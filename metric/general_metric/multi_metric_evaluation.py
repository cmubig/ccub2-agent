# Multi-metric Image Editing Evaluation: CLIPScore + Aesthetic Score + DreamSim
"""
Evaluate editing with three axes:
1) CLIPScore: text(prompt or editing_prompt) ↔ image alignment
2) Aesthetic: predicted human preference score from CLIP-ViT-L/14 image embedding
3) DreamSim: perceptual *distance* (higher = more different) vs step0 and vs previous step

Usage:
  python eval_clip_aesthetic_dreamsim.py \
    --csv /path/to/your.csv \
    --out /path/to/metrics.csv \
    --clip-model "ViT-L/14" \
    --dreamsim-type "ensemble" \
    --use-editing-prompt

CSV headers (flexible names; the script will look for both variants):
- prompt
- editing_prompt (optional; if --use-editing-prompt set but column missing, falls back to 'prompt')
- step0 / step0_path
- step1 / step1_path
...
- step5 / step5_path

Example row:
prompt,editing_prompt,step0_path,step1_path,step2_path,step3_path,step4_path,step5_path
"General House in China, photorealistic.","Change the image to represent china architecture house general.","/imgs/step0.png","/imgs/step1.png",...

Notes
- DreamSim returns a **distance**; we also report similarity = 1 - distance for convenience.
- Aesthetic predictor uses CLIP ViT-L/14 image embedding; keep the same backbone or replace weights accordingly.
"""
import argparse
import os
from typing import List, Tuple, Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# -------------------- CLIP --------------------
def load_clip(model_name: str, device: str):
    try:
        import clip  # OpenAI CLIP
    except Exception as e:
        raise RuntimeError("Install CLIP: pip install git+https://github.com/openai/CLIP.git") from e
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, clip


@torch.no_grad()
def clip_score_images(clip_model, clip_preprocess, clip_api, device: str, text: str, paths: List[str]):
    """Return list of (path, cosine, score_0_100). Processes in a batch."""
    tokens = clip_api.tokenize([text]).to(device)
    tfeat = clip_model.encode_text(tokens)
    tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    image_tensors = []
    valid_indices = []
    for i, p in enumerate(paths):
        if p and os.path.isfile(p):
            try:
                img = Image.open(p).convert("RGB")
                image_tensors.append(clip_preprocess(img))
                valid_indices.append(i)
            except Exception as e:
                print(f"      [WARN] Could not open image {p}: {e}")

    if not image_tensors:
        return [(p, float("nan"), float("nan")) for p in paths]

    image_batch = torch.stack(image_tensors).to(device)
    imfeats = clip_model.encode_image(image_batch)
    imfeats = imfeats / imfeats.norm(dim=-1, keepdim=True)

    cos_similarities = (imfeats @ tfeat.T).squeeze(-1).cpu().tolist()

    # Create a full result list with NaNs
    results = [(p, float("nan"), float("nan")) for p in paths]
    for i, sim in zip(valid_indices, cos_similarities):
        results[i] = (paths[i], sim, 100.0 * sim)

    return results


# -------------------- Aesthetic predictor (CLIP-ViT-L/14 embedding → MLP) --------------------
class AestheticMLP(nn.Module):
    """Architecture derived from state_dict error keys."""
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_aesthetic_predictor(device: str):
    """Load weights from HF Hub (trl-lib/ddpo-aesthetic-predictor)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub") from e

    mlp = AestheticMLP(in_dim=768).to(device).eval()
    cached = hf_hub_download("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")
    state = torch.load(cached, map_location=device)
    mlp.load_state_dict(state)
    return mlp


@torch.no_grad()
def aesthetic_from_clip_image(clip_model, clip_preprocess, device: str, mlp: nn.Module, paths: List[str]):
    """Get aesthetic score from CLIP ViT-L/14 image embedding. Processes in a batch."""
    image_tensors = []
    valid_indices = []
    for i, p in enumerate(paths):
        if p and os.path.isfile(p):
            try:
                img = Image.open(p).convert("RGB")
                image_tensors.append(clip_preprocess(img))
                valid_indices.append(i)
            except Exception as e:
                print(f"      [WARN] Could not open image {p}: {e}")

    if not image_tensors:
        return [(p, float("nan")) for p in paths]

    image_batch = torch.stack(image_tensors).to(device)
    imfeats = clip_model.encode_image(image_batch)
    imfeats = imfeats / imfeats.norm(dim=-1, keepdim=True)

    scores = mlp(imfeats.float()).squeeze(-1).cpu().tolist()
    if not isinstance(scores, list):
        scores = [scores] # Handle single-item batch case

    results = [(p, float("nan")) for p in paths]
    for i, score in zip(valid_indices, scores):
        results[i] = (paths[i], float(score))

    return results


# -------------------- DreamSim --------------------
def load_dreamsim(device: str, dreamsim_type: str = "ensemble", use_patch_model: bool = False):
    """
    dreamsim_type options include: "ensemble" (default), "dino_vitb16", "open_clip_vitb32", "clip_vitb32", "dinov2_vitb14", etc.
    """
    try:
        from dreamsim import dreamsim as dreamsim_fn
    except Exception as e:
        raise RuntimeError("Install DreamSim: pip install dreamsim") from e
    model, preprocess = dreamsim_fn(pretrained=True, device=device, dreamsim_type=dreamsim_type, use_patch_model=use_patch_model)
    model.eval()
    return model, preprocess


@torch.no_grad()
def dreamsim_distance(model, preprocess, device: str, a_path: str, b_path: str) -> float:
    """
    Returns the DreamSim *distance* between two images. Higher = more different, lower = more similar.
    """
    if (not a_path) or (not b_path) or (not os.path.isfile(a_path)) or (not os.path.isfile(b_path)):
        return float("nan")
    a = preprocess(Image.open(a_path).convert("RGB")).to(device)
    b = preprocess(Image.open(b_path).convert("RGB")).to(device)
    if a.dim() == 3:
        a = a.unsqueeze(0)
    if b.dim() == 3:
        b = b.unsqueeze(0)
    dist = model(a, b).item()
    return float(dist)


# -------------------- Main pipeline --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-model", default="ViT-L/14")
    ap.add_argument("--dreamsim-type", default="ensemble")
    ap.add_argument("--use-patch-model", action="store_true", help="Use DreamSim variant trained on CLS+patch features")
    ap.add_argument("--use-editing-prompt", action="store_true", help="Use 'editing_prompt' column for CLIPScore if present; else fall back to 'prompt'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Find step columns (support both 'stepK' and 'stepK_path')
    step_cols = []
    for k in range(6):
        for name in (f"step{k}", f"step{k}_path"):
            if name in df.columns:
                step_cols.append(name)
                break
    if not step_cols:
        raise ValueError("No step columns found. Expected one of step0..step5 or step0_path..step5_path")

    # Prompts
    use_text_col = None
    if args.use_editing_prompt and "editing_prompt" in df.columns:
        use_text_col = "editing_prompt"
    elif "prompt" in df.columns:
        use_text_col = "prompt"
    else:
        raise ValueError("CSV needs 'prompt' and/or 'editing_prompt'")

    # Load models
    device = args.device
    clip_model, clip_preprocess, clip_api = load_clip(args.clip_model, device)
    aest_mlp = load_aesthetic_predictor(device)
    ds_model, ds_preprocess = load_dreamsim(device, args.dreamsim_type, args.use_patch_model)

    # Iterate rows
    rows = []
    for ridx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating rows"):
        text = str(row[use_text_col])
        paths = [row[c] if isinstance(row[c], str) else "" for c in step_cols]
        base_path = paths[0] if paths else ""

        # CLIPScore
        clip_res = clip_score_images(clip_model, clip_preprocess, clip_api, device, text, paths)

        # Aesthetic
        aest_res = aesthetic_from_clip_image(clip_model, clip_preprocess, device, aest_mlp, paths)

        # DreamSim vs step0 and vs previous step (BATCHED)
        ds_vs_step0 = [float("nan")] * len(paths)
        ds_vs_prev = [float("nan")] * len(paths)

        valid_img_tensors = []
        valid_indices = []
        for i, p in enumerate(paths):
            if p and os.path.isfile(p):
                try:
                    tensor = ds_preprocess(Image.open(p).convert("RGB"))
                    if tensor.dim() == 4:
                        tensor = tensor.squeeze(0)
                    valid_img_tensors.append(tensor)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"      [WARN] Could not open image {p} for DreamSim: {e}")

        if len(valid_img_tensors) > 1:
            all_images_tensor = torch.stack(valid_img_tensors).to(device)

            # Batch compare all valid images to step0
            base_img_idx_in_batch = valid_indices.index(0) if 0 in valid_indices else -1
            if base_img_idx_in_batch != -1:
                base_img_tensor = all_images_tensor[base_img_idx_in_batch]
                batch_a_step0 = base_img_tensor.unsqueeze(0).repeat(len(valid_img_tensors), 1, 1, 1)
                batch_b_step0 = all_images_tensor
                with torch.no_grad():
                    d0_results = ds_model(batch_a_step0, batch_b_step0).cpu().tolist()
                for i, dist in zip(valid_indices, d0_results):
                    ds_vs_step0[i] = dist
                ds_vs_step0[0] = 0.0 # Distance to self is 0

            # Batch compare all valid images to previous valid image
            if len(valid_img_tensors) > 1:
                batch_a_prev = all_images_tensor[:-1]
                batch_b_prev = all_images_tensor[1:]
                with torch.no_grad():
                    dp_results = ds_model(batch_a_prev, batch_b_prev).cpu().tolist()

                # Map results back to the original sparse list of paths
                # Start from the second valid image, as the first has no predecessor
                for i in range(len(dp_results)):
                    original_idx = valid_indices[i+1]
                    ds_vs_prev[original_idx] = dp_results[i]

        # Merge
        for step_name, pth, (p1, cos, clip100), (p2, aest), d0, dp in zip(step_cols, paths, clip_res, aest_res, ds_vs_step0, ds_vs_prev):
            assert p1 == p2
            rows.append({
                "row_id": ridx,
                "prompt_used_for_clip": text,
                "step": step_name,
                "image_path": pth,
                "clip_cosine": cos,
                "clip_score_0_100": 100.0 * cos if cos == cos else float("nan"),
                "aesthetic_score": aest,
                "dreamsim_dist_vs_step0": d0,
                "dreamsim_sim_vs_step0": (1.0 - d0) if d0 == d0 else float("nan"),
                "dreamsim_dist_vs_prev_step": dp,
                "dreamsim_sim_vs_prev_step": (1.0 - dp) if dp == dp else float("nan"),
            })

    out_df = pd.DataFrame(rows)

    # Summaries
    # 1) best by CLIP
    best_clip_idx = out_df.groupby("row_id")["clip_score_0_100"].idxmax()
    best_clip_df = out_df.loc[best_clip_idx, ["row_id", "step", "clip_score_0_100"]].rename(
        columns={"step": "best_step_by_clip", "clip_score_0_100": "best_clip_score"}
    )

    # 2) best by Aesthetic
    best_aest_idx = out_df.groupby("row_id")["aesthetic_score"].idxmax()
    best_aest_df = out_df.loc[best_aest_idx, ["row_id", "step", "aesthetic_score"]].rename(
        columns={"step": "best_step_by_aesthetic", "aesthetic_score": "best_aesthetic"}
    )

    summary = df.reset_index().rename(columns={"index": "row_id"})
    summary = summary.merge(best_clip_df, on="row_id", how="left")
    summary = summary.merge(best_aest_df, on="row_id", how="left")

    # Save
    base_out = args.out
    out_df.to_csv(base_out, index=False)
    summary_out = os.path.splitext(base_out)[0] + "_summary.csv"
    summary.to_csv(summary_out, index=False)

    print(f"[OK] wrote per-image metrics: {base_out}")
    print(f"[OK] wrote per-row summary:  {summary_out}")


if __name__ == "__main__":
    main()
