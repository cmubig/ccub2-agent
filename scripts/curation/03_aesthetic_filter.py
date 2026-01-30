#!/usr/bin/env python3
"""Filter curated images by aesthetic quality using CLIP-based aesthetic predictor.

Uses the LAION Aesthetic Predictor v2 (linear probe on CLIP ViT-L/14 embeddings)
to score images and filter out low-quality ones.

Usage:
    python scripts/curation/03_aesthetic_filter.py --country korea
    python scripts/curation/03_aesthetic_filter.py --country korea --threshold 5.5 --device cuda:1
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AestheticResult:
    """Result for a single image."""

    image_id: str
    image_path: str
    score: float
    passed: bool
    error: str = ""


@dataclass
class AestheticReport:
    """Aggregated filtering report."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    mean_score: float = 0.0
    results: list[AestheticResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class AestheticPredictor(nn.Module):
    """LAION Aesthetic Predictor v2 - linear probe on CLIP ViT-L/14 embeddings.

    This is a simple MLP that takes 768-dim CLIP ViT-L/14 image embeddings
    and predicts an aesthetic score from 1-10.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticFilter:
    """Filter images by aesthetic quality using CLIP + aesthetic predictor."""

    def __init__(
        self,
        curated_dir: Path,
        threshold: float = 5.0,
        device: str = "auto",
        clip_model: str = "openai/clip-vit-large-patch14",
    ):
        self.curated_dir = Path(curated_dir)
        self.threshold = threshold
        self.clip_model_name = clip_model

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP ViT-L/14 on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained(
            clip_model, use_safetensors=True
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model.eval()

        # Load aesthetic predictor weights
        self.predictor = AestheticPredictor().to(self.device)
        self._load_predictor_weights()
        self.predictor.eval()

    def _load_predictor_weights(self):
        """Load pretrained aesthetic predictor weights."""
        weights_path = self.curated_dir / "models" / "aesthetic_predictor_v2.pth"

        if weights_path.exists():
            logger.info(f"Loading aesthetic predictor from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.predictor.load_state_dict(state_dict)
        else:
            logger.warning(
                f"No pretrained weights at {weights_path}. "
                "Attempting to download from HuggingFace..."
            )
            self._download_predictor_weights(weights_path)

    def _download_predictor_weights(self, save_path: Path):
        """Download aesthetic predictor weights from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
                filename="linear.pth",
            )
            # Load and adapt weights
            state_dict = torch.load(local_path, map_location=self.device, weights_only=True)

            # The downloaded weights may have different key names
            # Try loading directly first
            try:
                self.predictor.load_state_dict(state_dict)
                logger.info("Loaded aesthetic predictor weights (direct)")
            except RuntimeError:
                # If keys don't match, the model structure differs
                # Fall back to simple CLIP-norm scoring
                logger.warning(
                    "Weight keys mismatch. Using CLIP embedding norm as proxy score."
                )
                self._use_norm_fallback = True
                return

            # Save for next time
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.predictor.state_dict(), save_path)
            logger.info(f"Saved weights to {save_path}")

        except ImportError:
            logger.warning("huggingface_hub not installed. Using CLIP norm fallback.")
            self._use_norm_fallback = True
        except Exception as e:
            logger.warning(f"Failed to download weights: {e}. Using CLIP norm fallback.")
            self._use_norm_fallback = True

    @torch.no_grad()
    def score_image(self, image_path: Path) -> float:
        """Score a single image's aesthetic quality (1-10 scale)."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get CLIP ViT-L/14 image embedding (768-dim)
        vision_outputs = self.clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )
        embedding = vision_outputs.pooler_output  # [1, 768]

        if getattr(self, "_use_norm_fallback", False):
            # Fallback: use embedding norm as proxy (normalized to ~1-10 range)
            norm = embedding.norm(dim=-1).item()
            return min(max(norm / 3.0, 1.0), 10.0)

        # Run through aesthetic predictor
        score = self.predictor(embedding).item()
        return float(np.clip(score, 1.0, 10.0))

    def filter_country(self, country: str) -> AestheticReport:
        """Filter all curated images for a country."""
        report = AestheticReport()

        # Find provenance file
        provenance_path = self.curated_dir / "metadata" / "provenance.jsonl"
        if not provenance_path.exists():
            logger.warning(f"No provenance file at {provenance_path}")
            return report

        # Load records
        records = []
        with open(provenance_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("country", "").lower() == country.lower():
                    records.append(record)

        if not records:
            logger.warning(f"No records found for {country}")
            return report

        logger.info(f"Scoring {len(records)} images for {country}...")
        scores = []

        for i, record in enumerate(records):
            image_id = record.get("image_id", f"unknown_{i}")
            image_path = record.get("image_path", "")

            if not image_path or not Path(image_path).exists():
                report.results.append(
                    AestheticResult(
                        image_id=image_id,
                        image_path=image_path,
                        score=0.0,
                        passed=False,
                        error="Image file not found",
                    )
                )
                report.errored += 1
                report.total += 1
                continue

            try:
                score = self.score_image(Path(image_path))
                passed = score >= self.threshold
                scores.append(score)

                report.results.append(
                    AestheticResult(
                        image_id=image_id,
                        image_path=image_path,
                        score=score,
                        passed=passed,
                    )
                )
                report.total += 1
                if passed:
                    report.passed += 1
                else:
                    report.failed += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i + 1}/{len(records)} images...")

            except Exception as e:
                logger.warning(f"Error scoring {image_id}: {e}")
                report.results.append(
                    AestheticResult(
                        image_id=image_id,
                        image_path=image_path,
                        score=0.0,
                        passed=False,
                        error=str(e),
                    )
                )
                report.errored += 1
                report.total += 1

        if scores:
            report.mean_score = float(np.mean(scores))

        logger.info(
            f"Aesthetic filter for {country}: "
            f"{report.passed}/{report.total} passed "
            f"(mean={report.mean_score:.2f}, threshold={self.threshold})"
        )

        return report

    def save_report(self, report: AestheticReport, output_path: Path):
        """Save filtering report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "errored": report.errored,
            "mean_score": report.mean_score,
            "pass_rate": report.pass_rate,
            "threshold": self.threshold,
            "results": [
                {
                    "image_id": r.image_id,
                    "image_path": r.image_path,
                    "score": r.score,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in report.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to {output_path}")

    def update_provenance(self, report: AestheticReport):
        """Update provenance file with aesthetic scores and filter status."""
        provenance_path = self.curated_dir / "metadata" / "provenance.jsonl"
        if not provenance_path.exists():
            return

        # Build lookup
        score_map = {r.image_id: r for r in report.results}

        # Read, update, rewrite
        updated_records = []
        with open(provenance_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_id = record.get("image_id", "")
                if image_id in score_map:
                    result = score_map[image_id]
                    record["aesthetic_score"] = result.score
                    record["aesthetic_passed"] = result.passed
                updated_records.append(record)

        with open(provenance_path, "w") as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Updated {len(score_map)} records in provenance file")


def main():
    parser = argparse.ArgumentParser(description="Filter curated images by aesthetic quality")
    parser.add_argument("--country", required=True, help="Country code (e.g., korea)")
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="Minimum aesthetic score (default: 5.0)"
    )
    parser.add_argument("--curated-dir", default=None, help="Curated data directory")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda:0/cuda:1/cpu)")
    parser.add_argument(
        "--update-provenance",
        action="store_true",
        help="Update provenance file with scores",
    )
    args = parser.parse_args()

    curated_dir = (
        Path(args.curated_dir)
        if args.curated_dir
        else Path(__file__).parents[2] / "data" / "curated"
    )

    af = AestheticFilter(
        curated_dir=curated_dir,
        threshold=args.threshold,
        device=args.device,
    )

    report = af.filter_country(args.country)

    print(f"\n{'='*50}")
    print(f"Aesthetic Filter Report: {args.country}")
    print(f"{'='*50}")
    print(f"Total:      {report.total}")
    print(f"Passed:     {report.passed}")
    print(f"Failed:     {report.failed}")
    print(f"Errored:    {report.errored}")
    print(f"Mean Score: {report.mean_score:.2f}")
    print(f"Pass Rate:  {report.pass_rate:.0%}")
    print(f"Threshold:  {args.threshold}")

    # Save report
    report_path = curated_dir / "reports" / f"aesthetic_{args.country}.json"
    af.save_report(report, report_path)

    if args.update_provenance:
        af.update_provenance(report)

    # Print low-scoring images
    low = [r for r in report.results if r.passed is False and not r.error]
    if low:
        print(f"\nLowest scoring images (below {args.threshold}):")
        for r in sorted(low, key=lambda x: x.score)[:10]:
            print(f"  {r.image_id}: {r.score:.2f}")


if __name__ == "__main__":
    main()
