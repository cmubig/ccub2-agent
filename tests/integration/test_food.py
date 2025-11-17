#!/usr/bin/env python3
"""Test Korean food generation and editing"""

import torch
from pathlib import Path
from datetime import datetime
import sys
import os

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['HF_HOME'] = str(PROJECT_ROOT / 'data' / 'hf_cache')

print("="*70)
print("KOREAN FOOD TEST: T2I → VLM → I2I Pipeline")
print("="*70)
print()

# ========== Step 1: Generate Korean food image ==========
print("[1/6] Generating Korean food with FLUX...")
from diffusers import FluxPipeline
from PIL import Image

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()

prompt = "A traditional Korean bibimbap bowl with colorful vegetables, egg, and gochujang sauce, photorealistic food photography"
print(f"Prompt: {prompt}")
print()

output_dir = Path("test_outputs")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
initial_path = output_dir / f"food_initial_{timestamp}.png"

with torch.inference_mode():
    result = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=1024,
        height=1024,
        generator=torch.manual_seed(42),
    )

result.images[0].save(initial_path)
print(f"✓ Initial image saved: {initial_path}")
print()

# Clean up FLUX
del pipe
import gc
gc.collect()
torch.cuda.empty_cache()

# ========== Step 2: Load CLIP RAG for food ==========
print("[2/6] Loading CLIP RAG and retrieving food references...")
from ccub2_agent.modules.clip_image_rag import create_clip_rag

clip_rag = create_clip_rag(
    model_name="openai/clip-vit-base-patch32",
    index_dir=Path("data/clip_index/korea"),
    device="cuda",
)

results = clip_rag.retrieve_similar_images(
    image_path=initial_path,
    k=5,
    category="food",
)

print(f"✓ Found {len(results)} similar food images:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {Path(result['image_path']).name} (similarity: {result['similarity']:.3f})")
    if 'description' in result['metadata']:
        desc = result['metadata'].get('description', '')[:60]
        print(f"     {desc}...")

if len(results) > 0:
    ref_image_path = Path(results[0]['image_path'])
    ref_metadata = results[0]['metadata']
    print(f"\n✓ Using reference: {ref_image_path.name}")
else:
    print("⚠ No reference images found!")
    ref_image_path = None
    ref_metadata = {}
print()

# ========== Step 3: VLM Detection ==========
print("[3/6] Loading VLM detector...")
from ccub2_agent.modules.vlm_detector import create_vlm_detector

detector = create_vlm_detector(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    index_dir=Path("data/cultural_index/korea"),
    clip_index_dir=Path("data/clip_index/korea"),
    load_in_4bit=True,
    debug=False,
)
print("✓ VLM loaded")
print()

print("[4/6] Detecting cultural issues in food image...")
country = "korea"
category = "food"
original_prompt = "traditional Korean bibimbap"

issues = detector.detect(
    image_path=initial_path,
    prompt=original_prompt,
    country=country,
    category=category,
)

cultural_score, prompt_score = detector.score_cultural_quality(
    image_path=initial_path,
    prompt=original_prompt,
    country=country,
)

print(f"Cultural Score: {cultural_score}/10")
print(f"Prompt Score: {prompt_score}/10")
print(f"Issues Found: {len(issues)}")
for i, issue in enumerate(issues[:5], 1):
    print(f"  {i}. [{issue['type']}] {issue['description'][:80]}...")
print()

if cultural_score >= 8 and len(issues) == 0:
    print("✅ Food image is already culturally accurate!")
    sys.exit(0)

# ========== Step 5: I2I Editing ==========
print("[5/6] Editing with Qwen Image Editor...")
from ccub2_agent.adapters.image_editing_adapter import create_adapter
from ccub2_agent.modules.prompt_adapter import get_prompt_adapter, EditingContext

editor = create_adapter(model_type='qwen', t2i_model='sdxl')
adapter = get_prompt_adapter()

context = EditingContext(
    original_prompt=original_prompt,
    detected_issues=issues,
    cultural_elements="Traditional Korean food presentation",
    reference_images=[str(ref_image_path)] if ref_image_path else None,
    country=country,
    category=category,
    preserve_identity=True,
)

universal_instruction = f"Fix cultural accuracy issues in this Korean food image"
qwen_instruction = adapter.adapt(
    universal_instruction=universal_instruction,
    model_type='qwen',
    context=context,
)

print(f"Editing instruction: {qwen_instruction[:200]}...")
print()

input_img = Image.open(initial_path)

edited_img = editor.edit(
    image=input_img,
    instruction=qwen_instruction,
    reference_image=ref_image_path,
    reference_metadata=ref_metadata,
    strength=0.8,
)

edited_path = output_dir / f"food_edited_{timestamp}.png"
edited_img.save(edited_path)
print(f"✓ Edited image saved: {edited_path}")
print()

# ========== Step 6: Re-evaluate ==========
print("[6/6] Re-evaluating edited image...")
new_cultural_score, new_prompt_score = detector.score_cultural_quality(
    image_path=edited_path,
    prompt=original_prompt,
    country=country,
)

new_issues = detector.detect(
    image_path=edited_path,
    prompt=original_prompt,
    country=country,
    category=category,
)

print()
print("="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"Cultural Score:  {cultural_score}/10 → {new_cultural_score}/10 ({'+' if new_cultural_score > cultural_score else ''}{new_cultural_score - cultural_score})")
print(f"Prompt Score:    {prompt_score}/10 → {new_prompt_score}/10 ({'+' if new_prompt_score > prompt_score else ''}{new_prompt_score - prompt_score})")
print(f"Issues:          {len(issues)} → {len(new_issues)} ({'-' if len(new_issues) < len(issues) else '+'}{len(new_issues) - len(issues)})")
print()
print(f"Initial:  {initial_path}")
print(f"Edited:   {edited_path}")
if ref_image_path:
    print(f"Ref:      {ref_image_path}")
print()

if new_cultural_score > cultural_score:
    print("✅ Cultural accuracy IMPROVED!")
elif new_cultural_score == cultural_score:
    print("➡️ No change in cultural score")
else:
    print("⚠️ Cultural score decreased")
print()
