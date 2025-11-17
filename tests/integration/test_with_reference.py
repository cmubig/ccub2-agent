#!/usr/bin/env python3
"""Test I2I editing with RAG reference image"""

import torch
from pathlib import Path
from datetime import datetime
import sys
import os

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['HF_HOME'] = str(PROJECT_ROOT / 'data' / 'hf_cache')

print("="*70)
print("TEST: I2I Editing with RAG Reference Image")
print("="*70)
print()

input_image = Path("test_outputs/flux_test_20251110_113409.png")
if not input_image.exists():
    print(f"❌ Input image not found: {input_image}")
    sys.exit(1)

# ========== Step 1: Load CLIP RAG ==========
print("[1/5] Loading CLIP Image RAG...")
from ccub2_agent.modules.clip_image_rag import create_clip_rag

clip_rag = create_clip_rag(
    model_name="openai/clip-vit-base-patch32",
    index_dir=Path("data/clip_index/korea"),
    device="cuda",
)
print("✓ CLIP RAG loaded")
print()

# ========== Step 2: Retrieve Reference Image ==========
print("[2/5] Retrieving similar traditional_clothing images...")
results = clip_rag.retrieve_similar_images(
    image_path=input_image,
    k=3,
    category="traditional_clothing",
)

print(f"✓ Found {len(results)} reference images")
for i, result in enumerate(results, 1):
    print(f"  {i}. {Path(result['image_path']).name} (similarity: {result['similarity']:.3f})")
    if 'description_enhanced' in result:
        print(f"     {result['description_enhanced'][:80]}...")

# Use the top result as reference
ref_image_path = Path(results[0]['image_path'])
ref_metadata = results[0]
print()
print(f"Using reference: {ref_image_path.name}")
print()

# ========== Step 3: Load VLM Detector ==========
print("[3/5] Loading VLM detector...")
from ccub2_agent.modules.vlm_detector import create_vlm_detector

detector = create_vlm_detector(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    index_dir=Path("data/cultural_index/korea"),
    clip_index_dir=Path("data/clip_index/korea"),
    load_in_4bit=True,
    debug=False,
)
print("✓ VLM detector loaded")
print()

# ========== Step 4: Detect Issues ==========
print("[4/5] Detecting cultural issues...")
prompt = "A Korean woman in traditional hanbok"
country = "korea"
category = "traditional_clothing"

issues = detector.detect(
    image_path=input_image,
    prompt=prompt,
    country=country,
    category=category,
)

cultural_score, prompt_score = detector.score_cultural_quality(
    image_path=input_image,
    prompt=prompt,
    country=country,
)

print(f"Cultural Score: {cultural_score}/10")
print(f"Issues Found: {len(issues)}")
print()

# ========== Step 5: Edit with Reference ==========
print("[5/5] Editing with Qwen-Image-Edit + Reference Image...")
from ccub2_agent.adapters.image_editing_adapter import create_adapter
from ccub2_agent.modules.prompt_adapter import get_prompt_adapter, EditingContext
from PIL import Image

editor = create_adapter(model_type='qwen', t2i_model='sdxl')
adapter = get_prompt_adapter()

context = EditingContext(
    original_prompt=prompt,
    detected_issues=issues,
    cultural_elements="Traditional Korean hanbok with jeogori and chima",
    reference_images=[str(ref_image_path)],
    country=country,
    category=category,
    preserve_identity=True,
)

universal_instruction = f"Transform this garment to authentic Korean hanbok style"
qwen_instruction = adapter.adapt(
    universal_instruction=universal_instruction,
    model_type='qwen',
    context=context,
)

print(f"Editing instruction: {qwen_instruction[:150]}...")
print()

input_img = Image.open(input_image)

# Edit with reference image!
edited_img = editor.edit(
    image=input_img,
    instruction=qwen_instruction,
    reference_image=ref_image_path,  # ← KEY: Pass reference image!
    reference_metadata=ref_metadata,
    strength=0.8,
)

output_dir = Path("test_outputs")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"edited_with_ref_{timestamp}.png"
edited_img.save(output_path)

print(f"✓ Edited image saved: {output_path}")
print()

# ========== Step 6: Re-evaluate ==========
print("[BONUS] Re-evaluating edited image...")
new_cultural_score, new_prompt_score = detector.score_cultural_quality(
    image_path=output_path,
    prompt=prompt,
    country=country,
)

print()
print("="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"Cultural Score:  {cultural_score}/10 → {new_cultural_score}/10 ({'+' if new_cultural_score > cultural_score else ''}{new_cultural_score - cultural_score})")
print(f"Prompt Score:    {prompt_score}/10 → {new_prompt_score}/10 ({'+' if new_prompt_score > prompt_score else ''}{new_prompt_score - prompt_score})")
print()
print(f"Reference:  {ref_image_path}")
print(f"Original:   {input_image}")
print(f"Edited:     {output_path}")
print()

if new_cultural_score > cultural_score:
    print("✅ Cultural accuracy IMPROVED with reference image!")
else:
    print("⚠️ No significant improvement")
print()
