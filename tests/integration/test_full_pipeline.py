#!/usr/bin/env python3
"""Full pipeline test: T2I -> VLM Detection -> I2I Editing"""

import torch
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set HF cache
os.environ['HF_HOME'] = str(PROJECT_ROOT / 'data' / 'hf_cache')

print("="*70)
print("FULL PIPELINE TEST: VLM Detection + I2I Editing")
print("="*70)
print()

# Use the previously generated FLUX image
input_image = Path("test_outputs/flux_test_20251110_113409.png")
if not input_image.exists():
    print(f"❌ Input image not found: {input_image}")
    sys.exit(1)

print(f"Input image: {input_image}")
print()

# ========== Step 1: Load VLM Detector ==========
print("[1/5] Loading VLM Cultural Detector...")
from ccub2_agent.modules.vlm_detector import create_vlm_detector

detector = create_vlm_detector(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    index_dir=Path("data/cultural_index/korea"),
    clip_index_dir=Path("data/clip_index/korea"),
    load_in_4bit=True,
    debug=True,
)
print("✓ VLM detector loaded")
print()

# ========== Step 2: Detect Issues ==========
print("[2/5] Detecting cultural issues...")
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

print(f"✓ Cultural Score: {cultural_score}/10")
print(f"✓ Prompt Score: {prompt_score}/10")
print(f"✓ Issues Found: {len(issues)}")
for i, issue in enumerate(issues[:5], 1):
    print(f"  {i}. [{issue['type']}] {issue['description'][:80]}...")
print()

if len(issues) == 0:
    print("✓ No issues detected! Image is culturally accurate.")
    sys.exit(0)

# ========== Step 3: Load I2I Editor ==========
print("[3/5] Loading Qwen Image Editor (I2I)...")
from ccub2_agent.adapters.image_editing_adapter import create_adapter

editor = create_adapter(
    model_type='qwen',
    t2i_model='sdxl',
)
print("✓ I2I editor loaded")
print()

# ========== Step 4: Generate Editing Instruction ==========
print("[4/5] Generating editing instruction...")
from ccub2_agent.modules.prompt_adapter import get_prompt_adapter, EditingContext

adapter = get_prompt_adapter()

# Build editing context
context = EditingContext(
    original_prompt=prompt,
    detected_issues=issues,
    cultural_elements="Traditional Korean hanbok: jeogori (short jacket), chima (long flowing skirt), high waistline",
    reference_images=None,
    country=country,
    category=category,
    preserve_identity=True,
)

# Generate universal instruction
universal_instruction = f"Fix cultural accuracy issues in this {country} {category} image"

# Adapt to Qwen format
qwen_instruction = adapter.adapt(
    universal_instruction=universal_instruction,
    model_type='qwen',
    context=context,
)

print(f"Editing instruction:\n{qwen_instruction[:300]}...")
print()

# ========== Step 5: Edit Image ==========
print("[5/5] Editing image with Qwen-Image-Edit...")
from PIL import Image

input_img = Image.open(input_image)

edited_img = editor.edit(
    image=input_img,
    instruction=qwen_instruction,
    strength=0.8,
)

# Save result
output_dir = Path("test_outputs")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"edited_{timestamp}.png"
edited_img.save(output_path)

print(f"✓ Edited image saved: {output_path}")
print()

# ========== Step 6: Re-evaluate ==========
print("[BONUS] Re-evaluating edited image...")
new_issues = detector.detect(
    image_path=output_path,
    prompt=prompt,
    country=country,
    category=category,
)

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
print(f"Issues:          {len(issues)} → {len(new_issues)} ({'+' if len(new_issues) > len(issues) else ''}{len(new_issues) - len(issues)})")
print()
print(f"Original:  {input_image}")
print(f"Edited:    {output_path}")
print()

if new_cultural_score > cultural_score:
    print("✅ Cultural accuracy improved!")
else:
    print("⚠️ Cultural accuracy did not improve significantly")
print()
