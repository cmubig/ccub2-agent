#!/usr/bin/env python3
"""Quick FLUX model test"""

import torch
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set HF cache to use local cache
os.environ['HF_HOME'] = str(PROJECT_ROOT / 'data' / 'hf_cache')

print("="*70)
print("FLUX.1-dev Model Test")
print("="*70)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

print("[1/4] Loading FLUX model...")
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

print("[2/4] Enabling CPU offload for memory efficiency...")
if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cpu")

print("[3/4] Generating test image...")
prompt = "A Korean woman in traditional hanbok standing in front of Gyeongbokgung Palace, photorealistic, detailed"

print(f"Prompt: {prompt}")
print()

output_dir = Path("test_outputs")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"flux_test_{timestamp}.png"

with torch.inference_mode():
    result = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=1024,
        height=1024,
        generator=torch.manual_seed(42),
    )

print("[4/4] Saving image...")
result.images[0].save(output_path)

print()
print("="*70)
print("✅ FLUX Test Complete!")
print("="*70)
print(f"Output: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
print()
