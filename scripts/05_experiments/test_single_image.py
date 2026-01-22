#!/usr/bin/env python3
"""
Test single image with FULL pipeline (VLM + KB + CLIP + Prompt Adapter + I2I)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

from PIL import Image

# Test config
BASE_DIR = PROJECT_ROOT / "base_experimental"
TEST_IMAGE = BASE_DIR / "China" / "flux_china_art_painting_modern.png"
OUTPUT_DIR = BASE_DIR / "single_test"
OUTPUT_DIR.mkdir(exist_ok=True)

COUNTRY = "china"
CATEGORY = "art"
PROMPT = "Change the image to represent modern painting in China."

print("=" * 70)
print("SINGLE IMAGE TEST - FULL PIPELINE")
print("=" * 70)
print(f"Image: {TEST_IMAGE.name}")
print(f"Country: {COUNTRY}, Category: {CATEGORY}")

# ============================================
# Step 1: Load VLM Detector
# ============================================
print("\n[1/6] Loading VLM Detector...")
from ccub2_agent.modules.vlm_detector import VLMCulturalDetector
vlm = VLMCulturalDetector(load_in_4bit=True)
print("  ✓ VLM loaded")

# ============================================
# Step 2: Detect cultural issues
# ============================================
print("\n[2/6] Detecting cultural issues...")
issues = vlm.detect(
    image_path=TEST_IMAGE,
    prompt=PROMPT,
    country=COUNTRY,
    category=CATEGORY
)
print(f"  Found {len(issues)} issues:")
for i, issue in enumerate(issues):
    desc = issue.get('description', '')
    print(f"    [{i+1}] {desc[:150]}...")

# ============================================
# Step 3: Query Text Knowledge Base
# ============================================
print("\n[3/6] Querying Text Knowledge Base...")
cultural_context = ""
try:
    from metric.cultural_metric.enhanced_cultural_metric_pipeline import EnhancedCulturalKnowledgeBase
    kb_path = PROJECT_ROOT / "data" / "cultural_index" / COUNTRY
    if kb_path.exists() and (kb_path / "faiss.index").exists():
        kb = EnhancedCulturalKnowledgeBase(kb_path)
        query = f"{CATEGORY} {COUNTRY} " + " ".join(
            issue.get('description', '')[:50] for issue in issues[:3]
        )
        docs = kb.retrieve(query, top_k=5)
        if docs:
            cultural_context = "\n".join([doc.text for doc in docs[:3]])
            print(f"  ✓ Retrieved {len(docs)} docs")
            print(f"    Preview: {cultural_context[:200]}...")
        else:
            cultural_context = f"Traditional {CATEGORY} elements specific to {COUNTRY} culture"
            print("  ⚠ No docs found, using fallback")
    else:
        print(f"  ⚠ KB not found at {kb_path}")
except Exception as e:
    print(f"  ⚠ KB query failed: {e}")
    cultural_context = f"Traditional {CATEGORY} elements specific to {COUNTRY} culture"

# ============================================
# Step 4: CLIP RAG Search (optional, for metadata)
# ============================================
print("\n[4/6] CLIP RAG Search...")
reference_metadata = None
try:
    from ccub2_agent.modules.clip_image_rag import CLIPImageRAG
    clip_path = PROJECT_ROOT / "data" / "clip_index" / COUNTRY
    if clip_path.exists():
        clip_rag = CLIPImageRAG(index_dir=clip_path)
        results = clip_rag.retrieve_similar_images(image_path=TEST_IMAGE, k=3, category=CATEGORY)
        if results:
            # Get metadata only (not passing image to I2I)
            reference_metadata = results[0].get('metadata', {})
            print(f"  ✓ Found {len(results)} references")
            print(f"    Top: {Path(results[0].get('image_path', '')).name} (sim: {results[0].get('similarity', 0):.2%})")
        else:
            print("  ⚠ No CLIP results")
    else:
        print(f"  ⚠ CLIP index not found")
except Exception as e:
    print(f"  ⚠ CLIP search failed: {e}")

# ============================================
# Step 5: Generate Prompt with Adapter
# ============================================
print("\n[5/6] Generating prompt...")
from ccub2_agent.modules.prompt_adapter import UniversalPromptAdapter, EditingContext

adapter = UniversalPromptAdapter()
context = EditingContext(
    original_prompt=PROMPT,
    detected_issues=issues,
    cultural_elements=cultural_context,
    reference_images=None,  # Not passing reference image
    country=COUNTRY,
    category=CATEGORY,
    preserve_identity=True
)

adapted_prompt = adapter.adapt(PROMPT, "qwen", context)
print(f"\n  === FINAL PROMPT ===")
print(f"  {adapted_prompt}")
print(f"  =====================")
print(f"  Length: {len(adapted_prompt)} chars, {len(adapted_prompt.split())} words")

# ============================================
# Step 6: Edit with Qwen
# ============================================
print("\n[6/6] Editing image with Qwen...")
from ccub2_agent.adapters.image_editing_adapter import create_adapter

i2i = create_adapter(model_type='qwen', t2i_model='sd35')
image = Image.open(TEST_IMAGE).convert('RGB')

edited = i2i.edit(
    image=image,
    instruction=adapted_prompt,
    reference_metadata=reference_metadata,  # Text metadata only
    strength=0.35,
    num_inference_steps=40,
    seed=42
)

output_path = OUTPUT_DIR / "test_china_art_painting_modern.png"
edited.save(output_path)

print(f"\n{'=' * 70}")
print(f"✓ DONE! Saved: {output_path}")
print(f"{'=' * 70}")
