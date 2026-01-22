#!/usr/bin/env python3
"""
Full Pipeline Batch Experiment for OURS

Runs the COMPLETE pipeline for each image:
1. Input Image + I2I Prompt
2. VLM Detector (cultural issue detection)
3. Text KB Query (cultural knowledge retrieval)
4. CLIP RAG Search (reference image retrieval)
5. Reference Selector (best reference selection)
6. Prompt Adapter (model-specific prompt generation)
7. I2I Editor (Qwen-Image-Edit-2509 or FLUX.2-dev)

Usage:
    python run_ours_full_pipeline.py --model qwen    # Qwen-Image-Edit-2509
    python run_ours_full_pipeline.py --model flux2   # FLUX.2-dev (NEW!)
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from io import StringIO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch

# Setup logging
log_file = PROJECT_ROOT / 'logs' / f'ours_full_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = PROJECT_ROOT / "base_experimental"
OUTPUT_DIR = BASE_DIR / "ours"
CHINA_CSV = BASE_DIR / "edit-prompt.csv"
KOREA_CSV = BASE_DIR / "edit-prompt-kor.csv"


def load_all_prompts():
    """Load prompts from both CSV files."""
    prompts = []

    # Load China prompts
    if CHINA_CSV.exists():
        with open(CHINA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({
                    'prompt': row['I2I_prompt'],
                    'base_path': row['base_path'],
                    'output_name': Path(row['base_path']).name,
                    'country': 'china'
                })
        logger.info(f"Loaded {len(prompts)} China prompts")

    # Load Korea prompts
    korea_start = len(prompts)
    if KOREA_CSV.exists():
        with open(KOREA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({
                    'prompt': row['I2I_prompt'],
                    'base_path': row['base_path'],
                    'output_name': Path(row['base_path']).name,
                    'country': 'korea'
                })
        logger.info(f"Loaded {len(prompts) - korea_start} Korea prompts")

    return prompts


def resolve_image_path(base_path_str: str) -> Optional[Path]:
    """Resolve image path handling case sensitivity."""
    possible_paths = [
        BASE_DIR / base_path_str,
        BASE_DIR / base_path_str.replace('china/', 'China/'),
        BASE_DIR / base_path_str.replace('korea/', 'Korea/'),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def extract_category(filename: str) -> str:
    """Extract category from filename."""
    # flux_china_architecture_house_traditional.png -> architecture
    parts = filename.replace('.png', '').split('_')
    if len(parts) >= 3:
        return parts[2]
    return 'general'


class FullPipelineRunner:
    """Runs the complete CCUB2 pipeline."""

    def __init__(self, i2i_model: str = 'qwen'):
        self.vlm_detector = None
        self.i2i_adapter = None
        self.prompt_adapter = None
        self.clip_rags = {}  # country -> CLIPImageRAG
        self.text_kbs = {}   # country -> TextKnowledgeBase
        self.i2i_model = i2i_model

    def initialize(self):
        """Initialize all pipeline components."""
        logger.info("=" * 70)
        logger.info("Initializing Full Pipeline Components...")
        logger.info("=" * 70)

        # 1. VLM Detector
        logger.info("\n[1/4] Loading VLM Detector (Qwen3-VL-8B)...")
        from ccub2_agent.modules.vlm_detector import VLMCulturalDetector
        self.vlm_detector = VLMCulturalDetector(load_in_4bit=True)
        logger.info("  ✓ VLM Detector loaded")

        # 2. I2I Adapter
        model_names = {
            'qwen': 'Qwen-Image-Edit-2509',
            'flux2': 'FLUX.2-dev',
            'flux': 'FLUX.1 ControlNet',
            'sd35': 'SD 3.5 Medium',
        }
        model_display = model_names.get(self.i2i_model, self.i2i_model)
        logger.info(f"\n[2/4] Loading I2I Adapter ({model_display})...")
        from ccub2_agent.adapters.image_editing_adapter import create_adapter
        self.i2i_adapter = create_adapter(model_type=self.i2i_model, t2i_model='sd35')
        logger.info(f"  ✓ I2I Adapter loaded ({model_display})")

        # 3. Prompt Adapter
        logger.info("\n[3/4] Loading Prompt Adapter...")
        from ccub2_agent.modules.prompt_adapter import UniversalPromptAdapter
        self.prompt_adapter = UniversalPromptAdapter()
        logger.info("  ✓ Prompt Adapter loaded")

        # 4. CLIP RAG for each country
        logger.info("\n[4/4] Loading CLIP RAG indices...")
        from ccub2_agent.modules.clip_image_rag import CLIPImageRAG

        for country in ['china', 'korea']:
            clip_index_path = PROJECT_ROOT / "data" / "clip_index" / country
            if clip_index_path.exists():
                try:
                    self.clip_rags[country] = CLIPImageRAG(index_dir=clip_index_path)
                    logger.info(f"  ✓ CLIP RAG loaded for {country}")
                except Exception as e:
                    logger.warning(f"  ⚠ CLIP RAG failed for {country}: {e}")
            else:
                logger.warning(f"  ⚠ CLIP index not found for {country}")

        # 5. Text Knowledge Base (optional)
        logger.info("\nLoading Text Knowledge Bases...")
        try:
            from metric.cultural_metric.enhanced_cultural_metric_pipeline import EnhancedCulturalKnowledgeBase

            for country in ['china', 'korea']:
                # Use cultural_index folder (contains faiss.index, metadata.jsonl, index_config.json)
                kb_path = PROJECT_ROOT / "data" / "cultural_index" / country
                if kb_path.exists() and (kb_path / "faiss.index").exists():
                    self.text_kbs[country] = EnhancedCulturalKnowledgeBase(kb_path)
                    logger.info(f"  ✓ Text KB loaded for {country}")
                else:
                    logger.warning(f"  ⚠ Text KB index not found for {country}")
        except Exception as e:
            logger.warning(f"  ⚠ Text KB loading failed: {e}")

        logger.info("\n" + "=" * 70)
        logger.info("All components initialized!")
        logger.info("=" * 70 + "\n")

    def process_image(
        self,
        input_path: Path,
        i2i_prompt: str,
        country: str,
        category: str,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Process a single image through the full pipeline.

        Returns dict with scores and metadata.
        """
        result = {
            'input': str(input_path),
            'output': str(output_path),
            'country': country,
            'category': category,
            'i2i_prompt': i2i_prompt,
        }

        # Load image
        image = Image.open(input_path).convert('RGB')

        # ============================================
        # Step 1: VLM Detector - Detect cultural issues
        # ============================================
        logger.info("  [Step 1] VLM Detector - Detecting cultural issues...")
        try:
            issues = self.vlm_detector.detect(
                image_path=input_path,
                prompt=i2i_prompt,
                country=country,
                category=category
            )
            result['initial_issues'] = len(issues)
            result['issues'] = [
                {'description': issue.get('description', str(issue)), 'severity': issue.get('severity', 5)}
                for issue in issues[:5]
            ]
            logger.info(f"    Found {len(issues)} issues")
            for issue in issues[:3]:
                logger.info(f"      - {issue.get('description', issue)[:80]}")
        except Exception as e:
            logger.warning(f"    VLM detection failed: {e}")
            issues = []
            result['initial_issues'] = 0
            result['issues'] = []

        # ============================================
        # Step 2: Text KB Query - Get cultural knowledge
        # ============================================
        logger.info("  [Step 2] Text KB Query - Retrieving cultural knowledge...")
        cultural_context = ""
        if country in self.text_kbs:
            try:
                kb = self.text_kbs[country]
                # Query based on category and issues
                query = f"{category} {country} " + " ".join(
                    issue.get('description', '')[:50] for issue in issues[:3]
                )
                # Actually retrieve cultural context from KB
                docs = kb.retrieve(query, country=country, top_k=5)
                if docs:
                    cultural_context = "\n".join([doc.text for doc in docs[:3]])
                    logger.info(f"    Retrieved {len(docs)} cultural context entries")
                    logger.info(f"    Context preview: {cultural_context[:150]}...")
                else:
                    cultural_context = f"Traditional {category} elements specific to {country} culture"
                    logger.info(f"    No KB docs found, using fallback context")
            except Exception as e:
                logger.warning(f"    Text KB query failed: {e}")
                cultural_context = f"Traditional {category} elements specific to {country} culture"
        else:
            logger.info(f"    No Text KB available for {country}")

        # ============================================
        # Step 3: CLIP RAG Search - Find reference images
        # ============================================
        logger.info("  [Step 3] CLIP RAG Search - Finding reference images...")
        reference_image = None
        reference_metadata = None
        reference_path = None

        if country in self.clip_rags:
            try:
                clip_rag = self.clip_rags[country]
                search_results = clip_rag.retrieve_similar_images(
                    image_path=input_path,
                    k=5,
                    category=category
                )

                if search_results:
                    # Filter out self-reference
                    input_name = input_path.name
                    filtered = [r for r in search_results if Path(r.get('image_path', '')).name != input_name]

                    if filtered:
                        result['clip_results'] = [
                            {'path': Path(r['image_path']).name, 'score': r.get('similarity', 0)}
                            for r in filtered[:3]
                        ]
                        logger.info(f"    Found {len(filtered)} reference candidates")

                        # ============================================
                        # Step 4: Reference Selector - Pick best reference
                        # ============================================
                        logger.info("  [Step 4] Reference Selector - Selecting best reference...")

                        # Select top reference
                        top_ref = filtered[0]
                        reference_path = Path(top_ref['image_path'])
                        if reference_path.exists():
                            reference_image = Image.open(reference_path).convert('RGB')
                            reference_metadata = top_ref.get('metadata', {})
                            result['selected_reference'] = reference_path.name
                            logger.info(f"    Selected: {reference_path.name} (similarity: {top_ref.get('similarity', 0):.2%})")
                        else:
                            logger.warning(f"    Reference path not found: {reference_path}")
                    else:
                        logger.info("    All results were self-references")
                else:
                    logger.info("    No CLIP results found")
            except Exception as e:
                logger.warning(f"    CLIP RAG search failed: {e}")
        else:
            logger.info(f"    No CLIP RAG available for {country}")

        # ============================================
        # Step 5: Prompt Adapter - Generate model-specific prompt
        # ============================================
        logger.info("  [Step 5] Prompt Adapter - Generating optimized prompt...")
        try:
            from ccub2_agent.modules.prompt_adapter import EditingContext

            # Build context from detected issues
            context = EditingContext(
                original_prompt=i2i_prompt,
                detected_issues=issues[:5],
                cultural_elements=cultural_context,
                reference_images=[str(reference_path)] if reference_path else None,
                country=country,
                category=category,
                preserve_identity=True
            )

            adapted_prompt = self.prompt_adapter.adapt(
                universal_instruction=i2i_prompt,
                model_type=self.i2i_model if self.i2i_model != 'flux2' else 'flux',
                context=context
            )
            # Store full prompt (up to 2000 chars for CSV) - the full prompt is used for I2I
            result['adapted_prompt'] = adapted_prompt[:2000]
            logger.info(f"    Adapted prompt ({len(adapted_prompt)} chars): {adapted_prompt[:150]}...")
        except Exception as e:
            logger.warning(f"    Prompt adaptation failed: {e}")
            adapted_prompt = i2i_prompt

        # ============================================
        # Step 6: I2I Editor - Apply edits
        # ============================================
        model_names = {'qwen': 'Qwen-Image-Edit-2509', 'flux2': 'FLUX.2-dev', 'flux': 'FLUX.1', 'sd35': 'SD3.5'}
        logger.info(f"  [Step 6] I2I Editor - Applying edits with {model_names.get(self.i2i_model, self.i2i_model)}...")
        try:
            edited_image = self.i2i_adapter.edit(
                image=image,
                instruction=adapted_prompt,
                reference_image=reference_image,
                reference_metadata=reference_metadata,
                strength=0.35,
                num_inference_steps=40,
                seed=42
            )

            # Save result
            edited_image.save(output_path)
            result['success'] = True
            logger.info(f"    ✓ Saved: {output_path.name}")
        except Exception as e:
            logger.error(f"    I2I editing failed: {e}")
            result['success'] = False
            result['error'] = str(e)

        return result


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run OURS full pipeline experiment")
    parser.add_argument('--model', type=str, default='qwen',
                        choices=['qwen', 'flux2', 'flux', 'sd35'],
                        help='I2I model to use (default: qwen)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode: only process 8 representative samples')
    parser.add_argument('--test', type=str, default=None,
                        help='Test single image by filename (e.g., flux_china_art_painting_modern.png)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory (default: base_experimental/ours_{model})')
    args = parser.parse_args()

    # Quick test samples (8 total: 4 Korea + 4 China)
    QUICK_TEST_FILES = [
        'flux_korea_food_main_dish_traditional.png',
        'flux_korea_fashion_clothing_traditional.png',
        'flux_korea_architecture_landmark_traditional.png',
        'flux_korea_event_wedding_traditional.png',
        'flux_china_food_main_dish_general.png',
        'flux_china_fashion_clothing_general.png',
        'flux_china_architecture_house_traditional.png',
        'flux_china_event_wedding_traditional.png',
    ]

    i2i_model = args.model
    model_names = {
        'qwen': 'Qwen-Image-Edit-2509',
        'flux2': 'FLUX.2-dev',
        'flux': 'FLUX.1 ControlNet',
        'sd35': 'SD 3.5 Medium'
    }

    logger.info("=" * 70)
    logger.info("OURS FULL PIPELINE EXPERIMENT")
    logger.info(f"I2I Model: {model_names.get(i2i_model, i2i_model)}")
    logger.info("=" * 70)

    # Create model-specific output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = BASE_DIR / f"ours_{i2i_model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all prompts
    prompts = load_all_prompts()

    # Filter for test modes
    if args.test:
        # Single image test mode
        prompts = [p for p in prompts if p['output_name'] == args.test]
        if not prompts:
            logger.error(f"Image not found: {args.test}")
            logger.info("Available images:")
            all_prompts = load_all_prompts()
            for p in all_prompts[:10]:
                logger.info(f"  - {p['output_name']}")
            logger.info(f"  ... and {len(all_prompts) - 10} more")
            return
        logger.info(f"\n🧪 SINGLE TEST MODE: {args.test}")
        output_dir = BASE_DIR / "single_test"
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.quick:
        prompts = [p for p in prompts if p['output_name'] in QUICK_TEST_FILES]
        logger.info(f"\n🚀 QUICK TEST MODE: {len(prompts)} samples")
        output_dir = BASE_DIR / f"quick_test_{i2i_model}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"\nTotal images to process: {len(prompts)}")

    # Initialize pipeline with selected model
    pipeline = FullPipelineRunner(i2i_model=i2i_model)
    pipeline.initialize()

    # Results tracking
    results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    # CSV logging setup
    csv_path = output_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'index', 'filename', 'country', 'category',
        'original_prompt', 'adapted_prompt',
        'vlm_issues_count', 'vlm_issues_detail',
        'clip_reference', 'clip_similarity',
        'text_kb_used', 'success', 'error'
    ])

    start_time = time.time()

    for idx, item in enumerate(prompts):
        output_path = output_dir / item['output_name']

        # Skip if already exists
        if output_path.exists():
            logger.info(f"\n[{idx+1}/{len(prompts)}] SKIP (exists): {item['output_name']}")
            skip_count += 1
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"[{idx+1}/{len(prompts)}] {item['output_name']}")
        logger.info(f"{'='*70}")
        logger.info(f"  Country: {item['country']}")
        logger.info(f"  I2I Prompt: {item['prompt']}")

        try:
            # Resolve input path
            input_path = resolve_image_path(item['base_path'])
            if not input_path:
                logger.error(f"  ERROR: Image not found: {item['base_path']}")
                error_count += 1
                continue

            # Extract category
            category = extract_category(item['output_name'])

            # Process through full pipeline
            result = pipeline.process_image(
                input_path=input_path,
                i2i_prompt=item['prompt'],
                country=item['country'],
                category=category,
                output_path=output_path
            )

            results.append(result)

            # Write to CSV
            csv_writer.writerow([
                idx + 1,
                item['output_name'],
                item['country'],
                category,
                item['prompt'],
                result.get('adapted_prompt', ''),
                result.get('initial_issues', 0),
                '; '.join([i.get('description', '')[:100] for i in result.get('issues', [])]),
                result.get('selected_reference', ''),
                result.get('clip_results', [{}])[0].get('score', '') if result.get('clip_results') else '',
                'yes' if item['country'] in pipeline.text_kbs else 'no',
                'yes' if result.get('success') else 'no',
                result.get('error', '')
            ])
            csv_file.flush()  # Ensure data is written immediately

            if result.get('success'):
                success_count += 1
            else:
                error_count += 1

            # Clear GPU memory periodically
            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            # Still log to CSV on error
            csv_writer.writerow([
                idx + 1, item['output_name'], item['country'], '', item['prompt'],
                '', 0, '', '', '', 'no', 'no', str(e)
            ])
            csv_file.flush()

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total: {len(prompts)} | Success: {success_count} | Skipped: {skip_count} | Errors: {error_count}")
    logger.info(f"Time: {elapsed/60:.1f} minutes ({elapsed/max(success_count,1):.1f}s per image)")
    logger.info(f"Output folder: {output_dir}")

    # Save detailed results
    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': f'ours_full_pipeline_{i2i_model}',
            'i2i_model': i2i_model,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': len(prompts),
                'success': success_count,
                'skipped': skip_count,
                'errors': error_count,
                'elapsed_minutes': elapsed / 60
            },
            'results': results
        }, f, indent=2)
    logger.info(f"Results JSON: {results_path}")

    # Close CSV file
    csv_file.close()
    logger.info(f"Pipeline log CSV: {csv_path}")


if __name__ == "__main__":
    main()
