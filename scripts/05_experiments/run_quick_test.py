#!/usr/bin/env python3
"""
Quick test script - 8 representative samples only
Korea: 4, China: 4
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_ours_full_pipeline import FullPipelineRunner

# Representative samples (8 total)
TEST_SAMPLES = [
    # Korea (4)
    ("korea", "flux_korea_food_main_dish_traditional.png", "Change the image to represent traditional main dish in Korea."),
    ("korea", "flux_korea_fashion_clothing_traditional.png", "Change the image to represent traditional clothing in Korea."),
    ("korea", "flux_korea_architecture_landmark_traditional.png", "Change the image to represent traditional landmark in Korea."),
    ("korea", "flux_korea_event_wedding_traditional.png", "Change the image to represent traditional wedding in Korea."),
    # China (4)
    ("china", "flux_china_food_main_dish_general.png", "Change the image to represent main dish in China."),
    ("china", "flux_china_fashion_clothing_general.png", "Change the image to represent clothing in China."),
    ("china", "flux_china_architecture_house_traditional.png", "Change the image to represent traditional house in China."),
    ("china", "flux_china_event_wedding_traditional.png", "Change the image to represent traditional wedding in China."),
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "flux2"], default="qwen")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Setup paths
    base_dir = PROJECT_ROOT / "base_experimental"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / f"quick_test_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"QUICK TEST - {args.model.upper()} - 8 samples")
    print(f"=" * 60)

    # Initialize pipeline
    pipeline = FullPipelineRunner(i2i_model=args.model)

    # Process samples
    results = []
    for idx, (country, filename, prompt) in enumerate(TEST_SAMPLES):
        print(f"\n[{idx+1}/8] {filename}")

        # Find input image
        input_path = base_dir / country.capitalize() / filename
        if not input_path.exists():
            print(f"  ⚠ Not found: {input_path}")
            continue

        output_path = output_dir / filename

        # Extract category from filename
        parts = filename.replace(".png", "").split("_")
        category = "_".join(parts[2:-1]) if len(parts) > 3 else "general"

        try:
            result = pipeline.process_image(
                input_path=input_path,
                output_path=output_path,
                country=country,
                category=category,
                i2i_prompt=prompt
            )
            result['filename'] = filename
            results.append(result)
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Save results to CSV
    import csv
    from datetime import datetime

    csv_path = output_dir / f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'country', 'category', 'original_prompt', 'adapted_prompt', 'clip_reference', 'success'])
        for r in results:
            writer.writerow([
                r.get('filename', ''),
                r.get('country', ''),
                r.get('category', ''),
                r.get('i2i_prompt', ''),
                r.get('adapted_prompt', ''),
                r.get('selected_reference', ''),
                'yes' if r.get('success') else 'no'
            ])

    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} images saved to {output_dir}")
    print(f"CSV: {csv_path}")

    # Create comparison images
    print(f"\nCreating comparison images...")
    from create_comparison_images import create_comparison, load_prompts_from_csv

    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)

    for r in results:
        filename = r.get('filename')
        country = r.get('country')

        before_path = base_dir / country.capitalize() / filename
        after_path = output_dir / filename

        if before_path.exists() and after_path.exists():
            prompt_info = {
                'original': r.get('i2i_prompt', ''),
                'adapted': r.get('adapted_prompt', ''),
                'reference': r.get('selected_reference', ''),
            }
            output_compare = comparison_dir / f"compare_{filename}"
            try:
                create_comparison(before_path, after_path, output_compare, prompt_info)
                print(f"  ✓ {filename}")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")

    # Zip for download
    import shutil
    zip_path = output_dir / "comparisons"
    shutil.make_archive(str(output_dir / f"quick_test_{args.model}_comparisons"), 'zip', comparison_dir)
    print(f"\nZip: {output_dir}/quick_test_{args.model}_comparisons.zip")

if __name__ == "__main__":
    main()
