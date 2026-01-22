#!/usr/bin/env python3
"""
Create side-by-side comparison images (Before | After) with prompts
"""

import csv
import sys
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = PROJECT_ROOT / "base_experimental"
CHINA_DIR = BASE_DIR / "China"
KOREA_DIR = BASE_DIR / "Korea"
OURS_QWEN_DIR = BASE_DIR / "ours_qwen"
OUTPUT_DIR = BASE_DIR / "comparison_qwen"


def load_prompts_from_csv():
    """Load prompts from pipeline log CSV."""
    prompts = {}
    csv_files = list(OURS_QWEN_DIR.glob("pipeline_log_*.csv"))
    if not csv_files:
        return prompts

    csv_path = csv_files[0]  # Use first one
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            prompts[filename] = {
                'original': row.get('original_prompt', ''),
                'adapted': row.get('adapted_prompt', ''),
                'issues': row.get('vlm_issues_detail', ''),
                'reference': row.get('clip_reference', ''),
            }
    return prompts


def create_comparison(before_path: Path, after_path: Path, output_path: Path, prompt_info: dict = None):
    """Create side-by-side comparison image with prompts."""
    before = Image.open(before_path).convert('RGB')
    after = Image.open(after_path).convert('RGB')

    # Resize to same height
    target_height = 512

    before_ratio = target_height / before.height
    before = before.resize((int(before.width * before_ratio), target_height), Image.LANCZOS)

    after_ratio = target_height / after.height
    after = after.resize((int(after.width * after_ratio), target_height), Image.LANCZOS)

    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Create combined image with labels and prompt area
    label_height = 40
    prompt_height = 200  # Increased space for full prompts at bottom
    gap = 20
    total_width = before.width + gap + after.width
    total_height = target_height + label_height + prompt_height

    combined = Image.new('RGB', (total_width, total_height), color='white')

    # Paste images
    combined.paste(before, (0, label_height))
    combined.paste(after, (before.width + gap, label_height))

    # Add labels
    draw = ImageDraw.Draw(combined)

    # Before label
    before_text = "BEFORE (Bias)"
    draw.text((before.width // 2, 10), before_text, fill='red', font=font_title, anchor='mt')

    # After label
    after_text = "AFTER (Ours-Qwen)"
    draw.text((before.width + gap + after.width // 2, 10), after_text, fill='green', font=font_title, anchor='mt')

    # Add prompt info at bottom
    if prompt_info:
        prompt_y = label_height + target_height + 10

        # Original prompt
        original = prompt_info.get('original', 'N/A')
        wrapped_original = textwrap.fill(f"Input Prompt: {original}", width=80)
        draw.text((10, prompt_y), wrapped_original, fill='black', font=font_text)

        # Adapted prompt (show more text with better wrapping)
        adapted = prompt_info.get('adapted', 'N/A')
        # Truncate at 500 chars if too long, but show much more than before
        if len(adapted) > 500:
            adapted = adapted[:500] + "..."
        wrapped_adapted = textwrap.fill(f"Adapted Prompt: {adapted}", width=100)
        draw.text((10, prompt_y + 35), wrapped_adapted, fill='#0066cc', font=font_small)

        # Reference image if available
        ref = prompt_info.get('reference', '')
        if ref:
            draw.text((10, prompt_y + 150), f"Reference: {ref}", fill='#666666', font=font_small)

    combined.save(output_path)
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompt info from CSV
    prompts = load_prompts_from_csv()
    print(f"Loaded prompts for {len(prompts)} images")

    # Get all output images
    output_images = list(OURS_QWEN_DIR.glob("*.png"))
    print(f"Found {len(output_images)} output images")

    success = 0
    for after_path in output_images:
        filename = after_path.name

        # Find corresponding before image
        if 'china' in filename.lower():
            before_path = CHINA_DIR / filename
        elif 'korea' in filename.lower():
            before_path = KOREA_DIR / filename
        else:
            print(f"Unknown country: {filename}")
            continue

        if not before_path.exists():
            print(f"Before not found: {before_path}")
            continue

        output_path = OUTPUT_DIR / f"compare_{filename}"

        # Get prompt info for this image
        prompt_info = prompts.get(filename, {})

        try:
            create_comparison(before_path, after_path, output_path, prompt_info)
            print(f"✓ {filename}")
            success += 1
        except Exception as e:
            print(f"✗ {filename}: {e}")

    print(f"\nDone! {success} comparison images saved to {OUTPUT_DIR}")

    # Create zip
    import shutil
    zip_path = BASE_DIR / "comparison_qwen"
    shutil.make_archive(str(zip_path), 'zip', OUTPUT_DIR)
    print(f"Zip created: {zip_path}.zip")


if __name__ == "__main__":
    main()
