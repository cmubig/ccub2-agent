#!/usr/bin/env python3
"""
Create comparison images: Original vs Edited with captions from log
"""

import csv
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

def wrap_text(text, width=80):
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))

def create_comparison_row(original_path, edited_path, caption_info, output_path):
    """Create a single comparison image with original and edited side by side."""

    # Load images
    try:
        original = Image.open(original_path).convert('RGB')
        edited = Image.open(edited_path).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return False

    # Resize to same height
    target_height = 512

    orig_ratio = original.width / original.height
    orig_new_width = int(target_height * orig_ratio)
    original = original.resize((orig_new_width, target_height), Image.Resampling.LANCZOS)

    edit_ratio = edited.width / edited.height
    edit_new_width = int(target_height * edit_ratio)
    edited = edited.resize((edit_new_width, target_height), Image.Resampling.LANCZOS)

    # Create canvas
    padding = 20
    caption_height = 200
    total_width = original.width + edited.width + padding * 3
    total_height = target_height + caption_height + padding * 2

    canvas = Image.new('RGB', (total_width, total_height), 'white')

    # Paste images
    canvas.paste(original, (padding, padding))
    canvas.paste(edited, (padding * 2 + original.width, padding))

    # Add text
    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Labels
    draw.text((padding, target_height + padding + 5), "ORIGINAL", fill='blue', font=font_large)
    draw.text((padding * 2 + original.width, target_height + padding + 5), "EDITED (Qwen)", fill='green', font=font_large)

    # Caption info
    y_offset = target_height + padding + 30

    # Filename and country
    info_text = f"File: {caption_info.get('filename', 'N/A')} | Country: {caption_info.get('country', 'N/A')} | Category: {caption_info.get('category', 'N/A')}"
    draw.text((padding, y_offset), info_text, fill='black', font=font_small)
    y_offset += 18

    # Original prompt (truncated)
    orig_prompt = caption_info.get('original_prompt', '')[:100]
    draw.text((padding, y_offset), f"Prompt: {orig_prompt}...", fill='gray', font=font_small)
    y_offset += 18

    # VLM issues (truncated)
    vlm_issues = caption_info.get('vlm_issues', '')[:150]
    draw.text((padding, y_offset), f"VLM: {vlm_issues}...", fill='darkred', font=font_small)
    y_offset += 18

    # CLIP reference
    clip_ref = caption_info.get('clip_reference', 'None')
    clip_sim = caption_info.get('clip_similarity', '')
    draw.text((padding, y_offset), f"CLIP Ref: {clip_ref} (sim: {clip_sim})", fill='purple', font=font_small)

    # Save
    canvas.save(output_path, quality=95)
    return True


def main():
    # Paths
    base_dir = Path("/home/chans/ccub2-agent/base_experimental")
    result_dir = base_dir / "final_improved_qwen_20251210_001134"
    output_dir = base_dir / "comparison_grid_qwen"
    output_dir.mkdir(exist_ok=True)

    # Find CSV log
    csv_files = list(result_dir.glob("pipeline_log_*.csv"))
    if not csv_files:
        print("No pipeline log found!")
        return

    csv_path = csv_files[0]
    print(f"Using log: {csv_path}")

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} entries")

    # Process each entry
    success_count = 0
    for idx, row in enumerate(rows):
        filename = row.get('filename', '')
        country = row.get('country', '')

        # Find original image
        if country == 'china':
            original_path = base_dir / "China" / filename
        elif country == 'korea':
            original_path = base_dir / "Korea" / filename
        else:
            continue

        edited_path = result_dir / filename

        if not original_path.exists():
            print(f"Original not found: {original_path}")
            continue
        if not edited_path.exists():
            print(f"Edited not found: {edited_path}")
            continue

        # Prepare caption info
        caption_info = {
            'filename': filename,
            'country': country,
            'category': row.get('category', ''),
            'original_prompt': row.get('original_prompt', ''),
            'vlm_issues': row.get('vlm_issues_detail', ''),
            'clip_reference': Path(row.get('clip_reference', '')).name if row.get('clip_reference') else 'None',
            'clip_similarity': row.get('clip_similarity', ''),
        }

        output_path = output_dir / f"compare_{idx+1:02d}_{filename}"

        if create_comparison_row(original_path, edited_path, caption_info, output_path):
            print(f"[{idx+1}/{len(rows)}] ✓ {filename}")
            success_count += 1
        else:
            print(f"[{idx+1}/{len(rows)}] ✗ {filename}")

    print(f"\nDone! {success_count}/{len(rows)} comparisons created")
    print(f"Output: {output_dir}")

    # Create zip
    import shutil
    zip_path = base_dir / "comparison_grid_qwen"
    shutil.make_archive(str(zip_path), 'zip', output_dir)
    print(f"Zip: {zip_path}.zip")


if __name__ == "__main__":
    main()
