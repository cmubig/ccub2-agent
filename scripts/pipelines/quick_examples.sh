#!/bin/bash
###############################################################################
# Quick Examples for Building Country Indices
#
# Copy and use these commands!
###############################################################################

# Do not execute this script! Just copy and paste the commands you need.
echo "This file contains examples only. Copy and paste the commands you need."
exit 1

###############################################################################
# Example 1: Process all countries (fully automatic)
###############################################################################
conda run -n ccub2 python scripts/build_all_country_indices.py

###############################################################################
# Example 2: Specific countries only (Korea, China, Japan)
###############################################################################
conda run -n ccub2 python scripts/build_all_country_indices.py --countries korea china japan

###############################################################################
# Example 3: Dry run (check without actual execution)
###############################################################################
conda run -n ccub2 python scripts/build_all_country_indices.py --dry-run

###############################################################################
# Example 4: Use existing knowledge (build only CLIP and RAG)
###############################################################################
conda run -n ccub2 python scripts/build_all_country_indices.py --skip-knowledge

###############################################################################
# Example 5: Extract knowledge only (skip CLIP and RAG)
###############################################################################
conda run -n ccub2 python scripts/build_all_country_indices.py --skip-clip --skip-rag

###############################################################################
# Example 6: Force rebuild (regenerate even if exists)
###############################################################################
conda run -n ccub2 python scripts/build_all_country_indices.py --force

###############################################################################
# Example 7: Run in background + save logs
###############################################################################
nohup conda run -n ccub2 python scripts/build_all_country_indices.py \
  2>&1 | tee /tmp/build_all_$(date +%Y%m%d_%H%M%S).log &

# Check progress
tail -f /tmp/build_all_*.log

###############################################################################
# Example 8: Using shell script (convenient)
###############################################################################
# All countries
./scripts/quick_build_all.sh

# Specific countries only
./scripts/quick_build_all.sh --countries korea japan

# Dry run
./scripts/quick_build_all.sh --dry-run

###############################################################################
# Example 9: Step-by-step safe processing
###############################################################################
# Step 1: Dry run first
python scripts/build_all_country_indices.py --dry-run

# Step 2: Test with Korea
python scripts/build_all_country_indices.py --countries korea

# Step 3: Run all if no issues
python scripts/build_all_country_indices.py

###############################################################################
# Example 10: Process one country at a time (save memory)
###############################################################################
for country in korea china japan usa; do
    echo "Processing $country..."
    conda run -n ccub2 python scripts/build_all_country_indices.py \
        --countries $country
done

###############################################################################
# Recommended workflow
###############################################################################

# 👉 Good preparation: Check with dry-run first
python scripts/build_all_country_indices.py --dry-run

# 👉 Quick test: Process Korea only
python scripts/build_all_country_indices.py --countries korea

# 👉 Run overnight: Full automatic processing (save logs)
nohup conda run -n ccub2 python scripts/build_all_country_indices.py \
  2>&1 | tee /tmp/build_all_$(date +%Y%m%d_%H%M%S).log &
