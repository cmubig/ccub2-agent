#!/bin/bash
###############################################################################
# Quick Build All Country Indices
#
# Automatically build knowledge base, CLIP, and RAG indices for all countries.
#
# Usage:
#   ./scripts/quick_build_all.sh                    # All countries
#   ./scripts/quick_build_all.sh korea china        # Specific countries only
#   ./scripts/quick_build_all.sh --dry-run          # Dry run
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/build_all_indices_${TIMESTAMP}.log"

echo "================================================================================"
echo "Multi-Country Index Builder"
echo "================================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Log File:     $LOG_FILE"
echo ""

# If arguments provided, pass them; otherwise process all countries
if [ $# -eq 0 ]; then
    echo "Processing: ALL countries"
    echo ""
    conda run -n ccub2 python "$SCRIPT_DIR/build_all_country_indices.py" 2>&1 | tee "$LOG_FILE"
else
    echo "Arguments: $@"
    echo ""
    conda run -n ccub2 python "$SCRIPT_DIR/build_all_country_indices.py" "$@" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "================================================================================"
echo "Completed! Log saved to: $LOG_FILE"
echo "================================================================================"
