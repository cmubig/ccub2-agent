#!/bin/bash
# Complete CCUB2 Agent Pipeline
#
# This script runs the full pipeline:
# 1. Extract cultural knowledge from images
# 2. Integrate knowledge into RAG index
# 3. (Optional) Run test

set -e  # Exit on error

echo "=========================================================================="
echo "CCUB2 AGENT - COMPLETE PIPELINE"
echo "=========================================================================="
echo ""

# Configuration
COUNTRY="korea"
DATA_DIR="${DATA_DIR:-$HOME/ccub2-agent-data}"  # Override with env var if set
COUNTRY_PACK_DIR="$DATA_DIR/country_packs/$COUNTRY"
KNOWLEDGE_FILE="$DATA_DIR/cultural_knowledge/${COUNTRY}_knowledge.json"
INDEX_DIR="$DATA_DIR/cultural_index/$COUNTRY"

# Check if running in test mode
TEST_MODE=${1:-"full"}  # full | test
if [ "$TEST_MODE" = "test" ]; then
    MAX_IMAGES="--max-images 5"
    echo "🧪 Running in TEST mode (5 images only)"
else
    MAX_IMAGES=""
    echo "🚀 Running in FULL mode (all images)"
fi
echo ""

# Step 1: Extract Cultural Knowledge
echo "=========================================================================="
echo "STEP 1: Extract Cultural Knowledge from Images"
echo "=========================================================================="
echo "Input: $COUNTRY_PACK_DIR/approved_dataset_enhanced.json"
echo "Output: $KNOWLEDGE_FILE"
echo ""

python scripts/extract_cultural_knowledge.py \
    --data-dir "$COUNTRY_PACK_DIR" \
    --output "$KNOWLEDGE_FILE" \
    --model-name "Qwen/Qwen3-VL-8B-Instruct" \
    --load-in-4bit \
    $MAX_IMAGES \
    --resume

if [ $? -ne 0 ]; then
    echo "❌ Knowledge extraction failed!"
    exit 1
fi

echo ""
echo "✅ Step 1 complete!"
echo ""

# Step 2: Integrate into RAG
echo "=========================================================================="
echo "STEP 2: Integrate Knowledge into RAG Index"
echo "=========================================================================="
echo "Input: $KNOWLEDGE_FILE"
echo "Output: $INDEX_DIR/faiss.index"
echo ""

python scripts/integrate_knowledge_to_rag.py \
    --knowledge-file "$KNOWLEDGE_FILE" \
    --index-dir "$INDEX_DIR"

if [ $? -ne 0 ]; then
    echo "❌ RAG integration failed!"
    exit 1
fi

echo ""
echo "✅ Step 2 complete!"
echo ""

# Step 3: Summary
echo "=========================================================================="
echo "PIPELINE COMPLETE!"
echo "=========================================================================="
echo ""
echo "📊 Outputs:"
echo "  - Knowledge: $KNOWLEDGE_FILE"
echo "  - RAG Index: $INDEX_DIR/faiss.index"
echo "  - Metadata: $INDEX_DIR/metadata.jsonl"
echo ""
echo "🎯 Next Steps:"
echo "  1. Test VLM detection:"
echo "     python scripts/test_vlm_detector.py"
echo ""
echo "  2. Test full editing pipeline:"
echo "     python scripts/test_model_agnostic_editing.py --model qwen"
echo ""
echo "  3. Evaluate improvements:"
echo "     Compare detection accuracy before/after knowledge integration"
echo ""
