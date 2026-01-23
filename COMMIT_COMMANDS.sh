#!/bin/bash
# Git commit commands for logical grouping
# Execute each section separately
# NOTE: Korean markdown files are excluded from commits

set -e

echo "=========================================="
echo "Git Commit Plan Execution"
echo "=========================================="
echo ""
echo "⚠️  Note: Korean markdown files will be excluded"
echo ""

# 1. Core Architecture Refactoring
echo "1. Committing Core Architecture Refactoring..."
git add ccub2_agent/__init__.py
git add ccub2_agent/detection/
git add ccub2_agent/retrieval/
git add ccub2_agent/adaptation/
git add ccub2_agent/editing/
git add ccub2_agent/modules/
git add ccub2_agent/adapters/
git add ccub2_agent/pipelines/
git commit -m "refactor: restructure core modules into logical categories (detection, retrieval, adaptation, editing)"
echo "✓ Done"
echo ""

# 2. Agents Reorganization
echo "2. Committing Agents Reorganization..."
# Add agents directory but exclude Korean markdown files
git add ccub2_agent/agents/*.py
git add ccub2_agent/agents/**/*.py
git add ccub2_agent/agents/**/__init__.py
# Explicitly exclude Korean markdown files
find ccub2_agent/agents -name "*.md" -exec grep -l "[가-힣]" {} \; 2>/dev/null | xargs -r git reset HEAD 2>/dev/null || true
git commit -m "refactor: reorganize agents into logical sub-packages (core, evaluation, data, governance)"
echo "✓ Done"
echo ""

# 3. Evaluation System Restructure
echo "3. Committing Evaluation System Restructure..."
git add metric/
git add ccub2_agent/evaluation/
# Exclude Korean markdown files
find ccub2_agent/evaluation -name "*.md" -exec grep -l "[가-힣]" {} \; 2>/dev/null | xargs -r git reset HEAD 2>/dev/null || true
git commit -m "refactor: move metrics to evaluation/metrics/ directory"
echo "✓ Done"
echo ""

# 4. Data Layer Implementation
echo "4. Committing Data Layer Implementation..."
git add ccub2_agent/data/
git commit -m "feat: add data management layer with gap analysis components"
echo "✓ Done"
echo ""

# 5. NeurIPS Submission Layers
echo "5. Committing NeurIPS Submission Layers..."
git add ccub2_agent/schemas/
git add ccub2_agent/orchestration/
git add ccub2_agent/reproducibility/*.py
git add ccub2_agent/reproducibility/**/*.py
git add ccub2_agent/reproducibility/**/__init__.py
git add ccub2_agent/models/model_registry.py
git add ccub2_agent/tests/
# Exclude Korean markdown files
find ccub2_agent/reproducibility -name "*.md" -exec grep -l "[가-힣]" {} \; 2>/dev/null | xargs -r git reset HEAD 2>/dev/null || true
git commit -m "feat: add NeurIPS D&B submission layers (schemas, orchestration, reproducibility, model registry, tests)"
echo "✓ Done"
echo ""

# 6. Scripts Reorganization
echo "6. Committing Scripts Reorganization..."
git add scripts/**/*.py
git add scripts/**/*.sh
# Only add translated markdown files
git add scripts/STRUCTURE.md scripts/MIGRATION.md scripts/STRUCTURE_FINAL.md 2>/dev/null || true
# Exclude Korean markdown files
find scripts -name "*.md" -exec grep -l "[가-힣]" {} \; 2>/dev/null | xargs -r git reset HEAD 2>/dev/null || true
git commit -m "refactor: reorganize scripts into functional categories (setup, data_processing, indexing, testing, experiments, utils, pipelines, analysis)"
echo "✓ Done"
echo ""

# 7. Documentation Translation (only translated files)
echo "7. Committing Translated Documentation..."
# Only add translated markdown files
git add ccub2_agent/docs/README.md 2>/dev/null || true
git add ccub2_agent/docs/STRUCTURE.md 2>/dev/null || true
git add COMMIT_PLAN.md COMMIT_COMMANDS.sh TRANSLATION_STATUS.md 2>/dev/null || true
# Exclude Korean markdown files from docs
find ccub2_agent/docs -name "*.md" -exec grep -l "[가-힣]" {} \; 2>/dev/null | xargs -r git reset HEAD 2>/dev/null || true
git commit -m "docs: translate core documentation to English (partial - Korean files excluded for later translation)" || echo "  (No changes to commit)"
echo "✓ Done"
echo ""

# 8. Legacy Cleanup Documentation
echo "8. Committing Legacy Cleanup Documentation..."
# Only add if not Korean
if ! grep -q "[가-힣]" LEGACY_CLEANUP_SUMMARY.md 2>/dev/null; then
    git add LEGACY_CLEANUP_SUMMARY.md 2>/dev/null || true
fi
if ! grep -q "[가-힣]" LEGACY_CLEANUP.md 2>/dev/null; then
    git add LEGACY_CLEANUP.md 2>/dev/null || true
fi
git commit -m "docs: add legacy cleanup documentation" || echo "  (No changes to commit)"
echo "✓ Done"
echo ""

echo "=========================================="
echo "All commits completed!"
echo "=========================================="
echo ""
echo "📝 Remaining (not committed):"
echo "  - Korean markdown files (~31 files)"
echo "  - These will be translated and committed separately"
echo ""
echo "Next steps:"
echo "  1. Review commits: git log --oneline -8"
echo "  2. Push: git push origin main"
echo "  3. Translate remaining Korean files in follow-up commits"
