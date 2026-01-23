# Git Commit Plan

## Overview
This document outlines the logical grouping of changes for separate commits. All Korean text has been translated to English for the global repository.

## Commit Groups

### 1. Core Architecture Refactoring
**Purpose:** Restructure core modules into logical categories

**Files:**
- Deleted: `ccub2_agent/modules/*`, `ccub2_agent/adapters/*`, `ccub2_agent/pipelines/*`
- Added: `ccub2_agent/detection/`, `ccub2_agent/retrieval/`, `ccub2_agent/adaptation/`, `ccub2_agent/editing/`
- Modified: `ccub2_agent/__init__.py`

**Command:**
```bash
git add ccub2_agent/__init__.py
git add ccub2_agent/detection/
git add ccub2_agent/retrieval/
git add ccub2_agent/adaptation/
git add ccub2_agent/editing/
git add ccub2_agent/modules/
git add ccub2_agent/adapters/
git add ccub2_agent/pipelines/
git commit -m "refactor: restructure core modules into logical categories (detection, retrieval, adaptation, editing)"
```

### 2. Agents Reorganization
**Purpose:** Organize agents into sub-packages (core, evaluation, data, governance)

**Files:**
- Added: `ccub2_agent/agents/core/`, `ccub2_agent/agents/evaluation/`, `ccub2_agent/agents/data/`, `ccub2_agent/agents/governance/`

**Command:**
```bash
git add ccub2_agent/agents/
git commit -m "refactor: reorganize agents into logical sub-packages (core, evaluation, data, governance)"
```

### 3. Evaluation System Restructure
**Purpose:** Move metrics from root to evaluation/metrics/

**Files:**
- Deleted: `metric/cultural_metric/*`, `metric/general_metric/*`
- Added: `ccub2_agent/evaluation/metrics/`

**Command:**
```bash
git add metric/
git add ccub2_agent/evaluation/
git commit -m "refactor: move metrics to evaluation/metrics/ directory"
```

### 4. Data Layer Implementation
**Purpose:** Add data management layer with gap analysis

**Files:**
- Added: `ccub2_agent/data/`, `ccub2_agent/data/gap_analysis/`

**Command:**
```bash
git add ccub2_agent/data/
git commit -m "feat: add data management layer with gap analysis components"
```

### 5. NeurIPS Submission Layers
**Purpose:** Add 8 architectural layers for NeurIPS D&B submission

**Files:**
- Added: `ccub2_agent/schemas/`, `ccub2_agent/orchestration/`, `ccub2_agent/reproducibility/`, `ccub2_agent/models/model_registry.py`, `ccub2_agent/tests/`

**Command:**
```bash
git add ccub2_agent/schemas/
git add ccub2_agent/orchestration/
git add ccub2_agent/reproducibility/
git add ccub2_agent/models/model_registry.py
git add ccub2_agent/tests/
git commit -m "feat: add NeurIPS D&B submission layers (schemas, orchestration, reproducibility, model registry, tests)"
```

### 6. Scripts Reorganization
**Purpose:** Reorganize scripts from numbered directories to functional categories

**Files:**
- Deleted: `scripts/01_setup/*`, `scripts/02_data_processing/*`, `scripts/03_indexing/*`, `scripts/04_testing/*`, `scripts/05_experiments/*`, `scripts/05_utils/*`, root scripts
- Added: `scripts/setup/`, `scripts/data_processing/`, `scripts/indexing/`, `scripts/testing/`, `scripts/experiments/`, `scripts/utils/`, `scripts/pipelines/`, `scripts/analysis/`
- Modified: `scripts/README.md`

**Command:**
```bash
git add scripts/
git commit -m "refactor: reorganize scripts into functional categories (setup, data_processing, indexing, testing, experiments, utils, pipelines, analysis)"
```

### 7. Documentation Translation
**Purpose:** Translate all Korean documentation to English

**Files:**
- Modified: All `.md` files in `ccub2_agent/docs/`, `ccub2_agent/agents/`, `scripts/`

**Command:**
```bash
git add ccub2_agent/docs/
git add ccub2_agent/agents/*.md
git add scripts/*.md
git commit -m "docs: translate all Korean documentation to English for global repository"
```

### 8. Legacy Cleanup Documentation
**Purpose:** Add cleanup documentation

**Files:**
- Added: `LEGACY_CLEANUP.md`, `LEGACY_CLEANUP_SUMMARY.md` (translated to English)

**Command:**
```bash
git add LEGACY_CLEANUP.md LEGACY_CLEANUP_SUMMARY.md
git commit -m "docs: add legacy cleanup documentation"
```

## Execution Order

Execute commits in the order listed above to maintain logical dependencies.

## Notes

- All Korean text has been translated to English
- Python comments and docstrings are in English
- All markdown documentation is in English
- Commit messages follow conventional commits format
