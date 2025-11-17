# Data Directory

This directory contains all project data, both committed and generated.

## Committed Files (in git)

- **_contributions.csv** - Raw contributions data from WorldCCUB app (607KB)
  - Fallback data source when Firebase Admin SDK is unavailable
  - Contains image URLs, captions, categories, and metadata

- **_jobs.csv** - Job listings for data collection (80KB)
  - Used by the agent to create new data collection jobs
  - Contains category, country, and target information

## Generated Data (excluded from git)

The following directories are auto-generated during initialization and excluded from git:

```
data/
├── country_packs/korea/
│   ├── approved_dataset.json          # Processed contributions (306KB)
│   ├── approved_dataset_enhanced.json # VLM-enhanced captions (459KB)
│   └── images/                        # Downloaded images (338 files, 696MB)
├── cultural_knowledge/
│   └── korea_knowledge.json           # Extracted cultural knowledge (949KB)
├── cultural_index/korea/
│   ├── faiss.index                    # Text RAG index (493KB)
│   └── metadata.jsonl
├── clip_index/korea/
│   ├── clip.index                     # Image similarity index (657KB)
│   └── clip_metadata.jsonl
└── hf_cache/                          # HuggingFace model cache (150GB)
```

All large files and generated data are excluded from git (see `.gitignore`).

## Initialization

All scripts use `PROJECT_ROOT/data/` by default. To initialize:

```bash
python scripts/01_setup/init_dataset.py --country korea
```

Or run any test script, which will offer to initialize automatically:

```bash
python scripts/04_testing/test_model_agnostic_editing.py
```
