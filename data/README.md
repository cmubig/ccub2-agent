# Data Directory

This directory contains the contributions data that will be processed during initialization.

## Files

- **_contributions.csv** - Raw contributions data from WorldCCUB app
  - Contains image URLs, captions, categories, and metadata
  - This file should be committed to the repository

## Generated Data (Not in Repository)

The following will be generated during initialization:

```
data/
├── country_packs/
│   └── korea/
│       ├── approved_dataset.json          # Processed contributions
│       ├── approved_dataset_enhanced.json # VLM-enhanced captions
│       └── images/                        # Downloaded images (338 files)
├── cultural_knowledge/
│   └── korea_knowledge.json               # Extracted cultural knowledge
├── cultural_index/
│   └── korea/                             # Text RAG index (FAISS)
└── clip_index/
    └── korea/                             # Image similarity index (CLIP)
```

All generated data is excluded from git (see `.gitignore`).

## Initialization

To generate the required data files, run:

```bash
python scripts/init_dataset.py --country korea
```

Or simply run the main script, which will offer to initialize automatically:

```bash
python scripts/test_model_agnostic_editing.py
```
