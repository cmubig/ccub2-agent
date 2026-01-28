# Scripts Directory

All scripts are organized by workflow stage and purpose for easy navigation.

## 📁 Directory Structure

```
scripts/
├── setup/                    # Initial setup (one-time)
│   ├── init_dataset.py              # ⭐ Complete initialization
│   ├── batch_init_countries.py      # Batch initialization
│   ├── create_country_datasets.py   # Dataset creation
│   └── detect_available_countries.py # Country detection
│
├── data_processing/          # Data enhancement & knowledge extraction
│   ├── enhance_captions.py          # Caption enhancement
│   ├── batch_enhance_captions.py    # Batch caption enhancement
│   └── extract_cultural_knowledge.py # Cultural knowledge extraction
│
├── indexing/                 # Building search indices
│   ├── build_clip_image_index.py    # CLIP image index
│   ├── build_country_pack_index.py  # Text RAG index
│   └── integrate_knowledge_to_rag.py # Knowledge integration
│
├── testing/                  # Testing & evaluation
│   ├── test_model_agnostic_editing.py # ⭐ Main interactive interface
│   ├── test_vlm_detector.py          # VLM detector test
│   └── test_single_image.py          # Single image test
│
├── experiments/              # Experimental scripts
│   ├── run_ours_experiment.py        # Single experiment
│   ├── run_ours_batch.py             # Batch experiments
│   ├── run_ours_full_pipeline.py    # Full experimental pipeline
│   ├── run_quick_test.py             # Quick test
│   └── create_comparison_images.py   # Comparison grid creation
│
├── pipelines/                # Automated pipelines
│   ├── complete_pipeline.py          # Single country pipeline
│   ├── complete_pipeline_all_countries.py # All countries pipeline
│   ├── stable_extract_all_countries.py    # Stable extraction
│   ├── build_all_country_indices.py        # Build all indices
│   ├── parallel_extract_knowledge.py      # Parallel extraction
│   ├── run_complete_pipeline.sh            # Shell script
│   ├── quick_build_all.sh                  # Quick build
│   ├── quick_examples.sh                   # Quick examples
│   └── BUILD_INDICES_README.md             # Index build guide
│
├── utils/                    # Utility scripts
│   ├── download_images.py                 # Image downloader
│   ├── download_country_images.py          # Country image download
│   ├── batch_download_images.py           # Batch download
│   ├── test_firebase_connection.py         # Firebase test
│   ├── test_job_creation_flow.py            # Job creation test
│   └── test_multi_country_support.py        # Multi-country test
│
└── analysis/                 # Analysis & inspection
    ├── firebase_storage_analyzer.py        # Firebase Storage analysis
    └── create_comparison_grid.py           # Comparison grid
```

---

## 🚀 Quick Start

### For First-Time Users

```bash
# Interactive setup (recommended)
python scripts/testing/test_model_agnostic_editing.py

# The script will detect missing data and offer to initialize automatically.
```

### For Advanced Users

Run scripts in order:

#### 1️⃣ Setup (One-time)
```bash
# Initialize single country
python scripts/setup/init_dataset.py --country korea

# Initialize multiple countries
python scripts/setup/batch_init_countries.py
```

#### 2️⃣ Data Processing
```bash
# Enhance captions with VLM
python scripts/data_processing/enhance_captions.py

# Extract cultural knowledge
python scripts/data_processing/extract_cultural_knowledge.py --load-in-4bit
```

#### 3️⃣ Build Indices
```bash
# Build all indices for all countries
python scripts/pipelines/build_all_country_indices.py

# Or build individually
python scripts/indexing/build_clip_image_index.py --country korea
python scripts/indexing/build_country_pack_index.py --country korea
```

#### 4️⃣ Test & Evaluate
```bash
# Interactive testing (main interface)
python scripts/testing/test_model_agnostic_editing.py

# Test VLM detector only
python scripts/testing/test_vlm_detector.py
```

---

## 📋 Script Categories

### setup/ - Initial Setup
**Purpose**: One-time initialization for countries

| Script | Purpose |
|--------|---------|
| `init_dataset.py` | ⭐ Complete initialization for single country |
| `batch_init_countries.py` | Initialize multiple countries |
| `create_country_datasets.py` | Create country datasets from CSV |
| `detect_available_countries.py` | Detect available countries in data |

---

### data_processing/ - Data Enhancement
**Purpose**: Enhance and extract knowledge from data

| Script | Purpose |
|--------|---------|
| `enhance_captions.py` | Enhance captions with VLM |
| `batch_enhance_captions.py` | Batch caption enhancement |
| `extract_cultural_knowledge.py` | Extract cultural knowledge from images |

---

### indexing/ - Index Building
**Purpose**: Build search indices for RAG

| Script | Purpose |
|--------|---------|
| `build_clip_image_index.py` | Build CLIP image similarity index |
| `build_country_pack_index.py` | Build text RAG index |
| `integrate_knowledge_to_rag.py` | Integrate knowledge into FAISS |

---

### testing/ - Testing & Evaluation
**Purpose**: Test and evaluate the system

| Script | Purpose |
|--------|---------|
| `test_model_agnostic_editing.py` | ⭐ Main interactive interface |
| `test_vlm_detector.py` | Test VLM detector only |
| `test_single_image.py` | Test single image processing |

---

### experiments/ - Experimental Scripts
**Purpose**: Run experiments and benchmarks

| Script | Purpose |
|--------|---------|
| `run_ours_experiment.py` | Run single experiment |
| `run_ours_batch.py` | Run batch experiments |
| `run_ours_full_pipeline.py` | Run full experimental pipeline |
| `run_quick_test.py` | Quick test run |
| `create_comparison_images.py` | Create comparison grids |

---

### pipelines/ - Automated Pipelines
**Purpose**: Automated multi-step pipelines

| Script | Purpose |
|--------|---------|
| `complete_pipeline.py` | Complete pipeline for single country |
| `complete_pipeline_all_countries.py` | Pipeline for all countries |
| `stable_extract_all_countries.py` | Stable extraction for all countries |
| `build_all_country_indices.py` | Build all indices for all countries |
| `parallel_extract_knowledge.py` | Parallel knowledge extraction |
| `run_complete_pipeline.sh` | Shell script for full pipeline |
| `quick_build_all.sh` | Quick build script |
| `quick_examples.sh` | Quick examples script |

---

### utils/ - Utility Scripts
**Purpose**: Utility functions and helpers

| Script | Purpose |
|--------|---------|
| `download_images.py` | Generic image downloader |
| `download_country_images.py` | Download country pack images |
| `batch_download_images.py` | Batch image download |
| `test_firebase_connection.py` | Test Firebase connection |
| `test_job_creation_flow.py` | Test job creation |
| `test_multi_country_support.py` | Test multi-country support |

---

### analysis/ - Analysis & Inspection
**Purpose**: Analyze and inspect data

| Script | Purpose |
|--------|---------|
| `firebase_storage_analyzer.py` | Analyze Firebase Storage structure |
| `create_comparison_grid.py` | Create comparison image grids |

---

## 💡 Quick Reference

| Task | Script |
|------|--------|
| **First-time setup** | `setup/init_dataset.py` |
| **Interactive testing** | `testing/test_model_agnostic_editing.py` |
| **Extract knowledge** | `data_processing/extract_cultural_knowledge.py` |
| **Build all indices** | `pipelines/build_all_country_indices.py` |
| **Run experiment** | `experiments/run_ours_experiment.py` |
| **Analyze Firebase** | `analysis/firebase_storage_analyzer.py` |

---

## 📝 Notes

### Import Paths

All scripts import from the `ccub2_agent` package. Use new import paths:

```python
# New paths (use these)
from ccub2_agent.detection import VLMCulturalDetector
from ccub2_agent.retrieval import CLIPImageRAG
from ccub2_agent.adaptation import UniversalPromptAdapter
from ccub2_agent.editing import ImageEditingAdapter
from ccub2_agent.data import CountryDataPack, FirebaseClient
```

### Script Organization

- **setup/**: One-time initialization
- **data_processing/**: Data enhancement
- **indexing/**: Index building
- **testing/**: Testing and evaluation
- **experiments/**: Experimental runs
- **pipelines/**: Automated pipelines
- **utils/**: Utility functions
- **analysis/**: Analysis and inspection

---

## 🎯 Best Practices

1. **Use interactive mode** for first-time setup
2. **Build indices** after data processing
3. **Test before experiments** using testing scripts
4. **Use pipelines** for batch operations
5. **Check analysis scripts** for data inspection
