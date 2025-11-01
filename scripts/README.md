# Scripts Directory

All scripts are organized by workflow stage for easy navigation.

## 📁 Directory Structure

```
scripts/
├── 01_setup/              # Initial setup (run once)
├── 02_data_processing/    # Data enhancement & knowledge extraction
├── 03_indexing/           # Building search indices
├── 04_testing/            # Testing & evaluation
├── 05_utils/              # Utility scripts
└── run_complete_pipeline.sh  # Automated full pipeline
```

---

## 🚀 Recommended Workflow

### For First-Time Users

```bash
# Step 1: Initialize everything (automatic)
python scripts/04_testing/test_model_agnostic_editing.py

# The script will detect missing data and offer to initialize.
# It will automatically run all necessary setup steps.
```

### For Advanced Users

Run scripts manually in order:

#### 1️⃣ Setup (One-time)
```bash
# Initialize complete dataset
python scripts/01_setup/init_dataset.py --country korea
```

#### 2️⃣ Data Processing
```bash
# Enhance captions with VLM
python scripts/02_data_processing/enhance_captions.py

# Extract cultural knowledge from images
python scripts/02_data_processing/extract_cultural_knowledge.py --load-in-4bit
```

#### 3️⃣ Build Indices
```bash
# Integrate knowledge to RAG
python scripts/03_indexing/integrate_knowledge_to_rag.py

# Build CLIP image index
python scripts/03_indexing/build_clip_image_index.py

# Build text RAG index
python scripts/03_indexing/build_country_pack_index.py
```

#### 4️⃣ Test & Evaluate
```bash
# Interactive testing (main interface)
python scripts/04_testing/test_model_agnostic_editing.py

# Test VLM detector only
python scripts/04_testing/test_vlm_detector.py
```

---

## 📋 Script Details

### 01_setup/

#### **init_dataset.py** ⭐ PRIMARY ENTRY POINT
Complete one-time initialization. Runs all necessary steps automatically.

**Usage:**
```bash
python scripts/01_setup/init_dataset.py --country korea
```

**What it does:**
1. Converts contributions.csv to dataset JSON
2. Downloads images from Firebase
3. Enhances captions with VLM
4. Extracts cultural knowledge
5. Builds all RAG indices

---

### 02_data_processing/

#### **enhance_captions.py**
Enhance SNS captions using VLM for better descriptions.

**Usage:**
```bash
python scripts/02_data_processing/enhance_captions.py \
  --input data/country_packs/korea/approved_dataset.json \
  --output data/country_packs/korea/approved_dataset_enhanced.json \
  --load-in-4bit
```

#### **extract_cultural_knowledge.py**
Extract structured cultural knowledge from verified images.

**Usage:**
```bash
# Test with 5 images
python scripts/02_data_processing/extract_cultural_knowledge.py \
  --max-images 5 --load-in-4bit

# Full dataset
python scripts/02_data_processing/extract_cultural_knowledge.py --load-in-4bit
```

**Output:** Structured JSON with visual features, cultural elements, correct aspects, and common mistakes.

---

### 03_indexing/

#### **integrate_knowledge_to_rag.py**
Integrate extracted knowledge into FAISS text index.

**Usage:**
```bash
python scripts/03_indexing/integrate_knowledge_to_rag.py \
  --knowledge-file data/cultural_knowledge/korea_knowledge.json \
  --index-dir data/cultural_index/korea
```

#### **build_clip_image_index.py**
Build CLIP image similarity index for reference image retrieval.

**Usage:**
```bash
python scripts/03_indexing/build_clip_image_index.py \
  --data-dir data/country_packs/korea \
  --output-dir data/clip_index/korea
```

#### **build_country_pack_index.py**
Build text RAG index from Wikipedia + captions.

**Usage:**
```bash
python scripts/03_indexing/build_country_pack_index.py \
  --country korea
```

---

### 04_testing/

#### **test_model_agnostic_editing.py** ⭐ MAIN INTERFACE
Interactive CLI for testing the complete T2I → Detection → I2I workflow.

**Usage:**
```bash
# Interactive mode (recommended)
python scripts/04_testing/test_model_agnostic_editing.py

# Command-line mode
python scripts/04_testing/test_model_agnostic_editing.py \
  --prompt "A Korean woman in traditional hanbok" \
  --model qwen \
  --country korea
```

**Features:**
- Interactive configuration wizard
- Automatic initialization for first-time users
- Support for multiple models (Qwen, SDXL, FLUX)
- Model-specific prompt optimization

#### **test_vlm_detector.py**
Test VLM cultural detection on existing images.

**Usage:**
```bash
python scripts/04_testing/test_vlm_detector.py \
  --image-path path/to/image.jpg \
  --country korea
```

---

### 05_utils/

#### **download_images.py**
Generic image downloader from URLs.

#### **download_country_images.py**
Download country pack images from Firebase.

**Usage:**
```bash
python scripts/05_utils/download_country_images.py \
  --country korea \
  --output-dir ~/ccub2-agent-data/country_packs/korea/images
```

---

## 🔧 Automated Pipeline

### **run_complete_pipeline.sh**
Automated pipeline: Extract → Integrate → Test

**Usage:**
```bash
# Test mode (5 images)
bash scripts/run_complete_pipeline.sh test

# Full mode (all images)
bash scripts/run_complete_pipeline.sh full
```

---

## 💡 Quick Reference

| Task | Script |
|------|--------|
| **First-time setup** | `01_setup/init_dataset.py` |
| **Interactive testing** | `04_testing/test_model_agnostic_editing.py` |
| **Extract knowledge** | `02_data_processing/extract_cultural_knowledge.py` |
| **Build indices** | `03_indexing/integrate_knowledge_to_rag.py` |
| **Full pipeline** | `run_complete_pipeline.sh` |

---

## ℹ️ About ccub2_agent Package

The `ccub2_agent/` directory contains the core Python library that all scripts use:

```
ccub2_agent/
├── modules/     # VLM detector, CLIP RAG, prompt adapter, etc.
├── models/      # Universal I2I interface, model wrappers
├── pipelines/   # Iterative editing pipeline
└── adapters/    # Image editing adapters
```

Scripts import from this package:
```python
from ccub2_agent.modules import VLMCulturalDetector
from ccub2_agent.models import UniversalI2IInterface
```
