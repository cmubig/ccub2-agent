# CCUB2 Agent

**Model-Agnostic Cultural Bias Mitigation System**

Automatically detect and correct cultural biases in generative image models using VLM-based evaluation, RAG-enhanced cultural knowledge, and **automatic model-specific prompt optimization**.

## 🎯 Key Innovation

**One Universal Instruction → 6+ Model-Optimized Prompts**

Our system automatically adapts editing instructions to each model's optimal format:
- FLUX Kontext: Context-preserving instructions
- Qwen Image Edit: Detailed, specific requirements
- Stable Diffusion 3.5: Structured with quality tags
- HiDream, NextStep, Custom models...

**Result**: Best performance from every model, zero manual tuning.

## 🚀 Quick Start

### Setup

```bash
# 1. Clone repository
[git clone https://github.com/YOUR_USERNAME/ccub2-agent.git](https://github.com/cmubig/ccub2-agent)
cd ccub2-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Firebase Setup (Required for data access)
# Download your Firebase service account key from:
# Firebase Console → Project Settings → Service Accounts → Generate new private key
# Save as: firebase-service-account.json

# Test Firebase connection
python scripts/05_utils/test_firebase_connection.py

# 4. Run the interactive workflow (first-time setup will be automatic!)
python scripts/04_testing/test_model_agnostic_editing.py
```

**📚 Need help with Firebase?** See [FIREBASE_SETUP.md](FIREBASE_SETUP.md) for detailed instructions.

**First-time users**: The script will detect missing data and offer to initialize automatically (takes ~2-5 hours).

**Existing users**: If you already have data, it will start immediately.

The interactive CLI will guide you through:
1. Select T2I model (SDXL or FLUX)
2. Select I2I model (Qwen, SDXL, FLUX, or test all)
3. Choose target country
4. Pick image category
5. Enter your prompt

### Command-Line Mode

```bash
# Direct execution with parameters
python scripts/04_testing/test_model_agnostic_editing.py \
  --prompt "A Korean woman in traditional hanbok" \
  --model qwen \
  --t2i-model sdxl \
  --country korea \
  --category traditional_clothing
```

## 📂 Project Structure

```
ccub2-agent/                    # Code repository
├── scripts/                    # Organized workflow scripts
│   ├── 01_setup/               # Initial setup
│   ├── 02_data_processing/     # Data enhancement
│   ├── 03_indexing/            # Build indices
│   ├── 04_testing/             # Main testing interface
│   └── 05_utils/               # Utilities
├── ccub2_agent/                # Core Python library
│   ├── modules/                # VLM, CLIP, RAG, prompt adapter
│   ├── models/                 # Universal I2I interface
│   ├── pipelines/              # Iterative editing
│   └── adapters/               # Image editing adapters
├── metric/                     # Cultural metric evaluation
├── docs/                       # Documentation
└── data/                       # Contributions CSV

~/ccub2-agent-data/             # Generated data (not in repo)
├── country_packs/korea/
│   ├── approved_dataset_enhanced.json    # VLM-enhanced
│   └── images/                           # 338 images
├── cultural_knowledge/         # Extracted knowledge
├── cultural_index/korea/       # RAG text index
└── clip_index/korea/           # CLIP image index
```

## 🎯 Current Status

- ✅ **Qwen3-VL-8B** upgraded across all files
- ✅ **328 images** with VLM-enhanced captions (100%)
- ✅ Project cleaned and organized
- ❌ **Next step**: Extract cultural knowledge from images

## 📝 Key Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `extract_cultural_knowledge.py` | Extract structured knowledge from verified images | `--max-images 5 --load-in-4bit` |
| `test_model_agnostic_editing.py` | Test full T2I→I2I pipeline with VLM | `--model qwen` |
| `test_vlm_detector.py` | Test VLM cultural detection | `--image-path <path>` |
| `build_clip_image_index.py` | Build CLIP FAISS index | `--data-dir <path>` |

## 💡 What This Does

### Problem
Current RAG uses only Wikipedia text → VLM lacks specific visual guidance → Gives perfect 5/5 scores to culturally incorrect images.

### Solution
Extract structured cultural knowledge directly from verified images:

```json
{
  "visual_features": "Two-piece garment: jeogori (short jacket) + chima (high-waisted skirt)...",
  "materials_textures": "Silk fabric with flowing layers, not tight-fitting...",
  "cultural_elements": "Authentic Korean hanbok with traditional proportions...",
  "correct_aspects": [
    "Proper jeogori length at hip level",
    "High-waisted chima from chest",
    "Curved seam lines on sleeves"
  ],
  "common_mistakes": "Avoid: Chinese collar, Japanese obi, Western corset, tight fit..."
}
```

### Impact
- **Before**: Wikipedia-only context → 30-40% detection accuracy
- **After**: Image-derived knowledge → 70-90% detection accuracy (expected)

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- Qwen3-VL-8B-Instruct
- GPU with 8GB+ VRAM (4-bit quantization)

```bash
pip install -r requirements.txt
```

## 📊 Data Pipeline

```
1. Images (338) + SNS Captions
   ↓
2. VLM Caption Enhancement (Qwen3-VL) ✅ 100%
   ↓
3. Cultural Knowledge Extraction ← NOW
   ↓
4. RAG Index Integration
   ↓
5. VLM Evaluation with Enhanced Context
```

## 📚 Documentation

- [Complete Process Guide](docs/FULL_PROCESS_GUIDE.md) - Detailed workflow explanation
- [Future Work & Architecture](docs/FUTURE_WORK.md) - Design decisions & roadmap

## 💾 Data Paths

All data paths can be configured via command-line arguments. Default structure:

| Data | Default Path |
|------|--------------|
| Images | `data/country_packs/korea/images/` |
| Enhanced captions | `data/country_packs/korea/approved_dataset_enhanced.json` |
| Output knowledge | `data/cultural_knowledge/korea_knowledge.json` |
| RAG index | `data/cultural_index/korea/` |

**Note**: Large data files are not included in the repository. Download separately or use your own dataset.

## 🐛 Troubleshooting

**Out of GPU memory?**
```bash
python scripts/extract_cultural_knowledge.py --load-in-4bit
```

**Resume from checkpoint?**
```bash
python scripts/extract_cultural_knowledge.py --resume
```

**Test before full run?**
```bash
python scripts/extract_cultural_knowledge.py --max-images 5
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was conducted at Carnegie Mellon University.
