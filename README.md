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
git clone https://github.com/cmubig/ccub2-agent.git
cd ccub2-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Firebase Setup (Required)
# Contact chans@andrew.cmu.edu for credentials:
#   - firebase-service-account.json
#   - .firebase_config.json
# Save both files to project root

# 4. Test Firebase connection
python scripts/05_utils/test_firebase_connection.py

# 5. Initialize dataset (first-time: ~2-5 hours)
python scripts/01_setup/init_dataset.py --country korea

# 6. Run interactive workflow
python scripts/04_testing/test_model_agnostic_editing.py
```

**The interactive CLI guides you through:**
- T2I model selection (SDXL, FLUX)
- I2I model selection (Qwen, SDXL, FLUX, or test all)
- Country & category selection
- Prompt input
- Automatic cultural evaluation & refinement

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
│   ├── data_processing/     # Data enhancement
│   ├── indexing/            # Build indices
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

- ✅ **Firebase Direct Integration** - Real-time data access from Firestore
- ✅ **GPT-OSS-20B** - Upgraded question model for better cultural evaluation (20B params)
- ✅ **Qwen3-VL-8B** - Vision-Language Model for image analysis
- ✅ **Self-Improving System** - Automatic gap detection → job creation → retraining
- ✅ **Model-Agnostic I2I** - Universal interface for 6+ image editing models
- ✅ **575+ Cultural Images** - VLM-enhanced captions with cultural knowledge

## 📝 Key Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_firebase_connection.py` | Test Firebase connectivity and data access | No arguments needed |
| `init_dataset.py` | Initialize dataset from Firebase (auto-detects new data) | `--country korea` |
| `test_model_agnostic_editing.py` | Interactive T2I→I2I pipeline with cultural evaluation | `--prompt "text" --model qwen` |
| `extract_cultural_knowledge.py` | Extract structured knowledge from verified images | `--max-images 5 --load-in-4bit` |
| `test_vlm_detector.py` | Test VLM cultural bias detection | `--image-path <path>` |
| `build_clip_image_index.py` | Build CLIP FAISS index for reference images | `--data-dir <path>` |

## 💡 How It Works

### The Problem
Generative AI models often produce culturally inaccurate images due to:
- Limited cultural knowledge in training data
- Bias towards Western/dominant culture representations
- Lack of visual details about authentic cultural elements

### Our Solution: Self-Improving Cultural Agent

**1. Firebase-Powered Knowledge Base**
- Direct integration with Firestore (575+ verified cultural images)
- Real-time data updates from crowd-sourced contributions
- Automatic detection of data gaps

**2. Dual-Model Evaluation**
- **GPT-OSS-20B** (20B params): Generates detailed cultural verification questions
- **Qwen3-VL-8B**: Analyzes images and answers questions about cultural accuracy

**3. RAG-Enhanced Context**
- CLIP-based reference image retrieval
- VLM-extracted cultural knowledge from verified images
- Text + visual guidance for precise evaluation

**4. Model-Agnostic Image Editing**
- Universal prompt adapter for 6+ I2I models
- Automatic optimization for each model's format
- Iterative refinement based on VLM feedback

**5. Self-Improving Loop**
```
User generates → VLM detects gap ("Not enough jeogori collar data")
→ System creates Firebase job → Users upload authentic images
→ RAG auto-updates (89% faster!) → Accuracy improves (15% → 95%)
```

### Impact
- **Cultural Accuracy**: 30-40% → 70-90%+ with visual knowledge
- **Model Coverage**: Works with FLUX, SD3.5, Qwen, HiDream, etc.
- **Update Speed**: 89% faster with incremental FAISS updates
- **Continuous Learning**: Gets smarter with each use

## 🔧 Requirements

### Hardware
- **GPU**: 8GB+ VRAM (4-bit quantization) or 24GB+ for full precision
- **Storage**: ~50GB for models + data
- **RAM**: 16GB+ recommended

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Models Used
- **GPT-OSS-20B** - Question generation (16GB VRAM)
- **Qwen3-VL-8B-Instruct** - Image evaluation (8GB VRAM)
- **CLIP** - Image similarity search
- **FLUX/SD3.5/Qwen** - Image generation & editing

### Installation
```bash
pip install -r requirements.txt
```

**Note**: Firebase credentials required for data access (contact: chans@andrew.cmu.edu)

## 📊 Data Pipeline

```
1. Firebase Firestore (575+ contributions)
   ↓
2. init_dataset.py - Auto-detects new data (incremental update)
   ↓
3. VLM Caption Enhancement (Qwen3-VL)
   ↓
4. Cultural Knowledge Extraction (GPT-OSS-20B + Qwen3-VL)
   ↓
5. FAISS Index Building (Text RAG + CLIP Image Index)
   ↓
6. Ready for Cultural Evaluation!
   ↓
7. VLM Evaluation → Gap Detection → Job Creation → Loop back to step 1
```

**Key Feature**: Incremental updates only process new data (89% time savings!)

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md) - Get started in 30 minutes
- [Architecture](ARCHITECTURE.md) - System design and component details
- [FAQ](FAQ.md) - Frequently asked questions
- [Contributing](CONTRIBUTING.md) - Development setup and guidelines
- [Changelog](CHANGELOG.md) - Version history and updates

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

### Firebase Issues

**Firebase connection failed?**
```bash
# Test Firebase connectivity
python scripts/05_utils/test_firebase_connection.py

# System automatically falls back to CSV if Firebase unavailable
```

**Need Firebase credentials?**
- Contact: chans@andrew.cmu.edu
- You'll receive: `firebase-service-account.json` and `.firebase_config.json`
- Place both files in project root directory

### GPU/Memory Issues

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

## 📝 Citation

If you use CCUB2-Agent in your research, please cite our paper:

```bibtex
@misc{seo2025exposingblindspotsculturalbias,
      title={Exposing Blindspots: Cultural Bias Evaluation in Generative Image Models},
      author={Huichan Seo and Sieun Choi and Minki Hong and Yi Zhou and Junseo Kim and Lukman Ismaila and Naome Etori and Mehul Agarwal and Zhixuan Liu and Jihie Kim and Jean Oh},
      year={2025},
      eprint={2510.20042},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.20042},
}
```

**Paper**: [Exposing Blindspots: Cultural Bias Evaluation in Generative Image Models](https://arxiv.org/abs/2510.20042)

## 📧 Contact

For Firebase credentials or questions about the project:
- **Email**: chans@andrew.cmu.edu
- **Institution**: Carnegie Mellon University

## 🔗 Related Projects

- [WorldCCUB App](https://github.com/cmubig/WorldCCUB) - Crowdsourcing platform for cultural data collection
