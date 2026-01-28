# CCUB2 Agent

**Model-Agnostic Cultural Bias Mitigation System**

Automatically detect and correct cultural biases in generative image models using a multi-agent loop with VLM-based evaluation, RAG-enhanced cultural knowledge, and model-specific prompt optimization.

> **Project Milestone**: [seochan99.github.io/worldccub-agent-milestone](https://seochan99.github.io/worldccub-agent-milestone/) — NeurIPS 2026 D&B submission tracker

## Key Innovation

**One Universal Instruction → 6+ Model-Optimized Prompts**

The system automatically adapts editing instructions to each model's optimal format:
- **FLUX Kontext**: Context-preserving instructions
- **Qwen Image Edit**: Detailed, specific requirements
- **Stable Diffusion 3.5**: Structured with quality tags
- **HiDream, NextStep, Custom models**...

Best performance from every model, zero manual tuning.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/cmubig/ccub2-agent.git
cd ccub2-agent
pip install -r requirements.txt

# 2. Firebase credentials (required)
# Contact chans@andrew.cmu.edu for:
#   - firebase-service-account.json
#   - .firebase_config.json
# Place both in project root

# 3. Test connection
python scripts/utils/test_firebase_connection.py

# 4. Initialize dataset
python scripts/setup/init_dataset.py --country korea

# 5. Run full pipeline
python scripts/run_full_pipeline.py --country korea
```

### Interactive Mode

```bash
python scripts/testing/test_model_agnostic_editing.py \
  --prompt "A Korean woman in traditional hanbok" \
  --model qwen \
  --t2i-model sdxl \
  --country korea \
  --category traditional_clothing
```

## How It Works

```
Input Image
  |
  v
Scout Agent -----> detect cultural data gaps
  |
  v
VLM Detector -----> identify cultural errors (Qwen3-VL)
  |
  v
CLIP RAG ---------> retrieve reference images from 575+ verified cultural images
  |
  v
Edit Agent -------> model-agnostic I2I correction (FLUX / Qwen / SD3.5 / ...)
  |
  v
Judge Agent ------> evaluate result, loop if score < threshold
  |
  v
Job Agent --------> if data insufficient, create collection job on WorldCCUB app
```

## Project Structure

```
ccub2-agent/
├── ccub2_agent/                    # Core Python package
│   ├── agents/                     # Multi-agent loop
│   │   ├── core/                   #   Orchestrator, Scout, Edit, Judge, Job, Verification
│   │   ├── evaluation/             #   Metric, Benchmark, ReviewQA
│   │   ├── data/                   #   Caption, IndexRelease, DataValidator
│   │   └── governance/             #   CountryRep
│   ├── detection/                  # VLM cultural bias detection
│   ├── retrieval/                  # CLIP image RAG + reference selector
│   ├── adaptation/                 # Universal prompt adapter
│   ├── editing/                    # I2I adapters + iterative pipeline
│   ├── evaluation/                 # Cultural & general metrics
│   ├── data/                       # CountryPack, Firebase, gap analysis, curation
│   ├── models/                     # Universal interface + model registry
│   ├── schemas/                    # Agent message protocols + provenance
│   ├── orchestration/              # Decision logging
│   └── reproducibility/            # Hyperparameters, splits, configs
│
├── scripts/                        # Workflow scripts
│   ├── setup/                      #   Dataset initialization
│   ├── data_processing/            #   Caption enhancement, knowledge extraction
│   ├── indexing/                   #   CLIP & text RAG index building
│   ├── curation/                   #   Download, license validation, merge
│   ├── pipelines/                  #   Multi-step orchestration
│   ├── testing/                    #   Interactive testing interface
│   ├── experiments/                #   Experiment execution
│   ├── analysis/                   #   Firebase storage analysis
│   ├── utils/                      #   Downloads, Firebase tests
│   ├── run_full_pipeline.py        #   Full pipeline entry point
│   └── test_e2e_loop.py            #   End-to-end loop test
│
├── gui/                            # Web GUI (FastAPI + Next.js)
├── examples/                       # Usage examples
├── tests/                          # Integration tests
└── data/                           # Country packs, indices (gitignored)
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_full_pipeline.py` | Full pipeline entry point |
| `scripts/setup/init_dataset.py` | Initialize dataset from Firebase |
| `scripts/testing/test_model_agnostic_editing.py` | Interactive T2I/I2I with cultural evaluation |
| `scripts/data_processing/extract_cultural_knowledge.py` | Extract cultural knowledge from images |
| `scripts/indexing/build_all_country_indices.py` | Build all FAISS indices for all countries |
| `scripts/utils/test_firebase_connection.py` | Test Firebase connectivity |
| `scripts/test_e2e_loop.py` | End-to-end agent loop test |

## Data Pipeline

```
Firebase Firestore (575+ contributions)
  → init_dataset.py (incremental sync)
  → VLM Caption Enhancement (Qwen3-VL)
  → Cultural Knowledge Extraction (GPT-OSS-20B + Qwen3-VL)
  → FAISS Index Building (Text RAG + CLIP Image)
  → Cultural Evaluation → Gap Detection → Job Creation → loop
```

## Requirements

### Hardware
- **GPU**: 8GB+ VRAM (4-bit quantization) or 24GB+ for full precision
- **Storage**: ~50GB for models + data
- **RAM**: 16GB+

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### Models
| Model | Role | VRAM |
|-------|------|------|
| GPT-OSS-20B | Question generation | 16GB |
| Qwen3-VL-8B-Instruct | Image evaluation | 8GB |
| CLIP | Image similarity search | 2GB |
| FLUX / SD3.5 / Qwen | Image generation & editing | varies |

## Troubleshooting

**Firebase connection failed?**
```bash
python scripts/utils/test_firebase_connection.py
# Falls back to CSV if Firebase unavailable
```

**Out of GPU memory?**
```bash
python scripts/data_processing/extract_cultural_knowledge.py --load-in-4bit
```

**Resume from checkpoint?**
```bash
python scripts/data_processing/extract_cultural_knowledge.py --resume
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

- **Email**: chans@andrew.cmu.edu
- **Institution**: Carnegie Mellon University

## Related

- [WorldCCUB App](https://github.com/cmubig/WorldCCUB) — Crowdsourcing platform for cultural data collection
- [Project Milestone](https://seochan99.github.io/worldccub-agent-milestone/) — NeurIPS 2026 D&B progress tracker
