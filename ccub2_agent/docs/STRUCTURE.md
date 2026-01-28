# CCUB2-Agent Complete Folder Structure

## Overall Principles

```
WorldCCUB-Agent/
├── WorldCCUB/              # Platform (Mobile App, Web, Admin)
└── ccub2-agent/            # Agent System (Multi-Agent Loop)
```

## ccub2-agent Complete Structure

```
ccub2-agent/
├── ccub2_agent/                    # Core agent system
│   │
│   ├── agents/                     # Multi-agent loop
│   │   ├── core/                   # Core loop agents (6)
│   │   │   ├── orchestrator_agent.py
│   │   │   ├── scout_agent.py
│   │   │   ├── edit_agent.py
│   │   │   ├── judge_agent.py
│   │   │   ├── job_agent.py
│   │   │   └── verification_agent.py
│   │   ├── evaluation/             # Evaluation agents (3)
│   │   │   ├── metric_agent.py
│   │   │   ├── benchmark_agent.py
│   │   │   └── review_qa_agent.py
│   │   ├── data/                   # Data pipeline agents (3)
│   │   │   ├── caption_agent.py
│   │   │   ├── index_release_agent.py
│   │   │   └── data_validator_agent.py
│   │   ├── governance/             # Governance agents (1)
│   │   │   └── country_lead_agent.py
│   │   ├── base_agent.py           # BaseAgent class
│   │   └── utils.py                # Utilities
│   │
│   ├── detection/                  # Cultural bias detection system
│   │   └── vlm_detector.py         # VLM-based cultural bias detection
│   │
│   ├── retrieval/                  # RAG search system
│   │   ├── clip_image_rag.py       # CLIP image RAG
│   │   └── reference_selector.py   # Reference image selection
│   │
│   ├── adaptation/                 # Prompt adaptation system
│   │   └── prompt_adapter.py       # Model-specific prompt adaptation
│   │
│   ├── editing/                    # Image editing system
│   │   ├── adapters/               # I2I model adapters
│   │   │   └── image_editing_adapter.py
│   │   └── pipelines/              # Editing pipelines
│   │       └── iterative_editing.py
│   │
│   ├── evaluation/                 # Evaluation system
│   │   └── metrics/                # Metric toolkit
│   │       ├── cultural_metric/    # Cultural metric
│   │       └── general_metric/     # General metric
│   │
│   ├── data/                       # Data management system
│   │   ├── country_pack.py         # Country-specific data pack
│   │   ├── firebase_client.py      # Firebase client
│   │   ├── gap_analyzer.py         # Data gap analysis
│   │   ├── data_gap_detector.py    # Data gap detection
│   │   └── job_creator.py          # Job creation
│   │
│   └── models/                     # Common model interface
│       └── universal_interface.py
│
├── gui/                            # GUI system
│   ├── backend/                    # FastAPI backend
│   └── frontend/                   # Next.js frontend
│
├── scripts/                        # Utility scripts
│   ├── setup/                      # Initial setup
│   ├── data_processing/            # Data processing
│   ├── indexing/                   # Indexing
│   ├── testing/                    # Testing
│   └── experiments/               # Experiments
│
├── data/                           # Data repository
│   └── country_packs/              # Country-specific data packs
│
└── examples/                       # Usage examples
```

## Import Path Mapping

### Old Path → New Path

| Old | New | Description |
|------|--------|------|
| `modules.vlm_detector` | `detection.vlm_detector` | Detection system |
| `modules.clip_image_rag` | `retrieval.clip_image_rag` | RAG search |
| `modules.reference_selector` | `retrieval.reference_selector` | Reference selection |
| `modules.prompt_adapter` | `adaptation.prompt_adapter` | Prompt adaptation |
| `adapters.image_editing_adapter` | `editing.adapters.image_editing_adapter` | I2I adapter |
| `pipelines.iterative_editing` | `editing.pipelines.iterative_editing` | Editing pipeline |
| `modules.country_pack` | `data.country_pack` | Data pack |
| `modules.firebase_client` | `data.firebase_client` | Firebase |
| `modules.gap_analyzer` | `data.gap_analyzer` | Gap analysis |
| `modules.agent_job_creator` | `data.job_creator` | Job creation |
| `metric.cultural_metric` | `evaluation.metrics.cultural_metric` | Cultural metric |

## Category Descriptions

### 1. agents/ - Multi-Agent Loop
**Purpose**: Multi-agent collaboration system for cultural improvement

- **core/**: Core iterative loop (Orchestrator → Scout → Verification → Edit → Judge → Job)
- **evaluation/**: Evaluation and benchmark execution
- **data/**: Data pipeline management
- **governance/**: Organizational governance

### 2. detection/ - Detection System
**Purpose**: Cultural bias detection using VLM

- `vlm_detector.py`: Qwen3-VL based cultural quality evaluation and issue detection

### 3. retrieval/ - RAG Search System
**Purpose**: Reference image and knowledge search

- `clip_image_rag.py`: CLIP-based image similarity search
- `reference_selector.py`: Optimal reference image selection

### 4. adaptation/ - Prompt Adaptation System
**Purpose**: Model-specific prompt format conversion

- `prompt_adapter.py`: Universal → Model-specific prompt conversion

### 5. editing/ - Image Editing System
**Purpose**: Model-agnostic I2I editing

- `adapters/`: Model-specific I2I adapters (Qwen, FLUX, SDXL, Gemini, etc.)
- `pipelines/`: Iterative editing pipeline

### 6. evaluation/ - Evaluation System
**Purpose**: Cultural metrics and benchmarks

- `metrics/cultural_metric/`: Cultural metric toolkit
- `metrics/general_metric/`: General metrics

### 7. data/ - Data Management System
**Purpose**: Data collection, storage, and analysis

- `country_pack.py`: Country-specific data pack management
- `firebase_client.py`: Firebase integration
- `gap_analyzer.py`: Data gap analysis
- `job_creator.py`: Data collection job creation

### 8. models/ - Common Interface
**Purpose**: Model-agnostic interface

- `universal_interface.py`: Unified model interface
