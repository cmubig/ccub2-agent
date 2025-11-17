# CCUB2-Agent Architecture

This document explains the system architecture, design decisions, and data flow of CCUB2-Agent.

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Self-Improving Loop](#self-improving-loop)
- [Technology Stack](#technology-stack)

---

## System Overview

CCUB2-Agent is a **model-agnostic cultural bias mitigation system** for AI-generated images. It uses a combination of Vision-Language Models (VLM), Retrieval-Augmented Generation (RAG), and iterative refinement to detect and correct cultural inaccuracies.

### Key Features

1. **Automatic Detection**: VLM evaluates cultural accuracy (1-10 scale)
2. **Reference-Based Correction**: RAG retrieves authentic cultural images
3. **Iterative Refinement**: Repeats until cultural score ≥ 8/10
4. **Model-Agnostic**: Supports 6+ T2I/I2I models via universal adapter pattern
5. **Self-Improving**: Automatically detects data gaps and creates collection jobs

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CCUB2-Agent System                                │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  External Data Source                                                      │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  WorldCCUB App (Mobile)                          │                    │
│  │  • User uploads authentic cultural images         │                    │
│  │  • Auto-sync to Firebase                          │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Firebase (Cloud Database)                       │                    │
│  │  • contributions (approved images)                │                    │
│  │  • jobs (data collection requests)                │                    │
│  └────────────────┬─────────────────────────────────┘                    │
└───────────────────┼────────────────────────────────────────────────────────┘
                    │
                    │ sync
                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Data Layer                                                                │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  data/country_packs/{country}/                   │                    │
│  │  • approved_dataset.json (original captions)      │                    │
│  │  • approved_dataset_enhanced.json (VLM-enhanced)  │                    │
│  │  • images/ (downloaded reference images)          │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   │ indexing                                               │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  RAG Indices                                     │                    │
│  │  ┌────────────────────┐  ┌───────────────────┐  │                    │
│  │  │ CLIP Image Index   │  │ Text Knowledge KB │  │                    │
│  │  │ • clip.index       │  │ • faiss.index     │  │                    │
│  │  │ • FAISS cosine sim │  │ • Sentence embed  │  │                    │
│  │  │ • Visual retrieval │  │ • Text retrieval  │  │                    │
│  │  └────────────────────┘  └───────────────────┘  │                    │
│  └──────────────────────────────────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    │ retrieval
                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Processing Pipeline (Iterative Refinement)                               │
│                                                                            │
│  Step 0: Initial Generation                                               │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Text-to-Image (T2I) Adapter                     │                    │
│  │  • FLUX / SDXL / SD3.5 / Gemini                  │                    │
│  │  • Universal interface: generate(prompt, w, h)   │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  VLM Cultural Detector (Qwen3-VL-8B)             │                    │
│  │  • Evaluates cultural accuracy (1-10)            │                    │
│  │  • Evaluates prompt alignment (1-10)             │                    │
│  │  • Detects specific issues                       │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   │ if score < 8                                          │
│                   ▼                                                        │
│  Steps 1-5: Iterative Editing                                             │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Reference Retrieval                             │                    │
│  │  • Query CLIP index (top-k similar images)       │                    │
│  │  • Query Text KB (relevant cultural knowledge)   │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Prompt Adapter                                  │                    │
│  │  • Generate culturally-aware editing prompt      │                    │
│  │  • Incorporate VLM feedback + reference context  │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Image-to-Image (I2I) Adapter                    │                    │
│  │  • Qwen-Image-Edit / FLUX ControlNet / SDXL      │                    │
│  │  • Universal interface: edit(img, prompt, refs)  │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  VLM Re-evaluation                               │                    │
│  │  • Check if cultural score improved              │                    │
│  │  • Loop back if score < 8 (max 5 iterations)     │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   │ if score >= 8 or max iterations                       │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Final Output                                    │                    │
│  │  • Culturally accurate image                     │                    │
│  │  • Metadata (scores, detected issues, iterations)│                    │
│  └──────────────────────────────────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    │ detect gaps
                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Self-Improving Loop                                                       │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Gap Detection                                   │                    │
│  │  • VLM detects missing cultural elements         │                    │
│  │  • "Not enough jeogori collar data"              │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Job Creation Agent                              │                    │
│  │  • Automatically create Firebase job             │                    │
│  │  • Request: "Upload jeogori collar examples"     │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                    │
│  │  Firebase (jobs collection)                      │                    │
│  │  • Syncs to WorldCCUB app                        │                    │
│  │  • Users see new job and upload images           │                    │
│  └────────────────┬─────────────────────────────────┘                    │
│                   │                                                        │
│                   │ cycle back to Data Layer                              │
│                   └───────────────────┐                                   │
└───────────────────────────────────────┼───────────────────────────────────┘
                                        │
                                        ▼
                          [System improves over time]
```

---

## Core Components

### 1. VLM Cultural Detector

**Location**: `metric/cultural_metric/enhanced_cultural_metric_pipeline.py`

**Purpose**: Evaluate cultural accuracy and prompt alignment of generated images.

**Model**: Qwen3-VL-8B-Instruct (8-bit quantized)

**Functionality**:
- Accepts image + prompt + country context
- Returns dual scores:
  - **Cultural Score** (1-10): Authenticity of cultural elements
  - **Prompt Score** (1-10): Alignment with original prompt
- Detects specific issues:
  - "Collar design inauthentic"
  - "Wrong color palette"
  - "Missing traditional elements"

**Key Method**:
```python
def score_cultural_quality(self, image, prompt, country, category, reference_images=None):
    # Constructs culturally-aware evaluation prompt
    # Returns (cultural_score, prompt_score, detected_issues)
```

**Why Qwen3-VL?**
- Multilingual understanding (English + Korean + Chinese + Japanese)
- Strong visual reasoning capabilities
- Supports 8-bit quantization (fits on 8GB VRAM)
- Open-source (no API costs)

---

### 2. Dual RAG System

#### 2.1 CLIP Image RAG

**Location**: `ccub2_agent/modules/clip_image_rag.py`

**Purpose**: Retrieve visually similar reference images.

**Technology**:
- **CLIP Model**: `openai/clip-vit-base-patch32`
- **Index**: FAISS (IndexFlatIP - cosine similarity)
- **Dimension**: 512 (CLIP embedding size)

**Workflow**:
1. Encode query image with CLIP → 512-dim vector
2. Search FAISS index for top-k similar images (default k=5)
3. Filter by category (optional)
4. Return paths to most relevant reference images

**Index Structure**:
```
data/clip_index/{country}/
├── clip.index              # FAISS index
├── clip_metadata.jsonl     # Image metadata (path, category)
└── clip_config.json        # Index configuration
```

#### 2.2 Cultural Knowledge Base

**Location**: `ccub2_agent/modules/cultural_rag.py`

**Purpose**: Retrieve textual cultural knowledge.

**Technology**:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Index**: FAISS (IndexFlatIP)
- **Dimension**: 384 (sentence-transformers size)

**Workflow**:
1. Extract cultural knowledge from image captions with Qwen3-VL
2. Embed knowledge chunks with sentence-transformers
3. Index with FAISS for fast retrieval
4. Query returns relevant cultural context

**Index Structure**:
```
data/cultural_index/{country}/
├── faiss.index             # FAISS index
├── metadata.jsonl          # Knowledge chunks
└── index_config.json       # Index configuration
```

**Knowledge Example**:
```
Query: "traditional Korean hanbok"
Retrieved:
- "Hanbok features jeogori (jacket) with dongjeong (white collar)"
- "Goreum (ribbon ties) should match jeogori color"
- "Traditional colors: red, blue, yellow, green, white"
```

---

### 3. Universal Image Editing Adapter

**Location**: `ccub2_agent/adapters/image_editing_adapter.py`

**Purpose**: Provide model-agnostic interface for T2I and I2I generation.

**Design Pattern**: Strategy Pattern + Factory

**Supported Models**:

**Text-to-Image (T2I)**:
- `flux` - FLUX.1-dev (best quality)
- `sd35` - Stable Diffusion 3.5 Medium (balanced)
- `sdxl` - SDXL 1.0 (fast)
- `gemini` - Gemini 2.5 Flash Image API (photorealistic)

**Image-to-Image (I2I)**:
- `qwen` - Qwen-Image-Edit-2509 (best cultural accuracy)
- `flux` - FLUX ControlNet (style preservation)
- `sdxl` - SDXL InstructPix2Pix (fast)
- `sd35` - SD 3.5 Medium (versatile)
- `gemini` - Gemini API (photorealistic)

**Interface**:
```python
class BaseImageEditor(ABC):
    @abstractmethod
    def generate(self, prompt: str, width: int, height: int) -> Image:
        """Text-to-Image generation"""
        pass

    @abstractmethod
    def edit(self, image: Image, prompt: str, reference_images: List[Image] = None) -> Image:
        """Image-to-Image editing with optional references"""
        pass
```

**Factory**:
```python
def create_adapter(model_type: str, quantization: str = "4bit") -> BaseImageEditor:
    if model_type == "qwen":
        return QwenImageEditor(load_in_4bit=True)
    elif model_type == "flux":
        return FluxImageEditor(load_in_4bit=True)
    # ... etc
```

**Why Universal Adapter?**
- Decouple pipeline logic from specific models
- Easy to add new models (just implement BaseImageEditor)
- Swap models without changing pipeline code
- Compare models on same dataset

---

### 4. Prompt Adapter

**Location**: `ccub2_agent/modules/prompt_adapter.py`

**Purpose**: Generate culturally-aware prompts for I2I editing.

**Strategy**: Different models need different prompt styles.

**Example Adaptation**:

**Original**: `"Traditional Korean hanbok"`

**VLM Feedback**: `"Collar design incorrect, missing white dongjeong"`

**Retrieved Context**: `"Hanbok jeogori must have white dongjeong collar"`

**Adapted Prompt (for Qwen)**:
```
"Edit the traditional Korean hanbok to have authentic design:
- Add white dongjeong (collar) to jeogori
- Ensure collar follows traditional V-shape
- Maintain original pose and background
Reference authentic hanbok styles."
```

**Adapted Prompt (for FLUX)**:
```
"traditional korean hanbok, authentic white dongjeong collar,
proper jeogori design, cultural accuracy, high quality"
```

**Key Methods**:
```python
def adapt_for_editing(self, original_prompt, vlm_feedback, cultural_context, model_type):
    # Returns model-specific editing prompt

def adapt_for_generation(self, base_prompt, country, category, model_type):
    # Returns culturally-aware generation prompt
```

---

### 5. Reference Selector

**Location**: `ccub2_agent/modules/reference_selector.py`

**Purpose**: Intelligently select best reference images from RAG results.

**Selection Criteria**:
1. **Visual similarity** (CLIP score)
2. **Category match** (exact > similar > general)
3. **Quality** (resolution, clarity)
4. **Diversity** (avoid too-similar references)

**Workflow**:
```python
def select_references(clip_results, text_results, category, k=3):
    # 1. Filter by category
    # 2. Sort by CLIP similarity
    # 3. Deduplicate near-identical images
    # 4. Return top-k diverse references
```

**Why Smart Selection?**
- Too many references → confusion
- Too few references → insufficient guidance
- Similar references → redundant information
- Diverse references → comprehensive coverage

---

## Data Flow

### End-to-End Generation Flow

```
User Request
  ↓
"Traditional Korean hanbok"
  ↓
┌──────────────────────────────────────┐
│ Step 0: Initial Generation            │
│ T2I Model: FLUX / SDXL / SD3.5        │
│ Input: User prompt + country context  │
│ Output: initial_image.png             │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│ VLM Evaluation                        │
│ Input: initial_image + prompt         │
│ Output: cultural_score=6, issues=[    │
│   "Collar design incorrect",          │
│   "Missing white dongjeong"           │
│ ]                                     │
└───────────────┬──────────────────────┘
                ↓
          [Score < 8? → YES]
                ↓
┌──────────────────────────────────────┐
│ Step 1: Reference Retrieval           │
│ CLIP RAG: Query=initial_image         │
│ Returns: [ref1.jpg, ref2.jpg, ...]   │
│ Text RAG: Query="hanbok collar"       │
│ Returns: [knowledge about dongjeong]  │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│ Prompt Adaptation                     │
│ Input: original prompt + VLM feedback │
│       + reference context             │
│ Output: "Edit hanbok to add white    │
│         dongjeong collar..."          │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│ I2I Editing                           │
│ Model: Qwen-Image-Edit / FLUX         │
│ Input: initial_image + edit_prompt   │
│       + reference_images              │
│ Output: edited_image.png              │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│ VLM Re-evaluation                     │
│ Output: cultural_score=9              │
└───────────────┬──────────────────────┘
                ↓
          [Score >= 8? → YES]
                ↓
┌──────────────────────────────────────┐
│ Final Output                          │
│ • Culturally accurate image           │
│ • Metadata with scores                │
└──────────────────────────────────────┘
```

### Data Indexing Flow

```
Firebase contributions
  ↓
┌──────────────────────────────────────┐
│ init_dataset.py                       │
│ • Download images from Firebase       │
│ • Save to data/country_packs/{country}│
│ • Create approved_dataset.json        │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│ enhance_captions.py (optional)        │
│ • Use Qwen3-VL to generate detailed   │
│   descriptions from SNS captions      │
│ • Save to approved_dataset_enhanced   │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│ extract_cultural_knowledge.py         │
│ • Use Qwen3-VL to extract cultural    │
│   knowledge from images               │
│ • Save to data/cultural_knowledge/    │
└───────────────┬──────────────────────┘
                ↓
        ┌───────┴───────┐
        ↓               ↓
┌──────────────┐  ┌─────────────────────┐
│ build_clip_  │  │ integrate_knowledge_│
│ image_index  │  │ to_rag.py           │
│              │  │                     │
│ • Encode all │  │ • Embed knowledge   │
│   images     │  │   with sentence-    │
│   with CLIP  │  │   transformers      │
│ • Build      │  │ • Build FAISS index │
│   FAISS index│  │                     │
│ • Save to    │  │ • Save to           │
│   clip_index/│  │   cultural_index/   │
└──────────────┘  └─────────────────────┘
        │               │
        └───────┬───────┘
                ↓
        [Indices Ready]
                ↓
    [Used by pipeline for retrieval]
```

---

## Design Decisions

### 1. Why Model-Agnostic Architecture?

**Problem**: AI models evolve rapidly. New models emerge, old models improve.

**Solution**: Universal adapter pattern decouples pipeline from specific models.

**Benefits**:
- Add new models by implementing single interface
- Compare models on same dataset
- Swap models without changing pipeline code
- Future-proof: works with models not yet released

**Trade-off**: Some model-specific optimizations sacrificed for generality.

---

### 2. Why Dual RAG (CLIP + Text)?

**Problem**: Visual similarity alone is insufficient.

**Example**:
- CLIP might retrieve hanbok images with wrong collar
- Text KB provides explicit knowledge: "dongjeong must be white"

**Solution**: Combine visual similarity (CLIP) + semantic knowledge (text).

**Benefits**:
- CLIP: Finds visually similar examples
- Text: Provides explicit cultural rules
- Together: Comprehensive cultural context

**Implementation**:
- Parallel retrieval (both run simultaneously)
- Results merged by Reference Selector

---

### 3. Why Iterative Refinement?

**Problem**: Single-pass correction often insufficient.

**Example**:
- Pass 1: Fix collar → score 7 → still missing proper color
- Pass 2: Fix color → score 9 → acceptable

**Solution**: Loop until cultural score ≥ 8 (max 5 iterations).

**Benefits**:
- Handles complex issues requiring multiple edits
- Self-correcting: VLM guides each iteration
- Graceful degradation: stops at max iterations even if not perfect

**Trade-off**: Slower than single-pass (but more accurate).

---

### 4. Why 4-bit Quantization?

**Problem**: Full-precision models require 24GB+ VRAM.

**Solution**: 4-bit quantization reduces memory to ~8GB.

**Implementation**:
```python
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

**Trade-off**:
- Memory: 24GB → 8GB (75% reduction)
- Speed: ~10% slower
- Quality: ~2% accuracy loss (acceptable)

**Result**: Runs on consumer GPUs (RTX 3090, RTX 4080).

---

### 5. Why Self-Improving Loop?

**Problem**: Static datasets become outdated. New cultural gaps discovered.

**Example**: User generates hanbok → VLM detects "not enough jeogori collar data".

**Solution**: Automatically create Firebase job requesting that data.

**Workflow**:
1. VLM detects gap: "Not enough X"
2. Job Creation Agent → Firebase job
3. WorldCCUB app syncs job
4. Users upload missing data
5. System re-indexes with new data
6. Future generations benefit

**Result**: System improves continuously without manual intervention.

---

### 6. Why Qwen3-VL for Detection?

**Alternatives Considered**:
- GPT-4V: Expensive ($0.01 per image), API-only, no fine-tuning
- LLaVA: Weaker at multilingual understanding
- Gemini Vision: API-only, no local deployment

**Why Qwen3-VL**:
- ✅ Open-source (can run locally)
- ✅ Multilingual (understands Korean/Chinese/Japanese)
- ✅ Strong visual reasoning
- ✅ Supports quantization (8GB VRAM)
- ✅ Free (no API costs)
- ✅ Active development (Alibaba)

**Trade-off**: Slightly weaker than GPT-4V, but good enough + free + local.

---

### 7. Why FAISS for Indexing?

**Alternatives Considered**:
- Pinecone/Weaviate: Cloud-based, costs scale with usage
- ChromaDB: Good but slower for large datasets
- HNSW: Fast but complex to maintain

**Why FAISS**:
- ✅ Fast (C++ implementation)
- ✅ Local (no API calls)
- ✅ Free (open-source by Meta)
- ✅ Scalable (handles millions of vectors)
- ✅ Simple API
- ✅ Battle-tested (used in production at Meta)

**Index Type**: `IndexFlatIP` (Inner Product = Cosine Similarity)
- Exact search (no approximation)
- Perfect for datasets < 10K images per country

---

## Self-Improving Loop

### Overview

The self-improving loop ensures CCUB2-Agent gets better over time by detecting data gaps and automatically requesting missing data.

### Components

#### 1. Gap Detection

**Trigger**: VLM detects low cultural score + identifies missing elements.

**Example Detection**:
```
cultural_score: 6/10
issues: [
  "Jeogori collar design incorrect - insufficient reference data",
  "Traditional color palette not well represented"
]
```

**Pattern Recognition**:
- "insufficient reference data" → data gap
- "not well represented" → data gap
- "inconsistent with traditional X" → missing knowledge

#### 2. Job Creation Agent

**Location**: `ccub2_agent/modules/job_creation_agent.py`

**Functionality**:
```python
def create_data_gap_job(country, category, missing_element, context):
    job = {
        "title": f"{country.capitalize()} {category} - {missing_element}",
        "description": f"Upload authentic examples of {missing_element}",
        "country": country,
        "category": category,
        "targetCount": 20,
        "created": timestamp,
        "status": "active"
    }
    firebase.create_job(job)
```

**Example Job**:
```json
{
  "title": "Korea traditional_clothing - jeogori collar details",
  "description": "Upload authentic examples of jeogori with traditional collar (dongjeong) designs. Focus on collar shape, white dongjeong placement, and proper attachment.",
  "country": "korea",
  "category": "traditional_clothing",
  "targetCount": 20,
  "status": "active"
}
```

#### 3. WorldCCUB App Integration

**App Behavior**:
1. Syncs jobs from Firebase
2. Shows job to users with clear instructions
3. Users upload images with metadata
4. Images approved and added to contributions

#### 4. Automatic Re-indexing

**Trigger**: New contributions detected (Firebase listener or scheduled task)

**Workflow**:
```bash
# Detect new contributions
python scripts/01_setup/init_dataset.py --country korea

# Re-build indices
python scripts/03_indexing/build_clip_image_index.py --country korea
python scripts/03_indexing/integrate_knowledge_to_rag.py --country korea
```

**Result**: New references available for next generation.

### Example Flow

```
Day 1: User generates hanbok image
  ↓
VLM: "Score 6/10 - collar design incorrect, insufficient reference data"
  ↓
Job Agent: Creates "Korea traditional_clothing - jeogori collar"
  ↓
Job synced to WorldCCUB app
  ↓
Day 2-7: Users upload 20 authentic hanbok collar images
  ↓
Images approved → Firebase contributions
  ↓
Automatic re-indexing (scheduled task)
  ↓
Day 8: User generates hanbok again
  ↓
VLM: "Score 9/10 - collar design authentic" ✓
```

### Metrics

Track improvement over time:

```python
# Before new data
korea_hanbok_scores = [6, 7, 6, 5, 7]  # avg: 6.2

# After 20 new collar images
korea_hanbok_scores = [9, 8, 9, 8, 8]  # avg: 8.4
```

---

## Technology Stack

### Core ML Models

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| VLM Detector | Qwen3-VL-8B-Instruct | 8B params | Cultural evaluation |
| T2I Generation | FLUX.1-dev | 12B params | Initial image generation |
| I2I Editing | Qwen-Image-Edit-2509 | 7B params | Reference-based editing |
| CLIP Embeddings | clip-vit-base-patch32 | 151M params | Visual similarity |
| Text Embeddings | all-MiniLM-L6-v2 | 22M params | Semantic similarity |

### Frameworks & Libraries

**ML Frameworks**:
- `transformers` (HuggingFace) - Model loading and inference
- `diffusers` - Stable Diffusion models
- `torch` - Deep learning framework

**RAG & Search**:
- `faiss` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `clip` - Vision-language embeddings

**Backend (GUI)**:
- `FastAPI` - REST API server
- `WebSocket` - Real-time updates
- `Pydantic` - Data validation

**Frontend (GUI)**:
- `Next.js` - React framework
- `React Flow` - Node visualization
- `TailwindCSS` - Styling

**Database**:
- `Firebase` - Cloud database + authentication
- `JSON` - Local dataset storage

**Other**:
- `Pillow` - Image processing
- `opencv-python` - Computer vision
- `pytest` - Testing

### Deployment

**GPU Requirements**:
- Minimum: 8GB VRAM (with 4-bit quantization)
- Recommended: 12GB VRAM (8-bit quantization)
- Optimal: 24GB VRAM (full precision)

**Compute**:
- Local: NVIDIA RTX 3090 / 4080
- Cloud: AWS p3.2xlarge / Google Cloud T4

**Storage**:
- Models: ~20GB (cached in `~/.cache/huggingface/`)
- Data: ~1GB per country
- Total: ~50GB recommended

---

## Future Enhancements

### Planned Features

1. **Multi-GPU Support**: Distribute models across GPUs for faster processing
2. **Video Generation**: Extend to video T2V/V2V models
3. **3D Asset Generation**: Support 3D cultural artifacts
4. **Real-time Mode**: Sub-second feedback for interactive editing
5. **Fine-tuning**: Fine-tune VLM on cultural datasets
6. **Human-in-the-Loop**: Optional manual verification step
7. **A/B Testing**: Compare different model combinations
8. **Automated Benchmarking**: Track performance metrics over time

### Research Directions

1. **Better Cultural Metrics**: Beyond 1-10 scoring
2. **Explainable AI**: Visualize VLM attention on cultural elements
3. **Cross-Cultural Transfer**: Apply learned patterns across countries
4. **Adversarial Testing**: Stress-test with edge cases

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- How to add new models
- How to add new countries
- Testing requirements

---

## References

### Papers

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966)

### Repositories

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)

---

**Questions?** See [FAQ.md](FAQ.md) or open a [GitHub Discussion](https://github.com/cmubig/ccub2-agent/discussions).
