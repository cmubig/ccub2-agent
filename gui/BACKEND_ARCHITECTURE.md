# GUI Backend Architecture

This document explains the backend architecture of CCUB2-Agent GUI and where to find each component.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Pipeline Nodes](#pipeline-nodes)
- [Core Components](#core-components)
- [Adding New Nodes](#adding-new-nodes)

---

## Overview

The GUI backend is built with **FastAPI** and manages a visual pipeline of nodes. Each node represents a step in the cultural bias correction workflow:

```
Input → T2I → VLM → RAG Search → Reference Selection → Prompt Adaptation → I2I Edit → Output
```

---

## Directory Structure

```
gui/backend/
├── main.py                      # FastAPI server entry point
├── config.py                    # Configuration (models, paths)
├── models/                      # Pydantic data models
│   ├── node.py                  # Node data structures
│   ├── pipeline.py              # Pipeline configuration
│   └── session.py               # Session management
├── services/                    # Business logic
│   ├── pipeline_runner.py       # ⭐ MAIN: Node execution logic
│   ├── component_manager.py     # Component initialization
│   └── websocket_manager.py     # Real-time updates
└── routes/                      # API endpoints
    ├── pipeline.py              # Pipeline control endpoints
    └── history.py               # History and logs
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| **services/pipeline_runner.py** | All node execution logic | ~1300 |
| **services/component_manager.py** | Initialize VLM, CLIP, adapters | ~200 |
| **models/node.py** | Node data models | ~150 |
| **config.py** | Model IDs, paths, settings | ~100 |

---

## Pipeline Nodes

Each node has its execution logic in `services/pipeline_runner.py`. Here's where to find each node:

### Node Execution Methods

| Node | Method | Lines | Purpose |
|------|--------|-------|---------|
| **Input** | `_execute_input_node()` | 402-437 | Parse user input (prompt, country, category) |
| **T2I Generator** | `_execute_t2i_generator()` | 439-560 | Generate initial image with T2I model |
| **VLM Detector** | `_execute_vlm_detector()` | 562-761 | Detect cultural issues with Qwen3-VL |
| **Text KB Query** | (inline in `_execute_reference_selection`) | 786-813 | Query text knowledge base |
| **CLIP RAG Search** | (inline in `_execute_reference_selection`) | 815-861 | Find similar reference images |
| **Reference Selector** | `_execute_reference_selection()` | 763-920 | Select best reference image |
| **Prompt Adapter** | `_execute_prompt_adapter()` | 922-1080 | Adapt prompt for I2I model |
| **I2I Editor** | `_execute_i2i_editor()` | 1082-1242 | Edit image with I2I model |
| **Output** | `_execute_output_node()` | 1244-1277 | Finalize and save results |

### Quick Reference

Want to modify a specific node? Jump to these lines in `pipeline_runner.py`:

```python
# Input handling
Line 402: _execute_input_node()

# Image generation
Line 439: _execute_t2i_generator()
Line 1082: _execute_i2i_editor()

# VLM evaluation
Line 562: _execute_vlm_detector()

# Reference retrieval
Line 763: _execute_reference_selection()
  ├─ Line 786: Text KB Query
  ├─ Line 815: CLIP RAG Search
  └─ Line 863: Reference Selector

# Prompt engineering
Line 922: _execute_prompt_adapter()

# Output
Line 1244: _execute_output_node()
```

---

## Core Components

Each node uses underlying components from the main codebase. Here's the mapping:

### 1. VLM Detector
**Node**: VLM Detector
**Location**: `pipeline_runner.py:562-761`
**Component**: `ccub2_agent/modules/vlm_detector.py`
**Manager**: `component_manager.py:get_vlm_detector()`

**Key Methods**:
```python
vlm.score_cultural_quality(image, prompt, country, category)
# Returns: (cultural_score, prompt_score, issues)
```

**What to modify**:
- Scoring logic: `vlm_detector.py:score_cultural_quality()`
- Prompt template: `vlm_detector.py:_build_evaluation_prompt()`
- Issue detection: `vlm_detector.py:_parse_vlm_response()`

---

### 2. CLIP RAG Search
**Node**: CLIP RAG Search
**Location**: `pipeline_runner.py:815-861`
**Component**: `ccub2_agent/modules/clip_image_rag.py`
**Manager**: `component_manager.py:get_clip_rag()`

**Key Methods**:
```python
clip_rag.retrieve_similar_images(image_path, k=10, category=None)
# Returns: [{'image_path', 'similarity', 'category', 'metadata'}]
```

**What to modify**:
- Search algorithm: `clip_image_rag.py:retrieve_similar_images()`
- Similarity metric: `clip_image_rag.py:encode_image()`
- Filtering: `clip_image_rag.py` line 160-162

---

### 3. Reference Selector
**Node**: Reference Selector
**Location**: `pipeline_runner.py:863-920`
**Component**: `ccub2_agent/modules/reference_selector.py`
**Manager**: `component_manager.py:get_reference_selector()`

**Key Methods**:
```python
selector.select_best_reference(query_image, issues, category, k=10)
# Returns: {'image_path', 'similarity', 'total_score', 'reason'}
```

**What to modify**:
- Selection criteria: `reference_selector.py:_score_candidate()`
- Fallback logic: `pipeline_runner.py:1106-1117` (uses CLIP results if selector fails)

---

### 4. Prompt Adapter
**Node**: Prompt Adapter
**Location**: `pipeline_runner.py:922-1080`
**Component**: `ccub2_agent/modules/prompt_adapter.py`
**Manager**: `component_manager.py:get_prompt_adapter()`

**Key Methods**:
```python
prompt_adapter.adapt(universal_instruction, model_type, context)
# Returns: model-specific prompt string
```

**What to modify**:
- Model-specific formats: `prompt_adapter.py:_adapt_qwen()`, `_adapt_flux()`, etc.
- Issue extraction: `pipeline_runner.py:973-987` (sequential issue fixing)
- Context building: `pipeline_runner.py:989-1004`

---

### 5. T2I Generator
**Node**: T2I Generator
**Location**: `pipeline_runner.py:439-560`
**Component**: `ccub2_agent/adapters/image_editing_adapter.py`
**Manager**: `component_manager.py:get_t2i_adapter()`

**Supported Models**:
- `flux`: FLUX.1-dev
- `sd35`: SD 3.5 Medium
- `sdxl`: Stable Diffusion XL
- `gemini`: Gemini API

**Key Methods**:
```python
t2i_adapter.generate(prompt, width=1024, height=1024)
# Returns: PIL Image
```

**What to modify**:
- Model loading: `image_editing_adapter.py:FluxImageEditor.__init__()`
- Generation params: `image_editing_adapter.py:generate()` for each model

---

### 6. I2I Editor
**Node**: I2I Editor
**Location**: `pipeline_runner.py:1082-1242`
**Component**: `ccub2_agent/adapters/image_editing_adapter.py`
**Manager**: `component_manager.py:get_i2i_adapter()`

**Supported Models**:
- `qwen`: Qwen-Image-Edit-2509 (recommended)
- `flux`: FLUX ControlNet
- `sdxl`: SDXL InstructPix2Pix
- `sd35`: SD 3.5 Medium

**Key Methods**:
```python
i2i_adapter.edit(
    image=current_img,
    instruction=editing_prompt,
    reference_image=ref_img,
    reference_metadata=ref_metadata,
    true_cfg_scale=7.0,
    num_inference_steps=50,
)
# Returns: PIL Image
```

**What to modify**:
- Editing params: `pipeline_runner.py:1178-1187`
- Reference handling: `image_editing_adapter.py:QwenImageEditor.edit()` line 140-248
- Multi-image support: `image_editing_adapter.py` line 216-224

---

## Node Data Flow

### How data flows between nodes:

```python
# In pipeline_runner.py, each node stores data in self.nodes[node_id].data

# Example: VLM Detector → Prompt Adapter
vlm_data = self.nodes["vlm_detector"].data
issues = vlm_data.get("issues", [])  # Used by Prompt Adapter

# Example: CLIP RAG → I2I Editor
clip_data = self.nodes["clip_rag_search"].data
search_results = clip_data.get("search_results", [])
reference_paths = [r["path"] for r in search_results[:3]]  # Used by I2I Editor
```

### Node Data Structure

Each node has this structure:

```python
class NodeData:
    id: str              # "vlm_detector", "clip_rag_search", etc.
    type: NodeType       # Enum: T2I_GENERATOR, VLM_DETECTOR, etc.
    label: str           # Display name
    status: NodeStatus   # PENDING, PROCESSING, COMPLETED, ERROR
    data: Dict           # Node-specific output data
    error: Optional[str] # Error message if failed
```

**Node Data Contents**:

| Node | `data` Keys | Example |
|------|-------------|---------|
| VLM Detector | `issues`, `cultural_score`, `prompt_score` | `{"cultural_score": 6, "issues": [...]}` |
| CLIP RAG | `search_results`, `results_count` | `{"search_results": [{"path": "...", "score": 0.95}]}` |
| Prompt Adapter | `adapted_prompt`, `model`, `issues_addressed` | `{"adapted_prompt": "Modify the food..."}` |
| I2I Editor | `iteration_history`, `editing_time`, `issues_addressed` | `{"editing_time": 371.2}` |

---

## Adding New Nodes

Want to add a new node to the pipeline? Follow these steps:

### Step 1: Define Node in `pipeline_runner.py`

Add to `__init__()` method (~line 80):

```python
self.nodes = {
    # ... existing nodes ...
    "my_new_node": NodeData(
        id="my_new_node",
        type=NodeType.CUSTOM,  # Add new enum in models/node.py
        label="My New Node",
        position=NodePosition(x=100, y=500),
        status=NodeStatus.PENDING
    ),
}
```

### Step 2: Create Execution Method

Add new method in `pipeline_runner.py`:

```python
async def _execute_my_new_node(self, config: PipelineConfig):
    """Execute my new node."""
    node = self.nodes["my_new_node"]
    node.status = NodeStatus.PROCESSING
    node.start_time = time.time()

    await manager.broadcast_node_update("my_new_node", "processing", {})

    try:
        # Get data from previous nodes
        vlm_data = self.nodes["vlm_detector"].data

        # Your logic here
        result = do_something(vlm_data)

        # Store result
        node.status = NodeStatus.COMPLETED
        node.end_time = time.time()
        node.data = {
            "result": result,
        }

        await manager.broadcast_node_update("my_new_node", "completed", node.data)

    except Exception as e:
        logger.error(f"My new node failed: {e}", exc_info=True)
        node.status = NodeStatus.ERROR
        node.error = str(e)
        await manager.broadcast_error(str(e), "my_new_node")
        raise
```

### Step 3: Add to Pipeline Flow

Update `run()` method (~line 250):

```python
async def run(self):
    # ... existing nodes ...

    # Add your node
    await self._execute_my_new_node(config)

    # Continue with remaining nodes
```

### Step 4: Add Component Manager (Optional)

If your node needs a reusable component, add to `component_manager.py`:

```python
def get_my_component(self):
    """Get or create my component."""
    if not hasattr(self, '_my_component'):
        from my_module import MyComponent
        self._my_component = MyComponent(...)
    return self._my_component
```

### Step 5: Frontend Display (Optional)

Update frontend to display your node:

```typescript
// gui/frontend/src/components/PipelineCanvas.tsx
const nodeTypes = {
  // ... existing nodes ...
  my_new_node: MyNewNodeComponent,
};
```

---

## Component Manager

**Location**: `services/component_manager.py`

The Component Manager is responsible for:
1. **Lazy initialization**: Components loaded only when needed
2. **Caching**: Reuse components across iterations
3. **Country-specific setup**: Different indices per country

**Available Components**:

```python
comp_mgr = get_component_manager(country="korea")

# Get components
vlm = comp_mgr.get_vlm_detector()
clip_rag = comp_mgr.get_clip_rag()
ref_selector = comp_mgr.get_reference_selector()
prompt_adapter = comp_mgr.get_prompt_adapter()
t2i_adapter = comp_mgr.get_t2i_adapter("flux")
i2i_adapter = comp_mgr.get_i2i_adapter("qwen")
```

**Component Lifecycle**:
1. First call: Initialize and cache
2. Subsequent calls: Return cached instance
3. On cleanup: Free GPU memory

---

## Configuration

**Location**: `config.py`

Central configuration for all models and paths:

```python
# Model IDs
MODELS = {
    "vlm": "Qwen/Qwen3-VL-8B-Instruct",
    "clip": "openai/clip-vit-base-patch32",
}

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("gui/outputs")

# Performance
MAX_CONCURRENT_PIPELINES = 3
GPU_MEMORY_FRACTION = 0.9
```

**Modify settings here** instead of hardcoding in node logic.

---

## WebSocket Updates

**Location**: `services/websocket_manager.py`

Real-time updates to frontend via WebSocket:

```python
# Broadcast node update
await manager.broadcast_node_update(
    node_id="vlm_detector",
    status="completed",
    data={"cultural_score": 8}
)

# Broadcast progress
await manager.broadcast_progress(
    node_id="i2i_editor",
    current_step=25,
    total_steps=50,
    eta_seconds=60,
    message="Editing image..."
)

# Broadcast error
await manager.broadcast_error(
    error_message="CUDA out of memory",
    node_id="t2i_generator"
)
```

---

## Common Modifications

### Change VLM Scoring Threshold

**File**: `pipeline_runner.py`
**Line**: ~307

```python
# Change from 8.0 to your desired threshold
if cultural_score >= 8.0:
    logger.info("✓ Cultural accuracy acceptable")
    break
```

### Adjust I2I Editing Strength

**File**: `pipeline_runner.py`
**Line**: ~1184

```python
true_cfg_scale=7.0,  # Higher = stronger instruction following (1-15)
num_inference_steps=50,  # More steps = better quality (20-100)
```

### Change Reference Selection Logic

**File**: `ccub2_agent/modules/reference_selector.py`
**Line**: 99-131

```python
def _score_candidate(self, candidate, issues):
    score = candidate['similarity']  # Base CLIP score

    # Add your custom scoring here
    if "traditional" in candidate.get('metadata', {}).get('category', ''):
        score += 0.1  # Boost traditional items

    return score
```

### Add New Model Support

**File**: `ccub2_agent/adapters/image_editing_adapter.py`

1. Create new class inheriting `BaseImageEditor`
2. Implement `edit()` and `generate()` methods
3. Register in `create_adapter()` function

Example:
```python
class MyCustomEditor(BaseImageEditor):
    def edit(self, image, instruction, **kwargs):
        # Your model logic
        return edited_image

# Register
def create_adapter(model_type, **kwargs):
    if model_type == "my_custom":
        return MyCustomEditor(**kwargs)
    # ... existing models
```

---

## Debugging Tips

### 1. Enable Detailed Logging

**File**: `main.py`

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. Check Node History

Each node execution is logged in session history:

```python
# Access via API
GET /api/history/{session_id}

# Or check file
cat gui/outputs/{session_id}/history.json
```

### 3. Inspect Node Data

Add debug logging in pipeline_runner.py:

```python
logger.debug(f"VLM data: {self.nodes['vlm_detector'].data}")
logger.debug(f"CLIP results: {self.nodes['clip_rag_search'].data}")
```

### 4. Test Individual Nodes

Run components standalone:

```python
# Test VLM detector
from ccub2_agent.modules.vlm_detector import create_vlm_detector
vlm = create_vlm_detector(country="korea")
score = vlm.score_cultural_quality(image, prompt, "korea", "food")
print(score)
```

---

## Performance Optimization

### GPU Memory Management

**File**: `component_manager.py`

```python
# Sequential loading: Load one component at a time
vlm = comp_mgr.get_vlm_detector()
result = vlm.detect(...)
del vlm  # Free memory
torch.cuda.empty_cache()

# Then load next component
i2i = comp_mgr.get_i2i_adapter("qwen")
```

### Parallel Processing

**File**: `pipeline_runner.py`

Currently sequential. For parallel processing:

```python
import asyncio

# Run independent nodes in parallel
await asyncio.gather(
    self._execute_text_kb_query(config),
    self._execute_clip_rag_search(config),
)
```

---

## Testing

### Unit Tests

Test individual components:

```bash
pytest tests/unit/test_vlm_detector.py
pytest tests/unit/test_clip_rag.py
```

### Integration Tests

Test full pipeline:

```bash
pytest tests/integration/test_full_pipeline.py
```

### Manual Testing

```bash
# Start backend
cd gui/backend
python main.py

# Test endpoint
curl http://localhost:8000/api/pipeline/start \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Korean hanbok", "country": "korea"}'
```

---

## Need Help?

- **Backend Issues**: Check `pipeline_runner.py` and component implementations
- **Model Issues**: Check `ccub2_agent/adapters/` and `ccub2_agent/modules/`
- **API Issues**: Check `routes/` and `main.py`
- **Data Flow**: Read this document's [Node Data Flow](#node-data-flow) section

For more details, see:
- [Main README](../README.md)
- [Architecture Overview](../ARCHITECTURE.md)
- [Contributing Guide](../CONTRIBUTING.md)

---

**Last Updated**: November 2025
