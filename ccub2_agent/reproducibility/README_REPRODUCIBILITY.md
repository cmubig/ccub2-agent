# Reproducibility Guide

This document explains how to reproduce all experiments in the NeurIPS submission.

## Quick Start

```bash
# 1. Load hyperparameters
from ccub2_agent.reproducibility import load_hyperparameters
config = load_hyperparameters(Path("reproducibility/configs/hyperparameters.yaml"))

# 2. Apply random seeds
config.apply_seeds()

# 3. Load benchmark splits
from ccub2_agent.reproducibility.splits import load_benchmark_split
split = load_benchmark_split(Path("reproducibility/splits/task_a_test.json"))

# 4. Run experiment
# ... (use exact config and split)
```

## Components

### 1. Hyperparameters (`configs/`)

All hyperparameters are tracked in `hyperparameters.yaml`:

- Loop parameters (max_iterations, score_threshold)
- VLM parameters (model, confidence_threshold)
- Retrieval parameters (top_k, method)
- Editing parameters (strength, model)
- Random seeds (for exact reproducibility)

**Usage:**
```python
from ccub2_agent.reproducibility import load_hyperparameters

config = load_hyperparameters(Path("reproducibility/configs/hyperparameters.yaml"))
config.apply_seeds()  # Apply random seeds
```

### 2. Benchmark Splits (`splits/`)

Standardized splits for each task:

- `task_a_train.json` / `task_a_test.json` - Cultural Fidelity
- `task_b_test.json` - Iterative Degradation
- `task_c_test.json` - Cross-Country Transfer
- `task_d_test.json` - Contrastive Evaluation

Each split contains:
- Exact image IDs
- Countries and categories
- Random seed used

**Usage:**
```python
from ccub2_agent.reproducibility.splits import load_benchmark_split

split = load_benchmark_split(Path("reproducibility/splits/task_a_test.json"))
# Use split.image_ids, split.countries, etc.
```

### 3. Model Registry (`models/model_registry.py`)

Tracks exact model versions and hashes:

**Usage:**
```python
from ccub2_agent.models.model_registry import register_model, get_model_registry

# Register a model
register_model(
    model_name="Qwen3-VL-8B",
    model_type="vlm",
    version="1.0",
    checkpoint_path="path/to/checkpoint",
)

# Get registered model
registry = get_model_registry()
model = registry.get_model("Qwen3-VL-8B", "vlm", "1.0")
```

### 4. Decision Logging (`orchestration/logging/`)

Tracks all agent decisions:

**Usage:**
```python
from ccub2_agent.orchestration.logging import log_agent_decision, DecisionReason

log_agent_decision(
    agent_name="JudgeAgent",
    decision_type="STOP",
    decision_value="score >= 8.0",
    reason=DecisionReason.SCORE_THRESHOLD,
    context={"score": 8.5},
)
```

## Reproducing Experiments

### Ablation Study

```python
from ccub2_agent.agents.core.variants import run_ablation_study, AblationVariant
from ccub2_agent.agents.base_agent import AgentConfig

input_data = {
    "image_path": "path/to/image.png",
    "prompt": "Korean traditional clothing",
    "country": "korea",
    "category": "traditional_clothing",
}

agent_config = AgentConfig(
    country="korea",
    category="traditional_clothing",
    output_dir=Path("results/"),
)

results = run_ablation_study(
    input_data=input_data,
    agent_config=agent_config,
    output_dir=Path("ablation_results/"),
    variants=[
        AblationVariant.NO_CORRECTION,
        AblationVariant.RETRIEVAL_ONLY,
        AblationVariant.SINGLE_AGENT,
        AblationVariant.MULTI_AGENT_LOOP,
    ],
)
```

### Benchmark Run

```python
from ccub2_agent.reproducibility import load_hyperparameters
from ccub2_agent.reproducibility.splits import load_benchmark_split
from ccub2_agent.agents.evaluation.benchmark_agent import BenchmarkAgent

# Load config and split
config = load_hyperparameters(Path("reproducibility/configs/hyperparameters.yaml"))
config.apply_seeds()

split = load_benchmark_split(Path("reproducibility/splits/task_a_test.json"))

# Run benchmark
benchmark_agent = BenchmarkAgent(agent_config)
results = benchmark_agent.execute({
    "task": "fidelity",
    "split": split,
    "config": config,
})
```

## Output Files

All experiments produce:

1. **Decision logs** (`logs/decisions/decisions_*.json`)
   - All agent decisions with timestamps and reasons

2. **Ablation results** (`ablation_results_*.json`)
   - Scores, gains, regression rates for each variant

3. **Benchmark results** (`benchmark_results_*.json`)
   - Task-specific results with metrics

4. **Model registry** (`models/registry.json`)
   - All model versions and hashes used

## Verification

To verify reproducibility:

1. Check that random seeds match
2. Verify model versions in registry
3. Compare decision logs (should be identical)
4. Check hyperparameters match exactly

## Notes

- All random operations use fixed seeds
- Model versions are tracked with hashes
- All decisions are logged with reasons
- Splits are saved with exact image IDs
