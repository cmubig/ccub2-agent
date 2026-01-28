# Agents Folder Structure

## Overview

The `agents/` package is organized into logical sub-packages based on agent responsibilities:

```
agents/
├── core/          # Core multi-agent loop (6 agents)
├── evaluation/    # Evaluation & benchmarking (3 agents)
├── data/          # Data pipeline (3 agents)
└── governance/    # Governance (1 agent)
```

## Core Agents (`core/`)

**Purpose**: Form the core iterative improvement loop

| Agent | Responsibility | Dependencies |
|-------|--------------|--------------|
| `orchestrator_agent.py` | Master controller, coordinates all agents | All core agents |
| `scout_agent.py` | Gap detection, missing reference identification | DataGapAnalyzer, CountryDataPack |
| `verification_agent.py` | Reference relevance verification (MA-RAG pattern) | VLM Client |
| `edit_agent.py` | Model-agnostic I2I editing | PromptAdapter, I2I Adapter |
| `judge_agent.py` | Cultural quality evaluation, loop decisions | VLMCulturalDetector |
| `job_agent.py` | WorldCCUB job creation for data collection | AgentJobCreator |

**Import**: `from ccub2_agent.agents.core import OrchestratorAgent`

## Evaluation Agents (`evaluation/`)

**Purpose**: Handle cultural evaluation and benchmarking

| Agent | Responsibility | Dependencies |
|-------|--------------|--------------|
| `metric_agent.py` | Cultural metric toolkit execution | EnhancedVLMClient, KnowledgeBase |
| `benchmark_agent.py` | CultureBench-Global execution | OrchestratorAgent, JudgeAgent, MetricAgent |
| `review_qa_agent.py` | Peer review quality monitoring | FirebaseClient |

**Import**: `from ccub2_agent.agents.evaluation import MetricAgent`

## Data Pipeline Agents (`data/`)

**Purpose**: Handle data processing and management

| Agent | Responsibility | Dependencies |
|-------|--------------|--------------|
| `caption_agent.py` | Caption normalization (translation + VLM + refinement) | VLM Client |
| `index_release_agent.py` | RAG indices and dataset release management | CLIPImageRAG, CountryDataPack |
| `data_validator_agent.py` | Data quality and schema validation | FirebaseClient |

**Import**: `from ccub2_agent.agents.data import CaptionAgent`

## Governance Agents (`governance/`)

**Purpose**: Handle organizational and quality governance

| Agent | Responsibility | Dependencies |
|-------|--------------|--------------|
| `country_lead_agent.py` | Country Lead coordination and management | FirebaseClient |

**Import**: `from ccub2_agent.agents.governance import CountryLeadAgent`

## Base Classes

- `base_agent.py`: `BaseAgent` abstract class, `AgentConfig`, `AgentResult`
- `utils.py`: Utility functions for agents

## Import Patterns

### Recommended (from package root)
```python
from ccub2_agent.agents import (
    OrchestratorAgent,  # core
    MetricAgent,        # evaluation
    CaptionAgent,       # data
    CountryLeadAgent,   # governance
)
```

### Direct (from sub-packages)
```python
from ccub2_agent.agents.core import OrchestratorAgent
from ccub2_agent.agents.evaluation import MetricAgent
from ccub2_agent.agents.data import CaptionAgent
from ccub2_agent.agents.governance import CountryLeadAgent
```

## Internal Imports

Within agent files:
- `from ...base_agent import BaseAgent, AgentConfig, AgentResult` (3 levels up to root)
- `from ...modules.xxx import XXX` (3 levels up, then modules)
- `from .xxx import XXX` (same package)
- `from ..core.xxx import XXX` (parent package, then core)

## Migration Notes

All agents were moved from flat structure to organized sub-packages. Import paths updated:
- `from .base_agent` → `from ...base_agent`
- `from ..modules` → `from ...modules`
- Cross-package imports use `..core`, `..evaluation`, etc.
