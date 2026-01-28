# WorldCCUB Multi-Agent Loop System

This package contains the Python implementation of WorldCCUB multi-agent loop agents. These are the actual executable agents that run the cultural improvement pipeline.

## Folder Structure

```
agents/
├── __init__.py              # Main package exports
├── base_agent.py            # BaseAgent class and common types
├── utils.py                  # Utility functions
├── README.md                 # This file
│
├── core/                     # Core Multi-Agent Loop
│   ├── __init__.py
│   ├── orchestrator_agent.py    # Master controller
│   ├── scout_agent.py           # Gap detection
│   ├── edit_agent.py            # I2I editing
│   ├── judge_agent.py           # Quality evaluation
│   ├── job_agent.py             # Data collection
│   └── verification_agent.py    # Reference verification (NEW)
│
├── evaluation/               # Evaluation & Benchmark
│   ├── __init__.py
│   ├── metric_agent.py          # Cultural metric toolkit
│   ├── benchmark_agent.py        # CultureBench-Global execution
│   └── review_qa_agent.py        # Peer review monitoring
│
├── data/                     # Data Pipeline
│   ├── __init__.py
│   ├── caption_agent.py         # Caption normalization
│   ├── index_release_agent.py   # RAG indices & releases
│   └── data_validator_agent.py  # Data quality validation
│
└── governance/               # Governance
    ├── __init__.py
    └── country_lead_agent.py    # Country Lead coordination
```

## Architecture

```
OrchestratorAgent (core/)
    ├── ScoutAgent (core/) - Gap Detection
    ├── VerificationAgent (core/) - Reference Verification
    ├── EditAgent (core/) - I2I Editing
    ├── JudgeAgent (core/) - Quality Evaluation
    └── JobAgent (core/) - Job Creation

Support Agents:
    ├── MetricAgent (evaluation/) - Cultural Metric Toolkit
    ├── BenchmarkAgent (evaluation/) - CultureBench Execution
    ├── ReviewQAAgent (evaluation/) - Peer Review Monitoring
    ├── CaptionAgent (data/) - Caption Normalization
    ├── IndexReleaseAgent (data/) - Index & Release Management
    ├── DataValidatorAgent (data/) - Data Quality Validation
    └── CountryLeadAgent (governance/) - CL Coordination
```

## Usage

### Basic Example

```python
from ccub2_agent.agents import OrchestratorAgent, AgentConfig
from pathlib import Path

# Create config
config = AgentConfig(
    country="korea",
    category="traditional_clothing",
    output_dir=Path("results/"),
    verbose=True
)

# Initialize orchestrator
orchestrator = OrchestratorAgent(config)

# Run full loop
result = orchestrator.execute({
    "image_path": "input.png",
    "prompt": "Traditional Korean hanbok",
    "country": "korea",
    "category": "traditional_clothing",
    "max_iterations": 5,
    "score_threshold": 8.0
})

if result.success:
    print(f"Final score: {result.data['final_score']:.1f}/10")
    print(f"Iterations: {result.data['iterations']}")
    print(f"Final image: {result.data['final_image']}")
```

### Individual Agent Usage

```python
from ccub2_agent.agents import JudgeAgent, AgentConfig

config = AgentConfig(country="korea")
judge = JudgeAgent(config)

result = judge.execute({
    "image_path": "image.png",
    "prompt": "Korean hanbok",
    "country": "korea",
    "category": "traditional_clothing"
})

print(f"Cultural score: {result.data['cultural_score']}/10")
print(f"Decision: {result.data['decision']}")
```

## Agent Categories

### Core Loop Agents (`core/`)

These agents form the core iterative improvement loop:

- **OrchestratorAgent**: Coordinates the full multi-agent loop
- **ScoutAgent**: Detects coverage gaps and missing references
- **VerificationAgent**: Verifies reference relevance (MA-RAG pattern)
- **EditAgent**: Executes model-agnostic I2I editing
- **JudgeAgent**: Evaluates cultural quality and makes loop decisions
- **JobAgent**: Creates WorldCCUB collection jobs

### Evaluation Agents (`evaluation/`)

These agents handle cultural evaluation and benchmarking:

- **MetricAgent**: Runs Cultural Metric Toolkit for detailed scoring
- **BenchmarkAgent**: Executes CultureBench-Global benchmark tasks
- **ReviewQAAgent**: Monitors peer review integrity

### Data Pipeline Agents (`data/`)

These agents handle data processing and management:

- **CaptionAgent**: Normalizes captions (translation + VLM + refinement)
- **IndexReleaseAgent**: Manages RAG indices and dataset releases
- **DataValidatorAgent**: Validates data quality and schema

### Governance Agents (`governance/`)

These agents handle organizational and quality governance:

- **CountryLeadAgent**: Manages Country Lead coordination

## Configuration

All agents use `AgentConfig`:

```python
@dataclass
class AgentConfig:
    country: str
    category: Optional[str] = None
    output_dir: Optional[Path] = None
    verbose: bool = False
```

## Integration with Existing Modules

Agents use existing modules from `ccub2_agent.modules`:
- `VLMCulturalDetector` - Cultural evaluation
- `CLIPImageRAG` - Reference image retrieval
- `UniversalPromptAdapter` - Model-specific prompt adaptation
- `AgentJobCreator` - Job creation
- `DataGapAnalyzer` - Gap detection
- `FirebaseClient` - Data access

## Output Format

All agents return `AgentResult`:

```python
@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

## Import Paths

### Recommended (from package root)
```python
from ccub2_agent.agents import OrchestratorAgent, JudgeAgent
```

### Direct (from sub-packages)
```python
from ccub2_agent.agents.core import OrchestratorAgent
from ccub2_agent.agents.evaluation import MetricAgent
from ccub2_agent.agents.data import CaptionAgent
from ccub2_agent.agents.governance import CountryLeadAgent
```

## Related Documentation

- `.claude/agents/` - Claude Code agents (for code development assistance)
- `ccub2-agent/ARCHITECTURE.md` - System architecture
- `docs/proposal/WorldCCUB_Proposal.md` - Project proposal
- `docs/Agent_Architecture_Improvements.md` - Academic references & improvements
