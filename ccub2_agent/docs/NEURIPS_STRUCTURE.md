# NeurIPS D&B Submission Structure

## ✅ 완료된 8개 계층

### 1. ✅ schemas/ - Agent간 메시지 프로토콜
**목적**: Type-safe agent communication for reproducibility

**구조:**
```
schemas/
├── __init__.py
└── agent_messages.py
    ├── DetectionOutput
    ├── RetrievalOutput
    ├── EditingOutput
    ├── EvaluationOutput
    ├── JobCreationOutput
    ├── GapAnalysisOutput
    └── AgentMessage
```

**사용 예:**
```python
from ccub2_agent.schemas import DetectionOutput

output = DetectionOutput(
    failure_modes=["over_modernization"],
    cultural_score=6.5,
    confidence=0.8,
    reference_needed=True,
)
```

### 2. ✅ orchestration/logging/ - Decision Tracking
**목적**: 모든 decision을 추적하여 "왜 이 선택이 이뤄졌나" 답변

**구조:**
```
orchestration/
├── __init__.py
└── logging/
    ├── __init__.py
    └── decision_logger.py
        ├── DecisionLogger
        ├── DecisionLogEntry
        └── DecisionReason
```

**사용 예:**
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

### 3. ✅ agents/core/variants/ - Ablation 체계화
**목적**: 4개 ablation variant 체계적 실행

**구조:**
```
agents/core/variants/
├── __init__.py
└── ablation_runner.py
    ├── AblationVariant (no_correction, retrieval_only, single_agent, multi_agent_loop)
    ├── AblationRunner
    └── AblationResult
```

**사용 예:**
```python
from ccub2_agent.agents.core.variants import run_ablation_study, AblationVariant

results = run_ablation_study(
    input_data=input_data,
    agent_config=agent_config,
    output_dir=Path("results/"),
    variants=[AblationVariant.MULTI_AGENT_LOOP],
)
```

### 4. ✅ reproducibility/ - 재현성 패키지
**목적**: 정확한 hyperparams, splits, seeds

**구조:**
```
reproducibility/
├── __init__.py
├── configs/
│   ├── __init__.py
│   ├── hyperparameters.py
│   └── hyperparameters.yaml
├── splits/
│   ├── __init__.py
│   └── benchmark_splits.py
└── README_REPRODUCIBILITY.md
```

**사용 예:**
```python
from ccub2_agent.reproducibility import load_hyperparameters

config = load_hyperparameters(Path("reproducibility/configs/hyperparameters.yaml"))
config.apply_seeds()  # Apply random seeds
```

### 5. 🚧 cultural_metric/components/ - 메트릭 컴포넌트
**목적**: VQA, RAG, failure mode detector 명시

**구조:**
```
evaluation/metrics/cultural_metric/
├── components/          # NEW
│   ├── __init__.py
│   ├── vqa_scorer.py    # VQA-based scoring
│   ├── rag_retriever.py # RAG knowledge retrieval
│   └── failure_detector.py # Failure mode detection
└── calibration/         # NEW
    ├── __init__.py
    └── human_validation.py # Human validation protocol
```

**Status**: 디렉토리 생성 완료, 구현 필요

### 6. 🚧 data/gap_analysis/ - 갭 분석 구체화
**목적**: Coverage analyzer + job creator 구체화

**구조:**
```
data/
├── gap_analysis/        # NEW
│   ├── __init__.py
│   ├── coverage_analyzer.py  # Coverage analysis
│   └── job_creator.py         # Job creation logic
└── ...
```

**Status**: 디렉토리 생성 완료, 구현 필요

### 7. ✅ models/model_registry.py - 모델 버전 Tracking
**목적**: 어떤 모델 버전 사용했나 추적

**구조:**
```
models/
├── model_registry.py    # NEW
│   ├── ModelVersion
│   ├── ModelRegistry
│   └── register_model()
└── ...
```

**사용 예:**
```python
from ccub2_agent.models.model_registry import register_model

register_model(
    model_name="Qwen3-VL-8B",
    model_type="vlm",
    version="1.0",
    checkpoint_path="path/to/checkpoint",
)
```

### 8. 🚧 tests/ - 테스트 구조화
**목적**: 품질 보증

**구조:**
```
tests/
├── unit/                # NEW
│   ├── test_agents.py
│   ├── test_detection.py
│   └── test_retrieval.py
├── integration/         # EXISTING (update needed)
│   ├── test_full_pipeline.py
│   └── test_with_reference.py
└── validation/          # NEW
    ├── test_metric_validity.py
    └── test_human_correlation.py
```

**Status**: 디렉토리 생성 완료, 테스트 작성 필요

---

## 구현 우선순위

### ✅ Phase 1 완료 (2주)
1. ✅ schemas/ + orchestration/logging
2. ✅ agents/core/variants/ (4개 ablation)
3. ✅ reproducibility/configs/ (hyperparameters.yaml)
4. ✅ models/model_registry.py

### 🚧 Phase 2 진행 중 (1개월)
5. 🚧 cultural_metric/components/ + calibration/
6. 🚧 data/gap_analysis/ 구체화
7. 🚧 tests/ (unit + integration + validation)

### 📋 Phase 3 예정 (제출 직전)
8. reproducibility/notebooks/ (minimal_example.ipynb)
9. README_REPRODUCIBILITY.md 완성 (✅ 완료)
10. 최종 검증 및 문서화

---

## 다음 단계

### 즉시 구현 필요 (High Priority)

1. **cultural_metric/components/** 구현
   - `vqa_scorer.py`: VQA-based cultural scoring
   - `rag_retriever.py`: Cultural knowledge RAG
   - `failure_detector.py`: Failure mode classification

2. **data/gap_analysis/** 구체화
   - `coverage_analyzer.py`: Coverage analysis logic
   - `job_creator.py`: Job creation with priorities

3. **tests/** 작성
   - Unit tests for core components
   - Integration tests for full pipeline
   - Validation tests for metric validity

---

## NeurIPS Reviewers를 위한 체크리스트

- [x] **Reproducibility**: Hyperparameters, splits, seeds 모두 추적
- [x] **Ablation Study**: 4개 variant 체계적 실행 가능
- [x] **Decision Transparency**: 모든 decision이 로깅됨
- [x] **Model Tracking**: 모델 버전과 hash 추적
- [ ] **Metric Validation**: Human correlation 검증 (구현 필요)
- [ ] **Test Coverage**: Unit/integration tests (작성 필요)

---

## 파일 구조 요약

```
ccub2_agent/
├── schemas/                    # ✅ 완료
├── orchestration/logging/       # ✅ 완료
├── agents/core/variants/       # ✅ 완료
├── reproducibility/            # ✅ 완료
│   ├── configs/
│   ├── splits/
│   └── README_REPRODUCIBILITY.md
├── evaluation/metrics/cultural_metric/
│   └── components/             # 🚧 구현 필요
├── data/gap_analysis/          # 🚧 구현 필요
├── models/model_registry.py    # ✅ 완료
└── tests/                      # 🚧 작성 필요
    ├── unit/
    ├── integration/
    └── validation/
```
