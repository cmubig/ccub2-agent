# Multi-Country Index Builder

자동으로 모든 나라의 지식베이스, CLIP 인덱스, RAG 인덱스를 구축하는 스크립트입니다.

## 파일 위치
```
scripts/build_all_country_indices.py
```

## 기능

각 나라별로 자동으로:
1. **문화 지식 추출** - Qwen3-VL-8B로 이미지에서 문화 지식 추출
2. **CLIP 인덱스 구축** - 이미지 유사도 검색용 CLIP 임베딩 인덱스
3. **RAG 인덱스 통합** - 문화 지식을 FAISS 벡터 인덱스에 통합

## 사용법

### 1. 모든 나라 처리 (전체 자동)
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py
```

현재 가능한 나라들:
- china, france, germany, italy, japan, kenya, korea, mexico, nigeria, usa

### 2. 특정 나라만 처리
```bash
# 한국, 중국, 일본만
conda run -n ccub2 python scripts/build_all_country_indices.py --countries korea china japan

# 미국만
conda run -n ccub2 python scripts/build_all_country_indices.py --countries usa
```

### 3. 특정 단계 건너뛰기

#### 지식 추출 건너뛰기 (이미 있는 지식 파일 사용)
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py --skip-knowledge
```

#### CLIP 인덱스 건너뛰기
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py --skip-clip
```

#### RAG 인덱스 건너뛰기
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py --skip-rag
```

#### 여러 개 동시에
```bash
# 지식 추출만 하고 CLIP/RAG는 건너뛰기
conda run -n ccub2 python scripts/build_all_country_indices.py --skip-clip --skip-rag
```

### 4. 이미 있어도 강제로 다시 생성
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py --force
```

### 5. Dry Run (실제 실행 안하고 뭘 할지만 확인)
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py --dry-run
```

### 6. 백그라운드 실행 + 로그 저장
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py 2>&1 | tee /tmp/build_all_indices.log &
```

## 고급 사용법

### GPU 병렬 처리 (예정 - 아직 미구현)
```bash
# GPU 0, 1을 사용해서 병렬 처리
conda run -n ccub2 python scripts/build_all_country_indices.py --parallel-gpus 0,1
```

### 특정 나라만 강제 재생성
```bash
# 한국만 모든 인덱스 다시 만들기
conda run -n ccub2 python scripts/build_all_country_indices.py --countries korea --force
```

### 중국, 일본만 지식 추출 (CLIP/RAG는 기존 것 사용)
```bash
conda run -n ccub2 python scripts/build_all_country_indices.py \
  --countries china japan \
  --skip-clip \
  --skip-rag
```

## 출력 결과

스크립트는 다음 위치에 파일들을 생성합니다:

```
data/
├── cultural_knowledge/
│   ├── korea_knowledge.json      # 한국 문화 지식
│   ├── china_knowledge.json      # 중국 문화 지식
│   └── ...
├── clip_index/
│   ├── korea/
│   │   ├── clip.index           # CLIP FAISS 인덱스
│   │   └── clip_metadata.jsonl  # 이미지 메타데이터
│   ├── china/
│   └── ...
└── cultural_index/
    ├── korea/
    │   ├── faiss.index          # RAG FAISS 인덱스
    │   └── metadata.jsonl       # 지식 메타데이터
    ├── china/
    └── ...
```

## 진행 상황 확인

스크립트는 실시간으로 진행 상황을 출력합니다:

```
================================================================================
Processing: KOREA
================================================================================
Current status:
  Knowledge: ✓
  CLIP:      ✓
  RAG:       ✓

[korea] Knowledge already exists, skipping (use --force to rebuild)
[korea] CLIP index already exists, skipping (use --force to rebuild)
[korea] RAG index already exists, skipping (use --force to rebuild)

[korea] Results:
  Knowledge: ✓
  CLIP:      ✓
  RAG:       ✓
```

## 실행 시간 예상

각 나라당 (이미지 300-400개 기준):
- **지식 추출**: 2-4시간 (GPU 사용, 가장 오래 걸림)
- **CLIP 인덱스**: 5-10분
- **RAG 인덱스**: 2-5분

**전체 10개 나라 처리**: 약 20-40시간 (순차 처리 시)

## 에러 처리

- 특정 나라에서 에러 발생해도 다른 나라는 계속 처리됩니다
- 각 단계별로 성공/실패 로그가 출력됩니다
- 최종 요약에서 전체 결과를 확인할 수 있습니다

## 주의사항

1. **GPU 메모리**: 지식 추출은 약 10GB GPU 메모리 필요
2. **디스크 공간**: 각 나라당 약 1-2GB 필요
3. **실행 시간**: 전체 처리는 매우 오래 걸리니 백그라운드 실행 권장
4. **기존 파일**: `--force` 없이는 이미 있는 인덱스 건너뜀 (안전)

## 문제 해결

### 메모리 부족
```bash
# 한 번에 하나씩만 처리
python scripts/build_all_country_indices.py --countries korea
python scripts/build_all_country_indices.py --countries china
```

### 지식 추출만 나중에
```bash
# 1단계: CLIP, RAG만 빠르게 구축
python scripts/build_all_country_indices.py --skip-knowledge

# 2단계: 나중에 지식 추출
python scripts/build_all_country_indices.py --skip-clip --skip-rag
```

### 특정 나라 실패 시
```bash
# 실패한 나라만 다시
python scripts/build_all_country_indices.py --countries nigeria --force
```

## 예시: 추천 실행 순서

### 빠른 테스트용 (한국만)
```bash
python scripts/build_all_country_indices.py --countries korea
```

### 전체 처리 (밤새 실행)
```bash
nohup conda run -n ccub2 python scripts/build_all_country_indices.py \
  2>&1 | tee /tmp/build_all_$(date +%Y%m%d_%H%M%S).log &
```

### 단계별 처리 (안전)
```bash
# 1. 먼저 Dry run으로 확인
python scripts/build_all_country_indices.py --dry-run

# 2. 특정 나라로 테스트
python scripts/build_all_country_indices.py --countries korea japan

# 3. 문제 없으면 전체 실행
python scripts/build_all_country_indices.py
```
