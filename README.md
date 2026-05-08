# static-rec-lab

`static-rec-lab`는 STATIC-style constrained decoding을 추천 시스템 맥락에서 구현하고
검증하기 위한 실험용 레포지토리입니다.

목표는 단순 추천 API를 만드는 것이 아니라, 사용자 행동 이력을 기반으로 Transformer가
추천 item의 Semantic ID를 생성하고, 존재하는 item에 해당하는 token sequence만 생성되도록
decoding을 제약하는 전체 흐름을 구현하는 것입니다.

## 핵심 메시지

> 추천 시스템을 단순 API로 만든 것이 아니라, Generative Retrieval,
> Constrained Decoding, Benchmark, Serving까지 구현한다.

이 프로젝트는 추천 정확도 1등을 목표로 하지 않습니다. 핵심은 Generative Retrieval에서
유효한 item sequence만 생성하도록 제약하는 decoder를 구현하고, naive trie 방식과
STATIC-style sparse transition 방식의 정확성 및 지연시간 차이를 실험으로 보여주는 것입니다.

## 구현 범위

- MovieLens sequential recommendation 데이터 파이프라인
- Recall@K, NDCG@K, MRR 평가 지표
- Popularity 및 item co-occurrence baseline
- item_id와 Semantic ID 간 codec
- hierarchical clustering 기반 Semantic ID 생성
- naive trie constrained decoder
- STATIC-style matrix constrained decoder
- decoder latency 및 throughput benchmark
- FastAPI 기반 추천 endpoint
- 실험 결과와 trade-off를 설명하는 포트폴리오 리포트

## 현재 단계

현재 레포지토리는 Milestone 0, 즉 개발 환경과 프로젝트 구조를 고정하는 단계입니다.

완료 기준:

```bash
make lint
make test
make format
```

## 개발 환경

- Python 3.12
- uv
- ruff
- mypy
- pytest

설치:

```bash
uv sync --group dev
```

검사:

```bash
make lint
make test
```

포맷:

```bash
make format
```

## MovieLens 전처리

`ratings.csv`를 사용자별 시간순 interaction sequence로 정렬한 뒤 sequential recommendation용
prefix-target pair로 변환합니다.

실제 데이터는 Git에 커밋하지 않습니다. 대신 GroupLens의 MovieLens Latest Small 데이터를
로컬 `data/raw/` 아래에 다운로드해서 사용합니다.

다운로드:

```bash
make download-movielens
```

기본 입력 경로:

```text
data/raw/ml-latest-small/ratings.csv
```

실행:

```bash
RAW_RATINGS=data/raw/ml-latest-small/ratings.csv make preprocess
```

환경 변수로 입력/출력과 필터 기준을 바꿀 수 있습니다.

```bash
RAW_RATINGS=data/raw/ml-latest-small/ratings.csv \
PROCESSED_DIR=data/processed \
MIN_INTERACTIONS=5 \
MAX_HISTORY_LENGTH=50 \
make preprocess
```

생성 파일:

```text
data/processed/train.parquet
data/processed/valid.parquet
data/processed/test.parquet
```

split 정책:

- 사용자별 interaction을 `timestamp` 기준으로 정렬합니다.
- `MIN_INTERACTIONS`보다 interaction 수가 적은 사용자는 제외합니다.
- 각 사용자 sequence의 마지막 item은 test target으로 사용합니다.
- 마지막 직전 item은 valid target으로 사용합니다.
- 그 이전 prefix-target pair는 train 예제로 사용합니다.

## 프로젝트 구조

```text
static-rec-lab/
├── apps/
│   └── api/
├── recsys/
│   ├── data/
│   ├── baseline/
│   ├── semantic_id/
│   ├── models/
│   ├── decoding/
│   ├── evaluation/
│   └── benchmark/
├── scripts/
├── tests/
├── reports/
├── data/
│   ├── raw/
│   └── processed/
├── Makefile
├── docker-compose.yml
└── pyproject.toml
```

## 마일스톤

1. 레포 초기화
2. MovieLens 데이터셋 파이프라인
3. baseline recommender
4. Semantic ID 생성
5. Generative Retrieval 모델
6. constrained decoding
7. Recommendation API
8. 포트폴리오 리포트

## 구현 원칙

- 데이터셋, 평가 지표, baseline을 먼저 고정한 뒤 모델 학습으로 넘어갑니다.
- decoder는 naive trie 구현으로 정답 동작을 먼저 검증한 뒤 vectorized/static 방식으로 확장합니다.
- benchmark는 항상 naive 구현과 결과 동일성을 확인한 뒤 지연시간을 비교합니다.
- 모델 코드는 API 코드와 분리합니다.
- 문서와 리포트는 한국어로 정리합니다.
