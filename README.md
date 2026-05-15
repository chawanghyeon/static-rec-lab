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
- hierarchical balanced k-means 기반 Semantic ID 생성
- naive trie constrained decoder
- STATIC-style matrix constrained decoder
- decoder latency 및 throughput benchmark
- FastAPI 기반 추천 endpoint
- 실험 결과와 trade-off를 설명하는 포트폴리오 리포트

## 현재 단계

현재 레포지토리는 데이터셋 파이프라인, 평가 지표, baseline, Semantic ID codec,
Semantic ID 생성, Generative Retrieval 학습/평가 코드, naive trie constrained decoder,
STATIC-style sparse transition matrix decoder, decoder latency 및 throughput benchmark,
constrained beam search, 학습 checkpoint 기반 API service와 mock service 경로까지 구현된
상태입니다.
다음 단계는 실제 MovieLens 학습 checkpoint를 만들고 serving 품질/지연시간 리포트를 확정하는
것입니다.

완료 기준:

```bash
make lint
make test
make format
make train-baseline
make eval-baseline
make build-semantic-ids
make validate-semantic-ids
make train-generative
make eval-generative
make eval-generative-ranking
make benchmark-decoder
make benchmark-serving
make serve-api
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

## 평가 지표

추천 ranking 평가는 `recsys.evaluation`의 순수 함수로 계산합니다.

- `Recall@K`: 정답 item 중 top-K 추천에 포함된 비율
- `NDCG@K`: 정답 item이 ranking 상위에 있을수록 높은 점수를 주는 discounted gain
- `MRR`: 첫 번째 정답 item이 등장한 rank의 reciprocal

중복 추천은 여러 번 맞힌 것으로 세지 않습니다. 다만 중복 item도 ranking 위치를 차지하므로
NDCG에서는 낮은 순위의 정답처럼 penalty가 반영됩니다.

## Baseline

Generative Retrieval 모델과 비교하기 위한 baseline입니다.

- Popularity baseline: `train.parquet`의 `target_item_id` 빈도로 item ranking을 만듭니다.
- Item co-occurrence baseline: 사용자 history item과 target item의 co-occurrence count로
  개인화된 ranking을 만듭니다.

두 baseline 모두 평가 시 사용자 history에 이미 등장한 item은 추천에서 제외합니다.

학습:

```bash
make train-baseline
```

평가:

```bash
make eval-baseline
```

생성 파일:

```text
artifacts/baseline/popularity.json
artifacts/baseline/item_knn.json
reports/baseline.md
```

## Semantic ID Codec

Generative Retrieval 모델은 item을 직접 분류하지 않고, item에 대응하는 고정 길이 token
sequence인 Semantic ID를 생성합니다. codec은 clustering 단계에서 만들어진 mapping을 받아
다음 변환을 담당합니다.

```text
item_id -> semantic_id
semantic_id -> item_id
```

현재 codec은 다음 조건을 검증합니다.

- 모든 Semantic ID는 비어 있지 않아야 합니다.
- 모든 Semantic ID는 같은 길이여야 합니다.
- 서로 다른 item이 같은 Semantic ID를 공유하면 오류로 처리합니다.
- 알 수 없는 item_id 또는 Semantic ID 조회는 명시적인 예외로 처리합니다.
- JSON 저장/로드를 지원합니다.

## Semantic ID 생성

`train.parquet`의 전체 history-target co-occurrence를 배치 단위로 읽고, source item을
hashed projection으로 누적해 dense item embedding을 만든 뒤 hierarchical balanced k-means로
고정 길이 Semantic ID를 생성합니다. 각 subtree의 capacity를 넘지 않도록 cluster 순서를
balanced chunk로 나누기 때문에 모든 item에 고유한 Semantic ID를 부여할 수 있습니다.

실행:

```bash
make build-semantic-ids
make validate-semantic-ids
```

MovieLens 32M처럼 item 수가 많은 데이터셋에서는 capacity가 충분하도록 branching factor를
키워야 합니다.

```bash
PROCESSED_DIR=data/processed/ml-32m \
SEMANTIC_ID_PATH=artifacts/semantic_id/ml-32m/semantic_ids.json \
SEMANTIC_ID_REPORT=reports/ml_32m_semantic_id.md \
SEMANTIC_ID_BRANCHING_FACTOR=32 \
make build-semantic-ids
```

생성 파일:

```text
artifacts/semantic_id/semantic_ids.json
reports/semantic_id.md
```

대용량 검증 결과는 `reports/ml_32m_validation.md`에 정리합니다.

## Generative Retrieval Model

사용자 history item sequence를 입력으로 받고 target item의 Semantic ID token sequence를
생성하는 Transformer encoder-decoder 모델을 제공합니다. 학습은 teacher forcing으로 진행하며,
평가는 validation loss, token accuracy, sequence accuracy를 기록합니다.

학습:

```bash
make train-generative
```

평가:

```bash
make eval-generative
```

추천 ranking 평가:

```bash
make eval-generative-ranking
```

생성 파일:

```text
artifacts/generative/model.pt
reports/generative.md
reports/generative_eval.md
```

현재 모델 구현은 학습/평가 루프, checkpoint format, STATIC-style constrained beam search
inference까지 제공합니다. API는 checkpoint, Semantic ID, user history parquet 경로가 모두
환경변수로 주어지면 model-backed service를 사용합니다. 세 환경변수가 모두 없을 때만
deterministic mock service를 사용하고, 일부만 설정되었거나 파일이 없으면 명시적으로
실패합니다.

`make eval-generative`는 teacher-forcing 기준의 validation loss, token accuracy,
sequence accuracy를 측정합니다. `make eval-generative-ranking`은 실제 constrained beam
search로 추천 item ranking을 만든 뒤 baseline과 같은 Recall@K, NDCG@K, MRR@K 및 invalid
generation rate를 측정합니다.

## Constrained Decoding

Generative Retrieval 모델이 Semantic ID token을 생성할 때 존재하지 않는 item sequence를
만들지 못하도록 decoder 단계에서 다음 token 후보를 제한합니다.

naive trie decoder는 유효한 Semantic ID sequence를 prefix tree에 삽입하고, 입력 prefix 뒤에
올 수 있는 token만 반환합니다. STATIC-style decoder는 이 trie의 state transition을 CSR sparse
matrix로 flatten해서 batch state update와 allowed-token mask 생성을 지원합니다.

```python
from recsys.decoding import SemanticIdTrie, StaticTransitionMatrixDecoder

trie = SemanticIdTrie([(12, 4, 81, 7), (12, 4, 82, 3)])
trie.allowed_next_tokens([12, 4])
# (81, 82)

decoder = StaticTransitionMatrixDecoder.from_trie(trie)
decoder.allowed_next_tokens([12, 4])
# (81, 82)
```

보장하는 동작:

- 유효한 Semantic ID sequence 삽입
- `allowed_next_tokens(prefix)`로 다음 후보 반환
- 존재하지 않는 prefix에는 빈 후보 반환
- 길이가 다른 sequence도 trie에 저장 가능
- state 기반 transition API 제공
- trie node를 integer state로 flatten
- CSR sparse transition matrix로 유효 transition 표현
- batch prefix state update 지원
- STATIC-style mask가 naive trie 결과와 일치하는지 테스트로 검증
- constrained beam search로 유효한 Semantic ID만 생성
- `make benchmark-decoder`로 batch size별 latency와 throughput 리포트 생성

기본 benchmark는 synthetic Semantic ID로 실행되며 결과는 다음 파일에 저장됩니다.

```text
reports/decoder_benchmark.md
```

## Serving 벤치마크

Recommendation API의 service layer를 직접 호출해 사용자별 추천 생성 latency를 측정합니다.
HTTP 서버를 띄우지 않기 때문에 FastAPI 직렬화, validation, network overhead를 제외한 모델 및
decoder 비용을 볼 수 있습니다.

```bash
make benchmark-serving
```

model-backed service를 측정하려면 API 실행과 같은 `STATIC_REC_*` 환경변수를 설정하고
`SERVING_BENCHMARK_SERVICE=environment`로 실행합니다.

```bash
STATIC_REC_GENERATIVE_CHECKPOINT=artifacts/generative/model.pt \
STATIC_REC_SEMANTIC_ID_PATH=artifacts/semantic_id/semantic_ids.json \
STATIC_REC_USER_HISTORY_PARQUET=data/processed/valid.parquet \
SERVING_BENCHMARK_SERVICE=environment \
make benchmark-serving
```

생성 파일:

```text
reports/serving_benchmark.md
```

## Recommendation API

학습된 generative retrieval model이 준비되기 전에는 deterministic mock recommender로 serving
contract를 검증합니다. 아래 환경변수를 모두 설정하면 checkpoint 기반 generative retrieval
service가 대신 사용됩니다.

실행:

```bash
make serve-api
```

Model-backed 실행 환경변수:

```bash
STATIC_REC_GENERATIVE_CHECKPOINT=artifacts/generative/model.pt \
STATIC_REC_SEMANTIC_ID_PATH=artifacts/semantic_id/semantic_ids.json \
STATIC_REC_USER_HISTORY_PARQUET=data/processed/valid.parquet \
make serve-api
```

Endpoint:

```http
GET /recommendations/users/{user_id}?k=20
```

응답 필드:

- `user_id`: 요청한 사용자 ID
- `model`: 현재 추천 모델 이름
- `decoder`: decoder 종류
- `items`: 추천 item 목록
- `latency_ms`: endpoint 내부 추천 생성 지연시간

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
