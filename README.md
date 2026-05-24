# static-rec-lab

`static-rec-lab`는 Generative Retrieval 추천 시스템에 YouTube `static_decoding` package를
적용해, 존재하는 item에 해당하는 Semantic ID sequence만 생성하도록 제약하는 실험용
레포지토리입니다.

## 핵심 메시지

> 추천 시스템을 단순 API로 만든 것이 아니라, Generative Retrieval,
> Constrained Decoding, Evaluation, Serving까지 구현한다.

사용자 행동 이력을 Transformer에 넣어 추천 item의 Semantic ID token sequence를 생성하고,
decoding 단계에서 유효하지 않은 sequence를 차단합니다. 추천 정확도 1등이 목표가 아니라,
Generative Retrieval에서 constrained decoding을 실제 데이터 파이프라인, 모델 학습, 평가,
API serving까지 연결해 검증하는 것이 목표입니다.

## 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| Dataset | MovieLens Latest Small, MovieLens 32M |
| Model | PyTorch Transformer encoder-decoder |
| Target | item_id가 아니라 고정 길이 Semantic ID token sequence |
| Constraint | YouTube `static_decoding` sparse mask kernel |
| Baseline | Popularity, item co-occurrence |
| Serving | FastAPI checkpoint-backed service |
| Validation | `make lint`, `make test`, generated evaluation reports |

## 빠른 검토 가이드

이 레포를 처음 볼 때는 아래 순서로 보면 됩니다.

1. [프로젝트 요약](reports/project_summary.md): 문제 정의, 구현 범위, 재현 경로를
   한 페이지로 정리했습니다.
2. [static_decoding 통합 리포트](reports/static_decoding_integration.md): YouTube
   `static_decoding` package를 어떤 경로로 직접 호출하는지 정리했습니다.
3. `reports/ml_32m_preprocess.md`: MovieLens 32M feedback-aware 전처리 통계를 확인합니다.
4. `reports/ml_32m_baseline.md`: baseline 추천 성능을 확인합니다.
5. `reports/ml_32m_semantic_id.md`: Semantic ID catalog와 STATIC index 생성 결과를 확인합니다.
6. `reports/ml_32m_generative.md`: teacher-forcing loss/accuracy와 unknown target 수를 확인합니다.
7. `reports/ml_32m_generative_eval.md`: constrained decoding 기반 ranking 성능과 invalid
   generation rate를 확인합니다.

로컬 검증:

```bash
uv sync --group dev
make check
```

## 구현 범위

- MovieLens feedback-aware sequential recommendation 데이터 파이프라인
- Recall@K, NDCG@K, MRR 평가 지표
- Popularity 및 item co-occurrence baseline
- item_id와 Semantic ID 간 codec
- hierarchical balanced k-means 기반 Semantic ID 생성
- naive trie constrained decoder 검증 구현
- `static_decoding` 기반 constrained decoder
- decoder validity check 및 선택 benchmark script
- FastAPI 기반 추천 endpoint
- 실험 결과와 trade-off를 설명하는 프로젝트 리포트

## 현재 상태

현재 레포지토리는 데이터셋 파이프라인, 평가 지표, baseline, Semantic ID codec,
Semantic ID 생성, Generative Retrieval 학습/평가 코드, naive trie constrained decoder,
`static_decoding` sparse transition decoder, decoder latency 및 throughput benchmark,
constrained beam search, 학습 checkpoint 기반 API service까지 구현된 상태입니다.
Latest Small은 빠른 개발과 smoke test 용도로 사용하고, 대표 산출물은 MovieLens 32M으로
재생성합니다.

전체 재현 명령:

```bash
make download-ml32m
make reproduce-ml32m
```

## MovieLens 32M 결과 요약

feedback-aware 전처리 기준으로 MovieLens 32M 전체 pipeline을 재실행한 대표 결과입니다.

| 항목 | 값 |
| --- | ---: |
| raw ratings | 32,000,204 |
| eligible full history interactions | 31,935,252 |
| train examples | 15,411,717 |
| Semantic ID catalog items | 54,711 |
| Popularity test Recall@20 | 0.069374 |
| Item co-occurrence test Recall@20 | 0.105408 |
| Generative Retrieval test Recall@20 | 0.134848 |
| Generative Retrieval test NDCG@20 | 0.064866 |
| Generative Retrieval test MRR@20 | 0.044822 |
| test invalid generation rate | 0.000000 |

## 재실행 산출물

현재 전처리는 target은 `rating >= 4.0` positive item만 사용하고, history에는 negative,
neutral, positive interaction을 모두 남깁니다. 이 기준을 바꾸면 baseline, Semantic ID,
generative checkpoint, 평가 리포트가 모두 달라지므로 전체 pipeline을 다시 실행해야 합니다.
checkpoint와 artifact는 로컬 산출물로만 사용하며 Git에는 커밋하지 않습니다.

```bash
make preprocess-ml32m
make baseline-ml32m
make semantic-ids-ml32m
make generative-ml32m
```

재실행 후 생성되는 주요 리포트:

- `reports/ml_32m_preprocess.md`: rating filter와 split 예제 수
- `reports/ml_32m_baseline.md`: Popularity, item co-occurrence baseline 성능
- `reports/ml_32m_semantic_id.md`: Semantic ID 생성 설정과 catalog 검증
- `reports/ml_32m_generative.md`: teacher-forcing loss/accuracy와 unknown target 수
- `reports/ml_32m_generative_eval.md`: STATIC constrained decoding 기반 ranking 평가

실제 latency benchmark는 필수 재현 경로에서 제외했습니다. 필요할 때만
`make benchmark-decoder`, `make benchmark-serving`, `make benchmark-api`를 따로 실행합니다.

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

`ratings.csv`를 사용자별 시간순 sequence로 정렬한 뒤 feedback-aware prefix-target pair로
변환합니다. target은 `rating >= 4.0` positive item만 사용하지만, history에는 negative,
neutral, positive interaction을 모두 유지하고 `history_feedback_ids`를 함께 저장합니다.

실제 데이터는 Git에 커밋하지 않습니다. 기본 명령은 빠른 개발용 MovieLens Latest Small을
받지만, 대표 산출물은 MovieLens 32M으로 생성합니다.

다운로드:

```bash
make download-movielens
```

기본 입력 경로:

```text
data/raw/ml-latest-small/ratings.csv
```

MovieLens 32M 다운로드:

```bash
make download-ml32m
```

실행:

```bash
RAW_RATINGS=data/raw/ml-latest-small/ratings.csv make preprocess
```

환경 변수로 입력/출력과 필터 기준을 바꿀 수 있습니다.

```bash
RAW_RATINGS=data/raw/ml-latest-small/ratings.csv \
PROCESSED_DIR=data/processed \
PREPROCESS_REPORT=reports/local/preprocess.md \
MIN_RATING=4.0 \
NEUTRAL_RATING=3.0 \
MIN_INTERACTIONS=5 \
MAX_HISTORY_LENGTH=50 \
make preprocess
```

MovieLens 32M 전처리:

```bash
make preprocess-ml32m
```

생성 파일:

```text
data/processed/train.parquet
data/processed/valid.parquet
data/processed/test.parquet
reports/local/preprocess.md
```

split 정책:

- 사용자별 interaction을 `timestamp` 기준으로 정렬합니다.
- `MIN_RATING` 이상 interaction만 추천 target 후보로 사용합니다.
- `NEUTRAL_RATING <= rating < MIN_RATING`은 neutral history로 유지합니다.
- `rating < NEUTRAL_RATING`은 negative history로 유지합니다.
- positive interaction 수가 `MIN_INTERACTIONS`보다 적은 사용자는 제외합니다.
- 각 사용자의 마지막 positive item은 test target으로 사용합니다.
- 마지막 직전 positive item은 valid target으로 사용합니다.
- 그 이전 positive target으로 train 예제를 만듭니다.
- parquet에는 full `history_item_ids`, `history_feedback_ids`, baseline/Semantic ID용
  `positive_history_item_ids`를 함께 저장합니다.

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
- Item co-occurrence baseline: 사용자 positive history item과 target item의 co-occurrence
  count로 개인화된 ranking을 만듭니다.

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
reports/ml_32m_baseline.md
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
make build-static-decoding-index
```

MovieLens 32M처럼 item 수가 많은 데이터셋에서는 capacity가 충분하도록 branching factor를
키워야 합니다.

```bash
make semantic-ids-ml32m
```

생성 파일:

```text
artifacts/semantic_id/semantic_ids.json
artifacts/semantic_id/static_decoding_index.npz
reports/ml_32m_semantic_id.md
```

`build-static-decoding-index`는 `static_decoding.csr_utils.build_static_index`가 만든
`packed_csr`, `csr_indptr`, `start_mask`, `dense_mask`, `dense_states`,
`layer_max_branches`를 npz artifact로 저장합니다.

## Generative Retrieval Model

사용자 full history item sequence와 feedback id sequence를 입력으로 받고 target item의
Semantic ID token sequence를 생성하는 Transformer encoder-decoder 모델을 제공합니다.
학습은 teacher forcing으로 진행하며, 평가는 validation loss, token accuracy, sequence
accuracy를 기록합니다.

학습과 teacher-forcing 평가는 parquet를 메모리에 모두 올리지 않고
`GenerativeParquetBatchIterableDataset`으로 batch 단위 streaming 처리합니다. 이 경로를 기본값으로
고정했기 때문에 MovieLens 32M과 빠른 smoke test가 같은 데이터 로딩 방식을 사용합니다.
대용량 학습도 기본값은 `GENERATIVE_NUM_WORKERS=0`입니다. 이 프로젝트의 Dataset은 이미
parquet에서 tensor batch를 만들어 넘기기 때문에, macOS/MPS 환경에서는 멀티프로세스
DataLoader가 대형 tensor batch를 프로세스 간 복사하면서 오히려 크게 느려질 수 있습니다.
필요하면 `GENERATIVE_NUM_WORKERS`와 `GENERATIVE_PREFETCH_FACTOR`를 올려 비교할 수 있지만,
기본 경로는 단일 프로세스 streaming입니다.

학습:

```bash
make train-generative
```

장비 자원을 더 적극적으로 쓰는 예시:

```bash
GENERATIVE_DEVICE=mps \
GENERATIVE_BATCH_SIZE=4096 \
GENERATIVE_PARQUET_BATCH_SIZE=131072 \
GENERATIVE_NUM_WORKERS=0 \
make train-generative
```

macOS/MPS에서는 `GENERATIVE_NUM_WORKERS=0` 경로가 가장 안정적입니다. worker를 늘리거나
AMP를 켜는 경로는 환경별 차이가 크므로, 필요할 때만 별도로 비교한 뒤 사용합니다.

학습 속도를 우선할 때는 training loop에서 batch별 token/sequence accuracy를 계산하지 않고
loss만 집계합니다. 학습 중 accuracy까지 보고 싶으면 `scripts/generative.py train`에
`--full-train-metrics`를 추가하면 됩니다.

평가:

```bash
make eval-generative
```

추천 ranking 평가:

```bash
make eval-generative-ranking
```

ranking 평가는 `static_decoding` PyTorch sparse mask kernel을 사용하는 decoder로 실행됩니다.
로컬 naive trie와 matrix decoder는 정확성 확인 및 벤치마크 비교용으로만 사용합니다.
`make eval-generative-ranking`의 기본 inference batch size는 `128`입니다.

`artifacts/semantic_id/static_decoding_index.npz`가 존재하면 `make eval-generative-ranking`은
해당 static_decoding index artifact를 자동으로 로드합니다. 직접 지정할 수도 있습니다.

```bash
uv run python scripts/generative.py eval-ranking \
  --static-decoding-index-path artifacts/semantic_id/static_decoding_index.npz \
  --inference-batch-size 128
```

생성 파일:

```text
artifacts/generative/model.pt
reports/ml_32m_generative.md
reports/ml_32m_generative_eval.md
```

MovieLens 32M 최종 학습/평가 예시:

```bash
make generative-ml32m
```

현재 모델 구현은 item embedding, feedback embedding, position embedding을 더한 encoder 입력,
학습/평가 루프, checkpoint format, `static_decoding` 기반 constrained beam search inference까지
제공합니다. API는 기본 checkpoint, Semantic ID, user history parquet, static_decoding index
artifact를 고정 경로에서 로드합니다. artifact가 없으면 명시적으로 실패합니다.

`make eval-generative`는 teacher-forcing 기준의 validation loss, token accuracy,
sequence accuracy를 측정합니다. `make eval-generative-ranking`은 실제 constrained beam
search로 추천 item ranking을 만든 뒤 baseline과 같은 Recall@K, NDCG@K, MRR@K 및 invalid
generation rate를 측정합니다. ranking 평가도 valid/test parquet를 batch 단위로 읽어 처리합니다.

## Constrained Decoding

Generative Retrieval 모델이 Semantic ID token을 생성할 때 존재하지 않는 item sequence를
만들지 못하도록 decoder 단계에서 다음 token 후보를 제한합니다.

기본 constrained decoder는 `static_decoding.csr_utils.build_static_index`가 만든 CSR sparse
index와 `static_decoding.decoding_pt.generate_and_apply_logprobs_mask`를 사용합니다. 로컬 naive
trie와 matrix decoder는 같은 mask가 나오는지 확인하는 검증용 구현입니다.

```python
from recsys.decoding import StaticDecodingIndex

index = StaticDecodingIndex.from_semantic_ids(
    [(12, 4, 81, 7), (12, 4, 82, 3)],
    vocab_size=128,
)
index.allowed_next_tokens([12, 4])
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
- `static_decoding` mask가 naive trie 결과와 일치하는지 테스트로 검증
- constrained beam search로 유효한 Semantic ID만 생성
- `static_decoding.decoding_pt.generate_and_apply_logprobs_mask` 후보 추출 경로 검증
- `static_decoding.decoding_jax.generate_and_apply_logprobs_mask` 후보 추출 경로 검증
- `static_decoding.decoding_pt.sparse_transition_torch` harness 호출 검증
- `static_decoding.decoding_jax.sparse_transition_jax` harness 호출 검증
- `static_decoding.decoding_pt.RandomModel` benchmark model 호출 검증
- `static_decoding.csr_utils.build_static_index` 산출물 npz 저장 CLI 제공
- `make benchmark-decoder`로 선택 latency와 throughput 리포트 생성

## 선택 벤치마크

Benchmark는 필수 재현 경로에서 제외했습니다. 필요할 때만 두 경로를 분리해서 봅니다.

- `benchmark-serving`: service layer를 직접 호출해 모델과 decoder 중심 latency를 측정합니다.
- `benchmark-api`: ASGI test client로 FastAPI endpoint를 호출해 routing, validation,
  serialization, JSON decode 비용까지 포함합니다.

```bash
make benchmark-serving
```

```bash
make benchmark-api
```

serving benchmark는 기본 artifact 경로를 고정 사용합니다.

생성 파일:

```text
reports/ml_32m_serving_benchmark.md
reports/ml_32m_http_serving_benchmark.md
```

## Recommendation API

Recommendation API는 checkpoint 기반 generative retrieval service로만 동작합니다.
API는 기본 artifact 경로를 고정 사용합니다.

실행:

```bash
make serve-api
```

사용되는 artifact:

```text
artifacts/generative/ml-32m/model.pt
artifacts/semantic_id/ml-32m/semantic_ids.json
artifacts/semantic_id/ml-32m/static_decoding_index.npz
data/processed/ml-32m/valid.parquet
data/raw/ml-32m/movies.csv
```

API serving은 `static_decoding` 기반 STATIC decoder를 사용합니다.
static_decoding `build_static_index` 산출물 `.npz`를 직접 로드하고,
응답 item title은 MovieLens `movies.csv` metadata에서 조회합니다.
사용자의 입력 history에 이미 포함된 item은 추천 응답에서 제외합니다.

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

## 시스템 아키텍처

전체 흐름은 데이터 전처리, Semantic ID 생성, 모델 학습, constrained decoding, serving으로
나뉩니다.

```text
MovieLens ratings.csv
  -> sequential recommendation split
  -> train / valid / test parquet

train.parquet
  -> item interaction embedding
  -> hierarchical balanced k-means
  -> item_id <-> Semantic ID codec

user history
  -> item index sequence
  -> Transformer encoder-decoder
  -> Semantic ID token logits
  -> static_decoding constrained beam search
  -> valid Semantic ID only
  -> item_id recommendation

FastAPI
  -> RecommendationService
  -> model-backed service
  -> /recommendations/users/{user_id}?k=20
```

모델 코드는 `recsys.models`, constrained decoding은 `recsys.decoding`, 추천 평가와 benchmark는
`recsys.evaluation` 및 `recsys.benchmark`, API는 `apps.api`에 분리했습니다. API는 service
interface를 유지하지만 런타임에서는 checkpoint, Semantic ID codec, static_decoding index,
user history parquet로 구성된 model-backed service를 사용합니다.

## 설계 결정과 Trade-off

Semantic ID:

- item을 직접 분류하는 대신 고정 길이 Semantic ID token sequence를 생성합니다.
- hierarchical balanced k-means를 사용해 interaction embedding 기반 token path를 만들었습니다.
- MovieLens 32M에서는 item 수가 많아져 depth와 branching factor의 capacity가 중요합니다.
  `ml-32m` 검증에서는 Semantic ID 길이 4, branching factor 32로 capacity를 확보했습니다.

Constrained decoding:

- naive trie decoder와 로컬 matrix decoder는 정답 동작을 검증하고 benchmark 비교군을 만들기 위한 구현입니다.
- 실제 STATIC decoder는 YouTube `static-constraint-decoding`의 `static_decoding` package를 사용합니다.
  `build_static_index`, `generate_and_apply_logprobs_mask`, `sparse_transition_torch`를 직접 호출합니다.
  실제 추천 모델 inference는 사용자 history와 decoder prefix를 함께 넣어야 하므로,
  beam search loop만 모델 입력 형태에 맞추고 constraint 계산은 `static_decoding` sparse mask
  kernel을 사용합니다.

Generative model:

- 현재 모델은 작은 Transformer encoder-decoder를 1 epoch 학습한 baseline입니다.
- `ml-32m` 최종 평가에서는 Popularity와 item co-occurrence baseline보다 높은 ranking 성능을
  기록했습니다. 다만 모델 크기와 학습 epoch는 아직 초기 검증용 baseline 수준입니다.
- constrained decoding 적용 후 invalid generation rate가 0으로 유지됩니다.
- 이 프로젝트의 핵심은 SOTA 추천 정확도가 아니라, generative retrieval에서 존재하지 않는 item
  sequence 생성을 막고 그 latency/validity를 측정하는 것입니다.

Serving benchmark:

- `benchmark-serving`은 HTTP 서버를 띄우지 않고 `RecommendationService.recommend()`를 직접
  호출합니다.
- 따라서 측정값은 모델과 decoder 중심의 serving 비용이며, 실제 HTTP latency에는 FastAPI
  validation, serialization, network overhead가 추가됩니다.
- `benchmark-api`는 ASGI test client로 FastAPI endpoint를 호출합니다. Network hop은 제외하지만
  routing, request validation, response model serialization, client-side JSON decode 비용을
  포함합니다.

## 한계와 다음 개선

- Generative model은 작은 설정으로 1 epoch만 학습했습니다. 더 긴 학습, larger model, learning
  rate schedule을 적용하면 ranking 품질을 더 확인할 수 있습니다.
- Semantic ID는 interaction embedding 기반입니다. 영화 metadata, text embedding, collaborative
  embedding을 결합하면 token hierarchy 품질을 개선할 수 있습니다.
- 현재 ranking 평가는 beam search 결과만 사용합니다. score calibration, diversity constraint,
  candidate reranking은 이번 범위에서 제외했습니다.
- `static_decoding` package는 GitHub commit으로 고정해 index builder, PyTorch/JAX sparse mask
  kernel, PyTorch/JAX sparse transition harness, benchmark용 RandomModel을 통합했습니다. 다만
  TPU, GPU, `torch.compile` 기반 benchmark는 별도 환경 검증 대상으로 남겼습니다.
- HTTP endpoint benchmark는 ASGI test client 기준입니다. 실제 외부 client 기준 latency는 이번
  프로젝트 범위에서 제외했습니다.

## static_decoding 통합 방식

이 프로젝트는 YouTube `static-constraint-decoding` repository를 복사하지 않고,
`static-decoding` GitHub dependency를 commit
`c24f9dc8b9b8045716fff7ef750a1f0cb31c6f57`로 고정해 사용합니다. naive trie와
로컬 matrix decoder는 정확성 확인과 벤치마크 비교군을 위한 검증용 코드입니다.

`static_decoding` package는 JAX/TPU와 PyTorch/GPU를 대상으로 `start_mask`, `dense_mask`,
`dense_states`, `packed_csr`, `csr_indptr`, `layer_max_branches`를 사용하는 dense/sparse
hybrid index를 구성합니다. `static-rec-lab`의 STATIC decoder는
`static_decoding.csr_utils.build_static_index`, PyTorch/JAX `generate_and_apply_logprobs_mask`,
PyTorch/JAX sparse transition harness, 벤치마크용 `RandomModel`을 직접 호출하고,
`make build-static-decoding-index`로
static_decoding index artifact를 저장합니다.

자세한 통합 방식은 [reports/static_decoding_integration.md](reports/static_decoding_integration.md)에
정리했습니다.

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
8. 프로젝트 리포트

## 구현 원칙

- 데이터셋, 평가 지표, baseline을 먼저 고정한 뒤 모델 학습으로 넘어갑니다.
- decoder는 naive trie 구현으로 정답 동작을 먼저 검증한 뒤 vectorized/static 방식으로 확장합니다.
- benchmark는 항상 naive 구현과 결과 동일성을 확인한 뒤 지연시간을 비교합니다.
- 모델 코드는 API 코드와 분리합니다.
- 문서와 리포트는 한국어로 정리합니다.
