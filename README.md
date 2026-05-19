# static-rec-lab

`static-rec-lab`는 YouTube `static-constraint-decoding`의 `static_decoding` 패키지를
추천 시스템 맥락에 적용하고 검증하기 위한 실험용 레포지토리입니다.

목표는 단순 추천 API를 만드는 것이 아니라, 사용자 행동 이력을 기반으로 Transformer가
추천 item의 Semantic ID를 생성하고, 존재하는 item에 해당하는 token sequence만 생성되도록
decoding을 제약하는 전체 흐름을 구현하는 것입니다.

## 핵심 메시지

> 추천 시스템을 단순 API로 만든 것이 아니라, Generative Retrieval,
> Constrained Decoding, Benchmark, Serving까지 구현한다.

이 프로젝트는 추천 정확도 1등을 목표로 하지 않습니다. 핵심은 Generative Retrieval에서
유효한 item sequence만 생성하도록 제약하는 decoder를 적용하고, naive trie 방식과
`static_decoding` sparse transition 방식의 정확성 및 지연시간 차이를 실험으로 보여주는 것입니다.

## 빠른 검토 가이드

이 레포를 처음 볼 때는 아래 순서로 보면 됩니다.

1. [포트폴리오 요약](reports/portfolio_summary.md): 문제 정의, 구현 범위, 핵심 수치, 해석을
   한 페이지로 정리했습니다.
2. [static_decoding 통합 리포트](reports/static_decoding_integration.md): YouTube
   `static_decoding` package를 어떤 경로로 직접 호출하는지 정리했습니다.
3. [MovieLens 32M 검증](reports/ml_32m_validation.md): 대용량 전처리, baseline,
   Semantic ID 생성 결과를 확인합니다.
4. [ml-32m decoder benchmark](reports/ml_32m_decoder_benchmark.md): naive trie, 검증용 matrix,
   `static_decoding` PyTorch/JAX kernel 및 harness의 validity와 latency를 비교합니다.
5. [ml-32m Generative Retrieval 추천 평가](reports/ml_32m_generative_eval.md):
   constrained decoding을 적용한 추천 ranking 성능과 invalid generation rate를 확인합니다.
6. [ml-32m Serving benchmark](reports/ml_32m_serving_benchmark.md): model-backed recommendation
   service의 latency를 확인합니다.

최소 검증 명령:

```bash
uv sync --group dev
make lint
make test
make benchmark-decoder
```

## 구현 범위

- MovieLens sequential recommendation 데이터 파이프라인
- Recall@K, NDCG@K, MRR 평가 지표
- Popularity 및 item co-occurrence baseline
- item_id와 Semantic ID 간 codec
- hierarchical balanced k-means 기반 Semantic ID 생성
- naive trie constrained decoder 검증 구현
- `static_decoding` 기반 constrained decoder
- decoder latency 및 throughput benchmark
- FastAPI 기반 추천 endpoint
- 실험 결과와 trade-off를 설명하는 포트폴리오 리포트

## 현재 단계

현재 레포지토리는 데이터셋 파이프라인, 평가 지표, baseline, Semantic ID codec,
Semantic ID 생성, Generative Retrieval 학습/평가 코드, naive trie constrained decoder,
`static_decoding` sparse transition decoder, decoder latency 및 throughput benchmark,
constrained beam search, 학습 checkpoint 기반 API service와 mock service 경로까지 구현된
상태입니다.
최종 포트폴리오 수치는 MovieLens 32M 기준으로 생성했습니다. Latest Small은 빠른 개발과
smoke test 용도로만 남기고, README의 대표 성능 수치는 `ml-32m` 결과를 사용합니다.

완료 기준:

```bash
make lint
make test
make format
make train-baseline
make eval-baseline
make build-semantic-ids
make validate-semantic-ids
make build-static-decoding-index
make train-generative
make eval-generative
make eval-generative-ranking
make benchmark-decoder
make benchmark-serving
make serve-api
```

## 실험 결과 요약

MovieLens 32M 전처리 결과와 Semantic ID artifact를 사용해 Generative Retrieval 모델을
1 epoch 학습했습니다. checkpoint 파일은 로컬 산출물로만 사용하며 Git에는 커밋하지 않습니다.

Teacher-forcing 평가:

| split | examples | loss | token accuracy | sequence accuracy |
| --- | ---: | ---: | ---: | ---: |
| valid | 200,873 | 1.797266 | 0.409831 | 0.013222 |

추천 ranking 평가:

| model | split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | test | 0.035138 | 0.058100 | 0.016817 | 0.022527 | 0.012848 |
| Item co-occurrence | test | 0.063748 | 0.099016 | 0.034091 | 0.042943 | 0.027565 |
| Generative Retrieval + static_decoding | test | 0.098195 | 0.127978 | 0.054127 | 0.061858 | 0.043031 |

생성 품질:

- valid invalid generation rate: `0.000000`
- test invalid generation rate: `0.000000`
- test unknown targets: `102`
- full ranking inference: valid `285,347.85 ms`, test `285,259.78 ms`
- ranking inference batch size: `128`

Decoder benchmark:

- `ml-32m` Semantic ID 84,259개, trie state 86,980개 기준으로 측정했습니다.
- batch size 512 기준 검증용 matrix mask 생성은 naive trie 대비 `6.66x` 빠릅니다.
- 모든 sampled state batch에서 naive trie와 검증용 matrix decoder mask 일치를 확인했습니다.
- `static_decoding.decoding_pt.generate_and_apply_logprobs_mask` 후보 추출과
  `static_decoding.decoding_pt.sparse_transition_torch` harness도 별도 benchmark에서 호출합니다.
  harness의 dummy model도 `static_decoding.decoding_pt.RandomModel`을 사용하며,
  생성된 Semantic ID가 모두 유효한지 검증합니다.

Serving benchmark:

- `static_decoding` decoder를 사용하는 `ml-32m` model-backed service 기준 `k=20`,
  user_id `1..10000`, batch size `1, 32, 128`을 측정했습니다.
- batch size 128 기준 평균 latency는 `11.3422 ms`, p95 latency는 `14.0557 ms`,
  throughput은 `88.15 req/s`입니다.
- 요청 단위 serving benchmark에서는 MPS보다 CPU가 더 빨랐습니다. 작은 단건 추론은 device
  전환 비용이 커서 CPU 대표 수치를 사용합니다.

해석:

- 현재 generative model은 1 epoch의 작은 Transformer baseline이지만, `ml-32m`에서는
  Popularity와 item co-occurrence baseline보다 높은 Recall@20/NDCG@20을 기록했습니다.
- constrained decoding 적용 후 invalid generation rate가 0으로 유지되어, 존재하지 않는
  item Semantic ID를 추천하지 않는다는 핵심 목표도 만족합니다.
- portfolio 관점에서 핵심 비교 대상은 추천 정확도 1등이 아니라, baseline 추천 성능과
  constrained decoding latency/validity를 함께 제시하는 것입니다.

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

실제 데이터는 Git에 커밋하지 않습니다. 기본 명령은 빠른 개발용 MovieLens Latest Small을
받지만, 최종 포트폴리오 수치는 MovieLens 32M으로 생성했습니다.

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
MOVIELENS_DATASET=ml-32m \
MOVIELENS_URL=https://files.grouplens.org/datasets/movielens/ml-32m.zip \
make download-movielens
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

MovieLens 32M 전처리:

```bash
RAW_RATINGS=data/raw/ml-32m/ratings.csv \
PROCESSED_DIR=data/processed/ml-32m \
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
make build-static-decoding-index
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
artifacts/semantic_id/static_decoding_index.npz
reports/semantic_id.md
```

`build-static-decoding-index`는 `static_decoding.csr_utils.build_static_index`가 만든
`packed_csr`, `csr_indptr`, `start_mask`, `dense_mask`, `dense_states`,
`layer_max_branches`를 npz artifact로 저장합니다.

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

ranking 평가는 `static_decoding` PyTorch sparse mask kernel을 사용하는 decoder로 실행됩니다.
로컬 naive trie와 matrix decoder는 정확성 확인 및 벤치마크 비교용으로만 사용합니다.
`make eval-generative-ranking`의 기본 inference batch size는 `128`입니다.

`artifacts/semantic_id/static_decoding_index.npz`가 존재하면 `make eval-generative-ranking`은
해당 static_decoding index artifact를 자동으로 로드합니다. 직접 지정할 수도 있습니다.

```bash
uv run python scripts/eval_generative_ranking.py \
  --static-decoding-index-path artifacts/semantic_id/static_decoding_index.npz \
  --inference-batch-size 128
```

생성 파일:

```text
artifacts/generative/model.pt
reports/generative.md
reports/generative_eval.md
```

MovieLens 32M 최종 학습/평가 예시:

```bash
uv run python scripts/train_generative.py \
  --train-parquet data/processed/ml-32m/train.parquet \
  --valid-parquet data/processed/ml-32m/valid.parquet \
  --semantic-id-path artifacts/semantic_id/ml-32m/semantic_ids.json \
  --output-path artifacts/generative/ml-32m/model.pt \
  --epochs 1 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --max-history-length 50 \
  --streaming \
  --parquet-batch-size 131072 \
  --device auto \
  --log-every-batches 250

uv run python scripts/eval_generative_ranking.py \
  --checkpoint-path artifacts/generative/ml-32m/model.pt \
  --semantic-id-path artifacts/semantic_id/ml-32m/semantic_ids.json \
  --valid-parquet data/processed/ml-32m/valid.parquet \
  --test-parquet data/processed/ml-32m/test.parquet \
  --report-path reports/ml_32m_generative_eval.md \
  --beam-size 20 \
  --inference-batch-size 128 \
  --static-decoding-index-path artifacts/semantic_id/ml-32m/static_decoding_index.npz \
  --device auto
```

현재 모델 구현은 학습/평가 루프, checkpoint format, `static_decoding` 기반 constrained
beam search inference까지 제공합니다. API는 checkpoint, Semantic ID, user history parquet 경로가 모두
환경변수로 주어지면 model-backed service를 사용합니다. 세 환경변수가 모두 없을 때만
deterministic mock service를 사용하고, 일부만 설정되었거나 파일이 없으면 명시적으로 실패합니다.

`make eval-generative`는 teacher-forcing 기준의 validation loss, token accuracy,
sequence accuracy를 측정합니다. `make eval-generative-ranking`은 실제 constrained beam
search로 추천 item ranking을 만든 뒤 baseline과 같은 Recall@K, NDCG@K, MRR@K 및 invalid
generation rate를 측정합니다.

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
- `make benchmark-decoder`로 batch size별 latency와 throughput 리포트 생성

기본 benchmark는 synthetic Semantic ID로 실행되며 결과는 다음 파일에 저장됩니다.

```text
reports/decoder_benchmark.md
```

MovieLens 32M Semantic ID 기준 benchmark:

```bash
uv run python scripts/benchmark_decoder.py \
  --semantic-id-path artifacts/semantic_id/ml-32m/semantic_ids.json \
  --report-path reports/ml_32m_decoder_benchmark.md \
  --batch-sizes 1 32 128 512
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
STATIC_REC_STATIC_DECODING_INDEX_PATH=artifacts/semantic_id/static_decoding_index.npz \
STATIC_REC_USER_HISTORY_PARQUET=data/processed/valid.parquet \
SERVING_BENCHMARK_SERVICE=environment \
make benchmark-serving
```

MovieLens 32M checkpoint 기준:

```bash
STATIC_REC_GENERATIVE_CHECKPOINT=artifacts/generative/ml-32m/model.pt \
STATIC_REC_SEMANTIC_ID_PATH=artifacts/semantic_id/ml-32m/semantic_ids.json \
STATIC_REC_STATIC_DECODING_INDEX_PATH=artifacts/semantic_id/ml-32m/static_decoding_index.npz \
STATIC_REC_USER_HISTORY_PARQUET=data/processed/ml-32m/valid.parquet \
STATIC_REC_DEVICE=cpu \
SERVING_BENCHMARK_SERVICE=environment \
SERVING_BENCHMARK_REPORT=reports/ml_32m_serving_benchmark.md \
make benchmark-serving
```

생성 파일:

```text
reports/serving_benchmark.md
reports/ml_32m_serving_benchmark.md
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
STATIC_REC_STATIC_DECODING_INDEX_PATH=artifacts/semantic_id/static_decoding_index.npz \
STATIC_REC_USER_HISTORY_PARQUET=data/processed/valid.parquet \
make serve-api
```

API serving은 `static_decoding` 기반 STATIC decoder를 사용합니다.
`STATIC_REC_STATIC_DECODING_INDEX_PATH`는 선택값이지만, 지정하면 static_decoding
`build_static_index` 산출물 `.npz`를 직접 로드합니다.

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
  -> Mock service or model-backed service
  -> /recommendations/users/{user_id}?k=20
```

모델 코드는 `recsys.models`, constrained decoding은 `recsys.decoding`, 추천 평가와 benchmark는
`recsys.evaluation` 및 `recsys.benchmark`, API는 `apps.api`에 분리했습니다. API는 service
interface를 통해 mock service와 model-backed service를 교체할 수 있으므로, 학습 checkpoint가
없어도 endpoint contract를 테스트할 수 있습니다.

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
  기록했습니다. 다만 모델 크기와 학습 epoch는 아직 포트폴리오 검증용 baseline 수준입니다.
- constrained decoding 적용 후 invalid generation rate가 0으로 유지됩니다.
- 이 프로젝트의 핵심은 SOTA 추천 정확도가 아니라, generative retrieval에서 존재하지 않는 item
  sequence 생성을 막고 그 latency/validity를 측정하는 것입니다.

Serving benchmark:

- `benchmark-serving`은 HTTP 서버를 띄우지 않고 `RecommendationService.recommend()`를 직접
  호출합니다.
- 따라서 측정값은 모델과 decoder 중심의 serving 비용이며, 실제 HTTP latency에는 FastAPI
  validation, serialization, network overhead가 추가됩니다.

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
- serving benchmark는 service 직접 호출 기준입니다. 실제 API endpoint benchmark는 별도 HTTP
  client 기반 측정으로 확장할 수 있습니다.

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
8. 포트폴리오 리포트

## 구현 원칙

- 데이터셋, 평가 지표, baseline을 먼저 고정한 뒤 모델 학습으로 넘어갑니다.
- decoder는 naive trie 구현으로 정답 동작을 먼저 검증한 뒤 vectorized/static 방식으로 확장합니다.
- benchmark는 항상 naive 구현과 결과 동일성을 확인한 뒤 지연시간을 비교합니다.
- 모델 코드는 API 코드와 분리합니다.
- 문서와 리포트는 한국어로 정리합니다.
