# 포트폴리오 요약

## 한 줄 요약

`static-rec-lab`는 MovieLens 사용자 행동 이력으로 Transformer가 추천 item의 Semantic ID를
생성하고, YouTube `static_decoding` package로 존재하는 item sequence만 생성하도록 제약하는
Generative Retrieval 추천 시스템입니다.

## 왜 만들었나

일반적인 추천 API 구현은 candidate generation, ranking, serving 중 일부만 보여주는 경우가
많습니다. 이 프로젝트는 추천 문제를 Generative Retrieval로 풀면서 다음 질문에 답하는 것을
목표로 합니다.

- item을 token sequence로 생성하는 추천 모델을 만들 수 있는가
- 생성 모델이 존재하지 않는 item ID를 만들지 못하게 막을 수 있는가
- constrained decoding을 naive trie와 accelerator-friendly sparse transition 방식으로 비교할 수 있는가
- 실험 결과를 API serving까지 연결할 수 있는가

## 구현한 것

- MovieLens sequential recommendation 전처리
- Recall@K, NDCG@K, MRR 평가 지표
- Popularity 및 item co-occurrence baseline
- interaction embedding 기반 Semantic ID 생성
- hierarchical balanced k-means 기반 item tokenization
- PyTorch Transformer 기반 Generative Retrieval 모델
- YouTube `static_decoding` package 기반 constrained beam search
- naive trie 및 검증용 matrix decoder와의 mask 일치 검증
- decoder latency, throughput, generated sequence validity benchmark
- FastAPI recommendation endpoint와 model-backed serving benchmark

## 핵심 결과

MovieLens 32M 기준 결과입니다. Latest Small은 개발용 smoke test로만 사용하고,
최종 성능 수치는 `ml-32m` 산출물을 기준으로 정리했습니다.

| 항목 | 결과 |
| --- | ---: |
| Generative Retrieval Recall@20 | 0.127978 |
| Generative Retrieval NDCG@20 | 0.061858 |
| Generative Retrieval MRR@20 | 0.043031 |
| test invalid generation rate | 0.000000 |
| decoder benchmark batch 512 speedup | 6.66x |
| serving benchmark batch 128 평균 latency | 11.3422 ms |
| serving benchmark batch 128 throughput | 88.15 req/s |

해석:

- 현재 generative model은 작은 Transformer를 1 epoch 학습한 baseline입니다.
- `ml-32m` test split에서 Popularity baseline의 Recall@20 `0.058100`, item co-occurrence
  baseline의 Recall@20 `0.099016`보다 높은 `0.127978`을 기록했습니다.
- constrained decoding 적용 후 존재하지 않는 Semantic ID 생성률을 0으로 유지했습니다.
- 이 프로젝트의 핵심 성과는 추천 정확도 1등이 아니라, Generative Retrieval에서 constrained
  decoding을 실제 추천 pipeline과 benchmark, serving까지 연결한 것입니다.

## static_decoding 적용 방식

`static-decoding` Git dependency를 commit `c24f9dc8b9b8045716fff7ef750a1f0cb31c6f57`로
고정했습니다. 프로젝트는 다음 함수를 직접 호출합니다.

- `static_decoding.csr_utils.build_static_index`
- `static_decoding.decoding_pt.generate_and_apply_logprobs_mask`
- `static_decoding.decoding_jax.generate_and_apply_logprobs_mask`
- `static_decoding.decoding_pt.sparse_transition_torch`
- `static_decoding.decoding_jax.sparse_transition_jax`

추천 모델 inference는 사용자 history와 decoder prefix를 함께 입력해야 하므로 beam search loop는
프로젝트 모델 형태에 맞게 감싸고, constraint 계산은 `static_decoding` sparse mask kernel을
호출합니다.

## 재현 명령

```bash
uv sync --group dev
make lint
make test
MOVIELENS_DATASET=ml-32m \
MOVIELENS_URL=https://files.grouplens.org/datasets/movielens/ml-32m.zip \
make download-movielens
RAW_RATINGS=data/raw/ml-32m/ratings.csv \
PROCESSED_DIR=data/processed/ml-32m \
make preprocess

PROCESSED_DIR=data/processed/ml-32m \
BASELINE_DIR=artifacts/baseline/ml-32m \
make train-baseline

PROCESSED_DIR=data/processed/ml-32m \
BASELINE_DIR=artifacts/baseline/ml-32m \
BASELINE_REPORT=reports/ml_32m_baseline.md \
make eval-baseline

PROCESSED_DIR=data/processed/ml-32m \
SEMANTIC_ID_PATH=artifacts/semantic_id/ml-32m/semantic_ids.json \
SEMANTIC_ID_REPORT=reports/ml_32m_semantic_id.md \
SEMANTIC_ID_BRANCHING_FACTOR=32 \
make build-semantic-ids

SEMANTIC_ID_PATH=artifacts/semantic_id/ml-32m/semantic_ids.json \
make validate-semantic-ids

SEMANTIC_ID_PATH=artifacts/semantic_id/ml-32m/semantic_ids.json \
STATIC_DECODING_INDEX_PATH=artifacts/semantic_id/ml-32m/static_decoding_index.npz \
make build-static-decoding-index

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

uv run python scripts/benchmark_decoder.py \
  --semantic-id-path artifacts/semantic_id/ml-32m/semantic_ids.json \
  --report-path reports/ml_32m_decoder_benchmark.md \
  --batch-sizes 1 32 128 512
```

model-backed serving benchmark:

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

## 대표 리포트

- [MovieLens 32M 대용량 검증](ml_32m_validation.md)
- [ml-32m baseline](ml_32m_baseline.md)
- [ml-32m Semantic ID](ml_32m_semantic_id.md)
- [ml-32m Generative Retrieval teacher-forcing](ml_32m_generative.md)
- [ml-32m Generative Retrieval ranking](ml_32m_generative_eval.md)
- [ml-32m decoder benchmark](ml_32m_decoder_benchmark.md)
- [ml-32m serving benchmark](ml_32m_serving_benchmark.md)

## 산출물 정책

데이터셋, parquet, checkpoint, Semantic ID artifact, static decoding index는 로컬에서 생성합니다.
Git에는 코드, 테스트, 설정, 한국어 문서, 재현 가능한 실험 리포트만 커밋합니다.

## 다음 개선

- GPU 또는 TPU 환경에서 `static_decoding` kernel benchmark 재현
- 더 큰 Transformer 설정과 longer training schedule 적용
- 영화 metadata 또는 text embedding을 Semantic ID 생성에 결합
- score calibration, diversity constraint, candidate reranking 추가
- FastAPI HTTP endpoint 기준 latency benchmark 추가
