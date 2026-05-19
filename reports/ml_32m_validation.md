# MovieLens 32M 대용량 검증 리포트

## 목적

`ml-latest-small`에서 동작하던 데이터 파이프라인과 Semantic ID 생성이
MovieLens 32M 규모에서도 실제로 실행 가능한지 확인했습니다.

이번 검증은 generative model 학습이나 decoder benchmark가 아니라, 다음 경로의
대용량 안정성을 확인하는 smoke/performance validation입니다.

- MovieLens 32M 다운로드 및 압축 해제
- sequential recommendation 전처리
- popularity baseline 학습 및 평가
- item co-occurrence baseline 학습 및 평가
- Semantic ID 생성 및 검증

이후 같은 `ml-32m` 산출물로 generative model 학습, constrained ranking 평가, decoder benchmark,
serving benchmark까지 추가로 실행했습니다. 대표 수치는 이 문서 하단과 각 세부 리포트에
정리했습니다.

## 데이터

- dataset: MovieLens `ml-32m`
- ratings: `32,000,204`
- users after filtering: `200,948`
- train examples: `31,397,360`
- valid examples: `200,948`
- test examples: `200,948`
- train target item 수: `84,249`

## 실행 결과

| 단계 | 결과 | 소요 시간 |
| --- | --- | ---: |
| preprocess | `data/processed/ml-32m/*.parquet` 생성 | `287.47s` |
| popularity train | `artifacts/baseline/ml-32m/popularity.json` 생성 | `9.29s` |
| item co-occurrence train | `artifacts/baseline/ml-32m/item_knn.json` 생성 | `184.39s` |
| baseline eval | `reports/ml_32m_baseline.md` 생성 | `414.35s` |
| Semantic ID build | `artifacts/semantic_id/ml-32m/semantic_ids.json` 생성 | `141.64s` |
| Semantic ID validate | 중복 없는 고정 길이 ID 검증 통과 | `<1s` |
| static_decoding index build | `artifacts/semantic_id/ml-32m/static_decoding_index.npz` 생성 | `2.31s` |
| Generative Retrieval train | `artifacts/generative/ml-32m/model.pt` 생성 | `3162.64s` |
| Generative Retrieval ranking eval | `reports/ml_32m_generative_eval.md` 생성 | `576.29s` |
| decoder benchmark | `reports/ml_32m_decoder_benchmark.md` 생성 | `9.05s` |
| serving benchmark | `reports/ml_32m_serving_benchmark.md` 생성 | `28.42s` |

## 산출물 크기

- raw dataset: `911M`
- processed parquet: `923M`
- popularity artifact: `6.1M`
- item co-occurrence artifact: `333M`
- Semantic ID artifact: `9.1M`

데이터와 artifact는 Git에 커밋하지 않습니다. 리포트만 Git에 포함합니다.

## Baseline

자세한 결과는 `reports/ml_32m_baseline.md`에 기록했습니다.

| model | split | k | Recall@K | NDCG@K | MRR@K |
| --- | --- | ---: | ---: | ---: | ---: |
| Popularity | valid | 10 | 0.035581 | 0.017258 | 0.011773 |
| Popularity | valid | 20 | 0.059926 | 0.023336 | 0.013401 |
| Popularity | test | 10 | 0.035138 | 0.016817 | 0.011331 |
| Popularity | test | 20 | 0.058100 | 0.022527 | 0.012848 |
| Item co-occurrence | valid | 10 | 0.065559 | 0.033768 | 0.024201 |
| Item co-occurrence | valid | 20 | 0.103086 | 0.043182 | 0.026745 |
| Item co-occurrence | test | 10 | 0.063748 | 0.034091 | 0.025169 |
| Item co-occurrence | test | 20 | 0.099016 | 0.042943 | 0.027565 |

## Semantic ID

`ml-32m`은 train target item 수가 `84,249`개라 기본 설정인
`branching_factor=16`, `depth=4`의 capacity `65,536`을 초과합니다.
따라서 대용량 검증에서는 Semantic ID 길이 4를 유지하면서
`branching_factor=32`를 사용했습니다.

- depth: `4`
- branching factor: `32`
- capacity: `1,048,576`
- Semantic ID item 수: `84,259`
- Semantic ID 길이: `4`
- embedding 계산 example 수: `31,397,360`
- context edge 수: `1,342,214,049`

최종 검증 결과:

- 모든 item에 Semantic ID가 부여되었습니다.
- 모든 Semantic ID는 길이 4입니다.
- 중복 Semantic ID는 없습니다.

## Generative Retrieval

자세한 결과는 `reports/ml_32m_generative.md`와
`reports/ml_32m_generative_eval.md`에 기록했습니다.

Teacher-forcing 평가:

| split | examples | loss | token accuracy | sequence accuracy |
| --- | ---: | ---: | ---: | ---: |
| valid | 200,873 | 1.797266 | 0.409831 | 0.013222 |

추천 ranking 평가:

| split | k | Recall@K | NDCG@K | MRR@K |
| --- | ---: | ---: | ---: | ---: |
| valid | 10 | 0.111287 | 0.060276 | 0.044855 |
| valid | 20 | 0.143888 | 0.068742 | 0.047294 |
| test | 10 | 0.098195 | 0.054127 | 0.040806 |
| test | 20 | 0.127978 | 0.061858 | 0.043031 |

생성 품질:

- valid invalid generation rate: `0.000000`
- test invalid generation rate: `0.000000`
- ranking inference batch size: `128`
- valid elapsed: `285,347.85 ms`
- test elapsed: `285,259.78 ms`

## Decoder와 serving

- `ml-32m` decoder benchmark batch 512 기준 검증용 matrix decoder는 naive trie 대비 `6.66x`
  빠르게 mask를 생성했습니다.
- `static_decoding` PyTorch/JAX candidate gather와 sparse transition harness는 모두 후보 일치
  및 생성 ID validity 검증을 통과했습니다.
- `ml-32m` model-backed serving benchmark는 CPU 기준 batch 128 평균 latency `11.3422 ms`,
  p95 latency `14.0557 ms`, throughput `88.15 req/s`를 기록했습니다.
- MPS는 full ranking 평가에는 유효했지만, 작은 단건 serving 호출에서는 device 전환 비용 때문에
  CPU보다 느렸습니다.

## 경고 처리

초기 대용량 실행에서는 일부 node에서 KMeans가 요청한 cluster 수보다 적은 cluster로
수렴하는 `ConvergenceWarning`이 발생했습니다. 원인은 interaction embedding에 중복 또는
매우 유사한 row가 존재하기 때문입니다.

이를 단순히 숨기지 않고, cluster collapse가 감지되면 더 작은 cluster 수로 재시도하도록
구현을 수정했습니다. 최종 재실행에서는 `ConvergenceWarning` 없이 Semantic ID 생성이
완료되었습니다.

## 구현 메모

- Semantic ID 생성은 full `31,397,360` train example을 모두 사용했습니다.
- Python list에 전체 edge를 쌓지 않고 parquet를 배치 단위로 읽어 hashed co-occurrence
  projection을 누적합니다.
- `Semantic ID item 수`가 train target item 수보다 큰 이유는 train history에만 등장한 item도
  item universe에 포함하기 때문입니다.
- item co-occurrence baseline은 full `31,397,360` train example을 모두 사용했습니다.
- item co-occurrence 학습은 DuckDB로 `history_item_ids` list column을 풀고 source-target
  pair를 집계한 뒤 source별 top-200 candidate를 저장합니다.
