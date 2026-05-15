# Serving 벤치마크 리포트

추천 서비스의 사용자별 추천 생성 latency를 측정합니다.
이 벤치마크는 HTTP 서버를 띄우지 않고 서비스 contract를 직접 호출하므로,
FastAPI 직렬화, validation, network overhead를 제외한 모델 및 decoder 비용을 봅니다.

## 설정

- 서비스 소스: `environment`
- model: `generative-retrieval-static`
- decoder: `static_sparse_matrix`
- k: 20
- batch sizes: 1, 8, 32
- warmup iterations: 1
- measured iterations: 3
- random seed: 42
- user_id range: 1..610

## 결과

| batch_size | 요청 수 | 평균 latency ms | p50 latency ms | p95 latency ms | 최대 latency ms | throughput req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 13.6479 | 13.7175 | 13.9666 | 13.9943 | 73.26 |
| 8 | 24 | 13.7883 | 13.7450 | 14.5645 | 14.7064 | 72.52 |
| 32 | 96 | 13.5059 | 13.6805 | 14.9834 | 16.0165 | 74.04 |

## 해석

- latency는 `RecommendationService.recommend(user_id, k)` 단일 호출 기준입니다.
- batch_size는 한 측정 iteration에서 순차 호출한 user request 수입니다.
- 실제 HTTP endpoint latency는 FastAPI 직렬화, validation, network overhead가 추가될 수 있습니다.
