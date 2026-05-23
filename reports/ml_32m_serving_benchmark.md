# Serving 벤치마크 리포트

추천 서비스의 사용자별 추천 생성 latency를 측정합니다.
이 벤치마크는 HTTP 서버를 띄우지 않고 서비스 contract를 직접 호출하므로,
FastAPI 직렬화, validation, network overhead를 제외한 모델 및 decoder 비용을 봅니다.

## 설정

- 서비스 소스: `model-backed`
- model: `generative-retrieval-static`
- decoder: `static_decoding_pt`
- device: `cpu`
- k: 20
- batch sizes: 1, 32, 128
- warmup iterations: 3
- measured iterations: 10
- random seed: 42
- user_id range: 1..10000

## 결과

| batch_size | 요청 수 | 평균 latency ms | p50 latency ms | p95 latency ms | 최대 latency ms | throughput req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10.8240 | 11.5296 | 11.8903 | 11.9668 | 92.36 |
| 32 | 320 | 11.3692 | 11.5907 | 13.9118 | 24.0993 | 87.94 |
| 128 | 1,280 | 11.3422 | 11.4198 | 14.0557 | 29.5062 | 88.15 |

## 해석

- latency는 `RecommendationService.recommend(user_id, k)` 단일 호출 기준입니다.
- batch_size는 한 측정 iteration에서 순차 호출한 user request 수입니다.
- 실제 HTTP endpoint latency는 FastAPI 직렬화, validation, network overhead가 추가될 수 있습니다.
