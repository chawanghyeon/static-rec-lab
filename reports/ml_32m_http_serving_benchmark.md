# HTTP Endpoint 벤치마크 리포트

FastAPI recommendation endpoint를 HTTP client contract로 호출해 latency를 측정합니다.
이 벤치마크는 endpoint routing, request validation, response model serialization,
client-side JSON decode 비용을 포함합니다. 별도 network hop은 포함하지 않습니다.

## 설정

- 서비스 소스: `model-backed`
- endpoint: `/recommendations/users/{user_id}?k={k}`
- model: `generative-retrieval-static`
- decoder: `static_decoding_pt`
- k: 20
- batch sizes: 1, 32, 128
- warmup iterations: 3
- measured iterations: 10
- random seed: 42
- user_id range: 1..10000

## 결과

| batch_size | 요청 수 | 평균 latency ms | p50 latency ms | p95 latency ms | 최대 latency ms | throughput req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 13.7949 | 13.6424 | 17.4176 | 20.3080 | 72.46 |
| 32 | 320 | 12.9125 | 13.2220 | 15.2779 | 34.4323 | 77.43 |
| 128 | 1,280 | 14.1033 | 13.5673 | 18.3580 | 181.5605 | 70.89 |

## 해석

- latency는 `GET /recommendations/users/{user_id}?k=...` 호출 기준입니다.
- ASGI test client를 사용하므로 FastAPI endpoint 비용은 포함하지만 network hop은 포함하지 않습니다.
- service 직접 호출 benchmark와 비교하면 API routing, validation, serialization, JSON decode overhead를 볼 수 있습니다.
