# Decoder 벤치마크 리포트

naive trie decoder와 검증용 sparse transition matrix decoder의 allowed-token mask 생성 성능을 비교합니다.

## 설정

- 데이터 소스: `synthetic`
- Semantic ID 수: 4,096
- trie state 수: 11,927
- vocab size: 128
- batch sizes: 1, 32, 128, 512
- warmup iterations: 10
- measured iterations: 100
- random seed: 42

## 결과

| batch_size | mask 일치 | naive latency ms | matrix latency ms | naive rows/s | matrix rows/s | speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 예 | 0.0022 | 0.0270 | 450,874.92 | 37,084.53 | 0.08x |
| 32 | 예 | 0.0157 | 0.0304 | 2,039,840.64 | 1,052,170.22 | 0.52x |
| 128 | 예 | 0.0570 | 0.0381 | 2,245,860.29 | 3,361,123.88 | 1.50x |
| 512 | 예 | 0.2178 | 0.0617 | 2,350,384.81 | 8,295,583.30 | 3.53x |

## static_decoding PyTorch kernel

`static_decoding.decoding_pt.generate_and_apply_logprobs_mask`를 사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 측정합니다. 이 값은 전체 vocab boolean mask 생성이 아니라 candidate gather latency입니다.

| batch_size | 후보 일치 | static_decoding latency ms | static_decoding rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.0251 | 39,828.09 |
| 32 | 예 | 0.0299 | 1,071,518.50 |
| 128 | 예 | 0.0308 | 4,150,790.43 |
| 512 | 예 | 0.0912 | 5,612,727.30 |

## static_decoding sparse_transition_torch harness

`static_decoding.decoding_pt.sparse_transition_torch`를 그대로 호출해 upstream PyTorch decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 end-to-end harness latency를 측정합니다. 이 harness는 `static_decoding.decoding_pt.RandomModel`을 사용하므로 추천 모델 품질 평가는 아니며, static_decoding 호출 경로와 constrained generation 동작 검증에 초점을 둡니다.

| batch_size | 생성 ID 유효 | static_decoding harness latency ms | static_decoding harness rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.1291 | 7,743.48 |
| 32 | 예 | 0.2965 | 107,912.76 |
| 128 | 예 | 0.4092 | 312,828.10 |
| 512 | 예 | 0.8791 | 582,392.30 |

## static_decoding JAX kernel

`static_decoding.decoding_jax.generate_and_apply_logprobs_mask`를 사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 측정합니다. 이 값은 CPU 환경의 JAX candidate gather latency입니다.

| batch_size | 후보 일치 | static_decoding JAX latency ms | static_decoding JAX rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.5994 | 1,668.20 |
| 32 | 예 | 0.5858 | 54,624.09 |
| 128 | 예 | 0.5772 | 221,773.99 |
| 512 | 예 | 0.6109 | 838,170.03 |

## static_decoding sparse_transition_jax harness

`static_decoding.decoding_jax.sparse_transition_jax`를 그대로 호출해 static_decoding JAX decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 end-to-end harness latency를 측정합니다. 이 harness는 `static_decoding.decoding_jax.RandomModel`을 사용합니다.

| batch_size | 생성 ID 유효 | static_decoding JAX harness latency ms | static_decoding JAX harness rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.1638 | 6,105.02 |
| 32 | 예 | 0.4082 | 78,393.50 |
| 128 | 예 | 0.7581 | 168,841.58 |
| 512 | 예 | 1.3137 | 389,737.17 |

## 검증

- 각 batch size의 모든 sampled state batch에서 naive trie와 검증용 matrix decoder의 allowed-token mask가 동일한지 먼저 확인합니다.
- benchmark 실행 중 생성된 mask checksum도 비교해서 측정 루프 안의 결과 차이를 감지합니다.
- throughput은 measured iterations 동안 생성한 mask row 수를 총 소요 시간으로 나눈 값입니다.
- static_decoding PyTorch kernel은 동일 prefix batch에서 static_decoding index의 allowed-token 후보와 일치하는지 검증합니다.
- static_decoding sparse_transition_torch harness는 생성된 모든 Semantic ID가 static_decoding index에 존재하는지 검증합니다.
- static_decoding JAX kernel과 sparse_transition_jax harness도 동일한 static_decoding index에서 후보 token 및 생성 Semantic ID validity를 검증합니다.
