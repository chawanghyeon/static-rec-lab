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
| 1 | 예 | 0.0021 | 0.0265 | 466,563.71 | 37,673.07 | 0.08x |
| 32 | 예 | 0.0143 | 0.0292 | 2,231,389.51 | 1,095,749.69 | 0.49x |
| 128 | 예 | 0.0534 | 0.0366 | 2,394,910.67 | 3,495,516.18 | 1.46x |
| 512 | 예 | 0.2115 | 0.0603 | 2,420,326.93 | 8,486,480.89 | 3.51x |

## static_decoding PyTorch kernel

`static_decoding.decoding_pt.generate_and_apply_logprobs_mask`를 사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 측정합니다. 이 값은 전체 vocab boolean mask 생성이 아니라 candidate gather latency입니다.

| batch_size | 후보 일치 | static_decoding latency ms | static_decoding rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.0253 | 39,480.83 |
| 32 | 예 | 0.0277 | 1,157,009.85 |
| 128 | 예 | 0.0316 | 4,050,366.31 |
| 512 | 예 | 0.0887 | 5,769,664.19 |

## static_decoding sparse_transition_torch harness

`static_decoding.decoding_pt.sparse_transition_torch`를 그대로 호출해 `static_decoding` PyTorch decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 end-to-end harness latency를 측정합니다. 이 harness는 `static_decoding.decoding_pt.RandomModel`을 사용하므로 추천 모델 품질 평가는 아니며, static_decoding 호출 경로와 constrained generation 동작 검증에 초점을 둡니다.

| batch_size | 생성 ID 유효 | static_decoding harness latency ms | static_decoding harness rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.1234 | 8,101.79 |
| 32 | 예 | 0.2955 | 108,288.13 |
| 128 | 예 | 0.4052 | 315,892.74 |
| 512 | 예 | 0.8640 | 592,592.88 |

## static_decoding JAX kernel

`static_decoding.decoding_jax.generate_and_apply_logprobs_mask`를 사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 측정합니다. 이 값은 CPU 환경의 JAX candidate gather latency입니다.

| batch_size | 후보 일치 | static_decoding JAX latency ms | static_decoding JAX rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.5704 | 1,753.19 |
| 32 | 예 | 0.5750 | 55,654.19 |
| 128 | 예 | 0.5613 | 228,042.05 |
| 512 | 예 | 0.6009 | 852,096.61 |

## static_decoding sparse_transition_jax harness

`static_decoding.decoding_jax.sparse_transition_jax`를 그대로 호출해 static_decoding JAX decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 end-to-end harness latency를 측정합니다. 이 harness는 `static_decoding.decoding_jax.RandomModel`을 사용합니다.

| batch_size | 생성 ID 유효 | static_decoding JAX harness latency ms | static_decoding JAX harness rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.1522 | 6,571.98 |
| 32 | 예 | 0.3905 | 81,937.83 |
| 128 | 예 | 0.7248 | 176,603.79 |
| 512 | 예 | 1.2715 | 402,679.29 |

## 검증

- 각 batch size의 모든 sampled state batch에서 naive trie와 검증용 matrix decoder의 allowed-token mask가 동일한지 먼저 확인합니다.
- benchmark 실행 중 생성된 mask checksum도 비교해서 측정 루프 안의 결과 차이를 감지합니다.
- throughput은 measured iterations 동안 생성한 mask row 수를 총 소요 시간으로 나눈 값입니다.
- static_decoding PyTorch kernel은 동일 prefix batch에서 static_decoding index의 allowed-token 후보와 일치하는지 검증합니다.
- static_decoding sparse_transition_torch harness는 생성된 모든 Semantic ID가 static_decoding index에 존재하는지 검증합니다.
- static_decoding JAX kernel과 sparse_transition_jax harness도 동일한 static_decoding index에서 후보 token 및 생성 Semantic ID validity를 검증합니다.
