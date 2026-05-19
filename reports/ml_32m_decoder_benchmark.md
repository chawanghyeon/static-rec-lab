# Decoder 벤치마크 리포트

naive trie decoder와 검증용 sparse transition matrix decoder의 allowed-token mask 생성 성능을 비교합니다.

## 설정

- 데이터 소스: `artifacts/semantic_id/ml-32m/semantic_ids.json`
- Semantic ID 수: 84,259
- trie state 수: 86,980
- vocab size: 128
- batch sizes: 1, 32, 128, 512
- warmup iterations: 10
- measured iterations: 100
- random seed: 42

## 결과

| batch_size | mask 일치 | naive latency ms | matrix latency ms | naive rows/s | matrix rows/s | speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 예 | 0.0030 | 0.0329 | 330,351.36 | 30,385.13 | 0.09x |
| 32 | 예 | 0.0192 | 0.0307 | 1,667,426.56 | 1,041,751.44 | 0.62x |
| 128 | 예 | 0.0835 | 0.0395 | 1,533,531.02 | 3,242,319.57 | 2.11x |
| 512 | 예 | 0.4277 | 0.0642 | 1,197,002.82 | 7,974,663.00 | 6.66x |

## static_decoding PyTorch kernel

`static_decoding.decoding_pt.generate_and_apply_logprobs_mask`를 사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 측정합니다. 이 값은 전체 vocab boolean mask 생성이 아니라 candidate gather latency입니다.

| batch_size | 후보 일치 | static_decoding latency ms | static_decoding rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.0286 | 34,969.11 |
| 32 | 예 | 0.0357 | 896,777.29 |
| 128 | 예 | 0.0710 | 1,803,293.12 |
| 512 | 예 | 0.1419 | 3,607,009.55 |

## static_decoding sparse_transition_torch harness

`static_decoding.decoding_pt.sparse_transition_torch`를 그대로 호출해 `static_decoding` PyTorch decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 end-to-end harness latency를 측정합니다. 이 harness는 `static_decoding.decoding_pt.RandomModel`을 사용하므로 추천 모델 품질 평가는 아니며, static_decoding 호출 경로와 constrained generation 동작 검증에 초점을 둡니다.

| batch_size | 생성 ID 유효 | static_decoding harness latency ms | static_decoding harness rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.1234 | 8,102.39 |
| 32 | 예 | 0.4882 | 65,541.98 |
| 128 | 예 | 0.5428 | 235,805.79 |
| 512 | 예 | 1.2590 | 406,657.29 |

## static_decoding JAX kernel

`static_decoding.decoding_jax.generate_and_apply_logprobs_mask`를 사용해 CSR sparse tail에서 유효 child token 후보를 추출하는 성능을 측정합니다. 이 값은 CPU 환경의 JAX candidate gather latency입니다.

| batch_size | 후보 일치 | static_decoding JAX latency ms | static_decoding JAX rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.6093 | 1,641.31 |
| 32 | 예 | 0.6125 | 52,241.70 |
| 128 | 예 | 0.6425 | 199,216.88 |
| 512 | 예 | 0.6466 | 791,818.39 |

## static_decoding sparse_transition_jax harness

`static_decoding.decoding_jax.sparse_transition_jax`를 그대로 호출해 static_decoding JAX decoding loop가 생성한 Semantic ID가 모두 유효한지 검증하고 end-to-end harness latency를 측정합니다. 이 harness는 `static_decoding.decoding_jax.RandomModel`을 사용합니다.

| batch_size | 생성 ID 유효 | static_decoding JAX harness latency ms | static_decoding JAX harness rows/s |
|---:|:---:|---:|---:|
| 1 | 예 | 0.1773 | 5,639.75 |
| 32 | 예 | 0.4158 | 76,960.31 |
| 128 | 예 | 0.7620 | 167,977.62 |
| 512 | 예 | 1.3547 | 377,945.08 |

## 검증

- 각 batch size의 모든 sampled state batch에서 naive trie와 검증용 matrix decoder의 allowed-token mask가 동일한지 먼저 확인합니다.
- benchmark 실행 중 생성된 mask checksum도 비교해서 측정 루프 안의 결과 차이를 감지합니다.
- throughput은 measured iterations 동안 생성한 mask row 수를 총 소요 시간으로 나눈 값입니다.
- static_decoding PyTorch kernel은 동일 prefix batch에서 static_decoding index의 allowed-token 후보와 일치하는지 검증합니다.
- static_decoding sparse_transition_torch harness는 생성된 모든 Semantic ID가 static_decoding index에 존재하는지 검증합니다.
- static_decoding JAX kernel과 sparse_transition_jax harness도 동일한 static_decoding index에서 후보 token 및 생성 Semantic ID validity를 검증합니다.
