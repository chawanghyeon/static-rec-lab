# Decoder 벤치마크 리포트

naive trie decoder와 STATIC-style sparse transition matrix decoder의 allowed-token mask 생성 성능을 비교합니다.

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

| batch_size | mask 일치 | naive latency ms | STATIC latency ms | naive rows/s | STATIC rows/s | speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 예 | 0.0023 | 0.0269 | 438,196.73 | 37,170.69 | 0.08x |
| 32 | 예 | 0.0145 | 0.0293 | 2,203,413.64 | 1,090,769.40 | 0.50x |
| 128 | 예 | 0.0526 | 0.0376 | 2,432,804.70 | 3,408,221.00 | 1.40x |
| 512 | 예 | 0.2090 | 0.0621 | 2,449,374.96 | 8,240,509.57 | 3.36x |

## 검증

- 각 batch size의 모든 sampled state batch에서 naive trie와 STATIC-style decoder의 allowed-token mask가 동일한지 먼저 확인합니다.
- benchmark 실행 중 생성된 mask checksum도 비교해서 측정 루프 안의 결과 차이를 감지합니다.
- throughput은 measured iterations 동안 생성한 mask row 수를 총 소요 시간으로 나눈 값입니다.
