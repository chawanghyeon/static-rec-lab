# static_decoding 통합 리포트

## 목적

이 프로젝트의 STATIC decoder는 YouTube `static-constraint-decoding` repository에서 제공하는
`static_decoding` package를 기본 구현으로 사용합니다.

이 문서의 목적은 `static-rec-lab`의 STATIC decoder가 `static_decoding` package를 어떻게
직접 호출하는지 기록하는
것입니다.

사용하는 `static_decoding` package 정보:

- GitHub: <https://github.com/youtube/static-constraint-decoding>
- 고정 커밋: `c24f9dc8b9b8045716fff7ef750a1f0cb31c6f57`
- 라이선스: Apache-2.0

## 기본 원칙

STATIC constraint 계산은 `static_decoding` package가 담당합니다.

이 repository 안의 naive trie와 로컬 matrix decoder는 정확성 확인과 벤치마크
비교군을 위한 검증용 코드입니다. 추천 평가와 API serving의 실행 경로는
`static_decoding` 기반 STATIC decoder입니다.

## 연결된 `static_decoding` 함수

| `static_decoding` 함수 | 프로젝트 사용 위치 | 역할 |
| --- | --- | --- |
| `static_decoding.csr_utils.build_static_index` | `recsys/decoding/static_decoding.py` | Semantic ID catalog를 STATIC index 배열로 변환 |
| `static_decoding.decoding_pt.generate_and_apply_logprobs_mask` | 추론, 벤치마크, 테스트 | PyTorch sparse tail 후보 수집 |
| `static_decoding.decoding_jax.generate_and_apply_logprobs_mask` | 벤치마크, 테스트 | JAX sparse tail 후보 수집 |
| `static_decoding.decoding_pt.sparse_transition_torch` | 벤치마크, 테스트 | `static_decoding` PyTorch decoding harness 동작 검증 |
| `static_decoding.decoding_jax.sparse_transition_jax` | 벤치마크, 테스트 | `static_decoding` JAX decoding harness 동작 검증 |
| `static_decoding.decoding_pt.RandomModel` | 벤치마크, 테스트 | `static_decoding` harness용 dummy model |
| `static_decoding.decoding_jax.RandomModel` | 벤치마크, 테스트 | `static_decoding` JAX harness용 dummy model |

## 프로젝트 실행 경로

`make build-static-decoding-index`는 `build_static_index` 산출물을 그대로 저장합니다.

```text
artifacts/semantic_id/static_decoding_index.npz
```

저장되는 배열:

- `packed_csr`
- `csr_indptr`
- `layer_max_branches`
- `start_mask`
- `dense_mask`
- `dense_states`

`make eval-generative-ranking`은 위 artifact가 있으면 다시 빌드하지 않고 로드합니다. API도
`STATIC_REC_STATIC_DECODING_INDEX_PATH`가 지정되면 같은 artifact를 사용합니다.

## 모델 inference에서의 적용 방식

추천 모델은 사용자 history encoder 출력과 Semantic ID decoder prefix를 함께 사용합니다.
`static_decoding.decoding_pt.sparse_transition_torch` harness는 `model(input_ids[:, -1:])` 형태의 decoder-only
모델 호출을 전제합니다.

그래서 추천 모델 inference에서는 beam search loop만 프로젝트 모델 입력 형태에 맞게 감싸고,
constraint index와 sparse candidate gather는 `static_decoding` package 함수를 직접 호출합니다.

정리하면 다음과 같습니다.

- constraint index 생성: `static_decoding.csr_utils.build_static_index`
- sparse tail masking: `static_decoding.decoding_pt.generate_and_apply_logprobs_mask`
- `static_decoding` harness 검증: `static_decoding.decoding_pt.sparse_transition_torch`,
  `static_decoding.decoding_jax.sparse_transition_jax`

## 검증

현재 로컬 CPU 환경에서 확인한 항목:

- `make lint`
- `make test`
- `make build-static-decoding-index`
- `make benchmark-decoder`

테스트와 benchmark는 다음을 검증합니다.

- `static_decoding` index가 Semantic ID codec의 모든 ID를 포함하는지
- PyTorch sparse candidate gather가 index의 allowed token과 일치하는지
- JAX sparse candidate gather가 index의 allowed token과 일치하는지
- `static_decoding` PyTorch/JAX harness가 생성한 Semantic ID가 모두 유효한지
- Generative Retrieval ranking 평가에서 invalid generation rate가 0인지

## 추가 환경 검증 대상

로컬 CPU 환경 밖에서 추가로 확인할 항목:

- GPU latency benchmark
- TPU latency benchmark
- `torch.compile` 적용 benchmark
- `static_decoding` benchmark suite의 Hash bitmap, PPV baseline까지 포함한 동일 조건 재현

현재 상태를 표현할 때는 다음 문장이 가장 정확합니다.

```text
static-rec-lab은 STATIC constraint 계산에 YouTube static_decoding package를 사용한다.
Semantic ID catalog를 static_decoding index로 변환하고, PyTorch/JAX sparse candidate kernel과
static_decoding harness를 테스트와 benchmark에서 검증한다. 추천 모델 inference는 프로젝트 모델의
history-conditioned 입력 형태에 맞춘 beam loop에서 static_decoding sparse mask kernel을 호출한다.
```
