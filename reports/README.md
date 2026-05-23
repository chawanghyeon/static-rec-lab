# 리포트

이 디렉터리에는 실험 결과와 구현 trade-off를 한국어로 정리합니다.

리포트 목록:

- [portfolio_summary.md](portfolio_summary.md): 포트폴리오 제출용 1페이지 요약
- `ml_32m_validation.md`: MovieLens 32M 대용량 smoke/performance 검증
- `ml_32m_baseline.md`: MovieLens 32M baseline 추천 성능
- `ml_32m_semantic_id.md`: MovieLens 32M Semantic ID 생성 방식과 검증 결과
- `ml_32m_generative.md`: MovieLens 32M Generative Retrieval teacher-forcing 평가
- `ml_32m_generative_eval.md`: MovieLens 32M Generative Retrieval 추천 ranking 평가
- `ml_32m_decoder_benchmark.md`: MovieLens 32M Semantic ID 기준 decoder latency/validity 벤치마크
- `ml_32m_serving_benchmark.md`: MovieLens 32M model-backed recommendation service latency 벤치마크
- `ml_32m_http_serving_benchmark.md`: MovieLens 32M FastAPI HTTP endpoint latency 벤치마크
- [static_decoding_integration.md](static_decoding_integration.md): `static_decoding` package 통합 방식

static_decoding index artifact는 `make build-static-decoding-index`로
`artifacts/semantic_id/static_decoding_index.npz`에 생성합니다.
MovieLens 32M 최종 index artifact는 `artifacts/semantic_id/ml-32m/static_decoding_index.npz`에
생성합니다.
