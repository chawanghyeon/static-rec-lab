# 리포트

이 디렉터리에는 실험 결과와 구현 trade-off를 한국어로 정리합니다. `ml_32m_*`
리포트는 MovieLens 32M pipeline 재실행 후 생성되는 산출물입니다.

리포트 목록:

- [project_summary.md](project_summary.md): 문제 정의, 구현 범위, 재현 경로 요약
- `ml_32m_preprocess.md`: MovieLens 32M feedback-aware 전처리 통계
- `ml_32m_baseline.md`: MovieLens 32M baseline 추천 성능
- `ml_32m_semantic_id.md`: MovieLens 32M Semantic ID 생성 방식과 검증 결과
- `ml_32m_generative.md`: MovieLens 32M Generative Retrieval teacher-forcing 평가
- `ml_32m_generative_eval.md`: MovieLens 32M Generative Retrieval 추천 ranking 평가
- [static_decoding_integration.md](static_decoding_integration.md): `static_decoding` package 통합 방식

static_decoding index artifact는 `make build-static-decoding-index`로
`artifacts/semantic_id/static_decoding_index.npz`에 생성합니다.
MovieLens 32M 최종 index artifact는 `artifacts/semantic_id/ml-32m/static_decoding_index.npz`에
생성합니다.
