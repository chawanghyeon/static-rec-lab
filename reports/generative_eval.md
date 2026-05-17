# Generative Retrieval 추천 평가 리포트

## 설정

- checkpoint: `artifacts/generative/model.pt`
- Semantic ID artifact: `artifacts/semantic_id/semantic_ids.json`
- decoder: `static_decoding_pt`
- STATIC index artifact: `artifacts/semantic_id/static_decoding_index.npz`
- beam size: 50
- 평가 cutoff: 10, 20
- 추천 ranking 생성 후 사용자 history에 이미 포함된 item은 제외합니다.

## 추천 성능

| split | k | Recall@K | NDCG@K | MRR@K |
| --- | ---: | ---: | ---: | ---: |
| valid | 10 | 0.029508 | 0.016399 | 0.012486 |
| valid | 20 | 0.060656 | 0.024308 | 0.014671 |
| test | 10 | 0.037705 | 0.016930 | 0.010723 |
| test | 20 | 0.072131 | 0.025476 | 0.012989 |

## 생성 품질

| split | examples | unknown targets | generated sequences | invalid sequences | invalid generation rate | history filtered | duplicate filtered | avg recs/query | elapsed ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 610 | 20 | 30,500 | 0 | 0.000000 | 2,073 | 0 | 20.00 | 27438.01 |
| test | 610 | 24 | 30,500 | 0 | 0.000000 | 2,054 | 0 | 20.00 | 22215.41 |

## 해석

이 평가는 teacher-forcing token accuracy가 아니라 실제 추천 ranking 품질을 봅니다.
모델 logits에 선택한 constrained decoder를 적용해 존재하는 Semantic ID만 생성한 뒤 item_id로 복원하고, baseline과 같은 Recall/NDCG/MRR 기준으로 비교합니다.
unknown target은 Semantic ID catalog에 없는 cold-start target이므로 추천 가능 후보에는 없지만, 평가 denominator에는 남겨 실제 추천 실패로 반영합니다.
