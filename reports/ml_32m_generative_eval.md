# Generative Retrieval 추천 평가 리포트

## 설정

- checkpoint: `artifacts/generative/ml-32m/model.pt`
- Semantic ID artifact: `artifacts/semantic_id/ml-32m/semantic_ids.json`
- decoder: `static_decoding_pt`
- STATIC index artifact: `artifacts/semantic_id/ml-32m/static_decoding_index.npz`
- beam size: 20
- inference batch size: 128
- 평가 cutoff: 10, 20
- 추천 ranking 생성 후 사용자 history에 이미 포함된 item은 제외합니다.

## 추천 성능

| split | k | Recall@K | NDCG@K | MRR@K |
| --- | ---: | ---: | ---: | ---: |
| valid | 10 | 0.111287 | 0.060276 | 0.044855 |
| valid | 20 | 0.143888 | 0.068742 | 0.047294 |
| test | 10 | 0.098195 | 0.054127 | 0.040806 |
| test | 20 | 0.127978 | 0.061858 | 0.043031 |

## 생성 품질

| split | examples | unknown targets | generated sequences | invalid sequences | invalid generation rate | history filtered | duplicate filtered | avg recs/query | elapsed ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 200,948 | 75 | 4,018,960 | 0 | 0.000000 | 719,934 | 0 | 16.42 | 285347.85 |
| test | 200,948 | 102 | 4,018,960 | 0 | 0.000000 | 718,930 | 0 | 16.42 | 285259.78 |

## 해석

이 평가는 teacher-forcing token accuracy가 아니라 실제 추천 ranking 품질을 봅니다.
모델 logits에 선택한 constrained decoder를 적용해 존재하는 Semantic ID만 생성한 뒤 item_id로 복원하고, baseline과 같은 Recall/NDCG/MRR 기준으로 비교합니다.
unknown target은 Semantic ID catalog에 없는 cold-start target이므로 추천 가능 후보에는 없지만, 평가 denominator에는 남겨 실제 추천 실패로 반영합니다.
