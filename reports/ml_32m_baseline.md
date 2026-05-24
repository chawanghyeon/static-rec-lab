# Baseline 평가 리포트

## 설정

### Popularity

- 설명: 전체 train split의 target item 빈도 기반 baseline
- 학습 예제 수: 15411717
- item 수: 54689

### Item co-occurrence

- 설명: 사용자 positive history item과 target item의 co-occurrence count 기반 baseline
- 학습 예제 수: 15411717
- item 수: 54483

## 평가 결과

| model | split | k | Recall@K | NDCG@K | MRR@K |
| --- | --- | ---: | ---: | ---: | ---: |
| Popularity | valid | 10 | 0.043939 | 0.021913 | 0.015279 |
| Popularity | valid | 20 | 0.075073 | 0.029733 | 0.017398 |
| Popularity | test | 10 | 0.040446 | 0.019996 | 0.013820 |
| Popularity | test | 20 | 0.069374 | 0.027231 | 0.015765 |
| Item co-occurrence | valid | 10 | 0.075963 | 0.039857 | 0.028978 |
| Item co-occurrence | valid | 20 | 0.115012 | 0.049655 | 0.031628 |
| Item co-occurrence | test | 10 | 0.068756 | 0.036168 | 0.026363 |
| Item co-occurrence | test | 20 | 0.105408 | 0.045366 | 0.028851 |

## 해석

Popularity baseline은 개인화 없이 전체 train split에서 자주 등장한 item을 추천합니다.
Item co-occurrence baseline은 사용자 positive history item과 함께 등장한 target item을 집계해 개인화된 추천을 만듭니다.
이 결과는 이후 Generative Retrieval 모델의 비교 기준으로 사용합니다.
