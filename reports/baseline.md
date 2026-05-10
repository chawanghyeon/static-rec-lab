# Baseline 평가 리포트

## 설정

### Popularity

- 설명: 전체 train split의 target item 빈도 기반 baseline
- 학습 예제 수: 99006
- item 수: 9680

### Item co-occurrence

- 설명: 사용자 history item과 target item의 co-occurrence count 기반 baseline
- 학습 예제 수: 99006
- item 수: 9658

## 평가 결과

| model | split | k | Recall@K | NDCG@K | MRR@K |
| --- | --- | ---: | ---: | ---: | ---: |
| Popularity | valid | 10 | 0.027869 | 0.015914 | 0.012359 |
| Popularity | valid | 20 | 0.055738 | 0.022892 | 0.014234 |
| Popularity | test | 10 | 0.029508 | 0.013207 | 0.008267 |
| Popularity | test | 20 | 0.055738 | 0.019912 | 0.010147 |
| Item co-occurrence | valid | 10 | 0.059016 | 0.030234 | 0.021545 |
| Item co-occurrence | valid | 20 | 0.088525 | 0.037759 | 0.023645 |
| Item co-occurrence | test | 10 | 0.060656 | 0.031351 | 0.022596 |
| Item co-occurrence | test | 20 | 0.098361 | 0.040789 | 0.025138 |

## 해석

Popularity baseline은 개인화 없이 전체 train split에서 자주 등장한 item을 추천합니다.
Item co-occurrence baseline은 사용자 history의 item과 함께 등장한 target item을 집계해 개인화된 추천을 만듭니다.
이 결과는 이후 Generative Retrieval 모델의 비교 기준으로 사용합니다.
