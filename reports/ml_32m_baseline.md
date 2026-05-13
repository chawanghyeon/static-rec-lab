# Baseline 평가 리포트

## 설정

### Popularity

- 설명: 전체 train split의 target item 빈도 기반 baseline
- 학습 예제 수: 31397360
- item 수: 84249

### Item co-occurrence

- 설명: 사용자 history item과 target item의 co-occurrence count 기반 baseline
- 학습 예제 수: 31397360
- item 수: 84167

## 평가 결과

| model | split | k | Recall@K | NDCG@K | MRR@K |
| --- | --- | ---: | ---: | ---: | ---: |
| Popularity | valid | 10 | 0.035581 | 0.017258 | 0.011773 |
| Popularity | valid | 20 | 0.059926 | 0.023336 | 0.013401 |
| Popularity | test | 10 | 0.035138 | 0.016817 | 0.011331 |
| Popularity | test | 20 | 0.058100 | 0.022527 | 0.012848 |
| Item co-occurrence | valid | 10 | 0.065559 | 0.033768 | 0.024201 |
| Item co-occurrence | valid | 20 | 0.103086 | 0.043182 | 0.026745 |
| Item co-occurrence | test | 10 | 0.063748 | 0.034091 | 0.025169 |
| Item co-occurrence | test | 20 | 0.099016 | 0.042943 | 0.027565 |

## 해석

Popularity baseline은 개인화 없이 전체 train split에서 자주 등장한 item을 추천합니다.
Item co-occurrence baseline은 사용자 history의 item과 함께 등장한 target item을 집계해 개인화된 추천을 만듭니다.
이 결과는 이후 Generative Retrieval 모델의 비교 기준으로 사용합니다.
