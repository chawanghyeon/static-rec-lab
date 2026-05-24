# MovieLens feedback-aware 전처리 리포트

## 설정

- positive interaction 기준: rating >= 4
- neutral interaction 기준: rating >= 3
- 사용자별 최소 positive interaction 수: 5
- 최대 history 길이: 50

## 필터링 요약

| 단계 | users | items | interactions |
| --- | ---: | ---: | ---: |
| raw ratings | 200,948 | 84,432 | 32,000,204 |
| positive target candidates | 200,726 | 55,174 | 15,938,231 |
| eligible full histories | 198,979 | 84,381 | 31,935,252 |

## Split 예제 수

| split | examples | output |
| --- | ---: | --- |
| train | 15,411,717 | `data/processed/ml-32m/train.parquet` |
| valid | 198,979 | `data/processed/ml-32m/valid.parquet` |
| test | 198,979 | `data/processed/ml-32m/test.parquet` |
