# 데이터

이 디렉터리는 원본 데이터와 전처리 결과를 보관합니다.

- `raw/`: 다운로드한 원본 데이터
- `processed/`: train, valid, test parquet 파일

대용량 데이터 파일은 Git에 커밋하지 않습니다.

## 전처리 입력

MovieLens `ratings.csv`는 다음 컬럼을 사용합니다.

- `userId`
- `movieId`
- `rating`
- `timestamp`

전처리 결과는 다음 파일로 저장됩니다.

- `processed/train.parquet`
- `processed/valid.parquet`
- `processed/test.parquet`
