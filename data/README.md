# 데이터

이 디렉터리는 원본 데이터와 전처리 결과를 보관합니다.

- `raw/`: 다운로드한 원본 데이터
- `processed/`: train, valid, test parquet 파일

대용량 데이터 파일은 Git에 커밋하지 않습니다.

## 전처리 입력

MovieLens 데이터는 다음 명령으로 다운로드합니다.

```bash
make download-movielens
```

기본 다운로드 대상은 GroupLens의 MovieLens Latest Small 데이터셋입니다.

```text
data/raw/ml-latest-small/ratings.csv
```

MovieLens `ratings.csv`는 다음 컬럼을 사용합니다.

- `userId`
- `movieId`
- `rating`
- `timestamp`

전처리 결과는 다음 파일로 저장됩니다.

- `processed/train.parquet`
- `processed/valid.parquet`
- `processed/test.parquet`

원본 zip, CSV, parquet 결과물은 재생성 가능한 파일이므로 Git에 커밋하지 않습니다.
