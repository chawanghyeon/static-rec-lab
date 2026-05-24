# Semantic ID 리포트

## 생성 방식

- train split의 history-target co-occurrence로 item interaction matrix를 만듭니다.
- 전체 train example을 배치 단위로 읽고, source item을 hashed projection으로 누적해 dense item embedding을 만듭니다.
- item 빈도와 정규화된 item 순서 feature를 embedding에 더합니다.
- hierarchical balanced k-means로 token path를 만들고, 각 subtree capacity를 넘지 않게 balanced chunk로 나눕니다.
- 이 방식은 clustering 구조를 사용하면서도 모든 item에 중복 없는 고정 길이 Semantic ID를 부여하기 위한 구현입니다.

## 설정

- depth: 4
- branching factor: 32
- capacity: 1048576
- item 수: 54711
- Semantic ID 길이: 4
- train example 수: 15411717
- embedding 계산 example 수: 15411717
- embedding dimension: 35
- context edge 수: 384030358
- output: `artifacts/semantic_id/ml-32m/semantic_ids.json`

## 검증

- 모든 item에 Semantic ID가 부여되었습니다.
- 모든 Semantic ID는 고정 길이입니다.
- 중복 Semantic ID는 없습니다.
