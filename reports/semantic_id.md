# Semantic ID 리포트

## 생성 방식

- train split의 history-target co-occurrence로 item interaction matrix를 만듭니다.
- sparse matrix를 Truncated SVD로 축소한 뒤 item 빈도와 정규화된 item 순서 feature를 더합니다.
- hierarchical balanced k-means로 token path를 만들고, 각 subtree capacity를 넘지 않게 balanced chunk로 나눕니다.
- 이 방식은 clustering 구조를 사용하면서도 모든 item에 중복 없는 고정 길이 Semantic ID를 부여하기 위한 구현입니다.

## 설정

- depth: 4
- branching factor: 16
- capacity: 65536
- item 수: 9681
- Semantic ID 길이: 4
- embedding dimension: 35
- context edge 수: 4261752
- output: `artifacts/semantic_id/semantic_ids.json`

## 검증

- 모든 item에 Semantic ID가 부여되었습니다.
- 모든 Semantic ID는 고정 길이입니다.
- 중복 Semantic ID는 없습니다.
