import json
from collections import Counter

# 어휘 사전 로드
with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# 인코딩된 시퀀스 로드
with open('hamlet_encoded.json', 'r', encoding='utf-8') as f:
    encoded = json.load(f)

# ID -> 토큰 매핑 생성
id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}

print("="*60)
print("토큰화 결과 요약")
print("="*60)
print(f"\n어휘 크기: {vocab['vocab_size']:,} 고유 토큰")
print(f"총 토큰 수: {len(encoded):,}")
print(f"\n가장 자주 나타나는 토큰 20개:")
print("-"*60)

# 빈도 계산
counts = Counter(encoded)
top20 = counts.most_common(20)

for i, (token_id, count) in enumerate(top20, 1):
    token = id_to_token.get(token_id, "?")
    percentage = (count / len(encoded)) * 100
    print(f"{i:2d}. '{token}' (ID: {token_id:4d}): {count:6,}회 ({percentage:5.2f}%)")

print("\n" + "="*60)
print("예시: 숫자 시퀀스 -> 토큰 변환")
print("="*60)
print("\n처음 30개 숫자:")
print(encoded[:30])
print("\n해당하는 토큰들:")
tokens = [id_to_token.get(id, "?") for id in encoded[:30]]
print(" ".join(tokens))

print("\n" + "="*60)

