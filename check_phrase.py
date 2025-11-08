import json
import re

# 어휘 사전 로드
with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# 인코딩된 시퀀스 로드
with open('hamlet_encoded.json', 'r', encoding='utf-8') as f:
    encoded = json.load(f)

# ID -> 토큰 매핑 생성
id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}
token_to_id = vocab['token_to_id']

# 원본 텍스트 읽기
with open('hamlet_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# "To be, or not to be" 찾기
phrase = "To be, or not to be"
print("="*70)
print(f"검색할 문구: '{phrase}'")
print("="*70)

# 원본 텍스트에서 위치 찾기
idx = text.find(phrase)
if idx == -1:
    print("문구를 찾을 수 없습니다.")
else:
    # 주변 텍스트 출력
    start = max(0, idx - 50)
    end = min(len(text), idx + len(phrase) + 50)
    context = text[start:end]
    print(f"\n원본 텍스트 (주변 포함):")
    print("-"*70)
    print(context)
    print("-"*70)
    
    # 해당 문구를 토큰화
    print(f"\n1. 토큰화 과정:")
    print("-"*70)
    tokens = re.findall(r'\b\w+\b|[^\w\s]', phrase)
    print(f"문구: '{phrase}'")
    print(f"토큰으로 분할: {tokens}")
    
    # 각 토큰의 ID 찾기
    print(f"\n2. 각 토큰의 숫자 ID:")
    print("-"*70)
    token_ids = []
    for token in tokens:
        token_id = token_to_id.get(token, token_to_id.get('<UNK>', -1))
        token_ids.append(token_id)
        print(f"  '{token}' -> ID: {token_id}")
    
    print(f"\n3. 최종 숫자 나열:")
    print("-"*70)
    print(f"  {token_ids}")
    print(f"  또는: {' '.join(map(str, token_ids))}")
    
    # 전체 텍스트에서 해당 부분 찾기
    print(f"\n4. 전체 인코딩된 시퀀스에서 찾기:")
    print("-"*70)
    
    # 전체 텍스트를 토큰화
    all_tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    
    # 문구의 토큰들이 시작하는 위치 찾기
    phrase_start_in_tokens = None
    for i in range(len(all_tokens) - len(tokens) + 1):
        if all_tokens[i:i+len(tokens)] == tokens:
            phrase_start_in_tokens = i
            break
    
    if phrase_start_in_tokens is not None:
        print(f"  토큰 인덱스: {phrase_start_in_tokens}부터 {phrase_start_in_tokens + len(tokens) - 1}까지")
        print(f"  인코딩된 숫자: {encoded[phrase_start_in_tokens:phrase_start_in_tokens + len(tokens)]}")
        print(f"  주변 숫자 (앞 5개 + 문구 + 뒤 5개):")
        start_idx = max(0, phrase_start_in_tokens - 5)
        end_idx = min(len(encoded), phrase_start_in_tokens + len(tokens) + 5)
        print(f"    {encoded[start_idx:end_idx]}")
        
        # 디코딩 검증
        decoded_tokens = [id_to_token.get(id, "?") for id in encoded[phrase_start_in_tokens:phrase_start_in_tokens + len(tokens)]]
        print(f"  디코딩 검증: {' '.join(decoded_tokens)}")
    else:
        print("  인코딩된 시퀀스에서 해당 문구를 찾을 수 없습니다.")

print("\n" + "="*70)

