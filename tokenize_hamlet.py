import re
import json
from collections import Counter

def tokenize_text(text, method='word'):
    """
    텍스트를 토큰으로 분할
    
    Args:
        text: 입력 텍스트
        method: 토큰화 방법 ('word', 'char')
    
    Returns:
        토큰 리스트
    """
    if method == 'word':
        # 단어 단위 토큰화 (구두점 포함)
        # 단어와 구두점을 모두 토큰으로 유지
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        # 빈 토큰 제거
        tokens = [token for token in tokens if token.strip()]
    elif method == 'char':
        # 문자 단위 토큰화
        tokens = list(text)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return tokens

def create_vocabulary(tokens, min_freq=1, add_special_tokens=True):
    """
    토큰 리스트로부터 어휘 사전 생성
    
    Args:
        tokens: 토큰 리스트
        min_freq: 최소 빈도 (이보다 적게 나타나는 토큰은 <UNK>로 처리)
        add_special_tokens: 특수 토큰 추가 여부
    
    Returns:
        token_to_id: 토큰 -> ID 매핑 딕셔너리
        id_to_token: ID -> 토큰 매핑 딕셔너리
    """
    # 토큰 빈도 계산
    token_counts = Counter(tokens)
    
    # 특수 토큰 정의
    special_tokens = {}
    if add_special_tokens:
        special_tokens = {
            '<PAD>': 0,  # 패딩
            '<UNK>': 1,  # 알 수 없는 토큰
            '<START>': 2,  # 시작 토큰
            '<END>': 3,   # 끝 토큰
        }
    
    # 어휘 사전 생성
    token_to_id = special_tokens.copy()
    id_to_token = {v: k for k, v in special_tokens.items()}
    
    # 빈도가 min_freq 이상인 토큰만 추가
    next_id = len(special_tokens)
    for token, count in token_counts.items():
        if count >= min_freq:
            if token not in token_to_id:
                token_to_id[token] = next_id
                id_to_token[next_id] = token
                next_id += 1
    
    return token_to_id, id_to_token

def encode_text(tokens, token_to_id, use_unk=True):
    """
    토큰 리스트를 숫자 ID 리스트로 변환
    
    Args:
        tokens: 토큰 리스트
        token_to_id: 토큰 -> ID 매핑
        use_unk: 알 수 없는 토큰을 <UNK>로 처리할지 여부
    
    Returns:
        숫자 ID 리스트
    """
    encoded = []
    unk_id = token_to_id.get('<UNK>', -1)
    
    for token in tokens:
        if token in token_to_id:
            encoded.append(token_to_id[token])
        elif use_unk and unk_id != -1:
            encoded.append(unk_id)
        else:
            # <UNK>가 없으면 건너뛰기
            continue
    
    return encoded

def decode_text(encoded_ids, id_to_token):
    """
    숫자 ID 리스트를 토큰 리스트로 변환
    
    Args:
        encoded_ids: 숫자 ID 리스트
        id_to_token: ID -> 토큰 매핑
    
    Returns:
        토큰 리스트
    """
    tokens = []
    for id in encoded_ids:
        if id in id_to_token:
            token = id_to_token[id]
            # 특수 토큰은 제외
            if not token.startswith('<') or not token.endswith('>'):
                tokens.append(token)
    return tokens

def main():
    print("햄릿 텍스트 토큰화 및 숫자 변환 시작...")
    print("="*60)
    
    # 텍스트 읽기
    print("\n1. 텍스트 파일 읽기...")
    with open('hamlet_clean.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"   텍스트 길이: {len(text):,} 문자")
    
    # 토큰화 (단어 단위)
    print("\n2. 텍스트 토큰화 (단어 단위)...")
    tokens = tokenize_text(text, method='word')
    print(f"   총 토큰 수: {len(tokens):,}")
    print(f"   예시 토큰 (처음 20개): {tokens[:20]}")
    
    # 어휘 사전 생성
    print("\n3. 어휘 사전 생성...")
    token_to_id, id_to_token = create_vocabulary(tokens, min_freq=1, add_special_tokens=True)
    vocab_size = len(token_to_id)
    print(f"   어휘 크기: {vocab_size:,} 고유 토큰")
    print(f"   예시 매핑 (처음 10개):")
    for i, (token, token_id) in enumerate(list(token_to_id.items())[:10]):
        print(f"     '{token}' -> {token_id}")
    
    # 텍스트를 숫자로 변환
    print("\n4. 텍스트를 숫자 시퀀스로 변환...")
    encoded = encode_text(tokens, token_to_id, use_unk=True)
    print(f"   인코딩된 시퀀스 길이: {len(encoded):,}")
    print(f"   예시 숫자 시퀀스 (처음 30개): {encoded[:30]}")
    
    # 결과 저장
    print("\n5. 결과 저장...")
    
    # 숫자 시퀀스 저장 (텍스트 파일)
    with open('hamlet_encoded.txt', 'w', encoding='utf-8') as f:
        # 한 줄에 하나의 숫자로 저장
        f.write(' '.join(map(str, encoded)))
    print("   [OK] hamlet_encoded.txt: 숫자 시퀀스 저장")
    
    # 숫자 시푼스 저장 (JSON 형식)
    with open('hamlet_encoded.json', 'w', encoding='utf-8') as f:
        json.dump(encoded, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_encoded.json: 숫자 시퀀스 저장 (JSON)")
    
    # 어휘 사전 저장
    vocab_data = {
        'token_to_id': token_to_id,
        'id_to_token': {str(k): v for k, v in id_to_token.items()},  # JSON은 키가 문자열이어야 함
        'vocab_size': vocab_size,
        'total_tokens': len(tokens),
        'encoded_length': len(encoded)
    }
    with open('hamlet_vocabulary.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_vocabulary.json: 어휘 사전 저장")
    
    # 통계 정보 저장
    stats = {
        '원본_텍스트_길이': len(text),
        '총_토큰_수': len(tokens),
        '고유_토큰_수': vocab_size,
        '인코딩된_시퀀스_길이': len(encoded),
        '어휘_크기': vocab_size,
        '압축_비율': f"{len(encoded) / len(tokens) * 100:.2f}%"
    }
    
    with open('hamlet_tokenization_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_tokenization_stats.json: 통계 정보 저장")
    
    # 검증: 디코딩 테스트
    print("\n6. 검증: 디코딩 테스트...")
    decoded_tokens = decode_text(encoded[:50], id_to_token)
    decoded_text = ' '.join(decoded_tokens)
    print(f"   원본 (처음 50 토큰): {' '.join(tokens[:50])}")
    print(f"   디코딩 (처음 50 토큰): {decoded_text}")
    
    # 최종 요약
    print("\n" + "="*60)
    print("토큰화 및 숫자 변환 완료!")
    print("="*60)
    print(f"\n생성된 파일:")
    print(f"  - hamlet_encoded.txt: 숫자 시퀀스 (공백으로 구분)")
    print(f"  - hamlet_encoded.json: 숫자 시퀀스 (JSON 배열)")
    print(f"  - hamlet_vocabulary.json: 어휘 사전 (토큰 <-> ID 매핑)")
    print(f"  - hamlet_tokenization_stats.json: 통계 정보")
    print(f"\n통계:")
    print(f"  - 원본 텍스트: {len(text):,} 문자")
    print(f"  - 총 토큰 수: {len(tokens):,}")
    print(f"  - 고유 토큰 수: {vocab_size:,}")
    print(f"  - 인코딩된 시퀀스: {len(encoded):,} 숫자")
    print("="*60)

if __name__ == "__main__":
    main()

