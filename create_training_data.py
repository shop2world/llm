import json
import numpy as np

def create_input_target_pairs(encoded_sequence):
    """
    인코딩된 시퀀스에서 입력-정답 쌍 생성
    입력을 오른쪽으로 한 칸 밀어서 정답으로 설정
    
    Args:
        encoded_sequence: 인코딩된 숫자 시퀀스 [t1, t2, t3, ..., tn]
    
    Returns:
        inputs: 입력 시퀀스 [t1, t2, t3, ..., t(n-1)]
        targets: 정답 시퀀스 [t2, t3, t4, ..., tn]
    """
    # 입력: 처음부터 마지막-1까지
    inputs = encoded_sequence[:-1]
    # 정답: 두 번째부터 마지막까지 (입력을 한 칸 오른쪽으로 민 것)
    targets = encoded_sequence[1:]
    
    return inputs, targets

def create_sequence_pairs(encoded_sequence, sequence_length=None):
    """
    시퀀스를 여러 개의 입력-정답 쌍으로 분할
    (선택적: 긴 시퀀스를 여러 개의 짧은 시퀀스로 나눌 때 사용)
    
    Args:
        encoded_sequence: 인코딩된 숫자 시퀀스
        sequence_length: 각 시퀀스의 길이 (None이면 전체를 하나로)
    
    Returns:
        input_sequences: 입력 시퀀스 리스트
        target_sequences: 정답 시퀀스 리스트
    """
    if sequence_length is None:
        # 전체를 하나의 시퀀스로
        inputs, targets = create_input_target_pairs(encoded_sequence)
        return [inputs], [targets]
    else:
        # 여러 개의 짧은 시퀀스로 분할
        input_sequences = []
        target_sequences = []
        
        for i in range(0, len(encoded_sequence) - sequence_length, sequence_length):
            seq = encoded_sequence[i:i + sequence_length + 1]
            inputs, targets = create_input_target_pairs(seq)
            input_sequences.append(inputs)
            target_sequences.append(targets)
        
        return input_sequences, target_sequences

def main():
    print("="*70)
    print("다음 단어 예측을 위한 입력-정답 쌍 생성")
    print("="*70)
    
    # 인코딩된 시퀀스 로드
    print("\n1. 인코딩된 시퀀스 로드...")
    with open('hamlet_encoded.json', 'r', encoding='utf-8') as f:
        encoded = json.load(f)
    print(f"   총 토큰 수: {len(encoded):,}")
    
    # 어휘 사전 로드
    print("\n2. 어휘 사전 로드...")
    with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}
    token_to_id = vocab['token_to_id']
    print(f"   어휘 크기: {vocab['vocab_size']:,}")
    
    # 입력-정답 쌍 생성
    print("\n3. 입력-정답 쌍 생성...")
    inputs, targets = create_input_target_pairs(encoded)
    print(f"   입력 시퀀스 길이: {len(inputs):,}")
    print(f"   정답 시퀀스 길이: {len(targets):,}")
    
    # 예시 출력
    print("\n4. 예시 확인 (처음 20개):")
    print("-"*70)
    print("인덱스 | 입력 (ID) | 정답 (ID) | 입력 (토큰) | 정답 (토큰)")
    print("-"*70)
    for i in range(min(20, len(inputs))):
        input_id = inputs[i]
        target_id = targets[i]
        input_token = id_to_token.get(input_id, "?")
        target_token = id_to_token.get(target_id, "?")
        print(f"  {i:4d} |    {input_id:4d}   |    {target_id:4d}   |    {input_token:10s} |    {target_token}")
    
    # "To be, or not to be" 예시
    print("\n5. 'To be, or not to be' 예시:")
    print("-"*70)
    # "To be, or not to be"의 토큰 ID: [471, 309, 8, 594, 220, 70, 309]
    phrase_ids = [471, 309, 8, 594, 220, 70, 309]
    
    # 해당 부분 찾기
    found = False
    for i in range(len(encoded) - len(phrase_ids) + 1):
        if encoded[i:i+len(phrase_ids)] == phrase_ids:
            found = True
            print(f"   위치: 인덱스 {i}부터")
            print(f"   원본 시퀀스: {encoded[i:i+len(phrase_ids)]}")
            
            # 해당 부분의 입력-정답 쌍
            phrase_inputs = inputs[i:i+len(phrase_ids)-1]
            phrase_targets = targets[i:i+len(phrase_ids)-1]
            
            print(f"\n   입력-정답 쌍:")
            for j, (inp, tgt) in enumerate(zip(phrase_inputs, phrase_targets)):
                inp_token = id_to_token.get(inp, "?")
                tgt_token = id_to_token.get(tgt, "?")
                print(f"     입력[{i+j}]: {inp:4d} ('{inp_token}') -> 정답: {tgt:4d} ('{tgt_token}')")
            break
    
    if not found:
        print("   'To be, or not to be'를 찾을 수 없습니다.")
    
    # 결과 저장
    print("\n6. 결과 저장...")
    
    # 전체 입력-정답 쌍 저장 (JSON)
    training_data = {
        'inputs': inputs,
        'targets': targets,
        'vocab_size': vocab['vocab_size'],
        'sequence_length': len(inputs),
        'description': '입력을 오른쪽으로 한 칸 밀어서 정답으로 설정한 다음 단어 예측 데이터'
    }
    
    with open('hamlet_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_training_data.json: 전체 입력-정답 쌍 저장")
    
    # NumPy 배열로도 저장 (머신러닝 프레임워크에서 사용하기 편함)
    np_inputs = np.array(inputs, dtype=np.int32)
    np_targets = np.array(targets, dtype=np.int32)
    
    np.savez('hamlet_training_data.npz', 
             inputs=np_inputs, 
             targets=np_targets,
             vocab_size=vocab['vocab_size'])
    print("   [OK] hamlet_training_data.npz: NumPy 배열 형식으로 저장")
    
    # 통계 정보
    stats = {
        '총_토큰_수': len(encoded),
        '입력_시퀀스_길이': len(inputs),
        '정답_시퀀스_길이': len(targets),
        '총_입력_정답_쌍_수': len(inputs),
        '어휘_크기': vocab['vocab_size'],
        '설명': '입력을 오른쪽으로 한 칸 밀어서 정답으로 설정. 모델은 입력의 다음 토큰을 예측해야 함.'
    }
    
    with open('hamlet_training_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_training_stats.json: 통계 정보 저장")
    
    # 최종 요약
    print("\n" + "="*70)
    print("입력-정답 쌍 생성 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  - hamlet_training_data.json: 입력-정답 쌍 (JSON)")
    print(f"  - hamlet_training_data.npz: 입력-정답 쌍 (NumPy 배열)")
    print(f"  - hamlet_training_stats.json: 통계 정보")
    print(f"\n데이터 구조:")
    print(f"  - 입력: {len(inputs):,}개 토큰 (인덱스 0부터 {len(inputs)-1}까지)")
    print(f"  - 정답: {len(targets):,}개 토큰 (인덱스 1부터 {len(encoded)-1}까지)")
    print(f"  - 각 위치 i에서: 입력[i] -> 정답[i] (다음 토큰 예측)")
    print("="*70)

if __name__ == "__main__":
    main()

