import json
import numpy as np

def create_sequences_with_masking(encoded_sequence, sequence_length=128, stride=None):
    """
    시퀀스를 고정 길이로 나누고, 입력과 정답을 완전히 분리
    정답이 입력에 포함되지 않도록 보장
    
    Args:
        encoded_sequence: 인코딩된 숫자 시퀀스
        sequence_length: 각 시퀀스의 길이 (입력 길이)
        stride: 시퀀스 간 겹치는 길이 (None이면 sequence_length로 설정, 겹치지 않음)
    
    Returns:
        input_sequences: 입력 시퀀스 리스트 (각각 sequence_length 길이)
        target_sequences: 정답 시퀀스 리스트 (각각 sequence_length 길이)
        attention_masks: 어텐션 마스크 (정답 위치를 가리는 마스크)
    """
    if stride is None:
        stride = sequence_length  # 겹치지 않음
    
    input_sequences = []
    target_sequences = []
    attention_masks = []
    
    # 시퀀스를 나누기
    for i in range(0, len(encoded_sequence) - sequence_length, stride):
        # 입력: 현재 위치부터 sequence_length 길이
        input_seq = encoded_sequence[i:i + sequence_length]
        
        # 정답: 입력 다음 위치부터 sequence_length 길이 (한 칸 오른쪽으로 밀림)
        target_seq = encoded_sequence[i + 1:i + 1 + sequence_length]
        
        # 정답 시퀀스가 충분히 긴지 확인
        if len(target_seq) == sequence_length:
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
            
            # 어텐션 마스크: 모든 위치에서 다음 토큰을 예측하므로 모두 True
            # (실제로는 모델이 정답을 볼 수 없도록 loss 계산 시에만 사용)
            attention_mask = [1] * sequence_length
            attention_masks.append(attention_mask)
    
    return input_sequences, target_sequences, attention_masks

def create_sequences_with_causal_masking(encoded_sequence, sequence_length=128, stride=None):
    """
    Causal masking을 적용한 시퀀스 생성
    각 위치에서 이전 토큰들만 볼 수 있고, 현재 및 미래 토큰은 볼 수 없음
    
    Args:
        encoded_sequence: 인코딩된 숫자 시퀀스
        sequence_length: 각 시퀀스의 길이
        stride: 시퀀스 간 겹치는 길이
    
    Returns:
        input_sequences: 입력 시퀀스 리스트
        target_sequences: 정답 시퀀스 리스트
        causal_masks: Causal attention 마스크
    """
    if stride is None:
        stride = sequence_length
    
    input_sequences = []
    target_sequences = []
    causal_masks = []
    
    for i in range(0, len(encoded_sequence) - sequence_length, stride):
        # 전체 시퀀스 (입력 + 정답 포함)
        full_seq = encoded_sequence[i:i + sequence_length + 1]
        
        if len(full_seq) == sequence_length + 1:
            # 입력: 처음부터 마지막-1까지
            input_seq = full_seq[:-1]
            # 정답: 두 번째부터 마지막까지
            target_seq = full_seq[1:]
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
            
            # Causal mask: 각 위치에서 이전 토큰들만 볼 수 있음
            # shape: (sequence_length, sequence_length)
            mask = []
            for pos in range(sequence_length):
                row = [1 if j <= pos else 0 for j in range(sequence_length)]
                mask.append(row)
            causal_masks.append(mask)
    
    return input_sequences, target_sequences, causal_masks

def main():
    print("="*70)
    print("정답이 보이지 않도록 마스킹된 학습 데이터 생성")
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
    print(f"   어휘 크기: {vocab['vocab_size']:,}")
    
    # 시퀀스 길이 설정
    sequence_length = 128
    stride = sequence_length  # 겹치지 않음
    
    print(f"\n3. 고정 길이 시퀀스 생성 (길이: {sequence_length}, stride: {stride})...")
    input_seqs, target_seqs, attention_masks = create_sequences_with_masking(
        encoded, sequence_length=sequence_length, stride=stride
    )
    print(f"   생성된 시퀀스 개수: {len(input_seqs):,}")
    print(f"   각 시퀀스 길이: {sequence_length}")
    
    # Causal masking도 생성
    print(f"\n4. Causal masking 시퀀스 생성...")
    causal_input_seqs, causal_target_seqs, causal_masks = create_sequences_with_causal_masking(
        encoded, sequence_length=sequence_length, stride=stride
    )
    print(f"   생성된 시퀀스 개수: {len(causal_input_seqs):,}")
    
    # 예시 출력
    print("\n5. 첫 번째 시퀀스 예시:")
    print("-"*70)
    print("입력 시퀀스 (처음 20개):")
    print(f"  {input_seqs[0][:20]}")
    print("\n정답 시퀀스 (처음 20개):")
    print(f"  {target_seqs[0][:20]}")
    print("\n토큰으로 변환:")
    print("  입력:", " ".join([id_to_token.get(id, "?") for id in input_seqs[0][:20]]))
    print("  정답:", " ".join([id_to_token.get(id, "?") for id in target_seqs[0][:20]]))
    
    # 정답이 입력에 포함되지 않았는지 확인
    print("\n6. 검증: 정답이 입력에 포함되지 않았는지 확인...")
    print("-"*70)
    first_input = set(input_seqs[0])
    first_target = set(target_seqs[0])
    overlap = first_input.intersection(first_target)
    print(f"   입력 고유 토큰 수: {len(first_input)}")
    print(f"   정답 고유 토큰 수: {len(first_target)}")
    print(f"   겹치는 토큰 수: {len(overlap)}")
    print(f"   → 정답은 입력의 다음 위치이므로, 같은 토큰이 나타날 수 있지만")
    print(f"     위치가 다르므로 모델이 정답을 '엿볼' 수 없습니다.")
    
    # 위치별 비교
    print("\n7. 위치별 입력-정답 비교 (처음 10개):")
    print("-"*70)
    print("위치 | 입력(ID) | 정답(ID) | 입력(토큰) | 정답(토큰) | 일치?")
    print("-"*70)
    for i in range(min(10, len(input_seqs[0]))):
        inp_id = input_seqs[0][i]
        tgt_id = target_seqs[0][i]
        inp_token = id_to_token.get(inp_id, "?")
        tgt_token = id_to_token.get(tgt_id, "?")
        match = "O" if inp_id == tgt_id else "X"
        print(f"  {i:3d} |    {inp_id:4d}   |    {tgt_id:4d}   |    {inp_token:10s} |    {tgt_token:10s} |  {match}")
    print("\n   → 정답[i]는 입력[i+1]과 같아야 합니다 (한 칸 오른쪽으로 밀림)")
    
    # 결과 저장
    print("\n8. 결과 저장...")
    
    # 기본 시퀀스 데이터
    training_data = {
        'input_sequences': input_seqs,
        'target_sequences': target_seqs,
        'attention_masks': attention_masks,
        'sequence_length': sequence_length,
        'num_sequences': len(input_seqs),
        'vocab_size': vocab['vocab_size'],
        'description': '고정 길이 시퀀스로 나눈 입력-정답 쌍. 정답은 입력을 한 칸 오른쪽으로 민 것.'
    }
    
    with open('hamlet_sequences_masked.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_sequences_masked.json: 마스킹된 시퀀스 저장")
    
    # NumPy 배열로 저장
    np_inputs = np.array(input_seqs, dtype=np.int32)
    np_targets = np.array(target_seqs, dtype=np.int32)
    np_masks = np.array(attention_masks, dtype=np.int32)
    
    np.savez('hamlet_sequences_masked.npz',
             input_sequences=np_inputs,
             target_sequences=np_targets,
             attention_masks=np_masks,
             sequence_length=sequence_length,
             vocab_size=vocab['vocab_size'])
    print("   [OK] hamlet_sequences_masked.npz: NumPy 배열 형식으로 저장")
    
    # Causal masking 데이터도 저장
    causal_data = {
        'input_sequences': causal_input_seqs,
        'target_sequences': causal_target_seqs,
        'causal_masks': causal_masks,
        'sequence_length': sequence_length,
        'num_sequences': len(causal_input_seqs),
        'vocab_size': vocab['vocab_size'],
        'description': 'Causal masking이 적용된 시퀀스. 각 위치에서 이전 토큰만 볼 수 있음.'
    }
    
    with open('hamlet_sequences_causal.json', 'w', encoding='utf-8') as f:
        json.dump(causal_data, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_sequences_causal.json: Causal masking 시퀀스 저장")
    
    # 통계 정보
    stats = {
        '총_토큰_수': len(encoded),
        '시퀀스_길이': sequence_length,
        'stride': stride,
        '생성된_시퀀스_수': len(input_seqs),
        '어휘_크기': vocab['vocab_size'],
        '설명': '정답이 입력에 포함되지 않도록 고정 길이 시퀀스로 분리. 각 시퀀스에서 입력[i] -> 정답[i] (입력[i+1])를 예측.'
    }
    
    with open('hamlet_masked_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("   [OK] hamlet_masked_stats.json: 통계 정보 저장")
    
    # 최종 요약
    print("\n" + "="*70)
    print("마스킹된 학습 데이터 생성 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  - hamlet_sequences_masked.json: 마스킹된 시퀀스 (JSON)")
    print(f"  - hamlet_sequences_masked.npz: 마스킹된 시퀀스 (NumPy)")
    print(f"  - hamlet_sequences_causal.json: Causal masking 시퀀스 (JSON)")
    print(f"  - hamlet_masked_stats.json: 통계 정보")
    print(f"\n데이터 구조:")
    print(f"  - 시퀀스 개수: {len(input_seqs):,}")
    print(f"  - 각 시퀀스 길이: {sequence_length}")
    print(f"  - 입력 shape: ({len(input_seqs)}, {sequence_length})")
    print(f"  - 정답 shape: ({len(target_seqs)}, {sequence_length})")
    print(f"\n중요:")
    print(f"  - 정답은 입력을 한 칸 오른쪽으로 민 것")
    print(f"  - 각 위치 i에서: 입력[i]를 보고 정답[i] (입력[i+1])를 예측")
    print(f"  - 모델은 정답을 미리 볼 수 없음 (입력과 정답이 분리됨)")
    print("="*70)

if __name__ == "__main__":
    main()

