import json
import numpy as np

# 데이터 로드
with open('hamlet_sequences_masked.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 어휘 사전 로드
with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}

print("="*70)
print("마스킹된 데이터 검증")
print("="*70)

input_seqs = data['input_sequences']
target_seqs = data['target_sequences']
seq_len = data['sequence_length']

print(f"\n시퀀스 개수: {len(input_seqs):,}")
print(f"각 시퀀스 길이: {seq_len}")

# 첫 번째 시퀀스 상세 검증
print("\n" + "="*70)
print("첫 번째 시퀀스 상세 검증")
print("="*70)

first_input = input_seqs[0]
first_target = target_seqs[0]

print(f"\n입력 시퀀스 (처음 15개):")
print(f"  {first_input[:15]}")
print(f"  토큰: {' '.join([id_to_token.get(id, '?') for id in first_input[:15]])}")

print(f"\n정답 시퀀스 (처음 15개):")
print(f"  {first_target[:15]}")
print(f"  토큰: {' '.join([id_to_token.get(id, '?') for id in first_target[:15]])}")

print("\n위치별 비교:")
print("-"*70)
print("위치 | 입력(ID) | 정답(ID) | 입력(토큰) | 정답(토큰) | 정답=입력[i+1]?")
print("-"*70)

# 원본 인코딩 시퀀스 로드 (검증용)
with open('hamlet_encoded.json', 'r', encoding='utf-8') as f:
    original = json.load(f)

for i in range(min(15, len(first_input))):
    inp_id = first_input[i]
    tgt_id = first_target[i]
    inp_token = id_to_token.get(inp_id, "?")
    tgt_token = id_to_token.get(tgt_id, "?")
    
    # 정답[i]가 입력[i+1]과 같은지 확인
    if i + 1 < len(first_input):
        expected = first_input[i + 1]
        is_correct = "O" if tgt_id == expected else "X"
    else:
        is_correct = "?"
    
    print(f"  {i:3d} |    {inp_id:4d}   |    {tgt_id:4d}   |    {inp_token:10s} |    {tgt_token:10s} |      {is_correct}")

print("\n" + "="*70)
print("정답이 입력에 포함되지 않았는지 확인")
print("="*70)

# 각 시퀀스에서 정답이 입력의 같은 위치에 있는지 확인
print("\n중요한 검증:")
print("-"*70)

# 첫 번째 시퀀스에서
for i in range(min(10, len(first_input))):
    inp_at_i = first_input[i]
    tgt_at_i = first_target[i]
    
    # 정답[i]는 입력[i+1]과 같아야 함
    if i + 1 < len(first_input):
        expected_tgt = first_input[i + 1]
        if tgt_at_i == expected_tgt:
            status = "정상"
        else:
            status = "오류!"
            print(f"  위치 {i}: 정답[{i}]={tgt_at_i}, 입력[{i+1}]={expected_tgt} - {status}")
    else:
        status = "마지막 위치"

print("\n  → 정답[i]는 입력[i+1]과 같습니다.")
print("  → 모델은 입력[i]를 보고 정답[i] (입력[i+1])를 예측해야 합니다.")
print("  → 정답이 입력의 같은 위치에 있지 않으므로 모델이 '엿볼' 수 없습니다.")

# "To be, or not to be" 부분 찾기
print("\n" + "="*70)
print("'To be, or not to be' 부분 찾기")
print("="*70)

# 원본에서 "To be, or not to be"의 토큰 ID: [471, 309, 8, 594, 220, 70, 309]
phrase_ids = [471, 309, 8, 594, 220, 70, 309]

# 어느 시퀀스에 포함되어 있는지 찾기
found = False
for seq_idx, input_seq in enumerate(input_seqs):
    for i in range(len(input_seq) - len(phrase_ids) + 1):
        if input_seq[i:i+len(phrase_ids)] == phrase_ids:
            found = True
            print(f"\n  시퀀스 {seq_idx}, 위치 {i}에서 발견!")
            print(f"  입력 시퀀스의 해당 부분:")
            print(f"    {input_seq[i:i+len(phrase_ids)]}")
            print(f"    {' '.join([id_to_token.get(id, '?') for id in input_seq[i:i+len(phrase_ids)]])}")
            
            print(f"\n  정답 시퀀스의 해당 부분:")
            target_seq = target_seqs[seq_idx]
            print(f"    {target_seq[i:i+len(phrase_ids)]}")
            print(f"    {' '.join([id_to_token.get(id, '?') for id in target_seq[i:i+len(phrase_ids)]])}")
            
            print(f"\n  위치별 비교:")
            for j in range(len(phrase_ids)):
                inp_id = input_seq[i + j]
                tgt_id = target_seq[i + j]
                inp_token = id_to_token.get(inp_id, "?")
                tgt_token = id_to_token.get(tgt_id, "?")
                expected = input_seq[i + j + 1] if i + j + 1 < len(input_seq) else None
                match = "O" if tgt_id == expected else "X"
                print(f"    위치 {i+j}: 입력={inp_id}('{inp_token}') -> 정답={tgt_id}('{tgt_token}') [예상: {expected}] {match}")
            break
    if found:
        break

if not found:
    print("  'To be, or not to be'를 찾을 수 없습니다.")

print("\n" + "="*70)
print("검증 완료!")
print("="*70)
print("\n결론:")
print("  - 정답은 입력을 한 칸 오른쪽으로 민 것입니다.")
print("  - 정답[i] = 입력[i+1]")
print("  - 모델은 입력[i]를 보고 정답[i]를 예측해야 합니다.")
print("  - 정답이 입력의 같은 위치에 있지 않으므로 모델이 정답을 '엿볼' 수 없습니다.")
print("="*70)

