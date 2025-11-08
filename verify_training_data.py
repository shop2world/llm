import json
import numpy as np

# JSON 파일 로드
with open('hamlet_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 어휘 사전 로드
with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}

print("="*70)
print("입력-정답 쌍 검증")
print("="*70)
print(f"\n입력 길이: {len(data['inputs']):,}")
print(f"정답 길이: {len(data['targets']):,}")
print(f"어휘 크기: {data['vocab_size']:,}")

print("\n처음 15개 입력-정답 쌍:")
print("-"*70)
print("인덱스 | 입력(ID) | 정답(ID) | 입력(토큰) | 정답(토큰)")
print("-"*70)
for i in range(15):
    inp_id = data['inputs'][i]
    tgt_id = data['targets'][i]
    inp_token = id_to_token.get(inp_id, "?")
    tgt_token = id_to_token.get(tgt_id, "?")
    print(f"  {i:4d} |    {inp_id:4d}   |    {tgt_id:4d}   |    {inp_token:10s} |    {tgt_token}")

# "To be, or not to be" 부분 확인
print("\n'To be, or not to be' 부분:")
print("-"*70)
# 이 부분은 인덱스 17982부터 시작
start_idx = 17982
for i in range(start_idx, start_idx + 6):
    inp_id = data['inputs'][i]
    tgt_id = data['targets'][i]
    inp_token = id_to_token.get(inp_id, "?")
    tgt_token = id_to_token.get(tgt_id, "?")
    print(f"  입력[{i}]: {inp_id:4d} ('{inp_token}') -> 정답: {tgt_id:4d} ('{tgt_token}')")

# NumPy 파일도 확인
print("\nNumPy 파일 확인:")
print("-"*70)
np_data = np.load('hamlet_training_data.npz')
print(f"  inputs shape: {np_data['inputs'].shape}")
print(f"  targets shape: {np_data['targets'].shape}")
print(f"  vocab_size: {np_data['vocab_size']}")
print(f"  처음 10개 입력: {np_data['inputs'][:10]}")
print(f"  처음 10개 정답: {np_data['targets'][:10]}")

print("\n" + "="*70)
print("검증 완료!")
print("="*70)

