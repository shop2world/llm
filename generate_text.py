import json
import torch
import torch.nn as nn
import math
import re
import numpy as np
import argparse

# 모델 아키텍처 (train_model.py와 동일)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        output = self.transformer(src, src_key_padding_mask=None, mask=src_mask)
        output = self.fc_out(output)
        return output

def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)

def tokenize_text(text, method='word'):
    """텍스트를 토큰으로 분할"""
    if method == 'word':
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        tokens = [token for token in tokens if token.strip()]
    else:
        tokens = list(text)
    return tokens

def encode_text(tokens, token_to_id, use_unk=True):
    """토큰 리스트를 숫자 ID 리스트로 변환"""
    encoded = []
    unk_id = token_to_id.get('<UNK>', -1)
    
    for token in tokens:
        if token in token_to_id:
            encoded.append(token_to_id[token])
        elif use_unk and unk_id != -1:
            encoded.append(unk_id)
        else:
            continue
    
    return encoded

def decode_text(encoded_ids, id_to_token):
    """숫자 ID 리스트를 토큰 리스트로 변환"""
    tokens = []
    for id in encoded_ids:
        if id in id_to_token:
            token = id_to_token[id]
            if not (token.startswith('<') and token.endswith('>')):
                tokens.append(token)
    return tokens

def generate_text(model, token_to_id, id_to_token, prompt, max_length=50, temperature=1.0, top_k=50, device='cpu'):
    """
    모델을 사용하여 텍스트 생성
    
    Args:
        model: 학습된 모델
        token_to_id: 토큰 -> ID 매핑
        id_to_token: ID -> 토큰 매핑
        prompt: 시작 텍스트
        max_length: 생성할 최대 토큰 수
        temperature: 샘플링 온도 (높을수록 다양함)
        top_k: 상위 k개 토큰만 고려
        device: 사용할 디바이스
    """
    model.eval()
    
    # 프롬프트 토큰화 및 인코딩
    prompt_tokens = tokenize_text(prompt, method='word')
    prompt_encoded = encode_text(prompt_tokens, token_to_id, use_unk=True)
    
    if len(prompt_encoded) == 0:
        print("경고: 프롬프트를 인코딩할 수 없습니다.")
        return prompt
    
    # 입력을 텐서로 변환
    input_ids = torch.tensor([prompt_encoded], dtype=torch.long).to(device)
    input_ids = input_ids.transpose(0, 1)  # (seq_len, batch_size)
    
    generated_ids = prompt_encoded.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # 현재 시퀀스 길이
            seq_len = len(generated_ids)
            
            # 시퀀스가 너무 길면 잘라내기
            if seq_len > 128:
                generated_ids = generated_ids[-128:]
                seq_len = 128
            
            # 입력 준비
            input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
            input_tensor = input_tensor.transpose(0, 1)  # (seq_len, batch_size)
            
            # Causal mask 생성
            causal_mask = create_causal_mask(seq_len, device)
            
            # 모델 예측
            outputs = model(input_tensor, src_mask=causal_mask)
            
            # 마지막 토큰의 예측 확률 가져오기
            logits = outputs[-1, 0, :]  # (vocab_size)
            
            # Temperature 적용
            logits = logits / temperature
            
            # Top-k 샘플링
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                # Top-k 외의 토큰 확률을 매우 낮게 설정
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                logits = filtered_logits
            
            # 확률 분포로 변환
            probs = torch.softmax(logits, dim=-1)
            
            # 샘플링
            next_token_id = torch.multinomial(probs, 1).item()
            
            # 생성된 토큰 추가
            generated_ids.append(next_token_id)
            
            # 종료 조건 (선택적: 특정 토큰에서 멈출 수 있음)
            # 여기서는 max_length까지 생성
    
    # 생성된 토큰을 텍스트로 변환
    generated_tokens = decode_text(generated_ids, id_to_token)
    generated_text = ' '.join(generated_tokens)
    
    return generated_text

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='햄릿 언어 모델 텍스트 생성')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='생성 온도 (0.2: 보수적/예측 가능, 1.5: 창의적/다양함, 기본값: 0.8)')
    parser.add_argument('--max_length', type=int, default=100,
                        help='생성할 최대 토큰 수 (기본값: 100)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k 샘플링 (기본값: 50)')
    parser.add_argument('--prompt', type=str, default='To be, or not to be,',
                        help='시작 프롬프트 (기본값: "To be, or not to be,")')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='생성할 샘플 수 (기본값: 3)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("햄릿 언어 모델 텍스트 생성")
    print("="*70)
    
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/best_model.pt'
    
    # 온도 설정 출력
    print(f"\n생성 설정:")
    print(f"  온도 (Temperature): {args.temperature}")
    if args.temperature < 0.5:
        print(f"    → 낮은 온도: 보수적이고 예측 가능한 생성")
    elif args.temperature < 1.0:
        print(f"    → 중간 온도: 균형잡힌 생성")
    else:
        print(f"    → 높은 온도: 창의적이고 다양한 생성")
    print(f"  최대 길이: {args.max_length} 토큰")
    print(f"  Top-k: {args.top_k}")
    print(f"  생성 샘플 수: {args.num_samples}")
    
    # 어휘 사전 로드
    print("\n1. 어휘 사전 로드...")
    with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    token_to_id = vocab['token_to_id']
    id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}
    vocab_size = vocab['vocab_size']
    
    print(f"   어휘 크기: {vocab_size:,}")
    
    # 모델 설정 로드
    print("\n2. 모델 로드...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"   모델 경로: {model_path}")
    print(f"   학습된 에폭: {checkpoint['epoch']}")
    print(f"   검증 Loss: {checkpoint['loss']:.4f}")
    
    # 모델 생성
    model = TransformerLanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout']
    ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   모델 로드 완료!")
    
    # 프롬프트
    prompt = args.prompt
    
    print(f"\n3. 텍스트 생성...")
    print(f"   시작 프롬프트: '{prompt}'")
    print("-"*70)
    
    # 텍스트 생성
    print(f"\n[텍스트 생성 중... (온도: {args.temperature})]")
    generated = generate_text(
        model=model,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        prompt=prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(f"\n생성된 텍스트:")
    print("="*70)
    try:
        print(generated)
    except UnicodeEncodeError:
        # 인코딩 오류 시 ASCII로 변환
        print(generated.encode('ascii', 'ignore').decode('ascii'))
    print("="*70)
    
    # 여러 버전 생성
    if args.num_samples > 1:
        print(f"\n4. 추가 생성 (총 {args.num_samples}개 샘플)...")
        print("-"*70)
        for i in range(args.num_samples - 1):
            print(f"\n[샘플 {i+2}]")
            print("-"*70)
            generated = generate_text(
                model=model,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            try:
                print(generated)
            except UnicodeEncodeError:
                print(generated.encode('ascii', 'ignore').decode('ascii'))
            print("-"*70)
    
    print("\n" + "="*70)
    print("텍스트 생성 완료!")
    print("="*70)

if __name__ == "__main__":
    main()

