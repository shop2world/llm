import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import os

# 설정
CONFIG = {
    'vocab_size': 5169,
    'd_model': 256,  # 임베딩 차원
    'nhead': 8,  # 어텐션 헤드 수
    'num_layers': 4,  # Transformer 레이어 수
    'dim_feedforward': 1024,  # Feedforward 네트워크 차원
    'max_seq_length': 128,
    'dropout': 0.1,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints',
    'log_interval': 10
}

class LanguageModelDataset(Dataset):
    """언어 모델 학습용 데이터셋"""
    def __init__(self, input_sequences, target_sequences):
        self.inputs = torch.tensor(input_sequences, dtype=torch.long)
        self.targets = torch.tensor(target_sequences, dtype=torch.long)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
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
    """Transformer 기반 언어 모델"""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        self.d_model = d_model
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer 인코더
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 출력 레이어
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch_size)
        # 임베딩 및 위치 인코딩
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Transformer 인코더
        output = self.transformer(src, src_key_padding_mask=None, mask=src_mask)
        
        # 출력
        output = self.fc_out(output)
        return output  # (seq_len, batch_size, vocab_size)
    
    def generate_causal_mask(self, sz):
        """Causal mask 생성 (미래 토큰을 볼 수 없도록)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

def create_causal_mask(seq_len, device):
    """Causal attention mask 생성"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, config):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Causal mask 생성
    causal_mask = create_causal_mask(config['max_seq_length'], device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # 데이터를 device로 이동
        # inputs shape: (batch_size, seq_len)
        # targets shape: (batch_size, seq_len)
        inputs = inputs.transpose(0, 1).to(device)  # (seq_len, batch_size)
        targets = targets.transpose(0, 1).to(device)  # (seq_len, batch_size)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, src_mask=causal_mask)  # (seq_len, batch_size, vocab_size)
        
        # Loss 계산
        # outputs를 (seq_len * batch_size, vocab_size)로 reshape
        outputs = outputs.reshape(-1, outputs.size(-1))  # (seq_len * batch_size, vocab_size)
        targets = targets.reshape(-1)  # (seq_len * batch_size)
        
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 진행 상황 업데이트
        if batch_idx % config['log_interval'] == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate(model, dataloader, criterion, device, config):
    """모델 평가"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    causal_mask = create_causal_mask(config['max_seq_length'], device)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.transpose(0, 1).to(device)
            targets = targets.transpose(0, 1).to(device)
            
            outputs = model(inputs, src_mask=causal_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def save_model(model, optimizer, epoch, loss, config, filepath):
    """모델 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"모델 저장: {filepath}")

def load_model(filepath, device):
    """모델 로드"""
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

def main():
    print("="*70)
    print("햄릿 언어 모델 학습 시작")
    print("="*70)
    
    # 설정 출력
    print("\n학습 설정:")
    print("-"*70)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("-"*70)
    
    # 디바이스 설정
    device = torch.device(CONFIG['device'])
    print(f"\n사용 디바이스: {device}")
    
    # 데이터 로드
    print("\n1. 데이터 로드...")
    data = np.load('hamlet_sequences_masked.npz')
    input_sequences = data['input_sequences']
    target_sequences = data['target_sequences']
    
    print(f"   입력 시퀀스 shape: {input_sequences.shape}")
    print(f"   정답 시퀀스 shape: {target_sequences.shape}")
    
    # Train/Validation 분할 (80/20)
    num_samples = len(input_sequences)
    train_size = int(0.8 * num_samples)
    
    train_inputs = input_sequences[:train_size]
    train_targets = target_sequences[:train_size]
    val_inputs = input_sequences[train_size:]
    val_targets = target_sequences[train_size:]
    
    print(f"   학습 데이터: {len(train_inputs)}개")
    print(f"   검증 데이터: {len(val_inputs)}개")
    
    # 데이터셋 및 데이터로더 생성
    print("\n2. 데이터로더 생성...")
    train_dataset = LanguageModelDataset(train_inputs, train_targets)
    val_dataset = LanguageModelDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 모델 생성
    print("\n3. 모델 생성...")
    model = TransformerLanguageModel(
        vocab_size=CONFIG['vocab_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        max_seq_length=CONFIG['max_seq_length'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   모델 파라미터 수: {num_params:,}")
    
    # Loss 함수 및 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 체크포인트 디렉토리 생성
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 학습 시작
    print("\n4. 학습 시작...")
    print("="*70)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, CONFIG)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = evaluate(model, val_loader, criterion, device, CONFIG)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')
            save_model(model, optimizer, epoch, val_loss, CONFIG, save_path)
        
        # 주기적으로 모델 저장
        if epoch % 5 == 0:
            save_path = os.path.join(CONFIG['save_dir'], f'model_epoch_{epoch}.pt')
            save_model(model, optimizer, epoch, val_loss, CONFIG, save_path)
        
        print("-"*70)
    
    # 최종 모델 저장
    final_save_path = os.path.join(CONFIG['save_dir'], 'final_model.pt')
    save_model(model, optimizer, CONFIG['num_epochs'], val_loss, CONFIG, final_save_path)
    
    # 학습 결과 저장
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': CONFIG
    }
    
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("학습 완료!")
    print("="*70)
    print(f"\n최종 검증 Loss: {val_loss:.4f}")
    print(f"최고 검증 Loss: {best_val_loss:.4f}")
    print(f"\n저장된 파일:")
    print(f"  - {CONFIG['save_dir']}/best_model.pt: 최고 성능 모델")
    print(f"  - {CONFIG['save_dir']}/final_model.pt: 최종 모델")
    print(f"  - training_results.json: 학습 결과")
    print("="*70)

if __name__ == "__main__":
    main()

