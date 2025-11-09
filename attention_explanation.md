# 어텐션 메커니즘 사용 부분 설명

## 📍 어텐션 메커니즘이 사용되는 위치

### 1. TransformerEncoderLayer 내부 (핵심)

**파일**: `train_model.py`, `generate_text.py`

```python
# train_model.py 라인 67-74
encoder_layers = nn.TransformerEncoderLayer(
    d_model=d_model,           # 256
    nhead=nhead,               # 8 (Multi-Head Attention의 헤드 수)
    dim_feedforward=dim_feedforward,  # 1024
    dropout=dropout,
    batch_first=False
)
self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
```

**설명**:
- `nn.TransformerEncoderLayer`는 내부적으로 **Multi-Head Self-Attention** 메커니즘을 포함합니다
- `nhead=8`은 8개의 어텐션 헤드를 사용한다는 의미입니다
- 각 헤드는 독립적으로 어텐션을 계산하고, 결과를 결합합니다

### 2. Forward Pass에서 어텐션 호출

**파일**: `train_model.py` 라인 89-101

```python
def forward(self, src, src_mask=None):
    # 임베딩 및 위치 인코딩
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    src = self.dropout(src)
    
    # Transformer 인코더 (어텐션 메커니즘이 여기서 실행됨)
    output = self.transformer(src, src_key_padding_mask=None, mask=src_mask)
    
    # 출력
    output = self.fc_out(output)
    return output
```

**설명**:
- `self.transformer()` 호출 시 내부적으로 어텐션 메커니즘이 실행됩니다
- `src_mask`는 Causal Attention Mask로, 미래 토큰을 볼 수 없도록 합니다

### 3. Causal Attention Mask 생성

**파일**: `train_model.py` 라인 103-107, `generate_text.py` 라인 60-63

```python
def create_causal_mask(seq_len, device):
    """Causal mask 생성 (미래 토큰을 볼 수 없도록)"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)
```

**설명**:
- 이 마스크는 어텐션 메커니즘이 미래 토큰을 보지 못하도록 합니다
- 상삼각 행렬(upper triangular matrix)을 만들어서 미래 위치를 `-inf`로 설정
- 언어 모델에서 다음 단어를 예측할 때 미래 정보를 사용하지 않도록 보장

### 4. 학습 및 생성 시 Mask 사용

**파일**: `train_model.py` 라인 120-150

```python
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, config):
    # Causal mask 생성
    causal_mask = create_causal_mask(config['max_seq_length'], device)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.transpose(0, 1).to(device)
        targets = targets.transpose(0, 1).to(device)
        
        # Forward pass (어텐션 메커니즘 실행)
        outputs = model(inputs, src_mask=causal_mask)  # <-- 여기서 어텐션 실행
```

**파일**: `generate_text.py` 라인 140-160

```python
def generate_text(...):
    # Causal mask 생성
    causal_mask = create_causal_mask(seq_len, device)
    
    # 모델 예측 (어텐션 메커니즘 실행)
    outputs = model(input_tensor, src_mask=causal_mask)  # <-- 여기서 어텐션 실행
```

## 🔍 어텐션 메커니즘이 작동하는 방식

### Multi-Head Self-Attention 내부 동작 (개념적 설명)

PyTorch의 `TransformerEncoderLayer` 내부에서는 다음과 같은 어텐션 계산이 이루어집니다:

1. **Query, Key, Value 생성**:
   ```
   Q = src × W_q  (Query)
   K = src × W_k  (Key)
   V = src × W_v  (Value)
   ```

2. **어텐션 스코어 계산**:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ```

3. **Multi-Head로 분할 및 결합**:
   - 8개의 헤드로 나누어 각각 어텐션 계산
   - 결과를 결합(concatenate)하여 최종 출력 생성

4. **Causal Mask 적용**:
   - 마스크를 통해 미래 토큰의 어텐션 스코어를 `-inf`로 설정
   - softmax 후 미래 토큰의 가중치는 0이 됨

## 📊 어텐션 메커니즘 설정

현재 프로젝트의 어텐션 설정:

- **어텐션 헤드 수**: 8개 (`nhead=8`)
- **임베딩 차원**: 256 (`d_model=256`)
- **각 헤드의 차원**: 256 / 8 = 32
- **레이어 수**: 4개 (각 레이어마다 어텐션 실행)
- **마스크 타입**: Causal Mask (미래 토큰 차단)

## 💡 어텐션 메커니즘의 역할

1. **문맥 이해**: 각 토큰이 시퀀스 내 다른 토큰들과의 관계를 학습
2. **장거리 의존성**: 멀리 떨어진 토큰들 간의 관계도 포착
3. **가중치 계산**: 어떤 토큰에 더 집중할지 자동으로 학습
4. **병렬 처리**: 모든 토큰 쌍의 어텐션을 동시에 계산 가능

## 🔧 어텐션 메커니즘 커스터마이징

어텐션 메커니즘을 직접 구현하려면:

```python
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Query, Key, Value 생성
        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # 어텐션 스코어 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 마스크 적용
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 및 Value와 곱하기
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # 헤드 결합
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.W_o(attn_output)
```

## 📝 요약

**어텐션 메커니즘이 사용되는 주요 위치:**

1. ✅ `nn.TransformerEncoderLayer` - Multi-Head Self-Attention 내장
2. ✅ `self.transformer()` 호출 시 - 어텐션 메커니즘 실행
3. ✅ `create_causal_mask()` - Causal Attention Mask 생성
4. ✅ 학습 및 생성 시 `src_mask` 파라미터로 전달

**현재 프로젝트에서는 PyTorch의 내장 구현을 사용**하고 있어, 어텐션 메커니즘이 자동으로 실행됩니다.

