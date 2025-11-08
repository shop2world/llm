# 햄릿 언어 모델 (Hamlet Language Model)

셰익스피어의 햄릿 텍스트로 학습한 Transformer 기반 언어 모델입니다. 이 프로젝트는 텍스트 데이터 수집부터 모델 학습, 텍스트 생성까지의 전체 과정을 단계별로 안내합니다.

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [필수 요구사항](#필수-요구사항)
3. [설치 방법](#설치-방법)
4. [사용 방법](#사용-방법)
   - [1단계: 텍스트 수집 및 정리](#1단계-텍스트-수집-및-정리)
   - [2단계: 토큰화 및 숫자 변환](#2단계-토큰화-및-숫자-변환)
   - [3단계: 학습 데이터 준비](#3단계-학습-데이터-준비)
   - [4단계: 모델 학습](#4단계-모델-학습)
   - [5단계: 텍스트 생성](#5단계-텍스트-생성)
5. [파일 구조](#파일-구조)
6. [모델 아키텍처](#모델-아키텍처)
7. [고급 사용법](#고급-사용법)

## 🎯 프로젝트 개요

이 프로젝트는 언어 모델(Language Model)의 작동 원리를 이해하기 위한 교육용 프로젝트입니다. 다음과 같은 과정을 통해 LLM이 어떻게 작동하는지 학습할 수 있습니다:

- **텍스트 전처리**: 원시 텍스트를 모델이 이해할 수 있는 형태로 변환
- **토큰화**: 텍스트를 작은 단위(토큰)로 분할하고 각 토큰에 고유 번호 부여
- **학습 데이터 준비**: 입력과 정답을 분리하여 모델이 다음 단어를 예측하도록 설정
- **모델 학습**: Transformer 아키텍처를 사용하여 언어 패턴 학습
- **텍스트 생성**: 학습된 모델로 새로운 텍스트 생성

## 📦 필수 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)

## 🔧 설치 방법

1. 저장소 클론 또는 파일 다운로드

2. 필요한 패키지 설치:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install requests beautifulsoup4 tqdm numpy
```

또는 한 번에 설치:

```bash
pip install torch torchvision torchaudio requests beautifulsoup4 tqdm numpy --index-url https://download.pytorch.org/whl/cpu
```

## 📖 사용 방법

### 1단계: 텍스트 수집 및 정리

먼저 Project Gutenberg에서 햄릿 텍스트를 다운로드하고 불필요한 요소를 제거합니다.

```bash
python download_hamlet.py
```

**실행 결과:**
- `hamlet_clean.txt`: 정리된 텍스트 파일
- `hamlet_raw.txt`: 원본 텍스트 파일 (참고용)

**작동 원리:**
- Project Gutenberg에서 햄릿 텍스트를 다운로드
- HTML 태그, 페이지 번호, 라이선스 정보 등 불필요한 요소 제거
- 깨끗한 텍스트만 추출하여 저장

### 2단계: 토큰화 및 숫자 변환

텍스트를 작은 조각(토큰)으로 자르고, 각 토큰에 고유 번호를 부여하여 숫자로 변환합니다.

```bash
python tokenize_hamlet.py
```

**실행 결과:**
- `hamlet_encoded.txt`: 숫자 시퀀스 (공백으로 구분)
- `hamlet_encoded.json`: 숫자 시퀀스 (JSON 배열)
- `hamlet_vocabulary.json`: 어휘 사전 (토큰 ↔ ID 매핑)
- `hamlet_tokenization_stats.json`: 통계 정보

**작동 원리:**
1. **토큰화**: 텍스트를 단어와 구두점으로 분할
   - 예: "To be, or not to be" → ['To', 'be', ',', 'or', 'not', 'to', 'be']
2. **어휘 사전 생성**: 각 고유 토큰에 번호 부여
   - 예: 'To' → 471, 'be' → 309, ',' → 8
3. **숫자 변환**: 텍스트를 숫자 시퀀스로 변환
   - 예: "To be, or not to be" → [471, 309, 8, 594, 220, 70, 309]

**확인 방법:**
```bash
python check_phrase.py  # "To be, or not to be"가 어떻게 변환되었는지 확인
```

### 3단계: 학습 데이터 준비

입력을 오른쪽으로 한 칸 밀어서 정답으로 설정하여, 모델이 다음 단어를 예측하도록 데이터를 준비합니다.

```bash
python create_training_data.py
```

**실행 결과:**
- `hamlet_training_data.json`: 입력-정답 쌍 (JSON)
- `hamlet_training_data.npz`: 입력-정답 쌍 (NumPy 배열)
- `hamlet_training_stats.json`: 통계 정보

**작동 원리:**
- 원본 시퀀스: [t1, t2, t3, ..., tn]
- 입력: [t1, t2, t3, ..., t(n-1)]
- 정답: [t2, t3, t4, ..., tn] (입력을 한 칸 오른쪽으로 이동)

각 위치 i에서 모델은 입력[i]를 보고 정답[i] (다음 토큰)를 예측합니다.

**정답이 보이지 않도록 마스킹:**
```bash
python create_masked_training_data.py
```

**실행 결과:**
- `hamlet_sequences_masked.json`: 고정 길이 시퀀스로 나눈 입력-정답 쌍
- `hamlet_sequences_masked.npz`: NumPy 배열 형식
- `hamlet_masked_stats.json`: 통계 정보

이 방법은 모델이 정답을 미리 볼 수 없도록 보장합니다.

### 4단계: 모델 학습

Transformer 아키텍처를 사용하여 언어 모델을 학습시킵니다.

```bash
python train_model.py
```

**학습 설정 (기본값):**
- 모델 크기: 5.8M 파라미터
- 임베딩 차원: 256
- Transformer 레이어: 4개
- 어텐션 헤드: 8개
- 시퀀스 길이: 128 토큰
- 배치 크기: 8
- 학습 에폭: 10
- 학습률: 0.0001

**실행 결과:**
- `checkpoints/best_model.pt`: 최고 성능 모델
- `checkpoints/final_model.pt`: 최종 모델
- `checkpoints/model_epoch_5.pt`: 5 에폭 모델
- `checkpoints/model_epoch_10.pt`: 10 에폭 모델
- `training_results.json`: 학습 결과 및 통계

**학습 과정:**
1. 데이터를 학습(80%)과 검증(20%) 세트로 분할
2. 각 에폭마다:
   - 학습 데이터로 모델 가중치 업데이트
   - 검증 데이터로 성능 평가
   - 최고 성능 모델 저장
3. Loss가 감소하는 것을 확인

**예상 소요 시간:**
- CPU: 약 3-5분 (10 에폭)
- GPU: 약 1-2분 (10 에폭)

### 5단계: 텍스트 생성

학습된 모델을 사용하여 새로운 텍스트를 생성합니다.

#### 기본 사용법

```bash
python generate_text.py
```

기본 프롬프트 "To be, or not to be,"로 텍스트를 생성합니다.

#### 온도 조절 (창의성 조절)

온도(temperature)는 생성 텍스트의 창의성을 조절합니다:

```bash
# 낮은 온도 (0.2): 보수적이고 예측 가능한 생성
python generate_text.py --temperature 0.2

# 중간 온도 (0.8): 균형잡힌 생성 (기본값)
python generate_text.py --temperature 0.8

# 높은 온도 (1.5): 창의적이고 다양한 생성
python generate_text.py --temperature 1.5
```

**온도 가이드:**
- **0.1-0.3**: 매우 보수적, 반복적 패턴
- **0.4-0.7**: 균형잡힌 생성 (권장)
- **0.8-1.2**: 창의적이면서도 일관성 유지
- **1.3-2.0**: 매우 창의적, 예측 불가능

#### 다른 옵션들

```bash
# 커스텀 프롬프트 사용
python generate_text.py --prompt "HAMLET. What a piece of work is a man,"

# 더 긴 텍스트 생성
python generate_text.py --max_length 200

# 여러 샘플 생성
python generate_text.py --num_samples 5

# 모든 옵션 조합
python generate_text.py --temperature 0.9 --max_length 150 --prompt "To be, or not to be," --num_samples 3
```

#### Python에서 직접 사용하기

```python
import torch
import json
from generate_text import TransformerLanguageModel, generate_text, tokenize_text, encode_text, decode_text

# 모델 및 어휘 사전 로드
device = torch.device('cpu')
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
config = checkpoint['config']

with open('hamlet_vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

token_to_id = vocab['token_to_id']
id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}

# 모델 생성 및 로드
model = TransformerLanguageModel(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    dim_feedforward=config['dim_feedforward'],
    max_seq_length=config['max_seq_length'],
    dropout=config['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 텍스트 생성
prompt = "To be, or not to be,"
generated = generate_text(
    model=model,
    token_to_id=token_to_id,
    id_to_token=id_to_token,
    prompt=prompt,
    max_length=50,
    temperature=0.8,
    top_k=50,
    device=device
)

print(f"프롬프트: {prompt}")
print(f"생성된 텍스트: {generated}")
```

## 📁 파일 구조

```
.
├── README.md                          # 이 파일
├── download_hamlet.py                 # 1단계: 텍스트 수집 및 정리
├── tokenize_hamlet.py                 # 2단계: 토큰화 및 숫자 변환
├── create_training_data.py            # 3단계: 학습 데이터 준비 (기본)
├── create_masked_training_data.py     # 3단계: 학습 데이터 준비 (마스킹)
├── train_model.py                     # 4단계: 모델 학습
├── generate_text.py                   # 5단계: 텍스트 생성
│
├── check_phrase.py                    # 유틸리티: 특정 문구 확인
├── check_tokens.py                    # 유틸리티: 토큰 통계 확인
├── verify_training_data.py            # 유틸리티: 학습 데이터 검증
├── verify_masked_data.py              # 유틸리티: 마스킹 데이터 검증
│
├── hamlet_clean.txt                   # 정리된 텍스트
├── hamlet_raw.txt                     # 원본 텍스트
├── hamlet_encoded.json                # 인코딩된 숫자 시퀀스
├── hamlet_vocabulary.json             # 어휘 사전
├── hamlet_training_data.json          # 학습 데이터
├── hamlet_sequences_masked.json       # 마스킹된 학습 데이터
│
└── checkpoints/                       # 학습된 모델 저장 폴더
    ├── best_model.pt                  # 최고 성능 모델
    ├── final_model.pt                 # 최종 모델
    └── model_epoch_*.pt               # 특정 에폭의 모델
```

## 🏗️ 모델 아키텍처

이 프로젝트는 **Transformer** 아키텍처를 사용합니다:

### 주요 구성 요소

1. **임베딩 레이어 (Embedding Layer)**
   - 각 토큰을 고정 크기 벡터로 변환
   - 차원: 256

2. **위치 인코딩 (Positional Encoding)**
   - 토큰의 위치 정보를 추가
   - Sinusoidal 함수 사용

3. **Transformer 인코더 (Transformer Encoder)**
   - 4개의 레이어
   - 각 레이어는:
     - Multi-Head Self-Attention (8개 헤드)
     - Feed-Forward Network (1024 차원)
     - Residual Connection & Layer Normalization

4. **출력 레이어 (Output Layer)**
   - 어휘 크기(5,169)만큼의 확률 분포 출력
   - 다음 토큰 예측

### 모델 크기

- **총 파라미터**: 약 5.8M
- **어휘 크기**: 5,169 토큰
- **최대 시퀀스 길이**: 128 토큰

## 🎓 고급 사용법

### 학습 설정 변경

`train_model.py` 파일의 `CONFIG` 딕셔너리를 수정하여 모델 크기와 학습 설정을 변경할 수 있습니다:

```python
CONFIG = {
    'vocab_size': 5169,
    'd_model': 256,          # 임베딩 차원 (더 크게 하면 더 강력하지만 느림)
    'nhead': 8,              # 어텐션 헤드 수
    'num_layers': 4,         # Transformer 레이어 수
    'dim_feedforward': 1024, # Feedforward 네트워크 차원
    'max_seq_length': 128,   # 최대 시퀀스 길이
    'dropout': 0.1,          # 드롭아웃 비율
    'batch_size': 8,         # 배치 크기
    'learning_rate': 1e-4,   # 학습률
    'num_epochs': 10,        # 학습 에폭 수
}
```

### 더 긴 텍스트 생성

기본적으로 모델은 128 토큰 길이의 시퀀스로 학습되었지만, 더 긴 텍스트를 생성하려면:

1. 학습 시 `max_seq_length`를 더 크게 설정
2. 생성 시 `--max_length` 옵션으로 원하는 길이 지정

### 다른 텍스트로 학습하기

1. `hamlet_clean.txt` 대신 원하는 텍스트 파일 준비
2. `tokenize_hamlet.py`에서 파일 경로 수정
3. 나머지 단계는 동일하게 진행

## 📊 학습 결과 예시

10 에폭 학습 후 예상 결과:

- **초기 Loss**: ~7.5
- **최종 Loss**: ~5.8
- **개선율**: 약 23% 감소

## ❓ 문제 해결

### 메모리 부족 오류

- `batch_size`를 줄이기 (예: 8 → 4)
- `max_seq_length`를 줄이기 (예: 128 → 64)
- `d_model`을 줄이기 (예: 256 → 128)

### 생성 텍스트가 반복적임

- 온도를 높이기 (예: 0.8 → 1.2)
- `top_k` 값을 조정하기

### 생성 텍스트가 이상함

- 온도를 낮추기 (예: 1.5 → 0.8)
- 더 많은 에폭으로 학습하기

## 📚 참고 자료

- [Transformer 논문](https://arxiv.org/abs/1706.03762)
- [Project Gutenberg](https://www.gutenberg.org/)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 햄릿 텍스트는 저작권이 만료된 공개 도메인 작품입니다.

## 🤝 기여

이 프로젝트는 LLM의 작동 원리를 이해하기 위한 교육용 프로젝트입니다. 개선 사항이나 버그를 발견하시면 이슈를 등록해주세요.

---

**즐거운 학습 되세요! 🎭**

