import requests
import re
from bs4 import BeautifulSoup

def download_hamlet():
    """Project Gutenberg에서 햄릿 텍스트 다운로드"""
    # Project Gutenberg의 햄릿 URL (텍스트 버전)
    url = "https://www.gutenberg.org/files/1524/1524-0.txt"
    
    try:
        print("햄릿 텍스트 다운로드 중...")
        response = requests.get(url, timeout=30)
        response.encoding = 'utf-8'
        text = response.text
        print(f"다운로드 완료: {len(text)} 문자")
        return text
    except Exception as e:
        print(f"다운로드 실패: {e}")
        # 대체 방법: 다른 소스 시도
        try:
            url2 = "https://www.gutenberg.org/cache/epub/1524/pg1524.txt"
            response = requests.get(url2, timeout=30)
            response.encoding = 'utf-8'
            text = response.text
            print(f"대체 소스에서 다운로드 완료: {len(text)} 문자")
            return text
        except Exception as e2:
            print(f"대체 소스도 실패: {e2}")
            return None

def clean_hamlet_text(text):
    """햄릿 텍스트 정리"""
    if not text:
        return None
    
    original_length = len(text)
    print("텍스트 정리 중...")
    
    # 1. Project Gutenberg 헤더 제거 (START OF THE PROJECT GUTENBERG부터 시작)
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "START OF THIS PROJECT GUTENBERG",
    ]
    
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # 다음 줄로 이동
            start_idx = text.find('\n', idx) + 1
            break
    
    # 2. Project Gutenberg 푸터 제거 (END OF THE PROJECT GUTENBERG까지)
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "END OF THIS PROJECT GUTENBERG",
    ]
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    
    text = text[start_idx:end_idx]
    
    # 3. HTML 태그 제거 (있다면)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. 페이지 번호 제거 (예: [Page 1], Page 1 등)
    text = re.sub(r'\[?Page\s+\d+\]?', '', text, flags=re.IGNORECASE)
    
    # 5. Project Gutenberg 라이선스/저작권 정보 제거
    text = re.sub(r'This eBook.*?Project Gutenberg.*?\n', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 6. 과도한 공백 정리
    text = re.sub(r'[ \t]+', ' ', text)  # 여러 공백을 하나로
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3개 이상의 연속 줄바꿈을 2개로
    
    # 7. 줄 끝의 공백 제거
    lines = text.split('\n')
    text = '\n'.join(line.rstrip() for line in lines)
    
    # 8. 앞뒤 공백 제거
    text = text.strip()
    
    cleaned_length = len(text)
    removed = original_length - cleaned_length
    print(f"정리 완료: {cleaned_length} 문자 (제거됨: {removed} 문자)")
    return text

def main():
    # 텍스트 다운로드
    raw_text = download_hamlet()
    
    if raw_text:
        # 텍스트 정리
        clean_text = clean_hamlet_text(raw_text)
        
        if clean_text:
            # 정리된 텍스트 저장
            output_file = "hamlet_clean.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            # 통계 정보
            lines = clean_text.split('\n')
            words = len(clean_text.split())
            chars = len(clean_text)
            
            print(f"\n{'='*50}")
            print("정리 완료!")
            print(f"{'='*50}")
            print(f"정리된 텍스트: '{output_file}'")
            print(f"  - 총 문자 수: {chars:,}")
            print(f"  - 총 줄 수: {len(lines):,}")
            print(f"  - 총 단어 수: {words:,}")
            print(f"\n원본 텍스트: 'hamlet_raw.txt' (참고용)")
            print(f"{'='*50}")
        else:
            print("텍스트 정리 실패")
    else:
        print("텍스트 다운로드 실패")

if __name__ == "__main__":
    main()

