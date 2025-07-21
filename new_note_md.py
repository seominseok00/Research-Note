import os
import sys
from datetime import datetime
import re

def sanitize_filename(title):
    """파일 이름으로 사용할 수 있도록 문자열 정리 (공백은 _로 유지)"""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)  # 금지된 문자 제거
    return sanitized.strip().replace(" ", "_")      # 공백은 _로

def create_research_note(paper_title):
    # 오늘 날짜
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")

    # 논문 제목 기반 파일 이름
    paper_title = sanitize_filename(paper_title)
    filename = f"{paper_title}.md"

    # 베이스 폴더 설정 (Research-Note 폴더 바로 아래에 저장)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(script_dir, "Research-Note", paper_title)
    os.makedirs(base_folder, exist_ok=True)

    filepath = os.path.join(base_folder, filename)

    # 마크다운 템플릿
    template = f"""# {paper_title}

> 작성일: {date_str}

---

## 🪄 Abstract


---

## 🔍 Introduction


---

## 📚 Related Works


---

## 🛠️ Method


---

## 🔬 Experiment


---

## 💬 Discussion


---

## 📝 Conclusion


"""

    # 파일 생성
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(template)
        print(f"✅ 논문 노트 생성 완료: {filepath}")
    else:
        print(f"⚠️ 이미 존재하는 파일입니다: {filepath}")

# 명령줄에서 사용할 경우
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 논문 제목을 인자로 입력해주세요.")
        print("사용법: python create_note.py \"Your Paper Title Here\"")
    else:
        paper_title = sys.argv[1]
        create_research_note(paper_title)