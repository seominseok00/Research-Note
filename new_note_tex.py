import os
import sys
from datetime import datetime
import re

def sanitize_filename(title):
    """파일 이름으로 사용할 수 있도록 문자열 정리 (공백은 _로 유지)"""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
    return sanitized.strip().replace(" ", "_")

def create_tex_note(paper_title):
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")

    safe_title = sanitize_filename(paper_title)

    # 디렉토리 설정: Research-Note/논문제목/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(script_dir, "Research-Note", safe_title)
    os.makedirs(base_folder, exist_ok=True)

    # 파일 경로 설정
    tex_filename = f"{safe_title}.tex"
    filepath = os.path.join(base_folder, tex_filename)

    # LaTeX 템플릿
    template = f"""\\documentclass{{article}}
\\input{{../macros/macros_math.tex}}

\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath, amssymb}}
\\usepackage[colorlinks=true,linkcolor=blue,citecolor=black]{{hyperref}}

\\title{{{paper_title}}}
\\date{{{date_str}}}
\\author{{}}

\\begin{{document}}

\\maketitle

\\tableofcontents

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap Abstract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section*{{Abstract}}

% Your abstract here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap1 Introduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{{Introduction}}

% Your introduction here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap2 Related Works
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{{Related Works}}

% Related work goes here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap3 Method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{{Method}}

% Method description

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap4 Experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{{Experiment}}

% Experiment setup/results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap5 Discussion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{{Discussion}}

% Discussion points

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chap6 Conclusion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\section{{Conclusion}}

% Final conclusion

\\end{{document}}
"""

    # 파일 생성
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(template)
        print(f"✅ LaTeX 파일 생성 완료: {filepath}")
    else:
        print(f"⚠️ 이미 존재하는 LaTeX 파일입니다: {filepath}")

# 명령줄에서 사용할 경우
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 논문 제목을 인자로 입력해주세요.")
        print("사용법: python create_note.py \"Your Paper Title Here\"")
    else:
        paper_title = sys.argv[1]
        create_tex_note(paper_title)
