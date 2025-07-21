# 📝 소개

이 프로젝트는 research note를 자동으로 생성해주는 파이썬 스크립트를 제공합니다.

생성되는 마크다운 파일에는 아래와 같은 구조의 템플릿이 자동 포함됩니다:

```markdown
# Your Paper Title

> 작성일: 2025-07-21

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
```





---

## 🚀 alias 설정 (한 줄 커맨드 실행용)

### Mac

1. 터미널에서 다음 명령어 실행

```
vi ~/.zshrc
```

2. alias 추가

```
alias new-note-md="python ~/Research-Note/new_note_md.py"
alias new-note-tex="python ~/Research-Note/new_note_tex.py"
```

3. 설정 적용

```
source ~/.zshrc
```





----

## ✅ 사용법

```
new-note-md "Attention is All You Need"
```

터미널에서 위 명령어를 실행하면 자동으로 다음과 같은 구조로 노트가 생성됩니다.

```
Research-Note/
└── Attention_is_All_You_Need/
    └── Attention_is_All_You_Need.md
```

내용은 마크다운 템플릿으로 채워집니다. (new-note-tex 명령어로 실행시 LaTex 템플릿으로 생성됩니다)