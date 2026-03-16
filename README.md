# PicFind

PicFind는 로컬 사진 폴더를 인덱싱하고, 이미지 설명과 의미 기반 임베딩을 저장한 뒤, 자연어 문장으로 비슷한 사진을 찾는 도구입니다.

## 현재 MVP 범위

- 폴더를 순회하면서 이미지 파일 찾기
- BLIP로 짧은 설명 문장 생성
- CLIP으로 이미지 임베딩 생성
- 메타데이터, 설명, 임베딩을 SQLite에 저장
- 명령줄과 웹 UI에서 문장 검색 수행
- HEIC/HEIF 파일 읽기 지원
- 인덱싱 진행률 표시
- 배치 단위 중간 저장 지원

## 빠른 시작

1. Python 3.10 이상 환경을 준비합니다.
2. 가상환경을 만듭니다.
3. editable 모드로 설치합니다.
4. 사진 폴더를 인덱싱합니다.
5. 문장으로 검색합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
picfind init
picfind --no-caption index --source "D:\Photos"
picfind search --query "해변에서 뛰는 강아지"
```

## 설치 주의사항

- `torch` 는 2.6 이상이 필요합니다.
- 최근 `transformers` 는 보안 이슈 대응 때문에 일부 `.bin` 체크포인트 로딩 시 `torch>=2.6` 을 요구합니다.
- 설치된 `torch` 가 2.5.x 이하라면 업그레이드가 필요합니다.
- `transformers` 버전에 따라 CLIP 출력 타입이 약간 달라질 수 있어서, 프로젝트는 텐서와 모델 출력 객체를 모두 처리하도록 맞춰져 있습니다.

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## RTX 3070 8GB 기준 권장 사용법

- 기본 설정으로도 `CLIP + BLIP base` 조합은 충분히 시도할 수 있습니다.
- VRAM이 부족하거나 인덱싱 속도를 우선하면 `--no-caption` 으로 캡션 생성을 끄고 임베딩만 저장할 수 있습니다.
- 검색은 저장된 임베딩만 읽기 때문에 인덱싱보다 훨씬 가볍습니다.
- 인덱싱 중에는 전체 개수와 처리 진행률이 콘솔에 표시됩니다.
- 인덱싱 결과는 25장 단위로 SQLite에 저장되므로, 중간에 `Ctrl-C`로 끊어도 이미 저장된 배치는 유지됩니다.

```bash
picfind --no-caption index --source "D:\Photos"
```

## 지원 이미지 형식

- JPG / JPEG
- PNG
- BMP
- GIF
- WebP
- TIFF
- HEIC / HEIF

## 웹 UI 실행

```bash
streamlit run app.py
```

웹 UI에서는 다음 기능을 제공합니다.

- 자연어 질의 입력
- 상위 검색 결과 미리보기
- 파일 경로, 점수, 캡션 확인
- 결과 이미지를 썸네일로 확인

## 명령어

- `picfind init`: SQLite 데이터베이스 초기화
- `picfind index`: 새 사진 또는 변경된 사진 인덱싱
- `picfind search`: 자연어 문장으로 사진 검색
- `picfind stats`: 현재 인덱스 통계 출력

상세 설계와 결정 사항은 `docs/` 아래에 있습니다.
