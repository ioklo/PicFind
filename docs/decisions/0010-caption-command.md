# 결정 0010: caption 전용 명령으로 설명 생성과 인덱싱을 분리

## 상태

채택

## 배경

캡션은 검색의 핵심이 아니라 보조 정보에 가깝다. 따라서 임베딩 인덱싱과 캡션 생성을 한 번에 묶으면 시간이 오래 걸리고 운영이 불편하다.

## 결정

다음 구조를 도입한다.

- `index`: 이미지 임베딩 중심 인덱싱
- `caption`: DB에 저장된 이미지 경로를 기준으로 caption 필드만 갱신

caption 명령의 동작은 다음과 같다.

- 기본값은 기존 caption이 있으면 건너뜀
- `--overwrite` 는 실행 시작 시 기존 caption을 전부 비우고 전체 재생성
- 캡션 생성 모델은 `microsoft/Florence-2-base-ft` 로 고정
- 기본 캡션 프롬프트는 `<MORE_DETAILED_CAPTION>`
- 두 경로 모두 배치 저장과 중단 복구를 지원

## 결과

### 장점

- 대량 인덱싱을 `--no-caption` 으로 빠르게 끝낼 수 있다.
- caption 품질을 BLIP보다 더 자세한 Florence-2 경로로 높일 수 있다.
- 캡션 모델 교체나 재생성이 쉬워진다.
- 중간에 끊긴 caption 작업을 다시 이어서 수행하기 쉽다.

### 단점

- 인덱싱과 caption 생성이 별도 단계가 된다.
- `--overwrite` 실행 중 중단되면 일부 레코드는 빈 caption 상태로 남을 수 있다.
- Florence-2는 `trust_remote_code=True` 가 필요하다.

## 후속 작업

- `<DETAILED_CAPTION>` 과 `<MORE_DETAILED_CAPTION>` 결과 비교
- 상세 caption 품질 평가
