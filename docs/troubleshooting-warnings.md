# 경고와 오류 구분 가이드

## 목적

실행 중 출력되는 메시지 중에서 무시 가능한 경고와 실제로 대응해야 하는 오류를 구분한다.

## 무시 가능한 경고 예시

### CLIP position_ids 로드 리포트

예시:

- `text_model.embeddings.position_ids | UNEXPECTED`
- `vision_model.embeddings.position_ids | UNEXPECTED`

의미:

- 체크포인트에 들어 있는 보조 항목과 현재 모델 클래스 해석 사이의 차이로 생기는 로드 리포트다.
- 대개 실제 추론에는 영향을 주지 않는다.

대응:

- 추가 오류가 없다면 우선 무시해도 된다.

### HF Hub 비인증 다운로드 경고

예시:

- `You are sending unauthenticated requests to the HF Hub ...`

의미:

- 인증 토큰 없이 모델을 내려받고 있다는 뜻이다.

대응:

- 기능은 동작한다.
- 속도를 높이려면 `HF_TOKEN`을 설정한다.

## 대응이 필요한 오류 예시

### torch 버전이 너무 낮음

예시:

- `require users to upgrade torch to at least v2.6`

의미:

- 현재 `transformers` 와 체크포인트 조합을 로드할 수 없는 환경이다.

대응:

- `torch` 를 2.6 이상으로 업그레이드한다.

### Florence-2 processor 호환성 오류

예시:

- `TokenizersBackend has no attribute additional_special_tokens`

의미:

- 현재 설치된 `transformers` 버전이 Florence-2 remote processor와 맞지 않는 경우다.

대응:

- `transformers>=4.49,<4.50` 로 맞춘다.

```bash
pip install "transformers>=4.49,<4.50" --force-reinstall
```
