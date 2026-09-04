# 미리 계산된 산물 (F5a — GPU 없이 재생)

`CTP49906_avllm_molab_kr.py`에서 `USE_PRECOMPUTED = True`로 두면, 노트북은 모델을
실행하는 대신 이 디렉터리의 파일로 고정 파라미터 **W7-W9** 그래프를 재생합니다 —
molab의 GPU를 쓸 수 없을 때도 수업이 굴러가도록 하기 위한 것입니다
(`USE_PRECOMPUTED`의 기본값은 `False`이며, 이것은 비상용 모드입니다).

영어판 노트북의 팩은 [`../precomputed/`](../precomputed/)에 따로 있습니다. 두 팩은
프롬프트가 다르므로 서로 바꿔 쓸 수 없습니다. 재생 시 `meta.json` 검증이 이를
막습니다.

## 여기에 들어 있는 것

`scripts/generate_precompute.py`가 샘플 클립 `assets/02321.mp4`에 대해
**한국어 프롬프트로** 생성한 것입니다.

| 파일 | 쓰이는 곳 |
|---|---|
| `logit_lens_audio_token_analysis.csv` | W8 Logit-lens 다양성 / 최빈 비율 그래프 |
| `captions.json` | 생성된 캡션 + 기준선/녹아웃 캡션 |
| `attention_summary.json` | 축약된 기준선 + 녹아웃 생성-쿼리 어텐션 질량 행렬 (디코딩 스텝만; Δ는 노트북에서 계산) |
| `meta.json` | 클립 / 프롬프트 / 레이어 수 출처 정보 |

원본 어텐션 텐서는 **커밋하지 않습니다**(eager prefill 레이어 하나가 fp32로 약
64 MB) — 축약된 행렬만 담습니다. 재생할 때 `meta.json`을 노트북의 고정 클립,
프롬프트, 프레임 수, 생성 길이, 캡처 레이어, 녹아웃 규칙, 모델, 그리고 불변 모델
revision과 대조합니다. 하나라도 어긋나면 오래된 결과에 새 이름표를 붙이는 대신
시끄럽게 실패합니다.

## 다시 생성하기

GPU가 있는 머신에서 저장소 루트를 기준으로 한 번 실행하세요.

```bash
python avllm_interpretability/scripts/generate_precompute.py \
  --out_dir avllm_interpretability/precomputed_kr \
  --logit_prompt "영상에서 들리는 소리를 설명해 주세요" \
  --attention_prompt "영상에서 보이는 것과 들리는 소리를 설명해 주세요"
```

샘플 클립, 프롬프트, 모델이 바뀔 때마다 다시 생성하세요. 대화형 플레이그라운드와
티처 포싱 섹션은 미리 계산할 수 **없으므로**(학생이 임의의 입력을 넣습니다) 여전히
GPU가 필요하고, `USE_PRECOMPUTED = True`인 상태로 제출하면 시끄럽게 실패합니다.
