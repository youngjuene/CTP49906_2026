# 오디오-비주얼 대형 언어 모델은 정말로 보고 듣는가?

<p align="center">
  <a href="https://arxiv.org/abs/2604.02605">
    <img src="https://img.shields.io/badge/arXiv-2604.02605-b31b1b" alt="arXiv"/>
  </a>
  <a href="https://avllm-interpretability.github.io">
    <img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"/>
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>한국어</b>
</p>

논문에서 수행한 실험 코드이며, 대표 모델로 Qwen 2.5 Omni를 사용합니다.


## 설치
```bash
pip3 install -r requirements.txt
```

## 실험

### Logit Lens 실험
```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

### Attention Knockout 실험
```bash
python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

## 수업용 노트북: `CTP49906_avllm_molab.py`

`CTP49906_avllm_molab.py`는 위의 두 실험을 **강의용 워크스루 + 실시간 플레이그라운드**로
바꿔 놓은, 독립 실행 가능한 [marimo](https://marimo.io) 노트북입니다. [molab](https://marimo.io/molab)
(무료 호스팅 GPU 런타임)에서 열어, 우선 위에서 아래로 한 번 읽어 동작 원리를 파악한 뒤,
직접 *만져 보도록* — 영상, 프롬프트, 프레임 수, 그리고 무엇보다 **어텐션 녹아웃(attention
knockout)**을 바꿔 가며 — 설계되었습니다. 목표는 논문의 핵심 질문에 대한 감을 기르는
것입니다: **오디오-비주얼 LLM이 답할 때, 정말로 본 것과 들은 것을 모두 사용하는가?**

### 왜 필요한가

*"어떤 사람이 피아노를 치고 있다"* 같은 캡션은, 모델이 영상을 봤든, 소리를 들었든, 아니면
그냥 언어적 사전 지식(language prior)에 기댔든 똑같이 그럴듯해 보입니다. 이 노트북은 그런
경우들을 구분해 낼 두 가지 해석 가능성(interpretability) 도구를 학생들에게 제공합니다.

- **Logit Lens** — 각 레이어에서 모델의 *중간* 예측을 디코딩하되 **오디오 토큰 위치**에서
  읽어 냅니다. 그래서 네트워크를 위로 올라가면서 오디오 내용이 언제(그리고 과연) 해독
  가능해지는지 관찰할 수 있습니다.
- **Attention Knockout** — 하나의 정보 경로를 정밀하게 잘라 내고(예: *답변이 더 이상 영상
  프레임을 볼 수 없게* 함) 다시 실행합니다. 그래서 한 모달리티를 빼앗았을 때 무엇이
  무너지는지를 **인과적으로** 볼 수 있습니다.

목표는 어떤 수치를 재현하는 것이 아니라, *모델을 내부에서 읽어 내는* 감각을 기르고
"X를 막으면 출력이 Y처럼 바뀔 것이다"라는 **반증 가능한(falsifiable)** 가설을 세우는
연습을 하는 것입니다.

### 실행 방법 (molab)

1. molab에서 노트북을 열고 헤더의 notebook-specs 버튼으로 **GPU를 연결**하세요
   (`cuda:0`을 사용하며, 3B thinker는 넉넉히 올라갑니다).
2. 셀을 위에서 아래로 실행하세요. 셋업 셀은 의존성을 커널에 pip로 설치하고, 작은 PyAV
   심(shim)으로 `torchvision.io.read_video`를 복원하며(최근 torchvision이 비디오
   디코더를 제거함), 실험 코드를 import할 수 있도록 이 저장소(`src/` + 샘플 영상)를
   클론합니다.
3. 첫 실행은 모델 가중치(~8 GB)를 내려받으므로 가장 느립니다. 이후 셀들은 이미 로드된
   모델을 재사용합니다.

CUDA 머신이라면 `uvx marimo edit CTP49906_avllm_molab.py`로 로컬에서도 실행할 수
있습니다 — `# /// script` 헤더가 호환되는 의존성 버전을 고정해 둡니다.

### 셀 둘러보기 (읽어 보는 단계)

| 셀 | 하는 일 | 눈여겨볼 점 |
| --- | --- | --- |
| **Setup** | 의존성 설치, 비디오 리더 패치, 저장소 클론. | 조정할 것 없음. 끝날 때까지 기다리면 됩니다. |
| **Parameters** | 핵심 노브: `VIDEO_PATH`, `NFRAMES`, `LOGIT_PROMPT`, `ATTENTION_PROMPT`, `KNOCKOUT_RULES`, `MAX_NEW_TOKENS`. | *고정* 실행을 바꾸려면 편집하는 유일한 셀. |
| **Video preview** | Qwen에 보낼 바로 그 영상(프레임 **및** 내장 오디오)을 재생. | 여기서 모델이 인지할 수 없는 것은, 답변의 근거가 될 수도 없습니다. |
| **Model + helpers** | Qwen2.5-Omni-3B 로드(talker 해제 — 여기선 *thinker*만 필요) 및 토큰 타입 맵 구성. | `token counts:`를 출력 — 프롬프트가 만든 `audio` / `video` / `query_text` 토큰 수. |
| **Logit Lens** | 순전파 1회; 오디오 위치의 레이어별 예측을 CSV로 디코딩하고 캡션도 출력. | 캡션은 비교 기준이 되는 모델의 "최종 답". |
| **Diversity by layer** | 그래프 두 개: 각 레이어가 오디오 위치에서 디코딩하는 *서로 다른* 토큰 수, 그리고 최상위 예측이 얼마나 지배적인지. | 다양성이 낮은 레이어는 "확정된" 상태, 높은 레이어는 아직 "고민 중". |
| **Attention Knockout** | `KNOCKOUT_RULES`를 사용해 **기준(baseline)** 캡션과 **녹아웃(knockout)** 캡션을 나란히 생성. | 핵심: 경로를 잘랐을 때 답변이 *바뀌는가*? |
| **Captured attention** | 캡처된 레이어별로, 마지막 쿼리가 각 모달리티에 얼마나 어텐션하는지 히트맵. **서술적(descriptive)이며 인과적이지 않음.** | 텍스트 비교와 *함께* 읽을 것. 대체물이 아님. |
| **🎛️ Playground** | 여러분의 선택으로 logit-lens 다양성 측정을 다시 돌리는 대화형 폼(아래). | 학생들이 가장 오래 머무는 곳. |

### 녹아웃 규칙 읽는 법

모든 녹아웃은 튜플 **`(source, target, start_layer, end_layer)`**이며, 의미는 다음과
같습니다.

> *thinker 레이어 `[start_layer, end_layer)` 구간에서, `source` 타입 토큰이 `target`
> 타입 토큰에 어텐션하는 것을 금지한다.*

학생들이 자주 헷갈리는 두 가지를 먼저 짚고 갑니다.

- **방향성이 있습니다.** 어텐션은 source(보는 쪽 토큰)에서 target(읽히는 쪽 토큰)으로
  흐릅니다. `generated → video`(답변이 프레임을 볼 수 없음)와 `video → generated`는 완전히
  다른 개입입니다. 의미 있는 방향은 거의 항상 *뒤쪽 토큰이 앞쪽 토큰을 읽는* 것인데,
  모델이 인과 마스킹(causal masking)되어 있어 한 토큰은 자기 자신과 그 이전의 모든
  것에만 어텐션할 수 있기 때문입니다.
- **`end_layer`는 배타적(exclusive)입니다.** `[0, 36)`은 레이어 0부터 35까지, 즉 3B
  thinker에서는 전부를 의미합니다(노트북이 실제 레이어 수를 출력합니다). 창을 좁혀
  효과를 *국소화(localize)*하세요: 초기 레이어만 막으면 융합(fusion)이 어디서 일어나는지,
  후기 레이어만 막으면 답이 어디서 구성되는지 검증할 수 있습니다.

모델이 모든 위치에 붙이는 다섯 가지 토큰 타입(모달리티):

| 타입 | 무엇인가 |
| --- | --- |
| `query_text` | 프롬프트 / 지시문의 텍스트. |
| `audio` | 영상의 **사운드트랙**에서 나온 토큰. |
| `video` | 샘플링된 **프레임**에서 나온 토큰. |
| `image` | 정지 이미지 토큰 — 영상 입력에는 없으므로 여기선 작동하지 않음. |
| `generated` | 모델이 답으로 **생성하는** 토큰(생성 중에만 존재). |

### 녹아웃 쌍(pair) 카탈로그 (각각 무엇을 묻는가)

시도해 볼 만한 개입들입니다. 처음 세 개는 `source`가 `generated`이므로 **Attention
Knockout 셀**에 속하고(생성 중에 작동), 마지막 두 개는 **오디오 위치 logit lens**를 바꾸므로
**플레이그라운드**에 속합니다.

| 규칙 | 던지는 질문 | 출력 변화가 의미하는 것 |
| --- | --- | --- |
| `generated → video` | *답변이 프레임을 볼 수 없다면, 그래도 화면에 있는 것을 묘사하는가?* | 출력이 바뀜 → 캡션이 **시각에 근거**했음. 출력이 동일 → 오디오나 사전 지식으로 서술한 것. |
| `generated → audio` | *답변이 사운드트랙을 들을 수 없다면, 그래도 소리를 묘사하는가?* | 출력이 바뀜 → 진짜 **듣기**. 오디오 프롬프트인데도 그대로 → "듣기"는 형식적이었음. |
| `generated → query_text` | *답변이 지시문을 다시 읽을 수 없다면, 과제에서 벗어나는가?* | 크게 벗어남 → 모델은 과제를 유지하려 프롬프트를 계속 다시 참조하고 있었음. |
| `audio → video` | *오디오 토큰이 의미를 형성할 때 프레임에서 빌려 오는가?* (시각 → 오디오 융합) | 오디오 위치의 다양성이 붕괴 → 시각 스트림이 오디오 표현을 실제로 형성하고 있었음. |
| `video → audio` | *비디오 토큰이 사운드트랙에 기대는가?* (오디오 → 시각 융합) | 이후 동작의 변화 → 교차 모달 결합이 반대 방향으로도 작동함. |

> **왜 플레이그라운드에서는 `generated`가 아무 일도 하지 않는가.** 플레이그라운드는
> 순전파(forward pass) 1회만 돌리므로(자기회귀 디코딩 없음), 규칙이 작용할 **`generated`
> 토큰이 없습니다.** 거기서 `generated → …` 규칙은 아무것도 막지 못하고 Δ가 평평합니다 —
> 시도하면 노트북이 경고를 띄웁니다. 오디오 위치 점수를 움직이려면 **source**를 실제로
> 존재하는 모달리티(`audio`, `video`, `query_text`)로 두세요.

### 플레이그라운드 (직접 만지는 단계)

마지막 `🎛️` 섹션은 다양성 측정을 폼으로 감싸며 — **▶를 누르기 전에는 아무것도 실행되지
않습니다** — 이미 로드된 모델을 재사용하므로 반복이 빠르고 추가 VRAM이 필요 없습니다.
컨트롤:

- **Video** — `mp4 / mov / mkv / webm / avi`를 직접 업로드하거나, 비워 두면 샘플 영상을
  재사용합니다. (오디오 트랙이 없는 영상은 오디오 토큰을 만들지 못해 → 점수 매길 것이
  없습니다. 노트북이 이를 알려 줍니다.)
- **Frames** — 2–32; 프레임이 많을수록 시각 맥락이 풍부해지고(그리고 느려짐).
- **Prompt** — 지시문; 소리 vs 시각 쪽으로 유도해 보세요.
- **Knockout 켜기/끄기**, 그다음 **단일 규칙** 드롭다운(source, target, 레이어 범위)을
  쓰거나, 여러 규칙을 `source,target,start,end` 형식으로 `;`로 구분해 넣는 **고급(advanced)**
  필드(예: `audio,video,0,36 ; audio,image,0,36`)를 씁니다. 고급 필드가 채워지면 그쪽이
  우선합니다.
- **Compare** — 녹아웃 없는 기준선도 함께 실행해, 스코어보드가 레이어별
  **Δ (녹아웃 − 기준)**를 보여 주도록 합니다.

스코어보드 읽는 법: 각 레이어는 **오디오 위치에서 디코딩하는 서로 다른 토큰의 수**로
점수가 매겨집니다. **음(−)의 Δ**는 녹아웃이 그 위치들에서 *더 적은* 수의 서로 다른 토큰을
디코딩하게 만들었다는 뜻입니다 — 즉 막힌 경로가 잘리자 오디오 표현이 더 균질해졌고, 그
경로가 정보를 나르고 있었다는 증거입니다. Δ가 평평하면 그 개입이 중요하지 않았거나
(`generated` source의 경우처럼) 애초에 작용할 수 없었던 것입니다.

### 학생을 위한 실험 제안

1. **같은 영상에서 시각 vs 청각.** 녹아웃 셀에서 *"see and hear"* 프롬프트로
   `generated → video`를 돌린 뒤 `generated → audio`를 돌려 보세요. 어느 녹아웃이 캡션을
   더 많이 바꾸나요? 이 영상에서 모델이 어느 감각에 더 기대는지에 대해 무엇을 말해 주나요?
2. **융합은 어디에 사는가.** 플레이그라운드에서 `audio → video`를 `[0, 12)`, 그다음
   `[12, 24)`, 그다음 `[24, 36)` 구간으로 돌려 보세요. 어느 레이어 대역을 잘랐을 때 오디오
   다양성이 가장 크게 붕괴하나요 — 초기, 중기, 후기?
3. **프롬프트 유도.** 영상을 고정한 채 프롬프트를 *"what you hear"*와 *"what you see"*
   사이에서 바꿔 보세요. 어떤 녹아웃도 하기 전에, 오디오 위치가 이미 다르게 디코딩되나요?
4. **직접 영상 가져오기.** 오디오와 시각이 *서로 어긋나는* 영상(예: 무관한 영상 위에 얹은
   내레이션)을 업로드해, 모델이 어느 모달리티를 보고하는지 확인해 보세요.
5. **규칙 쌓기.** 고급 필드로 `audio → video`와 `audio → query_text`를 동시에 녹아웃해
   보세요 — 오디오 스트림에서 *양쪽* 이웃을 모두 굶기면 붕괴가 더 심해지나요?

## 인용

```bibtex
@misc{selvakumar2026audiovisuallargelanguagemodels,
      title={Do Audio-Visual Large Language Models Really See and Hear?},
      author={Ramaneswaran Selvakumar and Kaousheik Jayakumar and S Sakshi and Sreyan Ghosh and Ruohan Gao and Dinesh Manocha},
      year={2026},
      eprint={2604.02605},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.02605},
}
```
