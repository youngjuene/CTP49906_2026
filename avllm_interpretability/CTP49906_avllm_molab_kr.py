# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "torch==2.6.0",
#     "torchvision==0.21.0",
#     "transformers==4.52.4",
#     "accelerate==1.14.0",
#     "qwen-omni-utils==0.0.9",
#     "wigglystuff==0.5.21",
#     "anywidget>=0.9.2",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AVLLM 해석 가능성 — molab 실습 (한국어판)

    **Qwen2.5-Omni-3B**로 영상 한 편에 대해 두 가지 해석 가능성(interpretability)
    실험을 수행합니다.

    **오늘의 목표:** 같은 클립에서 설정 하나를 바꾸고, 달라진 결과와 다른 가능한 설명을
    함께 남깁니다. 숫자가 변했다는 관찰과 모델의 능력에 대한 결론을 구분해 봅시다.

    | 순서 | 할 일 | 남길 것 |
    |---|---|---|
    | 1 · 준비 | GPU 연결 → Run all → 시범 결과 읽기 | 입력·출력·두 지표의 의미 |
    | 2 · 비교 | 🎯에서 한국어 무음/원본 쌍, 이어서 영어 쌍 | 같은 설정의 두 실행 ID와 캡션 |
    | 3 · 탐색 | 🎛️ 프레임 8→4 또는 🎯 레이어 한 대역만 변경 | 바꾼 변수, 관측, 경쟁 설명 |
    | 4 · 제출 | 📓에서 해석 기록 → Markdown와 JSON 내려받기 | 재현 가능한 결과와 다음 질문 |

    앞부분은 **교수자와 읽는 시범**, 🎯·🎛️는 **여러분이 제출 버튼으로 실행하는 실험**입니다.
    처음에는 제공된 클립과 4·8프레임을 사용하세요. 자유 탐색은 기본 쌍을 기록한 뒤 시작합니다.

    1. **Logit Lens** — thinker 레이어들을 가로질러, **오디오 토큰 위치**에서 모델의
       중간 표현을 어휘로 투사한 **보정되지 않은 raw probe**를 읽습니다. 모델의 생각이나 확신을 그대로 보여 주는 것은 아닙니다.
    2. **Attention Knockout(어텐션 녹아웃)** — 선택한 source→target 어텐션 경로를
       막고 생성한 답변을, 막지 않은 **기준선(baseline)** 답변과 비교합니다.

    Qwen2.5-Omni는 **thinker**(보고 듣고 글을 쓰는 부분)와 **talker**(그 글을 음성으로
    바꾸는 부분)로 나뉩니다. 이 실습은 talker를 내려놓고 thinker만 쓰기 때문에, 아래에서
    계속 "thinker 레이어"라고 부릅니다.

    이 노트북은 `CTP49906_avllm_molab.py`의 **한국어판**입니다. 노브 셀의
    `LOGIT_PROMPT`와 `ATTENTION_PROMPT`가 한국어이므로 `query_text` 위치에 한국어
    토큰이 놓이고 캡션도 한국어로 나옵니다. 그 밖의 것 — 클립, 모델, 규칙, 레이어
    창 — 은 원본과 동일합니다. 그래야 두 노트북을 나란히 놓고 비교할 수 있습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## molab에서 실행하기

    1. `+ New notebook → Mirror from GitHub`에 이 노트북 주소를 붙여 넣습니다.
       미리보기가 읽기 전용이면 **Fork notebook**으로 자신의 사본을 만드세요.
    2. 위쪽 **Configure compute**에서 GPU를 선택하고 **Save and restart**를 누릅니다.
       CPU만 보이면 실행 전에 GPU 배정을 확인하세요.
    3. **Run all**을 누르고 설치·모델 로드·시범 실행이 끝날 때까지 기다립니다.
       첫 실행과 캐시 사용 실행의 시간은 다릅니다. 기다리는 동안 제출을 반복하지 마세요.
    4. 발표/App view에서는 코드 없이 읽을 수 있습니다. 코드 편집은 고급 탐색 단계에서 합니다.

    **진행이 멈췄다면:** 실행 중인 셀과 오류 문장을 먼저 확인하세요. GPU가 없으면 제공된
    replay로 시범 결과를 토론할 수 있지만 새 실험은 실행되지 않습니다. 메모리 오류 뒤에는
    먼저 결과를 내보내고 커널을 재시작한 다음 기본 클립·4프레임으로 돌아오세요.

    **저장은 직접 확인하세요.** 실행 기록은 세션 안의 파일에도 쓰지만 저장 실패와 세션 종료에
    대비해 Markdown와 JSON을 내려받습니다. 자신의 영상은 원격 GPU 서버로 업로드됩니다.
    수업에서는 제공된 샘플이나 공유에 동의받은 짧은 클립을 사용하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧭 시작하기 전에 — 그림 넷

    이 노트북의 두 실험은 전부 **토큰**과 **어텐션 화살표** 위에서 벌어집니다. 아래 넷만
    알면 나머지는 따라옵니다.

    ### 1. 영상도 소리도 질문도, 전부 한 줄의 토큰이 됩니다

    <svg viewBox="0 0 720 196" style="width:100%;max-width:940px;height:auto" >
      <title>영상·소리·질문이 토큰 한 줄로 바뀌는 그림</title>
      <g font-size="13" fill="currentColor" text-anchor="middle">
        <rect x="10" y="6" width="180" height="34" rx="6" fill="#54A24B" fill-opacity="0.18" stroke="#54A24B"/><text x="100" y="28">질문 (글자)</text>
        <rect x="205" y="6" width="180" height="34" rx="6" fill="#F58518" fill-opacity="0.18" stroke="#F58518"/><text x="295" y="28">소리 (사운드트랙)</text>
        <rect x="400" y="6" width="300" height="34" rx="6" fill="#4C78A8" fill-opacity="0.18" stroke="#4C78A8"/><text x="550" y="28">영상 (프레임)</text>
      </g>
      <path d="M355 46 L355 74" stroke="currentColor" stroke-width="2" marker-end="url(#ar1k)"/>
      <defs><marker id="ar1k" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="currentColor"/></marker></defs>
      <g stroke-width="1">
        <rect x="10"  y="86" width="14"  height="44" fill="#54A24B" fill-opacity="0.55" stroke="#54A24B"/>
        <rect x="24"  y="86" width="279" height="44" fill="#4C78A8" fill-opacity="0.55" stroke="#4C78A8"/>
        <rect x="303" y="86" width="23"  height="44" fill="#F58518" fill-opacity="0.55" stroke="#F58518"/>
        <rect x="326" y="86" width="279" height="44" fill="#4C78A8" fill-opacity="0.55" stroke="#4C78A8"/>
        <rect x="605" y="86" width="92"  height="44" fill="#F58518" fill-opacity="0.55" stroke="#F58518"/>
        <rect x="697" y="86" width="3"   height="44" fill="#54A24B" stroke="#54A24B"/>
      </g>
      <path d="M660 156 L698 134" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar1k)"/>
      <text x="655" y="162" font-size="12" fill="currentColor" text-anchor="end">질문은 여기</text>
      <g font-size="12" fill="currentColor">
        <rect x="10"  y="176" width="12" height="12" fill="#4C78A8" fill-opacity="0.55" stroke="#4C78A8"/><text x="28"  y="186">video ~80%</text>
        <rect x="150" y="176" width="12" height="12" fill="#F58518" fill-opacity="0.55" stroke="#F58518"/><text x="168" y="186">audio ~17%</text>
        <rect x="290" y="176" width="12" height="12" fill="#54A24B" fill-opacity="0.55" stroke="#54A24B"/><text x="308" y="186">query_text ~2%</text>
      </g>
    </svg>

    모델은 영상·오디오를 인코더로 변환한 표현을 **토큰 위치**의 배열로 처리합니다. 토큰마다
    **타입**(`video` · `audio` · `query_text`)이 있고, 이 실습의 모든 개입은 타입 단위로
    이뤄집니다. 영상과 소리는 **한 덩어리씩 번갈아** 놓입니다 — 영상 전부 다음에 소리 전부가
    오는 것이 아닙니다. 정지 이미지가 없으므로 `image`는 0개입니다. (비율은 기본 설정 기준.
    실제 개수는 아래 셀이 출력합니다.)

    ### 2. 어텐션은 화살표입니다. 녹아웃은 그 화살표를 자릅니다

    <svg viewBox="0 0 720 250" style="width:100%;max-width:940px;height:auto" >
      <title>어텐션 화살표 하나를 자르는 그림</title>
      <defs><marker id="a22k" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="currentColor"/></marker></defs>
      <g font-size="13" fill="currentColor" text-anchor="middle">
        <rect x="10"  y="112" width="120" height="40" rx="6" fill="#4C78A8" fill-opacity="0.18" stroke="#4C78A8"/><text x="70"  y="137">video</text>
        <rect x="145" y="112" width="120" height="40" rx="6" fill="#F58518" fill-opacity="0.18" stroke="#F58518"/><text x="205" y="137">audio</text>
        <rect x="280" y="112" width="120" height="40" rx="6" fill="#54A24B" fill-opacity="0.18" stroke="#54A24B"/><text x="340" y="137">query_text</text>
        <rect x="470" y="112" width="140" height="40" rx="6" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-dasharray="4 3"/><text x="540" y="137">generated</text>
      </g>
      <g fill="none" stroke="currentColor" stroke-width="1.8" marker-end="url(#a22k)">
        <path d="M480 110 Q345 46 209 108"/>
        <path d="M488 110 Q418 62 344 108"/>
      </g>
      <path d="M478 110 Q275 34 74 108" fill="none" stroke="currentColor" stroke-width="1.8" stroke-opacity="0.35" stroke-dasharray="7 6"/>
      <g stroke="#E45756" stroke-width="4" stroke-linecap="round">
        <path d="M261 55 L289 77"/><path d="M289 55 L261 77"/>
      </g>
      <text x="275" y="34" font-size="13" fill="#E45756" text-anchor="middle" font-weight="600">녹아웃</text>
      <text x="540" y="176" font-size="12" fill="currentColor" text-anchor="middle">source = 보는 쪽</text>
      <path d="M12 162 L12 168 L398 168 L398 162" fill="none" stroke="currentColor" stroke-opacity="0.45"/>
      <text x="205" y="184" font-size="12" fill="currentColor" text-anchor="middle">target = 보이는 쪽</text>
      <text x="10"  y="205" font-size="12" fill="currentColor">레이어 [0, 12) — 0번부터 11번까지. 12번은 포함하지 않습니다.</text>
      <g><rect x="10.0" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="29.2" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="48.4" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="67.6" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="86.8" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="106.0" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="125.2" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="144.4" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="163.6" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="182.8" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="202.0" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="221.2" y="212" width="17" height="20" fill="#E45756" fill-opacity="0.55" stroke="#E45756"/><rect x="240.4" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="259.6" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="278.8" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="298.0" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="317.2" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="336.4" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="355.6" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="374.8" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="394.0" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="413.2" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="432.4" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="451.6" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="470.8" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="490.0" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="509.2" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="528.4" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="547.6" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="566.8" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="586.0" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="605.2" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="624.4" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="643.6" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="662.8" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/><rect x="682.0" y="212" width="17" height="20" fill="none" stroke="currentColor" stroke-opacity="0.45"/></g>
      <text x="10"  y="245" font-size="11" fill="currentColor">0</text>
      <text x="223" y="245" font-size="11" fill="currentColor">11</text>
      <text x="688" y="245" font-size="11" fill="currentColor">35</text>
    </svg>

    토큰은 답을 만들 때 **자기보다 앞에 있는** 토큰들을 봅니다. 그 "봄"이 어텐션입니다.
    규칙 `(source, target, start, end)`는 *어느 레이어 구간에서 어떤 화살표를 자를지*를
    적은 것입니다. `generated → video` = 생성 중인 토큰이 영상 토큰을 못 보게 막기.

    ### 3. "한 번 읽기"와 "한 토큰씩 쓰기"는 다릅니다

    <svg viewBox="0 0 720 150" style="width:100%;max-width:940px;height:auto" >
      <title>순전파·생성·티처 포싱을 비교한 그림</title>
      <defs><marker id="a33k" markerWidth="8" markerHeight="8" refX="6" refY="2.5" orient="auto"><path d="M0,0 L6,2.5 L0,5 z" fill="currentColor"/></marker></defs>
      <g font-size="12.5" fill="currentColor">
        <text x="10"  y="16" font-weight="600">① 순전파 1회</text>
        <text x="250" y="16" font-weight="600">② 생성</text>
        <text x="490" y="16" font-weight="600">③ 티처 포싱</text>
      </g>
      <g font-size="12" fill="currentColor" text-anchor="middle">
        <rect x="10" y="28" width="150" height="34" rx="5" fill="#4C78A8" fill-opacity="0.18" stroke="#4C78A8"/><text x="85" y="49">입력 토큰</text>
        <rect x="250" y="28" width="150" height="34" rx="5" fill="#4C78A8" fill-opacity="0.18" stroke="#4C78A8"/><text x="325" y="49">입력 토큰</text>
        <rect x="490" y="28" width="150" height="34" rx="5" fill="#4C78A8" fill-opacity="0.18" stroke="#4C78A8"/><text x="565" y="49">입력 토큰</text>
        <rect x="646" y="28" width="64" height="34" rx="5" fill="#E45756" fill-opacity="0.20" stroke="#E45756"/><text x="678" y="49">answer</text>
      </g>
      <g fill="none" stroke="currentColor" stroke-width="1.6" marker-end="url(#a33k)">
        <path d="M85 66 L85 88"/><path d="M325 66 L325 82"/><path d="M565 66 L565 88"/>
      </g>
      <g font-size="12" fill="currentColor" text-anchor="middle">
        <rect x="250" y="88" width="44" height="28" rx="4" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-dasharray="3 2"/><text x="272" y="107">1</text>
        <rect x="303" y="88" width="44" height="28" rx="4" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-dasharray="3 2"/><text x="325" y="107">2</text>
        <rect x="356" y="88" width="44" height="28" rx="4" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-dasharray="3 2"/><text x="378" y="107">3</text>
      </g>
      <g fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#a33k)">
        <path d="M296 102 L301 102"/><path d="M349 102 L354 102"/>
      </g>
      <g font-size="11.5" fill="currentColor">
        <text x="10"  y="100">생성 토큰이 아직 없음</text><text x="10"  y="116">→ generated 규칙은 자를 것이 없음</text>
        <text x="250" y="136">→ generated 규칙이 작동</text>
        <text x="490" y="100">캡션을 answer로 되돌려 넣음</text><text x="490" y="116">→ answer 규칙이 작동</text>
      </g>
    </svg>

    이 차이 때문에, 같은 규칙이 어떤 섹션에서는 작동하고 어떤 섹션에서는 아무 일도
    하지 않습니다. 없는 토큰은 막을 수 없기 때문입니다.

    ### 4. 결과 숫자 두 개

    <svg viewBox="0 0 720 178" style="width:100%;max-width:940px;height:auto" >
      <title>두 측정값을 읽는 법</title>
      <defs><marker id="a44k" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="currentColor"/></marker></defs>
      <text x="10" y="16" font-size="12.5" fill="currentColor" font-weight="600">다양성 — 레이어별 서로 다른 예측 수</text>
      <g fill="#4C78A8" fill-opacity="0.55" stroke="#4C78A8">
        <rect x="14" y="70" width="20" height="34"/><rect x="40" y="52" width="20" height="52"/>
        <rect x="66" y="38" width="20" height="66"/><rect x="92" y="60" width="20" height="44"/>
        <rect x="118" y="46" width="20" height="58"/><rect x="144" y="78" width="20" height="26"/>
        <rect x="170" y="64" width="20" height="40"/><rect x="196" y="86" width="20" height="18"/>
      </g>
      <path d="M10 104 L300 104" stroke="currentColor" stroke-width="1.2"/>
      <text x="10" y="122" font-size="11.5" fill="currentColor">서술적 통계입니다.</text>
      <text x="10" y="138" font-size="11.5" fill="currentColor">크다고 좋은 것도, 작다고 나쁜 것도 아닙니다.</text>
      <text x="390" y="16" font-size="12.5" fill="currentColor" font-weight="600">Δ log-우도 — 토큰당</text>
      <path d="M400 74 L700 74" stroke="currentColor" stroke-width="1.2"/>
      <path d="M550 62 L550 86" stroke="currentColor" stroke-width="1.2"/>
      <text x="550" y="54" font-size="11.5" fill="currentColor" text-anchor="middle">0</text>
      <circle cx="550" cy="74" r="6" fill="#54A24B" stroke="#54A24B"/>
      <text x="550" y="100" font-size="11.5" fill="currentColor" text-anchor="middle">무음 효과도 직접 측정</text>
      <path d="M534 124 L406 124" stroke="currentColor" stroke-width="1.4" marker-end="url(#a44k)"/>
      <path d="M566 124 L694 124" stroke="currentColor" stroke-width="1.4" marker-end="url(#a44k)"/>
      <text x="400" y="144" font-size="11.5" fill="currentColor">← 덜 믿게 됨</text>
      <text x="700" y="144" font-size="11.5" fill="currentColor" text-anchor="end">더 믿게 됨 →</text>
      <text x="390" y="164" font-size="11.5" fill="currentColor" font-style="italic">여러분의 클립은 0에서 얼마나 멀어지나?</text>
    </svg>

    **다양성**은 "무슨 일이 있었나"를 적은 것이지 "좋아졌나"가 아닙니다. **Δ log-우도**는
    **부호**가 방향을(음수 = 자기 답을 덜 믿게 됨, 양수 = 더 믿게 됨), **크기**가 세기를
    말합니다. 같은 설정의 **원본·무음 쌍**과 캡션을 함께 보세요. 무음의 효과가 꼭 0인 것은
    아닙니다. 확률 변화의 크기는 정답 여부나 듣기 능력의 점수가 아닙니다.
    """)
    return


@app.cell
def _(mo):
    import importlib.metadata
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    def _ver_tuple(v):
        out = []
        for part in v.split(".")[:3]:
            digits = "".join(ch for ch in part if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return tuple(out)

    def _ensure_packages(specs):
        # specs: (import_name, dist_name, want, pip_spec), where `want` is a
        # minimum version, `None` for "any version will do", or `("==", version)`
        # when only that exact one will — a *newer* release is wrong too, so a
        # kernel that ships one gets it replaced rather than silently used.
        # molab does not install the `# /// script` block into the running
        # kernel, so pip-install anything missing, too old, or simply different.
        to_install = []
        for import_name, dist_name, want, pip_spec in specs:
            if importlib.util.find_spec(import_name) is None:
                to_install.append(pip_spec)
                continue
            if want is not None:
                try:
                    have = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    to_install.append(pip_spec)
                    continue
                _exact = isinstance(want, tuple)
                _need = want[1] if _exact else want
                _wrong = (
                    _ver_tuple(have) != _ver_tuple(_need) if _exact
                    else _ver_tuple(have) < _ver_tuple(_need)
                )
                if _wrong:
                    print(
                        f"{dist_name}: 커널에는 {have}이(가) 있고 이 노트북에는 "
                        f"{_need}{'만' if _exact else ' 이상'} 필요합니다 — 설치합니다"
                    )
                    to_install.append(pip_spec)
        if to_install:
            with mo.status.spinner(title=f"설치 중: {', '.join(to_install)}…"):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *to_install], check=True
                )

    _ensure_packages([
        # Exact, not a minimum. `src/attention_knockout_experiment.py` rewrites
        # the `attention_mask` kwarg of `layer.self_attn`, and the logit lens
        # reads `thinker.model.layers`, `thinker.lm_head` and
        # `config.thinker_config.audio_token_index` — internals, all of them. A
        # kernel that already ships a *newer* transformers passes a `>=` check
        # and would run this lab against a version nobody validated it on, which
        # for a knockout is worse than a crash: a mask that stops matching is a
        # null result, not an error. 4.52.4 rather than the 4.52.0 this notebook
        # used to pin: 4.52.0 is yanked on PyPI (an exact pin still installs it,
        # but nothing else resolves to it), the two ship byte-identical
        # `models/qwen2_5_omni` and `generation/utils.py`, and 4.52.4 adds three
        # `from_pretrained` fixes.
        ("transformers", "transformers", ("==", "4.52.4"), "transformers==4.52.4"),
        ("accelerate", "accelerate", ("==", "1.14.0"), "accelerate==1.14.0"),
        ("qwen_omni_utils", "qwen-omni-utils", ("==", "0.0.9"), "qwen-omni-utils==0.0.9"),
        ("av", "av", None, "av"),  # PyAV — backs the video-decode shim below
        # qwen-omni-utils 0.0.9 imports these at module scope but declares
        # neither: `audioread` (used for real, via audioread.ffdec) is missing
        # from its metadata entirely, and `librosa` is declared but absent on any
        # kernel where its dependency set was not resolved. Both are listed here
        # so the common case is fixed deterministically rather than by the
        # import-repair loop below.
        ("audioread", "audioread", None, "audioread"),
        ("librosa", "librosa", None, "librosa"),
        # anywidget-based classroom widgets (caption diff, Δ threshold).
        # 0.5.15+ needs Python >= 3.11 — molab qualifies.
        ("wigglystuff", "wigglystuff", ("==", "0.5.21"), "wigglystuff==0.5.21"),
        # Listed explicitly even though wigglystuff pulls it in: `src/probe_grid.py`
        # is a first-party anywidget and should not depend on another package's
        # dependency graph to be importable.
        ("anywidget", "anywidget", "0.9.2", "anywidget>=0.9.2"),
    ])

    def _ensure_video_reader():
        import torchvision
        from src.classroom_safety import streaming_read_video

        # Apply the same cumulative decode budget on native and shim runtimes.
        torchvision.io.read_video = streaming_read_video
        from qwen_omni_utils.v2_5 import vision_process as _vision
        _vision.FORCE_QWENVL_VIDEO_READER = "torchvision"
        _vision.get_video_reader_backend.cache_clear()
        print("영상 디코딩: 크기 제한이 있는 PyAV 리더 사용")

    def _ensure_audio_decoder():
        # qwen-omni-utils opens the clip's audio itself and hands `librosa.load`
        # an already-constructed `audioread.ffdec.FFmpegAudioFile`. Whether librosa
        # accepts that object depends on which librosa the kernel resolved:
        #
        #   <= 0.11.0  reads audioread objects, but dispatches to its audioread
        #              reader only for classes in `audioread.available_backends()`
        #              — a list built once and cached, holding the ffmpeg backend
        #              only if `ffdec.available()` succeeded at that moment.
        #   >= 1.0.0   dropped audioread altogether (it declares only soundfile),
        #              so the object goes straight to `sf.SoundFile(...)`, which
        #              raises `TypeError: Invalid file:
        #              <audioread.ffdec.FFmpegAudioFile object ...>`.
        #
        # molab is the second case: it runs Python 3.13, librosa 1.0.0 requires
        # >= 3.12, and nothing here pins librosa — while a local 3.11 venv still
        # resolves 0.11.0 and can hit the first. Both are the same missing
        # capability, so restore it once rather than repair two registries: wrap
        # `librosa.load` and decode audioread objects with the reader librosa 1.0
        # deleted. Same shape as the read_video shim above — qwen-omni-utils
        # 0.0.9 calls an API its dependency has since removed — and it beats
        # pinning librosa < 1.0, which would downgrade a package molab already
        # ships and re-resolve numba/soxr on a 3.13 kernel.
        import librosa
        import numpy as np

        if getattr(librosa.load, "__audioread_budget_version__", None) == 1:
            return  # already wrapped; this cell re-ran
        _librosa_load = librosa.load

        def _audioread_load(reader, offset, duration, dtype):
            # Ported from librosa 0.11.0's `__audioread_load`, minus its lookup
            # of the backend registry: audioread readers yield blocks of
            # interleaved little-endian 16-bit PCM, which scale to [-1, 1).
            from src.classroom_safety import MAX_AUDIO_DECODED_BYTES, validate_audio_metadata
            buf = []
            _decoded_bytes = 0
            with reader as input_file:  # closes it, and so the ffmpeg child
                sr_native = input_file.samplerate
                n_channels = input_file.channels
                validate_audio_metadata(sr_native, n_channels, 120.0)
                s_start = int(sr_native * offset) * n_channels
                s_end = (
                    np.inf if duration is None
                    else s_start + int(sr_native * duration) * n_channels
                )
                n = 0
                for frame in input_file:
                    _decoded_bytes += (len(frame) // 2) * np.dtype(dtype).itemsize
                    if _decoded_bytes > MAX_AUDIO_DECODED_BYTES:
                        raise ValueError("오디오 디코딩이 128 MiB 한도를 넘었습니다. 더 짧은 클립을 사용하세요.")
                    frame = np.frombuffer(frame, "<i2").astype(dtype) / 2**15
                    n_prev, n = n, n + len(frame)
                    if n < s_start:
                        continue  # offset is past this block
                    if s_end < n_prev:
                        break  # past the requested duration
                    if s_end < n:
                        frame = frame[: int(s_end - n_prev)]  # end is in this block
                    if n_prev <= s_start <= n:
                        frame = frame[s_start - n_prev :]  # start is in this block
                    buf.append(frame)
            if buf:
                y = np.concatenate(buf)
                if n_channels > 1:
                    y = y.reshape((-1, n_channels)).T  # de-interleave
            else:
                y = np.empty(0, dtype=dtype)
            return y, sr_native

        def _load(path, *, sr=22050, mono=True, offset=0.0, duration=None,
                  dtype=np.float32, res_type="soxr_hq", **kwargs):
            # Anything librosa can open itself — paths, file objects, SoundFile —
            # is still librosa's job; only audioread readers come here.
            if not (hasattr(path, "read_data") and hasattr(path, "samplerate")):
                return _librosa_load(path, sr=sr, mono=mono, offset=offset,
                                     duration=duration, dtype=dtype,
                                     res_type=res_type, **kwargs)
            y, sr_native = _audioread_load(path, offset, duration, dtype)
            # The tail of librosa's own `load`, so mono/sr behave identically.
            if mono:
                y = librosa.to_mono(y)
            if sr is not None:
                y = librosa.resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)
            else:
                sr = sr_native
            return y, sr

        _load.__audioread_shim__ = True
        _load.__audioread_budget_version__ = 1
        for _mod in (librosa, librosa.core, librosa.core.audio):
            if getattr(_mod, "load", None) is _librosa_load:
                _mod.load = _load
        print("qwen-omni-utils 호환을 위해 librosa.load(audioread 리더)를 패치했습니다")

    _ensure_audio_decoder()

    def _ensure_importable(module, attempts=4):
        # `find_spec` above only proves a package is on disk, not that it
        # imports. qwen-omni-utils 0.0.9 imports `audioread`, `numpy`, `torch`,
        # `torchvision` and `torchcodec` at module scope while declaring none of
        # them, so a kernel whose dependency set was resolved differently has the
        # package present and unimportable — and the failure surfaces as a
        # ModuleNotFoundError deep inside a later cell, long after setup claimed
        # success.
        #
        # So import it here and install whatever it actually asks for. Bounded,
        # and every install is printed: this repairs a broken environment, it
        # does not paper over one.
        for _ in range(attempts):
            try:
                importlib.import_module(module)
                return
            except ModuleNotFoundError as _exc:
                _missing = (_exc.name or "").split(".")[0]
                if not _missing or _missing == module:
                    raise
                if _missing.startswith("torch"):
                    # Never reinstall torch/torchvision: molab ships a build
                    # matched to its GPU, and replacing it yields an unrunnable
                    # one. Fail with the reason instead.
                    raise ModuleNotFoundError(
                        f"{module}에 필요한 {_missing!r}이(가) 없습니다. pip으로 설치하지 "
                        "않습니다: molab의 torch 빌드는 GPU에 맞춰져 있어 재설치하면 그것을 "
                        "덮어쓰게 됩니다. 해당 패키지를 제공하는 GPU 런타임을 연결하세요."
                    ) from _exc
                print(f"{module}에 {_missing!r}이(가) 필요합니다 (미선언) — 설치합니다")
                with mo.status.spinner(title=f"{module}에 필요한 {_missing} 설치 중…"):
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", _missing], check=True
                    )
                importlib.invalidate_caches()
        importlib.import_module(module)  # last try; let it raise if still broken

    _ensure_importable("qwen_omni_utils")

    def _print_versions():
        # What this kernel actually resolved, on one line. molab supplies its own
        # torch/torchvision and ignores the `# /// script` block, and three of the
        # packages above are deliberately unpinned, so "what is installed" is a
        # per-session fact — and every molab failure so far has been one of these
        # numbers, discovered from a traceback several cells later.
        _report = [f"python={sys.version.split()[0]}"]
        # matplotlib and numpy are not in the list above — molab has always
        # supplied them — so the banner is where their absence would show up.
        for _dist in ("torch", "torchvision", "transformers", "accelerate", "numpy",
                      "matplotlib", "librosa", "audioread", "av", "qwen-omni-utils",
                      "marimo", "wigglystuff", "anywidget"):
            try:
                _report.append(f"{_dist}={importlib.metadata.version(_dist)}")
            except importlib.metadata.PackageNotFoundError:
                _report.append(f"{_dist}=MISSING")
        print("versions: " + ", ".join(_report))

    _print_versions()

    # A release identity is explicit; never destroy a student's edited clone.
    import os as _os
    REPO_REF = _os.environ.get("CTP49906_REPO_REF", "1df22a98697db72e4c2a5725156951fcbf9348b7")
    REPO_DIR = Path("CTP49906_2026").resolve()
    if REPO_DIR.exists():
        _dirty = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        if _dirty:
            raise RuntimeError(
                "실험 코드에 저장되지 않은 수정이 있습니다. 덮어쓰지 않았습니다. "
                "수정 파일을 보관한 뒤 별도 노트북 사본에서 다시 시작하세요."
            )
    else:
        subprocess.run(["git", "clone", "--no-checkout", "--depth", "1",
                        "https://github.com/youngjuene/CTP49906_2026.git", str(REPO_DIR)], check=True)
    with mo.status.spinner(title="수업용 코드 버전 준비 중…"):
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1",
                        "origin", REPO_REF], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", "FETCH_HEAD"], check=True)
    PROJECT_DIR = REPO_DIR / "avllm_interpretability"
    assert PROJECT_DIR.is_dir(), f"코드 디렉터리를 찾을 수 없습니다: {PROJECT_DIR}"
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    _ensure_video_reader()
    print("프로젝트 디렉터리:", PROJECT_DIR)
    return PROJECT_DIR, Path


@app.cell
def _(PROJECT_DIR):
    _ = PROJECT_DIR
    from src.classroom_safety import preflight_clip, validate_encoded_inputs, validate_experiment
    from src.classroom_display import caption_cache_key, selected_drop_share

    return (
        caption_cache_key,
        preflight_clip,
        selected_drop_share,
        validate_encoded_inputs,
        validate_experiment,
    )


@app.cell
def _(MODEL_PATH, MODEL_REVISION, PROJECT_DIR):
    import hashlib as _hash
    import importlib.metadata as _versions
    import platform as _platform
    import subprocess as _sp

    _repo_revision = _sp.check_output(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], text=True).strip()
    try:
        _notebook_sha256 = _hash.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest()
    except (OSError, NameError):
        _notebook_sha256 = "unavailable"
    run_provenance = {
        "classroom_version": "2026-09-14", "model_id": MODEL_PATH,
        "model_revision": MODEL_REVISION, "repo_revision": _repo_revision, "notebook_sha256": _notebook_sha256,
        "python": _platform.python_version(),
        "packages": {name: _versions.version(name) for name in
                     ("torch", "torchvision", "transformers", "qwen-omni-utils", "marimo",
                      "numpy", "matplotlib", "av", "wigglystuff", "anywidget", "accelerate", "librosa", "audioread")},
    }

    def experiment_config(clip_path, **settings):
        config = {
            "clip": clip_path.name,
            "clip_sha256": _hash.sha256(clip_path.read_bytes()).hexdigest(),
            "model_id": MODEL_PATH, "model_revision": MODEL_REVISION,
            "repo_revision": _repo_revision, "notebook_sha256": _notebook_sha256,
            "runtime_packages": dict(run_provenance["packages"]),
            "python": run_provenance["python"], **settings,
        }
        # Only this supplied pair has established visual correspondence.
        if clip_path.resolve() in ((PROJECT_DIR / "assets" / "02321.mp4").resolve(), (PROJECT_DIR / "assets" / "02321_silent.mp4").resolve()):
            config["comparison_key"] = "builtin-02321-av-pair-v1"
        return config

    return experiment_config, run_provenance


@app.cell
def _(PROJECT_DIR):
    # F5a — GPU-free replay. Flip to True to render every non-interactive W7-W9
    # plot from committed artifacts (no GPU, no 8 GB download): a break-glass mode
    # for when molab's GPU is unavailable. Default False = live model. The
    # interactive playground / teacher-forcing sections still need a GPU and fail
    # loudly if submitted in this mode. Generate the artifacts on a GPU with:
    #   python avllm_interpretability/scripts/generate_precompute.py
    #
    # This notebook replays its OWN pack: `precomputed/` was generated with the
    # English prompts, and `validate_precompute_meta` refuses to relabel it as a
    # Korean run. `precomputed_kr/` is the same artifacts regenerated against the
    # Korean prompts below — regenerate it with the command in its README if you
    # edit LOGIT_PROMPT or ATTENTION_PROMPT.
    USE_PRECOMPUTED = False
    PRECOMPUTED_DIR = PROJECT_DIR / "precomputed_kr"
    if USE_PRECOMPUTED:
        print(f"USE_PRECOMPUTED=True — {PRECOMPUTED_DIR}에서 가이드 예제를 재생합니다 (GPU 불필요)")
    return PRECOMPUTED_DIR, USE_PRECOMPUTED


@app.cell
def _(USE_PRECOMPUTED):
    import torch

    if USE_PRECOMPUTED:
        DEVICE = torch.device("cpu")
        print(f"torch={torch.__version__}, USE_PRECOMPUTED=True → CPU (GPU 불필요)")
    else:
        assert torch.cuda.is_available(), (
            "GPU가 보이지 않습니다. molab에서는 헤더의 notebook-specs 버튼으로 GPU를 연결하세요. "
            "(또는 위 셀에서 USE_PRECOMPUTED=True로 두면 커밋된 산물로 가이드 예제를 재생합니다.)"
        )
        DEVICE = torch.device("cuda:0")
        _free, _total = torch.cuda.mem_get_info(0)
        print(f"torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}, VRAM={_total / 2**30:.0f} GiB")
    return DEVICE, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 파라미터 — 먼저 아래 실험 폼을 사용하세요

    **처음에는 코드 셀을 편집할 필요가 없습니다.** 아래 🎛️·🎯 폼에서 값 하나를 바꾸고
    ▶를 누르세요. 폼 값을 움직이는 것만으로 GPU 실험이 실행되지는 않습니다.

    | 바꿀 것 | 예상되는 변화 | 비교할 때 고정할 것 |
    |---|---|---|
    | 프레임 8→4 | 비디오 토큰 수가 줄어듦. 같은 클립의 오디오 길이는 유지 | 클립·프롬프트·규칙 |
    | 한국어→영어 프롬프트 | 질문 토큰과 생성 캡션이 함께 달라질 수 있음 | 원본/무음 각각 같은 프롬프트로 쌍 만들기 |
    | `audio`→`video` 타깃 | 다른 직접 어텐션 연결을 차단 | 클립·프롬프트·레이어 |
    | 레이어 `[0,12)`→`[12,24)` | 이 개입에 민감한 대역 비교 | `end`는 포함하지 않음; 다른 설정은 유지 |
    | 🎯 캡션 상한 32→64 | 더 긴 캡션을 생성할 수 있음 | 실제 길이·문장을 함께 기록 |

    **고급:** 아래 코드의 전역 노브를 바꾸면 여러 시범 셀과 폼이 다시 실행/초기화될 수 있습니다.
    모델은 재사용하지만 추가 연산과 메모리는 필요합니다. 먼저 결과를 내보내세요.
    프레임은 2–16의 짝수, 시범 어텐션 캡처는 최대 2개 레이어로 제한합니다.
    입력 토큰 수와 디코딩 크기도 검사하므로 짧고 작은 클립부터 시도하세요.
    `ATTENTION_CAPTURE_LAYERS=(0,2)`는 히트맵 저장 범위이며 **녹아웃 레이어 범위와 다릅니다.**
    """)
    return


@app.cell
def _():
    # Own cell on purpose: nothing but a genuine model change should ever
    # invalidate the loader cells below.
    MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
    MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    return MODEL_PATH, MODEL_REVISION


@app.cell
def _(PROJECT_DIR):
    VIDEO_PATH = PROJECT_DIR / "assets" / "02321.mp4"

    RESULTS_DIR = PROJECT_DIR / "notebook_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGIT_CSV_PATH = RESULTS_DIR / "logit_lens_audio_token_analysis_ko.csv"

    # The silent-clip control ships in the repo next to the default clip. It is
    # the only control in this lab that can fail, so its absence is a hard error
    # rather than something the forms below discover at submit time.
    SILENT_VIDEO_PATH = PROJECT_DIR / "assets" / "02321_silent.mp4"

    assert VIDEO_PATH.is_file(), f"영상을 찾을 수 없습니다: {VIDEO_PATH}"
    assert SILENT_VIDEO_PATH.is_file(), f"무음 대조군을 찾을 수 없습니다: {SILENT_VIDEO_PATH}"
    print("영상:", VIDEO_PATH)
    print("무음 대조군:", SILENT_VIDEO_PATH)
    return LOGIT_CSV_PATH, RESULTS_DIR, SILENT_VIDEO_PATH, VIDEO_PATH


@app.cell
def _(validate_experiment):
    # The knobs — editing these reruns guided experiments: re-runs the experiments, not the model loads.
    NFRAMES = 8
    LOGIT_PROMPT = "영상에서 들리는 소리를 설명해 주세요"
    ATTENTION_PROMPT = "영상에서 보이는 것과 들리는 소리를 설명해 주세요"
    KNOCKOUT_RULES = [("generated", "video", 0, 36)]  # block generated→video, all 36 thinker layers
    MAX_NEW_TOKENS = 32
    ATTENTION_CAPTURE_LAYERS = (0, 2)  # heatmap rows; widening this costs VRAM per decode step

    validate_experiment(NFRAMES, LOGIT_PROMPT, MAX_NEW_TOKENS, ATTENTION_CAPTURE_LAYERS)
    validate_experiment(NFRAMES, ATTENTION_PROMPT, MAX_NEW_TOKENS)

    # `end` is exclusive, so `(…, 12, 12)` masks nothing and would come back
    # labelled "no effect". The layer *count* is deliberately not checked here:
    # that would make this cell depend on the loaded model and undo the split
    # that keeps knob edits cheap. `block_attention` does the model-aware check.
    assert all(
        len(_r) == 4 and 0 <= _r[2] < _r[3] for _r in KNOCKOUT_RULES
    ), f"각 규칙은 0 <= start < end인 (source, target, start, end)여야 합니다: {KNOCKOUT_RULES}"
    return (
        ATTENTION_CAPTURE_LAYERS,
        ATTENTION_PROMPT,
        KNOCKOUT_RULES,
        LOGIT_PROMPT,
        MAX_NEW_TOKENS,
        NFRAMES,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 영상 미리 보기 (프레임 + 내장 오디오가 그대로 Qwen에 들어갑니다)
    """)
    return


@app.cell
def _(VIDEO_PATH, mo):
    mo.video(src=VIDEO_PATH.read_bytes(), width=640)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 모델과 입력 헬퍼
    """)
    return


@app.cell
def _(
    DEVICE,
    MODEL_PATH,
    MODEL_REVISION,
    PROJECT_DIR,
    preflight_clip,
    validate_encoded_inputs,
    validate_experiment,
):
    import csv
    from collections import Counter

    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fonts
    _font_path = PROJECT_DIR / "assets" / "fonts" / "NotoSansKR-VF.ttf"
    _fonts.fontManager.addfont(str(_font_path))
    plt.rcParams["font.family"] = [_fonts.FontProperties(fname=str(_font_path)).get_name()]
    plt.rcParams["axes.unicode_minus"] = False
    import numpy as np
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    _ = PROJECT_DIR  # ensure the clone / sys.path cell ran first
    from src.attention_knockout_experiment import block_attention
    from src.attention_knockout_experiment import (
        create_token_type_mapping as create_attention_token_mapping,
    )
    from src.logitlens_experiment import (
        analyze_and_save_audio_logits_to_csv,
        clear_logit_lens_hooks,
        create_token_type_mapping,
        register_logit_lens_hooks,
    )

    def load_model_and_processor(attn_implementation):
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH, revision=MODEL_REVISION, torch_dtype="auto", attn_implementation=attn_implementation
        )
        # Free the talker + (float32) token2wav BEFORE moving to GPU so they never
        # occupy VRAM — this experiment only needs the thinker.
        _model.disable_talker()
        _model = _model.to(DEVICE)
        _model.eval()
        _proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
        return _model, _proc

    # video_path/nframes are arguments, not closures: this cell must depend only
    # on the model constants, or a knob tweak would cascade into the loaders.
    def prepare_video_inputs(model, processor, prompt, token_mapping_fn, video_path, nframes):
        validate_experiment(nframes, prompt)
        _why = preflight_clip(video_path)
        if _why:
            raise ValueError(_why)
        _conv = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": str(video_path), "nframes": nframes},
        ]}]
        _text = processor.apply_chat_template(_conv, add_generation_prompt=True, tokenize=False)
        _audios, _images, _videos = process_mm_info(_conv, use_audio_in_video=True)
        _inputs = processor(
            text=_text, audio=_audios, images=_images, videos=_videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        validate_encoded_inputs(_inputs)
        _inputs = {k: v.to(model.device) for k, v in _inputs.items()}
        _types = token_mapping_fn(_inputs["input_ids"], model.config.thinker_config)
        # Spell out every modality including the zeros. `Counter` omits absent
        # keys, so `image` — which is always 0 for a video clip — simply did not
        # appear, and an absence reads as an oversight rather than as a fact.
        _counts = Counter(_types)
        print("token counts:", ", ".join(
            f"{_m}={_counts.get(_m, 0)}"
            for _m in ("query_text", "audio", "video", "image")
        ))
        return _inputs, _types

    return (
        Counter,
        analyze_and_save_audio_logits_to_csv,
        block_attention,
        clear_logit_lens_hooks,
        create_attention_token_mapping,
        create_token_type_mapping,
        csv,
        load_model_and_processor,
        np,
        plt,
        prepare_video_inputs,
        register_logit_lens_hooks,
    )


@app.cell
def _(attention_model, attention_processor):
    # The logit lens shares the eager model rather than loading a second SDPA
    # copy. *Measured on an RTX 3090:* each copy is 8.88 GiB (this "3B" Omni model
    # is 4.70 B parameters once the talker is freed), so two copies plus eager
    # attention over the 1,476-token multimodal prompt peaked at 22.08 GiB and
    # died with `CUDA out of memory` on a 24 GB card -- the exact floor the README
    # advertises. One shared model peaks at 13.70 GiB and completes.
    #
    # Nothing is lost: the logit lens hooks each layer's *output* and projects it
    # through `lm_head`. It never reads attention weights, so it has no reason to
    # prefer SDPA -- that choice only ever bought speed on one forward pass, at
    # the cost of the notebook not running at all.
    logit_model, logit_processor = attention_model, attention_processor
    return logit_model, logit_processor


@app.cell
def _(PRECOMPUTED_DIR, USE_PRECOMPUTED, load_model_and_processor, mo):
    # Dedicated loader cell for the eager model (knockout hooks + both
    # playgrounds). In precomputed mode a layer-count stub stands in so the
    # playground forms can render; it cannot compute, and the forms fail loudly
    # if submitted.
    if USE_PRECOMPUTED:
        from src.precompute import StubModel as _StubModel
        from src.precompute import load_precompute as _stub_pre

        attention_model = _StubModel(_stub_pre(PRECOMPUTED_DIR)["meta"].get("n_layers", 36))
        attention_processor = None
    else:
        with mo.status.spinner(title="Qwen2.5-Omni-3B 로드 중 (eager attention)…"):
            attention_model, attention_processor = load_model_and_processor("eager")
    return attention_model, attention_processor


@app.cell
def _(attention_model):
    # Submit-to-submit caches for the two playground forms, keyed on
    # (clip name, clip bytes, nframes, prompt): "encode" holds prepared inputs +
    # token types, "caption" holds greedy caption ids for teacher forcing — so a
    # layer-band sweep re-encodes and re-captions nothing after the first ▶.
    # Depending on attention_model flushes them whenever the model is reloaded.
    _ = attention_model
    playground_caches = {"encode": {}, "caption": {}}

    def cache_put(cache, key, value, keep=2):
        cache[key] = value
        while len(cache) > keep:  # bound GPU-resident entries; FIFO eviction
            cache.pop(next(iter(cache)))
        return value

    return cache_put, playground_caches


@app.cell
def _(RESULTS_DIR, mo):
    # The run ledger's state. Its only references are `mo` and `RESULTS_DIR`, both
    # computed once and never touched again — deliberately. marimo mints a fresh
    # `SetFunctor` every time a state cell re-runs, and every cell holding the old
    # setter re-runs with it; since the three GPU cells below call `set_runs`, a
    # dependency here on anything reactive would re-fire all three at once.
    #
    # Seeded from the JSONL rather than from `[]`: the log is the durability
    # promise this section makes to a student whose molab kernel dies mid-lab,
    # and a promise nothing reads back is not a promise.
    from src.run_ledger import load_log as _load_log
    #
    # Not an anywidget: a widget the GPU cells push to would have to be *named* by
    # them, which makes them referring cells, and any synced trait changing from
    # the browser re-runs every referring cell (marimo has no per-trait
    # subscription). One click on a verdict chip would re-run a 60-second
    # generation. An HTML string rendered with `mo.Html` has no such edge —
    # the same shape `render_delta_strip` already uses.
    get_runs, set_runs = mo.state(_load_log(RESULTS_DIR / "lab_log.jsonl"))
    return get_runs, set_runs


@app.cell
def _(RESULTS_DIR):
    # Writers only. Deliberately does NOT reference `get_runs`: the three GPU
    # cells import `run_record`/`append_run` from here, so if this cell also
    # referred to the state it would re-run on every append — and take all three
    # generations with it.
    from src.run_ledger import append_run, apply_verdict, run_record

    LEDGER_LOG = RESULTS_DIR / "lab_log.jsonl"
    return LEDGER_LOG, append_run, run_record


@app.cell
def _(Path, RESULTS_DIR, SILENT_VIDEO_PATH, VIDEO_PATH, preflight_clip):
    import hashlib as _hashlib

    # Upload limits. The PyAV shim above decodes *every* frame into a Python list
    # before stacking, so a two-minute 4K clip materializes tens of gigabytes and
    # takes the kernel — and both loaded models — with it. "Bring your own clip"
    # is the point of the last session, and the natural student clip is a 1080p60
    # phone video, so these are checked before a single frame is decoded.
    # Korean labels over the English values the rest of the notebook (and the
    # ledger, and `resolve_clip` below) already speaks. Defined here, in the cell
    # that owns clip semantics, so both playground forms read one definition.
    CLIP_CHOICES = {
        "기본 클립": "Default clip",
        "무음 대조군": "Silent control",
        "업로드": "Upload",
    }
    CLIP_DEFAULT = "기본 클립"
    CLIP_UPLOAD = "업로드"

    from src.classroom_safety import MAX_UPLOAD_BYTES

    def resolve_clip(choice, uploads):
        """Turn a clip choice into `(path, is_control, error)`.

        The silent control lives in the repo, so in molab it exists only on the
        kernel side — there is nothing on the student's machine for a browser file
        picker to select. Choosing it by name and resolving server-side is what
        makes the one control in this lab that can fail actually reachable.
        """
        if choice == "Default clip":
            return VIDEO_PATH, False, preflight_clip(VIDEO_PATH)
        if choice == "Silent control":
            return SILENT_VIDEO_PATH, True, preflight_clip(SILENT_VIDEO_PATH)
        if not uploads or not uploads[0].contents:
            # Deliberately an error, not a fallback: silently substituting the
            # default clip is how a "silent control" run became a duplicate of
            # the experiment it was supposed to falsify.
            return None, False, "**업로드**를 선택했지만 파일을 고르지 않았습니다."
        _blob = uploads[0].contents
        if len(_blob) > MAX_UPLOAD_BYTES:
            return None, False, (
                f"그 파일은 {len(_blob) / 2**20:.0f} MB입니다. 상한은 "
                f"{MAX_UPLOAD_BYTES // 2**20} MB입니다."
            )
        # Hash into the filename so two different clips that happen to share a
        # name and byte size cannot collide in the encode cache below. The
        # student-supplied name is reduced to its basename and stripped of
        # anything but word characters, dots and dashes: it arrives from a
        # browser and is interpolated straight into a path, so `../` in it would
        # write outside RESULTS_DIR.
        _digest = _hashlib.sha256(_blob).hexdigest()[:12]
        _safe = "".join(
            _c for _c in Path(uploads[0].name).name if _c.isalnum() or _c in "._-"
        )[:80] or "clip.mp4"
        _dest = RESULTS_DIR / f"upload_{_digest}_{_safe}"
        if not _dest.exists():
            _dest.write_bytes(_blob)
        _why = preflight_clip(_dest)
        if _why is not None:
            return None, False, f"`{uploads[0].name}`이(가) 거부됐습니다: {_why}"
        return _dest, False, None

    return CLIP_CHOICES, CLIP_DEFAULT, CLIP_UPLOAD, resolve_clip


@app.cell
def _(get_runs, mo):
    from src.run_ledger import build_worksheet_md as worksheet_md
    from src.run_ledger import render_ledger_html as _render_ledger
    def ledger_view(highlight=()):
        return mo.Html(_render_ledger(get_runs(), highlight_ids=highlight, lang="ko"))

    return ledger_view, worksheet_md


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logit Lens (로짓 렌즈)

    멀티모달 순전파(forward pass) 1회를 돌립니다. CSV 분석은 `audio` 토큰 위치에
    집중합니다.

    아래 캡션이 문장 중간에서 끊겨 보인다면 정상입니다. `MAX_NEW_TOKENS = 32`에서
    생성을 멈추기 때문입니다 — 고장이 아니라 길이 제한입니다.
    """)
    return


@app.cell
def _(
    LOGIT_CSV_PATH,
    LOGIT_PROMPT,
    MAX_NEW_TOKENS,
    NFRAMES,
    PRECOMPUTED_DIR,
    USE_PRECOMPUTED,
    VIDEO_PATH,
    analyze_and_save_audio_logits_to_csv,
    clear_logit_lens_hooks,
    create_token_type_mapping,
    logit_model,
    logit_processor,
    mo,
    prepare_video_inputs,
    register_logit_lens_hooks,
    torch,
):
    if USE_PRECOMPUTED:
        from src.precompute import load_precompute as _load_pre
        from src.precompute import validate_precompute_meta as _validate_pre

        _pre = _load_pre(PRECOMPUTED_DIR)
        # Replay mode keeps LOGIT_PROMPT / NFRAMES / VIDEO_PATH as dependencies of
        # this cell, so editing them re-runs a cell that ignores them. Naming the
        # pinned values — and refusing to relabel a cached run as if it came from
        # the current knobs — is what stops every knob quietly lying for a session.
        _validate_pre(
            _pre["meta"], clip=VIDEO_PATH.name, nframes=NFRAMES, logit_prompt=LOGIT_PROMPT
        )
        logit_csv_written = _pre["logit_csv"]
        _logit_out = mo.vstack([
            mo.callout(
                mo.md(
                    "**캐시에서 재생됨** — 미리 계산된 결과이며 GPU를 쓰지 않았습니다. 이 산물은 "
                    f"클립 `{_pre['meta'].get('clip')}`, "
                    f"`nframes={_pre['meta'].get('nframes')}`, 프롬프트 "
                    f"_{_pre['meta'].get('logit_prompt')}_로 생성됐습니다. 위의 노브를 "
                    "편집해도 이 값들은 바뀌지 않습니다."
                ),
                kind="neutral",
            ),
            mo.md(f"**생성된 캡션:**\n\n> {_pre['logit_caption']}"),
        ])
    else:
        logit_inputs, logit_token_types = prepare_video_inputs(
            logit_model, logit_processor, LOGIT_PROMPT, create_token_type_mapping,
            VIDEO_PATH, NFRAMES,
        )

        register_logit_lens_hooks(logit_model)
        try:
            with mo.status.spinner(title="순전파 + 레이어별 예측 디코딩 중…"):
                with torch.no_grad():
                    _ = logit_model.thinker(**logit_inputs, output_hidden_states=True)
                # Keep the return value. `build_compact_probe_result` already ran
                # inside this call and carries top-5, entropy and the top1-top2
                # margin for every (layer, position); the CSV adapter keeps only
                # `top_tokens[0]["token_text"]`. Throwing the rest away here is
                # what left the probe's degeneracy invisible.
                _probe_result = analyze_and_save_audio_logits_to_csv(
                    logit_model, logit_processor, logit_token_types, filename=str(LOGIT_CSV_PATH)
                )
        finally:
            clear_logit_lens_hooks()
        # `analyze_and_save_audio_logits_to_csv` returns None on every failure path
        # (no capture, no audio tokens, unwritable file) and deletes the previous
        # CSV first, so an unconditional assignment here pointed downstream cells
        # at a file that may not exist.
        logit_csv_written = LOGIT_CSV_PATH if _probe_result is not None else None

        with mo.status.spinner(title="캡션 생성 중…"):
            with torch.no_grad():
                # Generate from the thinker directly: the omni wrapper's generate()
                # defaults to audio output and errors because we freed the talker
                # (transformers >=5 dropped the has-talker fallback). The thinker is a
                # plain causal LM and yields the same text, version-agnostically.
                # do_sample=False pins greedy decoding explicitly: the shipped
                # generation_config is an empty stub that happens to resolve to
                # greedy today; an upstream change must not silently flip it.
                _ids = logit_model.thinker.generate(
                    **logit_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                )
        _logit_caption = logit_processor.batch_decode(
            _ids[:, logit_inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        _logit_out = mo.md(f"**생성된 캡션:**\n\n> {_logit_caption}")
    _logit_out
    return (logit_csv_written,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 레이어별 Logit-lens 다양성

    왼쪽: 각 레이어에서 오디오 토큰 위치들을 가로질러 **서로 다른** 예측이 몇 개
    디코딩되는지. 오른쪽: 그중 최빈 예측이 얼마나 지배적인지.

    둘 다 **무슨 일이 있었는지**를 적은 서술적 통계입니다. 값이 크다고 표현이 좋아진
    것도, 작다고 나빠진 것도 아닙니다.
    """)
    return


@app.cell
def _(Counter, USE_PRECOMPUTED, csv, logit_csv_written, mo, np, plt):
    # Twenty lines below, the scrubber handles the identical case with a callout.
    # A bare open() here meant a clip with no audio track ended the notebook in a
    # raw traceback instead of a sentence naming the cause.
    mo.stop(
        logit_csv_written is None or not logit_csv_written.is_file(),
        mo.callout(
            mo.md(
                "**Logit-lens CSV가 없습니다** — 위 실행이 오디오 토큰 행을 하나도 쓰지 "
                "못했습니다. 가장 흔한 원인은 오디오 트랙이 없는 클립입니다. 프로브는 "
                "`audio` 위치에서만 측정됩니다."
            ),
            kind="warn",
        ),
    )
    with open(logit_csv_written, newline="", encoding="utf-8") as _fh:
        _all = list(csv.reader(_fh))
    _header, _data = _all[0], _all[1:]
    _layer_names = _header[2:]
    _preds = list(zip(*(r[2:] for r in _data)))
    _unique = [len(set(p)) for p in _preds]
    _dominant = [Counter(p).most_common(1)[0][1] / len(p) for p in _preds]

    _x = np.arange(len(_layer_names))
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    _axes[0].bar(_x, _unique, color="#4C78A8")
    _axes[0].set(title="레이어별 Logit-lens 다양성", xlabel="Thinker 레이어", ylabel="서로 다른 예측 수")
    _axes[1].plot(_x, _dominant, marker="o", color="#F58518")
    _axes[1].set(title="최빈 예측 비율", xlabel="Thinker 레이어", ylabel="비율", ylim=(0, 1))
    for _ax in _axes:
        _ax.grid(axis="y", alpha=0.25)
    if USE_PRECOMPUTED:
        _div_out = mo.vstack([
            mo.callout(mo.md("**캐시에서 재생됨** — 미리 계산된 결과이며 GPU를 쓰지 않았습니다."), kind="neutral"),
            _fig,
        ])
    else:
        _div_out = _fig
    _div_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎞️ 인터랙티브: 프로브 표면, 36개 레이어를 한눈에

    위의 다양성 막대그래프는 모델이 *무엇을* 예측하는지를 뭉개 버립니다. 이 그리드는
    모든 레이어 × 모든 오디오 위치를 한 번에 보여 줍니다. **y = thinker 레이어,
    x = 오디오 위치**이고, 색은 프로브가 어떤 종류의 토큰으로 디코딩되는지를
    나타냅니다 — **내용(content)**, **잡토큰(junk)**(구두점·공백·기호), 또는
    **디코딩 불가(undecodable)**. 얇은 테두리는 그 위치의 마지막 레이어 토큰과 이미
    같아진 칸을 표시합니다.

    디코딩된 토큰에 한자나 다른 언어 조각이 섞일 수 있습니다. 보정되지 않은 raw probe의
    어휘 투사 결과이므로 다국어 추론이나 모델의 생각을 증명하지 않습니다. 실제 비율은
    아래 집계로 확인하세요. 문자가 깨진 경우와 읽을 수 있는 다른 문자도 구분합니다.

    **그리드 위를 드래그**하면 활성 레이어가 움직이고, 아래 칩 띠가 파이썬을 거치지
    않고 갱신됩니다. **열을 클릭**하면 그 위치가 고정되어, 36개 레이어를 지나는
    궤적 전체를 토큰으로 읽을 수 있습니다.

    > **여기서 눈여겨볼 것.** "초기 레이어의 잡음이 최종 예측으로 정리되어 간다"고
    > 읽고 싶어지지만, 그 전에 마지막 레이어들이 실제로 무엇으로 디코딩되는지 보세요.
    > 프로브는 오디오 위치에서 **보정되어 있지 않습니다**. 그 퇴화(degeneracy) 자체가
    > 이번 주의 결과이고,
    > 잡토큰/내용 구분이 그것을 눈에 보이게 만듭니다. 잡토큰 판정 규칙은 위젯 안에
    > 그대로 출력되므로 그 규칙에 이의를 제기할 수 있습니다.

    위에서 쓴 CSV를 다시 그리는 것뿐입니다. GPU가 필요 없고 `USE_PRECOMPUTED` 재생
    모드에서도 동작합니다.
    """)
    return


@app.cell(hide_code=True)
def _(logit_csv_written, mo):
    from src.probe_grid import ProbeGrid as _ProbeGrid
    from src.probe_grid import build_probe_grid_pack as _build_pack
    from src.probe_grid import probe_grid_layer_summary as _layer_summary

    _pack = (
        _build_pack(logit_csv_written)
        if logit_csv_written is not None and logit_csv_written.is_file()
        else None
    )
    mo.stop(
        _pack is None,
        mo.callout(
            mo.md("**표시할 것이 없습니다** — 위 Logit-lens 실행이 오디오 토큰 행을 쓰지 못했습니다."),
            kind="warn",
        ),
    )

    # Constructed and displayed in one cell, and nothing anywhere reads its
    # `.value`: every trait is py→js, so no interaction with this widget can make
    # marimo re-run anything. That is what lets it sit upstream of the GPU cells.
    probe_grid = mo.ui.anywidget(_ProbeGrid(**_pack))
    probe_summary = _layer_summary(_pack)
    probe_grid
    return (probe_summary,)


@app.cell(hide_code=True)
def _(mo, probe_summary):
    # The numbers the widget states, restated from the same pure function that
    # feeds it — so the claim in the prose can never drift from the data.
    _worst = max(probe_summary, key=lambda _r: _r["junk"])
    _rows = [
        {
            "레이어": _r["name"].replace("Layer_", ""),
            "서로 다른 토큰": _r["unique"],
            "잡토큰(junk)": _r["junk"],
            "디코딩 불가": _r["undecodable"],
            "마지막 레이어와 일치": _r["matches_final"],
            "최빈 토큰": repr(_r["modal_token"]),
        }
        for _r in probe_summary
    ]
    mo.vstack([
        mo.md(
            f"<span style=\"font-size:1.15rem;font-weight:600\">레이어별 집계 — 잡토큰이 가장 많은 레이어는 **{_worst['name']}**로, "
            f"잡토큰 칸 **{_worst['junk']}**개에 모든 오디오 위치를 통틀어 서로 다른 토큰이 "
            f"**{_worst['unique']}**개뿐입니다</span>"
        ),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=12),
    ], gap=0.4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attention Knockout (어텐션 녹아웃)

    `KNOCKOUT_RULES`는 `(source_type, target_type, start_layer, end_layer)` 튜플입니다.
    기본값은 레이어 0–35에서 생성 토큰이 비디오 토큰에 어텐션하는 것을 막습니다.
    """)
    return


@app.cell
def _(
    ATTENTION_CAPTURE_LAYERS,
    ATTENTION_PROMPT,
    KNOCKOUT_RULES,
    MAX_NEW_TOKENS,
    NFRAMES,
    PRECOMPUTED_DIR,
    USE_PRECOMPUTED,
    VIDEO_PATH,
    attention_model,
    attention_processor,
    block_attention,
    create_attention_token_mapping,
    mo,
    prepare_video_inputs,
    torch,
):
    if USE_PRECOMPUTED:
        from src.precompute import load_precompute as _load_pre
        from src.precompute import validate_precompute_meta as _validate_pre

        _pre = _load_pre(PRECOMPUTED_DIR)
        # This cell takes four knobs it cannot honour in replay mode. All four are
        # pinned in the artifact's meta, so check them rather than let each one
        # quietly lie for the whole session — the heatmap caption below reads
        # ATTENTION_CAPTURE_LAYERS to say which layers are on screen, and would
        # otherwise describe the current knob rather than the cached data.
        _validate_pre(
            _pre["meta"],
            clip=VIDEO_PATH.name, nframes=NFRAMES,
            attention_prompt=ATTENTION_PROMPT,
            knockout_rules=[list(_r) for _r in KNOCKOUT_RULES],
            max_new_tokens=MAX_NEW_TOKENS,
            attention_capture_layers=list(ATTENTION_CAPTURE_LAYERS),
        )
        baseline_text = _pre["baseline_text"]
        knockout_text = _pre["knockout_text"]
        attention_summary = _pre["knockout_attention_summary"]
        baseline_attention_summary = _pre["baseline_attention_summary"]
        attention_token_types = _pre["attention_token_types"]
        attention_inputs = None
        attention_baseline_ids = None
        _ko_rules = _pre["knockout_rules"]
        _ko_banner = mo.callout(mo.md("**캐시에서 재생됨** — 미리 계산된 결과이며 GPU를 쓰지 않았습니다."), kind="neutral")
    else:
        from src.precompute import summarize_attention as _summarize_attention

        attention_inputs, attention_token_types = prepare_video_inputs(
            attention_model, attention_processor, ATTENTION_PROMPT, create_attention_token_mapping,
            VIDEO_PATH, NFRAMES,
        )

        # Capture during the baseline too. The heatmap below used to show *only*
        # the knockout run while calling itself "captured attention", so the
        # blocked modality read as ~0 — which is the intervention, not a finding.
        # This costs nothing extra: the baseline generation already happens, and
        # an empty rule list means no knockout hooks are registered.
        with block_attention(
            attention_model, [], attention_token_types, len(attention_token_types),
            track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
        ) as _base_cap:
            with mo.status.spinner(title="기준선 생성 (+ 어텐션 캡처) 중…"):
                with torch.no_grad():
                    # Thinker-direct generation (see the logit cell): avoids the omni
                    # wrapper's talker requirement. Knockout hooks live on the thinker's
                    # layers, so they still fire below. Greedy (do_sample=False) so the
                    # baseline caption — reused as C by the teacher-forced cell below —
                    # is deterministic.
                    # No `output_attentions=True` here. The capture is done by
                    # per-module hooks (`force_attention_output_hook` +
                    # `attention_hook_fn`), which copy only the selected layers to
                    # CPU. Setting the flag at the `generate` level additionally
                    # makes Transformers retain *every* layer's attention for
                    # *every* decode step in the return object — precisely the
                    # 24 GB-classroom OOM that hook exists to avoid.
                    _base = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        return_dict_in_generate=True,
                    )
            _base_captured = {layer: list(v) for layer, v in _base_cap.items()}
        attention_baseline_ids = _base.sequences
        baseline_text = attention_processor.batch_decode(
            _base.sequences[:, attention_inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        with block_attention(
            attention_model, KNOCKOUT_RULES, attention_token_types, len(attention_token_types),
            track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
        ) as _cap:
            with mo.status.spinner(title="녹아웃 생성 중…"):
                with torch.no_grad():
                    # Same as the baseline: the per-module capture hooks do the
                    # work, so the model-level flag is redundant here and costs
                    # every layer x every step of retained attention.
                    _ko = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        return_dict_in_generate=True,
                    )
            _captured = {layer: list(v) for layer, v in _cap.items()}
        knockout_text = attention_processor.batch_decode(
            _ko.sequences[:, attention_inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # Reduce to the plot-ready matrix now, so the heatmap cell consumes the
        # same shape whether live or replayed (raw tensors are never committed).
        # `decode_only=True` matches `scripts/generate_precompute.py`, which
        # produced the committed matrices: it drops the multi-query prefill
        # snapshot, whose row has zero `generated` key mass. Without it the live
        # numbers would be the same *shape* but a different *quantity* than the
        # replayed ones (~1/MAX_NEW_TOKENS of the mass shifted off `generated`).
        attention_summary = _summarize_attention(
            _captured, attention_token_types, decode_only=True
        )
        baseline_attention_summary = _summarize_attention(
            _base_captured, attention_token_types, decode_only=True
        )
        _ko_rules = KNOCKOUT_RULES
        _ko_banner = None

    from wigglystuff import TextCompare as _TextCompare

    _ko_cmp = mo.vstack([
        mo.md(
            f"**기준선**(왼쪽) vs **녹아웃** `{_ko_rules}`(오른쪽) — 공통 구절은 마우스를 "
            "올리면 강조됩니다. **강조되지 않은 부분이 녹아웃이 캡션을 바꾼 곳**입니다. "
            "여기는 고정된 레이어 대역 하나이며, 아래 🎚️ 섹션에서 대역을 직접 훑어 "
            "볼 수 있습니다."
        ),
        mo.ui.anywidget(_TextCompare(
            text_a=baseline_text, text_b=knockout_text, min_match_words=2
        )),
    ])
    _ko_display = mo.vstack([_ko_banner, _ko_cmp]) if _ko_banner is not None else _ko_cmp
    _ko_display
    return (
        attention_baseline_ids,
        attention_inputs,
        attention_summary,
        attention_token_types,
        baseline_attention_summary,
        knockout_text,
    )


@app.cell(hide_code=True)
def _(ATTENTION_CAPTURE_LAYERS, mo):
    mo.md(f"""
    ## 키 모달리티별 캡처된 어텐션 — 기준선 **과** 녹아웃

    인과적 중요도가 아니라 **서술적** 요약입니다. 캡처한 레이어마다 헤드를 평균 내고,
    마지막 쿼리의 어텐션을 토큰 그룹별로 합산합니다.

    **읽는 법.** 한 행(레이어)의 값을 모두 더하면 1이 됩니다 — 즉 **비율**입니다
    (0.90 = 90%). 그래서 기준선 패널에서 `generated` 열이 가장 큰 것은 **정상**입니다.
    토큰은 대체로 자기 자신과 방금 쓴 이웃을 봅니다. 구조이지 발견이 아닙니다.

    두 패널은 반드시 함께 읽으세요. 색 스케일을 공유하므로, 더 어두워 보이는 칸은
    실제로 더 작은 값입니다. **녹아웃** 패널에서 막힌 열이 0에 가까운 것은 **설계상
    당연한 결과**입니다. 개입이 작동했다는 뜻이지 모델에 대한 발견이 아닙니다. 이
    모델의 어텐션이 평소 어디로 가는지는 **기준선** 패널만이 말해 주고, 마스크가
    질량을 어디로 밀어냈는지는 **Δ** 패널만이 보여 줍니다. 녹아웃 패널만 보고
    "모델이 영상을 무시한다"고 결론짓는 것은, 영상을 무시하라고 지시받은 셀에서
    그 결론을 읽어 내는 일입니다.

    레이어 `{ATTENTION_CAPTURE_LAYERS[0]}`–`{ATTENTION_CAPTURE_LAYERS[1] - 1}`만
    표시됩니다. 파라미터 셀의 `ATTENTION_CAPTURE_LAYERS` 값입니다. 이 창을 넓히는
    것은 정당한 실험이지만 실제 비용이 듭니다 — 위의 노브 표를 보세요.
    """)
    return


@app.cell
def _(attention_summary, baseline_attention_summary, mo, np, plt):
    # Each summary is `(layers, modalities, matrix)` — computed live from captured
    # tensors, or loaded from the committed matrices in USE_PRECOMPUTED mode. Same
    # shape either way, so this plot is unchanged between the two.
    if attention_summary is None:
        _out = mo.md("> 이 빌드는 어텐션 텐서를 반환하지 않았습니다. 위의 텍스트 비교가 결과입니다.")
    else:
        _layers, _mods, _ko_mat = attention_summary
        _ko_mat = np.asarray(_ko_mat, dtype=float)
        # Panels are `(title, matrix, cmap, vmin, vmax)`. The two mass panels
        # SHARE a scale: they exist to be read against each other, and letting
        # each autoscale to its own max is the same defect as a per-run color
        # scale on the Δ strip — two pictures that look alike while describing
        # different numbers. The Δ panel keeps its own symmetric diverging scale,
        # because it is a different quantity with a meaningful zero.
        _mass_hi = max(1e-9, float(_ko_mat.max()))
        _panels = [("녹아웃 실행", _ko_mat, "magma", 0.0, _mass_hi)]
        if baseline_attention_summary is not None:
            _bl_mat = np.asarray(baseline_attention_summary[2], dtype=float)
            if _bl_mat.shape == _ko_mat.shape:
                _delta = _ko_mat - _bl_mat
                _lim = max(1e-9, float(np.abs(_delta).max()))
                _mass_hi = max(1e-9, float(max(_bl_mat.max(), _ko_mat.max())))
                _panels = [
                    ("기준선 (녹아웃 없음)", _bl_mat, "magma", 0.0, _mass_hi),
                    ("녹아웃 실행", _ko_mat, "magma", 0.0, _mass_hi),
                    ("Δ = 녹아웃 − 기준선", _delta, "RdBu_r", -_lim, _lim),
                ]

        _fig, _axes = plt.subplots(
            1, len(_panels),
            figsize=(4.6 * len(_panels), max(3, len(_layers) * 0.6)),
            constrained_layout=True,
        )
        _axes = np.atleast_1d(_axes)
        for _ax, (_title, _mat, _cmap, _vmin, _vmax) in zip(_axes, _panels):
            _diverging = _vmin < 0
            _im = _ax.imshow(_mat, aspect="auto", cmap=_cmap, vmin=_vmin, vmax=_vmax)
            _ax.set(
                title=_title, xlabel="키 모달리티", ylabel="Thinker 레이어",
                xticks=np.arange(len(_mods)), xticklabels=_mods,
                yticks=np.arange(len(_layers)), yticklabels=_layers,
            )
            # Annotate against the cell's own background: magma runs dark->bright
            # with value, so white only reads on the low end; the diverging Δ panel
            # is lightest in the middle, where black always reads. Contrast is
            # judged against the *drawn* scale, which is now shared across the two
            # mass panels.
            for _ri in range(_mat.shape[0]):
                for _ci in range(_mat.shape[1]):
                    _v = float(_mat[_ri, _ci])
                    if _diverging:
                        _color = "#111"
                        _label = f"{_v:+.2f}"
                    else:
                        _color = "white" if _v < 0.6 * _vmax else "#111"
                        _label = f"{_v:.2f}"
                    _ax.text(_ci, _ri, _label, ha="center", va="center",
                             color=_color, fontsize=9)
            _fig.colorbar(_im, ax=_ax, label="어텐션 질량")
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎚️ 선택 탐색: 어느 레이어 대역의 개입에 더 민감한가?

    위의 비교는 **단 하나의** 녹아웃입니다 — 모달리티 하나, 고정된 레이어 대역 하나
    (36개 레이어 전부). 전 구간을 막은 결과만으로 경로의 중요성이나 위치를 확정할 수 없습니다. 이 섹션은 그 대역을 훑습니다. 타깃 모달리티와
    레이어 창을 고르고 다시 생성해서, 캡션이 기준선에서 얼마나 멀어지는지 보세요.

    같은 타깃에 대해 `[0, 12)` · `[12, 24)` · `[24, 36)`을 비교해 보세요.

    **널(null) 결과를 읽는 법.** 캡션이 그대로인 대역은 *이 측정에서는 효과가 없음*을
    보여 줄 뿐입니다. 중복(redundancy), 이 규칙이 자르지 못한 간접 경로, 또는 변화를
    보기에 너무 거친 지표 — 어느 쪽과도 모순되지 않습니다. 경로가 **없다는 증거는
    아닙니다**. 문자열 비교는 이분법적입니다. 아래의 티처 포싱 Δ가 같은 질문의 연속적
    버전이며, *작은* 효과를 보여 줄 수 있는 쪽입니다.

    ▶를 누를 때마다 이미 인코딩된 클립에 대해 greedy 생성 1회가 돌아갑니다. 입력과 런타임에 따라 시간이 달라집니다.
    기준선은 재사용하며 다시 생성하지 않습니다. 두 캡션 모두 **답변만** 표시됩니다 —
    공통 프롬프트를 잘라 내야 차이가 지시문이 아니라 모델의 말에 대한 것이 됩니다.
    서로 다른 설정·결과는 아래 **실습 기록(ledger)**에 남습니다. 동일 재실행은 한 행으로 합쳐집니다. 직전 대역을 기록에서
    비교할 수 있습니다.
    """)
    return


@app.cell
def _(Counter, attention_token_types, mo):
    # The modality census, shown next to the controls rather than folded into a
    # dropdown label. `Counter` omits absent keys, so `image` — offered in every
    # picker — used to be invisible rather than visibly zero, and a rule targeting
    # it masked nothing while reporting "no effect".
    _census = Counter(attention_token_types)
    _rows = [
        {
            "모달리티": _m,
            "이 입력에 있는 토큰 수": _census.get(_m, 0),
            "여기서 타깃으로 쓸 수 있는가": "예" if _census.get(_m, 0) else "아니오 — 하나도 없음",
        }
        for _m in ("video", "audio", "query_text", "image")
    ]
    mo.vstack([
        mo.md("<span style=\"font-size:1.15rem;font-weight:600\">이 인코딩된 입력에 실제로 들어 있는 것</span>"),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(
            "`generated`는 표에 없습니다. 그 위치는 모델이 디코딩하기 전에는 존재하지 "
            "않기 때문입니다. 그래서 `generated` source는 **생성 중에는 살아 있고**"
            "(이 섹션) **순전파에서는 작동하지 않습니다**(아래 🎛️ 스코어보드)."
        ),
    ], gap=0.4)
    return


@app.cell
def _(KNOCKOUT_RULES, attention_model, mo):
    _band_layers = len(attention_model.thinker.model.layers)
    _band_targets = ["video", "audio", "query_text"]
    _band_default = KNOCKOUT_RULES[0][1] if KNOCKOUT_RULES else "video"

    def _band_validate(_v):
        if not _v:
            return None
        # `.get` with a default throughout: a batch's value is a partial dict
        # until the frontend has pushed state for every child, so indexing
        # directly raises KeyError on the first render instead of validating.
        _lo, _hi = _v.get("layers") or (0, 1)
        if int(_hi) <= int(_lo):
            return (
                f"[{int(_lo)}, {int(_hi)})는 0개 레이어를 마스킹합니다 — `end`는 배타적입니다. "
                "이대로면 기준선을 돌려 놓고 '효과 없음'이라고 보고하게 됩니다."
            )
        return None

    band_controls = mo.md(
        "**generated** 토큰이 {target} 에 어텐션하는 것을 thinker 레이어 {layers} "
        "구간에서 금지\n\n"
        "{null_band} — 실행 전에 **효과가 작을 것으로 예상한 대역**입니다. "
        "이 체크는 예상의 기록이며, 대조군 검증이나 무효과 판정이 아닙니다.\n\n"
        f"(`end`는 배타적입니다. 이 thinker는 레이어가 **{_band_layers}**개입니다. "
        "클립·프롬프트·프레임 수는 파라미터 셀에 설정된 값 그대로입니다.)"
    ).batch(
        target=mo.ui.dropdown(
            _band_targets,
            value=_band_default if _band_default in _band_targets else "video",
        ),
        layers=mo.ui.range_slider(
            0, _band_layers, step=1, value=[0, _band_layers // 3], show_value=True, full_width=True
        ),
        null_band=mo.ui.checkbox(value=False),
    ).form(
        submit_button_label="▶ 이 대역으로 다시 생성",
        bordered=True,
        # Refuse the empty band *before* the GPU runs. Dragging both handles onto
        # the same layer is what a student does to ask "is it exactly layer 12?",
        # and the answer used to be a confident, bordered "no effect" tile.
        validate=_band_validate,
    )
    band_controls
    return (band_controls,)


@app.cell
def _(
    ATTENTION_PROMPT,
    LEDGER_LOG,
    MAX_NEW_TOKENS,
    NFRAMES,
    USE_PRECOMPUTED,
    VIDEO_PATH,
    append_run,
    attention_baseline_ids,
    attention_inputs,
    attention_model,
    attention_processor,
    attention_token_types,
    band_controls,
    block_attention,
    experiment_config,
    mo,
    run_record,
    set_runs,
    torch,
):
    import difflib as _difflib

    from wigglystuff import TextCompare as _BandCompare

    _bp = band_controls.value
    mo.stop(
        _bp is None,
        mo.callout(
            mo.md("타깃과 레이어 대역을 고른 뒤 **▶ 이 대역으로 다시 생성**을 누르세요."),
            kind="info",
        ),
    )
    mo.stop(
        USE_PRECOMPUTED or attention_inputs is None,
        mo.callout(
            mo.md(
                "**이 스윕은 라이브 모델이 필요합니다** — 대역마다 캡션을 다시 생성하므로 "
                "`USE_PRECOMPUTED=True`인 동안에는 건너뜁니다."
            ),
            kind="warn",
        ),
    )

    _lo, _hi = int(_bp["layers"][0]), int(_bp["layers"][1])
    _band_rules = [("generated", _bp["target"], _lo, _hi)]
    _plen = attention_inputs["input_ids"].shape[1]
    # Answer-only text: slicing off the shared prompt keeps the diff focused on
    # the generated words (the prompt would otherwise dominate as one big match).
    _base_ans = attention_processor.batch_decode(
        attention_baseline_ids[:, _plen:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    _band_out = None
    try:
        with mo.status.spinner(
            title=f"녹아웃 생성 · generated→{_bp['target']} [{_lo},{_hi})…"
        ):
            with block_attention(
                attention_model, _band_rules, attention_token_types,
                len(attention_token_types), track_attention=False,
            ):
                with torch.no_grad():
                    _band_ids = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    )
        _band_ans = attention_processor.batch_decode(
            _band_ids[:, _plen:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except Exception as _e:  # noqa: BLE001 — surface any run failure in-notebook
        _band_out = mo.callout(
            mo.md(f"**실행 실패** — `{type(_e).__name__}: {_e}`"), kind="danger"
        )

    if _band_out is None:
        _ratio = _difflib.SequenceMatcher(
            None, _base_ans.split(), _band_ans.split()
        ).ratio()
        _unchanged = _band_ans.strip() == _base_ans.strip()
        _band_out = mo.vstack([
            mo.md(
                f"**녹아웃** `generated→{_bp['target']}` **[{_lo}, {_hi})** "
                f"&nbsp;·&nbsp; 전체 {len(attention_model.thinker.model.layers)}개 중 "
                f"{_hi - _lo}개 레이어 차단"
            ),
            mo.hstack([
                mo.stat(
                    value=f"{_ratio:.0%}",
                    label="기준선 대비 캡션 유사도",
                    caption="어절 문자열의 겹침 · 100%로 반올림돼도 내부 확률이 같다는 뜻은 아님",
                    # No `direction=`: an unchanged caption used to get the green
                    # up-arrow and a moved one the red down-arrow, so a three-band
                    # sweep read as two failures and one success rather than as a
                    # localization. Neither outcome is the good one here.
                    bordered=True,
                ),
                mo.stat(
                    value="변화 없음" if _unchanged else "변화 있음",
                    label="이 대역의 효과",
                    caption=(
                        "이 측정에서는 효과 없음 — 중복이나 간접 경로와 모순되지 않으며, "
                        "부재의 증거가 아님"
                        if _unchanged else "여기를 막으니 캡션이 움직였음"
                    ),
                    bordered=True,
                ),
            ], widths="equal", gap=1),
            mo.md(
                "**기준선**(왼쪽) vs **이 대역**(오른쪽) — 공통 구절은 마우스를 올리면 "
                "강조됩니다. **강조되지 않은 부분이 이 대역이 바꾼 것**입니다."
            ),
            mo.ui.anywidget(_BandCompare(
                text_a=_base_ans, text_b=_band_ans, min_match_words=2
            )),
            mo.md(
                "<span style=\"color:#4C78A8;font-weight:600\">다음 →</span> 같은 타깃으로 `[12,24)`와 `[24,36)`도 돌려 보세요. 세 대역의 "
                "캡션과 유사도를 나란히 놓고 이 개입에 더 민감한 대역을 찾아보세요. 위치의 확정은 아닙니다."
            ),
        ])
        # Record it. `set_runs` is a SetFunctor, not the State object, so a cell
        # that only *sets* never re-runs itself — this append cannot re-trigger
        # the generation above. `run_record(...)` is bound as a default argument
        # so it evaluates here, in the GPU cell, leaving `prev` as the only lazy
        # input to the lambda.
        try:
            set_runs(
                lambda _prev, _r=run_record(
                    kind="band_sweep",
                    condition=f"generated→{_bp['target']} [{_lo},{_hi})",
                    metric_name="caption_similarity",
                    metric_value=round(_ratio, 4),
                    metric_unit="ratio",
                    # Every input that can move the number belongs in `config`:
                    # `run_id` digests config *and* metric, so an input left out
                    # produces a second row with a different number, an identical
                    # `condition`, and an empty `changed` column — two runs that
                    # look controlled and are not.
                    config=experiment_config(
                        VIDEO_PATH, nframes=NFRAMES, prompt=ATTENTION_PROMPT,
                        target=_bp["target"], start=_lo, end=_hi, max_new_tokens=MAX_NEW_TOKENS,
                    ),
                    is_control=False,
                    note="효과가 작을 것으로 예상한 대역" if _bp.get("null_band") else "",
                    extra={"baseline_caption": _base_ans, "knockout_caption": _band_ans},
                ): append_run(_prev, _r, log_path=LEDGER_LOG)
            )
        except Exception as _le:  # noqa: BLE001 — a ledger bug must never eat a run
            print("ledger append failed:", type(_le).__name__, _le)
    _band_out
    return


@app.cell
def _(ledger_view):
    ledger_view()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 티처 포싱 Δ log-우도 (고정 파라미터)

    **쉽게 말해:** 모델이 방금 생성한 캡션을 고정하고, 각 다음 토큰에 부여하는 조건부
    확률을 연결 차단 전후로 비교합니다. 정답 여부나 보정된 확신을 측정하지는 않습니다.

    위의 문자열 비교는 **직관적이지만 이분법적**입니다. *작은* 효과는 보이지 않고,
    생성이 어떻게 이어지느냐에 따라 달라집니다. 이 셀은 같은 질문을 **측정**으로
    바꿉니다. 기준선 캡션을 `answer`로 태그해 다시 입력한 뒤, 그 답변이
    `KNOCKOUT_RULES`와 같은 타깃 모달리티로부터 차단됐을 때 **그 캡션의 토큰 확률이
    얼마나 달라지는지**를 점수화합니다(같은 클립, 같은 프롬프트, 같은
    레이어 — source만 `answer`가 됩니다. 이제 캡션은 생성물이 아니라 *입력*이기
    때문입니다).

    캡션 토큰마다 **Δ = 녹아웃 − 기준선**이며, *음수 = 덜 믿음 = 뜨거운 색*입니다.
    아래 🎯 플레이그라운드 섹션에서는 같은 측정을 여러분의 클립·프롬프트·레이어
    대역으로 직접 돌려 볼 수 있습니다.
    """)
    return


@app.cell
def _(
    KNOCKOUT_RULES,
    MAX_NEW_TOKENS,
    USE_PRECOMPUTED,
    attention_baseline_ids,
    attention_inputs,
    attention_model,
    attention_processor,
    attention_token_types,
    mo,
):
    if USE_PRECOMPUTED:
        w9_tf_result = None
        _w9_out = mo.callout(
            mo.md(
                "**티처 포싱은 라이브 모델이 필요합니다** — `USE_PRECOMPUTED=True`인 동안 "
                "이 셀은 건너뜁니다. (이 측정은 아직 캐시 재생을 지원하지 않습니다.)"
            ),
            kind="warn",
        )
    else:
        from src.teacher_forcing import teacher_forced_delta as _w9_tfd

        # Mirror the params-cell intervention with `answer` as the source: the
        # caption is input now, so `answer → target` is the measurable counterpart
        # of the generation-time `generated → target` diff above.
        _w9_rules = [("answer", _t, _a, _b) for (_s, _t, _a, _b) in KNOCKOUT_RULES]
        _w9_prompt_len = attention_inputs["input_ids"].shape[1]
        _w9_c_ids = attention_baseline_ids[:, _w9_prompt_len:]

        w9_tf_result = None
        try:
            with mo.status.spinner(title="티처 포싱 채점 (순전파 2회) 중…"):
                w9_tf_result = _w9_tfd(
                    attention_model,
                    attention_processor,
                    attention_inputs,
                    attention_token_types,
                    _w9_rules,
                    max_new_tokens=MAX_NEW_TOKENS,
                    cached_caption_ids=_w9_c_ids,
                )
        except Exception as _e:  # noqa: BLE001 — surface any failure in-notebook
            _w9_out = mo.callout(
                mo.md(f"**티처 포싱 채점 실패** — `{type(_e).__name__}: {_e}`"),
                kind="danger",
            )

        if w9_tf_result is not None:
            _w9_delta = [float(x) for x in w9_tf_result["delta"].detach().cpu().float().tolist()]
            _w9_total = w9_tf_result["delta_total"]
            _w9_rule_txt = " + ".join(f"`answer→{_r[1]}` [{_r[2]},{_r[3]})" for _r in _w9_rules)
            _w9_out = mo.vstack([
                mo.md(f"**녹아웃** {_w9_rule_txt} &nbsp;·&nbsp; 기준선 캡션을 `answer`로 티처 포싱"),
                mo.callout(mo.md("**채점한 전체 캡션**\n\n" + w9_tf_result["caption_text"]
                                 + ("\n\n⚠ 생성 상한에 도달한 캡션입니다." if w9_tf_result["generation_truncated"] else "")),
                           kind="warn" if w9_tf_result["generation_truncated"] else "neutral"),
                mo.hstack([
                    mo.stat(
                        value=f"{_w9_total:+.2f}",
                        label="Σ Δ log-우도 (nats)",
                        caption="녹아웃 − 기준선 · 음수 = 해당 캡션의 확률 감소",
                        direction="decrease" if _w9_total < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{w9_tf_result['delta_mean']:+.3f}",
                        label="토큰당 Δ (nats)",
                        caption=(
                            "길이로 나눈 값 · 같은 설정의 쌍과 전체 캡션을 함께 비교하세요"
                        ),
                        direction="decrease" if w9_tf_result["delta_mean"] < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(len(_w9_delta)),
                        label="채점한 캡션 토큰 수",
                        caption="greedy 기준선, 티처 포싱",
                        bordered=True,
                    ),
                ], widths="equal", gap=1),
            ])
    _w9_out
    return (w9_tf_result,)


@app.cell
def _(mo, w9_tf_result):
    # Skipped quietly in USE_PRECOMPUTED mode / after a scoring failure.
    mo.stop(w9_tf_result is None)
    from src.teacher_forcing import threshold_slider_params as _w9_params
    from wigglystuff import TangleSlider as _W9Tangle

    # Bounds/step/default derived from this caption's own drops, so the drag
    # spans the range in ~300 px and starts with a meaningful set outlined.
    w9_threshold = mo.ui.anywidget(_W9Tangle(
        suffix=" nats",
        **_w9_params(w9_tf_result["caption_tokens"], w9_tf_result["delta"], token_kinds=w9_tf_result.get("caption_token_kinds")),
    ))
    mo.md(
        "<span style=\"font-size:1.15rem;font-weight:600\">토큰별 Δ log-우도 (표시 단위에 마우스를 올리면 그 토큰들의 nats가 보입니다)</span>\n\n"
        f"{w9_threshold} 이상 잃은 표시 단위만 표시합니다 — **밑줄 친 숫자를 옆으로 드래그**하거나 "
        "클릭해서 입력하세요. 임계값을 넘은 표시 단위는 **굵게 테두리**가 생기고 나머지는 "
        "흐려지므로, 드래그하는 대로 띠가 다시 정렬되는 것이 보입니다. 여기서는 모델을 "
        "전혀 건드리지 않습니다."
    )
    return (w9_threshold,)


@app.cell
def _(mo, selected_drop_share, w9_tf_result, w9_threshold):
    from src.teacher_forcing import group_tokens_into_words as _w9_group
    from src.teacher_forcing import render_delta_strip as _w9_strip

    _delta = [float(_x) for _x in w9_tf_result["delta"].detach().cpu().float().tolist()]
    _th = abs(float(w9_threshold.value.get("amount", 0.0)))
    _words = _w9_group(w9_tf_result["caption_tokens"], _delta, token_kinds=w9_tf_result.get("caption_token_kinds"))
    _hit = [_w for _w in _words if _w[1] < -_th]
    _share = selected_drop_share(w9_tf_result["caption_tokens"], _delta, _th, token_kinds=w9_tf_result.get("caption_token_kinds"))
    mo.vstack([
        mo.Html(
            "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
            + _w9_strip(w9_tf_result["caption_tokens"], _delta, highlight_below=_th, vmax=8.0, token_kinds=w9_tf_result.get("caption_token_kinds"))
            + "</div>"
        ),
        mo.md(
            f"**{len(_hit)}/{len(_words)}** 표시 단위가 −{_th:.2f} nats보다 크게 떨어졌습니다 — 합쳐서 "
            f"Δ = {sum(_w[1] for _w in _hit):+.2f} nats입니다. 감소한 표시 단위의 총 감소량 중 "
            f"선택된 단위가 **{_share:.0f}%**를 차지합니다. 색 범위는 모든 실행에서 ±8 nats입니다."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리
    """)
    return


@app.cell
def _(
    USE_PRECOMPUTED,
    attention_summary,
    baseline_attention_summary,
    knockout_text,
    logit_csv_written,
    mo,
    w9_tf_result,
):
    _csv_ok = bool(logit_csv_written and logit_csv_written.is_file() and logit_csv_written.stat().st_size)
    _attention_ok = bool(attention_summary and baseline_attention_summary)
    _tf_status = "replay에서는 생략" if USE_PRECOMPUTED else ("완료" if w9_tf_result is not None else "실패 또는 미실행")
    mo.md(
        "### 시범 실행 상태\n\n"
        f"- Logit-lens CSV: {'완료' if _csv_ok else '기록되지 않음'}\n"
        f"- 기준선/녹아웃 캡션: {'표시됨' if knockout_text else '확인 필요'}\n"
        f"- 어텐션 비교: {'캡처 결과 있음' if _attention_ok else '캡처 결과 없음'}\n"
        f"- 고정 티처 포싱: {_tf_status}\n\n"
        "**이제 직접 실험할 차례입니다.** 🎯에서 한국어 무음/원본 쌍부터 시작하세요. "
        "🎛️에서는 프레임 수 하나를 바꾸고 오디오·비디오 토큰 수가 어떻게 달라지는지 보세요. "
        "시범 출력은 아래 인터랙티브 기록에 자동으로 포함되지 않습니다. 시범을 근거로 쓸 때는 별도로 저장하세요."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎛️ 인터랙티브: Logit-lens 다양성 스코어보드

    여기까지는 고정 파라미터로 한 번씩 돌린 것입니다. 이 섹션은 **Logit-lens 다양성**
    측정을 살아 있는 플레이그라운드로 바꿉니다. 클립, 프레임 수, 프롬프트, 그리고
    (원한다면) 순전파 **도중에** 적용할 어텐션 녹아웃을 고르고 제출하면, 각 thinker
    레이어가 오디오 토큰 위치들에서 *서로 다른* 토큰을 몇 개나 디코딩하는지로 점수를
    매깁니다.

    제출을 누르기 전에는 아무것도 실행되지 않으며(컨트롤이 폼으로 감싸여 있습니다),
    녹아웃 실험에서 쓰던 모델을 재사용합니다. 모델을 다시 로드하지는 않지만 새 입력과
    순전파에는 추가 시간·메모리가 필요합니다.

    **무엇이 무엇을 움직이는가.** 점수는 **오디오** 토큰 위치에서 측정되고, 그 개수는
    클립의 *재생 시간*으로 고정됩니다. 따라서 `프레임 수`는 **비디오** 토큰 수를 바꿀
    뿐 "채점한 오디오 토큰" 값은 전혀 움직이지 않습니다 — 그 숫자가 변하기를
    기대하며 값을 훑었다면, 고장 난 것이 아닙니다. 아래에 두 개수를 모두 표시하므로
    자신이 무엇을 움직였는지 확인할 수 있습니다.

    이것은 프롬프트에 대한 순전파 1회이므로, **`generated`와 `answer`는 규칙의 어느
    쪽에 놓이든 여기서는 작동하지 않습니다** — 모델이 디코딩하기 전까지 그런 위치는
    존재하지 않습니다(맨 위 🧭 그림 ③). 아무것에도 걸리지 않는 규칙은 실행되지 않고
    거부됩니다. `audio`, `video`, `query_text`를 쓰세요. 드롭다운으로 규칙 하나를 만들거나,
    고급 필드에 여러 개를 입력하면 됩니다.
    """)
    return


@app.cell
def _(
    CLIP_CHOICES,
    CLIP_DEFAULT,
    CLIP_UPLOAD,
    KNOCKOUT_RULES,
    LOGIT_PROMPT,
    NFRAMES,
    attention_model,
    mo,
    validate_experiment,
):
    _n_layers = len(attention_model.thinker.model.layers)
    _modalities = ["audio", "video", "query_text"]
    # Scoreboard-appropriate defaults: the source must be a modality that is
    # actually PRESENT in the prompt, so `audio` (the positions being scored) —
    # not the params cell's `generated`, which is inert in a forward pass. The
    # target follows the params rule; the window spans every layer ([0, N)).
    _def_source = "audio"
    _def_target = KNOCKOUT_RULES[0][1] if KNOCKOUT_RULES else "video"

    _hint = (
        f"source/target ∈ `audio · video · query_text` — 다만 "
        f"`generated`는 규칙의 *어느 쪽*에 놓이든 여기서는 **작동하지 않고**(순전파 "
        f"중에는 생성 위치가 존재하지 않습니다), `image`는 영상 클립에서 0 토큰입니다. "
        f"아무것에도 걸리지 않는 규칙은 실행되지 않고 거부됩니다. 레이어 `end`는 "
        f"배타적입니다. 이 thinker는 레이어가 **{_n_layers}**개이므로 "
        f"`[0, {_n_layers})`가 전체를 뜻합니다."
    )
    _template = (
        "**클립** {clip} &nbsp; (무음 대조군은 저장소 안에 있습니다 — 이름으로 고르면 "
        "되고, 업로드할 것이 없습니다)\n\n"
        "**업로드**를 골랐을 때만 — `mp4 / mov / mkv / webm`, 250 MB·120초·1080p 이하이며, 디코딩 크기 512 MiB도 충족해야 합니다. 짧은 클립 권장:\n\n"
        "{video}\n\n"
        "**클립에서 샘플링할 프레임 수** {nframes} &nbsp; *(**비디오** 토큰 수를 움직입니다. "
        "오디오 위치는 재생 시간으로 고정입니다)*\n\n"
        "**프롬프트** {prompt}\n\n"
        "---\n\n"
        "**순전파 도중 어텐션 녹아웃 적용** {ko_enable}\n\n"
        "단일 규칙 — thinker 레이어 {ko_layers} 구간에서 {ko_source} → {ko_target} 차단\n\n"
        "고급 — `source,target,start,end` 형식의 규칙 여러 개를 `;`로 구분해 입력 "
        "(채우면 위의 단일 규칙보다 우선합니다):\n\n"
        "{ko_rules_text}\n\n"
        + _hint + "\n\n"
        "**비교를 위해 녹아웃 없는 기준선도 함께 실행** {compare}"
    )

    def _ko_validate(_v):
        if not _v:
            return None
        try:
            validate_experiment(_v.get("nframes", NFRAMES), _v.get("prompt", LOGIT_PROMPT),
                                _v.get("max_new_tokens", 32))
        except ValueError as _err:
            return str(_err)
        # The radio carries Korean labels over English values, and `validate` sees
        # the frontend value — which for a radio is the *label* (`_convert_value`
        # indexes `options` with it). Comparing against the English value alone
        # would silently never match and let this check fail open, so accept both.
        if _v.get("clip") in ("Upload", CLIP_UPLOAD) and not _v.get("video"):
            return "업로드를 선택했지만 파일을 고르지 않았습니다."
        # `validate` receives the *frontend* value, not the converted Python one
        # (`form._validate` calls `self.validate(value.value)`), and a dropdown
        # arrives as `list[str]` — `['generated']`, not `'generated'`. Comparing
        # the raw value to a string silently never matches, which would let the
        # inert-rule refusal this section promises fail open: the student pays a
        # full clip encode and then gets a raw ValueError from the backstop.
        def _one(_x):
            return _x[0] if isinstance(_x, list) and _x else _x

        # `.get` with defaults: the batch value is partial until the frontend has
        # pushed state for every child.
        _lo, _hi = _v.get("ko_layers") or (0, 1)
        if _v.get("ko_enable") and not (_v.get("ko_rules_text") or "").strip():
            if int(_hi) <= int(_lo):
                return (
                    f"[{int(_lo)}, {int(_hi)})는 0개 레이어를 마스킹합니다 — `end`는 배타적입니다."
                )
            if _one(_v.get("ko_source")) == "generated" or _one(_v.get("ko_target")) == "generated":
                return (
                    "`generated`는 순전파에서 규칙의 어느 쪽에 놓이든 작동하지 않습니다. "
                    "audio / video / query_text를 쓰세요. 그러지 않으면 이 실행은 그냥 기준선입니다."
                )
        return None

    ko_controls = mo.md(_template).batch(
        clip=mo.ui.radio(CLIP_CHOICES, value=CLIP_DEFAULT, inline=True),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm"],
            multiple=False,
            kind="area",
        ),
        nframes=mo.ui.slider(
            2, 16, step=2, value=NFRAMES, show_value=True, include_input=True
        ),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        ko_enable=mo.ui.checkbox(value=bool(KNOCKOUT_RULES)),
        ko_source=mo.ui.dropdown(_modalities, value=_def_source),
        ko_target=mo.ui.dropdown(_modalities, value=_def_target),
        ko_layers=mo.ui.range_slider(
            0, _n_layers, step=1, value=[0, _n_layers], show_value=True, full_width=True
        ),
        ko_rules_text=mo.ui.text(
            placeholder="예:  audio,video,0,36 ; audio,query_text,0,36", full_width=True
        ),
        compare=mo.ui.checkbox(value=True),
    ).form(
        submit_button_label="▶ Logit-lens 다양성 실행",
        bordered=True,
        validate=_ko_validate,
    )
    ko_controls
    return (ko_controls,)


@app.cell
def _(
    Counter,
    LEDGER_LOG,
    LOGIT_CSV_PATH,
    USE_PRECOMPUTED,
    analyze_and_save_audio_logits_to_csv,
    append_run,
    attention_model,
    attention_processor,
    block_attention,
    cache_put,
    caption_cache_key,
    clear_logit_lens_hooks,
    create_attention_token_mapping,
    csv,
    experiment_config,
    ko_controls,
    mo,
    np,
    playground_caches,
    plt,
    register_logit_lens_hooks,
    resolve_clip,
    run_provenance,
    run_record,
    set_runs,
    torch,
    validate_encoded_inputs,
    validate_experiment,
):
    from contextlib import nullcontext as _nullcontext

    from qwen_omni_utils import process_mm_info as _process_mm_info

    _p = ko_controls.value
    mo.stop(
        _p is None,
        mo.callout(
            mo.md("위에서 파라미터를 설정하고 **▶ Logit-lens 다양성 실행**을 누르세요."),
            kind="info",
        ),
    )
    # The band cell already guards replay mode; these two did not, and submitting
    # either called `apply_chat_template` on a `None` processor — an AttributeError
    # instead of the honest sentence the band cell knows how to print.
    mo.stop(
        USE_PRECOMPUTED,
        mo.callout(
            mo.md(
                "**이 플레이그라운드는 라이브 모델이 필요합니다** — 제출할 때마다 새 순전파를 "
                "돌리므로 `USE_PRECOMPUTED=True`인 동안에는 건너뜁니다."
            ),
            kind="warn",
        ),
    )

    _results_dir = LOGIT_CSV_PATH.parent

    _video_path, _is_control, _clip_err = resolve_clip(_p["clip"], _p["video"])
    mo.stop(
        _clip_err is not None,
        mo.callout(mo.md(f"**클립을 쓸 수 없습니다** — {_clip_err}"), kind="danger"),
    )
    _nframes = int(_p["nframes"])
    _prompt = _p["prompt"].strip()
    validate_experiment(_nframes, _prompt)

    # Build the knockout rules. The advanced text field (several `src,tgt,start,end`
    # rules separated by `;`) overrides the single-rule builder when it is filled.
    _modalities = ["audio", "video", "query_text"]
    _n_layers = len(attention_model.thinker.model.layers)

    def _parse_rules(text):
        _out = []
        for _seg in text.split(";"):
            _seg = _seg.strip()
            if not _seg:
                continue
            _f = [c.strip() for c in _seg.split(",")]
            if len(_f) != 4:
                return [], f"`{_seg}`에는 4개 필드가 필요합니다: `source,target,start,end`"
            _s, _t, _a, _b = _f
            if _s not in _modalities:
                return [], f"알 수 없는 source `{_s}` — {' / '.join(_modalities)} 중에서 쓰세요"
            if _t not in _modalities:
                return [], f"알 수 없는 target `{_t}` — {' / '.join(_modalities)} 중에서 쓰세요"
            try:
                _a, _b = int(_a), int(_b)
            except ValueError:
                return [], f"`{_seg}`의 start/end는 정수여야 합니다"
            if not (0 <= _a < _b <= _n_layers):
                return [], f"`{_seg}`에서는 0 ≤ start < end ≤ {_n_layers}여야 합니다"
            _out.append((_s, _t, _a, _b))
        if not _out:
            return [], "규칙을 하나도 읽지 못했습니다 — `audio,video,0,36` 형식으로 써 보세요"
        return _out, None

    _rules_err = None
    if not _p["ko_enable"]:
        _rules = []
    elif _p["ko_rules_text"].strip():
        _rules, _rules_err = _parse_rules(_p["ko_rules_text"])
    else:
        _lo, _hi = _p["ko_layers"]
        _rules = [(_p["ko_source"], _p["ko_target"], int(_lo), int(_hi))]
    mo.stop(
        _rules_err is not None,
        mo.callout(mo.md(f"**잘못된 녹아웃 규칙** — {_rules_err}"), kind="danger"),
    )
    _compare = bool(_p["compare"])

    def _prep(video_path, nframes, prompt):
        # Encoding (video decode + feature extraction) dominates a submit when
        # only the rule/layer band changed — cache it across ▶ presses.
        _key = caption_cache_key(video_path, nframes, prompt, 0, run_provenance["model_revision"])
        if _key in playground_caches["encode"]:
            return playground_caches["encode"][_key]
        _conv = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": str(video_path), "nframes": nframes},
        ]}]
        _text = attention_processor.apply_chat_template(
            _conv, add_generation_prompt=True, tokenize=False
        )
        _audios, _images, _videos = _process_mm_info(_conv, use_audio_in_video=True)
        _inp = attention_processor(
            text=_text, audio=_audios, images=_images, videos=_videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        validate_encoded_inputs(_inp)
        _inp = {k: v.to(attention_model.device) for k, v in _inp.items()}
        _types = create_attention_token_mapping(
            _inp["input_ids"], attention_model.config.thinker_config
        )
        return cache_put(playground_caches["encode"], _key, (_inp, _types))

    def _diversity(csv_path):
        # Reproduce the "diversity by layer" logic: per layer, count distinct decoded
        # tokens across the audio-token rows, and the most-common prediction's share.
        with open(csv_path, newline="", encoding="utf-8") as _fh:
            _data = list(csv.reader(_fh))[1:]  # drop the header row
        if not _data:
            return [], [], 0
        _cols = list(zip(*(_r[2:] for _r in _data)))  # one tuple per thinker layer
        _uniq = [len(set(_c)) for _c in _cols]
        _dom = [Counter(_c).most_common(1)[0][1] / len(_c) for _c in _cols]
        return _uniq, _dom, len(_data)

    def _run_pass(rules, tag, inp, types):
        _csv_path = _results_dir / f"interactive_logit_lens_{tag}.csv"
        if _csv_path.exists():
            _csv_path.unlink()  # no stale results if this run has no audio tokens
        register_logit_lens_hooks(attention_model)
        try:
            _ctx = (
                block_attention(
                    attention_model, rules, types, len(types), track_attention=False,
                    # One forward pass over the prompt: only the prefill branch
                    # fires, so `generated`/`answer` rules would mask nothing.
                    # Saying so here is what turns a silent baseline into a
                    # refusal the student can read.
                    context="forward",
                )
                if rules else _nullcontext()
            )
            with _ctx:
                with torch.no_grad():
                    attention_model.thinker(**inp, output_hidden_states=True)
            analyze_and_save_audio_logits_to_csv(
                attention_model, attention_processor, types, filename=str(_csv_path)
            )
        finally:
            clear_logit_lens_hooks()
        if not _csv_path.exists():
            return [], [], 0
        return _diversity(_csv_path)

    _scoreboard = None
    try:
        with mo.status.spinner(
            title=f"Logit-lens 순전파 · {_nframes} 프레임 · {_video_path.name}…"
        ):
            _inp, _types = _prep(_video_path, _nframes, _prompt)  # encode the clip once
            if _rules:
                _ko_u, _ko_d, _n_audio = _run_pass(_rules, "knockout", _inp, _types)
                _bl_u, _bl_d = (None, None)
                if _compare:
                    _bl_u, _bl_d, _ = _run_pass([], "baseline", _inp, _types)
            else:
                _bl_u, _bl_d, _n_audio = _run_pass([], "baseline", _inp, _types)
                _ko_u, _ko_d = (None, None)
    except Exception as _e:  # noqa: BLE001 — surface any run failure in-notebook
        _scoreboard = mo.callout(
            mo.md(f"**실행 실패** — `{type(_e).__name__}: {_e}`"), kind="danger"
        )

    if _scoreboard is None:
        _primary_u = _ko_u if _ko_u else _bl_u
        _primary_d = _ko_d if _ko_d else _bl_d
        _both = bool(_ko_u) and bool(_bl_u)

    if _scoreboard is not None:
        pass
    elif not _primary_u:
        _scoreboard = mo.callout(
            mo.md(
                f"이 프롬프트로 `{_video_path.name}`에서 **오디오 토큰이 하나도** 나오지 "
                "않았습니다. 채점할 오디오 위치 예측이 없습니다. 오디오 트랙이 있는 클립으로 "
                "시도해 보세요."
            ),
            kind="warn",
        )
    else:
        _n_l = len(_primary_u)
        _order = sorted(range(_n_l), key=lambda k: _primary_u[k], reverse=True)

        _rows = []
        for _rank, _i in enumerate(_order, 1):
            _row = {"순위": _rank, "레이어": _i, "서로 다른 예측": _primary_u[_i]}
            if _both:
                _row["기준선"] = _bl_u[_i]
                _row["Δ (기준선 대비)"] = _ko_u[_i] - _bl_u[_i]
            _row["최빈 비율"] = round(_primary_d[_i], 3)
            _rows.append(_row)
        _table = mo.ui.table(_rows, selection=None, pagination=True, page_size=12)

        _peak = _order[0]
        _stats = [
            mo.stat(
                value=f"레이어 {_peak}",
                label="다양성 최고점",
                caption=f"서로 다른 예측 {_primary_u[_peak]}개",
                bordered=True,
            ),
            mo.stat(
                value=f"{sum(_primary_u) / _n_l:.1f}",
                label="레이어당 평균 고유 예측 수",
                caption=f"thinker 레이어 {_n_l}개 전체",
                bordered=True,
            ),
            mo.stat(
                value=str(_n_audio),
                label="채점한 오디오 토큰",
                caption="클립 재생 시간으로 고정 — 프레임 수는 이 값을 움직이지 않습니다",
                bordered=True,
            ),
            mo.stat(
                value=str(Counter(_types).get("video", 0)),
                label="인코딩된 비디오 토큰",
                caption=f"프레임 수={_nframes}가 움직이는 것은 이 값입니다",
                bordered=True,
            ),
        ]
        if _both:
            _mean_delta = sum(_ko_u[k] - _bl_u[k] for k in range(_n_l)) / _n_l
            _less = sum(1 for k in range(_n_l) if _ko_u[k] < _bl_u[k])
            _stats.append(
                mo.stat(
                    value=f"{_mean_delta:+.1f}",
                    label="녹아웃에 의한 평균 Δ",
                    caption=f"{_n_l}개 중 {_less}개 레이어에서 다양성 감소",
                    direction="decrease" if _mean_delta < 0 else "increase",
                    bordered=True,
                )
            )

        _x = np.arange(_n_l)
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
        if _both:
            _axes[0].bar(_x, _ko_u, color="#4C78A8", label="녹아웃")
            _axes[0].plot(_x, _bl_u, color="#F58518", marker="o", ms=3, lw=1.5, label="기준선")
            _axes[0].legend()
            _axes[0].set(title="레이어별 서로 다른 예측 수",
                         xlabel="Thinker 레이어", ylabel="서로 다른 예측 수")
            _delta = [_ko_u[k] - _bl_u[k] for k in range(_n_l)]
            _axes[1].bar(_x, _delta, color=["#E45756" if d < 0 else "#54A24B" for d in _delta])
            _axes[1].axhline(0, color="black", lw=0.8)
            _axes[1].set(title="Δ 다양성 (녹아웃 − 기준선)",
                         xlabel="Thinker 레이어", ylabel="Δ 서로 다른 예측 수")
        else:
            _axes[0].bar(_x, _primary_u, color="#4C78A8")
            _axes[0].set(title="레이어별 Logit-lens 다양성",
                         xlabel="Thinker 레이어", ylabel="서로 다른 예측 수")
            _axes[1].plot(_x, _primary_d, marker="o", color="#F58518")
            _axes[1].set(title="최빈 예측 비율",
                         xlabel="Thinker 레이어", ylabel="비율", ylim=(0, 1))
        for _ax in _axes:
            _ax.grid(axis="y", alpha=0.25)

        _rule_txt = (
            " + ".join(f"`{r[0]}→{r[1]}` [{r[2]},{r[3]})" for r in _rules)
            if _rules else "_없음 (기준선만)_"
        )
        _children = [
            mo.md(
                f"**클립** `{_video_path.name}`"
                + (" _(무음 대조군)_" if _is_control else "")
                + f" &nbsp;·&nbsp; **프레임 수** {_nframes} "
                f"&nbsp;·&nbsp; **프롬프트** _{_prompt}_ &nbsp;·&nbsp; **녹아웃** {_rule_txt}"
            ),
            mo.hstack(_stats, widths="equal", gap=1),
            _fig,
            mo.md("<span style=\"font-size:1.15rem;font-weight:600\">디코딩된 예측의 다양성 순으로 정렬한 레이어 (클수록 오디오 토큰 예측이 더 다양함)</span>"),
            _table,
            mo.md(
                "<span style=\"color:#4C78A8;font-weight:600\">다음 →</span> source를 `video`로 바꾸거나, 같은 규칙을 **무음 대조군**에 "
                "돌려 두 스코어보드를 비교해 보세요."
            ),
        ]
        _scoreboard = mo.vstack(_children)

        # Record it. `_mean_delta` only exists when a baseline was run alongside
        # the knockout; without one there is no effect size to log, only a
        # description — so the metric reported changes with it, rather than
        # stacking two different quantities in one column.
        try:
            if _both:
                _metric = ("mean_delta_diversity", round(_mean_delta, 3), "unique preds")
            else:
                _metric = (
                    "mean_unique_per_layer", round(sum(_primary_u) / _n_l, 3), "unique preds"
                )
            set_runs(
                lambda _prev, _r=run_record(
                    kind="diversity",
                    condition=(
                        " + ".join(f"{r[0]}→{r[1]} [{r[2]},{r[3]})" for r in _rules)
                        if _rules else "기준선 (녹아웃 없음)"
                    ),
                    metric_name=_metric[0],
                    metric_value=_metric[1],
                    metric_unit=_metric[2],
                    config=experiment_config(
                        _video_path, nframes=_nframes, prompt=_prompt,
                        rules=[list(r) for r in _rules], compare=_compare,
                    ),
                    is_control=_is_control,
                    extra={"audio_tokens": _n_audio, "peak_layer": _peak,
                           "baseline_unique": _bl_u, "knockout_unique": _ko_u,
                           "baseline_dominance": _bl_d, "knockout_dominance": _ko_d,
                           "layers": _rows,
                           "measurement": "decoded raw-probe top-1 string diversity; not model quality"},
                ): append_run(_prev, _r, log_path=LEDGER_LOG)
            )
        except Exception as _le:  # noqa: BLE001 — a ledger bug must never eat a run
            print("ledger append failed:", type(_le).__name__, _le)
    _scoreboard
    return


@app.cell
def _(ledger_view):
    ledger_view()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 인터랙티브: 티처 포싱 Δ log-우도

    **질문:** 같은 캡션을 채점할 때, 선택한 직접 어텐션 연결을 차단하면 토큰 확률이
    얼마나 달라질까요? 먼저 모델이 만든 캡션을 고정하고, 기준선과 녹아웃에서 그 캡션을
    다시 입력해 점수를 매깁니다. **Δ = 녹아웃 − 기준선**, 단위는 nats입니다.
    음수는 그 캡션에 부여한 확률이 줄었다는 뜻이며 정답 여부나 모델의 확신 점수는 아닙니다.

    ### 탐색 1 · 원본과 무음을 같은 설정으로 비교하기

    1. 한국어 프롬프트, 8프레임, `audio`, `[0,36)`, 상한32로 **무음 대조군**을 실행합니다.
    2. **클립만 기본 클립으로** 바꾸어 실행합니다. 두 실행의 ID·전체 캡션·토큰당 Δ를 기록하세요.
    3. 프롬프트를 `Describe what you hear in the video`로 바꾸고 **영어 무음/원본 쌍을 새로** 만듭니다.
       영어 원본을 한국어 무음과 짝지으면 프롬프트와 오디오가 동시에 바뀝니다.

    **예상과 관측을 구분하세요.** 이 예제에서는 한국어 쌍의 효과가 작고 영어 쌍에서 차이가
    커질 수 있습니다. 직접 재현해 확인하세요. 무음도 표현을 만들므로 Δ=0이 보장되지 않습니다.
    비슷한 캡션이나 작은 Δ는 **이 설정에서 선택한 직접 연결에 대한 민감도가 작았다**는 결과입니다.
    모델이 소리를 전혀 쓰지 않았거나 측정이 완전하다는 증거는 아닙니다. 다른 연결,
    중복된 정보, 문장 내용과 지표의 민감도도 가능한 설명입니다.

    ### 탐색 2 · 한 변수만 더 바꾸기

    타깃을 `video`로 바꾸거나 레이어를 `[0,12)`로 좁혀 보세요. 새 설정에 맞는 무음 실행도
    함께 남깁니다. 캡션이 잘렸다면 최대 토큰 수를 32→64로 바꾸고 같은 쌍을 다시 만드세요.
    서로 다른 캡션의 평균 Δ는 길이 영향을 줄일 뿐 **내용·언어 차이를 통제하지는 않습니다.**

    `answer`는 채점할 캡션 토큰입니다. 첫 토큰은 마지막 프롬프트 위치에서 예측하므로
    answer-only 규칙의 직접 차단 범위 밖에 있습니다. `query_text`에는 질문뿐 아니라
    채팅 구조·특수 토큰도 포함됩니다. 표시 단위(어절·특수 토큰)의 색은 여러 토큰의 Δ를 합친 값입니다.
    `⟨special⟩`은 문장 종료 같은 특수 토큰으로, 이 점수를 이웃 단어의 의미로 해석하지 마세요.
    제출 전에는 실행되지 않습니다.
    """)
    return


@app.cell
def _(
    CLIP_CHOICES,
    CLIP_DEFAULT,
    CLIP_UPLOAD,
    LOGIT_PROMPT,
    NFRAMES,
    attention_model,
    mo,
    validate_experiment,
):
    _n_layers = len(attention_model.thinker.model.layers)
    _tf_targets = ["audio", "video", "query_text"]
    _tf_template = (
        "**클립** {clip} &nbsp; (반증 가능한 쪽은 `무음 대조군`입니다 — 저장소 안에 "
        "있으니 업로드하지 말고 이름으로 고르세요)\n\n"
        "**업로드**를 골랐을 때만 — `mp4 / mov / mkv / webm`, 250 MB·120초·1080p 이하이며, 디코딩 크기 512 MiB도 충족해야 합니다. 짧은 클립 권장:\n\n"
        "{video}\n\n"
        "**클립에서 샘플링할 프레임 수** {nframes}\n\n"
        "**프롬프트** {prompt}\n\n"
        "---\n\n"
        "**캡션 최대 토큰 수** {max_new_tokens} — 상한에 닿으면 문장이 끊길 수 있습니다.\n\n"
        "**answer**가 {target} 에 어텐션하는 것을 thinker 레이어 {layers} 구간에서 금지\n\n"
        f"(`answer`는 모델 자신의 캡션을 티처 포싱으로 다시 넣은 것입니다. 이 thinker는 "
        f"레이어가 **{_n_layers}**개이고 `end`는 배타적입니다.)"
    )

    def _tf_validate(_v):
        if not _v:
            return None
        try:
            validate_experiment(_v.get("nframes", NFRAMES), _v.get("prompt", LOGIT_PROMPT),
                                _v.get("max_new_tokens", 32))
        except ValueError as _err:
            return str(_err)
        # Label *or* value: see the note in the 🎛️ form's validator.
        if _v.get("clip") in ("Upload", CLIP_UPLOAD) and not _v.get("video"):
            return "업로드를 선택했지만 파일을 고르지 않았습니다."
        # `.get` with a default: the batch value is partial on first render.
        _lo, _hi = _v.get("layers") or (0, 1)
        if int(_hi) <= int(_lo):
            return f"[{int(_lo)}, {int(_hi)})는 0개 레이어를 마스킹합니다 — `end`는 배타적입니다."
        return None

    tf_controls = mo.md(_tf_template).batch(
        clip=mo.ui.radio(CLIP_CHOICES, value=CLIP_DEFAULT, inline=True),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm"], multiple=False, kind="area"
        ),
        nframes=mo.ui.slider(2, 16, step=2, value=NFRAMES, show_value=True, include_input=True),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        max_new_tokens=mo.ui.slider(8, 128, step=8, value=32, show_value=True, include_input=True),
        target=mo.ui.dropdown(_tf_targets, value="audio"),
        layers=mo.ui.range_slider(0, _n_layers, step=1, value=[0, _n_layers], show_value=True, full_width=True),
    ).form(
        submit_button_label="▶ 티처 포싱 Δ log-우도 실행",
        bordered=True,
        validate=_tf_validate,
    )
    tf_controls
    return (tf_controls,)


@app.cell
def _(
    LEDGER_LOG,
    USE_PRECOMPUTED,
    append_run,
    attention_model,
    attention_processor,
    cache_put,
    caption_cache_key,
    create_attention_token_mapping,
    experiment_config,
    mo,
    np,
    playground_caches,
    resolve_clip,
    run_provenance,
    run_record,
    set_runs,
    tf_controls,
    validate_encoded_inputs,
    validate_experiment,
):
    from qwen_omni_utils import process_mm_info as _tf_mm_info

    from src.teacher_forcing import teacher_forced_delta as _tfd

    _tp = tf_controls.value
    mo.stop(
        _tp is None,
        mo.callout(
            mo.md("위에서 파라미터를 설정하고 **▶ 티처 포싱 Δ log-우도 실행**을 누르세요."),
            kind="info",
        ),
    )
    mo.stop(
        USE_PRECOMPUTED,
        mo.callout(
            mo.md(
                "**이 측정은 라이브 모델이 필요합니다** — 캡션을 생성하고 순전파 2회로 "
                "채점하므로 `USE_PRECOMPUTED=True`인 동안에는 건너뜁니다."
            ),
            kind="warn",
        ),
    )

    _tf_video, _tf_is_control, _tf_clip_err = resolve_clip(_tp["clip"], _tp["video"])
    mo.stop(
        _tf_clip_err is not None,
        mo.callout(mo.md(f"**클립을 쓸 수 없습니다** — {_tf_clip_err}"), kind="danger"),
    )
    _tf_nframes = int(_tp["nframes"])
    _tf_prompt = _tp["prompt"].strip()
    _tf_max_tokens = int(_tp["max_new_tokens"])
    validate_experiment(_tf_nframes, _tf_prompt, _tf_max_tokens)
    _tf_lo, _tf_hi = int(_tp["layers"][0]), int(_tp["layers"][1])
    _tf_rules = [("answer", _tp["target"], _tf_lo, _tf_hi)]

    def _tf_prep(video_path, nframes, prompt):
        # Shared encode cache with the 🎛️ section: a layer-band or target sweep
        # on the same clip/prompt re-encodes nothing after the first ▶.
        _key = caption_cache_key(video_path, nframes, prompt, 0, run_provenance["model_revision"])
        if _key in playground_caches["encode"]:
            return playground_caches["encode"][_key]
        _conv = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": str(video_path), "nframes": nframes},
        ]}]
        _text = attention_processor.apply_chat_template(
            _conv, add_generation_prompt=True, tokenize=False
        )
        _audios, _images, _videos = _tf_mm_info(_conv, use_audio_in_video=True)
        _inp = attention_processor(
            text=_text, audio=_audios, images=_images, videos=_videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        validate_encoded_inputs(_inp)
        _inp = {k: v.to(attention_model.device) for k, v in _inp.items()}
        _types = create_attention_token_mapping(
            _inp["input_ids"], attention_model.config.thinker_config
        )
        return cache_put(playground_caches["encode"], _key, (_inp, _types))

    tf_result = None
    _tf_out = None
    try:
        # The caption cache includes file content, prompt, frames, model revision,
        # and the generation cap. A rule/layer sweep can reuse the same caption.
        _tf_cap_key = caption_cache_key(_tf_video, _tf_nframes, _tf_prompt, _tf_max_tokens, run_provenance["model_revision"])
        _tf_cached_c = playground_caches["caption"].get(_tf_cap_key)
        with mo.status.spinner(
            title=f"티처 포싱 · {_tf_nframes} 프레임 · {_tf_video.name}"
            + (" · 캡션 캐시 사용…" if _tf_cached_c is not None else "…")
        ):
            _tf_inp, _tf_types = _tf_prep(_tf_video, _tf_nframes, _tf_prompt)
            validate_encoded_inputs(_tf_inp, max_tokens=4096 - _tf_max_tokens)
            _tf_res = _tfd(
                attention_model, attention_processor, _tf_inp, _tf_types, _tf_rules,
                # Without this the playground scored 32-token captions (the
                # function's own default) while the fixed cell above used
                # MAX_NEW_TOKENS — two different caption lengths, one Σ column.
                max_new_tokens=_tf_max_tokens,
                cached_caption_ids=_tf_cached_c,
            )
            cache_put(playground_caches["caption"], _tf_cap_key, _tf_res["caption_ids"])
            tf_result = _tf_res
    except Exception as _e:  # noqa: BLE001 — surface any failure in-notebook
        _tf_out = mo.callout(
            mo.md(f"**실행 실패** — `{type(_e).__name__}: {_e}`"), kind="danger"
        )

    if _tf_out is None:
        _tf_delta = [float(x) for x in _tf_res["delta"].detach().cpu().float().tolist()]
        _tf_total = _tf_res["delta_total"]
        _tf_toks = _tf_res["caption_tokens"]
        _tf_worst = int(np.argmin(_tf_delta)) if _tf_delta else 0
        _tf_rule_txt = f"`answer→{_tp['target']}` [{_tf_lo},{_tf_hi})"
        _tf_mean = _tf_res["delta_mean"]
        _tf_stats = [
            mo.stat(
                value=f"{_tf_mean:+.3f}",
                label="토큰당 Δ (nats)",
                caption="길이로 나눈 값 · 같은 설정의 쌍과 캡션 내용도 함께 비교",
                direction="decrease" if _tf_mean < 0 else "increase",
                bordered=True,
            ),
            mo.stat(
                value=f"{_tf_total:+.2f}",
                label="Σ Δ log-우도 (nats)",
                caption="녹아웃 − 기준선 · 음수 = 해당 캡션의 확률 감소",
                direction="decrease" if _tf_total < 0 else "increase",
                bordered=True,
            ),
            mo.stat(
                value=(_tf_toks[_tf_worst].strip() or "·") if _tf_toks else "—",
                label="Δ가 가장 작은 토큰",
                caption=(f"Δ = {_tf_delta[_tf_worst]:+.2f} nats" if _tf_delta else ""),
                bordered=True,
            ),
            mo.stat(
                value=str(len(_tf_toks)),
                label="채점한 캡션 토큰 수",
                caption="티처 포싱, greedy",
                bordered=True,
            ),
        ]
        _tf_out = mo.vstack([
            mo.md(
                f"**클립** `{_tf_video.name}`"
                + (" _(무음 대조군)_" if _tf_is_control else "")
                + f" &nbsp;·&nbsp; **프레임 수** {_tf_nframes} "
                f"&nbsp;·&nbsp; **프롬프트** _{_tf_prompt}_ &nbsp;·&nbsp; **녹아웃** {_tf_rule_txt}"
            ),
            mo.callout(mo.md("**채점한 전체 캡션**\n\n" + _tf_res["caption_text"]
                             + (f"\n\n⚠ 최대 {_tf_max_tokens}토큰에 도달했습니다. 문장이 미완성일 수 있습니다."
                                if _tf_res["generation_truncated"] else "\n\n생성이 끝났습니다.")),
                       kind="warn" if _tf_res["generation_truncated"] else "neutral"),
            mo.hstack(_tf_stats, widths="equal", gap=1),
            mo.md(
                "<span style=\"color:#4C78A8;font-weight:600\">다음 →</span> 타깃을 `video`로 바꾸거나 레이어를 `[0,12)`로 좁혀 보세요. "
                "영어로 바꾸면 차이가 날 수 있습니다. 같은 조건의 원본·무음 쌍으로 확인하세요."
            ),
        ])
        try:
            set_runs(
                lambda _prev, _r=run_record(
                    kind="teacher_forcing",
                    condition=f"answer→{_tp['target']} [{_tf_lo},{_tf_hi})",
                    # Δ/token, not Σ: the control and the experiment score
                    # different captions of different lengths, so the total is not
                    # the comparable quantity — logging Σ as the headline would
                    # rebuild the exact confusion this section exists to remove.
                    metric_name="delta_per_token",
                    metric_value=_tf_mean,
                    metric_unit="nats/token",
                    config=experiment_config(
                        _tf_video, nframes=_tf_nframes, prompt=_tf_prompt,
                        target=_tp["target"], start=_tf_lo, end=_tf_hi,
                        max_new_tokens=_tf_max_tokens,
                    ),
                    is_control=_tf_is_control,
                    extra={"delta_total": _tf_total, "delta_mean": _tf_mean,
                           "n_tokens": len(_tf_toks), "caption_text": _tf_res["caption_text"],
                           "caption_ids": _tf_res["caption_ids"][0].detach().cpu().tolist(),
                           "caption_tokens": _tf_toks, "caption_token_kinds": _tf_res["caption_token_kinds"],
                           "delta": _tf_delta,
                           "baseline_logprobs": _tf_res["baseline_logprobs"].detach().cpu().tolist(),
                           "knockout_logprobs": _tf_res["knockout_logprobs"].detach().cpu().tolist(),
                           "baseline_distribution": _tf_res["baseline_distribution"],
                           "knockout_distribution": _tf_res["knockout_distribution"],
                           "generation_truncated": _tf_res["generation_truncated"],
                           "generation_end_reason": _tf_res["generation_end_reason"]},
                ): append_run(_prev, _r, log_path=LEDGER_LOG)
            )
        except Exception as _le:  # noqa: BLE001 — a ledger bug must never eat a run
            print("ledger append failed:", type(_le).__name__, _le)
    _tf_out
    return (tf_result,)


@app.cell
def _(mo, tf_result):
    # No output until the form above has produced a result (and skipped after a
    # failed run) — mirrors the W9 threshold cells.
    mo.stop(tf_result is None)
    from src.teacher_forcing import threshold_slider_params as _tf_params
    from wigglystuff import TangleSlider as _TfTangle

    tf_threshold = mo.ui.anywidget(_TfTangle(
        suffix=" nats",
        **_tf_params(tf_result["caption_tokens"], tf_result["delta"], token_kinds=tf_result.get("caption_token_kinds")),
    ))
    mo.md(
        "<span style=\"font-size:1.15rem;font-weight:600\">토큰별 Δ log-우도 (뜨거운 색 = 녹아웃 뒤 log-확률 감소. 표시 단위에 마우스를 "
        "올리면 그 토큰들의 nats가 보입니다)</span>\n\n"
        f"{tf_threshold} 이상 잃은 표시 단위만 표시합니다 — **밑줄 친 숫자를 옆으로 드래그**하거나 "
        "클릭해서 입력하세요. 다시 그려지는 것은 이 띠뿐이며 모델은 건드리지 않습니다."
    )
    return (tf_threshold,)


@app.cell
def _(mo, selected_drop_share, tf_result, tf_threshold):
    from src.teacher_forcing import group_tokens_into_words as _tf_group
    from src.teacher_forcing import render_delta_strip as _tf_strip

    _delta = [float(_x) for _x in tf_result["delta"].detach().cpu().float().tolist()]
    _toks = tf_result["caption_tokens"]
    _th = abs(float(tf_threshold.value.get("amount", 0.0)))
    _words = _tf_group(_toks, _delta, token_kinds=tf_result.get("caption_token_kinds"))
    _hit = [_w for _w in _words if _w[1] < -_th]
    _share = selected_drop_share(tf_result["caption_tokens"], _delta, _th, token_kinds=tf_result.get("caption_token_kinds"))
    _rows = [
        {"위치": _i, "토큰": _t or ("특수 토큰" if tf_result["caption_token_kinds"][_i] == "special" else "문자 이어짐"),
         "토큰 종류": tf_result["caption_token_kinds"][_i], "Δ log-우도": round(_d, 3)}
        for _i, (_t, _d) in enumerate(zip(_toks, _delta))
    ]
    mo.vstack([
        mo.Html(
            "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
            + _tf_strip(_toks, _delta, highlight_below=_th, vmax=8.0, token_kinds=tf_result.get("caption_token_kinds"))
            + "</div>"
        ),
        mo.md(
            f"**{len(_hit)}/{len(_words)}** 표시 단위가 −{_th:.2f} nats보다 크게 떨어졌습니다 — 합쳐서 "
            f"Δ = {sum(_w[1] for _w in _hit):+.2f} nats입니다. 감소한 표시 단위의 총 감소량 중 "
            f"선택된 단위가 **{_share:.0f}%**를 차지합니다. 색 범위는 모든 실행에서 ±8 nats입니다."
        ),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=16),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📓 실습 기록 — 관측에서 주장으로

    서로 다른 설정·결과를 기록합니다. **같은 실행은 한 행으로 합치며 판정은 보존합니다.**
    프롬프트·프레임·규칙·캡션 상한·코드/모델 버전이 같은 제공 클립 쌍만 연결됩니다.
    “짝 없음”은 아직 대응 실행이 없다는 뜻입니다. 짝이 있다는 표시도 인과 결론을 보증하지는 않습니다.
    업로드와 대역의 무효과 예상은 자동으로 검증된 대조군이 되지 않습니다.

    **기록할 세 문장:** 무엇을 하나 바꾸었나? 무엇을 관측했나? 같은 결과의 다른 설명은 무엇인가?
    판정은 그 문장에 대해 내립니다. 가설이 없거나 근거가 부족하면 **미검증**을 선택하세요.

    **Markdown는 읽는 제출물, JSON은 전체 결과와 설정을 보존하는 파일**입니다. 두 파일을 함께
    내려받으세요. 기록표의 저장 상태가 실패/메모리 전용이면 세션 안에서 보이더라도 디스크에
    저장됐다고 가정하지 마세요. 세션 종료 전에 내보내고 실제 파일을 열어 확인하세요.
    """)
    return


@app.cell
def _(mo):
    # Refs = {mo} only, deliberately. A form rebuilt on every append would clear
    # itself the moment the run it is describing is recorded.
    # The dropdown shows Korean but submits the English values `run_ledger.VERDICTS`
    # validates — the ledger vocabulary is shared with the English notebook.
    verdict_form = mo.md(
        "**실행에 판정 내리기** — 위 기록에서 `id`를 복사해 오세요.\n\n"
        "실행 {run_id}에서 남길 **내 주장 또는 관측**: {claim}\n\n"
        "이 주장에 대한 판정 {verdict}. 지지됨/반박됨에는 주장을 적어야 합니다.\n\n"
        "같은 숫자를 낳을 수 있는 **경쟁 설명**: {rival}"
    ).batch(
        run_id=mo.ui.text(placeholder="예: 3f9a1c02"),
        claim=mo.ui.text_area(placeholder="예: 이 설정의 직접 audio 연결 차단에서 캡션 확률 감소가 관측됐다", full_width=True),
        verdict=mo.ui.dropdown(
            {"지지됨 (supported)": "supported",
             "반박됨 (refuted)": "refuted",
             "미검증 (untested)": "untested"},
            value="미검증 (untested)",
        ),
        rival=mo.ui.text(placeholder="예: 캡션이 그냥 짧아졌을 뿐이다", full_width=True),
    ).form(submit_button_label="판정 기록", bordered=True)
    verdict_form
    return (verdict_form,)


@app.cell
def _(LEDGER_LOG, mo, set_runs, verdict_form):
    from src.run_ledger import apply_verdict_checked as _checked
    _v = verdict_form.value
    mo.stop(_v is None)
    _feedback = []
    def _save_verdict(_prev):
        _updated, _status = _checked(_prev, (_v.get("run_id") or "").strip(),
                                    _v.get("verdict", "untested"),
                                    rival=_v.get("rival", ""), log_path=LEDGER_LOG,
                                    claim=_v.get("claim", ""))
        _feedback.append(_status)
        return _updated
    set_runs(_save_verdict)
    _status = _feedback[0]
    if _status["ok"]:
        _save = _status.get("save_status", {})
        _saved = _save.get("state") == "saved"
        _message = f"실행 `{_status['run_id']}`의 판정과 경쟁 설명을 반영했습니다. "
        _message += "세션 파일에도 저장했습니다." if _saved else "파일 저장을 확인하지 못했습니다. 지금 내보내세요."
        _feedback_view = mo.callout(mo.md(_message), kind="success" if _saved else "warn")
    else:
        _why = ("지지됨/반박됨의 대상인 주장을 먼저 적어 주세요."
                if _status.get("error") == "missing_claim"
                else "위 표의 실행 ID를 그대로 복사하고 판정 값을 확인하세요.")
        _feedback_view = mo.callout(mo.md("**판정을 저장하지 않았습니다.** " + _why), kind="danger")
    _feedback_view
    return


@app.cell
def _(ledger_view):
    ledger_view()
    return


@app.cell
def _(get_runs, mo, run_provenance, worksheet_md):
    from src.run_ledger import build_evidence_json as _evidence_json
    _runs = get_runs()
    _md = worksheet_md(_runs)
    _json = _evidence_json(_runs, provenance=run_provenance)
    mo.vstack([
        mo.md("### 제출할 결과 묶음 — 두 파일을 함께 보관하세요"),
        mo.hstack([
            mo.download(_md.encode("utf-8"), filename="lab_log.md", mimetype="text/markdown",
                        label=f"⬇ 읽는 워크시트 ({len(_runs)}건)"),
            mo.download(_json.encode("utf-8"), filename="lab_evidence.json", mimetype="application/json",
                        label=f"⬇ 전체 설정·캡션·토큰 결과 JSON ({len(_runs)}건)"),
        ]),
        mo.accordion({"Markdown 보기 · 다운로드가 안 되면 복사": mo.md(f"````markdown\n{_md}\n````"),
                      "JSON 보기 · 다운로드가 안 되면 복사": mo.md(f"````json\n{_json}\n````")}),
        mo.md("파일을 열어 실행 ID·클립·프롬프트·캡션이 있는지 확인하세요. JSON은 원본 영상 파일을 포함하지 않습니다."),
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
