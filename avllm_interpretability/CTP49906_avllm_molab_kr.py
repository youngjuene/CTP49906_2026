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

__generated_with = "0.23.14"
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

    1. **Logit Lens** — thinker 레이어들을 가로질러, **오디오 토큰 위치**에서 모델의
       중간 예측을 디코딩합니다.
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

    - **GPU:** 헤더의 notebook-specs 버튼으로 GPU를 연결하세요. 이 노트북은
      `cuda:0`을 사용하며, 3B 모델은 molab의 VRAM에 넉넉히 올라갑니다.
    - **의존성:** 셋업 셀이 커널에 pip로 설치합니다(molab은 `# /// script` 블록을
      자동으로 반영하지 않습니다). 또한 molab에 들어 있는 torchvision에는 비디오
      디코더가 없으므로, 작은 PyAV 심(shim)으로 `torchvision.io.read_video`를
      복원합니다.
    - 실험 코드(`src/`)와 샘플 클립은 아래 셋업 셀이
      `youngjuene/CTP49906_2026`에서 클론해 옵니다.
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

    모델은 영상을 "보지" 않습니다. 잘게 쪼갠 **토큰** 한 줄을 읽을 뿐입니다. 토큰마다
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
      <text x="550" y="100" font-size="11.5" fill="currentColor" text-anchor="middle">무음 대조군 ≈ 0</text>
      <path d="M534 124 L406 124" stroke="currentColor" stroke-width="1.4" marker-end="url(#a44k)"/>
      <path d="M566 124 L694 124" stroke="currentColor" stroke-width="1.4" marker-end="url(#a44k)"/>
      <text x="400" y="144" font-size="11.5" fill="currentColor">← 덜 믿게 됨</text>
      <text x="700" y="144" font-size="11.5" fill="currentColor" text-anchor="end">더 믿게 됨 →</text>
      <text x="390" y="164" font-size="11.5" fill="currentColor" font-style="italic">여러분의 클립은 0에서 얼마나 멀어지나?</text>
    </svg>

    **다양성**은 "무슨 일이 있었나"를 적은 것이지 "좋아졌나"가 아닙니다. **Δ log-우도**는
    **부호**가 방향을(음수 = 자기 답을 덜 믿게 됨, 양수 = 더 믿게 됨), **크기**가 세기를
    말합니다. 그리고 그 크기는 **무음 대조군 옆에 놓아야** 의미가 생깁니다 — 그래서
    대조군을 먼저 돌립니다.
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
        ("accelerate", "accelerate", "1.14.0", "accelerate==1.14.0"),
        ("qwen_omni_utils", "qwen-omni-utils", None, "qwen-omni-utils==0.0.9"),
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
        ("wigglystuff", "wigglystuff", "0.5.21", "wigglystuff==0.5.21"),
        # Listed explicitly even though wigglystuff pulls it in: `src/probe_grid.py`
        # is a first-party anywidget and should not depend on another package's
        # dependency graph to be importable.
        ("anywidget", "anywidget", "0.9.2", "anywidget>=0.9.2"),
    ])

    def _ensure_video_reader():
        # molab ships its own recent torch/torchvision and ignores the
        # `# /// script` pins above. torchvision >= 0.23 dropped the built-in
        # video decoder, so `torchvision.io.read_video` no longer exists and
        # qwen-omni-utils' default torchvision backend dies with
        # `AttributeError: module 'torchvision.io' has no attribute 'read_video'`.
        # PyAV is already installed (qwen uses it to read the audio track), so
        # restore read_video on top of PyAV — no version-fragile CUDA wheels
        # (torchcodec/decord) and no reliance on system codecs.
        import torchvision

        if hasattr(torchvision.io, "read_video"):
            return  # normal torchvision (e.g. the pinned 0.21.0) — nothing to do
        import av
        import numpy as np
        import torch

        def _read_video_pyav(
            filename, start_pts=0.0, end_pts=None, pts_unit="sec", output_format="TCHW"
        ):
            # Minimal torchvision.io.read_video replacement covering the single
            # call qwen makes: it only reads `video.size(0)` and `info["video_fps"]`.
            if isinstance(filename, str) and filename.startswith("file://"):
                filename = filename[len("file://") :]
            container = av.open(filename)
            try:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                rate = stream.average_rate or stream.guessed_rate or stream.base_rate
                video_fps = float(rate) if rate else 30.0
                frames = []
                for frame in container.decode(video=0):
                    ts = frame.time
                    if pts_unit == "sec" and ts is not None:
                        if ts < start_pts:
                            continue
                        if end_pts is not None and ts > end_pts:
                            break
                    frames.append(frame.to_ndarray(format="rgb24"))  # (H, W, C) uint8
            finally:
                container.close()
            if frames:
                video = torch.from_numpy(np.stack(frames))  # (T, H, W, C)
            else:
                video = torch.zeros((0, 0, 0, 3), dtype=torch.uint8)
            if output_format.upper() == "TCHW":
                video = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
            # qwen extracts audio separately (process_audio_info), so an empty
            # placeholder here is fine; it only unpacks and discards this value.
            audio = torch.zeros((1, 0), dtype=torch.float32)
            return video, audio, {"video_fps": video_fps, "audio_fps": None}

        torchvision.io.read_video = _read_video_pyav
        print("molab 호환을 위해 torchvision.io.read_video를 PyAV 심으로 패치했습니다")

    _ensure_video_reader()

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

        if getattr(librosa.load, "__audioread_shim__", False):
            return  # already wrapped; this cell re-ran
        _librosa_load = librosa.load

        def _audioread_load(reader, offset, duration, dtype):
            # Ported from librosa 0.11.0's `__audioread_load`, minus its lookup
            # of the backend registry: audioread readers yield blocks of
            # interleaved little-endian 16-bit PCM, which scale to [-1, 1).
            buf = []
            with reader as input_file:  # closes it, and so the ffmpeg child
                sr_native = input_file.samplerate
                n_channels = input_file.channels
                s_start = int(sr_native * offset) * n_channels
                s_end = (
                    np.inf if duration is None
                    else s_start + int(sr_native * duration) * n_channels
                )
                n = 0
                for frame in input_file:
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

    # The experiment code (src/) and sample video live under the
    # `avllm_interpretability/` subdirectory of this repo. If the clone already
    # exists, hard-sync it to REPO_REF so pushed fixes reach molab (a kernel
    # restart is still needed to re-import updated modules).
    #
    # REPO_REF selects which branch or tag to sync: "main" for normal class use;
    # a feature branch to smoke-test unmerged work; a release tag (risk R7 in
    # the PRD) to pin the semester so September pushes can't change what
    # students execute mid-course. Works for branches and tags alike (fetch +
    # FETCH_HEAD, not origin/<branch>).
    REPO_REF = "main"
    REPO_DIR = Path("CTP49906_2026").resolve()
    if REPO_REF != "main":
        print(f"⚠️ REPO_REF={REPO_REF!r} — 이 노트북은 main이 아닌 ref에 고정되어 있습니다.")
    if REPO_DIR.exists():
        with mo.status.spinner(title=f"CTP49906_2026을 최신 {REPO_REF}로 업데이트하는 중…"):
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_REF],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "reset", "--hard", "FETCH_HEAD"], check=True
            )
    else:
        with mo.status.spinner(title=f"CTP49906_2026 @ {REPO_REF} 클론 중 (src + 샘플 영상)…"):
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", REPO_REF,
                 "https://github.com/youngjuene/CTP49906_2026.git", str(REPO_DIR)],
                check=True,
            )
    PROJECT_DIR = REPO_DIR / "avllm_interpretability"
    assert PROJECT_DIR.is_dir(), f"코드 디렉터리를 찾을 수 없습니다: {PROJECT_DIR}"
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    print("프로젝트 디렉터리:", PROJECT_DIR)
    return (PROJECT_DIR,)


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
    ## 파라미터

    자신의 영상을 쓰거나 개입(intervention)을 바꾸려면 여기를 편집하세요.
    `NFRAMES`는 조심해서 올리세요 — molab의 GPU는 96 GB로 넉넉하지만 병목은 **호스트
    RAM 32 GB**이고, 아래 어텐션 캡처가 `seq²`로 커집니다. 캡처를 켠 기본 설정에서는
    16 정도가 안전한 상한입니다.

    파라미터는 **재실행 비용**을 기준으로 세 셀로 나뉘어 있습니다(marimo는 편집한
    셀의 하위 셀을 모두 다시 실행합니다). 모델 id — 편집하면 모델을 다시 로드합니다;
    경로; 그리고 **노브** — 프롬프트, 규칙, 프레임 수. 노브 셀은 마음껏 만지세요.
    이미 로드된 모델로 실험만 다시 돌리므로(수 초) 모델 로드는 다시 하지 않습니다.

    각 노브가 실제로 무엇을 움직이는지:

    | 노브 | 무엇이 움직이는가 |
    |---|---|
    | `NFRAMES` | **비디오** 프레임을 몇 장 샘플링할지. 오디오 토큰 수는 클립의 재생 시간으로 고정되므로, 이 값은 Logit Lens 스코어보드의 행 수를 바꾸지 *않습니다*. |
    | `LOGIT_PROMPT` / `ATTENTION_PROMPT` | 지시문. `query_text` 위치를 바꾸며 캡션도 바뀔 수 있습니다. |
    | `KNOCKOUT_RULES` | `(source, target, start_layer, end_layer)`이며 `end`는 **배타적**입니다. 어떤 레이어에도, 어떤 토큰에도 걸리지 않는 규칙은 조용히 기준선을 돌려주는 대신 거부됩니다. |
    | `MAX_NEW_TOKENS` | 캡션 길이 — 따라서 Σ Δ log-우도도 함께 바뀝니다. 실행끼리 비교할 때 토큰당 평균을 봐야 하는 이유입니다. |
    | `ATTENTION_CAPTURE_LAYERS` | 아래 어텐션 히트맵이 다루는 레이어. 기본값은 `(0, 2)`입니다. 이 창을 넓히는 것은 진짜 실험이지만 진짜 비용이 듭니다. 캡처는 VRAM이 아니라 **호스트 RAM**에 쌓이며, 캡처한 레이어마다 **디코딩 스텝 하나당** `seq × seq` 텐서를 붙듭니다. 기본값은 약 4 GiB지만 `(0, 36)`은 약 **75 GiB**로, molab의 32 GB RAM을 넘겨 커널이 죽습니다. |
    """)
    return


@app.cell
def _():
    # Own cell on purpose: nothing but a genuine model change should ever
    # invalidate the loader cells below.
    MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
    return (MODEL_PATH,)


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
def _():
    # The knobs — cheap to tweak: re-runs the experiments, not the model loads.
    NFRAMES = 8
    LOGIT_PROMPT = "영상에서 들리는 소리를 설명해 주세요"
    ATTENTION_PROMPT = "영상에서 보이는 것과 들리는 소리를 설명해 주세요"
    KNOCKOUT_RULES = [("generated", "video", 0, 36)]  # block generated→video, all 36 thinker layers
    MAX_NEW_TOKENS = 32
    ATTENTION_CAPTURE_LAYERS = (0, 2)  # heatmap rows; widening this costs VRAM per decode step

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
def _(DEVICE, MODEL_PATH, PROJECT_DIR):
    import csv
    from collections import Counter

    import matplotlib.pyplot as plt
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
            MODEL_PATH, torch_dtype="auto", attn_implementation=attn_implementation
        )
        # Free the talker + (float32) token2wav BEFORE moving to GPU so they never
        # occupy VRAM — this experiment only needs the thinker.
        _model.disable_talker()
        _model = _model.to(DEVICE)
        _model.eval()
        _proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        return _model, _proc

    # video_path/nframes are arguments, not closures: this cell must depend only
    # on the model constants, or a knob tweak would cascade into the loaders.
    def prepare_video_inputs(model, processor, prompt, token_mapping_fn, video_path, nframes):
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

    def cache_put(cache, key, value, keep=4):
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
    return LEDGER_LOG, append_run, apply_verdict, run_record


@app.cell
def _(RESULTS_DIR, SILENT_VIDEO_PATH, VIDEO_PATH):
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

    MAX_UPLOAD_BYTES = 250 * 2**20
    MAX_DURATION_S = 120.0
    MAX_PIXELS = 1920 * 1080
    MAX_FPS = 60.0
    # The three limits above are checked independently, so a clip sitting on all
    # of them (120 s x 1080p x 60 fps) still decodes to ~42 GiB -- and molab gives
    # 32 GB of RAM. The PyAV shim materialises every frame in a list, np.stack
    # copies it, and `.permute(...).contiguous()` copies again, so peak host RAM is
    # roughly 3x the decoded size. Bound the product, not just the factors.
    # 6 GiB decoded ~= 18 GiB peak, which clears molab's 32 GB with the models on
    # the GPU -- and still admits a 30 s 1080p30 or 60 s 720p30 phone clip.
    MAX_DECODED_BYTES = 6 * 2**30

    def preflight_clip(path):
        """디코딩해도 안전하면 `None`, 아니면 그 이유를 설명하는 한국어 문장."""
        import av

        try:
            with av.open(str(path)) as _c:
                if not _c.streams.video:
                    return "이 파일에는 비디오 스트림이 없습니다."
                _v = _c.streams.video[0]
                _dur = float(_c.duration / 1_000_000) if _c.duration else None
                _rate = _v.average_rate or _v.guessed_rate
                _fps = float(_rate) if _rate else None
                _px = int(_v.width or 0) * int(_v.height or 0)
                if _dur is not None and _dur > MAX_DURATION_S:
                    return (
                        f"길이가 {_dur:.0f}초입니다. 여기서의 상한은 "
                        f"{MAX_DURATION_S:.0f}초입니다. 잘라서 다시 시도하세요."
                    )
                if _px > MAX_PIXELS:
                    return (
                        f"해상도가 {_v.width}×{_v.height}입니다. 상한은 1920×1080입니다. "
                        "모든 프레임이 압축 해제된 상태로 메모리에 올라갑니다."
                    )
                if _fps is not None and _fps > MAX_FPS:
                    return f"{_fps:.0f} fps입니다. 상한은 {MAX_FPS:.0f} fps입니다."
                if _dur and _fps and _px:
                    _decoded = _dur * _fps * _px * 3
                    if _decoded > MAX_DECODED_BYTES:
                        return (
                            f"디코딩하면 약 {_decoded / 2**30:.1f} GiB가 됩니다 "
                            f"({_dur:.0f}초 × {_fps:.0f} fps × {_v.width}×{_v.height}). "
                            f"한도는 {MAX_DECODED_BYTES / 2**30:.0f} GiB입니다 — 길이를 "
                            "줄이거나 해상도를 낮춰서 다시 시도하세요."
                        )
                if not _c.streams.audio:
                    return (
                        "이 파일에는 오디오 트랙이 없습니다. 두 플레이그라운드 모두 `audio` "
                        "토큰 위치에서 측정하므로 채점할 것이 없습니다. "
                        "(소리 없는 대조군을 *원한다면* 무음 클립을 쓰세요. 그쪽은 디지털 "
                        "무음이지만 실제 오디오 트랙이 있습니다.)"
                    )
        except Exception as _e:  # noqa: BLE001 — a broken container is a message, not a crash
            return f"파일을 열 수 없습니다 ({type(_e).__name__}: {_e})."
        return None

    def resolve_clip(choice, uploads):
        """Turn a clip choice into `(path, is_control, error)`.

        The silent control lives in the repo, so in molab it exists only on the
        kernel side — there is nothing on the student's machine for a browser file
        picker to select. Choosing it by name and resolving server-side is what
        makes the one control in this lab that can fail actually reachable.
        """
        if choice == "Default clip":
            return VIDEO_PATH, False, None
        if choice == "Silent control":
            return SILENT_VIDEO_PATH, True, None
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
    # The reader. This one *does* reference the state, so it re-runs on every
    # append — which is what keeps the views and the worksheet export fresh.
    import re as _re

    from src.run_ledger import build_worksheet_md as worksheet_md
    from src.run_ledger import ledger_counts as _counts
    from src.run_ledger import render_ledger_html as _render_ledger

    # `render_ledger_html` lives in `src/`, which both notebooks import from
    # GitHub — so its chrome cannot be translated at the source without also
    # translating the English notebook. Rewrite it here instead, scoped to the
    # two places the renderer emits fixed English: the `<th>` row, and the
    # placeholder for a run with no verdict yet. A student's own prediction text
    # is interpolated into `<td>`s, never a `<th>`, so this cannot touch it.
    # `test_notebook_kr_replay.py` asserts every key below still matches; if the
    # renderer's vocabulary drifts, that test fails rather than this silently
    # leaving English on screen.
    LEDGER_HEADINGS = {
        "# / id": "# / id",
        "kind": "종류",
        "condition": "조건",
        "metric": "지표",
        "changed": "바뀐 것",
        "control": "대조군",
        "prediction": "가설",
        "verdict": "판정",
    }
    _UNRESOLVED_HTML = '<span style="opacity:0.55">unresolved</span>'

    # Verdict *values* stay English in the log (`run_ledger.VERDICTS`), so only the
    # display is swapped — and only inside the verdict cell, the one `<span>` that
    # carries a `title="rival: …"`. A student's own prediction sits in a
    # differently-titled span and can never be rewritten by this.
    LEDGER_VERDICTS = {"supported": "지지됨", "refuted": "반박됨", "untested": "미검증"}

    def _localize_ledger(html):
        def _th(match):
            return f"{match.group(1)}{LEDGER_HEADINGS.get(match.group(2), match.group(2))}</th>"

        html = _re.sub(r"(<th[^>]*>)([^<]*)</th>", _th, html)
        html = _re.sub(
            r'(<span title="rival:[^"]*">)(' + "|".join(LEDGER_VERDICTS) + r")</span>",
            lambda m: f"{m.group(1)}{LEDGER_VERDICTS[m.group(2)]}</span>",
            html,
        )
        # The summary line above the table. Its uncontrolled-claims clause is the
        # one number the ledger section tells students to drive to zero, so it is
        # the last thing that should still be reaching them in English.
        html = _re.sub(
            r"(\d+) run\(s\) · (\d+) in a family that has a control · (\d+) still unresolved",
            r"실행 \1건 · 대조군이 있는 계열의 실행 \2건 · 판정 없음 \3건",
            html,
        )
        html = _re.sub(r"(\d+) claim\(s\) with NO control", r"대조군 없는 주장 \1건", html)
        html = html.replace(
            "No runs logged yet. Write a prediction, press ▶, and the run "
            "lands here — with or without a control.",
            "아직 기록된 실행이 없습니다. 가설을 적고 ▶를 누르면 실행이 여기에 "
            "쌓입니다 — 대조군이 있든 없든.",
        )
        html = html.replace(">none written<", ">가설 없음<")
        # The control column's chip. `is_control` is a bool, so this cell is pure
        # display — unlike `kind` and `metric`, which are the keys a student uses
        # to match a row against `lab_log.jsonl` and stay English on purpose.
        html = html.replace(';font-weight:600">control</span>',
                            ';font-weight:600">대조군</span>')
        return html.replace(
            _UNRESOLVED_HTML, '<span style="opacity:0.55">판정 없음</span>'
        )

    def ledger_view(highlight=()):
        """The ledger, rendered wherever a result lands.

        Called from cheap display cells only. A cell calling this depends on this
        cell and therefore re-renders on every append — the point for a view, and
        exactly why no GPU cell and no form cell may call it.
        """
        _runs = get_runs()
        _c = _counts(_runs)
        _head = mo.md(
            f"###### 실습 기록 — 실행 {_c['n']}건 · "
            f"대조군 없음 **{_c['n_uncontrolled_claims']}건** · "
            f"판정 없음 {_c['n_unresolved']}건"
        )
        _html = _localize_ledger(_render_ledger(_runs, highlight_ids=highlight))
        return mo.vstack([_head, mo.Html(_html)], gap=0.3)

    return LEDGER_HEADINGS, LEDGER_VERDICTS, ledger_view, worksheet_md


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
            _ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
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

    디코딩된 토큰에 한자나 다른 언어 조각이 섞여 나오는 것도 정상입니다 — 이 클립에서는
    프로브 칸의 약 4분의 1이 그렇습니다. 모델이 언어를 넘나든 것이지 노트북이 깨진 것이
    아닙니다. 같은 이유로 위의 캡션에 `牛` 같은 한자가 끼어 있을 수 있습니다.

    **그리드 위를 드래그**하면 활성 레이어가 움직이고, 아래 칩 띠가 파이썬을 거치지
    않고 갱신됩니다. **열을 클릭**하면 그 위치가 고정되어, 36개 레이어를 지나는
    궤적 전체를 토큰으로 읽을 수 있습니다.

    > **여기서 눈여겨볼 것.** 이 섹션의 예전 버전은 "초기 레이어의 잡음이 최종
    > 예측으로 결정화되는 과정을 지켜보라"고 안내했습니다. 그 말을 믿기 전에, 마지막
    > 레이어들이 실제로 무엇으로 디코딩되는지 보세요. 프로브는 오디오 위치에서
    > **보정되어 있지 않습니다**. 그 퇴화(degeneracy) 자체가 이번 주의 결과이고,
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
    return probe_summary,


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
            f"###### 레이어별 집계 — 잡토큰이 가장 많은 레이어는 **{_worst['name']}**로, "
            f"잡토큰 칸 **{_worst['junk']}**개에 모든 오디오 위치를 통틀어 서로 다른 토큰이 "
            f"**{_worst['unique']}**개뿐입니다"
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
            _base.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
            _ko.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
    ## 🎚️ 인터랙티브: 어느 레이어 대역이 경로를 나르는가?

    위의 비교는 **단 하나의** 녹아웃입니다 — 모달리티 하나, 고정된 레이어 대역 하나
    (36개 레이어 전부). 전 구간을 막으면 그 경로가 *중요하다*는 것은 알 수 있어도
    *어디서* 쓰이는지는 알 수 없습니다. 이 섹션은 그 대역을 훑습니다. 타깃 모달리티와
    레이어 창을 고르고 다시 생성해서, 캡션이 기준선에서 얼마나 멀어지는지 보세요.

    같은 타깃에 대해 `[0, 12)` · `[12, 24)` · `[24, 36)`을 비교해 보세요.

    **널(null) 결과를 읽는 법.** 캡션이 그대로인 대역은 *이 측정에서는 효과가 없음*을
    보여 줄 뿐입니다. 중복(redundancy), 이 규칙이 자르지 못한 간접 경로, 또는 변화를
    보기에 너무 거친 지표 — 어느 쪽과도 모순되지 않습니다. 경로가 **없다는 증거는
    아닙니다**. 문자열 비교는 이분법적입니다. 아래의 티처 포싱 Δ가 같은 질문의 연속적
    버전이며, *작은* 효과를 보여 줄 수 있는 쪽입니다.

    ▶를 누를 때마다 이미 인코딩된 클립에 대해 greedy 생성 1회가 돌아갑니다(수 초).
    기준선은 재사용하며 다시 생성하지 않습니다. 두 캡션 모두 **답변만** 표시됩니다 —
    공통 프롬프트를 잘라 내야 차이가 지시문이 아니라 모델의 말에 대한 것이 됩니다.
    모든 ▶는 아래 **실습 기록(ledger)**에 추가되므로, 직전 대역이 화면에 남아
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
        mo.md("###### 이 인코딩된 입력에 실제로 들어 있는 것"),
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
    _band_targets = ["video", "audio", "image", "query_text"]
    _band_default = KNOCKOUT_RULES[0][1] if KNOCKOUT_RULES else "video"

    def _band_validate(_v):
        if not _v:
            return None
        if not (_v.get("prediction") or "").strip():
            return "실행하기 전에 가설을 적으세요 — 틀릴 수 있는 가설이어야 합니다."
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
        "**▶ 누르기 전 가설** — 이 대역이 경로를 나른다면 캡션에 무슨 일이 일어나야 "
        "하며, 무엇이 관찰되면 가설이 틀린 것입니까?\n\n"
        "{prediction}\n\n"
        "**generated** 토큰이 {target} 에 어텐션하는 것을 thinker 레이어 {layers} "
        "구간에서 금지\n\n"
        "{null_band} — 이 실행을 **무효과 대역(null band)**으로 표시합니다: 아무 일도 "
        "없으리라 예상하는 대역이라는 뜻입니다. 이것이 이 계열 실험의 대조군이며, "
        "기록의 *대조군 없음* 개수를 빨간색에서 벗어나게 하는 것입니다.\n\n"
        f"(`end`는 배타적입니다. 이 thinker는 레이어가 **{_band_layers}**개입니다. "
        "클립·프롬프트·프레임 수는 파라미터 셀에 설정된 값 그대로입니다.)"
    ).batch(
        prediction=mo.ui.text_area(
            placeholder="예: 설명이 앞쪽에서 조립되므로, [0,12)에서 video를 막으면 "
                        "[24,36)에서 막을 때보다 캡션이 더 많이 바뀔 것이다.",
            rows=2,
            full_width=True,
        ),
        target=mo.ui.dropdown(
            _band_targets,
            value=_band_default if _band_default in _band_targets else "video",
        ),
        layers=mo.ui.range_slider(
            0, _band_layers, step=1, value=[0, _band_layers // 3], show_value=True
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
    # The form's `validate=` runs only in the submit-button handler; marimo's
    # Ctrl/Cmd+Enter shortcut sets the value directly and skips it. The empty-band
    # check has a backstop (`rule_reach` raises inside `block_attention`), but the
    # prediction gate has none — and it is the one that makes hypothesis-before-▶
    # structurally unavoidable rather than merely suggested.
    mo.stop(
        not (_bp.get("prediction") or "").strip(),
        mo.callout(
            mo.md(
                "**가설을 먼저 적으세요** — 틀릴 수 있는 가설이어야 합니다. "
                "(Ctrl/Cmd+Enter는 폼 자체의 검사를 건너뛰므로, 실행이 여기서 "
                "멈춰 있습니다.)"
            ),
            kind="warn",
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
                    caption="단어 단위. 100% = 이 대역은 아무것도 바꾸지 않음",
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
                    config={
                        "clip": VIDEO_PATH.name, "nframes": NFRAMES,
                        "prompt": ATTENTION_PROMPT,
                        "target": _bp["target"], "start": _lo, "end": _hi,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    prediction=_bp.get("prediction", ""),
                    is_control=bool(_bp.get("null_band")),
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

    **쉽게 말해:** 모델이 방금 한 말을 그대로 되돌려 보여 주고 *"이 말을 얼마나
    확신하니?"* 라고 묻는 것입니다. 경로를 끊기 전과 후의 확신을 비교합니다.

    위의 문자열 비교는 **직관적이지만 이분법적**입니다. *작은* 효과는 보이지 않고,
    생성이 어떻게 이어지느냐에 따라 달라집니다. 이 셀은 같은 질문을 **측정**으로
    바꿉니다. 기준선 캡션을 `answer`로 태그해 다시 입력한 뒤, 그 답변이
    `KNOCKOUT_RULES`와 같은 타깃 모달리티로부터 차단됐을 때 **모델이 자기 말을
    얼마나 덜 믿게 되는지**를 토큰 단위로 점수화합니다(같은 클립, 같은 프롬프트, 같은
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
                mo.hstack([
                    mo.stat(
                        value=f"{_w9_total:+.2f}",
                        label="Σ Δ log-우도 (nats)",
                        caption="녹아웃 − 기준선 · 음수 = 덜 믿게 됨",
                        direction="decrease" if _w9_total < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{w9_tf_result['delta_mean']:+.3f}",
                        label="토큰당 Δ (nats)",
                        caption=(
                            "길이로 정규화한 값 — 실행끼리 비교할 때는 **이 숫자**를 보세요. "
                            "Σ는 캡션 길이에 비례합니다"
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
        **_w9_params(w9_tf_result["caption_tokens"], w9_tf_result["delta"]),
    ))
    mo.md(
        "###### 토큰별 Δ log-우도 (단어에 마우스를 올리면 그 토큰들의 nats가 보입니다)\n\n"
        f"{w9_threshold} 이상 잃은 단어만 표시합니다 — **밑줄 친 숫자를 옆으로 드래그**하거나 "
        "클릭해서 입력하세요. 임계값을 넘은 단어는 **굵게 테두리**가 생기고 나머지는 "
        "흐려지므로, 드래그하는 대로 띠가 다시 정렬되는 것이 보입니다. 여기서는 모델을 "
        "전혀 건드리지 않습니다."
    )
    return (w9_threshold,)


@app.cell
def _(mo, w9_threshold, w9_tf_result):
    from src.teacher_forcing import group_tokens_into_words as _w9_group
    from src.teacher_forcing import render_delta_strip as _w9_strip

    _delta = [float(_x) for _x in w9_tf_result["delta"].detach().cpu().float().tolist()]
    _th = abs(float(w9_threshold.value.get("amount", 0.0)))
    _words = _w9_group(w9_tf_result["caption_tokens"], _delta)
    _hit = [_w for _w in _words if _w[1] < -_th]
    _share = (
        100.0 * sum(_w[1] for _w in _hit) / w9_tf_result["delta_total"]
        if w9_tf_result["delta_total"]
        else 0.0
    )
    mo.vstack([
        mo.Html(
            "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
            + _w9_strip(w9_tf_result["caption_tokens"], _delta, highlight_below=_th)
            + "</div>"
        ),
        mo.md(
            f"**{len(_hit)}/{len(_words)}** 단어가 −{_th:.2f} nats보다 크게 떨어졌습니다 — 합쳐서 "
            f"Δ = {sum(_w[1] for _w in _hit):+.2f} nats이며, 전체 "
            f"{w9_tf_result['delta_total']:+.2f} 중 **{_share:.0f}%**입니다."
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
def _(knockout_text, logit_csv_written, mo):
    _ = knockout_text  # depend on the knockout run
    _ok = (
        logit_csv_written is not None
        and logit_csv_written.is_file()
        and logit_csv_written.stat().st_size > 0
    )
    mo.md(
        f"### 고정 실행 완료\n\n"
        + (
            f"- Logit-lens CSV 기록: **완료** — `{logit_csv_written}`\n"
            if _ok
            else "- Logit-lens CSV: **기록되지 않음** — 이 실행은 오디오 토큰 행을 "
                 "만들지 못했습니다(오디오 트랙이 없는 클립일 가능성이 큽니다).\n"
        )
        + "- 기준선 vs 녹아웃 비교 완료, 두 어텐션 패널 모두 표시했습니다.\n\n"
        "**여기까지는 시범 예제입니다 — 설정 하나를 대신 돌려 드린 것입니다.** 아래 두 "
        "플레이그라운드가 여러분이 직접 돌리는 곳입니다. ▶를 누를 때마다 먼저 가설을 "
        "요구하고 결과를 실습 기록에 추가하므로, 직전 실행이 화면에 남아 비교할 수 "
        "있습니다. 🎯 섹션의 무음 클립 대조군부터 시작하세요."
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
    녹아웃 실험에서 쓰던 모델을 그대로 재사용합니다 — 그래서 빠르고 추가 메모리가
    필요 없습니다.

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
):
    _n_layers = len(attention_model.thinker.model.layers)
    _modalities = ["audio", "video", "query_text", "image", "generated"]
    # Scoreboard-appropriate defaults: the source must be a modality that is
    # actually PRESENT in the prompt, so `audio` (the positions being scored) —
    # not the params cell's `generated`, which is inert in a forward pass. The
    # target follows the params rule; the window spans every layer ([0, N)).
    _def_source = "audio"
    _def_target = KNOCKOUT_RULES[0][1] if KNOCKOUT_RULES else "video"

    _hint = (
        f"source/target ∈ `audio · video · query_text · image · generated` — 다만 "
        f"`generated`는 규칙의 *어느 쪽*에 놓이든 여기서는 **작동하지 않고**(순전파 "
        f"중에는 생성 위치가 존재하지 않습니다), `image`는 영상 클립에서 0 토큰입니다. "
        f"아무것에도 걸리지 않는 규칙은 실행되지 않고 거부됩니다. 레이어 `end`는 "
        f"배타적입니다. 이 thinker는 레이어가 **{_n_layers}**개이므로 "
        f"`[0, {_n_layers})`가 전체를 뜻합니다."
    )
    _template = (
        "**▶ 누르기 전 가설** — 어느 레이어에서 다양성이 줄어야 하며, 그 이유는?\n\n"
        "{prediction}\n\n"
        "**클립** {clip} &nbsp; (무음 대조군은 저장소 안에 있습니다 — 이름으로 고르면 "
        "되고, 업로드할 것이 없습니다)\n\n"
        "**업로드**를 골랐을 때만 — `mp4 / mov / mkv / webm`, 250 MB 이하, 120초 이하, 1080p 이하:\n\n"
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
        if not (_v.get("prediction") or "").strip():
            return "실행하기 전에 가설을 적으세요 — 틀릴 수 있는 가설이어야 합니다."
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
        prediction=mo.ui.text_area(
            placeholder="예: 두 스트림이 융합되는 중간 레이어에서 audio→video 차단이 "
                        "다양성을 가장 크게 떨어뜨릴 것이다.",
            rows=2,
            full_width=True,
        ),
        clip=mo.ui.radio(CLIP_CHOICES, value=CLIP_DEFAULT, inline=True),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            multiple=False,
            kind="area",
        ),
        nframes=mo.ui.slider(
            2, 32, step=2, value=NFRAMES, show_value=True, include_input=True
        ),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        ko_enable=mo.ui.checkbox(value=bool(KNOCKOUT_RULES)),
        ko_source=mo.ui.dropdown(_modalities, value=_def_source),
        ko_target=mo.ui.dropdown(_modalities, value=_def_target),
        ko_layers=mo.ui.range_slider(
            0, _n_layers, step=1, value=[0, _n_layers], show_value=True
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
    LOGIT_PROMPT,
    USE_PRECOMPUTED,
    analyze_and_save_audio_logits_to_csv,
    append_run,
    attention_model,
    attention_processor,
    block_attention,
    cache_put,
    clear_logit_lens_hooks,
    create_attention_token_mapping,
    csv,
    ko_controls,
    mo,
    np,
    playground_caches,
    plt,
    register_logit_lens_hooks,
    resolve_clip,
    run_record,
    set_runs,
    torch,
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
    # Backstop for the prediction gate: `validate=` is skipped by marimo's
    # Ctrl/Cmd+Enter shortcut. See the band-sweep cell above.
    mo.stop(
        not (_p.get("prediction") or "").strip(),
        mo.callout(
            mo.md("**가설을 먼저 적으세요** — 틀릴 수 있는 가설이어야 합니다."),
            kind="warn",
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
    _prompt = _p["prompt"].strip() or LOGIT_PROMPT

    # Build the knockout rules. The advanced text field (several `src,tgt,start,end`
    # rules separated by `;`) overrides the single-rule builder when it is filled.
    _modalities = ["audio", "video", "query_text", "image", "generated"]
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
        _key = (video_path.name, video_path.stat().st_size, nframes, prompt)
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
            mo.md("###### 디코딩된 예측의 다양성 순으로 정렬한 레이어 (클수록 오디오 토큰 예측이 더 다양함)"),
            _table,
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
                    config={
                        "clip": _video_path.name, "nframes": _nframes, "prompt": _prompt,
                        "rules": [list(r) for r in _rules], "compare": _compare,
                    },
                    prediction=_p.get("prediction", ""),
                    is_control=_is_control,
                    extra={"audio_tokens": _n_audio, "peak_layer": _peak},
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

    위의 다양성 스코어보드는 **프롬프트**에 대해 순전파를 1회 돌리므로, `generated`와
    똑같이 **`answer`** source도 거기서는 작동하지 않습니다(막을 answer 토큰이 아예
    없습니다). 이 섹션이 그 빈틈을 메웁니다. 캡션을 한 번 생성하고, **`answer`**로
    태그해 다시 입력한 뒤, 그 답변이 어떤 모달리티에 어텐션하지 못하게 됐을 때
    **모델이 자기 말을 얼마나 덜 믿게 되는지**를 측정합니다.

    지표는 **Δ log-우도 = `녹아웃 − 기준선`**입니다. *음수*는 녹아웃 뒤 모델이 자기
    캡션을 **덜** 믿게 됐다는 뜻, 즉 그 경로가 캡션을 떠받치고 있었다는 뜻입니다.
    위 🎚️ 섹션의 자유 생성 문자열 비교와 달리 **연속적**이고(작은 효과도 보입니다)
    **결정론적**입니다(greedy 캡션, 순전파만으로 채점). ▶를 누르기 전에는 아무것도
    실행되지 않습니다.

    > **대조군을 먼저 돌리세요.** **클립**을 `무음 대조군`으로 놓고 `answer → audio`를
    > 실행하세요. 프레임은 같고 오디오 트랙만 디지털 무음인 클립입니다. 오디오 토큰은
    > 존재하지만 신호가 없으므로 **토큰당 Δ는 ≈ 0이어야 합니다**. 그다음 같은
    > 프롬프트·레이어로 `기본 클립`으로 바꾸세요. 진짜 오디오 의존성이 있다면 토큰당
    > Δ가 뚜렷하게 더 큰 음수로 나타납니다.
    >
    > Σ가 아니라 **토큰당 Δ**를 비교하세요. 두 클립은 길이가 다른 서로 다른 캡션을
    > 만들고 Σ는 길이에 비례하므로, 합계를 비교하는 것은 의미가 없습니다. 판정
    > 기준은 `|Δ_무음| / 토큰 ≪ |Δ_오디오| / 토큰`입니다.
    >
    > **숫자만 보지 말고 캡션도 비교하세요.** 결과 바로 아래 토큰 띠에 캡션이 그대로
    > 나옵니다. 소리를 완전히 지웠는데도 답이 거의 같다면, 모델은 사운드트랙을 읽은
    > 적이 없다는 뜻입니다.
    >
    > **실패할 수 있는** 대조군이라는 점이 핵심이며, 두 실행 모두 기록에 남으므로 두
    > 숫자를 나란히 놓고 볼 수 있습니다.

    ### 이 노트북의 기본 프롬프트에서는 판정 기준이 충족되지 않습니다

    미리 밝혀 둡니다. 위 절차를 그대로 따르면 **두 클립 모두 토큰당 Δ가 0 근처**로
    나옵니다. 고장이 아니라 **그것이 결과입니다.**

    캡션을 비교해 보면 이유가 바로 보입니다. 오디오 트랙을 완전히 지워도 답이 거의
    그대로입니다. 모델은 화면에서 본 것을 근거로 소리를 **추측**했을 뿐, 사운드트랙을
    읽지 않았습니다. 자를 오디오 경로가 애초에 없으니 잘라도 Δ가 움직이지 않습니다 —
    그러므로 **Δ ≈ 0은 측정 실패가 아니라 정확한 측정**입니다.

    이것이 이 수업의 질문 그 자체입니다: *오디오-비주얼 LLM은 정말로 듣는가?* 이
    프롬프트·이 클립·이 모델에서 답은 **아니오**이고, 그 사실을 밝혀낸 것은 그럴듯해
    보이는 캡션이 아니라 **무음 대조군**입니다. 캡션만 읽었다면 "소리를 잘 묘사하네"
    하고 넘어갔을 것입니다.

    > **직접 반증해 보세요.** 판정 기준을 만족시키는 설정이 존재할까요? 프롬프트 칸은
    > 편집할 수 있습니다. 출발점 하나: 같은 클립·같은 모델에 **영어로** 물으면 결과가
    > 달라집니다. `Describe what you hear in the video`를 넣고 두 클립을 다시 돌린 뒤,
    > 캡션과 토큰당 Δ가 어떻게 변하는지 기록에 남기세요.
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
):
    _n_layers = len(attention_model.thinker.model.layers)
    _tf_targets = ["audio", "video", "query_text", "image"]
    _tf_template = (
        "**▶ 누르기 전 가설** — 예상하는 토큰당 Δ 값과, 무엇이 관찰되면 그 가설이 "
        "반박되는지 적으세요:\n\n"
        "{prediction}\n\n"
        "**클립** {clip} &nbsp; (반증 가능한 쪽은 `무음 대조군`입니다 — 저장소 안에 "
        "있으니 업로드하지 말고 이름으로 고르세요)\n\n"
        "**업로드**를 골랐을 때만 — `mp4 / mov / mkv / webm`, 250 MB 이하, 120초 이하, 1080p 이하:\n\n"
        "{video}\n\n"
        "**클립에서 샘플링할 프레임 수** {nframes}\n\n"
        "**프롬프트** {prompt}\n\n"
        "---\n\n"
        "**answer**가 {target} 에 어텐션하는 것을 thinker 레이어 {layers} 구간에서 금지\n\n"
        f"(`answer`는 모델 자신의 캡션을 티처 포싱으로 다시 넣은 것입니다. 이 thinker는 "
        f"레이어가 **{_n_layers}**개이고 `end`는 배타적입니다.)"
    )

    def _tf_validate(_v):
        if not _v:
            return None
        if not (_v.get("prediction") or "").strip():
            return "실행하기 전에 가설을 적으세요 — 틀릴 수 있는 가설이어야 합니다."
        # Label *or* value: see the note in the 🎛️ form's validator.
        if _v.get("clip") in ("Upload", CLIP_UPLOAD) and not _v.get("video"):
            return "업로드를 선택했지만 파일을 고르지 않았습니다."
        # `.get` with a default: the batch value is partial on first render.
        _lo, _hi = _v.get("layers") or (0, 1)
        if int(_hi) <= int(_lo):
            return f"[{int(_lo)}, {int(_hi)})는 0개 레이어를 마스킹합니다 — `end`는 배타적입니다."
        return None

    tf_controls = mo.md(_tf_template).batch(
        prediction=mo.ui.text_area(
            placeholder="예: 무음 클립에서 토큰당 Δ는 ±0.02 nats 이내일 것이고, "
                        "실제 클립에서는 최소 5배 더 큰 음수일 것이다.",
            rows=2,
            full_width=True,
        ),
        clip=mo.ui.radio(CLIP_CHOICES, value=CLIP_DEFAULT, inline=True),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"], multiple=False, kind="area"
        ),
        nframes=mo.ui.slider(2, 32, step=2, value=NFRAMES, show_value=True, include_input=True),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        target=mo.ui.dropdown(_tf_targets, value="audio"),
        layers=mo.ui.range_slider(0, _n_layers, step=1, value=[0, _n_layers], show_value=True),
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
    LOGIT_PROMPT,
    MAX_NEW_TOKENS,
    USE_PRECOMPUTED,
    append_run,
    attention_model,
    attention_processor,
    cache_put,
    create_attention_token_mapping,
    mo,
    np,
    playground_caches,
    resolve_clip,
    run_record,
    set_runs,
    tf_controls,
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
    # Backstop for the prediction gate: `validate=` is skipped by marimo's
    # Ctrl/Cmd+Enter shortcut. See the band-sweep cell above.
    mo.stop(
        not (_tp.get("prediction") or "").strip(),
        mo.callout(
            mo.md("**가설을 먼저 적으세요** — 틀릴 수 있는 가설이어야 합니다."),
            kind="warn",
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
    _tf_prompt = _tp["prompt"].strip() or LOGIT_PROMPT
    _tf_lo, _tf_hi = int(_tp["layers"][0]), int(_tp["layers"][1])
    _tf_rules = [("answer", _tp["target"], _tf_lo, _tf_hi)]

    def _tf_prep(video_path, nframes, prompt):
        # Shared encode cache with the 🎛️ section: a layer-band or target sweep
        # on the same clip/prompt re-encodes nothing after the first ▶.
        _key = (video_path.name, video_path.stat().st_size, nframes, prompt)
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
        _inp = {k: v.to(attention_model.device) for k, v in _inp.items()}
        _types = create_attention_token_mapping(
            _inp["input_ids"], attention_model.config.thinker_config
        )
        return cache_put(playground_caches["encode"], _key, (_inp, _types))

    tf_result = None
    _tf_out = None
    try:
        # Caption cache (the F1 spec's "cached keyed on (clip, prompt, nframes)"):
        # the greedy caption depends only on the encoded inputs, so a rule/layer
        # sweep reuses C instead of regenerating it every submit.
        _tf_cap_key = (_tf_video.name, _tf_video.stat().st_size, _tf_nframes, _tf_prompt)
        _tf_cached_c = playground_caches["caption"].get(_tf_cap_key)
        with mo.status.spinner(
            title=f"티처 포싱 · {_tf_nframes} 프레임 · {_tf_video.name}"
            + (" · 캡션 캐시 사용…" if _tf_cached_c is not None else "…")
        ):
            _tf_inp, _tf_types = _tf_prep(_tf_video, _tf_nframes, _tf_prompt)
            _tf_res = _tfd(
                attention_model, attention_processor, _tf_inp, _tf_types, _tf_rules,
                # Without this the playground scored 32-token captions (the
                # function's own default) while the fixed cell above used
                # MAX_NEW_TOKENS — two different caption lengths, one Σ column.
                max_new_tokens=MAX_NEW_TOKENS,
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
                caption="**실행끼리 비교할 때는 이 값** — Σ는 캡션 길이에 비례합니다",
                direction="decrease" if _tf_mean < 0 else "increase",
                bordered=True,
            ),
            mo.stat(
                value=f"{_tf_total:+.2f}",
                label="Σ Δ log-우도 (nats)",
                caption="녹아웃 − 기준선 · 음수 = 덜 믿게 됨",
                direction="decrease" if _tf_total < 0 else "increase",
                bordered=True,
            ),
            mo.stat(
                value=(_tf_toks[_tf_worst].strip() or "·") if _tf_toks else "—",
                label="가장 영향을 받은 토큰",
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
            mo.hstack(_tf_stats, widths="equal", gap=1),
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
                    metric_value=round(_tf_mean, 4),
                    metric_unit="nats/token",
                    config={
                        "clip": _tf_video.name, "nframes": _tf_nframes,
                        "prompt": _tf_prompt, "target": _tp["target"],
                        "start": _tf_lo, "end": _tf_hi,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    prediction=_tp.get("prediction", ""),
                    is_control=_tf_is_control,
                    extra={"delta_total": round(_tf_total, 4), "n_tokens": len(_tf_toks)},
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
        **_tf_params(tf_result["caption_tokens"], tf_result["delta"]),
    ))
    mo.md(
        "###### 토큰별 Δ log-우도 (뜨거운 색 = 녹아웃 뒤 덜 믿게 됨. 단어에 마우스를 "
        "올리면 그 토큰들의 nats가 보입니다)\n\n"
        f"{tf_threshold} 이상 잃은 단어만 표시합니다 — **밑줄 친 숫자를 옆으로 드래그**하거나 "
        "클릭해서 입력하세요. 다시 그려지는 것은 이 띠뿐이며 모델은 건드리지 않습니다."
    )
    return (tf_threshold,)


@app.cell
def _(mo, tf_result, tf_threshold):
    from src.teacher_forcing import group_tokens_into_words as _tf_group
    from src.teacher_forcing import render_delta_strip as _tf_strip

    _delta = [float(_x) for _x in tf_result["delta"].detach().cpu().float().tolist()]
    _toks = tf_result["caption_tokens"]
    _th = abs(float(tf_threshold.value.get("amount", 0.0)))
    _words = _tf_group(_toks, _delta)
    _hit = [_w for _w in _words if _w[1] < -_th]
    _share = (
        100.0 * sum(_w[1] for _w in _hit) / tf_result["delta_total"]
        if tf_result["delta_total"]
        else 0.0
    )
    _rows = [
        {"위치": _i, "토큰": _t, "Δ log-우도": round(_d, 3)}
        for _i, (_t, _d) in enumerate(zip(_toks, _delta))
    ]
    mo.vstack([
        mo.Html(
            "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
            + _tf_strip(_toks, _delta, highlight_below=_th)
            + "</div>"
        ),
        mo.md(
            f"**{len(_hit)}/{len(_words)}** 단어가 −{_th:.2f} nats보다 크게 떨어졌습니다 — 합쳐서 "
            f"Δ = {sum(_w[1] for _w in _hit):+.2f} nats이며, 전체 "
            f"{tf_result['delta_total']:+.2f} 중 **{_share:.0f}%**입니다."
        ),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=16),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📓 실습 기록 — 실제로 무엇을 돌렸는가

    이 노트북에서 누른 모든 ▶가, 미리 적어 둔 가설과 같은 종류의 직전 실행 대비
    바꾼 설정, 그리고 나온 측정값과 함께 아래에 기록됩니다. 지표는 **이름과 단위**를
    유지합니다. 캡션 유사도와 토큰당 Δ는 같은 양이 아니므로 절대 한 열에 섞이지
    않습니다.

    표 위 머리글에서 지켜볼 숫자가 둘 있습니다. **대조군 없음**은 지금으로서는
    방어할 수 없는 주장의 개수입니다 — 각 실험을 무음 클립이나 아무 효과도 없으리라
    예상하는 레이어 대역과 짝지어 0으로 만드세요. **판정 없음**은 아직 *지지됨* ·
    *반박됨* · *미검증* 중 어느 것도 말하지 않은 실행의 개수입니다.

    이 기록은 폼을 초기화해도 남고, 모든 실행 **과 판정**이 그때그때
    `notebook_results/lab_log.jsonl`에 추가되므로 molab 커널이 재시작돼도
    살아남습니다 — 노트북이 시작할 때 표는 그 파일에서 다시 읽어 들입니다.

    다만 **molab 세션 자체가 종료되면 그 파일도 사라집니다**(90분 유휴 또는 12시간
    경과 시 자동 종료). 자리를 뜨기 전에 아래 **워크시트 내려받기**를 눌러 두세요.
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
        "실행 {run_id}은(는) {verdict}. 같은 숫자를 낳을 수 있는 경쟁 설명: {rival}"
    ).batch(
        run_id=mo.ui.text(placeholder="예: 3f9a1c02"),
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
def _(LEDGER_LOG, apply_verdict, mo, set_runs, verdict_form):
    # Sets but never reads: referencing `get_runs` here would re-run this cell on
    # every append and re-apply the last verdict.
    _v = verdict_form.value
    mo.stop(_v is None or not (_v.get("run_id") or "").strip())
    set_runs(
        lambda _prev, _id=_v["run_id"].strip(), _k=_v.get("verdict", "untested"),
        _r=_v.get("rival", ""): apply_verdict(
            _prev, _id, _k, rival=_r, log_path=LEDGER_LOG
        )
    )
    # Deliberately *not* `kind="success"`. This cell cannot read `get_runs`
    # (that would make it re-run on every append), so it cannot know whether the
    # id matched — and `apply_verdict` is a no-op on an unknown one. Claiming
    # success here would be a confident, wrong report in the one notebook whose
    # whole subject is that those are the enemy.
    mo.callout(
        mo.md(
            f"실행 `{_v['run_id'].strip()}`에 **{_v.get('verdict', 'untested')}**을(를) "
            "보냈습니다. 아무것에도 맞지 않는 id는 무시됩니다 — 아래 **판정** 열이 "
            "실제로 바뀌었는지 확인하세요."
        ),
        kind="info",
    )
    return


@app.cell
def _(ledger_view):
    ledger_view()
    return


@app.cell
def _(RESULTS_DIR, get_runs, mo, worksheet_md):
    # References `get_runs`, which is exactly what keeps the download current: the
    # cell re-runs on every append, so the file offered is never a stale snapshot.
    _runs = get_runs()
    _md = worksheet_md(_runs)
    mo.vstack([
        mo.md("###### `WORKSHEET.md`용 내보내기"),
        mo.download(
            _md.encode(),
            filename="lab_log.md",
            mimetype="text/markdown",
            label=f"⬇ 실행 {len(_runs)}건을 워크시트 표로 내려받기",
        ),
        mo.accordion({
            "마크다운 보기 (선택해서 복사)": mo.md(f"```markdown\n{_md}\n```")
        }),
        mo.md(
            f"_같은 내용이 `{RESULTS_DIR / 'lab_log.jsonl'}`에도 실시간으로 기록되고 있습니다._"
        ),
    ], gap=0.4)
    return


if __name__ == "__main__":
    app.run()
