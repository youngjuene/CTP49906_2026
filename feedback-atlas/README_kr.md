# 실시간 피드백 지도

<p align="center">
  <a href="README.md">English</a> · <b>한국어</b>
</p>

CTP49906 *Multimodal AI* 워크숍용 강의실 도구입니다. 강사가 서버에서 피드백과 학생들의
LLM 대화를 등록하면, 링크를 가진 사람은 누구나 로그인 없이 휴대폰이나 프로젝터로 지도를
엽니다. 각 의견은 외부 API 호출 없이 **로컬에서** 임베딩되어 2D로 투영되고, 등록되는 즉시
접속한 모든 화면에 전달됩니다.

첫 화면(`/`, `/chats.html`과 같음)은 [대화 경로 지도](#대화-경로-지도-chatshtml)이고, 의견 지도는
`/map`, 관리자 모드는 `/admin`에 있습니다.

색은 **어느 학생에 대한 의견인지**, 모양은 **AI인지 사람인지**를 나타냅니다. 두 번째
채널이 이 도구의 핵심입니다. 이 세션은 모델이 주관적 경험을 어디까지 서술할 수 있는지를
묻고, 이 지도는 두 종류의 코멘트를 같은 의미 공간에 놓아 둘이 같은 자리에 떨어지는지를
눈으로 확인하게 합니다.

하나의 코퍼스 위에 화면이 둘입니다. 휴대폰과 프로젝터를 위해 손으로 쓴 지도, 그리고 표와
연동 차트와 교차 필터를 위한 Apple의 Embedding Atlas 뷰어. 둘은 같은 행을 읽습니다.

명세는 [`realtime-feedback-atlas-prd.md`](realtime-feedback-atlas-prd.md)입니다.

## 이것이 아닌 것

의견 제출 양식이 아닙니다. 참가자 화면은 익명이고 읽기 전용입니다. 개인 코드도 세션도
없고, 브라우저가 보내는 어떤 메시지로도 의견을 추가할 수 없습니다. 의견은 서버에서
[일괄 등록](#ai-의견-일괄-등록-prd-43)으로만, 대화는 [대화 경로 지도](#대화-경로-지도-chatshtml)로만
들어옵니다.

누가 무엇을 썼는지를 공개하는 화면도 아닙니다. `reviewer_id`는 서버에 저장되지만 참가자
연결로는 절대 전달되지 않습니다. 관리자 코드 뒤의 관리자 모드만 그것을 보여줍니다. 대상
학생 이름(작업이 논의되는 학생)과 대화 경로 지도의 모든 내용은 링크를 가진 누구에게나
보입니다.

## 설치

[uv](https://docs.astral.sh/uv/)와 **Python 3.12 이상**이 필요합니다. 취향이 아니라
필요 조건입니다. numpy 2.5와 scipy 1.18이 모두 3.12를 요구하므로, 3.11 가상환경은 조용히
구버전으로 해석됩니다.

```bash
cd feedback-atlas
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 첫 실행 전에 Gemma 라이선스에 동의하세요

기본 모델 `google/embeddinggemma-300m`은 **게이트가 걸린** 다운로드입니다. 새로 클론한
저장소는 설치 시점이 아니라 모델을 불러오는 시점에 401 또는 403으로 실패합니다. 그러니
수업 당일이 아니라 미리 해두세요.

1. <https://huggingface.co/google/embeddinggemma-300m>에서 라이선스에 동의합니다.
2. **동의한 그 계정으로** 로그인합니다. `huggingface-cli login` 또는 `HF_TOKEN` 설정.
   라이선스에 동의한 계정과 다른 계정의 토큰은 토큰이 아예 없는 것과 똑같이 실패합니다.
3. 첫 실행이 1.2 GB 다운로드가 되지 않도록 가중치를 미리 받아둡니다.
   `hf download google/embeddinggemma-300m`

> **라이선스에 동의했는데도 계속 "gated"라고 나오면 `HF_TOKEN`을 확인하세요.**
> 환경 변수가 `huggingface-cli login`이 저장해 둔 토큰보다 우선하므로, 만료되었거나
> 잘못된 `HF_TOKEN`이 멀쩡한 로그인을 가려버립니다. 그리고 허브는 이를 gated 오류로
> 보고하기 때문에, 이미 모든 것이 정상으로 보이는 라이선스 페이지로 다시 보내집니다.
> `env -u HF_TOKEN python -m uvicorn …`으로 실행해 보면 둘을 가장 빨리 구분할 수
> 있습니다. 이 저장소의 첫 실행에서 실제로 시간을 잡아먹은 문제입니다.

게이트가 걸림돌이라면 `ATLAS_EMBEDDING_MODEL=e5-small-ko`가 있습니다. 게이트 없음,
Apache-2.0, 384차원입니다. 서버는 스스로 다른 모델로 넘어가지 않고 읽을 수 있는 오류와
함께 시작을 거부합니다. 아무도 고르지 않은 모델이 만든 지도는 지도가 없는 것보다 나쁩니다.

### 참가자 명단

수업 전에 등록하는 `roster.csv`입니다. [`roster.example.csv`](roster.example.csv) 참고.

```csv
id,display_name,role
kim.seoyeon,김서연,student
kang.minsu,강민수,observer
ta.youngjun,영준,ta
```

역할은 세 가지입니다. `student`(수강생) 행은 **대상**이기도 합니다. 즉 피드백을 받을 수
있는 프로젝트이고, 대화를 등록할 수 있는 사람입니다. `observer`(청강생)와 `ta`(조교)는
등록한 의견의 작성자로 기록될 수 있지만 대상이 되지는 않습니다. 대조는 공백·대소문자·전각
문자에 관대하고(한국어 IME가 전각 모드면 `ｋｉｍ`이 입력됩니다) 그 외에는 관대하지
않습니다. 편집 거리 기반 매칭은 한 학생의 ID가 다른 학생의 ID로 해석되게 만듭니다.

서버가 가진 사람 목록은 명단 하나뿐이고, 사람별로 만들 자격 증명은 없습니다. 나중에
명단을 고치면 `SIGHUP`을 보냅니다. 재시작도, 연결 끊김도 없습니다.

```bash
kill -HUP $(pgrep -f 'uvicorn.*src.server')
```

### 환경 설정

[`.env.example`](.env.example)을 `.env`로 복사하세요. `ATLAS_ADMIN_CODE`와
`ATLAS_ROSTER`에는 기본값이 없습니다. 기본값이 있는 관리자 코드는 이미 공개된 관리자
코드입니다. `ATLAS_CHATS`(기본값 `.run/chats/atlas.json`)는 대화 지도를 읽어 오는
위치입니다.

이전 관리자 코드가 평문 HTTP를 지난 적이 있다면 수업 전에 회전하세요.

```bash
python scripts/configure_security.py
```

새 관리자 코드는 비공개 권한으로 `.run/admin-code.txt`에 저장됩니다. 그 내용을 문서나
슬라이드에 넣지 마세요.

## 실행

강의실 배포 주소는 고정 숫자 HTTPS 주소입니다.

```text
https://143.248.249.7:8888
```

`143.248.249.7`용으로 이미 발급받은 신뢰된 IP 인증서를 사용하며, 인증서는
`.run/tls/live/feedback-atlas-ip/` 아래에 있습니다. 일반 수업 경로에는 도메인도
Cloudflare 터널도 필요 없습니다.

사용자 systemd 서비스로 HTTPS 서버를 실행합니다.

```bash
systemctl --user link "$(pwd)/deploy/feedback-atlas.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-renew.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-renew.timer"
systemctl --user daemon-reload
systemctl --user enable --now feedback-atlas.service feedback-atlas-renew.timer
loginctl enable-linger "$USER"
```

`scripts/serve.sh`는 uvicorn을 TLS로 직접 실행하고, 프록시 헤더를 끄고,
`ATLAS_ALLOW_INSECURE_HTTP=0`을 강제하며, Hugging Face 캐시는 오프라인으로만 사용합니다.
갱신 타이머는 하루 두 번 인증서를 확인하고 인증서 파일이 바뀐 경우에만 서비스를 다시
시작합니다.

로컬 개발과 테스트에서만 `ATLAS_ALLOW_INSECURE_HTTP=1`로 평문 HTTP를 허용할 수 있습니다.
이때도 loopback에만 바인딩하세요.

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
export ATLAS_ALLOW_INSECURE_HTTP=1
python -m uvicorn --factory src.server:create_app \
  --host 127.0.0.1 --port 8888
```

서버 진입점은 안전하지 않은 공개 HTTP·프록시 신뢰 fallback을 거부하므로, 개발용 평문
명령을 `0.0.0.0`에 바인딩하지 마세요.

**5분 일찍** 켜세요. 첫 실행은 모델을 불러오고 UMAP의 numba 커널을 예열합니다. 수업 시작에
맞춰 켜면 그 비용이 그 학기 첫 의견 등록에 그대로 얹힙니다.

`GET /healthz`는 모델 캐시 키, 누적 의견 수, 사용 중인 투영 방식, 접속자 수를 보고합니다.
관리자 코드는 절대 보고하지 않습니다.

### 강의실 접속

공유할 주소는 HTTPS IP 주소 하나입니다.

```text
https://143.248.249.7:8888
```

페이지, 웹소켓, Embedding Atlas 질의 엔드포인트가 모두 같은 origin에 있고, 어느 것도
코드를 묻지 않습니다. 서버는 강의실 IP 대역으로 제한되어 있지 않으므로 링크를 가진 사람은
누구나 지도와 대화 지도를 읽을 수 있습니다. 보호되는 것은 관리자 코드 뒤의 작성자
정보입니다.

**SSH 터널은 수업용 경로가 아닙니다.** `ssh -L 8888:127.0.0.1:8888 …`은 명령을 입력한
*그 노트북 한 대에만* 포트를 전달합니다. 강사가 SSH 너머로 지도를 미리 보는 데는 맞는
방법이고, 학생 휴대폰 30대에는 아무 역할도 하지 않습니다.

Cloudflare는 권장 실행 경로 밖입니다. 나중에 수동으로 다른 터널을 쓴다면 같은 관리자 코드,
HTTPS 브라우저 origin, 웹소켓 업그레이드, origin 검사를 유지하세요.

### 수업 당일

```bash
scripts/class.sh start     # 관리되는 HTTPS 서비스와 갱신 타이머 시작, 주소와 QR 출력
scripts/class.sh url       # https://143.248.249.7:8888 출력 및 .run/qr.png 갱신
scripts/class.sh status    # 서비스 상태와 /healthz 확인
scripts/class.sh stop
curl --noproxy '*' --resolve 143.248.249.7:8888:127.0.0.1 https://143.248.249.7:8888/healthz
journalctl --user -u feedback-atlas.service -f
```

5분 전에 실행하세요. 첫 실행은 모델을 불러오고 UMAP의 numba 커널을 예열합니다. 수업 시작에
맞춰 켜면 그 비용이 그 학기 첫 의견 등록에 그대로 얹힙니다.

운영 절차는 [`SECURITY_DEPLOYMENT.md`](SECURITY_DEPLOYMENT.md)에 정리되어 있습니다.

## 관리자 모드

`/admin`을 열고 관리자 코드를 입력합니다. 코드는 UI 플래그가 아니라 **다른 웹소켓**을
엽니다. 두 엔드포인트가 서로 다른 직렬화를 쓰기 때문에, 참가자 연결이 관리자 연결이 되는
경로는 존재하지 않습니다.

관리자 전용 도구 네 가지입니다.

| | |
|---|---|
| **검색** | 의견 본문 부분 일치 검색. 일치 부분을 강조하며, 주차 필터·대상 강조·작성자 강조와 함께 겹쳐서 동작합니다 |
| **선택** | 지도 위 사각 또는 올가미 선택. 고른 의견이 읽을 수 있는 목록으로 나옵니다. 군집을 한 점씩 마우스로 짚는 대신 **이야기할 수 있게** 만드는 기능입니다 |
| **가까운 의견** | 점을 누르면 코사인 거리 기준 가장 가까운 8건. 이 워크숍의 질문을 점 하나 단위로 묻는 것입니다. 사람이 쓴 이 문장에 가장 가까이 떨어진 모델의 문장은 무엇인가? |
| **CSV 내보내기** | 전체 또는 현재 선택분을, 작성자와 좌표까지 포함해서 |

## Embedding Atlas 뷰어

지도와 나란히, 이 앱은 Apple의
[Embedding Atlas](https://github.com/apple/embedding-atlas) 뷰어 자체를 서빙합니다.
비슷하게 만든 것이 아니라 실제 컴포넌트입니다. 정렬 가능한 표, 칼럼마다의 분포 차트와
상호 교차 필터, 전문 검색, 그리고 밀도 등고선과 자동 군집 이름표를 갖춘
WebGPU 임베딩 뷰가 들어 있습니다.

왼쪽 레일의 패널 버튼으로 전환합니다. 넓은 화면과 지원되는 브라우저에서는 시작할 때 켜지고,
휴대폰에서는 꺼진 채로 둡니다. 휴대폰 화면에서는 대시보드가 제 몫을 하지 못합니다.

**Mosaic 애플리케이션이라 데이터를 받는 게 아니라 질의합니다.** 서버가 코퍼스를 담은
인메모리 DuckDB를 두고 `POST /data/query`로 답합니다. `embedding_atlas.server`의 엔드포인트를
명령 세 종류까지 그대로 따랐기 때문에 컴포넌트는 수정 없이 붙습니다. 재계산 때마다 웹소켓
브로드캐스트 **전에** 이 릴레이션을 갱신하므로 차트와 지도가 어긋나지 않습니다.

**데이터베이스는 둘이고, 엔드포인트는 SQL을 파싱하기 전에 하나를 고릅니다.** 질의를
작성하는 쪽이 클라이언트이므로 내보내는 쪽의 필터는 버틸 수 없습니다.
`SELECT reviewer_id FROM dataset`은 Mosaic 입장에서 평범한 요청입니다. 그래서 참가자용
데이터베이스에는 그 칼럼이 아예 없고, 요청하면 데이터가 아니라 DuckDB의 바인더 오류가
돌아옵니다. 참가자 질의에는 자격 증명이 필요 없고 주소별로 속도가 제한됩니다. 관리자 질의는
관리자 코드를 요청 본문으로 보냅니다. URL이 아니라 본문으로 전달되는 것은 CSV 내보내기와
같습니다.

**AI/사람 구분은 모양에서 필터로 옮겨졌습니다.** `EmbeddingView`는 범주 색 하나만
인코딩하므로 지도의 두 번째 채널에 대응하는 것이 없습니다. 대신 `source` 차트가 대시보드
전체를 교차 필터합니다. `ai`를 누르면 다른 모든 차트와 표와 임베딩 뷰가 거기에 맞춰
좁혀집니다. 두 군집 비교가 범례가 아니라 클릭 한 번이 됩니다.

**번들 다시 만들기.** `web/vendor/`는 커밋되어 있고 수업 중에 받아오는 것은 없습니다.
버전을 바꾼 뒤에는:

```bash
cd frontend && npm install && npm run build   # -> ../web/vendor
```

Node는 이 빌드에만 필요하고 다른 곳에는 필요 없습니다. 서버는 실행하지 않습니다.

같은 패키지로 내보낸 CSV를 공식 CLI로 열 수도 있습니다. 서버가 전혀 필요 없습니다:

```bash
embedding-atlas feedback-atlas.csv --text text --x x --y y
```

## AI 의견 일괄 등록 (PRD 4.3)

의견이 지도에 들어오는 유일한 경로입니다. `scripts/seed_ai_opinions.py`가 데이터베이스에
쓰고, `SIGUSR1`로 실행 중인 서버가 다시 읽게 합니다. 작성자는 명단에 있는 ID여야 합니다.

```bash
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer ta.youngjun --dry-run
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer ta.youngjun
kill -USR1 $(pgrep -f 'uvicorn.*src.server')   # 재시작 없이 다시 읽기
```

입력은 CSV 또는 JSONL이며 `target_id,text[,source][,week]` 형식입니다. 대상을 찾을 수 없는
행은 건너뛰고 목록으로 알려줍니다. 임의로 추측하지 않습니다. 먼저 `--dry-run`으로
확인하세요. 의견 ID는 행마다 새로 생성되므로 같은 파일을 두 번 넣으면 두 번 들어갑니다.

## 대화 경로 지도 (`/chats.html`)

학생이 LLM과 나눈 대화를 위한 별도 화면입니다. 대화를 문장 단위로 나누어 같은 임베딩
공간에 놓고, 말해진 순서대로 선을 그립니다. 수업 지도에는 일부러 넣지 않았습니다. 대화
하나가 약 800문장인데 수업 지도는 한 학기 약 1000건을 기준으로 만들어졌고, 학생이 모델에게
쓴 질문은 동료 피드백이 아니기 때문입니다.

1. claude.ai 공유 링크를 열고 끝까지 스크롤한 뒤 **Ctrl+S → "웹페이지, 전체"**로 저장합니다
   ("HTML만"은 안 됩니다. 이 페이지는 스크립트로 그려집니다). 저장한 페이지에는 누가 언제
   말했는지가 표시되어 있고, 복사·붙여넣기에는 없습니다.
2. `.run/chats/`에 `<target_id>__w<주차>__<번호>.html` 이름으로 둡니다. 예:
   `kim.seoyeon__w1__01.html`. 대상은 명단의 수강생이어야 합니다.
3. 지도를 다시 만듭니다. 새 문장만 임베딩하며, 캐시는 수업 데이터베이스가 아니라
   `.run/chats/embeddings.db`입니다.

```bash
HF_HUB_OFFLINE=1 python scripts/build_chat_atlas.py --chats .run/chats --roster roster.csv
```

재시작은 필요 없습니다. `/api/chats`는 요청마다 `.run/chats/atlas.json`을 읽습니다. 페이지와
마찬가지로 링크를 가진 누구에게나 열려 있으니, 학생이 이렇게 보여도 된다고 동의한 대화만
등록하세요. 만들기 전에 나뉜 결과를 보려면
`scripts/chat_to_trajectory.py`가 읽기 쉬운 `<파일명>.preview.md`를 씁니다.

다시 만들어도 지도는 안정적입니다. 배치는 이전 결과에서 출발해 같은 방향·크기로 맞추고,
군집은 번호·색·범례 순서를 유지하므로 대화를 추가해도 지도가 돌거나 색이 바뀌지 않습니다
(새 문장이 자리를 잡으면서 문장들이 국소적으로 조금 움직이기는 합니다). 새 문장이 없으면 아무것도
움직이지 않습니다. `--fresh`는 처음부터 다시 배치합니다. 같은 대화를 두 번 저장한 파일(첫 메시지와
그 시각이 같음)은 더 긴 쪽만 쓰고, 건너뛴 파일은 빌드 출력에 알려 줍니다.

남기는 것과 빼는 것: 학생의 질문(네모)과 모델의 문장(원)은 모두 점이 됩니다. "웹 검색됨" 같은
상태 표시는 표식(마름모)으로 남기되 배치하지 않고 다음 문장 옆에 고정합니다. 코드 블록(붙여
넣은 Claude Code 프롬프트), 출처 칩, 화면 낭독기용 제목은 뺍니다. 20자 미만 조각은 이웃 문장과
합칩니다. 주차는 메시지 시각이 아니라 파일 이름에서 가져옵니다.

화면은 세 가지 굵기로 길을 그립니다. 지금 턴은 문장 단위로 진하게, 지난 문장들은 흐리게,
그리고 각 턴 문장들의 중심을 잇는 번호 붙은 **턴 경로**로 대화 전체의 흐름을 보여 줍니다.

## 학기가 끝난 뒤 (PRD 6, 7)

```bash
python scripts/deidentify.py --db .run/classroom/atlas.db --roster roster.csv --out archive/
```

모든 ID를 무작위 코드로 바꾼 `archive/archive.csv`와, 코드 대응표인
`archive/mapping.csv`를 만듭니다. 원본 데이터베이스는 건드리지 않습니다.

둘을 반드시 떼어놓으세요. 대응표는 아카이브를 다시 식별할 수 있는 유일한 수단입니다.
파기하거나, 아카이브가 없는 곳에 따로 보관하세요. 코드는 ID에서 유도하지 않고 무작위로
섞어 배정합니다. 해시나 명단 순서 번호는 학생 명단을 가진 사람이면 누구나 되돌릴 수 있고,
그 사람은 곧 아카이브를 공유받을 모든 사람입니다.

**이 아카이브는 가명 처리된 것이지 익명이 아닙니다.** 의견 본문은 그대로이고, 사람들은
"제가 발표에서 말했듯이" 같은 문장을 씁니다. 연구 목적으로 쓴다면 진짜 통제 수단은 수업
참여와 별도로 받은 동의와, 성적 확정 이후에만 분석하는 절차라는 PRD 11의 지적이 맞습니다.
이 스크립트는 치환 단계이지 윤리 절차가 아닙니다.

## 테스트

```bash
python -m pytest feedback-atlas
```

CPU만 사용하고, 모델 가중치도 네트워크도 쓰지 않습니다. 저장소의 다른 부분과 같은
규칙입니다. 필수 의존성은 numpy와 pytest뿐이며, umap·fastapi·httpx·duckdb가 필요한 테스트는
실패가 아니라 건너뜁니다. 그래서 맨 인터프리터에서도 전체 suite가 돌아갑니다.

그중 셋은 알아둘 값어치가 있습니다. 보통은 "그러길 바라는" 수준에 그치는 것을 실제로
확인하기 때문입니다.

- `test_embedder_contract.py`는 `src.embedder`를 **별도 프로세스에서** import한 뒤 torch가
  `sys.modules`에 없음을 확인합니다. 이 규칙 하나가 이 suite를 오프라인으로 유지합니다.
  다른 테스트가 torch를 한 번 불러오고 나면, 같은 프로세스 안의 검사로는 누가 불러왔는지
  알 수 없습니다.
- `test_server_routes.py`는 참가자와 관리자를 **동시에** 하나의 서버에 붙여놓고, 한 번의
  브로드캐스트에서 관리자에게는 `reviewer_id`가 보이고 참가자에게는 보이지 않음을
  확인합니다. 이 누출이 살아 있는 동안에도 모듈별 테스트는 전부 통과했습니다. 두 채널이
  동시에 연결되어 있을 때만 존재하는 결함이기 때문입니다.
- `test_viewer_pipeline.py`는 돌아가는 서버에 문장을 등록한 뒤 SQL로 그 문장을
  되묻습니다. 뷰어 쪽을 정직하게 검사하는 방법은 이것뿐입니다. HTTP로
  `reviewer_id`를 요청해 칼럼이 없어서 DuckDB가 거절하는지 확인하고, `/etc/passwd`를
  요청해 같은 거절을 확인합니다. duckdb·pyarrow·embedding-atlas가 없으면 건너뜁니다.

브라우저 동작 검사는 별도로 실행합니다. Playwright와 Chromium이 설치된 환경에서
`feedback-atlas/` 디렉터리의 `.venv/bin/python -m pytest tests/browser_viewer.py -q`를
실행하세요. 임시 데이터베이스와 로컬 테스트 서버로 WebGPU 미지원 시 지도 유지,
재연결·지연된 기능 검사 중 뷰어 선택 유지, 명단 변경 알림을 확인합니다.
뷰어 동작을 대체한 테스트이므로 실제 GPU 렌더링은 검증하지 않습니다.

## 알려진 한계

- **작성자 정보를 뺀 모든 것은 링크를 가진 누구에게나 공개됩니다.** 참가자 로그인도
  IP 대역 제한도 없으므로 `https://143.248.249.7:8888`에 도달할 수 있는 사람은 지도, 분석
  보기, `/chats.html`의 모든 대화를 읽을 수 있습니다. 관리자 코드 뒤에 있는 것은
  `reviewer_id`, 관리자 도구, CSV 내보내기뿐입니다.
- **전용 OS 사용자 격리는 아직 설정되어 있지 않습니다.** 서비스는 현재 이 저장소에서 기존
  `june` 계정으로 실행됩니다.
- **보안 배포 과정에서 애플리케이션 의존성은 바꾸지 않았습니다.** 이 문서 작업에서는
  dependency advisory scan을 다시 실행하지 않았습니다.
- **의견이 80건이 되는 순간 지도가 한 번 재배치됩니다.** 투영 방식이 PCA에서 UMAP으로
  바뀌기 때문이며, 두 방식은 구조적으로 다른 배치를 만들어 어떤 정렬로도 메울 수 없습니다.
  `ATLAS_PCA_UMAP_THRESHOLD`로 조정할 수 있으니, 발표 도중이 아니라 세션 사이에 의도적으로
  넘기세요.
- **AI/사람 구분은 자기 신고입니다.** 누구나 어느 쪽으로든 표시할 수 있으므로(PRD 5.2),
  AI-사람 비교의 신뢰도는 표시의 정확도만큼입니다. PRD 11이 지적한 사항이며 앱은 이를
  탐지하려 하지 않습니다.
- **데이터베이스는 로컬 디스크에 두세요.** SQLite의 WAL 모드는 NFS/SMB에서 신뢰할 수
  없습니다.

## 설계 노트

Apple의 [Embedding Atlas](https://github.com/apple/embedding-atlas)(MIT)는 여기서
세 가지로 쓰이며, 그중 마지막만 겉모습입니다.

**분석 방식 — 호출이 아니라 재현입니다.** `src/ea_projection.py`는 umap의
`nearest_neighbors`로 근사 k-NN 그래프를 만들고 그것을 `precomputed_knn`으로 넘겨
UMAP을 학습시킵니다. `embedding_atlas.projection`이 쓰는 것과 같은 순서, 같은
파라미터입니다. 다만 그쪽 함수를 부르지는 않습니다. 그 함수는 데이터프레임을 만들고,
입력을 키로 하는 파일 캐시로 감싸고, 학습된 reducer를 버립니다. reducer가 없으면 의견이
하나 들어올 때마다 전체를 다시 학습해야 하고, 발표 중에 읽히는 지도는 그 비용을 감당할 수
없습니다. 그래서 알고리즘과 기본값은 그쪽 것이고, 호출은 이쪽 것입니다. 그 학습이 만들어낸
그래프가 `neighbors` 칼럼을 채웁니다.

혼동하기 쉬우니 분명히 적어 둡니다. 실행 경로에서 실제로 도는 상위 패키지의 파이썬 코드는
`/data/query`가 차트 질의에 답할 때 쓰는 Arrow 작성기 하나뿐입니다.

**뷰어와 렌더러.** `web/vendor/`에 번들로 넣고 서버의 DuckDB에 붙였습니다. 위 절을
보세요.

**시각 언어** — 손으로 쓴 지도 쪽입니다. 슬레이트 바탕, 흰색/검정 카드, 1px 테두리,
단일 파란 강조색, d3의 `category10`, 반투명 툴팁, 후광 처리된 지도 이름표. 한국어를
위한 두 가지 변경: 의견 본문을 13px이 아닌 14px/1.6로, 그리고
`word-break: keep-all` — 없으면 브라우저가 한국어를 단어 중간에서 끊습니다.

`web/atlas.js`의 손으로 쓴 산점도는 남아 있고, 중복이 아닙니다. `EmbeddingView`가
채널 하나를 인코딩하는 자리에서 둘을 인코딩하고, WebGPU 없이 동작하며, 의견이 도착하는
순간에 애니메이션을 줍니다. 휴대폰이 받는 화면이자, 뷰어를 그릴 수 없는 노트북이 받는
화면입니다.

CDN도, 웹폰트도, 수업 중에 받아오는 것도 없습니다. 프런트엔드는 여전히 `web/`에서
그대로 서빙되는 순수 ES 모듈이고, 번들러는 개발 시점에 `frontend/`에서 한 번만 돌며 그
결과물은 저장소에 커밋되어 있습니다.
