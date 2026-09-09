# 실시간 피드백 지도

<p align="center">
  <a href="README.md">English</a> · <b>한국어</b>
</p>

CTP49906 *Multimodal AI* 워크숍용 강의실 도구입니다. 학생이 발표하는 동안 참석자들은
각자 휴대폰으로 링크를 열고 떠오른 생각을 적습니다. 각 의견은 외부 API 호출 없이
**로컬에서** 임베딩되어 2D로 투영되고, 1초 안에 접속한 모든 화면에 전달됩니다.

색은 **어느 학생에 대한 의견인지**, 모양은 **AI인지 사람인지**를 나타냅니다. 두 번째
채널이 이 도구의 핵심입니다. 이 세션은 모델이 주관적 경험을 어디까지 서술할 수 있는지를
묻고, 이 지도는 두 종류의 코멘트를 같은 의미 공간에 놓아 둘이 같은 자리에 떨어지는지를
눈으로 확인하게 합니다.

하나의 코퍼스 위에 화면이 둘입니다. 휴대폰과 프로젝터를 위해 손으로 쓴 지도, 그리고 표와
연동 차트와 교차 필터를 위한 Apple의 Embedding Atlas 뷰어. 둘은 같은 행을 읽습니다.

명세는 [`realtime-feedback-atlas-prd.md`](realtime-feedback-atlas-prd.md)입니다.

## 이것이 아닌 것

캠퍼스 로그인 시스템은 아닙니다. 참가자는 본인의 고정 개인 접근 코드 하나만 입력하며,
서버가 연결된 명단의 ID와 역할을 확인합니다. 참가자 웹소켓은 `{code}`를 받아
8시간짜리 서명된 세션 토큰을 돌려주거나, 이후 `{session_token}`으로 세션을 이어 받습니다.
서버를 재시작해도 개인 코드는 바뀌지 않으며, 세션이 만료되면 같은 코드로 다시 들어옵니다.
한 코드가 두 사람을 가리키지 않도록 중복 코드는 허용하지 않습니다.

익명이 아니며, 익명인 척하지도 않습니다. `reviewer_id`는 서버에 저장되지만 참가자
연결로는 절대 전달되지 않습니다. 고정 접근 코드 뒤에 있는 관리자 모드만이 누가 무엇을
썼는지 보여주는 유일한 화면입니다.

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
있는 프로젝트입니다. `observer`(청강생)와 `ta`(조교)는 의견을 쓰지만 대상이 되지는
않습니다. 수업의 실제 모습 그대로입니다 — 조교는 작업에 코멘트를 하지만, 조교의 작업에
코멘트하는 사람은 없습니다.

PRD 7의 표에는 `student`와 `observer`만 있습니다. `ta`는 의도적으로 추가했습니다. PRD 3이
이미 조교를 의견을 제출할 수 있는 사용자로 적고 있고, 제출하려면 명단 항목이 필요하기
때문입니다. 고유한 역할이 없으면 조교를 청강생으로 등록해야 하는데, 그러면 귀속을 보라고
만든 관리자 패널이 조교를 있는 그대로 표시하지 못합니다. 대조는 공백·대소문자·전각 문자에 관대하고(한국어 IME가
전각 모드면 `ｋｉｍ`이 입력됩니다) 그 외에는 관대하지 않습니다. 편집 거리 기반 매칭은 한
학생의 ID가 다른 학생의 ID로 해석되게 만듭니다.

참가자별 비공개 코드는 명단이 공유 가능한 상태가 된 뒤 생성합니다. JSON 파일에는
SHA-256 해시만 들어가고 서버가 그것을 읽습니다. CSV에는 원문 코드가 들어가므로 각
참가자에게 자기 코드 한 줄만 따로 보내세요. CSV 전체를 공개 채팅이나 슬라이드에 올리지
마세요.

```bash
python scripts/create_access_codes.py \
  --roster roster.csv \
  --hashes .run/access-codes.json \
  --out .run/access-codes.csv
```

스크립트는 `.run/` 아래에 권한 `0600`인 비공개 파일을 만들고, 기존 자격 증명은 덮어쓰지
않습니다. 회전하려면 기존 파일을 옮겨 두고 새 코드를 개인별로 다시 배포하세요.

나중에 사람을 추가하려면 명단을 고치고, 새 명단 상태에 맞는 코드를 생성한 뒤 `SIGHUP`을
보냅니다. 재시작도, 연결 끊김도 없습니다.

```bash
kill -HUP $(pgrep -f 'uvicorn.*src.server')
```

### 환경 설정

[`.env.example`](.env.example)을 `.env`로 복사하세요. `ATLAS_ADMIN_CODE`와
`ATLAS_ROSTER`에는 기본값이 없습니다. `ATLAS_ACCESS_CODES`의 기본값은
`.run/access-codes.json`이며, 서버를 시작하기 전에 모든 명단 ID의 해시가 이 파일에 있어야
합니다. 기본값이 있는 관리자 코드는 이미 공개된 관리자 코드입니다.

이전 관리자 코드가 평문 HTTP를 지난 적이 있다면 수업 전에 회전하세요.

```bash
python scripts/configure_security.py
```

새 관리자 코드는 비공개 권한으로 `.run/admin-code.txt`에 저장됩니다. 그 내용을 문서나
슬라이드에 넣지 마세요.

## 실행

실제 수업 메인(`/`)과 연습용 데모(`/demo/`)는 같은 화면을 사용하지만 데이터는
분리됩니다. 메인은 개인 코드로 입장하는 1~4주차 수업 공간이고, 데모에서는 기본
계정 `demo` 하나로 코드 입력 없이 바로 들어갑니다. 기존 의견 30건은 데모로 분리하고,
메인은 의견 0건에서 시작합니다. 데모의 세션·입력·조회·내보내기는 메인에 섞이지
않습니다. 데이터 경로와 이전 방법은 [운영 문서](SECURITY_DEPLOYMENT.md#classroom-and-demo)를
참고하세요. 새로 설치할 때는 `.run/classroom`, `.run/demo` 디렉터리를 먼저 만듭니다.

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
맞춰 켜면 그 비용이 그 학기 첫 의견 제출에 그대로 얹힙니다.

`GET /healthz`는 모델 캐시 키, 누적 의견 수, 사용 중인 투영 방식, 접속자 수를 보고합니다.
접근 코드는 절대 보고하지 않습니다.

### 강의실 접속

공유할 주소는 HTTPS IP 주소 하나입니다.

```text
https://143.248.249.7:8888
```

페이지, 웹소켓, Embedding Atlas 질의 엔드포인트가 모두 같은 origin에 있습니다. 알려진
강의실 CIDR이 없어 서버는 강의실 IP 대역으로 제한되어 있지 않습니다. 보안은 IP 필터가
아니라 TLS, 개인별 코드, 서명된 세션, 관리자 코드로 성립합니다.

**SSH 터널은 수업용 경로가 아닙니다.** `ssh -L 8888:127.0.0.1:8888 …`은 명령을 입력한
*그 노트북 한 대에만* 포트를 전달합니다. 강사가 SSH 너머로 지도를 미리 보는 데는 맞는
방법이고, 학생 휴대폰 30대에는 아무 역할도 하지 않습니다.

Cloudflare는 권장 실행 경로 밖입니다. 나중에 수동으로 다른 터널을 쓴다면 같은 참가자·관리자
자격 증명, HTTPS 브라우저 origin, 웹소켓 업그레이드, origin 검사를 유지하세요.

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
맞춰 켜면 그 비용이 그 학기 첫 의견 제출에 그대로 얹힙니다.

운영 절차는 [`SECURITY_DEPLOYMENT.md`](SECURITY_DEPLOYMENT.md)에 정리되어 있습니다.

## 관리자 모드

**주차 일정 설정** 버튼에서 네 주차의 시작일을 달력으로 지정하고 각 7일 범위를
확인할 수 있습니다. 메인은 **한국 시간(Asia/Seoul)의 서버 접수 시각**으로 주차를
분류하며, 참가자가 보내는 주차 값은 사용하지 않습니다. 기본 일정은 2026년
10/15–10/21, 10/22–10/28, 11/5–11/11, 11/12–11/18입니다.
10/29–11/4는 중간고사 휴강 기간입니다.

기간 밖의 새 의견은 제출되지 않으며 작성 중인 문장은 유지됩니다. 이미 접수된
의견의 재전송 확인은 기간이 끝난 뒤에도 가능합니다. 저장한 일정은 재시작 후에도
유지되며, 변경 후 새로 접수되는 의견부터 적용합니다. 기존 의견의 주차는 바꾸지
않습니다. 데모는 별도 설정을 사용하며 날짜에 관계없이 1주차로 연습할 수 있습니다.
참가자 입력란의 주차 드롭다운은 없고, AI/사람 구분에 맞는 입력 안내가 표시됩니다.

`/admin`을 열고 접근 코드를 입력합니다. 코드는 UI 플래그가 아니라 **다른 웹소켓**을
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
휴대폰에서는 꺼진 채로 둡니다. 휴대폰에서 할 일은 대시보드 탐색이 아니라 한 문장 쓰기입니다.

**Mosaic 애플리케이션이라 데이터를 받는 게 아니라 질의합니다.** 서버가 코퍼스를 담은
인메모리 DuckDB를 두고 `POST /data/query`로 답합니다. `embedding_atlas.server`의 엔드포인트를
명령 세 종류까지 그대로 따랐기 때문에 컴포넌트는 수정 없이 붙습니다. 재계산 때마다 웹소켓
브로드캐스트 **전에** 이 릴레이션을 갱신하므로 차트와 지도가 어긋나지 않습니다.

**데이터베이스는 둘이고, 엔드포인트는 SQL을 파싱하기 전에 하나를 고릅니다.** 질의를
작성하는 쪽이 클라이언트이므로 내보내는 쪽의 필터는 버틸 수 없습니다.
`SELECT reviewer_id FROM dataset`은 Mosaic 입장에서 평범한 요청입니다. 그래서 참가자용
데이터베이스에는 그 칼럼이 아예 없고, 요청하면 데이터가 아니라 DuckDB의 바인더 오류가
돌아옵니다. 참가자 질의에는 웹소켓 hello에서 받은 `Authorization: Bearer <session_token>`도
필요합니다. 관리자 질의는 계속 관리자 코드를 요청 본문으로 보냅니다. URL이 아니라 본문으로
전달되는 것은 CSV 내보내기와 같습니다.

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

제출 경로는 연결당 초당 한 건 정도로 제한됩니다. 사람에게는 맞고 스크립트에는 맞지 않는
값입니다. `scripts/seed_ai_opinions.py`는 데이터베이스에 직접 씁니다.

```bash
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer instructor --dry-run
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer instructor
kill -USR1 $(pgrep -f 'uvicorn.*src.server')   # 재시작 없이 다시 읽기
```

입력은 CSV 또는 JSONL이며 `target_id,text[,source][,week]` 형식입니다. 대상을 찾을 수 없는
행은 건너뛰고 목록으로 알려줍니다. 임의로 추측하지 않습니다. 먼저 `--dry-run`으로
확인하세요. 의견 ID는 행마다 새로 생성되므로 같은 파일을 두 번 넣으면 두 번 들어갑니다.

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
- `test_viewer_pipeline.py`는 실제 웹소켓으로 문장을 제출한 뒤 돌아가는 서버에 SQL로 그
  문장을 되묻습니다. 뷰어 쪽을 정직하게 검사하는 방법은 이것뿐입니다. HTTP로
  `reviewer_id`를 요청해 칼럼이 없어서 DuckDB가 거절하는지 확인하고, `/etc/passwd`를
  요청해 같은 거절을 확인합니다. duckdb·pyarrow·embedding-atlas가 없으면 건너뜁니다.

브라우저 동작 검사는 별도로 실행합니다. Playwright와 Chromium이 설치된 환경에서
`feedback-atlas/` 디렉터리의 `.venv/bin/python -m pytest tests/browser_viewer.py -q`를
실행하세요. 임시 데이터베이스와 로컬 테스트 서버로 WebGPU 미지원 시 지도 유지,
재연결·지연된 기능 검사 중 뷰어 선택 유지, 명단 변경 알림을 확인합니다.
뷰어 동작을 대체한 테스트이므로 실제 GPU 렌더링은 검증하지 않습니다.

## 알려진 한계

- **개인별 코드는 캠퍼스 SSO가 아니라 수업용 자격 증명입니다.** 누군가 다른 사람의 코드를
  받거나 훔치면 그 사람인 척 쓸 수 있습니다. 각 참가자에게 자기 코드만 주고, 노출되면
  코드를 회전하세요. 관리자 작성자 조회를 성적 판단의 근거로 삼지 마세요.
- **서비스는 IP 대역으로 제한되어 있지 않습니다.** 현재 강의실 CIDR이 설치되어 있지
  않습니다. `https://143.248.249.7:8888`에 도달할 수 있는 사람은 페이지를 열 수 있지만,
  보호된 엔드포인트를 쓰려면 유효한 참가자 코드 또는 관리자 코드가 필요합니다.
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
