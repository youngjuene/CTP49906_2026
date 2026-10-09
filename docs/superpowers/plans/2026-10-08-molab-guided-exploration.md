# 비전공자용 Molab 탐구 활동 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 사용자가 별도로 위임을 요청하지 않으면 한 작업자가 순서대로 실행한다.

**Goal:** 새 영상 2종과 한국어 질문 탐색을 포함해, 비전공자가 조건을 바꾸고 결과와 자기 설명을 구분할 수 있는 한국어 Molab 수업 흐름을 만든다.

**Architecture:** 검증된 영상 카탈로그로 원본·무음 쌍을 선택하고, 질문 정의·실행 어댑터·비교 기록·저장 예제를 작은 Python 모듈로 나눈다. 기존 계산 함수를 canonical marimo 노트북에서 호출하고 현재 생성기로 HTML 실행판을 만든다. 라이브와 저장 결과는 동일한 표시 경로를 사용하되 출처를 구별한다.

**Tech Stack:** Python 3.11+, marimo 0.25.0, 기존 PyTorch·Transformers·Qwen 모델, JSON/JSONL, pytest. FFmpeg/ffprobe는 영상 준비·검증 환경에만 필요하다.

**Spec:** [비전공자용 Molab 탐구 활동 설계](../specs/2026-10-08-molab-guided-exploration-design.md).

**기준:** `0d2eb6f7713a2c6e107b1fb3b54284f38eb1fe22`, 2026-10-08 1차 검토 반영본. 기존 384개 CPU 검사와 이전 GPU 검증은 출발점이며 새 기능 검증을 대신하지 않는다. 계획 검토 뒤 사용자의 후속 요청으로 새 영상의 10초 파일 준비와 기존 노트북 드롭다운 연결을 수행했다. 아래 진행 기록은 이 두 요청의 범위이며 전체 계획의 구현 완료를 뜻하지 않는다.

## 영상 준비 진행 기록 (2026-10-08 후속 요청)

- [x] 기존 `02321.mp4`의 비디오 길이 10.000초를 ffprobe로 확인했다.
- [x] 타블라 75–85초와 비 오는 장면 5–15초를 각각 소리 포함본/무음본으로 준비했다. 파일은 assets/scene02*.mp4, assets/scene03*.mp4다.
- [x] 네 파일의 10초 길이, 각 쌍의 240개 RGB 프레임·시간축 일치, 원본 오디오 신호·무음본 0신호, 기존 미디어 안전 검사를 확인했다.
- [x] 출처·라이선스·가공 내역·파일 해시·재현 스크립트와 검증 JSON을 저장했다.
- [ ] 직접 청취, processor/GPU 검사와 나머지 Task는 남아 있다. Task 1 전체 완료로 표시하지 않는다.

## 드롭다운 연결 진행 기록 (2026-10-09 후속 요청)

- [x] `src/classroom_media.py`에서 세 장면의 원본·무음 경로, 파일 해시, 장면별 비교 키를 관리한다. 다른 장면이나 업로드를 같은 대조군으로 연결하지 않는다.
- [x] 별도 미리보기 드롭다운과 기존 다양성·티처 포싱 폼에 여섯 파일을 연결했다. 미리보기 변경은 모델 계산을 실행하지 않으며 실험 폼은 제출 시점에 적용된다.
- [x] 생성기를 통해 내장 HTML판에 반영하고 한국어 노트북의 영어 비교 안내를 한국어 질문 탐색으로 바꿨다.
- [x] 전체 CPU 검사 400개, 엄격 marimo 검사, 생성본 동기화 및 실제 로컬 marimo 브라우저의 새 영상 선택을 확인했다. GPU 실행 증거와 구분한다.
- [ ] 계획의 A/B 가이드 활동, 한국어 질문 프리셋, 18개 저장 결과, 새 영상의 Molab GPU 검증은 아직 구현·검증하지 않았다. 가이드와 경로 차단 시범은 기존 장면 1을 유지한다.

## Global Constraints

- Python >=3.11, marimo==0.25.0, transformers==4.52.4, 모델 Qwen/Qwen2.5-Omni-3B, MODEL_REVISION=f75b40e3da2003cdd6e1829b1f420ca70797c34e.
- 기존 GPU용 torch를 UI 개선 때문에 교체하지 않는다. 새 웹 서버·터널·프런트엔드 프레임워크를 추가하지 않는다.
- `CTP49906_avllm_molab_html_kr.py`는 직접 편집하지 않고 canonical 노트북과 `scripts/build_molab_html.py`에서 생성한다.
- `src/teacher_forcing.py`와 `src/attention_knockout_experiment.py`의 측정 정의를 바꾸지 않는다. 새 실행 경로는 기존 계산 함수를 호출한다.
- 비교 사이에는 의도한 변수만 바꾼다. 프레임·프롬프트·모델·코드·상한의 숨은 변경을 허용하지 않는다.
- 한국어판의 추천 프롬프트·학생 활동은 한국어로 구성하며 영어 비교 활동을 추가하지 않는다.
- 새 영상 2종의 출처·이용 조건·원본/무음 정합성과 실제 GPU 결과를 첫 배포 완료 조건에 포함한다.
- 사용자 입력과 모델 출력은 HTML/Markdown에서 이스케이프하고 JSON에는 원문을 보존한다.
- 저장 예제를 라이브 실행으로 표시하지 않는다. 출처의 해시·시간·모델 정보를 현재 값으로 덮어쓰지 않는다.
- 예제 선택·초안 편집·관찰 기록·다운로드는 추가 추론 0회를 만족해야 한다.
- 기술 검사 통과, Molab GPU 검증, 학생 시범 활동은 각각 별도의 완료 상태로 보고한다.

## Review Focus

1. 같은 파일명에 다른 영상이 들어오거나 무음본의 영상·시간축까지 바뀐 경우: 등록된 짝으로 인정하지 않는다. Task 1의 해시·디코딩 검증과 Task 3의 대조 키 검사로 확인한다.
2. 영상·한국어 질문·프레임을 바꿔도 같은 활동 ID인 경우: 캡션 캐시와 저장 결과를 잘못 재사용하지 않는다. Task 2·3·6의 전체 조건 식별 검사로 확인한다.
3. 원본 성공 뒤 무음 계산 실패 또는 세션 재연결: 첫 증거와 메모를 보존하고 미완료를 표시한다. Task 3·4·5의 부분 실패·복구 검사로 확인한다.
4. 저장 결과 재열기·관찰 메모 편집·긴 한국어와 HTML 문자 입력: 추론을 실행하거나 원래 출처·다른 메모를 덮어쓰지 않는다. Task 4·5·6의 ID·이스케이프·실행 횟수 검사로 확인한다.
5. GPU가 없거나 일부 저장 파일만 손상된 경우: 정상 조합은 계속 읽고 잘못된 조합은 명시적으로 차단한다. Task 6·8의 18개 조합·부분 손상·가중치 로드 금지 검사로 확인한다.

## 검토에서 바꾼 결정

- 새 영상 2종을 첫 배포에 포함했다. 기존 영상까지 총 3개 원본/무음 쌍을 제공한다.
- 한국어 질문 `sound / visual / combined`와 직접 입력을 추가했다. 영어 비교는 한국어 수업 범위에서 제외했다.
- 활동 ID만으로 저장 예제를 고르던 설계를 버리고 영상·프롬프트·조건·모델을 포함하는 키로 바꿨다.
- 기존 `02321` 전용 대조 키 발급을 검증된 카탈로그로 확장한다. 짝짓기 검사 자체를 느슨하게 만들지 않는다.
- 저장 결과는 2개에서 **18개 비교(2활동 × 3영상 × 3질문)**로 늘린다. CPU에서도 질문·영상을 바꿔 탐색할 수 있게 한다.
- 저장 예제의 원본 ID와 학생 관찰 기록 ID를 분리하고, float32 측정에 맞게 배열 검증 오차를 정했다.

## 적용 순서와 범위

`Task 1 영상 → Task 2 질문·계획 → Task 3 실행 → Task 4 기록 → Task 5 화면 → Task 6 실제 저장 결과 → Task 7 수업 문서 → Task 8 통합 검증` 순서다. Task 6의 로더 개발은 CPU에서 가능하지만 실제 배포 자료 생성에는 GPU가 필요하다.

첫 화면은 A·장면 1·소리 질문만 선택한 상태로 시작한다. 18개 조합을 한 화면에 펼치거나 모두 수행하게 하지 않는다. 별도 Studio 개편·업로드 자동 무음 제작·자동 채점은 범위 밖이다. 기존 영어판의 회귀 검사는 영어 비교 수업을 뜻하지 않는다.

## 파일 책임

아래 경로는 저장소 루트 기준이다. 이후 Task의 짧은 경로와 테스트 명령은 `avllm_interpretability/` 기준이다.

| 파일 | 책임 |
|---|---|
| `avllm_interpretability/assets/lesson_media.json` (신규) | 세 영상의 ID·버전·상대 경로·해시·짝 키·출처·가공 내역·검증 보고서 연결 |
| `avllm_interpretability/assets/scene02.mp4`, `scene02_silent.mp4`, `scene03.mp4`, `scene03_silent.mp4` (신규) | 새 두 영상의 수업용 원본·무음 파생 파일. 기존 `assets/02321*.mp4`는 재사용 |
| `avllm_interpretability/assets/LESSON_MEDIA_SOURCES.md`, `lesson_media_validation.json` (신규) | 이용 조건·저작자·가공 표시·검증 근거. 학생에게 정답을 암시하는 해설은 교사용 문서에 둠 |
| `avllm_interpretability/src/lesson_media.py` (신규) | 카탈로그 읽기·상대 경로/파일 해시 검증·검증된 짝 식별. torch/marimo 의존 없음 |
| `avllm_interpretability/scripts/prepare_lesson_media.py`, `validate_lesson_media.py` (신규) | 교사용 구간 변환·무음 제작, 디코딩·정합성 감사. 노트북 실행 중 호출하지 않음 |
| `avllm_interpretability/src/lesson_scenarios.py` (신규) | 두 활동·세 한국어 질문·직접 입력, 조건 검증·계획 키·요청 스냅샷 |
| `avllm_interpretability/src/lesson_runtime.py` (신규) | 기존 캡션/TF 함수 연결·순차 실행·부분 실패 보존 |
| `avllm_interpretability/src/lesson_records.py` (신규) | 비교/관찰 기록·JSONL·라이브/저장 출처·HTML·내보내기 |
| `avllm_interpretability/src/lesson_examples.py` (신규) | manifest/해시/전체 조건/배열 검증·정확한 예제 선택 |
| `avllm_interpretability/CTP49906_avllm_molab_kr.py` | 새 활동 셀·실행 경계·카탈로그 연결·기존 자동 시범 실행 차단 |
| `avllm_interpretability/scripts/build_molab_html.py`, 생성본 `CTP49906_avllm_molab_html_kr.py` | 첫 학습 위치·HTML 폼 생성. 계산 셀은 canonical과 동일 |
| `avllm_interpretability/scripts/generate_lesson_examples.py` (신규) | 실제 GPU에서 18개 비교 생성·출처와 측정 dtype 기록 |
| `avllm_interpretability/precomputed_kr/lessons/` (신규) | `manifest.json`과 `examples/<example_id>.json` 18개 |
| `avllm_interpretability/tests/test_lesson_media.py`, `test_lesson_scenarios.py`, `test_lesson_runtime.py`, `test_lesson_records.py`, `test_lesson_examples.py`, `test_lesson_notebook.py` (신규) | 미디어·조건·실행·기록·저장 자료·UI 계약 검사 |
| `avllm_interpretability/tests/conftest.py` | 공용 카탈로그 및 합성 비교 fixture. 배포 자료에는 합성값 사용 금지 |
| 기존 `test_molab_html.py`, `test_notebook_kr_replay.py`, `test_notebook_replay.py`, `test_studio_execution.py`, `test_studio_idle.py` | 생성본·한국어/영어 기존 기능·실행 경계 회귀 검사 |
| `avllm_interpretability/WORKSHEET_kr.md`, `CLASSROOM_GUIDE_kr.md`, `README_kr.md`, `MOLAB_HTML_QA.md` | 학습 흐름·출처·저장 결과 범위·실제 검증 상태 |

## 공통 자료 구조와 인터페이스

모든 영속 자료는 JSON으로 저장 가능한 `dict`다. 입력은 복사하고 GPU tensor는 어댑터 경계에서 CPU 목록/숫자로 변환한다. 원시 값은 표시용으로 반올림하지 않는다.

```python
PROMPTS = {
    "sound": "영상에서 들리는 소리를 한 문장으로 설명해 주세요.",
    "visual": "영상에 보이는 장면을 한 문장으로 설명해 주세요.",
    "combined": "영상에 보이는 장면과 들리는 소리를 함께 한 문장으로 설명해 주세요.",
}
DEFAULTS = {"nframes": 8, "max_new_tokens": 64, "do_sample": False}
COMPARISON_TYPES = {
    "sound_vs_silence": "input_condition",
    "audio_edge_effect": "within_caption",
}
```

`Plan`에는 `scenario_id/version`, `comparison_type`, `media`(ID·버전·두 파일 경로/해시·comparison_key), `prompt_id/version`, `common`(정확한 prompt·nframes·상한·do_sample), `model`(model_id·model_revision), `steps`가 있다. A steps는 `original/silent`, operation `caption` 두 개이며 B는 `fixed_caption`, operation `teacher_forcing`, rules `[["answer","audio",0,36]]` 하나다. 각 step에는 `path`, `sha256`을 포함한다. B도 카탈로그 식별을 보존하지만 무음 대조 실행을 만들지 않는다.

`plan_key`는 위 실행 조건의 canonical JSON SHA-256이다. 요청 ID·예상·메모·UI 표시명·저장 시각은 포함하지 않는다. `example_id`는 plan_key, 생성 헬퍼/스크립트 해시·모델 리비전, 생성 묶음마다 새로 만든 UUID인 generation_id로 만든다. 같은 조건을 다시 생성해도 이전 저장 결과와 ID가 겹치지 않는다. payload 해시는 별도 manifest 필드로 두어 자기 참조를 만들지 않는다. 모델·생성 코드 출처는 실제 생성 당시 값으로 보존한다.

`Example` payload는 `{schema_version,example_id,plan_key,generation_provenance,comparison}`이며 comparison은 아래 Comparison이다. generation_provenance에는 generation_id, helper_sha256, generator_sha256, model, created_at, repo_revision, repo_dirty, packages, python을 기록한다. manifest에도 같은 provenance를 넣어 대조한다.

`Comparison` 필수 필드: `schema_version`, `comparison_id`, `request_id`, `submitted_plan`, `plan_key`, `origin`, `created_at`, `status`, `members`, `errors`, `prediction`, `reflection`. `members=[{step_id,run}]`의 run은 기존 run_record 결과다. status는 complete/partial/failed, origin은 live/stored다. 저장 예제를 가져온 경우 `source_example_id`, `source_comparison_id`, `imported_at`을 추가한다. reflection은 changed/held_constant/observed/explanation/next_question 다섯 필드다. prediction은 `{choice,note}`다.

```text
# lesson_media.py: catalog는 media_id -> media 레코드 매핑
load_media_catalog(project_dir) -> dict
resolve_media(catalog: dict, media_id: str, *, project_dir) -> dict
# 위 함수는 실제 경로와 해시를 검증한 JSON 가능한 사본을 반환한다.
verified_pair_config(catalog: dict, clip_path, *, project_dir) -> dict
# 등록 파일이면 두 변형에 동일한 pair 메타데이터, 미등록이면 빈 dict.

# lesson_scenarios.py
make_plan(scenario_id: str, *, catalog: dict, media_id: str = "scene01",
          prompt_id: str = "sound", custom_prompt: str | None = None,
          nframes: int = 8, layer_band: tuple[int, int] = (0, 36)) -> dict
plan_key(plan: dict) -> str
snapshot_request(plan: dict, prediction: dict, *, request_id: str) -> dict

# lesson_runtime.py
execute_plan(request: dict, run_step, on_member=None) -> dict
make_step_runner(*, project_dir, catalog, model, processor, prepare_inputs,
                 experiment_config, caption_cache: dict) -> Callable
# run_step(step: dict, common: dict, media: dict) -> run_record dict
# execute_plan -> {members,errors,status,persistence_errors}

# lesson_records.py
new_comparison(request: dict, outcome: dict, *, created_at: str) -> dict
import_stored_comparison(example: dict, *, request_id: str,
                         prediction: dict, imported_at: str) -> dict
update_reflection(comparison: dict, reflection: dict) -> dict
save_comparison(comparison: dict, path) -> dict
load_comparisons(path) -> list[dict]
render_comparison_html(comparison: dict) -> str
build_lesson_evidence_json(runs, comparisons, provenance=None) -> str
build_lesson_worksheet_md(comparisons) -> str

# lesson_examples.py: load는 검증된 Example envelope를 반환한다.
load_lesson_example(directory, plan: dict, *, example_id: str | None = None) -> dict
example_differences(comparison: dict, plan: dict) -> list[str]
```

`Callable`은 `collections.abc.Callable`이다. `prepare_inputs(clip_path,nframes,prompt)`는 기존 `prepare_video_inputs`와 model/processor/type mapper를 묶어 `(inputs,types)`를 반환한다. 라이브 어댑터와 생성 스크립트는 동일한 경로를 사용한다.

## Task 1: 새 영상 선정·제작과 검증된 카탈로그

**Files:** 신규 assets 4개·카탈로그·출처/검증 문서, `src/lesson_media.py`, 영상 준비/검증 스크립트, `tests/test_lesson_media.py`, `tests/conftest.py`.

**Interfaces:** `load_media_catalog`, `resolve_media`, `verified_pair_config`. 카탈로그 ID는 `scene01/02/03`이며 각 레코드에 동일한 media_id를 명시한다. clips는 original/silent 각각 path·sha256만 보관하고 출처·표시명·검증 보고서는 레코드의 별도 필드로 둔다. scene01은 기존 `02321` 쌍, 새 두 역할은 설계 문서 §3을 따른다.

- [ ] 설계의 후보 원본을 재생·청취하고 기존 소 영상과 같은 10초 구간을 고른다. 타격 동작과 실제 소리, 환경음의 단서를 각각 기록한다. 부적합한 후보는 같은 역할의 다른 공개 영상으로 교체하고 선정 이유를 남긴다. 얼굴·대화·배경 음악·정답 자막이 수업을 방해하지 않는 구간을 우선한다.
- [ ] 원천 URL·설명 페이지의 고정 revision·제작자·라이선스 URL·확인일·다운로드 SHA-256·가공 구간을 출처 목록에 기록한다. 기존 영상의 출처도 확인한다. 이용 조건이 확인되지 않은 파일을 새로운 수업 추천 목록에 배포하지 않는다. 저장소의 코드 라이선스로 대체하지 않는다.
- [ ] `prepare_lesson_media.py`에 `--source`, `--media-id`, `--start`, `--reference`, `--out-dir` 인자를 둔다. duration은 기준 영상에서 읽고 기본 reference는 assets/02321.mp4다. source는 사전에 받은 로컬 원천 파일이다. 기존 출력이 있으면 덮어쓰지 않고 실패한다. 실제 subprocess는 문자열 셸 조합 없이 인자 목록으로 실행하고 ffmpeg 버전과 인자를 기록한다.
- [ ] 새 원본은 H.264/yuv420p, 종횡비 보존·확대 없음·640×360 경계 안, 최대 24fps, AAC 모노 16kHz로 한 번 변환한다. 무음본은 그 비디오를 `-c:v copy`로 복사하고 오디오에 `volume=0`을 적용한다. 두 변형에서 동일한 오디오 인코딩 설정·구간을 사용한다. 배포 파일당 3MiB, 새 파일 합계 12MiB 상한을 검사한다.
- [ ] `validate_lesson_media.py`는 ffprobe 메타데이터, 디코딩 RGB 프레임 해시와 PTS, 오디오 sample count/rate/channels/PTS를 비교한다. 무음의 float32 파형 peak가 `1e-6` 이하이고 원본 RMS가 `1e-4`보다 큰지 확인하되 수치만으로 가청성 판정을 대신하지 않는다. 인코더 지연으로 시간축·샘플 수가 다르면 검사 허용치를 넓히지 말고 파일을 다시 만든다. 영상 신호 자체의 의도적 변경은 금지한다.
- [ ] 기존 `preflight_clip`의 512MiB 디코딩 제한과 `validate_encoded_inputs`의 4096 입력 토큰 제한을 유지한다. 새 클립의 기본 8프레임은 실제 processor로 검증한다. 4/8 샘플링에서 원본/무음 시각 입력이 같고 오디오 입력 구간 길이도 같은지 확인한다. 기존 쌍에도 정합성 감사를 수행하고 문제가 있으면 검증 완료로 표기하지 않는다.
- [ ] 카탈로그에 version, comparison_key, original/silent의 path·sha256, 출처, 변환 및 검증 보고서 해시를 넣는다. 검증 보고서는 대상 영상 해시와 측정값을 담되 catalog 해시를 역참조하지 않아 순환 해시를 만들지 않는다. 기존 키 `builtin-02321-av-pair-v1`은 유지한다. 새 키는 `builtin-scene02-av-pair-v1`, `builtin-scene03-av-pair-v1`로 한다. 영상이 바뀌면 version·키·해시·저장 결과를 함께 갱신한다.
- [ ] 경로 이탈·심볼릭 링크로 assets 밖에 도달하는 경로·중복 ID/키·잘못된 해시·같은 파일을 양쪽에 둔 카탈로그를 거부한다. torch를 import하지 않는 로더를 구현한다. 파일명만 일치하거나 업로드 파일인 경우 pair key를 발급하지 않는다.

공용 fixture는 실제 배포 카탈로그를 읽는다. 아래 검사는 등록 파일을 다른 위치에 같은 이름으로 복제해도 등록 쌍으로 인정하지 않는지를 확인한다.

```python
from pathlib import Path
import pytest
from src.lesson_media import load_media_catalog, verified_pair_config

@pytest.fixture
def media_catalog():
    project_dir = Path(__file__).resolve().parents[1]
    return load_media_catalog(project_dir)

def test_same_name_upload_is_not_a_registered_pair(tmp_path, media_catalog):
    project_dir = Path(__file__).resolve().parents[1]
    source = project_dir / media_catalog["scene02"]["clips"]["original"]["path"]
    uploaded = tmp_path / source.name
    uploaded.write_bytes(source.read_bytes())
    assert verified_pair_config(media_catalog, uploaded, project_dir=project_dir) == {}
```

- [ ] 해당 검사를 먼저 작성해 새 모듈 부재의 실패를 확인하고 구현 후 통과시킨다. 잘못된 해시와 프레임/PTS가 달라진 짝도 별도 실패 사례로 추가한다. CPU 단위 검사와 실제 미디어 감사를 구별해 다음을 실행한다. FFmpeg가 없으면 미디어 감사를 통과했다고 기록하지 않는다.

```bash
python -m pytest tests/test_lesson_media.py tests/test_classroom_safety.py -q
python scripts/validate_lesson_media.py --catalog assets/lesson_media.json --out assets/lesson_media_validation.json
```

- [ ] 영상 네 파일·출처·카탈로그·검증 근거를 함께 검토하고 `feat: add verified classroom video pairs`로 커밋한다.

## Task 2: 한국어 질문과 변경 조건을 고정하기

**Files:** 신규 `src/lesson_scenarios.py`, `tests/test_lesson_scenarios.py`.

**Interfaces:** `make_plan`, `plan_key`, `snapshot_request`. catalog 레코드를 deep copy하고 상대 경로·해시를 steps에 담는다. 실제 바이트의 최종 확인은 실행 직전 Task 3에서 수행한다.

- [ ] 영상·질문 변경에 따라 실행 키가 달라지고 요청은 동결되는 검사를 먼저 작성한다.

```python
from src.lesson_scenarios import make_plan, plan_key, snapshot_request

def test_media_and_korean_prompt_are_part_of_identity(media_catalog):
    a = make_plan("sound_vs_silence", catalog=media_catalog)
    b = make_plan("sound_vs_silence", catalog=media_catalog, media_id="scene02")
    c = make_plan("sound_vs_silence", catalog=media_catalog, prompt_id="visual")
    assert len({plan_key(a), plan_key(b), plan_key(c)}) == 3
    request = snapshot_request(a, {"choice": "아직 모르겠다", "note": ""}, request_id="r1")
    a["common"]["prompt"] = "이 초안은 실행 조건이 아니다."
    a["media"]["version"] = 99
    assert request["plan"]["common"]["prompt"] != a["common"]["prompt"]
    assert request["plan"]["media"]["version"] != 99
```

- [ ] `python -m pytest tests/test_lesson_scenarios.py -q`의 실패를 확인한 뒤 구현한다. 미지 활동/영상/질문, A의 4/8 외 프레임, B의 8 외 프레임, 잘못된 레이어 구간을 거부한다. A에서 기본 전체 구간 외 layer_band를 넘겨도 거부한다.
- [ ] `prompt_id="custom"`일 때만 custom_prompt를 허용하고 원문 그대로 보존한다. preset과 custom_prompt를 동시에 지정하면 거부한다. 빈 입력·4000자 초과를 기존 안전 함수로 검증하며, 숫자·알파벳 고유명사를 자동 삭제하거나 영어를 한국어로 번역하지 않는다.
- [ ] plan_key는 아래처럼 실행 조건만 선택해 해시한다. 정확한 prompt와 prompt version, 양쪽 파일 해시, 모델 revision, 규칙 변경에 대한 키 변경 검사도 추가한다.

```python
import hashlib
import json

def plan_key(plan):
    identity = {key: plan[key] for key in (
        "scenario_id", "scenario_version", "comparison_type", "prompt_id",
        "prompt_version", "common", "model", "steps",
    )}
    media = plan["media"]
    identity["media"] = {key: media[key] for key in (
        "media_id", "version", "comparison_key", "clips",
    )}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] A는 caption 2단계, B는 teacher_forcing 1단계인지 검사한다. 비교 종류를 바꾸거나 A에 차단 규칙이 들어가면 실패하도록 한다. 통과 후 `feat: define Korean prompt exploration plans`로 커밋한다.

## Task 3: 실행·대조군 연결·부분 실패 보존

**Files:** 신규 `src/lesson_runtime.py`, `tests/test_lesson_runtime.py`; canonical `experiment_config`; 기존 `teacher_forcing.py`, `classroom_display.py`, `run_ledger.py` 재사용.

**Interfaces:** `execute_plan`은 members/errors/status/persistence_errors를 반환한다. `run_step(step,common,media)`는 기존 run_record 형태를 반환한다.

- [ ] 부분 실패 계약을 먼저 검사한다.

```python
from src.lesson_scenarios import make_plan, snapshot_request
from src.lesson_runtime import execute_plan

def test_second_failure_keeps_first_result(media_catalog):
    plan = make_plan("sound_vs_silence", catalog=media_catalog, media_id="scene02")
    request = snapshot_request(plan, {"choice": "", "note": ""}, request_id="r2")
    saved = []
    def run(step, common, media):
        if step["step_id"] == "silent":
            raise RuntimeError("CUDA out of memory")
        return {"run_id": "first", "config": dict(common)}
    outcome = execute_plan(request, run, saved.append)
    assert outcome["status"] == "partial"
    assert [m["run"]["run_id"] for m in outcome["members"]] == ["first"]
    assert saved == outcome["members"]
    assert outcome["errors"][0]["step_id"] == "silent"
```

- [ ] 순차 실행으로 각 성공 직후 on_member를 호출한다. 첫 계산 실패에서 중단한다. 성공 0개는 failed, 일부는 partial, 전부는 complete다. 저장 콜백의 예외는 persistence_errors에 별도로 남기고 계산된 결과를 보존한다. kernel 강제 종료의 복구는 Task 5의 즉시 저장으로 처리한다.
- [ ] 실제 runner는 각 실행 직전 카탈로그 경로·해시·현재 파일을 대조한다. `experiment_config`의 기존 `02321` 전용 조건을 `verified_pair_config`로 확장하되 `matched_control_ids`의 비교 규칙은 바꾸지 않는다. config에 media_id/media_version/comparison_key/pair_hashes를 넣을 때 두 변형에서 같은 값이어야 한다. original/silent 역할은 config가 아닌 member.step_id와 extra에 둔다. config에 role/path처럼 서로 다른 필드를 추가해 짝 검사를 무효화하지 않는다.
- [ ] A는 기존 전처리 후 `model.thinker.generate`의 답변 부분 ID를 추출하고 `decode_caption_tokens`로 복원한다. 공통 max_new_tokens와 do_sample=False를 사용한다. 기록 kind=`caption_observation`, condition=`baseline_generation`, metric=`caption_token_count`/tokens이며 캡션·원시 IDs·잘림·종료 이유를 extra에 저장한다. silent만 is_control=True다. 길이 지표를 성능 점수로 표시하지 않는다.
- [ ] B는 `teacher_forced_delta`를 그대로 호출한다. 기존 `caption_cache_key`에 실제 파일 내용·프레임·정확한 질문·상한·모델 identity를 넣고, 저장된 IDs를 수정/재토큰화하지 않는다. 동일 조건의 A 원본에서 생성한 C가 있으면 재사용하고, 입력이나 질문이 바뀌면 캐시를 재사용하지 않는다. B는 is_control=False이며 무음 run을 만들지 않는다.
- [ ] B 기록에는 기존 TF 전체 증거·원시 caption IDs·두 log-prob 배열·Δ·합계/평균·측정 dtype·생성 정보가 들어간다. 출력이 영어거나 혼합 문자여도 사후 번역하지 않는다. 빈 캡션은 오류 상태로 보존하며 빈 배열을 정상 Δ로 포장하지 않는다.
- [ ] fake model로 A의 generate 2회·차단 0회, B의 C 생성 후 같은 IDs로 채점 forward 2회, 캐시 사용 시 generate 0회를 검사한다. EOS/상한/빈 답변, 새 영상·새 질문의 cache miss, 다른 질문의 무음 run과는 짝이 되지 않음을 검사한다.
- [ ] `python -m pytest tests/test_lesson_runtime.py tests/test_teacher_forcing.py tests/test_classroom_display.py tests/test_run_ledger.py -q` 통과 후 `feat: run verified video and prompt comparisons`로 커밋한다.

## Task 4: 비교·관찰 기록과 내보내기

**Files:** 신규 `src/lesson_records.py`, `tests/test_lesson_records.py`; 공용 fixture `tests/conftest.py`.

**Interfaces:** 라이브는 comparison_id=request_id, origin=live. 저장 예제 가져오기는 새 request_id를 comparison_id로 쓰고 origin=stored, source_example_id/source_comparison_id/imported_at을 추가한다. Example.comparison의 원래 created_at·멤버 run ID·출처·수치는 보존하고 Example의 generation_provenance도 보관한다.

- [ ] 공용 fixture를 tests/conftest.py에 둔다. Task 1의 media_catalog도 같은 파일에 둔다. 합성 수치와 문장은 배포 디렉터리에 쓰지 않는다.

```python
import pytest

@pytest.fixture
def lesson_comparison(media_catalog):
    from src.lesson_scenarios import make_plan, snapshot_request
    from src.lesson_records import new_comparison
    from src.run_ledger import run_record
    plan = make_plan("sound_vs_silence", catalog=media_catalog)
    request = snapshot_request(plan, {"choice": "", "note": ""}, request_id="fixture")
    members = []
    for step in plan["steps"]:
        run = run_record(
            kind="caption_observation", condition="baseline_generation",
            metric_name="caption_token_count", metric_value=2, metric_unit="tokens",
            config={**plan["common"], "clip": step["path"].rsplit("/", 1)[-1],
                    "clip_sha256": step["sha256"], **plan["model"],
                    "media_id": plan["media"]["media_id"],
                    "media_version": plan["media"]["version"],
                    "comparison_key": plan["media"]["comparison_key"],
                    "pair_hashes": {role: clip["sha256"] for role, clip in plan["media"]["clips"].items()},
                    "helper_sha256": "0" * 64, "repo_revision": "2" * 40,
                    "repo_dirty": False, "runtime_packages": {}, "python": "3.11.0"},
            is_control=step["step_id"] == "silent",
            extra={"caption_text": "테스트용 문장", "caption_ids": [1, 2],
                   "generation_truncated": False, "generation_end_reason": "eos"},
        )
        members.append({"step_id": step["step_id"], "run": run})
    outcome = {"members": members, "errors": [], "status": "complete", "persistence_errors": []}
    return new_comparison(request, outcome, created_at="2026-10-08T00:00:00Z")
```

- [ ] 메모 변경이 증거를 바꾸지 않는 검사를 먼저 작성한다.

```python
import json
from src.lesson_records import update_reflection, build_lesson_evidence_json

def test_reflection_keeps_original_evidence(lesson_comparison):
    before = json.dumps(lesson_comparison, ensure_ascii=False, sort_keys=True)
    note = "<script>alert(1)</script> | 관찰\n```"
    updated = update_reflection(lesson_comparison, {"observed": note})
    assert updated["comparison_id"] == lesson_comparison["comparison_id"]
    assert updated["members"] == lesson_comparison["members"]
    assert json.dumps(lesson_comparison, ensure_ascii=False, sort_keys=True) == before
    exported = json.loads(build_lesson_evidence_json([], [updated]))
    assert exported["lessons"][0]["reflection"]["observed"] == note
```

- [ ] update_reflection은 다섯 필드만 허용하고 미지 필드는 거부한다. 미제공 값은 보존한다. HTML은 html.escape, Markdown은 원문보다 긴 fence와 표 셀 이스케이프를 사용한다. 모델 출력·사용자 입력의 JSON 원문은 보존한다.
- [ ] save_comparison은 `{state,path,error}` save_status를 추가한 사본을 반환한다. JSONL은 comparison_id별 마지막 기록으로 복구한다. 끝의 끊긴 행은 경고와 원본을 보존하고 무시한다. 중간 행의 손상도 위치를 보고하고 이후 유효 기록을 잃지 않는다. 쓰기 실패 시 메모리 사본을 유지한다.
- [ ] 저장 예제를 단순히 다시 보기만 할 때는 새 관찰 기록을 만들지 않는다. `이 결과로 관찰 기록하기`로 가져오면 새 ID를 발급한다. 같은 source_example_id로 다시 시작해도 앞선 메모를 덮어쓰지 않는다. 저장 원본 파일은 읽기 전용으로 취급한다.
- [ ] build_lesson_evidence_json은 기존 build_evidence_json 반환 객체의 schema_version/runs/provenance를 유지하고 lesson_schema_version=1과 lessons를 추가한다. 저장 증거는 lessons 안에 두고 라이브 runs에 새 실행으로 추가하지 않는다. 고급 실험 runs도 보존한다.
- [ ] HTML은 A의 두 캡션, B의 같은 C와 변화 방향을 구분한다. 완료는 계산 완료일 뿐 가설 지지를 뜻하지 않는다. 원본/무음 이름 옆에 실제 영상·프롬프트·프레임·출처·잘림을 보여 준다. 다른 C의 Δ를 영상/질문 간 성능 순위로 그리지 않는다.
- [ ] 저장 실패·중복 ID·재연결·저장 예제 반복 가져오기·partial/failed·이스케이프·기존 영어 export와 runs 동일성을 검사한다. `python -m pytest tests/test_lesson_records.py tests/test_run_ledger.py tests/test_run_ledger_evidence.py -q` 통과 후 `feat: preserve comparison evidence and Korean observations`로 커밋한다.

## Task 5: 단계적으로 펼치는 화면과 실행 경계

**Files:** canonical, HTML 생성기·생성본, 신규 `tests/test_lesson_notebook.py`, 기존 실행/idle/parity 검사.

**Interfaces:** `lesson_controls`, `lesson_session`, `lesson_execute`, `lesson_result`, `lesson_reflection`, `lesson_downloads` 셀. 상태는 UI 재생성에 의존하지 않고, 표시·메모 셀만 기록 getter를 읽는다.

- [ ] 기존 AST 셀 추출/fake runner 패턴으로 최초 로드·활동/영상/질문 변경·직접 입력·예상/메모 편집·다운로드의 추가 추론이 0인지 검사한다. A의 명시적 제출은 두 단계, B는 한 TF 단계만 호출해야 한다.
- [ ] 화면 상단은 활동·장면 선택, 수동 재생 미리보기, 바뀌는 것/고정되는 것, 예상(선택), 실행/저장 결과 버튼, 마지막 결과, 관찰 기록 순서다. 한글 질문 변경과 직접 입력·프레임·구간은 `직접 바꿔 보기`에서 펼친다. 활동 선택에는 추천 순서 A→B를 표시한다.
- [ ] 프롬프트 프리셋을 바꿔도 실제 시청각 입력은 유지됨을 알려 준다. 복수 설정을 바꾸면 이전 실행과 달라진 항목을 나열한다. 질문 변경은 한 쌍 양쪽에 동일하게 적용한다. 새 초안은 이전 결과의 제목이나 캡션을 바꾸지 않는다.
- [ ] marimo 0.25.0의 카운터 버튼을 사용한다. run_button.value의 일시적인 bool을 클릭 ID로 쓰지 않는다.

```python
lesson_run = mo.ui.button(
    value=0, on_click=lambda count: count + 1,
    label="선택한 조건으로 실행",
)
# lesson_session은 UI와 기록 getter에 의존하지 않는 객체다.
# last_click=0, outcomes={}, last_result=None, busy=False를 보관한다.
click = lesson_run.value
if click > session.last_click and not session.busy:
    session.last_click = click
    request = snapshot_request(plan, prediction, request_id=str(uuid.uuid4()))
    outcome = execute_plan(request, run_step, on_member=persist_member)
```

이 코드는 실행 경계의 핵심이다. `plan/prediction`은 controls 초안, `run_step`은 Task 3 콜백, uuid는 표준 라이브러리다. 실제 구현에서 busy 설정/해제는 try/finally로 하고 실행 중 버튼을 비활성화한다. 유효하지 않은 요청도 클릭을 소비하며 수정 후 다시 눌러야 한다. 동일 클릭의 재평가는 재실행하지 않는다.

- [ ] persist_member는 append_run과 같은 comparison_id의 partial 기록을 성공 직후 저장한다. Task 3 종료 후 complete/partial/failed로 갱신한다. A의 두 번째 계산 중 kernel 종료 뒤 복구해도 첫 run과 비교 연결이 남아야 한다. 복구한 partial은 자동 재실행하지 않는다.
- [ ] 활동/영상/질문을 바꿔도 버튼 카운터와 세션 객체를 재생성하지 않는다. 관찰 메모는 comparison_id에 연결한다. 다른 결과를 열었다 돌아와도 미저장 메모를 유지한다. 저장 결과 보기와 가져오기는 라이브 실행 버튼과 다른 동작이다.
- [ ] 기존 고정 로짓 렌즈·guided_captions·guided_tf_panel은 별도 `시범 결과 계산` 제출로 실행한다. idle 반환값은 명시적으로 초기화한다. band_result_panel은 필요한 기준선이 없으면 계산 순서 안내를 표시하며 몰래 forward하지 않는다. 기존 고급 폼의 명시적 제출 기능은 유지한다.
- [ ] 라이브의 기존 모델 준비 단계는 유지하고 로딩과 추론을 구별한다. CPU 진입점은 `CTP49906_REPLAY=1`이며 새 모듈 import와 저장 결과 보기에 모델 로드가 없어야 한다. 라이브에서 저장 결과만 열어도 추가 추론은 0이다.
- [ ] 셋업의 필수 파일 확인에 새 lesson 헬퍼 5개·카탈로그·세 영상 쌍·출처·manifest를 포함하고, 다운로드한 manifest가 가리키는 18개 payload도 검사한다. 기존 소스 해시 계산이 신규 헬퍼를 포함하는지 확인한다.
- [ ] 생성기에서 anchor·안내·폼을 생성한다. 계산 셀의 canonical/native AST parity를 유지하고 예외 범위를 넓히지 않는다. 820px/1280px와 키보드로 선택·실행·기록을 확인한다.
- [ ] `python scripts/build_molab_html.py`, strict marimo 검사, lesson_notebook·molab_html·studio_idle/execution 검사를 통과시킨 후 `feat: add guided Korean exploration to the notebook`로 커밋한다.

## Task 6: 18개 실제 저장 결과와 CPU 탐색

**Files:** 신규 `src/lesson_examples.py`, `scripts/generate_lesson_examples.py`, `tests/test_lesson_examples.py`, `precomputed_kr/lessons/manifest.json` 및 `examples/<example_id>.json`; Task 5 보기 연결.

**Interfaces:** 기본 로더는 plan_key가 정확히 맞는 Example envelope를 반환한다. 일치 결과가 없으면 KeyError, 손상/잘못된 형식은 ValueError다. 명시적 example_id로 다른 조건을 열 때만 원래 조건을 표시하며 example_differences로 차이를 알린다. 조용한 대체는 금지한다.

- [ ] manifest 구조는 schema_version=1, coverage(2활동·3영상·3질문·기본 조건), examples(각 example_id·plan_key·상대 파일 경로·payload SHA-256·생성 provenance)다. 조회 시 manifest 스키마와 해당 파일을 검증하고, 다른 파일 한 개의 손상이 정상 결과 읽기를 막지 않게 한다. 배포 전에는 전체 묶음을 검증한다.
- [ ] 임시 비교 하나를 정상 형식으로 쓰는 아래 helper를 테스트 파일 안에 정의한다. 부분 묶음도 개발 중 읽을 수 있지만 생성 스크립트의 배포 검사에서는 coverage 18개와 synthetic_test_only 부재를 필수로 확인한다. 테스트 helper는 production 코드에서 import하지 않는다. 이어지는 키 혼동 검사를 먼저 만든다.

```python
import hashlib
import json
import uuid
from src.lesson_scenarios import plan_key

def write_test_bundle(directory, comparisons):
    entries = []
    for comparison in comparisons:
        plan = comparison["submitted_plan"]
        key = plan_key(plan)
        provenance = {
            "generation_id": str(uuid.uuid4()), "helper_sha256": "0" * 64,
            "generator_sha256": "1" * 64, "model": plan["model"],
            "created_at": comparison["created_at"], "repo_revision": "2" * 40,
            "repo_dirty": False, "packages": {}, "python": "3.11.0",
            "synthetic_test_only": True,
        }
        identity = {"plan_key": key, "provenance": provenance}
        example_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        payload = {"schema_version": 1, "example_id": example_id, "plan_key": key,
                   "generation_provenance": provenance, "comparison": comparison}
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        name = "examples/" + example_id + ".json"
        (directory / "examples").mkdir(exist_ok=True)
        (directory / name).write_bytes(data)
        entries.append({"example_id": example_id, "plan_key": key, "file": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "generation_provenance": provenance})
    manifest = {"schema_version": 1, "coverage": {"plan_keys": [e["plan_key"] for e in entries]},
                "examples": entries, "synthetic_test_only": True}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
```


```python
import pytest
from src.lesson_examples import load_lesson_example
from src.lesson_scenarios import make_plan

def test_no_silent_fallback_for_another_video(tmp_path, media_catalog, lesson_comparison):
    write_test_bundle(tmp_path, [lesson_comparison])
    other = make_plan("sound_vs_silence", catalog=media_catalog, media_id="scene03")
    with pytest.raises(KeyError):
        load_lesson_example(tmp_path, other)
```

- [ ] 정상 파일 읽기·payload 1바이트 변조·경로 이탈·미지 schema·중복 plan_key·미완료 status·media/prompt/config 불일치를 검사한다. 요청한 키와 manifest와 payload의 submitted_plan이 모두 일치해야 한다. provenance의 helper/model/생성 스크립트 해시도 manifest와 payload에서 일치해야 한다.
- [ ] TF는 caption_ids의 길이를 기준으로 배열·토큰 종류 길이와 유한성을 확인한다. per-token Δ는 knockout−baseline과 `rtol=1e-5, atol=1e-5`로 비교한다. 합계/평균은 기록된 dtype(float32)의 반올림을 반영해 재계산하고 `rtol=1e-5, atol=1e-4`로 확인한다. 원시 수치를 고쳐 맞추지 않는다. 큰 합계의 정상 float32 반올림 사례와 명백한 Δ 변조가 각각 통과/실패하는 검사를 둔다. 이 허용치는 수학적 파일 정합성 검사이며 GPU 재실행 간 동일성 허용치가 아니다.
- [ ] 생성 스크립트는 make_plan/runner/execute_plan/new_comparison을 사용해 각 영상×질문에서 A 다음 B를 실행한다. eager 모델 하나를 재사용한다. A 원본의 C를 같은 키의 B에서 재사용할 수 있으나 실제 generation_from_cache를 남긴다. 서로 다른 질문/영상의 C는 공유하지 않는다.
- [ ] 실제 GPU에서 18개 complete 결과를 생성한다. helper/스크립트 해시·Git revision/dirty 상태·모델 revision·패키지·UTC 생성 시각을 실제 값으로 기록한다. 노트북 밖에서 생성했다면 notebook hash를 꾸며 넣지 않는다. 전체 검증 뒤 manifest를 마지막에 원자적으로 저장하고, 중간 실패 결과는 출시 묶음 밖에 보존한다.

```bash
python scripts/generate_lesson_examples.py --out-dir precomputed_kr/lessons
python -m pytest tests/test_lesson_examples.py tests/test_lesson_notebook.py -q
```

- [ ] 18개를 사람이 읽고 잘림·외국어 출력·소리 언급·같거나 다른 답변을 기록한다. 기대한 방향의 결과만 고르지 않는다. 잘림이 반복돼 기본 상한을 바꾸면 두 활동 정의와 질문 조건·버전을 갱신해 전체 묶음을 다시 만든다. 일회성 낯선 결과도 근거 없이 삭제하지 않는다.
- [ ] 가중치 로더/추론을 호출하면 예외가 나도록 한 CPU 재생에서 18개 조합과 가져오기·메모·다운로드를 검사한다. custom·4프레임·부분 레이어에는 정확한 저장 결과 없음과 라이브 필요를 표시한다. 다른 조합의 기본값으로 자동 되돌리지 않는다.
- [ ] 실제 payload만 `feat: ship verified Korean lesson result matrix`로 커밋한다. GPU 접근이 없으면 로더 개발 완료와 실제 자료 생성 미완료를 구분한다.

## Task 7: 한국어 설명·워크시트·교사용 흐름

**Files:** canonical 도입·안내, 생성기·생성본, `WORKSHEET_kr.md`, `CLASSROOM_GUIDE_kr.md`, `README_kr.md`, `MOLAB_HTML_QA.md`.

**Interfaces:** 제출 파일명 `lab_log.md`, `lab_evidence.json`은 유지한다. 기본 관찰 기록과 자동 출처 부록, 기존 고급 runs를 함께 포함한다.

- [ ] 첫 개요는 `보고 듣는 정보 → 질문에 따른 답변 → 조건을 바꿔 관찰 → 내부 연결 탐색` 순서로 쓴다. RMSNorm·로짓 렌즈·generated/answer 용어는 심화 활동 가까이 둔다. 두 활동의 차이와 질문 변경≠입력 제거를 화면 가까이 설명한다.
- [ ] 한국어판 노트북·워크시트·교사용 필수 안내에서 영어 비교와 언어별 예상 효과 방향을 제거한다. 교사용 참고에 영어 비교를 다시 필수 절차로 남기지 않는다. 별도 영어판 기능과 데이터는 유지한다.
- [ ] 워크시트는 선택한 활동/영상/질문과 다섯 관찰 필드로 구성한다. 질문을 직접 고쳐 봤는지, 예상과 다른 점은 무엇인지, 다음에 바꿀 한 가지는 무엇인지 묻는다. 해시 수기 전사와 기본 활동의 supported/refuted 필수 입력을 제거한다.
- [ ] 추천 최소 경로를 안내하고 18개 전체 수행을 요구하지 않는다. 실제 출력이 같거나 Δ가 작아도 관찰 가치가 있음을 설명한다. 모델 출력 언어를 자동 교정하지 않는 이유와 저장 결과의 범위를 적는다.
- [ ] README의 영어 위젯 유지 설명과 `묶음가` 오류를 고친다. 세 영상 출처 링크·원본/무음 제작 사실·저장 18개 조합·라이브 전용 조건을 문서와 화면에서 일치시킨다. 출처 파일·원문 제목을 모델 prompt에 넣지 않는다.
- [ ] 교사용 안내에 영상 미리보기/볼륨 확인, GPU 준비, CPU 전환, 부분 실패 복구, 종료 전 다운로드 확인을 넣는다. 비전공자 3~5명 시범 활동 양식에는 두 활동의 차이·동일 답변 해석·다음 변수 선택을 기록한다. 시행 전에는 완료라고 쓰지 않는다.
- [ ] 문자열마다 새 테스트를 만들지 않는다. 기존 문구/노트북 계약 검사, 생성본 동기화, 실제 화면 읽기와 `git diff --check`를 수행하고 `docs: align Korean lessons with video and prompt exploration`로 커밋한다.

## Task 8: 통합 검증과 배포 조합 확인

**Files:** `MOLAB_HTML_QA.md`, canonical REPO_REF·생성본·README의 배포 참조. 실제 점검 산출물은 `outputs/molab-guided-exploration-20261008/`에 둔다.

**Interfaces:** 게시할 notebook과 그 notebook이 받는 helpers/media/examples 커밋의 조합을 검증한다. 생성한 자료의 과거 provenance를 새 배포 커밋으로 덮어쓰지 않는다.

- [ ] 실제 Python 3.11+·marimo 0.25.0 환경에서 다음을 실행한다. 이전 검증 환경은 저장소 루트 `.venv/bin/python` 및 `/private/tmp/ctp49906-review-deps-20261008`이며 경로가 없으면 호환 환경을 준비하고 기록한다. 오래된 tests/README 버전 안내를 그대로 사용하지 않는다.

```bash
python -m pytest -q
python -m marimo check --strict CTP49906_avllm_molab_kr.py CTP49906_avllm_molab_html_kr.py
python scripts/build_molab_html.py --check
python scripts/build_studio_bundle.py --check
python scripts/validate_lesson_media.py --catalog assets/lesson_media.json --check
git diff --check
```

- [ ] canonical/native 계산 셀 parity와 별도 영어판의 기존 CPU 재생을 확인한다. 테스트가 통과한 뒤 새 변경·실패·우려가 없으면 전체 검사를 반복하지 않는다.
- [ ] 실제 Molab GPU와 별도 CPU 세션에서 아래를 점검한다. 호출 횟수 hook은 계산을 바꾸지 않는다. 고의 OOM은 학생 환경에서 재현하지 않고 fake 실패 주입으로 검사한다.

| 조작 | 통과 기준 |
|---|---|
| 새 사본 설치·Run all | 영상·문서·18개 저장 결과 파일 확보. 숨겨진 시범 추론 0. 로딩과 추론 구별 |
| 활동·영상·세 한국어 질문·직접 입력 편집 | 추가 forward 0, 이전 결과 설정·메모 보존 |
| 세 영상 원본/무음 미리보기 | 수동 재생, 정상 영상·가청 원본·무음 확인, 출처 접근 가능 |
| 18개 기본 조합 실제 실행 | 모두 complete, 조건·파일 해시·IDs·출처 정확. 예상 효과 방향은 통과 기준 아님 |
| 한 한국어 직접 질문으로 A/B 실행 | 정확한 원문 prompt 사용, 이전 질문 캐시 재사용 없음 |
| A 4프레임·B 앞/중간/뒤 구간 | 허용 설정만 적용, 변경 설정 표시, 같은 C 채점과 입력 비교 구별 |
| 무음 단계 실패·재연결 | 첫 결과/비교 ID·메모 복구, partial, 자동 재실행 없음 |
| 메모·세부 보기·저장 결과 가져오기·다운로드 | 추가 추론 0, 다른 관찰 기록/생성 출처 불변 |
| CPU 재생의 18개 선택 | 모델 가중치·GPU 없이 읽기/기록/내보내기, 라이브 전용 조건은 정확한 안내 |
| 저장 파일 하나 손상 | 해당 결과만 거부, 다른 정상 결과 사용 가능 |
| 내려받은 두 파일 다시 열기 | 화면의 ID·질문·영상·캡션·출처·메모·raw TF 배열과 일치 |
| 820px/1280px·키보드 | 한국어 문장이 잘리지 않고 비교·도움말·입력·버튼 조작 가능 |

- [ ] 생성 스크립트의 GPU 검사와 실제 Molab UI 검사를 구별해 기록한다. UI 점검은 생성 때 기대한 출력 문자열과 완전히 같은지를 강요하지 않는다. 실행 조건·완료·증거 정합성과 실제 사용 가능성을 확인한다.
- [ ] 비전공자 시범 활동은 교사와 수행하고 막힌 위치·오해·수정 사항을 QA에 남긴다. 학생 확인이 없으면 그 제한을 명시한다. 코드 검사만으로 교육적 결함이 없다고 결론 내리지 않는다.
- [ ] helpers·영상·카탈로그·18개 결과를 포함한 커밋 H를 먼저 만든다. canonical REPO_REF를 H로 바꾸고 native를 생성한 뒤 README와 파일 SHA-256을 갱신한다. 생성본 직접 편집은 금지한다.
- [ ] 빈 디렉터리에서 최종 notebook의 셋업 함수로 H를 받아 helper/media/example 바이트를 대조하고 CPU 저장 활동을 실행한다. 새 API와 과거 pin이 섞이지 않아야 한다. 최종 notebook+pin 조합의 Molab 확인을 QA에 남긴다.
- [ ] 구현·게시 요청이 이어질 때 기존 사용자 권한 범위에 따라 GitHub에 반영하고 remote main/파일 해시를 확인한다. 현재 계획 검토 단계에서는 push하지 않는다.

## 단계별 완료 판정

| 단계 | 완료 증거 |
|---|---|
| 계획 검토 | 새 영상·한국어 질문·저장 범위를 반영한 설계/계획과 검토 기록 |
| 영상 준비 | 출처가 확인된 세 쌍, 새 파일 4개, 디코딩/시간축/가청성 검사 |
| CPU 개발 검증 | Task 1~5의 조건·캐시·대조 연결·부분 실패·기록·실행 경계 검사 |
| 저장 자료 준비 | 실제 GPU의 18개 complete 비교와 전체 manifest 검증 |
| 수업용 배포 검증 | 최종 notebook+pin의 Molab 조작·다운로드·CPU 탐색 |
| 학습 사용성 확인 | 비전공자 시범 활동의 관찰·막힘·수정 기록 |

## 문서 자체 검토

- [x] 새 영상 2종을 후속 과제가 아닌 첫 배포 작업·검사·완료 조건으로 옮겼다.
- [x] 추천 질문을 한국어 3종으로 정하고 직접 입력·캐시·내보내기·CPU 범위를 연결했다.
- [x] 영어 비교는 한국어 수업 범위에서 제외하고 별도 영어판 호환성만 유지했다.
- [x] 활동 ID만으로 파일/결과를 재사용하는 위험과 무음 대조 키의 하드코딩을 해결할 계약을 정했다.
- [x] A의 입력 비교와 B의 같은 C 채점, 질문 변경과 입력 제거의 차이를 명시했다.
- [x] 영상 검증·18개 GPU 결과·Molab 확인·학생 활동을 이미 끝낸 것처럼 쓰지 않았다.
- [x] 파일 책임·인터페이스·Task 번호·검사·설계의 전역 제약을 대조했다.

**실행 방식:** 현재는 계획 검토, 후속 요청의 10초 영상 파일 준비 및 기존 노트북 드롭다운 연결까지 수행했다. 전체 계획 구현 요청이 이어지면 한 작업자가 Task 1의 남은 확인부터 진행한다. 미디어 직접 청취·processor/GPU 검사와 위에 표시한 나머지 기능은 아직 남아 있다.
