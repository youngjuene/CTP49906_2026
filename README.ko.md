# CTP49906 시청각 해석가능성 스튜디오

<p align="center"><a href="README.md">English</a> · <b>한국어</b></p>

이 저장소는 시청각 작품을 만들고, 살펴보고, 비교하고, 수정하는 예술 통합
수업 경로를 제공합니다. 주 처치 노트북은 **Counterpoint Lens**이며, 별도의
GPU 없는 청중 화면이 사람의 해석을 블라인드 상태로 수집합니다. 중립적인
제작, 세션, 청중, 내보내기 계약은 `curriculum_common/`에 있습니다.

## 수업 시작 위치

- [Counterpoint Lens 노트북](avllm_interpretability/CTP49906_avllm_molab.py)
  — 저장된 가이드 재생과 선택적인 라이브 해석가능성 탐구.
- [블라인드 청중 응답](audience/CTP49906_audience_response_molab.py)
  — GPU나 모델 없이 허용 목록 제시 패킷을 검증하고 독립 읽기 하나를
  다운로드합니다.
- [영문/한글 수업 안내](study_materials/wp6/README.md) — 교수자 운영 안내,
  학생 빠른 시작, 워크시트 전환, 개인정보·데이터 사전, 문제 해결,
  재생 ID, 시각 자료 대안.
- [AVLLM 기술·수업 안내](avllm_interpretability/README.ko.md).
- [Jacobian Lens 보조 자료](jacobian-lens/README.md#classroom-marimo-demo).

## 안전한 빠른 시작

필수 경로는 **저장된 수업 재생**으로 시작합니다. 커밋된 출력을 사용하므로
GPU를 할당하거나 모델 가중치를 다운로드하지 않습니다.

```bash
uvx --python 3.10 marimo@0.23.14 edit \
  avllm_interpretability/CTP49906_avllm_molab.py
```

블라인드 청중 화면은 별도로 엽니다.

```bash
uvx --python 3.10 marimo@0.23.14 run \
  audience/CTP49906_audience_response_molab.py
```

수업 모드는 기본값이자 실패 시 안전 모드입니다. 노트북은 학생 작품,
과정 기록, 청중 응답을 교수자에게 자동 전송하지 않습니다. 사용자가
컨트롤을 눌렀을 때만 로컬 개인 다운로드나 권한이 유효한 블라인드 패킷이
생성됩니다.

## 재생 및 배포 상태

[재생 매니페스트](study_materials/wp6/replay_manifest.json)는 불변 모델
revision, 코드화 자극 체크섬, 산출물 해시, 정확한 GPU 생성 명령을
공개합니다. 현재 묶음은 수업 후보 재생입니다. 이 CPU 전용 변경은
바이트를 검증하지만 GPU 출력을 재생성했다고 주장하지 않습니다.

소프트웨어와 자료는 **research-ready가 아닙니다**. 기관 거버넌스, 자극
라이선스, 도구 품질, 비교 충실도, 청중 타당도, 접근성·현지화 검토, 불변
강좌 태그, 파일럿 기준은 담당 사람이 승인해야 하는 게이트입니다.

## 개발 검사

```bash
python3.10 -m py_compile audience/*.py curriculum_common/*.py
uvx --python 3.10 marimo@0.23.14 check --strict \
  audience/CTP49906_audience_response_molab.py
PYTHONPATH=. uvx --python 3.10 pytest -q tests
uvx --python 3.10 ruff check audience curriculum_common tests
```

라이브 AVLLM 경로의 CUDA/모델 의존성은
[avllm_interpretability/README.ko.md](avllm_interpretability/README.ko.md)에
설명되어 있습니다.
