# Counterpoint Lens AVLLM 수업 노트북

<p align="center"><a href="README.md">English</a> · <b>한국어</b></p>

`CTP49906_avllm_molab.py`는 *Do Audio-Visual Large Language Models Really See
and Hear?*의 Qwen2.5-Omni 해석가능성 코드를 바탕으로 만든 Marimo 수업
스튜디오입니다. 시청각 제작·수정 경로와 제한된 로짓 프로브, 직접 어텐션
엣지 개입, 티처 포싱 답변 분포 관찰을 결합합니다.

그럴듯한 캡션이 모델의 “진짜 보기/듣기”를 증명한다고 가정하지 않고,
현재 근거가 어디까지 말할 수 있는지를 묻습니다.

## 필수 수업 경로

렌더링된 노트북은 네 부분과 여덟 단계로 구성됩니다.

1. **프로젝트 준비**
   - 안내 — 만들고, 시험하고, 수정하기
   - 구성 — 첫 컷 계획하기
   - 등록 — 첫 컷(V1)
2. **가이드 시연**
   - 관찰 — 공통 기준
3. **탐색 플레이그라운드**
   - 실험 — 한 번에 하나만 바꾸기
   - 비교 — 한 작품의 세 가지 읽기
   - 수정 — 설명과 두 번째 컷(V2)
4. **종합 및 아키텍처 과제**
   - 종합 — 포트폴리오와 제한된 제안

필수, 선택, 고급 활동을 눈에 띄게 구분합니다. 폼은 제출할 때 입력을
스냅샷하며 초안을 편집하는 것만으로 실행이나 성찰이 확정되지 않습니다.

## 저장 재생으로 시작

첫 컨트롤에서 **저장된 수업 재생**을 선택하세요. `precomputed/`의
체크섬 결합 파일을 사용하므로 GPU와 모델 다운로드가 필요 없습니다.
라이브 전용 기능은 사용할 수 없는 이유와 함께 비활성화됩니다.

불변 모델·자극 ID, 산출물 체크섬, 생성 명령, 후보 배포 한계는
[`../study_materials/wp6/replay_manifest.json`](../study_materials/wp6/replay_manifest.json)에
있습니다. 현재 CPU 검증은 커밋된 바이트를 확인하지만 새 GPU 재생성을
주장하지 않습니다.

```bash
uvx --python 3.10 marimo@0.23.14 edit CTP49906_avllm_molab.py
```

라이브 경로는 스크립트 헤더의 고정 의존성을 사용하며 교수자가 승인한
자원 범위의 CUDA 런타임이 필요합니다. 첫 실행은 Qwen2.5-Omni 가중치를
다운로드합니다. 배포 환경을 리허설하지 않았다면 라이브 실행을 필수로
만들지 마세요.

## 근거 읽기

- **원시 프로브 점수 분산**은 레이어/위치 진단이며 보정된 불확실성이나
  중간 다음 토큰 예측이 아닙니다.
- **캡처 어텐션 질량**은 서술 값입니다. 마스킹은 어텐션을 기계적으로
  재분배하며 어텐션 자체를 인과 설명으로 만들지 않습니다.
- **직접 어텐션 엣지 녹아웃**은 선택 레이어 구간에서 source 쿼리에서
  target 키로 가는 엣지를 막습니다. 모달리티 제거가 아닙니다.
- **티처 포싱 답변 분포 변화**는 고정 답변에 대한 직접 엣지 개입 점수이며
  자유 생성 불확실성이 아닙니다.
- 위치 비교에는 호환되는 토큰 레이아웃 지문이 필요합니다. 호환되지 않으면
  노트북이 비교를 막거나 낮은 수준으로 제한합니다.

모든 해석은 관찰, 제한된 추론, 한계, 경쟁 설명, 다음 대조군을 구분해야 합니다.

## 블라인드 청중 비교

창작자는 허용 목록 제시 패킷을 내보냅니다. 응답자는 별도 GPU 없는
[청중 화면](../audience/CTP49906_audience_response_molab.py)을 사용합니다.
공개 전에는 창작 의도, 조건, 개인 ID, 모델 출력이 없습니다.

창작자 노트북은 같은 교환의 서로 다른 스키마 유효 블라인드 응답 두 개를
받은 뒤 창작자/기계 읽기를 공개합니다. 비교는 일치, 차이, 빠진 읽기를
기록하며 일치를 정답으로, 두 응답자를 연구 표본으로 취급하지 않습니다.

## 데이터, 접근, 복구

수업 모드가 기본값입니다. 노트북은 학생 작품, 과정 기록, 청중 응답을
교수자에게 자동 전송하지 않습니다. 사용자가 시작한 개인 다운로드와
권한이 유효한 청중 패킷은 별도 egress입니다. 유효한 승인 설정과 독립
권한이 없으면 연구 내보내기는 사용할 수 없습니다.

자막, 전사, 설명, 키보드 검사, 교수자/학생 경로, 개인정보 필드, 문제 해결,
이중언어 자료는 [수업 안내 묶음](../study_materials/wp6/README.md)에 있습니다.
이 지원은 사람이 수정할 수 있으며 몰래 모델 입력으로 바뀌지 않습니다.

기존 [WORKSHEET.md](WORKSHEET.md)는 수업 메모로 남습니다. 새 활동은
노트북의 버전 있는 준비 → 실행 → 성찰 기록을 사용합니다.
[워크시트 전환 안내](../study_materials/wp6/worksheet_migration.ko.md)를 보세요.

## 라이브 실험 명령

`requirements.txt` 설치 뒤 원래 명령줄 실험도 사용할 수 있습니다.

```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4

python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

## 배포 상태

이 저장소는 수업 후보 자료이며 research-ready 배포가 아닙니다. 접근성·
현지화, 라이선스 자극, 청중 타당도, 도구 품질, 윤리·데이터 거버넌스,
비교 충실도, 불변 강좌 태그, 파일럿 기준은 담당 사람의 검토가 필요합니다.

## 인용

```bibtex
@misc{selvakumar2026audiovisuallargelanguagemodels,
  title={Do Audio-Visual Large Language Models Really See and Hear?},
  author={Ramaneswaran Selvakumar and Kaousheik Jayakumar and S Sakshi and Sreyan Ghosh and Ruohan Gao and Dinesh Manocha},
  year={2026},
  eprint={2604.02605},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2604.02605}
}
```
