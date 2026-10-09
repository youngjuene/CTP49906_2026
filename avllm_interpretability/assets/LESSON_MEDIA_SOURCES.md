# 한국어 수업용 추가 영상

준비일: 2026-10-08. 사용자 요청에 따라 기존 소 영상 `02321.mp4`의 비디오 길이 **10.000초**에 맞췄다. 새 영상 두 개의 소리 포함본과 무음본을 준비했다. 2026-10-09에는 한국어 노트북의 미리보기·다양성·티처 포싱 드롭다운에 연결했다. 새 영상의 실제 모델 추론은 아직 수행하지 않았다.

## 파일과 출처

| 수업 파일 | 원천 영상의 구간 | 제작자 | 출처와 이용 조건 |
|---|---|---|---|
| [scene02.mp4](scene02.mp4), [scene02_silent.mp4](scene02_silent.mp4) | 75.000–85.000초 | tabladrumsonline | [Tabla drums demo](https://commons.wikimedia.org/wiki/File:Tabla_drums_demo.webm), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| [scene03.mp4](scene03.mp4), [scene03_silent.mp4](scene03_silent.mp4) | 5.000–15.000초 | Ата | [Rain in Rivne 1](https://commons.wikimedia.org/wiki/File:Rain_in_Rivne_1.webm), [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

타블라 영상은 [제작자의 원 게시물](https://www.youtube.com/watch?v=j0okP-Gq13s)에서 유래한 Commons 파일이다. 출처와 이용 조건은 [Commons 설명의 고정 판본](https://commons.wikimedia.org/w/index.php?title=File:Tabla_drums_demo.webm&oldid=1037829467)을 참고했다. 비 오는 장면은 [Commons 설명의 고정 판본](https://commons.wikimedia.org/w/index.php?title=File:Rain_in_Rivne_1.webm&oldid=609848589)을 참고했다.

타블라 파생 파일에는 CC BY 3.0을 적용하고 저작자·출처·라이선스·아래 변경 내역을 함께 표시한다. 비 오는 장면의 파생 파일은 CC0 1.0으로 제공한다. 이 영상들에 저장소 코드 라이선스를 자동 적용하지 않는다. `02321.mp4`와 `02321_silent.mp4`는 기존 자료이며 이번 작업에서 수정하지 않았다. 기존 영상의 원천 권리 확인을 새 두 영상의 출처 확인과 혼동하지 않는다.

## 변경 내역

- 재생 속도는 유지하고 지정한 10초 구간만 추출했다.
- 타블라: 480×360, 24fps, H.264/yuv420p. 손과 악기가 보이는 연주 구간의 샘플 프레임 20개를 확인했다.
- 비 오는 장면: 636×360, 24fps, H.264/yuv420p. 원본 장면을 잘라내지 않고 크기를 줄였다. 인코딩을 위해 가로·세로를 짝수 픽셀로 맞췄다.
- 원래 오디오를 모노 16kHz로 변환하고 AAC로 저장했다. 다른 소리를 합성하거나 추가하지 않았다.
- 무음본은 같은 비디오 스트림을 복사하고 오디오 신호만 0으로 만들었다. 오디오 트랙은 유지했다.
- 자막·정답 설명·로고를 새로 넣지 않았다. 기존 소 영상에 맞추는 기준은 재생 길이이며 서로 다른 장면의 화질이나 음량을 같게 만든 것은 아니다.

파일 해시·정확한 추출 구간·실행 명령·도구 버전은 [scene02.preparation.json](scene02.preparation.json)과 [scene03.preparation.json](scene03.preparation.json)에 있다. 원천 WebM 파일은 저장소에 중복 포함하지 않았다.

## 기술 검증

[검증 결과](lesson_media_validation.json)에 실제 측정값을 저장했다.

- 새 파일 네 개의 컨테이너·비디오·오디오 표시 길이는 모두 10.000초다.
- 각 원본/무음 쌍의 240개 디코딩 RGB 프레임과 타임스탬프가 모두 일치한다. 4/8프레임 표본도 일치한다.
- 두 쌍 모두 원본 오디오에는 0이 아닌 신호가 있고 무음본의 peak/RMS는 0이다.
- 쌍 안에서 오디오 프레임의 시간축·샘플 수·샘플레이트·채널이 같다. AAC를 디코딩하면 끝의 패딩을 포함해 160,768샘플이며, 의도한 재생 구간은 160,000샘플(10초)이다. 두 변형에 같은 패딩이 있다.
- 네 파일 모두 기존 `preflight_clip`과 디코딩 메모리 제한 검사를 통과했다. 총 크기는 약 2.59MiB다.

이는 파일·신호 검증이다. 직접 청취에 의한 최종 수업 점검과 모델의 답변·GPU 메모리 검증은 아직 수행하지 않았다. 특히 비 오는 장면을 화면 밖 소리의 확정적인 예제로 분류하지 않는다.

## 다시 만들기

Python 3.11+와 FFmpeg/ffprobe가 있는 준비 환경에서 실행한다. 아래 명령은 저장소 루트 기준이며, `/path/to/sources`에는 다음 원천 파일을 내려받아 둔다.

- [Tabla_drums_demo.webm](https://upload.wikimedia.org/wikipedia/commons/4/44/Tabla_drums_demo.webm)
- [Rain_in_Rivne_1.webm](https://upload.wikimedia.org/wikipedia/commons/1/1f/Rain_in_Rivne_1.webm)

```bash
python avllm_interpretability/scripts/prepare_lesson_media.py --source /path/to/sources/Tabla_drums_demo.webm --media-id scene02 --start 75 --out-dir /path/to/new-output
python avllm_interpretability/scripts/prepare_lesson_media.py --source /path/to/sources/Rain_in_Rivne_1.webm --media-id scene03 --start 5 --out-dir /path/to/new-output
```

스크립트는 기본 기준 파일 `assets/02321.mp4`에서 길이를 직접 읽으며, 기존 출력이 있으면 덮어쓰지 않는다. 원천 파일 해시는 preparation JSON과 대조한다. FFmpeg 버전이 바뀌면 인코딩 바이트가 달라질 수 있으므로 다시 만든 파일은 별도로 검증한다. 이 준비 도구는 학생의 노트북 실행 경로에서 호출하지 않는다.
