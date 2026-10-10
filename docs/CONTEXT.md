# 연구와 Isaac Lab 개발 맥락

최종 갱신: **2026-10-10, Asia/Seoul**.

밀집 선반에서 Target 접근 공간을 확보하는 비파지 재배치 연구의 목적, 설계가
바뀐 이유, 현재 구현과 검증 결과, 미결 질문을 유지한다. 새 대화와 개발 작업은
이 문서를 먼저 읽고 관련 코드 및 세부 문서로 이어간다.

연구 기준은 Notion의 [연구 맥락 — 생각의 흐름과 현재 설계](https://app.notion.com/p/3f34968094ee81ca8eb6feb1f700e0f9)
이다. 해당 페이지의 연구 기준일은 2026-10-08이며, 2026-10-09에 다시 확인했다.
이 문서는 그 맥락과 이후 이 저장소에서의 개발을 통합한다.

## 1. 현재 위치와 상태 구분

**스폰·센서 위에 학습용 MDP와 모델·PPO를 구현했으며, 시간 입력을 현재·직전
두 sample로 변경했다.**
장면 확인용 `ManagerBasedEnv`와 학습용 `ManagerBasedRLEnv`를 구분한다. 사용자가
초기에는 최근 16 step 이력을 선택했으나 후속 요청에 따라 **현재 t와 직전 t−1의
센서·로봇 상태·행동 두 sample**로 줄였다. 18D OSC/hand action,
Reward v2·종료 조건, 초기 장면 GNN·역할 조건부 attention·유한 이력 encoder,
공동학습 PPO를 구현했다. 물체 입력은 승인한 초기 RGB-D에서 추정한 기하만
사용한다. 초기 기하를 robot base에 저장하고 매 policy step에 **현재 EEF
좌표계로 재표현**하며 현재 실제 물체 pose를 actor에 넣지 않는다. 제어 기준
좌표계는 `examples`의 sweeping 프로젝트처럼 robot root/base다. 아래의
스폰·센서 GPU 검증을 새 MDP의 학습 성공이나 제어 검증으로 확대하지 않는다.
Reward·제어 수치는 provisional default이며 정책 성능은 아직 입증하지 않았다.
코드 구조는 `examples`의 `sweeping_policy`를 따라 공통 환경 설정,
`config/ur5e`의 로봇·학습 설정, `mdp`의 manager 함수, `src`의 장면 지원 구현으로
정리했다. 학습은 `Isaac-Contact-Rearrangement-UR5e-v0`의 등록된 설정으로 실행한다.
새 패키지 `src/contact_rearrangement`는 **10종 asset 중 4–10종을 선택하고
전체 선반 밀도에 필요한 수의 instance를 배치**한다. 이전에 4–10종을
active instance 4–10개로 해석한 것은 잘못이며 종류 수와 개수를 분리했다.
`--all_asset_types`는 10종 모두 선택하고 non-target 복사본을 추가할 수 있다.
**Target 종류는 active 장면에 한 번만 등장**하고 Blocker·주변 후보에서 같은
종류의 모든 복사본을 제외한다. 종류마다 최소 한 instance를 선택한 뒤 footprint
면적에 따라 개수를 정한다. 추가 가능한 non-target 종류 중 active 개수가 가장
적은 종류를 고르고 동률은 무작위로 정해 중복 편중을 줄이는 구성으로 변경했다.
Blocker도 해당 종류의 첫 instance로 센다. Target은 항상 하나이며 현재 동일
pool capacity에서는 non-target 종류별 개수 차이가 최대 1이다. 별도 종류별
최대값은 없고 pool의 10-copy·전체 cap 40을 유지한다. 작은 물체 위주 종류
집합은 같은 면적을 채우려면 더 많은 instance가 필요할 수 있다.
기본 instance 최소 4·최대 40은 safety limit이다.
10종 × 10개 복사본의 100-body pool을 유지한다. Pool 전체의 rigid-body
simulation과 native gravity를 켜 두고,
나머지는 높은 격리 위치에 주차해 매 physics substep마다 COM에 `-m*g` 외력을
적용한다. Active 물체에는 추가 힘을 주지 않아 원래 중력과 접촉을 유지한다.
Episode의 Target·Blocker를 먼저 지정하고 **종류 수·집합, 전체 active instance
ID와 개수**를 pose sampling 전에 고정한다. Retry에서는 yaw와 위치만 바꾼다.
Target은 Blocker가 앞에 들어갈 수 있는 전체 X 범위에서 선택한다. 기존 뒤쪽
10 cm 제한과 38 cm cluster·35% tight-rectangle fill 조건은 폐기했다.
현재 구현은 전체 usable shelf 점유율 **35% 이상**, X 2 × Y 3의 **6개 구역
각각 10% 이상**, 4 cm footprint 이웃을 가진 active 비율 **75% 이상**을 hard
acceptance 조건으로 요구한다. X/Y coverage와 병진·평면 pivot·회전 후 이동의
다단계 clearance witness도 검사한다. 사용자는 구현 가능한 범위를 먼저 측정해
밀도를 정하도록 요청했고, 25–45% 비교에서 생성 속도와 성공률을 고려해 35%를
초기 기본값으로 선택했다. 사용자가 직접 정한 최종 연구 난이도 수치는 아니다.
앞쪽 +X camera의 실제 초기 영상에서 Target 가시 비율 **40% 이상**을 요구하기로
했다. 분모는 같은 camera·같은 pose에서 가림 없이 렌더링한 전체 Target
silhouette 픽셀 수다. 현재 앞쪽 RGB-D camera와 실제 silhouette acceptance를
구현하며 **Blocker 전체 가시성**도 함께 요구한다. Target·Blocker 둘 다 전체
bounds가 frame 안에 있어야 한다. 승인한 초기 snapshot은 episode 동안 고정한다.
YCB 003·004·005·006의 초기 회전을 **X축 −90°**로 교정하고, 원본 USD의
수정 전·후 각 8장 렌더에서 라벨·캔 고리·병뚜껑으로 상하를 확인했다.
이전 큰 footprint 우선 복사본 구성의 35% 규칙은 실제 USD 측정 bounds로 일반·all-types 각각 seed 0–19,
**총 40개 기하 sample**을 통과했다. 일반 모드는 4–10종·10–24 instance,
all-types는 10종·14 instance였다. 일반·all-types 각각 **4개 GPU 환경·10초**의
정착·전체/부분 reset 검사도 통과했다. 각 모드의 8개 layout을 각각 5초 진행한
범위이며 기하 benchmark와 물리 검사를 구분한다. 당시 unit test 21개도 통과했다.
이 분포와 통과 결과를 현재 균형 구성의 결과로 사용하지 않는다. 현재 균형
구성은 Blocker 앞 reserve·camera gate 추가 전 기하 sample **40/40개**와
일반 모드 **4개 GPU 환경·151 step·5.03초**의
정착·전체/부분 reset 검사를 통과했다. 당시 unit test **23개**도 통과했으며
현재 구성의 별도 all-types GPU 실행은 완료한 것으로 기록하지 않는다.
**전물체의 2 mm 낙하 여유를 제거**하고 003·004에만 visible mesh를 유지한
**명시적 Cube collision proxy**를 적용해 초기 흔들림을 줄였다. 원본
Nucleus USD·authored mass·contact 속성·physics material·active gravity는 유지한다.
이전 정확히 10개 instance 버전에서는 이 collider로 **4개 GPU 환경·10초**의
정착·전체/부분 reset 검사가 통과했다. 이 결과와 이전 실패·비채택 approximation,
새 종류/instance 분리 benchmark는 8절에서 구분한다.
판단·행동 모델과 PPO 코드는 추가했으며 학습 성능은 아직 확인하지 않았다.
사용자 요청으로 장면 확인 스크립트에 **Target 초록색·Blocker 빨간색 OBB**의
debug draw 표시를 추가했다. 물리·USD asset을 수정하지 않는 확인용 표시이며
2개 CUDA 환경의 headless reset 검사와 1개 GUI 환경의 실제 표시 검사가 통과했다.
2026-10-09에는 사용자가 파일이 지나치게 나뉜 구조를 정리하도록 요청해
구현을 책임별 **6개 모듈**로 통합했다. 당시 `catalog`·`assets`·`env_cfg`·`spawn`·
`sampling`·`visualization` 구조였으며 스폰 규칙·기본값·실행 명령을 유지했다. 통합 후 unit test **23개**와
일반 모드 **4개 GPU 환경·151 step·5.03초** 검사가 통과했으며 초기·자동 reset
뒤 8개 장면 설명이 통합 전과 같았다. 아래 기존 실행 결과와 구분해 기록한다.
사용자는 **Blocker 앞의 충돌 없는 hand 시작 영역 randomization과
elbow-down 금지**를 요청했다. RobotStartRules·실제 USD FK/IK·초기 collision
판정을 구현했고 2026-10-09의 24개 시작·반복 reset·5초 물리 검사와 unit test
**29개**가 통과했다. 당시 손가락/tool은 아래를 향했다. 사용자의 후속 요청에
따라 tool 전방을 **선반 방향인 world −X**로 교정했고 palm normal이 위를
향하던 이 단계의 4개 GPU 환경·151 step·5.03초 검사도 통과했다. 이어
사용자가 tool 기준 Z축으로 hand를 **−90° 회전**하도록 요청해 mount에 roll을
추가한다. 현재 tool 전방은 −X를 유지하고 palm normal은 world −Y를 향한다.
새 hand roll의 4개 GPU 환경·151 step·13개 시작 pose 검사도 통과했다.
앞 단계의 통과 결과와 새 hand roll의 검사를 구분한다. 후속 요청으로 실제
센서·연결부의 공간을 고려해 UR5e 말단과 hand 사이에 **길이 5 cm·직경
5 cm의 원통 연결부**를 추가한다. 연결부는 wrist-3의 compound collider이며
별도 body나 DOF를 만들지 않는다. 연결부 추가 전 검증과 새 조립 검사를 구분한다.
새 연결부의 4개 GPU 환경·151 step·13개 시작 pose 검사도 통과했다.
후속 센서 단계에서는 원통을 25 mm 두 rigid half로 나누고 중앙 fixed joint의
실제 F/T를 읽는다. 앞쪽 camera pose를 무작위로 제안해 Target ≥40%·Blocker
100%(면적 2 pixel 허용)를 실제 렌더로 승인하며 raw F/T와 초기 RGB-D/mask를
`sensors` observation group에 제공한다. 이번 센서 단계의 검증은 별도로 기록한다.
최종 4개 GPU 환경 검사에서 18개 시작 pose의 실제 가시성과 capture 중 native
state 무변경, 초기 snapshot 보존·F/T 유효성·알려진 force/torque 응답이 통과했다.
가시성 proposal을 개선하려고 `reserve_blocker_front=True`로 Blocker 앞의
직선 +X 통로를 주변 물체로부터 비운다. 기존 전체 선반 밀도와 Target A₀는
유지하며 이 통로가 실제 가시성을 보장하는 것으로 취급하지 않는다.
현재 hand의 17개 tactile link 각각에 독립 `ContactSensor`를 붙이고 world net
normal force norm **>0.01 N**을 binary contact로 제공한다. 2개 GPU 환경에서
17개 pad의 실제 contact ON/OFF·부분 reset·render cache 보존 검증도 통과했다.

| 상태 | 의미 |
|---|---|
| 결정 또는 방향 | 사용자가 선택한 연구 방향. 구현 완료나 성능 입증을 뜻하지 않는다. |
| 현재 작업안 | 현재 구체화 기준으로 삼는 설계. 세부 구조와 수치는 더 정해야 한다. |
| 구현 기본값 | 지금 코드 실행을 위한 설정. 연구 전체의 최종 명세로 간주하지 않는다. |
| 제안 및 미채택 | 검토 후보. 사용자의 질문이나 설명 요청을 채택으로 기록하지 않는다. |
| 검증 결과 | 명시한 조건과 검사에서 실제 확인한 결과. 검사 범위 밖으로 확대하지 않는다. |

우선순위는 **최신 사용자 결정 → 현재 연구 맥락 → 과거 설계·발표 자료**다.
코드의 실제 동작은 해당 코드와 실행 결과로 확인한다. 연구 설계와 코드가 다르면
어떤 부분이 구현 전인지, 이전 실험의 설정인지 명시한다.

현재 코드 구조는 사용자의 후속 요청에 따라
`examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy`를 기준으로 정리했다.
공통 환경 설정, `config/ur5e`의 로봇별 설정·Gym task 등록·agent 설정,
`mdp`의 manager 함수, `src`의 장면 지원 구현을 분리한다. Shared transition은
`mdp/state.py`가 한 번만 갱신하며 물리·관측·reward·18D action 명세를 유지한다.
현재 파일 위치와 task registry 기반 실행은 7절과 패키지 README에 정리한다.

## 2. 연구 목적과 범위

근사적인 초기 물체 기하와 촉각·힘/토크 피드백을 이용해, 명시적 Blocker 종착
pose 없이 Target 접근 공간을 확보하는 신속한 비파지 재배치 정책을 학습한다.

- **Target:** 접근 공간을 확보하려는 물체. 역할은 외부에서 지정한다.
- **Blocker:** 접근을 방해하며 주로 조작할 물체. 역할은 외부에서 지정한다.
- **주변 물체:** 추가 접촉·충돌·차단이 발생할 수 있는 물체.
- **조작:** 선반의 지지를 받으며 밀기·회전으로 재배치한다. 파지해서 들어 옮기는 것을 주된 해법으로 삼지 않는다.
- **정책 역할:** Blocker의 최종 goal pose를 사전에 입력하지 않고 재배치 판단과 로봇 행동을 함께 학습한다.
- **실행 목표:** 접촉 변화에 빠르게 대응하면서 성공·안정성을 유지하고 수행시간을 줄인다. Rapid는 아직 입증되지 않은 연구 목표다.
- **환경과 장비:** 고정 선반, UR5e + RH56E2 hand, wrist F/T, 17개 위치의 binary tactile. 선반 형상은 현재 observation 명세에 포함하지 않는다.

가림과 근사 형상 때문에 유효한 종착 pose를 사전에 안정적으로 특정하기 어려울
수 있다는 문제 설정이다. 상위 모델이 goal pose를 생성할 능력이 없다는 주장은 아니다.
실제 Target 파지·인출까지 학습할지, 여러 Blocker의 조작 순서를 다룰지는 미정이다.
발표용 제목은 **신속한 비파지 재배치를 위한 촉각 및 힘·토크 피드백 기반
판단·행동 통합 심층 강화학습 정책**이었다. 학위논문의 최종 제목 확정과 구분한다.

## 3. 현재 관측과 정책 작업안

### 관측과 시간 구조

최신 방향은 **초기 vision을 사용하고, 이후 robot state·F/T·tactile·행동 이력으로
이번 episode의 환경 변화를 해석하는 것**이다. 지속적인 vision 갱신을 기다리는
의존성을 줄여 촉각·힘 피드백의 빠른 대응을 살리려는 의도다. 초기에는 16-step
window를 구현했으나 사용자가 **previous step까지만 제공**하도록 변경했다.
현재 입력은 **현재 t와 직전 t−1의 두 sample**이며 newest-first로 저장한다.
Policy는 30 Hz, physics는 120 Hz이므로 sample 사이 간격은 약 33 ms다.
Reset 직후 현재 sample만 valid, 직전 sample은 zero/invalid로 두며 그보다 오래된
이력은 제공하지 않는다. Episode를 넘어 유지하는 GRU/LSTM은 도입하지 않았다.

| 입력 | 현재 구현 | 후속 교정 |
|---|---|---|
| 초기 물체 기하 | 물체별 위치 3D + quaternion(wxyz) 4D + OBB extent 3D = 10D, 현재 EEF 기준 | 단일 시점 visible depth의 근사 오차와 실기 detector |
| 역할과 유효성 | K=40 슬롯, valid 1D와 Target·Blocker role 2D를 별도로 전달 | 실기에서의 identity/role association |
| Robot state | Arm 6개·hand 독립 6개 관절의 q/dq 24D + base 기준 EEF pose 7D = 31D | 실제 구동 관절 정규화·동기화 |
| Wrist F/T | Sensor frame의 중력·bias 보정 force/torque 6D | Filtering·실제 질량과 bias 교정 |
| Tactile | 접촉 17D + channel validity 17D | 0.01 N은 변경 가능한 초기 threshold |
| Previous action | 해당 관측 직전 적용한 normalized action 18D | Policy/control 주기 교정 |

K는 Target·Blocker를 포함한 전체 최대 슬롯 수다. 같은 물체 기하를 역할별로
중복 계산하지 않는다. 모든 가려진 물체를 초기 vision으로 알고 있다는 가정도 하지 않는다.
관절 관측 차원은 URDF의 전체 물리 관절 수와 독립 구동 관절 수를 구분해 구현해야 한다.

초기 기하는 승인한 camera snapshot의 instance mask 안에 있는 visible depth를
back-project하고 PCA OBB로 근사한다. Simulator instance ID는 물체별 분할과
외부 Target/Blocker 역할의 association에 사용하므로 이 부분은 oracle이다.
USD의 전체 bounds나 현재 실제 물체 pose를 geometry feature로 복사하지 않는다.
관측하지 못한 물체는 invalid padding이다. 가림으로 extent가 작게 추정될 수 있으며
완전한 물체 형상을 복원했다고 해석하지 않는다.

Camera extrinsic으로 변환한 초기 geometry의 canonical copy를 robot base에
저장한다. 매 policy step마다 현재 EEF pose의 역변환을 적용해 위치·orientation을
현재 EEF 기준으로 표현하고 extent는 유지한다. 이는 world에 고정한 **초기
추정 장면의 재표현**이다. 조작 후의 실제 물체 이동을 vision으로 갱신하거나
정답으로 대체하지 않는다. 제어·robot EEF pose는 base, 물체 입력은 현재 EEF,
F/T 입력은 sensor frame을 각각 사용한다.

한 history sample은 `31 + 6 + 17 + 17 + 18 = 89D`다. Actor flat observation은
`geometry 40×10 + valid 40 + roles 40×2 + history 2×89 + history_valid 2`
순서로 **700D**다. 현재 sample에는 해당 관측 직전 적용한 action aₜ₋₁,
직전 sample에는 aₜ₋₂를 포함한다. Reset 첫 현재 sample의 action은 0이다.
Role·valid·history_valid는 별도 mask로 유지하고 padding을
encoder에서 제외한다. 관절 q/dq는 π·2 rad/s, force/torque는 50 N·5 N·m로
정규화하는 초기 설정이다. 과거 `72 + 11K`의 current-only 입력안은 이력 구조로 대체했다.

Actor가 현재 실제 object pose·velocity를 받기로 한 것은 아니다. 시뮬레이터
정답을 reward·검증·학습 label에 쓰는 것과 actor 실행 입력에 쓰는 것은 구분한다.
History를 외부 상태 추정기로 해석한 뒤 추정 상태를 actor에 넣는 구조는 지양한다.

### 관계 처리와 학습

현재 구현 구조는 다음과 같다.

```text
초기 물체 기하(현재 EEF 재표현)와 역할·valid
  → 공유 node encoder / dense GNN 2 rounds
  → Target·Blocker 조건부 masked attention readout
현재 t·직전 t−1의 robot state·F/T·tactile·action와 history_valid
  → 두 sample을 받는 causal convolution
두 encoder feature 결합
  → bounded Gaussian actor의 18D action + shared critic
```

- 슬롯 나열 순서를 바꿔도 같은 장면에 대해 같은 판단을 하도록 집합을 처리한다. 실제 공간 배치는 입력에 남아야 한다.
- GNN edge는 메시지 교환을 허용하는 연결이다. 접촉·방해 관계의 정답 label을 미리 제공하는 것으로 정하지 않았다.
- 로봇에 대한 물체 위치를 판별할 정보가 필요하다. 물체 간 상대 관계만 남기는 구조로 단순화하지 않는다.
- Role mask와 valid mask는 목적이 다르다. Zero padding에서 valid를 만들면 noise·정규화 전에 생성하고 message passing과 attention에서 padding을 제외한다.
- GNN의 같은 forward 내 message-passing 반복과 episode의 시간적 기억은 다르다.
- Node 64D, message passing 2 rounds의 dense valid edge를 사용하며 self/padding edge를 제외한다. Edge feature는 초기 물체 간 상대 위치·extent 차이다. Role-conditioned query는 Target·Blocker embedding에서 만들며 두 attention readout과 role/global feature를 결합한다.
- History encoder는 64D causal convolution이며 기본 1개 layer로 현재·직전 두 sample을 처리한다. Padding을 제외하고 유효 sample만 pooling한다. Fused latent는 128D이며 기본 모델은 248,997개 parameter다. 더 오래된 sample이나 지속 hidden state는 제공하지 않는다.
- Custom PPO가 scene/history encoder와 actor/critic을 공동학습한다. Actor와 critic은 같은 700D 입력과 encoder를 공유하며 privileged critic·현재 물체 GT 입력·보조 예측 head/loss는 추가하지 않았다.
- Tanh-Gaussian action의 log probability에는 변환 Jacobian을 반영한다. GAE는 실제 terminal state에서 bootstrap하지 않고, timeout에서는 auto-reset 이전 실제 final observation의 value로 bootstrap하며 다음 episode의 advantage를 이어 붙이지 않는다.

이전 16-step 모델은 관측 1960D·286,053 parameter였다. 현재 2-sample 모델의
관측·network 규격이 달라 그 checkpoint로 이어 학습할 수 없다. 아래의 기존
16-step 검증 기록은 당시 모델의 결과이며 현재 규격의 검증과 구분한다.

### Action과 제어

| 출력 | 차원 | 제어 경로 |
|---|---:|---|
| EEF motion | 6 | 위치·회전 증분 → variable-stiffness OSC |
| Cartesian stiffness | 6 | 병진·회전 각 3축 강성 → OSC |
| Hand configuration | 6 | RH56E2 독립 구동 관절 목표 위치 → position controller |

총 **18D action**이며 arm의 12D 출력은 OSC를 통해 관절 토크로 연결한다.
목표를 이전 명령에 계속 누적하지 않는 방향을 유지한다. 직전 action을 관측하는
것과 목표 누적은 별개다. Examples의 robot-root reference 구현을 따라 증분은
**base 기준**이며 매 policy step의 실제 TCP pose에서 position과 orientation
목표를 새로 만든다. 회전 증분은 base 축 기준으로 현재 quaternion에 left-multiply한다.
기본 scale은 위치 축별 ±15 mm·회전 축별 ±0.08 rad, 강성은 병진 100–800 N/m·
회전 5–40 N·m/rad, damping ratio는 1이다. 현재 inertia decoupling은 꺼진
설정만 지원하며 이를 물리 Cartesian stiffness로 사용한다. Arm의 기존
PD/mimic PD hold를 비활성화하고 OSC torque를 적용하며 hand의 독립 6개
관절만 position target으로 제어한다.
Hand의 authored mimic constraint는 유지한다. TCP Jacobian의 COM→link origin 이동과
world→base 변환, native effort limit clamp를 적용한다. 이 수치는 실기 허용치나
검증된 제어 gain이 아니며, spawn-only 스크립트의 PD hold와 학습 제어를 구분한다.

SDK의 unit-mass `2√K` explicit damping은 실제 가벼운 wrist rotational inertia에서
120 Hz 이산화와 맞지 않아 새 제어의 기준으로 사용하지 않는다. Native arm
mass matrix `M`과 TCP Jacobian `J`로 coupled task inertia
`Λ=(J M⁻¹ Jᵀ)⁻¹`를 구하고 inverse-inertia eigenvalue floor `1e−6`을 적용한다.
대각 물리 stiffness `K`에 대해 coupled critical damping matrix를 계산한다.

$$
D=2\zeta\Lambda^{1/2}
\left(\Lambda^{-1/2}K\Lambda^{-1/2}\right)^{1/2}\Lambda^{1/2},
\qquad \zeta=1.
$$

매 physics substep의 `dt=1/120 s`에 spring/damper를 backward Euler로 계산한다.
`e`는 측정 TCP와 latch한 목표 사이의 base-frame pose error, `v`는 같은
TCP origin/base 축의 twist다.

$$
v_{next}=(\Lambda+dtD+dt^2K)^{-1}(\Lambda v+dtKe),
\qquad \tau=J^T\left[Ke-(D+dtK)v_{next}\right]+G.
$$

SDK OSC의 spring·gravity 계산에서 explicit damping을 제외하고 위 보정항을
적용한 뒤 native effort limit로 clamp한다. `damping_ratio=1`은 관성 반영
critical matrix의 비율이며 unit-mass scalar `2√K`를 뜻하지 않는다. 이 선형
substep 계산의 안정화와 실제 접촉·비선형 task의 성능 검증은 구분한다.

Hand 6D는 soft limit 안의 **원하는 절대 관절 목표**다. 실제 PD target은 매
physics substep의 측정 q에서 `min(native velocity limit, hand_target_rate)`×dt만큼
목표 방향으로 제한한다. 기본 `hand_target_rate=1 rad/s`이며 reset 직후 큰
목표 위치 오차가 한 번에 적용되지 않게 한다. `hand_command`와 rate-bounded
`hand_applied_command`는 별도다. Policy action의 차원·의미는 바꾸지 않는다.

## 4. Reward 기준점과 사용자 교정

**Formulation v2를 새 학습 환경에 구현했으며 수치는 초기 실행 기본값이다.**
Transition `s_t → s_{t+1}`에 대해 계산한다. 상위 weight는 비음수, scale과
`Δt`는 양수다. 최종 weight·scale·soft threshold·종료 허용오차와 학습 충분성은 미정이다.
`MdpCfg`의 현재 상위 weight 4개는 모두 1, `d_0=0.5 m`, `T_ref=10 s`,
성공 bonus 500·실패 penalty 100이다. Episode timeout 8초는 별도 초기 설정이다.

$$
r_t=w_{\mathrm{clear}}r_t^{\mathrm{clear}}
+w_{\mathrm{rearr}}r_t^{\mathrm{rearr}}
+w_{\mathrm{motion}}r_t^{\mathrm{motion}}
+w_{\mathrm{time}}r_t^{\mathrm{time}}
+r_t^{\mathrm{terminal}}
$$

### 접근성

지정 Blocker의 차단 비율을 줄인다.

$$
r_t^{\mathrm{clear}}
=\exp\left(-\frac{A_{\mathrm{blocked},t+1}}{A_{\mathrm{blocker},t+1}}\right)
$$

분자는 고정 접근영역 `A_0`와 지정 Blocker footprint의 겹침 면적, 분모는 같은
shelf 평면에 투영한 **Blocker의 전체 footprint 면적**이다. 접근영역 면적으로
나누지 않는다. `A_0`는 초기 Target pose·접근 방향·hand footprint·margin으로
고정해 Target 이동에 따라 성공 영역이 같이 움직이지 않게 한다. 차단 비율이
0이면 1, 1이면 `e^-1`이다. 이 점수만으로 전체 경로 확보를 판정하지 않는다.

### 최소 재배치

전체 유효 물체의 실제 누적 평면 이동량을 줄인다.

$$
D_0=0,\qquad D_{t+1}=D_t+
\sum_{i\in\mathcal O}\|p^{xy}_{i,t+1}-p^{xy}_{i,t}\|_2,\qquad
r_t^{\mathrm{rearr}}=\exp(-D_{t+1}/d_0)
$$

`O`에는 Target·Blocker·주변 물체를 포함하고 padding은 제외한다. `D`는
초기–최종 변위가 아닌 물체 중심의 경로 길이 합이므로 왕복 이동도 비용으로 남는다.
사용자는 **같은 실제 이동은 같은 비용**으로 취급하도록 교정했다. 물체 크기로
나누거나 물체 수로 평균내지 않고 공통 `d_0`를 사용한다. 회전 환산 거리와
역할별 가중치는 현재 식에 포함하지 않은 대안이다.

### 운동과 접촉 안정성

네 무차원 비용을 지수 점수로 바꾸어 동일 계수로 평균한다.

$$
r_t^{\mathrm{motion}}=\frac14
\sum_{j\in\{\mathrm{vel,acc,load,rate}\}}\exp(-x_{j,t})
$$

| 비용 | 현재 계산 의미 |
|---|---|
| `x_vel` | 전체 물체의 평면 선속도·각속도의 soft 기준 초과분을 각각 ref로 나누고 제곱해 합산 |
| `x_acc` | 전체 물체의 선가속도·각가속도를 각각 ref로 나누고 제곱해 합산 |
| `x_load` | 보정 wrist force·moment norm의 soft 기준 초과분을 각각 ref로 나누어 합산. 현재 식은 제곱하지 않음 |
| `x_rate` | 힘·토크 시간 변화율을 각각 ref로 나누고 제곱해 합산 |

네 그룹 사이의 내부 λ는 두지 않는다. 다른 reward와의 비중은 상위 `w_motion`으로
조절한다. 가속도와 wrench 변화율은 `Δt`로 차분한다. Wrench는 중력·bias 보정
후 같은 좌표축·토크 기준점에서 비교해야 한다. Actor의 sensor-frame 입력과
reward 계산 좌표계를 혼동하지 않는다. Actor에는 움직이는 sensor center의
sensor-frame 보정 wrench를 제공한다. 중력 support와 downstream COM의 moment를
빼고 명시적 tare에서 남은 bias만 보정하며 관성·접촉 신호는 남긴다. Reward에는
이를 world 축으로 회전하고 **episode 시작 sensor center라는 고정 기준점**으로
moment를 옮긴 값을 사용하므로 sensor 회전·이동 자체가 wrench rate로 섞이지 않는다.

### 시간과 종료

$$
r_t^{\mathrm{time}}=-\frac{\Delta t}{T_{\mathrm{ref}}},\qquad
r_t^{\mathrm{terminal}}=
R_{\mathrm{success}}\mathbf1_{\mathrm{success},t}
-R_{\mathrm{failure}}\mathbf1_{\mathrm{failure},t}
$$

사용자는 `T_ref` 유지를 선택했다. Step마다 −1로 교체하지 않는다. `T_ref`는
보상 크기를 맞추는 시간 scale이며 timeout이나 물리적 허용시간으로 정한 값이 아니다.
종료 보상은 한 번 지급한다.

- 성공은 접근영역의 **전체 non-target 차단 면적**이 허용오차 이하이고 Target 보호·안정 조건을 만족하는 것이다. 지정 Blocker만 치우고 다른 물체가 막은 상태를 성공으로 처리하지 않는다.
- 성공의 초기 기준은 전체 non-target 겹침 합 ≤1e−5 m², Target 이동 ≤2 cm·회전 ≤0.15 rad이며 모든 active 물체가 선속도 ≤0.015 m/s·각속도 ≤0.10 rad/s로 5 policy step 연속 안정한 상태다.
- 실패의 초기 기준은 낙하 6 cm, 지지면에서 4 cm 초과 들어올림, shelf workspace 여유 1.5 cm 이탈, upright 대비 tilt 50° 초과, 보정 force 80 N·torque 4 N·m 초과 또는 금지 접촉이다. 성공과 실패를 동시에 검출하면 실패를 우선한다.
- 학습 scene에만 UR arm 7개 body와 원통 2개 half의 ContactSensor **9개**를 둔다. 선반·물체·ground의 각 partner와의 normal force norm **>1 N**을 금지 접촉으로 보는 provisional 기준이다. GPU contact filter는 동적/kinematic body의 자식 mesh가 아닌 RigidBodyAPI prim을 사용한다. 기본 pool 100개와 선반 1개의 **101 rigid-body filter**, rigid body가 없는 static ground의 collision-shape filter를 각각 지정한다. 최근 4 physics substep 중 하나라도 초과하면 실패하며 서로 다른 partner의 반대 방향 force를 합산해 상쇄하지 않는다. Hand 접촉이나 hand–Target 접촉 자체는 이 판정에서 금지하지 않는다. Target 이동/회전은 별도 보호 조건으로 판정한다.
- 8초 time limit는 PPO의 truncation으로 처리하며 성공/실패는 termination이다. 실기에서 초기 vision만으로 성공을 판별하는 방법은 미정이다.

### 구현과 검증에서 남은 문제

양의 매-step 점수는 대기하면서 보상을 누적하는 행동을 유리하게 만들 수 있다.
성공 bonus를 **할인된 누적 대기 return**과 비교해야 한다. 안정성 항이 rapid
동작을 과도하게 막는지, 물체 수에 따른 합산 scale 변화, 평면 면적이 실제 hand/arm
접근 가능성을 충분히 나타내는지도 검증해야 한다.

로컬 Isaac Lab [RewardManager](../IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py)는
각 term에 `weight * dt`를 곱한다. Policy-step 수식을 그대로 옮길 때 시간 벌점의
`Δt`와 one-shot 종료 보상을 포함해 시간 scale을 중복 적용하지 않도록,
구현은 완성한 discrete transition reward를 `dt`로 나눈 값을 manager에 반환한다.
위 reward를 잘 받는다는 사실만으로 내부 상태 이해의 정확성을 인증할 수는 없다.

## 5. 환경 변화 이해와 미채택 제안

사용자가 요구한 것은 초기 장면–행동–sensing–환경 변화의 관계를 정책이 배우도록
유도하는 학습 신호와, 제대로 배웠는지 판별할 근거다. 연산 지연은 구현 후 측정할
문제로 두었으며, 이 논의를 반복해서 latency 구조 논의로 되돌리지 않는다.

같은 초기 장면에서 Blocker를 서로 다른 만큼 밀고 손이 같은 pose로 돌아온
두 episode를 생각한다. 현재 q·dq·F/T·tactile이 같고 밀기 사건이 유한 history
window 밖이라면 기억 없는 정책은 두 상태를 구분하지 못할 수 있다. Reward를
바꾸어도 현재 입력에서 사라진 사건 정보가 생기지는 않는다.

Current-only, 유한 window, 지속 hidden state를 갖는 recurrent policy는 서로
다르다. 가중치는 episode 전반의 일반적인 관계를, hidden state는 이번 episode의
상호작용 정보를 담는다. GNN round 수와 history 길이, hidden state 유지 기간과
학습 sequence 길이도 구분한다. **LSTM/GRU 도입은 아직 승인되지 않았다.**

현재 task reward만 사용하는 finite-window baseline을 먼저 구현했다. 다음은
아직 미채택 후보이며 baseline에 포함하지 않는다.

- 정책의 공유 표현에서 물체 이동·회전·차단비율 등을 예측하는 학습용 head와 `L_train = L_PPO + λ L_pred` 추가. Label·정규화·loss·λ는 미정이며 실행 시 head를 제거할 수 있다는 제안이다.
- GT 상태 label 대신 action-conditioned latent forward dynamics 사용.
- 초기 scene encoder와 recurrent temporal encoder 결합, sensor-conditioned query와 복수 readout.

예측 결과나 simulator 정답을 actor에 직접 주기로 한 것은 아니다. 보조 loss도
사용자가 수용한 상태가 아니다. 예측 정확도를 scalar reward에 더하면 예측하기
쉬운 상태에 머무르는 행동을 선호할 수 있다는 쟁점이 남아 있다.

평가 후보는 고정 분포의 변화 예측 오차, no-change predictor 대비 개선,
history 제거·교란 시 성능 변화, 비슷한 현재 관측과 서로 다른 과거를 가진 조건의
다음 행동 비교다. 모두 확정 실험 계획과 구분한다.

## 6. 설계가 바뀐 흐름과 문헌 교정

| 시점 또는 단계 | 변경과 현재 의미 |
|---|---|
| 초기 BrainStorming | 초기 vision·상위 VLM·pull-forward/lateral sweeping skill 구상. 접근 공간 확보 문제는 유지되지만 두 skill/VLM 구조 전체를 현재 정책으로 채택하지 않았다. |
| Track B | 상위 판단기가 회전·이동 명령을 주고 순차 수행, vision 지속 갱신. 명시적 종착 목표와 고정 단계 순서는 현재 명세로 가져오지 않는다. |
| 2026-09-30 중간발표 | Target 접근성을 목표로 no-blocker-goal-pose 판단·행동 통합, 18D action, variable stiffness, 네 reward 목적을 구체화했다. |
| 2026-10-01∼04 | Planar 관측 후보에서 3D OBB10D·quaternion·단일 카메라·외부 역할 지정·집합 처리로 구체화. Map은 채택 방향에서 제외했다. |
| 2026-10-06 | GNN·mask·EEF 기준 정보 보존을 검토하고 message-passing 반복과 temporal memory를 구분했다. |
| 2026-10-08 | Reward v2, 공통 거리 scale, T_ref 유지. 지속 tracking에서 초기 vision + 상호작용 이력으로 전환하는 방향이 추가됐다. |
| 2026-10-09 초기 구현 | 사용자가 Isaac Lab 구현 시작과 새 환경의 스폰부터 진행하도록 요청. `contact_rearrangement`의 고정 6개 장면·reset 구현과 원본 USD 검증을 완료했다. |
| 2026-10-09 장면 구체화 | 사용자가 장면당 4–6개, 더 밀집한 배치, 같은 종류 중복을 선택했다. YCB visual 폴더의 9개 사용 요청은 실제 `Axis_Aligned_Physics` 폴더의 physics USD 4개 사용으로 대체됐다. 기존 6종과 합친 10종 pool, instance별 역할, geometry 기반 sampling으로 확장했다. |
| 2026-10-09 역할·pool 수명주기 교정 | 사용자가 episode Target·Blocker를 먼저 정한 뒤 배치 retry에서 유지하도록 선택했다. GPU 반복 reset의 simulation flag 전환과 live gravity flag 전환에서 문제가 재현돼 둘 다 폐기했다. Pool topology·native gravity를 유지하고 inactive COM에 countergravity를 주는 `TaskObjectPool`로 변경했다. 8개 환경·16초 반복 reset과 동일 종류 역할 선지정 검증이 통과했다. |
| 2026-10-09 YCB 상하 방향 교정 | 사용자가 006의 뒤집힘 수정 및 다른 YCB 확인을 요청했다. 실제 렌더에서 003·004·005도 뒤집혀 있어 4종 모두 X +90°에서 −90°로 교정했다. 수정 후 양쪽 렌더와 기본 8개 환경 물리·reset 검사가 통과했으며 별도 고정 pair의 5.5 cm drift 실패도 기록한다. 상세 배치 기준은 `SPAWN_RULES.md`에 분리했다. |
| 2026-10-09 선반 전체·Target 유일성·초기 가시성 구체화 | 사용자가 10종 전체 사용 허용, Target 종류 중복 금지, 앞쪽 Target, 선반 전반의 배치와 이동 여유, pivot/회전+밀기 경로를 요청했다. 4–10개·100-body pool·all-types 모드, coverage/국소 밀집도 분리, 다양한 평면 path로 변경한다. Camera의 40% 기준은 전체 비가림 silhouette 대비 실제 픽셀 수로 선택했으며 실제 camera 검사는 후속 구현이다. |
| 2026-10-09 YCB 박스 초기 정착 수정 | 사용자가 스폰 직후 낙하·흔들림 수정을 요청했다. 전물체의 2 mm 여유 제거만으로 003의 흔들림이 남아 003·004에 명시적 AABB Cube collider를 적용했다. 원본 visible mesh·Nucleus USD·mass/contact/material·active gravity는 유지한다. 4개 환경·10종 전체·10초 정착 및 reset 검사가 통과했다. |
| 2026-10-09 종류 수·instance 수와 선반 전체 밀도 교정 | 사용자가 4–10은 종류 선정 기준이었다고 교정했다. 이전 4–10 instance 제한·all-types 정확히 10개를 대체해 종류 4–10, 면적 기반 instance 수·cap 40으로 분리했다. 전체 점유율과 6개 구역 하한을 hard 조건으로 적용하며, 25–45% 생성 비교로 global 35%·구역 10%·이웃 75%를 현재 구현 기본값으로 선택했다. |
| 2026-10-09 역할 OBB 표시 | 사용자가 Target은 초록색, Blocker는 빨간색 OBB로 표시하도록 요청했다. 현재 pose의 debug draw 표시를 스크립트에 추가했고 USD·물리 상태 무변경, headless reset 갱신과 실제 GUI 표시를 제한된 실행에서 확인했다. |
| 2026-10-09 중복 종류 편중 교정 | 사용자가 한 종류가 공간을 과하게 채우는 문제를 지적했다. 큰 footprint 우선 복사본 선택을 폐기하고 선택된 non-target 종류의 최소 active 개수를 우선하는 균형 구성으로 변경했다. 종류·역할·density·clearance 조건을 유지하고 현재 source의 기하 40개와 일반 모드 4개 GPU 환경·5.03초 검사가 통과했다. |
| 2026-10-09 책임별 모듈 통합 | 사용자가 과하게 세분화된 패키지 구조를 정리하도록 요청했다. Metadata·asset 생성·환경 설정·spawn/reset·순수 sampling·visualization의 6개 모듈로 통합하고 기본값·동작을 유지했다. 이전 모듈은 wrapper 없이 제거했다. 통합 후 23개 unit test·simulator 없는 import·CLI와 4개 GPU 환경·5.03초 검사가 통과했다. |
| 2026-10-09 Blocker 앞 hand 시작 randomization | 사용자가 충돌 없는 hand 시작 영역 randomization과 elbow-down 금지를 요청했다. USD FK·SciPy IK·scene collision과 별도 robot seed를 구현했다. 초기 151-step 검사 뒤 callback·실패 시 state 유지·COM cache·arm 이웃 OBB 처리를 보강했고 최종 24개 시작·5초 stress와 29개 unit test가 통과했다. |
| 2026-10-10 Tool 전방 교정 | 사용자가 아래를 향하던 tool을 선반 정면으로 향하게 요청했다. Palm normal을 전방으로 맞췄던 초기 orientation을 교정해 TCP +Y와 wrist-3 +Z가 world −X를 향하게 한다. 시작 영역·elbow-up·15 mm scene 여유는 유지하고 실제 PhysX TCP 회전과 tool 축을 검증한다. |
| 2026-10-10 Tool 기준 hand −90° roll | 사용자가 tool Z축 기준 hand 회전을 요청했다. 기존 vendor adapter `(0,-90,0)` deg에 별도 `mount_tool_roll_deg=-90`을 추가해 physical tool body인 `wrist_3_link`의 +Z축으로 hand 조립 transform을 회전한다. Tool 전방 −X와 기존 시작 조건은 유지하며 TCP 목표는 X→+Z, Y→−X, Z→−Y다. |
| 2026-10-10 말단–hand 원통 연결부 | 사용자가 실제 센서·연결부 공간을 고려해 길이 5 cm 원통을 요청했다. Wrist-3 compound collider로 직경 5 cm의 visible `SensorMount`를 추가하고 hand fixed anchor를 physical tool +Z로 5 cm 옮긴다. Hand roll·TCP 방향·39 body·18 DOF를 유지하며 enabled robot collider는 40개가 된다. 원본 USD·기존 mass/inertia는 유지하고 실제 센서 질량·F/T 구현과 구분한다. |
| 2026-10-10 중앙 F/T·초기 vision | 원통 중앙 fixed-joint F/T와 앞쪽 camera를 추가한다. 원통의 두 25 mm rigid half를 통해 hand load가 중앙 joint를 지나게 하고 raw sensor-frame wrench를 읽는다. 실제 같은 pose의 비가림 silhouette로 Target ≥40%·Blocker 전체 가시성을 승인하고 초기 RGB-D를 보존한다. Nominal mount mass 0.20 kg는 구현값이며 실제 하드웨어 질량으로 확정하지 않는다. |
| 2026-10-10 독립 tactile 17채널 | 기존 URDF tactile link 17개에 독립 ContactSensor를 연결하고 net normal force norm >0.01 N을 binary로 관측한다. Palm·finger pad-2의 원래 CAD slot을 메우던 부모 5개 convex hull을 convex decomposition으로 바꿔 접촉이 pad까지 닿게 한다. Source URDF·visual·mass/inertia·41 body·18 DOF·41 collider prim은 유지한다. 실제 sphere contact로 모든 채널 ON/OFF와 reset/다중 환경 cache 보존을 확인했다. |
| 2026-10-10 시간 입력 축소 | 사용자가 16-step history를 previous step까지만 제공하도록 요청했다. 현재 t·직전 t−1의 두 sample로 줄여 기본 입력을 1960D에서 700D로 변경하고 temporal convolution을 1개 layer로 줄인다. GNN·역할 attention·18D action·좌표계·PPO rollout 길이는 유지한다. 기존 16-step checkpoint와 새 규격의 이어 학습은 호환되지 않는다. |

Notion에 정리된 기존 문헌 논의에서 다음 교정을 유지한다. 논문의 성과를 우리
과업의 성공·속도·상태 이해 결과로 기록하지 않는다.

| 문헌 | 유지할 근거와 적용 한계 |
|---|---|
| [Rotating without Seeing](https://arxiv.org/html/2303.10880) | Binary tactile·짧은 history·PPO/task reward로 접촉 이력을 활용한 선례. 별도 이해 보상이 항상 필수라는 주장에는 반례지만, 우리 다물체 과업에서 충분하다는 증거는 아니다. |
| [Enhancing Tactile-based Reinforcement Learning for Robotic Control](https://arxiv.org/html/2510.21609v1) | Forward dynamics 등 보조 목적의 근거. Auxiliary replay buffer와 실행 recurrent memory를 구분하고, FD+확장 buffer의 개선을 loss 단독 효과로 해석하지 않는다. |
| [RoboPack](https://arxiv.org/html/2407.01418) | Dense packing의 초기 vision+tactile 선례. Recurrent GNN 추정·dynamics·지도학습·MPC 구조이며 우리 단일 정책을 검증한 것이 아니다. Box pushing은 push 사이 vision을 갱신한다. |
| [HORA](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf) | Privileged 정책 학습 후 고정하고 history adaptation module을 latent L2 오차로 별도 학습한다. Task reward만으로 history를 해석한 선례로 인용하지 않는다. |
| [ReNN](https://richardrl.github.io/relational-rl/) | 관계 처리와 RL 공동학습의 선례. 명시적 물체 goal/velocity, SAC+HER·curriculum의 stacking 실험이다. GNN 단독 효과나 rapid 실행 성능으로 확대하지 않고 64D·3 rounds 등도 우리 명세로 채택하지 않았다. |

## 7. 현재 Isaac Lab 구현

새 환경의 위치는 [src/contact_rearrangement](../src/contact_rearrangement/README.md)다.
장면 확인용 `ManagerBasedEnv`는 빈 action manager와 PD hold를 사용하며
`sensors` group에서 raw F/T·17채널 tactile/valid·초기 RGB-D·역할 mask·camera
pose를 반환한다. 학습용 `ContactRearrangementRLEnv`는 별도 설정으로 18D
OSC/hand action·Reward v2·종료 조건과 700D `policy` group을 사용한다.
초기 visible-depth 기하·현재 EEF 재표현·현재/직전 두 sample, GNN/attention/temporal
encoder와 shared actor/critic·custom PPO를 구현했다. `train.py`는
`--task Isaac-Contact-Rearrangement-UR5e-v0`의 registry에서 환경·agent 설정을
읽으며 `env.py`가 auto-reset 이전 final observation을 PPO에 보존한다.

### 장면과 초기 설정

- UR5e + RH56E2 **오른손**을 fixed joint로 하나의 articulation에 조립한다. 손의 fixed link와 mimic 관계를 유지하고 contact-report API를 활성화한다. 기존 vendor adapter `mount_rotation_deg=(0,-90,0)` deg에 `mount_tool_roll_deg=-90` deg를 추가한다. 추가 roll은 physical tool body인 `wrist_3_link` local +Z축을 기준으로 hand 조립 transform에 적용한다. 같은 tool +Z로 hand anchor를 5 cm 옮기고 길이 5 cm·직경 5 cm 원통을 두 25 mm rigid half로 조립한다. 중앙 fixed joint를 F/T 측정점으로 삼아 hand load가 통과하게 한다. 현재 41 body·18 DOF·41 enabled robot collider이며 nominal 원통 질량 0.20 kg는 실제 센서 질량이 아닌 구현 기본값이다. 원본 asset과 기존 wrist mass/inertia는 유지한다.
- Kinematic speedrack shelf와 **기존 6종 + YCB 4종 = 10종** asset을 사용한다. YCB는 `NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics`의 `003_cracker_box`, `004_sugar_box`, `005_tomato_soup_can`, `006_mustard_bottle`이다. 해당 폴더에서 확인한 4개 USD는 rigid body와 collision이 이미 authoring되어 있다.
- 장면당 **4–10종**을 선택하고 종류별 최소 하나씩 사용한다. 요구 면적에 도달할 때까지 추가 가능한 non-target 종류 중 active 개수가 가장 적은 종류에 복사본을 추가하며 동률은 무작위로 정한다. Blocker도 해당 종류의 첫 instance로 센다. 현재 같은 pool capacity에서는 non-target 개수 차이가 최대 1이다. Target은 정확히 1개이며 별도 종류별 고정 최대값을 추가하지 않는다. `all_asset_types=True`는 catalog 10종을 모두 선택하고 같은 균형 원칙으로 중복을 추가한다. Instance 최소 4·cap 40, 종류당 10-copy·환경당 100-body pool은 별도 값이다. 공간 배치는 packing을 위해 큰 footprint부터 진행하며 복사본 종류 선택 순서와 구별한다. Inactive는 `z=4.0/4.4/4.8/5.2 m`에 주차하고 속도를 0으로 설정한다. Parking은 작업층 camera ROI 밖에 두며 USD visibility를 수정해 숨기지 않는다. **Pool의 모든 body는 simulation enabled와 native gravity를 유지**해 topology를 바꾸지 않는다. 후속 actor·reward·종료에는 active mask를 적용한다. Observation K를 100으로 확정하는 뜻은 아니다.
- `TaskObjectPool`은 매 physics write/substep에서 inactive body에만 world force `-m*g`를 추가해 중력을 상쇄한다. PhysX `apply_forces_and_torques_at_position`에 실제 **world COM 위치**를 명시해 asset root pivot과 COM의 차이로 불필요한 토크가 생기지 않도록 한다. Active body의 추가 힘은 0이며 native gravity와 접촉 동역학을 유지한다. Live CPU simulation/gravity flag나 USD visibility를 매 reset마다 바꾸지 않는다.
- `TaskObjectPool.write_object_link_pose_to_sim`은 상위 writer 호출 뒤 `_object_com_pose_w.timestamp=-1.0`으로 COM cache를 무효화한다. Physics step 없이 같은 timestamp에서 연속 reset해도 다음 COM 읽기를 갱신하며 countergravity의 적용점을 최신 COM으로 유지한다. 추가 physics step이나 simulation/native gravity flag 변경은 없다.
- **Target 종류는 active 장면에 정확히 한 번** 등장한다. 해당 종류는 Blocker·주변 물체에서 제외하고, 선택한 다른 종류의 중복은 허용한다. 초기 baseline은 Target 1개와 Blocker 1개이며 episode 시작에 쌍을 먼저 선택한다. 종류 수·집합, 전체 active instance ID와 개수를 pose retry 전에 고정하며 yaw·위치만 다시 뽑는다. Cap·복사본 수 안에서 면적을 충족하지 못하거나 300 layout 시도에서 배치하지 못하면 구성을 교체하거나 조건을 낮추지 않고 실패한다.
- `cfg.target_asset`·`cfg.blocker_asset`과 CLI의 `--target_asset`·`--blocker_asset`으로 역할의 종류를 고정할 수 있다. 기본 `None`은 episode 역할을 처음에 한 번 추출한다. 두 옵션에 같은 종류를 지정하면 Target 유일성에 위배되어 오류다. Blocker 종류만 고정한 경우 Target 후보에서도 그 종류를 제외한다. 이는 외부 역할 지정이며 Blocker의 최종 목표 pose를 제공하는 것이 아니다.
- CLI `--min_asset_types`·`--max_asset_types`는 종류 수, `--min_objects`·`--max_objects`는 instance 최소값·cap이다. `--min_shelf_occupancy`·`--min_region_occupancy`는 global/구역 하한을 지정한다. 종류와 instance 개수를 혼용하지 않는다.
- 로봇 base는 환경 로컬 `(0, 0, 0.79505)`, quaternion `(w,x,y,z)=(0,0,0,1)`, 선반은 `(-0.7,0,0)`에 놓인다. Quaternion 표기는 Isaac Lab 순서다.
- 기존 2×3 고정 배열과 `cup_1` Target/`mug_2` Blocker는 초기 구현 이력이다. 현재는 환경별 seed와 reset 횟수로 episode 역할을 먼저 정하고 그 역할에 맞는 upright yaw·배치를 뽑는다. 선택되지 않은 환경의 reset sequence는 유지한다. 종류·역할별 배치 실패율은 따로 확인하며, 역할을 placement 성공률에 따라 바꿔 분포를 감추지 않는다.
- 물체의 spawned rigid-body local bounds를 읽고 canonical upright rotation을 적용한다. YCB 4종은 local −Y를 world +Z로 옮기는 X축 −90° 회전 후 world Z yaw를 적용한다. 수정 전 양쪽 렌더에서 003·004·005의 라벨과 006의 뚜껑이 모두 아래를 향해 기존 +90°를 교정했다. Root pivot과 bounding center를 구분해 XY를 보정하고, 실제 lower Z bound로 선반 위 높이를 정한다.
- 모든 물체의 root Z는 `shelf_top - lower_z`로 정해 bounds 바닥을 선반 상면에 바로 맞춘다. 기존 2 mm 여유는 제거했다. 003·004만 `FlatBoxUsdCfg`로 visible scanned mesh의 collision을 끄고 동일 body-local AABB 크기·center offset의 invisible Cube collider를 추가한다. 기존 single body를 유지하고 clone spawn 시점에만 적용하며 Nucleus 원본을 수정하지 않는다. Authored mass·원본 contact 속성·physics material을 유지하고 active body의 gravity·simulation을 그대로 사용한다. 자동 fit `boundingCube` approximation은 최종 구현이 아니다.
- 실제 shelf collision에서 support와 상부 panel을 읽는다. 현재 기본 asset의 작업층은 환경 로컬 `x=[-0.9,-0.5]`, `y=[-0.455,0.455]`, 상면 `z=1.05 m`, 위쪽 panel 하단 `z=1.48 m`이며 물체 footprint에 edge margin과 상부 여유를 적용한다. Asset override에서는 해당 geometry를 다시 측정한다.
- 기본 물리 주기는 `1/120 s`, decimation은 4로 환경 step은 `1/30 s`다. 기본 환경 수 1, 간격 2.5 m다.
- 장면 확인용 `spawn_scene.py`는 기본적으로 시뮬레이션 시간 5초마다 전체 환경을 reset해 새 배치를 생성한다. 실제 `env.step_dt`로 간격을 계산하며 현재 설정에서는 150 step이다. `--reset_interval N`으로 step 간격을 지정하고 0이면 자동 reset을 끈다. 이 설정은 실행 스크립트의 확인 주기이며 학습 episode timeout을 확정하는 것은 아니다.
- `RoleObbVisualizer`는 Target 초록색·Blocker 빨간색 OBB를 표시한다. 각 역할의 12개 모서리를 현재 body link pose·local bounds에서 계산해 YCB upright·root offset·합성 scale을 반영하고 매 step과 전체/부분 reset 뒤 갱신한다. Debug draw만 사용하며 USD prim·collider·material·physics state를 쓰지 않는다. GUI·livestream에서 기본 표시하고 `--disable_role_obb`로 끈다. 일반 headless에서는 생략하며 `--headless --validate`에서 viewport 없이 overlay 검사를 실행한다. 제한된 headless·GUI 검사 결과는 8절에 기록한다.
- Reset은 물체 sample 뒤 Blocker 앞의 robot 시작 후보를 찾고 실제 앞쪽 camera 렌더로 초기 가시성을 승인한다. Camera 32개 후보가 실패하면 같은 역할·종류·active ID·개수로 pose를 다시 제안하며 최대 8개 layout을 검사한다. 승인된 pose·A₀·robot 시작 metadata와 initial snapshot을 해당 환경에 저장하고 후보 실패 시 기존 physical pose·metadata로 복구한다. Render는 physics/settling step을 추가하지 않으며 pool의 simulation/native gravity와 inactive parking/countergravity를 유지한다.
- 17개 tactile link에 각각 독립 ContactSensor를 붙이고 binary contact·valid mask를 관측한다. 중앙 fixed-joint raw F/T를 유지하고 학습 actor에는 sensor-frame 중력·bias 보정값을 제공한다. Filtering·실제 하드웨어 threshold/질량 교정은 후속 단계다.
- 학습 `UR5eHandLearningSceneCfg`에만 arm/spacer의 9개 금지 접촉 sensor를 추가한다. 설정 factory는 `src/sensors.py`, 최근 4-substep/각 partner >1 N 판정은 `mdp/actions.py`, threshold는 `ContactRearrangementRLEnvCfg.forbidden_contact_threshold`에 둔다. Hand 조작 접촉을 이 판정에 포함하지 않으며 Target의 실제 이동 보호는 MDP가 맡는다. 이 접촉 분류·threshold는 초기 task 기본값이다.

### Blocker 앞의 robot 시작 randomization

`cfg.robot_start`에 `RobotStartRules`를 둔다. 기본 제안 영역은 Blocker의
**회전 footprint +X 최전방보다 12–40 cm 앞**, 동시에 **선반 앞 경계에서
6 cm 이상 밖**, Blocker bounding center의 **Y ±8 cm**, shelf top 위
**20–28 cm**다. 거리 기준은 palm TCP `right_plam_force_sensor`이며 손가락
표면이 아니다. 현재 TCP local +Y인 손가락/tool 전방은 **world −X, 선반 방향**,
local +Z인 palm normal은 **world −Y**, local +X는 world +Z를 향하게 고정한다.
조립된 USD에서 TCP +Y는 UR `wrist_3_link`의 local +Z인 tool 전방과 일치한다.
기존 vendor adapter `(0,-90,0)` deg는 유지하고 tool +Z 기준 **hand −90° roll**을
mount에 추가한다. Arm wrist 관절각만 바꾸는 것이 아니라 hand의 mount 회전과
translation을 함께 변환한다. `tool0` attachment의 축은 physical tool body와
다르므로 `tool0` 자체 local Z를 회전축으로 쓰지 않는다. 구체적인 영역 식과
회전 행렬은 [SPAWN_RULES.md](SPAWN_RULES.md)에 있다.

실제 센서·연결부 공간을 반영하는 길이 5 cm 원통을 wrist-3와 hand 사이에
둔다. Hand fixed anchor는 physical tool +Z로 5 cm 이동하고 현재 −90° roll은
유지한다. 현재 원통은 중앙 F/T joint를 둔 두 rigid half이므로 **41-link FK·
18 DOF**, 외부 scene·arm self 검사에는 **41개** robot collider를 쓴다.
Nominal mass/inertia는 구현값이며 실제 하드웨어 교정과 구분한다.

`assets.py`의 `RobotPoseModel`은 실제 조립 USD의 41-link joint frame·axis·scale로
FK를 구성하고 SciPy bounded IK로 arm 해를 찾는다. Finger q는 기존 열린 손의
기본값을 유지한다. Elbow-up은 관절각 부호로 결정하지 않고 elbow joint center가
shoulder–wrist-1 chord의 가장 가까운 점보다 **world Z로 5 cm 이상 위**인지
검사한다. Elbow-down/낮은 elbow 해는 채택하지 않는다.
현재 실제 pose에서 모델 FK를 대조하므로 default 자세로 선행 reset하지 않는다.
`accept_candidate` callback으로 한 TCP의 IK 해가 scene collision에서 탈락하면
같은 TCP의 다른 seed/branch도 탐색한다. 모든 branch의 완전 탐색을 의미하지 않는다.

모든 enabled arm/hand collision OBB를 모든 active object(역할 포함)·선반 개별
enabled collider·ground와 **15 mm 여유**로 검사한다. UR arm 7개 link의 21쌍 중
두 arm joint 이내의 11쌍은 제외하고 거리 3 이상인 10쌍을 2 mm 여유로 검사한다.
손 자체의 모든 self pair를 검사하지 않으므로 손 전체의 self collision 안전성을
검증했다고 기록하지 않는다.
이 판정은 초기 pose의 보수적인 OBB 조건이며 실제 mesh 거리·접촉·재배치 경로의
도달성이나 swept collision을 보장하지 않는다.

물체 sample을 고정한 뒤 `episode_seed XOR 0x5D2EA91C`의 독립 RNG로 최대
**80개** robot 시작 후보를 찾는다. 실패하면 이유를 포함해 명시적으로 중단하며
elbow-down fallback이나 물체 배치 재선정으로 조건을 완화하지 않는다.
`robot_initial_joint_pos`·`robot_start_tcp_pos_w`·`robot_start_region`·
`robot_start_elbow_height`·`robot_start_attempts`를 기록한다. 부분 reset에서는
선택하지 않은 환경의 state·목표·reset sequence를 유지한다.
선택된 모든 환경의 물체·robot 후보가 CPU에서 성공한 뒤 default reset과
실제 state/PD target·metadata 적용을 진행한다. 후보 실패 시에는 기존 선택
환경의 physical pose·metadata도 유지한다.
`cfg.robot_start.enabled=False`이면 기존 기본 자세를 사용한다. 수치는 후속
제어·학습 분포에 맞춰 조정할 수 있다. Reset 검증은 실제 PhysX TCP의 전체
회전 오차를 **0.01 rad 이하**로 확인하고 wrist-3 local +Z가 world −X인지도
대조한다. 2026-10-09의 24개 시작과 반복 reset·5초 물리 검사는 이전 아래를
향하던 tool 자세의 결과이며 조건·범위는 8절에 기록한다.

### 밀집 배치와 해소 가능성의 초기 조건

좌표·수치·샘플링 순서·계산식은 [상세 배치 기준](SPAWN_RULES.md)에 정리했다.

샘플러는 처음부터 Target 접근을 막되 간단한 재배치 해법이 존재하는 baseline을
만든다. Blocker는 Target의 로봇 쪽인 +X에 두고, 초기 Target과 +X 접근 방향으로
고정한 `A_0`에 대해 Blocker footprint의 30% 이상이 겹치도록 한다. 주변 물체는
초기 `A_0` 밖에 둔다. Footprint는 실제 body bounds에서 얻은 보수적 upright
box의 평면 투영이며 mesh의 실제 점유 면적과 동일하지 않다.

Target X는 Blocker가 앞에 들어갈 수 있는 전체 유효 범위에서 뽑는다.
Blocker의 Y도 ±30 mm로 제한하지 않고 footprint가 A₀와 겹칠 수 있는 범위에서
제안한 뒤 실제 겹침을 검사한다. 이 offset 제안은 부분 가림을 허용하지만 영상
가시성의 40% 조건을 증명하지 않는다.

현재 구현 기본값은 최소 footprint gap **3 mm**, edge margin **15 mm**,
margin을 뺀 shelf 범위 대비 footprint 집합의 **X span 55%·Y span 65% 이상**이다.
별도로 active 물체의 **75% 이상이 4 cm 이내 footprint 이웃**을 갖도록 한다.
전체 footprint 면적 합/전체 usable floor는 **35% 이상**이고, X 2 × Y 3의
각 구역은 footprint 교차 면적 합/구역 면적이 **10% 이상**이어야 한다. 면적
기준이 미달이면 넓은 X/Y span만으로 채택하지 않는다. 35%는 실제 측정 bounds의
25–45% 생성 비교에서 선택한 초기 구현값이며 최종 연구 명세가 아니다. Reserved
path는 점유 면적으로 계산하지 않는다. 정확한 mesh 점유율·동역학 안정성·영상
가시성을 보장하는 수치도 아니다.

**Clearance witness**는 병진·footprint 모서리의 평면 pivot·중심 회전 후 병진·
작은 앞쪽 이동 후 회전/병진 등의 후보에서 하나를 찾는다. Pivot은 고정점 주변의
원호로 검사하고 회전 구간을 최대 5°씩 나눠 arc sagitta로 확장한 보수적 swept
polygon을 사용한다. 전체 경로가 support 안에 있고 Target·주변 물체와 떨어져야
한다. 경로를 예약한 뒤 부족한 6개 구역을 우선 채우는 위치·이웃 proposal을
평가해 주변 물체를 놓는다. Episode 구성은 유지하고 최종 hard threshold를 모두 검사한다.

Witness는 생성·검증용 privileged metadata다. Actor에 Blocker의 최종 goal pose로
입력하는 것이 아니며 정책이 이 병진 해법만 수행하도록 결정한 것도 아니다.
이 조건은 보수적 평면 기하에서 경로 하나가 존재한다는 뜻이다. Bounding corner
pivot은 실제 hand contact point를 인증하는 것이 아니다. UR5e의 접근
가능성, 손의 접촉점과 충돌, OSC로 밀 수 있는지, 마찰·회전·전도를 포함한 동역학,
센서 기반 종료 판정과 실제 재배치 성공은 아직 입증하지 않았다.

### 중앙 fixed-joint F/T와 초기 센서 observation

원통 중앙의 fixed joint에서 PhysX incoming-joint reaction을 직접 읽는다.
측정 frame은 중앙점에 놓인 physical tool XYZ이며 parent→child 방향의
`[Fx,Fy,Fz,Tx,Ty,Tz]`, 단위는 `[N,N,N,N·m,N·m,N·m]`다. 힘·토크에는 중력·
관성·접촉·외력이 포함된다. Raw 관측에 자동 tare나 중력 보정을 적용하지 않는다.
명시적인 `tare()`는 그 시점의 bias만 저장하며 중력 보정과 다르다. Reset은
선택한 환경의 cache·bias를 비우고 `wrench_valid=False`로 둔다. 다음 완료된
physics step 뒤 timestamp가 증가했을 때 반력을 읽어 valid로 갱신한다.

`ObservationsCfg.sensors`는 concatenate하지 않는 dictionary다. `wrench` 6D,
`wrench_valid` 1D, `tactile`·`tactile_valid` 각 bool 17D, RGB `480×640×3`,
depth `480×640×1` m, Target/Blocker 순서의
실제 instance mask `480×640×2`, camera world position+ROS quaternion(wxyz) 7D를
반환한다. RGB/depth/mask/camera pose는 승인한 초기 snapshot의 cache를 읽으며
episode 중 image를 갱신하지 않는다. 비가림 reference·가시성 비율은 reset의
acceptance metadata이며 observation으로 제공하지 않는다. 학습용 `policy` group은
이 초기 snapshot의 visible depth에서 추정한 OBB10과 robot/sensor/action 이력을
별도로 구성한다. Raw sensor dictionary와 actor의 중력·bias 보정 F/T를 구분한다.

### 기존 hand link의 독립 tactile 17채널

Scene에 `contact_` + canonical tactile body name으로 17개 독립 ContactSensor를
둔다. 각각 `{ENV_REGEX_NS}/Robot/ROASRightHand/{body_name}` 한 body만 검사한다.
순서는 **palm 1개 → thumb 1–4 → index 1–3 → middle 1–3 → ring 1–3 → little 1–3**다.
기존 URDF의 palm 이름 `right_plam_force_sensor`는 철자를 그대로 유지한다.
초기 URDF 검사에서 정확한 17개 link와 기존 collision mesh를 확인한다.

`cfg.tactile.force_threshold=0.01 N`은 finite positive 값이어야 한다.
`env.binary_tactile_sensor.net_forces_w`의 world net normal contact force
`[N_env,17,3]`에 대해 norm **> threshold**를 `contacts[N_env,17]` bool로 계산한다.
`valid`도 같은 shape의 bool이다. SDK `ContactSensorCfg.force_threshold`는
사용하지 않는 air/contact-time용 설정이므로 binary threshold로 대신 쓰지 않는다.
각 sensor는 `update_period=0`, `history_length=0`이며 120 Hz physics의 마지막
완료 step, 즉 30 Hz 관측의 마지막 substep 값을 읽는다. 자유 공간의 중력·
관절하중 자체는 normal contact가 아니다. Render/reset의 같은 timestamp에서는
stale report를 다시 읽지 않는다. 선택 reset은 cache를 zero·invalid로 비우고
다음 완료 physics step 뒤 valid로 갱신하며 미선택 환경 cache는 유지한다.

기존 단일 convex hull이 palm·finger pad-2의 CAD slot을 메워 pad collider까지
접촉이 닿지 않는 문제를 교정했다. `assets._preserve_tactile_parent_slots()`는
`right_plam_1`, `right_index_2`, `right_middle_2`, `right_ring_2`, `right_little_2`의
기존 부모 collider 5개만 convex decomposition으로 바꾼다. 기본값은 최대
128 hull·hull당 64 vertex·1M voxel·error 0.5%·shrink wrap·min thickness 0.1 mm다.
17개 pad collider·source URDF·visual·mass/inertia는 유지한다. Robot은
41 body·18 DOF·**41 collider prim**이며 decomposition의 cooked hull 수와
collider prim 수를 구분한다. Sensor 책임은 기존 `sensors.py`에 통합한다.

### 초기 vision의 실제 silhouette 판정

`InitialVisionCfg`는 앞쪽 **+X camera에서 −X를 보는 방향**,
`min_target_visible_fraction=0.40`, `min_blocker_visible_fraction=1.0`을 설정한다.
분모는 같은 camera intrinsics·상대 pose에서 해당 역할 물체만 외부 가림 없이
렌더링한 **전체 silhouette 픽셀 수**다. 실제 장면의 해당 instance mask 픽셀
수를 분자로 사용한다. 다른 물체뿐 아니라 shelf·robot 가림도 포함하며 물체
self-occlusion은 reference silhouette에 자연스럽게 반영한다. 둘 다 전체 bounds가
frame 안에 있어야 하고 reference 최소 32 pixel을 요구한다. Blocker는 전체
reference 면적에서 최대 2 pixel의 손실을 허용하며 contour 경계 band로 빼지 않는다.
가시 비율의 상한은 추가하지 않는다.

`InitialCamera`는 640×480 RGB·image-plane depth·instance segmentation을 읽는다.
카메라 X는 선반 전면보다 0.55–0.95 m 앞, Y는 Blocker 중심 ±0.35 m에서 제안해
±0.50 m로 제한하며 높이는 선반 상면 **+0.18–0.48 m**다. 두 역할 사이를 보되
aim Y ±0.06 m, aim 높이 상면 +0.10±0.03 m, optical-axis roll ±3°를 무작위로
정한다. Pinhole focal length 18 mm·horizontal aperture 24 mm로 수평 시야각은
약 67°다.
두 역할의 전체 bounds를 실제 camera near/far clip `0.02–2.5 m`와 화면 안에서
검사한다. Nonfinite/range 오류·음수 jitter는 거절하며 `camera_side="+X"`,
전체 frame 포함과 같은 pose 비가림 reference 규칙은 필수다.

동일 형상의 10종 visual reference를 physics 시작 전에 작업층 밖의 격리 lane
`z+(20+5×env_id) m, y+5 m×type index`에 만들고 physics API를 제거한다. 환경
사이도 Z로 5 m 분리한다. Reference 물체와
`SilhouetteCamera`를 같은 offset으로 평행 이동해 실제 물체·camera의 상대 pose를
보존한다. Live body의 visibility·physics topology는 바꾸지 않는다.
순서는 **역할/종류/active ID/개수 고정 → pose·elbow-up 제안 → native state 적용 →
camera·실제 mask·동일 pose reference → pixel 승인 → 초기 snapshot·A₀ commit**이다.
최초 5회·이후 2회 `sim.render()`로 texture를 준비하며 physics step은 진행하지
않는다. 따라서 settled pose가 아니라 승인된 reset pose의 가시성을 검사한다.
32개 camera 후보가 실패하면 같은 composition의 pose를 재시도하고 최대 8개
layout에서도 실패하면 기존 state·metadata로 복구해 명시적으로 실패한다.
실패 복구는 물체 COM velocity와 robot state/목표뿐 아니라 모든 초기 image·
camera·Target/Blocker 가시성 buffer도 보존한다.

Camera는 로컬 `InitialCamera` subclass를 쓴다. Isaac Lab 2.3.2의 첫 Fabric pose
조회는 `Camera.reset → _update_poses → XformPrimView`의 stage sync에서 Kit
`app.update()`를 호출하므로 physics가 진행될 수 있다. `_update_poses`·
`set_world_poses`·`camera.update` 동안 `/app/player/playSimulations=False`로
보호하고 `try/finally`로 이전 값을 복구한다. 첫 camera data 읽기 전에 cache를
준비하며 SDK·live body physics flag는 수정하지 않는다.

`SpawnRules.reserve_blocker_front=True`는 rotated Blocker footprint의 가장
큰 X부터 shelf front까지, Y bounds 양쪽에 `gap=3 mm`를 더한 직선 통로를
주변 물체로부터 비운다. Target A₀와 clearance witness는 유지한다. Global
35%·구역 10% 등 density 조건도 그대로 검사한다. 이는 camera proposal의
성공 가능성을 높이는 기하 조건이며 실제 silhouette 판정을 대체하지 않는다.

| 파일 | 역할 |
|---|---|
| [shelf_rearrangement_env_cfg.py](../src/contact_rearrangement/contact_rearrangement/shelf_rearrangement_env_cfg.py) | 공통 scene·camera·MDP manager term과 환경 기본값 |
| [config/ur5e/osc_env_cfg.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/osc_env_cfg.py) | UR5e/RH56E2 장면 확인·학습 설정, OSC용 actuator |
| [config/ur5e/__init__.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/__init__.py) | Gym task와 env/agent entry point 등록 |
| [env.py](../src/contact_rearrangement/contact_rearrangement/env.py) | Auto-reset 전 실제 terminal observation 보존 |
| [mdp/actions.py](../src/contact_rearrangement/contact_rearrangement/mdp/actions.py) | Base 기준 18D OSC/hand action과 arm/spacer 금지 접촉 이력 판정 |
| [mdp/state.py](../src/contact_rearrangement/contact_rearrangement/mdp/state.py) | MdpCfg·기하·부분 reset·한 transition에 한 번 갱신하는 이력/누적량 |
| [mdp/observations.py](../src/contact_rearrangement/contact_rearrangement/mdp/observations.py) | 초기 기하의 현재 EEF 재표현·현재/직전 두 sample·sensor-frame F/T 보정 |
| [mdp/rewards.py](../src/contact_rearrangement/contact_rearrangement/mdp/rewards.py) | Reward v2와 dt 보정 |
| [mdp/terminations.py](../src/contact_rearrangement/contact_rearrangement/mdp/terminations.py) | 성공·실패 판정 |
| [mdp/events.py](../src/contact_rearrangement/contact_rearrangement/mdp/events.py) | USD 측정·역할/reset·robot 후보·camera retry·commit/rollback·MDP reset |
| [src/catalog.py](../src/contact_rearrangement/contact_rearrangement/src/catalog.py) | Simulator 없이 읽는 10종 metadata·100개 pool·upright·parking·asset 경로 |
| [src/assets.py](../src/contact_rearrangement/contact_rearrangement/src/assets.py) | 로봇/F/T 조립·raw reader·tactile CAD slot·FK/IK·물체/pool |
| [src/sampling.py](../src/contact_rearrangement/contact_rearrangement/src/sampling.py) | 순수 density·SE(2) 경로·robot 시작 영역/elbow/3D OBB helpers |
| [src/sensors.py](../src/contact_rearrangement/contact_rearrangement/src/sensors.py) | 초기 RGB-D/PCA 기하·raw F/T·17채널 tactile·금지 접촉 sensor factory |
| [src/visualization.py](../src/contact_rearrangement/contact_rearrangement/src/visualization.py) | Target/Blocker OBB debug draw |
| [agents/model.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/model.py) | 공유 GNN·역할 attention·유한 temporal encoder·actor/critic |
| [agents/ppo.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/ppo.py) | 공동학습 PPO·terminal/timeout GAE·checkpoint |
| [agents/ppo_cfg.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/ppo_cfg.py) | Model/PPO와 runner 기본값 |
| [spawn_scene.py](../src/contact_rearrangement/scripts/spawn_scene.py) | 장면 실행·5초 reset·역할 OBB·검증 |
| [train.py](../src/contact_rearrangement/scripts/train.py) | Task 선택·학습 rollout·PPO update·checkpoint |

구조는 [sweeping_policy](../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy)의
공통 `shelf_sweep_env_cfg.py`, `config/ur5e`의 concrete 환경과 등록,
`mdp`의 manager 함수, `config/ur5e/agents`의 runner 설정에 대응한다.
`Isaac-Contact-Rearrangement-UR5e-v0`의 `env_cfg_entry_point`와
`ppo_cfg_entry_point`를 통해 설정을 불러오며 model·algorithm·기본 iteration·
저장 간격은 `ContactRearrangementPPORunnerCfg`에 둔다. GNN·현재/직전 모델과
custom PPO 구현을 유지하므로 예제의 RSL-RL 설정을 그대로 채택하지 않는다.
장면 지원의 asset·sensor·sampling 구현은 각각 하나의 모듈로 유지한다.
이전 flat 모듈을 다시 import하는 wrapper는 만들지 않는다.

스폰 실행은 **catalog·CLI → camera 활성화 AppLauncher → scene/reference/camera
생성 → USD 측정/reset·pixel acceptance → physics step·F/T 갱신·OBB 표시** 순서다.
Metadata와 순수 sampling은 simulator 없이 읽으며 simulator 의존 모듈은
AppLauncher 뒤에 로드한다. `mdp/events.py`는 실제 geometry를 sampler에 전달하고
그 결과를 state로 적용한다. Pool은 매 substep countergravity를 적용한다.
학습에서는 초기 pixel acceptance 후 visible-depth geometry를 저장하고
history/reward 누적량을 reset한 뒤 OSC→physics→센서→transition reward/종료→policy
관측 순서로 진행한다. `mdp/state.py`가 공통 transition을 한 번만 갱신해
manager 호출에 따른 누적량·history 중복 갱신을 막는다. Package tree와 실행·
수정 위치는 [README](../src/contact_rearrangement/README.md)에 있다.

2026-10-09의 6개 모듈 통합 기록과 해당 검사 결과는 8절에 보존한다. 이후
학습 코드가 추가되고 sweeping example 구조로 재배치되었으며 현재 위치는 위
표를 기준으로 읽는다.

Asset 기본 경로는 `omniverse://192.168.0.13`이며
`CONTACT_REARRANGEMENT_*` 환경변수로 override할 수 있다. 손은 저장소의
`RH56E2_R_2025_9_11_relative.urdf`를 사용한다. 세부 명령과 변수는 패키지 README에 둔다.

### 기존 환경에서 참고한 범위

Examples와 `roas_sweeping`에서 shelf·6개 물체 종류와 배치를,
`hand_rl`에서 RH56E2 오른손 조립을, Isaac Lab의
`scripts/tutorials/03_envs/create_cartpole_base_env.py`에서 base environment
생명주기를 참고했다. 새 제어의 기준 frame은 examples의 sweeping 프로젝트에서
`subtract_frame_transforms(root_pose_w, object_or_ee_pose_w)`와 `base_link` source
frame을 사용하는 방식을 참조한다. 구체적인 참조는
[Sweeping observations.py](../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/mdp/observations.py)의
`object_position_in_robot_root_frame`와 EEF pose 변환이다. `sweep_jh`의 기존 goal/reward나 가상 F/T를
새 MDP의 실제 센서로 대체하지 않는다.
기존 환경의 목표 pose·reaching IK·sweep reward·단일 active-object reset을
새 연구 명세로 그대로 사용하지 않는다. 오래된 `docs/hand_rl`의 왼손 설명보다
현재 오른손 코드를 확인해야 한다.

## 8. 개발과 검증 기록

과거 기록의 모듈 이름은 당시 구조다. 현재 위치는 위 모듈 표를 따르며,
통합 전 검사 결과를 통합 후 runtime 검사의 통과로 사용하지 않는다.

### 2026-10-09 초기 고정 6개 장면 스폰과 reset — 이전 버전

**구현:** 독립 패키지 `contact_rearrangement`와 실행 스크립트를 추가했다.
Arm·hand는 초기 PD 목표를 유지하고 물체 6개는 중력과 접촉에 따라 시뮬레이션된다.

**검증 조건:** 이 저장소의 Isaac Lab source 2.3.2, 설치된 Isaac Sim 5.1,
RTX 3090에서 원본 Nucleus UR5e·선반·물체 USD 및 로컬 RH56E2 오른손 URDF를 사용했다.
2개 환경을 headless CUDA로 120 environment step, 4초 실행했다.

**통과한 검사:** 6개 물체 이름·순서, arm/hand 독립 관절과 17개 tactile용 링크,
fixed robot base, finite robot joint position·물체 state, 초기 높이보다 10 cm 이상
낙하하지 않음, XY 이동이 5 cm를 넘지 않음, 전체 reset 및 한 환경만 교란한 뒤
부분 reset의 물체 pose/velocity·robot 관절 위치/목표 복원. 부분 reset에서
다른 환경의 물체 state가 유지되는 것도 확인했다.

이 결과는 초기 장면과 reset의 제한된 smoke 검증이다. 재배치 성공, 센서 신호
품질, OSC 제어 성능, 파지·비파지 판별, 장기 안정성, 정책 학습과 rapid 성능을 입증하지 않는다.

아래는 **이전 고정 6개 버전 검증에 사용한 명령**이다. 같은 경로의 현재 코드가
확장되고 있으므로 지금 실행한 결과를 이전 결과와 동일하게 해석하지 않는다.

```bash
conda activate env_isaaclab
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 2 --headless --device cuda:0 --max_steps 120 --validate
```

이전 버전의 최종 로그는 `outputs/contact_rearrangement/spawn_validation.log`에 있다.
`outputs/`는 Git에서 제외되므로 다른 checkout에 로그가 없을 수 있다.

**유지할 개발 교훈:** 로컬 기본 `reset_scene_to_default`는 rigid-object collection을
복원하지 않아 별도 처리를 추가했다. Collection default state에는 link pose와
COM velocity가 들어가므로 대응 writer인 `write_object_state_to_sim`을 사용한다.
설치된 `isaaclab`이 다른 checkout을 가리켜 실행 스크립트에서 이 저장소의 source를
우선한다. 중간에 Nucleus 인증 만료와 `fast_shutdown=False` 종료 시 충돌이 있었으나,
최종 실행은 원본 asset 연결과 기본 AppLauncher 종료 설정으로 통과했다. 임시
테스트 물체 검증과 최종 원본 USD 검증을 혼동하지 않는다.

### 2026-10-09 물체 종류·중복·밀집도·역할 선지정 확장 — 검증 완료

**사용자 결정:** 장면당 4–6개 active 물체, 더 밀집한 배치, 같은 종류의 중복을
허용한다. Target과 Blocker의 종류가 같아도 서로 다른 instance이면 된다.
초기 YCB visual 폴더의 9개 사용 요청은 실제 `Axis_Aligned_Physics`에 존재하는
authored physics USD 4개를 기존 6종과 함께 사용하는 것으로 대체됐다.
Target·Blocker는 episode 시작에 먼저 지정하고 배치 재시도 동안 유지한다.
옵션으로 종류를 고정해도 같은 종류의 두 역할은 서로 다른 instance를 사용한다.

**구현 변경:** 10종 × 6-copy pool, active mask와 instance별 역할, 환경·reset별
seed, 실제 body/shelf geometry 기반 높이·support 검사, 고정 `A_0`의 초기
차단 조건, 가까운 이웃 배치와 밀집도 threshold, 전체 swept footprint의
lateral clearance witness를 추가했다. 수치와 단일 Blocker 가정은
bootstrap baseline이며 최종 curriculum·최대 K·일반 다중 Blocker 설계와 구분한다.

**중간 GPU 검증과 변경 이유:** 기존 simulation-disable 방식에서 8개 환경의
한 reset cycle은 통과했지만, active 개수를 6으로 고정한 반복 reset의 두 번째
cycle에서 body pose가 다른 환경의 정확한 parking 좌표로 바뀌었다. 보통의
낙하·밀림과 다른 현상이며, `set_disable_simulations` 전환으로 인한 GPU body
index/cache 대응 변화가 원인이라는 추론에 따라 해당 방식을 폐기한다. 엔진
내부 원인을 확정한 것은 아니다.

후속 live gravity flag 방식도 이 direct GPU 실행에서는 동작을 보장하지 못했다.
Inactive getter가 disable-gravity `1`을 반환해도 첫 4개 physics substep 뒤
`vel_z=-0.327 m/s`, `delta_z=-0.006812 m`로 정확히 `9.81 m/s²`의 낙하가
관측됐다. Nucleus asset·로봇 없이 480개 cube body만 사용하는 최소 재현에서도
active→parking 전환 뒤 gravity-disable이 반영되지 않는 현상을 확인했다. 이는
이 환경에서의 API/runtime 관측이며 Isaac Sim 전체의 일반적인 엔진 결함을
입증한 것은 아니다.

따라서 live simulation/gravity flag 전환을 피하고 **모든 pool body의 native
gravity와 simulation을 유지**한다. 당시 `object_pool.py`(이후 `assets.py`에 통합)의 `TaskObjectPool`이
inactive COM에만 매 substep world `-m*g` 외력을 적용한다. 높은 격리 parking으로
작업층과 분리하고 USD visibility는 수정하지 않는다. COM 적용은 pivot 기준
외력이 만드는 torque를 피하기 위한 선택이다.

**당시 최종 검증 (YCB 방향 교정 전):** 이 저장소의 Isaac Lab 2.3.2, Isaac Sim 5.1, RTX 3090 CUDA에서
원본 Nucleus USD와 기본 fast shutdown으로 다음 두 실행이 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 8 --min_objects 6 --max_objects 6 --headless --device cuda:0 \
  --max_steps 480 --reset_interval 120 --validate

./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 2 --seed 7 --target_asset ycb_003_cracker_box \
  --blocker_asset ycb_003_cracker_box --headless --device cuda:0 \
  --max_steps 240 --reset_interval 120 --validate
```

첫 실행은 8개 환경에서 6개 active 고정, 4초 간격 reset, 총 16초다. 두 번째는
Target·Blocker를 cracker box의 서로 다른 복사본으로 먼저 지정하고 4–6개를
생성했다. 로그에서 4·5·6개 모두 확인했으며 총 8초를 실행했다. 전체·부분 reset의
PhysX 직접 pose 읽기, quaternion·velocity·PD 목표, unselected 환경의 state·mask·
역할·고정 접근영역·reset 횟수 유지가 통과했다. Physics 후 active 낙하 10 cm 초과,
XY drift 5 cm 초과, 초기 upright 대비 tilt 15° 초과를 검사했고 inactive position·
velocity는 1e-5 허용오차 안에서 정지했다. 초기 배치의 지지·간격·밀집도·차단·
swept clearance와 native gravity/simulation flag 유지도 통과했다.

샘플러 unit test **14개**가 통과했다. 역할 선지정·rejection 뒤 역할 유지,
중복 instance·수 고정·seed 재현, clipping/SAT, 불가능한 배치의 명시적 실패,
끝점이 비어 있어도 중간 경로가 막히는 경우 등을 검사한다. 별도 기하 검사에서
측정된 10종 bounds의 60-slot pool, 6개 고정, 역할을 먼저 정한 500 seed가 모두
생성됐다. 로그는 ignored `outputs/contact_rearrangement/clutter_com_validation.log`와
`clutter_fixed_roles_validation.log`에 있다.

이는 스폰과 제한된 무동작 안정성 검증이다. Settle 이후 모든 기하 threshold의
유지, 손의 실제 접촉·도달성, 센서 신호, 재배치 성공이나 PPO 학습은 입증하지 않았다.

### 2026-10-09 YCB 상하 방향 교정과 배치 기준 문서화

**사용자 요청:** 006 mustard bottle의 뒤집힘을 수정하고 다른 YCB의 상하도
확인한다. 현재 Target·Blocker 중심 배치 기준을 자세히 설명한다.

**확인과 교정:** `Axis_Aligned_Physics` 원본 003·004·005·006을 기존 X축
+90° 상태로 각각 양쪽에서 렌더링했다. 003·004의 라벨, 005의 라벨과 캔 윗면,
006의 병뚜껑이 모두 뒤집혀 있었다. 모든 YCB의 canonical quaternion을
`(w,x,y,z)=(sqrt(0.5),-sqrt(0.5),0,0)`으로 통일해 local −Y를 world +Z로 옮긴다.
복사본도 같은 catalog 값을 사용하고 이후 world Z yaw만 추가한다.
당시 USD의 mass·collision approximation은 변경하지 않았다.

교정 후에도 같은 4종의 양쪽을 8장 렌더링했다. 003·004 라벨, 005 라벨과 여는
고리, 006 뚜껑이 정상 방향임을 직접 확인했다. 렌더는 각 asset을 바닥 위 2 mm에
kinematic으로 두고 640×640 RGB camera 8개로 촬영했다. 이 검사는 실제 라벨과
형상을 통한 상하 확인이며 무동작 물리 안정성 검사와 구분한다.
이미지와 사용한 quaternion·높이 manifest는 ignored
`outputs/contact_rearrangement/ycb_orientation/{before,corrected}/`에 있다.

**교정 후 통과한 물리 검증:** 같은 Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090,
기본 seed 0에서 아래 실행이 exit code 0으로 통과했다. 각 환경에 6개 active,
총 8초, 4초마다 전체 reset을 수행했고 마지막 부분 reset도 검사했다. 로그의
active 장면들에 YCB 4종이 모두 포함됐다. 초기 기하·PhysX 직접 pose·quaternion·
velocity·pool flag·COM 주차 정지·관절 목표·제한된 낙하/이동/기울기 검사를 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 8 --min_objects 6 --max_objects 6 --headless --device cuda:0 \
  --max_steps 240 --reset_interval 120 --validate
```

로그: `outputs/contact_rearrangement/ycb_upright_reference_validation.log`.
기존 sampling unit test 14개도 다시 통과했다.

**별도 고정 pair 검사의 실패:** `--seed 7 --target_asset ycb_006_mustard_bottle
--blocker_asset ycb_003_cracker_box`를 추가한 8개 환경·6개 active 검사에서는
첫 step의 검사를 통과했지만 120 step (4초) 시점의 drift 검사에 실패했다.
Env 3의 Blocker `ycb_003_cracker_box_copy_06`가
`Δx=-0.00868 m`, `Δy=+0.05434 m`, `Δz=-0.00515 m` 움직여 XY 5 cm 기준을
넘었다. Runtime error에 물체 이름·초기/현재 위치·속도를 포함하도록 진단을
보강했다. 해당 결과를 모든 초기 장면이 정적으로 안정하다는 근거로 사용하지 않는다.
로그: `ycb_upright_validation.log`, `ycb_upright_diagnostic.log`.

같은 scene seed `3000016`으로 1개 환경을 재현하고 Blocker의 접촉을 매
physics substep (1/120초) 기록했다. 1초 시점의 기울기는 6.79°, 2초 이후에는
29.91°이며 `Δx=-0.00863 m`, `Δy=+0.05409 m`에서 멈췄다. 처음에는 선반과만
접촉했고, step 175 (1.458초)부터 옆의 `ycb_004_sugar_box_copy_04`와 접촉했다.
480개 substep 동안 Blocker–robot 접촉은 관측되지 않았다. 따라서 이 실패를
로봇이 밀어서 생긴 drift로 설명하지 않는다. 원본 convexHull collider와 바닥의
실제 지지 형상이 기울기에 미치는 영향은 별도로 확인해야 한다. Mass는
0.411 kg이고 local COM 위치는 root와 같았다. 접촉 reporter는 임시 진단 환경의
spawn 시점에만 활성화했으며 본 환경의 asset mass/collision은 바꾸지 않았다.
진단 로그·JSON은 `outputs/contact_rearrangement/ycb_upright_contact_diagnostic.*`다.

**배치 기준:** [SPAWN_RULES.md](SPAWN_RULES.md)에 실제 좌표·role/count 선지정·
회전/pivot·pair gap·고정 A₀·차단 비율의 분모·swept witness·주변 물체 이웃 제안·
cluster 밀집도·retry/seed·inactive pool을 정리했다. 당시 baseline은 주변 물체가
초기 A₀와 한쪽 witness 경로를 비우므로 연쇄 재배치가 필수인 clutter를 생성하지
않는다. 10 cm hand width와 모든 threshold는 구현 기본값이며 손의 실제 형상과
제어·동역학·학습 성공의 보장은 아니다.
실제 측정 geometry와 기본 seed 0·6개 active의 샘플을 이용한 평면 배치도는
`docs/assets/spawn_rules_seed_0.svg`다. 통과한 기본 실행의 env 0과 같은
Target·Blocker·active 종류, Y span 32.3 cm, footprint fill 36.2%를 재현했다.

### 2026-10-09 선반 전체 분포·10종 허용·Target 유일성·회전 경로 확장

이 절은 당시 구현의 이력이다. **4–10종을 active instance 4–10개로 잘못 해석한
설정**은 뒤의 종류/instance 분리 교정으로 대체했다. 당시 all-types의 정확히
10개 검증을 현재 density 산정 개수의 검증으로 읽지 않는다.

**사용자 교정:** 4–6개 제한을 확장해 10종 전체 사용을 허용한다. Target 종류는
한 장면에 하나이며 중복하지 않는다. 앞에 Blocker가 들어갈 수 있으면 Target을
앞쪽에도 놓는다. 좁은 cluster 대신 선반 전체에 분포시키되 해소 공간을 남긴다.
Lateral push 외에 pivoting, 회전 후 밀기와 복합 동작을 고려한다.
초기 vision은 선반 앞 +X camera에서 Target의 40% 이상이 보여야 한다.
사용자는 분모를 **같은 pose의 가림 없는 전체 silhouette 픽셀 면적**으로 선택했다.

**구현:** 기본 4–10개·100-body pool, 10종 각각 하나씩 사용하는 all-types 옵션,
Target 종류 전체의 중복 배제, 전체 pair-feasible X와 넓은 Blocker Y 제안,
coverage/국소 근접 판정, 실제 고정점 pivot 원호와 회전/병진 다단계 witness를
추가했다. 기본 X coverage 55%·Y coverage 65%, 4 cm 이웃 비율 50%는 이번
요구를 구현하기 위한 수치이며 사용자가 정한 최종 난이도 명세는 아니다.
기존 38 cm cluster 최대 폭·tight rectangle fill 35% 하한을 제거했다.
당시 YCB upright 회전·authored mass/collider·active native gravity를 유지했다.

`InitialVisionCfg`와 reset metadata에 40% 요구와 **미검증 상태**를 기록했다.
실제 camera·segmentation·reference와 settle 이후 acceptance/rejection은
구현 전이다. 이 후속 단계에서 역할/개수를 유지하고 승인한 초기 pose와 A₀를
일치시켜야 한다. 현재 geometric reset을 camera gate의 통과로 해석하지 않는다.

**기하와 unit 검사:** unit test **20개**가 통과했다. Target 종류 유일성·10종
전체 사용·앞쪽 Target·역할/개수 retry 유지·true polygon 거리·중간 pivot
충돌·회전 구간 내부 corner의 보수적 envelope 포함 등을 검사한다.
기존 실제 측정 USD bounds로 일반 seed 0–99·all-types seed 0–29를 확인했고
모두 생성됐다. 일반 모드의 active 개수는 4–10 모두 나왔고, 경로는 병진 15개·
회전/병진 36개·pivot 49개였다. All-types는 병진 6개·회전/병진 8개·pivot 16개였다.
이 경로 분포는 이 seed 집합의 결과이며 균등 분포나 실제 로봇 성공률을 뜻하지 않는다.

**GPU pool/reset 검사 — 통과:** Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090에서
아래 두 실행이 exit code 0으로 통과했다. 각각 4개 환경·24 env step·0.8초다.
매 step reset하므로 각 layout은 1/30초만 진행한다. 이는 100-body pool의 상태
쓰기·gravity/주차·직접 PhysX readback·전체/부분 reset과 기하 metadata 검사다.
장시간 settle 안정성을 검증한 것으로 기록하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --all_asset_types --headless --device cuda:0 \
  --max_steps 24 --reset_interval 1 --validate

./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 \
  --max_steps 24 --reset_interval 1 --validate
```

로그는 `outputs/contact_rearrangement/full_shelf_pool_reset_validation.log`와
`full_shelf_mixed_reset_validation.log`다. 각 로그에서 초기+24회 reset의 100개
장면을 기술했고, 마지막 부분 reset도 통과했다. 일반 모드에서는 active 4–10,
Target 종류 1회 등장, 다른 종류 중복을 확인했다. All-types 모드에서는 매번
10종 각각 하나씩 사용했고, 경로 종류는 병진 15·회전/병진 35·pivot 50개였다.

**별도 settle 검사 — 실패 유지:** 같은 기본 seed 0·4개 환경·all-types에서
`--max_steps 240 --reset_interval 120 --validate`를 실행했다. 초기 규칙과 첫 step
검사는 통과했지만 120 step (4초) 시점에 env 0의 003이 XY 약 11.2 cm,
env 3의 003이 약 13.8 cm 움직여 실패했다. 로그는
`outputs/contact_rearrangement/full_shelf_all_types_validation.log`다.
당시 버전에는 003의 자연 정착 문제가 남았다. 그 실행에서는 collider/mass를
바꾸거나 검사의 5 cm·15° 기준을 완화해 통과시키지 않았다. 이후 사용자의
흔들림 수정 요청에 따라 collider를 바꾸는 현 구현과 구분한다.

**배치도:** `docs/assets/full_shelf_clearance_examples.svg`는 실제 측정 geometry의
10종 전체 seed 0·2·10으로 병진·pivot 후 이동·회전 후 이동을 보여준다.
이는 생성 기하와 경로 예약의 설명이며 실제 camera visibility나 조작 실행 결과는 아니다.

### 2026-10-09 실행 스크립트의 5초 자동 reset

사용자 요청으로 `spawn_scene.py`의 기본 동작을 시뮬레이션 시간 5초마다 전체
환경을 reset하도록 변경했다. `ceil(5 / env.step_dt)`로 계산하며 현재 150 step이다.
기존 `--reset_interval N`의 step 단위 지정과 0의 비활성 동작을 유지한다.

Headless CUDA 환경 1개를 `--max_steps 151`로 실행해 exit code 0으로 종료했다.
150 step·5.00초에서 reset하고 seed 0의 4개 장면이 seed 97409의 6개 장면으로
바뀌는 것을 확인했다. 로그는 `outputs/contact_rearrangement/five_second_reset_validation.log`다.
이 실행은 자동 reset 주기 확인이며 `--validate`를 사용한 settle 안정성 검사는 아니다.
스크립트 문법 검사와 `git diff --check`도 통과했다.

### 2026-10-09 YCB 박스의 초기 낙하·흔들림 수정 — 제한된 GPU 정착·reset 검증 통과

**사용자 요청:** 스폰 직후 YCB 박스가 낙하하며 흔들리는 현상을 해결한다.
먼저 모든 종류의 2 mm 낙하 여유를 제거해 `root_z=shelf_top-lower_z`로 바꿨다.
높이를 바꾸는 것만으로 원본 scanned mesh collider의 접촉면이 평평해지는 것은
아니므로 collision 형상을 별도로 비교했다.

| 중간 시험 | 확인한 결과와 상태 |
|---|---|
| 여유 0·원본 collider, 4개 환경·10종 전체·5초 | 003 최대 기울기 66.96°·XY 이동 11.3 cm, 004 최대 3.80°·XY 5.79 mm. 높이 수정만으로 안정화하지 못했다. |
| 원본 mesh의 `boundingCube` fitted approximation | 0.83°·XY 1.54 mm의 움직임이 남아 최종 방식으로 사용하지 않았다. |
| 명시적 body-local AABB Cube proxy의 초기 진단 | 4개 환경·10종 전체·5초 동안 박스 8개를 120 Hz로 기록. 최대 XY 이동 0.007894 mm, 계산 정밀도 내 거의 직립. |

**구현:** 당시 `object_spawner.py`(현재 `assets.py`에 통합)의 `FlatBoxUsdCfg`는 003·004의 visible scanned
mesh를 보존하고 원본 collider를 disabled로 바꾸며, 기존 single rigid body 아래에
같은 unscaled body-local AABB 크기·offset의 invisible `UsdGeom.Cube`를 추가한다.
Cube size 1·extent ±0.5를 명시하고 body·ancestor scale을 상속하며 spawn/clone
시점에만 override한다. 원본
Nucleus USD는 무수정이고 body의 authored mass는 유지한다. 원본 collider의
authored `physxCollision:*` contact 속성을 가져오고 실제 `UsdPhysics.MaterialAPI`가
있는 physics material binding만 proxy에 적용한다. Visual material을 physics
material로 오인하지 않도록 guard한다. Active native gravity·simulation과 inactive
COM countergravity도 유지한다.
003·004의 접촉 형상을 평평한 box로 근사하는 변경이며 원본 mesh collision과
동일하다고 표현하지 않는다. 005·006과 기존 6종은 원본 collider를 사용한다.

**Unit 검사:** 새 높이 0에 맞춰 fixture를 수정한 20개 sampling test가 통과했다.
이는 placement·역할·경로 명세 검사이며 새 collision의 물리 정착 성공을 증명하지
않는다. 실행 시간은 2.934초였고 `py_compile`·`git diff --check`도 통과했다.

**초기 정량 진단:** 4개 GPU 환경·10종 전체 장면을 5초 유지하면서 박스 8개를
매 physics substep (120 Hz) 기록했다. 최대 XY 이동은 **0.007894 mm**였다.
Float32 `acos` 기울기 값은 계산 정밀도 내 0이었으므로 **거의 직립**으로
기록하고 수학적으로 정확한 0° 보장으로 취급하지 않는다. 003 mass 0.411 kg,
004 mass 0.514 kg를 유지했고 진단한 박스 8개 각각에서 rigid body 1개·enabled Cube
collider 1개, 원본 mesh collision disabled를 확인했다. 로그·JSON은
`outputs/contact_rearrangement/box_supported_proxy_diagnostic.{log,json}`다.

**최종 공식 실행:** Cube extent·physics material guard를 보강한 버전에서
아래 명령이 exit code 0으로 통과했다. Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090,
기본 seed 0, 환경당 100-body pool에서 10종 각각 하나씩 active다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

총 300 environment step·10.00초다. Reset interval을 생략해 기본 150 step을
사용했고 150·300 step에서 전체 reset, 끝에서 부분 reset을 확인했다.
최초 4개 환경의 layout과 첫 reset 뒤 4개 layout, **총 8개 layout을 각각
5초** 진행했다. 300 step에서 새로 생성한 장면에는 물리 step을 진행하지
않았으므로 추가 정착 검증으로 세지 않는다. 5 cm drift·15° tilt 기준은
완화하지 않았다. 전체/부분 reset과 bounded settle 검사가 통과한 범위이며,
다른 seed·모든 배치·실제 robot push·camera visibility의 검증으로 확대하지 않는다.
로그는 `outputs/contact_rearrangement/box_stable_reset_validation.log`다.

### 2026-10-09 종류·instance 분리와 전체 선반 밀도 — 이전 면적 우선 구성의 검사

아래 수치·실행·배치도는 큰 footprint부터 복사본을 고르던 **이전 구성**의
기록이다. 현재 균형 구성의 instance 분포·생성 성능·정착 결과로 사용하지 않는다.

사용자가 **4–10은 종류 선정 기준**이라고 교정했고 선반 전체의 density를
충족하도록 요청했다. 이전 active 4–10개·all-types 정확히 10개 해석을 폐기했다.
당시는 종류 4–10개를 고른 뒤 선택한 non-target 종류의 큰 footprint 복사본부터 더해 요구 면적을
충족한다. Target 종류는 한 번만 사용하며 instance 최소 4·cap 40,
10종 × 10-copy pool을 유지한다. 역할·종류 집합·전체 instance ID·N은 배치
재시도 전에 고정한다. 구성에 필요한 면적 하한은
`max(min_shelf_occupancy, min_region_occupancy) × usable_area`다.
Global 면적과 X 2 × Y 3의 6개 구역 면적 하한을 모두 실제 배치에서 검사한다.

**기하 benchmark와 수치 선택:** 실제 USD 측정 bounds와 현재 clearance 조건으로
global 25–45%를 비교했다. 아래는 일반·all-types 두 모드의 고유 sample 수다.

| Global 요구 | Seed·표본 | 생성 통과 |
|---|---|---|
| 25% | 각 모드 seed 0–4, 총 10개 | 10/10 |
| 30% | 각 모드 seed 0–4, 총 10개 | 10/10 |
| 35% | 각 모드 seed 0–19, 총 40개 | 40/40 |
| 40% | 각 모드 seed 0–4·10–19, 총 30개 | 30/30 |
| 45% | 각 모드 seed 0–4, 총 10개 | 3/10 — 일반 1/5, all-types 2/5 |

35%·40%의 공통 seed 집합에서 평균 생성 시간은 각각 약 0.206초와
0.99–1.00초였다. 45% 실패는 일반 모드의 capacity 1개·packing 3개,
all-types의 packing 3개였다. 생성 성공률과 약 5배의 시간 차이를 고려해
**global 35%·각 구역 10%·4 cm 이웃 비율 75%**를 초기 구현 기본값으로 선택했다.
사용자가 직접 선택한 최종 연구 난이도 수치는 아니다.

35%의 일반 모드는 4–10종·10–24 instance, all-types는 10종·14 instance였다.
실제 global 점유율은 각각 35.39–38.68%·36.32–37.36%였고, 장면별 최소
구역 점유율은 전체 sample에서 12.98–35.22%였다. 자료는
`outputs/contact_rearrangement/density_calibration.json`의 초기 비교·독립
확인·확장 기본값 확인에 저장했다. 같은 seed의 중복 검사를 표본 수에 합산하지
않는다. 이는 geometry 생성 검사이며 물리 정착·camera·robot push 검사는 아니다.

**당시 GPU 검사:** 다음 두 실행이 모두 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

각 실행은 기본 35%·seed 0·4개 GPU 환경·300 step·10.00초다. 기본 150-step
간격으로 150·300 step의 전체 reset과 끝부분의 partial reset을 확인했다.
각 모드에서 초기 4개·첫 reset 뒤 4개, **8개 layout을 각각 5초** 진행했고
5 cm drift·15° tilt 기준을 완화하지 않았다. 마지막 reset에서 생성한 장면은
추가 물리 step이 없어 정착 표본으로 세지 않는다. 로그는
`outputs/contact_rearrangement/dense_shelf_default_validation.log`와
`outputs/contact_rearrangement/dense_shelf_all_types_validation.log`다.
미검사 seed·실제 로봇 밀기·camera visibility의 성공으로 확대하지 않는다.

**당시 단위 검사:** 21개가 18.219초에 통과했고 `py_compile`·`git diff --check`도
통과했다. Region 요구가 global보다 큰 경우의 composition 면적 하한 검사도
포함하며 기본 35%/10% 동작은 같다.

**당시 배치도:** 실제 측정 geometry의 4종·10개, 4종·24개, 10종·14개를
비교했다. 해당 파일은 이후 아래의 현재 균형 구성 예시로 교체됐으며 이전
개수의 그림을 현재 배치도로 연결하지 않는다.

### 2026-10-09 Target·Blocker의 현재 OBB 표시

사용자 요청에 따라 `visualization.py`의 `RoleObbVisualizer`를 장면 확인
스크립트에 연결했다. Target은 초록색, Blocker는 빨간색으로 각 12개 edge를
debug draw한다. 현재 body link pose와 upright frame bounds를 사용해 YCB 방향,
중심과 다른 root 위치, 합성 scale을 반영하며 전체·부분 reset의 역할 ID도 갱신한다.
USD prim·API·collider·material·physics state를 추가하거나 변경하지 않는다.
GUI·livestream에서는 기본 표시하고 `--disable_role_obb`로 끈다. 일반 headless는
생략하고 `--headless --validate`는 viewport 없이 같은 debug draw 경로를 검사한다.

**제한된 실행 검사:** YCB 003 Target·006 Blocker를 지정한 2개 CUDA 환경에서
151 step·5.03초의 headless `--validate`가 exit code 0으로 통과했다. 총 48개
line과 draw 전후 USD prim/API 집합·PhysX pose/velocity·native flag 유지,
150-step 자동 reset과 마지막 전체·부분 reset 갱신을 확인했다. 별도의 1개 GUI
환경·45 step·1.5초 `--validate`도 exit code 0이었다. 실제 RTX viewport 렌더
`outputs/contact_rearrangement/role_obb_preview.png`에서 초록 cracker box·빨간
mustard bottle의 OBB가 물체를 감싸는 것을 확인했다. 이 검사는 역할 표시와
무변경 동작의 범위이며 task camera의 Target 40% 가시성이나 로봇 조작 검사가 아니다.

**독립 수치 검사:** 별도의 CPU 행렬 기준과 비교해 legacy·YCB upright, yaw/tilt,
scaled/noncentered bounds, 두 환경의 world offset, 이동·기울어진 pose와 부분
역할 교체를 확인했다. 48개 edge의 연결·초록/빨강 색상·3 px 두께, redraw/cleanup과
body pose의 정확한 유지도 통과했다. 검사 스크립트는
`/tmp/contact_rearrangement_obb_numerical_review.py`이며 `py_compile`과
`git diff --check`도 통과했다.

### 2026-10-09 non-target 복사본의 종류별 개수 균형

사용자가 한 asset의 복사본이 공간을 지나치게 채우는 문제를 지적했다.
이전 큰 footprint 우선 복사본 선택은 채울 면적에 필요한 instance 수를 줄이는
대신 큰 종류에 중복을 집중시켰다. 현재는 추가 가능한 non-target 종류 중
**active 개수가 가장 적은 종류**를 고르고 동률만 무작위로 정한다. 지정
Blocker도 해당 종류의 첫 instance로 세며 모든 종류의 개수가 같아지면
다음 회차를 시작한다. Target 종류는 항상 하나다.

현재 같은 pool capacity에서는 non-target 종류별 개수 차이가 최대 1이다.
종류별 새 고정 최대값은 두지 않고 기존 종류당 10-copy·전체 cap 40을 유지한다.
작은 종류 위주 집합은 같은 global 점유율에 더 많은 instance가 필요할 수 있다.
종류 4–10개, 역할·종류 집합·전체 instance ID·N의 retry 전 고정,
global 35%·6개 구역 각 10%·이웃 75%와 clearance 조건은 유지한다.
공간 배치는 packing을 위해 큰 footprint부터 진행하므로 **종류 선택과 위치
배치의 순서는 별개**다. 이전 calibration·instance 분포·배치도를 현재 결과로
사용하지 않으며 다음은 최종 source에서 별도로 확인한 결과다.

**현재 기하 검사:** 실제 USD bounds로 일반·all-types 각각 seed 0–19,
**40/40개**가 통과했다. 모든 장면에서 non-target 종류별 개수 차이가 최대 1이었다.
일반 모드는 12–25 instance·global 35.05–38.48%·검사 전체의 최소 구역 22.86%,
all-types는 15–18 instance·global 35.14–38.71%·최소 구역 24.75%였다.
Source SHA를 포함한 자료는
`outputs/contact_rearrangement/balanced_composition_benchmark.json`에 있다.
이는 geometry 생성 검사이며 camera·robot push 검사와 구분한다.

**현재 GPU 검사:** 다음 일반 모드 실행이 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 --max_steps 151 --validate
```

4개 CUDA 환경·151 step·5.03초에서 150-step 자동 reset과 마지막 전체·부분
reset, 기존 정착 guard를 확인했다. 총 96개 역할 OBB line과 draw 전후 USD·
PhysX 상태 유지도 검사했다. Reset sample 중 10종 장면이 있었지만 현재 구성의
별도 `--all_asset_types` GPU 실행을 완료한 것으로 해석하지 않는다. 로그는
`outputs/contact_rearrangement/balanced_composition_gpu_validation.log`다.
미검사 seed·실제 로봇 밀기·camera visibility의 성공으로 확대하지 않는다.
당시 unit test **23개**가 24.384초에 통과했으며 균형 선정·복사본 재고 소진
회귀 검사를 포함한다.

**현재 배치도:** [균형 구성 예시](assets/dense_shelf_examples.svg)는 일반 seed 0의
4종·12개, 일반 seed 2의 4종·25개, all-types seed 1의 10종·17개를 보여준다.
Target은 하나이며 non-target 종류별 개수는 각각 3/4/4, 8/8/8,
1/1/2/2/2/2/2/2/2다. Console의 `[SPAWN]`에도 반복 목록 대신 종류별
`asset_counts`를 출력해 편중을 확인할 수 있다.

### 2026-10-09 책임별 6개 모듈 통합 — 동작 유지 검사 통과

과하게 세분화된 파일을 `catalog.py`·`assets.py`·`env_cfg.py`·`spawn.py`·
`sampling.py`·`visualization.py`로 통합하고 빈 `__init__.py`를 유지했다.
이전 모듈·`mdp/`는 wrapper 없이 제거했다. 실행 진입점과 CLI, 스폰 규칙·
기본값은 유지했으며 새 시뮬레이션·camera 기능을 추가하지 않았다.

모든 모듈의 문법 검사가 통과했고 `catalog`·`sampling`을 simulator 없이
import해 `isaaclab`·`isaacsim`·`pxr`·`torch`가 로드되지 않는 것을 확인했다.
CLI `--help`는 exit code 0이었다. Unit test **23개**가 24.552초에 통과했다.
6개 모듈의 의존관계에는 cycle이 없고 상대 Markdown 링크도 유효했다.

일반 모드의 `--num_envs 4 --headless --device cuda:0 --max_steps 151 --validate`가
exit code 0으로 통과했다. 4개 CUDA 환경·5.03초에서 150-step 자동 reset과
마지막 전체/부분 reset, 역할 OBB 96개 line·draw 전후 USD/PhysX 상태 유지를
검사했다. 물리 guard를 완화하지 않았다. 초기·150-step reset 뒤의 **8개 장면
설명**은 이전 `balanced_composition_gpu_validation.log` 실행과 정확히 같았다.
로봇·물체 18개 정의, USD 측정/reset 정의와 scene/env 설정의 AST도 통합 전과
일치했다. 이 결과를 새 camera 가시성·로봇 조작의 검증으로 확대하지 않는다.
로그는 `outputs/contact_rearrangement/refactor_cli_help.log`,
`refactor_unit_tests.log`, `refactor_gpu_validation.log`에 남겼다.

### 2026-10-09 Blocker 앞 hand 시작과 elbow-down 금지 — 이전 tool 방향의 검증

사용자 요청으로 기존 고정 robot 시작 pose를 episode별 Blocker 앞 영역의
collision-checked elbow-up 시작으로 확장했다. 6개 모듈 구조를 유지하고
`assets.py`가 USD FK/IK·collision geometry, `sampling.py`가 순수 영역/elbow/OBB
계산, `spawn.py`가 독립 seed·후보 승인·state/PD target·metadata를 담당한다.
물체 선정·배치 RNG를 유지하고 추가 physics step 없이 선택한 환경에만 적용한다.
거리·높이·여유·80번 cap은 현재 구현 기본값이며 후속 제어·학습 분포에 맞춰
조정할 수 있다.

이 절의 수치와 렌더는 tool 전방이 world −Z였던 당시 orientation의 검사다.
2026-10-10의 선반 방향 교정 결과로 사용하지 않는다.

초기 구현은 4개 GPU 환경·151 step·5.03초의 검사에서 150-step·5초 자동 reset과
마지막 전체/부분 reset, 선택하지 않은 환경의 q/velocity/PD target과 robot metadata
유지를 통과했다. Robot collider 39개·shelf collider 17개와 모델 FK 대 실제 link
position 오차 5.13e−7 m도 확인했다. 이후 solver에 같은 TCP의 다른 IK branch를
scene 조건까지 검사하는 callback을 넣고 물체·robot 후보 모두 성공한 뒤
reset/state를 적용하도록 보강했다. 초기 검사와 아래의 최종 변경본 검사를 구분한다.

후속 stress에서 **4개 환경 × 6회 reset의 24개 robot 시작 pose**는 실제 PhysX
조건을 통과했다. `min_elbow_height=2 m`·1회 시도로 실패를 강제해 joint q·object
pose·PD target·reset count·역할·robot metadata 유지도 확인했다. 다만 physics 없이
6번 reset한 뒤 150 step에서 parked body angular velocity가 32–71 rad/s로 증가했다.
해당 경로의 상위 `RigidObjectCollection.write_object_link_pose_to_sim`이 별도 COM
pose cache를 무효화하지 않아 같은 timestamp의 이전 COM을 읽는 것을 확인했다.
Pool의 writer에서 그 cache를 무효화하도록 수정했다. 이후 UR arm의 가까운
joint housing OBB가 충돌로 잡히는 false positive도 수정했다. 직렬 7-link arm의
21쌍에서 두 arm joint 이내의 11쌍을 제외하고 거리 3 이상인 10쌍만 self 검사한다.
전체 39개 robot collider 대 물체·17개 shelf collider·ground의 **15 mm 기준은
완화하지 않았다.**

**최종 변경본 검사:** 4개 GPU 환경 × 6회 reset의 **24개 초기 pose**가 실제
PhysX 조건을 통과했고 8개는 10종 전체 장면이었다. 같은 timestamp에서 physics
없이 6번 reset한 뒤 **150 step·5초**의 parked 안정성, 전체/부분 reset도 통과했다.
`min_elbow_height=2 m`·1회 시도로 실패를 강제했을 때도 physical joint q·object
pose·PD target·reset count·역할·TCP metadata가 유지됐다. 최종 FK position 오차는
**2.8458e−7 m**, elbow chord 높이는 **11.49–16.07 cm**, 후보 시도는 **1–3회**였다.
현재 10종의 enabled collider mesh points·primitive OBB가 측정한 물체 bounds 안에
포함되는 것도 확인해 초기 scene obstacle OBB의 보수적인 형상을 대조했다.
최종 unit test **29개**(기존 23개 + 영역/SAT/elbow 6개)가 **24.673초**에 통과했다.
로그는 `outputs/contact_rearrangement/robot_randomization_unit_tests.log`다. Stress 자료는
`outputs/contact_rearrangement/robot_randomization_benchmark.{json,log}`다.

4환경 reset batch의 wall time은 물체 배치를 포함해 **1.420–7.601초**였다.
현재 CPU 후보 계산 비용은 환경 수를 확대할 때 batch/vectorization 검토가 필요한
한계다.

**GUI 검사:** 45 step·1.5초의 실제 GUI 실행과 전체/부분 reset, COM cache를
미리 읽은 뒤 reset하는 검사, OBB draw 전후 USD/PhysX 불변 조건도 통과했다.
실제 렌더에서 손이 Blocker 앞 공간에 있고 elbow가 높게 위치한 것을 확인했다.
당시 palm normal은 선반을 향했지만 tool 전방은 아래를 향했다. 자료는
`outputs/contact_rearrangement/robot_start_preview.{png,log}`다.
Elbow 높이 계산은 `assets.py`의 FK와 `sampling.elbow_above_chord`를 공유한다.

**최종 자동 reset 검사:** 현재 변경본의 일반 모드
`--num_envs 4 --headless --device cuda:0 --max_steps 151 --validate`도 exit code 0으로
통과했다. 151 step·5.03초에서 150 step·5.00초의 자동 reset, 부분 reset의
미선택 환경 보존, COM cache를 미리 읽고 pose write 뒤 새로 읽는 검사,
역할 OBB draw 전후 USD/PhysX 상태 유지를 확인했다. 로그는
`outputs/contact_rearrangement/robot_randomization_gpu.log`다.
위 통과를 손 전체 self collision·재배치 제어 성공·task camera 가시성의 보장으로
확대하지 않는다.

### 2026-10-10 Tool 전방을 선반 방향으로 교정 — hand roll 추가 전 검증

사용자가 시작 자세에서 tool이 바닥을 향한다고 지적하고 정면인 선반 방향으로
수정하도록 요청했다. 이전 목표 회전은 palm normal인 TCP +Z를 world −X로
맞춰 실제 tool 전방인 TCP +Y·wrist-3 +Z를 world −Z로 향하게 했다.
`sampling.ROBOT_START_TCP_ROTATION`을 TCP +X→world +Y, +Y→world −X,
+Z→world +Z로 교정했다. 손 mount·시작 영역·elbow-up·15 mm 외부 충돌 여유와
물체 배치 규칙은 유지한다. Reset 검증에 실제 PhysX TCP 전체 회전과 wrist-3
tool 축 검사를 추가했다.

**선반 방향 GPU 검사:** 일반 모드 4개 GPU 환경·151 step·5.03초의
`--validate` 실행이 exit code 0으로 통과했다. 초기 4개·5초 자동 reset 4개·
부분 reset 1개·마지막 전체 reset 4개, **총 13개 시작 pose**에서 실제 PhysX
TCP 회전 오차 ≤0.01001 rad와 wrist-3 tool 전방의 world −X 축 검사를 통과했다.
Elbow-up·초기 scene 충돌 여유·부분 reset의 미선택 환경 유지·COM cache 갱신·
OBB 표시 전후 물리 상태 불변도 확인했다. 로그는
`outputs/contact_rearrangement/tool_forward_gpu_validation.log`다.
이번 수정에서 순수 helper는 바꾸지 않았으므로 앞의 29개 unit test를 새 결과로
기록하지 않는다.

**선반 방향 GUI 검사:** 1개 환경·45 step·1.50초의 실제 GUI 실행과 전체/부분
reset, OBB 표시 전후 물리 상태 불변 검사가 exit code 0으로 통과했다. 렌더에서
tool·손가락이 선반 정면을 향하는 것을 확인했다. 45-step GUI 실행 중 캡처
시점의 PD 유지 pose에서
wrist-3 tool 전방은 world `(-0.99947666,-0.0005177,-0.03234399)`이며 정확한
world −X와의 각도는 **0.03235 rad, 약 1.85°**였다. 초기 reset의 0.01 rad 조건과
구분해 캡처 시점의 유지 pose는 별도 0.05 rad 검사를 통과했으며, 물리 진행 내내 정확한
수평 회전이 유지된다는 의미는 아니다. 자료는
`outputs/contact_rearrangement/tool_forward_preview.{png,log}`다. 이전
`robot_start_preview` 렌더는 과거 orientation의 자료로 남긴다.

### 2026-10-10 Tool +Z축 기준 hand −90° roll — 원통 연결부 추가 전 검증

사용자가 tool 기준 Z축으로 hand를 −90° 회전하도록 요청했다.
`Ur5eHandSpawnerCfg.mount_tool_roll_deg=-90.0`을 추가하고 기존 vendor adapter
`mount_rotation_deg=(0,-90,0)` deg는 유지한다. 실제 조립에서는 `tool0` attachment
frame과 physical tool body `wrist_3_link`의 축이 서로 달라, 두 frame 사이의
transform을 반영한 뒤 physical tool +Z를 기준으로 hand mount를 회전한다.
Hand의 상대 회전뿐 아니라 상대 translation도 함께 회전하며 UR wrist 관절의
기본값만 바꾸는 방식으로 처리하지 않는다.

TCP 목표는 X→world +Z, Y→world −X, Z→world −Y로 바꾼다. 손가락/tool 전방
−X, 시작 영역, elbow-up, 15 mm 외부 충돌 여유와 물체 생성 규칙은 유지한다.
앞 절의 `tool_forward_*` 검사·렌더는 hand roll 추가 전의 결과로 남긴다.

**Hand roll GPU 검사:** 일반 모드 **4개 GPU 환경·151 step·5.03초**의
`--validate`가 exit code 0으로 통과했다. 초기 4개·5초 자동 reset 4개·부분 reset
1개·마지막 전체 reset 4개, **총 13개 시작 pose**에서 실제 PhysX TCP의 새
회전과 wrist-3 tool 전방 −X, elbow-up·15 mm scene 여유를 확인했다. 미선택
환경 유지·COM cache·pool flag 유지·역할 OBB 표시의 물리 상태 불변도 통과했다.
로그는 `outputs/contact_rearrangement/hand_tool_roll_gpu_validation.log`다.

**Hand roll 조립·GUI 검사:** 첫 reset의 실제 PhysX pose에서 physical tool body
기준 새 hand 상대 회전이 **`R_new = Rz(-90°) @ R_old`**임을 확인했다. 회전
오차는 **3.23141e−7 rad**였고 TCP 상대 translation에도 같은 회전이 적용된
것을 1e−5 m 허용 오차로 확인했다. 기존 tool 정면 목표에 대한 전체 world
회전 오차는 **1.15236e−5 rad**였다. 이로써 hand mount의 roll과 tool 전방
유지를 실제 조립된 body pose로 대조했다.
1개 GUI 환경·45 step·1.50초와 마지막 전체/부분 reset·역할 OBB 불변 검사도
exit code 0으로 통과했다. 렌더는 실행 중 약 20번째 step의 캡처이며 앞의
상대 transform 수치는 첫 update의 초기 pose에서 측정했다. 자료는
`outputs/contact_rearrangement/hand_tool_roll_preview.{png,log}`다. 순수 helper를
바꾸지 않았고 이번 수정에서 새 unit test 실행 결과를 추가하지 않는다.

### 2026-10-10 UR5e 말단과 hand 사이 5 cm 원통 연결부 — 중앙 F/T 추가 전 검증

사용자가 실제 센서·연결부를 고려해 UR5e 말단과 hand 사이에 길이 5 cm의
원통을 요청했다. `Ur5eHandSpawnerCfg.sensor_mount_length=0.05 m`,
`sensor_mount_radius=0.025 m`로 설정한다. 직경 50 mm는 실제 hand
`base_link.STL`의 mount X/Y bounds ±25 mm를 기준으로 선택한 구현 기본값이다.
Wrist-3 아래 visible collision Cylinder `SensorMount`를 추가하며 physical tool
+Z 방향 z=0–50 mm, 중심 z=25 mm에 둔다. World rigid rotation을 반영해 원본
asset scale과 무관하게 실제 치수를 적용한다.

Hand fixed-joint anchor를 같은 tool +Z 방향으로 50 mm 옮기고 −90° hand roll과
TCP 목표 회전은 유지한다. 원통은 wrist-3 compound collider이므로 별도 rigid
body나 DOF를 추가하지 않는다. FK body 39개·DOF 18개는 유지하며 robot collider는
39개에서 **40개**로 늘어난다. 원통도 기존 15 mm scene 여유와 arm self
collision 판정에 포함한다. 원본 USD를 수정하지 않고 기존 wrist mass/inertia를
유지한다. 실제 센서 질량 보정·F/T 센서 출력은 아직 구현되지 않았다.
위 `hand_tool_roll_*` 결과는 원통 추가 전 검증이며 새 조립 검사는 별도로 기록한다.

**연결부 GPU 검사:** 일반 모드 **4개 GPU 환경·151 step·5.03초**의
`--validate`가 exit code 0으로 통과했다. 초기 4개·5초 자동 reset 4개·부분 reset
1개·마지막 전체 reset 4개, **총 13개 시작 pose**에서 실제 PhysX TCP 회전·tool
전방 −X·elbow-up·15 mm scene 여유를 확인했다. Robot collider **40개**·shelf
collider 17개, 실제 link 대 FK position 오차 **5.15e−7 m**였다. 미선택 환경
유지·parking COM cache·pool flag·역할 OBB 표시의 물리 상태 불변도 통과했다.
로그는 `outputs/contact_rearrangement/sensor_mount_gpu_validation.log`다.

**연결부 조립·GUI 검사:** 첫 reset의 실제 PhysX pose에서 wrist-3→hand base
offset은 tool body local `[0,0,0.05] m`, gap은 **0.049999997 m**였다. 원통의
near/far end가 wrist와 hand base origin에 각각 1e−5 m 이내로 일치했고 길이
50 mm·반지름 25 mm를 확인했다. 기존 −90° 상대 hand roll의 오차는
**4.51901e−7 rad**였으며 TCP 상대 offset도 원통 추가 전 값에 `[0,0,0.05] m`를
더한 값과 일치했다. 39 body·18 DOF·40 collider를 유지한 실제 조립을 대조했다.
1개 GUI 환경·45 step·1.50초와 마지막 전체/부분 reset·역할 OBB 상태 불변
검사도 exit code 0으로 통과했다. 실행 중 약 20번째 step의 넓은 시야 렌더와
약 32번째 step의 연결부 근접 렌더에서 원통 공간을 확인했다. 자료는
`outputs/contact_rearrangement/sensor_mount_preview.{png,log}`와
`sensor_mount_detail.png`다. 위 상대 transform 수치는 첫 reset에서 측정했으며
이번 수정에서 새 unit test 실행 결과를 추가하지 않는다.

### 2026-10-10 원통 중앙 F/T와 initial vision acceptance

사용자가 원통 중앙의 articulation fixed-joint F/T와 앞쪽 초기 vision camera를
요청했다. 원통을 두 25 mm rigid half로 분리하고 중앙 fixed joint를 통해 hand
load를 전달한다. Nominal 질량 합 0.20 kg와 원통 inertia를 명시하며 원본 wrist
질량은 유지한다. Body·collider는 각각 41개, DOF는 18개다. Sensor frame의 raw
반력을 읽고 reset 직후 stale sample은 valid=False로 제외한다. 자동 tare·중력
보정·filtering은 추가하지 않는다.

실제 같은 pose의 비가림 reference로 Target ≥40%·Blocker 전체(2 pixel tolerance)
가시성을 검사한다. 앞쪽 camera 위치·aim·roll을 무작위로 정하고 승인한 RGB-D와
실제 role mask를 초기 snapshot으로 보존한다. Physics step 없이 camera/layout
retry를 진행하며 역할·종류·active ID·개수는 유지한다. `sensors.py` 한 모듈에
render·snapshot·F/T observation 연결을 모으고 script는 camera를 자동 활성화한다.
검증 결과는 이전 원통 geometry 검사와 구분해 기록한다.

Blocker 앞 reserve 통로와 환경별 reference lane 분리를 추가했다. 변경 후
순수 sampling 검사 **32개**가 **27.993초**에 통과했다. 이 결과는 렌더 가시성·
F/T 정확도·GPU reset 검사의 통과가 아니며 실제 센서 실행 결과는 별도로 기록한다.

낮은 camera 후보에서는 실제 robot arm·hand가 역할 물체를 가리는 것을 확인해
기본 camera 높이를 상면 +0.18–0.48 m로 넓혔다. 1개 환경의 진단에서는 같은
composition의 layout을 한 번 재시도한 뒤 높은 camera 후보가 Target **56.370%**·
Blocker **100%**로 2번째 camera 시도에 승인됐다. 이는 한 진단 장면의 결과이며
다중 환경 GPU 검증이나 전체 생성 성공률로 해석하지 않는다.

Reset의 native physics counter 추적에서 첫 camera pose의 SDK Fabric sync가
Kit update를 통해 physics를 진행시키는 경로를 확인했다. 로컬 `InitialCamera`와
camera update guard로 이 경로를 보호한다. 이 수정의 reset/render/time/native
counter 검증은 아래 최종 GPU 실행에서 통과했다.

**최종 센서 GPU 검사:** `outputs/contact_rearrangement/initial_sensors_gpu_validation.log`
실행이 exit code 0으로 통과했다. 4개 환경의 public script **151 env step·
5.03초**, 추가 2 env step과 F/T bench 7 physics step을 검사했다. 전체 reset
6회 중 2회는 부분 reset이며 **18개 승인 시작 pose**였다. Successful capture
call은 **6회**, failed layout은 **2개**, render는 두 warmup call을 포함해
**464회**였다. Capture 중 object/robot native pose·velocity·joint q/v·PD target·
physics time/step은 정확히 유지됐으며 추가 physics step이 없었다. SDK 초기화의
별도 physics step 2회는 env step·reset/capture 검사와 분리해 센다.

18개 시작의 Target fraction은 **47.1253–100%**였다. Blocker는 17개가 정확히
100%, 1개가 **99.97586%**로 정확히 **2 pixel** 손실을 허용한 경우였다. 승인한
camera 후보 시도는 1–30회였다. Native robot은 **41 body·18 DOF·41 collider**다.
Reset 직후 선택된 F/T는 zero·invalid였고 다음 physics step 뒤 valid·finite로
갱신됐다. 미선택 F/T·snapshot cache를 보존했으며 153 env step 동안 승인한
initial snapshot은 고정됐다.

F/T bench는 distal link에 sensor XYZ별 **3 N force**와 **0.10 N·m torque**를
독립 적용해 baseline 및 Newton–Euler의 중력·관성 항과 대조했다. Downstream은
33개 body·질량 합 **0.876663 kg**였다. 최대 vector norm 오차는 force
**0.0206053 N**, torque **0.00132915 N·m**로 기준 0.10 N·0.015 N·m보다 작았다.
중앙 측정점·sensor frame·parent→child 부호·성분 순서를 실제 물리 반력으로
확인했다. 이는 해당 제한된 초기 장면과 bench의 검증이며 모든 seed·실제
재배치 성공·실제 하드웨어 교정·보정 F/T observation의 입증은 아니다.

**센서 GUI·실패 복구 검사:** 1개 GUI 환경·45 env step·1.50초가 exit code 0으로
통과했다. Target `ycb_003_cracker_box`는 **7724/9286 pixel, 83.178979%**, Blocker
`ycb_006_mustard_bottle`은 **3608/3608 pixel, 100%**, camera 후보 3회였다.
실제 원통은 25 mm half 두 개, 중앙점 wrist +25 mm·hand anchor +50 mm와
41 body·18 DOF·41 collider를 확인했다. Capture 실패와 중간 commit 오류를
강제한 이동 상태에서도 물체 COM velocity·robot root pose/velocity·q/qd·
position/velocity target·camera pose·모든 초기 snapshot/역할/seed/count/sample·
F/T raw/valid/tare·clock/step 복구가 통과했다. Authoritative root·DOF state는
정확히 복구됐으며 derived robot link velocity는 kinematic 재계산 전 solver
출력과 최대 **4.47e−5** 차이가 있어 exact 복구로 기록하지 않는다.
자료는 `outputs/contact_rearrangement/initial_sensors_preview.log`,
실제 초기 camera `initial_camera_preview.{png,json}`, GUI overview
`initial_sensors_preview.png`, 연결부 확대 `initial_sensors_detail.png`다.
Camera/F/T·전체/부분 reset의 재사용 검증 진입점은
[check_sensors.py](../src/contact_rearrangement/tests/check_sensors.py)이며
실행 명령은 [README](../src/contact_rearrangement/README.md)에 기록한다.

**Repository 진입점 재현:** `check_sensors.py --num_envs 1`도 exit code 0으로
통과했다. 151+2 env step·bench 7 physics step, reset/capture 각 6회(부분 reset
0회), render 179회였다. 승인 6개의 Target **41.0632–100%**·Blocker 전부 100%,
reset/capture의 physics step 증가 0·snapshot 보존·41 body/18 DOF/41 collider를
확인했다. 6축 bench의 최대 norm 오차는 **0.0122181 N·0.000850297 N·m**였다.
로그는 `outputs/contact_rearrangement/initial_sensors_reusable_check.log`다.
기존 4환경·GUI 결과와 별도 실행이며 `compileall`·`git diff --check`도 통과했다.

### 2026-10-10 Tactile 17채널 실제 접촉·reset 검증

[check_tactile.py](../src/contact_rearrangement/tests/check_tactile.py) `--num_envs 2`
GPU 실행이 exit code 0으로 통과했다. Public 151 env step·5.03초, reset 6회
(부분 2회), render 116회에서 tactile cache와 physics time/step의 보존을 확인했다.
반지름 3 mm의 test-only sphere로 모든 17개 pad의 해당 채널 ON, 분리 뒤
4 physics step에서 모두 OFF를 확인했다. Probe penetration은 16개 pad 1 mm·
`right_little_force_sensor_3`만 2 mm, 측정력은 0.108537–31.0263 N이었다. 이는
binary 실제 접촉 검증이며 하중 교정으로 해석하지 않는다. 중력·PD hold만으로
모두 OFF, 둘째 환경 palm ON을 유지한 env0 부분 reset의 contacts/valid/raw cache
보존과 reset zero/invalid·다음 완료 step valid를 확인했다. Native 독립 body·
reporter 17개, 41 body·18 DOF·41 collider prim, dense/역할/robot elbow-up·support·
camera visibility·OBB 무변경도 통과했으며 FK 오차는 4.96e−7 m였다. 로그는
`outputs/contact_rearrangement/tactile_gpu_validation.log`다. Probe는 검증 스크립트에만
있으며 기본 task geometry에는 추가하지 않는다. Compile·diff check도 통과했다.

### 2026-10-10 학습용 MDP·초기 장면/16-step 모델·PPO

이 절은 현재/직전 두 sample로 변경하기 전의 구현·검증 기록이다.

사용자가 설계했던 MDP/모델 구현을 요청했고 최근 16 step 이력부터 시작하도록
선택했다. 기준 frame은 examples를 참조하고 물체 기하는 현재 EEF 기준으로
입력한다. 초기 geometry의 base copy와 현재 EEF pose로 재표현하며 현재 object
GT를 actor에 제공하지 않는다. Initial RGB-D의 visible point cloud PCA OBB,
valid/role mask, sensor-frame 보정 F/T, 18D OSC/hand action과 Reward v2·성공/실패,
true-final-observation timeout bootstrap을 구현했다. GRU·privileged critic·
보조 prediction loss는 도입하지 않는다. Model/PPO의 **14개 CPU 검사**에서
슬롯 순열 불변성, NaN padding/all-invalid, 16-step 전체 반영, bounded/saturated
action log-probability, 양 encoder gradient, terminal/timeout GAE 분리와 inference-mode
rollout update, model/optimizer checkpoint 복원이 통과했다.

**통합 검증:** 전체 CPU 검사 **89개**(sampling 32, action/contact/implicit control 22,
MDP 21, model/PPO 14)와 compile/diff 검사가 통과했다. 실제 GPU에서
`train.py --headless --num_envs 2 --iterations 2 --rollout_steps 16
--episode_length_s 0.8 --validate`가 exit code 0으로 완료했다.
열린 손의 unloaded OSC hold 16 step에서 EEF 위치 drift는 최대 **0.0914 mm**,
orientation drift는 최대 **0.00195 rad**, 보정 F/T norm peak는 **0.0108 N /
0.00138 N·m**였다. 실제 Blocker를 proximal cylinder와 겹치게 만든 검증 fixture에서
filtered arm/spacer contact failure가 ON임을 확인했다. 이 fixture는 검증 옵션에서만
실행하며 학습 시작 전에 장면을 다시 reset한다.

학습 rollout은 **64 transition**, PPO optimizer update **2회**를 완료했다.
첫 rollout에서 양 환경의 history 16개가 모두 유효했고 다음 rollout에서는
timeout **2회**의 실제 final observation을 auto-reset 이전에 보존했다.
학습 rollout의 failure는 0회, 보정 wrench peak는 **1.407 N / 0.1885 N·m**였다.
Geometry가 episode 동안 고정된 canonical 초기 값임을 확인하고, 현재 EEF 입력을
SDK frame transform으로 다시 base에 옮겨 원본 초기 기하와 일치함을 검사했다.
Terminal reward의 dt 적용, reset history/action 분리, native arm/mimic PD 비활성,
finite effort, 실제 parameter update 및 GPU checkpoint 복원도 통과했다.
로그는 `outputs/contact_rearrangement/learning_gpu_validation.log`, checkpoint는
`outputs/contact_rearrangement/learning_smoke/model_latest.pt`다. 0.8초 time limit는
reset 검증용이며 기본 학습 limit는 8초다. 이 검증은 구현 연결과 짧은 제어 안정성을
보여주며 재배치 task 성공·정책 수렴·실기 성능을 입증하지 않는다.

초기 진단에서는 SDK unit-mass explicit damping이 손을 열린 채 유지해도 팔을
진동시켰다. Native damping mode의 `dt×eigenvalue`가 최대 **51.78**이었다.
Hard-load 기준을 높이지 않고 coupled mass-aware damping과 backward-Euler 계산으로
수정했다. 또한 GPU에서 child collision mesh filter가 지원되지 않아 rigid-body
partner 경로로 교체했고, manager 초기화 시 이름이 겹치는 hand base 90 g을 포함해
F/T 중력 보상 대상 **33개 body**를 정확히 선택했다. 이 진단의 실패 결과를 위의
최종 통합 검증과 혼합하지 않는다. 기본 학습 출력은
`outputs/contact_rearrangement/learning/<시작 시각>/`에 저장한다.

### 2026-10-10 sweeping_policy 기준 코드 구조 정리

이 절의 학습 검증은 당시 16-step 모델을 사용했다.

사용자가 `examples`의 `sweeping_policy` 구조를 참고하도록 요청했다.
공통 scene·MDP 설정은 `shelf_rearrangement_env_cfg.py`, 로봇별 설정과 Gym
등록은 `config/ur5e`, 모델·PPO·runner 설정은 `config/ur5e/agents`, manager
함수와 공통 transition 상태는 `mdp`, 장면 지원 구현은 `src`로 옮겼다.
장면·센서·샘플러를 작은 helper 파일로 추가 분해하지 않으며 이전 flat 경로의
wrapper도 남기지 않는다. `train.py`는 등록된 task의 환경·agent 설정을 읽는다.
물체의 현재 EEF 기준 재표현, F/T sensor frame, 최근 16-step 이력, 18D action과
Reward v2 명세는 유지한다. 이전 통합 검증 수치를 이번 구조 변경의 검증 결과로
사용하지 않는다.

**구조 변경 후 실제 검증:** Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090·기존
Nucleus asset에서 CPU 검사 **92개**가 **33.203초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 21, model 9, PPO/runner 6, Gym registration 2다.
새 Gym ID의 env/agent entry point를 실제로 읽어 환경을 생성하고 이전
`learning_smoke/model_latest.pt`를 이어서 학습했다. **2환경·64 transition·추가
PPO update 2회(누적 4회)**, failure 0회·timeout 2회, 16-step history·부분 reset,
EEF 초기 기하의 SDK transform 대조, 실제 arm/spacer 접촉 실패 판정,
reward dt·native actuator 설정·parameter 갱신·GPU checkpoint 복원이 통과했다.
Hold drift 최대 **0.0914 mm**, 학습 중 보정 wrench peak는
**1.3860 N / 0.1841 N·m**였다. 실행 옵션은
`train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless --num_envs 2
--iterations 2 --rollout_steps 16 --episode_length_s 0.8
--resume outputs/contact_rearrangement/learning_smoke/model_latest.pt
--log_dir outputs/contact_rearrangement/structure_learning_smoke --validate`이며
로그는 `outputs/contact_rearrangement/structure_learning_validation.log`다.

장면 확인 환경도 `spawn_scene.py --headless --num_envs 2 --max_steps 151
--validate`로 exit code 0을 확인했다. **151 step·5.03초** 동안 150-step 자동
reset과 정착, 역할·밀도·elbow-up 시작, 전체/부분 reset, raw F/T·17채널 tactile
observation, OBB 표시의 USD/PhysX 불변 검사를 통과했다. 로그는
`outputs/contact_rearrangement/structure_spawn_validation.log`다. 이 구조 변경
검증을 재배치 정책의 수렴이나 task 성공률로 해석하지 않는다.

### 2026-10-10 시간 입력을 현재·직전 관측으로 축소

사용자가 previous step 정보까지만 제공하도록 요청해 관측 window를
**현재 t와 직전 t−1의 두 sample**로 변경했다. `MdpCfg.history_length`와
`ModelCfg.history_steps`는 2, actor/critic 입력은 **700D**, 기본 모델은
**248,997 parameter**다. Temporal convolution은 두 sample에 맞춰 1개 layer로
구성하며 초기 장면 GNN·역할 attention·18D action·Reward v2·좌표계를 유지한다.
Reset 직후 현재 sample만 valid이고 직전 sample·현재 action은 0이다.
현재 sample은 그 관측 직전 적용한 aₜ₋₁, 직전 sample은 aₜ₋₂를 담는다.
기존 16-step checkpoint는 새 모델로 이어 학습하지 않으며 새 checkpoint를 쓴다.

**검증:** CPU 검사 **97개**가 **33.587초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 25, model 10, PPO/runner 6, registration 2다.
이력의 오래된 sample 제거·반복 manager 조회·행동 정렬·전체/부분 reset과
이전 episode padding 차단을 검사했다. Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090의
실제 GPU 실행은 **2환경·64 transition·PPO update 2회**, failure 0회·timeout
2회로 exit code 0을 확인했다. 매 전이에서 마지막 두 sample의 현재 action과
직전 관측을 대조하고, timeout의 true final observation·reset previous invalid,
initial geometry의 EEF/SDK transform, 실제 금지 접촉, actuator·reward dt,
parameter 갱신·GPU checkpoint 복원을 확인했다. Rollout history는 두 환경
모두 **2개**였고 보정 wrench peak는 **1.4059 N / 0.1883 N·m**였다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/previous_step_learning_smoke --validate`다.
`rollout_steps`는 PPO batch 수집 길이를 정한다. 로그는
`outputs/contact_rearrangement/previous_step_learning_validation.log`, checkpoint는
`outputs/contact_rearrangement/previous_step_learning_smoke/model_latest.pt`다.
짧은 통합 검증이며 정책 수렴이나 task 성공률을 입증하지 않는다.

## 9. 다음 구체화와 구현에서 정할 사항

아래는 미결 사항이며 다음 작업의 순서가 확정된 것은 아니다.

1. 실제 학습 분포 구체화. 종류별 채택률·실패율, settle 이후 기하 조건 유지, 밀집도 수치와 sampler bias, 로봇의 실제 접촉·접근 가능성을 확인한다.
2. 구현한 tactile 17채널과 raw F/T의 실기 threshold·filtering·동기화 교정, 구현한 F/T 중력·bias 보정의 실제 센서 질량 교정.
3. Base 기준 18D OSC/hand 제어의 gain·workspace·effort limit와 접촉 안정성 검증, 실기 제어 경로.
4. Visible-depth 초기 PCA OBB의 가림·부분 형상 오차, oracle instance-ID association을 대체할 실기 perception, 현재/직전 두 sample에 남지 않는 정보 손실 평가.
5. 구현한 전체 차단·Target 보호·비파지 조건·성공/실패·timeout과 arm/spacer 금지 접촉 기본값의 task 적합성.
6. Reward v2의 초기 수치 조정, 대기 보상 악용과 안정성/속도 trade-off 검증.
7. 구현한 GNN·attention·finite temporal encoder·shared actor/critic·PPO의 학습 수렴과 일반화, 필요 시 구조 변경.
8. 환경 변화 이해를 무엇으로 평가할지와 task reward만의 baseline, 추가 reward/보조 loss 비교의 채택 여부.

연구 논의를 이어갈 핵심 사례는 같은 초기 OBB와 같은 명령에 대해 마찰·접촉
차이로 Blocker 이동량이 달라진 두 episode다. 이력에 어떤 차이가 남는지,
그 차이가 다음 행동을 바꿔야 하는지, 이를 기억·평가 label·reward/loss 중
무엇이 담당하는지 구체화한다. 모든 숨은 상태의 정확한 복원이 목표로 확정된 것은 아니다.

## 10. 이 맥락을 갱신하는 방법

개발 진행, 중요한 연구 결정·사용자 교정·검증 결과가 생기면 이 문서에 반영한다.
대화 전문을 누적하거나 질문마다 별도 보고서로 분리하지 않는다.

- **현재 상태부터 수정:** 1절과 해당 설계·구현 절을 먼저 갱신한다. 과거 문장을 그대로 두고 로그만 추가하지 않는다.
- **이유와 이전 상태 보존:** 무엇을 선택했고 왜 바꿨는지, 이전 선택이 폐기·대체·보류 중 무엇인지 간결하게 남긴다.
- **제안의 상태 유지:** 승인 전에는 미채택으로 두고 사용자 확인 질문을 도입 승인으로 승격하지 않는다.
- **검증 근거 기록:** 날짜, 코드/명령, 버전·환경·asset, 검사 범위와 결과를 남긴다. 미실행 계획을 성공 결과로 기록하지 않는다.
- **주기와 상태 일관성 확인:** Scene, action/observation, reward, termination, reset 간 전제를 확인한다. 복제 환경의 부분 reset과 actor 입력의 GT 유출도 해당 구현에서 확인한다.
- **세부 문서 연결:** 실행·asset 설정은 패키지 README, 큰 MDP/실험 명세는 관련 세부 문서로 연결하고 여기에는 현재 판단을 복원할 핵심을 유지한다.
- **출처 충돌 처리:** 나중의 명시적 사용자 변경을 우선하고 연구 작업안·실제 코드·과거 자료의 차이를 적는다. Notion과 로컬 문서의 동기화는 별도 요청 범위에 따라 수행한다.

## 11. 기준 자료

- [Road to 졸업](https://app.notion.com/p/39e4968094ee8088b12af2849757fa73): 프로젝트 상위 페이지.
- [연구 맥락 — 생각의 흐름과 현재 설계](https://app.notion.com/p/3f34968094ee81ca8eb6feb1f700e0f9): 설계 변경 이유·사용자 교정·미채택 제안의 기준.
- [학습 정리](https://app.notion.com/p/3f24968094ee80c6a0d4f15c94c5e081): Framework·Observation·Reward v2·Action 정리. 초기 vision/history 전환 이전 문구는 최신 맥락으로 읽는다.
- [Reward 설계 근거](https://app.notion.com/p/3f24968094ee819fb1a4df6ff4bdc9a1): 원문 수식·우리 적용·선택 이유·대안. Potential-based shaping은 미채택 대안이다.
- [Track B](https://app.notion.com/p/3da4968094ee804d8f5afeeb17ba5279), [Task BrainStorming](https://app.notion.com/p/3d14968094ee80448d6fe2f3189b8b17): 이전 구상의 역사 자료.
- [새 환경 README](../src/contact_rearrangement/README.md): 현재 실행·asset 경로·구현 위치.

Notion에 언급된 이전 대화 전체·발표 원본·`docs/연구논의/` 등 모든 자료가
이 checkout에 존재하거나 이번 작업에서 직접 확인된 것은 아니다. 필요하면 원문을
확인하고, 이 문서에 보존된 요약을 전체 transcript 확인으로 확대하지 않는다.
