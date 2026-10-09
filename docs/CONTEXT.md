# 연구와 Isaac Lab 개발 맥락

최종 갱신: **2026-10-09, Asia/Seoul**.

밀집 선반에서 Target 접근 공간을 확보하는 비파지 재배치 연구의 목적, 설계가
바뀐 이유, 현재 구현과 검증 결과, 미결 질문을 유지한다. 새 대화와 개발 작업은
이 문서를 먼저 읽고 관련 코드 및 세부 문서로 이어간다.

연구 기준은 Notion의 [연구 맥락 — 생각의 흐름과 현재 설계](https://app.notion.com/p/3f34968094ee81ca8eb6feb1f700e0f9)
이다. 해당 페이지의 연구 기준일은 2026-10-08이며, 2026-10-09에 다시 확인했다.
이 문서는 그 맥락과 이후 이 저장소에서의 개발을 통합한다.

## 1. 현재 위치와 상태 구분

**현재 개발 단계는 로봇·선반·물체 스폰과 reset이다.**
새 패키지 `src/contact_rearrangement`는 **10종 asset 중 4–10개 instance를
선반 전체에 배치하는 밀집 장면**을 만든다. `--all_asset_types`는 10종을 하나씩
모두 사용한다. **Target 종류는 active 장면에 한 번만 등장**하고 Blocker·주변
물체 후보에서 같은 종류의 모든 복사본을 제외한다. 나머지 종류의 중복은 허용한다.
10종 × 10개 복사본의 100-body pool 중 선택된 4–10개만 작업층의 active 물체로
취급한다. Pool 전체의 rigid-body simulation과 native gravity를 켜 두고,
나머지는 높은 격리 위치에 주차해 매 physics substep마다 COM에 `-m*g` 외력을
적용한다. Active 물체에는 추가 힘을 주지 않아 원래 중력과 접촉을 유지한다.
Episode의 Target·Blocker를 먼저 지정하고 배치 재시도에서도 같은 instance 쌍을
유지한다.
Target은 Blocker가 앞에 들어갈 수 있는 전체 X 범위에서 선택한다. 기존 뒤쪽
10 cm 제한과 38 cm cluster·35% tight-rectangle fill 조건은 폐기했다.
선반 전반의 X/Y coverage와 국소 이웃 간격을 따로 검사하고, 병진·평면 pivot·
회전 후 이동 등의 다단계 clearance witness 중 하나를 확보한다.
앞쪽 +X camera의 실제 초기 영상에서 Target 가시 비율 **40% 이상**을 요구하기로
했다. 분모는 같은 camera·같은 pose에서 가림 없이 렌더링한 전체 Target
silhouette 픽셀 수다. **Camera와 영상 acceptance gate는 아직 구현 전**이며
스폰 직후 visibility는 미검증(`False`, fraction `NaN`)으로 남긴다.
YCB 003·004·005·006의 초기 회전을 **X축 −90°**로 교정하고, 원본 USD의
수정 전·후 각 8장 렌더에서 라벨·캔 고리·병뚜껑으로 상하를 확인했다.
전체 선반 규칙의 초기 버전은 실제 USD에서 측정한 bounds로 일반 100 seed·10종 전체 30 seed에서
배치·Target 유일성·coverage·국소 밀집도·다양한 해소 경로 조건을 통과했다.
전체 선반 규칙의 이전 버전에서 unit test 20개와 100-body pool의 4개 GPU
환경의 일반/10종 전체 반복·부분 reset 검사가 통과했다. 각 layout은 한 step만
진행한 검사이며, 10종 전체 장면의 별도 4초 settle 검사는 003 이동으로 실패했다.
현재는 **전물체의 2 mm 낙하 여유를 제거**하고 003·004에만 visible mesh를 유지한
**명시적 Cube collision proxy**를 적용해 초기 흔들림을 줄였다. 원본
Nucleus USD·authored mass·contact 속성·physics material·active gravity는 유지한다.
새 높이에 맞춘 unit fixture 수정 후 20개 테스트와 이 collider 버전의 **4개 GPU
환경·10종 전체·10초** 정착·전체/부분 reset 검사가 통과했다. 초기+첫 reset 후
총 8개 layout을 각각 5초 진행한 범위다. 이전 실패·비채택 approximation과
박스의 정량 이동 측정은 8절에서 구분한다.
판단·행동 정책과 PPO 학습은 아직 구현되지 않았다.

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
의존성을 줄여 촉각·힘 피드백의 빠른 대응을 살리려는 의도다. 초기 관측 시점,
history 길이와 주기, 시간적 기억의 구현은 미정이다.

| 입력 | 현재 의미 | 미정 사항 |
|---|---|---|
| 초기 물체 기하 | 물체별 위치 3D + quaternion 4D + OBB extent 3D = 10D | 단일 카메라의 관측 범위, 초기 기하 오차와 가림 처리 |
| 역할과 유효성 | Target·Blocker 지정, K개 물체 슬롯과 padding | K, role/valid mask 전달 형식 |
| Robot state | Arm·hand 관절 위치/속도 24D + base 기준 EEF pose 7D | 관절 선택·정규화·시간 이력 연결 |
| Wrist F/T | Sensor frame의 힘 3D + 토크 3D, 중력·bias 보정 | Filtering과 동기화 |
| Tactile | 17개 위치의 binary contact | 0.01 N은 변경 가능한 초기 threshold |
| Previous action | 직전 action 18D | 더 긴 행동 history의 사용 여부 |

K는 Target·Blocker를 포함한 전체 최대 슬롯 수다. 같은 물체 기하를 역할별로
중복 계산하지 않는다. 모든 가려진 물체를 초기 vision으로 알고 있다는 가정도 하지 않는다.
관절 관측 차원은 URDF의 전체 물리 관절 수와 독립 구동 관절 수를 구분해 구현해야 한다.

초기 EEF frame에서 얻은 물체 pose를 현재 EEF 기준 pose처럼 취급하면 안 된다.
초기 EEF pose를 보존하거나 base frame으로 변환하는 방법은 후보이며 미채택이다.
과거의 한 step 입력 차원은 explicit valid mask 포함 시 `72 + 11K`, raw zero
padding에서 mask를 내부 생성하면 `72 + 10K`였다. 이를 미정인 temporal input의
최종 차원으로 확정하지 않는다.

Actor가 현재 실제 object pose·velocity를 받기로 한 것은 아니다. 시뮬레이터
정답을 reward·검증·학습 label에 쓰는 것과 actor 실행 입력에 쓰는 것은 구분한다.
History를 외부 상태 추정기로 해석한 뒤 추정 상태를 actor에 넣는 구조는 지양한다.

### 관계 처리와 학습

현재 구조 작업안은 다음과 같다.

```text
물체별 기하와 역할
  → 공유 처리 / GNN
  → Target·Blocker 조건부 masked attention readout
  → scene feature와 robot state·F/T·tactile·previous action 결합
  → actor의 18D action
```

- 슬롯 나열 순서를 바꿔도 같은 장면에 대해 같은 판단을 하도록 집합을 처리한다. 실제 공간 배치는 입력에 남아야 한다.
- GNN edge는 메시지 교환을 허용하는 연결이다. 접촉·방해 관계의 정답 label을 미리 제공하는 것으로 정하지 않았다.
- 로봇에 대한 물체 위치를 판별할 정보가 필요하다. 물체 간 상대 관계만 남기는 구조로 단순화하지 않는다.
- Role mask와 valid mask는 목적이 다르다. Zero padding에서 valid를 만들면 noise·정규화 전에 생성하고 message passing과 attention에서 padding을 제외한다.
- GNN의 같은 forward 내 message-passing 반복과 episode의 시간적 기억은 다르다.
- PPO와 custom encoder 공동학습 및 Isaac Lab 연계를 계획한다. Query·edge 구성·층 수·feature dimension, critic의 privileged input, actor/critic 공유 범위는 미정이다.

### Action과 제어

| 출력 | 차원 | 제어 경로 |
|---|---:|---|
| EEF motion | 6 | 위치·회전 증분 → variable-stiffness OSC |
| Cartesian stiffness | 6 | 병진·회전 각 3축 강성 → OSC |
| Hand configuration | 6 | RH56E2 독립 구동 관절 목표 위치 → position controller |

총 **18D action**이며 arm의 12D 출력은 OSC를 통해 관절 토크로 연결한다.
목표를 이전 명령에 계속 누적하지 않는 방향을 유지한다. 직전 action을 관측하는
것과 목표 누적은 별개다. 증분 기준 frame·scale·강성/감쇠 범위·정책 주기·실기
UR5e 인터페이스는 미정이다. 현재 스폰용 PD hold는 이 학습용 제어 구현과 구분한다.

## 4. Reward 기준점과 사용자 교정

**Formulation v2는 현재 작업안이며 새 환경에는 아직 구현되지 않았다.**
Transition `s_t → s_{t+1}`에 대해 계산한다. 상위 weight는 비음수, scale과
`Δt`는 양수다. 실제 weight·scale·soft threshold·종료 허용오차와 학습 충분성은 미정이다.

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
reward 계산 좌표계를 혼동하지 않는다.

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
- 실패 범주는 낙하·전도·workspace 이탈·금지 접촉·hard-force limit 등이다. 구체적인 threshold는 미정이며 성공과 실패가 동시에 검출되면 실패를 우선한다.
- Timeout의 과업 실패와 학습용 truncation 구분, 실기에서 초기 vision만으로 성공을 판별하는 방법은 미정이다.

### 구현과 검증에서 남은 문제

양의 매-step 점수는 대기하면서 보상을 누적하는 행동을 유리하게 만들 수 있다.
성공 bonus를 **할인된 누적 대기 return**과 비교해야 한다. 안정성 항이 rapid
동작을 과도하게 막는지, 물체 수에 따른 합산 scale 변화, 평면 면적이 실제 hand/arm
접근 가능성을 충분히 나타내는지도 검증해야 한다.

로컬 Isaac Lab [RewardManager](../IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py)는
각 term에 `weight * dt`를 곱한다. Policy-step 수식을 그대로 옮길 때 시간 벌점의
`Δt`와 one-shot 종료 보상을 포함해 시간 scale을 중복 적용하지 않도록 명세를 정해야 한다.
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

현재 미채택 후보는 다음과 같다.

- 시간적 구조를 명확히 한 정책을 기존 task reward만으로 학습하는 baseline.
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
`ManagerBasedEnv`를 사용하는 스폰 단계이며 action/observation manager는 비어 있다.
Gym 학습 task ID·Reward v2·종료 조건·18D OSC action·PPO·GNN·history는 후속 구현 범위다.

### 장면과 초기 설정

- UR5e + RH56E2 **오른손**을 fixed joint로 하나의 articulation에 조립한다. 손의 fixed link와 mimic 관계를 유지하고 contact-report API를 활성화한다. 손 mount 회전은 tool frame 기준 `(0,-90,0)` deg다.
- Kinematic speedrack shelf와 **기존 6종 + YCB 4종 = 10종** asset을 사용한다. YCB는 `NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics`의 `003_cracker_box`, `004_sugar_box`, `005_tomato_soup_can`, `006_mustard_bottle`이다. 해당 폴더에서 확인한 4개 USD는 rigid body와 collision이 이미 authoring되어 있다.
- 장면당 **4–10개 active instance**를 사용한다. `all_asset_types=True`는 catalog 10종에서 하나씩 총 10개를 사용한다. 각 종류의 복사본 10개씩, 환경당 100개 rigid body를 미리 로드하고 reset에서 선택한 instance만 작업층으로 옮긴다. 나머지는 `z=4.0/4.4/4.8/5.2 m`의 격리 위치에 주차하고 속도를 0으로 설정한다. Parking은 작업층 camera ROI 밖에 두며 USD visibility를 수정해 숨기지 않는다. **Pool의 모든 body는 simulation enabled와 native gravity를 유지**해 topology를 바꾸지 않는다. Active mask로 task 물체를 구분하며 이후 actor·reward·종료 계산에도 이를 적용해야 한다. Observation 최대 슬롯 K를 100으로 확정하는 뜻은 아니다.
- `TaskObjectPool`은 매 physics write/substep에서 inactive body에만 world force `-m*g`를 추가해 중력을 상쇄한다. PhysX `apply_forces_and_torques_at_position`에 실제 **world COM 위치**를 명시해 asset root pivot과 COM의 차이로 불필요한 토크가 생기지 않도록 한다. Active body의 추가 힘은 0이며 native gravity와 접촉 동역학을 유지한다. Live CPU simulation/gravity flag나 USD visibility를 매 reset마다 바꾸지 않는다.
- **Target 종류는 active 장면에 정확히 한 번** 등장한다. 해당 종류는 Blocker·주변 물체에서 제외하고, 다른 종류의 중복은 허용한다. 역할은 서로 다른 instance ID에 지정한다. 초기 baseline은 Target 1개와 Blocker 1개이며 episode 시작에 쌍을 먼저 선택한다. 배치 rejection/retry가 발생해도 쌍과 active 개수를 유지하고 주변 instance·yaw·위치만 다시 뽑는다. 제한 횟수 안에 배치하지 못하면 다른 역할로 교체하지 않고 실패를 보고한다.
- `cfg.target_asset`·`cfg.blocker_asset`과 CLI의 `--target_asset`·`--blocker_asset`으로 역할의 종류를 고정할 수 있다. 기본 `None`은 episode 역할을 처음에 한 번 추출한다. 두 옵션에 같은 종류를 지정하면 Target 유일성에 위배되어 오류다. Blocker 종류만 고정한 경우 Target 후보에서도 그 종류를 제외한다. 이는 외부 역할 지정이며 Blocker의 최종 목표 pose를 제공하는 것이 아니다.
- 로봇 base는 환경 로컬 `(0, 0, 0.79505)`, quaternion `(w,x,y,z)=(0,0,0,1)`, 선반은 `(-0.7,0,0)`에 놓인다. Quaternion 표기는 Isaac Lab 순서다.
- 기존 2×3 고정 배열과 `cup_1` Target/`mug_2` Blocker는 초기 구현 이력이다. 현재는 환경별 seed와 reset 횟수로 episode 역할을 먼저 정하고 그 역할에 맞는 upright yaw·배치를 뽑는다. 선택되지 않은 환경의 reset sequence는 유지한다. 종류·역할별 배치 실패율은 따로 확인하며, 역할을 placement 성공률에 따라 바꿔 분포를 감추지 않는다.
- 물체의 spawned rigid-body local bounds를 읽고 canonical upright rotation을 적용한다. YCB 4종은 local −Y를 world +Z로 옮기는 X축 −90° 회전 후 world Z yaw를 적용한다. 수정 전 양쪽 렌더에서 003·004·005의 라벨과 006의 뚜껑이 모두 아래를 향해 기존 +90°를 교정했다. Root pivot과 bounding center를 구분해 XY를 보정하고, 실제 lower Z bound로 선반 위 높이를 정한다.
- 모든 물체의 root Z는 `shelf_top - lower_z`로 정해 bounds 바닥을 선반 상면에 바로 맞춘다. 기존 2 mm 여유는 제거했다. 003·004만 `FlatBoxUsdCfg`로 visible scanned mesh의 collision을 끄고 동일 body-local AABB 크기·center offset의 invisible Cube collider를 추가한다. 기존 single body를 유지하고 clone spawn 시점에만 적용하며 Nucleus 원본을 수정하지 않는다. Authored mass·원본 contact 속성·physics material을 유지하고 active body의 gravity·simulation을 그대로 사용한다. 자동 fit `boundingCube` approximation은 최종 구현이 아니다.
- 실제 shelf collision에서 support와 상부 panel을 읽는다. 현재 기본 asset의 작업층은 환경 로컬 `x=[-0.9,-0.5]`, `y=[-0.455,0.455]`, 상면 `z=1.05 m`, 위쪽 panel 하단 `z=1.48 m`이며 물체 footprint에 edge margin과 상부 여유를 적용한다. Asset override에서는 해당 geometry를 다시 측정한다.
- 기본 물리 주기는 `1/120 s`, decimation은 4로 환경 step은 `1/30 s`다. 기본 환경 수 1, 간격 2.5 m다.
- 장면 확인용 `spawn_scene.py`는 기본적으로 시뮬레이션 시간 5초마다 전체 환경을 reset해 새 배치를 생성한다. 실제 `env.step_dt`로 간격을 계산하며 현재 설정에서는 150 step이다. `--reset_interval N`으로 step 간격을 지정하고 0이면 자동 reset을 끈다. 이 설정은 실행 스크립트의 확인 주기이며 학습 episode timeout을 확정하는 것은 아니다.
- Reset은 선택한 환경의 robot state·PD 관절 목표를 복원하고, 새 active mask·역할·초기 state·고정 접근영역을 기록한다. 물체의 pose/velocity에는 각 `env_origins`를 적용한다. 모든 body의 simulation/native gravity flag는 유지하고 inactive body의 parking pose·zero velocity와 countergravity mask를 갱신한다. 새 flat-box 버전도 4개 환경·10종 전체의 10초 실행에서 전체·부분 reset을 통과했다.
- 17개 tactile용 링크와 독립 구동 관절의 존재는 확장 버전에서도 검증했다. 새 환경의 `ContactSensorCfg`, binary tactile observation, wrist F/T 신호·보정은 아직 구현되지 않았다.

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
별도로 active 물체의 **50% 이상이 4 cm 이내 footprint 이웃**을 갖도록 한다.
두 조건으로 선반 전반의 분포와 국소 clutter를 구분한다. 전체 shelf footprint
occupancy는 기록하지만 최소 35%를 요구하지 않는다. 모든 수치는 사용자 요구를
구현하기 위한 현재 기본값이며 최종 연구 명세나 동역학 안정성 보장이 아니다.

**Clearance witness**는 병진·footprint 모서리의 평면 pivot·중심 회전 후 병진·
작은 앞쪽 이동 후 회전/병진 등의 후보에서 하나를 찾는다. Pivot은 고정점 주변의
원호로 검사하고 회전 구간을 최대 5°씩 나눠 arc sagitta로 확장한 보수적 swept
polygon을 사용한다. 전체 경로가 support 안에 있고 Target·주변 물체와 떨어져야
한다. 경로를 예약한 뒤 주변 물체를 선반 전체에 분산하며 가까운 이웃도 제안한다.

Witness는 생성·검증용 privileged metadata다. Actor에 Blocker의 최종 goal pose로
입력하는 것이 아니며 정책이 이 병진 해법만 수행하도록 결정한 것도 아니다.
이 조건은 보수적 평면 기하에서 경로 하나가 존재한다는 뜻이다. Bounding corner
pivot은 실제 hand contact point를 인증하는 것이 아니다. UR5e의 접근
가능성, 손의 접촉점과 충돌, OSC로 밀 수 있는지, 마찰·회전·전도를 포함한 동역학,
센서 기반 종료 판정과 실제 재배치 성공은 아직 입증하지 않았다.

### 초기 vision의 가시성 조건 — 실제 camera 검사 후속

`InitialVisionCfg`는 앞쪽 **+X camera에서 −X를 보는 방향**,
`min_target_visible_fraction=0.40`, `require_full_target_in_frame=True`를 기록한다.
선택한 분모는 같은 camera intrinsics·pose와 같은 settled Target pose에서
Target만 외부 가림 없이 렌더링한 **전체 silhouette 픽셀 수**다. 실제 장면의
Target mask 픽셀 수를 분자로 사용한다. 다른 물체뿐 아니라 shelf·robot 가림도
포함하고 Target self-occlusion은 reference silhouette에 자연스럽게 반영한다.
가시 비율의 상한은 추가하지 않는다.

현재 task camera는 없고 `target_visibility_validated=False`,
`target_visible_fraction=NaN`이다. 향후 **고정 역할/개수 → pose 제안 → settle와
안정성 검사 → RGB·instance mask·reference → 40% 조건 검사** 순으로 초기
장면을 승인한다. 실패 시 같은 역할/개수의 pose만 재시도한다. 승인된 초기
vision과 같은 settled pose로 A₀를 정한 뒤 episode 내에서는 고정해야 한다.
현재 A₀는 spawn pose 기준이며 이 후속 gate를 구현한 것으로 취급하지 않는다.

| 파일 | 역할 |
|---|---|
| [layout.py](../src/contact_rearrangement/contact_rearrangement/layout.py) | 10종 catalog, 100개 instance pool, upright 회전·주차 위치·asset 경로 |
| [assets.py](../src/contact_rearrangement/contact_rearrangement/assets.py) | UR5e/RH56E2 조립, 관절·링크 이름, 초기 자세·PD hold |
| [scene_cfg.py](../src/contact_rearrangement/contact_rearrangement/scene_cfg.py) | 선반·로봇·물체 pool·ground/light 구성 |
| [object_pool.py](../src/contact_rearrangement/contact_rearrangement/object_pool.py) | `TaskObjectPool`, 매 physics substep의 inactive COM countergravity 적용 |
| [object_spawner.py](../src/contact_rearrangement/contact_rearrangement/object_spawner.py) | 003·004 visible scanned mesh 유지, body-local AABB Cube collision override |
| [geometry.py](../src/contact_rearrangement/contact_rearrangement/geometry.py) | Spawn된 body bound·authored physics 검사, shelf support/ceiling 측정 |
| [sampling.py](../src/contact_rearrangement/contact_rearrangement/sampling.py) | Seeded 4–10개/all-types 배치, Target 유일성·coverage·국소 밀집도·pivot/회전 path 검사 |
| [env_cfg.py](../src/contact_rearrangement/contact_rearrangement/env_cfg.py) | ManagerBasedEnv, 주기, 환경 복제, reset event, spawn rules, 후속 initial-vision 요구값 |
| [mdp/events.py](../src/contact_rearrangement/contact_rearrangement/mdp/events.py) | Episode 역할 선지정, 전체/부분 resampling, active mask·초기 state, parking·countergravity mask 관리 |
| [spawn_scene.py](../src/contact_rearrangement/scripts/spawn_scene.py) | 실행, 스폰·상태·reset 검증 |

Asset 기본 경로는 `omniverse://192.168.0.13`이며
`CONTACT_REARRANGEMENT_*` 환경변수로 override할 수 있다. 손은 저장소의
`RH56E2_R_2025_9_11_relative.urdf`를 사용한다. 세부 명령과 변수는 패키지 README에 둔다.

### 기존 환경에서 참고한 범위

Examples와 `roas_sweeping`에서 shelf·6개 물체 종류와 배치를,
`hand_rl`에서 RH56E2 오른손 조립을, Isaac Lab의
`scripts/tutorials/03_envs/create_cartpole_base_env.py`에서 base environment
생명주기를 참고했다. `sweep_jh`의 OSC·가상 F/T는 후속 구현 참고 후보이며 새 환경에 통합되지 않았다.
기존 환경의 목표 pose·reaching IK·sweep reward·단일 active-object reset을
새 연구 명세로 그대로 사용하지 않는다. 오래된 `docs/hand_rl`의 왼손 설명보다
현재 오른손 코드를 확인해야 한다.

## 8. 개발과 검증 기록

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
gravity와 simulation을 유지**한다. 새 `object_pool.py`의 `TaskObjectPool`이
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

**현재 구현:** `object_spawner.py`의 `FlatBoxUsdCfg`는 003·004의 visible scanned
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

## 9. 다음 구체화와 구현에서 정할 사항

아래는 미결 사항이며 다음 작업의 순서가 확정된 것은 아니다.

1. 실제 학습 분포 구체화. 종류별 채택률·실패율, settle 이후 기하 조건 유지, 밀집도 수치와 sampler bias, 로봇의 실제 접촉·접근 가능성을 확인한다.
2. 17개 binary tactile과 wrist F/T의 센서 정의, 중력·bias 보정, filtering·동기화·threshold.
3. 18D OSC/hand 제어의 frame·scale·강성/감쇠·정책 주기 및 실기 제어 경로.
4. 초기 vision snapshot의 좌표계·보존 시점, 관측 가능한 물체·padding mask, history 또는 hidden state의 reset.
5. 고정 접근영역·footprint·전체 차단 계산과 Target 보호·비파지 조건·성공/실패·timeout의 정확한 정의.
6. Reward v2의 수치·시간 scale·누적량 갱신 주기, 대기 보상 악용과 안정성/속도 trade-off 검증.
7. GNN·attention·temporal 표현·actor/critic·PPO의 공동학습 구조.
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
