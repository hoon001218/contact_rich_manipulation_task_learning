# Contact Manipulation 학습 기록과 수정 과제

작성·검토: **2026-10-21, Asia/Seoul**. 현재 프로젝트는
`src/contact_manipulation`, task는 `Isaac-Contact-Manipulation-UR5e-v0`다.
연구·개발 결정의 전체 이력은 [CONTEXT.md](CONTEXT.md), 현재 구현은
[프로젝트 README](../src/contact_manipulation/README.md)를 참고한다.

**최신 판단:** `2026-10-21_19-59-36`은 value loss의 일시적 극단값이 후반의 지속적 붕괴로
이어진다. 첫 spike285회 전후 저장 Critic 가중치는 정상 규모이며, 실제 입력·Value·GAE target은
저장되지 않아 최초 원인은 미확정이다. 보상 감소를 접촉 회피의 합리적 수렴으로 단정하지 않고
가치 추정·입력 이상치를 먼저 분리한다. 정확한 loss 의미와 수치는 **37절**을 따른다.

**초기 분석:** 10월19일의 추가 학습·reward 조합5개를 분석하고 저장정책을 고정조건에서 재생했다.
현재 정책들의 Clearance 개선은 거의 없으며, 조합별 생존시간 차이는 관측된다.
공통checkpoint750의320episode는 성공·통로확보0회, 과부하/센서범위초과278회였다.
장기실행은 Gaussian std뿐 아니라 deterministic 평균 명령도 비정상적으로 커졌다.
상세 비교·실제평가·한계는 **10절**에 둔다. 해당 분석에서는 새 학습이나 production수정을
하지 않았다. 이후 손바닥 EEF action 변경은 **11절**, 물체 종료 조건 완화는 **12절**,
현재 UR5e 말단 제어·관측 통일과 중력 보상은 **13절**에 기록한다.
최신 학습 후반 불안정은 **17절**, 물체 물성은 **18절**, Hold0.05·정책100Hz 적용과
self-collision slowdown 실측은 **19절**을 참고한다. 이전 실행의50Hz 설정과 구분한다.

아래1–6절은 이전 본 학습6개에 대한10월19일 분석이다.
사용자가 관찰한 **“초반에 물체를 툭 치고 hand가 선반 반대 방향으로 이동”**하는 행동을
주요 문제로 기록한다. 최초 로그 정리의 분석·수정안은1–6절에 보존했다.
이후 적용한 Gaussian 출력 전환은 **7절**, Reward penalty·IK복구와 검증은 **8절**,
Terminal스케일 조정은 **9절**이다.
T01–T06은 이 변경 전 학습이며, 변경 후 새 본 학습 결과를 섞지 않는다.

## 1. 기록 범위와 로그 해석

`logs/rsl_rl/contact_manipulation/`의 72개 실행을 TensorBoard scalar,
`params/env.yaml`, `params/agent.yaml`, checkpoint 목록과 기존 조사 기록으로 분류했다.
다른 task의 학습과 이전 `contact_rearrangement` 프로젝트는 섞지 않았다.

| 분류 | 수 | 이번 기록에서의 처리 |
|---|---:|---|
| 대화·기존 문서에서 본 학습임을 확인 | 6 | 아래 본 학습 이력에 포함 |
| 본 학습 후보, 실행 목적 확인 필요 | 3 | 별도 보류 목록에 보존 |
| 이름 없는 짧은 실행, 목적 불명 | 36 | 테스트로 단정하지 않고 보류 |
| scalar 없이 초기화까지만 남은 실행 | 2 | 목적·학습 진행 여부 미확인 |
| 명시적 profile/probe/audit/validation 등 | 15 | 성능·정책 학습 비교에서 제외 |
| 설정상 최대 64 iteration 이하의 검증 실행 | 10 | 성능·정책 학습 비교에서 제외 |

**짧게 종료됐다는 이유만으로 테스트라고 분류하지 않는다.** 전체 분류 근거·파일 경로·
scalar 요약은 [실행 목록 JSON](../outputs/contact_manipulation/training_review_20261019/run_inventory.json)에
보존했다. 원래 CLI 명령이 저장되지 않은 실행은 명령을 추정해 쓰지 않고 저장된 설정을 기준으로 한다.
저장 git hash만으로 실행 버전을 단정하지 않는다. 당시 미커밋 diff와 실행 시 저장한 설정도 함께 봐야 한다.

로그를 읽을 때 다음을 적용한다.

- Iteration 번호는 0부터 시작한다. `0…29573`은 **29,574회 기록**이며
  `model_29550.pt`와 마지막 완료 iteration은 다르다.
- `Train/mean_reward`, `Train/mean_episode_length`는 최근 완료 episode 100개의
  이동 통계다. 이후 이 문서의 “마지막 100 iteration 평균”은 이 로그 값 100개의 평균이다.
- Native `Episode_Reward/*`는 **weight가 적용된 episode 누적 기여량 ÷ 최대 episode 8초**다.
  순간 0–1 품질 점수나 실제 episode 시간으로 정규화한 평균이 아니다.
  Reset subset 평균을 RSL이 다시 평균하며, reset 없는 step에는 이전 extras가 반복될 수 있다.
- `Episode_Termination/*`는 환경별 마지막 종료 상태를 바탕으로 한 snapshot 통계다.
  로그가 양수인 iteration 수를 성공 episode 수로, 평균값을 정확한 종료 사건 비율로 읽지 않는다.
  여러 실패 원인이 동시에 참일 수도 있다.
- 당시50Hz 공유 장면의400-step 경계에서도 전체 환경을 truncation한다. 따라서 `time_out`에는
  **개별 episode가 8초를 모두 사용하지 않은 공통 장면 종료**도 포함된다.
  이50Hz 기준을 과거30Hz 실행에 소급 적용하지 않는다. 19절에서 적용한100Hz 설정은800step이다.
- 제어 주기·action scale·reward 수식과 weight가 변했다. 서로 다른 실행의 return 크기로
  정책 우열을 매기지 않는다. `Episode_Reward` 합과 `Train/mean_reward`도 모집단이 달라 직접 대조하지 않는다.

## 2. 본 학습 이력

6개 모두 1,024환경, seed 0, rollout 32, `resume=False`, 목표 90,000 iteration으로
설정했다. 아래 시간은 **첫 scalar부터 마지막 scalar까지의 기록 구간**이며 초기 로딩 시간을 포함하지 않는다.

| ID / 실행 디렉터리 | 기록 iteration 수 | 최신 checkpoint | 기록 구간 | 마지막 100회 return / 길이(step) | 마지막 100회 collection+learning |
|---|---:|---:|---:|---:|---:|
| T01 [2026-10-13_17-25-25](../logs/rsl_rl/contact_manipulation/2026-10-13_17-25-25/) | 184 | 150 | 1시간 56분 | 326.76 / 185.15 | 40.96초 |
| T02 [2026-10-13_23-18-25](../logs/rsl_rl/contact_manipulation/2026-10-13_23-18-25/) | 2,082 | 2050 | 10시간 56분 | 327.12 / 191.34 | 18.23초 |
| T03 [2026-10-14_23-25-08](../logs/rsl_rl/contact_manipulation/2026-10-14_23-25-08/) | 3,557 | 3550 | 3시간 8분 | 436.65 / 254.69 | 3.10초 |
| T04 [2026-10-15_17-57-45](../logs/rsl_rl/contact_manipulation/2026-10-15_17-57-45/) | 19,100 | 19050 | 9시간 16분 | 172.15 / 133.56 | 1.62초 |
| T05 [2026-10-16_21-36-16](../logs/rsl_rl/contact_manipulation/2026-10-16_21-36-16/) | 940 | 900 | 29분 | 91.34 / 95.28 | 1.76초 |
| T06 [2026-10-17_15-52-37](../logs/rsl_rl/contact_manipulation/2026-10-17_15-52-37/) | 29,574 | 29550 | 14시간 26분 | 243.12 / 161.11 | 1.73초 |

Collection에는 physics뿐 아니라 observation·reward·termination·reset·정책 추론이 들어간다.
위 표는 PhysX 단독 시간이나 제어된 동일 trajectory 성능 비교가 아니다.

### 실행별 설정 변화

| 실행 | Physics / policy | Arm action과 actuator | Clearance / Rearrangement / Motion weight | 종료 로그 |
|---|---|---|---|---|
| T01·T02 | 120 / 30 Hz | Custom 통합 action, 위치 0.015 m·회전 0.08 rad scale, Implicit arm | 1 / 1 / 1 | success·timeout·통합 failure |
| T03 | 100 / 50 Hz | Custom ArmOscAction·HandPositionAction, 위치 0.015·회전 0.08, Implicit arm | 1 / 1 / 1 | 동일 |
| T04 | 100 / 50 Hz | Native OSC·joint position, 위치 1 m·회전 1 rad, stiffness scale 100, Implicit arm | 1 / 1 / 1 | 동일 |
| T05 | 100 / 50 Hz | Native action 유지, arm을 **IdealPD 150/28 N·m, P/D=0**으로 변경 | 1.8 / 0.6 / 0.6 | 실패 원인 7개로 분리 |
| T06 | 100 / 50 Hz | T05와 동일 | 1.8 / 0.6 / **4개 × 0.15** | 동일 |

Time·Terminal manager weight는 위 실행 모두 각각 1이다. 과거 Motion과 현재 Motion은
weight만 다른 것이 아니라 수식도 변경됐다. T05부터 action-rate·joint-velocity·force-growth·
moment-growth의 네 독립 exp 점수를 평균하며, T06은 **그 합산 기여량을 바꾸지 않고 로그만 분리**했다.
조사한 저장 physics/action/actuator/PPO 설정에서 T05→T06 차이는 Motion term 분리뿐이다.
과거 수식은 실행 당시 snapshot·당시 분석을 기준으로 읽으며 현재 수식을 소급 적용하지 않는다.

### 실행별 결과와 종료 원인

- **T01:** 초기 장기 학습·속도 문제 조사 대상. Success 로그가 양수인 iteration은 20개,
  마지막 139다. 종료 원인은 저장 자료로 확정하지 못했다. 기록이 짧고 후속 설계와 달라
  현재 정책 성능의 기준으로 사용하지 않는다.
- **T02:** 기존 `model_2050.pt` 재생에 사용한 본 학습. Success 양수 로그는 129개,
  마지막 2078이다. 종료 원인은 미확인이다. 현재 제어·reward와 달라 성공 통계를 직접 비교하지 않는다.
- **T03:** 사용자는 물체를 접촉하는 행동을 관찰했다. Success 로그는 전 구간 0이다.
  **10월 15일 02:33:10 KST에 PhysX CUDA 719**로 중단됐다.
  [당시 장애 조사](../outputs/contact_manipulation/cuda719_20261015/incident.json)와
  [reward 검토](../outputs/contact_manipulation/reward_review_20261015/summary.json)를 남겼다.
- **T04:** 사용자가 약 19,100 iteration 시점 정책을 IdealPD 변경 전에 재생했을 때
  **물체를 조작하지 않고 선반 앞에서 움직이다 timeout**되는 행동을 보고했다.
  첫 1,000회→마지막 1,000회 return은 80.91→193.56, 길이는 90.29→144.51 step이었다.
  Success 양수 로그 10개, 마지막 8099로 후반에는 나타나지 않았다.
  **10월 16일 03:14:15 KST에 CUDA 719**로 중단됐다.
  [기존 reward 분석](../outputs/contact_manipulation/reward_log_review_20261016/summary.json)을 참고한다.
- **T05:** IdealPD 제한과 수정된 Motion·weight를 적용했으나
  **10월 16일 22:06:01 KST에 CUDA 719**가 다시 발생했다.
  마지막 10회 return −43.90, 길이 30.96 step, hard-load 종료 snapshot 평균 0.918로 악화됐다.
  Success 양수 로그 21개, 마지막 546이다. Scalar는 finite였고 `invalid_state` 로그는 0이었다.
  `model_900.pt`는 마지막 기록 939보다 **39 PPO update 이전**이므로 장애 직전 정책과 같지 않다.
- **T06:** 아래 최신 학습 분석 대상이다. 마지막 scalar는 **10월 18일 06:19:26 KST**다.
  해당 종료 구간에서 CUDA 719는 확인되지 않았다. 후속 보고된 초기자세 탐색 소진 예외의
  원본 stderr가 실행과 연결돼 저장되지 않아 **이 run의 종료 원인으로 확정하지 않는다**.

CUDA 719의 Python traceback은 GPU 장애 뒤 DOF velocity 조회가 실패한 지점이다.
OOM·버퍼 부족·특정 reward가 직접 원인이라고 단정하지 않는다. 이전 사고의 모델·optimizer가
finite였다는 사실도 실제 물리 상태나 제어 수치가 안전했다는 증명은 아니다.

## 3. 최신 학습 T06의 관찰과 해석

### 사용자 관찰과 로그 사실

**사용자 관찰:** 초기 물체 접촉 후 hand가 선반 반대 방향으로 이동하는 경향.
정확한 재생 checkpoint·seed·episode 영상은 아직 기록되지 않았다. 이 문서 작성 중
정책을 재생해 직접 확인한 행동은 아니며, 아래 scalar만으로 이동 방향·접촉 물체·인과를 복원할 수 없다.

| 지표 | 첫 1,000회: 0–999 | 마지막 1,000회: 28574–29573 |
|---|---:|---:|
| Train/mean_reward 평균 | 59.15 | 227.18 |
| Train/mean_episode_length 평균 | 75.64 step / 1.51초 | 152.08 step / 3.04초 |
| Timeout snapshot 평균 | 0.2108 | 0.7882 |
| Hard-load/overrange snapshot 평균 | 0.7817 | 0.2124 |
| Clearance / Rearrangement 상대 aggregate | 0.4865 | 0.4995 |
| Action-rate / Rearrangement 상대 aggregate | 0.9827 | 0.9582 |
| Joint-velocity / Rearrangement 상대 aggregate | 0.8623 | 0.8825 |
| Force-growth / Rearrangement 상대 aggregate | 0.9950 | 0.9996 |
| Moment-growth / Rearrangement 상대 aggregate | 0.9470 | 0.9622 |

상대 aggregate는 같은 구간의 **평균 native reward 로그에서 weight를 제거한 합의 비율**이다.
예를 들어 `C/R = (mean(C_log)/1.8)/(mean(R_log)/0.6)`,
`Motion_i/R = (mean(M_i_log)/0.15)/(mean(R_log)/0.6)`다.
Episode 길이 영향을 일부 함께 제거해 비교하지만, **정확한 평균 순간 quality·가림 비율·
물체 이동량은 아니다.** 특히 `C/R≈0.5`를 가림 면적 50%로 해석하지 않는다.

Success 로그가 양수인 iteration은 `6455, 6459, 16033–16037`의 **7개**다.
마지막 16037 이후 29573까지 양수 로그가 없었다. 성공 episode 7개라는 뜻은 아니다.
Return 상승만으로 작업을 학습했다고 판단하기 어렵고, 현재 자료는 **실패를 피하고 오래 유지하는
행동으로 보상을 늘렸을 가능성**을 지지한다. Episode 내 Clearance 개선량은 아직 측정하지 못했다.

![최신 학습 추이](../outputs/contact_manipulation/training_review_20261019/latest_trends.png)

그림은 로그 값과 100-iteration 이동 평균을 표시한다. 초록 점선은 success 로그가 양수인
iteration이며 성공 횟수·성공률이 아니다. 아래 두 패널은 weight를 제거한 aggregate의 비율이다.
자세한 통계는 [분석 JSON](../outputs/contact_manipulation/training_review_20261019/latest_metrics.json),
원본 scalar는 [CSV](../outputs/contact_manipulation/training_review_20261019/latest_scalars.csv)에 있다.

T06의 collection+learning 전체 평균은 **1.740초/iteration**, 마지막 1,000회는 **1.704초**다.
관측된 기록 구간은 14시간 26분이며, 이 속도가 유지돼도 90,000회는 약 43시간이다.
따라서 다음 장기 학습 전에 **작업 진행과 성공을 검증할 짧은 평가 기준**이 필요하다.

### 후퇴 행동을 설명할 수 있는 T06 당시 reward 구조

T06 당시 한 policy step의 보상은 다음과 같다. 이후 적용식은8절에 둔다.

$$
\begin{aligned}
r_k ={}& 1.8q_C + 0.6q_R + 0.15(q_A + q_V + q_F + q_T) \\
&+ r_{\mathrm{time}} + r_{\mathrm{terminal}}.
\end{aligned}
$$

- `q_C=exp(−ρ)`, `ρ=Blocker와 초기 접근영역 A₀의 교차 면적 / 현재 Blocker 투영 면적`.
  지정한 Blocker만 평가하며 hand 위치·지속 접촉은 요구하지 않는다.
- `q_R=exp(−2ΔL_corner)`. 물체마다 이번 step의 8개 OBB corner 변위를 평균한 뒤
  활성 물체 전체에 합산한다. 물체 간 평균은 아니다.
  이동이 멈추면 다시 1이며 과거 조작량을 계속 감점하지 않는다.
- `q_A=exp(−0.004Σ(a_k−a_{k−1})²)`, `q_V=exp(−0.004Σq̇²)`.
  일정한 큰 상대 pose 명령은 action-rate 비용이 작을 수도 있다.
- `q_F`, `q_T`는 보상된 F/T 크기의 **양의 증가율**에서 dead band를 뺀 제곱에 대한 독립 exp다.
  하중을 유지하거나 힘을 빼는 것은 감점하지 않는다. `q_T`는 F/T moment이며 joint torque가 아니다.

물체·관절이 멈추고 action 변화와 하중 증가가 없다면, terminal·time을 제외한 보상은
`1.8 exp(−ρ)+1.2`다. 통로를 전혀 더 비우지 않아도 ρ=1이면 **1.862/step**,
ρ=0.5이면 **2.292/step**다. 작은 접촉으로 ρ를 0.5→0.4로 바꾼 뒤 그 상태를 유지하면
Clearance 기여는 이후 매 step 약 **0.115** 늘어난다.
이는 **부분 진전 뒤 접촉을 풀고 기다리는 행동에도 계속 양의 보상**이 남는다는 뜻이다.
실제 후퇴 원인이나 전역 최적 정책이 증명됐다는 뜻은 아니다.

Time은 `−0.02∫(exp(0.7t)−1)dt`이며 8초 누적 **−7.538**,
마지막 20 ms 구간 **−0.107**이다. 현재 양의 step 보상에 비해 작다.
또한 γ=0.99, 50 Hz에서 8초 뒤 +500은 첫 step 기준 `500·γ^399≈9.07`
(보상 index를 0부터 둔 경우)로 할인된다. Time·Terminal·γ·양의 baseline을 함께 검토해야 한다.

단, **통로를 확보한 뒤 손을 빼고 물체가 안정될 때까지 기다리는 행동은 올바를 수도 있다.**
현재 성공에는 Blocker 교차 면적 ≤10⁻⁵ m², Target 초기 pose 유지 ≤2 cm/0.15 rad,
전체 active 물체 속도 ≤0.015 m/s·0.1 rad/s, 5-step 연속 안정, 실패 없음이 모두 필요하다.
다음 두 경우를 구별해야 한다.

1. 통로가 충분히 열리지 않았는데 후퇴해 timeout을 기다리는 경우.
2. 통로는 열렸지만 Target 변위·주변 물체 흔들림·안정 counter 때문에 성공하지 못하는 경우.

현재 Clearance의 비율과 성공의 절대 면적 기준은 다르다. 회전으로 Blocker 투영 면적이
변하면 상대 점수와 절대 통로 확보가 다르게 움직일 수 있다. Target pose 초과도 현재
성공을 막는 조건이지 독립 실패 종료가 아니므로, 이 상태가 된 뒤 보상만 받는지 확인해야 한다.

## 4. 구현된 변경과 남은 수정 과제

### 이미 반영된 사항

| 사항 | 현재 상태와 한계 |
|---|---|
| Native OSC·hand position action | 공식 ActionCfg 사용 중. Custom controller로 다시 바꾸는 작업은 필요하지 않음 |
| 팔 IdealPD, P/D=0, 150/28 N·m | T05부터 적용. 출력 제한 동작 확인. 큰 상대 목표·특이점·행렬 수치 문제까지 해결한 것은 아님 |
| 실패 원인 분리 | 낙하·기울어짐·영역 이탈·들림·hard load/overrange·선반 변위·invalid state 7개 |
| 선반 충돌에 대한 종료 | 초기 대비 **1 cm / 2°** 초과를 이미 검사. 금지 접촉 센서를 다시 추가할 필요 없음 |
| Motion 분리 | T06부터 native 4개 term 각각 0.15. 기존 총 Motion 0.6과 정상 유한 입력에서 동일 |
| Reset·vision 최적화 | 공통 배치·초기 ray 관측 공유, 환경마다 새로운 GPU batched IK. 실제 카메라 생성 없음 |

### 수정 목록

아래는 최초 검토 당시의 수정 목록이다. 이후 FIX-01의복구와 FIX-04의R/Mbaseline제거를
8절에 구현·검증했다. 후보별 세부탈락진단과 학습성능검증은 남아 있다.

| ID / 우선순위 | 문제 | 수정 방향 | 완료 판단 |
|---|---|---|---|
| FIX-01 / 우선 | 초기자세 검색 한 행 실패가 장기 학습 전체 중단 | 탈락 사유·seed·환경 ID 기록, 유효한 TCP 영역 확보, 검증된 조건을 지키는 bounded 복구 | 실패 행만 복구하고 다른 환경·난수열 유지, elbow·충돌 조건 보존 |
| FIX-02 / 우선 | 성공 부족·후퇴 원인을 현재 scalar로 구분 불가 | 접촉 전후 overlap·성공 gate·행동·종료 사건을 episode 단위로 기록 | 실제 성공 횟수·원인별 사건 수와 후퇴 전후 진행량을 재현 가능 |
| FIX-03 / 다음 | 상대 pose scale과 OSC 수치 증폭 | 공식 ActionCfg의 scale 검토, 목표/실제 변위·토크 포화·행렬 conditioning 측정 | 작업에 필요한 빠른 접근을 유지하면서 과도 목표·추종 실패·수치 증폭 감소 |
| FIX-04 / 다음 | 부분 Clearance 유지·비활동에도 지속 양의 보상 | 성공 gate와 정합성을 확인하고 exp 내부 비용·기여량/weight 대조 실험 | idle·tap-and-retreat보다 안전한 통로 확보가 평가 성능에서 우세 |
| FIX-05 / 다음 | 늦은 성공의 할인과 작은 Time 비용 | γ·Time 증가율/크기·Terminal·step baseline을 공동 검토 | 성공 시점별 discounted return과 실제 학습 성공이 일치하는지 확인 |
| FIX-06 / 보류 조사 | 접촉 후 실제 물체 상태를 정책이 식별하기 어려울 가능성 | 초기 vision 고정·현재/직전 센서만으로 구별 불가능한 사례를 수집 | 관측의 모호성이 확인된 경우에만 후속 모델/입력 설계 논의 |

### FIX-01: reset 초기자세 탐색 소진

대상 예외는 [events.py의 `_shared_reset_batch`](../src/contact_manipulation/contact_manipulation/mdp/events.py)에서
발생한 `Shared scene robot-start search exhausted retries for 1 environments`다.
CUDA 719와 다른 **명시적 후보 탐색 실패**다. `1`은 마지막 실패 행 수이며 환경 ID가 아니다.
외부 32회 × 내부 80후보로 실패 행당 최대 2,560후보를 검사하지만, 같은 공통 배치와
TCP 영역이 고정돼 유효 영역이 매우 작으면 계속 실패할 수 있다.
공통 장면 준비 시 env 0의 유효 자세 하나를 찾은 사실만으로 충분한 시작 영역이 보장되지는 않는다.

현재 탈락 단계가 합쳐져 있어 실제 원인이 IK 수렴·elbow·scene/self collision 중 무엇인지 알 수 없다.
다음 내용을 수정 대상으로 남긴다.

1. 실패 시 env ID, 배치 generation/seed, reset count, TCP 영역과 단계별 탈락 수를 저장한다.
2. 유효 시작 영역을 검증하고 그 안에서 **연속적인 pose randomization**을 유지한다.
   검증된 자세 주변 sampling도 검토하되 고정 posture bank로 대체하지 않는다.
3. 성공한 행은 유지하고 실패한 행만 검증된 대체 proposal/복구 경로로 처리한다.
   조건 무시, 무검증 자세 사용, 무한 재시도, 단순 반복 상한 증가는 해결책으로 삼지 않는다.
4. 1,024행 중 일부 실패·동일 seed 재현·부분 reset 독립성·추가 지연을 검증한다.

### FIX-02·03: 진단 항목과 제어 검토

다음 평가는 native logger·기존 평가 경로를 활용하고, 큰 tensor의 매 step CPU 복사나
전체 raw trace의 상시 저장은 피한다. 종료 시 작은 episode 통계와 선택한 trajectory만 보존한다.

- 작업 진행: 초기/최소/종료 Blocker 교차 면적, Blocker 투영 면적과 ρ,
  Target 보호 gate, 전체 물체 안정 gate, stable counter, EEF–Blocker 거리와 선반 방향 변위.
- 조작: 첫 tactile 접촉 시점·해제 이후 진행량, 실제 물체 변위·회전,
  F/T 증가율 분포와 hard-load 발생 시점. 지속 접촉 자체를 목표로 삼지는 않는다.
  현재 tactile 센서는 접촉 상대를 구분하지 않으므로 이 신호만으로 Blocker 접촉을 판정하지 않는다.
- 제어: raw/processed delta pose와 Kp, 20 ms 실제 EEF 변위, 추종 오차,
  raw OSC torque와 actuator 제한 후 제출 명령·포화 빈도, near-singular 사례.
  **제출 torque를 측정 joint torque로 부르지 않는다.**
- 종료: 실제 reset 대상의 사건 수를 누적해 성공·실패·공통 scene truncation을 나누고
  실패 원인이 겹치는 사건도 보존한다. 기존 snapshot 로그와 구분해 이름을 정한다.

별도 진단 재실행에서 T05 `model_900.pt`는 상대 위치 목표 norm 중앙값 **0.669 m**,
최대 **1.471 m**, raw OSC torque 최대 **1.636×10⁷ N·m**를 출력했다.
환경·physics substep의 **33.2%**에서 한 관절 이상이 제한됐고 실제 제출 명령은 150/28 N·m 이내였다.
Peak에서는 near-singular Jacobian과 FP32 task inertia의 양의 정부호성 손실이 확인됐다.
[재실행 자료](../outputs/contact_manipulation/cuda719_20261017/action_replay_stochastic.json)와
[독립 수치 분석](../outputs/contact_manipulation/cuda719_20261017/peak_independent_analysis.json)을 참고한다.
이것은 **본 학습 이력에 추가한 run도, 장애 직전 상태 재현도, 최신 T06 정책 측정도 아니다.**
재실행에서는 CUDA 719가 발생하지 않았다. TF32 단독 원인도 확정하지 않았다.

따라서 native ActionCfg의 위치/회전 scale을 물리적으로 추종 가능한 범위와 대조하고,
공식 OSC 설정의 수치 조건을 점검하는 것이 우선이다. 아직 새 scale 수치는 정하지 않았다.
IdealPD를 제거하거나 직접 만든 OSC inverse/controller로 교체하는 것으로 해석하지 않는다.

### FIX-04·05: reward 검토 기준

- **합의 유지:** Clearance·Rearrangement·Motion의 `exp(−a·x)` 구조,
  네 Motion 점수의 독립 계산과 weight 분할을 유지하며 우선 `x`와 실제 작업 진행의 관계를 본다.
  `exp(−x)−1` 또는 여러 비용을 먼저 더한 단일 exp로 이미 변경한 것이 아니다.
- Clearance는 지정한 Blocker만 평가한다. 다른 non-target 물체를 통로 reward에 다시 포함하지 않는다.
  비율 개선과 절대 통로 확보·Target 보호·안정화가 어디서 어긋나는지 먼저 확인한다.
- Rearrangement는 현재 step의 pose 조작량이다. 정지 상태 보상이 높다는 이유만으로
  이 항이 현재 실패의 지배 원인이라고 단정하지 않는다. 예를 들어 경로 1 mm의 즉시 손실은
  `0.6(1−exp(−0.002))≈0.0012`, 1 cm는 약 0.0119다.
- Motion은 빠른 접근과 필요한 감속을 불필요하게 억제하지 않아야 한다.
  F/T dead band 10 N/s·0.5 N·m/s와 half-quality 증가율 100 N/s·5 N·m/s는
  초기 설계값이며 검증된 안전 한계가 아니다. 실제 분포·접촉 성공과 대조해 결정한다.
- 성공 미달 상태의 보상 유지가 원인으로 확인되면 Clearance 내부 비용/기여 방식과 weight,
  Time·Terminal·γ를 개별 실험한다. 진행량 기반 shaping은 검토할 수 있지만
  **potential difference를 적용하면 기존 양의 exp 점수 계약도 달라지므로 미채택 후보**로 남긴다.
- Tactile은 17개 binary observation을 유지하되 reward 추가는 보류한다.
  Joint torque observation/reward, 추가 history encoder도 아직 확정하지 않는다.

## 5. 다음 실험의 기록 규칙

1. **현재 baseline 동결:** T06 checkpoint 16000 전후와 29550을 같은 배치·seed·초기
   pose 집합으로 평가한다. 정책 구조/activation/action 계약을 확인하고 버전이 다른 checkpoint를 섞지 않는다.
2. **평가 모드 구분:** deterministic과 stochastic 추론을 분리하고, 지정한 checkpoint·seed·
   episode 수·설정 diff를 저장한다. 접촉, Clearance 달성, Target 보호, 안정화, 실제 성공을 따로 집계한다.
3. **수정은 한 단계씩:** reset 복구/진단 → 제어 scale·수치 조건 → reward 대조 실험 순서로 진행한다.
   같은 평가 집합에서 성공 수·진행량·실패 사건·시간·토크 포화를 비교한다.
4. **장기 학습 전 판정:** 짧은 학습의 return 상승만으로 채택하지 않는다. 안전한 통로 확보가
   정지·tap-and-retreat보다 실제 평가에서 개선됐는지 확인한 뒤 장기 실행한다.
5. **각 본 학습 기록:** 실행 ID/목적, 부모 checkpoint 또는 새 학습, 정확한 명령과 git+diff,
   env/agent 설정, 적용 FIX ID, 마지막 iteration/checkpoint, 종료 stderr, scalar,
   동일 조건 평가와 사용자 관찰을 이 문서에 추가한다. 진단 실행은 별도 이름으로 저장한다.

## 6. 2026-10-19 후속 교정과 구체화

아래는 사용자의 후속 의견과 조사 결과다. **코드 적용·성능 검증과 구분한다.**

### Clearance를 주목표로 하는 보상 구성

사용자는 Clearance 상승을 우선하고, Rearrangement·Motion은 penalty 또는 조건부 누적으로
두는 방향을 제안했다. 우선 비교할 최소 변경안은 다음과 같다.

$$
\begin{aligned}
r_k ={}& w_Cq_{C,k} - w_R(1-q_{R,k}) \\
&- \frac{w_M}{4}\sum_{i\in\{A,V,F,T\}}(1-q_{i,k}) \\
&+ r_{\mathrm{time},k} + r_{\mathrm{terminal},k},
\qquad q = e^{-ax}.
\end{aligned}
$$

각 exp는 기존 quality 계산으로 유지한다. 정지 시 Rearrangement·Motion은 0이고,
조작량·속도·행동 변화·F/T 증가 비용이 발생할 때만 음수가 된다. 기존의 +1.2/step
비활동 보너스를 제거하는 **제안**이며, 아직 현재 코드의 양의 reward를 바꾼 것은 아니다.
여기서 x≥0, a,w>0이면 비용 `w(1−exp(−ax))`는 x=0에서 0이고 x>0에서 양수이며,
reward 기여는 그 음수다. x가 커질수록 비용은 증가하되 증가폭은 줄고 w에 포화한다.
작은 양의 x에도 작은 비용이 있다. F/T의 x는 기존 dead band를 뺀 뒤 0 이하를 잘라
만들므로 허용 증가율 이내에서는 센서 변화가 있어도 x=0이다. 이는 미채택 penalty 안의 설명이다.
Clearance의 양의 상태 보상은 남으므로 부분 진전 뒤 대기를 완전히 해결한다고 단정하지 않는다.
현재 weight를 먼저 고정해 지급 방식만 비교하고, 이후 의미 있는 Clearance 개선량 대비
실제 성공 조작의 비용을 보고 weight를 결정한다. `w_C>w_R+w_M`만으로 모든 작은 진전이
우선된다고 보장할 수는 없다.

Clearance가 상승하는 step에만 R/M 비용을 부과하면 생산적인 조작에만 벌점을 주거나,
반대 방식에서는 pivot 준비·일시적인 겹침 증가·재접촉을 잘라낼 수 있다. 우선 즉시 비용을
비교하고, 조건부 누적은 필요성이 확인된 경우에만 별도 검토한다. 성공 후 접촉을 풀고
물체 안정화를 기다리는 행동은 허용한다. Time·Terminal·할인과 실제 성공 gate도 함께 관찰한다.

### 공식 OSC scale 조사

로컬 Isaac Lab `VERSION`은 2.3.2다. 공식
[v2.3.2 Franka OSC Reach 설정](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/osc_env_cfg.py)은
`position_scale=1.0`, `orientation_scale=1.0`, `stiffness_scale=100.0`이지만
**`pose_abs`의 절대 위치·quaternion 명령**이다. 공식
[OSC tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_osc.html)도
절대 목표 pose·wrench를 직접 전달한다. 우리 `pose_rel`의 위치/axis-angle 증분 scale과
숫자만으로 동일시하지 않는다. 이전에 제안한 0.015 m/0.08 rad는 우리 과거 설정의
대조값이지 공식 OSC 상대 제어 예제에서 가져온 기본값이 아니다.

로컬 추가 UR10 OSC Reach에는 pose_rel 0.03/0.15가 있으나 upstream 공식 예제로
확인되지 않아 공식 권장값으로 인용하지 않는다. 현재 scale은 이번 조사에서 변경하지 않았다.

후속 질문으로 원격 min도 직접 조회했다. [min의 OSC 설정](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py)의
`UR5eSweepOSCEnvCfg`는 pose_rel·position/orientation_scale 1.0이고,
`UR5eSweepKpOSCRelativeObsEnvCfg`는 variable_kp·stiffness_scale 100·limits 50–200을 쓴다.
현재 우리 설정과 같은 scale·상대 입력 방식이다. 따라서 1.0 자체를 오류 또는 축소 필수로
판정하지 않는다. 큰 명령·OSC 수치 증폭은 별도 측정된 문제이므로 정책 출력 분포,
추종·포화·자세 조건을 보고 판단한다. 로컬 origin/min은 8월 snapshot으로 현재 원격과 달라
원격 파일을 GitHub 연결로 확인했다(file blob `d2f51e40b356683116a446763f38717546654ab7`).

후속 action 구성 조사에서 min의 fixed OSC는 pose6+binary gripper1=7D,
variable-Kp OSC는 pose6+Kp6+binary gripper1=13D임을 확인했다. Pose는 robot-root 기준
위치 증분3(m)·회전벡터3(rad)이며 velocity/관절 torque 명령이 아니다. Kp 입력은
`clip(100a,50,200)`이고, binary gripper 한 입력이 여러 관절의 open/close 목표를 선택한다.
우리 환경은 pose6+Kp6+독립 hand6=18D다.

등록된 OSC task의 [runner 설정](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py)은
표준 Gaussian 계열·초기 noise std1.0이며 repo 안에 별도 tanh/clip_actions override가 없다.
외부 Isaac Lab 실행 버전의 기본값은 별도로 고려해야 한다. 우리 actor는 초기 latent std0.3의
tanh Gaussian으로 약 ±1에 제한된다. 따라서 같은 scale이어도 action 분포와 실제 Kp 도달
범위는 같지 않다. Min의 scale1이 내부에서 작은 pose 증분으로 다시 축소된다고 설명하지 않는다.

Pose6D만 보면 native 경로는 `p*=p_actual+s_p*a_pos`,
`R*=Exp(s_R*a_rot)R_actual`이다. 현재는 robot-base 축이며 회전은 Euler가 아닌 회전벡터다.
매20ms policy 시점의 실제 EEF에서 목표를 만들고 두10ms physics step 동안 유지한다.
Delta에 dt를 곱하지 않으며 해당 거리를20ms에 실제 이동한다는 뜻도 아니다.
우리 bounded actor의 scale1은 위치 성분별 약±1m, 벡터 norm 최대약√3m다.
Min의 Gaussian 출력에는 그 자체로±1상한이 없으므로 scale1을최대1m라고 부르지 않는다.
과거 model900 replay의 norm최대1.471m는 이 성분별 범위를 위반한 것이 아니다.

### 기준 EEF pose 주변 position·orientation randomization 후 IK

사용자가 지정한 순서는 **Blocker 정면의 기준 EEF pose → 위치·회전 범위 sampling → IK**다.
현재는 위치 sampling 후 IK를 풀지만 orientation이 고정돼 있다.
기준 EEF는 OSC·IK가 사용하는 `right_plam_force_sensor`(손바닥 TCP)이며 tool0·F/T frame과 구분한다.
환경 좌표의 기준 pose를 `(p₀,R₀)`라 하면 기준 EEF 축의 변화량으로
`p*=p₀+R₀δp`, `R*=R₀δR`를 만든다. Quaternion은 `q*=q₀⊗δq` 순서다.
Position 범위를 무작위 회전으로 함께 돌리지 않도록 위치에는 기준 R₀를 사용한다.

기존 position 분포(Blocker 앞 12–40 cm, 좌우 ±8 cm, 선반 위 20–28 cm와 선반 앞 여유)는
기준 pose·EEF-local offset으로 재표현할 수 있다. 회전은 작은 범위부터 검증하며,
예를 들어 local rotation-vector 각 성분 ±10°는 **미확정 초기 후보값**이다.
기존 장착 회전에 −90°를 다시 적용하지 않는다.

위치·회전 후보 tensor를 기존 GPU batched IK에 함께 전달한다. IK는 이미 후보별 회전을
지원하므로 새 solver가 필요하지 않다. Elbow/전완 방향·관절 한계·scene/self collision은
유지하고, 물리 검증의 고정 회전 비교도 선택한 목표 회전을 따르도록 해야 한다.
실패 행 재제안·기준 자세 seed·유효 영역 확인은 이 sampling 안에서 처리한다.
사용자가 아직 최종 범위나 코드 적용을 확정한 것으로 기록하지 않는다.

**이 randomization 추가만으로 후보 탐색 소진이 해결되지는 않는다.** 회전 변화는 새로운
유효해를 만들 수도, 기존 위치를 더 어렵게 할 수도 있다. 새 공유 배치를 채택하기 전에
기준 p/R의 elbow-up 해와 주변 6D 후보를 검증하고, 그 관절해를 IK seed로 사용하며
실패 행만 같은 배치의 검증된 영역에서 유한하게 재제안해야 한다. 후보별 IK 잔차·관절 한계·
elbow·충돌 탈락 원인도 필요하다. 부분 reset 실패를 이유로 공유 물체 배치를 바꾸면
다른 환경에 영향을 주므로 그렇게 복구하지 않는다. 유효 기준점 하나 또는 유한한 재시도만으로
모든 random pose의 성공을 보장한다고 설명하지 않는다.

### IK 수정안의 함수별 범위

사용자는 현재 상태와 수정 위치·이유를 구체적으로 제안하도록 요청했다. 아래는 미적용 설계안이다.

| 위치 | 현재 | 제안하는 변경과 이유 |
|---|---|---|
| RobotStartRules | 위치 범위만 설정. Robot 외부 retry는 vision 설정을 재사용 | 회전 범위와 robot 전용 검색 budget을 선언. 위치·elbow·충돌 한계는 유지 |
| `_build_episode_batch` | 공유 배치에 안전한 시작 자세 하나만 확인 | Blocker 앞 기준 pose와 elbow-up `q_anchor`, 주변 pose probe를 준비해 시작 영역 품질을 검사하고 채택 |
| `_sample_robot_starts` | 목표 위치와 관절 seed를 동시에 변경, 회전 고정 | 기준 EEF 축에서 위치·회전 생성. 같은 목표에 anchor seed 우선, 실패 시 다른 branch를 적용해 목표와 IK 초기값 탐색을 분리 |
| `RobotPoseModel.solve_batch` | 후보별 회전을 지원하는 48회 DLS | Solver·오차 기준을 유지하고, 필요한 위치/회전 잔차·finite 진단만 반환 |
| `_shared_reset_batch` | 동일 영역을 seed만 바꿔 반복하다 전체 예외 | 성공 행 유지, 실패 행만 anchor 주변의 원래 범위 안에서 연속 pose 재제안. 재시도 횟수 증가보다 후보 품질 개선 |

현재 네 seed branch는 shoulder lift −2.0/−1.6/−0.6/−1.0, elbow1.8/1.2/−1.8/−1.2,
wrist1=−lift−elbow를 사용한다. 첫8개는 추가 관절 noise0, 다음8개는±0.5rad,
이후는±1rad다. 각 branch마다 목표 pose도 달라 같은 목표에 여러 seed를 푸는 구조가 아니다.
수정 시 목표 후보축과 seed축을 분리하되 anchor로 성공한 목표에 다른 seed를 계산하지 않는다.
GPU accepted mask·기존 chunk/scratch cap·per-env seed 독립성·공유 scene cache는 유지한다.

탈락 진단은 IK 잔차/finite·관절 한계·elbow·forearm 방향·바닥·scene/self collision을 구분한다.
후보별 작은 reason code를 GPU에서 모으고 reset 종료/실패 시 통계만 읽는다.
새 파일이나 별도 IK solver, 반복적인 실제 physics trial, 정상 step의 CPU trace는 추가하지 않는다.

최종 검색 소진으로 전체 학습이 중단되는 것을 막으려면 **별도의 예외 정책도 필요**하다.
실용적인 후보는 같은 초기 물체 배치에서 검증된 `q_anchor`로 실패 행만 복구하고 카운트를
남기는 것이다. 이 경우에 한해 연속 randomization을 포기한다는 tradeoff가 있으므로,
사용자가 이미 승인한 정책으로 기록하거나 일반 sampling에 몰래 적용하지 않는다.
이 예외까지 허용하지 않는다면 유한한 random search만으로 무조건 성공을 보장할 수 없다.
부분 reset에서 공유 배치를 임의 변경하지 않고, anchor의 안전성도 없으면 오류를 숨기지 않는다.

검증은 δR=0 기존 자세 재현, 같은 목표의 seed별 성공/실패, elbow/충돌 불가 후보 거절,
reset 순서·분할에 따른 seed 독립성, 1024행 중 한 행 실패 주입과 복구,
실제 GPU 전체/부분 reset의 오차·성공률·fallback 빈도·지연을 포함한다.

### CUDA 719

사용자는 추가 실행을 보며 확인하는 것으로 정리했다. 원인 미확정 상태를 유지하고,
위 보상/자세 설계 검토와 CUDA fault의 인과를 연결하지 않는다. 이번 조사에서 새 학습·재생은 하지 않았다.

## 7. 2026-10-19 적용 — 표준 Gaussian 정책 출력

사용자가 min과 유사한 출력으로 전환하는 방향을 제시했다. Tanh가 큰 latent 값을 압축하고
포화 구간에서 출력 변화가 작아지는 것은 사실이나, 이것이 큰 action 지속·후퇴 행동의
원인이었다는 증거는 아직 없다. 제거하면 같은 latent의 실제 목표가 더 커질 수도 있다.

이번 구현은 RSL-RL `GaussianDistribution` 하나를 공유 policy가 소유하도록 바꿨다.
학습은 `a=μ+σε`, 일반 play는 `a=μ`다. Native sampling·log-probability·analytic entropy·
KL을 사용하고 tanh/atanh/Jacobian 보정·경계 확률·Monte Carlo entropy 코드를 제거했다.
표준편차는 native `std_type="log"`로 양수를 유지하며, 이전 custom log_std clamp도 제거했다.
기존 초기 std **0.3은 유지**해 출력 변환 효과를 분리했다. Min의 초기 std1.0 및 다른 PPO
hyperparameter까지 모두 맞춘 것은 아니다. GNN·attention·ReLU·현재/직전 concat·244D/18D는 유지했다.

ActionCfg와 position/orientation scale1.0은 유지한다. Gaussian pose 입력에는 ±1 상한이 없다.
OSC Kp의 native50–200 clamp, hand의 native±1 clamp 후 관절 범위 매핑,
IdealPD의150/28N·m 제한은 유지한다. Actor std와 actual state의 관계는 별도 실행으로 검토해야 한다.
현재/직전 observation의 raw action을 두 곳에서 ±1로 자르던 처리도 제거해 실제 명령 크기를
보존한다. 비유한 입력 sanitization·invalid-state 검사는 유지한다.

Native PPO update와 큰 minibatch의 microbatch 경로 모두 같은 Gaussian 계산을 사용한다.
분포 entropy가 analytic이므로 추가 entropy noise 생성과 chunk별 noise 처리가 필요 없다.
이번 전환 자체의 물리 처리량·성공률 향상을 주장하지 않는다. Reward·reset 알고리즘은 바꾸지 않았다.

Checkpoint에 `output_distribution`(GaussianDistribution/log/init_std0.3)을 추가했다.
물리적 `action_contract`는 유지하되, 기존 tanh checkpoint는 actor·critic·optimizer를 읽기 전에
출력 분포 불일치로 거절한다. 기존 checkpoint를 그대로 새 출력 방식으로 resume/play하지 않는다.
README의 train/play 예시는 새 `--run_name gaussian` 실행을 대상으로 갱신했다.

검증:

- CPU learning: **57 passed, 2 skipped, 20 subtests passed**.
- CPU MDP: **152 passed, 12 skipped, 125 subtests passed**.
- RTX3090에서 learning 전체: **59 passed, 20 subtests passed**. GPU fused GNN·컴파일된
  추론/학습 전환·gradient·checkpoint key 검사를 포함한다. 3개 Inductor 성능 경고가 있었다.
- Gaussian 분석식 log-prob/entropy, 큰 평균의 무제한 출력, deterministic mean,
  entropy의 RNG 비소모, microbatch/native PPO 일치, std 학습·단일 소유,
  이전 출력 분포 checkpoint의 사전 거절과 current/previous action>1 보존을 검증했다.

명령은 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 <env_isaaclab python> -m pytest
src/contact_manipulation/tests/test_learning.py -q`와 같은 형식으로 각 test 파일을 실행했다.
이 검증은 실제 PhysX 환경의 장기 학습/재생이나 CUDA719 재현 검증이 아니다.

## 8. 2026-10-19 적용 — R/M penalty와 초기자세 탐색 복구

사용자의 “적용시키고 로그 남겨”에 따라 앞 절의 보상·IK 수정안을 반영했다.
새 본 학습을 시작한 기록은 아니며, 정책 출력은 7절의 Gaussian을 유지한다.

### 보상 지급 변경

품질 $q_i=\exp(-a_i x_i)$와 x·계수·weight를 유지하고 실제 step 보상을 다음과 같이 바꿨다.

$$
\begin{aligned}
r ={}& 1.8q_C + 0.6(q_R-1) \\
&+ 0.15\bigl[(q_A-1)+(q_V-1)+(q_F-1)+(q_T-1)\bigr] \\
&+ r_{\mathrm{time}} + r_{\mathrm{terminal}}.
\end{aligned}
$$

Clearance는 지정 Blocker가 초기 접근영역을 가리는 면적 비율의 양의 주 보상이다.
Rearrangement는 이번 step의 모든 활성 물체 OBB corner 이동량, Motion은 action-rate·
관절속도·F/T 크기 증가율의 기존 네 비용을 쓴다. 비용0에서 R/M 기여0, 비용이 커지면
각 항은 음수가 되어 weight의 음수에 접근한다. 과거 이동량을 반복 감점하지 않는다.
Tactile·joint torque 보상이나 접촉 gate, Clearance 증가량 보상은 추가하지 않았다.
Time·Terminal·γ·관측·ActionCfg는 이번 변경에서 유지했다.

예전 수식에서 R/M 상수1.2/step을 제거했다. 정지·후퇴 시의 양의 R/M 지급을 없애지만,
모든 작은 Clearance 개선이 다른 비용보다 우선한다는 수학적 보장은 아니다.
Clearance의 양의 state 보상은 남으므로 변경 후 성공률·접촉 후 진행·timeout을 다시 봐야 한다.

`reward_components`의 0~1 품질 캐시는 유지하고 native reward wrapper에서만1을 뺀다.
Native8항 이름과 logging방식은 유지하며 `Episode_Reward/rearrangement`와 네 Motion로그는
이제0이하다. 구버전 return·양의 항 로그와 직접 비교하지 않는다. 센서 조회·CPU변환은 늘리지 않았다.

### IK 및 reset

- Blocker앞 기존 위치 영역을 유지하고 `right_plam_force_sensor` 기준 local rotation-vector
  각 성분±10°를 적용한다. 임의 회전으로 위치 직사각형까지 회전시키지 않는다.
- 동일 목표 pose에 검증된 기준 관절값을 우선 사용하고 다른3개 branch seed도 시도한다.
  총80후보는20목표×4seed다. Elbow-up·전완방향·관절한계·scene/self충돌 검사를 유지한다.
- 새 공유배치는 기준해를 얻은 뒤 독립seed의6D목표8개를 각1번 검사해 최소4개가 유효해야
  채택한다. 이 probe는 유한 표본의 품질검사이며 전체 영역의 유효성을 보장하지 않는다.
- 부분reset은 성공행을 유지하고 실패행만 일반3회, 기준점근방2회로 탐색한다.
  근방위치는 환경축별±(2,2,1.5)cm와 원래영역의 교집합, 회전은±2.5°다.
  Robot retry는 `robot_start.reset_attempts/recovery_attempts`로 vision budget과 분리했다.
- 끝까지 실패한 행만 같은 초기장면에서 검증된 관절값·TCP·영역으로 복구한다.
  이 예외는 새 자세 randomization을 포기하며, 다른환경·물체배치·vision은 바꾸지 않는다.
  검증된 기준점도 없는 새 layout은 제한된 proposal후 명시적으로 실패한다.

GPU tensor·Warp IK·충돌kernel·CUDA graph와 임시후보상한을 유지한다. 기준관절값·방향은
graph replay입력으로 전달해 새 장면에 이전 anchor가 남지 않게 했다. 같은 버전에서
환경ID·reset횟수에 따른 난수 독립성과 chunk크기에 따른 목표/seed 불변성을 유지하지만,
회전난수·pose/seed축 변경으로 이전버전과 같은 seed의 구체적인 자세는 달라진다.

Native로그는 `Reset/robot_start_fallback_count`, `Reset/robot_start_fallback_fraction`,
`Reset/robot_start_fallback_total`, `Reset/robot_start_recovery_count`,
`Reset/robot_start_search_rounds`다. Reset event뒤 native `_reset_idx`가 log dict를 초기화하므로
pending진단을 이후 observation계산에서 합친다. 환경step이나runner를 override하지 않았다.
Native runner가 평균한 count를 누적횟수로 합산하지 말고 `fallback_total`을 함께 확인한다.

최종복구가 발생하면 실행log폴더의 `robot_start_fallbacks.jsonl`에 환경ID·seed·reset횟수·
장면번호·budget·탐색소진 사유를 남긴다. Normalreset에는 파일쓰기가 없다. 후보별IK잔차와
충돌 세부원인을 분리하는 진단은 이번에 추가하지 않았다.

검증자료는 [reward_ik_20261019](../outputs/contact_manipulation/reward_ik_20261019/)에 둔다.
CUDA719를 해결했다고 판단하거나 이번변경이 성공률을 개선했다고 주장하지 않는다.

검증 결과:

- CPU MDP 전체 **162 passed, 12 skipped, 125 subtests passed**. Learning은
  **57 passed, 2 skipped, 20 subtests passed**.
- RTX3090 learning+MDP 회귀검사 **224 passed, 7 skipped, 145 subtests passed**.
  이후 추가된 최종 IK·로그 검사9개도 별도 통과했다. 1,024행 중1행실패 주입,
  성공행·neighbor보존, domain gate/rollback, 회전·seed·recovery범위, cache갱신,
  native로그·JSONL 기록을 확인했다.
- 실제 Isaac Sim **4환경·32 policy step·부분reset2회**에서 native8개항의
  weight·dt적용을 확인했다. 독립식과 최대오차 **1.19e−7**, R/M은 모두0이하였으며
  다른환경 물리상태·공유template이 보존됐다. 이 실행에서 최종fallback은0회였다.
- 동일 실제장면의 GPU sampler를 **1,024행**으로 실행해 **1,024/1,024 유효해**를
  얻었다. Warm-up뒤3회 **29.5 / 29.8 / 29.5 ms**였다. 위치·회전·IK·충돌검사만의
  시간이며 전체scene reset·PhysX·학습iteration 시간은 아니다. 한장면·고정seed 검사다.
  기준방향 대비 실제 EEF 총회전각은 약1.02–16.33°였다(성분별±10°와 구분).
- 실제환경 검증 중 발견한 CUDAgraph 내부 `new_tensor(tuple)`의host→GPU전송을
  사전GPU상수cache로 수정했다. 별도 실제sampler graph회귀검사 **1 passed**로
  InferenceMode capture/replay와 template교체뒤 기준관절값·방향갱신도 확인했다.
  장기학습·다양한장면의fallback빈도는 아직측정하지 않았다.

## 9. 2026-10-19 적용 — Terminal 스케일 조정

최신 본 학습 T06(`2026-10-17_15-52-37`)을 다시 확인해 `MdpCfg`의 기본값을
성공 **+500→+200**, 실패 **−100→−50**으로 조정했다. Native Terminal weight는1이다.
성공·실패 조건, 실패 우선·episode당1회지급, timeout지급0과 bootstrap, γ=.99,
Time·Clearance·R/M 수식과 weight는 이번 변경에서 유지한다.

### 로그 근거와 선택 기준

| T06 native 평균 | 첫1,000 iteration | 마지막1,000 iteration |
|---|---:|---:|
| `Episode_Reward/clearance` | 8.279 | 19.151 |
| `Episode_Reward/terminal` | −12.424 | −12.426 |
| Terminal 절댓값 / Clearance | 1.501 | 0.649 |
| 실패벌점만 절반으로 본 같은 로그집계 | −6.212 | −6.213 |
| 그때 Terminal 절댓값 / Clearance | 0.750 | 0.324 |

이 두구간은 성공 snapshot이 모두0이며 마지막 성공 양수 로그는iteration16037이다.
관측된 Terminal은 주로 실패−100의 기여였고 **성공500이 로그를 지배했다는 근거는 없다**.
실패50은 반복되는 큰 실패지급을 완화하되 즉시 실패의 비용을 남기는 초기실험값이다.
표의 평균은 native reset-call/rollout집계다. Episode빈도나 실제정책을 재실행한 return이
아니며, penalty를 낮추면 조기실패도 더 싸진다는 tradeoff가 있다.

성공값은 현재 R/M penalty형태의 보상상한과 할인을 함께 고려했다.
Non-terminal step보상은 Clearance최대1.8에서 비양수R/M·Time을 더하므로1.8이하다.

$$
\sum_{k=0}^{\infty}\gamma^k r_{\mathrm{nonterminal},k}
\leq \frac{1.8}{1-0.99}=180 < R_{\mathrm{success}}=200.
$$

따라서 **이미 성공할 수 있는 상태에서 단순히 dense보상을 더 받으려고 성공을 미루는 것**은
이 상한 아래에서 즉시 성공200을 이기지 못한다. 성공100처럼180보다작은 값은 이 성질이
없다. 이는 return수준의 비교이며, 탐색·critic오차·도달가능성·성공학습을 보장하지 않는다.
Terminal500의 불필요하게 큰 크기를 줄이면서 이 기준은 유지했다.

첫 reward를index0으로두고 terminal만 할인하면 다음과 같다.

| 성공 시점 | 기존500 | 변경200 |
|---|---:|---:|
| 2초, index99 | 184.86 | 73.95 |
| 4초, index199 | 67.67 | 27.07 |
| 8초, index399 | 9.07 | 3.63 |

늦은 성공은 여전히 크게 할인된다. 이 조정만으로 접근·통로확보·후퇴 문제를 해결했다고
판단하지 않는다. T06는 양의R/M·tanh정책이므로 현재의 R/M penalty·Gaussian에 대해
같은trajectory의 return을 정확히 재구성한 것은 아니다. 새학습에서 Clearance진행·성공·
실패원인과 episode길이를 함께 비교한다. 기존checkpoint resume은 부모run과 스케일변화를
구분해서 기록해야 하며, 이번에 새학습·장기평가는 실행하지 않았다.

분석값·집계한계는 [analysis.json](../outputs/contact_manipulation/terminal_scale_20261019/analysis.json),
회귀검사 로그는 [tests.log](../outputs/contact_manipulation/terminal_scale_20261019/tests.log)에 둔다.
MDP회귀검사는 **162 passed, 13 skipped, 125 subtests passed**였다. 변경된 기본설정으로
성공안정조건·실패우선·일회지급과 native dt상쇄, 중복실패원인의일회감점을 확인했다.
추가 production파일·sensor조회·physics연산은 없다.

## 10. 2026-10-20 — 추가 학습·reward 조합 비교와 실제 정책 평가

사용자가 reward 조합을 바꿔 짧게 학습해도 Clearance가 일정 이상 증가하지 않는다고 보고했다.
저장된설정·TensorBoard전체·checkpoint를 확인하고, 학습을 하지 않는 고정조건 재생을 추가했다.
이 절의 짧은실행은 사용자가 명시한 **reward 조합 실험**이며 본 학습6개의 성능이력과 구분한다.

### 비교 대상과 통제 조건

C=Clearance, R=Rearrangement, A=action-rate, V=joint-velocity, F/T=힘/모멘트증가 penalty다.
M은 A/V/F/T 네 항을 뜻한다. R/M은 기존 exp−1이며 활성 항의 weight는 동일하다.

| ID | 10월19일 run | 활성 reward | 기록 iteration | 최신 checkpoint |
|---|---|---|---:|---:|
| A | `15-43-41` | C+R+M+Time+Terminal | 11,567 | 11,550 |
| B | `22-14-38` | C+A+V+Time+Terminal | 789 | 750 |
| C | `22-42-06` | C+A+V | 769 | 750 |
| D | `23-08-15` | C+A+V+F+T | 754 | 750 |
| E | `23-33-48` | C+R+A+V+F+T | 1,312 | 1,300 |

`params/env.yaml`·`agent.yaml` 전체를 재귀비교하면 log_dir와활성reward외의 설정은 같다.
모두 seed0·새학습·1,024환경·100/50Hz·8초공통장면·Gaussian초기std0.3·actionclip없음·
rollout32·PPO4epoch/4minibatch·lr3e−4 fixed·γ=.99·entropy coefficient .01이다.
Git snapshot은동일commit767a016이며 events.py의저장diff도 같다. 모든선택checkpoint의
모델·행동·F/T·분포metadata가 같다. model_0은 첫PPO update후 저장본으로 무학습모델이 아니다.

순수Clearance-only 실험은없다. 모든조합에 A/V가 있고, Terminal을뺀 C/D/E도 성공·실패
**종료조건은 유지**한다. 같은seed 한 번씩이며 독립seed 반복실험이 아니므로 인과의크기와
일반적인 조합우열을 확정하지 않는다. 750iteration은 약60회의 공통장면교체다.

### 학습 로그: 공통754 iteration으로 비교

다섯실행의 공통구간0–753중 마지막200회(554–753)를 비교했다.

| ID | native C로그 평균 | Train 길이(step) 평균 | 길이보정proxy | Gaussian mean std | hard-load 종료snapshot |
|---|---:|---:|---:|---:|---:|
| A | 10.890 | 100.10 | 0.4835 | 0.6448 | 0.6943 |
| B | 13.947 | 125.50 | 0.4939 | 0.6512 | 0.5322 |
| C | 11.481 | 101.62 | 0.5021 | 0.5756 | 0.6685 |
| D | 12.456 | 113.37 | 0.4883 | 0.5751 | 0.5997 |
| E | 9.669 | 86.76 | 0.4953 | 0.5806 | 0.7377 |

Native C로그는episode누적값/8초로 길이에 비례한다. 표의proxy는
$8\,\overline{C_{\mathrm{log}}}/(1.8\,\overline{L_{\mathrm{Train}}})$다.
서로다른모집단을 나눈 값이므로 실제step평균Clearance나 가림비율이 아니다.
Iteration별ratio는 물리적으로 가능한품질범위마저 벗어나므로 개별값을품질로 사용하지 않는다.
집계proxy가0.484–0.502에모이고 시간에따른변화가유사하다는것까지만 로그에서 말할수있다.
0.5를 "가림50%"나 수식의상한으로 해석하지 않는다.

![동일 학습길이 비교](../outputs/contact_manipulation/reward_ablation_20261020/matched_budget.png)

방문한상태에서 A의 R평균−0.00544·F평균−0.00346은 C평균10.890보다훨씬작다.
R/F가 현재로그의Clearance기여를 압도하지는 않는다. 다만 물체를안움직여서 비용이작을수
있으므로, 잠재적인 성공동작에서도 이 비용이작다는결론은 아니다. 리워드값의단순합이나
서로다른조합의 Train return크기로 정책성능을비교하지 않는다.

### 실제 재생: 같은 장면·시작 자세에서 진전이 있는가

모든모델의 `model_750.pt`를 새seed20261020의 동일한4개장면에서 비교했다.
각장면8개검증된초기자세를모든정책에복제하고, 초기관절·관측·카메라의일치를assert했다.
각모델은 deterministic mean과 stochastic sample을각각실행했다. Stochastic은 같은표준정규
noise를공유하되 모델별std를사용했다. 각행의첫episode만집계하며 native auto-reset **이전**
GT품질·교차면적·종료·행동·토크를기록했다. 반환reward는비교하지않아원래reward조합은재지급하지않았다.

총320episode(각모델·mode32개)의초기 raw품질평균은 모두 **0.462601**이다.
아래는 deterministic평가다. 각값은episode별집계후동일가중평균이다.

| ID | 최종 raw C품질 | 초기대비최종차이 | 생존step | 과부하/범위초과 실패 | 통로확보 / 성공 |
|---|---:|---:|---:|---:|---:|
| A | 0.461343 | −0.001258 | 102.4 | 32/32 | 0/32 / 0/32 |
| B | 0.462546 | −0.000055 | 324.3 | 9/32 | 0/32 / 0/32 |
| C | 0.461353 | −0.001248 | 86.8 | 32/32 | 0/32 / 0/32 |
| D | 0.461335 | −0.001265 | 144.4 | 28/32 | 0/32 / 0/32 |
| E | 0.461385 | −0.001215 | 65.8 | 32/32 | 0/32 / 0/32 |

Stochastic도 최종품질0.461344–0.462673으로 초기값과거의같았다. 전체320episode에서
**통로면적gate 도달0, 실제성공0, 실패278(86.875%)**였고 실패는모두
`hard_load_or_overrange`였다. Shelf변위·물체낙하/기울어짐·이탈·들림·invalid실패는0이었다.
17개tactile중접촉을관측한episode는8/320(2.5%)다. 이는특정Blocker접촉계측이아니며,
다른robot부위접촉·감지면외접촉까지 없었다는뜻은아니다.

실패마지막frame을제외한 best품질증가도 A/C/D/E평균은약0.00002다. B는평균0.0012–0.0013,
최대약0.04의드문증가가있었지만 해당사례의Target도2.84–3.07cm이동했고 과부하로실패했다.
단순순간 q상승을 목표를보호한 통로확보로 해석하지 않는다. 이 "실패frame제외"도 Target회전과
전체성공조건을모두검증한 안전성지표는 아니다.

![고정조건 Clearance 비교](../outputs/contact_manipulation/reward_ablation_20261020/paired_evaluation/clearance_paired.png)

조합에따라 행동은다르다. Deterministic 위치증분명령 norm은 평균0.205–0.463m,
회전증분 norm은0.260–2.101rad였다. 이는20ms마다생성하는 **목표pose증분**이며 실제이동이아니다.
예를들어 A와B는 생존102.4→324.3step, 위치명령0.463→0.273m, 회전명령2.101→0.260rad로
달랐다. Reward가전혀영향없다는결론은맞지않고, 차이가작업진전으로연결되지않는상태다.
현재실패항은 보정F/T 과부하와raw센서범위초과를합쳐검사하므로 그중무엇이먼저발생했는지는
이자료만으로분리하지못한다. 큰명령·토크포화·비접촉관성과 센서보상오차 중원인을단정하지않는다.

### 장기학습: plateau에 더해 정책 출력의 발산

A의마지막200iteration은 mean std **5,607.49**, 길이 **7.51step(0.150초)**,
hard-load snapshot **0.9845**, C로그 **0.7003**, Train return **−44.13**이었다.
Mean std는iteration924에서1,2335에서10,3916에서100,6481에서1000을넘었다.

`model_11550`의 실제Gaussian std평균은 위치 **0.542**, 회전 **7,677**,
강성 **4,676**, hand **8,748**이다. 위치의std까지수천이라는뜻은아니다.
모든actor·optimizer tensor는finite이므로 NaN검사만으로는탐지되지않는다.
E도iteration981에서meanstd1을넘었고 `model_1300`의meanstd는1.752였다.

추가로 A750/A11550/E750/E1300을같은조건에서 deterministic으로비교했다(각32episode).
이번세트는세트내시작조건을맞췄으며 앞80env세트와시작pose까지같다고가정하지않는다.
A11550도성공0·과부하실패32/32·생존19.28step이었다. Noise를빼도 위치목표증분norm평균
**12.14m**, 회전목표증분norm평균 **3,042.87rad**를출력했다. 첫action부터4개장면의
8start평균위치증분19.61–21.73m였다. **탐색std만의문제가아니라 평균정책도악화**됐다.
E1300의최종품질은0.46623으로소폭증가했지만 통로확보·성공0, 실패30/32였다.

Code상Gaussian entropy에는상한이없고 loss에 $-0.01H$가있다. Hand raw±1과stiffness
raw×100의50–200clamp로실제명령이포화돼도raw분포entropy는증가할수있다.
큰raw변화에서 action-rate exp penalty도−0.15에포화해차이를구별하기어렵다.
이는관측된std증가와일치하는기전후보다. Entropy만이원인이라고증명한대조실험은아니다.
모델은raw직전action도이력으로받으므로 큰분포가network입력규모에도영향줄수있지만,
평균정책악화가그경로때문인지는별도분리검사가필요하다.

### 실험의 해석과 다음 확인 순서

1. 현재관찰은 "수식이0.5에서포화"가아니라 **현정책이물체를유효하게조작하지못하는것**이다.
   $q_C=\exp(-\rho)$는 $\rho\in[0,1]$에서0.368–1이고0.5가상한이아니다.
2. 고정조건에서먼저과부하/범위초과로종료되는원인과큰pose명령을분리한다.
   보정전/후F/T·초과축·실패직전action/목표추종/torque포화·실제접촉을함께봐야한다.
3. Rewardweight추가sweep전 Gaussian std와평균명령규모가안정된조건을잡고같은평가를반복한다.
   Entropy항을고정/제거한대조와같은 한변수실험이원인분리에맞다. Tanh재도입을확정하지않는다.
4. C/D의최소step보상은각각 $1.8/e-0.3\simeq0.3622$, $1.8/e-0.6\simeq0.0622$로양수다.
   성공종료는남기고Terminal을삭제했으므로 성공완료유인도함께제거했다. 이둘의정체만으로
   R/F/T가무관하거나Clearance학습이불가능하다고결론내리지않는다.
5. 다음비교는실제 $\Delta q_C$, 겹침면적감소, Target보호·안정화별gate, episode별실제종료수를
   사용한다. 같은seed·4개장면의이번결과를환경전체의성공률이나seed평균의인과효과로확대하지않는다.

다섯run의reset fallback_total/fraction은전구간0, search_rounds평균1.006–1.015였다.
이번정체가 fallback 반복이나reset실패때문이라는증거는없다. 이번검사에서CUDA719는발생하지
않았지만 장기안정성을보장하지않는다. 현재working RewardsCfg는마지막실험E대로Time/Terminal이
주석처리돼있다. 분석중원래설정으로복구하거나production코드를수정하지않았다.

### 근거 파일과 재현

자료는 [reward_ablation_20261020](../outputs/contact_manipulation/reward_ablation_20261020/)에둔다.
`analyze.py`는5run설정·전체scalar를다시읽어 `metrics.json`, `scalars.csv`와학습추이그림을만든다.
`checkpoints.json`은실제CPUcheckpoint audit, `gaussian_bounds.json`은분포의수학적포화확률하한이다.
후자는실제trajectory에서측정한비율과구분한다.

`evaluate.py --headless --device cuda:0`은공통checkpoint750의고정조건평가를수행하고
`paired_evaluation/{evaluation.json,episodes.csv,trajectories.npz}`에auto-reset전데이터를저장한다.
`latest_evaluation/`은위에명시한4checkpoint의추가평가다.
`summarize_evaluation.py --directory <평가폴더>`로표·요약JSON·그림을재생성한다.
전체실행log는 `analysis.log`, `evaluation.log`, `latest_evaluation.log`다.

## 11. 2026-10-20 — OSC action을 현재 EEF frame으로 변경

이 절은 손바닥 EEF를 사용했던 중간 단계의 기록이다. 현재 기준은 13절의 `wrist_3_link`다.

사용자 요청에 따라 공식 `OperationalSpaceControllerActionCfg`에
`task_frame_rel_path="Robot/ROASRightHand/right_plam_force_sensor"`를 추가했다.
제어 TCP와 물체 observation의 EEF를 일치시킨다. 별도 action/controller는 만들지 않는다.
원격 min 최신 커밋은 확인 시점 `baa3740`(2026-10-14)이며, 해당 설정은 task frame 미지정으로
base 기준 증분이다. EEF observation을 사용하는 min 변형도 action은 base 기준이었다.

정책 step 시작 시 EEF pose를 $(p_t,R_t)$로 읽어
$p^*=p_t+R_t\Delta p_{\mathrm{EEF}}$,
$R^*=R_t\operatorname{Exp}(\Delta r_{\mathrm{EEF}})$를 만든다.
위치·회전 증분과 variable Kp 6축은 EEF 축으로 해석하고, native OSC가 base로 변환한다.
목표와 gain 방향은 다음 정책 명령 전까지 고정된다. 매 physics substep에 목표를 누적하지 않는다.
Position/orientation scale 1.0, stiffness scale 100·범위 50–200, arm IdealPD·hand 제어,
observation 244D·action 18D·reward 설정은 유지했다. Time/Terminal 비활성 실험 설정도 유지했다.

물리적 action 의미가 달라져 checkpoint action contract를
`isaaclab_eef_osc_variable_kp_joint_position_to_limits_v2`로 갱신했다.
이전 base-frame checkpoint는 actor·critic·optimizer를 읽기 전에 거절하며 새 학습이 필요하다.
10절의 평가와 기존 checkpoint는 이전 base-frame 제어에서 얻은 결과로 보존한다.
이번 변경만으로 큰 action·과부하·학습 정체가 해결됐다고 판단하지 않는다.

검증은 실제 CUDA/PhysX 4개 환경·16 policy step과 부분 reset으로 수행했다.
모든 검사 시점에서 native task frame과 실제 EEF pose가 일치했고 위치목표 오차는
float32에서 0이었다. 회전·비등방성 gain 변환과 physics substep 간 목표 유지도 확인했다.
부분 reset은 나머지 두 환경의 관절값을 변경하지 않았으며 종료·비유한 값은 없었다.
`NativeControllerTests` 6개가 통과했다. Learning 검사는 57개 통과·2개 건너뜀이며,
이전 base-frame 규약을 weight·optimizer 로드 전에 거절하는 회귀 검사를 포함한다.
MDP CPU 검사는 160개 통과·13개 건너뜀·기존 실패 1개다. 실패한
`test_native_reward_config_logs_four_motion_terms_without_duplicate_aggregate`는
Time/Terminal이 활성이라고 가정하나 현재 HEAD의 실험 설정은 비활성이다.
해당 테스트와 reward 설정 모두 이번 변경 전 HEAD와 동일하며 그대로 보존했다.
단기 제어 검증이며 학습 성공이나 장기 CUDA 안정성 검증으로 확대하지 않는다.
실행 코드·전체 log·수치 결과는
[eef_action_20261020](../outputs/contact_manipulation/eef_action_20261020/)의
`verify.py`, `verify.log`, `results.json`에 남겼다.
10절의 기존 평가를 재현할 때는 당시 base-frame cfg를 사용해야 한다.

## 12. 2026-10-20 — 물체 종료 조건 완화

사용자는 작업 영역 이탈·들림·순간 기울어짐의 즉시 실패가 밀기·pivoting을 과도하게 제한할 수
있음을 지적했다. `workspace_exit`와 `object_lift`를 제거하고, `object_tilt`는 기존 50° 초과
즉시 실패에서 **80° 이상인 상태가 같은 물체에서 0.2초 지속될 때 실패**하도록 변경했다.
80°·0.2초는 사용자가 선택한 초기 판정 기준이며 실제 모든 물체의 완전 전도를 보장하는
물리적 기준으로 검증한 값은 아니다.

기울기 누적은 환경·활성 물체별 tensor로 관리한다. 현재 policy 50 Hz에서는 같은 물체가
10 step 연속 기준을 만족해야 한다. 80° 미만으로 돌아오면 해당 물체 누적값을 0으로 만들고,
부분 reset에서는 해당 환경 행만 초기화한다. 서로 다른 물체의 교대 기울어짐은 합산하지
않는다. 정책 step당 한 번 갱신한 캐시를 native 종료 term과 reward·observation이 공유한다.
80° 이상으로 기울어진 유예 구간에는 성공도 보류한다. 5-step 성공이 10-step 전도 판정보다
먼저 확정되는 것을 막으며, 회복하면 기존 성공 안정화 카운터를 다시 시작한다.

| 항목 | 변경 후 |
|---|---|
| `object_drop` | 유지: 활성 물체 OBB 최저점이 지지면보다 6 cm 초과 낮음 |
| `object_tilt` | 변경: 같은 물체 기울기 ≥80°가 연속 0.2초 유지 |
| `workspace_exit` | 제거 |
| `object_lift` | 제거 |
| F/T·선반 변위·invalid state | 기존 임계값과 종료 유지 |
| 성공·timeout | 기존 면적·Target 보호·안정화·지속 시간 및 공통 8초 경계 유지 |

Native 실패 term은 7개에서 5개가 되고 성공·timeout을 포함한 종료 term은 총 7개다.
들림·허용 작업 영역 이탈 자체는 별도 실패나 성공 gate로 남기지 않는다. 다른 실패가 없고
기존 성공 조건을 충족하면 이 동작을 포함한 조작도 성공할 수 있다. 기존 all-object 안정화
조건과 Target pose 제한이 들림이나 영역 이탈을 대신 금지하는 것은 아니다.

EEF action frame·244D observation·18D action·reward 수식과 weight는 바꾸지 않는다.
현재 실험의 Time/Terminal 비활성도 유지한다. Terminal reward 활성 조합의 실패 판정에는
이 5개 원인의 OR를 사용하므로, 제거한 두 원인에 대한 실패 보상도 발생하지 않는다.
이 변경이 과부하 종료나 Clearance 정체를 해결했다고 판단하지 않으며, 이전 10절의 평가
수치에는 새 종료 조건을 소급 적용하지 않는다.

관련 Termination·State·NativeAdapter 검사 **49/49개 통과**. 전체 MDP CPU 검사는
180개 중 166개 통과·13개 건너뜀·기존 reward 구성 검사 실패 1개다. 이 실패는
Time/Terminal 비활성 실험 설정과 기존 테스트 기대값의 불일치이며 11절과 같은 기존 문제다.
실제 CUDA/PhysX 4개 환경에서 16 policy step과 부분 reset을 통과했고, 다른 환경의 관절값과
전도 카운터를 보존하며 선택된 환경의 카운터만 초기화함을 확인했다. GPU tensor 2,048행으로
80°의 10-step 지속·79.9° 허용·들림/영역 이탈 허용·비활성 NaN 제외를 확인했다.
Native controller 회귀 검사 6개도 통과했다. 학습은 실행하지 않았다.
실행 코드·log·결과는
[fall_termination_20261020](../outputs/contact_manipulation/fall_termination_20261020/)의
`verify.py`, `verify.log`, `results.json`에 남겼다.

## 13. 2026-10-20 — UR5e 말단 제어·관측 통일과 OSC 중력 보상

사용자는 `gravity_compensation=True`를 직접 활성화했고 OSC 제어 EEF와 task frame을
UR5e 말단으로 옮기도록 요청했다. 물체 기하와 robot EEF pose observation도 같은 말단으로
통일하는 방향을 선택했다. 실제 rigid body인 **`wrist_3_link`**를 사용하며 `tool0` 가상 frame이나
hand 링크를 제어 body로 사용하지 않는다.

| 항목 | 현재 설정 |
|---|---|
| Native OSC 제어 body | `body_name="wrist_3_link"`, body offset 없음 |
| Native task frame | `task_frame_rel_path="Robot/wrist_3_link"` |
| 관측 FrameTransformer | target `{ENV_REGEX_NS}/Robot/wrist_3_link`, offset 없음 |
| 물체 observation | Base에 저장한 초기 visible 기하를 매 policy step 현재 wrist frame으로 재표현 |
| Robot observation의 EEF pose | Base에서 표현한 현재 `wrist_3_link` pose 7D |
| OSC 중력 보상 | `gravity_compensation=True` |
| 초기자세 IK용 TCP | 기존 손바닥 `right_plam_force_sensor` 유지 |
| F/T·tactile | 기존 `ToolForceTorqueLink` sensor frame·distal 자중/관성 보상·17개 tactile 유지 |

Wrist의 현재 Base 기준 pose를 $(p_t,R_t)$라 하면 native OSC는
$p^*=p_t+R_t\Delta p_{\mathrm{wrist}}$,
$R^*=R_t\operatorname{Exp}(\Delta r_{\mathrm{wrist}})$를 만든다. Stiffness 6축도 현재 wrist 축이다.
20 ms policy step 시작 시 frame을 읽어 목표를 만들고 두 10 ms physics substep 동안 유지한다.
단순히 원점만 옮긴 것이 아니라 wrist와 hand의 고정 장착 회전 차이도 반영한다.
Position/orientation scale 1.0, stiffness scale 100·범위 50–200, IdealPD의 150/28 N·m 제한,
hand joint-position action은 유지한다.

OSC 중력 보상은 관절 중력 토크를 제어 출력에 feedforward로 더한다. F/T 자중·관성 보상은
센서 측정값에서 distal 링크의 예측 하중을 제거하는 별도 계산이다. 둘을 함께 사용하는 것이
동일 측정값을 두 번 보상하는 것은 아니다. 중력 보상 활성화가 적절한 목표 action이나 학습
성공을 보장하지는 않는다.

Reset IK는 제어 frame과 분리해 손바닥 TCP를 유지한다. Blocker 앞 위치·회전 범위와 elbow-up·
충돌 검사·실패 행 복구 정책을 바꾸지 않아 초기 hand의 배치 의미를 보존한다.
공유 배치·초기 mesh raycast·물체 pool·F/T·tactile·reward 수식과 weight·종료 조건은 유지한다.
12절의 80°·0.2초 전도 판정과 Time/Terminal 비활성 실험 설정도 그대로다.

Action contract는 **`isaaclab_ur5e_wrist_osc_variable_kp_gravity_v3`**로 갱신한다.
Observation 244D·action 18D의 크기는 같아도 제어 body·관측 의미·중력 보상이 달라져
이전 base-frame(v1)·손바닥 EEF(v2) checkpoint는 로드 전에 거절하며 새 학습이 필요하다.
기존 10절 정책 평가는 당시 설정의 결과로 보존하고 이번 변경에 소급 적용하지 않는다.

실제 CUDA/PhysX **4개 환경·16 policy step·부분 reset** 검증을 통과했다.
OSC task frame과 관측 EEF가 현재 wrist pose와 일치했고, 물체 관측 위치를 wrist에서 Base로
다시 변환하면 캐시한 초기 기하를 복원했다. 손바닥은 reset sampler가 반환한 위치를 유지했고,
wrist–palm 거리는 약 **0.17005 m**였다. F/T binding은 기존 `ToolForceTorqueLink`를 유지했다.
초기 정지 목표에서 OSC 출력과 PhysX 중력 보상 토크의 최대 차이는 **9.47e−5 N·m**였다.
Native controller 검사 6개도 통과했다. 학습 성능이나 장기 안정성을 검증한 결과는 아니다.
실행 코드·전체 log·수치 결과는
[wrist_frame_20261020](../outputs/contact_manipulation/wrist_frame_20261020/)의
`verify.py`, `verify.log`, `results.json`에 남겼다. 첫 실행의 검증 스크립트가 초기자세 모델을
잘못된 runtime 속성에서 찾았고, 실제 `env.robot_pose_model`로 교정 후 위 검증을 완료했다.
CPU `pytest -q tests/test_mdp.py tests/test_learning.py`는 **225 passed·18 skipped·
150 subtests passed·기존 실패 1개**였다. 실행에는 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`과
`env_isaaclab` Python을 사용했다. 새 관측 binding 검사는 독립 SciPy oracle과 비교하고,
checkpoint 검사는 v1/v2를 actor·critic·optimizer 로드 전에 거절함을 확인했다.
기존 실패는 Time/Terminal을 활성이라고 가정하는 reward 구성 검사이며 현재 비활성 실험
설정을 그대로 유지한 결과다. 이전 11·12절에서 기록한 동일 불일치다.

## 14. 2026-10-20 — 접근·접촉 보상과 1024환경 비용 검증

사용자는 거리 지원 항과 같은 tactile pad의 직전·현재 Blocker 접촉 유지 보상을 승인했다.
우선 연산량을 비교하고, 접촉 판별이 심한 병목을 만들면 기존 센서 구성으로 복구하도록 했다.
이번 실행은 **고정 명령에 대한 실제 환경 비용·기능 검증이며 본 학습이 아니다.**

### 적용한 보상과 native 연결

| 항 | Weight | Discrete step 지급 |
|---|---:|---|
| `blocker_distance` | 0.10 | `exp(−ln(2)/0.10 × d)`; d는 17개 pad 표면 대표점에서 현재 Blocker OBB까지의 최소 거리(m) |
| `blocker_contact_hold` | 0.01 | 직전·현재 모두 같은 Blocker와 접촉한 pad 수의 합; 최초 접촉0, 최대0.17/step |

대표점은 env_0의 실제 collision mesh 중 노출된 local +Z 표면에서 초기화 때 한 번 구한다.
이후 GPU native 링크 pose로 변환한다. 물체 pose도 현재 privileged 값을 사용하며,
actor의 초기 vision·244D observation·상대 구분 없는 tactile binary17은 그대로다.
OBB 근사 거리와 실제 물리 접촉은 구분한다. 접촉은 Blocker 상대 normal force norm이
0.01 N을 초과하는지 판단하고, 같은 pad의 이전 binary와 AND한다. 접촉 세기·유지 시간 배율·
접촉 pad 수 평균·Clearance gate는 추가하지 않았다. 20 ms 간격 두 sample의 접촉 판정이다.

`TaskContactSensor`는 하나의 공식 PhysX contact view에 환경별 실제 pad/body 경로와
object pool의 filter 목록을 연결한다. 17개 sensor instance로 되돌리지 않는다. `[N,17,10,3]`
pair force에서 실제 collection `blocker_id` 열을 GPU gather한다. 매 reset의 view 재생성은 없다.
`mdp/sensors.py`의 `BlockerContactHold(ManagerTermBase)`가 이전 binary를 관리하며,
**공식 RewardManager.reset**이 해당 환경의 이력·출력만 초기화한다. 같은 step의 중복 조회는
이력을 갱신하지 않는다. 기존 events/TaskState/transition CUDA graph 흐름은 바꾸지 않았다.
새 production 파일은 추가하지 않았다.

기존 C/R/M 수식·weight, UR5e wrist OSC·gravity compensation·IdealPD·물리 주기,
공유 배치·raycast·종료 조건을 유지한다. Time·Terminal은 사용자 설정대로 비활성이다.
새 dense 상한은2.07/step, γ=.99에서207이므로 예전 Terminal200의 근거였던180은
현재 상한으로 사용할 수 없다. 이번에는 Terminal의 재활성화·scale 조정을 하지 않았다.

### 비용 비교

RTX3090, **1,024환경·물체10개·physics100 Hz/policy50 Hz**를 사용했다.
기존 / 거리만 / 거리+Blocker 필터·유지 계산을 별도 프로세스에서 같은 seed·명령·reset
일정으로 비교했다. 초기 생성·바인딩·컴파일 warmup은 제외하고 24-step block 끝에서만
동기화했다. 단계별 진단은 별도 실행에서 동기화하며 headline에 포함하지 않는다.

| 시나리오 | 기존 ms/step | 거리만 ms/step | 거리+판별·유지 ms/step | 전체 증가 |
|---|---:|---:|---:|---:|
| 정지 명령, 1,008 step | 37.435 | 38.467 | 39.145 | 4.57% |
| 작은 접근, 역순 실행, 1,008 step | 36.743 | 37.791 | 38.940 | 5.98% |
| 실제 접촉 접근, 역순 실행, 432 step | 47.492 | 48.572 | 48.035 | 1.14% |

앞의 두 비교는 전체 reset2회·고정 부분 reset13회(각128환경)를 포함하며,
실제 tactile 접촉은 발생하지 않았다. 마지막 비교는 초기 wrist-local 방향으로 Blocker를
향하는15 mm 상대 목표를 사용했다. 기존과 추가 구성 모두 자연 종료365회·전체 timeout
1,024개가 발생했다. 이는 학습 action scale 변경이 아니라 검증용 고정 명령이다.

모든 비교에서 초기 상태·reset event 순서가 일치했다. 최종 상태 hash를 기록한 두 번째·
세 번째 비교에서는 최종 관절·물체 상태도 bit 단위로 일치했다. 마지막 비교의 24-step 끝
sample에서 any-object contact는 최대64 pad, Blocker contact는 최대4 pad였다.
Step168에서는 현재 Blocker 접촉4·유지3, step216에서는 접촉4·유지4,
step312에서는 any-object 접촉64·Blocker 접촉/유지0이었다. 이 수치는 전체 step의
접촉률·누적 접촉 횟수가 아니라 **block 끝 snapshot**이다. 모든 환경이 동시에 접촉한
최악 부하 시험으로 해석하지 않는다.

비접촉 비교에서 native 접촉 조회는 약0.20→0.25–0.28 ms였고, 거리 tensor 계산은 약1 ms였다.
추가 구성의 비용은 24-step 환경 수집 기준 약0.04–0.05초 증가였다. 접촉 비교에서 filtered가
거리만보다 조금 빠른 차이는 실행 변동으로 보고 성능 향상으로 주장하지 않는다.
첫 비교의 peak PyTorch allocation은117.80→122.56 MB, reserved는247.46 MB로 같았다.
이는 PhysX 내부 GPU 메모리를 포함한 전체 VRAM 측정이 아니다.

적용 후 시험용 계산 hook을 끄고 **실제8개 native reward 구성**으로1,024환경·1,008step을
다시 실행했다. 평균 **38.277 ms/step, 24-step 환경 수집0.91865초**였고,
첫 정지 baseline과 초기 상태 hash·전체/부분 reset 순서가 일치했다. 첫 baseline의0.89843초에
비해 약2.25% 증가지만, 별도 시점 단독 재측정이므로 엄밀한 paired 인과 추정치는 아니다.
자연 실패0·전체 timeout2회, peak PyTorch allocation121.63 MB였다.
`native_1024_hold_production.json`에 실제8개 활성 term과 전체 측정을 보존했다.

큰 새 병목은 관측되지 않아 두 보상을 유지했다. 접촉 항을 비활성화하려면
`env.rewards.blocker_contact_hold=null`을 사용한다. 이때 기존 grouped net-force backend와
거리 항은 유지하며, 다른 물체 접촉을 Blocker 보상으로 대신 사용하지 않는다.

**위 시간은 실제 환경 수집 시간이며 PPO actor/critic·update·logging을 포함한 전체 iteration
시간이 아니다.** 접촉한 채 정지하는 양의 지원 보상은 남으므로 이후 본 학습에서 접촉률뿐 아니라
접촉 후 Clearance 증가·성공률을 확인해야 한다. 장기 학습·CUDA719 해결을 검증한 결과는 아니다.

### 기능·회귀 검증

- 실제 CUDA/PhysX 4환경의 분리된 pad/object positive control에서 예상4개 pair만 검출되고,
  기존 view와 새 view의 net force 차이0, filtered 합과 net 차이0이었다. 부분 reset·원복도 확인했다.
- 활성화한 실제 native RewardManager 환경4개·16 policy step에서 두 항의 weight·8개 활성 term·
  부분 reset의 선택 행 이력 초기화·동일 step cache·기존 binary 관측·244D 유지·모드 전환을 확인했다.
- CPU 회귀 검사: **235 passed, 18 skipped, 150 subtests passed**. 이전 Time/Terminal 활성 가정의
  cfg 검사는 현재 비활성 설정과 새 두 활성 항을 명시적으로 검사하도록 갱신했다.

코드·실행 log·전체 timing JSON은
[contact_reward_cost_20261020](../outputs/contact_manipulation/contact_reward_cost_20261020/)에 남겼다.
`benchmark.py --mode all`은 재현을 위해 새 cfg 항을 끄고 세 후보를 비교하고,
`--mode native`는 실제 활성화한 두 native reward를 그대로 측정한다.

## 15. 2026-10-20 — 거리·접촉 항 도입 이후의 학습 결과와 재조정 검토

사용자는 거리·접촉 항을 추가한 뒤 Clearance와 success가 개선되는 반면 hard-load 종료가
학습을 너무 일찍 끊고, 거리 항의 상대 비중이 작다고 관찰했다. 저장된 설정·TensorBoard·
checkpoint와 `examples/`를 분석했다. **이번에는 production weight·계수·종료 설정을
수정하지 않았으며 새 학습이나 정책 재생도 실행하지 않았다.**

### 비교 가능한 실행과 로그 해석

10월20일 새7개 run 중 `01-17-28`, `01-32-11`, `01-34-34`, `01-44-35`는
4환경·22–38 iteration 검증 실행이므로 학습 비교에서 제외했다.

| Run | 마지막 iteration | 보상 | Native hard-load 종료 |
|---|---:|---|---|
| `01-54-22` | 421 | C/R/M | 활성 |
| `02-49-46` | 755 | C/R/M + 거리0.10 + 유지0.01 | 활성 |
| `03-16-36` | 3250 | 위와 동일 | 비활성 |

모두1,024환경·seed0·새 학습·wrist EEF/task frame·OSC 중력 보상·32-step rollout이며 agent
YAML은 같다. 첫째→둘째는 새 두 reward만, 둘째→셋째는 native hard-load 종료만 다르다
(각 log_dir 제외). 단일 seed이고 서로 다른 종료 이후 scene/robot 샘플이 달라지므로
완전히 paired인 episode 비교나 독립 seed 반복 실험은 아니다. 거리와 유지는 함께 켰으므로
각 항의 독립 효과도 분리할 수 없다.

공통 **321–420 iteration**의 평균은 다음과 같다.

| 지표 | C/R/M | +거리·유지, hard ON | +거리·유지, hard OFF |
|---|---:|---:|---:|
| Native Clearance 로그 | 11.440 | 13.048 | 24.092 |
| Train episode 길이(step) | 112.34 | 125.72 | 298.71 |
| 거리 / Clearance 지급 비율 | — | 1.007% | 0.893% |
| 유지 / Clearance 지급 비율 | — | 0.0000169% | 0.00000424% |
| hard-load 종료 snapshot | 64.83% | 49.73% | — |
| success snapshot | 0 | 0 | 0 |

`Episode_Reward/*`는 reset 그룹별 weighted 누적 지급을 최대 episode8초로 나눈 값을
RSL이 iteration의32개 step extras에서 평균한다. `Train/mean_episode_length`는 최근
완료100episode 버퍼이므로 두 통계를 나누어 실제 step 평균 품질을 복원할 수 없다.
그 방식은 일부 구간에서 Clearance의 이론적 하한 exp(−1)까지 위반했다. 따라서 여기서는
길이 보정 품질 proxy를 보고하지 않는다. **Clearance 로그가 두 배라고 가림 개선이 두 배인
것은 아니다.** 같은 집계의 D/C·H/C 지급 비율은 기여 크기 비교에 사용할 수 있다.

### 초반 성공 신호와 후반 약화

Hard ON의 두 학습은 전체 success scalar가0이었다. OFF 실행은190–2612 iteration 사이
**86개 iteration 로그에서 success가 양수**, 최대는1124의0.00552368(0.5524%)였다.
이는 일부 episode에서 실제 성공 판정이 발생했다는 근거다. 다만 `_last_episode_dones`의
환경 평균을 다시 iteration 평균한 **최근 종료 원인 snapshot**이며 누적 episode 성공률이 아니다.
따라서 “성공률0.55%”나 “86회 성공”으로 표현하지 않는다. 마지막100 iteration은 다시0이다.

최신 OFF 실행의 최초/최종100 iteration 비교:

| 지표 | 최초100 | 최종100 |
|---|---:|---:|
| Clearance 로그 | 27.016 | 27.703 |
| Episode 길이(step) | 303.52 | 373.23 |
| 거리 로그 | 0.60377 | 0.24939 |
| 거리 / Clearance | 2.235% | 0.900% |
| 유지 로그 | 8.2925e−5 | 3.9215e−7 |
| 유지 / Clearance | 0.0003069% | 0.00000142% |
| Action-rate 비용 로그 | −0.06145 | −4.89329 |
| Gaussian 평균 std | 0.3059 | 48.6417 |
| Timeout snapshot | 85.63% | 99.08% |

거리 신호는 상대적으로 작고, 같은 pad의 유지 보상은 사실상 드물다. 단, 접촉 유지가 거의
없다는 사실을 weight만 낮아서라고 단정하지 않는다. Blocker와 접촉한 채 다음 sample까지
유지되는 기회 자체가 드물 수 있다. 마지막 구간은 episode가 길어졌는데도 거리·유지 지급이
감소하고 success가 사라졌다. “접촉을 잘하는 방향으로 계속 개선 중”이라는 해석은 지지하지 않는다.
평균 실제 거리·접촉률·Clearance 개선량은 현재 scalar만으로 복원할 수 없어 추가 계측이 필요하다.

![새 학습 추이](../outputs/contact_manipulation/reward_followup_20261020/metrics_trends_full.png)

### Hard 종료 해제의 내부 불일치

현재 hard-load는 보상된 force norm>80 N, 보상된 torque norm>4 N·m, **보상 전 raw
SI-200-8 어느 축이든 범위 초과**의 OR이며,20 ms sample 한 번으로 발동한다. 세 원인을
분리한 로그는 없어서 실제로 무엇이 지배했는지는 확정하지 못한다. 기존80/4가 물리적으로
잘못된 한계라는 결론과, 지금 정책에서 너무 자주 학습을 끊는다는 관찰은 구분해야 한다.

`03-16-36`의 저장 git diff와 현재 코드는 native `TerminationTermCfg`만 제거했다.
`task_conditions`는 여전히 hard를 `failed=reasons.any(-1)`에 포함하고 success 후보에서
`~failed`를 요구한다. 따라서 **즉시 reset은 없어져도 과부하 동안 success stability는0으로
돌아가며 성공이 차단**된다. 이후 하중이 낮아지고 나머지 조건을 만족하면 다시 성공할 수 있다.
또 `extras['terminated']=success|internal_failure`를 사용하므로 공통8초 timeout 순간에는
native truncation인데도 PPO value bootstrap이 차단될 수 있다. 일반 step에서 false native
done이 생긴다는 뜻은 아니다. 현재 Terminal은 비활성이지만, 그대로 재활성화하면 transient
hard가 −50과 `terminal_paid`를 발생시켜 이후 성공 보너스까지 막을 수 있다.

CPU 순수 판정 재현에서 그 외 모든 성공 조건이 충족된 장면에 force81 N, force[60,60,0] N,
torque4.01 N·m 또는 **보상 wrench0+raw overrange**를 주면 내부 failure와 성공 차단을 확인했다.
Raw를 SI범위로 먼저 clipping한 뒤 자중·관성 보상하므로 raw 포화와 외력 허용 기준은 별개다.
포화된 측정에서 정확한 외력을 복원할 수 없으며 빠른 비접촉 motion도 raw 한계를 넘을 수 있다.
현재 로그는 이 경로가 실제 주원인이었다고 입증하지 않는다.

우선 수정은 **진단용 load flag와 실제 활성 failure gate의 분리**다. Native done·success·
Terminal·timeout bootstrap이 같은 활성 원인 mask를 쓰게 하고, force/torque norm 초과와
raw축 saturation은 각각 계속 기록한다. 추가 contact sensor나 raycast는 필요 없다.
그다음 episode 경과 시간·접촉 여부·초과 지속 시간을 보고 단발 외력 spike의 grace/persistence와
raw 센서 saturation 처리를 각각 정해야 한다. 현재 사용자 hard 종료 OFF 설정은 유지한다.

### 후반 Gaussian 분산 증가

최신 `model_3250.pt`의 실제18축 std를 분리하면 다음과 같다. 평균48이라는 이유로
위치 std도48 m라고 해석하면 안 된다. 이는 sample 분포의 std이며 정책 mean 크기를 측정한 것은 아니다.

| 채널 | 평균 std | 범위 |
|---|---:|---:|
| 상대 위치3축 | 0.3734 m | 0.3522–0.4002 |
| 상대 회전벡터3축 | 63.26 rad | 52.04–78.29 |
| Raw stiffness6축 | 54.19 | 34.31–77.38 |
| Raw hand6축 | 74.93 | 71.33–79.43 |

Kp가50–200으로 제한되고 raw scaling이100이므로 유효 raw 구간은[0.5,2]다. Gaussian의
최대 밀도로 interval 확률을 상계하면 **mean에 관계없이 Kp각축 포화 확률이 최소98.26%**,
hand[-1,1] 포화는 최소98.88%다. 실제 rollout에서 측정한 포화 빈도는 아니지만 저장 std만으로
성립하는 분포 확률 하한이다. 이 채널들의 std를 키워도 유효 명령 다양성은 제한되는 반면
Gaussian entropy는 커진다. Entropy coefficient는0.01, raw action clip은 없다.

Action-rate는 raw18축 차이 제곱합에 exp를 적용하므로 이런 큰 값에서 비용이 −0.15에
포화하기 쉽다. 큰 분산을 더 키우는 것과 작은 분산으로 줄이는 것을 구별하는 신호가 약해질 수
있다. 최종 Action-rate 비용은 거리 지급의 약19.6배다. 이는 분산 증가를 보상 weight만으로
해결할 수 있다는 증거가 아니며, entropy·유효 action 범위·std 관리에 대한 별도 실험이 필요하다.
모델 mean 진단/새 제한은 이번에 적용하지 않았다. 계산·중간 checkpoint std는
`reward_sensitivity.py/json`에 보존했다.

### Weight와 exp 계수 검토

우리 거리 항은 `0.1*exp(−6.931*d)`이다. Sweeping 기본/random 예제의 실제 식은
**`exp(−10*distance_ee)`**, 선형 거리(m)이며 reaching/main weight는2/6 또는3/6이다.
다만 예제는 manager의 dt=.02를 그대로 곱하고, 우리는 `/step_dt`로 상쇄하므로 raw
weight2/3을 우리0.1과 직접 비교하면 안 된다. 예제 거리도 offset EEF 점→목표 접근점이고,
우리 거리와 정의가 다르다. 우리 프로젝트 내 D/C weight 비율1/18은 예제1/3–1/2보다 작다.

| Pad–Blocker 거리 | 현재 w=.1,a=6.931 | w=.4,a=6.931 | w=.4,a=10 |
|---|---:|---:|---:|
| 5 cm | 0.07071 | 0.28284 | 0.24261 |
| 10 cm | 0.05000 | 0.20000 | 0.14715 |
| 20 cm | 0.02500 | 0.10000 | 0.05413 |
| 30 cm | 0.01250 | 0.05000 | 0.01991 |

모두 실제 지급step값이다. $r_D=w_D\exp(-a_Dd)$에서
$|\partial r_D/\partial d|=w_Da_D\exp(-a_Dd)$다. Weight는 전체 크기·기울기를
비례 확대하고, a는 강조할 거리 구간을 바꾼다. 고정 d>0에서 기울기를 최대화하는 a는1/d다.
예제의 a=10은 반감 거리6.93 cm이고 현재는10 cm다. **a만 올리면 먼 거리에서는 접근 신호가
약해질 수 있다.** 초기 robot range0.12–0.40 m는 palm 기준이며 실제 가장 가까운 pad 거리와
동일하지 않으므로 실제 d의 quantile을 기록한 뒤5/6.931/10을 비교해야 한다.

현재10→9 cm 접근의 step 지급 증가는0.003589이고 w=.4,a=6.931이면0.014355,
w=.4,a=10이면0.015476이다. 비교로 물체 한 개의1 cm 조작 비용은−0.011881,
Clearance overlap fraction0.5→0.4 개선은+0.114821이다. 조작비용과 접근지원의 상대 크기를
검토하는 수치이지, 한 action이 반드시 그만큼 변화시킨다는 예측은 아니다.

![거리 weight·계수 후보](../outputs/contact_manipulation/reward_followup_20261020/distance_sensitivity.png)

권장 최초 대조는 **C1.8/D0.4/H0.01/R0.6/M각0.15, a_D=6.931·a_C=1 유지**다.
한 번에 모든 항을 변경하지 않고 D weight만4배로 만든다. 그다음 같은 weight에서 a_D5/10을
분리 비교한다. Hold는 발생 기회가 거의 없는 상태라 weight 확대만으로 해결하려 하지 않는다.
R·F/T cost는 방문한 상태에서 주된 지배항이 아니고 Action-rate 증가는 분산 문제와 얽혀 있으므로
현재 단계에서 전부 낮추는 근거도 부족하다. 성공 조작이 생긴 뒤 해당 궤적의 비용을 다시 검토한다.
이는 진단용 후보이며 최적값으로 결정한 것도, production에 적용한 것도 아니다.

Time·Terminal 비활성 상태에서 C/D/H를 매 step 양수로 지급하면 접촉 후 정지/성공 직전 대기가
계속 이득일 수 있다. 성공은 즉시 reset되므로 종료 보상 없이 접근만 강화해도 완료를 계속
학습한다는 보장은 없다. 본 학습 설정을 정하기 전에 완료 인센티브를 별도 대조해야 한다.
이때 D를0.4로 올리면 dense 상한은2.37/step, γ=.99의 보수적 상한은237이므로 기존
Terminal200의 과거180 근거를 다시 사용하지 않는다. 이 상한만으로 새 terminal scale을
자동 결정하지도 않는다.

근거 코드: [예제 거리 식](../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/mdp/reward_sweep.py),
[예제 cfg](../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/shelf_sweep_env_cfg.py),
현재 `mdp/terminations.py`, `mdp/events.py`, `agents/ppo.py`와 저장된 각 run의 `params/`, `git/`다.
원자료·재현 script·공통 학습길이 그래프·설정 diff·predicate 재현은
[reward_followup_20261020](../outputs/contact_manipulation/reward_followup_20261020/)에 남겼다.

## 16. 2026-10-20 — 성공 판정 분리·거리 weight 적용과 PhysX 중단 조사

사용자가15절의 후보 weight를 승인하고, 성공은 Target을 유지하면서 지정 Blocker를 치운
상태만 평가하도록 요청했다. 이번 변경은 새 학습 결과가 아니라 코드 교정과 단기 검증이다.

### 적용한 MDP 변경

- Success: 지정 Blocker와 A₀의 교차 면적≤1e−5 m², 초기 Target 대비 이동≤2 cm·회전≤0.15 rad를
  기존처럼5개 policy sample(0.1초) 연속 만족하면 참이다. 유효한 Target이 하나 있어야 한다.
  전체 물체의 속도, 전도 유예, F/T, 선반 상태, 다른 실패 원인을 성공 predicate에서 제거했다.
- 활성 실패는 별도로 종료한다. 같은 step에 성공과 활성 실패가 모두 참이면 native 로그에
  둘 다 기록한다. Terminal을 켰을 때의 지급은 기존처럼 실패 우선이며 episode당1회다.
- Native TerminationsCfg에서 활성화한 실패만 내부 aggregate failure, Terminal 지급,
  PPO `extras['terminated']`에 반영한다. 설정은 초기화 때 한 번 읽고 GPU boolean mask로
  캐시한다. 현재 꺼진 hard-load는 진단 열에 남지만 성공·terminal/bootstrap을 막지 않는다.
- Reward weight: **C1.8 / D0.4 / H0.01 / R0.6 / Motion4항 각각0.15**.
  실제 수치 변경은 D0.1→0.4뿐이다. `a_D=ln(2)/0.10`, `a_C=1`은 유지했다.
  Time·Terminal·native hard-load의 비활성 상태도 유지했다. Dense 상한은2.37/step,
  γ=.99의 보수적 할인합 상한은237이며 Terminal scale을 자동 변경하지 않았다.

관측244D·action18D·controller·충돌 형상·PhysX capacity는 이번에 변경하지 않았다.
성공 속도 검사2개를 제거했고, 활성 실패 mask는 기존5열 boolean 연산에 적용하므로
환경별 Python loop나 CPU scalar read를 추가하지 않는다.

### 첨부 오류에서 확인한 사실

첨부626줄의 최초 표시 오류는 UTC2026-10-19 19:51:55의
`artiSolveInternalTendonAndMimicJointConstraints1T fail to launch kernel`이다.
약237 ms 동안 다른 articulation/solver/stream 실패가 이어진 뒤 CUDA719와 DOF 조회 예외로
끝난다. 마지막 `Failed to get DOF velocities`는 이미 실패한 CUDA context에서의 후속 오류다.

같은 시각 kernel journal에는 **PID1432859의 NVRM Xid31,
`FAULT_PDE ACCESS_TYPE_VIRT_WRITE`**가 있다. GPU 가상 메모리 쓰기 접근 오류의 직접 증거다.
TensorBoard 파일 PID도1432859이고, `2026-10-20_03-16-36`의 마지막 scalar는3250iteration,
UTC19:51:54.796283이다. 첨부 오류와 이 실행의 연결은 강하지만, 최초 오류시각의 초 단위
정밀도를 넘어서 정확한 지연을 계산하지 않는다.

Xid31은 application·driver·hardware 원인 모두 가능하다. 커널 기록의 `ENGINE GRAPHICS`는
camera가 원인이라는 증거가 아니며, GPU가 단순히 바쁜 상황과도 구분한다.
[NVIDIA Xid 문서](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html).
첨부 구간에는 contact/patch overflow, allocation 실패, OOM 경고가 없다. 첨부 이전의 경고까지
없었다는 뜻은 아니다. PhysX는 일부 고정 버퍼 부족 시 경고와 contact/constraint discard를
문서화한다. **이번 로그만으로 충돌량 과다·contact capacity 초과를 확정할 수 없다.**
[PhysX GPU 메모리 문서](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/GPURigidBodies.html).

### 실제 충돌 형상과 1,024환경 native 통계

현재 stage와 PhysX cooked collision representation을 직접 조사했다. 시각 메시의 원본
vertex/triangle 수와 실제 solver의 충돌 형상 수를 구분한다.

| 환경당 구성 | 실제 collision shape 수 |
|---|---:|
| UR5e+hand+센서 마운트 | 46 |
| 선반 | 17 |
| 물체 pool10개 | 45 |
| 물체 중 Mustard 한 개 | 위45개 중32 |

Hand의 과거5링크×128 hull 코드는 돌아오지 않았다. 현재 해당5개 parent collider는
각2개 convex proxy다. 전체1,024환경은 convex91,136개·box19,456개·공유 plane1개이며
triangle-mesh collider는0개다. Mustard의32 hull은 단순화 후보지만 이번 fault의 원인으로
확인된 것은 아니다. 이번 조사에서는 충돌 형상을 바꾸지 않았다.

1차 검사는 hold·약한 접근·최신model3250 Gaussian 재생을 각120 policy step, 합360step
수행했다. `PhysicsSceneStats`를24step마다 읽은 **sample 최대값**이며 장기 학습의 진짜
순간 peak가 아니다. 특히 이 검사에서는 손 접촉이 충분하지 않아 별도 접촉 검사를 추가했다.

| 항목 | sample 최대값 | 설정 용량 | 비율 |
|---|---:|---:|---:|
| Rigid contact count | 87,415 | 8,388,608 | 1.04% |
| Rigid patch count | 22,756 | 163,840 | 13.89% |
| Found/lost pairs | 69,632 | 2,097,152 | 3.32% |
| Found/lost aggregate pairs | 48,128 | 33,554,432 | 0.143% |
| Total aggregate pairs | 6,825 | 2,097,152 | 0.325% |
| GPU temporary buffer | 255,072 B | 16 MiB | 1.52% |

추가1,024환경 접근 검사에서는 native OSC에10 mm 증분을160step 입력하고4step마다
통계를 수집했다. 실제 tactile 접촉은 일부 sample의1개 pad로 희박했으며, 조밀한 hand
접촉 stress를 재현한 것은 아니다. 접촉 sample도 포함한 최대 contact는92,637(1.10%),
patch는24,889(15.19%), narrowphase pair는135,363이었다. 최소 pad–Blocker 거리는
0.78 mm까지 줄었다. 이 검사 역시 overflow·CUDA719 없이 끝났다. 명령 증분10 mm를
매step 실제 이동10 mm로 해석하지 않는다.

Heap 약2.50 GiB는 초기64 MiB보다 크지만 이 heap은 확장되므로 capacity overflow로
해석하지 않는다. Collision stack 통계가0인 항목은 실제 사용량을 확인할 수 없는 것으로
취급한다. 이 값으로 stack이 사용되지 않거나 충분하다고 주장하지 않는다.
검사 중 overflow나 CUDA719는 재현되지 않았다. 약95분 후 발생한 오류를 수초의 검사로
해결·배제했다고 판단하지 않는다. 체크포인트 재생 중 raw OSC 토크는 최대약619,930 N·m였지만
IdealPD 제출 arm 토크는150 N·m 이하였다. 큰 요구가 남아 있으나 이것이 메모리 fault의 직접
원인이라는 증거도 아니다.

### 별도로 확인된 Hand mimic 설정 불일치

원본 URDF에는 mimic 관계6개가 있지만, 실제4환경·1,024환경 stage의
`PhysxMimicJointAPI`, fixed tendon, spatial tendon은 모두0개였다. 설치된 Isaac Lab
`urdf_converter.py:150`은 `convert_mimic_joints_to_normal_joints` 값을
`set_parse_mimic`에 직접 넘긴다. 현재False는 `parse_mimic=False`이며, 공식 importer의
`test_urdf_ignore_parse_mimic`도 이때 mimic API가 생성되지 않음을 확인한다.
**설정 이름만 보고 native mimic이 유지된다고 판단하면 안 된다.**

현재 hand actuator는 독립6관절 외의 distal6관절 stiffness/damping을0으로 두므로,
이들은 mimic 비율 제약으로 움직이는 관절이 아니라 passive DOF다. 이는 학습 환경의
기계적 동작에 관한 별도 교정 항목이다. 이번 요청에서는 hand의 joint/import 설정을
변경하지 않았다. 오류 kernel 이름은 tendon/mimic을 포함하지만, 현재 runtime에 그
제약이 존재하거나 원인이라는 증거가 아니다.

### 후속 진단 기준

1. 실패 checkpoint와 같은 action stream에서 physics 직전 상태·raw/submitted torque와
   reset 여부를 보존하고, 실제 fault 이전의 최초 이상을 찾는다.
2. 별도 진단 실행에만 Compute Sanitizer 또는 동기 실행을 사용해 최초 실패 kernel을
   좁힌다. 표준 학습 hot path에 매step 동기화·대형 로그를 넣지 않는다.
3. 같은 제어·reset 조건에서 물체 hull 단순화와 filtered contact read 제외를 각각 비교한다.
   Buffer 증설은 실제 높은 사용률/경고가 확인된 항목에 한정하며 일괄 증설하지 않는다.

CPU 검증은 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`로 MDP·learning 검사를 실행해
**241 passed /18 skipped /152 subtests passed**였다. ROS의 외부 pytest plugin 자동 로딩은
별도 `lark` 의존성 오류를 내므로 이번 검증에서는 자동 로딩을 끄고 프로젝트 검사를 실행했다.
실제 GPU4환경×8 policy step에서244D 관측·reward가 모두 유한했고 종료는0회였다.
Native reward8개와 승인 weight, 종료6개, hard 비활성 mask를 확인했다. 부분 reset 후
나머지 환경의 관절 상태·history가 정확히 보존됐다. 별도 GPU 복사 입력에서 과부하와
높은 물체 속도를 주고, 차단 상태와 통로 확보 상태의 native success·bootstrap mask를
확인했다. 실제 PhysX에 과부하·가상 pose를 쓰거나 성공 조작을 학습시킨 검증은 아니다.
초기 검증 fixture는 reset 직후 유효 F/T sample이 없는 행의 `invalid_state`를 함께 복사해
실패했으며, 원본 진단과 이를 분리한 결과를 모두 `verify_success.json`에 남겼다.
원문·커널 기록·재현 script·native통계는
[physx_failure_20261020](../outputs/contact_manipulation/physx_failure_20261020/)에 보존한다.

## 17. 2026-10-20 — 승인 설정 이후 최신 학습: 정책 분산 증가와 후반 불안정

저장된 TensorBoard·환경/agent YAML·실행 당시 Git commit/diff와169개 checkpoint를 CPU로
분석했다. 최신 긴 실행은 `2026-10-20_13-08-06`이며 분석 시점의 읽을 수 있는 event를
**iteration4934, 2026-10-20 15:44:38 KST**로 고정했다. `/time` tag의 x축은 경과 초라서
iteration 최댓값 계산에서 제외했다. 본 절의 수치는 실제 새 조작을 재생한 평가가 아니다.

### 실제 적용 설정과 비교 범위

| 실행 | Scalar 마지막 iteration | 마지막 checkpoint | Distance weight | 성공 판정 |
|---|---:|---:|---:|---|
| 03-16-36 | 3250 | 3250 | 0.1 | 과거 정지·hard·실패 gate 포함 |
| 12-50-52 | 191 | 150 | 0.4 | Target 보호+지정 Blocker 해소 |
| 13-08-06 | 4934 | 4900 | 0.4 | Target 보호+지정 Blocker 해소 |

세 실행 모두1,024환경·seed0·resume=false·목표90,000iteration·rollout32step이다.
12:50 실행은 짧지만 설정만으로 test라고 제외할 근거는 없다. 최신 두 실행은 저장 소스와
agent 설정이 같고 log_dir만 다르다. 03:16 대비 새 실행은 D0.1→0.4와 success/disabled-hard
mask 교정이 함께 적용됐으므로 성공 수치를 같은 정의의 weight 효과로 해석하지 않는다.

현재 C1.8/D0.4/H0.01/R0.6/Motion4×0.15, `a_D=ln(2)/.1`, 모든 다른 지수 계수도 그대로다.
Time·Terminal은 꺼져 있다. Gaussian·entropy coefficient0.01·OSC/IdealPD·action scale1과
hand import는 세 실행에서 같다. 최신13:08은 clean commit `dd860512`로 현재 핵심 소스와 같다.

### 최신 실행은 후반에 크게 악화됐다

같은13:08 실행의 처음100개와 마지막100개 iteration scalar 평균이다.

| 지표 | 1–100 | 4835–4934 |
|---|---:|---:|
| Runner mean episode reward | 253.659 | 5.691 |
| Runner mean episode length (policy step) | 294.836 | 9.423 |
| 낙하 latest-cause snapshot | 0.448% | 78.112% |
| Timeout latest-cause snapshot | 84.910% | 14.843% |
| Success latest-cause snapshot | 0.188% | 0.768% |
| Policy mean std (서로 다른 단위18축 평균) | 0.306 | 148.235 |
| Collection time | 1.534 s | 2.908 s |
| PPO update time | 0.201 s | 0.188 s |

마지막 평균 episode는50Hz에서 약0.188초다. 이전의 timeout 위주 대기와 달리 낙하 등의
실패로 매우 짧게 끝나는 상태다. 후반 일부 Motion/Rearrangement episode 로그가 작아져도
짧아진 episode의 영향이 섞이므로 부드러운 조작을 배웠다고 해석하지 않는다.

중간501–2000의 success snapshot은 약0.98–1.08%로 일부 Goal 신호가 있었지만,2501–3000에는
0.83%,3501–4000에는0.33%로 줄어 지속적인 개선은 아니다. 마지막0.77%도 낙하78%와 함께
나타난다. Success와 failure는 동시에 참일 수 있고 native scalar는 환경별 최근 종료 원인의
snapshot이므로, 누적 성공률·실패 없는 성공률·낙하 물체의 역할을 이 로그에서 복원할 수 없다.

### 분산 증가 → critic loss 급증 → 짧은 실패 episode 순서

| 관측 사건 | 최초 iteration/값 |
|---|---|
| 전체 mean std >10 | 2294 |
| 전체 mean std >100 | 4247 |
| Critic `Loss/value` >1e3 | 4292 /2750.35 |
| Critic `Loss/value` >1e10 | 4304 /3.223e10 |
| Critic `Loss/value` >1e20 | 4309 /3.052e20 |
| 첫 invalid-state snapshot | 4412 /13.965% |
| 최대 value loss | 4555 /3.284e32 |

4201–4300의 평균 episode는255.81step·reward143.44였지만4601–4700에는9.44step·5.68로
줄었다. Critic 이상은 최초 기록된 invalid-state보다 먼저다. 다만 로그 간 선후관계가
원인 방향을 증명하지는 않는다. 마지막100회의 value loss 평균은28.14로 다시 작아졌지만
짧은 episode와 낙하 우세는 계속되므로 정책이 회복됐다고 볼 수 없다.

모든 scalar와169개 checkpoint의 actor/optimizer tensor는 유한했다. 4250–4600의 critic
head 최대 절댓값은0.1056–0.1117, fusion은1.4396–1.4670으로 거대한 parameter jump도 없다.
따라서 큰 **유한값**의 observation/activation/value prediction/bootstrapped target을 추적해야 한다.
현재 저장된 값에는 이들을 특정할 실제 observation·action mean·raw action·torque가 없다.

### 위치 채널까지 과도한 탐색 분산으로 커졌다

model4900의 상태 독립 Gaussian 표준편차다. 실제 EEF 이동량이나 actor mean을 뜻하지 않는다.

| 채널 | 축별/그룹 평균 σ |
|---|---:|
| EEF 상대 위치3축 | 16.018 /14.427 /32.082 m; 평균20.842 m |
| 회전3축 평균 | 179.011 rad |
| Kp raw6축 평균 | 161.076 |
| Hand raw6축 평균 | 189.863 |

동일 iteration3250에서 이전 실행의 위치 σ 평균은0.373m, 새 실행은5.251m다.
새 실행에서는 위치 σ가1250에1m,3200에5m,3900에10m를 넘었다. 최신 σ에서 Gaussian mean이
어떤 값이더라도 각 Kp 채널이 native 입력 범위[.5,2] 밖일 확률은 최소99.557%, hand[-1,1] 밖은
최소99.518%다. 이는 분포에서 계산한 하한이며 실측 clipping 빈도는 아니다. Pose 증분에는
이 채널 clamp가 적용되지 않는다. Deterministic play는 mean을 사용하므로 σ만으로 그 행동을
설명할 수도 없다.

### Reward 효과와 현재 판단

세 실행의 공통1–190구간 C native 통계는27.705/29.446/26.219, D는.481/1.815/2.089다.
D 지급값의 상승에는 weight4배가 직접 들어가므로 접근거리 개선량으로 읽지 않는다.
새 긴 실행의 C 통계는1001–3000에22–23 부근에서 정체하고, D 통계는초반2.761에서
3151–3250의.784,4001–4100의.515로 줄었다. 이 episode 통계는 순간 가림 면적·거리와 다르다.
Hold는 중간에 약1e−4 수준이지만 안정적인 증가가 없고 전체 지급 비중도 작다. 접촉률이나
유지한 센서 수는 해당 로그를 episode 길이로 나누어 복원하지 않는다.

수식상의 후보 원인은 남는다. Exp Action-rate 비용은 큰 x에서−0.15로 포화하는 반면
Gaussian entropy는 logσ에 따라 계속 증가한다. Native Kp/hand clamp 뒤에는 서로 다른 큰 raw
값이 같은 구동 명령이 될 수 있다. 또 raw previous action18D가 scaling 없이 history에 들어가
ReLU MLP의 입력이 되는 경로가 있다. 분산 증가·입력 증폭·bootstrap 오류 중 무엇이 먼저
문제 상태를 만드는지는 현재 자료로 확정하지 않는다. Entropy loss 계수와 episode reward를
단순히 수치 비교해 어느 쪽이 지배한다고 결론내리지 않는다.

현재 모든 dense 보상은 대략[−1.2,2.37]/step으로 유계이고 γ=.99의 실제 discounted value
상한은237이다. Value loss1e32를 정상적인 reward scale 효과라고 설명할 수 없다.
우선순위는 reward weight/a 추가 변경보다 **정책 분산·critic 수치 안정성 확인**이다.

다음 대조는 reward·장면·제어 설정을 유지하고 entropy coefficient만 줄이거나0으로 두어
std 증가가 완화되는지 보는 것이다. 동시에 action/observation 크기·V·target·timeout bootstrap·
gradient/KL을 GPU에서 집계해 낮은 빈도로 기록해야 한다. 평균행동 재생과 stochastic 재생을
같은 장면에서 분리하면 mean 자체의 문제와 sampling의 문제를 구분할 수 있다. Success
predicate는 유지하면서 `goal & ~active_failure`를 별도 평가 통계로 두는 것도 필요하다.
이들은 다음 실험 제안이며 이번 분석에서 production 설정을 변경하지 않았다.

Iteration 수집+update는 처음1.735초에서 마지막3.096초로 늘었고 update 시간은 비슷하다.
Episode가 짧아져 reset이 잦아지는 것은 유력한 수집 비용 증가 요인이지만, 이 로그에는
physics/reset/inference 분해가 없어 정확한 비용 배분은 확정할 수 없다. IK recovery/fallback
scalar는 모두0이며, 이번 학습이 CUDA719로 끝났다는 근거도 이 scalar에는 없다.

원자료·소스 복원·checkpoint 통계·이상 시점과 그래프는
[recent_training_20261020](../outputs/contact_manipulation/recent_training_20261020/)에 둔다.
전체 그림: [learning_diagnostics.png](../outputs/contact_manipulation/recent_training_20261020/learning_diagnostics.png).

## 18. 2026-10-20 — Contact hold 지급 규모와 실제 물체 물성

사용자가 접촉 유지 보상의 약한 영향과 물체 mass/friction 등의 정리를 요청했다.
16-54-31 실행의3772iteration까지 기록을 분석하고, 환경1개를 초기화해 source USD·composed
USD·native PhysX 값을 대조했다. Policy 실행·episode reset·rollout은 수행하지 않았다.

### Contact hold의 실제 기여

현재 `r_H = .01 × Σ_i[b^B_{i,k-1} & b^B_{i,k}]`이며17개 pad를 평균하지 않고 합산한다.
Blocker-only normal force norm>.01N을 같은 pad의 직전·현재 sample에서 만족해야 한다.
최초 접촉이나 다른 링크로 미는 접촉은0이다. Policy50Hz의 마지막 physics sample을 읽으므로
짧은 접촉이 다음 sample 전에 끝나면 유지 보상이 없다. 두 sample 사이 연속 접촉을 증명하는
판정은 아니다. Native dt 상쇄 후 pad당 정확히.01/policy step이다.

| 16:54 실행 구간 | Clearance native 평균 | Hold native 평균 | 지급액 H/C |
|---|---:|---:|---:|
| 801–900 | 22.185015 | .00042027 | .001894% |
| 1451–1550 | 17.706207 | .00033974 | .001919% |
| 3673–3772 | 24.993980 | .000009995 | .000040% |

실제 지급 비중은 매우 작다. 다만 native episode log는 reset그룹 누적값/최대8초를
iteration에서 평균한 것이므로 실제 접촉률·센서 수·접촉 지속시간으로 환산하지 않는다.
작은 weight와 같은 pad의 반복 접촉 기회 부족을 현재 scalar만으로 분리할 수 없다.
1550의 같은 궤적에5배를 적용해도 H/C는 약.0096%이므로 weight만으로 해결을 확정하지 않는다.

| H weight | 1 pad/step | 3 pads/step | 17 pads 최대/step |
|---|---:|---:|---:|
| 현재.01 | .01 | .03 | .17 |
| 대조.03 | .03 | .09 | .51 |
| 대조.05 | .05 | .15 | .85 |

Clearance는 유효 상태에서.662–1.8/step이다. 첫 weight 대조값으로.03/.05를 제안하며
binary 관측·17 pad 합산·유지 판정은 바꾸지 않는다. .05에서3pad 접촉은.15/step이므로
현재보다 의미 있는 조건부 기여가 된다. 다수 pad를 붙이고 통로를 치우지 않는 정지 행동에도
양의 점수를 주는 구조이므로 실제 pad count와 접촉 이후 Clearance 개선을 함께 봐야 한다.
Weight 변경 자체는 새 센서나 연산을 추가하지 않는다. 다음 진단은 GPU에서 Blocker 접촉
여부·held pad count·첫 접촉의 다음 sample 유지 여부를 집계하는 것이다.

16:54 학습 YAML은 R=.6으로 활성이고 현재 작업 코드에서는 사용자가 Rearrangement를
주석 처리했다. 이 차이를 유지했다. 이번 분석에서 hold weight나 물성은 변경하지 않았다.

### 물체별 실제 PhysX 질량

| 물체 | 질량 kg | 지정 방식 |
|---|---:|---|
| bottle_1 | .549279 | 형상·기본 밀도로 자동 계산 |
| cup_1 | .447285 | 형상·기본 밀도로 자동 계산 |
| cup_2 | .271988 | 형상·기본 밀도로 자동 계산 |
| mug_1 | .641424 | 형상·기본 밀도로 자동 계산 |
| mug_2 | .534807 | 형상·기본 밀도로 자동 계산 |
| can_1 | .368075 | 형상·기본 밀도로 자동 계산 |
| YCB003 cracker box | .411 | USD에 명시 |
| YCB004 sugar box | .514 | USD에 명시 |
| YCB005 tomato soup can | .349 | USD에 명시 |
| YCB006 mustard bottle | .603 | USD에 명시 |

기존6종에는 MassAPI·mass·density나 physics material의 density가 지정되지 않아
충돌 체적과 기본 밀도1,000kg/m³로 계산된다. 실물 측정 질량으로 설정한 값이 아니다.
Convex hull이 컵의 빈 공간을 채우면 자동 질량·관성에 영향을 줄 수 있다. YCB4종은 명시된
질량을 사용한다. 관성은 전부 PhysX 자동 계산이며 COM pose와3×3 tensor는 artifact에 보존했다.
전체10종의 mass·COM·inertia는10월14일의 native snapshot과 정확히 같았다(max difference0).
[공식 질량 계산 설명](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html).

### 마찰·접촉 및 나머지 설정

물체10종·선반의17개 shape·로봇 전체 shape(따라서 tactile17개 포함)의 **native 정지마찰.5,
동마찰.5, restitution0**을 확인했다. Source의 Looks binding은 물리 속성이 없는 시각 재질이다.
별도 material override가 없어 simulation default를 사용하며 combine은 average다.
따라서 현재 물체–선반과 물체–pad의 유효 마찰계수도.5다. 표면별 마찰 차이는 모델링하지 않는다.
선반은100kg dynamic body다.

| 속성 | 기존6종 | YCB4종 |
|---|---|---|
| Contact offset (native) | 1.962mm 자동값 | 5mm 명시 |
| Rest offset (native) | 0 | 0 |
| Linear/angular damping (composed USD) | 0/.05 | 0/.05 |
| 최대 각속도 (USD 설정) | 약100rad/s | 약1.745rad/s |
| Solver position/velocity iteration | 16/2 | 16/2 |
| Max depenetration velocity | 1m/s | 1m/s |
| 중력 | 활성 | 활성 |

각속도의 schema 단위는 **deg/s**다. 기존 default5729.58°/s가100rad/s이고 YCB의
authored100°/s는1.745rad/s다. 이것을 둘 다100rad/s로 읽으면 안 된다. 실제 rollout의
측정 각속도는 아니며 설정 상한의 차이다. CCD·speculative CCD는 비활성이고 episode마다
질량·마찰 randomization은 없다. Contact offset은 고정20mm 기본값이 아니라 scene/timestep에
따라 정해지는 자동값이므로 여기서는 native1.962mm를 기준으로 쓴다.

003/004는 원본 Mesh collision=False, Cube proxy만True다. SDK의 `max_shapes=2`에는 비활성
원본 slot이 포함되므로 active collider2개라는 뜻이 아니다. 이전 audit에도 두 offset slot이
있었으며 새로운 충돌 형상 regression은 확인되지 않았다.

물성을 명시적으로 관리하려면 기존6종의 실제 질량을 정하고, 재질별 마찰과 두 그룹의
각속도 상한 차이를 별도 검토해야 한다. Hold weight 실험과 물성 변경은 분리해 효과를 비교한다.
정리·strict JSON·재현 script·native 조회 log는
[contact_hold_physics_20261020](../outputs/contact_manipulation/contact_hold_physics_20261020/)에 둔다.

## 19. 2026-10-20 — Hold weight·정책100Hz 적용과 slowdown 분리

### 적용한 설정

사용자 승인으로 `blocker_contact_hold.weight`를 **0.01→0.05**로 변경했다. Same-pad의
직전·현재 Blocker 접촉을 합산하는 수식·센서·filter는 유지하므로 weight 변경으로 연산이
추가되지는 않는다. Pad1개·3개·17개 유지 시 각각0.05·0.15·0.85/step이다. 기존 지급량이
작았던 원인에 접촉 유지 기회의 부족도 포함될 수 있어 weight만으로 학습 개선을 보장하지 않는다.
접촉 이후 Clearance 진행과 성공을 함께 비교한다.

물체·선반·pad의 마찰은 현재 **μs=.5/μd=.5, restitution0**을 유지한다. 추후 마찰을
변경할 수 있는 항목으로 남기며, 재질별 마찰·질량 조정은 이번 weight·주기 변경과 분리한다.
사용자가 주석 처리한 Rearrangement는 복원하지 않았다. Time·Terminal·hard-load 종료도
비활성을 유지한다. 활성 보상은 C1.8/D.4/H.05/Motion4×.15의7개다.

추가 요청으로 물리 `dt=.01`은 유지하고 **decimation2→1, 정책50→100Hz**를 적용했다.
성공 연속 확인은5→10step으로 바꿔0.1초를 보존했다. 전도0.2초는 dt에 따라 자동으로
20step이 되며, 최대8초 episode·공통 장면 경계도800step이 된다. Current/previous 관측은
10ms 간격이 된다. 다른 weight·action scale·γ=.99·rollout32는 자동 변경하지 않았다.

Dense reward는 native dt 상쇄 후 **policy step마다** 지급하므로 같은8초 동안 지급 기회가
두 배다. 현재 step당 상한은1.8+.4+17×.05=3.05, γ=.99의 보수적 할인합 상한은305다.
Rollout32는0.64→0.32초의 simulation을 포함하며 γ의 물리 시간상 할인 길이도 절반이다.
따라서 과거50Hz 학습과 return·iteration 시간을 바로 비교하지 않는다. Terminal은 여전히
비활성이며 이 상한만으로 scale을 재조정하지 않았다.

기존50Hz checkpoint 재생은 `env.decimation=2 env.mdp.stable_steps=5`를 명시한다.
GUI 렌더링까지 당시 주기로 맞추려면 `env.sim.render_interval=2`도 추가한다. 현재 play는
checkpoint의 env.yaml에서 주기를 자동 복원하지 않는다. 이미 실행 중인 프로세스의 설정은
이 소스 변경으로 바뀌지 않으며 새 실행부터100Hz 기본값을 사용한다.

### 과거 학습 로그의 동일 구간 비교

분석 snapshot은 최신 `20-18-18`의 **iteration359,20:37:08KST**다. 세 실행 모두
1,024환경·물리100Hz/정책50Hz·rollout32이며, 같은101–359 구간을 비교했다.

| 실행 | Collection | PPO update | 합계 |
|---|---:|---:|---:|
| `13-08-06` | 1.6299초 | .2053초 | 1.8352초 |
| `16-54-31` | 2.8090초 | .2008초 | 3.0099초 |
| `20-18-18` | 2.8248초 | .2038초 | 3.0286초 |

증가는 **Collection 약1.19초,73.3%**에 있으며 PPO update는 약.20초로 일정하다.
저장 env.yaml·agent.yaml과 미커밋 diff를 비교하면13:08→16:54의 실질 설정 차이는
**robot self-collision False→True**뿐이다.16:54→20:18은 Rearrangement 비활성만 다르고,
Collection 차이는.56%다. 큰 slowdown은 이미16:54에 나타났다.

최신 평균 episode302.0step은13:08의301.2step과 비슷하다. 실제 reset 횟수는 scalar에
없지만 이 구간에서 reset 빈도가 급증했다는 근거는 없다. Recovery/fallback은0,
search-round는1이다.13:08 후반의9.42step·낙하 우세 구간은 별도의 reset-heavy
정책 붕괴 사례이므로 self-collision 비교의 baseline으로 쓰지 않는다.

Scalar wall 간격에서 collection+update를 뺀 일반 overhead는 약.0235초다. Console·logging·
checkpoint 등의 외부 overhead가1.19초 증가를 설명하지 못한다. Production hot path에
강제 CUDA synchronization·매-step flush·새 debug print도 없다. Solver·물체 형상·
contact offset·grouped sensor·headless 렌더링 및 reset 최적화가 되돌아간 흔적은 없다.

### 실제1,024환경의 native 비교

RTX3090에서 seed20261020,같은 초기 joint/object/role tensor 해시,공식 OSC·hand action,
7개 native reward를 사용했다. EEF delta는0, hand 목표는 실제 reset 관절 자세로 유지했다.
Policy 추론·PPO update·학습 logger는 포함하지 않았다. 각 경우별 별도 프로세스에서
warmup32를 제외하고,32step block 경계에서만 동기화했다.50Hz는448step,100Hz는832step으로
각각 새 공통 장면 생성1회를 포함했다. 중간 failure/reset은0이고400/800 경계 timeout만
각각1,024개였다. 실제 학습 행동의 처리량을 보장하는 benchmark는 아니다.

| Self collision / policy | 32step 환경 실행 | Physics 호출당 시간¹ | wall / simulation1초 |
|---|---:|---:|---:|
| OFF / 50Hz | 1.1380초 | 8.825ms | 1.778초 |
| ON / 50Hz | 2.3873초 | 22.817ms | 3.730초 |
| ON / 100Hz | 1.3069초 | 20.662ms | 4.084초 |

¹ Physics 시간은 headline과 별도의32step 진단에서 호출 전후 동기화한 inclusive wall 값이다.
동기화·nested 호출 때문에 stage 값을 headline에 더하거나 엄밀한 GPU kernel 비중으로 읽지 않는다.

Self-collision을 켜는 것만으로 같은50Hz의 환경 실행이 **109.8% 증가**했다. 물리 호출이
8.825→22.817ms로 늘었고, sampled native rigid contact 최대87,033→135,984,
patch22,528→36,684였다. Buffer overflow가 아니라 추가 접촉·constraint 처리 비용의
증거다. 전체 학습 slowdown의 모든 시간을 이 fixture만으로 설명할 수는 없지만,
Self-collision이 주요 원인이라는 로그상 후보를 실제 환경에서 재현했다.

100Hz는32step당 physics 호출64→32회로 줄어 같은32step wall은45.3% 감소했다.
다만 포함한 simulation 시간이 절반이어서 물리 시간1초당 비용은 오히려9.5% 늘었다.
이는 고속 정책 호출에 따른 MDP·센서·action 처리 빈도 증가도 있기 때문이다.1.3069초는
PPO 추론·update를 포함한 새 학습 iteration 시간이 아니다.

동기 진단의100Hz 대표 호출 시간은 physics20.662ms, OSC apply2.474ms,
action process3.755ms,termination+cached MDP5.713ms,reward2.400ms,
tactile read.330ms,observation.177ms였다. 별도128환경 partial reset은16.392ms였다.
이 reset은 한 번의 대표 fixture이며 장기학습 reset 분포 전체의 평균은 아니다.

Headline 이후 env0에만 추가한 native contact view에서 **손바닥 `right_plam_1`과
손가락 proximal 링크** 사이 큰 normal-force norm이 측정됐다. Middle9,689N,ring8,695N,
index6,037N,little3,134N,thumb2,929N이다. View는 ContactReportAPI가 있는34개 링크를
41개 filter 링크와 비교하며, row 매핑은 실제 `view.sensor_paths`를 사용한다. 이 값은
마지막으로 physics가 계산한 진단 상태의 값이며 reset 순간의 새 측정·실제 하드웨어 하중이 아니다.
서로 반대 방향의 동일 pair는 중복 집계하지 않는다.

URDF에서 palm housing과 각 proximal은 같은 `base_link`의 자식이라 native parent-child
collision 제외만으로는 이 sibling 쌍이 제외되지 않는다. 이는 **hand 내부의 계속 발생하는
하우징 접촉**을 우선 검토할 근거다. 다음 개선은 이5개 쌍의 형상·관절 범위에서의 의도된
중첩을 확인하고 필요한 pair만 공식 `FilteredPairsAPI`로 제외하는 것이다. 손가락–손가락,
hand–arm 충돌을 일괄 끄거나 UR5e 형상을 바꾸지 않는다. 이번 작업은 분석이므로 실제
filter와 self-collision=True 설정은 변경하지 않았다. CUDA719 원인이 해결됐다는 뜻도 아니다.

### 검증과 산출물

CPU MDP·모델 검사는 **241pass/18skip/152subtest**다. 실제100Hz native 환경에서도
800step horizon·success10step·Hold.05 및 활성 reward7개를 확인했고,832step headline과
별도 진단·부분reset을 정상 종료했다.100Hz 새 정책의 학습 성능은 아직 검증하지 않았다.

분석 코드·historical scalar/config diff·세 경우 native timing JSON과 재현 probe는
[slowdown_20261020](../outputs/contact_manipulation/slowdown_20261020/)에 보존한다.

## 20. 2026-10-20 — Hand 관절 한계값과 실제 뒤로 꺾임 재현

사용자가 손가락의 역방향 굽힘이 실제로 불가능하다는 점과 URDF 관절 범위 일치 여부를
질문했다. 제조사 원본과 사용하는 `_relative.urdf`의31개 joint를 비교했으며
axis/origin/parent/child/limit/mimic 정의는 동일했다. Relative 버전은 mesh 경로를 고친 파일이다.
현재 프로젝트에서 position limit을 다시 쓰는 코드는 없다.

실제1환경에서 composed USD(각도deg)와 native PhysX hard/soft limit(rad)을 읽었다.
손의12개 revolute 관절 모두 URDF와 일치하며 최대 float 변환 차이는 **1.389e−7rad**다.
`soft_joint_pos_limit_factor=1`이므로 action용 soft limit도 같은 범위다.

| Hand 관절 | URDF와 native position 범위 |
|---|---|
| thumb1 | 0–1.658rad, 약0–95° |
| thumb2 | 0–.62rad, 약0–35.52° |
| thumb3 | 0–.541052rad, 약0–31° |
| thumb4 | 0–.455531rad, 약0–26.1° |
| index/middle/ring/little의 각1번 관절 | 0–1.4381rad, 약0–82.40° |
| index/middle/ring/little의 각2번 관절 | 0–1.476374rad, 약0–84.59° |

공식 `JointPositionToLimitsAction`은 독립6관절만 제어하며, raw action을[-1,1]로 자른 뒤
이 범위로 매핑한다. 실제 raw−2/−1에서 목표0,raw0에서 각 관절 중간값,raw1/2에서
상한이 나오는 것을 확인했다. 따라서 음수 관절 목표를 명령해서 발생한 뒤로 꺾임은 아니다.
단, 제어 목표의 범위와 물리 solver가 실제로 만족하는 관절각 범위는 구분해야 한다.

### 동일 조건 self-collision 대조

Seed20261020,1환경,physics/policy100Hz,동일 reset 관절 상태,EEF delta0,
손 목표12개 초기각0 중 독립6개를0으로 유지하는 action으로200physics step(2초)을 실행했다.
Reward·policy·자동 episode reset 없이 공식 action manager와 native simulation을 사용했다.
별도 프로세스에서 **self-collision ON/OFF만** 바꿨으며 production 설정은 바꾸지 않았다.

| 실제 관절각 측정 | Self collision ON | OFF |
|---|---:|---:|
| 가장 큰 하한 이탈 | −30.20745° (`little2`) | −.00001713° |
| 가장 큰 상한 이탈 | 3.64695° | .00003927° |
| 표본 internal pair force norm 최대 | 12,641.65N | 0 |

ON에서는 첫 physics step부터 middle2−3.76°/ring2−6.97°,
4step에서 middle2−22.82°/ring2−26.76°였다.200step 동안 thumb2−29.69°,
middle2−26.53°/ring2−27.34°도 측정됐다. Palm housing↔proximal의 강한 접촉이 즉시
발생하며, OFF에서는 큰 범위 이탈이 사라졌다. **범위 설정 누락이 아니라 현재 hand 내부
충돌 제약 아래에서 실제 관절 한계가 크게 위반되는 현상**을 이 fixture에서 재현했다.
이는 단순한 시각적 착각이나 정상적으로 허용된 손 자세가 아니다. 장기학습의 모든 이상 자세나
CUDA719까지 같은 원인이라고 확정하지는 않는다.

다음 교정은 hand collision proxy의 연결부 중첩을 확인하고, 불필요한 housing–proximal
충돌 쌍만 공식 filter로 제외하거나 형상을 바로잡는 것이다. URDF position limit을 넓히거나
매 step 실제 joint pose를 강제 clamp하는 방식으로 처리하지 않는다. UR5e와 유효한
손가락–손가락/hand–arm 충돌은 유지할 대상이다.

### 별도로 확인한 실제 손과의 차이

URDF의 mimic 관계6개와 달리 실제stage Mimic/Tendon API는0개, native fixed/spatial
tendon도0개다. Distal6개는 gains0/0인 passive 관절이다. 설치 converter는 현재
`convert_mimic_joints_to_normal_joints=False`를 `set_parse_mimic(False)`로 전달한다.
OFF 대조에서도 thumb2가약.153°이하인 동안 thumb3가약30.99°까지 움직여,
정상적인 기계 연동이 없다는 점을 별도로 확인했다. 따라서 내부 충돌만 교정해도 손의
전체 kinematics가 URDF와 같아지는 것은 아니다.

Mimic을 복원할 때는 실제 native API 생성과 연동각 검증이 필요하다. URDF의 선형 mimic
비율과 개별 upper limit에도 일부 불일치가 있어 부모·자식 full-range가 동시에 가능한지
확인해야 한다. 임의의 custom PD/위치 덮어쓰기 없이 importer의 공식 경로를 먼저 검토한다.

또한 현재 hand actuator는 URDF `effort=1N·m`를 **10N·m**로 덮어쓰며,
thumb velocity는 URDF2rad/s 대신1rad/s다. 이 차이는 position 범위와 별개이고
이번 ON/OFF 대조에서는 동일하게 유지했다. 토크 설정의 영향은 이 실험으로 분리하지 않았다.

이 작업에서 URDF·production controller/물성/형상/충돌 설정은 변경하지 않았다.
원본·USD·native 한계값,action 매핑,200step 관절각 min/max와 시간별 접촉 표본은
[hand_limits_20261020](../outputs/contact_manipulation/hand_limits_20261020/)에 보존한다.

## 21. 2026-10-20 — 공식 API로 Hand URDF 연동 복원과 tactile 보존

사용자 요청에 따라 Hand를 공식 URDF importer·`ImplicitActuatorCfg`·
`JointPositionToLimitsActionCfg`와 native PhysX mimic 제약으로 구성했다.
Python에서 종속관절의 목표를 매 step 계산하거나 실제 관절각을 덮어쓰지 않는다.
기존 17개 tactile 링크·메시·채널 순서·binary observation·Blocker별 접촉 이력은 유지한다.

### 적용한 설정과 원인

- 설치된 converter는 `convert_mimic_joints_to_normal_joints`를 `set_parse_mimic`에
  직접 전달한다. 따라서 True로 설정해 실제 mimic API6개를 생성한다.
  `joint_drive=None`으로 importer의 mimic drive 구분을 유지하고, 공식 USD DriveAPI의
  force drive와 native implicit actuator로 독립6관절만 구동한다.
- Hand 토크 상한은 덮어썼던10N·m 대신 URDF의1N·m를 계승한다.
  독립관절 gain20/1·종속관절0/0이다. 원본 URDF 질량·COM·관성은 수정하지 않았다.
- URDF에는 master 최대각을 그대로 적용하면 child 최대각을 넘는 관계가 있다.
  구성 시 연동식 전체의 교집합을 계산해 공식 master hard limit에 전달한다.
  네 손가락 master는0–1.36159181rad, thumb2는0–.609220762rad이며 thumb1은0–1.658rad다.
  손가락 master 속도는1/1.0843=.922253966rad/s, thumb은2rad/s다.
  종속관절의 저장된 limit은 importer의 확장 범위를 유지하고 별도 velocity brake를
  추가하지 않는다. 중복 end-stop·속도 제약은 mimic과 충돌해 수치 불안정을 유발했다.
  실제12관절의 각도를 원본 URDF 범위와 별도로 비교한다.
- 기존 Python convex proxy 생성 코드를 제거했다. 원본 collision mesh 중 오목한
  palm1개와 finger parent4개만 공식 `ConvexDecompositionPropertiesCfg`로 처리한다.
  Palm은 최대16hull·500,000voxel·error1%, 네 finger parent는 각각4hull·100,000voxel,
  hull당64정점·shrink-wrap을 사용한다. 17개 pad의 메시·rigid body는 유지한다.
- 고정 housing2개와 proximal root5개의 의도된 연결부 중첩10쌍만 공식
  `FilteredPairsAPI`로 제외했다. 다른 self-collision·pad–물체 충돌은 유지한다.
  Geometry instance 공유 시 cross-body 관계가 reference 밖을 가리켜 유실되던 문제도
  수정했다. 관계를 editable body root에 보존해 환경 복제 후에도 해당 환경을 가리킨다.
- Hand에만 공식 max depenetration velocity .1m/s를 설정했다. UR5e는1m/s를 유지한다.
  결합된 로봇 articulation의 position solver는16→32회, velocity는2회다.
  이 수치 설정은 팔·손 전체에 적용되며 UR5e actuator·controller·collision mesh는 유지한다.

### 실제 여닫기 동작의 수렴 비교

동일한1환경·100Hz·800step에서 open/half/close/reopen 목표를 사용했다.
접촉 부하가 걸리는 완전 닫기까지 포함하며, 비교 중 Hand 물성과 목표는 동일했다.
아래 수치는 공식 물리 제약의 근사 오차이며 모든 외력에서의 절대 보장은 아니다.

| Position solver 반복 | 최대 mimic 오차(rad) | 완전 닫기 안정화 후 mimic 오차(rad) | 최대 URDF 상한 이탈(rad) |
|---|---:|---:|---:|
| 16 | .115387 | .04922 | .049298 |
| 32 — 적용 | .030736 | .006551 | .0000677 |
| 64 — 대조 | .012126 | .004397 | .0000132 |

32회에서 최대 하한 이탈은.0000465rad이며, 관절·F/T 값은 유한했다.
최대 위치 범위 이탈.0000677rad는 약.004°, 최대 mimic 오차는 약1.76°다.
기존 URDF 범위를 넓히거나 손가락 간 충돌을 전부 끄는 방법으로 처리하지 않았다.

동일한1,024환경 시작 관절값에서 별도 probe·diagnostic sensor가 없는128step을 비교했다.
Native32 policy step 시간은32회 solver1.32998초,64회1.95471초로64회가46.97% 느렸다.
NN 추론·PPO update는 빠진 환경 실행 시간이다. 초기에는64회를 후보로 삼았으나
실제 관절 범위와 실행 비용을 함께 고려해32회를 채택했다.
이 검증의 기존 mimic 오차.02rad 기준은32회에서 통과하지 않는다. 그 결과를 숨기거나
시험 기준을 소급 변경하지 않고 위의 실제 오차와 비용을 채택 근거로 남긴다.

**남은 속도 한계:** 독립관절의 native 속도 제한은 적용되지만 접촉 중 종속관절의
실제 순간 속도는 원본 상한을 넘었다. 채택한32회 설정에서 Thumb3/4 최대4.545/4.036rad/s(원본2),
index2 최대9.249rad/s(원본1), 다른 finger2는1.108–1.566rad/s(원본1)였다.
64회에서는 index2 최대7.720rad/s로 줄었지만 원본 상한을 엄격히 보장하지 못했다.
64회 solver·확장 child 위치 범위 상태에서 종속관절에도 공식 속도 제한을 추가한
별도 대조는 half 동작부터 master 속도53–138rad/s·mimic 오차.846rad로 불안정해 채택하지 않았다.
따라서 모든12관절의 위치·속도 제한이 수치적으로 엄격히 보장된다고 설명하지 않는다.
해당 대조는 artifact에서만 수행했고 production에는 반영하지 않았다.

### Contact sensor 검증

실제4환경에서 다음을 검증했다.

- 17개 링크·센서 행과 observation 순서, 물체별 filtered contact matrix shape.
- 공식 구형 rigid object로17개 pad 각각의 실제 접촉과 binary observation 확인,
  분리 후 접촉 해제. Palm pad가 housing보다 먼저 접촉함도 확인했다.
- 실제 YCB 물체를 이용한 Blocker 접촉과 다른 물체 접촉의 구분.
- 첫 접촉의 hold0, 같은 pad의 previous/current 접촉 유지 보상.
- 선택 환경의 센서·이력 초기화와 실제 `env._reset_idx([0])` 후 다른 환경의
  로봇·물체·접촉·이력 보존.

최종32회 설정에서 낮은 probe 힘도.08816N 이상으로 binary 기준.01N을 넘었다.
시험용 구체를 얇은 pad 내부까지 깊게 넣으면 housing 접촉으로
넘어갈 수 있으므로, 실제 표면의 정상 접근 깊이에서 접촉을 검사했다.
시험용 probe·관절 상태 지정은 검증 스크립트에만 있고 학습 경로에는 추가하지 않았다.

추가 진단 센서·probe가 없는 실제1,024환경에서100Hz로832개 policy step을 실행했다.
모든 측정 block에서 관절·토크·F/T·filtered tactile matrix·244D observation이 유한했다.
부분reset `[0,512]`의 다른 환경 보존과 common step800의 전체 장면 교체(세대0→1)를
통과했다. 전체reset을 포함한 native32step 평균은1.33313초, reset 없는 block은1.32752초였다.
NN 추론·PPO update를 포함한 iteration 시간과 구분한다. 상세 결과의 현재 진입점은
[검증 요약](../outputs/contact_manipulation/hand_urdf_native_20261020/SUMMARY.md)이다.

관측244D·action18D와 모델 구조·reward는 이번 Hand 변경으로 바꾸지 않았다.
기존 checkpoint는 다른 Hand dynamics로 학습됐으므로 새 dynamics의 학습 결과와 구분한다.
장기 CUDA719 오류가 해결됐다는 결론은 이번 짧은 동작 검사로 내리지 않는다.
재현 스크립트와 단계별 JSON은
[hand_urdf_native_20261020](../outputs/contact_manipulation/hand_urdf_native_20261020/)에 보존한다.
실패한 중간 실험과 최종 검증을 파일명으로 구분한다.
CPU 회귀 검사는241pass·18skip·296subtest pass이며 `git diff --check`도 통과했다.

공식 근거:
[URDF importer](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html),
[PhysX mimic API](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_mimic_joint_a_p_i.html),
[Articulation mimic 설정](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html).

## 22. 2026-10-20 — Clearance를 episode 초기 가림 대비 개선으로 변경

사용자가 초기 가림비율을 기준으로 개선에 보상을 주도록 요청했다.
기존에는 처음 상태를 유지해도 `1.8 exp(−x)`가 매 step 지급됐다.
이번에는 같은 exp 품질 함수에서 해당 episode 초기 품질을 뺀다.

$$
x_t=\frac{A_{\mathrm{blocked}}(t)}{A_{\mathrm{blocker}}(t)},\qquad
q_t=\exp(-a_Cx_t),\qquad
r_C(t)=1.8\,(q_t-q_0).
$$

`a_C=1`, weight1.8은 유지한다. 지정 Blocker만 접근영역 A₀와 교차하는 면적을 계산하며,
다른 물체의 overlap은 포함하지 않는다. 초기와 같으면0, 개선양수, 악화음수다.
예를 들어 초기 가림비율.5에서 .3으로 줄이면+.24172/step, .7로 늘면−.19790/step이다.
초기 배치로 돌아오면0이다. 직전 step 대비 증가량 방식이 아니므로 개선된 상태를
유지하면 같은 양의 보상이 계속 지급된다.

초기 품질 `TaskState.initial_clearance_quality`는 reset 대상 행의 실제 초기 OBB에서
기존 GPU `projected_areas`와 공통 `clearance_quality`를 사용해 한 번 계산한다.
부분 reset은 해당 행의 기준만 갱신한다. Raw `components[:,0]`은 절대 품질을 유지하고
reset 직후에도 초기 품질로 채워, 지급 어댑터의 차이가 정확히0이 되도록 한다.
어댑터는 차이를 `step_dt`로 나눠 공식 RewardManager의 dt 곱과 상쇄한다.
환경별 Python 반복·CPU 전송·새 파일·observation 채널을 학습 경로에 추가하지 않았다.

**지급 범위와 해석:** 장면별 Clearance 최대 기여는 `1.8(1−exp(−x₀))`다.
초기비율.5이면 최대.70824/step이며 전체 가능한 상한은1.13782/step이다.
거리.4·Hold.85를 포함한 현재 dense 상한은약2.38782/step이다.
추가 정규화·weight 변경은 하지 않았다. 가림 개선에 대한 기울기는 기존과 같고,
고정 episode 길이에서는 episode별 return에서 상수를 빼는 효과다.
조기 종료가 있으면 생존 기간의 유인이 달라지지만, 이 변경만으로 비조작 정체가
해결된다고 결론 내리지 않는다. 이후 weight·a·성공/실패 종료 설계를 함께 검토한다.
이후 native `Episode_Reward/clearance`는 초기 기준 보정 후 지급값이므로 과거의
절대 Clearance 로그와 숫자를 직접 비교하지 않는다.

검증: CPU244pass·18skip·332subtest pass. 정지0·개선/악화 부호·초기 복귀0·
개선 상태 반복 지급·서로 다른 초기 가림·비연속ID 부분 reset·dt 상쇄를 확인했다.
실제 RTX3090의1,024행 GPU tensor 검사도 통과했다. CPU double 대비 초기 품질의
최대 절대 오차는2.47e−8, 초기 기준 계산 핵심 query는평균.182ms였다.
이는 물리 simulation이나 전체 reset·PPO iteration 시간이 아니다.
재현 코드와 결과는
[clearance_initial_20261020](../outputs/contact_manipulation/clearance_initial_20261020/)에 보존한다.
이번 변경 후 새 정책의 학습 성과는 아직 측정하지 않았다.

## 23. 2026-10-20 — 성공 Terminal50과 최신 학습 기반 Clearance4 적용

사용자가 Terminal term을 활성화하고 성공50·그 외0을 요청했다.
추가로 이전 학습을 참고해 Clearance weight를 높이도록 요청했다.
현재 Terminal weight1.0을 유지하며 성공 보너스를+50으로 설정했다.

### Terminal 지급 규칙

- 활성 success 조건이 충족되면 episode당 한 번+50을 지급한다.
- 성공 없이 실패·timeout만 발생하거나 아직 성공 유지시간을 채우지 못하면0이다.
- 현재 성공·실패 flag는 독립 판정이다. 두 flag가 동시에 참이면 **success 기준으로+50**이다.
  기존 failure 우선 감점 분기를 제거했다. 실패 판정·종료·원인 로그는 유지한다.
- `/step_dt` 어댑터와 native dt 곱을 유지하므로100Hz에서도 지급액은+50이다.
  이미 지급한 episode에는 추가 지급하지 않고 부분 reset에서 해당 행만 초기화한다.
- 저장된 과거 cfg의 `failure_penalty` 필드는 호환용0으로 남기며 지급 계산에서 사용하지 않는다.

### 비교한 최근 학습

현재 snapshot은 [2026-10-20_22-33-03](../logs/rsl_rl/contact_manipulation/2026-10-20_22-33-03/)의361회다.
1,024환경·100Hz·Clearance1.8/Distance.4/Hold.05/Motion4×.15이며 Terminal/R은 비활성이다.
Saved git diff에서 `initial_clearance_quality` 차감을 확인했으므로,22절 변경 전 절대
Clearance 로그와 섞지 않고 초기 기준 보정 후의 지급을 비교했다.

| Native scalar 평균 | 101–200회 | 201–300회 |
|---|---:|---:|
| Episode_Reward/clearance | 5.95921 | 6.70647 |
| Episode_Reward/blocker_distance | 7.91652 | 5.23216 |
| Episode_Reward/blocker_contact_hold | .12512 | .06193 |
| Motion 네 항 합 | −1.34249 | −1.35254 |
| Success 종료 원인 snapshot | 7.34094% | 3.30649% |

지급 로그는 episode 누적값/최대episode시간8초의 native scalar이며 step당 보상이 아니다.
Success는 종료 원인 snapshot의 구간 평균으로 누적 성공률이나 failure 없는 성공률이 아니다.
두 구간에서 접촉·접근 신호는 있지만 성공 snapshot은 낮아졌으며,101–200회에는
Clearance가 Distance의.75276배였다. 이 결과만으로 원인을 weight 하나로 확정하지 않는다.

### 적용한 Clearance scale

**Weight1.8→4.0**, `a_C=1`을 유지한다. 같은101–200회 궤적에서 선형 환산하면
Clearance 로그는5.95921→13.24269, Distance 대비.75276→1.67279배다.
이는 같은 궤적의 지급 크기 비교이며 새로운 정책의 학습 결과를 예측하는 값은 아니다.
Weight4는2.22배 강화하는 첫 비교값이며 최적값이라는 증거는 없다.

초기 가림비율.5에서 .4로 줄이면+.11482→+.25516/step,
.3으로 줄이면+.24172→+.53715/step이다. 같은 pad 접촉 유지는 pad당.05,
Motion 한 항의 최대 비용은.15다. 이 비교는 목표 달성 항의 상대 영향력을 높이는 근거다.
초기 대비 개선 상태를 유지하면 지급이 반복되는 성질도 함께 강화되므로 완료를
서두르는 정책이 학습된다고 보장하지 않는다. Terminal50도 함께 바뀌므로 다음 학습의
차이를 Clearance weight 하나의 효과로 해석하지 않는다.

현재 non-terminal 상한은약3.77848/step이다. 다른 weight·a·제어·관측·성공/실패 조건은
유지했다. 변경은 이후 새로 시작하는 실행에 적용되며 이미 실행 중인 프로세스에는 자동 반영되지 않는다.

### 별도로 남은 수치 이슈와 검증

같은 학습에서 iteration324의 value loss는1.558e10으로 증가했고327(22:45:44 KST)에
처음Inf가 기록됐다. Invalid-state snapshot도327에서 처음 양수가 됐다.
327의 mean std는.347로, 과거처럼 std가수백이 된 사례와 동일한 원인으로 단정할 수 없다.
327 이후는 weight 결정 근거에서 제외했다. 이 수치 불안정이 이번 weight 변경으로
해결됐다는 증거는 없으며 별도 원인 검토가 필요하다.

CPU244pass·18skip·334subtest pass. 성공/실패 동시 판정·실패만·미완료·중복 지급·
success 비활성·50/100Hz에서 실제 지급액50·활성 reward8개/C4 설정을 검증했다.
원본 event와 saved cfg/diff를 기준으로 추출한 수치는
[comparison.json](../outputs/contact_manipulation/reward_update_20261020/comparison.json)에 보존한다.
이번 설정으로 새 학습은 실행하지 않았다.

## 24. 2026-10-20 — Rearrangement 활성화·Blocker 제외·선반 이동 포함

사용자가 활성화한 Rearrangement weight0.6을 유지하고, 비용 계산에서 지정 Blocker를
제외했다. Target과 다른 active 물체는 초기 vision에서 보이지 않았더라도 포함한다.
Blocker의 이동·회전·pivot은 이 항의 비용이 아니다. Blocker와 같은 종류의 다른 instance가
있으면 해당 물체는 제외하지 않으며 episode의 역할 mask로 지정한 한 instance만 제외한다.

$$
x_R(t)=\sum_{i\in\mathcal O_{\mathrm{active}}\setminus\{B\}}
\frac18\sum_{k=1}^{8}\|p_{i,k}^{t}-p_{i,k}^{t-1}\|_2
+\frac18\sum_{k=1}^{8}\|p_{\mathrm{shelf},k}^{t}-p_{\mathrm{shelf},k}^{t-1}\|_2,
\qquad r_R(t)=0.6\left[\exp(-2x_R(t))-1\right].
$$

선반 전체의 대응 OBB8corner 이동량도 같은 meter 단위로 더해 병진·회전을 함께 평가한다.
현재 step의 조작량만 지급하므로 정지하면0이며 왕복은 각 이동 step에 비용이 남는다.
누적 `path_length`는 진단용이다. Clearance4·Terminal성공50/그외0과 선반 변위 종료
기준1cm/2°는 유지한다. 현재 활성 reward는9개다.

### 선반 geometry와 tensor 처리

실제 native rigid body `/World/envs/env_0/Shelf/rack`에서 공식
`UsdGeom.BBoxCache.ComputeUntransformedBound`와 기존 scale 처리를 사용해 전체 local OBB를
한 번 측정한다. 작업층의 사용 가능 공간을 선반 전체 크기로 대신하지 않는다.
실제 측정 bounds는 local lower(−.205,−.505,0), upper(.2035,.505,1.8)m,
크기는약(.4085,1.01,1.8)m였다.

이를 `SceneBatch.shelf_local_corners[N,8,3]`의 공유 view로 제공하고 reset 행만
TaskState 버퍼에 복사한다. 매 step 이미 읽던 native `root_link_pose_w`로 GPU에서
변환하므로 추가 PhysX pose 조회·USD 조회·CPU 전송·환경별 Python 반복은 없다.
초기 선반 위치를 원점으로 하는 고정 frame을 사용해 큰 clone 좌표에서 회전 이동량의
FP32 상쇄를 줄인다. 선반 pose 자체는 기존 world frame을 유지한다.

부분 reset은 해당 행의 이전 corner를 현재 선반 pose로 설정해 복원 teleport를
과금하지 않는다. Reset의 Rearrangement 품질도1로 채워 즉시 penalty0이 되게 했다.
관측은244D 그대로이며 선반의 현재 pose와 실제 물체 pose는 reward의 privileged 정보다.
새 production 파일은 추가하지 않았다.

### 검증

- CPU248pass·18skip·374subtest pass. Blocker 병진/회전 제외, Target·비가시 물체 비용,
  선반의 해석적 병진/yaw 거리, 정지0·왕복 비용, 큰 world 좌표, 비연속ID 부분 reset을 확인했다.
- RTX3090 CUDA1,024행에서 같은 조건과 partial reset을 통과했다.
  해석적 변위 대비 최대 오차3.73e−9m, 물체+선반 이동량 tensor 계산은평균.280ms였다.
  이 수치는 물리 simulation·전체 reset·PPO iteration 시간과 구분한다.
- 실제 Isaac Lab4환경·8step에서 물리 선반 bounds,9개 reward 활성화, 유한 관측/보상,
  native 부분 reset의 다른 환경 선반 이력 보존을 확인했다.

재현 코드·native 로그·JSON은
[rearrangement_20261020](../outputs/contact_manipulation/rearrangement_20261020/)에 보존한다.
새 reward 설정으로 정책을 학습한 결과는 아직 없다.

## 25. 2026-10-21 — OSC 작업공간 역행렬 singular 오류 조사

사용자 traceback의 직접 실패 지점은 native `operational_space.py:442`다.
현재 `inertial_dynamics_decoupling=True`, `partial_inertial_dynamics_decoupling=False`이므로
6축 전체의 작업공간 관성을 다음과 같이 계산한다.

$$
A=J M^{-1}J^\top,\qquad \Lambda=A^{-1}.
$$

`Batch element 939`의 A가 singular로 판정돼 batched inverse 전체가 예외를 발생시켰다.
`diagonal element 6`은 행렬 분해의 여섯 번째 pivot을 뜻하며 여섯 번째 관절 고장이 아니다.
이 traceback은 PhysX CUDA719나 GPU 메모리 부족 오류와 다르다. M 자체의 역산은 앞선431행이며,
이번에 실패한 것은 작업공간 A의 역산이다. 이상적인 양의 정부호 M에서는 J의 rank가 부족하면
A도 역산할 수 없고, near-singular 상태에서는 FP32 오차로도 같은 문제가 나타날 수 있다.
이 단계의 A는 현재 자세의 J·M에 의존하며 이번 action의 delta pose·Kp가 A를 직접 결정하지 않는다.
큰 action이 이전 step들을 거쳐 불리한 자세로 유도했을 가능성과 직접 원인은 구분한다.

IdealPD의 effort clamp는 OSC 계산 이후이므로 역행렬 예외를 막지 못한다.
Reward·termination 역시 물리 step 뒤에서 평가하므로 실패한 이 호출을 보호하지 못한다.
당시939행의 q·J·M·action이 저장되지 않았고 보고에 실행 디렉터리도 없어,
현재 오류의 정확한 자세나 특정 학습 실행과의 연결은 확정하지 않았다.

### 과거 저장 상태의 CPU 재계산

10월17일 `2026-10-16_21-36-16/model_900.pt` replay의 torque peak를 재계산했다.
이 데이터는 이번939행의 재현이 아니며 당시 frame·정책 설정도 다르다.

| 지표 | 과거 torque peak |
|---|---:|
| Elbow 위치 | 0.0304° |
| J 최소 singular value | 8.664e−5 |
| J condition number | 2.492e4 |
| A condition number, FP64 | 9.489e9 |
| FP64 재계산 motion torque 최대 | 78,127N·m |
| CPU FP32 재계산 motion torque 최대 | 2,674,976N·m |
| 당시 저장 GPU raw torque 최대 | 16,358,637N·m |

재계산 torque는 중력·actuator 항을 제외한 motion 성분이다. FP64에서도 큰 값이므로
실제 기하학적 특이점 근접과 FP32 수치 증폭이 함께 존재한다.
CPU FP32 역행렬 잔차의 Frobenius norm은645.85였다. FP32 기본 tolerance의 수치 rank5를
기하학적 rank5로 단정하지 않는다. FP64 J의 rank는6이다.
`train.py`가 TF32를 허용하지만 이6×6 CUDA 연산에서 실제 선택됐는지는 미확인이다.
TF32를 꺼도 실제 near-singularity는 사라지지 않는다. 역행렬·정밀도 관련 한계는
[PyTorch 공식 수치 정확도 문서](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)를 따른다.

같은 과거 상태에서 공식 partial decoupling은273.4N·m, decoupling OFF는86.2N·m의
motion torque를 계산했다. 이는 단일 상태의 대수 비교이며 제어 응답이나 학습 결과의 검증이 아니다.
Partial 모드도3×3 역행렬을 사용하므로 모든 특이점에 대한 해결이 아니다.
OFF는 해당 역행렬을 제거하지만 관성 보상과 gain의 제어상 의미가 바뀐다.

### 후속 수정 방향과 이번 조사 범위

[최신 공식 develop API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.controllers.html#isaaclab.controllers.OperationalSpaceControllerCfg.inertia_conditioning_thresholds)에는
`inertia_conditioning_thresholds`와 FP64 관성 계산·작은 고유값 방향의 점진적 damping이 있다.
공식 수정은 [Handle near-singular operational-space control (#7903)](https://github.com/isaac-sim/IsaacLab/commit/f75767005f64eb38042d2cb6ff3f7880f50649e8)이며,
호환성 비교 시 움직이는 branch 대신 이 commit의 변경을 기준으로 삼을 수 있다.
현재 설치본에는 그 설정과 구현이 없으므로 cfg 한 줄만 추가할 수 없다.
공식 구현의 호환 가능한 반영과 연산 비용 검증을 우선 검토한다. 전체 SDK 업그레이드나
partial/OFF 전환을 확인 없이 수행하지 않는다.

다음 실패 시 기존 OSC 버퍼의 q·qdot·J·M·raw/processed action을 예외 경로에서 저장하면
추가적인 매-step 진단 연산 없이 실제 실패 행의 유한성·rank·conditioning을 구분할 수 있다.
이 저장 경로는 아직 추가하지 않았다. Production 제어·reward·reset 코드는 이번에 변경하지 않았다.
수치 근거는 [archived_peak_cpu_numeric_audit.json](../outputs/contact_manipulation/osc_singular_20261021/archived_peak_cpu_numeric_audit.json)에 보존했다.

## 26. 2026-10-21 — invalid-state 급증과 raw action 이력의 증폭 조사

### 로그에서 확인한 시간 순서

최신 본 학습 `2026-10-20_23-36-51`은1,024환경·100Hz·rollout32이며955개 scalar
(iteration0–954)와 checkpoint950까지 저장돼 있다. 사용자의 singular 오류 보고에는
실행 경로·발생 시각이 없으므로 이 실행에서 발생한 예외라고 확정하지 않는다.

| Iteration | Value loss | invalid-state 종료 원인 snapshot |
|---|---:|---:|
| 700 | 2.39e20 | 0% |
| 735 | 9.48e25 | 0% |
| 738 | 6.10e29 | 0% |
| 739 | 3.76e34 | 0% |
| 740 | Inf | 1.724% |
| 741 | Inf | 75.812% |
| 749 | Inf | 86.371% |
| 750 | 1.69e14 | 0.293% |

첫 Inf와 첫 invalid는740회(2026-10-21 00:06:02 KST)이며, 유한한 값의 비정상 증가는
그보다 먼저 시작됐다. Value loss는207회7.28e6,216회8.43e16도 기록했다.
735–749회의 Gaussian mean std는0.53111136으로 동일했다. Reward별 scalar는 모두finite다.
저장 checkpoint0–950의 actor/shared/critic parameter와 Adam state도 모두finite였으며,
700 이후 parameter 최대 절댓값은약1.06이었다. 따라서 모델 weight 전체가NaN이 되거나
Gaussian std가 폭증했다는 설명은 이 실행의 근거와 맞지 않는다. 실제 rollout 관측·행동은
저장돼 있지 않아 value loss 증가가 input·value·bootstrap 중 어디서 처음 시작했는지는 미확정이다.

### invalid-state의 의미와 OSC 오류의 관계

현재 판정은 active 물체의 pose/velocity·초기 reference·선반 pose의 finite/유효 quaternion,
Target 수1개, Blocker overlap 유효성, 로봇 sample·raw F/T·센서 validity·raw action·
전체 articulation의 applied torque finite 여부를 합친다. Tactile의 불능 채널은−1로 mask하며
그 자체로 전체환경을invalid 처리하지 않는다. 큰 유한 값·F/T overrange·near-singular
Jacobian은 이 조건에 포함되지 않는다.

특이점 부근의 J·A도 전부finite일 수 있으므로 invalid=False여도 OSC inverse는 실패할 수 있다.
실행 순서는action/OSC→physics→termination이며 inverse 예외가 먼저 발생하면 이 step의
invalid 판정은 실행되지 않는다. 반대로 불안정한 OSC/physics가NaN/Inf를 만들면 이후invalid가
발생할 수 있다. 따라서 공통 수치 불안정과 연관될 가능성은 있지만 두 사건은 동치가 아니다.

Native TerminationManager는 종료한 행의 `_last_episode_dones`만 갱신하고 reset 시
전체환경 평균을 기록한다. Runner는 다시 해당rollout의reset로그를 평균한다.
75.8%는현재75.8%가NaN이라는뜻도,이번iteration의새실패비율도아니다.
여러 행이차례로실패하면마지막종료원인에쌓여곡선이급격히상승할수있다.
800step마다공통timeout이모든행의종료원인을덮어쓰므로25iteration 경계의급락도
기계적으로발생할수있다. 750회하락만으로새scene이안전하거나물리가복구됐다고해석하지않는다.

### 확인된 구조적 위험: previous raw action 되먹임

Observation은current/previous sample각각에정규화·범위제한없는18D raw policy action을포함한다.
이력은최종MLP로직접들어가고Gaussian mean도출력상한이없다. Hand와stiffness의실제적용값은
native action에서제한되지만이력에는그이전의raw값이들어간다.

CPU에서합성된작은크기의finite geometry·관절·센서값을고정하고,현재평균action만다음step의
current sample로넣으며previous72D전체를한칸씩넘겼다. Noise없이수행했다.
초기raw action이력은0이며모델·mask는실제checkpoint와같다.

| Checkpoint | 첫 출력 max abs | 20번째 출력 | 100번째 출력 | 200번째 출력 |
|---|---:|---:|---:|---:|
| 700 | 2.85 | 2.85 | 2.85 | 2.85 |
| 750 | 13.88 | 1.30e3 | 1.18e9 | 3.16e16 |
| 950 | 26.23 | 7.02e8 | 96번째에이미nonfinite | — |

950은50번째에2.65e20까지커졌고45번째부터value제곱이FP32범위를초과했다.
이는실제PPO loss가아닌합성진단의value제곱이다. 두action이력슬롯을계속0으로두면200회내내26.23으로
유지됐다. 즉이checkpoint들은물리·OSC를실행하지않아도raw action 이력을통해자가증폭할수있다.
Finite인상태에서먼저커지고나중에overflow하므로value loss 선행증가와invalid 후행발생을
설명할수있는구체적기전이다. 다만합성geometry·joint·EEF사이FK/충돌일관성은검증하지않았고,
실제740회관측의재현도아니다. 실제실패의최초원인으로확정하지않는다.

### 현재 판단과 다음 수정 우선순위

현재가장구체적으로재현된위험은raw action 이력의양의되먹임이며, OSC full inverse의특이점
취약성은별도로확인돼있다. 큰유한관절속도·action은현재invalid검사를통과해모델입력을키울수있다.
공유scene·정책은여러환경에비슷한실패를유발할수있지만현재snapshot만으로그비중은분리할수없다.

1. Raw action 이력의채널별스케일·입력범위를정하고정규화/제한을검토한다. Gaussian분포유지와
   별개인관측전처리변경이다. 실제적용명령과이력에보존할값의규약을명확히정해야한다.
2. 25절의공식OSC conditioning을호환가능하게반영하고gain·응답·시간을검증한다.
3. Invalid를raw/보상후F/T,로봇q/qdot/TCP,action,applied torque,물체,선반의하위원인으로
   분리하고최초이상의관측·행동·q/J/M을보존한다. Snapshot과별도로새invalid사건수를기록한다.
4. Native PPO 경로도비유한loss/gradient를감지해야한다. 현재nativeclip은
   `error_if_nonfinite=False`이며FP32norm이overflow하면유한gradient가0으로줄어들수있다.
   Adam1차moment가극히작아지고actor head가850→900→950에서동일했던점은갱신정체와
   양립하지만gradient는저장되지않아원인을확정하지않는다. 비유한값의치환만으로계속학습하지않는다.

이번조사는CPU로그/checkpoint분석과합성policy재현만수행했고production코드는변경하지않았다.
재현코드·JSON·[시간축plot](../outputs/contact_manipulation/invalid_state_20261021/timeline_focus.png)은
[invalid_state_20261021](../outputs/contact_manipulation/invalid_state_20261021/)에보존했다.

## 27. 2026-10-21 — Singularity penalty·개별 종료 추가, OSC 유지

사용자가 공식OSC수정은보류하고reward추가를요청한뒤termination도추가하도록지시했다.
현재설치된IsaacLab2.3.2기반fork의OSC·Actuator·ActionCfg·물리주기·초기IK·관측및모델은
변경하지않았다. Raw previous-action 입력의전처리나클리핑도이번범위에포함하지않았다.

### 지표와 계산 공유

공식 `OperationalSpaceControllerAction.jacobian_w`로현재물리step이후의wrist_3·팔6관절
Jacobian을읽는다. OSC의private캐시는물리step이전값이므로사용하지않았다.
이상태는world축이지만공통좌표회전은지표를바꾸지않으므로base/EEF회전을추가하지않는다.
Hand·mimic관절은지표에포함하지않는다.

$$
\bar J=\begin{bmatrix}J_v/L\\J_\omega\end{bmatrix},\qquad L=0.25\,\mathrm m,
\qquad s=\frac1{\|\bar J^{-1}\|_F}
=\left(\sum_{i=1}^6\sigma_i^{-2}\right)^{-1/2}.
$$

L은기존IK의위치/회전weighting규약과맞춘초기값이다. 기준길이를정하지않은m/rad혼합
Jacobian이나condition ratio를사용하지않는다. 정확한최소singular value가아니며
`sigma_min/sqrt(6) <= s <= sigma_min`인보수적대리값이다. 여러방향이약하면더낮게평가한다.

설치PyTorch의 `svdvals`는CUDA에서도CPU동기화를하므로hot path에는사용하지않는다.
공식 `torch.linalg.inv_ex(check_errors=False)`가오류정보를device에남기도록하고,
입력비유한행은역산전에identity로대체한다. 원래유효성·info·역행렬·norm유한성을검사해
실패행은s=0으로처리한다. 다른환경을중단하지않으며pseudo-inverse로약한방향을버리지않는다.
이역산은진단용이며OSC역산이나토크출력에는개입하지않는다.

기존 `get_task_state`의native control counter별1회갱신에연결했다. Reward·termination·
observation이재조회해도추가Jacobian조회나분해를하지않는다. 부분reset에서는해당행의
지표만경고기준값으로초기화해첫완료sample전에는penalty0으로두고,이후현재Jacobian을읽는다.
새production파일이나observation채널은추가하지않았다.

### 보상과 종료

$$
x_S=[\max(0,0.05-s)]^2,\qquad
r_S=0.15\{\exp(-a_S x_S)-1\},\qquad
a_S=\frac{\ln2}{0.025^2}.
$$

`motion_singularity`는정상구간에서0이며s=.025에서−.075/step,s=0에서−.140625/step이다.
위험자세에서는정지해도penalty가있다. 기존네Motion항의weight0.15는그대로두고다섯번째
독립항으로추가했다. 기본Motion합의최소기여는−.740625이고양의정지보상은없다.
다른reward의weight·계수는유지했으며현재활성reward는10개다.

Native TerminationTerm `singularity`는s≤.005에서즉시종료한다. 기존FAILURE_REASON_NAMES에
별도열로추가해aggregatefailure·terminal_paid·PPO terminated에도같은활성mask를반영했다.
종료만비활성화해도reward는유지할수있고,비활성원인은failure에기여하지않는다.
Success의Target보호/Blocker통로확보조건과Terminal성공+50·그외0은유지한다.
현재활성실패는5개이며success·timeout까지native종료term7개다.

로그는공식방식의 `Episode_Reward/motion_singularity`,
`Episode_Termination/singularity`로분리한다. 이지표는자세기반이며action이취약방향을향하는지
검사하는방향조건부항은아니다. 임계값은초기실험기준이지검증된안전한계가아니다.
실행순서OSC→physics→termination은유지하므로현재OSC호출의역행렬예외방지를보장하지않는다.

### 검증과 비용

- CPU전체256pass·18skip·379subtest pass. 해석적diagonal/bounds·좌표회전/단위불변성,
  singular/NaN/Inf행분리·정확한hard경계·종료비활성mask·native dt×weight·현재Jacobian1회조회·
  부분reset선택행초기화를검사했다.
- RTX3090 CUDA1,024행의해석해대비최대오차6.71e−8. 역산실패행을격리했고CUDAgraph
  capture/replay를통과해지표kernel내숨은CPU오류검사동기화가없음을확인했다.
  지표+penalty tensor계산은평균0.488ms였다.
- 실제native1,024환경·32physics/policy step과부분reset을통과했다. 관측/보상이finite였고,
  공식OSC `variable_kp`·full inertia와10reward/7termination활성화·로그키를확인했다.
  현재Jacobian native조회+지표계산은평균0.623ms/step이다. 전체PPOiteration측정은아니다.
- 이번한공통장면의1,024개초기자세에서s의최솟값.08245,중앙값.28547,최댓값.35506이었다.
  초기경고/종료비율은모두0%였다. 모든가능장면의초기자세안전을보장하는결과는아니다.
- 과거10월17일의큰토크peak 저장J에서는FP32 s=.00032883으로종료기준에걸렸다.
  다른peak는s=.04071496으로경고구간이며종료하지않는다. 이번실패939행의재현은아니다.

재현코드·JSON·native로그는
[singularity_20261021](../outputs/contact_manipulation/singularity_20261021/)에보존했다.
이번조건으로새정책의학습성능은아직측정하지않았다.

## 28. 2026-10-21 — 성공 종료의 유인과 후속 학습 하락 분석

분석snapshot은 `2026-10-21_11-09-26`의iteration2058/checkpoint2050이다.
새singularity항이활성화됐으며이전rewardweight·Terminal50·gamma.99·100Hz설정은유지됐다.

| 구간 | Mean episode return | Success snapshot | Singularity snapshot |
|---|---:|---:|---:|
| 600–699 | 124.01 | 12.63% | 3.04% |
| 800–899 | 66.72 | 20.23% | 18.00% |
| 1500–1599 | 38.65 | 10.22% | 13.41% |
| 1600–1699 | −13.60 | 0.003% | 77.02% |

낮은return이항상성공감소를뜻하지는않는다. Native reward로그는episode누적값/고정8초이며
Train mean_reward는최근종료episode합의평균이다. 성공해서짧아지면dense누적값이줄수있다.
그러나후반에는성공도거의0이돼단순한빠른성공으로설명되지않는다.
1575부근의singularity급증은붕괴와동시에나타나지만원인인지불안정한정책의결과인지미확정이다.
그전1400–1499회에는drop45.79%·tilt33.88%·singularity2.13%로다른실패도컸다.
종료비율은신규사건률이아닌마지막종료원인snapshot이며동시원인은겹칠수있다.
Invalid는전구간0이고모든scalar가finite였으나value loss최대5.09e17(1724회),
std가초반.309→후반2.762로커져학습불안정은남아있다.

코드순서는termination계산→최종dense및Terminal+50지급→reset이다. 성공보상소실이나
성공에음수보상을주는구현은없다. Success/failure는true terminal로미래가치를0으로하며,
timeout만종료전관측으로bootstrap한다. 이는정상적인처리다.

다만반복보상과완료보상의크기는성공지연을유리하게만들수있다. 예를들어성공step의
총보상R_s=51(일반보상1+성공50),미완료유지순보상c=1/step을가정하면,

$$
G_{\mathrm{now}}=51,\qquad
G_{\mathrm{wait\,100}}=
\frac{1-0.99^{100}}{1-0.99}+0.99^{100}\,51\simeq82.06.
$$

이는실제trajectory의수치가아닌구조적비교다. 일반적으로c>(1−gamma)R_s이면같은성공을
미루는것이유리하다. 현재C는초기대비개선상태를유지해도매step반복지급하고,D≤.4와
Hold≤.85도반복된다. 안정정지시R/M대부분0이며Time은꺼져있다. Target보호는success조건이지만
C/D/H지급gate는아니다. 실제정책이이유인을이용했는지는별도trajectory근거가필요하다.

따라서success termination자체를제거하거나timeout으로분류하지않는다. Clearance의실제진행량
지급과D/H지원항의지급단계·조건을검토해미완료유지수입을줄이는것이우선검토할수정안이다.
이번에는production설정이나수식을변경하지않았다. 상세scalar·재현코드는
[terminal_incentive_20261021](../outputs/contact_manipulation/terminal_incentive_20261021/)에보존했다.

### 사용자 후속 명확화 — 학습 후반에는 처음부터 접촉하지 않음

사용자는한episode안에서민뒤후퇴한다는뜻이아니라,학습초기정책은물체와상호작용하고
후기정책은처음부터물체를건드리지않는다고명확히했다. 따라서후기회피를기존Clearance
개선수익의유지로설명하지않는다. Untouchedscene의Clearance는정확히0이다.

1500–1574→1600–1699회에서episode길이는57.38→56.67step(−1.25%)인데,
Clearance로그4.6872→.00388(−99.917%),Distance1.9994→.1493(−92.53%),
Hold.13728→.0000178(−99.987%)로하락했다. Action-rate비용절댓값은.531→.578로증가했고,
return은47.59→−13.60,success snapshot13.22%→.00275%,singularity3.38%→77.02%였다.
단순한episode길이감소로설명하기어려운접근·조작관여의상실이다.
Hold는연속접촉지표이므로단발접촉까지전혀없었음을scalar만으로증명하지는못한다.

가능한기전은접촉시즉각비용·실패가자주발생해첫접촉을회피하는국소정책이다.
정상자세에서거리20cm정지시D=.1/step,R/M대부분0인데,초기접촉에는F/T증가·action-rate·
jointvelocity·부수물체이동비용이먼저발생할수있다. 실제Blockeroverlap이개선돼야C가생기며
goal50도할인된다. 다만안전하고안정적인근접/접촉은같은C의먼거리정지보다D/H가높으므로,
후퇴를무조건최적해라고판정하지않는다. 정책분산·value loss증가와조작능력붕괴도함께검토한다.
원인분리는동일초기장면에서초기/후기checkpoint의첫접촉전누적보상·행동·종료원인을
비교해야한다. 이번에는code나weight를변경하지않았다. 추가집계근거는
[retreat_20261021](../outputs/contact_manipulation/retreat_20261021/)에보존했다.

## 29. 2026-10-21 — Critic을 공식 독립 MLP로 분리

사용자는 학습 후반의 return·접근·접촉 감소가 안정적인 회피 정책의 수렴이 아니라
정책의 퇴행일 수 있다고 지적했고, 공유 Critic의 영향을 분리하기 위해 일반적인 Isaac Lab
예제와 같은 Critic을 요청했다. 기존 공유 구조가 최초 붕괴 원인이었다고 확정한 변경은 아니다.

### 모델과 native API 변경

Critic은 `RslRlMLPModelCfg(hidden_dims=[256,128,64], activation="elu",
obs_normalization=False, distribution_cfg=None)`로 공식 RSL-RL `MLPModel`을 생성한다.
이는 로컬 Sweeping 예제와 Isaac Lab Franka Lift·Cabinet 예제의 Critic hidden 크기·활성화·
관측 정규화 설정을 따른다. 설치된 RSL-RL5.0.1의 분리 모델 API에서
`critic_class(obs, obs_groups, "critic", 1, **critic_cfg)`로 구성한다.

| 항목 | 기존 | 변경 후 |
|---|---|---|
| Actor | GNN·역할 attention·ReLU fusion→18D Gaussian | 동일한 계산과 출력 분포 유지 |
| Critic | Actor의 scene encoder·fusion을 공유하는 value head | 공식 독립 MLP `244→256→128→64→1` |
| Critic hidden activation | 공유 ReLU 표현 | 예제의 ELU |
| 입력 | 공개244D 관측 | 동일한 공개244D 관측, privileged 채널 추가 없음 |
| Critic observation normalization | 사용 안 함 | 사용 안 함 |
| 파라미터·latent·value cache 공유 | 공유 | 공유 없음 |

Actor 안의 value head·`value()`와 별도 `NativeContactCritic` 클래스를 제거했다.
Actor는 action mean만 계산하며 Critic은 관측을 직접 받아 scalar value를 계산한다.
공유 모델에서 Actor 직후 값을 재사용하던 cache도 제거했다. Observation 기반 mask는
기존처럼 Actor GNN의 padding·역할·결측 처리에 사용하며 Critic 입력 채널로 추가하지 않는다.

Native optimizer는 서로 겹치지 않는 Actor·Critic 파라미터를 모두 관리한다.
기본 `PPO.update()`를 유지하고 큰 minibatch의 기존 microbatch 경로도 양쪽 loss를 누적한다.
Gradient clipping은 native PPO와 같이 Actor와 Critic에 각각 적용한다.
이제 value loss가 Actor의 GNN·attention·fusion에 직접 gradient를 주지는 않는다.
다만 Critic의 값이 GAE·advantage에 사용되므로 Actor 학습에 대한 간접 영향까지 사라진 것은 아니다.

PPO 하이퍼파라미터·loss 수식·timeout 최종 관측 bootstrap, reward와 weight·scale,
controller·action 계약·244D observation·18D action·MDP·physics/policy 주기는 유지했다.
새로운 학습 성능이나 수치 안정성 개선이 확인됐다는 의미는 아니다.

### Checkpoint와 실행 기준

기존 공유 checkpoint는 Actor 내부 value head, 비어 있는 별도 Critic state dict,
공유 optimizer 상태를 저장하므로 새 독립 구조의 전체 resume와 호환되지 않는다.
Architecture metadata로 이를 구분하고 **새 학습을 시작한다**.
이전 공유 checkpoint를 현재 play에서 호환 로드하는 기능도 이번 변경에 포함하지 않았다.
학습·재생 예제 run 이름은 `independent_critic`으로 구분한다.

### 검증과 범위

- CPU 전체 검사: **259pass·18skip·381subtest pass**.
- Actor의 CUDA compile 검사 **1pass**. Actor GNN·attention 계산을 유지한 상태로 검사했다.
- 실제 native **1,024환경×2iteration** 학습을 exit0으로 완료했다. 공식 독립 `MLPModel`
  구조가 콘솔에 출력되고 rollout·PPO update·로그·checkpoint 저장까지 진행됐다.
- 저장된 checkpoint의 Actor234,660개·Critic103,937개 파라미터와 독립 architecture metadata를
  확인했다. Model·optimizer tensor 및 TensorBoard scalar는 모두 finite였다.
  같은 checkpoint를 공식 runner로 로드해 **4환경×8control step** play도 exit0으로 완료했다.
- 첫 iteration은 collection19.360초·learning10.603초, 합계29.96초로 최초 compile을 포함한다.
  다음 iteration은 collection1.662초·learning.205초, 합계1.87초였다.
  **두 iteration의 실행 확인이며 장기 평균 처리량, 공통 scene 교체 비용이나 정책 성공을
  입증하는 benchmark는 아니다.**

재현 로그는
[native_train.log](../outputs/contact_manipulation/independent_critic_20261021/native_train.log),
실행은 [2026-10-21_13-27-47_independent_critic_smoke](../logs/rsl_rl/contact_manipulation/2026-10-21_13-27-47_independent_critic_smoke/)에 보존한다.
Checkpoint·play 확인 결과는 [native_result.json](../outputs/contact_manipulation/independent_critic_20261021/native_result.json)에 기록했다.
이번 smoke는 기능 검증이므로 본 학습의 reward·success 경향 분석에 포함하지 않는다.

## 30. 2026-10-21 — 독립 Critic 학습의 선반 종료 급증과 reward scale 재검토

분석 snapshot은 독립 Critic 본 학습 `2026-10-21_13-33-02`의iteration1843까지다.
사용자의 선반 이동 종료 증가와 reward weight·scale 가설을 기존 코드·저장 설정·scalar로 검토했다.
**이번에는 production 코드·reward 수식·weight·종료 기준을 변경하지 않았다.**

### 선반 종료 급증 전후의 상태

| Iteration 구간 | Mean return | Distance 로그 | Hold 로그 | Shelf 종료 snapshot | Singularity 종료 snapshot | Mean std |
|---|---:|---:|---:|---:|---:|---:|
| 1775–1799 | −22.52 | 0.437 | 0.000636 | 7.37% | 69.05% | 56.99 |
| 1800–1824 | −14.64 | 0.479 | 0.000550 | 32.04% | 47.10% | 63.97 |
| 1825–1843 | −14.72 | 0.433 | 0.001021 | 4.30% | 76.39% | 70.90 |

선반 종료가 급증하기 전부터 접근·지속접촉 보상이 작고 return이 음수이며 singularity 종료가
많았다. 따라서 선반 종료 증가만을 최초 붕괴 원인으로 볼 수 없다. 평균 std는 초기.300에서
최종73.856으로 커졌고, value loss는403회에 이미1.827e11로 처음1e6을 넘었다.
독립 Critic은 value loss의 직접적인 공유 encoder 역전파를 제거했지만 학습 불안정을
단독으로 막지는 못했다. 이 결과가 공유 Critic의 영향이 전혀 없었다는 증명도 아니다.

`Episode_Termination/*`는 각 환경의 마지막으로 끝난 episode 원인 snapshot이며,
해당 iteration에 새로 발생한 실패율이나 현재 물리 상태의 불량률이 아니다.
여러 원인이 동시에 참일 수 있다. `Episode_Reward/*`는 weighted episode 누적값을 고정8초로
나눈 로그이며 순간 보상이나 실제 생존시간으로 나눈 평균이 아니다. `Train/mean_reward`도
최근 종료 episode 집계여서 동일한 cohort라고 단정하지 않는다.
공통 배치는800policy step, 즉 rollout32에서25iteration마다 바뀐다. 1800·1825 경계의
가파른 변화에는 장면 교체와 snapshot 갱신이 함께 작용할 수 있다.
현재 scalar에는 실제 선반 pose 궤적·생성된 실패 사건 수·warning 구간 체류시간이 없어
충돌 시점과 보상 간의 인과관계를 확정할 수 없다.

### 선반 penalty는 존재하지만 cm 단위 민감도가 작음

현재 Rearrangement는 지정 Blocker를 제외한 active 물체와 선반의 **이번 step** 대응 OBB
corner8개 변위 norm 평균을 합쳐 `0.6(exp(−2ΔL)−1)`을 지급한다. Target과 관측되지 않은
주변 물체도 포함한다. 선반의 초기 pose 대비 순변위를 직접 입력하는 항이 아니다.

| 선반의 이번 step 순수 병진 이동 | 선반만 움직였을 때 R 지급 |
|---|---:|
| 0.1 mm | −0.000120 |
| 1 mm | −0.001199 |
| 5 mm | −0.005970 |
| 10 mm | −0.011881 |

1cm를100step에 나눠 움직여도 할인 전 누적은 약−.012다. 초기 위치에서9mm 밀린 채
멈추면 이후 이동량은0이므로 이 penalty도0이다. 그러나 종료는 초기 위치 대비1cm 또는
회전2° 초과에서 즉시 발생한다. `a_R=2m⁻¹`의 half-quality 이동량은약34.7cm로,
종료 전 cm 단위 이동에 대한 감점 기울기가 작다. 정상 범위에서 지수항이 포화해서
정보를 잃는 상황은 아니다. OBB는 실제 선반 전체0.4085×1.01×1.8m를 사용한다.

비교하면 Distance는최대+.4/step, Hold는최대+.85/step이고, 가림비율.50→.49만 개선해도
Clearance가약+.02438/step 반복 지급된다. 성공은+50이며 failure만 발생하면 terminal은0이다.
성공과 선반 failure가 동시에 참일 때도 기존 사용자 결정대로+50을 지급한다.
이러한 크기 비교는 선반 이동 비용이 약할 가능성을 뒷받침하지만, 실제 정책의 행동 원인이나
바람직한 최종 weight를 계산한 것은 아니다.

선반 corner는 reset 행의 초기 pose와 같은 frame으로 이력을 초기화하며,
native 실행은 종료 판정→마지막 reward 계산→reset 순서여서 마지막 이동 비용이 누락되지 않는다.
정상적인 finite 선반 이동에서 reset 이력·reward 지급의 구현 오류는 찾지 못했다.
NaN/Inf 이동량을0으로 치환하는 경로는 있지만 별도 invalid 종료가 처리하며, 일반적인
finite 선반 이동 급증을 설명하는 근거는 없다.

### Singularity 로그의 크기와 경고 시점

현재 penalty는 `0.15(exp(−a_S[max(0,.05−s)]²)−1)`,
`a_S=ln(2)/.025²`이며 `s≤.005`에서 종료한다.

| 현재 지표s | Weighted penalty |
|---|---:|
| .05 이상 | 0 |
| .04 | −.01575 |
| .025 | −.075 |
| .005 | −.13412 |

Reward와 종료는 같은 post-physics 지표를 공유하므로 사전 경고 구간은 존재한다.
다만 평균 episode 로그가 작다는 이유만으로 곡선이 잘못됐다고 판단하지 않는다.
대부분 정상 구간에 있었거나 warning 구간을 짧게 통과한 뒤 종료해도 평균은 작을 수 있다.
경고 진입 이후 몇step 안에 종료되는지 확인해야 경고 시점과 크기 중 무엇이 부족한지 구분된다.

### 미적용 분리 실험 후보

기준 비교에서는 Clearance4·Distance.4·Hold.05·Terminal성공+50과 나머지 네 Motion의
weight각.15를 유지한다. 여러 penalty를 한꺼번에 키워 접촉 회피를 악화시키지 않도록
다음 후보를 **각각 별도 실험**으로 비교한다.

- **Rearrangement:** 우선 weight.6을 유지하고 `a_R=2→20m⁻¹`로 cm 단위 기울기를 비교한다.
  1mm는−.01188,1cm는−.10876이고 최대 비용은−.6으로 같다. 전체 R weight를 올리는 것과 달리
  상한을 그대로 두지만 Target·주변 물체 이동 비용도 커지므로 접근·접촉 상실을 함께 감시해야 한다.
- **Singularity:** 먼저 weight.15를 유지하고 warning threshold`.05→.07`, half-deficit`.025→.03`
  (`a_S=ln(2)/.03²`)으로 경고를 앞당기는 후보를 검토한다. 예를 들어s=.06에서약−.01112,
  s=.04에서−.075가 된다. 적용 전 새 장면·로봇 reset의 지표 분포를 검사해 초기 자세를
  무차별 감점하지 않는지 확인해야 한다. 종료 기준`.005`는 우선 유지한다.
  Weight`.15→.3`은 적절한 warning 구간이 확보돼도 비용 크기가 부족한 경우의 **후속 조건부 후보**이며,
  처음부터 함께 적용하는 기본 제안이 아니다.
- **선반 별도 조기 경고:** 초기 pose 대비 변위·회전에 warning 구간을 두면,9mm 이동 후 정지한
  상태의 위험도도 표현할 수 있다. 기존 shelf pose를 재사용하는 소량의 tensor 계산으로 가능하지만
  별도 수식·term 설계가 필요하며 아직 구현하지 않았다.

어느 후보도 std·value 폭증이나 기존 raw action 이력의 되먹임 문제를 해결한다고 주장하지 않는다.
보상 감도를 실험하는 동안 같은 장면에서 평균 행동과 sampling 행동의 차이, raw action·std·value,
선반 변위와 warning 체류시간을 분리해 확인해야 한다.

설정·구간 집계·분석 snapshot은
[scalar_analysis.json](../outputs/contact_manipulation/reward_scale_20261021/scalar_analysis.json)에 보존했다.
현재·후보 곡선의 수치 비교는 [sensitivity.json](../outputs/contact_manipulation/reward_scale_20261021/sensitivity.json),
그림은 [reward_curves.png](../outputs/contact_manipulation/reward_scale_20261021/reward_curves.png)와
[latest_training.png](../outputs/contact_manipulation/reward_scale_20261021/latest_training.png)에 보존했다.
후보 곡선은 계산상 비교이며 변경한 설정으로 학습한 결과가 아니다.

## 31. 2026-10-21 — Rearrangement·Singularity weight와 경고 곡선 조정

30절의 검토 후 사용자가 조정을 요청해 두 항의 weight와 곡선을 함께 변경했다.
30절은 이전 설정을 평가한 기록으로 유지하며, 이번 변경은 각 효과를 나누어 측정한
분리 실험과 구분한다. 학습 퇴행이나 std·value 폭증의 원인이 해결됐다는 판단은 아니다.

| 설정 | 이전 | 적용 |
|---|---:|---:|
| Rearrangement weight | .6 | 1.0 |
| `rearrangement_decay` | 2m⁻¹ | 20m⁻¹ |
| Singularity weight | .15 | .30 |
| `motion_singularity_threshold` | .05 | .07 |
| Singularity half-deficit | .025 | .03 |
| `motion_singularity_decay` | `ln(2)/.025²` | `ln(2)/.03²` |

### 지급되는 값과 유지한 조건

Rearrangement는 지정 Blocker를 제외한 active 물체와 선반의 직전step OBB corner 이동량을
합친 x_R에 대해 `exp(−20x_R)−1`을 지급한다. 비용 정의·회전 반영·선택 행 reset 이력은 유지한다.
Target·다른 물체 이동 비용도 함께 커지므로 접촉 회피 재발 여부를 함께 확인해야 한다.

| 이번 step x_R | 이전 지급 | 조정 후 지급 |
|---|---:|---:|
| 1mm | −.001199 | −.0198013 |
| 1cm | −.011881 | −.181269 |

이는 이번 step의 이동 비용이며 초기 pose 대비 순변위 penalty로 바꾼 것은 아니다.
선반이9mm 이동한 채 멈추면 이후 이동 비용은 여전히0이다. 선반 pose의 별도 조기 경고
term은 추가하지 않았다.

Singularity는 다음을 지급한다.

$$
r_S=0.30\left\{\exp\left(-\frac{\ln2}{.03^2}[\max(0,.07-s)]^2\right)-1\right\}.
$$

| 현재 지표s | 조정 후 지급 |
|---|---:|
| .07 이상 | 0 |
| .06 | −.0222376 |
| .04 | −.15 |
| .005 | −.288413 |
| 0 | −.293111 |

이전보다 큰 지표에서 경고가 시작되며, penalty 크기도 확대했다. 지표 정의와 계산 공유는
유지하며 종료는기존`s≤.005`다. 초기 reset 행은 첫 완료physics sample 전까지 warning 기준값으로
초기화해 penalty0을 유지한다. 새 warning 기준에 대한 실제 초기 자세 검증은 아래에 따로 기록한다.

Clearance4·Distance.4·Hold.05(17pad 합산), 나머지 네 Motion각.15,
Terminal성공+50·그외0, 선반1cm/2°와 singularity.005 종료 조건은 유지했다.
Controller·action·관측·물리 주기·PPO 하이퍼파라미터·독립 Critic 구조도 바꾸지 않았다.
현재10개 reward term의 구성은 같고, Motion 합의 하한은약−.893111,
Rearrangement의 하한은−1이다.

기존 tensor 식의 설정 상수만 변경했다. 센서 조회·physics step·추가 kernel 연산이나
새 production 파일을 추가하지 않는다. 기존 로그 키를 그대로 사용한다.

### 검증 상태

CPU 전체 검사는 **259pass·18skip·381subtest pass**, 소요15.40초였다.
실제 native 검사에서는 첫장면 **1,024환경×32holdstep(32,768transition)**의 관측·보상이
finite였고, 새 Rearrangement·Singularity의 native manager weight와 dt 적용 및
부분reset의 다른 행 보존을 확인했다. 결과의 `reward_and_reset_checks_passed`는true다.

이와 별도로 새 장면3개에서 reset시점의 native SDK Jacobian 지표를 조회했다.
**3개 장면 각각32physics step을 실행한 결과는 아니다.** Hold 실행은 첫장면에만 수행했다.

| 장면 | SDK 지표s 최솟값 | 기존 warning s<.05 | 새 warning s<.07 | 기존 hard s≤.005 |
|---|---:|---:|---:|---:|
| 0 | .0824477 | 0/1,024 | 0/1,024 | 0/1,024 |
| 1 | .000110765 | 229/1,024(22.36%) | 371/1,024(36.23%) | 2/1,024(.1953%) |
| 2 | .0588610 | 0/1,024 | 3/1,024(.2930%) | 0/1,024 |

**초기 SDK 지표에서 기존 종료 기준도 걸리는 행이 확인됐다.**
따라서 결과의 `all_initial_starts_above_termination`은false이며, 전체 초기 자세가 정상이라는
검증으로 해석하지 않는다. 특히 장면1에서는 warning 확대가 초기 감점 대상 행을22.36%에서
36.23%로 늘린다. 첫장면의 정상 분포를 모든 장면의 초기 상태로 일반화할 수 없다.

기존 초기 IK는 이 singularity 지표를 직접 검사하지 않는다. 다만 reset시점 SDK 읽기의
frame/기준점·갱신 시점 영향과 실제 자세를 아직 구분하지 못했다.
추가 analytic Jacobian 대조도 native 값과 최대약.0229m 차이가 남아 기준점·SDK refresh를
완전히 설명한 검증이 아니었다. 이 차이만으로 실제 초기 자세가 원인이라고 단정하지 않는다.
현재 범위에서는 production IK·초기 sampling 조건을 변경하지 않았으며, 해당 구분은 후속 과제다.

확정 결과는 [native_result.json](../outputs/contact_manipulation/reward_adjustment_20261021/native_result.json),
실행 로그는 [native_validation.log](../outputs/contact_manipulation/reward_adjustment_20261021/native_validation.log)에 보존했다.
이 검사는 수식·연동·일부 reset과 초기 SDK 분포의 확인이며, 조정 후 학습 성능이나
장기 안정성·모든 초기 자세의 건전성은 아직 확인하지 않았다.

## 32. 2026-10-21 — 물체 튕겨내기와 성공 판정의 허점 검토

사용자는 play에서 Blocker와 주변 물체를 정상적으로 밀어 재배치하기보다 튕겨내는 행동을
관찰했고, 이런 동작이 성공으로 집계될 가능성을 지적했다. 허용할 결과는 Target 접근 공간을
만들면서 물체가 선반에 남는 조작이다. 이번에는 기존 판정의 반례와 수정 우선순위를 검토했으며,
**production 코드·reward·weight·성공·종료 조건은 변경하지 않았다.**

### 현재 성공 조건이 보장하지 않는 것

현재 성공은 지정 Blocker의 XY접근영역 교차면적≤1e−5m²와 Target초기pose대비
이동≤2cm·회전≤.15rad를10step(.1초) 연속 만족하는 것이다. Blocker의 높이·작업영역 잔류·
선반 지지·선속도·각속도, 주변 물체의 비산 여부는 성공 gate가 아니다.
`stable_linear_velocity=.015m/s`, `stable_angular_velocity=.1rad/s`가 설정에 남아 있지만
현재 성공 판정에서는 사용하지 않는다. 현재의 stable count는 정지 확인이 아닌
통로 확보·Target 보호 조건의 지속 횟수다.

낙하는 물체 OBB최저점이 선반 지지면보다6cm 넘게 낮아졌을 때 검사하며, 높이 들림과
작업영역 이탈 자체는 종료하지 않는다. 전도는80° 이상이.2초 지속돼야 종료한다.
따라서 공중에 있거나 막 넘어지기 시작한 물체가 낙하·전도 실패보다 먼저 성공할 수 있다.
단순 자유낙하 예에서 초기 수직속도0이면.1초 낙하량은약4.905cm,
6cm까지는약.1106초다. 이는 시간차를 설명하는 계산이며 실제 PhysX 재생 결과는 아니다.

성공과 활성 failure는 독립 판정하며 같은step에둘다참이어도 Terminal은+50을 지급한다.
기존 사용자 결정대로 구현된 동작이지만, 물체를 잃거나 선반을 움직인 완료를 배제하려는
이번 요구와 충돌한다. 실패 원인만으로 음수 terminal이 지급되는 구조도 아니다.

### Production 판정 함수의 CPU 반례

현재 `task_conditions()`와 `terminal_transition()`을 합성 tensor 상태에 직접 호출했다.
정책·physics 재생이 아니며 해당 checkpoint에서 실제로 발생한 궤적을 복원한 검사는 아니다.

| 입력 상태 | 현재 성공 | 현재 failure | 최초 terminal 지급 |
|---|---|---|---:|
| 지지된 정지 Blocker가 통로를 확보한 대조군 | 참 | 거짓 | +50 |
| 공중의 Blocker가2.245m/s·10rad/s로 이동하며.1초 통로 확보 | 참 | 거짓 | +50 |
| Blocker가 작업 XY영역 밖에 있으나 통로·Target 조건 충족 | 참 | 거짓 | +50 |
| 선반 밖 자유낙하.1초로 최저점이 지지면보다4.905cm 낮음 | 참 | 거짓(6cm미만) | +50 |
| 새로90° 기울어진 상태가.1초 지속 | 참 | 거짓(.2초유예중) | +50 |
| 성공 조건과 `object_drop`이 같은step에 처음 충족 | 참 | 참 | +50 |
| 성공 조건과 `object_tilt`가 같은step에 처음 충족 | 참 | 참 | +50 |
| 성공 조건과 `shelf_displacement`가 같은step에 처음 충족 | 참 | 참 | +50 |

전도 반례를 성공 reset 없이.2초까지 계속 계산하면 전도 failure도 발생한다.
하지만 native 환경은.1초 성공에서 이미 reset하므로 이후 실패가 잘못된 완료를
사후 취소하지는 못한다. 실제 episode에서 성공은 한 번만 지급한다.
반례와 재현방법은
[predicate_counterexamples.json](../outputs/contact_manipulation/flinging_review_20261021/predicate_counterexamples.json)에 보존했다.

### Reward·action 구조와 학습 로그의 해석

Clearance는 XY투영만 보므로 실제 지지 여부를 구분하지 않는다. **순수 수직 이동만으로
XY가림이 줄어드는 것은 아니다.** 다만 수평으로 튕겨나가거나 회전해 투영면적·교차면적이
바뀌면, 여전히 공중·고속인 상태에서도 Clearance가 증가할 수 있다.
지정 Blocker는 Rearrangement에서 제외된다. 이는 정상적인 밀기·pivot을 허용하려는 규칙이며
비산 방지의 물리 유효성까지 검사하는 규칙은 아니다.

다른 active 물체의 이동은 R에 들어가지만 현재 R은 이번step 조작량의 합에 지수를 적용한다.
동일한 총이동량.1m를 한 번에 만들면 `exp(−20×.1)−1≈−.8647`,10step으로 균등 분할하면
누적약−1.8127,100step이면약−1.9801이다(weight1,할인전).
큰 순간 이동이 step당 비용 포화로 상대적으로 싸지는 구조가 있어, 별도 유효성 판정 없이
R weight만 키운다고 비산을 배제할 수 없다. 동일한 측정 경로길이를 가정한 수식 비교이며,
실제로 튕긴 물체는 reset까지 계속 이동 비용이 누적될 수 있다. 한 번 충돌하면 비용이
한 번만 발생한다는 의미나 실제 비산 행동의 원인을 입증한 것은 아니다.

Checkpoint1350의 위치 Gaussian std는[3.6202,4.0207,3.7551], 회전은[6.5975,6.7027,7.2941]이었다.
Pose scale1.0·unclipped Gaussian이므로 학습 sampling의 위치 명령 std는약3.6–4.0m,
회전 명령 std는약6.6–7.3rad다. 실제 EEF이동이나 action 궤적을 측정한 값은 아니다.
Play 기본값은평균μ를 사용하므로 이 std만으로 deterministic play의 특정 비산을 설명할 수도 없다.
Task 유효성과 별도로 action scale·탐색분산·entropy·제어기 제한의 영향을 진단할 필요가 있으나,
이번 검토에서 OSC 수정이나 controller 교체를 적용하지 않았다. 수식·checkpoint 근거는
[reward_bias.json](../outputs/contact_manipulation/flinging_review_20261021/reward_bias.json)에 보존했다.

`2026-10-21_15-33-47`을1804회까지 포착한 scalar에서 checkpoint1350직전50회
(1301–1350)의 success snapshot은44.76%,drop36.11%,tilt7.63%였다.
이 값은 각 원인의 주변집계이며 동일episode에서 성공과 실패가 겹쳤다는 뜻은 아니다.
기존에 선택한 상위 checkpoint는 scalar 기준의 평가 후보일 뿐 올바른 밀기 동작을 검증한
정책이 아니다. 사용자의 play 관찰은 허용 조작 여부를 판단하는 직접 증거이며,
현재 scalar만으로 잘못된 성공의 비율을 계산할 수는 없다.
근거는 [metrics.json](../outputs/contact_manipulation/flinging_review_20261021/metrics.json)에 둔다.

### 미적용 수정 우선순위

1. **최종 성공의 물리적 유효성부터 정의한다.** 기존 통로·Target보호에 더해 Blocker가
   실제 작업 선반의 지지영역에 남고, 완전히 넘어진 자세가 아니며,
   낮은 선속도·각속도로.1–.2초 유지되며,
   활성 failure가 없는 완료를 성공으로 삼는 방향이다. 주변 물체를 잃거나 튕겨낸 상태도
   유효한 완료에서 제외한다. 허용 속도·지지 tolerance·최종 유지시간의 수치는 후속 확정이 필요하다.
   조작 중 전도의.2초 종료 유예는 유지할 수 있지만, 유예 중인 완전 전도 자세가 먼저
   최종 성공을 통과하지 않도록 완료의 자세 조건과 실패 debounce를 구분해야 한다.
2. **Success와 failure를 동시에 참으로 완료 처리하지 않는다.** 유효하지 않은 완료에는
   +50을 지급하지 않도록 판정과 terminal 지급을 일치시킨다. 별도 음의 failure weight를
   추가할지는 아직 결정하지 않았다.
3. **양의 dense Clearance에 물리 유효성 gate를 검토한다.** 지지·잔류를 잃어 통로를
   확보한 경우 양의 보상을 지급하지 않되 음의 Clearance와 R/M 비용은 유지한다.
   미는 도중의 속도까지 최종 정지 기준으로 gate하면 필요한 조작을 막으므로,
   settled 속도는 완료 조건에 두고 진행 보상의 gate는 물리적인 지지·잔류 위주로 구분한다.
4. **Blocker의 정상 이동 R제외는 유지한다.** 대신 비산·이탈의 유효성 검사는 Blocker도
   포함한다. 모든 일시적 lift·pivot을 즉시 종료하거나 처음부터 정지를 요구하는 방식은 피한다.
   R의 물체별/step별 비용 포화·분할 편향은 이와 별도의 후속 수식 검토 대상으로 남긴다.

이미 보유한 물체 pose·OBB corner·속도·선반 pose와 실제 작업판의 지지 metadata를 활용하면
GPU tensor 계산으로 구성할 수 있다. 추가 contact sensor·환경별 Python 반복문·매step CPU
logging이 필요한 수정은 아니다. 다만 OBB의 지지 판정은 실제 비볼록 mesh 접촉의 근사이며,
**선반 전체1.8m높이 OBB를 작업판 지지영역으로 쓰면 안 된다.** 실제 작업 선반의 XY·지지면
높이와 움직인 선반 기준 변환을 사용해야 한다.

검증에는 raw success와 failure·공중·영역밖·고속 상태가 같은 종료에서 겹쳤는지 확인하는
최소 GPU episode counter를 추가하는 방향을 검토한다. 전체 step을 CPU로 기록하는 디버그
구조는 필요하지 않다. 합법적인 밀기·pivot·회복과 비산 반례를 나누어 검사한 뒤
이전 checkpoint의 성공률도 새 판정으로 재평가해야 한다.

## 33. 2026-10-21 — 비산을 성공과 보상에서 분리하는 구체 수정안

사용자가32절의 검토를 구체적인 수정안으로 정리하도록 요청했다.
**아래는 작성 당시의 구현 전 계획이다. 이후 적용 내용은34절을 따르며, 완료 유지시간.2초 제안은 사용자 요청으로.5초로 대체됐다.**
첫단계는 성공·진행보상의 물리적 의미를 바로잡는 범위로 한정하고,
Rearrangement 수식과 action·탐색 설정은 이후 별도 실험으로 구분한다.

### 기준 기하와 높이 gap

실제 작업판과 기둥에서 측정한 usable XY영역·지지면을 현재 선반 frame으로 저장한다.
물체의 기존 OBB corner를 현재 선반 pose의 역변환으로 같은 frame에 표현해 선반이
움직여도 지지 판정의 기준이 함께 움직이도록 한다. 선반 전체1.8m높이 OBB는 쓰지 않는다.

물체i의 위쪽 들림 gap은 다음처럼 한쪽 방향으로 계산한다.

$$
g_i=\max\left(z_{i,\min}^{\mathrm{shelf}}-z_{\mathrm{panel}}
-b_{i,\mathrm{rest}},\;0\right).
$$

`abs(z_min−z_panel)`을 쓰지 않아 작은 침투·solver 오차를 공중 들림과 혼동하지 않는다.
선택적인 rest bias는 **검증된 정지 지지 자세의 asset별 고정 보정값**으로만 얻는다.
임의의 reset높이를 기준으로 삼으면 잘못 생성된 공중 상태를 정상화하므로 그렇게 하지 않는다.
기존 bounds가 지지면과 맞으면 bias는0이다. 구현 전에 resting asset·정상밀기·pivot의
gap을 먼저 측정해 아래 mm후보가 실제 기하·contact offset과 맞는지 확인한다.
OBB는 실제 비볼록 mesh 접촉의 근사여서 이 판정을 접촉의 증명으로 해석하지 않는다.
특히 위 식만으로는 판 아래로 침투한 상태도 gap=0이 된다. 따라서 최종 지지 유효성은
이 gap 상한만으로 확정하지 않고, signed gap과 실제 collision 형상을 대조해 얻은 정상
지지 자세의 허용 하단 범위도 함께 검사한다. 이 하단 범위는 asset·허용 자세의 검증된
지지 기하를 기준으로 정하고 현재 reset 높이로 학습하지 않는다. 보수적인 OBB 범위가
합법적 pivot을 거절하는지도 calibration에 포함한다. 아래 높이 값은 이 지지 일관성 검사를
통과한 상태에서 허용할 **위쪽 들림의 후보 상한**이다.

### 진행 유효성·최종 완료·실패를 별도 판정

| 목적 | 초기 후보 조건 | 시간·처리 |
|---|---|---|
| 진행 유효성 Vp | 모든active물체 OBB중심이 작업판XY내부, XYbbox가 판과겹침, gap≤10mm, 현재80°이상전도·invalid없음 | 매step즉시판정, 속도제한없음 |
| 최종 완료 | 모든active물체 XYbbox가 작업판내부(+5mm허용), 검증된 하단 범위·gap≤5mm, 현재tilt<80°, 선속도≤.05m/s·각속도≤.3rad/s | 기존통로·Target조건과함께.2초유지 |
| `object_ejected` 실패 | 같은active물체의 XYbbox가 작업판과 완전히 분리되고 경계 gap>2cm | .05초연속 |
| `object_off_support` 실패 | 같은active물체의 gap>3cm | .1초연속 |

값은 확정된 물리 안전 한계가 아니라 **calibration후 검증할 초기 후보**다.
최종 bbox완전포함(+5mm)은 단순하고 보수적인 초기 완료 기준이므로 정상적인5mm초과
overhang도 거절할 수 있다. 실제 mesh지지·손잡이 bounds와 정상조작을 대조한 뒤 강제해야 한다.
진행조건은 더 느슨하게 두어 미는 중의 부분 overhang·작은 bounce·pivot 회복을 허용한다.
정상적인 조작의 속도가 빠르다는 이유로 진행 Clearance를 막거나 즉시 종료하지 않는다.

기존 통로면적≤1e−5m²와 Target이동≤2cm·회전≤.15rad는 유지한다.
최종 속도 검사는 Blocker만이 아니라 모든active물체에 적용해 주변 물체가 비산 중인
장면을 완료로 세지 않는다. .05m/s·.3rad/s는 **완료 판정 전용**이며 연속 Motion penalty를
추가하는 설정이 아니다. 완료 유지시간.2초는 seconds로 두고현재step_dt에서ceil해
100Hz에서는20sample이 되도록 한다. XY이탈은5sample, off-support는10sample이다.

기존 drop6cm, tilt80°·.2초, 선반1cm/2°, singularity·invalid 실패는 유지한다.
새 실패의 누적은 물체별로 수행하며 회복·비활성화하면 해당 물체의 counter를0으로 만든다.
다른 물체의 번갈아 발생한 들림·이탈을 한 물체의 지속으로 합치지 않는다.
Gate는 유예 없이 즉시 닫고 **실패만 debounce**하므로 유예기간에 비산 성공을 지급하지 않는다.
Hard failure가 한번 성립하면 native 종료로 episode를 끝내 이후 착지로 완료를 세탁하지 못하게 한다.

최종 valid success는 raw goal·위 완료조건·`~enabled_failure`를 함께 만족해야 한다.
기존 native enabled failure mask만 사용해, 비활성화한 F/T hard-load를 숨은 gate로 부활시키지 않는다.
Raw goal은 진단값으로 남기고 valid success와 failure는 상호배타적으로 만든다.
Terminal은 **valid success에만+50**, 그외0을 유지하며 별도 음의 failure weight는 아직 정하지 않는다.

### 진행 Clearance의 지급 방식

초기대비 품질차이 d와 현재진행유효성 Vp로 다음을 제안한다.

$$
d=q_C(t)-q_C(0),\qquad
r_C=4\left[\min(d,0)+V_p\max(d,0)\right].
$$

기존 exp품질·초기baseline·weight4를 유지하고 **양의 부분만** 진행 유효성으로 gate한다.
음의 Clearance와 R/M 비용은 그대로 지급하며 다른 reward까지 전체0으로 만들지 않는다.
공중·영역밖에서 보류된 양의 보상을 나중에 정산하지 않는다.
최종 정지 조건을 이 gate에 넣지 않으므로 물체를 움직이며 통로를 개선하는 중에도
진행 유효성이 유지되면 양의 Clearance를 받을 수 있다.

첫단계에서는 다른 weight와 Blocker의 일반 병진·회전 R제외를 유지한다.
Blocker도 지지·잔류·비산 유효성 검사에서는 제외하지 않는다.

### 코드·병렬화 계획

- `mdp/events.py`: 기존 `MdpCfg`에 threshold·seconds를, `TaskState`에 cached진행/완료/failure
  bool과 두개`(N,O)`정수counter를 추가한다. 기존 post-physics갱신에 현재corner·속도·선반pose를
  재사용하고 native manager가 호출하는 순서와 무관하게 step당한번 계산한다.
- 기존 `scene.py`의 실제작업판·기둥 metadata를 재사용/확장해 최초준비후 GPU에 저장한다.
  환경별 Pythonloop나 추가 sensor·SDK pose조회가 필요한 설계로 만들지 않는다.
- `mdp/terminations.py`: cached조건을 읽는 두 failure term과 valid success를 연결하고,
  `shelf_manipulation_env_cfg.py`에서 공식 TerminationTermCfg로 선언한다.
- `mdp/rewards.py`: cachedVp를 양의 Clearance에만 적용하고 valid success로 terminal지급을 일치시킨다.
  새production파일은 만들지 않는다. 부분reset은선택env_ids의counter·cache만초기화한다.
  Inactivepadding은기하·속도·counter검사에서모두mask한다.

이 계획은 기존 GPU tensor계산에 작은 reduction·bool·counter갱신을 추가하는 방식이다.
실제 overhead는 측정해야 하며0이라고 주장하지 않는다.

### 후속 단계의 분리 실험

**Rearrangement:** 현재 exp-per-step 비용의 .1m총경로를1/10/100step으로 나누는 독립 검사를
유지해 분할 편향을 정량화한다. 과거 누적이동량을 매step 반복 감점하는 수식으로 돌아가지 않는다.
exp품질을 유지한다는 조건에서 최종 지급방식을 별도로 결정해야 하며, 조용히 선형비용으로
교체하지 않는다. 이 수식 실험은 첫단계 성공·진행gate수정과 섞지 않는다.

**Action·탐색:** 공식 ActionCfg의 `position_scale`후보.02/.05와 `orientation_scale`후보.05/.10을
고정 명령으로 비교해 실제EEF추종·접촉·토크포화를 측정한다. 아직 선택한action상한은 아니며
Gaussian출력자체의상한도 아니다. Kp50–200·stiffness_scale100과 공식OSC core는 유지한다.
초기std.3에서 native entropy coefficient`.01→.001`, adaptive schedule·desiredKL`.01`도
각각 분리해서 비교할 수 있으나 이들이 std의 hard cap을 보장하지는 않는다.
설치RSL-RL5.0.1에 없는 `max_std`config를 있다고 가정하지 않는다.
전체action을±1로 자르면 Kp최대가100으로 줄어200까지 쓰지 못하므로 그런globalclip을 넣지 않는다.
Actionscale을 바꾸는 실험은 metadata계약을 버전으로 구분하고 새학습으로 평가한다.

### 구현 후 통과 기준

- 합법적밀기에서 움직이는동안 양의Clearance가 지급되고, 낮은속도로.2초유지하기전 성공은false다.
- 작은lift·pivot·bounce가회복되면 object별counter가초기화되고 과한일괄종료가발생하지않는다.
- 공중·영역밖·전도·주변물체비산 반례는성공false·+50지급0이며 success와failure가동시참이되지않는다.
- 음의Clearance·R/M이gate로사라지지않고, 비활성F/T종료가새gate에포함되지않는다.
- 물체별counter·inactivepadding·선택행reset·움직인선반frame을 CPU독립예제와CUDA1,024행에서검사한다.
- Native실행에서 rawgoal대비invalidcompletion의동시counter를집계하고 **수정된valid success**로
  checkpoint를재평가한다. 이전success snapshot을새유효성공률처럼사용하지않는다.
- 최소GPUepisode집계만사용하고매stepCPU로그는추가하지않는다. 같은조건baseline과비교해
  iteration overhead5%미만을목표로측정하되 통과를미리주장하지않는다.

초기height·XY·속도후보는 정지asset·정상밀기 calibration이 끝난 후 강제할 값으로 확정한다.
이 계획 작성 시점에는 구현·성능검증·새성공률측정을 수행하지 않았다.

## 34. 2026-10-21 — 유효 재배치 성공과 비산 차단 1단계 적용

사용자가33절 수정안의 완료 유지시간을0.2초에서 **0.5초**로 바꾸어 적용하도록 요청했다.
이번 범위는 성공·진행 Clearance·명백한 비산 실패의 정합성이다. 다른 reward weight,
Blocker의 일반 이동 Rearrangement 제외, action scale·OSC·탐색 설정은 유지한다.

### 적용한 판정

| 항목 | 적용 조건 |
|---|---|
| 기존 목표 | 지정 Blocker의 접근영역 교차 면적≤1e−5m², Target 초기 대비 이동≤2cm·회전≤.15rad |
| 최종 완료 | 모든 활성 물체가 작업판 XY 안에 잔류(+5mm), collision 지지 signed gap −5~5mm, tilt<80°, 선속도≤.05m/s·각속도≤.3rad/s |
| 유지시간 | 목표·최종완료·활성실패없음을 **.5초 연속**;100Hz50sample, 주기 변경 시 seconds에서 올림 |
| 진행 유효성 | 모든 활성 물체의 작업판 내 OBB중심·XY footprint overlap, signed gap −5~10mm, 현재전도·invalid·활성실패없음; 속도 제한 없음 |
| `object_ejected` | 같은 활성 물체 XYbbox가 작업판과 완전히 분리되고 경계 간격>2cm인 상태가.05초 지속 |
| `object_off_support` | 같은 활성 물체의 지지 gap>3cm인 상태가.1초 지속 |

새 실패의 지속시간은 `(N,O)` GPU정수 counter로 관리한다. 회복·비활성화하면 해당 물체만0이 되고,
부분reset은 선택 환경의 counter만 초기화한다. 서로 다른 물체의 교대 들림·이탈을 합치지 않는다.
진행 gate는 즉시 닫으며 실패 종료만 짧게 유예한다. 기존 drop6cm·tilt80°/.2초·선반1cm/2°·
invalid·singularity 실패는 유지했다. 공식 `TerminationTermCfg`를 사용해 새 두 원인을 개별 기록한다.

기존 통로·Target 조건은 `raw_goal` 진단과 분리하고 **활성 실패가 성공보다 우선**하도록 했다.
같은 step에 실패가 성립하면 성공false·terminal0이며, 유효 성공에만+50을 한 번 지급한다.
비활성화한 F/T hard-load는 숨은 성공 gate로 복구하지 않는다. 속도는 완료에만 적용하므로
지지된 상태의 빠른 밀기·pivot을 그 이유만으로 진행 보상에서 제외하지 않는다.

### Clearance와 지지 기하

기존 초기 품질 기준과exp형태·weight4를 유지하고 양의 부분만 gate한다.

$$
d=q_C(t)-q_C(0),\qquad
r_C=4\left[\min(d,0)+V_p\max(d,0)\right].
$$

음의 Clearance와 Rearrangement·Motion 비용은 유지하며, gate가 닫힌 동안 지급되지 않은
양의 보상을 나중에 정산하지 않는다. Distance·접촉 유지 보상은 이번 변경에서 바꾸지 않았다.

실제 작업판·기둥에서 측정한 영역과 현재 선반 frame을 사용한다. 높이는 보수적인 OBB 최저점만
쓰면 작은 pivot에서 빈 corner가 판 아래로 내려가므로, 활성 collider Mesh/Cube 등의 실제 지지
정점을 asset별 시작 시 한 번 추출하고 CPU convex hull로 내부 정점을 제거한다. **이 hull은
높이 계산용 cache이며 PhysX collision 형상이나 hull 분해 설정을 변경하지 않는다.**
GPU에서는 평면 법선 방향 정점 내적의 최솟값으로 collision 하단 높이를 구한다. Mesh의 선형
평면 극값은 원래 정점과 같고, 곡면 primitive는 촘촘한 근사와 scale을 반영한 미터 단위 오차 상한을
함께 저장한다. 이는 authored collider의 기하 근사이며 cooked PhysX 접촉력의 증명이 아니다.
Signed gap의하한−5mm로 판 아래 침투를 정상 지지로 오인하는 것도 차단한다. 작은 pivot은
회복 가능한 범위에서 허용하되 현재80° 이상 전도는 완료와 양의 진행 보상에서 제외한다.

기존 `scene.py`·`events.py`·`terminations.py`·`rewards.py`와 환경cfg 안에서 구현했다.
새 production 파일·카메라·접촉 센서를 추가하지 않았고, 매step CPU기하 조회·환경별Pythonloop 없이
기존 pose·OBB·속도와 cached GPU기하를 재사용한다. 기존 native manager 실행 흐름과dt 어댑터를 유지한다.

### GPU 비용과 native 로깅

실제 collision catalog는 asset별 hull 정점이8~28,209개였다. 초기의 환경별 padded정점 복제는
약3.47GB와support helper20.08ms/step을 요구해 제거했다. 최종 구현은 약3.39MB의 공통 catalog를
그대로 공유하고, Warp block reduction이 물체별 실제 정점 수만 처리한다. 정점 수를 줄여 근사를
늘린 최적화가 아니며, 결과는 full정점 torch계산과float32오차 범위에서 일치했다. 작은 tensor연산은
cached CUDA graph로 묶어 반복적인 CPU kernel launch비용을 줄였다.

실제 완료된 episode를 reset 전에GPU에서 집계해 기존 native logger로 내보낸다.
`Task_Count/completed`, `valid_success`, `raw_goal_seen`, `unsupported_goal_seen`, `goal_and_failure`는
프로세스 시작 후 누적 개수다. `Task_Rate/valid_success`는 누적 유효 성공/완료episode 비율이다.
`raw_goal_seen` 등은 episode 도중 한 번이라도 조건을 보였는지 집계하며 서로 배타적이지 않다.
`unsupported_goal_seen`은 raw목표와 진행유효성 위반의 동시 발생, `goal_and_failure`는 raw목표와
활성 실패의 동시 발생이다. 기존 `Episode_Termination/*`의 마지막episode snapshot 평균과
새로 완료된 episode의 실제 누적 비율을 혼동하지 않는다. 매step CPU출력은 추가하지 않았다.

### 검증 결과와 범위

- CPU MDP227 tests(16 CUDA skip), learning62 tests(2 CUDA skip)를 통과했다. CUDA4개 회귀검사도
  통과했으며 cache입력변경·물체별counter·부분reset·inference mode교대를 포함한다. 공중·영역밖·
  주변물체비산·음의Clearance유지·실패우선·비활성F/T·padding을 tensor반례로 검사했다.
  Kit의 실제 USD지원기하 fixture1개도 통과했고 app종료exit0을 확인했다.
- 실제10종asset의 최초 collision signed gap은 모두0이었다. 1,024환경에서64holdstep 후 최소
  **−0.182mm**, 최대 **1.711mm**였고 진행·완료의 물리유효성 mask는 모두true였다. 이는
  정지 장면의 기하 판정 결과이며 목표가 달성됐거나 유효 성공률100%라는 뜻이 아니다.
  선택 환경의 부분reset도 나머지 환경의 상태를 바꾸지 않았다.
- 최종 native support helper는 **.414ms/step**이었다. 같은1,024환경·물리·고정hold명령에서
  MDP갱신만 저장된이전구현과 교체한 비교는 baseline평균 **40.34ms/step**, 새구현 **39.82ms/step**
  (−1.27%)였다. 이 차이는 측정잡음 수준이며 뚜렷한 step지연 증가는 관찰하지 않았다.
- 위 비교에는 **PPO update와 새scene생성 경계reset이 포함되지 않는다.** 따라서 전체iteration
  overhead5%미만의 통과나90000iteration 학습시간을 입증한 측정으로 사용하지 않는다.
  장기학습·동적밀기/pivot·새checkpoint의 유효성공률 평가는 별도로 남아 있다.

근거와 재현 스크립트는
[valid_rearrangement_20261021](../outputs/contact_manipulation/valid_rearrangement_20261021/VALIDATION.md)에 둔다.
새 성공 판정으로 기존 checkpoint를 재평가하기 전까지 과거 scalar success를 유효 성공률로
해석하지 않는다. Rearrangement의 시간 분할 편향과action·entropy 조정은 후속 분리 실험이다.

## 35. 2026-10-21 — 손가락 관통·종료 전 위험 비용·성공 보상 재검토

사용자는 빠른 손 동작에서 물체/선반 관통, 비산 종료 경계 전 penalty 부재,
강화된 성공 조건 대비 +50 bonus 부족을 보고했다. 추가 답변에서 관통은 **주로 손가락**이라고 했다.
이 절은 코드·공식 API·로그를 대조한 **검토와 제안**이며 production 설정을 변경하지 않았다.

### 손가락 충돌

현재 물리/정책은100Hz(dt=.01, decimation1)이고, 손의 부모 collider5개는 contact offset .5mm,
손 rigid body의 max_depenetration_velocity는 .1m/s이다. 후자는 실제 손 속도 상한이 아니다.
OSC는 목표 pose 증분을 torque로 제어하므로 position_scale1이 매step1m teleport를 뜻하지 않는다.
1m/s인 실제 link는10ms동안1cm를 이동할 수 있어 얇은 collider를 검사 사이에 지나갈 가능성이 있다.
저속/고속의 동일 경로 대조와 collision overlay로 시간 이산화와 형상/필터 누락을 구분해야 한다.

설치된 Isaac Sim5.1의 SimulationManager.enable_ccd는 CUDA에서 활성화 요청을 무시한다.
[공식 OmniPhysics107.3 문서](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-continuous-collision-detection-on-a-rigid-body)도
Direct GPU API에서 두 CCD 모드가 작동하지 않음을 명시한다. 따라서 단순 CCDon을 해결책으로 적용하지 않는다.
공식 cfg의 solve_articulation_contact_last=True, 부모 contact offset의 점진적 확대를 단독 비교하고
17개 tactile 접촉 시점·self contact·실행 비용도 검사할 후보로 둔다. 지속 침투 해소와 이미 놓친
충돌은 다른 문제다. 여전히 관통하면 물리200Hz(dt=.005)/정책100Hz(decimation2)를 비교하되
물리 호출 증가를 측정한다. depenetration 상한을 무작정 올리면 비산을 악화할 수 있다.

### 위험 비용 후보

현재 support helper에는 gate·지속시간 counter만 있고 경계 전 연속 비용은 없다.
기존 계산의 gap·XY bounds·upright_z를 재사용해 exp(-a*x)-1 비용을 별도 항으로 캐시할 수 있다.
정상 선반 위 밀기/yaw pivot은0이며 Blocker도 위험 검사에서는 제외하지 않는다.
각 위험 비용을0~1로 계산하고 활성 물체/위험 중 최댓값에 총 weight2를 적용하는 것은 초기 후보다.
이는 물체 수에 따른 penalty배율을 피하는 결합이며, 종류별 크기는 별도로 기록한다.
온셋과 weight는 아직 확정/검증하지 않았다.

- 들림: gap10mm까지0,20mm에서 절반,30mm에서약.9375.
- 경계: footprint 전체 이탈 거리만 쓰면 늦을 수 있으므로, 최종 잔류 허용과 맞춘
  XY overhang5mm초과를 경고 후보로 삼는다. 과도한 pivot 억제 여부를 함께 측정한다.
- 낙하·전도도 soft warning을 검토한다. 단, 현재 hard drop은 world OBB 하단이고
  support gap은 현재 shelf-frame collision 하단이므로 기하 기준 차이를 먼저 정리한다.

기존 positive Clearance gate가 닫혀도 Distance최대.4와17pad Hold최대.85가 남는다.
따라서 작은 위험 weight만 추가하면 비산 경계에서조차 양의 보상이 남을 수 있다.
위험 비용은 물리적 충돌 방지나 종료 전 감속 보장이 아니며 한step에 경고 구간을 뛰어넘을 수 있다.

### 로그와 성공 bonus

2026-10-21_18-20-44는1,024환경 본 학습으로 iteration0–1396을 분석했다.
새 누적카운터의 rollout-평균 snapshot에서 추정한 완료episode 대비 비율은
유효 성공약.767%, raw goal경험16.127%, unsupported goal경험12.054%다.
동일episode에서 중복되는 지표이므로 서로 배타적인 분류가 아니며 정수 원시 횟수도 아니다.
마지막50iteration 유효 성공 추정은1.813%, raw goal27.033%, unsupported goal19.920%였다.
4환경/16iteration인18-18-47은 bonus 결정의 주 근거로 삼지 않았다.

성공 bonus의 첫 비교값은 **50→100**을 제안한다. gamma=.99/100Hz에서0.5초 뒤의
+50은 현재 시점30.25, +75는45.38, +100은60.50이다. 과거0.1초 뒤+50의45.22를
유지시간 변경만으로 보상하려면 약74.74면 된다. +100은 추가 완료 우대 실험이며 최적값은 아니다.
현재 std가.300→7.621, value loss peak약2.75e17까지 증가했으므로 bonus만으로 모든 현상을
설명하거나 해결했다고 보지 않는다. 실제 명령·관절 속도·접촉 침투를 함께 확인해야 한다.

수치 원본: [training_reward_metrics.json](../outputs/contact_manipulation/reward_safety_review_20261021/training_reward_metrics.json).
이 검토에서는 관통 재현 실험·새 penalty 구현·bonus 변경을 수행하지 않았다.

## 36. 2026-10-21 — Reward·Termination 통합과 지지 위험 비용

사용자는35절의 손가락 관통 수정은 재발 시 다루고, **종료 전 위험 비용과 성공 bonus 증가**를
진행하도록 요청했다. 동시에 전체 reward·termination을 검토해 목적이 겹치는 것은 단순 OR·합산이
아닌 하나의 기준으로 통합하고, 불필요해진 항목을 제거하도록 했다. 이번 변경에서는 물리 주기,
collision 형상·offset·solver·controller·action scale을 수정하지 않는다.

### 전체 term 검토와 유지·통합 결정

| 기존 항목 | 결정 | 이유 |
|---|---|---|
| Clearance | 유지 | 통로 개선이라는 주목표 |
| Blocker distance / contact hold | 각각 유지 | 접촉 전 접근과 동일pad의Blocker접촉 지속은 서로 다른 신호 |
| Rearrangement | 유지 | 비Blocker 물체·선반의 step간 조작 경로 비용; 지지 상태 위험과 다름 |
| Action rate / joint velocity | 각각 유지 | 명령 변화와 실제 속도 비용은 중복이 아니며 기존 사용자 선택을 유지 |
| Force 증가 / torque 증가 | 하나의 wrench 증가 비용으로 통합 | 보정된 결합 norm을 한 exp로 평가 |
| Singularity reward / termination | 유지 | 동일 margin의 연속 비용과 위험 경계 종료는 상호 보완적 |
| Terminal | 유효 성공+50→+100 | .5초 안정된 완료를 더 우대; 실패·timeout0 유지 |
| Drop / ejected / off-support | 단일 지지 거리 조건으로 교체 | 같은 지지 작업영역에서의 이탈을 서로 다른 frame·기하로 검사하던 중복 해소 |
| Object tilt | 유지 | 옆으로 누운 물체는 지지 거리0일 수 있어 지지와 자세는 별도 조건 |
| Shelf displacement | 유지 | 선반 자체의 작업 기반 보호; R의 경로 비용만으로 대체하지 않음 |
| Invalid state | 유지 | 비유한 물리·센서·제어 상태의 실행 안전성 |
| Success / timeout | 유지 | 완료와 native truncation의 구별 |

활성 reward는 **10→10개**다. F/T2항을1항으로 줄이고 물체 지지 비용1항을 추가한다.
활성 종료는 **9→7개**이며 실패 원인5개+success+timeout이다. 목적이 다른 항을 한 함수에
넣어 term 개수만 줄이지 않는다. Joint velocity를 없애려면 빠른 속도 자체의 약한 규제를 없애는
별도 설계 변경이며, 이번 중복 정리에서는 제거하지 않는다.

### 지지 위험의 단일 연속 기준

현재 선반 frame의 실제 작업판 영역을 유한 직사각형 $S$로 사용한다. 활성 물체마다 기존
OBB 중심XY와 실제 collider 최저 높이를 결합한 대표점 $p_i$를 만들고, 작업판 위 최근접점까지
**하나의 유클리드 거리**를 계산한다.

$$
\begin{aligned}
S&=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]\times\{z_{\mathrm{panel}}\},\\
p_i&=(c_{ix},c_{iy},z_{i,\min}),\\
d_i&=\|p_i-\Pi_S(p_i)\|_2,\qquad
d_{\max}=\max_{i\in\mathcal O_{\mathrm{active}}}d_i.
\end{aligned}
$$

`clamp`로 얻은 수평 최근접점과 지지면 높이를 사용하므로 위·아래·수평 방향 이탈을 같은 단위와
기하로 평가한다. 예를 들어 수평24mm·높이18mm 이탈은 거리30mm다. 예전 세 bool의 OR가
아니며 기존의 world OBB6cm낙하·완전XY분리2cm/.05초·양의높이3cm/.1초와 의미가 달라진다.
중심이 판 안에 있고 collider가 지지된 정상 평면 밀기·yaw pivot은0이며, Blocker도 제외하지 않는다.
대표점은 실제 질량중심이나 접촉 지지다각형이 아니므로 역학적 안정성·실제 접촉력의 증명으로
해석하지 않는다. 판 위에 누운 물체도 거리0일 수 있어80°/.2초 tilt조건은 유지한다.

보상은 가장 큰 활성 물체 거리에서 다음 하나의 exp비용을 만든다.

$$
r_{\mathrm{object\_stability}}
=2\left[\exp\left(-\frac{\ln2}{(0.01\,\mathrm m)^2}
\max(d_{\max}-0.01\,\mathrm m,0)^2\right)-1\right].
$$

| 거리 | weight 포함 step 비용 |
|---|---:|
| 0–1cm | 0 |
| 1.5cm | −0.318207 |
| 2cm | −1 |
| 3cm | −1.875 |

거리≥3cm가 **같은 물체에서.1초 연속** 지속되면 `object_support_lost`다. `>=` 경계를 포함하며
100Hz에서는10sample이다. 물체별 GPU카운터는 회복·비활성화 시0으로 만들고, 부분reset은 선택
환경만 초기화한다. 서로 다른 물체의 교대 들림을 누적하지 않는다. inactive padding은 제외한다.
현재 지지점·pose·지원 catalog와cachedCUDAgraph를 재사용하며 센서나 매step CPU조회는 추가하지 않는다.

완료의 작업판XY잔류·signed gap−5~5mm·모든활성물체저속·.5초유지는 유지한다. 진행 Clearance도
기존의더엄격한즉시gate를 유지하므로 실패 유예 중 양의 Clearance를 계속 지급하지 않는다.
음의 Clearance와 다른 비용은 gate하지 않는다. Distance와 Hold의최대합1.25는 남지만 지지거리
3cm에서 새비용−1.875가 발생한다. 이것이 모든 조작의 총return 우선순위나 경고 구간을 건너뛰는
동작의 충돌 방지를 보장하지는 않는다. 위험 비용의 초기 weight2는 이후 실제 조작으로 검토한다.

### F/T reward의 실제 통합

기존 센서중심frame·자중/관성보상·양의norm증가·10N/s와.5N·m/s deadband를 유지한다.
하중 감소나 norm이 같은 축방향 변화에는 새 비용을 주지 않는다. 기존 단위별 계수도 재사용한다.

$$
\begin{aligned}
x_F&=\left[\frac{[\|F_t\|_2-\|F_{t-1}\|_2]_+}{\Delta t}-10\right]_+^2,\\
x_T&=\left[\frac{[\|\tau_t\|_2-\|\tau_{t-1}\|_2]_+}{\Delta t}-0.5\right]_+^2,\\
a_F&=\frac{\ln2}{90^2},\qquad a_T=\frac{\ln2}{4.5^2},\\
x_W&=\left\|\begin{bmatrix}\sqrt{a_F/2}\sqrt{x_F}\\
\sqrt{a_T/2}\sqrt{x_T}\end{bmatrix}\right\|_2^2
=\frac{a_Fx_F+a_Tx_T}{2},\\
r_W&=0.30\bigl(\exp(-x_W)-1\bigr).
\end{aligned}
$$

이는 두 기존 reward를 합치는 것이 아니라 **보정된 증가량norm 하나의 exp**다. 힘 증가와 토크
감소가 상쇄되지 않고, 단위가 다른 값을 보정 없이 더하지 않는다. 작은 비용의기울기는
`0.30/2=.15`로 기존각.15를유지하며 총 비용하한−.30도 같지만, 중간영역의 곡선은 다르다.
한채널만 기존 half점이면 이전−.075→새−.087868, 두채널모두half점이면−.15로 같다.
하나의 채널만 매우 커도 결합 비용은−.30까지 접근해 기존 해당채널 하한−.15와 다르다.
`Episode_Reward/motion_wrench_growth`로 기록하고 힘·토크 증가량 자체는 진단으로 유지한다.

### 실제 제거 목록

| 제거 항목 | 대체 또는 남기는 정보 |
|---|---|
| `object_drop`, `object_ejected`, `object_off_support` 종료term·reason | `object_support_lost` 단일 거리 조건 |
| 별도 ejection/off-support 카운터 | 하나의 물체별 support-loss 카운터 |
| `motion_force_increase`, `motion_torque_increase` reward adapter | `motion_wrench_growth` 단일norm·단일exp |
| 비활성 `time_penalty` reward·매step 시간exp 계산 | timeout은 native truncation으로 유지; 새시간비용 없음 |
| 비활성 `hard_load_or_overrange` 종료reason·hard하중비교 | SI-200-8 raw범위초과·보정센서진단 유지; 숨은 성공gate 없음 |
| `failure_penalty` legacy field | 실패bonus0은 그대로 유지 |

제거할 불필요한 cfg10개는 `ejection_gap`, `ejection_duration`, `off_support_gap`,
`off_support_duration`, `drop_tolerance`, `force_hard`, `torque_hard`, `time_penalty_scale`,
`time_penalty_growth_rate`, `failure_penalty`다. 새 지지 거리·duration·soft 곡선 설정으로 필요한
값만 선언한다. 기존 물체·선반R, action-rate·velocity, singularity는 제거하지 않는다.

### 구현·검증 범위

기존 `mdp/events.py`, `rewards.py`, `terminations.py`와 native환경cfg에서 연결하고 새production
파일·물리센서·환경subclass를 추가하지 않는다. RewardManager의weight/dt어댑터, 실패우선·
성공+100한번·부분reset·rawgoal진단을유지한다. Reward/termination 이름이 바뀌므로 새run의
항목별 곡선과 과거 force/torque·세support실패 로그를 같은정의로직접비교하지않는다.
네트워크·action계약은바뀌지않지만 이전checkpoint를이어학습하면MDP보상/종료변경실험으로기록한다.

### 검증 결과와 한계

- CPU MDP는 225개 중 16개 제외, learning은 62개 중 2개 제외로 실패가 없었다. 검사 범위는 거리의
  수평·상하·대각선·경계 포함, 작은 lift/pivot·물체별 timer·inactive padding·선택행 reset,
  실패우선·.5초성공·+100한번·F/T exp의국소기울기를포함한다. 수평24mm와높이24mm가
  **각각3cm미만이어도 결합거리33.94mm로실패하는 반례**를 통과해 기존개별조건의OR가아님을확인했다.
- RTX3090 CUDA에서1,024환경×10물체×6transition의 CPU/GPU판정일치·hotpath host read부재,
  변경입력에대한CUDAgraph재생·2cm비용−1·inactiveNaN제외를통과했다. 기존CUDA회귀4개도
  실패·skip없이통과했다. 이는tensor·state검사이며 Kit/PhysX/PPO검사는아니다.
- 실제Kit/PhysX1,024환경에서64holdstep의관측·보상finite,성공/실패상호배타성, native reward10개·
  termination7개를확인했다. 지지거리2cm의weight포함비용−1, 결합wrench수식의native dt/weight,
  success설정.5초/+100, 선택행0·1023의부분reset지역성을확인했다. 이hold실행에서완료episode는0으로
  **실제성공조작을재현한검사가아니며 성공률을측정하지않았다.** +100의조건부1회지급은tensor회귀로검사했다.
- 초기10종asset의signed지지gap은모두0,hold후범위−.182~1.711mm였으며 진행·완료의물리유효성
  mask는모두true였다. 이는정지장면오판정검사이며목표달성이나성공률100%를뜻하지않는다.
- 지지helper평균은**.4226ms/step**,고정hold의전체환경step은**38.889ms**,32step환산은
  **1.244초**였다. policy추론·PPO update·새scene경계reset을제외한측정이며,이번변경전과의
  통제된baseline비교는수행하지않았다. 따라서이수치로속도개선율이나전체iteration시간을주장하지않는다.

F/T quality의reset초기값은1로두어센서의첫완료sample전에는결합비용0을유지한다.
새장기학습의성공률개선·동적밀기/전도성능·관통해소를입증한변경으로표현하지않는다.
검사원본과재현스크립트는
[unified_mdp_20261021](../outputs/contact_manipulation/unified_mdp_20261021/VALIDATION.md)에둔다.

## 37. 2026-10-21 — 반복되는 Value loss 폭증의 의미와 최초 이상치 검토

사용자는 학습 초반의 상호작용이 후반에는 사라지고 보상도 감소하는 현상을 반복해서 관찰했다.
이번 범위는 `Loss/value`의 정확한 의미와 최신 로그·저장 checkpoint를 통한 원인 범위 축소다.
**Production 코드·reward·termination·모델·PPO 설정을 변경하거나 새 학습을 실행하지 않았다.**
TensorBoard 분석은2026-10-21 13:07 UTC에 확보한 scalar까지이며, 최신 실행은
`2026-10-21_19-59-36`, 마지막 scalar는2230회다. Checkpoint 검토는2200회까지다.

### 로그의 Value loss가 계산하는 것

1,024환경×32step÷4minibatch=8,192이므로 현재 `ContactManipulationPPO.update()`는
설정한 microbatch 한도8,192 이내에서 설치된 RSL-RL5.0.1의 native `PPO.update()`를 호출한다.
Critic은 공개244D 관측을 받는 독립 공식 MLP `244→256→128→64→1`이고, ELU·입력 정규화OFF·
선형 출력이다. Actor와 파라미터를 공유하지 않는다.

Critic의 예측 V는 그 관측에서 앞으로 받을 할인 누적 보상의 추정이다. Loss는 보상의 크기 자체가
아니라 추정 V와 학습 목표 return 사이의 제곱 오차다. 실제 clipped value loss는 다음과 같다.

$$
\begin{aligned}
\bar V_\theta(o)&=V_{\mathrm{old}}(o)+
\operatorname{clip}\bigl(V_\theta(o)-V_{\mathrm{old}}(o),-0.2,0.2\bigr),\\
L_V&=\mathbb E\left[\max\left\{
\bigl(V_\theta(o)-\hat R^\lambda\bigr)^2,
\bigl(\bar V_\theta(o)-\hat R^\lambda\bigr)^2\right\}\right].
\end{aligned}
$$

`Loss/value`는 이 값을4epochs×4minibatch에 걸쳐 평균한 것으로 **value_loss_coef=.25를
곱하기 전**이다. 최종 objective는 surrogate+.25×value−.01×entropy다. `.2 clipping`은
이전 V에서의 변화에 대한 loss branch를 만드는 것이며 V의 절댓값이나 loss의 상한이 아니다.
Actor와 Critic에 각각 적용하는 gradient norm.5도 forward 값·target·loss의 크기를 제한하지
않는다. Native clipping은 nonfinite gradient에 즉시 예외를 내도록 설정돼 있지 않다.

목표는 완전히 관측된 미래 reward의 단순 합이 아니라 Critic을 재사용한 GAE return이다.

$$
\begin{aligned}
\delta_t&=r_t+\gamma(1-d_t)V_{\mathrm{old}}(o_{t+1})-V_{\mathrm{old}}(o_t),\\
\hat A_t&=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},\\
\hat R_t^\lambda&=\hat A_t+V_{\mathrm{old}}(o_t).
\end{aligned}
$$

γ=.99, λ=.95, rollout은32step이며 마지막 관측의 Critic V를 bootstrap에 사용한다.
Timeout에서는 reset 전 관측의γV를 stored reward에 추가한 뒤 done으로 재귀를 끊는다.
진짜 termination과 timeout이 동시에면 termination을 우선하며 native timeout의 이중 가산은
제거한다. 현재 코드 읽기에서는 reset 관측을 최종 관측으로 잘못 사용하는 alias 오류는 찾지
못했다. Advantage만 표준화하고 reward·return·Value는 표준화하지 않는다.

따라서 reward가 정상 범위라도 잘못된 V가 bootstrap target으로 들어가면 target도 비정상적으로
커질 수 있다. 현재 수식의 정상 finite 상태에서 dense reward의 느슨한 범위는
약−6.43~3.78/step이고 성공+100은1회다. 800step·γ=.99에서 실제 환경 할인 return의 느슨한
범위는 약−642.65~477.73이다. 이 범위는 **정상 task reward의 return**에 대한 것이며,
오류가 있는 Critic으로 계산한 GAE target을 제한하지 않는다.

### 최신 실행의 시간 순서

| Iteration | Loss/value |
|---:|---:|
| 284 | 42.49 |
| 285 | 4.276×10¹¹ |
| 286 | 9.899×10¹¹ |
| 287 | 39.10 |
| 631 | 189.21 |
| 632 | 1.231×10¹⁴ |
| 633 | 180.22 |
| 2228 | 6.855×10²⁸ |
| 2229 | inf |
| 2230 | nan |

첫285회의 제곱 오차를 RMS로 환산하면 약653,932지만, 모든 환경의 V가 이 크기라는 뜻은 아니다.
특정 환경·transition의 큰 오차 몇 개도 평균 제곱 오차를 지배할 수 있다. 처음부터 Critic 전체가
지속해서 거대한 출력을 내는 형태보다는 **일부 batch의 극단값이 간헐적으로 나타나는 형태**다.

| 구간 | Loss 중앙값 | Loss>10⁶인 iteration 수 |
|---|---:|---:|
| 0–284 | 73.68 | 0/285 |
| 285–499 | 86.18 | 2/215 |
| 500–999 | 124.25 | 30/500 |
| 1000–1499 | 187.05 | 136/500 |
| 1500–1999 | 1.427×10¹⁴ | 460/500 |
| 2000–2228 | 3.013×10¹⁷ | 227/229 |

285회 당시 평균 episode return은6.42, Gaussian 평균std는.496이다. 당시 이미 거대한 reward나
폭증한std가 확인된 것은 아니다. Mean reward는 최근 완료episode 통계라 단일 outlier reward를
배제하는 증거는 아니지만, 위의 수식상 bounded reward와 함께 보면 정상 보상 scale만으로
이 loss를 설명할 수 없다. Value loss의 첫 nonfinite는2229회이고 invalid_state snapshot의
첫 nonzero는2230회다. Invalid 종료가 finite한 큰 값의 선행 검출을 보장하지 않음을 보여준다.

| 지표의 구간 평균 | 0–49회 | 2000–2199회 |
|---|---:|---:|
| Train/mean_reward | 112.84 | −29.15 |
| Episode_Reward/clearance | 11.06 | .328 |
| Episode_Reward/blocker_contact_hold | .0231 | .000085 |
| Policy/mean_std | .315 | 56.79 |

최종2230회의std는82.55다. 이는 학습 Gaussian의 분산 지표이며 deterministic play의 출력이나
EEF 속도가 아니다. 보상·접촉 감소와 함께 탐색 분산이 매우 커져 있으므로 조용한 비접촉 정책으로
안정 수렴했다고 해석하지 않는다. 로그상의 선후관계만으로 loss가 행동 붕괴를 일방적으로
일으켰다고 확정하지도 않는다.

다른 독립 Critic 실행에서도 첫 Loss>10⁶는 `11-09-26`221회2.028×10¹⁰,
`13-33-02`403회1.827×10¹¹, `15-33-47`222회1.520×10¹⁶,
`18-20-44`257회2.154×10¹⁵로 반복됐다. 실행 간 reward/종료 설정은 다르므로 재발성의
증거이며 하나의 원인을 분리한 통제 실험은 아니다.

### Checkpoint로 확인한 것과 확인하지 못한 것

최신250→300회 저장 Critic의 최대 절댓값 weight는.211→.215, 첫 layer norm은11.41→11.44,
출력 layer norm은.821→.794다. 검토한2200회까지의 Actor·Critic·Adam state는 모두finite다.
따라서 첫285회 spike를 저장 가중치 자체의 거대한 폭발로 설명하는 근거는 없다. 50회 간격의
저장값이므로 그 사이의 일시적 상태까지 정상이라고 증명하는 것은 아니다.

실제 rollout이 아닌 합성 정상 범위 입력1,024개에 대한 최대|V|는250회45.49, 300회34.19,
1500회41.40, 2000회2,926, 2200회9,125다. 후반의 critic drift는 이 probe에서도 보이지만
합성 관측을 실제 학습 분포나 당시 target의 재현으로 취급하지 않는다.

확인한 입력 경로는 joint_vel/2와 raw previous_action18에 큰 finite 값을 제한하거나
통계적으로 정규화하는 처리가 없다는 점이다. Current·previous 두 sample 모두에 들어간다.
F/T는SI범위로 나눈 뒤±5 clamp하지만, 다른 모든 입력도 같은 범위로 제한되는 것은 아니다.
model250에서 합성 joint velocity 열만10⁶배 키우면 최대|V|≈2.27×10⁶, action 이력 열만
키우면≈1.09×10⁶가 된다. 정상 규모 weights로도 입력 이상치가 큰 V를 만드는 기작의 확인이지
당시 그런 입력이 있었다는 증거는 아니다.

반대로 정상 물리 입력을 고정하고 raw action 이력만500step 되먹임한 합성 검사에서는 최신
checkpoint들의 자가 지수 폭발을 재현하지 못했다. 따라서 action history만을 단독 원인으로
확정하지 않는다. Checkpoint에 실제 obs·V·return target·rollout buffer가 없어 최초 이상치가
관절 상태, raw action, 다른 관측, bootstrap target 중 어디서 시작됐는지는 미확정이다.

### Actor와의 연결과 다음 진단 기준

Actor와 Critic은 독립 파라미터·각각의 gradient clipping을 사용한다. 큰 value loss가 Actor의
gradient에 직접 더해지거나 Actor의 clipping 한도를 빼앗는 구조는 아니다. 다만 위 GAE의
잘못된 V가 advantage에 섞여 **실제 return이 감소해도 잘못된 행동을 강화**할 수 있다.
큰 outlier가 advantage 표준편차를 지배하면 대부분 정상 transition의 학습 신호가 약해질 수도
있다. 단순한 일정 offset·scale 오류는 차분/정규화로 일부 상쇄되므로 모든 큰 loss가 반드시
이 결과를 일으킨다고 단정하지 않는다. Advantage 신호가 약해지면 entropy.01의 분산 증가
유도가 상대적으로 남을 가능성도 있으나, 현재 자료만으로 실제 기여 비율은 정할 수 없다.

관측 aliasing과 초기 vision만의 부분관측은 return 예측 난도를 높이지만, bounded reward의
10²⁰ loss를 그 이유만으로 정당화하지 않는다. 우선 최초 spike에서 다음을 함께 보존해야 한다.

1. 환경 reward와 timeout bootstrap 추가 후 stored reward를 구분한 범위.
2. V_old, V_new, 마지막/timeout V, GAE target, raw advantage/TD error의 p99·최댓값·상위 환경.
3. 해당 환경의 관측 group별 크기, action mean/std/sample, q/qdot, 종료·reset 여부.
4. Actor/Critic별 clip 전 gradient norm, nonfinite 여부, KL과 advantage 정규화 표준편차.

이는 다음 구현의 진단 기준이며 이번 분석에서 추가한 runtime logging은 아니다. Reward weight나
모델 크기를 바꾸기 전에 reward scale 효과와 입력/target 수치 이상을 구분한다.
Native Episode_Reward는 누적 기여/고정8초, Train/mean_reward는 최근 완료episode 통계다.
종료 snapshot은 신규episode 실패 비율이 아니며 Task_Count도 logger의32step 평균으로 기록돼
차분을 정확한 정수 episode 수로 표현하지 않는다.

원본·재현 스크립트는
[value_loss_review_20261021](../outputs/contact_manipulation/value_loss_review_20261021/)에 둔다.
주요 그림은 [value_loss_timeline.png](../outputs/contact_manipulation/value_loss_review_20261021/value_loss_timeline.png),
수치 근거는 `timeline.json`, `derived_findings.json`, `checkpoint_diagnostics.json`,
`older_native_critic_comparison.json`이다. 학습 성능이나 최초 실패 장면의 실제 재현을 수행한 분석은 아니다.

## 38. 2026-10-21 — Gaussian std 오류와 학습 안정화 수정 방향

사용자가 후속 traceback을 제공했다. Native PPO update 중 Actor의 stochastic forward가
`torch.normal`에서 `normal expects all elements of std >= 0.0`으로 중단됐다. 이번에도 원인
검토와 수정안 정리만 수행하며 production 설정·코드·reward를 변경하지 않았다.

### 이번 traceback이 추가로 확인하는 것

현재 Actor는 공식 GaussianDistribution의 log parameterization으로 `std=exp(log_std_param)`을
사용한다. 유한한 log_std에서 음수 std는 나올 수 없다. CPU 재현에서 std=NaN은 정확히 같은
오류를 냈고 std=+inf는 inf를 샘플링했으며 std=0은 허용됐다. 따라서 현재 설정에서 이 오류는
**Actor의 std/log_std가 NaN이 됐다는 강한 근거**다. 단순히 음수 파라미터를 양수로 바꾸는
문제로 해석하지 않는다.

Native PPO는 update에서 distribution을 다시 구성하려고 stochastic Actor를 호출한다.
이때 새로 뽑은 action은 surrogate에 사용하지 않으며 저장된 action의 log_prob를 계산한다.
따라서 sample 호출은 이미 손상된 분포를 발견한 지점이다. 샘플링을 생략하는 것만으로 분포나
log_prob를 복구할 수 없다. 직전 minibatch optimizer update가 최초 오염 지점일 가능성은
있지만 당시 gradient와 분포 파라미터가 없어 확정하지 않는다.

Critic과 Actor가 독립이므로 Critic의 inf loss가 같은 scalar objective에 합쳐졌다는 이유만으로
Actor gradient가 자동으로 NaN이 되는 것은 아니다. CPU의 분리 파라미터 예제에서도
Critic 제곱 오차는 inf인 반면 Actor gradient는 정상으로 남았다. 가능한 연결은 잘못된 GAE/
advantage, 공통 관측 이상치, Actor log_prob/ratio overflow 등이다. 별도 합성 예제에서는
매우 큰 유한 action의 log_prob가−inf가 되고, PPO ratio에서−inf−(−inf)가NaN이 되어
log_std gradient까지 오염됐다. 이 역시 실제 run의 최초 transition을 재현한 증거는 아니다.

Native full-minibatch PPO는 `clip_grad_norm_`에 `error_if_nonfinite=True`를 지정하지 않는다.
CPU에서 inf gradient에 기본 clipping을 적용하면 inf×0으로NaN이 된다. 현재1,024환경의
native경로는 이 예외 검사가 없고 custom microbatch경로에는 있으므로 양 경로 보호가 다르다.

### 아직 적용하지 않은 수정 우선순위

1. **최초 이상값 확보와 optimizer 오염 방지.** 관측 group별 크기, V/return/timeout bootstrap,
   원시 advantage, action mean/std/log_prob, clip 전 gradient의 유한성을 함께 검사한다.
   비유한 gradient를 optimizer에 반영하기 전에 자료를 보존한다. 정상 시에는 GPU reduction과
   작은 통계만 기록하고 상세 CPU 복사는 최초 이상 때로 한정하는 방향이다. 오류를0으로 덮거나
   해당 update를 무기록으로 건너뛰며 계속 학습하는 것을 해결책으로 삼지 않는다.
2. **큰 유한 입력의 생성 경로 수정.** 현재 finite 검사만으로는 매우 큰 qdot/raw action을
   거절하지 않는다. 실제 이상 row의 물리 상태와 정책 출력을 구분하고, 물리 상태가 정상 범위를
   벗어난 경우에는 원인과 invalid 처리를 정한다. 범위는 URDF/actuator 설정과 실제 자료로
   결정하며 이번 검토에서 임의의 새 action 제한을 확정하지 않는다.
3. **공식 Critic 관측 정규화.** `RslRlMLPModelCfg(obs_normalization=True)`로 native
   EmpiricalNormalization을 사용한다. GPU tensor 연산이며 통계는 native checkpoint에 저장된다.
   평균/표준편차 정규화일 뿐 outlier clipping이나NaN 거절 기능은 없어1·2를 대체하지 못한다.
   현재 Actor의 flag는 명시적으로 거절되며 물리값으로 mask를 생성하므로 같이 켜지 않는다.
4. **PPO 분포 안정화의 분리 비교.** 공식 예제처럼 `schedule="adaptive"`, `desired_kl=.01`,
   entropy.01→.005를 비교할 수 있다. Adaptive는 KL에 따른LR조절이며 hard KL 종료나NaN
   보호가 아니다. 낮은 KL에서는LR을최대.01까지올리므로 항상보수적이라고 표현하지 않는다.
   Entropy 축소나0대조는 후반std증가의기여를분리하는 실험이며 최초Critic spike원인으로
   확정하지 않는다. value_loss_coef.25나 reward weight 증폭은 입력/target 오염의 해결책이 아니다.

우선 reward·MDP·Actor 구조를 유지하고1–3을 확인한 뒤4를 비교하는 방향이다. 정규화 추가는
Critic state_dict와 metadata를 바꾸므로 기존 불안정 checkpoint를 그대로 resume하지 않고
새 학습을 기준으로 한다. 정상 resume의 통계 복구는 native 기능을 사용하며 strict=False로
버퍼 불일치를 숨기지 않는다. 장기 학습 재개 전에는 기존 첫 spike285회와 붕괴2230회를 넘겨
V/target·분산·실제 접촉/성공이 함께 안정적인지 검토해야 한다. 이번에 그 검증을 실행한 것은 아니다.

근거는 설치본 RSL-RL5.0.1 `modules/distribution.py`, `modules/normalization.py`,
`algorithms/ppo.py`와 로컬 ShadowHand/UR10 OSC PPO 예제다. 외부 1차 자료는
[공식 정규화](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/modules/normalization.py),
[공식 Gaussian](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/modules/distribution.py),
[Andrychowicz et al. 2020, §3.2–3.3](https://arxiv.org/html/2006.05990v1)이다.
논문의 exp-std NaN 및 observation normalization 결과는 유사 위험의 근거이며 우리 run의
동일 원인이나 해당 수정만으로 해결될 것이라는 증명은 아니다.

## 39. 2026-10-21 — 공식 Actor·Critic·PPO와 GNN 연결 구조 비교

사용자는 GNN·Attention 추가 과정에서 공식 구조와 멀어진 것이 반복되는 학습 붕괴를 만들었는지
검토하도록 요청했다. 설치된 RSL-RL5.0.1, 최신 실행 `2026-10-21_19-59-36`의 저장 설정,
현재 Actor/PPO, Franka Lift/Reach·ShadowHand·Allegro 예제와 Sweeping min 캐시를 비교했다.
**이번 검토에서 production 코드·MDP·네트워크·학습 설정을 변경하지 않았다.**

### 비교 대상의 출처 정정

로컬 IsaacLab remote는 `7cmdehdrb/IsaacLab` fork다. UR10 OSC torque 설정은 tracked/clean이지만
fork의 `3456fb5566958c3a5f608dd978ec6b5f04114cd1`에서 추가된 예제다. 앞선 답변에서 이를
공식 upstream 예제로 표현한 것은 부정확했다. 이번 공식 비교는 upstream에서도 내용을 확인한
Franka Lift·ShadowHand 및 기존 Franka Reach/Allegro를 사용한다. Sweeping min은 별도 프로젝트의
참고 예제이며 공식 권장값으로 취급하지 않는다. 로컬 min 캐시는
`57caa46bd3fe9f8fb00fc0160d1c31beb23b4c78` 기준이다.

### 이미 native인 부분

- 1,024×32÷4=8,192이므로 PPO update는 그대로 `super().update()`다. Native act,
  RolloutStorage, Adam, GAE, advantage 정규화, clipped value/surrogate loss를 사용한다.
  별도 microbatch loss 구현은 이 실행에서 활성화되지 않는다.
- Critic은 공식 독립 MLP `244→256→128→64→1`, ELU다. GNN과 Attention은 Actor에만 있고
  Critic의 forward/backward가 해당 encoder를 사용하지 않는다. 공유 latent나 value head는 없다.
- Sampling·log_prob·entropy·KL은 공식 GaussianDistribution을 호출한다. Tanh 변환이나 그에
  필요한 확률 보정을 별도로 혼합하지 않는다. Native PPO도 update에서 stochastic forward를
  호출하므로 std 오류의 sample stack 자체가 custom 연결 오류의 증거는 아니다.
- 추가 timeout bootstrap은 pre-reset final observation을 평가하고 true termination을 제외한다.
  native time_outs 가산을 제거해 이중 적용하지 않으며 reset clone으로 최종 관측을 보존한다.
  현재 코드와 CPU 회귀에서는 episode 경계 누출을 찾지 못했다.
- Runner의 `check_for_nan=True`도 유지한다. 다만 공식 검사 대상은 환경 obs/reward/done의NaN이며
  큰finite입력, 모델 내부Value/return/gradient까지 검사하는 기능은 아니다.

### 설정의 실제 차이

최신 저장 설정과 현재 설정은 일치한다. 아래 std는 legacy cfg를 설치본의 distribution_cfg로
변환한 뒤의 의미다. 공식 cfg에서 생략한 noise_std_type의 기본값은scalar다.

| 항목 | 현재 | Franka Lift | ShadowHand | Franka Reach |
|---|---|---|---|---|
| Actor | GNN2층·역할Attention·MLP | MLP256/128/64 | MLP512/512/256/128 | MLP64/64 |
| Critic | 독립MLP256/128/64 | 독립MLP256/128/64 | 독립MLP512/512/256/128 | 독립MLP64/64 |
| Activation | ActorReLU/CriticELU | ELU | ELU | ELU |
| Actor/Critic 관측 정규화 | False/False | False/False | True/True | False/False |
| 초기std / parameterization | .3/log | 1/scalar | 1/scalar | 1/scalar |
| Entropy coefficient | .01 | .006 | .005 | .001 |
| Value coefficient | .25 | 1 | 1 | 1 |
| LR / schedule | 3e−4/fixed | 1e−4/adaptive | 5e−4/adaptive | 1e−3/adaptive |
| Desired KL | None | .01 | .016 | .01 |
| Gradient norm limit | .5 | 1 | 1 | 1 |
| Rollout / epochs / minibatches | 32/4/4 | 24/5/4 | 16/5/4 | 24/8/4 |
| γ / λ | .99/.95 | .98/.95 | .99/.95 | .99/.95 |

**정규화 OFF 자체가 공식 구조에서 벗어난 설정은 아니다.** Franka Lift의 Critic은 층 너비와
ELU, 정규화 OFF까지 현재와 같다. Raw previous action 관측도 Shadow/Allegro에 있다.
Hand 예제는 실제 관절 limit으로 위치를 [-1,1]에 매핑하고 속도에 .2를 곱하지만, 현재는
q/π와 qdot/2를 사용한다. Franka Cabinet은 action을 ±1, obs를 ±5로 clamp하지만 모든 공식
task에 같은 clamp가 있는 것은 아니다.

### 원인 후보로 우선 검토할 차이

1. **log std + entropy.01 + fixed LR + KL 제어 없음의 조합.**
   `H=Σ(logσ+C)`이고 `∂H/∂logσ=1`이므로 task advantage가 약해지면 entropy가 log_std를
   계속 올릴 수 있다. 파라미터가 조금씩 증가해도 σ는 지수적으로 증가한다. 비교 예제의 scalar는
   entropy를 σ로 미분하면 1/σ가 된다. 둘 다 공식 지원 방식이고 scalar에는 음수 가능성도 있어
   단순 교체가 해결을 보장하지 않는다. 첫285회의 std는 약.5이므로 후반의 std 증가만으로 최초
   Value 이상치를 설명하지도 않는다.
2. **Actor의 normalization API를 막은 독자 adapter.** `obs_normalization=True`를 예외로
   거부하고 `update_normalization()`도 no-op이다. Raw 관측으로 mask를 생성해야 한다는
   요구가 학습 입력 정규화 전체를 금지할 이유는 아니다. Mask를 먼저 만들고 padding과
   quaternion의 의미를 유지하면서 연속 입력을 따로 처리할 수 있다. GNN의 필수 조건이 아니라
   연결 과정에서 선택한 구조다.
3. **Scene LayerNorm이 history144D에는 적용되지 않는다.** Raw qdot/action 등이 scene encoder를
   거치지 않고 concat 후 fusion MLP로 들어가며, Critic도 raw244D를 받는다. 극단 입력은 양쪽에
   영향을 줄 수 있다. 공식에도 raw action 관측이 있으므로 그 자체를 버그로 부르지 않고 실제
   입력 범위와 최초 loss 이상치를 함께 확인해야 한다.
4. **직접 만든 Triton autograd와 torch.compile.** 기본 nn.Module 외에 검증해야 할 경로가 늘었다.
   아래 CPU/GPU 검사를 통과해 보통의 수식·역전파 오류를 지지하는 증거는 찾지 못했으나,
   장기 학습 중 실제 극단 관측에서의 동작까지 보증하는 검사는 아니다.
5. **value coefficient.25, ReLU, 네트워크 너비.** 비교 예제와 차이가 있지만 Critic은 이미
   일반적인 ELU MLP다. 이것만으로 10¹¹~10²⁰ loss를 설명할 근거는 없으며 계수 증가에 비례해
   clipped Adam의 Critic 업데이트가 빨라지는 것도 아니다.

### GNN 자체와 연결 방법의 분리

GNN은 degree에 따른 mean aggregation과 LayerNorm을 사용하고 attention은 sqrt(64)로 scale하며
전체 mask 행을0으로 처리한다. Dropout/BatchNorm/recurrent state는 사용하지 않는다.
CPU learning 검사는62개 실행, CUDA 전용2개 skip, 실패 없음이다. Message factorization의
출력·gradient, 빈 attention, native PPO 업데이트 및 checkpoint 복원 검사를 포함한다.

추가로 Kit를 띄우거나 학습하지 않고 RTX3090에서 실제 배치 크기1024/8192, 물체10개,
특징64개로 검사했다. Fused edge 집계·입력 gradient는 기본 Torch 식과 일치하며 비대칭 edge,
빈 graph, 입력 scale1000도 통과했다. Actor eager/compile의 출력과36개 파라미터 gradient도
두 배치에서 일치했다. 최대 출력 오차3.73e−9, gradient 오차1.16e−10이며
inference→backward→파라미터 변경→inference 전환과 기존 출력 보존도 통과했다.
최대 모델 메모리 할당은1024에서141MB, 8192에서611MB였다. 정상 범위의 무작위 관측과 초기
가중치를 사용했으므로 장기 학습의 실제 이상 transition을 재현한 결과는 아니다.
원본과 재현 스크립트는
[gpu_actor_parity.json](../outputs/contact_manipulation/native_model_audit_20261021/gpu_actor_parity.json)에 둔다.

공식 구조에 가깝게 정리하는 안은 **GNN/Attention을 feature extractor로 남기고 MLPModel의
get_latent를 확장해 scene 특징과 현재/직전 정보를 concat하는 것**이다. 공식 CNNModel도
MLPModel을 상속하고 encoder 외의 head/distribution/protocol을 공통 코드에 맡긴다.
현재는 공통 protocol을 NativeContactActor에서 다시 구현한다. 기존 fusion의 층 너비와 ReLU를
유지할 수 있어 별도 history encoder나 관측 차원 변경이 필요한 제안은 아니다. 기존 파일 안에서
정리하며 PPO/GAE나 올바른 최종 관측 bootstrap을 제거하는 방향도 아니다.

학습 원인의 분리 실험은 같은 reward·Critic·PPO·action 계약에서 Actor만 공식 MLP로 바꾼 경우,
동일 GNN에서 compile만 OFF인 경우, 집계만 Torch CUDA 식으로 바꾼 경우다. 위 GPU 검사는
수치 일치 검사이며 이 학습 대조군들을 실행한 것이 아니다. KL/entropy/normalization 역시 여러
항목을 한 번에 바꾸면 효과를 구분할 수 없으므로 비교 대상을 분리한다.

공식 1차 자료는
[Franka Lift](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rsl_rl_ppo_cfg.py),
[ShadowHand](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py),
[RSL-RL5.0.1 CNNModel](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/models/cnn_model.py)이다.

## 40. 2026-10-21 — 공식 MLPModel Actor 적용과 PPO 설정별 영향

사용자가39절의 공식 MLPModel 기반 Actor 방향에 동의했고, 비교표의 각 설정이 어떤 영향을
주는지 설명이 부족하다고 지적했다. Actor 공통 처리를 공식 구현으로 옮기고 설정 효과를 분리해
검토했다. **이번에는 구조만 바꾸며 std 방식·초기값·entropy·LR schedule·KL·value 계수와
reward·MDP·controller는 기존 값을 유지했다.**

### 적용한 Actor 구조

- `model.py`의 `NativeContactActor(MLPModel)`가 공식 head, forward, Gaussian 생성,
  log_prob/entropy/KL, reset/hidden-state 인터페이스를 상속한다.
- `ContactManipulationEncoder`는 기존 GNN2층·역할 조건부 attention과 현재/직전144D concat만
  수행한다. Native head는464→256→128→18, ReLU이며 별도 history encoder는 없다.
- 마지막 Linear의 gain.01/bias0과 encoder→head의 난수 초기화 순서를 보존했다. 기본 설정의
  표현식·학습 분포를 바꾸기 위한 개편이 아니다.
- Mask는 raw244D에서 생성한다. 선택적 공식 history normalizer는144D를 처리하며, 정규화
  전후 결측 history/tactile mask를 적용해 padding이 비영값으로 살아나지 않도록 한다.
  Actor/Critic의 정규화 기본값은 둘 다False다. Running statistics는 native API로 저장된다.
- cfg도 `RslRlMLPModelCfg`를 확장해 head·activation·분포 설정을 노출한다. 현재 Gaussian의
  log/scalar·초기std를 비교하려고 adapter를 다시 고칠 필요가 없어졌다.
- Native forward는 stochastic_output=True에서만 분포를 갱신한다. Native PPO는 원래 이 방식이며
  큰-batch microbatch와 KL 계산도 새 의미에 맞춰 deterministic mean으로 분포를 명시적으로
  갱신한다. 추가 sampling으로 microbatch의 난수 소비를 바꾸지 않는다.
- v1의 정확히 동등한 checkpoint는 encoder/head/distribution key를 변환하고 Adam ID 순서도
  재배치해 play·resume를 지원한다. 단순히 weight 이름만 바꾸면 Adam 상태가 다른 파라미터에
  연결되므로 충분하지 않다. Metadata·shape·moment를 먼저 검증하며 strict=False로 숨기지 않는다.

### 1. log 표준편차와 직접 표준편차

두 방식은 같은 μ와σ에서 같은Gaussian이지만 optimizer가 움직이는 좌표가 다르다.

$$
\sigma=\exp(s),\qquad
\sigma_{\mathrm{new}}=\sigma_{\mathrm{old}}\exp(\Delta s)
$$

직접 방식은σ에Δσ를 더한다. Log 방식은 양수를 보장하지만 파라미터가 꾸준히 증가하면 실제σ는
곱셈으로 커진다. 직접 방식은 설치본에서 양수 clamp가 없어 음수σ가 될 수 있으므로 무조건 더
안전한 대안은 아니다. 둘 다 공식 지원 설정이다.

18축 독립Gaussian에서entropy를H, 계수를β로 두면 다음과 같다.

$$
H=\sum_{j=1}^{18}\left(\log\sigma_j+\frac12\log(2\pi e)\right),\qquad
\frac{\partial(-\beta H)}{\partial s_j}=-\beta,\qquad
\frac{\partial(-\beta H)}{\partial\sigma_j}=-\frac{\beta}{\sigma_j}.
$$

Task gradient가 약하면 log parameter에는 entropy의 일정한 상승 압력이 남는다. 현재처럼
log방식·entropy.01·fixedLR·KL없음이 결합되면 후반의 분산 확대를 검토해야 한다. 다만 첫
Value spike285회에는std≈.5였으므로 뒤의σ>50만으로 최초이상을설명하지않는다.

### 2. 초기 std.3과1.0

초기 탐색 진폭은3.33배, 분산은11.11배 차이 난다. 같은 초기 평균에서 모든18축σ가같다면
전체entropy는3.869→25.541nats다. 현재OSC position/orientation scale은1이므로 위치한축의
평균주변noise 약95%범위는±.588m→±1.960m다. XYZ noise의RMS는.520m→1.732m다.
이는 **명령의 통계이며 실제TCP변위가 아니다.** Controller추종·토크/목표제한·접촉이 실제운동을
결정한다. 공식 다른task의초기std1을그대로복사하면우리pose명령의초기크기가매우커질수있다.
작은std는탐색부족가능성이있지만 현재후반분산폭증을초기std증가로해결할근거는없다.

### 3. entropy coefficient.01과.006/.005

환경reward가아니라optimizer의loss에−βH로들어간다. 현재는Lift비교값의1.67배,
ShadowHand의2배다. 현재처럼state-independentσ에서는entropy가직접미분하는대상은
std파라미터이며Actor평균을물체쪽으로직접유도하는항이아니다. 그후수집되는행동이달라져
평균정책학습에도간접영향을준다.

계수를줄이면taskgradient대비분산확대압력을줄일수있고, 너무줄이면탐색이일찍줄어들수있다.
그러나entropy만남은극단적상황에서Adam은gradient배율을대부분정규화하므로계수를절반으로
했다고σ증가속도가정확히절반이되는것은아니다. 유효한task신호를회복하는것과함께봐야한다.

### 4. fixed와adaptive

현재는정책분포가급격히바뀌어도lr3e−4가유지된다. 공식adaptive는이전/현재정책의KL로
학습률을조정한다. 한iteration의4epochs×4minibatches에서최대16번반응할수있다.
Actor와Critic이하나의optimizer를쓰므로양쪽의LR이함께변한다.

Adaptive는큰변화때업데이트를줄일수있지만작은변화때는오히려LR을올린다. 설치본의LR범위는
1e−5~1e−2다. NaN거절·hardKL상한·earlystop을제공하는설정은아니며,항상fixed보다안전하다고
표현하지않는다. 오랫동안작게누적되는std상승도KL기준만으로없어지지는않는다.

### 5. desired KL None과.01/.016

Adaptive와함께설정해야효과가있다. Fixed그대로두고desired_kl만바꾸면현재구현에서는
학습률이변하지않는다. 기준κ에대해KL>2κ이면LR÷1.5, 0<KL<κ/2이면LR×1.5다.

| Desired KL | LR 감소 기준 | LR 증가 기준 |
|---:|---:|---:|
| .01 | KL>.02 | 0<KL<.005 |
| .016 | KL>.032 | 0<KL<.008 |

.016이더큰정책변화를허용한다. KL은평균과분산모두의변화를보며초기σ에도영향받는다.
예를들어18축평균이모두같은d만큼변하고σ가고정이면KL=18d²/(2σ²)다. KL=.01에대응하는d는
σ=.3에서.01, σ=1에서.0333으로같지않다. 따라서초기std와KL을동시에복사하면두효과가섞인다.

### 6. value-loss coefficient.25와1.0

Critic의rawgradient를4배로만들지만, 현재Actor/Critic이독립이므로Actorgradient를직접줄이거나
Critic으로gradient예산을옮기는효과는없다. Log의Loss/value도계수를곱하기전값이다.

양쪽계수에서Criticgradientclip.5가발동하면같은방향이같은norm으로제한돼배율차이가
상쇄될수있다. Adam의m/sqrt(v)도일정배율의효과를대부분상쇄한다. 따라서1.0이정확히4배빠른
Critic학습을뜻하지않는다. Clipping상태·epsilon·기존Adammoment·학습중계수변경에따라차이는
남을수있으므로효과가전혀없다는뜻도아니다. 극단적인입력/target이나NaN을이계수만으로
해결할수는없다.

### 다음 비교의 기준

우선이번공식Actor구조에서기존설정과수치동등성을검사한다. 그다음초기std.3은유지하고
`adaptive+desired_kl=.01`을비교하고,그다음entropy만.005로비교하는순서가적절하다.
Scalar/log비교도초기std를같게두어야한다. Value계수1은그이후의분리비교로둔다.
이값들은이번에적용한설정이아니며장기학습결과로검증되지않았다.

### 검증 결과와 한계

- CPU learning은70개 실행, CUDA 전용2개 skip, 실패 없음이다. Native 메서드 상속, 기존 특징과
  head 수식의 출력/gradient, mask 전후 history 정규화, v1 weight·Adam·다음 업데이트 동등성,
  새 정규화 통계 복원, scalar/log 설정, adaptive native/microbatch 일치를 포함한다.
- CPU MDP는225개 실행,16개 skip, 실패 없음이다. 코드 변경은Actor/PPO연결에 한정했지만
  기존 관측·reset·보상·종료 회귀가 유지됨을 확인했다.
- GPU CUDA 전용2개 검사를 통과했다. Native Actor의1024/8192배치, 정규화OFF/ON에서
  eager/compile 출력·확률·entropy·37개파라미터gradient가일치했다. 최대오차는출력4.66e−9,
  log_prob4.77e−7, gradient1.26e−8이며모든gradient는finite다. 정상무작위관측과초기가중치검사다.
- 실제과거 `2026-10-21_19-59-36/model_250.pt`의Actor37개state와Adam45개state를
  새구조로strict로드했고출력도finite였다. CPU동등성검사에서는재개후다음업데이트까지일치했다.
- 실제Kit/PhysX1,024환경에서native2iteration을완료했다. Run은
  `2026-10-21_23-00-48_native_actor_refactor_smoke`이며본학습성능이력에서제외한다.
  Value loss는2.7747/3.2507, meanstd는.30033/.30088이었다. 첫iteration27.28초에는
  compile이포함됐고두번째는2.006초(collection1.788/learning.217)였다. 저장된Actor/Critic/
  optimizer텐서가모두finite이며종료코드0이다. 두iteration으로장기처리량이나학습안정성을
  입증하지않으며변경전과통제된속도비교도아니다.

검증원본은
[native_actor_refactor_20261021](../outputs/contact_manipulation/native_actor_refactor_20261021/)의
`validation_summary.json`, `gpu_validation.json/.py`, `native_training.log`에둔다.
이구조개편이반복되는Value이상치와NaN을해결했다는장기학습증거는아직없다.

## 41. 2026-10-21 — Actor 구조 외 학습을 examples random-sweep 기준으로 통일

사용자가 GNN·Attention을 포함한 Actor 네트워크 구조 외의 학습은 `examples/`를 따르도록
명시했다. 이번 변경은40절의 하이퍼파라미터 유지·자체 PPO adapter 방침을 대체한다.
연구 task의 물체 배치·초기 vision·관측·reward·물리·controller를 Sweeping task로 바꾼 것은 아니다.

### 기준과 적용값

기준은 `examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/config/ur5e/agents/`
`rsl_rl_ppo_cfg_02.py`다. 예제의 Random task 등록과 메인 학습 문서가 이 파일을 지정한다.
다른 non-random cfg의 γ.95·entropy.01·30,000회와 섞지 않는다.

| 학습 설정 | 적용값 |
|---|---:|
| Seed | 42, native 부모 기본값 |
| Rollout / 최대 iteration / save interval | 36 / 90,000 / 50 |
| Gaussian std parameter / 초기값 | scalar / 1.0 |
| Actor / Critic observation normalization | False / False |
| Critic hidden layers / activation | 256/128/64 / ELU |
| PPO epochs / minibatches | 8 / 4 |
| LR / schedule / desired KL | .001 / adaptive / .02 |
| γ / λ | .98 / .95 |
| Value coefficient / entropy coefficient | 1 / .005 |
| Clipped value loss / clip | True / .2 |
| Max gradient norm | 1 |
| Optimizer / advantage normalization | Adam / 전체 rollout |
| Wrapper action clipping / logger | None / TensorBoard |

Actor의 GNN2층·역할attention·scene320D와현재/직전144D concat·head256/128/18 ReLU는
기존 연구 구조로 유지한다. Mean-head의gain.01/bias0재초기화는학습초기화의차이이므로제거하고
native MLP의기본초기화를사용한다. ModelCfg의중복initial_std도제거해분포설정은nativecfg에둔다.

### 자체 학습 경로 제거

- `algorithm.class_name="PPO"`로 공식 클래스를 직접 사용한다. 자체 `ContactManipulationPPO`,
  `construct_algorithm`, microbatch loss/gradient accumulation, KL scheduler, optimizer 코드를 제거했다.
- 자체 pre-reset final-observation timeout bootstrap을 제거했다. 공식 PPO가 timeout 시 저장된
  해당 step의 V를 가산하고 GAE를 계산하는 예제 동작을 그대로 따른다. 이 차이를 숨기지 않는다.
- rollout36에서1,024환경의PPO minibatch는9,216이다. 기존8,192상한때문에자체업데이트로
  바뀌던분기는없다. Actor feature chunk8,192는메모리최적화로만남으며PPO수식을바꾸지않는다.
- 학습의 `init_at_random_ep_len=True`도 표준 스크립트와 맞췄다. 첫 episode의남은길이는
  무작위화되지만실제reset에서초기vision/current/previous를만드는동작은유지한다.
  공통장면교체deadline과episode별부분reset은MDP의조건으로남긴다.
- 전용 learner에서만쓰던 `time_outs_possible` hint는제거했다. TaskState의완료관측과종료진단
  export는남지만native PPO가이것으로별도학습계산을수행하지는않는다.

### Checkpoint와 이전 정책 재생

새 학습의저장/복원은native `OnPolicyRunner/PPO`다. Actor의입력·action·모델설정검사는
PyTorch의공식 `get_extra_state/set_extra_state`를사용해actor state_dict에저장한다.
`agents/ppo.py`에는학습클래스없이checkpoint호환함수만남겼다.

과거v1/v2의logstd/.3정책은play에서원래Actor/Critic설정과분포를복원하고필요한weight key만
변환한뒤공식loader로읽는다. Scalar/log파라미터를숫자만바꿔해석하거나새baseline의Adam으로
이전학습을이어가지않는다. 구형optimizerresume는명시적으로거절하고새학습을시작한다.
이번baseline으로저장한checkpoint의resume는native방식으로검사했다.

### 검증

- CPU learning56개 실행중54개통과·CUDA전용2개skip다. 예제의runner/PPO설정일치,
  nativePPO의실제타입과메서드, native timeout/GAE,로그·checkpoint,scalar1기본분포와
  nativehead초기화,정규화·독립gradient,새checkpointresume와구형정책play를포함한다.
- CPU MDP225개실행중16개skip·실패없음이다. 연구환경과종료조건의기존회귀를확인했다.
- CUDA learning전용2개검사도통과했다. Fused GNN의출력·backward와native Actor의
  compile추론·학습gradient·checkpoint key를검사했다.
- 실제1,024환경에서2iteration을완료했다. Run은
  `2026-10-21_23-18-08_example_learning_baseline_smoke`이며본학습성능비교에서제외한다.
  저장된agent.yaml에서nativePPO·seed42·36rollout·scalar1·8epochs등실행값을확인했다.
  최대iteration은검사용CLI로2로덮었으며기본값90,000과구분한다.
  Value loss는2.4813/10.2161, meanstd는.99833/.99860이었다. 첫iteration27.35초에는
  compile이포함됐고두번째는2.756초(collection2.226/learning.530)였다.
  저장된Actor37개·Critic8개·optimizer135개텐서가모두finite이며종료코드0이다.
- 실제이전 `2026-10-21_19-59-36/model_250.pt`를4환경·8step으로play했다. 원래logstd정책을
  복원하고iteration250을확인했으며정상종료했다. 새로운학습성능검사가아니다.

원본은
[example_learning_baseline_20261021](../outputs/contact_manipulation/example_learning_baseline_20261021/)에둔다.
예제일치는발산원인을해결했다는증거가아니며장기수렴·성공률은새baseline에서별도평가한다.
Rollout과epoch가동시에늘었으므로이전iteration시간과같은연산량으로비교하지않는다.

## 부록: 본 학습 여부가 확인되지 않은 실행

아래 3개는 모두 목표 90,000회·`resume=False`인 본 학습 후보지만 목적의 직접 근거가 부족해
위 6개와 분리했다. 이들을 테스트라고 확정한 것은 아니다.

| 실행 | 환경 수 | 기록 iteration 수 | 최신 checkpoint |
|---|---:|---:|---:|
| [2026-10-14_20-41-21](../logs/rsl_rl/contact_manipulation/2026-10-14_20-41-21/) | 512 | 120 | 100 |
| [2026-10-14_20-50-49](../logs/rsl_rl/contact_manipulation/2026-10-14_20-50-49/) | 1024 | 206 | 200 |
| [2026-10-15_15-53-06](../logs/rsl_rl/contact_manipulation/2026-10-15_15-53-06/) | 1024 | 349 | 300 |

이외 36개 짧은 실행과 초기화만 남은 2개는 실행 목록 JSON의
`short_attempt_ambiguous`, `startup_only_ambiguous`에 보존했다.
예를 들어 `2026-10-15_15-17-51`은 **2,048환경·62회 기록·model_50.pt**가 있지만
목표 90,000회였으므로 짧다는 이유로 테스트라고 제외하지 않는다.
목적이 확인되면 본 학습 이력으로 편입하며, 현재 성능 결론에는 사용하지 않는다.
