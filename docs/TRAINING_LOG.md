# Contact Manipulation 학습 기록과 수정 과제

작성·검토: **2026-10-26, Asia/Seoul**. 현재 프로젝트는
`src/contact_manipulation`, task는 `Isaac-Contact-Manipulation-UR5e-v0`다.
연구·개발 결정의 전체 이력은 [CONTEXT.md](CONTEXT.md), 현재 구현은
[프로젝트 README](../src/contact_manipulation/README.md)를 참고한다.

**최신 적용:** 기존 Terminal에 일회 실패−5를 추가했다. 성공+250·일반/timeout0이며,
현재 사용자 실험의 다른 weight는 유지한다. 변경·근거·검증은 **79절**에 기록한다.
직전 reward 조정 이력은 **78절**, GUI tactile/F/T 모니터는 **77절**, 저장 실행 분석은
**76절**을 따른다.

**이전 물리·관측 적용:** Patch 용량262,144·관측64×48과 GUI 관측 OBB의 변경·검증은
**48절**에 기록한다. 아래 과거 학습 기록은 당시 설정이며 현재 값은79절과 README를 따른다.

**직전 적용:** 2026-10-21_23-26-39 분석의1차 후보를 사용자 승인으로 적용했다.
Hold .05→.10·Stability2→3·Success100→250이며 민감도와 나머지 설정은 유지한다.
검증은 **47절**, 분석 당시의 근거·미적용 후속 민감도 후보는 **46절**에 기록했다.
새 학습은 수행하지 않았다.

**이전 판단:** `2026-10-21_19-59-36`은 value loss의 일시적 극단값이 후반의 지속적 붕괴로
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

## 42. 2026-10-22 — 규칙 기반 밀집 배치와 고정 기하·시점 공동 영역

초기 vision과 물체 배치의 재시도 소진이 반복돼, 사용자는 임의 배치를 승인할 때까지 다시
뽑는 흐름을 제거하고 정해진 생성 규칙의 파라미터를 랜덤화하도록 요청했다. 기본
`SpawnRules.generation_mode="constructive"`를 추가하고 기존 파일 안에서 reset과 연결했다.
새 production 파일은 만들지 않았다. PPO·controller·reward·termination은 이번 배치 변경으로
수정하지 않았으며, 물리 후보 pool 확대는 별도 비용으로 구분한다.

### 확정한 구성과 생성 규칙

| 항목 | 적용 내용 |
|---|---|
| 활성 개수 / 종류 수 | 기본 N=10 / 4–10종 |
| Target | 해당 종류 정확히1개 |
| non-Target | 중복 허용, 선택된 종류별 개수 차이≤1 |
| 전체 점유율 | 기존20% 하한을 강제하지 않음 |
| 초기 yaw | 각 물체의 전체360°에서 한 번 선택 |
| Clutter | 가까운 이웃의 convex collision footprint 간격3–10mm |
| 물리 후보 / 관측 슬롯 | 기본 P=30 / N=10, global pool ID와 관측 slot 분리 |
| 가시성 | 같은 자세의 비가림 실제 메시 실루엣 대비 Target40–90%, Blocker≥95% |

`CONTACT_MANIPULATION_NUM_OBJECTS`로 최초 topology 구성 전에 N을 정하고 종류별
`ceil((N−1)/3)`개를 준비한다. Reset은 기존 body의 pose/state를 쓰며 새 prim을 만들지 않는다.
관측 폭은 `10N+144`이며 Target/Blocker slot0/1·현재/직전 이력·비가시 padding 규약은 유지한다.
N=16의 CPU 검사에서 모델/MDP/reset 용량·304D 관측·18D Actor 출력과 finite backward를 확인했다.
용량 확장은 모든 종류 구성의 공간 수용이나 기존 checkpoint 호환성의 보장이 아니다.

한 scene seed에서 역할·종류 구성·yaw를 한 번 정한다. ±Y 두 방향의 pivot→병진 작업 구획을
먼저 계산하고 수용 가능한 영역의 합집합에서 선택한다. 주변 물체의 ordered slicing 분할은
유한 dynamic programming의 폭·깊이 Pareto frontier로 계산하며 임의 후보 수 제한을 두지 않는다.
이후 두 방향에서 각각 한 번 해석적 compaction을 한다. 이미 이동한 물체와 아직 이동하지 않은
물체를 모두 반영한다. 초기 구현에서 이 후자의 제약을 빠뜨려 archived bbox seed4개에 간격
위반이 생긴 회귀를 수정했고, 해당 seed10의 실제 입력을 단위 회귀로 추가했다.

배치용 형상은 최초 합성된 USD의 authored collider에서 얻은 upright XY convex hull이다.
외접원 전체 회전 여유를 모든 물체에 예약하는 비교안은 채택하지 않았다. Convex hull은
concave/compound 형상의 빈 공간을 메우므로 PhysX cooked 접촉 표면과 동일하다고 하지 않는다.
실제 cup의 XY hull은497개 정점이었다. Cartesian vertex-sum 대신 edge-angle 병합으로
Minkowski 경계를 계산하고 다각형 거리는 compiled CPU kernel로 처리한다.
이 CPU 계산은 공유 장면 하나에서 수행하며 clone별 physics step에 추가하지 않는다.

기존 주변 물체1–2개를 guard로 둘 수 있는 유한 영역도 계산한다. Guarded mode는 초기 X·yaw를
고정한 순수±Y 양쪽 탈출을 margin0에서도 막고 pivot→병진 witness를 유지한다. 일반 XY 병진이나
모든 비회전 해법이 불가능하다는 판정은 아니다. Guard가 쓰는 보수적 실제 이동 envelope와
이전 packing용 큰 빈 직사각형은 구분하며, guarded의 `reserved_regions=()`는 후자를 주장하지
않는 metadata다. Guard가 없으면 일반 `pivot_translate` mode만 준비한다.

### 적응적 재시도 없이 실제 메시 시점을 선택

각 기하 mode에 고정 `2×9×2=36`개 시점과 배치 규칙의 시점1개를 연결한다. Base/guarded가
있으면 최대74개의 mode·시점 쌍을 처음부터 모두 평가한다. 메모리 chunk 분할은 새 후보를
추가하는 재시도가 아니다. Mode별 ray casting·실루엣·유효 시점 선택·visible-surface OBB는
GPU에서 계산하고 최대2개 mode의 우선순위·결과 metadata는 CPU에서 정리한다.

유효한 guarded mode를 우선하고, 여기에 유효 쌍이 없으면 **이미 함께 평가한 base mode**를
선택한다. 두 mode의 균등 선택 분포는 아니다. Mode 안에서는 Target 가시율65%와의 거리,
Blocker 가시율, index 순서로 선택한다. 선택한 mode의 sample·물체 state·충돌 기하·관측을
함께 commit하므로 base 관측을 guarded 물리 배치에 붙이지 않는다. 모든 쌍이 실패해도 역할·
구성·yaw를 다시 뽑지 않으며 조건별 진단을 남긴다.

초기 단일 시점 V1/V2는 각각64개 중36개만 통과했다. V1의 주된 실패는 Target40% 미만,
V2는90% 초과였다. 기하 보장 없이 단일 시점을 통과한다고 가정하지 않고 사용자 확인 후
고정 GPU 시점 영역을 도입했다. 이후 guard만 적용한 native1,024환경 시작은 첫 장면에서
Target65.4321%·Blocker71.9178%로 거절됐다. 이 결과에 대응해 가시성 기준을 낮추지 않고
사전에 구성한 base/guarded와 시점의 공동 영역을 한 번에 평가하도록 수정했다.

### 확인된 검증 결과

| 검사 | 결과와 범위 |
|---|---|
| CPU MDP 전체 | 249개 실행,16개 skip, 실패0;25.425초 |
| CPU learning | 56개 실행,2개 skip, 실패0;0.811초 |
| 실제 collider hull 기하 stress | Raw `Random(seed)`0–1023의1,024개에서 빈 생성 영역0건 |
| 최종 실제 메시 공동 영역 | `_episode_seed` 기반256장면 모두 통과,11,470개 mode·시점 쌍 평가 |
| 초기 native smoke | Guard 확장 전4환경·2iteration 통과; 최종 버전 검증과 구분 |

기하 stress의 평균 생성 시간은75.215ms, p95는137.187ms였다. 최대3.704초에는 cold/concurrent
실행의 이상치가 포함돼 이를 일정한 장면 생성 시간이나 전체 학습 iteration 시간으로 사용하지
않는다. 해당 raw seed 스트림과 이후 실제 메시 검사의 episode seed 스트림도 구분한다.

최종 실제 메시256장면에서는202개가 singleton이고54개가 geometric guarded/base 두 mode를
가졌다. 실제 가시성까지 반영해24개는 guarded,30개는 함께 평가한 base를 선택했다.
선택된 Target 가시율은43.650794–79.436153%, Blocker는95.454544–100%였고 두 역할의 관측
OBB는 모두 유효했다. 후보별 실패 counts가 남는 것은 유효한 최종 쌍이 없었다는 뜻이 아니다.
사용한 전체 가시성 문턱과 화각·최소32/16픽셀 조건은 유지했다.

256장면 생성 합계22.891초, 공동 영역 측정·선택은4.427초였다. 후자는 object state packing,
ray tracing·가시성·OBB와 host metadata를 포함하므로 순수 GPU kernel 시간으로 부르지 않는다.
초기 setup이나 IK·PhysX step·PPO update를 포함하는 전체 학습 속도도 아니다.

원본은 [constructive_layout_20261022](../outputs/contact_manipulation/constructive_layout_20261022/)에 둔다.
실제 메시 결과는 [actual_mesh_joint_domain_v5/actual_mesh_probe.json](../outputs/contact_manipulation/constructive_layout_20261022/actual_mesh_joint_domain_v5/actual_mesh_probe.json),
기하 stress는 `actual_hull_1024_generation_stress.json`, 전체 CPU 결과는
`mdp_tests_final.log`와 `learning_tests_final.log`다. 단일 시점의 실패 원본과 guard-only native 실패도
삭제하지 않았다. 생성 규칙·관측 모델의 전체 설명은 [SPAWN_RULES.md](SPAWN_RULES.md)를 따른다.

### 유지한 물리 경로와 남은 검증

미사용 후보의 parking은 기존 native permanent wrench composer의 COM 중력 상쇄를 유지한다.
설치된 SDK에 gravity flag setter/getter가 있지만, 과거 같은 direct GPU 실행에서 getter가1이어도
물체가 실제 중력으로 낙하한 반례가 있으므로 API 존재만으로 이 경로를 대체하지 않았다.
실험적인 flag 변경은 원복했고 기존 parking 회귀2개가 통과했다. P=30 확대의 물리 비용을
배치 기하 계산 최적화와 섞어 해석하지 않는다.

### Native scene 교체에서 드러난 로봇 domain probe gate 수정

공동 영역 버전의 첫 native1,024환경·26iteration 목표 실행은iteration0–21의22회 update 뒤800step
scene 교체에서 중단됐다. 새 장면의 물체 배치·실제 메시 가시성이 아니라, 로봇의 임의6D목표
8개를 한 번씩 검사해 `2/8 < 4`였다는 추가 승인 gate가 중단 원인이었다.
이미 유효한 기준 자세를 찾았어도 임의 소표본 때문에 공유 장면을 버리는 조건이었다.

이 probe는 시작 영역의 부재를 증명하지 않는다. 예를 들어 각 probe의 유효 확률을 독립적으로
0.5라고 가정해도 `P(Binomial(8,0.5)<4)=93/256≈36.33%`다. 이는 실제 유효 확률을 측정한
수치가 아니라 gate의 표본 변동을 설명하는 예다. 단일 목표·IK 초기값의 수렴 실패와 기하적으로
가능한 시작 영역이 없다는 판단도 구분해야 한다.

구성적 경로에서는 이 random4/8 gate를 제거하고 legacy 비교 경로에서만 유지하도록 바꿨다.
새 scene은 유효한 anchor를 찾고 실제 PhysX link pose와 FK·elbow·충돌 검사를 통과해야 한다.
각 환경의 연속적 위치·회전 randomization, GPU batched IK, 실패 행만 일반/좁은 영역에서 복구,
끝까지 실패한 행의 **같은 장면에서 물리 검증된 anchor 복귀**는 유지한다. 물리적인 충돌·자세
조건을 완화하거나 고정 자세 bank를 일반 초기화로 바꾼 수정이 아니다.

수정 후 [2026-10-22_13-41-29_constructive_verified_anchor_smoke](../logs/rsl_rl/contact_manipulation/2026-10-22_13-41-29_constructive_verified_anchor_smoke/)는
native1,024환경·26iteration(iteration0–25),936개 정책 step을 정상 완료하고 exit0으로 종료했다.
800step의 새 장면 교체를 포함하며 해당 iteration22의 collection3.081초·총3.60초였다.
첫 iteration20.91초는 compile을 포함한다. 이후25개의 평균은3.3208초, 중앙값3.31초이며
학습 구간 합계105.05초는 환경 초기 setup과 구분한다. 이 속도는 전체 학습 연산 측정이며
장면 생성 시간과 동일하지 않다.

Native 로그의 anchor fallback total·좁은 영역 recovery count·invalid-state termination은 모두0이었다.
즉 이번 실행에서 임의 probe gate를 제거한 뒤 시작 자세를 고정 anchor로 반복 복사한 것은 아니다.
저장된 `model_25.pt`의 Actor37개·Critic8개·optimizer135개 텐서는 모두 finite였다.
최종 MDP249개 실행 중16개 skip·실패0이며, constructive 경로에서 probe를 호출하지 않는 경우와
실제 물리 검증에 실패한 anchor는 계속 거부·rollback하는 경우를 추가 검사했다.
검증 원본은 [native_final_validation.json](../outputs/contact_manipulation/constructive_layout_20261022/native_final_validation.json)과
`native_1024_verified_anchor_smoke.log`에 보관한다. 이 실행은 본 학습의 성능 비교에서 제외한다.

이번 유한 표본의 통과가 모든 역할·yaw·구성, 접촉 동역학,
장기 학습 수렴을 보장하는 것은 아니다. 일반 XY 병진까지 막힌 회전 필수 장면 비율도 별도 과제다.

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

## 43. 2026-10-22 — 연속 파라미터 구간과 직접 생성 경로

이 절은 **생성기 개발·짧은 학습 통합 검증**이다. 초기 구성 요소 시험부터 전체 catalog와
기본 경로 전환까지의 후속 결과를 순서대로 기록한다. 최종적으로90 ordered pair의 두 역할360°
구간을 확인하고 기본 reset을 `parameterized`로 교체했다. 장기 학습 성능의 검증은 아니다.

사용자는 사전 계산한 유효 연속 구간에서 직접 샘플링하는 방식, 비균등 yaw 빈도를 허용했다.
후속 확인으로 **각 ordered Target/Blocker 종류 조합에서 두 역할 각각의 yaw support가 전체360°**여야
한다. 종류 조합별로 각도 구간을 나누어 전체 catalog만360°가 되게 하는 방식은 허용되지 않는다.

구현한 구성 요소는 다음과 같다.

- `InitialVisionSensor.precompute_translation_domain`: Target의 고정 silhouette/depth를 한 번
  구하고 ray/triangle 교차의 affine 부등식에서 Blocker 이동량의 유효 구간을 역산한다.
  후보 pose query는0회다. 공통 rigid transform만 이 구간의 광학 관계를 보존한다.
- `cache_directory`는 메시·자세·내부 파라미터·pixel/depth 기준의 fingerprint와 payload SHA를
  확인하는 atomic JSON cache다. 같은 canonical Blocker 입력의 silhouette 계산은 다른 Target에서
  재사용한다. 유사 입력을 반올림하여 같은 key로 취급하지 않는다.
- `_certify_role_translation_cells`는 초기 A0 차단율·전 이동 경로의 지지·시각 보호 strip·camera 범위를
  연속 s/yaw cell 전체에 대해 보수적으로 인증한다. 현재 이 helper의 witness는 순수 local-Y 이동이다.
- 고정 두-row background chain은 이웃 간격을 Minkowski 경계에서 직접 계산한다. 실제 visual hull의
  돌출 여유를 포함하며, 타입별 프로파일은 한 번 계산한다. DP·compaction·후보 승인 루프를 호출하지 않는다.
- `prepare_certified_scene_domain`은 시작 단계에서 역할 구간과 background 용량을 결합한다.
  `sample_certified_scene`은 CDF에서 한 파라미터 tuple을 뽑아 직접 생성한다.
  `observe_at_viewpoint`는 정해진 시점에서 crop ray로 보이는 표면만 관측에 넣으며 가시성 승인 검사를 하지 않는다.
  이 관측의 `visible_fractions=None`은 미측정이라는 뜻이다. 캐시 예측값은 별도 provenance이며 정책 입력이 아니다.
- `certified_catalog_capacity`는 허용되는 종류별 개수의 유한 집합에서 필요한 공간의 최악값을 계산한다.
  `certified_role_yaw_gaps`는 각 ordered pair의 연속 각도 합집합에서 빠진 구간을 찾는다.

실제 mesh 컴포넌트의 단일 pair 검증에서는 camera 높이+0.18m에서100개 연속 표본 모두
Target44.17–88.33%, Blocker100%, Blocker259–302pixel을 만족했다. 예측·실측 pixel 불일치는0건이었다.
한 새 Blocker 관계의 warm 계산은 약0.445초, 같은 Blocker 계산 재사용 시 약0.0455초,
완전한 disk cache hit는 약5.5ms였다. 이는 전체 캐시 생성 시간이나 학습 iteration 시간이 아니다.

새 canonical 역할 helper에서 생성한 **10물체 전체 장면64개**도 별도로 GPU 검증했다.
Target `mug_2`, Blocker `ycb_004_sugar_box`, 공통 회전±2.5°의 한 연속 family다.
최초 구간 계산0.492초·직접 생성64개0.826초·관측64개0.042초였다.
Target41.67–87.5%, Blocker100%, 양 역할 OBB 유효, 예측·실측 pixel 불일치0건이었다.
독립 기하 검사에서 이웃 간격3.037–9.966mm, 초기 A0 차단율40.41–74.81%,
선반 지지·이동 경로·camera 위치·background 시야 분리를 확인했다.

원본:

- `outputs/contact_manipulation/constructive_layout_20261022/production_inverse_component_probe_h18_blocker_cache.json`
- `outputs/contact_manipulation/constructive_layout_20261022/canonical_whole_scene_probe.json`
- `outputs/contact_manipulation/constructive_layout_20261022/strict_cached_domain_integration_probe.json`

남은 기본 전환 기준은 모든 ordered pair에서 최종 **공간 인증까지 통과한 연속 구간의360° 합집합**,
pivot이 필요한 부분 family의 연속 영역 인증, 새 경로의 실제 robot-start/reset 통합 검증이다.
단일 family의64/64 통과를 이 세 조건의 완료로 해석하지 않는다.

후속 검증 및 확정:

- 간격의10mm hard upper bound는 요구가 아니다. 새 직접 생성 경로에서 `spread_gaps=True`를
  사용하여 짧은 간격과 남는 공간에 맞는 넓은 간격을 모두 지원한다. 실제 pivot 부분 family의
  후속64장면에서는 BG 간격3.09–381.18mm가 나왔으며 지지·시야·경로 조건을 유지했다.
- `_precompute_guarded_role_family`는 asset명이나 완성 scene을 hardcode하지 않고 guard·pivot·
  연속 범위를 계산한다. 충분조건을 인증하지 못하면 startup cache에 넣지 않는다.
  실제 부분 family에서 warm compile평균6.33ms·sample평균9.99ms였고, 양방향 순수Y 이동은 막히고
  pivot+slide witness는 유효했다. 일반 XY 이동 전체의 불가능성은 증명하지 않았다.
- 전체 catalog의 한 각도 dry 검사는90 pair/180 family 모두 통과했다. 전체 각도 검사 부분 결과는
  10 pair/720 family 중693개 유효·9 pair 연속360°다. `bottle_1 → cracker_box`는 Blocker yaw가
  90°에 그쳤으며 lateral±8cm의54개 추가 관계 후100°였다. 캐시·미커버 구간은 보존하고 full sweep를
  중단했다. 이것은 기본 경로 전환의 통과 결과가 아니다.
- 사용자는 각도를 위해 주변 구성을 바꾸는 방식도 거절했다. 같은 주변 종류·개수 구성에서 두 역할의
  360°가 가능해야 한다. 보완은 시점과 pivot 후 이동 경로의 연속 인증에서 수행하며 조건을 낮추지 않는다.

자료: `pivot_required_partial_domain_probe.json`, `certified_catalog_coverage_dry.json`,
`certified_catalog_coverage_full.json`, `certified_lateral_augmentation.json`.

**같은 주변 구성의360° 보완:** `_precompute_normal_pivot_family`가 초기 실제 메시 가시성/A0
cell을 유지한 채 pivot+slide motion의 정확한 연속 bounds를 계산한다. 첫 문제 pair
`bottle_1 → cracker_box`에서 기본45 family+normal-pivot133 family의 Target/Blocker 각도 합집합은
각각360°이며 빈 구간은0이다. 모든 허용 주변 구성에 공통인 최악 module 폭0.389202927m를
유지했다. 실제 inverse/A0의 추가87관계 계산45.84초, CPU 합집합·경로 검사2.13초였고,
56개 중심의 독립 dense witness 검사가 통과했다. 전체 catalog 완료 결과는 아니다.
자료: `normal_pivot_production_coverage.json`.

**새 경로1,024 reset 통합 smoke: 실패.** 단일 family를 주입한 시험은 관측 전용 ray 경로에
도달했지만 초기 anchor IK에서 중단됐다. Process exit0은 예외를 JSON에 기록한 harness의
종료 코드이며 검증 통과를 뜻하지 않는다. `parameterized_reset_1024_probe.json`의
`status="failed"`가 판정이다. 같은 장면의1환경 진단에서 기존80개 후보 중26개가 IK에
수렴했지만 elbow-up11개와 forearm-forward12개의 교집합은0개였다. 수렴한26개 모두 외부
물체·선반·바닥 충돌 검사를 통과했다. ±10° orientation만 적용한 동일 표본도 유효0개였다.
같은 영역의 고정 TCP/orientation grid58,212개에서는2,704개가 모든 조건을 통과했고,
대표 anchor 주변729개 표본은729개 통과했다. 장면 재생성·임계값 완화 대신 초기 anchor의
체계적인 pose/branch 준비를 수정한다. 이 grid는 진단이며 reset 때 매번58,212개를 탐색하는
구현이 아니다. 자료: `parameterized_anchor_failure_probe.json`, `parameterized_anchor_domain_probe.json`.

**초기 anchor 보완 후 통합 재검증:** 최초 template만 위치3×3×3 strata와 방향9개, 기존 IK branch4개로
구성한972개 고정 domain을 사용한다. 최종 clone의 시작은 기존 연속 randomization이며 부분 reset의
후보 예산80·검증 anchor 복구도 유지한다. 배치/가시성 재시도는 추가하지 않았다.
`parameterized_reset_1024_structured_probe.json`은 첫 reset1.939초,5-row 부분 reset0.263초,
새 scene reset0.106초로 완료됐다. Scene 교체 deadline을 시험에서만 앞당겼으며 실제800step이나
학습 iteration을 수행한 검증은 아니다. 새 두 scene에 각각 관측1회·ray1회만 있었고,
부분 reset에서는 추가 생성/관측/ray0회·나머지1,019 row bitexact·fallback0이었다.
같은 family의 다른32개 장면도 제외/재시도 없이 native IK/FK32/32 통과, 전체 producer0.855초였다.
단일 family의 통합 검증이며 전체 종류/각도 catalog의 통과를 뜻하지 않는다.

Zero-action10step에서는 `singularity` 종료2개, 나머지 실패 원인 및 success/timeout은0개였다.
따라서 초기 자세 검증에도 **기존 활성 singularity 종료 기준과 동일한 판정**을 연결하는 보완을
진행한다. 임계값을 완화하거나 이2개를 정상 초기 상태로 무시하지 않는다.

**전체 연속 영역 catalog 완료:** 실제10종의 ordered Target/Blocker90 pair 모두에서 두 역할의
연속 yaw 구간 합집합이 각각360°를 덮었다. 같은 주변 종류·개수 구성에 공통으로 유효하도록
pair별 최악 공간 예산을 사용한다. 총6,593 family·207,027 cell이며 normal pivot101개,
pivot이 필요한 guarded family1개를 포함한다.90개 pair checkpoint의 source/payload 검증과
reload가 모두 통과했다. Cold 구축2,020.679초(약33.68분), 이 중 categorical capacity 계산9.055초다.
이 비용은 시작 전 캐시 구축이며 reset/iteration 시간으로 해석하지 않는다.

구성적으로 직접 생성한 각 pair4장면, 총360장면은 제외/재생성 없이 독립 기하 검사·실제 메시
가시성·framing·역할 관측을 모두 통과했다. Normal-pivot 표본12개를 명시적으로 포함한다.
Target 실측41.279–87.952%, Blocker100%; 관측 전용360 views0.174초, 독립 full-role trace0.144초,
전체 기하/관측 회귀22.471초였다. Pivot-required guarded family의 parameter 끝점4장면도
경로·기하·framing·역할 관측을 통과했고 Target52.410%, Blocker100%였다.

정밀도 한계도 기록한다. 보통 장면2개에서 Target count 예측과 실측이 각각−2/−1 pixel,
1개에서 Blocker silhouette가+2 pixel 달랐다. 모든 실측 가시율은 조건 안에 있었으며 guarded
4장면은 count도 일치했다.2-pixel guard는 이 관측 오차를 덮었으나 모든 float32 변환에 대한
완전한 수치 증명이나 bit-exact matching을 주장하지 않는다. 연속 구간 인증은 모델링한 실제
메시의 기하식에 대한 것이며364회 ray 검사는 별도의 회귀다.

전체207,027 cell의 camera lateral 범위를 끝점 및 각도 정지점에서 계산한 결과
`[−0.255960,+0.254741]m`였다. OBB center/native body root 기준 모두 설정±0.35m 안이다.
자료는 `actual_compiler_v4/report.json`, `scene_regression.json`,
`guarded_scene_regression.json`, `camera_lateral_audit.json`이다.

**캐시 원점 불변성과 native 초기 자세:** world 변환의 상쇄 및 BBoxCache 내부 계산이
환경 원점에 따라 최대 약1e−15의 hull/bounds 차이를 만들어 정확한 cache key를 바꾸는 문제가
있었다. 상대 환경 변환 API와 임시 in-memory composed-body reference에서의 기존 BBox 계산으로
해결했다. 형상 근사/hash 반올림/live stage 변경은 하지 않았다. 실제1·128·1,024환경의 source가
기존 v4 fingerprint `b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`과
정확히 일치하여33.68분의 캐시를 그대로 재사용한다. 기본 경로에도8,326 JSON 파일을 복사했다.

90개 ordered pair에서 각각 한 번 직접 만든 장면의 native anchor는90/90 통과했다.
Native FK/특이점 실패0개, 최소 native margin0.0146109다. 새 COM 판정 추가 전의1024 비교에서
Jacobian 차이도 검출했다. Link-origin 판정의 선속도 최대 오차0.0227381, 각속도 오차8.05e−7였고,
native wrist_3 COM `(0,0,−0.02289999835)m`으로 기준점을 맞추자 전체 오차8.05e−7가 됐다.
기존 공유 singularity 식/threshold는 유지하고 초기 candidate의 기준점만 수정했다. COM snapshot은
PoseModel에 최초1회 캐시하며 CUDA graph/scene cache key에도 해당 COM tuple을 포함한다.
자료: `certified_all_pairs_anchor_probe.json`, `parameterized_native_jacobian_1env.json` 및 source 비교 기록.

**최종 1,024환경 reset smoke와 기본 전환:**
`parameterized_reset_1024_singularity_probe.json`에서 두 전체 reset 직후 native singularity 실패는
각각 0개였다. 최소 margin 0.005014884 / 0.054135174는 기존 임계값 0.005 이상이다.
첫 물리 step 후 경계 부근 1행이 singularity로 종료됐으나 초기 invalid pose는 아니었다.
그 외 invalid/shelf/support/tilt 종료는 0개였다. 추가 32장면의 producer도 모두 통과했다.
5행 부분 reset은 다른 1,019행을 bit-exact하게 유지하고 추가 scene/observation/ray query 0회,
anchor fallback 0회였다. 각 새 장면의 observation은 1회만 생성했다.
초기 reset 1.960초(cold graph), 5행 부분 reset 0.265초(새 graph bucket), warm 전체 reset 0.117초,
추가 32장면의 producer 합계 0.918초다. PPO iteration 측정과 혼동하지 않는다.

**기본 설정의 공식 학습 통합 시험 완료:**
로그는 `logs/rsl_rl/contact_manipulation/2026-10-22_17-19-09_parameterized_default_smoke`다.
Mode/cache override 없이 실제 `parameterized` 기본값과 기본 cache 경로를 사용했다.
전체 manifest를 재계산 없이 로드했으며 JSON/payload load 7.298초, catalog 초기화 8.754초였다.
이는 최초 시작 비용이며 reset마다 발생하지 않는다.

26 iteration·936 policy step을 완료했고 실제 global step 800에서 장면 generation이 0→1로
교체됐다. Step 0/800의 두 전체 reset 모두 초기 native singularity 위반 0개였고 최소 margin은
0.008071624 / 0.006082743이었다. Reset fallback은 0회였으며 actor·critic·optimizer tensor는
종료 시 모두 유한했다. Cold capture를 포함한 학습 구간은 108.254초였다.
자료는 `parameterized_train_native_probe.json`이다. 기존 학습 진입점·공식 PPO·controller·logger는
그대로 사용하고 경계 두 번의 기록과 종료 후 유한성 확인만 추가한 시험이다.
짧은 통합 실행으로 장기 안정성이나 정책 성능을 보장하지 않는다.

프로세스는 exit 0으로 종료했고 `model_0.pt`, `model_25.pt`를 저장했다. 첫 iteration의 cold 비용을
제외한 25 iteration은 평균 3.421초·중앙값 3.35초·범위 2.99–4.21초였다. 평균 collection은
2.899초, PPO learning은 0.522초다. `invalid_state`는 모든 26 iteration에서 0이었다.
정책 동작에 의한 singularity·tilt 등의 종료는 발생했고 success는 0이므로 이 결과를 학습 성능
개선으로 해석하지 않는다. 집계는 `parameterized_train_native_summary.json`에 보존했다.

## 44. 2026-10-22 — 통합 clutter 생성과 고정10-slot 관측 변경

이 절은 요구 변경과 개발 검증 기록이며 본 학습 결과가 아니다. 이전43절의10물체·26 iteration
smoke를 새20–40물체 설정의 검증으로 사용하지 않는다.

사용자는 역할 물체를 먼저 놓고 나머지를 두-row 잔여 영역에 배치하는 규칙에서 두 그룹이
분리되어 보인다고 지적했다. 새 생성기는 전체 물체가 하나의 랜덤 clutter를 이루면서
Target40–90%·Blocker95% 이상 가시성과 기존 조작·yaw 조건을 만족하도록 변경해야 한다.
90개 ordered pair는10종의 `10×9` 역할 조합이며90개 완성 scene의 목록이 아니다.

물리 개수는 같은 선반에서20–40개로 공간 수용 가능성에 연동한다. 기존4–10종 선택과 Target
종류1개·non-Target 균형 중복은 유지한다. 큰 종류를 제외해 개수를 채우는 방식은 쓰지 않는다.
관측은 실제 활성 개수와 독립적인10개 slot·244D를 유지한다. Target·Blocker의 slot0/1 뒤에는
초기 vision으로 인식한 물체 중 두 역할의 관측 중심까지 최소 유클리드 거리가 작은8개를 고른다.
비가시 물체는 정답 정보로 채우지 않는다. Reward·termination은 전체 활성 물체에 적용한다.

물리 최대값 `CONTACT_MANIPULATION_NUM_OBJECTS`의 기본40 변경은 `scene.py`에 적용했다.
최소4종의 임의 구성에 대응하려면 각 종류13개씩,130개 rigid-body pool이 필요하다. 현재의
native 주차·중력 상쇄 composer는 유지하며 per-reset USD 생성이나 topology 변경은 하지 않는다.
기존30-body pool보다1,024환경에서102,400개 body가 추가되므로 GPU 메모리와 PhysX·센서·reset
비용을 다시 측정해야 한다. 공식 MultiAssetSpawner의 초기 asset 선택만으로 이 pool을40개로
줄일 수는 없다. 종류 구성을 episode마다 바꿀 수 있어야 하기 때문이다.

변경한 기본값의 Python 문법 검사와 실제 source의 capacity 검사는 통과했다. Capacity 검사는
N=4/10/40/41/100의 모든 허용K에서 균형 개수가 복사본 상한 안에 들어오는지와 pool 이름 유일성을
확인했다. 이는 선반 packing·카메라·학습 성능 검사가 아니다. 통합 clutter와20–40개 공간 영역,
고정10-slot observation 선택 및 native 대규모 실행은 구현·검증 중이다.

후속 확정: 전체 이동·pivot 경로를 예약하는 조건은 제거하고 Blocker를 옮길 **빈 목적 영역만
확보**한다. 따라서43절의 full-path witness·guard 인증은 이전10물체 경로의 기록이며 새 통합
clutter의 연결 경로나 조작 성공 가능성을 보증하지 않는다. 목적 공간이 비어 있어도 도달하려면
주변 접촉·pivot·여러 단계 조작이 필요할 수 있다.

조건부 위치 mixing helper는 고정3회 sweep, 물체당 X·Y2회씩 정확한 free-interval union에서
직접 추출한다. 유효 seed를 승인할 때까지 다시 뽑는 과정은 없으며 yaw·종류 구성은 유지한다.
Minkowski pair 준비를 Numba로 묶어 반복 Python 배열 생성을 제거했다. 40개 물체의 전체 준비와
240개 좌표 draw는 footprint4/64/500정점에서 평균5.07/14.09/78.72ms, 순수 sweep은
0.475/1.429/8.365ms였다. Python pair 준비를 사용한 대조는114/144/360ms였다. 공유 장면 한 번의
CPU 측정이며 환경별 실행이나 학습 iteration 비용이 아니다. 다른 CPU 작업과의 동시 실행도
있어 이 미세 측정으로 end-to-end 속도 향상을 단정하지 않는다.

독립 polygon distance로 전체 pair·고정 영역·선반 지지를 검사한140장면과 기존
`footprint_line_limit`의 upper/lower171개 비교는 통과했다. 고정3mm gap의 접선·고립된 유효
중심, 입력 불변성·재현성·잘못된 초기 상태 명시 실패를 포함한6개 회귀도 통과했다.
자료는 `outputs/contact_manipulation/clutter_mixing_20261022/report.json`, `timing.json`과
`timing_before_numba.json`이다. 통합20–40개 생성기와 native GPU 실행 검증은 진행 중이다.

**130-body source-only native 검사:** 기본 물리 최대40과130-body pool로1환경을 초기화했다.
Episode 생성·reset·catalog 재계산은 예외 hook으로 차단하고 geometry/resources 준비 및 기존
reference10 역할 cache source만 비교했다. 최초 source 불일치는 pool 크기 증가로 바뀐 mug의
초기 주차 변환에서 약4e−16 bounds, cracker에서 약1.1e−16 collision 차이가 난 경우였다.
첫3개 복사본의 주차 slot을 기존30-body pool과 같게 두고 나머지를 뒤에 배정하는 순열로 수정했다.
130개 주차 위치의 집합 자체·활성 물체·reset state writer는 같으며 hash 반올림은 하지 않았다.

수정 후 native source는 기존 `b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`과
정확히 일치했다. 물리 ID `[1,40]`과 관측 geometry `[1,10,10]`를 확인했고166.99MB 기존 manifest의
source·payload checksum 검증과 읽기 전용 로드를7.961초에 마쳤다. Scene 초기화23.492초,
geometry/resources2.138초였다. Torch allocated57.6MB·reserved201.3MB였고 AppLauncher 완료
시점 대비 device used 증가는약2.70GB였다. 이 값에는 native PhysX·renderer 비용이 포함되므로
1,024환경의 추가 메모리를 단순 비례 추정하지 않는다. Reset·학습을 실행하지 않은 검사다.
결과는 `clutter_mixing_20261022/native_130_source_probe.json`; 최초 실패 자료는
`native_130_source_before_parking_fix.json`에 보존했다. 새 packing 수용 영역과 실제 scene의
가시성·로봇 초기 자세·학습 처리량은 이 결과로 검증됐다고 보지 않는다.

**현재 runtime 연결과 검증 대기:** 기본 모드를 `clutter`로 연결했다. Reference10 catalog의
T/B 상대 pose·시점·가림 정보만 쓰며 기존 background 용량 인증·full-path reserve는 사용하지
않는다. GPU physical seed 뒤에 단일 CPU 공동 제약 계산으로 필수 BG 전체와 빈 목적지를 함께
놓고 고정3회 free-interval mixing으로 주변 위치를 섞는다. 추가 수량은 동일 종류 구성의 균형
prefix다. 이 부분은 내부 최적화가 없는 GPU-only 공식이 아니며 무작위 장면 재생성 loop와
구분한다.

Production GPU seed는 mandatory head 재정렬·원래 row ID 보존·누락된 필수 행의 부분 graph
반환을 지원한다. Optional tail은 원래 순서를 유지한다. Collider/전체 OBB의 서로 다른 지지
경계, 벽에 불필요한 gap/2를 추가하지 않는 조건, 잘못된 입력·누락 행을 확인한 CUDA 회귀6개가
통과했다. 어려운8장면에서 자유 yaw1개와π/±.001rad 변화를 포함해 필수18개 BG가 모두 seed됐다.
독립 polygon distance의 최소 간격은3.199969mm, warm110–120ms, 최대 CUDA 할당480.45MB,
FFT 정수 roundtrip 오차는최대4.37e−11이었다. 자료는
`clutter_mixing_20261022/production_gpu_seed_free_yaw_probe.json`이다.

새 기본값4환경 reset·30step과 independent full-silhouette1회 검사를 위한
`clutter_mixing_20261022/native_clutter_reset_probe.py`를 준비했다. 아직 실행 전이며 앞의 GPU
seed 성공을 최종 pocket·물리·가시성·정책 학습 통과로 확대 해석하지 않는다.

최종 CPU 전체331개(33.084초·21skip), 학습 관련56개(2skip)와 camera-Y 표적3개를 통과했다.
별도90-pair initial-domain 검사는37.14초에 normal6,592 family·103,513 dedup cell에서 역할별
360° coverage와 camera lateral±.35m를 확인했다. 자료는
`clutter_initial_angle_coverage_with_lateral_audit.json`이다. Background 전체의 연속 공간 인증으로
해석하지 않는다.

**기본4환경 native 검사 통과:** `native_clutter_reset_4_probe.json`은 seed42의24물체·10종 장면을
새 기본 `clutter`로 생성했다. Target1개·non-Target2–3개씩이며 독립 실제 메시 검사에서
Target68.85246%·Blocker100%, framing·최소 pixel을 통과했다. 실제 collider pair 최소 간격은
3.36546mm, support 여유 최솟값0.180515mm였다. Scene1회에 GPU seed1회·CPU 공동 solve1회·
고정 mixing1회·초기 관측1회를 실행했고 full-silhouette query1회는 검증용 추가로 구분했다.

정책 `[4,244]`, 물리 ID `[4,40]`, geometry `[4,10,10]`, tactile force `[4,17,3]`; 초기 singularity는
0행·최소 margin0.14111이었다. 부분 reset은 다른3행을 bit-exact하게 유지하고 layout/관측/ray를
추가하지 않았다. 30 zero-action step의 상태는 모두 유한했고 모든 종료 원인·fallback은0이었다.
이 검사는4환경 단일 scene이며 새90개 조합 전체나1,024환경의 안정성을 입증하지 않는다.

Scene 초기화23.762초, 첫 reset20.477초에는 catalog9.489초·GPU seed cold0.791초·CPU joint0.124초·
Numba mixing cold4.838초·초기 anchor IK1.616초 등이 포함된다. 부분 reset0.725초는 새 IK bucket
비용을 포함한다. 30step3.590초는 진단·cold graph가 포함된 짧은 검사로 iteration 처리량이 아니다.
마지막 GPU used5.386GB/free19.901GB, Torch allocated1.090GB/reserved1.357GB, host RSS5.558GB였다.
1,024환경의130-body pool 비용은 별도 native 실행에서 측정해야 한다.

**최종 standalone98장면:** 모든90 ordered pair를 각각 고정 seed 하나로 검사하고 어려운 큰
물체4종 구성8개를 추가해 모두 통과했다. 90개 장면의 활성 개수는20–38개·종류4–10종이다.
첫 JIT2.516초를 제외한 생성 중앙값0.3433초·95백분위0.5740초·최대0.9173초였다. 최대 공동 제약
잔차0.678µm는10µm guard보다 작았다. 큰4종8개는 요청39–40개에서 같은 종류 구성을 유지하며
실제20–23개를 생성했다. 이 기록은40개 scene 실증이나 모든 종류·각도 조합의 증명이 아니다.
자료는 `constructive_layout_20261022/production_clutter_sampler_summary.json`과 해당 probe다.

**1,024환경 native 검사 통과:** `native_clutter_reset_1024_probe.json`에서130-body pool,
24 active·10종의 공통 scene을 생성했다. 정책 `[1024,244]`, 물리 ID `[1024,40]`, geometry
`[1024,10,10]`와 모든 환경의10개 유효 관측 slot을 확인했다. 독립 full-silhouette1회에서
Target68.85246%·Blocker100%, framing·pixel 조건을 통과했다. 실제 pair 최소 간격3.42087mm,
support 최솟값0.742268mm였다. 초기 singularity0행·최소 margin0.028813이었다.

부분 reset은 다른1,023행을 bit-exact하게 유지하고 layout/관측/ray를 추가하지 않았다.
10 zero-action step은 모두 유한했고 모든 종료 원인·fallback은0이었다. GPU used10.839GB·
free14.448GB, Torch allocated1.287GB·peak1.322GB, host RSS9.537GB였으며 OOM·PhysX 오류가 없었다.
이는 단일 scene·짧은 실행의 결과다.

Scene 초기화51.699초·첫 reset15.837초에는 catalog9.752초와 geometry·IK 준비가 포함됐다.
공유 장면 생성0.583초 중 GPU seed0.379초·CPU 공동 solve0.158초·mixing0.024초였고
초기 관측0.0475초, 부분 reset0.807초였다. 10step4.318초는 cold graph·진단 동작을 포함하므로
정규 학습 iteration 속도로 환산하지 않는다. 최종 CUDA 표적 회귀6개도 통과했으며 정규 PPO
학습 처리량·장기 수렴은 이번 변경의 검증 범위에 포함하지 않았다.

## 45. 2026-10-22 — Blocker 전용 접촉 필터로 축소

사용자 승인으로 `TaskContactSensor`의 reward용 filter를 전체80-body pool에서 공유 장면의
Blocker 하나로 줄였다. 17개 tactile body의 native 순서와 각 clone의 Blocker 경로를 공식
PhysX `create_rigid_contact_view`에 연결한다. 전체 net-force는 필터와 무관하므로
상대 구분 없는 binary observation17개·244D 관측·접촉 유지 reward 수식은 그대로다.

이전 `[N,17,80,3]` force matrix 복사와 GPU Blocker-column gather를 제거하고
`[N,17,3]`을 직접 읽는다. Native filter의 패턴은 변경 API가 없으므로 공유 Blocker ID가 바뀔
때 view만 교체한다. 물리 body나 ContactSensor를 다시 spawn하지 않는다. 부분 reset과
동일 Blocker ID는 재사용하며, 원본 net-only view 하나 외에 과거 filtered view를 캐시하지 않는다.
Native 초기화 후 scene reset에서 Python `sample.blocker_id`를 전달하므로 step마다 CUDA
role ID를 host로 읽지 않는다. Reset 없이 Stop/Play를 재개한 경우도 보관한 host ID로 다시 연결한다.
새 역할·새 view는 이전 유지 이력을 지우고 첫 접촉 sample의 유지 보상은0이다.

기본1,024환경에서 native matrix와 task cache 합은 약31.88MiB→0.40MiB다. 이는 pair buffer
크기이며 전체 GPU 메모리·물리 충돌 비용·PPO iteration의80배 감소를 의미하지 않는다.
접촉 유지 term을 제거하거나 시작부터 weight0이면 native filter를 생성하지 않는다.

**실제 PhysX4환경 검증:** 각 환경에17pad·80cuboid를 만들고 Blocker와 non-Blocker를
서로 다른 pad에 접촉시켰다. 기존80열 native view와 새1열의 force·net binary·유지 count가
일치했고 최대 force 차이는0이었다. 다른 물체만 접촉한 pad는 binary True·Blocker 접촉 False였다.
역할24회 변경, 같은 ID 재호출·부분 reset 시 view 재사용, 선택하지 않은 행의 force 보존,
native view 생성 실패 시 기존 view/ID/cache 보존, 원본 view 복귀를 통과했다.
마지막16회 교체에서 torch allocated/reserved 증가0이었다. 누적 Python view 참조가 없었고
device free 메모리는4.9MB 늘었다. 이는 짧은 반복 검사이며 장기 메모리 안정성 보증은 아니다.

동일 `TaskContactSensor.update`/native net·matrix read/cache 경로로 비교한4환경 미세 측정은
기존80열+gather 중앙값0.12314ms, 새1열0.09605ms로 접촉 버퍼 처리 구간이 약22% 감소했다.
20개 block·block당30회 read, block 경계 CUDA 동기화를 사용했다. 초기의 raw getter 대
production update 비대칭 비교는 별도 파일에 보관하고 이 수치에는 사용하지 않았다.
이 측정은 simulation step·reset·PPO 전체 시간이나1,024환경 iteration 개선폭이 아니다.

재현 자료는 `outputs/contact_manipulation/blocker_filter_20261022/`의
`native_check.py`, `native_check_4.json`이다. Probe의 `SimulationContext.stop()` 대기는
공식 `_app_control_on_stop_handle_fn`의 play 대기 loop임을 stack으로 확인했으며 검증 완료
후 해당 probe process만 정리했다. 실제 task 연결과 CPU 회귀 결과는 아래에 덧붙인다.

최종 CPU 전체 검사는340개 중319개 통과·CUDA 전용21개 skip이었다. 관련 contact/reset
46개 중45개 통과·CUDA 전용1개 skip이며, Stop/Play 후 reset 없이 cached host role로
재연결·최초 미설정 호출의 명확한 오류·다른 환경 history 보존을 포함한다. 구문 검사와
`git diff --check`도 통과했다. 테스트만을 위한 production 파일은 추가하지 않았다.

**실제 프로젝트4환경 연결도 통과:** 기본15–25개 설정·80-body pool에서 초기
RewardManager 생성 시에는 필터를 만들지 않고 첫 reset 뒤 Blocker20의1열 `[4,17,3]`을
연결했다. 244D 관측·17채널 tactile·2개 policy step의 유한 상태를 확인했다. 부분 reset은
view를 재사용하며 다른 행의 접촉 이력을 보존했다. 완료된 transition에서 공통 장면을
만료시켜 generation0→1·Blocker20→58로 바꿨을 때 native view가 교체됐고 이후2step도
정상 동작했다. `env.close()`와 `app.close()`는 정상 종료됐다. 첫 진단에서 common step을
임의로 건너뛴 경우 발생한 기존 completed-transition guard는 진단의 경계 설정을 수정해
재검증했으며 production guard는 변경하지 않았다. 최종 자료는 `native_task_check.py`와
`native_task_check_4.json`이다. 본 학습이나1,024환경 전체 iteration 측정은 실행하지 않았다.

## 46. 2026-10-22 — 23:26:39 학습의 weight와 민감도 검토

대상은 2026-10-21_23-26-39의0–5,221iteration이다. Git 기록은 clean commit
b37c615이며 저장 reward 공식·weight·decay와 현재 설정이 일치한다. 당시 물체는10개,
현재는15–25개이므로 합산 Rearrangement와 전체 물체의 지지/성공 조건에 같은 난도를
가정하지 않는다. **이번 작업은 분석만 수행했으며 아래 후보는 적용하지 않았다.**

로그 시각 범위는 약4.86시간이다. Episode_Reward는 weight가 적용된 episode 누적값을
고정8초로 나누고 reset callback별로 평균한 값이다. Adapter의 /dt로 native dt가 상쇄된다.
아래 표는 step reward나 물리량의 평균이 아니다. Episode_Termination은 각 환경의
마지막 종료 원인 snapshot이므로 실제 종료 건수 비율로 해석하지 않는다.
성공률은 Task_Count 누적값의 차이로 계산했다. Callback 평균에 따른 작은 경계 오차가 있다.

| 지표 | 최상250구간 1136–1385 | 마지막250구간 4972–5221 |
|---|---:|---:|
| 유효 성공률 | 4.923% | 2.345% |
| Clearance | 13.987 | 18.094 |
| Distance | 3.218 | 2.828 |
| Contact hold | 0.0557 | 0.0768 |
| Rearrangement | −0.693 | −0.586 |
| Action rate | −2.191 | −2.272 |
| Joint velocity | −0.507 | −0.478 |
| F/T growth | −2.902 | −2.795 |
| Singularity | −0.0834 | −0.0692 |
| Object stability | −0.946 | −1.411 |
| Terminal | 1.018 | 0.693 |
| 양의 항 합 | 18.279 | 21.692 |
| Penalty 합 | −7.323 | −7.612 |

전체 완료 약1,696,699episode 중 유효 성공은 약34,242건(2.018%)이다.
Raw goal을 한 번 이상 관측한 episode는23.72%, unsupported goal17.39%,
goal과 실패가 동시에 발생한 episode16.81%였다. 세 분류는 서로 배타적이지 않다.
Raw goal episode 중73.29%는 unsupported goal을 한 번 이상 포함했다.
이것은 들린 물체의 비율이 아니다. Progress validity에는 지지 gap 외에도
중심 XY 이탈, 전도, penetration, 유한값 검사 등이 포함된다.

Clearance 기여는29.4% 늘었지만 성공은 감소했고 stability 비용은49.2% 늘었다.
Penalty 총합이 진행 보상을 압도한다는 단독 설명의 근거는 약하다.
안전하게 마무리되지 않는 진행과 보상의 관계를 우선 검토한다. 단, 다른 장면·정책의
집계이므로 원인을 확정하지 않는다. Hold는 양의 지급 중0.30–0.35%지만 낮은 weight 외에
접촉 희소성·짧은 유지도 원인일 수 있다.

**수치 이상은 별도 문제다.** Value loss는11회에1.81e6,65회에3.61e13을 기록했다.
전체 median154.4, p95 3.90e12, 최대1.295e16이며 정상값과 극단값이 반복된다.
1e12이상은391/5,222회(7.49%)이고 후기 빈도가 증가한다. 영구적인 단조 붕괴는 아니다.
Gaussian std는 약1.0→6.81로 증가했다. CPU checkpoint0/500/1500/3000/5200의
Actor·Critic·Adam은 모두 finite였고 Critic parameter L2는12.29→40.54였다.
그 자체의 weight 폭발을 확인한 것은 아니다. 정상 reward의 보수적 할인 return 범위
약−321.43~+288.93에 비해 손실은 매우 크다. 큰 finite observation/예측값/bootstrapped
target을 의심할 수 있으나 실제 rollout이 없어 원인은 미확정이다. Invalid state는 전 구간0이다.
Weight 조정만으로 이 수치 문제가 해결된다고 판단하지 않는다.

1차 비교 후보는 수식·민감도를 고정하고 다음 weight/성공 보상을 검토하는 것이다.

| 항목 | 현재 | 1차 후보 |
|---|---:|---:|
| Clearance | 4.0 | 유지 |
| Distance | 0.4 | 유지 |
| Hold/pad | 0.05 | 0.10 |
| Rearrangement | 1.0 | 유지 |
| Action rate / Joint velocity | 각0.15 | 유지 |
| F/T growth / Singularity | 각0.30 | 유지 |
| Object stability | 2.0 | 3.0 |
| Success bonus (Terminal weight1) | 100 | 250 |

Clearance는 초기 대비 개선된 상태를 매 step 지급하므로 먼저 더 키우지 않는다.
Hold는 희소성의 원인이 불명확하므로2배 비교 후보로 제한한다. 이는 접촉한 채 기다리는
보상도 키우므로 Terminal도 함께 검토한다. 새 Hold 상한1.7/step을 포함한 양의 dense
전역 상한은4(1−exp(−1))+.4+1.7=4.62848/step이다. γ=.98에서 할인합 상한은231.424다.
Success250은 이를 넘는 충분조건이며 실제17pad 동시 접촉이나 성공 회피가 관측됐다는
뜻도, 최적 bonus가250이라는 실험 결과도 아니다.
Rearrangement는 물체 수 증가에 따라 같은 개별 흔들림에도 합계가 커지므로 더 높이지 않는다.

민감도는 다음처럼 해석한다.

- Clearance a1 유지: 접근 영역과 Blocker XY footprint의 교집합 비율이 x다.
  카메라 가시성 비율이 아니다. 예시 x=.8→.7의 지급은 a1에서+.189,
  a2에서+.179, a3에서+.127이다. a 증가가 초기 진행 신호를 약하게 할 수 있다.
- Distance a=ln2/.10m=6.931/m 유지: 거리0/10/20/30cm에서+.4/.2/.1/.05다.
  더 큰 a는 먼 거리의 접근 신호를 줄이므로 지금 올릴 근거가 부족하다.
- Rearrangement a20/m 유지: 합산1/5/10mm 변위에서−.0198/.0952/.1813이다.
- Action rate/Joint velocity a=.004 유지: 제곱합173.3에서 각 weight의 절반 비용이다.
- F/T aF=8.557e−5, aT=.034229 유지: 증가율 deadband10N/s·.5Nm/s 이후 제곱을
  .5배 혼합한다. 한 채널만 변하면 half penalty는137.3N/s 또는6.864Nm/s다.
  평균 penalty만으로 포화 비율을 알 수 없어 먼저 계수를 완화하지 않는다.
- Singularity warning .07·hard .005와 현재 decay도 유지한다.

Stability는1차에 weight만3으로 높이고 경고10mm·a=ln2/(10mm)^2를 유지한다.
5mm부터 경고하는 안은 지지 거리의 정지 오차·실제 실패 원인을 확인한 뒤 별도로 비교한다.
기존 급격한 곡선을 왼쪽으로만 옮기면 정상 조작의 접촉 회피를 강화할 수 있다.
후속 곡선 후보는 경고5mm, a=ln2/(15mm)^2≈3080.65/m²로 상승 구간도 넓히는 것이다.
Half penalty 지점은 기존과 같은 총 지지 거리20mm로 유지된다.

| 지지 거리 | 현재 w2·경고10mm | 1차 w3·경고10mm | 별도 후보 w3·경고5mm·넓은 상승 |
|---|---:|---:|---:|
| 5mm | 0 | 0 | 0 |
| 10mm | 0 | 0 | −.222 |
| 15mm | −.318 | −.477 | −.795 |
| 20mm | −1.000 | −1.500 | −1.500 |
| 30mm | −1.875 | −2.813 | −2.563 |

거리, 힘·토크 증가율, action 차이, 지지 거리, singularity margin의 원시 분포는 미기록이다.
평균 exp 점수에서 적정 a를 역추정하지 않는다. 후속 비교는 유효 성공·raw goal 전환·
지지 실패의 원인과 함께 각 x의 분포 및 exp 포화 비율을 확인해야 한다.
Value spike의 관측/Value/return 진단은 보상 재조정과 분리하며 공식 PPO 구조는 유지한다.

근거·산출물은 outputs/contact_manipulation/reward_review_20261021_232639/의
metrics_agent.json, config_agent.json, checkpoints_agent.json, value_sanity_agent.json,
recommendations.json, training_review.png에 보존했다. Production 설정과 기존 학습 파일은
변경하지 않았다.

## 47. 2026-10-22 — 보상 weight 1차 조정 적용

사용자가46절의1차 비교안을 승인해 다음 세 값만 변경했다. 기존 로그·checkpoint·저장된
params는 수정하지 않으며 다음 학습의 기본 설정부터 적용한다.

| 항목 | 이전 | 적용값 |
|---|---:|---:|
| `blocker_contact_hold.weight` | 0.05 | 0.10 |
| `object_stability.weight` | 2.0 | 3.0 |
| `MdpCfg.success_reward` | 100 | 250 |

Terminal의 native weight1.0과 `/step_dt` 어댑터를 유지하므로 유효 성공에+250을 한 번
지급하고 실패·timeout에는0을 지급한다. 활성 실패 우선과.5초 유효 완료 조건은 유지한다.
Hold는 동일 pad의 직전·현재 Blocker 접촉을 합산하며1/3/17pad 유지 시 각각.10/.30/1.70을
받는다. Binary tactile observation과 Blocker 전용 필터 동작은 바꾸지 않았다.

Stability 경고 시작10mm와 `a_P=ln(2)/(.01m)^2`는 유지했다. 지지 거리10mm까지0,
15mm에서약−.4773,20mm에서−1.5,30mm에서−2.8125다. 같은 물체의30mm/.1초 지지 실패
조건도 유지한다. 경고5mm와 더 넓은 상승 구간은 이번에 적용하지 않은 후속 실험 후보다.

Clearance4·Distance.4·Rearrangement1·Motion각 weight와 모든 지수 민감도·threshold·
action·공식 PPO 설정은 유지한다. 양의 dense 전역 상한은
`4(1−exp(−1))+.4+17×.10=4.628482/step`이고 현재γ=.98의 무한 할인합 상한은
`4.628482/(1−.98)=231.424112`다. 실제 장면의 Clearance 상한·접촉 pad 수는 더 작을 수 있다.
성공+250은 이 상한을 참고한1차 실험값이며 성공까지의 지연 할인과 실제 궤적까지 고려한
최적값이나 성공 선호의 보장은 아니다. Stability 비용의 하한은−2에서−3으로 바뀐다.

기존 회귀의 기대값을 갱신하고 관련109개 검사를 실행해106개 통과·CUDA3개 skip을 확인했다.
범위는 ContactReward/Reward/Termination/ObjectSupport/State/NativeAdapter/
BlockerContactManager이며 `py_compile`도 통과했다. Parameter만 조정했으므로 새 test 파일이나
simulation 실행은 추가하지 않았다. 새 학습·성공률·전체 iteration 성능은 측정하지 않았고,
기존 value-loss/std 이상이 해결됐다는 결과도 아니다.

## 48. 2026-10-23 — Patch 버퍼 확대·관측 ray 축소·관측 OBB 시각화

학습 중 첨부된 Patch overflow는 기본163,840보다 큰184,347–184,896개를 요구했다.
사용자가 학습 중 실행·수정을 금지했을 때는 설정·공식 문서의 읽기만 수행했다. 이후 학습을
종료했다고 알리고 수정을 승인한 뒤 task의 PhysxCfg에
gpu_max_rigid_patch_count=2**18(262,144)을 적용했다. Contact count·다른 buffer·solver 반복·
물리/정책100Hz는 유지한다. 이 용량은 첨부 피크보다 약41.8% 여유가 있지만 이후 다른 장면의
모든 접촉 피크를 보장하지 않으며 overflow가 난 기존 학습 데이터를 복구하지 않는다.

현재 사용하던 observation ray는160×120이었다. 실제 observation 생성 경로를
64×48=3,072rays로 줄여19,200개 대비84%(6.25배) 감소시켰다. 새 설정은
initial_vision.observation_ray_width/observation_ray_height/observation_min_depth_pixels이며
기본64/48/3이다. 표면점16개 기준을 면적비에 맞춰3개로 낮췄다. 실제 보이는 first-hit점만으로
OBB를 계산하며, 작은 물체가 충분히 관측되지 않으면 invalid로 남긴다. 관측10slot/244D는 같다.

배치 가시성 인증은160×120과 silhouette32/depth16 조건을 유지한다. 새 관측 전용3개 field만
catalog fingerprint에서 제외해 기존159MB·90-pair 캐시를 그대로 사용한다. 실제 정책 관측을
줄이기 위해 이 인증 전체를 재계산하지 않는다. Low-resolution grid/intrinsics는 센서 초기화
때 만들고 재사용하며 별도의 mesh나 카메라를 생성하지 않는다. 기본 clutter reset은 정해진
시점의 observation-only query1회이고 부분 reset은 재촬영하지 않는다.

GUI에는 기존 Target/Blocker 표시와 별도의 Observed objects OBB 체크박스를 추가했다.
청록색 선은 state.geometry_b/geometry_valid의 유효한 초기 정책 관측 OBB만 world로 변환해
그린다. 물체가 움직여도 초기 관측 snapshot을 유지하며 보이지 않은 물체의 실제 bounds로
채우지 않는다. 실제 Target의 초록/Blocker의 빨강과 추정 관측을 비교할 수 있다.
두 레이어는 독립 toggle이지만 하나의 DebugDraw callback을 공유해 clear_lines 간섭을 막는다.
모두 OFF일 때 callback을 제거한다. USD prim·rigid body·collision shape를 추가하지 않으며,
headless startup은 두 visualizer를 모두 만들지 않는다.

**실제4환경 통합 검사:** seed42,15–25개/80-pool에서262,144 설정과 observation64×48,
certification160×120을 확인했다. 기존b29c... cache의 정확한 source hash를 확인하고 cold build를
금지한 상태에서 읽기 전용 warm load를8.73초에 마쳤다. Runtime trace는1회/[1,3072]였고
debug depth[48,64,1]·role mask[48,64,2]·intrinsics fx=fy48,cx32,cy24를 확인했다.
정책[4,244]·기하[4,10,10], 각 환경10개 slot 모두 유효했으며 역할 두 개도 포함됐다.
2개 native step은 모두 유한했고 부분 reset에서 추가 관측/query는 없었다.

실제 native DebugDraw에 관측480선+역할96선을 함께 그리고 독립 toggle을 통과했다.
관측 꼭짓점의 독립 SciPy 변환과 최대 오차는1.46e−7m였다. 시각화 전후 USD prim996개,
rigid API488개·collision API77개의 path/hash와 물리 상태는 그대로였다. Headless 기본은
visualizer가 없었고 직접 draw 경로를 검사한 뒤 callback/선을 정리하고 정상 종료했다.
육안으로 GUI viewport를 확인한 검사는 아니다.

Scene 초기화17.71초·첫 reset20.16초(8.73초 warm load 포함)·부분 reset0.743초는 진단 실행
시간이며 PPO iteration 속도가 아니다. 이번에1,024환경 장기 학습이나 overflow 재발률은
측정하지 않았다. 검증 자료는 outputs/contact_manipulation/ray_obb_20261023/의
native_check.py/json/log다. CPU 전체 회귀 결과는 아래에 기록한다.

CPU 전체348개 검사에서327개 통과·CUDA 전용21개 skip이었다. 표적48개 중46개 통과·
CUDA2개 skip으로 low-resolution 실제 Warp query/최소점 수·인증 cache identity와 관측 OBB
좌표·독립 toggle·headless 무할당을 확인했다. 문법 검사와 git diff --check도 통과했다.
Production 파일 수는 늘리지 않았으며 학습은 시작하지 않았다.

## 49. 2026-10-23 — 1,024환경 collection 세부 시간 실측

사용자 요청으로 현재 코드를 수정하지 않고 별도 출력용 timing hook을 적용해 native
train.py / OnPolicyRunner / PPO 경로를 실행했다. RTX 3090, seed 42, headless,
물리·정책 100 Hz, rollout 36 step, 물체 pool 80개, 활성 물체 17개→19개,
Patch 262,144, 관측 ray 64×48 설정이다. 새 정책으로 총 32 iteration을 실행했고,
초기 10회 warmup, 단계 hook 없는 baseline 5회, 비동기 실행을 유지한 wall 측정 14회,
단계 경계에 CUDA 동기화를 추가한 진단 3회로 나눴다. 초기 환경 생성·컴파일은 아래 통계에서
제외했다. 800 policy step의 공유 장면 교체 1회를 포함한다.

**일반 collection 실측:** 장면 교체 iteration 22를 제외한 wall 13회의 평균은
3.460415초, 중앙값은 3.351608초다. 아래는 부모·자식 시간을 중복 합산하지 않은 분할이다.
‘일반’에도 조기 종료 환경의 부분 reset은 포함된다.

| 단계 | 평균 초 / iteration | Collection 비율 |
| --- | ---: | ---: |
| 물리 step 호출 | 1.922748 | 55.56% |
| 부분 reset 전체 | 0.745665 | 21.55% |
| 종료 판정 및 최초 공통 상태·센서 계산 | 0.257304 | 7.44% |
| Action 처리·목표 생성 | 0.133917 | 3.87% |
| Reward 계산 | 0.104057 | 3.01% |
| Actor·Critic 추론 및 action 분포 처리 | 0.095283 | 2.75% |
| OSC·hand 제어 적용 | 0.093028 | 2.69% |
| Scene 데이터 제출 | 0.045943 | 1.33% |
| Scene buffer 갱신 | 0.017475 | 0.50% |
| 나머지 runner·관측·관리 비용 | 0.044996 | 1.30% |
| 합계 | 3.460415 | 100% |

물리 step의 1.910102초는 PhysX fetch_results 대기였다. 이는 앞서 제출된 GPU 작업의
완료 대기를 일부 포함할 수 있으므로 순수 충돌 kernel 시간이라고 해석하지 않는다.
동기화 진단 3회에서도 collection 평균 3.338692초 중 물리 step이 2.055695초(61.57%)로
가장 컸다. 이 진단은 실행 중첩·대기 위치를 바꾸며 정책·reset 분포도 달라 실제 속도 개선
비교로 사용할 수 없다. 충돌 탐색과 solver 내부 비용은 이번 측정에서 분리하지 않았다.

부분 reset은 일반 13회 동안 총 466 batch / 3,562환경 행에서 발생했다. 평균 35.85번 /
iteration, batch당 7.64환경, batch당 20.80ms였다. Reset 내부 평균은 로봇 시작 자세 준비
0.333231초(그 안의 _sample_robot_starts 0.221823초), 상태 적용 0.152026초,
MDP state 초기화 0.088482초, snapshot 읽기 0.042840초 등이었다. CUDA graph replay 중
IK·충돌 연산은 내부 Python 함수 hook을 다시 통과하지 않으므로 내부 ik.solve 호출 시간만
합쳐서 전체 IK 비용이라고 해석하면 안 된다. 상위 robot-start 범위가 해당 replay를 포함한다.

공통 상태 계산은 termination.success가 최초 호출자여서 그 하위에 기록된다. 이 값을 성공
predicate 자체의 비용으로 보면 안 된다. TaskState.advance 0.119622초와 transition 읽기
0.065374초가 포함되고, 그 transition 중 SensorReader는 0.029059초였다. Actor forward는
0.080378초, native Critic forward는 0.008501초로 rollout.act에 포함된다. OSC apply의
0.090759초에는 controller.compute 0.030616초와 Jacobian·mass·pose·velocity 조회가 포함된다.

**장면 교체:** iteration 22의 collection은 7.066026초, reset 합계는 4.487477초였다.
이는 1,024환경 전체 reset 1번과 부분 reset 20번(95환경 행)의 합계다. 새 template 준비는
1.405342초였으며 clutter 배치 0.261612초, template 로봇 시작 자세 1.127006초,
ray casting·관측 생성 0.004351초를 포함한다. 별도의 전체 환경 로봇 자세 준비 범위는
2.626472초였다. Ray trace 호출은 1회 / 0.001059초, OBB fit은 0.002160초였고 부분
reset에서는 추가 관측 query가 없었다. 이 단일 교체에는 새 graph shape·cache 준비 비용이
섞일 수 있다. 교체 직후 iteration 23·24도 reset 비용이 커 collection 4.659504·3.827495초였고,
25–28회는 3.158373–3.438920초였다.

같은 17물체 장면의 baseline 10–14회 평균은 3.337719초, wall 15–21회는 3.335564초로
큰 계측 지연은 보이지 않았다. 서로 다른 시점의 갱신 중인 정책이므로 정확한 hook 오버헤드를
분리한 대조 실험은 아니다. 전체 wall 일반 평균에는 새 19물체 장면의 일시적인 reset 상승도
포함했다. 평균 learning은 0.552895초로 collection 밖이다. Console·TensorBoard logger 출력은
평균 0.035979초이며 역시 collection 밖이다. Collection 안의 native logging bookkeeping은
0.010353초, NaN 검사는 0.005133초, 최종 observation.compute는 0.004844초였다.

원시 32회 기록·config·중복 없는 집계는
outputs/contact_manipulation/collection_profile_20261023/의 collection_timing.py,
metadata.json, timing.json, summary.json/txt, run_fixed.log에 보존했다. 측정 경로의 부모·자식
합계 검증은 root·nested 모두 mismatch 0건이었다. 실행은 exit 0으로 종료했고 해당 실행의
Error·overflow는 없었다. Production 설정·보상·제어 코드는 수정하지 않았다. 초기 진단 파일의
이름 profile.py가 Python 표준 모듈과 충돌한 startup 실패는 run.log에 별도 보존했고,
파일을 collection_timing.py로 바꾼 정상 실행만 위 통계에 사용했다.

현재 우선 병목은 물리 step 완료 대기와 잦은 부분 reset이다. Ray casting·정책 추론·logging은
이번 실행의 주 병목이 아니다. 단, 새 정책 32 iteration과 두 장면의 측정이며 장기 학습된
checkpoint의 접촉·종료 분포나 90,000 iteration 전체 평균을 대표한다는 결과는 아니다.

## 50. 2026-10-23 — 미사용 물체 비활성화 실험과 원본 충돌·solver 확인

사용자 요청으로 pool·생성 규칙·활성 물체·17 tactile 채널을 유지하면서 미사용 body의
물리 처리를 줄이는 방법을 실제 SDK에서 검사했다. **검사한 방식 중 안전성과 성능을 함께
통과한 방법은 없어 production 변경은 모두 복원했다.** 현재 미사용 물체는 기존처럼 격리된
공중 위치에서 native permanent wrench composer의 COM countergravity로 유지한다.
이 방식이 simulation에서 body를 제외하는 것은 아니다. 이번 실험으로 미사용 body의
비활성화가 완료됐다고 해석하면 안 된다.

**Live disable flags:** 공식 rigid-body tensor view의 `set_disable_simulations`와 `wake_up`을
사용했다. Collection의 object-major native index인 `object_id * num_envs + env_id`를
적용했고 조회한 flag가 의도한 mask와 일치했다. 4환경에서 reset 직후 native pose와 writer
차이는 최대 2.38e-7m였지만, physics 3 step 뒤 미사용 body의 GPU pose readback이 기대한
주차 pose에서 최대 2.82m 달라졌다. 이 결과만으로 실제 actor 이동과 readback/index 문제를
구분하지는 못한다. 따라서 flag가 바뀌었다는 것만으로 성공으로 처리하지 않았다.
별도 8-cube 최소 예제에서 disable 뒤 view를 다시 만드는 경로도 native unresolved rigid
dynamic index와 allocator 오류로 종료됐다. Global simulation view를 재생성하면 robot과
sensor handle까지 영향을 받으므로 task reset 경로에 적용하지 않았다.

**Explicit sleep:** 공식 `IPhysxSimulation.put_to_sleep`과 native `wake_up`도 비교했다.
초기 장면의 전체·부분 reset은 통과했지만 다음 공유 장면 교체에서 새로 주차한 body가
3 step 동안 5.886mm 내려갔다. CPU sleep 조회와 GPU pose 결과가 일치하지 않았고,
한 step 뒤 재요청·wake 후 sleep도 해당 재현을 해결하지 못했다. 엔진 내부 원인은 확정하지
않았으며, CPU sleep 결과만으로 GPU actor가 안전하게 제외됐다고 판단하지 않았다.

**기존 Ground 위 주차:** Live flags 없이 자연스럽게 sleep하도록 기존 Ground의 로봇 도달
범위 밖에 미사용 body를 두고 countergravity 제출을 제거하는 대안도 구현·검사했다.
추가 collider나 환경별 Python loop는 없었으며 reset의 GPU tensor 연산으로 위치만 정했다.
4환경에서 12회 전체·부분 reset과 4개 장면을 통과했고, native reset pose 오차는 최대
1.20e-7m, 미선택 환경 pose는 보존됐다. 주차 body의 최대 접촉 settling은 1.141mm였다.
17 pad × 4환경의 실제 접촉·binary 감지·분리 검사와 실제 cracker-box Blocker를 이용한
3회 필터 역할 교체도 통과했다. 이는 주차·센서 대응의 검사이며 sleep 자체의 증명은 아니다.

**1,024환경 대조:** RTX 3090, 물리·정책 100Hz, pool 80개 중 활성 17개였다.
총 object body는 81,920개이고 주차 body는 64,512개다. 기존 공중 주차+countergravity와
Ground 주차를 3개 paired block에서 교대로 비교했다. 각 phase는 warmup 128 step 뒤
64 step을 측정해 모드별 192 sample이다. 활성 물체·로봇의 초기 상태를 복원하고 질량이
bit-exact하게 같은지 확인했다. 미사용 body의 주차 위치와 접촉 부하는 의도한 차이다.
각 단계 앞뒤 CUDA 동기화로 분리한 microbenchmark이며 PPO·reset·ray 시간은 포함하지 않는다.

| 평균 / physics step | 기존 공중 주차 | Ground 주차 | 변화 |
| --- | ---: | ---: | ---: |
| Scene write | 2.03480ms | 1.30047ms | −36.09% |
| Physics step | 34.66214ms | 36.78275ms | +6.12% |
| Action·write·physics·update 합계 | 42.47722ms | 43.87175ms | +3.28% |

3개 paired block 모두 합계가 +4.19%, +3.86%, +1.83%로 악화됐다. Force 제출 감소보다
physics 증가가 컸으므로 **Ground 주차는 채택하지 않고 복원했다.** Ground 주차 최대 변위는
1.148mm이고 모든 phase에서 state는 finite·질량은 유지됐다. 두 모드의 CPU sleep 조회는
모두 0이었으나 이 API로 GPU sleep 여부를 확정하지 않는다. 측정 로그에 CUDA/kernel/
Patch overflow 오류는 없었다. 짧은 제어 microbenchmark이므로 장기 안정성이나 전체 PPO
iteration 성능을 입증하는 검사는 아니다.

**Mustard 원본:** 기존 원본 USD 조사 자료를 재확인했다. `Axis_Aligned_Physics/006_mustard_bottle.usd`
자체가 `convexDecomposition`과 `maxConvexHulls=32`를 author하며 실제 cooking 결과도
32 hull·총 640 hull vertex다. 원본 mesh는 8,377 vertex·16,382 triangle이며 32개 mesh라는
뜻은 아니다. 현재 production에서 mustard hull 수를 늘리는 override는 없다. 과거 1 hull
비교는 별도 진단이었다. 근거는 object_collision_audit_20261014/assets.json의 authored와
cooked 레코드다.

**예제 solver:** examples/Sweeping-Policy-DRL-Sim-Train의 robot cfg는 articulation solver
값을 override하지 않고 외부 USD를 유지한다. 로컬 예제용
/home/hoon/Downloads/Shelf_USD/Robots/UR5e/Collected_UR5e_v4/UR5e_v4.usd를 CPU의
pxr.Usd로 직접 읽으면 /ur5e/root_joint의 authored Position 32 / Velocity 1이다.
SHA256은 1504fbe97e743cc697e0ffc3e2103beaf2e8a5035e1d7e94d879211b7e8e3614다.
예제에 하드코딩된 /home/ssu/... 파일은 없으므로 모든 외부 사본의 동일성을 주장하지 않는다.
예제의 물체 cfg는 명시적으로 16 / 1이다. 로컬 캐시 origin/min 57caa46도 같은 방식이다.

| 대상 | 예제 코드 + 확인한 로컬 USD | 현재 task |
| --- | ---: | ---: |
| 로봇 Position / Velocity | 32 / 1 | 32 / 2 |
| 물체 Position / Velocity | 16 / 1 | 16 / 2 |

따라서 로봇 Position 32 자체는 로컬 예제용 USD와 같고, 확인된 차이는 Velocity 1→2다.
이번 작업에서 solver·robot·active object의 충돌 형상이나 제어 설정은 변경하지 않았다.
비활성화 문제는 미해결이며, 현재 SDK에서 pool handle·GPU pose·접촉 센서 대응을 보존하는
방법이 확인되기 전까지 live flags나 sleep 성공을 가정한 코드를 넣지 않는다.

실험 자료는 outputs/contact_manipulation/inactive_pool_20261023/에 보존했다.
native_disable_failure.json·native_sleep_repeat_failure.json은 실패 근거,
native_check.json/native_final.log는 Ground 대안 통합 검사,
benchmark.json/benchmark_verified.log와 benchmark_review_summary.json은 최종 대조,
example_solver_audit.json은 원본 solver 확인이다. 초기 harness 실패 로그는 정상 대조와
구분해 별도 파일명으로 보존했다. Production 복원 뒤 관련 CPU 28개 검사를 모두 통과했고
자체 GPU 진단 프로세스는 종료됐다. 새 production 파일은 추가하지 않았다.

## 51. 2026-10-23 — 예제와 solver 일치, simulation 활성화 옵션 구분

사용자 요청으로 scene.py의 로봇 articulation solver를 Position32/Velocity2에서32/1로,
물체 rigid-body solver를16/2에서16/1로 변경했다. 앞 절에서 확인한 로컬 예제용 robot USD와
examples의 물체 cfg에 맞춘 값이다. Position 반복 수와 물리·정책100Hz는 유지한다.
Mustard는 요청대로 원본32 hull을 유지하며 충돌 mesh를 수정하지 않았다. README의 현재
설정도 갱신했고 과거 velocity2 측정 결과를 새 설정에서 측정한 것처럼 읽지 않도록 구분했다.

추가 질문의 kinematic 비활성화는 이미 적용된 상태다. 현재 물체의 kinematic_enabled=False는
힘·중력으로 움직이는 dynamic body를 뜻하며 simulation 제외가 아니다. 미사용 body만 True로
바꾸면 외부 pose로 움직이는 kinematic body가 되지만 collider와 동적 물체와의 충돌은 남는다.
설치된 RigidBodyView의 set_kinematic_targets는 이미 kinematic인 body의 목표를 바꾸는 API이며
dynamic/kinematic 타입을 바꾸지 않는다. 설치된 tensor API에는 타입 전환용 일괄 setter가 없고
Isaac Lab의 modify_rigid_body_properties는 USD RigidBodyAPI 속성을 수정한다. 일반 Kit의 공식
테스트에는 실행 중 USD kinematic→dynamic 전환 사례가 있으므로, DirectGPU에서 전환 자체가
불가능하다고 단정하지 않는다. 현재 collection의 반복 reset 호환성과 비용은 아직 검증하지
않았으며 이번에는 kinematic 전환 코드를 넣지 않았다.

물체별 PhysX simulation ON/OFF API는 별도로 존재한다. 설치된 공식 tensor RigidBodyView의
set_disable_simulations(data, indices)는1이면 simulation 비활성,0이면 활성이다. 재활성화 뒤
wake_up(indices)을 호출해야 sleep 상태 때문에 반응하지 않는 상황을 피할 수 있다고 공식
docstring에 명시돼 있다. 이는 앞 절에서 실제 사용한 API다. 기능이 없어서 미채택한 것이
아니라 GPU pose readback/반복 reset 검사가 실패해 production에 적용하지 않은 것이다.
이전 실패를 해결한 결과나 kinematic 전환 실패로 혼동하지 않는다. 현재 미사용 body의
simulation은 계속 활성 상태이며 기존 격리·countergravity 방식을 사용한다.

검증은 기존 HandUrdfBoundsTests·NativeControllerTests·ContactBatchUpdateTests·EpisodeResetTests·
NativeArmEffortLimitTests의 CPU36개 중30개 통과·CUDA 전용6개 skip이다. AST로 두 설정의
32/1·16/1 적용과 문법을 확인했고 git diff --check도 통과했다. 새 GPU 실행·성능 비교·
velocity1 설정의 native hand 동작 재측정은 하지 않았으므로 속도 개선이나 손 제약 정확도
유지를 실측한 결과로 설명하지 않는다. Production 파일 개수는 늘리지 않았다.

## 52. 2026-10-23 — 사용자 실행 확인을 위한 simulation ON/OFF 적용

사용자가 이전 비활성화 검사 결과를 설명받은 뒤 해당 API를 적용하고 직접 실행해 확인하도록
요청했다. 앞 절의 미채택 상태에서 변경해 현재 production은 공식
RigidBodyView.set_disable_simulations와 wake_up을 사용한다. 이전 GPU pose 조회 차이가
해결됐음을 확인한 변경은 아니며 이번 작업에서는 GPU simulation·학습을 실행하지 않았다.

TaskObjectPool.set_simulation_enabled는 공식 CPU property API의 전체 view 크기 uint8 flag
buffer와 object-major 인덱스를 캐시한다. 인덱스는 object_id*num_instances+env_id이며
1=simulation OFF·0=ON이다. Native view가 재생성되면 getter로 flag cache를 다시 초기화한다.
부분 reset은 선택된 환경의 인덱스만 setter에 전달하고 다른 환경의 flag는 유지한다.
Per-environment/body Python loop·매 step API 호출·native view 재생성은 추가하지 않았다.

_write_physical_reset은 대상 환경의 pool을 먼저 활성화(wake하지 않음)하고 native writer로
pose·velocity를 초기화한 뒤, unused flag를1로 설정하고 active body만 wake_up한다.
대상 ID와 active mask의 CPU 전송은 해당 reset에서 한 번씩 수행해 두 native 호출에서
재사용한다. 실제 state·기하·IK 처리는 기존 GPU tensor 경로다. Spawn 직후 parked_mask는
모두False이며 첫 reset에서 선택한 물체 구성에 따라 비활성 mask를 확정한다.

후보 준비 중 실패하는 rollback도 enable→native state write→unused OFF→active wake 순서를
사용한다. 비활성 물체의 GPU pose readback이 불확실했던 이전 결과를 고려해 rollback용
snapshot에서는 해당 행을 authored parking state+env origin으로 교체한다. 이는 미사용
물체가 기본 주차 상태에서 비활성화되는 현재 규칙을 사용한 것이며 활성 물체의 native 조회나
매 step 전체 상태를 덮어쓰지 않는다. 활성 body의 GPU 인덱스 문제가 해결됐다는 의미는 아니다.

기존 refresh_parking·gravity·countergravity buffer와 force 설정을 제거했다. Native collection의
reset/update/write_data_to_sim은 그대로 사용하며 주차용 permanent wrench composer를
활성화하지 않으므로 매 physics step parking force 제출이 없다. Pool 크기·물체 생성 규칙·
17 tactile 채널·Blocker 필터·관측 계약·solver32/1 및16/1·Mustard32 hull은 유지한다.

관련 CPU29개(API mock2개+episode reset27개) 검사를 통과했다. Unsorted 환경 선택의 native
인덱스·CPU uint8 전체 flag 크기·미선택 환경 보존·active-only wake·enable/write 순서·빈 reset·
cache 재바인딩·inactive NaN readback의 rollback 제외를 확인했다. 검사 중 bool tensor를
uint8 destination에 대입하는 오류를 발견해 명시적 uint8 변환으로 수정했다. 문법·diff 검사도
수행했다. 이 결과는 native PhysX에서 재활성화·접촉·장기 학습이 통과했다는 결과가 아니다.
README와 SPAWN_RULES의 현재 물체 처리 설명을 갱신했고 production 파일은 추가하지 않았다.

## 53. 2026-10-23 — PhysX GPU 단계 세분화와 최적화 후보 대조

사용자 요청으로 미사용 body ON/OFF 이외의 물리 병목을 조사했다. 학습용 코드·물리 설정을
변경하지 않고 outputs/contact_manipulation/physics_step_profile_20261023/의 별도 harness로
native train.py/OnPolicyRunner 경로와 고정 제어 microbenchmark를 실행했다. RTX3090,
1,024환경, headless, 물리·정책100Hz, robot32/1·objects16/1, 활성17개/pool80개다.
진단용 짧은 학습은 각10 iteration(warmup4/baseline2/wall4), 마지막2회만 Nsight Systems
2025.6.3의 CUDA/NVTX와 graph-node trace를 수집했다. 실제 학습 로그는 outputs 아래 별도
디렉터리에 두었고 기존 logs/rsl_rl의 학습을 재개하지 않았다. 모든 자체 실행은 종료됐다.

**측정 방법:** SimulationContext.step→native simulate/fetch_results의 CPU wall을 분리했고,
공식 PhysX benchmarks callback(zone_name/ms)과 scene 통계도 저장했다. Callback의 첫
sample에는 과거 초기화 누적 기록이 섞여 제외했다. 이후 native callback은 PhysX Update와
Physics Update Transforms만 제공해 broadphase/solver의 내부 GPU 시간은 알 수 없었다.
Nsight SQLite에서 실제 physics.step 구간의 GPU kernel과 copy를 분리했다. Native worker
thread가 제출한 kernel도 포함하고 PPO·reset·ray·MDP 구간은 제외했다. Kernel 이름별
분류표·원시 횟수·stream·launch geometry를 보존했으며 GPU 시간 합과 wall 시간을 구분했다.

**움직이는 정책에서 추가 상태 이상 발견:** 동일 seed의 짧은 실행에서도 극단적으로 큰 유한
손 관절값이 나타났다. 원본 TGS+disable control은 URDF limit 오차 최대48,079.9rad,
mimic52,109.8rad, PGS는14,938.5/16,173.1rad였다. 새 flag를 전혀 쓰지 않고 기존 native
countergravity로 유지한 대조에서도1,798,516.6/1,949,901.6rad가 관측됐다. 이 수치는 각
iteration 끝의1,024환경 표본 최대이며 모든 step의 최대가 아니다. 마지막 partial reset 뒤
작아진 값만 보면 앞선 이상을 놓칠 수 있다. finite=True가 물리적으로 정상이라는 뜻은 아니다.
실제 물리 상태·수치 불안정·readback 중 원인은 분리하지 못했으며 PGS 또는 새 flag 하나의
원인으로 단정하지 않는다. 최초 current/pads 실행에는 이 정확도 계측이 없었다.

이 때문에 정상 자세 유지 대조를 별도로 수행했다. Native OSC pose 증분0·kp100과 초기
hand 위치 유지만 사용하고 env.reset1회 후 PPO·reward·추가 reset 없이128 warmup+
72 capture step을 실행했다. 관절 범위·mimic 오차를 매 step GPU에서 집계했다. 측정72개
전체의 URDF limit 오차 최대0.000368591rad, mimic0.000434699rad였고 q/qd는 유한했다.
원본 mass는 유지됐으며 비활성 body64,512개가 유지됐다. 정상 자세 유지 guard를 통과한
결과이며17개 pad의 실제 접촉 fixture 검사나 움직이는 정책의 안정성 검증은 아니다.

**정상 자세 유지의 GPU 세분화:** 아래 비율은 GPU kernel 시간 합22.8297ms 대비다.
이는 실행 시간이 겹치는 kernel의 합이며 physics wall과 같은 값이 아니다.

| GPU kernel 분류 | ms / physics step | Kernel 합 비율 |
| --- | ---: | ---: |
| Articulation 내부 constraint | 8.1081 | 35.52% |
| Tendon·mimic constraint | 3.7401 | 16.38% |
| Contact·joint constraint solver | 3.5325 | 15.47% |
| Articulation dynamics·integration | 3.1874 | 13.96% |
| Solver propagation·bookkeeping | 2.4864 | 10.89% |
| Broadphase·pair·sorting | 0.7861 | 3.44% |
| Narrowphase | 0.6939 | 3.04% |
| Rigid integration·transform 갱신 | 0.2790 | 1.22% |
| Contact report | 0.0161 | 0.07% |

Native physics wall 평균은26.1204ms/step(simulate0.2037·fetch25.6769ms)이었다. NVTX 자체
구간으로는26.1242ms였으며 GPU kernel union22.4539ms, copy/memset 포함 GPU activity
union23.4044ms다. Wall과 GPU 점유 차이를 전부 CPU 계산으로 해석하지 않는다. 이 고정 제어
값을36 step에 곱하면 physics만0.9403초이나, reset·센서·정책·PPO를 포함한 iteration 예상이
아니다. 움직이는 정책의 uncaptured wall은44.20–45.17ms/physics step이었고 상태 이상·명령·
접촉 분포·추가 조회가 함께 달라26ms와의 차이를 상태 이상 한 원인의 비용으로 볼 수 없다.

정상 유지에서도 주요 내부 constraint/mimic kernel은 각각33회/step, articulation integration은
32회/step이었다. Position32/Velocity1의 반복 예산과 일치한다. 주요3개 kernel은32 blocks×
32 threads이며 registers/thread233·255·140이다. RTX3090의82SM에 대해 개별 launch의
병렬 실행 폭이 제한될 가능성은 있지만 achieved occupancy·SM utilization을 측정한 것은
아니다. 이 PhysX 내부 반복은 task의 Python for를 tensor로 바꿔 실행 횟수가 줄어드는 부분이
아니다. 내부 constraint와 mimic만51.90%, broadphase+narrowphase는6.48%였다.

**후보 대조:** hand contact report를 실제32→17개 pad만 유지하도록 진단 spawner를 수정해
비교했다. 모든 collision/joint/mass·17 pad report·UR5e는 유지했다. Uncaptured wall의
physics는1.59105→1.58665초/36step(−0.28%), collection3.21991→3.24593초로 뚜렷한
이득이 없었다. Captured GPU copy/report kernel도 큰 차이가 없었다. 같은 seed지만 reset
분포가 달라 작은 차이를 회귀나 개선으로 확정하지 않으며 production에는 적용하지 않았다.
공식 PGS32/1도 비교했지만 공통 상태 이상의 영향을 배제하지 못해 정상 동작의 속도 개선으로
채택하지 않았다. Solver는TGS를 유지한다. Env ID bounds encoding은 실제 기본4로 이미
사용 중임을 확인했고 broadphase 비중도 작아 추가 설정 대조를 늘리지 않았다.

**남은 구체적 후보:**17 tactile fixed joint를 공식 importer의 dont_collapse로 보존하고 나머지
fixed joint만 merge하는 방법은 가능하다. 다만 현재19개 fixed joint 중17개는 pad이고 나머지
2개만 palm housing이므로 hand32→30, 전체 robot41→39 수준이다. Housing collision/filter
경로와 합산 질량·COM·관성·F/T 보상·17 pad 검증이 필요하며 큰 향상을 약속할 근거는 없다.
Position 반복 수 축소는 가장 큰 비용에 직접 영향을 주지만 이전 손 제약 정확도 저하가 있어
단순 적용하지 않았다. Native articulation 내부 제약·mimic 비용과 움직이는 정책의 상태
이상부터 검토하는 것이 우선이며, Mustard32 hull·UR5e·주기·센서 계약은 유지한다.

한 step당8MiB H→D copy도 관측됐으나 어떤 buffer인지 크기만으로 특정할 수 없다.
GPU contact/patch 용량 축소가 이 copy를 줄인다는 근거가 없으므로 이전 overflow 대응 용량은
유지했다. F/T getter는 조회·변환·동기화 비용이 있지만 지속적인 solver 모드를 켠다는 근거는
찾지 못했다. 정상 유지와 정책 trace의 차이를 F/T 하나에 귀속하지 않는다.

원시 trace·SQLite·재현 helper와 분석은 해당 outputs 경로에 있다. 핵심 자료는
hold/status.json·timing.json·gpu_breakdown.json/txt(정상 유지), gpu_breakdown.json/txt(정책),
pads/comparison_vs_current.json·gpu_comparison.json, final_experiment_summary.json이다.
Native profile과 Nsight를 켠 collection은 current기준3.220→3.929초로 오버헤드가 있으므로
uncaptured 구간과 구분한다. 이번 작업에서 production 코드/학습 설정 변경은 없고,
문법 검사·분석 산술 검토·git diff --check를 수행했다. 추가 GPU 실험은 종료했다.


## 54. 2026-10-23 — 장면 교체 시 선반 통과 회귀와 수정

**요청:** 장면 변경 때 일부 물체가 선반을 통과하는 현상을 PhysX 활성화, 스폰 자세,
물리 주기로 분리해 수정한다. 재현/원시 자료는
`outputs/contact_manipulation/reset_support_20261023/`에 보존했다.

### 확인한 두 문제와 적용한 수정

1. **Live simulation OFF/ON 뒤 GPU actor 대응이 깨졌다.** 로봇 없는8개 Cube 대조에서도
   재활성화 후 다음 step에 XY 오차2.401m가 생겼다. 실제16환경에서는 두 번째 장면에
   활성 물체 좌표 오차8.9901m, 다음 장면에 CUDA700이 발생했다. 설치 버전의 native
   `set_disable_simulations`는 actor flag를 바꾸며 tensor view가 캐시한 GPU 인덱스를
   갱신하는 경로가 확인되지 않았다. 이 기록만으로 과거 모든 CUDA719의 원인을
   같은 문제로 확정하지 않는다.
2. **OFF/ON을 제거해도1,024환경의 GPU 충돌 누락이 남았다.** 기본 환경 ID bounds4bit에서
   세 번째 장면의4개 물체가 `env_id % 16 != 0`인960환경에서 자유 낙하했다.
   각 물체의 독립 native view와 collection view는 prim path로 대응했을 때 오차0이었다.
   Shelf의 GPU 위치 오차는0.1mm 미만이고 해당 위치의 native scene query도 윗면Z=1.05의
   Cube collider를 확인했다. 단순 OBB 표시나 잘못 기록된 루트 좌표로 설명되지 않았다.

**Production 변경:** `TaskObjectPool`의 live OFF/ON과 CPU flag/index 제출을 제거한다.
Native actor/view와 simulation·gravity flag를 유지하고, unused를 격리된 주차 위치에 놓아
공식 `permanent_wrench_composer`가 COM 중력을 상쇄한다. Active의 외력은0이다.
Reset·rollback 모두 선택 행에 적용한다. 고정 mass/gravity를 한 번 캐시하며 native collection이
매 physics step에 힘을 제출한다. 별도 Python step override는 없다. 모든 body가 simulation에
남으므로 미사용 물체의 물리 비용 제거 효과는 포기한 정확도 수정이다.

충돌 누락은 **환경 ID bounds encoding8bit + 환경별 유한 static 바닥**을 함께 적용해 회피한다.
바닥은 공식 `CuboidCfg`로2.49×2.49×0.10m, 윗면Z=0, `collision_group=0`, 기본 native 물리
material을 사용한다. 전역 무한 Ground는 제거한다. Physics attach 전 floor spawner에서
USD session layer에8을 기록해 native cloner가 root layer에 쓰는4보다 우선시한다.
바닥 생성/복제는 native spawner·cloner가 담당하며 reset 중 설정·actor·shape를 재생성하지 않는다.
설정은 매 시작에 적용되며 root layer만 별도로 export하면 session opinion은 포함되지 않는다.

8bit만 켜고 전역 바닥을 유지하면 `env_id % 256`이0·1·254·255인16환경에서 선반 자체가
자유 낙하했다. 공식107.3 GPU broadphase의 global-shape encoded bounds 상수
`[0x01800000, 0xfe7fffff]`와8bit 환경 prefix 경계가 이를 설명한다. 따라서 바닥 교체를
분리하거나 전역 collider를 다시 추가해서는 안 된다. 4bit 물체 누락의 정확한 내부 오류 행은
미확정이며, 비교 카운터 overflow 같은 추정을 확정 원인으로 기록하지 않는다.

### 주요 대조 결과

아래 장면은 seed42의 동일 active ID·배치 sequence이며, 별도 표기가 없으면1,024환경이다.
낙하는 활성 물체의 OBB 최저점이 초기 지지면보다3cm 이상 아래에 있는 경우로 집계했다.

| 구성 | 결과 |
| --- | --- |
| OFF/ON 제거·기존4bit·전역 바닥 | scene2에서4개×960환경 낙하 |
| 초기2mm 여유 또는200Hz | 같은 scene2 낙하 유지 |
| Parking 외력 없음 / Blocker 필터 없음 / shared geometry 없음 | 같은 낙하 유지 |
| Active wake / contact offset 재기록 / 후보 env0 물체 임시 write 생략 | 같은 낙하 유지 |
| 공식 ManagerBased `env.step()` | scene2 step10에서 같은3,840개 낙하; 이후 자동 reset이 복구 |
| 8bit + 전역 바닥 | ID 경계16환경에서 선반과 모든 활성 물체 낙하 |
| 4bit + 환경별 바닥 | scene2의3,840개 낙하 유지 |
| 8bit + 환경별 바닥 | 같은3장면에서 낙하0 |

일반 CCD는 이 GPU backend가 지원하지 않는다는 native 경고를 내며 무시했다.
Speculative CCD와 reset CUDA 동기화도 문제를 해결하지 못했다. Physics replication을
완전히 끈 별도 대조는 초기화 메모리 한계 또는 다른 초기 낙하가 있어 채택하지 않았다.
따라서 스폰 높이·upright·100Hz 물리/정책·solver32/1 및16/1·UR5e·Mustard32 hull은 유지한다.

### 실제 production 구성의 회귀 검사

- `production_1024/support.json`: seed42,1,024환경,8장면×80 physics step, 각 장면 뒤512행
  부분 reset. 활성 개수17/19/19/23/23/22/19/15, 모든 표본의 낙하0.
  전체 표본의 최저 지지면 차이−1.009mm, 최종 각 장면은 약−0.30~−0.34mm였다.
- Reset 직후 활성 좌표 오차 최대2.384e−7m, Shelf 위치 오차 최대0.0706mm.
  독립 native view5개의 prim-path 대응 오차0. 부분 reset의 다른512행 좌표는 전부 bit-exact,
  선택 행 오차 최대2.384e−7m. Disabled body0, native tactile body17개 유지.
- `production_native_1024_seed43/support.json`: 다른 seed43, 공식 ManagerBased `env.step()`의
  센서·보상·종료·자동 reset을 포함한4장면×80step과 부분 reset에서 표본 낙하0.
  이것은 고정 자세 유지 검증이며 움직이는 학습 정책의 장기 안정성 검증은 아니다.
- `production_native_4_seed44/support.json`:4환경·다른 seed44의3장면×40step도 낙하0,
  자동 종료/reset0. 각 장면 뒤2행 부분 reset의 미선택 행도 bit-exact였다.
- CPU reset·부분 reset·parking force 검사34개 통과/CUDA1개 skip. 실제 USD root/session
  합성과 native delegation을 포함한 geometry 검사10개 통과. 문법·`git diff --check` 통과.

`production_summary.json`에 원시 결과의 최대 오차와 집계를 저장했다. OFF/ON이 이미 실행된
프로세스의 GPU view를 hot patch로 복구했다고 볼 수 없으므로 새 프로세스로 실행해야 한다.


## 55. 2026-10-23 — 스폰·학습 변경 없는 미사용 물체 연산 제외 대안 조사

**후속 정정:** 56절의 실제 접촉 검사에서 shared geometry prototype의 collider 속성 변경이
native instance shape에는 반영되지 않는 것을 확인했다. 아래 조사는 당시 의도한 flag 조합을
기록한 것이며, 실제 충돌 제외나 그 성능 효과를 검증한 결과가 아니다. 최종 적용은 collider를
변경하지 않는 kinematic 전환뿐이다. 아래 약15% 물리 시간 차이를 collider 제외 효과로 해석하지 않는다.

사용자는54절의 바닥/encoding 변경으로 우회하기보다 기존 스폰·학습 구성을 유지하고
미사용 물체 연산을 제외할 다른 방법을 요청했다. 이번에는 **production 코드를 수정하지 않고**
output-only 진단에서 원래 전역 Ground·native4bit·동일 배치/로봇/100Hz/센서를 사용했다.
자료: `outputs/contact_manipulation/inactive_alternatives_20261023/`.

### 확인한 대안

미사용 body는 `physics:kinematicEnabled=True`, 원래 활성 collider만
`physics:collisionEnabled=False`로 전환한다. 다시 쓰면 같은 속성을 복원한 다음 native
pose·velocity를 기록한다. `set_disable_simulations`와 달리 body를 scene에서 제거하지 않는다.
공식107.3 `NpRigidDynamic::getGPUIndex`는 island node index를 반환하고,
`ScBodySim::switchToKinematic/switchToDynamic`은 같은 node를 유지하며 타입/group만 바꾼다.
Kinematic body는 transform 관리는 남지만 solve/integrate를 하지 않는다고 native 코드가
명시한다. Collider toggle도 기존 shape의 simulation/query flag와 filtering을 바꾸며 body를
삭제하지 않는다. Geometry/actor/GPU buffer 메모리는 남고 모든 관리 비용이0이 되는 것은 아니다.

공식 tensor API에는 runtime kinematic 타입 전환용 일괄 setter가 없다. 설치 버전의
GPU `set_kinematic_targets`는 not implemented이므로 사용하지 않았다. 공식 USD 속성 API를
사용한다. 현재 모든 환경이 같은 물체 구성을 공유하므로 env0 body 속성과 **프로세스 전용**
shared geometry library의 collider 속성을 바꿔 USD inherit/instance로 전파했다. 원본 Nucleus
asset·메시·mass/friction·구획·개수·배치 규칙은 수정하지 않았다. Deinstance나 파일 저장도 없다.

`RigidPrim.disable_rigid_body_physics`는 기존 simulation OFF API의 wrapper라 대안이 아니다.
중력 해제는 동역학·충돌 연산을 제외하지 않는다. Sleep는 공식 API가 있지만50절의 GPU
주차 유지 회귀를 통과하지 못했다. 최신SDK의 sleeping 관련 제약을 설치107.3과 동일하다고
단정하지 않는다.

### 정확성 검사

- 8-Cube 최소 예제의 kinematic+collider toggle8회×60step: unused는 주차 좌표 오차0,
  다시 dynamic으로 바뀐 Cube는Z=.8에서 중력으로 내려와 기존 테이블 윗면Z=.5에 지지됐다.
  Native 통계도 dynamic5/6개·kinematic3/2개로 실제 전환했다.
- 실제4환경4장면,1,024환경6장면×80step, 각 장면 뒤 절반 행 부분 reset을 검사했다.
  **원래 전역 바닥·4bit에서 표본 낙하0**. 이전4bit 문제를 재현했던 세 번째 배치도 포함한다.
- 1,024환경의 unused 주차 오차 최대2.384e−7m, 독립 native view prim-path 대응 오차0,
  미선택512행 pose bit-exact, 선택 행 reset 오차2.384e−7m, simulation disable flag0.
  Native kinematic 수64,512/62,464/62,464/58,368/58,368/59,392로각scene unused 수와일치했다.
- 17개 tactile body 수는 유지됐다. 이번에는 실제17pad 접촉/Blocker filter fixture까지 반복한
  검사는 아니며, 장기 학습 안정성 검사도 아니다. 최소 예제2개는 결과 저장 후 app.close가
  대기해 해당 진단 프로세스를 정리했다. 실제 환경 진단은 정상 종료했다.

### 단기 성능 대조

1,024환경·같은 첫 scene·자세 유지 제어에서 countergravity와 새 방식을3쌍으로 교대했다.
Phase마다64 warmup+64측정step, action/write/physics/update 경계에 CUDA barrier를 넣었다.
PPO·ray·배치 생성은 포함하지 않은 직렬화된 물리 microbenchmark다.

중간 한 쌍의 대조군에서 unused가12.56cm 이동해 기존5mm 주차 기준을 넘었다.
그 쌍 전체(새 방식 포함)를 비교에서 제외하고 원시 기록은 남겼다. 정상2쌍 결과:

| 평균 / physics step | Countergravity | Kinematic + collider OFF |
| --- | ---: | ---: |
| Physics |31.128ms|26.342ms|
| Action·write·physics·update 합|38.867ms|34.290ms|

Physics−15.38%, 합−11.78%이지만 작은 정상 표본2쌍의 결과이며 전체 PPO iteration의
개선율로 볼 수 없다. Mass는모든phase bit-exact·상태는finite였다. 전체 unused를 한 번
전환하고 상태를 복구하는 준비 구간은0.88–0.93초로, USD 변경의 CPU 비용도 있었다.
같은 mode로 다시 준비한 구간은약9ms였으며 이는 episode 전체 reset 시간이 아니다.

### Production으로 옮길 때 남은 작업

- 공통 active mask가 바뀌는 전체 장면 교체에서만 USD 속성을 바꾼다. 부분 reset은 mask가
  같아야 하며 변경/`flush_changes()`를 생략해야 한다. 환경별 별도 mask를 지원한다고 하면 안 된다.
- Flag→native USD change flush→pose/velocity write 순서를 commit/rollback에 통일한다.
  준비 단계env0 staging에서 전역 속성이 바뀌면 다른 환경의 velocity도 초기화되므로,
  단일 행 rollback만으로 전체 transaction 복원을 보장할 수 없다. 전역 변경을 최종 commit으로
  옮기는 편이 맞다.
- 원래 enabled collider만 복구해 FlatBox proxy로 대체한 원본 collider를 중복 활성화하지 않는다.
- 공유 layer 소유 정보를 보관하고 강한 참조를 유지한다. 진단의 `/tmp/contact_geometry_`
  경로 판별은 production의 소유권 확인 방법으로 쓰지 않는다.
- 실제17pad/Blocker 접촉 fixture·실패 rollback·다른 seed·짧은 native 학습을 검증한다.

이번 요청은 대안 조사였으므로 현재 production의54절 구현은 변경하지 않았다.
새 방식을 채택하면 기존 전역 바닥/4bit를 유지하는 방향이 가능하다는 실행 근거를 확보했다.


## 56. 2026-10-23 — 미사용 물체 kinematic 전환 적용

사용자의 적용 요청에 따라 미사용 pool body만 공식 USD `physics:kinematicEnabled=True`로
전환한다. 다시 사용할 때는 `False`로 돌리고 native 변경을 flush한 다음 해당 episode의
pose·velocity를 기록한다. Actor를 scene에서 제거하는 simulation OFF/ON은 사용하지 않는다.
기존 주차 위치·스폰 규칙·물체 종류/개수·질량/마찰·로봇·100Hz·MDP·PPO는 유지한다.
54절의 환경별 Cuboid 바닥과8bit 강제 설정은 제거하고 기존 전역 GroundPlane과 native4bit로 복원했다.

### Collider 제외를 적용하지 않은 이유

55절 진단은 shared geometry prototype에 collider OFF를 기록했지만 실제 instance 접촉까지
확인하지 않았다. 이번 실제 접촉 검사에서는 그 속성이 native instance shape에 전파되지 않아
kinematic 주차 물체와 probe가 충돌했다. 따라서 무효한 collider 속성 변경과 공유 layer 관리
코드를 제거했다. 원래 disabled collider도 포함해 기존 collision 설정을 그대로 유지한다.
Kinematic 전환은 native 통계와 중력/재활성화 검사로 확인됐다. 미사용 물체의 동역학 solve와
integration을 제외하지만 collider·broadphase·transform 관리·GPU 메모리는 남는다.
이 방식은 모든 PhysX 연산을 끄는 기능이 아니다. 기존 격리 주차 위치가 유지돼 작업 scene과 분리된다.

### Reset 경계와 실패 복구

- 공통 active mask가 바뀌는 전체 장면 commit에서만 속성을 수정한다. Env0 body의 USD 속성이
  inherited 환경 전체에 적용되므로 서로 다른 환경별 mask나 부분 reset의 mask 변경은 거부한다.
- 같은 mask의 부분 reset은 USD 변경·flush·CPU flag 복사를 생략한다. Actor·GPU view·17개
  tactile 채널과 Blocker contact filter view를 재생성하지 않는다.
- Episode 후보 준비는 물체를 수정하지 않고 로봇만 임시 배치한다. 성공/실패 모두 `finally`에서
  로봇 root·관절 상태·관절 목표를 복구한다. 전역 body flag 전환은 최종 물리 commit에만 있다.
- 속성/flush가 실패하면 이전 flag와 전체 물체 pose·velocity를 복구한다. Kinematic 진입이
  native velocity를 지울 수 있어 flag만 되돌리지 않는다. Mask 검증은 다른 asset/state 기록 전에 수행한다.
- 기존 COM-pose cache 무효화 보정은 유지한다. Countergravity와 매 step의 parking force 제출은 제거한다.

### 검증

- CPU reset·부분 reset·USD inheritance/instance·기존 collider 불변·속성 실패 복구 검사47개 통과,
  CUDA 전용1개 skip. 설치 collection에 `.stage`가 없어 공식 `get_current_stage()`를 사용하며
  CPU fixture도 없는 `.stage`를 가정하지 않도록 수정했다.
- 실제4환경에서 전체/부분 reset12회, unused 주차 오차0, native pose/rotation writer 오차
  최대2.384e−7, 다른 행 pose 오차0을 확인했다. 모든 simulation disable flag는0이다.
- 17개 tactile pad 각각의 실제 binary 접촉/분리와 Blocker 재활성화3회의 filtered contact를
  확인했다. Filtered force는약2.15–2.18N이며 view/path가 유지됐다. 재활성 물체는 중력으로
  12step 동안76.52mm 내려왔고 공식 `env.step()` 상태도 finite였다.
- 최종 production1,024환경에서6장면×80physics step, 각 장면 뒤512행 부분 reset을 통과했다.
  활성 개수17/19/19/23/23/22, 표본 낙하0(지지면보다3cm 아래 기준), 가장 낮은 OBB 최저점은
  지지면 대비−1.317mm였다. Unused 주차 좌표 오차와 선택 행 reset 오차는 최대2.384e−7m,
  미선택512행 pose는 모든 부분 reset에서 bit-exact였다. Native4bit·전역 plane1개를 확인했다.
  Kinematic 수64,512/62,464/62,464/58,368/58,368/59,392는 각각 unused 수와 일치했다.
  Simulation disable0, tactile17body가 유지됐다. `production1024/support.json`과
  `applied_summary.json`에 원시 결과와 집계를 저장했다.

자료: `outputs/contact_manipulation/inactive_applied_20261023/`. 이전 simulation OFF/ON이
실행된 프로세스를 복구하는 패치는 아니므로 적용 후에는 새 프로세스로 시작해야 한다.
고정 제어 회귀 검사이며 장기 학습 안정성이나 전체 PPO iteration 개선율을 보장하지 않는다.


## 57. 2026-10-23 — Reward 분모·Target gate·F/T 분리·Time 복원

사용자의 전체 reward 검토 후 지시에 따라 다음을 적용했다. Rearrangement와 Object stability는
수식 설명 후 계속 논의하기로 했으므로 이번 변경에서 수정하지 않았다. 물리·제어·배치·관측·PPO도 유지한다.

- **Target gate:** 초기 pose 대비 위치2cm·회전0.15rad라는 기존 성공 보호 조건을 재사용한다.
  별도 termination은 추가하지 않는다. 현재 pose가 범위를 벗어난 동안 양의 Clearance와
  Distance·Contact 보상만0이며, 음의 Clearance와 다른 penalty·Time은 유지한다.
  허용 범위로 복구하면 보상 지급도 복구한다. Contact의17개 binary 이전/현재 이력은 gate 중에도
  계속 갱신해 복구 시 오래된 접촉이 보상으로 연결되지 않도록 했다.
- **Clearance 분모:** reset의 물리 Blocker OBB 투영면적을 환경별로 캐시하고 episode 내에서
  고정한다. 분자는 현재 겹침면적이다. 초기 대비 품질 차이·weight4·초기 보상0은 유지한다.
  부분 reset은 선택 행의 면적·보호 상태만 갱신하며 가림 검사나 카메라를 추가하지 않는다.
- **F/T 분리:** `motion_wrench_growth`를 제거하고 `motion_force_growth`와
  `motion_torque_growth`를 각각 native weight0.15로 등록했다. 보정된 센서 frame의 norm 증가율,
  deadband10N/s·0.5Nm/s와 계수ln2/90²·ln2/4.5²를 유지한다. 개별 품질의 반감점은
  100N/s와5Nm/s이고, 힘을 빼거나 일정한 norm을 유지하면 비용0이다. 결합 exp 계산을 남기지 않는다.
  내부 component는 `[Clearance, Rearrangement, Force, Torque, Terminal]`5열이며 관측 폭은 동일하다.
- **Time 복원:** 이전에 선택한 비용률 `c(t)=0.02(exp(0.7t)−1)`을 복원하고 native weight1을
  등록했다. 완료된 control 구간의 평균 비용률을 함수에서 반환하여 공식 RewardManager의
  `×dt`가 정확한 구간 적분 `−integral[t−dt,t] c(s) ds`가 되도록 했다. 다른 discrete reward의
  `/dt` 상쇄와 구분된다. Clock은 완료 전이마다 한 번 증가하고 부분 reset은 선택 행만0이다.
  Scale0도 허용한다. 총 활성 term은12개다.

### 설명 후 논의가 남은 두 항

현재 Rearrangement는 초기 pose와의 차이나 누적 path가 아니라 **직전 control step의 코너
이동량**에 exp를 적용한다. `path_length`는 별도로 누적하지만 reward 입력은 increment다.
함수가 이미 `/step_dt`를 반환하므로 native `×dt`는 상쇄된다. 같은10cm를 첫1step에 이동하고
99step 정지하면 총−0.8647,100step에 나누면 총−1.9801이다. 초기 변위에 지속 비용을 주는
방식과는 다른 수식이며, 초기 변위/누적 조작량/step 조작량 중 의도를 다시 정해야 한다.

Object stability는 현재 선반 frame의 물체 중심XY와 실제 충돌형상 최저점을 사용한다.
중심XY에서 선반 사각형까지 수평 거리와 최저점의 높이 차이를 Euclidean 거리로 묶고,
10mm를 넘는 최악 활성 물체에 비용을 준다. 20mm에서−1.5/step,30mm에서−2.8125/step이다.
거리30mm가0.1초 지속되면 지지 이탈 종료이고,80° 전도0.2초는 별도 종료 조건이다.
지지가 유지되는 기울기·물체 속도·손 접촉 유지 자체는 이 비용의 직접 입력이 아니다.
사용자는 Blocker 불안정 상태에 추가 보상 gate를 두기보다 이 항의 처벌을 활용하길 원해
Distance·Contact에 새로운 Blocker 안정성 gate는 추가하지 않았다.

### 검증

CPU180개 중176개 통과, CUDA 전용4개 skip. 실제 기울어진 OBB의 고정 면적 기준,
Target 위치/회전 위반과 복구·음의 Clearance 유지·Contact 이력·부분 reset의 다른 행 보존,
F/T 독립 반감점과 native term 등록, Time의 구간 적분/시간 분할 불변성을 검사했다.
기존 관측·reset·지지·종료·1024행 tensor 연산 회귀도 포함한다. GPU/학습은 실행하지 않았다.
Time8초 누적값은10/20/100ms 모두약−7.537897이며100Hz의 첫step은약−7.016e−7,
8초 직전 마지막step은약−0.053696이다. 이는 과거 계수의 복원이며 학습 로그를 이용한 재튜닝은 아니다.


## 58. 2026-10-23 — 누적 Rearrangement와 OBB 이탈·기울기 안정성

사용자가 annotation으로 보완 방향을 지정했고 추가 질문에 다음처럼 확정했다.
Rearrangement는 초기 pose 순변위가 아니라 **episode 중 누적 경로**로 왕복 이동도 남는다.
Object stability의 XY 이탈은 물체 중심 대신 **전체 OBB의 돌출 거리**를 사용한다.
기울기는 yaw를 제외한 현재 선반 법선 대비 각도θ를 그대로 exp 지수에 넣으며,
45°에서 기울기 품질0.5인 계수로 시작한다. 이전57절 변경은 모두 유지한다.

### Rearrangement

기존 step별 OBB 평균 코너 이동량을 계산하고 Blocker를 제외한 활성 물체들 및 선반의
누적 경로 `path_length`에 더하는 부분은 재사용한다. 실제 reward 입력만 step increment에서
이 누적값으로 변경했다. 계수20m⁻¹·native weight1·`q−1` 지급을 유지한다.

`P_k=sum_{j<=k} L_j`, `r_R=exp(−20P_k)−1`이다. 정지하면 새 경로는 늘지 않지만
과거 조작량에 따른 비용은 계속 지급한다. 왕복하면 누적값이 커지며 부분 reset은 선택 행만0이다.
같은 누적 경로의 현재 비용은 이동을 나눈 step 수와 무관하지만, 빨리 움직인 뒤 기다리는 궤적은
높아진 누적 비용을 더 오래 받으므로 episode 총합까지 같다는 뜻은 아니다. 순변위가 아니다.
사용자가 양의 품질 보상도 허용했지만 기존 정지 보너스 문제를 추가하지 않도록
`exp(−20P)` 자체가 아니라 `exp(−20P)−1`을 지급한다. 모든 관측·action·PPO 설정은 동일하다.

### Object stability

현재 선반 frame에서 각 활성 물체의 8개 OBB 코너를 유한 작업판 사각형과 비교한다.
가장 멀리 돌출한 코너의 XY 거리 제곱과, 기존 실제 충돌형상 최저점의 지지 높이 차이 제곱을
더해 reward용 거리 `stability_distance`를 계산한다. 원래 중심XY 기준 `support_distance`는
기존 support-loss 종료 조건용으로 그대로 남긴다.

`d_i=sqrt(max_corner(dist_XY(corner,panel)^2)+(z_min−height)^2)`

`theta_i=acos(clamp(canonical_up dot shelf_normal,−1,1))`

`risk_i=(ln2/0.01²)*[d_i−0.01]_+²+(ln2/(pi/4))*theta_i`

`r_stability=3*(exp(−max_active(risk_i))−1)`

거리와 각도에는 각각 m⁻²·rad⁻¹ 계수를 적용하므로 지수는 무차원이다. 10mm 거리 여유,
거리20mm에서 절반 품질, native weight3 및 최악 활성 물체 집계를 유지한다.
기울기에는 θ² 대신 사용자 요청인 θ를 사용하며 물리적 각도 deadband는 추가하지 않았다.
거리 이탈이 없을 때45°에서−1.5/step,90°에서−2.25/step이다. Yaw는 비용이 아니다.
기울기와 이탈이 같은 물체에서 동시에 있으면 그 물체의 위험이 함께 증가한다.
Blocker도 안정성 평가에는 포함한다. 물체 속도 항이나 새 reward term은 추가하지 않았다.

기존 중심 기준 지지 거리30mm·0.1초 및 전도80°·0.2초의 종료 조건은 변경하지 않는다.
Support helper에서 계산한 거리·각도를 캐시하고 같은 GPU batch/기존 CUDA graph 경로로
reward에서 읽는다. 추가 USD/physics 조회나 환경별 Python 반복문은 없다.

### 검증

CPU185개 중181개 통과, CUDA 전용4개 skip. 동일 경로의 분할 비교·정지 뒤 비용·왕복·
선반 이동/회전·Blocker 제외·부분 reset, OBB 돌출/높이·기울기 반감점·yaw 무관성·
비활성 padding·기존 종료 조건 불변을 검사했다. 추가 yaw 검사와 문법/diff 검사도 통과했다.
GPU 시뮬레이션·학습 및 성능 측정은 실행하지 않았다. 누적 비용으로 지급 의미가 달라졌으므로
기존 step 비용 로그와 수치 크기를 그대로 비교하지 않는다. 기본 계수는 이번에 재튜닝하지 않았다.


## 59. 2026-10-24 — 독립 graph Critic과 Actor 실행 순서 확인

사용자 요청으로 Critic에도 Actor와 같은 scene encoder·GNN·역할 조건부 attention을 구성했다.
두 모델은 같은 공개244D 관측을 받지만 각각 새 encoder와 native MLP를 생성한다.
공통 클래스로 구조만 재사용하며 파라미터·정규화 통계·compile cache를 공유하지 않는다.
앞선57·58절의 reward·termination과 환경 생성·물리·제어 설정은 이번 작업에서 변경하지 않았다.

### 현재 구조와 native 경계

- 각 모델: 초기 vision100D → 64D node encoder·GNN2층 → Target/Blocker 조건부 attention과
  pooling → scene320D → 현재72D·직전72D와 concat → ReLU MLP464→256→128.
- Actor 출력은18D Gaussian, Critic 출력은 deterministic value1D다. 현재·직전 입력에는
  별도 학습 encoder를 추가하지 않고 기존 mask·raw history 연결을 그대로 사용한다.
- `NativeContactActor`와 `NativeContactCritic`은 공통 특징 추출 구현을 통해 공식 `MLPModel`을
  상속한다. Forward·분포·loss·GAE·optimizer·runner·logger는 native 구현이다.
  PPO 하이퍼파라미터와 Actor의 초기화·분포·state dict 규약은 유지한다.
- Critic의 value loss가 Actor encoder/head를 직접 갱신하지 않음을 확인했다.
  Critic의 value 추정이 GAE·advantage를 통해 Actor 학습에 미치는 영향은 남는다.

시작 로그에서 MLP가 먼저 보였던 원인은 native 생성자가 먼저 등록한 모듈을 PyTorch가
등록 순서로 나열하기 때문이다. 실제 `MLPModel.forward()`는 `get_latent()`를 호출하여
scene encoder와 concat을 마친 뒤 MLP를 실행한다. Forward hook으로 두 모델 모두
scene→concat→MLP 순서와144D raw history 연결을 확인했다. 로그 첫 부분에 이 실행 흐름을
명시했으며, 등록 순서를 바꿔 기존 Actor의 parameter/optimizer 나열 순서를 변경하지 않았다.

### Checkpoint와 검증

각 graph Critic은 자신의 설정 계약을 native extra state에 저장한다. 기존 raw MLP Critic
checkpoint와 새 Critic은 구조가 달라 전체 resume를 거절한다. 계약·키·shape를 Actor 가중치
로딩 전에 확인하므로 실패한 resume가 모델 일부를 갱신하지 않는다. 기존 native Actor v3 및
지원하던 legacy v1/v2의 play는 이전 Critic 설정을 복원하여 유지한다. 새 graph Critic으로는
새 학습을 시작해야 하며, 이전 Actor의 play 출력은 유지한다.

CPU 학습 테스트60개 중58개 통과·CUDA 전용2개 skip, 전체 프로젝트 테스트422개 중
399개 통과·GPU 관련23개 skip. 가중치 메모리·정규화 통계 비공유, 양방향 gradient 격리,
Critic GNN/attention gradient, native PPO update·로그·checkpoint 왕복 저장과 이전 play를
검사했다. 문법·diff 검사도 통과했다. 테스트는 CUDA 장치를 숨겨 실행했으며 실제 GPU
시뮬레이션·학습·처리량은 측정하지 않았다. 별도 scene encoder의 추가 계산 비용과
기존 value-loss 불안정 개선 여부는 새 학습에서 확인해야 한다.

추가 CPU 감사에서는 실제 Isaac Lab configclass·설정 migration·to_dict와 native PPO 생성까지
검사했다. 구 MLP 및 새 graph Critic의 저장 설정 복원·strict load 후 출력이 원본과 완전히 같았다.
HEAD의 이전 Actor와 새 Actor도 같은 seed에서 초기 tensor·parameter/state dict 순서·extra state·
forward 결과가 모두 같았다. 실제 기본 설정의36step rollout·8epoch native update에서 loss가
유한하고 Critic encoder 가중치가 갱신되는 것도 확인했다.


## 60. 2026-10-24 — 최초 인식 물체만 갱신하는 Critic 관측

사용자와 논의하여 Critic의 물체 정보만 현재 실제 상태로 갱신하는 실험을 적용했다.
reset에서 인식되지 않았던 물체를 Critic에 추가하거나, 재 ray casting으로 물체 목록을
교체하는 방식은 사용하지 않는다. Actor의 입력과 reward·termination·PPO 설정은 유지한다.

- **Actor:** `policy` 그룹의 초기 vision 추정 OBB100D와 현재·직전 robot/sensor/action144D.
  초기 물체 기하는 robot base에서 고정하고 현재 `wrist_3_link` frame으로 재표현한다.
- **Critic:** `critic` 그룹의 현재 실제 OBB100D와 같은144D history. 초기 인식 ID·슬롯·padding을
  그대로 고정하고 해당 물체들의 실제 중심·회전만 매 step 갱신한다. 실제 canonical OBB 크기는
  rigid body이므로 reset에서 캐시한다. 초기 Actor PCA OBB와 Critic의 실제 OBB 값은 다를 수 있다.
- Target/Blocker slot0/1과 인식된 주변 최대8개를 유지한다. 인식되었던 물체가 가려지거나
  선반에서 벗어나도 계속 추적하며, 초기 비인식 물체는 포함하지 않는다. Role ID나 mask 채널을
  추가하지 않았고 두 그룹 모두244D·GNN10node·최종 head464D를 유지한다.

`prepare_initial_geometry`가 이미 계산한 선택 순서를 반환하도록 확장하고, 그 순서의
인식 object ID를 물리 snapshot 슬롯에 reset당 한 번 연결한다. 같은 SceneBatch의 공통 기하
cache를 재사용하며 부분 reset은 선택 행의 연결만 갱신한다. 매 step에는 기존
`TransitionBatch`의 실제 center/rotation을 고정 index로 gather하고, 이미 읽은 native root pose
cache와 Isaac Lab quaternion 연산을 사용해 base→현재 EEF frame으로 변환한다.
추가 물체/mesh/PhysX 조회나 ray casting·per-env Python 반복문·per-step 정렬은 없다.
Robot root의 회전·이동·clone 원점, canonical upright 보정 및 OBB 중심의 link 원점 offset을 반영한다.

공식 ObservationManager에 두 그룹을 선언하고 native PPO의 obs_groups를 Actor=`policy`,
Critic=`critic`으로 연결했다. 두 모델은 독립 GNN·attention·MLP를 유지한다. Critic의 규약은
`native_graph_critic_v2`·`tracked_initial_objects_v1`로 기록하며, 이전 초기 vision Critic과는
같은244D여도 의미가 달라 전체 resume를 거절한다. 이전 v1 graph/MLP checkpoint play는
원래 `policy` 관측으로 복원한다. Actor의 입력/action 계약과 저장 가중치 이름은 유지한다.

### 검증과 한계

통합 CPU 테스트431개 중408개 통과·GPU 관련23개 skip, 문법·diff 검사 통과.
새 검사는 현재 GT 이동·회전·offset·upright 보정, 초기 비인식 물체 제외, 초기 슬롯과 padding
고정, 이동·회전된 robot root와 clone 좌표계, 부분 reset의 다른 행 보존·이전 관측 참조 불변,
native244D 그룹 연결·Actor GT 차단·공유 history·독립 gradient·PPO update·신구 checkpoint를
포함한다. 기존 reward 설정 검사 하나가 이전 weight를 가정해 현재 사용자 값인
Distance4/Hold1/Rearrangement0.5에 맞게 테스트만 수정했으며 실제 reward 설정은 바꾸지 않았다.
추가 CPU harness에서는 실제 Isaac Lab configclass·RSL-RL 설정 변환·ObservationManager와
native PPO 생성을 연결했다. 두 그룹 `(3,244)`·같은144D history·서로 다른100D geometry,
각 모델의 올바른 그룹 선택과 forward·독립 가중치를 확인했다. Kit와 asset adapter는 CPU fixture다.

현재 실행 환경에서 NVIDIA driver에 접근할 수 없어 실제 GPU/Kit 시뮬레이션·처리량·새 학습은
실행하지 않았다. 이번 변경은 현재 물체 상태를 Critic에 제공하는 실험이며 기존 value loss
폭증의 해결이나 성공률 개선을 입증하지 않는다. 미관측 물체 상태·누적 path·시간 등 모든
숨은 reward 상태를 Critic에 추가하는 변경도 아니다. 효과는 새 run에서 비교해야 한다.


## 61. 2026-10-24 — 18D action 분포 및 CPU 출력 진단

사용자 요청으로 `2026-10-24_01-11-37`의 모든18개 action을 조사했다. 해당 run의
TensorBoard는 scalar42종만 저장하며, checkpoint에는 Actor/Critic/optimizer와 metadata만
있다. 원래 학습의 observation·rollout action·차원별 μ가 저장되지 않아 당시 실제 최대 action은
복구할 수 없다. 마지막 scalar step은11684이며 mean std는5.06738이다.

### 저장된 분포에서 직접 확인한 사실

234개 checkpoint를 CPU로 순차 검사했다. 모든 floating Actor/Critic tensor가 유한했고
std도 양수였다. 18개 채널 모두 최대 std는 마지막 저장11650에서 발생했다.

| 0-based index | Action | checkpoint0 std | checkpoint11650 std |
|---:|---|---:|---:|
| 0 | Δx | 0.995 | 4.685 |
| 1 | Δy | 1.002 | 4.792 |
| 2 | Δz | 0.998 | 4.693 |
| 3 | rotation vector x | 1.004 | 5.080 |
| 4 | rotation vector y | 0.999 | 4.985 |
| 5 | rotation vector z | 1.006 | 5.136 |
| 6 | Kp x raw | 1.000 | 5.062 |
| 7 | Kp y raw | 1.003 | 5.169 |
| 8 | Kp z raw | 1.004 | 5.229 |
| 9 | Kp rx raw | 1.007 | 5.126 |
| 10 | Kp ry raw | 0.999 | 5.189 |
| 11 | Kp rz raw | 1.002 | 5.128 |
| 12 | thumb1 raw | 1.007 | 5.071 |
| 13 | thumb2 raw | 1.000 | 5.158 |
| 14 | index1 raw | 0.995 | 5.048 |
| 15 | middle1 raw | 0.994 | 5.131 |
| 16 | ring1 raw | 1.001 | 5.144 |
| 17 | little1 raw | 0.999 | 5.200 |

해당 run의 position/orientation scale은1.0이며 pose 입력에 추가 clip은 없다. Kp는 raw×100을
50–200에, hand는 raw를±1로 제한한 뒤 joint limit에 대응시킨다. 18D raw action은 모두
그대로 previous_action에 들어간다. Gaussian의μ와 무관하게 성립하는11650의 확률 하한은
각 병진 성분의 절대값10cm 초과98.30% 이상,1m 초과83.10% 이상이다. Hand의 끝값 clamp는
각 축84.30% 이상, Kp의50/200 clamp는 각 축88.22% 이상이다. 이는 저장된 분포의 수학적
성질이며 원래 학습에서 측정한 빈도나 실제 EEF 이동량을 뜻하지 않는다.

### CPU 진단과 한계

현재 native Actor를 원래 v3 계약으로 strict load하고,9 checkpoint에 같은256개의 유계 합성
관측을 제공했다. FK/IK로 정합성을 확인한 실제 시뮬레이션 관측은 아니다. 각 입력에서 새로
128회 샘플링하고 전체18D의μ·σ·raw action을 집계했다. 11650의 최대|μ|는9.9473,
새 표본의 최대|action|은28.9746이다. 450은 각각3.0988/8.4299,900은5.6779/11.0900이다.

물리 상태를 고정하고 current/previous raw action만200전이 갱신하는 추가 진단에서는
11650의 새 확률 출력의 전체 기간 최대값이61.0251이었다. 다만10^6 이상의 action이나
거대한 value는 이 조건에서 재현하지 않았으며, 원래 학습의 value loss 급증을 action
feedback만으로 설명하는 결과는 아니다. 원래 궤적·OSC 출력·실행 토크는 재현하지 않았다.

Production code·reward·controller·checkpoint는 변경하지 않았으며 GPU/Kit·학습도 실행하지 않았다.
이전의`action_feedback_initial_observation.pt` 역시 합성 데이터이며, 다른 run의 simulation 재생을
이번 학습의 실측으로 해석하지 않는다. 결과는`outputs/contact_manipulation/action_audit_20261024/`의
`checkpoint_std.json`과`cpu_policy_probe.json`에 저장했다. 과거 실제 action의 최대값을 특정하려면
당시 기록이 필요하며, 이후 run에서는 raw action·μ·σ를18축별로 기록해야 한다.


## 62. 2026-10-24 — Clearance 우선 reward weight 조정

사용자가 reward 항별 weight 추천안을 승인하여 다음 세 값만 변경했다. 적용 직전 코드는
Clearance4·Distance4·Hold/pad1이었으며, README의 과거 Distance0.4·Hold0.1 표를
현재 코드의 이전값으로 해석하지 않는다. 기존 학습의 params·로그·checkpoint는 수정하지 않는다.

| Native reward | 이전 weight | 적용 weight |
|---|---:|---:|
| Clearance | 4.0 | 8.0 |
| Blocker distance | 4.0 | 1.0 |
| Blocker contact hold / pad | 1.0 | 0.2 |

Rearrangement0.5, Action rate/Joint velocity/Force growth/Torque growth각0.15,
Singularity0.3, Object stability3, Time1, Terminal weight1 및 성공1회+250은 유지한다.
수식·exp 민감도·Target gate·성공/실패 조건·policy 주기·action·PPO·물리·reset은 변경하지 않는다.
기존 실행 중인 환경의 메모리 설정에는 소급 적용되지 않으며 다음 환경 생성부터 새 기본값을 쓴다.

### 조정 근거와 한계

직전 `2026-10-24_01-11-37`의 마지막25% 구간에서 weighted `Episode_Reward` 중앙값은
Clearance+0.411, Distance+12.64, Hold+0.101, Rearrangement−3.13, Stability−2.75였다.
이 값은 weighted episode 합을 최대 episode 길이8초로 나눈 native 로그이며 순간 reward나
실제 episode 길이로 나눈 평균이 아니다. Distance가 Clearance의 약31배여서, 접근 상태의
유지보다 초기 가림 대비 개선을 우선하는 비교 실험으로 C를2배·D를1/4로 조정한다.
같은 궤적의 단순 환산은 C0.822/D3.16이며 새 정책의 행동·성공률을 예측하는 결과가 아니다.

Hold의 낮은 로그 기여만으로 weight가 부족하다고 판단하지 않는다. 17개 binary의 합이므로
이전 weight1은3pad 유지 시+3/step·이론적 최대+17이었다. 새 weight0.2는 각각+0.6과+3.4다.
접촉 유지의 지원은 남기되 다수 pad 접촉에서 체류 보상이 커지는 것을 줄인다. Penalty는
이미 상당한 음의 기여가 있어 이번에는 함께 강화하지 않는다.

기본 a_C=1에서 새 Clearance 양의 전역 상한은8(1−exp(−1))≈5.056964/step,
Distance 상한1, Hold 상한3.4다. 양의 dense 합의 느슨한 상한은9.456964/step,
γ=.98 무한 할인합 상한은472.848224다. 실제 초기 가림과 동시 접촉 가능성을 고려한 값은
더 작을 수 있으며, 성공250이 이 상한을 넘거나 성공 선호를 보장한다고 주장하지 않는다.
거리·접촉의 양의 지속 지급과 누적 Rearrangement 비용 자체는 유지되므로 weight만으로
접촉 후 정체나 회피 해법이 불가능해지는 것도 아니다.

새 `2026-10-24_16-57-56`은 조사 시점0–5회뿐이어서 효과 판단에 쓰지 않았다.
직전 run 후반 value loss 중앙값약3.18×10¹¹·Gaussian std중앙값4.63이라는 별도
수치 불안정성이 있었으며 이번 변경을 그 원인의 확정이나 해결로 해석하지 않는다.

### 검증

기존 native reward 등록 검사의 기대값을 새3개 weight에 맞췄다. Reward/ContactReward/
SingularityReward/ObjectSupportValidity의 관련 CPU43개 중41개 통과·GPU 전용2개 skip.
변경한 Python 파일 문법 검사와 `git diff --check`도 통과했다. 새 테스트 파일이나 term은
추가하지 않았으며 GPU/Kit·학습·시뮬레이션은 실행하지 않았다. 프로젝트 README의 현재
weight·지급 예시와 CONTEXT 최신 진입점도 동기화한다.


## 63. 2026-10-24 — 예제 구조의 native dt reward와 움직임 비용

**이 절은 적용 당시 기록이다. Object 선속도 L1합의 tanh와 `object_linear_velocity` cache는
후속64절의 OBB 코너 속도·선형 비용으로 대체됐다. 나머지 native dt·weight·성공 조건은 유지한다.**

사용자가 로컬 `examples/Sweeping-Policy-DRL-Sim-Train`의 reward formulation과 코드 구조를
기준으로 수정하도록 요청했다. Dense score의 `/step_dt` 상쇄를 없애고 native manager가
`weight × dt`를 적용하게 한다. Action-rate·joint-velocity를 공식 L2 비용으로 연결하고,
누적 Rearrangement를 예제의 Shelf/Object collision 방식으로 대체한다. F/T 증가 항은
추가 질문에 **매 step 증가량의 제곱**으로 확정했다. 성공 후 종료·Terminal은 그대로 둔다.

### 수식·설정과 코드 구조

모든 term의 함수·weight·파라미터는 `shelf_manipulation_env_cfg.py::RewardsCfg`에 선언한다.
Action-rate·joint-velocity 계산은 공식 `action_rate_l2`·`joint_vel_l2`에 맡기고,
반환 NaN/Inf만0으로 바꾸는 얇은 wrapper를 cfg에 연결한다. `invalid_state`로 종료돼도
reset 전에 reward가 평가되므로 이전의 비유한 보상 보호를 유지한다. 정상 유한 비용을
clipping하거나 L2 수식을 재구현하지 않는다. Task별 수식은 `mdp/rewards.py`에 둔다.
물체 기하와 F/T는 기존 `TaskState`의 한-step GPU cache를 재사용한다. Native RewardManager가
합산·부분 reset·로그를 처리하며 자체 reward 합산이나 환경 subclass를 추가하지 않는다.

| Term | Native weight | Raw score / cost |
|---|---:|---|
| Clearance | 8 | 초기 대비 품질 차이, 기존 Target 보호·진행 gate 유지 |
| Blocker distance | 1 | 기존 현재 pad-to-OBB exp 품질·Target gate 유지 |
| Blocker contact hold | 0.2 | 기존17개 pad의 직전/현재 Blocker binary AND 합·Target gate 유지 |
| Shelf collision | −0.5 | 선반 pose 또는 속도 기준을 넘으면1 |
| Object collision | −0.5 | 활성 non-Blocker 선속도 XYZ L1합의 tanh |
| Motion action rate | −0.03 | 전체 raw18D action 차이 제곱합, 공식 함수 |
| Motion joint velocity | −0.03 | 팔6+hand 독립6관절 속도 제곱합, 공식 함수 |
| Motion force growth | −0.15 | `[norm(F_now)−norm(F_prev)]_+²` |
| Motion torque growth | −60 | `[norm(T_now)−norm(T_prev)]_+²` |
| Motion singularity | 0.3 | 기존 `exp(−a deficit²)−1` |
| Object stability | 0.5 | 기존 전체OBB XY·collision 최저Z·기울기의 `exp(−risk)−1` |
| Time | 1 | 기존 지수형 시간 비용률의 control 구간 평균 |
| Terminal | 1 | 성공일 때만 `250/dt` 반환, native dt 후 고정+250 |

수정 직전 코드에서 **Object stability는 사용자 설정0.5**였다. 62절과 그 이전 기록의3을
복원하지 않고0.5를 유지했다. Clearance8/Distance1/Hold0.2도 유지한다. 과거 기록의
weight와 수식을 덮어쓰지 않으며 현재 설정의 우선순위는 이 절이다.

Shelf의 기준은 초기 위치 대비5mm, 초기 자세 대비1°, 현재 선속도0.005m/s,
각속도0.05rad/s이며 하나라도 초과하면 비용1이다. 길이·각도·속도 기준을 cfg에서 분리한다.
예제의 `norm(position_delta)+norm(linear/angular_velocity)>0.005` 혼합식과 오래된 gripper
링크·고정 높이 기반 근접 항은 복사하지 않는다. 기존1cm/2° 선반 종료는 유지한다.

Object는 Target과 초기 비인식 물체를 포함한 **모든 활성 non-Blocker**의 현재 선속도를 본다.
Blocker·비활성 pool slot은 제외하고 방향·물체·XYZ를 환경별로 합산한다. 예제의 속도 소수
두 자리 반올림은 넣지 않아 작은 움직임도 연속적으로 반영한다. 이름은 collision이지만
접촉 impulse나 충돌 대상 판별을 추가하는 항은 아니다. 회전·누적 경로·초기 pose 오차를
직접 평가하지 않으며, 정지하면 Object 비용은0이다. Shelf는 초기 pose 변위가 남으면 비용도 남는다.

Motion의 네 L2 비용에는 exp·기준값 나눗셈·deadband를 사용하지 않는다. F/T는 자중·관성
보상 후 현재 sensor 중심에서 norm의 양의 차이만 제곱한다. 하중 감소·일정 norm의 방향 변화는0,
힘과 토크는 별도 term이므로 서로 상쇄하지 않는다. 사용자 선택은 증가율이 아닌 step 증가량이므로
`/dt`를 x에 넣지 않는다. Native dt만 마지막에 한 번 적용한다.

F/T 계수−0.15N⁻²·−60(N·m)⁻²는 ΔF=1N과 ΔT=0.05N·m의 weighted rate가 각각−0.15,
100Hz 지급이 각각−0.0015가 되도록 둔 **초기 설정**이다. 센서 SI 단위에 맞춘 출발점이며
성공 조작의 분포로 재튜닝한 값은 아니다. 실제 증가량·센서 보상 노이즈·접촉/비접촉 구간의
기여를 새 run에서 확인해야 한다. 제어주기를 바꾸면 step 증분 비용도 달라질 수 있다.
Action·velocity 비용도 더 이상−1로 포화되지 않아 전체 reward의 유한 하한은 없다.

### Native dt와 Terminal 예외

현재100Hz에서는 각 dense term이 `raw × weight × 0.01`로 지급된다. 이전 대부분 term은
`/step_dt`로 native dt를 상쇄했으므로 같은 raw score와 weight라도 step 지급이1/100이 된다.
Distance 최대1초 합은1, Hold의 이론적 최대1초 합은3.4다. F/T 등 수식도 달라졌으므로
모든 이전 로그를 일괄1/100로 바꿔 새 정책 결과를 예측할 수는 없다.

Time은 기존 구간 적분을 유지한다. 함수가 반환하는 평균 비용률의 dt는 discrete score 상쇄용
adapter와 다르며, native dt 후8초 할인 전 합−7.537897은 그대로다. Terminal도 사용자의
유지 요청에 따라 일회 성공+250을 그대로 지급한다. 이 항만 `250/dt`로 변환하여 native dt를
상쇄한다. 따라서 Terminal·Time은 다른 dense 항 대비 상대 강도가 크게 달라진다.
성공 정의·안정 유지0.5초·실패 우선·Target gate·실패/timeout의 Terminal0은 변경하지 않는다.
과거 bounded reward/step·Motion 하한·value 상한 수치는 현재 수식의 전체 범위가 아니다.

### 환경별 sum 확인과 제거한 코드

사용자가 예제 `torch.sum()`이 환경별인지 다시 확인하도록 요청했다. 실제 예제는 입력
`objects_lin_vel_w.shape=(num_envs,num_objects,3)`에 `torch.sum(torch.abs(...))`를 쓰며
`dim`이 없어 **환경축까지 모두 합친 scalar**를 반환한다. 환경0만 x속도0.2이고 환경1은0이면
원본은 tanh(.2)=.197375라는 단일 값이 되어 manager에서 두 환경 모두 같은 값으로 방송된다.
우리 구현은 `sum(dim=(1,2))`여서 `[.197375, 0]`을 반환한다. 예제의 이 부분은 그대로 복사하지 않는다.

누적 Rearrangement·`path_length`·이전 물체/선반 OBB 코너 이력·packed `reward_components`와
관련 누적 연산을 제거했다. `clearance_quality`, `terminal_reward`, `force_growth`,
`torque_growth`, `object_linear_velocity`를 이름 있는 tensor로 유지한다. 사용하지 않는
Motion exp decay·증가율 deadband 설정도 제거한다. Step별 센서 읽기·기하 생성·CPU 전송을
늘리지 않으며 부분 reset 행만 이력 초기화하는 규칙은 유지한다.

### 검증

MDP 전체364개 중343개 통과·GPU 전용21개 skip, 학습/model 전체62개 중60개 통과·GPU
전용2개 skip을 확인했다. 이후 비유한 L2 보호 회귀2개를 추가하고 RewardTests12개를 다시
실행해 모두 통과했다. 공식 L2 위임·유한 행 동일성·NaN/Inf 행 분리, 실제 native
RewardManager.compute 구현의 dt 합산과 성공250, 환경별 속도 합산·Blocker/padding 면제·
Target 포함·부분 reset·contact 이력·관측/독립 Critic·성공/실패 기존 동작을 검사했다.
변경 Python 파일 문법 검사와 `git diff --check`도 통과했다. 새 파일은 추가하지 않았다.
실제 GPU/Kit 학습·처리량·성공률 효과는 아직 검증하지 않았다. Scene 생성·물리·제어·
Actor/Critic 관측·모델·PPO 설정은 변경하지 않는다.


## 64. 2026-10-24 — Object 비용의 OBB 코너 속도와 tanh 제거

사용자가 병진·회전을 함께 반영하는 OBB 코너 속도 비용에 동의하고, 기존 Object 비용의
tanh를 빼서 진행하도록 요청했다. 동시에 이 선택을 기억하여 **나중에 수정 방향을 물으면
다시 설명할 것**을 명시했다. 이전 tanh는 예제의 `object_collision`을 옮기는 과정에서
추가했던 포화 함수이며, 사용자가 policy action에 tanh를 요청한 것이 아니다.

### 적용 수식과 의미

물체 $i$의 현재 OBB 중심 $\mathbf o_i$, 그 중심의 선속도 $\mathbf v_i$, 각속도
$\boldsymbol\omega_i$, 코너 $\mathbf p_{i,c}$에서 다음 값을 계산한다.

$$
\mathbf v_{i,c}=\mathbf v_i+\boldsymbol\omega_i\times(\mathbf p_{i,c}-\mathbf o_i),\qquad
c_{O,k}=\sum_{i\ne B,\,i\text{ active}}\frac18\sum_{c=1}^{8}\lVert\mathbf v_{i,c}\rVert_2,\qquad
r_{O,k}=-0.5c_{O,k}\Delta t.
$$

8코너는 평균하여 순수 병진의 이동을8번 세지 않는다. 물체들은 합산하여 여러 주변 물체의
움직임을 모두 비용에 포함하며 물체 개수로 나누지 않는다. 기존과 같이 reward의 privileged
물리 정보로 **Target과 초기 비인식 물체를 포함한 모든 활성 non-Blocker**를 평가한다.
Blocker의 일반 이동은 면제하고 미사용 pool slot도 제외한다. 환경별 계산이므로 다른 환경의
움직임이 현재 환경에 섞이지 않는다.

선속도는 OBB 중심의 속도이며 actor origin이나 COM 속도를 중심 속도로 오인해 사용하지 않는다.
현재 수식은 m/s 하나의 단위로 병진·회전을 결합한다. Native dt로 적분하면 정책 sample에서
근사한 총 코너 경로 길이를 억제한다. 빠른 속도 자체의 비용이나 초기 pose 오차가 아니며
정지하면0이다. 왕복은 양방향 경로가 남는다. Policy step 사이의 충격 궤적까지 정확히 복원하는
이동거리는 아니고 할인율까지 고려하면 빠른/느린 조작의 return이 완전히 같은 것도 아니다.

할인 전·일정 속도·순수 x방향 병진의 비교는 다음과 같다. 이전 항은 속도의 XYZ L1합에 tanh를
적용했으므로 순수 x방향으로 비교해야 두 식의 차이를 분리할 수 있다.

| 조작 | 이전 tanh 비용 | 현재 선형 코너 속도 비용 |
|---|---:|---:|
| 한 물체10cm를0.1초에 이동,1m/s | 약−0.038080 | −0.05 |
| 한 물체10cm를1초에 이동,0.1m/s | 약−0.049834 | −0.05 |
| 10×10cm 상자 제자리 yaw90° | 각속도는 직접 미평가 | 약−0.055536 |

마지막 행은 중심의 XY 코너 반경 $\sqrt{0.05^2+0.05^2}$에 $\pi/2$를 곱한
평균 코너 경로0.111072m를 기준으로 한다. 같은 방향·같은 경로에서 tanh의 포화가 빠른 burst에
더 작은 누적 비용을 주는 효과를 제거한다. 동시에 새 항은 큰 순간 속도·진동·노이즈에도
선형으로 반응하므로 실제 로그에서 Object 기여와 성공·상호작용을 함께 봐야 한다.

### 코드 구조와 변경 범위

`RewardsCfg.object_collision`의 함수 등록과 weight−0.5는 유지한다. 수식은
`mdp/rewards.py::object_motion_cost(center, corners, velocity, active, blocker)`에 두고,
기존 GPU physics snapshot을 처리하는 `events.py`에서 step당1회 계산한다.
이전 `(num_envs,num_objects,3)`의 `TaskState.object_linear_velocity` cache 대신
환경별 `(num_envs,)`인 `TaskState.object_motion_cost`를 보관하며 native reward 함수는
해당 scalar를 반환한다. 부분 reset은 선택 행만 초기화·갱신한다.

활성·Blocker mask는 코너 속도 계산 전에 적용하여 주차된 미사용 물체의 비유한 값이
정상 물체에 섞이지 않게 한다. 비유한 물체의 기여만0으로 처리하며 기존 `invalid_state`
termination은 변경하지 않는다. 코너의 직전step 이력·누적 path·추가 SDK 조회·ray casting·
환경별 Python 반복문·속도의 `/dt` 변환은 추가하지 않는다. 기존 기하와 tensor 연산을 재사용한다.

Time weight1→0.1, Shelf 이진 비용→연속 위험 비용, Distance 계수10은 앞선 **검토 제안이며
이번에는 적용하지 않는다.** 모든 weight, Time 수식, 성공1회+250, Shelf·Motion·Stability·
Target gate·termination·Actor/Critic·PPO·scene 생성·제어·물리 설정은 그대로다.
새 파일은 추가하지 않는다.

### 후속 수정 논의 시 기억할 사항

사용자가 다시 reward 수정 방향을 물으면 **현재의 선형 코너 속도 비용과 이전 tanh 포화의
차이를 다시 설명한다.** 큰 조작의 기여·sample 사이 진동·센서/상태 노이즈·주변 물체 수가
늘 때의 합산 규모를 실제 로그로 확인한다. 선형 비용이 강해 접촉을 회피하는지와 tanh가 빠른
burst에 유리한 편향을 주는지를 비교 대상으로 둘 수 있다. 사용자가 기억해 달라고 한 것은
이 선택과 재검토 가능성이며, tanh를 자동 복원하라는 승인이 아니다.

### 검증

MDP 전체373개 중352개 통과·GPU 전용21개 skip, 학습/model62개 중60개 통과·GPU 전용2개
skip을 확인했다. 전체435개 중412개 통과·23개 skip이다. SceneMotion11개도 모두 통과했다.
순수 병진·회전·pivot, 좌표계 불변성, COM→OBB 중심 속도 보정, Target·비인식 물체 포함,
Blocker·미사용·비유한 물체 분리, native RewardManager의 `weight × dt`, 같은 경로의
속도별 할인 전 비용, 부분 reset·step당1회 cache·1,024환경 tensor 계산을 검사했다.
변경 production Python2개 AST 문법 검사와 `git diff --check`도 통과했다.
실제 GPU/Kit·학습·simulation·처리량 benchmark는 실행하지 않았으며 장기 학습 효과나
추가 tensor 연산의 GPU 처리량은 아직 측정하지 않았다.


## 65. 2026-10-24 — 초기 finite 상태 폭발과 L2 비용·value loss 발산

사용자 첨부 로그와 `2026-10-24_22-25-22` TensorBoard의0–7회 기록이 일치한다.
실행 snapshot은 clean commit `f07f3aa47e19a31b3f7aedb334d112cbee97c4b7`이다.
1,024환경·100Hz·episode8초·rollout36step, native L2/F/T 제곱·Object 코너 속도 수식과
weight−.03/−.03/−.15/−60이 실제 실행됐다. 이전 수식의 로그가 아니다.

| 지표 | 2회 | 최초 이상3회 |
|---|---:|---:|
| Action rate | −0.0708 | −6.173e12 |
| Joint velocity | −0.1291 | −1.152e18 |
| Force growth | −3.493 | −4.183e16 |
| Torque growth | −21.15 | −3.867e30 |
| Value loss | 7,019 | Inf |
| Gaussian mean std | 1.0015 | 1.0038 |
| Invalid-state 종료 | 0 | 0 |

6회에도 같은 이상이 재발했다. Adaptive LR은2회 .0050625,3회1e−5였다.
이번에는 이전처럼 std가약5까지 커진 뒤가 아니라 **std≈1인 초기부터** 폭발했다.
2회에도 양의 항 합은약.02024로 Torque 비용이약1,045배다. −60은1N/0.05N·m 참조를
맞춘 미검증 초기 계수이며, 이 정상 구간의 불균형과 극단적인 수치 오류는 별개로 다룬다.

### 로그에서 확인 가능한 크기와 증폭 경로

`Episode_Reward`는 reset 행의 `sum(weight×raw×dt)` 평균을 최대 episode8초로 나눈 뒤
rollout callback별 평균한 값이다. 따라서 개별 step의 raw 값이나 실제 최대값은 아니다.
episode 길이≤8초를 이용하면 적어도 한 contributing transition의 norm은
`sqrt(abs(logged_cost)/abs(weight))` 이상이다.3회의 보수적 하한은 action 변화18D norm
1.435e7, 관절속도12D norm6.197e9rad/s, 힘 norm 증가5.281e8N, 토크 norm 증가2.539e14N·m다.
정상적인 빠른 조작으로 설명할 수 없으며, 정확한 최초 transition·최대값은 저장되지 않았다.

검사는 대부분 `isfinite`라 거대한 finite 상태를 통과시킨다. `nan_to_num`도 finite 값을
줄이지 않는다. Action-rate·관절속도는 공식 환경별 `sum(dim=1)`이며 환경 전체 합산 오류는
확인되지 않았다. F/T는 raw 측정치를 센서 범위로 clip한 **뒤** 자중·관성 부하를 뺀다.
이 부하는 가속도와 `ω×(Iω)`를 포함하므로 큰 관절/링크 속도가 gyro항을 거쳐 증폭되고,
보상 후 wrench 증가량을 다시 제곱하면 비용은 더 커질 수 있다. 보상 후 observation만의
clip은 reward 원본을 보호하지 않는다. 이는 가능한 증폭 경로이며 최초 물리 오류의 증명은 아니다.

CPU float32에서 finite target−1e31의 제곱 오차는 Inf다. 따라서 큰 finite reward가
GAE/return과 value 제곱오차를 오염시키는 수치 경로는 확인했다. Actor/Critic은 독립 모델이고
native gradient clipping도 각각이므로 Critic gradient가 Actor로 직접 공유된다는 뜻은 아니다.
Advantage·공통 상태 입력 등 간접 경로는 남는다. 마지막 `std >= 0.0`는 분포의 음수 또는 NaN
scale로 발생할 수 있지만 최종 파라미터가 없어 어느 쪽인지 확정하지 않는다. 저장된 checkpoint는
정상 `model_0.pt` 하나뿐이며 Actor/Critic tensor는 모두 finite다.

### 다음 수정의 우선순위와 변경하지 않은 사항

최초 이상 action·관절/링크 상태·가속도·raw/보상 F/T·OSC torque·reset 여부를 보존하고,
모델의 실제 물리 한계에 근거한 크기 검사와 **이상 transition의 PPO 유입 차단**을 먼저 설계한다.
Native 순서는 termination→reward→reset이므로 종료만 추가해도 그 step의 비용은 이미 지급된다.
원인을 숨기는 무차별 reward clipping 대신 비정상 행의 격리와 정상 행 보존을 검증해야 한다.
Action scale1·무clip은 위험 후보지만 이번 최초 원인으로 확정하지 않는다. 이상치 분리 후
정상 F/T 분포와 관성 보상 오차를 확인해 weight를 재조정한다.

사용자가 기억을 요청한 Object tanh 제거도 재검토 대상으로 설명하되 자동 복원하지 않는다.
이번3회의 Object 비용은−.0044로 폭발한 Motion과 다르며, Object 포화 복원만으로 이 오류가
해결된다는 근거는 없다. 이번 작업은 읽기·CPU 수치 확인·기록만 수행했으며 production·PPO·
reward·termination·테스트 파일은 수정하지 않았다. 새 학습·GPU/Kit 재현은 실행하지 않았다.


## 66. 2026-10-24 — Hand F/T observation의 자중·관성 보상 제거

사용자 요청으로 `src/contact_manipulation`의 F/T 측정 경로에서 중력·병진/회전 관성
보상을 제거했다. 기존 `contact_rearrangement` 프로젝트는 수정하지 않았다.

기존 흐름은 `native reaction → SI-200-8 clipping·양자화 → self-load/bias 차감`이었다.
이제 sensor frame의 native joint reaction에 **SI-200-8 clipping·양자화만** 적용한다.
Actor·Critic의 현재/직전 robot sample은 이 측정값을 full scale로 정규화한다.
공통 `TransitionBatch.sensor_wrench`를 사용하는 Force/Torque 증가량 reward도 같은
보상하지 않은 값을 사용하며, 자중과 관성에 따른 norm 증가도 비용에 반영된다.
Reward 수식·weight·dt 처리는 변경하지 않았다.

`self_motion_wrench`·`compensate_wrench`와 bias 상태, distal subtree 이름/ID,
손 링크 질량·관성·COM cache 및 매 step 가속도·각속도 조회를 제거했다.
따라서 SI 범위 제한 뒤 큰 관성 보상값을 빼서 wrench가 증폭되는 경로가 사라졌다.
Object COM→OBB 중심 속도 보정과 물체 local COM cache는 유지한다.
OSC의 **gravity_compensation=True** 및 물리 중력·로봇 질량·관성은 유지한다.

Sensor frame·센서 범위/분해능·raw overrange flag·유효하지 않은 sensor 행의0 처리,
reset current F/T=0·previous sample=0, 부분 reset 및 244D 입력 크기는 유지한다.
첫 완료 step에서는 previous wrench=0이므로 손 자중도 초기 F/T 증가량에 포함되고,
이후 일정한 하중의 증가 비용은0이다. 기존 checkpoint와 입력 크기는 같지만 F/T 의미는
달라지므로 전후 학습/재생 결과를 같은 observation 조건으로 해석하지 않는다.

이번 변경은 큰 action·관절 속도의 최초 원인에 대한 수정이나 발산 해결 검증이 아니다.
추가 clipping·PPO·action scale·reward weight·termination 변경은 하지 않았다.
README와 CONTEXT를 현재 측정 경로에 맞게 갱신했으며 새 파일은 만들지 않았다.

### 검증

기존 MDP·모델/PPO CPU 전체 suite **434개 중411개 통과·GPU 전용23개 skip**을 확인했다.
SI-200-8 포화·분해능·invalid mask, 관측과 reward의 측정값 일치, 현재/직전 이력·부분 reset,
Object COM 속도 변환 및 F/T 처리에서 손 질량·관성·COM·가속도 조회가 없는 것을 검사했다.
OSC gravity compensation=True 설정이 유지됨을 소스에서 확인했다.
Production Python3개 AST 문법 검사와 `git diff --check`가 통과했다.
CUDA 장치가 없는 환경이므로 실제 GPU/Kit·물리 simulation·학습은 실행하지 않았다.


## 67. 2026-10-24 — 예제 기준 팔·Hand 관절 속도 종료 추가

사용자는 `examples/`의 `hand_velocity`를 참고한 종료를 요청했고, 질의에서 우선 관절 속도
종료만 적용하기로 선택한 뒤 Hand 관절도 포함하도록 요청했다.
예제 `sweeping_policy/mdp/terminations.py:150`의 실제 조건은 EEF/손가락 속도가 아니라
팔6개 관절의 `any(abs(qdot)>1.0)`이다. 예제의 일반/random env cfg 모두 threshold1rad/s다.

동일한 공식 `isaaclab.envs.mdp.joint_vel_out_of_manual_limit`을 `TerminationTermCfg`에 직접
설정했다. `max_velocity=1.0`, `time_out=False`이며 UR5e specialization에서 관절 이름을
팔6개 + Hand 독립6개로 지정한다. Joint velocity reward와 같은12개이고 mimic/passive는
중복 검사하지 않는다. Joint ID는 native `SceneEntityCfg`가 해석하므로 asset 관절 순서의
앞12개를 가정하지 않는다. **정확히±1rad/s는 통과하고 어느 관절이든 절대값>1이면 즉시 종료**한다.
여러 관절의 norm/합이1을 넘는 조건이나 일정 시간 유지 조건은 아니다.

속도 종료는 native `Episode_Termination/hand_velocity`로 독립 기록한다.
기존5개 연구 task의 `failure_reasons` 배열은 유지한다. `get_task_state`는 native manager의
활성 `hand_velocity` term과 resolved asset cfg를 읽어 공식 함수를 평가하고,
동일한 결과를 `TaskState.advance`의 실패/성공 판정에도 반영한다. 속도 초과 행은
실패로 표시하고 성공 지속 카운터를0으로 만들어 성공·terminal 지급을 차단한다.
Termination term을None으로 비활성화하면 이 성공 차단도 적용하지 않는다.
공식 함수는 기존 native 관절속도 buffer를 환경별 tensor로 평가하며 추가 PhysX fetch나
환경별 Python loop·CPU tensor 전송을 넣지 않는다. 기존 물리/actuator/OSC 설정은 유지한다.

사용자 선택에 따라 **action-rate 별도 종료·action clipping·reward 수식/weight 변경은 없다.**
팔 속도만 검사하는 예제와 달리 Hand 독립 관절도 같은1rad/s를 적용한 것이 연구 환경의 차이다.
Action 변화량 자체를 제한하지 않으며 native `termination → reward → reset` 순서도 그대로다.
따라서 종료 직전의 큰 finite action/관절속도 제곱 비용은 여전히 reward로 계산될 수 있고,
65절의 전체 수치 발산 방지가 완료된 것은 아니다. 새 파일은 만들지 않았다.

### 검증

CPU 전체 suite **438개 중415개 통과·GPU 전용23개 skip**을 확인했다. 관련41개 집중 검사도
통과했다. Checkout의 공식 속도 함수를 실행해±1rad/s 경계·팔/Hand 양방향 초과·비연속
관절ID·mimic 제외·1,024행 tensor 처리를 검사했다. 실제 UR5e cfg의12관절 이름 연결,
resolved manager cfg 사용, 속도 초과 시 성공/250 보상 차단, term 비활성화 후 성공 복원,
native 원인 로그와 부분 reset 보존도 확인했다. Python 문법과 `git diff --check`가 통과했다.
CUDA 장치가 없어 실제 GPU/Kit·물리 simulation·학습은 실행하지 않았다.


## 68. 2026-10-24 — 최신 min Inspire 설정을 확인해 관절 속도 종료 임계값3rad/s 적용

최신 원격 `IROL-SSU/Sweeping-Policy-DRL-Sim-Train`의 `min` HEAD를 다시 조회했다.
확인한 commit은 `baa3740e38c1a693b2094dc4d141b72de0dd3142`이며 commit 시각은
2026-10-14T07:46:11Z다. 기본 shelf cfg의1rad/s만으로 원격 전체의 실효 설정을 설명했던 것은
Inspire specialization의 override를 놓친 불완전한 설명이었다.

[InspireOpen cfg63–68행](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/baa3740e38c1a693b2094dc4d141b72de0dd3142/sweeping_policy/config/ur5e/joint_pos_random_single_inspire_open_env_cfg.py#L63-L68)은
`hand_velocity`를 `threshold=3.0`, `time_out=False`로 재설정한다.
InspireHand는Open을, HandMass는Hand를 상속한다. Sensors·MassSensors·InitialMass·PrivilegedMass·
MassReaching도 이3rad/s를 상속하며, 별도의
[HandMassLimit cfg53–58행](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/baa3740e38c1a693b2094dc4d141b72de0dd3142/sweeping_policy/config/ur5e/joint_pos_random_single_inspire_hand_mass_env_cfg.py#L53-L58)만
1rad/s로 복원한다. 원격의
[실제 종료 함수152–165행](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/baa3740e38c1a693b2094dc4d141b72de0dd3142/sweeping_policy/mdp/terminations.py#L152-L165)은
이름으로 선택한 팔6개를 검사하며 손가락은 검사하지 않는다.

사용자는 이 확인 후3rad/s 적용을 승인했다. 우리 환경의 공식
`joint_vel_out_of_manual_limit`의 `max_velocity`만 **1.0→3.0**으로 변경한다.
67절에서 사용자 요청으로 확장한 **팔6개+Hand 독립6개**와 이름 기반 `SceneEntityCfg`는
유지한다. 어느 선택 관절이든 `abs(qdot)>3.0`이면 즉시 종료하며 정확히±3.0은 통과한다.
Native `Episode_Termination/hand_velocity` 로그, 속도 실패와 성공 보상의 동시 지급 차단,
term 비활성화 시 성공 판정 복원은 동일하다. 67절의1rad/s는 당시 구현·검증의 역사이며,
현재 임계값은 이68절의3rad/s가 우선한다.

이는 episode 종료 조건만의 변경이다. URDF 관절 범위·actuator 물리 속도/토크 상한·
OSC·action scale/clipping·reward 수식/weight·관측·모델/PPO는 변경하지 않는다.
특히 종료 임계값3rad/s가 Hand의 물리 속도 한계를3rad/s로 바꾸는 것은 아니다.
Action-rate 자체의 제한과 최종 transition 격리도 추가하지 않았으므로, 큰 finite action/
관절속도 비용의 전체 발산 방지가 해결됐다는 뜻은 아니다. 새 파일은 만들지 않았다.

### 검증

임계값 변경 후 CPU 집중 검사 **41개가1.006초에 통과**했다. 공식 함수의±3rad/s 경계 통과,
±3.0001rad/s 종료, ±2rad/s 허용과 팔·Hand12관절 선택을 확인했다. Native 원인 로그,
속도 실패의 성공 보상 차단·조건 비활성화 시 복원, 부분 reset 보존도 검사했다.
변경한 cfg·테스트 Python2개 AST 문법 검사와 `git diff --check`가 통과했다.
실제 GPU/Kit·물리 simulation·학습은 실행하지 않았다.


## 69. 2026-10-25 — 관절 속도 종료 후 즉시 reset과 연산 증가 분석

사용자는 `hand_velocity` 추가 후 초기에 즉시 종료가 반복되고 IK·연산이 늘어난다고 관찰했다.
각 run의 저장 cfg와 TensorBoard를 읽어1rad/s/3rad/s 적용 여부·환경 수를 분리했다.

| 실행2026-10-24 | 환경 수 | 속도 종료 | 초기 episode step | Collection1–3회 중앙값 |
|---|---:|---|---:|---:|
| 23-03-35 | 1,024 | 없음 | 27–66 | 2.65초 |
| 23-19-11 | 1,024 | 12관절·1rad/s | 1 | 5.29초 |
| 23-22-30 | 4 | 12관절·1rad/s | 1 | 2.53초 |
| 23-31-31 | 4 | 12관절·3rad/s | 1.4–1.7 | 2.47초 |
| 23-33-13 | 1,024 | 12관절·3rad/s | 1.4–1.7 | 15.07초 |

최초 compile iteration0은 collection 비교에서 제외했다.100Hz이므로 최신 초기 episode는
14–17ms이며302회에도약4step/40ms다. 마지막10회 collection 중앙값도6.53초다.
속도 종료 snapshot은 거의1이지만 이는 완료 episode의 정확한 원인 비율은 아니다.
`Task_Count/completed`의 callback 평균 차이는 초기약2.4–2.5만/iteration, 후반약9,300이다.
이전 초기약370–490보다 크게 늘었지만 callback 평균이므로 정확한 reset 총수로 쓰지 않는다.
IK search rounds는 초기약2.8 대 이전약1.4이며 fallback count/total은0이다.

### 짧은 episode와 제어·reset의 관계

현재 `_shared_reset_batch`는 부분 reset마다 Blocker 앞 연속 EEF 시작 후보를 새로 뽑아
IK·충돌 검사를 수행한다. 물체 배치·vision은 공유한 값을 재사용한다. 따라서 IK 실패 자체보다
reset 횟수 증가가 유력한 지연 요인이다. 상태 쓰기·센서/manager 초기화·physics forward 등도
포함되므로 collection 증가를 IK만의 시간으로 단정하지 않는다. 이번 단계별 profiling은 없다.

`_write_robot_reset`은 관절속도를0으로 쓰고 position target도 같은 초기 관절 위치로 맞춘다.
그러므로 초기 상태에 임의의 큰 qdot를 직접 쓰는 코드는 아니다. 이후 첫 Gaussian 입력은
std1·무clip, 상대 위치/회전 scale1이다.100Hz에서 큰 arm 목표가 나오는 것은 위험 후보지만
원인 확정은 아니다. 현재 로그에는 처음 초과한 관절/episode age의 구분이 없어 팔과 Hand 중
누가 trigger인지 알 수 없다. 공칭 속도는 Hand 독립관절2/.922rad/s, 팔3.14/6.28rad/s이며
solver의 실제 순간 속도와 별개다. Hand action0도 열린 reset 자세 유지가 아니라 범위 중점이다.
min Inspire의3rad/s는 joint-position 제어와 팔6개 검사 기준이어서 우리 OSC·12관절과 다르다.

302회까지 valid success/raw goal은0, 접촉 유지도 사실상0이다. Reward−49.5→−.0297,
value loss1,961→.0016, std1→.478은 과제 성공의 증거가 아니다. 짧은 종료 상태에서 비용을
줄이는 변화와 구분해야 하며, 특히 충분한 조작 transition이 수집되지 않는 점이 우선 문제다.

### 다음 비교와 변경 범위

초과 관절 ID·팔/Hand별 최대속도·episode age를 기록하고 첫 step에 정지/자세 유지 명령을
넣어 reset 후 물리 응답과 정책 입력을 분리한다. 그다음 정상 조작 속도와 추종 오차를 기준으로
공식 OSC position/orientation scale과 속도 종료 전략을 함께 검토한다. Hand가 원인이면
공식 joint-position EMA도 비교 후보이나 팔 원인을 해결하는 방법으로 오인하지 않는다.
PPO std를 임의 변경하거나 유예 시간만 늘려 문제를 숨기는 수정은 하지 않았다.
이번 작업은 저장 로그·코드의 읽기 전용 분석과 문서 기록이며 runtime·테스트·GUI 수정이나
새 학습·GPU/Kit simulation은 실행하지 않았다. 최초 물리 원인과 단계별 시간은 미확정이다.


## 70. 2026-10-25 — Hand를 관절 속도 종료에서 제외

사용자 요청으로 `hand_velocity`의 검사 관절을 팔6개+Hand 독립6개에서 **UR5e 팔6개만**으로
변경했다. UR5e specialization의 종료용 `SceneEntityCfg.joint_names`만 팔 action의 관절 이름에
맞춘다. 공식 `joint_vel_out_of_manual_limit`, 임계값3rad/s와 즉시 종료 방식은 유지한다.
따라서 팔의 어느 관절이든 `abs(qdot)>3.0`이면 종료하고 정확히±3.0은 통과한다.
Hand·mimic/passive 관절 속도는 이 종료를 유발하지 않는다.

**Joint velocity reward는 변경하지 않았다.** 팔6개+Hand 독립6개의 실제 관절 속도 제곱합과
기존 weight를 유지한다. Native `Episode_Termination/hand_velocity` 원인 로그와 resolved
manager cfg를 사용한 성공·terminal 동시 지급 차단도 동일하다. Hand를 제외하는 것은
손가락의 물리 속도·관절 범위·mimic 제약을 제거하는 변경이 아니다.

사용자가 최신 코드에서 직접 조정한 **OSC position scale0.5·orientation scale0.5·Hand scale0.5**는
그대로 둔다. 초기 Gaussian std1, 다른 PPO 설정·action/controller·URDF·actuator 한계·reward·
reset/IK는 변경하지 않는다. 67–69절은 당시12관절 종료의 기록으로 보존하며 현재 검사 범위는
이70절의 팔6개가 우선한다. 새 파일은 만들지 않았다.

### 검증

CPU 집중 검사 **41개가1.055초에 통과**했다. 팔±3rad/s 경계 통과·양방향 초과 종료,
Hand 독립관절/mimic 초과의 종료 제외, joint velocity reward의12관절 이름 유지를 검사했다.
Hand 초과만으로 이 속도 term이 성공을 차단하지 않는 것과 native 원인 로그·부분 reset 보존도
확인했다. 변경한 Python3개 AST 문법 검사와 `git diff --check`가 통과했다.
실제 GPU/Kit·물리 simulation·학습은 실행하지 않았다.


## 71. 2026-10-25 — 02-40-51 학습의 접근 편향·느린 실제 동작과 CUDA 오류 검토

사용자는 play에서 로봇이 느리고 Hand 움직임이 적으며 접근 후 치우지 않는 경향을 관찰했다.
여기서 속도는 **로봇의 실제 동작**이며 collection/iteration 시간 문제가 아니다.
`2026-10-25_02-40-51`의 저장 env/agent cfg·git snapshot·TensorBoard를 확인했다.
마지막 기록은15497회, 마지막 checkpoint는15450회다. 실행 기준은 commit
`2d559f2b0f76dd1c04c43f28f1634ce853b37b23`과 저장된 dirty diff이며 현재 HEAD만으로 추정하지 않았다.

### 실제 설정과 reward 기여

OSC 위치 scale.03m·회전.15rad, Hand scale.5, 팔6개만3rad/s 종료다. 초기 std1·무clip,
100Hz·episode8초·γ=.98이며 C8/D3(decay10)/Hold.2, Action-rate/Joint-velocity/F/T각−.03이다.
특히 이전 Torque−60·Distance decay6.931·12관절 종료 설정은 이번 run에 적용되지 않는다.
성공 보상+250과 기존 Target gate·안정 유지.5초는 유지됐다.

| 구간 | Clearance 평균 | Distance 평균 | Contact 평균 | 양의 dense 합 중 Distance |
|---|---:|---:|---:|---:|
| 100–999 | .01537 | .77293 | .01042 | 96.77% |
| 1000–3999 | .07067 | .75910 | .01910 | 89.42% |
| 8000–11999 | .10549 | .73926 | .03727 | 83.81% |
| 12000–15497 | .11262 | .70816 | .04958 | 81.36% |

위 값은 weight를 포함한 `Episode_Reward`의 callback 평균으로, 개별 step raw값이나 균등한
episode 평균이 아니다. Native manager는 episode 누적을 최대8초로 나누고 runner가 callback을
평균한다. 후반 각 중앙값은 C .01721/D .54559/H .01223으로 D/C가약31.7배다.
후반 평균 비용은 Action-rate−.16795, Time−.07285, Force−.02836, Joint velocity−.01667,
Object−.01353, Torque−.000424다. 현재는 Torque보다 Action-rate가 가장 큰 dense 비용이며,
Clearance·Contact가 개선되면서도 접근 상태의 수익이 크다. 아직 학습 초반이라는 점도 고려한다.

Distance는17pad의 OBB까지 **최소거리**이므로 한 pad만 가까워도 되고, 추가 손가락을 닫는
효과나 Blocker 이동량은 직접 평가하지 않는다. Contact hold도 기존 접촉 유지이며 첫 접촉이나
진행 자체는 아니다. Action-rate는 물리 scale 전18D 전체, Joint velocity는 Hand 포함12관절에
비용을 준다. 같은 관절 이동량Θ를 시간T에 수행하면 속도 제곱 적분은Θ²/T여서 느릴수록 저렴하다.
동일한 정규화 위치 target 변화에 Hand scale.5는1보다 raw action 차이가2배, 제곱 비용은4배다.
이는 같은 동작을 지시할 때의 비교이고 실속도를 측정한 결과는 아니다. 최종 Hand std는.710–.841,
평균.781로 사라지지 않았으나 play는 deterministic mean을 쓴다. Std만으로 손가락을 닫는 mean이나
추종 여부는 판단하지 않는다. Joint velocity 비용 저하도 episode 길이·scene의 영향을 받으므로
그대로 실제 속도 저하율로 바꾸지 않는다.

γ=.98·100Hz의 할인 반감기는약.343초다.3초/4초 뒤 성공250의 현재 가치는약.583/.0773이다.
2cm 접근 상태의 Distance는3exp(−.2)=2.456/s를 유지할 수 있다. 이는 목적함수 편향의 계산이며
실제 전체 return을 비교한 결과는 아니다. γ=.995의 반감기는약1.383초여서 비교 후보지만,
γ만 바꿔도 접근 상태의 정기 지급 문제가 해결되는 것은 아니다.

### 성공은 드물고 scene별 편차가 크다

최종 누적은 완료1,303,568·성공3,036으로약.2329%다. Raw goal을 본 episode는25,668이고
성공/Raw goal은11.83%의 episode 전환율 대리 지표이며 transition 확률은 아니다.
Unsupported goal10,671과 goal-and-failure11,447은 겹칠 수 있어 합산하지 않는다.
12944–13443의500회 구간은 counter 차이 성공1163/완료34962≈3.33%였지만,
14500–15497은7/약90627≈.00772%,15000–15497은1/약53242≈.00188%다.
구간 차이는 callback 평균 counter에 근거한 근사이고 공통 scene 난이도·종류 편차를 포함한다.
13376회의 성공 snapshot .39947도 전체 성공률이 아니다. 직후만 보고 policy 붕괴로 단정하지 않는다.

후반 episode 길이 중앙값은136.705step/1.367초, 마지막1000회는146.905step/1.469초다.
후반 마지막 원인 snapshot 평균은 속도.2852/timeout .6892/singularity .00353/선반 변위0이므로
선반 회피만으로 설명하기 어렵다. Snapshot은 실제 새 종료 원인 비율이 아니며 timeout에는 공통
scene 갱신도 포함된다. 성공 기회가 부족한 상황과 속도 종료의 영향은 나누어 검증한다.

### CUDA 오류에서 확인한 사실과 미확정 부분

저장 scalar681,912개는 모두 finite다. 직전 value loss .003764·std .7208로 이전 run의 극단적
비용·Inf 연발은 없다. Value 최대721.2인13370회는 성공 bonus가 집중된 구간으로 과거1e30급
발산과 다르다.0/1000/5000/7750/10000/15000/15450 checkpoint의 Actor/Critic·Adam은
finite였으나 마지막 실패 transition과 GPU 내부 상태의 정상성을 보장하는 증거는 아니다.

Traceback의 `value_loss.item()`은 native PPO의 optimizer.step 후 CPU 읽기로, 선행한 비동기
CUDA 실패가 그곳에서 드러날 수 있다. 최초 실패 kernel을 특정하는 증거는 아니다.
[PyTorch 공식 비동기 실행 설명](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
처럼 오류가 드러난 동기화 위치와 원인을 구별한다. Patch capacity는 이미262144이고 이번 제시
로그에는 overflow가 없다. Warp/CUDA graph의 범위 밖 참조나 race도 확인된 원인이 아니다.
사용자는 headless 학습이며 제시한 오류가 보유한 로그의 전부라고 확인했다. 추가 선행 로그가
있다고 가정하지 않으며, 현재 자료만으로 최초 실패 연산이나 렌더링과의 관련성을 특정하지 않는다.

### 다음 비교 방향과 변경 범위

Distance는 접근 개선량에 지급해 정지 중 추가 보상이 생기지 않도록 하는 안을 우선 검토하며,
Clearance를 주목적으로 유지한다. 접촉 직후 Distance만 끄는 단순 gate는 경계 밖에 머무는
유인을 만들 수 있으므로 그대로 적용하지 않는다. 개선량 지급으로 바꾸면 native dt에 따른
크기도 다시 정해야 한다. Hold만 올리면 접촉한 채 정지하는 보상이 될 수 있다.
Play에서 arm pose/Kp/Hand별 raw action→처리 target→실제 관절 위치·속도를 먼저 확인한다.
위치 scale·속도 기준·std를 일괄 높이지 않으며, 물리 비용 변경도 mean 동작 진단 후 비교한다.

CUDA는 진단 실행에서만 physics→MDP→reset→PPO 경계에 동기화를 넣어 최초 실패 구간을 기록한다.
그 뒤 native cfg의 `compile_model=False`와 task graph/eager를 하나씩 비교한다. Warp CUDA
검증과 graph capture의 비호환도 고려하며 상시 동기화를 본 학습의 기본으로 만들지 않는다.
이번 작업은 읽기·CPU 분석·문서 기록뿐이다. Runtime·reward/PPO/제어는 변경하지 않았고,
새 GPU/Kit simulation·학습·오류 재현도 실행하지 않았다.

## 72. 2026-10-25 — Distance의 전체 pad 반영과 정책 50 Hz 비교 설정

사용자는 한 pad의 최소거리 대신 모든 pad를 평가하고, γ 변경에 앞서 정책 50 Hz로
비교하기로 했다. `mdp/rewards.py`와 공통 `ShelfManipulationEnvCfg`에 반영했다.

Distance는 각 pad 표면 대표점에서 현재 Blocker OBB까지 거리 $d_i$를 구한 뒤 평가한다.

$$
r_D=\frac{V_T}{17}\sum_{i=1}^{17}\exp(-10d_i).
$$

거리의 평균을 exp에 넣는 식이 아니라 각 pad 품질을 평균한다. 모든 pad가 같은 거리라면
기존 exp 점수와 같고 범위는0–1이다. 한 pad만 거리0이고 나머지가 매우 멀면 약1/17이며,
다른 pad도 접근할 때 추가 기여가 생긴다. Invalid pad는0점으로 두고 분모17은 고정한다.
계산 중 비유한 거리가 나온 pad도0점으로 처리한다. Helper의 두 번째 반환값은 평균 거리이며,
invalid pad가 있으면 inf다. Production은 품질만 사용한다. Weight3·decay10·Target gate는
유지했고 초기 대비/직전 대비 개선량 reward로 바꾸지는 않았다. Contact hold는 기존17개
Blocker binary의 직전·현재 AND 합산을 유지한다. 새 loop나 CPU 전송은 추가하지 않았다.

| 항목 | 이전 | 이번 비교 |
|---|---:|---:|
| 물리 dt / 주파수 | .01초 / 100Hz | 동일 |
| Decimation / policy | 1 / 100Hz | 2 / 50Hz |
| OSC torque 재계산 | 100Hz | 동일 |
| 상대 목표 갱신·previous 간격 | 10ms | 20ms |
| 최대 episode / 공통 scene round | 8초 / 800step | 8초 / 400step |
| 성공 연속 유지 | .5초 / 50sample | .5초 / 25sample |
| 전도 / 지지 상실 지속시간 | .2초 / .1초 | 동일, 10 / 5sample |
| Rollout36의 simulation 시간 | .36초 | .72초 |
| γ | .98 | 동일 |

Native 환경은 `step_dt=sim.dt*decimation`을 사용하므로 지속시간·round 경계는 자동 반영된다.
OSC 목표는20ms 동안 유지하고 물리 substep마다 토크를 재계산한다. Scale(.03m/.15rad,
Hand.5), actuator·종료 기준·reward weight·PPO 설정은 바꾸지 않았다. 같은 상대 action을
유지하면 목표 갱신 빈도 감소로 실제 이동이 느려질 수도 있어 속도 개선을 보장하는 변경은 아니다.
γ=.98의 할인 반감기는약.343→.686초다. Action-rate·F/T 증가량 제곱은 policy sample 간격에
의존하므로 그 비용이 주기 불변이라고 해석하지 않는다. Contact hold도20ms 간격의 두 sample을
사용하며 두 sample 사이의 모든 순간 접촉을 검사하는 것은 아니다.

**검증:** 기존 contact 검사5개가 수정 전 통과하는 것을 확인했다. 수정 후 contact·reward·
termination·support 검사48개 중46개 통과, CUDA 전용2개는 GPU 부재로 skip했다. 개별 pad
추가 접근, 평균 exp와 exp(평균거리)의 구분, 회전 OBB, Target gate, invalid pad,1,024환경의
dtype/host scalar read 금지,50Hz 성공.5초·전도.2초·고정 Terminal 지급을 포함한다.
기존 config 검사에 남은 Distance weight1 기대값도 현재 사용자가 설정한3에 맞췄다.
실제 simulation·새 학습·GPU 성능 측정은 실행하지 않았다. README와 이 문서에 변경을 기록했다.

## 73. 2026-10-25 — 정책 주기 100 Hz 복원

사용자 요청으로 `ShelfManipulationEnvCfg.decimation`을2에서1로 되돌렸다.
물리 dt.01초·OSC100Hz는 유지하므로 정책도100Hz가 된다. GUI render interval은 native
설정의 `self.decimation`에 따라1로 연동된다. 성공 유지.5초=50sample, 최대 episode와
공통 scene round8초=800step, rollout36=.36초다. Distance의17개 pad별 exp 평균,
weight3·decay10·Target gate, γ=.98·action scale·그 외 reward/termination은 유지한다.
README와 CONTEXT의 현재 기본값을 갱신했다. 학습·simulation은 실행하지 않았다.

## 74. 2026-10-25 — 배치 생성 FFT의 CUDA OOM과 커널 수명 축소

첨부 traceback은 `_clutter_gpu_seed`의 `torch.fft.rfft2`에서12MiB 할당을 실패한 명시적인
`torch.OutOfMemoryError`다. 이전 PhysX719의 최초 원인이 증명된 것으로 해석하지 않는다.

| 첨부 OOM 시점의 항목 | 보고값 |
|---|---:|
| GPU 전체 / free | 23.55GiB / 50.75MiB |
| 별도 PID858212 | 11.98GiB |
| 실패한 학습 process 전체, native 포함 | 10.85GiB |
| 학습 process PyTorch allocated | 1.92GiB |
| PyTorch reserved 중 unallocated | 957.81MiB |
| 위 allocated 중 private pool | 121.41MiB |

PyTorch allocator는 자신의 PID에만 `this process`, 다른 NVML PID에 `Process <pid>`를
출력한다. 따라서11.98GiB와10.85GiB는 같은 process를 중복 집계한 수치가 아니다.
[공식 allocator 소스](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.0/c10/cuda/CUDACachingAllocator.cpp)
에서 이 분기를 확인했다. 다른 process와 합22.83GiB가 사용된 상황이며, PID858212의 프로그램과
현재 생존 여부는 확인하지 못했다. 현재 작업 환경에서는 NVIDIA driver 접근이 되지 않아
실시간 NVML 조회나 GPU 재현은 불가능했다. Process 종료는 하지 않았다.

Reserved 중 unallocated957.81MiB는 전부 즉시 반환 가능한 VRAM이라는 뜻이 아니다.
Fragmentation 여부는 단일 로그로 확정하지 않으며, `expandable_segments:True`도 다른 process의
점유를 해소하지는 않는다. 매 reset `empty_cache()`나 allocator 환경변수 변경은 추가하지 않았다.
[공식 메모리 관리 문서](https://docs.pytorch.org/docs/2.7/notes/cuda.html#memory-management)를 따른다.

코드에서는 형상별 전체-grid `complex128` FFT2개를 함수 종료까지 유지했다.
기존 `clutter_mixing_20261022/native_clutter_reset_1024_probe.json`의 FFT shape756×1782와
고유 커널43개를 기준으로 spectrum pair당약20.58MiB, 총약884.92MiB다. 이는 기존 자료에
대한 산술 추산이며 이번 OOM의 정확한 cache 크기를 실측한 값은 아니다. 함수 local cache여서
정상 반환 후에도 지속되는 누수라고 단정하지 않는다. 이전 variant의 큰 중간 tensor가 다음
body의 scene FFT 생성까지 참조된 점도 peak 사용량을 높이는 구조였다.

수정은 `mdp/events.py`의 해당 함수에 한정했다. 실제 처리 순서에서 각 kernel의 마지막
사용 위치를 미리 구하고 마지막 사용이 끝난 행에서 cache 참조를 제거한다. 재사용이 남으면
유지하므로 추가 FFT 계산은 없다. 선택 결과에는 작은 bool mask와 origin만 남기고,
counts/rounded/free/contact/score 및 scene spectrum은 사용 직후 해제한다. 기존 고유 커널
진단값은 총 생성 개수로 유지하고 `peak_cached_shape_kernels`를 추가했다. 이 peak가 모든
입력에서2로 고정되는 것은 아니며 향후 재사용 패턴에 따라 달라진다.

배치 규칙·FP64·0.5mm raster·argmax 순서·난수 사용·최종 좌표·물리/정책100Hz·reward·
termination은 바꾸지 않았다. CPU fallback이나 OOM 후 다른 장면 재추첨도 추가하지 않았다.

**검증:** 수정 전 함수를 임시 보존해 CPU allocator를 사용한 동일 FFT 수치 경로로4사례
(44고유 커널, 반복2커널, optional 실패, mandatory 실패)를 비교했다. 좌표·variant IDs와
모든 기존 provenance 값이 정확히 같았다. 새 회귀 검사는 weak reference로 spectrum 수명,
44고유 커널 입력의 cache peak2,반복 커널 재사용과 FFT 횟수 유지,독립 polygon 간격을 확인한다.
관련17검사 중12통과·실제 CUDA가 필요한5개는 skip했다. `git diff --check`도 통과했다.
이는 CPU 수치·수명 검증이며 실제 GPU peak·throughput·장시간 OOM 해결을 보장하는 측정은 아니다.

## 75. 2026-10-26 — Time을 매 step 양의 감소형 보상으로 전환

사용자는 Time을 penalty에서 reward로 바꾸고, 성공 시 일회 지급 대신 **매 step 초반에 크고
시간이 지날수록 작은 양의 보상**을 선택했다. 이에 따라 현재 episode의 경과 시간 $t$에 대해
기준 품질 $q_T(t)=\exp(-0.7t)$를 사용한다. Weight는1이며 decay0.7은 공통 env cfg의
`RewardsCfg.time_reward.params`에서 조정한다.

$$
\bar q_{T,k}=\exp[-\beta(t_k-\Delta t)]
\frac{1-\exp(-\beta\Delta t)}{\beta\Delta t},\qquad
r_{T,k}=w_T\bar q_{T,k}\Delta t,\quad \beta=0.7,\quad w_T=1.
$$

함수는 control 구간 평균을 반환하고 공식 RewardManager가 weight·dt를 한 번 적용한다.
초기 곡선값은1,1초약.496585,4초약.060810,8초약.003698이다. Reset 직후 완료 transition이
없으면 지급0이며100Hz의 첫 지급은약+.009965다. 8초 할인 전 누적은
$(1-\exp(-5.6))/0.7\approx+1.423289$다. 기존−7.537897의 부호만 바꾼 것이 아니라,
증가하는 음의 비용률을 감소하는 양의 품질로 바꾼 것이다.

`mdp/rewards.py`에서 기존 `time_penalty`를 `time_reward`로 교체했다. 구형 MdpCfg의
`time_penalty_scale`·`time_penalty_growth_rate`와 validation 예외도 제거했으며 호환 alias를
추가하지 않았다. Console/TensorBoard term 이름은 `Episode_Reward/time_reward`다.
기존 실행의 old term/구형 parameter override와는 구분한다. 별도 캐시나 새 파일 없이 기존
GPU elapsed tensor를 재사용한다. 새 연산은 환경별 exp와곱셈이며 CPU tensor 복사/동기화는 없다.

경과 시간은 native randomized episode counter가 아니라 실제 완료 transition의 누적 시간이다.
반복 조회는 시계를 늘리지 않고 부분 reset은 선택 행만 초기화한다. 물리/정책100Hz·γ=.98·
Terminal250과 다른 reward·종료 조건·gate는 바꾸지 않았다. 성공 여부에 따른 새 지급 gate도 없다.
이 선택에서는 계속 생존할수록 양의 Time 누적합 자체는 늘어난다. 따라서 빠른 성공의
직접적인 진행 보상으로 해석하지 않으며, 그 학습 효과는 다른 항들과 함께 관찰해야 한다.

**검증:** RewardTests·TerminationTests의 CPU32검사가 모두 통과했다. 양수·단조 감소,
.01/.02/.1초 partition의 같은 구간 적분, reset0·선택행 clock 복구,decay 유효성,
host scalar/CPU전송 금지,기존 성공250 지급·종료 판정 유지가 포함된다. `git diff --check`도
통과했다. 실제 GPU simulation·새 학습·정책 성능 검증은 실행하지 않았다.

## 76. 2026-10-26 — 접근 후 정지·손등 밀기 관찰과 최신 실행의 보상 기여 분석

사용자는 play에서 손등으로 미는 듯한 동작과 Blocker를 조금 민 뒤 멈추는 경향을 관찰했다.
Clearance·Contact hold의 영향과 빠른 조작을 유도하는 Time의 역할을 검토하기 위해 최신 저장 실행
`2026-10-25_18-57-11`의 **iteration17183까지**를 CPU에서 읽었다. TensorBoard의
`Train/*/time` 마지막 step57091은 시간축이며 iteration57091로 해석하지 않는다.

**실행 설정과 현재 코드의 구분:** 저장 env는100Hz(dt.01·decimation1), γ=.98,
OSC 위치/회전 scale.03m/.15rad·Hand scale.5다. Clearance8·Distance3(decay10)·Hold.2,
Action-rate·Joint-velocity·Force-growth·Torque-growth는 각각−.03이다. 저장 git diff에서
Distance는 이미 `mean_i(exp(-10*d_i))`의17개 pad 평균이었다. Time은 **기존 음의
`time_penalty`**, scale.02·growth.7이다. 따라서 이 실행을75절에서 변경한 현재 양의
`time_reward`의 실험 결과로 해석하지 않는다.75절의 구현과 사용자 선택은 이번 분석에서
변경하지 않았다.

| Weighted `Episode_Reward` 항 | 최신1000회 평균,16184–17183 |
|---|---:|
| Clearance | +.108105 |
| Distance | +.495214 |
| Blocker contact hold | +.049747 |
| Action rate | −.131306 |
| Joint velocity | −.054964 |
| Force growth | −.015432 |
| Torque growth | −.000382 |
| 기존 Time penalty | −.064176 |

이 값은 native RewardManager의 **weighted episode 누적합을 최대 episode 길이8초로 나눈
로그**를 reset callback/iteration에서 평균한 값이다. 실제 매 step 값이나 같은 수의 episode를
직접 평균한 비용률이 아니며, 항의 기여를 비교하는 지표로 사용한다. C+D+Hold의 평균 합에서
Distance가약75.83%,Clearance16.55%,Hold7.62%다. Terminal은 이 비중 계산에서 제외한다.

초기0–999회의 C/D/H 평균은 .0145/.4592/.0144였다. 최신1000회에는 .1081/.4952/.04975로,
Clearance와 Hold도 실제로 증가했다. 따라서 이 항들이 전혀 작동하지 않는다거나 학습 전체가
붕괴했다고 단정할 수는 없다. 다만 접근 보상이 여전히 가장 크게 실현된다. Hold의 낮은 실현값은
작은 weight뿐 아니라 접촉 자체가 드물거나 짧아도 발생하므로, 이 값만으로 weight 부족을
확정하지 않는다. 손등 동작도 action·pad별 접촉·손 자세 trajectory가 기록되지 않아 현재
집계 로그로 원인을 특정할 수 없다.

누적 완료 counter는약1,885,772,유효 성공1546으로약.08198%다. 최신1000회의 counter 차이는
완료약149,192.875·성공218,약.1461%다. Counter는 rollout 내 여러 reset extras의 평균을
통해 기록되어 window 경계에 소수가 생길 수 있으므로 이 비율은 근사다.1000회 구간 성공은
11000–11999에서약.3326%,16000–16999에서약.1657% 등 장면별 변동이 있다. 성공은 증가한
구간도 있으나 아직 드물며, 같은 배치에서 정책이 멈추는 원인을 분리한 실험은 아니다.

최신1000회 native 종료 snapshot 평균은 팔 속도44.06%,timeout50.85%,전도4.71%,
선반 변위0이다. 이는 각 환경의 마지막 종료 원인 snapshot을 평균한 것이므로 해당 구간에서
새로 종료한 episode의 원인별 비율로 해석하지 않는다. 팔 속도 종료가 여전히 관찰되지만
그 사실만으로 play에서 멈추는 행동의 직접 원인이라고 단정하지 않는다.

기존 음의 Time 비용률 `.02*(exp(.7*t)-1)`은 .5초에약.00838/s,1초.02028/s,
2초.06110/s,4초.30889/s다. 최신1000회의 `Train/mean_episode_length` 평균은약202.17step
(100Hz에서약2.02초)로, 초기1–2초 동안에는 시간 비용이 작고 뒤쪽에서 가팔라지는 구조였다.
이는 초반 빠른 조작을 우대하는 신호가 약할 가능성을 보여주지만 계수 변경의 충분조건은 아니다.
현재 양의 Time은 생존할수록 자체 누적 보상이 증가한다는75절의 성질과 구분해 논의한다.

이번 작업은 기존 로그·저장 cfg·git diff의 읽기 전용 분석이다. 행동·모델 코드, reward·종료
설정은 변경하지 않았고 학습·simulation·GPU 연산을 실행하지 않았다. 새로운 프로젝트 파일은
추가하지 않았다. 향후 비교에서는 같은 유효 조작의 완료 시간·pad별 Blocker 접촉·누적 항별
return을 함께 확인해 시간 유인 부족,접촉 부족,접촉 방향 문제를 구분해야 한다.

코드상 Distance는 pad 노출면 대표점의 **위치**만 사용하고 면 법선이 Blocker를 향하는지는
평가하지 않는다. 손등이 앞을 향해도 pad 점들이 OBB에 가까우면 점수를 받을 수 있다.
Hold는17개 tactile body의 Blocker-only force norm으로 평가하므로 별도 손등/하우징만의
접촉에는 직접 지급하지 않지만, pad의 앞/옆/뒷면은 구분하지 않는다. 손등으로 보이는 순간에도
pad가 함께 닿는지 확인하지 않으면 높은 weight가 필요한지,접촉 방향 정의가 필요한지 모른다.
Clearance와 Hold 모두 정지 상태에서도 기존 개선/접촉이 유지되면 계속 지급한다.
예시로 초기 가림비율.5→.45의 작은 개선을 유지하면 Clearance는약+.24878/s,
pad 하나 유지접촉은+.2/s다. 기존 Time의1초 비용률은−.02028/s에 불과하다. 이는 실제 play의
상태를 측정한 것이 아니라 현재 수식이 소폭 개선 후 정지에도 수익을 남길 수 있음을 보인 계산이다.
따라서 Contact/Clearance weight 부족을 확정하거나 무조건 높이기보다 Time 초반 역할과
정지/진행/손등/패드 접촉 구간의 항별 return을 비교한다. 사용자가 유지하기로 한 다른 항과
제어·종료 조건은 이번 분석에서 변경하지 않았다.

## 77. 2026-10-26 — GUI tactile·F/T 시각 모니터

사용자 요청에 따라 기존 OBB 토글과 같은 **IsaacLab → Scene Debug Visualization**에
**Tactile / F/T monitor**를 추가했다. 기본 OFF이며 기존 Environment Index로 선택한 환경만
읽는다. GUI1은 `env_0`이다. 기존 `sensors.py`의 SensorMonitor가 표시를 담당하고,
`events.py`의 OBB visualizer가 하나의 render callback과 DebugDraw 선·점 버퍼를 관리한다.
별도 프로젝트 파일·USD prim·rigid body·collision shape·contact view는 추가하지 않았다.

- Tactile:17개 pad의 노출면 대표점을 손 위에 표시하고, Palm·Thumb·Index·Middle·Ring·Little
  타일로 채널별 상태를 보여준다. 접촉은 초록색, 미접촉은 회색, invalid는 주황색이다.
  Observation과 같은 threshold의 **any-object binary**이며 Blocker 구분을 새로 계산하지 않는다.
- F/T:물리 센서 frame의 RGB축과 청록색 힘 화살표, 오른손 법칙의 주황색 토크 원호를 표시한다.
  힘1cm/N·길이상한25cm, 토크반경6cm·1N·m당288°·각도상한288°다.
  축별 양방향 막대는 SI-200-8 한계200/200/360N·8/8/8N·m에 정규화한다.
  공식 `LiveLinePlot` 두 개에 Force[N]와 Torque[N·m]의 최대200개 sample을 유지한다.
  초기 plot 범위는±20N/±1N·m이고 native Autoscale checkbox로 변경할 수 있다.
- Data:기존 `task_sensor_reader.batch`와 현재 `TaskState.previous_wrench`만 읽는다.
  표시값은 SI-200-8 보정·양자화된 측정 반력으로 Hand 자중·관성을 포함한다. GUI에서
  SensorReader·get_task_state·reward를 실행하지 않으며 물리/MDP 상태를 변경하지 않는다.
- Lifetime:패널 최대20Hz 갱신, 환경 선택·episode reset 시 그래프 초기화, invalid 값은
  미접촉/정상0값과 구분한다. 비유한 body pose는 DebugDraw에 보내지 않는다.
  Monitor OFF에서는 센서 조회·기하 연산·CPU 전송·UI 갱신이 없고, Headless에서는 monitor와
  UI 자체를 생성하지 않는다. 다른 OBB 표시가 ON이면 기존 callback은 유지한다.

CPU에서 신규 monitor10개·기존 sensor6개·OBB6개, **총22개 검사 통과**를 확인했다.
채널 순서·선택 환경17점·센서축 회전 공변성·부호 막대·invalid 안전 처리·캐시 읽기 전용·
OFF/headless 무동작·OBB 공존·종료 정리·STOP 시 PhysX 무조회를 검증했다.
Fake UI에서 실제 panel build와17개 Rectangle 배경 타일·native plot 설정도 확인했다.
실제 Kit GUI 레이아웃·GPU 렌더링 및
실제 센서 접촉 장면은 실행 검증하지 않았다. Reward·termination·학습 설정은 이번 작업에서
변경하지 않았다.

## 78. 2026-10-26 — Reward weight·exp 민감도 동시 조정과 일정 Time 비용

사용자는 Time만 조정하는 대신 전체 reward의 상대 관계와 exp 내부 계수를 함께 검토한 뒤
제안한 실험 설정을 적용하도록 승인했다. 접근·유지 상태의 수익을 줄이고 Clearance 개선의
민감도를 높이며, 움직임 비용은 완화하고 주변 물체·선반 교란 비용은 강화했다.

| 항 | 직전 값 | 적용 값 |
|---|---:|---:|
| Clearance weight / decay | 8 / 1 | 12 / 2 |
| Distance weight / decay | 3 / 10 | 1 / 15 |
| Contact hold | .2 | .2 유지 |
| Shelf collision | −.5 | −2 |
| Object collision | −.5 | −1 |
| Action rate / Joint velocity | 각각−.03 | 각각−.01 |
| Force growth / Torque growth | 각각−.03 | 유지 |
| Singularity | .3 | 유지 |
| Object stability | .5 | 1 |
| Time | 양의 exp 감소형, weight1 | 공식 is_alive, weight−2 |
| Terminal | 성공+250 | 유지 |

설정은 `RewardsCfg`와 `MdpCfg.clearance_decay`에서 수정했다. Clearance는 여전히 초기 대비
품질 차이이며, 양의 부분에 대한 Target 보호·지지 유효성 gate는 유지한다. Distance는17개
pad 품질 평균, Hold는 같은 pad의 직전·현재 Blocker 접촉 binary 합이다. 둘의 Target gate,
모든 termination, action/관측/모델·PPO·정책/물리100Hz와 물체 생성 규칙은 변경하지 않았다.
F/T의 오래된 주석·README에 남아 있던−.15/−60 설명을 실제−.03/−.03 값에 맞췄으며,
같은 숫자라도 N²와(N·m)²에서 같은 민감도를 뜻하지 않는다고 명시했다.

**민감도:** weighted exp의 기울기 크기는 `w*a*exp(-a*x)`다. Clearance는 가림비율.5에서
4.85225→8.82911로약82% 커지며,0~1 전체에서 개선 기울기가 커진다. 가림비율.5→.45의
보상률은+.248780→+.464283, .5→0이면+3.147755→+7.585447이다. 계수만2로 올리고
weight8을 유지하면 큰 가림에서는 기울기가 약해지므로 두 값을 함께 바꿨다.
Distance의 반감거리는6.93→4.62cm다. 모든 pad가10cm일 때 weighted rate는
3*exp(-1)=1.103638에서exp(-1.5)=.223130으로 줄어든다. 가까운 영역에 상대적으로 집중하되
접근 자체의 절대 기울기도 감소하므로 새 학습에서 접근 신호가 유지되는지 확인해야 한다.
Stability의 거리/기울기 계수와 Singularity 계수는 유지했다. Stability weight1이면 다른
위험이 없을 때45°기울기나20mm지지 오차의 비용은−.5/s다.

**Time 구현과 지급:** custom `mdp.rewards.time_reward` 함수·export·cfg term을 제거하고,
공식 `isaaclab.envs.mdp.is_alive`를 `time_penalty`의 함수로 직접 설정했다. 별도의 wrapper나
새 파일은 없다. 공식 함수는 `(~env.termination_manager.terminated).float()`를 반환한다.
Native RewardManager가 weight·dt를 한 번 곱한다.

$$
f_{T,k}=\mathbf1[\neg\mathrm{terminated}_k],\qquad
r_{T,k}=-2\Delta t\,f_{T,k}.
$$

일반 step에서는100Hz의dt=.01이므로−.02다. 초기부터 비용률이 일정−2/s이며 exp 계수나
T_ref, elapsed clock·누적 이력은 없다. 기존 exp 구간 적분도 필요 없다. 동작 여부·접촉 여부와
무관하게 시간이 흐르면 비용이 발생하고, 성공이 빨리 끝나면 남은 비용을 피한다. 모든 step이
과금되는1/2/4/8초의 할인 전 합은−2/−4/−8/−16이다. dt와 weight를 함께 고려하므로
이 항만 보면50Hz에서−.04/step이어도 같은 실제1초 비용은−2지만, 정책 주기는 변경하지 않았다.

공식 env 순서는 termination 계산 → reward 계산 → reset이다. 성공·실패 종료 step에는
`terminated=True`여서 Time 비용0이다. Timeout 또는 공통 장면 교체 truncation만 있으면
`terminated=False`이므로 해당 마지막 step도−.02다. 동시에 성공/실패가 있으면 Time0이다.
따라서2초째200번째step에서 성공한 episode의 Time합은−3.98,4초 성공은−7.98,
8초 timeout800step은−16이다. 성공 Terminal+250은 별도로 한 번 지급한다. Reset 자체에는
과금하지 않으며 새 episode의 첫 정상 step부터−.02다. 부분 reset에 복구할 Time history는 없다.
센서·PhysX 추가 조회나 host scalar 읽기 없이 기존 termination tensor만 사용한다.

**해석과 한계:** 비용은 성공뿐 아니라 실패로도 더 이상 발생하지 않는다. 큰 음의 Time이
조기 실패를 유리하게 만들 수 있어 실제 종료 원인 counter와 성공률을 함께 확인해야 한다.
γ=.98·100Hz의 할인반감기는약.343초다. Time 보상열 자체의 할인합은1초−.86738,
2초−.98241,8초약−1이며 무할인 누적값과 다르다. Timeout에서는 PPO가 다음 가치도
bootstrap하므로 이 숫자는 전체 학습 return이 아니다. γ와 PPO 설정은 이번에 변경하지 않았다.

정상 지지·Target 보호 상태에서4개 pad 접촉,나머지13개5cm,가림비율.5→.45로 정체하면
다른 비용0 가정의 새C+D+Hold+Time 합은−.139202/s다. 같은 기하에서가림비율.3이면
+1.567702/s,0이면+6.981962/s다. 실제 trajectory 측정이 아닌 수식 예시다. 유지접촉6개면
작은 개선 상태도+.322873/s이므로 weight만으로 모든 정체 상태를 제거하지는 못한다.
손등 방향을 평가하는 수식도 새로 추가하지 않았다. GUI pad monitor와 접촉 후 Clearance
변화·정체 시간, 유효 성공과 실제 종료 원인별 counter로 새 학습 효과를 확인한다.

**검증:** Reward·Contact·SceneMotion·Singularity·State·Termination·ObjectSupportValidity의
총94개 검사 중 **CPU91개 통과, CUDA전용3개 skip**. 실제 공식 `is_alive`와
`RewardManager.compute()`를 사용해 일반/timeout/성공/실패의 비용,dt별8초−16 누적을
검증했다. 전체 설정·Clearance 새 계수와 초기 품질·부분 reset·기존 성공250/gate도 확인했다.
이는 수식·CPU manager 검증이며 simulation·학습·새 정책 성능은 실행 검증하지 않았다.

## 79. 2026-10-26 — 조기 실패 회피에 대한 일회 Terminal 비용

사용자는 여러 조합의 짧은 학습에서 빠르게 실패 종료하여 Time 비용을 피하는 방향이
학습되는 것 같다고 보고했다. 아래 로그를 읽기 전용으로 확인한 뒤, 실패와 일반 진행·
timeout의 의미 및 보상 차이를 논의했고 성공+250·실패−5·그 외0 적용을 승인했다.

| 실행 | 저장 Time weight | 확인한 마지막 iteration | 평균 episode 길이 | 평균 reward |
|---|---:|---:|---:|---:|
| 2026-10-26_15-51-06 | −2 | 109 | 1.87step | −.0621 |
| 2026-10-26_16-06-39 | −1 | 104 | 2.34step | −.0317 |

두 초기 실험의 확인 구간에서 성공은0이었다. `hand_velocity` 종료 snapshot은 각각약.99962,
.99984로 최근 종료 원인이 팔 속도 초과인 경향이 강했다. 이는 전체 전이의 정확한 종료 확률이
아니며 짧은 실험으로 장기 성능을 단정하지 않는다. 두 번째 실행의 평균 reward는 초기−.6618에서
−.0317로 높아져 episode 단축과 비용 회피 가설을 강하게 뒷받침했다. 저장 reward 전체가 동일한
대조 실험이라고 주장하지 않는다.

핵심은 실패step의 작은 Time 면제보다 true terminal 이후 남은 음의 return이 사라지는데
실패 지급이0이었던 점이다. 현재 Time−1·dt.01·γ.98의 순수 Time 보상열 할인합은 최대약−.5다.
즉시 실패는 이를 피할 수 있고 미래 Motion 비용도 피한다. 마지막step에−.01을 추가하는
수정만으로는 이 이득을 제거하지 못한다. Timeout은 truncation이며 기존 PPO bootstrap을 유지한다.

**구현:** 기존 `terminal_transition`에 실패 분기를 추가했다. 새로운 reward term이나 파일은
추가하지 않았다. `MdpCfg.failure_penalty=5`는 양의 비용 크기이며 기존 유한·비음수 cfg 검증을
따른다. 기존 `state.failure`에는 활성 물체 지지 상실·전도·선반 변위·invalid·singularity와
공식 `hand_velocity`의 결과가 OR되어 있다. 따라서 여러 실패가 겹쳐도−5 한 번만 지급한다.

$$
P_k=(1-\mathrm{paid}_{k-1})
\begin{cases}
-5,&\text{활성 실패},\\
250,&\text{실패 없이 활성 성공 확정},\\
0,&\text{일반 진행 또는 timeout 단독}.
\end{cases}
$$

실패가 성공 및 timeout보다 우선한다. 성공 termination을 꺼도 실패 비용은 독립적으로 남으며,
비활성화한 실패 조건은 기존 mask 규칙대로 비용을 발생시키지 않는다. 기존 paid flag는 성공·
실패 지급 후 켜지고 부분 reset에서 선택 행만 비워 다음 episode에서 다시 평가한다.
RewardManager가 계산하기 전 이미 종료 판정이 확정되므로 별도의 센서·PhysX 조회는 필요 없다.
기존 terminal 함수가 `P_k/dt`를 반환하고 native weight1·dt를 적용하여 실제 일회−5/+250이
유지된다. 예를 들어100Hz raw−500에dt.01이 곱해져−5다. Bool에단순weight−5를주어 실제−.05가
되는 방식은 사용하지 않았다. 첫step 실패로 Time0이어도 Terminal−5를 지급한다.

사용자가 시험 중인 다른 수치는 유지했다. 변경 당시 실제 cfg는 Clearance12/decay2,
Distance3/decay10,Hold.2,Object/Shelf−.5,Action/Joint-velocity각−.03,F/T각−.03,
Singularity.3,Stability1,Time−1,Terminalweight1이었다. 78절의 이전 제안값으로 복구하지 않았다.
현재 README 표와 Time 예시를 이 사용자 설정에 맞췄고, cfg Time 주석의 고정−2 표현만 제거했다.
기존 termination 원인별 로그·Time의 공식is_alive 동작·정책/물리100Hz·모델·관측·PPO는 유지한다.

**검증:** Reward·Termination·NativeVelocityTermination·State·NativeAdapter의 총103개 검사 중
**CPU102개 통과·CUDA전용1개 skip**. 성공/실패 우선·지급 반복0·일반/timeout0·실패비용설정5→7·
성공term비활성시실패·중복원인1회·팔속도/invalid실패·부분reset재지급을 확인했다.
실제 native RewardManager의dt=.01/.02/.1에서 일회 금액이−5/+250인지 검증했다.
실험용 weight의 정확값을 고정하던 cfg 검사는 공식 함수 연결·비용/보상 부호·양의 decay·
Terminalweight1 등 구조를 확인하도록 정리하여 유효한 사용자 조합을 허용한다.

이는 첫 실패 비용 실험값이며 새 학습 효과는 아직 미측정이다. 즉시 속도 종료가 줄었는지에
더해 접촉 후 Clearance 개선·유효 성공이 함께 늘었는지 확인해야 한다. 정지·timeout으로만
바뀌어도 성공으로 보지 않는다. 무동작에서도 초기 속도 종료가 난다면 reset·제어 문제를
별도로 확인해야 한다. Simulation·새 학습은 실행하지 않았다.

## 80. 2026-10-30 — Parallel MLP의 선형 Clearance와 성공 유지 보상

**대상은 `src/contact_manipulation_parallel`만이다.** 다른 PC에서 접근 후 멈추거나 느리게
움직이며 통로 확보를 배우지 못하는 경향이 보고되었다. 해당 PC의 현재 로그를 직접 분석한
결과는 아니며, 아래는 사용자와 논의해 승인한 다음 실험의 코드 변경이다. 원본 프로젝트와
진행 중인 학습은 수정하거나 실행하지 않았다.

Clearance는 초기 Blocker 전체 면적 대신 **초기 Blocker–Target 접근 영역의 교차 면적**
`O_0`을 episode 기준으로 고정한다. 현재 교차 면적을 `O_t`라고 하면 원점수는 다음이다.

```text
C_t = 1 - O_t / O_0
step reward = 12 * dt * C_t
```

초기0, 완전 확보1, 초기의 두 배를 가리면−1이다. 상한 clip·exponential·추가 초기 품질
차감은 제거했다. 개선된 상태를 유지하면 같은 점수를 매 step 지급한다. 기존 보호·지지
gate가 무효이면 양수만 차단하고 음수는 유지한다. 부분 reset은 선택된 환경의 `O_0`만
갱신한다. 유효한 양수 초기 가림이 없으면 Clearance는0으로 처리해 분모 문제로 보상을
만들지 않는다. 기존 `clearance_decay`와 불필요해진 초기 품질 buffer도 제거했다.

사용자는 weight 조정 가능성을 논의한 뒤 첫 비교에서는 기존 **Clearance weight12를 유지**
하기로 했다. 100Hz의 최대 step 보상은0.12다. 직전 논의한 `exp(-2 O_t/O_0)-exp(-2)`의
최대0.8647보다 원점수 최대가 약15.7% 커진다. 실제 이전 코드의 전체 Blocker 면적 분모
지수식과는 초기 가림에 따라 차이가 달라진다. 개선뿐 아니라 악화 구간의 선형 비용도
강해지므로, 실제 항별 지급량·유효 Clearance·접촉 유지·실패율을 보고 추후 scale을 결정한다.

Time weight−1은 유지하고 공식 `is_alive`에 현재 유효 성공 mask를 적용했다. 기존 성공
조건을 현재 만족하는 동안 비용은0이며, 조건이 깨지면 비용을 재개한다. 과거 성공 기록으로
영구 면제하지 않는다. 실제 episode 시간·timeout과 native 실패 종료 step 면제는 유지한다.

성공 보상과 성공 termination은 분리했다. 현재 success 종료는 꺼진 상태를 유지하며,
기존 조건을0.5초 만족하면 기존 설정인+250을 episode당 한 번 지급한다. 성공 이후 실패에도
기존−3을 한 번 지급하도록 `success_paid`와 `failure_paid`를 따로 관리한다. 같은 step은 실패
우선이고, 성공 상실·재획득으로 보너스를 반복 지급하지 않는다. 부분 reset은 선택 행의
두 flag만 초기화한다. Terminal weight1과 native manager의 일회 지급 방식은 유지한다.

할인율은 사용자가 별도로 승인한 `gamma=0.98→0.995`다. 초기 learning rate1e-3와 adaptive
schedule, 다른 weight·종료 조건·정책100Hz·물리·모델·관측은 변경하지 않았다. 수식 비교
그림3개와 parallel README를 업데이트했다. 비교 그림의 일부 선형 계수0.8647은 최대값을
맞추기 위한 시각화용이며, 실제 구현은 계수1의 위 식을 사용한다.

**검증:** Reward·Termination·State·NativeAdapter·ContactReward·SceneMotion의124개 검사 중
CPU123개 통과·CUDA전용1개 생략. 초기/개선/악화·분모 고정·gate·잘못된 초기 겹침에서의
유한 출력·부분reset·성공 후 실패 지급·Time 면제와 재개·native RewardManager dt 계산을
확인했다. 이는 수식과 CPU 연동 검증이며 새 학습의 성능이나 연산 시간 개선을 의미하지 않는다.

## 81. 2026-10-30 — GNN 프로젝트의 Reward·Termination 동기화

사용자 요청에 따라 `src/contact_manipulation`의 reward와 termination을80절의 parallel
프로젝트 현재 설정에 맞췄다. 이번 대상은 GNN 프로젝트이며, parallel의 코드·설정은 유지한다.

Clearance는 초기 Blocker–Target 접근 영역 교차 면적 `O_0`을 고정하고
`C_t = 1 - O_t/O_0`으로 계산한다. Weight12와 native `dt` 적분을 적용하며 초기0·완전 확보1,
초기보다 두 배를 가리면−1이다. 상한 clip·exp·추가 초기 품질 차감은 없다. 기존 Target 보호·
지지 유효성 gate는 양수만 차단하고 음수는 유지한다. 부분 reset은 선택 행의 분모만 갱신한다.

활성 weight는 Clearance12, Distance3/decay10, Contact hold.2, Shelf/Object collision각−.5,
Action rate/Force growth/Torque growth각−.03, Object stability.5, Time−1, Terminal1이다.
Joint velocity·Motion singularity reward와 success termination은 parallel처럼 비활성화했다.
팔 속도3rad/s·지지 상실·전도·선반 변위·invalid·singularity 실패 및 timeout은 유지한다.

성공 기하·안정 조건은 바꾸지 않는다. 기존 조건을0.5초 연속 만족하면 성공 종료를 켜지 않아도
+250을 episode당 한 번 지급한다. 실패 비용은−5에서 parallel의−3으로 맞추고, `success_paid`와
`failure_paid`를 분리해 성공 후 실패에도 비용을 한 번 지급한다. 같은 step은 실패 우선이며
성공 상실·재획득은 성공 보상을 반복 지급하지 않는다. 선택 환경의 두 flag만 reset한다.

Time은 공식 `is_alive`에 현재 유효 성공 mask를 곱한다. 현재 성공을 유지하는 동안 비용0,
조건을 잃으면 비용을 재개한다. 실제 episode 시간·8초 timeout은 계속 진행한다.
GNN·attention, Actor/Critic 관측, action·물리·reset 생성 구조·PPO는 변경하지 않는다.
특히 GNN 프로젝트의 할인율은 기존 `gamma=.98`을 유지하며 parallel의`.995`를 복사하지 않는다.

**검토 범위:** 소스·설정·호출부의 정적 비교만 수행했다. 사용자 요청대로 학습·simulation·
play·테스트를 실행하지 않았으며, 이번 변경에 대한 실행 성공이나 학습 성능 개선은 미검증이다.
80절의 CPU 검사 결과는 이전 parallel 변경의 기록으로만 유지한다.

## 82. 2026-10-30 — GNN 프로젝트의 다른 PC 초기 IK 실패와 이식성 보완

다른 PC에서 원본 `src/contact_manipulation/scripts/train.py` 실행 시 첫 `env.reset()`의
`_build_episode_batch`가 `No collision-free elbow-up pose in the bounded batched IK search`
오류로 중단되었다. 이는 PPO 학습 전의 초기 로봇 자세 준비 실패이며, 이번 로그만으로
GPU 차이·설치 문제·특정 충돌 조건을 직접 원인으로 확정할 수는 없다.

**확인한 누락:** 앞서 다른 PC 실행을 위해 parallel에 넣었던 초기 카탈로그 번들·제한된
TF32 비활성화·실패 전용 eager 재검사·실제 articulation FK 조회 보완이 원본 GNN에는
적용되지 않았다. Reward·termination 동기화와 별개의 초기화 경로 차이였다.
이번에는 해당 보완을 원본에도 독립적으로 반영하고 parallel package를 import하지 않는다.

- 같은 기본 초기 카탈로그를 manifest와 일반 Git 압축 조각4개로 원본 package에 포함했다.
  setuptools package-data에도 등록한다. 입력 지문이 일치하면 PC별 로컬 생성 cache보다
  번들을 우선하고, SHA-256·경로·파일 수·크기를 검사한 뒤
  `outputs/contact_manipulation/parameter_domains/bundled_<archive SHA>/`에 원자적으로 복원한다.
  복원본은 재사용한다. 명시한 cache 경로와 변경된 입력의 기존 생성 경로는 유지한다.
- 초기 로봇 자세 생성·기하 검사의 FP32 행렬 연산은 해당 범위에서만 TF32를 끈다.
  완료 후 기존 학습 설정으로 복원하며, 정책의 전역 정밀도를 변경하지 않는다.
- 첫 장면의 고정 위치27개×방향9개×branch4개에서 정상 탐색이 실패할 때만 같은 후보를
  기존 tensor eager 경로로 재검사한다. CUDA graph·통합 IK/clearance kernel을 우회하지만
  IK iteration48회와 후보 budget은 유지한다. 충돌·elbow-up·forearm·TCP 범위·특이점 조건을
  완화하지 않고 새 배치를 뽑아 우회하지 않는다.
- 실제 관절값 적용 후 공식 `robot.data.body_link_pose_w`로 FK가 갱신된 링크 자세를 읽어
  Float64 기준 FK와 비교한다. 위치0.2mm·회전 행렬 오차0.0003 기준은 유지하고, 불일치 시
  가장 크게 어긋난 링크와 실제/예상 위치·오차를 메시지에 남긴다.

기본 입력 지문은
`b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`,
압축 전체 SHA-256은
`18eba4c7e98625839d03dcc05776ccc3971313c702d3daa32e00e4f71afef3fc`다.
필요하면 `env.initial_catalog_fingerprint=<위 입력 지문>`으로 입력 일치를 강제 확인한다.
번들은 초기 입력을 통일하지만 GPU 계산의 완전한 동일성이나 IK 성공을 보장하지 않는다.

실패 전용 `[Initial IK replay]` 메시지와
`logs/rsl_rl/contact_manipulation/<실행 시각>/initial_robot_start_diagnostics.jsonl`에
eager 재검사의 독립/누적 조건별 통과 수·실제 배치·로봇 형상·joint frame·fingerprint·seed·
라이브러리 버전을 기록한다. 정상 reset에는 이 파일 쓰기를 추가하지 않는다.

다른 PC에는 이 변경을 Git push/pull로 전달한 뒤 기존 명령을 사용한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 1024 --headless
```

**범위와 검증:** GNN·attention·Actor/Critic observation·물체 순서·action·reward·termination은
변경하지 않았다. 사용자 요청대로 학습·simulation·play·테스트를 실행하지 않고 소스와 설정을
정적으로 대조했다. 누락 반영은 확인했지만, 이번 변경이 원격 PC의 실패를 해결했는지는
재실행 결과로 확인해야 한다. 같은 실패가 나면 위 JSONL로 어느 조건이 후보를 배제했는지
구분한다. 이전 parallel CPU 검사 결과를 이번 원본 변경의 실행 검증으로 사용하지 않는다.
