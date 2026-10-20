# Contact Manipulation 학습 기록과 수정 과제

작성·검토: **2026-10-20, Asia/Seoul**. 현재 프로젝트는
`src/contact_manipulation`, task는 `Isaac-Contact-Manipulation-UR5e-v0`다.
연구·개발 결정의 전체 이력은 [CONTEXT.md](CONTEXT.md), 현재 구현은
[프로젝트 README](../src/contact_manipulation/README.md)를 참고한다.

**현재 판단:** 10월19일의 추가 학습·reward 조합5개를 분석하고 저장정책을 고정조건에서 재생했다.
현재 정책들의 Clearance 개선은 거의 없으며, 조합별 생존시간 차이는 관측된다.
공통checkpoint750의320episode는 성공·통로확보0회, 과부하/센서범위초과278회였다.
장기실행은 Gaussian std뿐 아니라 deterministic 평균 명령도 비정상적으로 커졌다.
상세 비교·실제평가·한계는 **10절**에 둔다. 해당 분석에서는 새 학습이나 production수정을
하지 않았다. 이후 손바닥 EEF action 변경은 **11절**, 물체 종료 조건 완화는 **12절**,
현재 UR5e 말단 제어·관측 통일과 중력 보상은 **13절**에 기록한다.

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
- 현재 공유 장면의 400-step 경계에서도 전체 환경을 truncation한다. 따라서 `time_out`에는
  **개별 episode가 8초를 모두 사용하지 않은 공통 장면 종료**도 포함된다.
  이 50 Hz 기준을 과거 30 Hz 실행에 소급 적용하지 않는다.
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
