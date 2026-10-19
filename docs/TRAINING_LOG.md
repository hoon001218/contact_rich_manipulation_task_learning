# Contact Manipulation 학습 기록과 수정 과제

작성·검토: **2026-10-20, Asia/Seoul**. 현재 프로젝트는
`src/contact_manipulation`, task는 `Isaac-Contact-Manipulation-UR5e-v0`다.
연구·개발 결정의 전체 이력은 [CONTEXT.md](CONTEXT.md), 현재 구현은
[프로젝트 README](../src/contact_manipulation/README.md)를 참고한다.

**현재 판단:** 10월19일의 추가 학습·reward 조합5개를 분석하고 저장정책을 고정조건에서 재생했다.
현재 정책들의 Clearance 개선은 거의 없으며, 조합별 생존시간 차이는 관측된다.
공통checkpoint750의320episode는 성공·통로확보0회, 과부하/센서범위초과278회였다.
장기실행은 Gaussian std뿐 아니라 deterministic 평균 명령도 비정상적으로 커졌다.
상세 비교·실제평가·한계는 **10절**에 둔다. 새 학습이나 production수정은 하지 않았다.

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
