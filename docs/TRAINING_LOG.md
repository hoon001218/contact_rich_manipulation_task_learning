# Contact Manipulation 학습 기록과 수정 과제

최종 검토: **2026-11-07, Asia/Seoul**. 현재 메인 프로젝트는
`src/contact_manipulation_symmetric`, task는
`Isaac-Contact-Manipulation-Symmetric-UR5e-v0`다. 원본 `contact_manipulation`과
별도 MLP `contact_manipulation_parallel`의 기록은 실행별 프로젝트를 구분하여 보존한다.
연구·개발 결정의 전체 이력은 [CONTEXT.md](CONTEXT.md), 현재 구현은
[Symmetric README](../src/contact_manipulation_symmetric/README.md)를 참고한다.

**최신 오류 분석(2026-11-07):** 원본 `2026-11-06_13-53-02` 실행의 CUDA719와
동시각 커널 **Xid32**는 **112절**에 기록했다. 같은 Kit 로그의 선행 broadphase
`found null elements` 오류도 보존한다. 최근 Xid69와 구분하며 근본 원인은 미확정이다.
현재 메인은 Symmetric이고 공통 물리 경로의 안정성 위험은 계속 추적한다.

**재발 시 참고(2026-11-05):** `2026-11-04_19-52-40` 실행의 CUDA 실패와 동시각 NVIDIA
`Xid 32` 확인 기록은 **99절**을 따른다. 근본 원인은 미확정이며 사용자 요청으로 해결·재현은
보류했다. 이후 유사 오류를 조사할 때 이 실행과 비교하되 과거 Hand 속도 발산과 동일시하지 않는다.

**보상 후보 속도 실측(2026-11-05):** 동적 Blocker goal 지도 갱신의 독립 GPU 측정은 **100절**을
참고한다. 검증용 코드와 결과만 추가했으며 학습 환경·reward에는 아직 반영하지 않았다.

**새 보상 적용(2026-11-05):** 원본 `contact_manipulation`에 초기 빈 재배치 영역의 중심거리
보상을 추가하고, 손–Blocker 거리는 episode 최소값에서 현재값으로 복원했다. 수식·GPU 비용·
검증 범위는 **101절**을 따른다. 100절의 도달 가능 자세 탐색 보상은 계속 보류한다.

**GPU 오류 추가 기록(2026-11-05 21:51:57 KST):** 새 실행의 Vulkan page fault와 커널
**Xid31**은 **102절**에 기록했다. 99절의 Xid32와 오류 종류가 다르며 원인은 미확정이다.

**후속 CUDA 오류(2026-11-05 22:44:51 KST):** **103절**의 Xid69는10월25일과
Class·Offset·Data·ErrorCode가 같다. PPO loss 출력 위치를 최초 실패 연산으로 단정하지 않는다.

**동일 Xid69 재발(2026-11-06 02:27:56 KST):** **104절**은 OSC 관절 조회에서 보고된
실패다. 저장 설정은103절과 동일하며 compile_model=true가 유지됐다. 이번에는2298회가 기록됐다.

**컴파일 경로 분리·재검증(2026-11-06):** **105절**에 모델 컴파일 기본값 변경, 단계별 환경
CUDA graph 스위치, 1,024환경 검사와 Compute Sanitizer 결과를 기록했다. 장기 안정성 대조를
시작했으며, Xid69의 근본 원인을 해결했다고 판단한 상태는 아니다.

**2026-10-26 당시 적용:** 기존 Terminal에 일회 실패−5를 추가했다. 성공+250·일반/timeout0이며,
당시 사용자 실험의 다른 weight는 유지했다. 현재 수치로 사용하지 않으며 변경·근거·검증은 **79절**에 기록한다.
직전 reward 조정 이력은 **78절**, GUI tactile/F/T 모니터는 **77절**, 저장 실행 분석은
**76절**을 따른다.

**이전 물리·관측 적용:** Patch 용량262,144·관측64×48과 GUI 관측 OBB의 변경·검증은
**48절**에 기록한다. 아래 과거 학습 기록은 당시 설정이며 현재 값은 Symmetric README와 최신 실행 cfg를 따른다.

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

## 83. 2026-10-31 — Parallel 다음 학습 최종 명세 반영

사용자가 [다음 학습 최종 명세 — Codex 구현 기준](https://app.notion.com/p/40a4968094ee813ba809e044a2db975a)의
구현을 요청했다. 페이지 최종 수정 시각은 2026-10-31 09:16:51 UTC이며, 작업 시작 HEAD는 명세의
기준과 같은 `563e3760728049879e4451e5b713faa5cb03c0cf`, 작업트리는 깨끗했다.
강제 checkout은 하지 않았다. 수정 대상은 `src/contact_manipulation_parallel`과 이 기록뿐이다.

### 실제 코드 변경

| 파일·키 | 이전 | 다음 학습 |
|---|---:|---:|
| `config/ur5e/agents/rsl_rl_ppo_cfg.py` · `algorithm.gamma` | 0.995 | 0.999 |
| `shelf_manipulation_env_cfg.py` · `motion_action_rate.weight` | −0.03 | −0.003 |
| 같은 파일 · `motion_force_growth.weight` | −0.03 | −0.01 |
| 같은 파일 · `motion_torque_growth.weight` | −0.03 | −0.01 |
| 같은 파일 · `TerminationsCfg.success` | 비활성 | 기존 `mdp.success` 활성 |
| `mdp/events.py` · `MdpCfg.failure_penalty` | 3 | 10 |

Production 변경은 위 세 파일의 설정과 비용 주석뿐이다. `rewards.py`, `terminations.py`,
`sensors.py`, `observations.py`, `scripts/train.py`는 수정하지 않았다. 초기 카탈로그·배치 생성·
로봇 시작 영역·IK·물리 설정과 Original `contact_manipulation`, Isaac Lab 및 RSL-RL도 유지했다.
Parallel은 자체 상대 import와 자체 cfg를 사용한다. Original과 알고리즘 AST를 같게 강제하던
`tests/test_learning.py` 검사는 Parallel의 최종 기대값을 명시적으로 검사하도록 교체했다.

기존 단일 MLP actor/critic은 각각 256→128·ReLU·독립 파라미터다. 관측은 Actor의 초기 장면과
로봇/센서 현재·직전 sample, Critic의 최초 인식 ID에 대한 현재 GT 정보를 유지한다.
100Hz·8초 truncation·rollout36·epoch8·minibatch4·LR0.001/adaptive·KL0.02·λ0.95·clip0.2·
entropy0.005·max-grad-norm1을 유지하며 curriculum·새 Target 비용·정규화·cap은 추가하지 않았다.

성공 조건·gate·50회 연속 카운터·기존 이벤트 latch를 재작성하지 않고 native success term만
연결했다. 실패가 없고 기존 조건을 연속 0.5초 만족하는 첫 transition에 +250을 한 번 지급한 뒤
non-timeout 종료한다. 실패는 −10이며 동시 성공보다 우선한다. 순수 timeout은 실패 비용 없이
기존 truncation/bootstrap을 따른다. `terminal_reward / dt`, terminal weight1, manager의 dt 곱은
유지하여 실제 이벤트가 +250/−10이 된다. 기존 alive/success Time mask도 유지했다.

### 검증 결과

CUDA를 숨기고 CPU thread를 1로 제한하여 기존 사용자 프로젝트 테스트를 실행했다.
`test_learning.NativeMlpTests` 6개 통과, 아래 MDP 12개 class의 159개 중 156개 통과·CUDA 전용
3개 생략으로 **총 165개 중 162개 통과·3개 생략**이다. 초기 검증에서 확인된 혼합 batch의
mock manager 누락과 비활성 velocity reward에 대한 오래된 테스트 가정은 테스트에서 수정했다.
이 때문에 production 동작이나 native 라이브러리를 수정하지 않았다.

MDP 실행 class는 `RewardTests`, `TerminationTests`, `StateTests`, `NativeAdapterTests`,
`ContactRewardTests`, `BlockerContactManagerTests`, `NativeVelocityTerminationTests`,
`ObjectSupportValidityTests`, `SceneMotionRewardTests`, `ImageOrderedObservationTests`,
`DynamicObservationTests`, `WrenchTests`다. CPU 실행에는 `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`과 `env_isaaclab/bin/python -B -m unittest`를 사용했다.

| 명세 acceptance 항목 | 결과·근거 |
|---|---|
| 최종 설정 | 통과 — Parallel 전용 PPO 기대값과 모든 활성 reward weight·decay·성공 term 확인 |
| dt·단위 | 통과 — 실제 native RewardManager 산술에서 C/D/H=1이면 +0.12/+0.03/+0.002; raw action/F/T cost=1이면 −0.00003/−0.0001/−0.0001; 이벤트 +250/−10 |
| Clearance 악화·분모 | 통과 — A=1.2A0에서 −0.2, 무효 gate에서도 음수 유지, finite 초기 threshold·초기 active Blocker overlap·분모 고정 |
| 기존 gate | 통과 — Target 보호 및 support/retention gate, 이동 중 진행 허용·속도는 성공 정착에만 사용 |
| Palmar hold | 통과(CPU) — 기존 canonical 17 pad·Blocker 필터·동일 pad의 연속 접촉 합, 단일 sample/교대 pad 제외·동일 step 중복 갱신 방지 |
| 연속 성공 50회 | 통과 — 49회 무지급, 조건 해제 시 0, 새 50회에 최초 +250·success; 기존 pose/support/velocity 조건 유지 |
| 혼합 vector batch | 통과 — 한 transition의 진행/성공/실패/순수 timeout 및 원인 mask·일회 이벤트·Time·Clearance 지급 확인 |
| F/T growth | 통과 — 일정 norm의 방향 변화·감소는 0, 양의 norm 증가분만 제곱·매 step 이전 wrench 갱신 |
| Reset 격리 | 통과 — 선택 행의 stable/latch/wrench/contact history만 초기화, 다른 행 보존 |
| 이벤트 배타성 | 통과 — 성공·실패 동시 −10만, 이후 반복 지급 없음·성공 보상과 종료 연결 중복 없음 |
| ARM6 과속 | 통과 — interleaved 관절 ID에서도 팔 abs(qdot)>3만 종료, Hand·mimic 제외·경계 3 허용 |
| 보존 범위 | 통과(정적 diff 및 관련 회귀 검사) — Original·기본 PPO/래퍼·reset 분포·관측·함수·hard failure predicate 불변 |

생략된 세 검사는 `StateTests`의 support replay cache와 `ObjectSupportValidityTests`의
collision-support reduction/CUDA graph다. 실제 PhysX의 손바닥 접촉, GPU 동작·처리량,
학습 수렴·clearance/성공률 개선은 **미실행·미검증**이다. Canonical pad 검출은 접촉력의
방향이나 F/T 안전성을 증명하는 것으로 해석하지 않는다. 학습·simulation·play·commit·push 및
Notion 페이지 쓰기는 실행하지 않았다. −10 이벤트만으로 모든 조기 실패 유인이 제거되거나,
γ 변경만으로 빠른 조작이 학습된다고 판단하지 않는다.

## 84. 2026-11-01 — Actor 비전의 10step 실제 ray 갱신

사용자가 `contact_manipulation_parallel`의 Actor 비전을 10step마다 갱신하도록 요청했다.
10월31일 명세의 초기 비전 고정 방식에서 변경한 별도 후속 요청이다. 이번 변경은 parallel에만
적용하며 Original GNN·reward·termination·PPO·제어·물리·reset 분포는 수정하지 않았다.

**관측:** `actor_vision_update_interval=10`을 기본값으로 추가했다. 현재100Hz에서 비전은10Hz다.
초기 reset에는 기존 초기 ray snapshot을 즉시 넣고, 이후 현재 물체의 실제 visual mesh에
ray casting하여 보이는 first-hit 표면에서 OBB를 다시 추정한다. GT pose는 센서의 표면 위치를
놓는 데만 사용하고, 보이지 않는 물체의 실제 중심·크기로 Actor 슬롯을 채우지 않는다.
새로 보이는 물체를 넣고 가려진 물체는 zero padding한다. Target·Blocker는0·1번이고 나머지는
보이는 역할 물체와의 거리로 선택한 뒤 현재 ray 이미지의 좌→우 순서로 정렬한다.

Episode 초기의 인접 swap 패턴은 새 RNG 없이 재사용한다. 두 슬롯이 현재 모두 유효할 때만
교환하여 padding으로 물체가 밀려나지 않게 한다. 중간9step에는 최근 ray snapshot의 base-frame
좌표를 고정하고 현재 EEF frame으로 재표현한다. 로봇·F/T·tactile의 현재/직전144D는 매step 유지하고
전체 관측은244D다. GUI의 observed-object OBB도 같은 최신 Actor cache를 읽는다.

**Critic:** 최초 인식 ID와 slot을 유지하여 현재 GT를 갱신하는 기존 계약을 보존했다. Actor의
가시성 변경이 Critic에 섞이지 않도록 `critic_geometry_valid`를 reset 때 별도로 저장한다.
Critic의 장면·공유 current/previous 접근만으로는 Actor ray를 실행하지 않는다.

**연산 구조:** env ID를10개 phase로 나눈 고정 GPU index bucket을 초기화 때 한 번 만든다.
1024환경은 매step102개 또는103개씩 처리하고 각 환경은10step 간격으로 갱신한다. 초기 snapshot
이후 첫 정기 갱신은 해당 phase에 따라1–10step 안에 온다. 부분 reset도 같은 phase를 유지한다.
이는 매10step1024개를 동시에 처리하는 방식과 총 ray 수는 같고, 순간 부하를 분산하는 방식이다.

- 기존64×48 observation grid와 shared mesh BVH를 재사용하고 view32개씩 처리한다.
  정기 query는 step당약31.3–31.6만 image ray다. 실루엣 분모·가시성 합격 검사·시점 탐색·재시도는
  실행하지 않는다. 물체만 기존 규칙대로 검사하며 새 카메라 prim/render product는 만들지 않는다.
- `InitialVisionSensor.observe_tensors`가 environment/active/role/camera tensor를 직접 받고
  OBB·valid·pool ID tensor를 반환한다. CPU 요청 dict·환경별 Snapshot list·진단 파일·이미지 저장이
  hot path에 없으며, 새 scheduling의 GPU `nonzero()`·tensor scalar 조회·CPU 복사도 없다.
  Python 반복은 초기 phase bucket 구성과 고정 크기 ray chunk launch에만 있다.
- 같은 native counter의 반복 관측은 ray를 중복 실행하지 않는다. 새 reset 행과 recorder의
  pre-reset 종료 행은 active ID를−1로 마스킹해 실제 물체 ray query를 차단하고 해당 cache를 유지한다.
  고정 batch 모양을 유지하므로 미갱신 행을 찾기 위한 GPU→CPU 동기화를 추가하지 않는다.
- 갱신 후 Actor packed cache는 새 tensor로 만든다. 기존 `extras.terminal_observation`과
  reset하지 않은 환경의 history·Critic ID·mask는 수정하지 않는다. 최신 물체 pose 조회는 기존
  native asset cache를 재사용한다.

설정값0은 기존 reset-only Actor 비전이고 양의 정수는 policy step 간격이다. 기존 수치 단독
호출의 `TaskState`는 기본값0을 유지하며 실제 환경 cfg의 기본값만10이다. 카메라 pose는 해당
episode 초기 world pose를 재사용하고, ray 결과는 robot-base에 저장한 뒤 현재 EEF로 변환한다.

**검증:** `CUDA_VISIBLE_DEVICES=''` 및 CPU thread1에서 관련13개 test class의188개 중
**185개 통과·CUDA 전용3개 생략**. 새 검사는 실제 Warp CPU 메시 ray와 기존 fixed-view API의
결과 일치, world/root 좌표 및 unsorted env ID,32+3 view chunk·64×48 image-only query,
가려짐/재노출·GT 미보충·입력 pose 불변을 확인한다. Scheduling은1024행의 균등 phase와 각10step
간격·중복 호출·동기화 없는 tensor 처리·reset/종료 행 마스크를 검증했다. 별도 Critic mask/ID,
현재 EEF 변환, 기존 terminal 관측 보존, 부분 reset·stored swap/padding도 확인했다.

관련 class는 `PeriodicActorVisionTests`, `GeometricVisionTests`, `ImageOrderedObservationTests`,
`InitialGeometryTests`, `DynamicObservationTests`, `StateTests`, `NativeAdapterTests`,
`ObservationObbVisualizationTests`, `PhysicalStateCapacityTests`, `ObservationCapacityTests`,
`RewardTests`, `TerminationTests`, `BlockerContactManagerTests`다. 생략된3개는 GPU ray parity·
visible-point CUDA reduction·support replay cache다. 실제 GPU 처리 시간·메모리 증가·학습 효과는
**미측정**이며, phase 분산만으로 평균 연산량이 줄었다고 해석하지 않는다. 학습·PhysX simulation·
play·commit·push는 실행하지 않았다. `git diff --check`와 보존 경로의 diff도 확인했다.

## 85. 2026-11-01 — GNN Actor에도 10step 비전 갱신 적용

사용자 후속 요청으로 Original `contact_manipulation`에도 84절의 주기적 실제 ray 관측을
적용했다. 기본 `actor_vision_update_interval=10`이며 현재 정책100Hz에서 비전은10Hz다.
0으로 설정하면 기존 reset-only Actor 비전으로 돌아간다. Parallel 프로젝트와 Original의
reward·termination·PPO·제어·물리·reset 분포는 이번에 수정하지 않았다.

초기 reset snapshot은 즉시 제공한다. 이후 고정 env-ID phase10개로 ray 부하를 분산하여
1024환경 중 매step102개 또는103개를 갱신한다. 첫 갱신은 reset 후1–10step이며 이후 간격은
10step이다. 해당 episode의 가상 카메라 pose,64×48 observation grid,shared mesh BVH와
32view chunk를 재사용한다. 현재 물체의 보이는 first-hit 표면에서만 OBB를 추정하고,
가려진 물체는 padding한다. 시점 재탐색·가시성 합격 검사·실루엣 분모 계산·카메라 prim·렌더링은
추가하지 않았다. 중간 step은 최근 snapshot을 현재 UR5e 말단 EEF frame으로 재표현한다.

Original GNN의 순서 규칙을 유지했다. Target·Blocker는0·1번이며 나머지는 보이는 역할 물체와
가까운 순서로 선택한다. Parallel MLP의 이미지 좌→우 정렬·인접 swap은 가져오지 않았다.
기존 GNN·Attention은 매 forward 현재 관측에서 mask와 scene feature를 계산하므로 모델의
실행 구조는 바꾸지 않았다. 기존 compile cache는 실행 함수만 보관하며 장면 feature를 고정하지 않는다.

Critic은 별도 reset-time valid mask를 사용하여 최초 인식 ID·slot만 유지한 채 현재 GT pose를
갱신한다. Actor에 새 물체가 등장하거나 기존 물체가 가려져도 Critic의 목록은 바뀌지 않는다.
Critic/robot history 조회만으로는 ray를 실행하지 않는다. 같은 step의 중복 관측,
reset 직후 행과 종료 직후 행은 중복 ray를 막고 기존 terminal 관측 tensor를 보존한다.
주기 query는 GPU tensor를 직접 주고받으며 CPU 요청 dict·환경별 snapshot list·GPU nonzero나
tensor scalar 조회를 추가하지 않는다. phase bucket 초기 구성과 고정 ray chunk launch는 Python에서 수행한다.

**검증:** `CUDA_VISIBLE_DEVICES=''`,CPU thread1에서 관련14개 test class의241개 중
**236개 통과·CUDA 전용5개 생략**(8.373초). 새6개 검사는1024환경 phase/cadence·중복 호출 방지,
실제 Warp CPU 메시 ray와 기존 fixed-view API 일치,32+3view chunk,world/root 좌표와 입력 불변,
가려짐·재노출·GT 미보충,변경된 GNN 입력 mask와 고정 Critic mask,부분 reset 및 terminal 보존을 확인했다.
기존 GNN/Attention·native Actor/Critic·관측·보상·종료 회귀 검사도 통과했다.

검사 class는 `PeriodicActorVisionTests`, `GeometricVisionTests`, `InitialGeometryTests`,
`DynamicObservationTests`, `StateTests`, `NativeAdapterTests`, `ObservationObbVisualizationTests`,
`PhysicalStateCapacityTests`, `ObservationCapacityTests`, `RewardTests`, `TerminationTests`,
`BlockerContactManagerTests`, `ModelTests`, `NativeLearningTests`다. 생략 항목은 GPU ray parity,
visible-point CUDA reduction,support replay cache,fused edge reduction,CUDA Inductor 통합이다.
실제 GPU 처리 시간·메모리·학습 효과는 **미측정**이다. 학습·PhysX simulation·play·commit·push는
실행하지 않았다.

## 86. 2026-11-02 — GNN Reward·Termination 최종 명세 적용

사용자 요청으로 [다음 학습 수정 명세 — Reward·Termination 최종 설정 (2026-11-02)](https://app.notion.com/p/40c4968094ee81b4aa3bf980ebbf03d3)를 읽고
`contact_manipulation`에 확정된 A–F절을 적용했다. 페이지의 접힌 과거 검토·미확정 후보보다
최종 구현 지침을 우선한다. 원격 문서는 수정하지 않았다.

**기준 확인:** 작업 시작 HEAD는 명세의 검토 commit과 같은
`b768449df278b84ff21782f3aae7c6474ae33ac4`였고 작업트리는 clean이었다.
대상 task는 `Isaac-Contact-Manipulation-UR5e-v0`이다. 로컬의 최근 의미 있는 Original run은
`logs/rsl_rl/contact_manipulation/2026-10-26_17-04-28`이며 `params/env.yaml`, `params/agent.yaml`과
최종 `model_32750.pt`의 존재를 확인했다. 저장 run은 joint-velocity·singularity reward 활성,
stability weight1,실패−5,success termination 활성 및 reset-only Actor 비전인 과거 설정이다.
현재 source의 후속 변경을 이 YAML로 되돌리지 않았다. Action rate·F/T growth 각−.03,
Distance weight3/decay10,PPO γ=.98,100Hz/8초,Actor10step ray 갱신,현재 GNN·관측·OSC·spawn 분포를
유지했다. Parallel package와 외부 라이브러리도 변경하지 않았다.

| 항목 | 변경 전 | 이번 고정 설정 |
|---|---|---|
| Clearance 지급 | Signed 점수,양수에 support·Target·failure gate | `max(0,1−At/A0)`,상태 gate 없음,weight12 |
| Contact | 같은 pad 연속 접촉 count,weight.2 | Raw=`held+.5·new`,weight.4; 새 접촉.2/s·유지.4/s |
| Target 비용 | 일반 물체 코너 속력 비용에 포함 | 기존 Shelf 함수·네 threshold 그대로,별도 weight−.5 |
| 일반 물체 비용 | Blocker 제외 | 기존 Blocker 제외를 유지하고 Target 추가 제외 |
| Stability dense reward | 모든 활성 물체의 worst 지지 거리+tilt | Blocker tilt만,5° 여유 후 기존 β,weight.5 |
| Success | 물체 정착 속도·.5초 연속,종료 비활성 | 현재 기하 유효성을 충족한 첫 step +250,success 종료 활성 |
| Failure | −3 일회 | −10 일회 |
| ARM6 과속 종료 | 실제 abs(qdot)>3rad/s | 실제 abs(qdot)>1rad/s,Hand 제외 |

Clearance의 초기 면적과 접근 영역은 reset 때만 고정하며 signed raw는 진단용으로 유지한다.
초기보다 더 가리면 지급0,초기보다 덜 가리면 개선된 상태에 매step 반복 지급한다. 유효 면적이면
실패 transition에도 지급하며 실패 여부·Target 이탈·다른 물체 지지는 Clearance 지급 mask가 아니다.
Area helper의 finite·초기 면적>1e−12·현재 면적≥0 검사는 유지했다. 원래 `safe_corners`로
비정상 Blocker 좌표가 유효한 면적0처럼 보이는 경우도 면적 입력 자체의 유효성으로 차단한다.
다른 물체의 invalid 상태로 유효한 Blocker 면적 보상을 차단하지 않는다.

Target 항은 `shelf_motion_cost`를 재사용한다. 초기·현재 pose 모두 native WORLD body-link
origin/wxyz이며 velocity는 Shelf와 같은 native COM twist다. Reset에서 실제 pool ID로 선택한
초기 Target pose를 기록하고 부분 reset에서는 선택 행만 갱신한다. 병진5mm·회전1°·선속력.005m/s·
각속력.05rad/s 중 하나라도 strict 초과하면 raw1,여러 개를 위반해도1이다. 이동 후 정지해도
초기 pose 오차가 남으면 비용이 지속된다. 일반 물체 비용은 기존 평균8개 OBB 코너 속력의 합이며
cached reset/advance 경로 모두 Target·Blocker를 제외한다. 다른 물체의 active mask는 바꾸지 않는다.

Contact는 같은 snapshot에서 pad별 `held=current&previous`, `new=current&~previous`를 계산해
두 count를 cache한 다음 previous를 한 번 갱신한다.17개 canonical pad·Blocker-only force·.01N·
validity와 Target gate는 유지한다. Gate가 닫혀도 history는 갱신하며 reset/Blocker 교체는 기존
이력과 새 count를 함께 초기화한다. Native RewardManager가 weight와 dt를 한 번 곱한다.

Stability는 `expm1(-β·max(θB−5°,0))`이며 β=ln2/(π/4)를 유지한다.
Blocker 실제50°는 초과45°로 raw−.5,weight.5·dt.01 적용 시−.0025/step이다.
일반 물체·Target의 tilt와 dense 지지 거리 비용만 제외했다. 모든 활성 물체의 raw support/tilt,
낙하·전도 timer,물리 실패 및 성공 순간의 기하 유효성은 보존했다.

Success는 지정 Blocker overlap≤1e−5m²,Target 초기 OBB 중심 이동≤2cm·회전≤.15rad,
모든 활성 물체의 전체 OBB XY가 panel±5mm 이내·collision bottom gap[−5,+5]mm·tilt<80° 및
finite·활성 실패 없음이 필요하다. 물체 속도의 finite 검사는 남기고 정착 속도 상한만 제거했다.
`success_duration=0`을 허용하며 첫 candidate의 갱신 count1로 같은 step에 일회+250을 지급한다.
물리 failure와 ARM 과속은 성공보다 우선하고−10을 한 번 지급한다. 순수 timeout은 기존 truncation이며
실패로 바꾸지 않는다. Terminal만 기존 `/step_dt` 보정을 유지한다.

별도 속도 warning·L2·중복 과속 penalty·방향성 Distance·curriculum은 추가하지 않았다.
Spawn 수·손 시작 분포·weight schedule·외부 PPO/GAE/timeout bootstrap·네트워크는 변경하지 않았다.

**검증:** `CUDA_VISIBLE_DEVICES=''`,CPU thread1에서 관련24개 class의331개 중
**317개 통과·CUDA/AppLauncher 전용14개 생략**했다. 보상·상태·종료 집중150개는147개 통과·3개 생략,
추가 reset·센서·vision 회귀119개는 최종110개 통과·9개 생략,기존 모델/native 학습62개는60개 통과·
2개 생략이다. 추가 회귀에서 발견한 기존 COM 테스트의 Target 이동 비용 기대값은 일반 물체
회전 사례로 바꾸어 COM offset·코너 속력 검증 목적을 유지했고,수정한 `PartialResetBatchTests`
7개를 다시 실행해6개 통과·CUDA1개 생략을 확인했다.

- Native RewardManager 산술: 면적 비율1.2/1/.8/.5/0의 실제 Clearance0/0/.024/.060/.120;
  상태 gate 독립성,비정상 면적0,유효 Clearance+실패−10의 합−9.88,부분 reset 기준 격리.
- Target의 네 strict 경계·복수 위반 시 raw1·dt−.005,Shelf 함수 재사용,off-center OBB와
  world link origin 구분,환경별 role/pool ID와 부분 reset 기준,실제 cached 일반 물체 비용에서 Target 제외.
- 최초 유효 step의+250·성공 일회 지급,물체 정착 속도 제거,유지 시간0,즉시 물리 유효성은
  debounce 전부터 요구,과속·물리 실패 우선,순수 timeout 구분과 선택 행 latch 초기화.
- ARM6의±1rad/s 경계와±1.01rad/s 초과,Hand-only 초과 제외,해석된 native joint 선택,
  속도 실패의 terminal 연결 및 현재 GNN/PPO 설정 보존.
- Contact 새/유지 배타적 pad 합,0→1→1→0의.002/.004/0,새2·유지3개의.016,
  17pad·교대 pad·invalid·반복 호출·Target gate 중 이력·부분 reset·Blocker 교체.
- Blocker0/5/8/50°와yaw-only,거리·다른 물체 tilt 무관성,기존 모든 물체 실패와 raw support 보존.
  독립 Critic·GNN·Attention,10step 실제 메시 관측·sensor/filter·reset 회귀도 통과했다.

수정 파일은 기존 `mdp/rewards.py`, `mdp/events.py`, `mdp/sensors.py`, `mdp/terminations.py`,
`shelf_manipulation_env_cfg.py`, `tests/test_mdp.py`, README와 이 기록이다. 새 코드 파일은 없다.
`git diff --check`와 Parallel·관측·model/PPO·UR5e/scene·scripts·IsaacLab 보존 경로의 diff를 확인했다.
실제 GPU/PhysX·학습·play·commit·push는 실행하지 않았다. 이번 CPU 검증은 동작 계약의 검증이며,
학습 개선·성공률·iteration 시간과 실물 운동을 검증한 결과는 아니다.

## 87. 2026-11-02 — Time penalty를 학습 중간부터 활성화

후속 사용자 요청으로 `contact_manipulation`의 Time weight curriculum을 추가했다.
86절의 curriculum 없음 결정은 이 후속 요청으로 Time 항에 한해 변경된다.
시작 HEAD는 `73bb36e`였으며 사용자가 이미 환경 cfg를 수정한 상태였다. Distance5,
joint-velocity reward 활성,F/T growth 비활성,Time weight−.3을 그대로 보존했다.
PPO의 γ=.95·entropy=.01 등 현재 실험 설정도 변경하지 않았다.

`env.time_penalty_start_fraction=0.5`를 기본값으로 두었다. 학습 entrypoint에서 CLI·Hydra 적용 후
실제 `max_iterations`의 비율을 올림해 시작 iteration을 정하고 `num_steps_per_env`를 곱한다.
기본90,000×.5=45,000 iteration,rollout36이면1,620,000 control step이다. 환경 수1024는 곱하지 않는다.
전반부는 weight0,후반부는 launch 시 `RewardsCfg.time_penalty.weight` 값(현재−.3)을 적용한다.
후반부100Hz 일반 step의 비용은−.003이며 기존 성공·실패 step 면제와 timeout 처리는 유지한다.

공식 `CurriculumTermCfg(func=modify_reward_weight)`와 `CurriculumManager`를 사용했다.
Native 조건이 `common_step_counter > num_steps`여서 설정값은 기준 step−1이다. Curriculum은
reset 때 실행되므로 기준에 도달한 뒤 첫 reset에서 전체 환경의 weight를 바꾸고 다음 reward부터
적용한다. 부분 reset 하나로도 전역 계수가 전환되며 episode 진행률 curriculum은 아니다.
점진 ramp는 없다. Native RewardManager는 weight0인 동안 함수 계산을 건너뛴다.
환경 step/reset override,새 module,추가 per-step tensor·host synchronization은 없다.

재개 학습은 checkpoint의0-based iteration label+1을 진행량으로 삼아 남은 threshold를 줄인다.
예를 들어 model34999는35,000회 진행으로 계산하여10,000×36=360,000 local step이 남고,
model44999 이후라면 실제 첫 학습 step 전에 native manager를 한 번 계산해 활성화한다.
`common_step_counter`나 MDP·scene·vision 시계는 바꾸지 않는다. 기준은 현재 지정한
`max_iterations`에서 계산하므로 이 값을 바꾸면 기준도 바뀐다. Native runner의 resume 시
추가 iteration 실행 의미는 수정하지 않았다.

시작 비율0은 즉시 활성화,null은 curriculum 없이 기존 계수를 처음부터 사용한다.
Time reward 자체가null이면 schedule도 생성하지 않는다. 목표 weight는 사용자 launch override를
포함해 읽으며 `params/env.yaml`에 초반 reward weight0과 curriculum의 목표 weight·남은 기준을
함께 기록한다. Native `Curriculum/time_penalty` 로그로 현재0/−.3을 확인할 수 있다.
Standalone 환경·Play는 학습 schedule을 준비하지 않고 설정 계수를 사용한다. 이미 실행 중인
학습 프로세스는 변경되지 않으며 다음 실행 또는 checkpoint 재개부터 적용된다.

**검증:** CPU 전용 curriculum9개가 통과했다. 실제 checkout의 native curriculum·RewardManager
코드를 사용해0인 동안 함수 생략,strict/reset 경계·전역 전환,dt 및 성공·실패 면제,
CLI/Hydra budget·분수 올림·환경 수 독립성,재개 offset·중간 이후 즉시 활성화,
환경 시계 불변·설정 유효성·기존 native main 실행 순서를 확인했다.
기존 model·native runner 검사까지71개를 확인해 최종 **69개 통과·CUDA 전용2개 생략**이다.
처음 발견한 기존 예제 비교 테스트의 γ=.98·entropy=.005 기대값은 사용자 현재 설정과 달라,
해당 두 기대값만 .95/.01로 명시하고 나머지 예제 비교를 유지했다. 수정한 그 검사는 재실행해 통과했다.
실제 PPO 설정을 예전 값으로 복구하지 않았다.

기존 환경 cfg·train entrypoint·테스트·README와 이 기록만 수정했고 새 파일은 추가하지 않았다.
`git diff --check` 및 Parallel·reward 함수·관측·model/PPO·UR5e/scene·외부 라이브러리의 보존을
정적 diff로 확인했다. 실제 프로젝트 학습·GPU/PhysX simulation·play·commit·push는 실행하지 않았다.

## 88. 2026-11-04 — 하나의 Action-rate 함수에서 Stiffness 분리

사용자 요청으로 Original `contact_manipulation`의 `motion_action_rate` 한 함수 안에서
pose·hand raw 변화와 stiffness 변화를 분리했다. 별도 reward term·코드 파일·controller는 추가하지 않았다.
시작 HEAD는 `6e93059`였으며 기존 미커밋 변경인 팔 과속 threshold3rad/s를 보존했다.

기존 공식 `action_rate_l2`는18D 전체 raw 차이를 합산하고 채널 선택 파라미터가 없다.
따라서 이 연구 reward에 필요한 선택·가중만 로컬 tensor 연산으로 구현하고 공식 함수를 수정하지 않았다.
단일 함수의 입력은 `motion_action_rate(env, stiffness_weight=0.0)`이다.

```
cost = ||Δraw_pose[0:6]||² + ||Δraw_hand[12:18]||²
       + stiffness_weight * ||(Kp_now - Kp_prev) / (Kmax - Kmin)||²
Kp = clip(raw_stiffness[6:12] * actual_scale, actual_Kmin, actual_Kmax)
reward = -0.03 * step_dt * cost
```

기본 `stiffness_weight=0`에서는 Kp 계산과 cfg 조회도 생략한다. Pose·hand12D만 평가하며
현재·직전 input을 수정하지 않는다. Kp 계수를 켜면 실제 arm action cfg의 scalar scale·limits를
읽는다. 현재100과50–200을 상수로 중복 정의하지 않는다. 다른 raw 입력이 같은 적용 Kp로 clipping되면
Kp 변화 비용0이고,상·하한이 같으면 고정 Kp로 보고0이다. 유효한 정규화 Kp 비용 범위는0–6이다.
계수.1이면 기존 전체 weight−.03에 대한 Kp의 상대 계수.1이며 별도 dt/weight를 중복 적용하지 않는다.
Pose·hand 비용은 기존 raw 제곱합이고,최종 비유한 비용은 기존 adapter처럼 환경별0으로 처리한다.

Cfg에는 기존 `motion_action_rate` 항의 weight−.03을 유지하고
`params={"stiffness_weight": 0.0}`만 추가했다. Environment별 Python 반복·CPU tensor 복사·scalar
조회는 없으며 batched tensor slice·elementwise·마지막 차원 reduction으로 처리한다.
기존 Time curriculum·사용자 reward/termination 설정·action/observation18D/244D·GNN/PPO·OSC·
Parallel package·외부 라이브러리는 변경하지 않았다.

**검증:** CPU 집중 검사11개가 모두 통과했다. 새 `ActionRateRewardTests`8개는 pose/hand/Kp의
기여 분리,기본 Kp 완전 제외와 cfg 미조회,같은 clipping 결과0,사용자 scale/limits와 음수 scale,
고정 bounds·scale0,전체 Kp 범위 비용6,상대 계수,원본 input 불변,32/64bit 및1024행·부분/빈 batch,
host scalar/CPU 읽기 금지,형상/파라미터 검증,NaN/Inf·overflow와 native dt 산술을 확인했다.
기존 공식18D L2 참조 검사는 유지하고,선택한12D adapter·native joint velocity 위임·native reward
manager 검사를 새 목적에 맞춰 통과시켰다. 전체 suite는 실행하지 않았다. 사용자 F/T 비활성·
joint velocity 활성 설정과 맞지 않는 기존 별도 cfg 기대값은 이번 변경에서 수정하지 않았다.

`git diff --check`와 보존 경로의 diff를 확인했고 README에 수식·파라미터를 반영했다.
실제 GPU/PhysX·프로젝트 학습·play·commit·push는 실행하지 않았다. 기존 checkpoint의 action 의미는
바뀌지 않지만 이후 학습의 reward 목적은 달라지므로 이 CPU 검사만으로 정책 개선을 판단하지 않는다.

## 89. 2026-11-04 — Episode 최소거리를 사용하는 별도 Blocker distance 함수

사용자 요청으로 Original `contact_manipulation`에 `blocker_min_distance(env, decay=...)`를 추가했다.
기존 `blocker_distance` 함수는 현재 거리만 평가하는 동작·signature로 보존했다. 공통 OBB 거리 계산만
private helper로 공유하며, 새 코드 파일·reward term·센서는 추가하지 않았다.
`RewardsCfg.blocker_distance.func`만 새 함수로 연결하고 기존 사용자 weight5·decay10을 유지했다.
Term 이름과 TensorBoard key도 기존 `blocker_distance`다.

```
min_distance[env, pad] = min(min_distance[env, pad], current_distance[env, pad])
raw_score = target_protected * mean_pad(exp(-decay * min_distance))
paid_reward = weight * step_dt * raw_score
```

17개 pad 각각의 실제 접촉면 대표점과 현재 Blocker OBB 사이 거리를 평가한다.
최소거리는 episode에서 보상이 평가된 control step들의 기록이다. 초기 reset 순간에 추가 거리
조회를 넣지 않으며, 첫 평가에서 기록을 시작한다. Pad마다 최단거리에 도달한 시점이 달라도 된다.
평균거리의 시간 최소값이나 한 시점의 가장 좋은 손 자세를 고르는 방식은 아니다.
손이 멀어져도 유효한 기록은 유지되며, 매 step 기록에 따른 양의 점수를 지급한다.
최소거리 개선 순간에만 지급하는 delta 보상으로 바꾸지는 않았다.

`TaskState`에 `[num_envs,17]` buffer를 추가하고 초기값·reset 값을 `+inf`로 둔다.
기존 reset 경로가 선택 행만 비우므로 부분 reset에서도 다른 환경의 기록은 보존한다.
같은 sample의 반복 평가는 `minimum`의 성질상 중복 누적되지 않는다. Target gate가 닫혀도 이력은
갱신하고 지급만 차단한다. Invalid 현재 pad는 이번 점수에서0이며 분모17을 유지한다.
그 pad의 이전 유효 최소거리는 보존해 이후 유효 측정에서 재개한다. Blocker 전체가 invalid이면0점이다.

GPU tensor의 기존 거리 계산에 elementwise minimum을 추가했다. 환경별 Python 반복문·CPU tensor
조회·전송은 없고, 최소거리 buffer는1024환경 float32에서68KiB다. 이 buffer는 reward 내부 상태이며
Actor/Critic observation에 추가하지 않았다. Parallel·GNN/PPO·OSC/action·spawn·termination·
사용자의 기존 팔 과속3rad/s 및 직전 Action-rate 수정은 유지했다.

**검증:** CPU 집중 검사13개가 통과했다. 새5개는 pad별 서로 다른 시점의 최소값·후퇴 시 점수 유지,
기존 현재거리 함수의 독립 동작,실제 TaskState의 unsorted partial reset 및 selected-batch reset,
Target gate·invalid·복구,1024행 float32/64와 host scalar/CPU 조회 금지·in-place buffer·gradient 없음,
cfg의 새 함수 연결 및 기존 weight/decay를 확인했다. 기존 ContactRewardTests6개와 partial reset
회귀2개도 통과했다. 전체 suite는 실행하지 않았다. `git diff --check` 및 보존 경로를 확인했고,
실제 학습·GPU/PhysX simulation·play·commit·push는 실행하지 않았다.

## 90. 2026-11-04 — Shelf displacement를 따르는 Target displacement 종료

사용자 요청으로 Original `contact_manipulation`에 `target_displacement`를 추가했다.
기존 `shelf_displacement`처럼 초기 pose 대비 위치 norm·quaternion 회전 차이를 검사하고,
공식 `TerminationTermCfg(func=mdp.failure, params={"reason": "target_displacement"})`로 등록했다.
새 controller·센서·파일·시뮬레이션 step은 없다.

Target은 기존 보호 gate와 성공 조건의 허용값을 공유한다. 초기 canonical OBB 중심에서3D 이동이
2cm를 초과하거나 초기 대비 회전이0.15rad(약8.59°)를 초과하면 한 sample에서 즉시 실패 종료한다.
Shelf의 별도 임계값1cm·2°는 유지한다. 기존 고정 환경 좌표계에서 pose 차이를 계산하므로
로봇의 움직임·EEF frame 변화가 Target 이동으로 계산되지 않는다.
Shelf와 같은 부호 불변 `cos(theta/2)` 비교를 Target 보호·실패에 함께 적용하며,
`q`와 `-q`는 같은 회전이다. Invalid/inactive 물체를 Target 이동으로 오인하지 않는다.

`FAILURE_REASON_NAMES` 끝에 새 원인을 추가해 기존 열 순서를 보존했다. Native 활성 term mask·
TaskState 캐시·부분 reset·통합 failure·성공 우선순위와 자동으로 연동되며, 같은 step에 성공 보상을
지급하지 않는다. 로그는 `Episode_Termination/target_displacement`다.
`env.terminations.target_displacement=null`이면 이 원인의 종료만 비활성화하고 Target gate·진단은 남긴다.
기존 pose 계산을 재사용하는 batched tensor 조건이며 환경별 반복문·CPU tensor 조회는 추가하지 않는다.

현재 사용자가 설정한 `failure_penalty=0`·팔 속도3rad/s·Actor vision interval0을 보존했다.
거리 최소값 reward·Action-rate 수정·다른 reward와 성공/지지/전도 조건·Parallel·PPO·제어·spawn은
바꾸지 않았다. Target gate 위반이 기본 설정에서 즉시 종료되므로 기존 회복 허용 동작은 해당
종료 term을 끈 실험에서만 적용된다. Clearance의 기존 ungated 지급도 변경하지 않았다.

**검증:** 새 CPU 검사6개가 위치·회전 경계값, quaternion 부호·크기, Target/활성 mask·invalid 구분,
native enable/disable,1024환경 tensor 처리 및 host scalar/CPU 복사 금지,성공 배타성·부분 reset을 확인했다.
관련 기존 검사13개(공식 TerminationManager의 원인별 로그·reset 포함)까지 총19개가 통과했다.
기존 gate-only 회복 검사는 새 종료 term을 명시적으로 끈 상태로 보존했다. 전체 suite는 실행하지
않았으며 `git diff --check`가 통과했다. 기존5개 reason 열에 의존한 검사는 새 원인 수를 반영했다.
실제 학습·GPU/PhysX simulation·play·commit·push는 실행하지 않았다.

## 91. 2026-11-04 — F/T 관측의 자중·관성 보상과 Sensor 기준 통일; Wrench 방식 확인 중

사용자는 F/T observation의 자중·관성 제거와 sensor-frame OSC wrench 제어를 요청했다.
추가 답변에서 force/torque scale10N·1N·m와 물체·로봇 EEF observation의 sensor frame 통일을 선택했다.
다만 stiffness가 force에 영향을 주는지 질문했고, 순수 wrench12D와 Pose·stiffness18D 중 최종 선택은
아직 답변되지 않았다. 아래 승인된 보상·frame 변경만 적용했으며 **OSC target은 아직 기존
`pose_rel`/`variable_kp`18D를 유지한다. Wrench 모드 전환을 완료했다고 판단하면 안 된다.**

`observations.self_motion_wrench`는 distal sensor half와 연결된 Hand의 모든 native 링크에 대해
Newton–Euler 자중·관성 반력을 계산한다. `F_i=m_i(a_i-g)`,
`tau_i=I_i*alpha_i+omega_i cross(I_i*omega_i)+(r_i-r_sensor) cross F_i`를 world에서 합산해
sensor 축으로 표현한다. Native 실제 COM 가속도와 각속도를 쓰므로 reset pose를 차분하지 않는다.
Inertia는 native API의 COM 기준 link-frame tensor이며 principal-axis quaternion을 중복 적용하지 않는다.
고정 metadata인 mass/inertia/local COM은 scene binding에서 한 번만 CPU→GPU 캐시한다.
매 step 계산은 기존 transition CUDA graph 안의 batched tensor 연산이며 환경별 반복은 없다.

물리 SI-200-8 모델은 raw 반력을 먼저 clipping·양자화한다. Actor·Critic observation만
`measured_reaction - modeled_self_load`를 정규화해 사용한다. Native parent-on-child 반력 부호를
유지하므로 보상 후 residual은 Hand에 작용한 외력과 반대 부호다. Raw reward F/T·GUI 모니터·
overrange 판정은 기존 측정값을 유지한다. Sensor 포화로 잃은 정보는 보상으로 복구하지 않는다.
잘못된 payload dynamics는 observation을0으로 만들면서 기존 invalid-state 판정에 반영한다.

제어 body·task frame과 scene `ee_frame`을 센서 중심 `ToolForceTorqueLink`로 통일했다.
기존 reset IK의 손바닥 목표는 유지한다. Observation은 기존244D/72D sample이며 바뀐 F/T·frame
의미를 보호하도록 contract를 `isaaclab_ur5e_sensor_osc_variable_kp_compensated_ft_v4`로 갱신했다.
이전 wrist/raw F/T checkpoint의 재개·play는 새 contract에서 거부된다.
사용자 position/orientation scale0.03/0.15·hand scale0.5,stiffness weight1·실패 비용0·팔3rad/s·
Actor vision interval0을 보존했다. Parallel·PPO 설정·배치 생성·termination·reward 수식은 변경하지 않았다.

공식 OSC 조사 결과 순수 `wrench_abs`6축은 motion stiffness를 사용하지 않는다.
`contact_wrench_stiffness_task`는 별도 force-error gain이며 `variable_kp` action이 바꾸는 값이 아니다.
또한 현재 버전은 force_scale/torque_scale 별도 필드 대신 `wrench_scale`6축 값으로10/1을 지정한다.
순수 wrench를 선택하면 불필요한 motion Kp6채널을 제외한12D action과66D sample/232D observation으로
계약을 함께 변경해야 한다. Native task-frame wrench의 원점 변환과 COM Jacobian 기준점 차이도
검증 중이며, sensor 중심 wrench를 정확히 전달하려면 최소 frame adapter가 필요할 수 있다.
아직 이 adapter나 wrench 설정은 적용하지 않았다.

**검증:** CPU60개 통과·CUDA 전용1개 skip. F/T 관련58개는 독립 Newton–Euler oracle,
회전된 sensor와 COM 레버암·비대칭 관성과 gyro·multibody 합산,1024행 float32/64·부분/빈 batch,
host 조회 금지,관측만 보상하고 raw reward를 보존하는 경로,metadata 캐시·동일 timestamp·부분 reset,
잘못된 dynamics 및 기존 native sensor 경로를 확인했다. 추가2개는 모델 input/action contract와
checkpoint 불일치 거부를 확인했다. Native Kit 전용 cfg 검사는 실제 실행하지 않았다.
`git diff --check`가 통과했으며 실제 학습·simulation·GPU benchmark·play·commit·push는 실행하지 않았다.
현재 실행 중인 학습 프로세스는 수정·중단하지 않았다.

## 92. 2026-11-04 — Sensor-frame Wrench 직접 제어와12D Action 전환 완료

사용자가 목표 wrench 직접 출력(Arm6D+Hand6D)을 선택해91절에서 보류했던 전환을 완료했다.
Original `contact_manipulation`만 수정했으며 현재 실행 중인 학습은 건드리지 않았다.
Raw action 순서는 `[Fx,Fy,Fz,Tx,Ty,Tz,hand_1,...,hand_6]`다.
공식 `wrench_scale=(10,10,10,1,1,1)`로 raw1당 Force10N·Torque1N·m를 지정한다.
Stiffness6출력을 제거하고 공식 OSC `target_types=["wrench_abs"]`, `impedance_mode="fixed"`,
motion축0·wrench축1·gravity=True·inertia decoupling=False·nullspace 없음으로 설정했다.
`contact_wrench_stiffness_task=None`이므로 native open-loop wrench이며 추가 contact sensor는 없다.
F/T observation을 활용하는 feedback은 정책이 수행한다. 0 wrench는 gravity compensation만
남고 pose hold를 제공하지 않는다. Native IdealPD의 기존 관절 effort/velocity 제한과
Hand `JointPositionToLimitsActionCfg`·scale0.5·URDF 범위·mimic/tactile 설정은 유지한다.

**Sensor 원점 보정:** 설치된 native OSC는 task-frame wrench의 moment에 root 원점까지의
`p_sensor cross F_base`를 추가하고, 이를 COM geometric Jacobian에 곱한다.
원하는 sensor 원점 wrench에 대해서는 `T_COM=R*T_sensor+(p_sensor-p_COM) cross F_base`가
필요하므로 cfg만 바꾸면 불필요한 moment가 더해진다.
기존 `ur5e_env_cfg.py`에 `SensorFrameWrenchAction`을 두어 공식 Action을 상속하고
전처리의 moment 기준점만 보정한다. Native scale 적용 후 sensor 축에서
`(R^T*p_sensor+c_COM) cross F_sensor`를 빼면 공식 변환 후 올바른 COM moment가 된다.
COM offset은 초기화 시 한 번 캐시하고 매 step GPU 회전·cross만 추가한다.
공식 controller·Jacobian·동역학 계산·gravity·effort 제출·actuator는 교체하지 않았다.
Raw action도 변하지 않는다. 별도 action 파일이나 controller 파일은 만들지 않았다.

제어·관측 기준은91절의 sensor 중심 `ToolForceTorqueLink`를 사용한다. 기존 sensor-frame 자중·
관성 보상도 유지한다. 단일 robot sample은54개 상태와12개 action으로66D이며 current/previous
history132D + scene100D = Actor·Critic 각각232D다. Scene slot10개·17tactile·현재/직전 구조·
독립 GNN/attention/native MLP·PPO 설정은 유지한다. `ModelCfg`와 observation slice·TaskState
buffer·reset snapshot을 함께 수정했다. 검사에서 reset snapshot의 기존 zero41개(72D)가
남아 있음을 찾아 `ROBOT_SAMPLE_DIM-31`(35개)로 수정하고 부분 reset까지 재검증했다.

`motion_action_rate`는 공식 `action_rate_l2`로 wrench6+hand6 raw 변화 제곱합을 계산한다.
기존 전체 weight−.03·native dt·비유한 최종 비용 보호는 유지한다. 없어지는 stiffness 출력에
대한 구간 정규화와 `stiffness_weight` 파라미터(직전 사용자 값1)는 제거했다.
기존 현재거리 `blocker_distance` 및 새 episode 최소거리 함수는 둘 다 보존했다.
다른 reward·termination·실패 비용0·팔3rad/s·Actor vision interval0·Parallel은 변경하지 않았다.

새 계약은 `isaaclab_ur5e_sensor_osc_wrench_compensated_ft_v5`다. 기존 wrist/pose18D v3와
sensor/pose18D v4 checkpoint는 설정이나 Actor를 변경하기 전에 재개·play에서 거부한다.
새 학습이 필요하다. Policy output Gaussian·초기std1·clipping 없음·PPO gamma.95/entropy.01 등
기존 학습 설정과 native runner 흐름은 유지했다.

**검증:** 관련 CPU195개 통과·CUDA 전용4개 skip.
MDP127개 중125개가 통과해12D/66D/232D, reset·부분 reset·vision·F/T 자중/관성 보상,
native action-rate와 reward/dt,Target/속도 종료 및 상태 갱신을 확인했다.
실제 native OSC/action과 production adapter의 CPU 검사는1024개 임의 회전·COM offset·
zero/force/torque/mixed 명령·gravity on/off·root 이동/회전의 물리 torque 일치를 확인했다.
원점 보정 없이2N·m·보정 후 정답1N·m인 반례도 확인했으며 fp32 최대 오차는 약6.34e-5N·m다.
Learning72개 중70개가 통과해 model/mask·독립 Actor/Critic·native PPO CPU fixture 갱신·
checkpoint roundtrip·기존 계약 거부·1024행 CPU forward를 확인했다.
기존 FT 비활성·JV 활성·실패 비용0과 충돌하던 test 기대값3개는 현재 설정을 존중하도록 수정했다.
전체 기하 생성 suite나 native Kit scene은 실행하지 않았다. `git diff --check` 및 보존 경로를
확인했다. 실제 프로젝트 학습·simulation·GPU benchmark·play·commit·push는 실행하지 않았다.

## 93. 2026-11-04 — Action-rate의 이전 분리 계산 함수 복원

사용자 요청으로 Original `motion_action_rate(env)`를 native `action_rate_l2` 위임에서
이전의 단일 함수 내 그룹별 tensor 계산으로 복원했다. 현재12D wrench 제어에 맞춰
`wrench[0:6]`와 `hand[6:12]`의 raw 변화 제곱합을 각각 구한 뒤 더한다.
18D 시절의 Hand index12나 더 이상 존재하지 않는 stiffness 출력·파라미터는 복원하지 않았다.

```
cost = sum((wrench_now - wrench_prev)^2) + sum((hand_now - hand_prev)^2)
paid_reward = -0.03 * step_dt * cost
```

Cfg는 이미 이 함수로 연결되어 있으므로 기존 함수명·weight−.03·native dt 적용을 유지했다.
Force10N/Torque1N·m의 SI scale을 곱하기 전 raw action을 평가한다. 두 그룹 계수가 모두1이므로
직전 전체12D native L2와 수학적으로 동일한 비용이며 합산 순서에 따른 roundoff만 가능하다.
현재·직전 입력의 `[num_envs,12]` 형상을 검사해 이전18D 인덱스와 섞이는 것을 방지하고,
기존 비유한 최종 비용 보호·환경별 GPU tensor 연산·입력 불변을 유지했다.
Wrench OSC·sensor frame·F/T 보상·232D 관측·GNN/PPO·다른 reward·termination·Parallel은 바꾸지 않았다.

**검증:** 관련 CPU7개가 모두 통과했다. Wrench와 Hand 기여·raw/SI 구분·1024행 float32/64·
host scalar/CPU 조회 금지·native weight/dt·비유한 입력 보호·18D 및 잘못된 형상 거부·
기존 native joint velocity 위임을 확인했다. `git diff --check`가 통과했으며 새 파일은 없다.
전체 suite·실제 학습·simulation·GPU benchmark·play·commit·push는 실행하지 않았다.

## 94. 2026-11-04 — TensorBoard의 관절 속도·F/T 모니터 추가

Original `contact_manipulation`의 학습에서 기본 활성화되는 센서 모니터를 추가했다.
기존 native RSL-RL Logger와 writer를 그대로 호출하는 작은 adapter를 사용하며,
runner·PPO·기존 Loss/Reward/console 로그는 변경하지 않는다. 별도 writer나 파일은 만들지 않았다.

- `JointVelocity/{Mean,RMS,AbsMax}/<joint_name>`: UR5e6·Hand 독립6관절의 실제 속도, rad/s.
- `FT/Measured/{Mean,RMS,AbsMax}/{Fx,Fy,Fz,Tx,Ty,Tz}`: SI-200-8 clipping·양자화 후 전체 반력.
  자중·관성을 포함하며 Force는 N, Torque는 N·m다.
- `FT/CompensatedObservation/{Mean,RMS,AbsMax}/{Fx,Fy,Fz,Tx,Ty,Tz}`: 자중·관성을 제거한
  실제 환경 observation을 SI로 환산한다. 관측의 최종 clipping도 반영한다.
- `TensorboardMonitor/ValidFraction`과 `FT/OverrangeFraction`: 측정 유효성 및 유효 sample 중
  센서 범위 초과 비율. `hand_velocity` 종료 조건이 활성화된 경우에만
  `JointVelocity/ArmOverLimitFraction`도 실제 설정 한계로 기록한다. 현재 주석 설정은 유지했다.

통계 window는 한 learning iteration의 모든 환경·control step이다. Mean은 부호 있는 평균,
RMS는 제곱평균제곱근, AbsMax는 절댓값 최대다. Episode reward와 달리 weight·dt·최대 episode 시간
나눗셈은 없다. Invalid sample과 비유한 채널은 통계에서 제외하고 전부 invalid면0을 기록하므로
ValidFraction과 함께 본다. F/T 좌표계·반력 부호는 기존 sensor frame 규약을 유지한다.

`get_task_state`의 새 completed transition을 reset 전에 한 번 기록해 종료 직전 피크를 보존한다.
부분 reset은 iteration buffer를 지우지 않으며 reset 관측0·동일 step 재조회는 기록하지 않는다.
매 step 이미 조회한 tensor의 세 slice와 두 mask만 GPU buffer에 복사하고 추가 PhysX 조회나
CPU scalar 변환은 하지 않는다. Iteration 끝에서 통계를 GPU로 계산해 최대75개 scalar를 하나의
벡터로 묶어 CPU로 한 번 옮긴다. 36step×1024환경의 float32 sample buffer는 약3.4MiB이며
mask·통계 계산용 임시 tensor는 별도다. GPU 성능 수치는 이번에 측정하지 않았다.

`env.tensorboard_sensor_monitor=False`로 끌 수 있다. Headless에서 동작하며 GUI 모니터와 독립적이다.
다음 새 학습 또는 checkpoint 재개부터 적용되며 실행 중인 프로세스에는 소급 적용되지 않는다.
Reward·termination·controller·observation/policy 계약·Parallel 프로젝트는 바꾸지 않았다.

**검증:** 기존 테스트 두 파일에 CPU 검사8개를 추가했다. 관련 회귀 검사103개 통과·CUDA 전용2개
skip. SI 환산·raw/보상 후 F/T 구분·invalid 처리·1024행 hot path의 host 조회 금지·iteration당
CPU 복사1회·부분 reset 전 피크 보존·동일 step 중복 방지·실제 native Logger와 SummaryWriter의
TensorBoard event 및 기존 Loss/Reward tag 보존을 확인했다. 기존 검사 하나는 사용자가 주석 처리한
`hand_velocity` 활성화를 가정하므로 회귀 실행에서 제외했으며 해당 검사·사용자 설정은 수정하지 않았다.
실제 프로젝트 학습·simulation·play·GPU benchmark는 실행하지 않았다.

## 95. 2026-11-04 — Action 모니터와 sensor 범위 clipping·관측 이력 반영

Original `contact_manipulation`에 사용자 요청의 Arm wrench clipping과 action TensorBoard 기록을
추가했다. 목표 sensor-frame wrench의 범위는 Fx·Fy ±200N, Fz ±360N, Tx·Ty·Tz 각 ±8N·m다.
현재 `wrench_scale=(10,10,10,1,1,1)`에서는 raw action을 ±[20,20,36,8,8,8]로 제한한다.

공통 `wrench_action_limits()`가 sensor 사양과 cfg scale로 raw 한계를 초기화 때 계산한다.
설치된 공식 OSC Action은 상속받은 `clip` 필드를 실제 적용하지 않으므로 기존
`SensorFrameWrenchAction`의 native 전처리 진입 전에 GPU clamp를 추가했다.
Clipping은 native scale과 sensor→COM moment 보정 전에 수행한다. 기준점 변환으로 생긴 moment를
다시 ±8로 자르면 올바른 sensor wrench가 아니므로 보정 후에는 sensor 한계로 자르지 않는다.
Controller·gravity compensation·effort 제출·IdealPD는 기존 native 경로를 유지한다.
이 범위는 목표 wrench의 축별 한계이며 실제 측정 반력이나 최종 관절 torque를 보장하는 한계가 아니다.

Observation의 current/previous sample에도 각 transition에서 사용한 clipping 후 **raw wrench**를
넣는다. CUDA graph 내부에서 같은 cached bounds로 제한하므로 매 step CPU 계산이나 새 SDK 조회는
없다. `ActionManager.prev_action`으로 바꾸어 한 step 더 늦추지 않으며 부분 reset의 이력은0이다.
PPO sample·log probability·ActionManager 원본 history는 변경하지 않는다. 따라서 기존
action-rate reward도 원본 action 차이로 계산한다. Invalid 판정은 clipping 전 NaN/Inf를 검사해
Inf가 clamp로 유한해져도 invalid를 숨기지 않는다. Hand6입력은 기존 native scale0.5·joint limit
mapping을 유지하고 sensor 범위로 제한하지 않는다. 12D action·232D observation·v5 frame 계약과
PPO·reward/termination 설정·Parallel은 유지했다. 기존 checkpoint 재개 시에도 새 환경 clipping이 적용된다.

기존 TensorBoard 모니터에 다음을 추가했다. 통계는 iteration 내 모든 환경·control step의
Mean·RMS·AbsMax이며 reset 전에 원본과 관측용 command를 각각 저장한다.

- `Action/PolicyRaw/{Mean,RMS,AbsMax}/{Fx,Fy,Fz,Tx,Ty,Tz,<hand_joint>}`: clipping 전12D 정책 출력.
- `Action/Applied/{Mean,RMS,AbsMax}/...`: 제한 후 sensor 원점 wrench6은 N·N·m로 환산하고,
  Hand6은 기존 raw 입력 단위로 기록한다. 실제 센서 반력·관절 torque·Hand 목표각과 구분한다.
- `Action/WrenchClippedFraction`: 유한 action sample 중 Arm6축 하나라도 clipping된 환경·step 비율.
- `Action/ValidFraction`: 원본과 관측 command가 모두 유한한 환경·step 비율.

물리 상태가 invalid여도 유한한 action 채널은 기록한다. 원본이 비유한 채널은 관측의
`nan_to_num` 대체값까지 통계에서 제외해0을 정상 명령처럼 표시하지 않는다.
기존24 + 원본 action12 + command12의48채널 buffer로 확장했다. 36×1024환경의 float32 저장소는
약6.8MiB이고 mask·통계 임시 tensor는 별도다. 최대149개 scalar를 iteration당 한 번 묶어 CPU로
옮긴다. `hand_velocity`가 비활성인 현재 설정에서는148개다. 추가 hot-path host 조회나 PhysX 조회는 없다.

**검증:** 관련 회귀 검사121개 중 CPU119개 통과·CUDA 전용2개 skip. 실제 native OSC를 CPU에서
실행해6축 양·음 포화,1024개 회전·COM offset에서의 올바른 `J^T*wrench`와 moment 보정 순서,
입력 불변·raw/적용 command 구분·현재/직전 이력·부분 reset·NaN/Inf 원본 invalid 유지·Hand 보존을
확인했다. TensorBoard 원본/제한 후 값·clipping 비율·실제 native writer·한 번의 CPU 전송도 확인했다.
사용자 주석 설정과 충돌하는 기존 `hand_velocity` 활성 가정 검사 하나는 제외하고 수정하지 않았다.
`git diff --check` 통과. 실제 프로젝트 학습·simulation·play·GPU benchmark·commit·push는 실행하지 않았다.

## 96. 2026-11-04 — Clipped Action-rate와 Hand 속도 발산 원인 검토

사용자 요청으로 `motion_action_rate`의 현재·직전 입력에 실제 command 포화를 반영했다.
기존 `bindings.action_limits` GPU cache를 재사용하며 wrench6·Hand6을 한 함수에서 계산한다.

```
u_now  = clamp(raw_now,  -limits, limits)
u_prev = clamp(raw_prev, -limits, limits)
cost   = sum((u_now[:6] - u_prev[:6])^2) + sum((u_now[6:] - u_prev[6:])^2)
reward = -0.03 * step_dt * cost
```

95절에서는 Hand 입력을 raw로 남겼지만, 실제 native Hand는 `clamp(0.5*raw,-1,1)` 뒤
URDF joint limit에 매핑한다. 따라서 Hand까지 raw ±2로 제한해야 같은 포화 명령에 서로 다른
action-rate 비용이 발생하지 않는다. `action_input_limits()`가 현재 전체 한계를
`[20,20,36,8,8,8,2,2,2,2,2,2]`로 한 번 계산한다. 현재 Hand 계약은 양의 scalar scale과
`rescale_to_limits=True`이며 이를 초기화 때 검사한다.
Hand의 실제 controller·position target mapping은 바꾸지 않는다. 관측 current/previous 이력과
TensorBoard `Action/Applied`의 Hand 채널도 native 포화에 해당하는 raw ±2를 사용한다.
정책 원본 sample·ActionManager의 raw 이력·PPO log probability·joint-velocity 수식은 유지한다.
입력 tensor를 수정하지 않고 cached bounds로 GPU clamp하므로 추가 host 조회나 SDK 호출은 없다.
현재 범위에서 최대 cost는9248, 100Hz의 한 step 비용 최대 크기는2.7744다.

**Joint velocity 분석:** 최신 `2026-11-04_16-37-44` 기록에서 적용 wrench는95절의 범위를
넘지 않았지만 실제 native `get_dof_velocities()`의 반환값이 폭증했다. Iteration9에는
raw action 최대가 약4.91인 상태에서도 Middle35.11rad/s·Ring19.91rad/s가 기록됐고,
iteration10에는 Ring약279만rad/s, iteration38에는 Middle약1.83e11rad/s가 기록됐다.
이 값들은 유한하므로 현재 finite 검사 중심 `invalid_state`를 통과한다. 속도 termination도
현재 사용자 설정에서 주석 처리되어 있다. Iteration 집계만으로 최초 이상 환경·step 또는
raw action과 물리 상태 중 어느 것이 먼저 폭증했는지 확정할 수는 없다.

우선 검토할 물리 원인은 Hand의 hard mimic·접촉·self-collision·drive 제약 경쟁이다.
현재 `_configure_hand_joints`는 importer의 compliant mimic을 natural frequency0·damping ratio0의
강체 mimic으로 덮는다. 모터 effort 제한은 drive torque에 적용되며 접촉·mimic·관절 한계
해결 impulse 전체의 상한이 아니다. PhysX는 제약을 순차적으로 풀기 때문에 뒤의 제약이
앞서 적용한 속도 제한을 바꿀 수 있고, 강체 제약들이 충돌하면 동시에 만족하지 못할 수 있다.
[PhysX Drive Stability](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#articulation-drive-stability)와
[Omni Physics Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/guides/articulation_stability_guide.html)
에서도 mimic compliance와 drive·관성·time-step 관계를 검토하도록 안내한다.
이 조합은 유력한 후보이며 직접 원인으로 확정하거나 이번에 물리 설정을 변경하지는 않았다.
URDF 질량·관성은 양수이고 각도/속도 단위 변환에도 명백한 오류는 찾지 못했다.
작은 fixed tactile 링크의 관성을 곧바로 관절 유효 관성으로 간주하지 않는다.
후속 진단은 최초 이상 환경의 physics 전후 q/qdot·Hand target·mimic 오차·reset 여부를 보존하고
공식 compliant mimic과 비교하는 것이다. 비정상 속도를 reward에서 잘라내는 것만으로 물리 원인이 해결되지는 않는다.

**검증:** 관련 CPU107개 통과·CUDA 전용2개 skip. 현재·직전 모두 clipping·동일 포화 명령의 비용0·반대 포화9248·
입력 불변·1024행 host 조회 금지·native weight/dt·빈 batch·부분 reset 행 독립성·관측/로그 일치를
검사했다. 실제 native `JointPositionToLimitsAction.process_actions`를 CPU에서 실행해 큰 raw 입력과
±2에 제한한 입력이 동일한 Hand position target을 내는 것도 확인했다.
추가 Model/NativeLearning 회귀로 raw Gaussian과 native PPO 경로 보존도 확인했다.
기존 `hand_velocity` 활성화를 가정하는 stale 검사 하나는 사용자 설정을 바꾸지 않고 제외했다.
실제 프로젝트 학습·simulation·play·GPU benchmark는 실행하지 않았다.

## 97. 2026-11-04 — Hand mimic의 importer 기본 설정 보존

사용자 요청으로 Original `contact_manipulation`의 강체 mimic 후처리를 제거했다.
`_configure_hand_joints()`는 이제 importer가 작성한6개 mimic API의 존재만 검사하며,
reference joint·gearing·offset·natural frequency·damping ratio를 덮어쓰지 않는다.
종속 관절의 DriveAPI type·max force도 다시 작성하지 않고 importer 값을 보존한다.
공식 converter의 mimic parsing을 켜는 기존 flag=True와 `joint_drive=None`은 유지한다.

독립 master6개의 URDF 위치 교집합·속도 한계·force drive/effort 설정은 유지한다.
17개 tactile 링크·충돌 형상·필터·contact sensor·관측 순서, UR5e actuator·OSC·
solver32/1·100Hz·reward/termination·PPO·Parallel은 바꾸지 않았다.
새 환경 생성부터 적용되며 현재 프로세스의 이미 생성된 articulation을 변경하지 않는다.

캐시된 importer 원본 USD를 CPU `pxr.Usd`로 읽어 실제 URDF와 대조했다.
6개 reference joint·gearing·offset이 모두 원본 연동식과 일치하므로 재작성할 필요가 없었다.
일반 손가락4개는 rotZ에 natural frequency25·damping ratio약0.005를 작성한다.
엄지2개는 실제 mimic 축 rotX와 달리 compliance 속성이 rot에 작성되는 차이를 확인했다.
따라서 이번 변경은 **설치된 importer 결과의 보존**이며6개 모두 동일한 compliant mimic이
효과적으로 적용됨을 검증한 변경은 아니다. 엄지의 compliance 축 차이는 별도 조사 대상으로
남기고 이번에 임의로 수정하지 않았다. 숫자25/0.005도 코드에서 새로 고정하지 않는다.
현재 `usd_dir=None`인 converter는 새 실행마다 임시 USD를 생성하고, 기존 강체 설정은
import 후 composed stage에만 작성됐으므로 별도 캐시 삭제는 필요 없다.

기존21절의 강체 mimic·velocity solver2회 개폐/센서 검증 결과를 새 기본 설정의 검증으로
사용하지 않는다. README에도 당시 기록이라는 점을 표시했다. 이번에 실제 physics를 실행해
다양한 접촉·장기 학습의 안정성을 확인한 것은 아니다.

**검증:** 기존 테스트 파일에3개 검사를 추가하고 관련 CPU24개가 통과했다.
서로 다른 imported mimic sentinel 값과 child drive·범위가 보존되는지,6개 master의 한계·effort가
유지되는지,API 누락·중복 거부와 parsing/drive cfg를 확인했다. 기존 tactile/contact/runtime
회귀도 포함한다. 추가 범위 검사에서 사용자 주석 설정의 velocity binding을 가정한 기존 검사
1개가 실패했으며 이번 변경과 무관해 해당 설정·검사를 수정하지 않았다.
`git diff --check` 통과. 실제 학습·simulation·play·GPU benchmark·commit·push는 실행하지 않았다.

## 98. 2026-11-04 — Parallel의 비네트워크 학습 설정을 원본과 동기화

사용자는 학습 중이므로 어떤 학습·시험 실행도 하지 말고, 네트워크 구조를 제외한 Parallel의
설정을 현재 `contact_manipulation`과 일치시키도록 요청했다. 원본 파일을 읽어 사용자 최신
수정값까지 반영했으며 Original 소스와 실행 중인 프로세스는 변경하지 않았다.

Parallel의 Actor·Critic은 기존 공식 `MLPModel`·[256,128]·ReLU·독립 가중치를 유지한다.
GNN·attention·custom PPO/model 파일을 복사하지 않았다. PPO gamma는0.999→0.95,
entropy coefficient는0.005→0.01로 맞췄으며 rollout36·epochs8·minibatches4·lr1e-3 adaptive·
GAE0.95·clip0.2·desired KL0.02·value coefficient1·gradient norm1도 원본과 동일하다.
프로젝트·Gym task·로그·cache·환경변수 namespace와 MLP용 좌우 정렬·초기 인접 교환20%는 유지한다.

동기화한 주요 설정·동작은 다음과 같다.

| 항목 | 현재 공통 설정 |
|---|---|
| Action / observation | sensor-frame wrench6 + Hand6 =12D / scene100 + sample66×2 =232D |
| Arm 제어 | native OSC wrench_abs·fixed·gravity compensation, sensor→COM 원점 adapter |
| 명령 범위 | sensor SI-200-8 한계, Hand native 포화, clipped 관측 이력·action-rate |
| F/T | 센서 축에서 distal 자중·관성 보상, raw 측정 경로와 구분 |
| Hand | native controller·URDF master 한계·importer 기본 mimic·17 tactile |
| Physics / policy | dt0.01·decimation2:100Hz /50Hz |
| Actor vision | interval0: reset 때만; Critic은 초기 인식 ID의 현재 GT를 갱신 |
| Reward | Clearance12, episode 최소거리5, Hold0.4, Shelf/Target/Object 각각−0.5 |
| Motion | clipped action-rate−0.03, 실제12관절 velocity−0.03; F/T growth·singularity reward 비활성 |
| Stability / Time | Blocker 기울기5° 여유·weight0.5, Time−0.3·학습 절반부터 curriculum |
| Terminal | 성공10·실패0·success_duration0 |
| Termination | Target displacement 포함, 팔 속도1rad/s 종료 등 현재 원본 조건 |
| Logging | 기존 native writer + joint velocity·raw/보상 F/T·raw/적용 action 모니터 |

MDP runtime도 현재 원본에서 동기화했다. `rewards.py`·`terminations.py`는 원본과 동일하다.
Parallel 전용 차이는 ray first-hit 픽셀 중심으로 선택된 주변 물체를 좌→우 정렬하는 기능과
reset 때20% 인접 교환을 Actor geometry·Critic ID에 함께 적용하는 기능이다. 역할0/1은 고정한다.
Periodic Actor vision의 기존 순서 유지 코드도 보존해 추후 양수 주기를 설정할 수 있다.

Train은 공식 Time curriculum 준비·resume 시점 동기화·TensorBoard 연결을 반영한다.
Checkpoint는 기존 native MLP `runner.load`를 사용하며 GNN 전용 loader를 가져오지 않는다.
이전 Parallel18D/244D checkpoint는 새12D/232D 모델과 크기가 달라 호환되지 않으므로 새 학습이
필요하다. 실행 중인 학습에는 소급 적용되지 않고 다음 환경 생성부터 새 설정을 사용한다.

README와 기존 테스트 정의도232D/12D·현재 default에 맞췄다. MLP와 이미지 정렬 검사는 유지했다.
기존 original 테스트의 graph mask 의존 코드는 Parallel에 가져오지 않았다. Nonzero terminal 동작을
검사하는 fixture는250/10을 명시하고 현재 default 검사는10/0을 확인하도록 구분했다.

**확인 범위:** 사용자 지시에 따라 파일 읽기·수정·텍스트 diff·공백 검사만 수행했다.
테스트·모듈 import·Python 컴파일·checkpoint 로딩·학습·simulation·play·GPU 측정은 실행하지 않았다.
따라서 실행 검증 통과나 성능·안정성 개선을 주장하지 않는다. 소스 diff에서 잔여 차이가 MLP·
의도한 정렬·프로젝트 namespace뿐임을 검토했고 `git diff --check`가 통과했다.

## 99. 2026-11-05 — CUDA 실패와 NVIDIA Xid 32 조사 기록·해결 보류

사용자 요청으로 이번 오류의 증거만 남기고 다른 안건으로 이동한다. **원인 해결·재현 실험은
보류했으며 해결 완료로 취급하지 않는다.** 재발하면 먼저 이번 기록과 같은 종류의 실패인지 비교한다.

| 항목 | 확인 내용 |
|---|---|
| 실행 | `logs/rsl_rl/contact_manipulation/2026-11-04_19-52-40` — Original GNN 프로젝트 |
| 발생 시각 | 2026-11-05 06:54:33 KST = 2026-11-04 21:54:33 UTC |
| 실행 경과 | 약 11시간 2분, Kit elapsed 39,729,855ms |
| GPU / 드라이버 | RTX 3090, PCI `0000:26:00.0` / `580.173.02` |
| 학습 프로세스 | PID `3227629` |
| 마지막 완료 기록 | iteration 7368, 06:54:32.7778 KST — 오류 약 0.22초 전 |
| 당시 설정 | 1024환경, physics100Hz·policy50Hz, robot solver32/1, patch buffer262144 |

첨부 로그와 전체 Kit 로그에서 최초 CUDA 오류는 같은 시각의
`omni.physx.tensors.plugin / GpuRigidBodyView.cpp:192–201 / unspecified launch failure`다.
후속 traceback은 `events.py`의 `reset_task → state.reset` 중 다음 대입에서 끝난다.

```python
self.blocker_min_distance[ids] = math.inf
```

이 대입은 episode 최소거리 이력을 위한 `[num_envs, 17]` tensor의 정상 초기화다.
Reset ID의 범위·중복 검사와 관련 gather 경로에서도 명백한 인덱스 오류는 찾지 못했다.
**비동기 GPU 실패가 이 줄에서 보고됐다는 사실만으로 이 대입이나 reset이 최초 원인이라고
판정하지 않는다.** 오류 전 전체 Kit 로그에는 OOM·patch overflow·singular inverse·PhysX
수치 경고가 없었으며, allocator 경고는 약 2초 뒤 종료 처리 과정에서 나타났다.

시스템 kernel journal에는 오류와 정확히 같은 시각·학습 PID·GPU의 다음 기록이 있었다.

```text
Nov 05 06:54:33 hoon kernel: NVRM: Xid (PCI:0000:26:00): 32, pid=3227629, name=python, channel 0x02000028 intr1 00000008 HCE_DBG0 00000518 HCE_DBG1 00000001
```

NVIDIA는 Xid32를 드라이버와 GPU 사이 PCIe 명령 전달을 담당하는 DMA 경로의 오류로 설명한다.
따라서 드라이버·PCIe 경로를 조사할 중요한 단서지만 **일회성 GPU 오류, 하드웨어 고장 또는
프로젝트 코드와 무관한 오류라고 확정한 것은 아니다.** 드라이버·PhysX·Warp·CUDA graph 등의
소프트웨어 문제도 배제하지 않았다.
[NVIDIA Xid32 설명](https://docs.nvidia.com/deploy/xid-errors/archive/index.html#xid-32-pbdma-error)

마지막 완료 iteration의 TensorBoard 값은 다음과 같다. Reward 로그는 기록된 통계값이며
순간 step reward나 아래 관절 최대속도와 같은 환경·시점의 값으로 해석하지 않는다.

| 지표 | iteration 7368 |
|---|---:|
| Value loss | 0.003498 |
| UR5e 독립 6관절 속도 최대 | 2.7753 rad/s |
| Hand 독립 6관절 속도 최대 | 2.00001 rad/s |
| Joint velocity reward 로그 | −0.007753 |
| Action-rate reward 로그 | −0.006209 |
| Sensor/action valid fraction | 각각 1.0 |
| Invalid-state 종료 로그 | 0 |
| Wrench clipping 비율 | 0 |

이번 실행에서는 이전 `2026-11-04_16-37-44`의 `1e11 rad/s` 같은 지속적 속도·loss 폭증 전조를
확인하지 못했다. 전체 실행의 Value loss 최대는0.821878, 독립 Hand 속도 최대는39.7505rad/s였다.
Hand raw action은 마지막에 약41.4까지 커졌지만 적용 입력은 native position mapping의 포화에
해당하는 raw ±2로 제한됐다. Raw drift는 별도 정책 검토 대상이며 이번 오류 직전 물리 명령
폭증이 확인됐다는 뜻은 아니다. **모니터는 완료된 policy transition과 독립 관절12개만 집계하므로
미완료 iteration·physics substep·종속 mimic 관절에서 순간 문제가 생긴 가능성은 남는다.**

이 실행에는 이미97절의 **importer 기본 mimic 보존**이 반영됐다. 미사용 object pool도
`set_disable_simulations()` 대신 **kinematic 전환**을 사용했다. 따라서 예전 강체 mimic 또는
simulation 비활성화 API를 이번 실행의 원인으로 그대로 적용해서 설명하지 않는다.
시작 시 IOMMU enabled 경고도 있었지만, 그것이 이번 오류를 유발했다는 증거는 없다.

드라이버 논의에서 Isaac Sim5.1의 Linux 검증 버전은580.65.06, 연결된 Omniverse 공통 표의
Ampere·GeForce R580 검증 버전은580.95.05로 확인했다. 현재580.173.02가 더 높다는 이유만으로
비호환이라고 판단하거나 downgrade를 해결책으로 확정하지 않았다.
R580 데이터센터 드라이버580.178.04에는 `CUDA_SCALE_LAUNCH_QUEUES=2x/4x`와 많은 CUDA graph
연결 노드를 사용할 때 발생하는 Xid32 수정 내역이 있다. 하지만 이번 실행이 그 조건에 해당하는지,
RTX3090에 동일 수정이 적용되는지는 미확인이다. 해당 데이터센터 패키지를 설치하라는 결정은 없다.
[Isaac Sim5.1 사양](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
[Omniverse 드라이버 검증표](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html#ada-ampere-and-turing),
[조건부 Xid32 수정 내역](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-178-04/index.html)

**재발 시 확인할 사항:** kernel journal의 최초 Xid·시각·PID·PCI 주소를 전체 Kit 로그와 맞추고,
실행에 저장된 cfg·git diff와 마지막 정상 TensorBoard 기록을 비교한다. 같은 Xid32가 반복되는지
확인한 뒤 호환 드라이버·PCIe 연결·전원·온도와 GPU 실행 경로를 원인별로 검토한다.
Checkpoint에서 새 프로세스로 재개하는 것은 손실 완화이며 재발 방지나 원인 해결과 구분한다.

**원본 증거 위치:** 첨부 `621ebc55-c662-4fec-b099-1258e4ea4bc0/Pasted text.txt`, 위 실행의
`params/env.yaml`·`params/agent.yaml`·저장 git diff·`events.out.tfevents.1793789609.hoon.3227629.0`,
Isaac Sim 설치 환경의 `kit/logs/Kit/Isaac-Sim/5.1/kit_20261104_195223.log`, 시스템 kernel journal.

**수행 범위:** 소스·기존 로그 읽기와 CPU TensorBoard 분석만 수행했고 학습·simulation·play·
checkpoint 로딩·GPU 진단·재현 테스트는 실행하지 않았다. 이번 기록 외 소스·학습 설정·
드라이버·하드웨어 설정을 변경하지 않았다.

## 100. 2026-11-05 — 동적 Blocker goal 지도 갱신 독립 GPU 벤치마크

[노션 보상 후보](https://app.notion.com/p/40e4968094ee81508608c1d3589c4aa5)를 검토한 뒤,
사용자가 변하는 환경에서 goal을 계속 갱신할 때의 실제 비용을 간단한 독립 코드로 측정하도록
요청했다. 이번 변경은 **검증 스크립트·결과·이 기록에 한정**하며, production MDP·reward·
termination·관측·제어·Actor/Critic·학습 cfg는 변경하지 않았다. 실제 학습·PhysX도 실행하지 않았다.

확정된 설계 논의는 다음과 같다. 새 항은 기존 Clearance처럼 추가적인 Target 보호·지지 gate를
두지 않는다. 초기 지도 무효 시 reset까지 새 항을0으로 두는 안에는 사용자 동의가 있었지만,
이후 기존 penalty와 보상 차단이 중복 억제를 만들 수 있다는 우려로 동적 재계산을 검토 중이다.
최종 지급·지도 교체 시 최저기록 처리와 학습 반영은 아직 확정하지 않았다.

**측정 조건:** RTX3090·driver580.173.02, PyTorch2.7.0+cu128, Warp1.15.0.
활성 CUDA 학습 프로세스가 없는 상태에서 실행했다. XY10mm·yaw10°의
`39×92×36=129168` 자세 격자와 1024환경 저장소를 사용했다. 초기 import·버퍼 준비·
Warp 컴파일은 steady-state 시간과 분리하고 CUDA event elapsed와 동기화 포함 wall time을
함께 기록했다. 아래 표는 warmup2회 후3회 측정의 **wall time 중앙값**이다.

실제 자료는 `clutter_mixing_20261022/native_clutter_reset_1024_probe.json`에 저장된24개
물체 배치다. 원본 convex collision polygon을 보수적인 최소면적 OBB로 감쌌으며, 실제 접근영역을
사용했다. 환경별0.1mm·0.05° 수준의 작은 변형을 줬다. 별도의 비교 장면은 장애물23개와
더 작은 직사각형 Blocker를 사용한 합성 배치이며 도달 가능한 goal 약40318개가 있었다.
두 장면은 물체 크기·배치·연결 영역이 다르므로 goal 개수만의 효과를 비교한 실험이 아니다.

초기 eager Torch 전수 구현에서 실제1024환경은 자세·goal 판정12.731초, edge 검사7.403초,
도달성0.500초였다(각1회). 이는 하드웨어 하한이 아니다. 중복 trig·막힌 자세의 clipping과
충돌 검사를 줄이고 Warp kernel로 호출을 합친 뒤 최종 측정을 수행했다.

**동적 지도 갱신 전체:** 장애물에0.1mm 변화를 준 뒤 free/goal node·연속 swept edge·
실제 시작 자세의 격자 연결·도달성을 순서대로 다시 계산했다. 매 step의 거리 평가와
PhysX·PPO는 아래 시간에 포함하지 않는다.

| 이번에 재계산한 환경 수 | 저장된24개 배치 | goal이 있는 합성 배치 |
|---|---:|---:|
| 1 | 3.252ms | 6.295ms |
| 16 | 3.596ms | 11.908ms |
| 64 | 13.940ms | 40.997ms |
| 1024 | 214.063ms | 751.618ms |

매 policy step에1024개를 모두 다시 계산한다면 rollout36 step의 지도 계산만 약7.71초 또는
27.06초가 추가되는 산술 추산이다. 실제 학습 iteration을 실행해 측정한 시간이 아니다.
16개만 매번 갱신하면 같은 추산은 약0.13초·0.43초다. 충분한 최적화 전 전체 환경의 매step
전수 갱신을 그대로 학습에 넣을 근거는 없다.

**비용 분해:** 저장 배치의1024환경에서 node 판정 약7.36ms, edge 검사 약2.12ms,
시작 연결점 검색 약181.86ms, 순수 graph 탐색 약30.65ms였다. 합성 장면은 각각
124.57ms·27.47ms·183.97ms·424.91ms였다. 각 단계의 독립 중앙값이므로 전체 갱신 중앙값과
정확히 합산 일치하지 않는다. 시작 연결점은 아직 전체 격자에서 nearest-free 후보를 찾는
Torch 경로이고, graph는8회마다 host convergence check가 남아 있다. 지역 연결·유효 노드만의
탐색·변화한 node/edge만 갱신하는 개선 여지가 있으며, 현 결과를 최종 최적 성능으로 주장하지 않는다.
별도 프로세스의 peak PyTorch allocated memory는 최대 약2160MiB였고 학습과 동시 측정하지 않았다.

**거리 평가만의 측정:** `D=min_g(||p-pg||+R*|wrap(yaw-yawg)|)`를 GPU에서 계산했다.
1024개 query를 한 번에 처리하고, goal4096개 단위로 나눠 임시 메모리를 제한했다.
무작위 연속 goal 좌표의 산술 부하이며 도달성 생성 비용을 포함하지 않는다.

| 후보 goal 수 | 공유 집합 | query별 mask |
|---|---:|---:|
| 2500 | 0.421ms | 0.457ms |
| 10000 | 1.501ms | 1.656ms |
| 40000 | 6.000ms | 7.080ms |

7회 중앙값이며 query별 mask는4가지 패턴의 반복이고 유효 후보는75%다. 이를1024개의 실제
독립 재계획 장면으로 해석하지 않는다. 거리 계산보다 지도 구축·연결성 갱신이 큰 비용인 사례가
확인됐지만, 작은 goal 집합을 공유할 때의 거리 평가까지 어렵다는 결론은 아니다.

**정확도·범위 확인:** 독립 scalar SAT·polygon clipping160개씩, 중간 자세5050개를 이용한
sweep envelope 확인, 작은 격자의 CPU BFS·yaw 주기·부분 갱신 행 격리를 통과했다.
GPU에서도 실제/합성 장면 각각4환경 전체 격자의 free·goal·edge·reachable·reachable-goal
mask가 Torch 참조와 모두 일치했다. 코드가 검사하는 것은 보수적 OBB·6방향 연결이며,
26방향 복합 이동·pivot·실제 mesh·마찰·손의 실행 가능성을 검증한 것은 아니다.
현재 SAT는 접선 접촉도 충돌로 취급하므로 노션안의 접촉 허용 경계 처리와도 차이가 있다.

특히 저장 배치에서는 **이 해상도와 보수적 연결 모델에서 reachable goal이0개**였다.
초기 자세는 격자에 연결됐지만 이것이 실제 물리적 해법 부재를 뜻하지 않는다. 거친 격자·
OBB proxy·6방향·단일 seed 선택으로 경로를 놓칠 수 있다. 빈 goal만으로 비용을 과소평가하지
않도록 도달 가능한 goal이 충분한 합성 장면도 별도로 측정했다. 이 정확도/coverage 문제도
학습 반영 전 해결해야 한다.

재현 스크립트와 원자료는
[`blocker_goal_benchmark_20261105`](../outputs/contact_manipulation/blocker_goal_benchmark_20261105/)에 둔다.
`benchmark.py`는 Torch 참조와 compact 거리 부하, `fused_benchmark.py`는 fused GPU 비교,
`validate.py`는 독립 CPU oracle이다. `scene_input.json`은 실제 배치의 필요한 필드만 보존한다.
주요 결과는 `actual_final.json`·`synthetic_final.json`·`distance_final.json`·`summary.json`,
검증은 `geometry_validation.json`·`gpu_validation.json`에 있다.

```bash
./IsaacLab/isaaclab.sh -p outputs/contact_manipulation/blocker_goal_benchmark_20261105/fused_benchmark.py \
  --synthetic --counts 1 16 64 1024 --repeats 3 \
  --output outputs/contact_manipulation/blocker_goal_benchmark_20261105/repeat_synthetic.json
```

**다음 검토:** 거리 평가는 매step 유지하면서, 시작 연결점의 전수 검색 제거와 환경 안에서
영향받은 node/edge만 갱신하는 방식을 먼저 검증한다. 동적 goal 교체만으로 개선 보상이 생기는
문제와 기존 신기록 이력의 재평가 문제도 속도와 별개로 남아 있다. 새 reward를 학습에 넣거나
현재 보상 수식·weight를 변경하지 않았다.

## 101. 초기 재배치 영역 보상과 현재 손–Blocker 거리 복원 (2026-11-05)

**사용자 요청:** 동적 도달 가능 목표 자세 탐색은 별도 테스트로 보류하고, 초기 Clearance 양옆의
빈 재배치 영역으로 Blocker가 가까워지는 보상 하나를 추가한다. 사용자가 선택한 판정은
**Blocker 중심점의 영역 진입**이며, 영역 내부에서는 같은 보상을 유지한다. 여러 영역이면
거리의 최솟값을 사용한다. 기존 `blocker_distance`는 episode 최소거리 평가에서 현재 거리로
복원한다. 적용 대상은 원본 `src/contact_manipulation`이다.

### 수식과 초기 계수

환경 고정 XY 좌표계에서 보정된 Blocker OBB 중심을 $p_t$, 초기 빈 영역을 $R_j$라 한다.

$$
d_t=\min_j\operatorname{dist}(p_t,R_j),\qquad
r_{\mathrm{relocation},t}=\exp(-a d_t),\qquad
a=\frac{\ln 2}{0.10}\;\mathrm{m}^{-1}.
$$

영역 내부·경계에서는 $d_t=0$, 밖에서는 polygon 선분까지의 최단 거리를 사용한다.
raw score는 내부1, 5cm에서0.7071, 10cm에서0.5, 20cm에서0.25다.
`blocker_relocation.weight=1.0`으로 시작하며 native RewardManager의 `weight × step_dt`를
그대로 적용한다. 매 step 현재 거리로 지급하며 신기록 이력·추가 상태 gate·`/dt` 상쇄는 없다.
유효 영역이 없거나 중심이 비유한 경우 해당 항만0이며 scene 재생성·실패 예외를 추가하지 않는다.

`blocker_distance` term은 기존 `mdp.blocker_distance` 함수로 되돌렸다.

$$
r_{\mathrm{distance},t}=\frac{V_T}{17}\sum_{i=1}^{17}\exp(-10d_{i,t}).
$$

17개 pad 모두 현재 OBB 거리를 평가하므로 손이 멀어지면 보상도 감소한다. weight5.0과
Target 보호 gate는 유지한다. 이전 `blocker_min_distance` 함수와 이력 필드는 비교용으로 보존한다.

### 빈 영역과 실행 경로

- 초기 Clearance의 양옆 Y band에서 모든 활성 non-Blocker 충돌 footprint를 장애물로 사용한다.
  그 AABB 경계의 X 구간과 Y interval union으로 maximal empty rectangle을 추출하고,
  Blocker의 principal 방향 중 하나에서 들어갈 크기의 사각형만 등록한다. 공간 격자·SE(2)
  자세 탐색·IK·경로 연결성 계산은 수행하지 않는다.
- 생성기가 확보한 pocket은 certificate의 center/yaw로 실제 convex footprint를 복원하고
  선반 지지 범위·Y side band·다른 물체와의 convex SAT 분리를 재검증한다. 외접 AABB 전체가
  비었다고 간주하지 않는다. Y side band 밖의 pocket은 이 항의 목표로 사용하지 않는다.
- 다른 물체의 AABB를 사용하는 추가 공간 추출은 보수적이다. 대각선 물체 사이에서 실제로
  가능한 모든 비정형 빈 공간까지 찾는 완전한 알고리즘은 아니다. 실제 pocket은 이 보수적
  rectangle 근사와 별도로 검사한다.
- 새 공유 배치에서만1회 생성한다. `SceneBatch`의 immutable polygon/count/mask를
  zero-stride broadcast view로 공유하고 부분 reset에서는 재계산하지 않는다. `EpisodeBatch`나
  물체 생성 규칙·PhysX actor·collision 설정은 변경하지 않았다.
- `TaskState`가 기존 transition의 `batch.center`에서 현재 Blocker 중심을 보관한다. 매 step
  보상은 GPU tensor의 point–polygon 거리와 region min만 수행하며 추가 PhysX 조회가 없다.
  초기 영역은 주변 물체·선반이 움직여도 고정한다.

중심 진입 기준이므로 전체 Blocker가 현재 yaw에서 영역 안에 들어갔다는 보장은 없다.
최댓값도 실제 Clearance 성공을 뜻하지 않는다. 현재 상태의 성공·전도·주변 물체 이동 등은
기존 보상과 종료 조건이 계속 판단한다. 도달 가능성이나 실제 push 경로의 존재를 증명하는
보상은 아니며, 학습 성능 향상 여부는 새 실행으로 확인해야 한다.

### 검증과 연산 비용

Isaac Sim·PhysX·train/play 없이 CPU 수식·native adapter/reset mock과 독립 GPU 연산만 검사했다.

- 원시 거리/보상8개 테스트와 초기 영역3개 테스트: 내부·경계 plateau, CW/CCW polygon,
  polygon padding, 여러 영역의 최근접 선택, invalid/빈 영역, 1024행 tensor, actual rotated
  pocket과 그 외접 AABB 차이, 물체가 영역을 점유한 경우 제외를 확인했다.
- Native adapter에서 비대칭 USD 원점의 보정 중심, nonzero env origin, step cache와 부분
  reset 선택행을 검사했다. 공유 reset에서는 새 배치당1회 추출과 metadata 재사용을 확인했다.
- 관련 CPU 테스트144개 실행: **142통과·2skip·새 실패0**. 별도4개는 수정 전 HEAD에서도
  동일하게 실패했다. 과거 기대값250과 현재 success_reward10의 불일치3개 및 arm velocity
  기대값3과 현재1의 불일치1개다. 이4개를 통과시키려고 사용자 설정이나 무관한 판정을 바꾸지 않았다.
- 저장된24개 물체 배치에서 유효 영역3개를 찾았다. 모든 영역의 실제 물체/A0 overlap이
  $10^{-9}\;\mathrm m^2$ 미만임을 CPU에서 확인하고, GPU polygon·보상과 CPU float64 결과를
  비교했다.

RTX3090, PyTorch2.7.0+cu128의 독립 wall-clock 중앙값은 다음과 같다.

| 단계 | 중앙값 |
|---|---:|
| 저장 배치의 초기 영역 생성: 공유 scene당1회 | 5.269ms |
| 저장 영역3개로1024행 보상 계산: 매 step | 0.876ms |
| 합성 영역64개로1024행 보상 계산: 매 step | 0.924ms |

저장 배치의 region metadata 자체는123bytes이며 환경 수만큼 복제하지 않는다. 위 값에는
PPO·물리 step·기존 snapshot 비용이 포함되지 않는다. 36step이면 거리/보상 계산만 약31.5ms에
해당하지만 실제 training iteration을 측정한 값은 아니다. 실제 빈 영역 수와 polygon vertex 수에
따라 비용이 달라진다. 합성64영역은 arithmetic stress case이며 실제 장면64개 영역을 뜻하지 않는다.

재현 코드·원자료:
[`relocation_reward_20261105`](../outputs/contact_manipulation/relocation_reward_20261105/)
의 `benchmark.py`, `results.json`, `cpu_tests.json`.

## 102. Headless 초기 실행의 Vulkan page fault·Xid31 (2026-11-05)

사용자 제공 오류는 **2026-11-05 12:51:57 UTC = 21:51:57 Asia/Seoul**에 발생했다.
해당 실행은 `logs/rsl_rl/contact_manipulation/2026-11-05_21-50-04`, PID71361이다.
읽은 자료는 다음과 같다.

- Kit 전체 로그:
  `/home/hoon/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261105_214947.log`
- Vulkan crash dump:
  같은 디렉터리의 `kit_20261105_214947-0.nv-gpudmp`.
- 커널 근거: `/var/log/kern.log:1176–1177`, `/var/log/syslog:82660–82661`와
  `journalctl -k -n 80`.

### 확인된 사실

```text
Nov 5 21:51:57 KST
NVRM: Xid (PCI:0000:26:00): 31, pid=71361, name=python,
channel 0x0000003c, intr 00000000.
MMU Fault: ENGINE GR0_PBDMA1 HUBCLIENT_ESC
faulted @ 0x40_05740000.
Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ
```

1. Vulkan이 기록한 가상주소 `0x0000004005740000`와 커널 주소 `0x40_05740000`가
   일치한다. GPU 가상메모리 읽기 중 MMU fault가 있었고 이후 Vulkan은
   `ERROR_DEVICE_LOST`를 보고하며 앱을 종료했다.
2. GPU는 기존과 같은 PCI0000:26:00, UUID `GPU-87dfa229-670d-63dd-0199-0e6629448650`,
   드라이버는580.173.02 Open Kernel Module이다. **99절은 Xid32**, 이번은 **Xid31**로
   동일 종류의 재발이라고 단정할 수 없다.
3. Kit 로그에서 page fault 전에 CUDA/PhysX error·patch overflow·명시적인 OOM은 확인되지
   않았다. 같은 커널 구간에도 AER/PCIe Bus Error·fallen-off/reset·OOM/killed-process 기록이
   없다. 이는 원인 배제의 증거가 아니라 해당 로그에 그런 선행 기록이 없다는 뜻이다.
   오류 직전 peak VRAM은 저장되지 않았다. `device lost, out of memory, or an unexpected bug`
   문장은 일반적인 원인 목록이며 OOM 확정 메시지가 아니다.
4. 실제 실행은 headless Kit와 `--no-window`를 사용했다. 설정은1024환경, physics dt0.01,
   decimation2, render_interval2147483647, actor/critic compile_model true다. Headless에서도
   이 실행은 Vulkan renderer를 초기화했다. Vulkan이 오류를 보고했다는 것만으로 최초 원인
   연산이 렌더링이라고 단정할 수 없다.
5. 해당 TensorBoard 파일은88bytes, scalar tag0개, checkpoint0개다. 첫 iteration 통계를
   기록하기 전에 종료됐으며 reward/loss 상승이나 행동 발산을 근거로 원인을 설명할 수 없다.
6. 새 relocation 보상은 활성(weight1)이고 손–Blocker distance도 현재값 함수다. 신규 코드
   검토에서 명백한 gather 범위 초과·공유 polygon의 수명 문제·새 CUDA graph capture는
   찾지 못했다. 새 영역 계산은 stage/PhysX actor/collider를 바꾸지 않는다. 다만101절의 GPU
   검증은 독립 연산이므로 SDK·물리·Vulkan과의 통합 상호작용을 배제하지는 못한다.

### 판단과 다음 구분 실험

[NVIDIA Xid31 공식 설명](https://docs.nvidia.com/deploy/xid-errors/archive/index.html#xid-31-fifo-mmu-error)은
MMU의 잘못된 주소 접근 계열 오류로 분류하며 애플리케이션·드라이버·하드웨어 원인이 모두
가능하다고 설명한다. 이번 오류를 단순한 일시 GPU 오류나 드라이버 문제로 확정하지 않는다.

신규 변경의 영향을 구분하려면 같은 명령·seed에서 relocation 항의 **초기 영역 준비와 매 step
평가를 함께 제외한 대조**가 필요하다. 현재 `weight=0`만 설정하면 native RewardManager가
매 step 평가만 생략하고 초기 영역 생성은 계속하므로 전체 신규 경로를 제거한 대조가 아니다.
그 대조에서도 재발하면 기존 CUDA graph·torch.compile 및 SDK/드라이버 경로를 분리하는
실험을 이어갈 수 있다. CUDA 측 최초 접근은 Compute Sanitizer 등으로 추가 추적할 수 있지만
현재 자료에는 원인 함수·행을 확정할 증거가 없다.

이번 조사에서는 로그 읽기와 CPU TensorBoard 파일 확인만 수행했다. 학습·simulation·GPU
재현 테스트나 driver/cache/config 변경은 하지 않았다. 신규 오류와 이전 Xid32를 각각 보존하여
이후 재발 조사 시 같은 GPU·다른 fault 종류임을 비교한다.

## 103. 학습135 iteration 후 CUDA 실패와 과거와 동일한 Xid69 (2026-11-05)

실행은 `logs/rsl_rl/contact_manipulation/2026-11-05_22-26-35`, PID93333이다.
사용자 traceback은 PhysX `GpuRigidBodyView.cpp:192–201`의 unspecified launch failure와
PPO `mean_value_loss += value_loss.item()`에서 보고된 CUDA 실패다.

### 오류 순서와 과거 비교

커널 `/var/log/kern.log:1179`, `/var/log/syslog:92146`에서 다음을 확인했다.

```text
Nov 5 22:44:51 KST = 2026-11-05 13:44:51 UTC
NVRM: Xid (PCI:0000:26:00): 69, pid=93333, name=python,
Class Error: channel 0x00000013, Class 0000c7c0,
Offset 000001b0, Data 00000041, ErrorCode 00000053
```

같은 PID의 Kit 로그는
`/home/hoon/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261105_222618.log`다.
최초 `[Error]`는5372줄, 13:44:52UTC의 `GpuRigidBodyView.cpp:192`이며10개 CUDA 오류가
이어진다. allocator 경고는 종료 중13:44:54UTC에 발생했다. 선행 OOM·patch overflow·
singular inverse·Vulkan pagefault/device-lost 기록은 없으며 같은 커널 구간에도 AER·OOM·GPU
reset/fallen-off 기록이 없다. 오류 직전 peak VRAM이 없어 OOM을 완전히 배제하는 근거는 아니다.

**과거 동일 signature:** `/var/log/kern.log.1:18`의10월25일16:32:21KST, PID858212,
실행 `2026-10-25_02-40-51`에서도 Xid69와 Class`c7c0`·Offset`1b0`·Data`41`·ErrorCode`53`이
같았다. 채널 번호는 달랐다. 이번은102절의 Xid31이 아니라, 새 relocation 항 추가 전에도
존재했던 Class Error 유형과 일치한다. 신규 항만이 이 signature를 만들 수 있다는 가설은
맞지 않지만, 현재 변경이 발생 빈도에 영향을 주는지까지 배제할 수는 없다.

[NVIDIA Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)는69를
Graphics Engine class error로 분류하고 `CHECK_APP/CUDA` 조사를 안내한다.
공개 [Ampere compute header](https://raw.githubusercontent.com/NVIDIA/open-gpu-doc/master/classes/compute/clc7c0.h)에서
Class`0xC7C0`는 `AMPERE_COMPUTE_B`, Offset`0x01b0`는 `NVC7C0_LAUNCH_DMA`다.
따라서 이름의 Graphics만 보고 viewport/렌더링 문제로 해석하지 않는다. 이 명령 정보만으로
PhysX·Torch·Warp·드라이버 중 호출 주체나 실패 데이터의 근원을 특정할 수는 없다.
ErrorCode`0x53`의 상세 의미를 다른 API의 숫자 코드와 임의로 대응시키지 않는다.

CUDA 비동기 실행에서는 `.item()` 같은 동기화 시점에 앞선 실패가 보고될 수 있다.
[PyTorch2.7 문서](https://docs.pytorch.org/docs/2.7/notes/cuda.html#asynchronous-execution)에 따라
traceback의 loss 출력 행을 수식/critic의 직접 실패 지점으로 보지 않는다. 커널 Xid가 PhysX
보고보다 먼저 있었지만, 기록 시각만으로 최초 원인 연산을 확정하지 않는다.

### 실패 전 저장된 수치

TensorBoard194개 scalar series를 CPU로 확인했다. iteration0–134, 총135개가 기록됐으며
마지막 wall time은13:44:47.220100UTC로 오류보다 약4.78초 앞선다.

| 지표 | 마지막134회 | 전체 기록 |
|---|---:|---:|
| Value loss | 0.0589255 | 최대0.0589255 |
| Surrogate loss | -0.000797685 | -0.0530615–0.0486085 |
| 정책 평균 std | 0.534005 | 0.534005–0.961274 |
| Action-rate reward | -0.0110162 | 최소-0.0164486 |
| Joint-velocity reward | -0.0129296 | 최소-0.0172378 |
| Invalid-state 종료 | 0 | 전체0 |
| 상태·action 유효 비율 | 1 | 전체1 |
| Wrench clipping 비율 | 0 | 전체0 |
| Arm 관절속도 최대값 | 4.05262rad/s | 최대6.27946rad/s |
| 독립 Hand 관절속도 최대값 | 3.32495rad/s | 최대19.6902rad/s |
| 적용 force 최대 축값 | 23.6959N | 최대46.4871N |
| 적용 torque 최대 축값 | 1.68438N·m | 최대4.61083N·m |

scalar에 NaN/Inf는 없다. Value loss는 마지막0.0322→0.0486→0.0589로 올랐지만 과거의
극단적인 수치 폭발과 같은 증거는 없다. 속도 수치를 안전하다고 인증하는 뜻도 아니며,
실패한 마지막4.78초의 rollout/update 자체는 이 기록에 포함되지 않는다. 속도 모니터는
36×1024 transition의 실제 최대값으로12개 독립 관절만 포함하고 mimic 관절은 포함하지 않는다.

마지막 mean return1.85146, mean length26.42step, hand_velocity 종료비율0.78814,
singularity0.05941, success0이다. 종료 편향은 학습 관점의 별도 문제이며 GPU Class Error의
원인으로 연결할 근거는 없다.

저장 설정은1024환경, physics dt0.01·decimation2(정책50Hz), rollout36step, Actor/Critic
compile_model true다. 당시 relocation은 **weight3·decay10**, 현재 거리 보상은weight5·decay10,
Contact hold는weight3이다.101절 최초 적용값1·ln2/.10과 구분한다.10월25일 동일 signature의
실행도1024환경과 두 모델 compile_model true였다. 공통점은 원인 증명이 아니다.

### 후속 검증 방향

loss나 reward weight를 GPU 오류의 직접 원인으로 단정해 바꾸기보다, 동일 조건에서
GPU 실행 경로를 분리한다. Actor/Critic `torch.compile`을 끈 대조, 기존 transition/support/IK
CUDA graph 경로를 분리한 대조를 한 종류씩 비교할 수 있다. 현재 모델 compile 코드 자체는
`options={"triton.cudagraphs": False}`를 사용하므로 compile off와 환경 CUDA graph off는
동일한 변경이 아니다. `CUDA_LAUNCH_BLOCKING=1`은 최초 실패 호출을 좁히는 진단 수단이며
해결책이나 정상 학습 설정으로 간주하지 않는다. 필요하면 NVIDIA 버그 리포트와 메모리
검사 도구로 최초 잘못된 연산을 추가 추적한다.

이번 조사에서는 로그·저장 설정·설치된 PPO 코드·TensorBoard만 읽었다. 새로운 학습·재현
테스트·GPU 연산이나 driver/실행 설정 변경은 수행하지 않았다. 코드·SDK·드라이버·하드웨어
중 근본 원인 귀속은 아직 미확정이다.

## 104. OSC 관절 조회에서 보고된 동일 Xid69 재발 (2026-11-06)

실행 `logs/rsl_rl/contact_manipulation/2026-11-05_22-52-36`, PID106662에서 발생했다.
사용자 traceback은 공식 OSC Action의 `_compute_joint_states()`가
`self._asset.data.joint_pos[:, self._joint_ids]`를 읽는 중 unspecified launch failure를
보고한다. 관절 인덱스 오류나 OSC 수식 오류가 직접 확인됐다는 뜻은 아니다.

### 커널·Kit 대조

```text
2026-11-06 02:27:56 KST = 2026-11-05 17:27:56 UTC
NVRM: Xid (PCI:0000:26:00): 69, pid=106662, name=python,
Class Error: channel 0x00000013, Class 0000c7c0,
Offset 000001b0, Data 00000041, ErrorCode 00000053
```

- 근거는 `/var/log/kern.log:1184`, `/var/log/syslog:94175`다.103절의 Xid69와는
  Class·Offset·Data·ErrorCode뿐 아니라 채널 번호까지 같다.10월25일에도 같은4개 필드가
  존재한다.102절 Xid31의 MMU fault 주소와는 연결하지 않는다.
- Kit는 `kit_20261105_225220.log`, 시작 시각은11월5일22:52:20KST다.
  이 로그에는 선행 `[Error]`가 없고,11월6일02:27:58KST에 종료 시작과 allocator 경고를
  기록했다. 첨부된 `12,938,517ms` 경고가 이 파일5378줄과 일치한다. allocator 경고는
  kernel fault 약2초 뒤 종료 중 발생하므로 최초 원인으로 판단하지 않는다.
- OOM·patch overflow·singular inverse·Vulkan pagefault/device-lost·AER·GPU reset/fallen-off
  선행 기록은 발견하지 못했다. peak VRAM 및 실패 순간의 CUDA 입력은 저장되지 않았다.
- close 이후에도 Kit에 omniclient token 갱신 로그가 남아 종료 완료가 지연된 정황은 있다.
  조사 당시 `ps -p 106662`에는 해당 PID가 없었다. 이를 최초 GPU fault의 원인으로 연결하지 않는다.

### 저장된 학습 상태

TensorBoard에는 iteration0–2297, **2298회**가 기록됐다. 마지막 기록은
11월6일02:27:56.142679KST로 사용자 CUDA 보고보다 약1.86초 앞선다.

| 지표 | 마지막2297회 | 전체 기록 |
|---|---:|---:|
| Value loss | 0.0548914 | 최대0.853210 @2058 |
| Surrogate loss | 0.00440623 | -0.0525547–0.125904 |
| 평균 std | 0.955622 | 0.540090–0.959331 |
| Action-rate reward | -0.0176155 | 최소-0.0443837 |
| Joint-velocity reward | -0.0272967 | 최소-0.0379490 |
| Arm 속도 최대값 | 3.73472rad/s | 10.9939rad/s @1438 |
| 독립 Hand 속도 최대값 | 2.00941rad/s | 29.8748rad/s @1638 |
| 적용 force 최대 축값 | 14.4631N | 46.7431N |
| 적용 torque 최대 축값 | 1.03902N·m | 4.61083N·m |

194개 scalar series에 NaN/Inf가 없고 invalid_state는 전체0, 상태/action 유효 비율은1이다.
마지막 value loss10개는0.12834,0.00852,0.01021,0.01246,0.01220,0.01805,0.03195,
0.03077,0.03978,0.05489다. 기록된 구간에는 과거1e11급 상태나 극단적인 loss 폭발의 증거가
없지만 실패한 마지막 약1.86초의 상태까지 배제할 수는 없다. 속도 최대값을 안전하다고
인증하는 결과도 아니다.

Hand raw action은 마지막15.2821·전체16.5939지만 적용 채널 절댓값은 계속2 이하였다.
Wrench clipping 비율은 전체0이다. 마지막 mean return14.0806·episode length214.10step,
timeout0.609484·hand_velocity0.375705·success0이다. 학습 품질은 GPU failure와 별도로 평가한다.

### 설정과 판단

저장 `agent.yaml`은103절 실행과 byte-identical이며 `env.yaml`도 log_dir만 다르다.
따라서 **두 모델 compile_model=true**와1024환경·reward·제어 설정이 그대로다.
이번 실행을 compile off 대조로 해석하지 않는다. 동일 GPU Class Error가 PPO loss 집계와
OSC 상태 조회 등 다른 호출에서 보고된 것으로, 특정 traceback 행을 고치는 근거는 없다.

다음 비교는 기존 명령에 `agent.actor.compile_model=false agent.critic.compile_model=false`를
추가해 모델 컴파일만 제외하는 방식으로 수행할 수 있다. 환경의 transition/support/IK CUDA
graph는 별도 경로이므로 이 설정으로 꺼지지 않는다. 모델의 Triton CUDA graph도 현재 코드에서
이미 false이므로 두 종류의 graph를 혼동하지 않는다. 이 대조가 원인 증명이나 해결 보장은
아니며 재발하면 환경 graph와 동기화 진단을 별도로 비교해야 한다.

이번 조사에서는 로그·저장 설정·공식 OSC 코드와 CPU scalar만 읽었다. 추가 실행·GPU 작업·
driver/학습 설정 변경은 하지 않았다. 기존 설정을 보존하고 실패 유형의 재발 기록을 추가했다.

## 105. 모델 컴파일 분리와 GPU 검증 (2026-11-06)

사용자가 반복 오류의 원인 해결·학습 재개를 요청했고, 실행 중이던 GUI play를 직접 종료한 뒤
검증을 진행했다. GPU 사용 프로세스가 없는 것을 확인하고 각 GPU 작업을 순서대로 실행했다.

### 변경 범위와 해석

- 원본 Actor·Critic의 `compile_model` 기본값을 `true → false`로 변경했다. `torch.compile`은
  GNN·attention 등의 tensor feature 계산을 TorchInductor로 최적화하는 선택 기능이다.
  네트워크 구조·파라미터 이름·checkpoint contract·native PPO·reward·제어 설정은 유지한다.
  이 변경은 **컴파일 원인 확정에 따른 수정이 아니라 원인 분리를 위한 실행 기준 변경**이다.
- 환경 설정에 `transition_cuda_graph`, `support_cuda_graph`, `robot_start_cuda_graph`를 추가했다.
  기본값은 모두 `true`다. 각각 `false`로 지정하면 해당 단계의 자체 capture/replay만 우회하고,
  GPU tensor 계산·Warp kernel·동일한 후보 순서는 유지한다. PhysX SDK 내부 graph를 끄는 API가 아니다.
- 모델의 `triton.cudagraphs=false`는 이전부터 적용되어 있었다. 모델 컴파일과 환경의 CUDA graph는
  다른 경로다. 컴파일을 꺼도 직접 작성한 Triton edge reduction과 Warp 기하 kernel은 남는다.
- `contact_manipulation_parallel`, 보상 weight·수식, 물리·로봇·센서 설정은 이번 조사에서 변경하지 않았다.
  드라이버·BIOS·IOMMU·GPU clock도 변경하지 않았다.

### GPU 오류에 대해 확인한 범위

104절과103절의 동일 `Xid69 / c7c0 / 1b0 / 41 / 53`은
[해당 드라이버의 NVIDIA class 정의](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.173.02/src/common/sdk/nvidia/inc/class/clc7c0.h#L212)에서
Ampere compute class의 `LAUNCH_DMA`에 해당한다. Data `0x41`은 정의된 flag 조합이며,
그 숫자 자체를 비정상 action이나 torque 값으로 해석하지 않는다. Hardware class ErrorCode `0x53`은
별도 `NV_STATUS` enum의 같은 숫자로 해독할 수 없다.

[NVIDIA Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)는 Xid69에
앱 재시작과 앱/CUDA 경로 조사를 안내한다. 현재 로그만으로 앱·SDK·driver·hardware 중 하나를 확정할
수 없다. 검토한 Warp/Triton kernel의 인덱스 mask, stream, mesh·graph buffer 수명에서 구체적인
메모리 오류는 찾지 못했지만 이는 전체 경로의 무결성 증명이 아니다.

[2023년 PyTorch 이슈의 RTX3090 보고](https://github.com/pytorch/pytorch/issues/74235#issuecomment-1612030397)에도
정확히 같은 오류 서명이 있다. 해당 사용자는 BIOS 업데이트 후 개선을 보고했지만 CPU·보드·driver가
다르므로 우리 시스템의 해결책으로 단정하지 않는다. 현재 로컬 시스템은 Ryzen7 3700X,
MSI B450M MORTAR MAX, BIOS2.P1(2025-09-18), RTX3090, driver580.173.02다.
IOMMU 활성화는 확인했지만 단일 GPU인 이번 실패가 PCIe peer transfer와 연관됐다는 증거는 없다.
580.178.04의 특정 launch-queue/CUDA graph 관련 Xid32 수정도 이번 Xid69의 확인된 해결책은 아니다.

### 실제 검증

검증 자료는 `outputs/contact_manipulation/gpu_stability_20261106/`에 저장했다.

| 검사 | 조건 | 결과 |
|---|---|---|
| CPU 회귀 검사 | native Actor/Critic·checkpoint·Time curriculum·새 graph switch | 41개 중40통과, GPU 전용1skip |
| 기본값 변경 검사 | native 학습 설정의 compile=false | 통과 |
| 실제 환경16 iteration | 1024환경, checkpoint2250, 모델 compile=false, 환경 graph=true | 정상 종료,147.66초, invalid_state=0,193개 scalar 모두 finite |
| 실제 환경4 iteration | 1024환경, checkpoint2250, 모델/환경 graph off, CUDA_LAUNCH_BLOCKING=1 | 정상 종료,97.46초, invalid_state=0,193개 scalar 모두 finite |
| 모델 독립 memcheck | 실제 checkpoint,1024 추론·9216 backward(8192+1024), compile on/off, full FP32 | 출력·gradient 일치, GPU 메모리 오류0 |
| Encoder 갱신 후 memcheck | 두 모드의 node encoder weight를 동일하게 변경 후 재평가 | 새 weight 반영·출력·gradient 일치, GPU 메모리 오류0 |

독립 모델 검사는 실제 production 모델과 checkpoint를 사용하지만 observation은 유효한 형식의
합성 입력이며 optimizer step이나 PhysX simulation은 없다. Full-FP32 비교에서 최대 출력 차이는
1.91e-6, gradient 차이는1.70e-6였다. Encoder weight 갱신 검사는 출력2.03e-6,
gradient1.70e-6 이내였다. 두 검사 모두 Compute Sanitizer `ERROR SUMMARY: 0 errors`로 종료했다.

처음에는 기존 학습과 같은 TF32 설정으로 비교했다. Critic 출력의 최대 절대 차이0.0045967 때문에
미리 정한 일치 기준(atol5e-4,rtol2e-3)을 통과하지 못했다. 그 구간의 memcheck 오류는0이었다.
기준을 완화하지 않고 TF32만 꺼서 재검사했으며, 위의 full-FP32 결과를 얻었다. 이는 관찰된 출력
차이가 정밀도·연산 결합 차이와 일치한다는 단서이며 **TF32가 Xid69의 원인이라는 결과가 아니다**.
장기 대조 학습의 TF32 설정은 기존대로 유지한다.

짧은 환경 검사는 checkpoint2250을 새 장면에서 재개한 안정성 검사다. 첫 rollout 뒤에는
`hand_velocity=1`, 평균 episode 길이1step이 관찰되어 학습 품질이 좋은 결과로 해석하지 않는다.
환경 상태·RNG 전체를 복원한 동일 trajectory 재현도 아니다. 짧은 검사의 max_iterations 때문에
Time curriculum이 앞당겨지지 않도록 해당 검사에서만 Time을0으로 유지했다. 원래45000회 이전과
같은 지급 상태이며, 본 학습의 curriculum을 비활성화한 변경은 아니다. 위 시간은 이 높은 reset
빈도와 동기화 조건의 수치이므로 정상 학습의 처리량 비교에 그대로 사용하지 않는다.

### 장시간 대조 학습

검증 뒤 1024환경·seed42·90000회 예산으로 **새 학습**을 시작했다. 모델 컴파일만 끄고,
환경 graph 세 개·TF32·Time curriculum·모든 reward/termination/제어 설정은 기존대로 둔다.
Checkpoint2250 재개 검사의 결과물은 새 학습의 초기 weight로 사용하지 않는다.

실행 정보·PID·정확한 인수·소스 hash는 `training_launch.json`, 전체 콘솔은
`training_model_eager.console.log`에 보존한다. 자동 재시작은 하지 않는다.
이 실행은 **재발 여부를 확인하는 대조 실행이며 원인 해결이 확정된 학습은 아니다**.
새 Xid69가 발생하면 컴파일만 제거해서는 해결되지 않는다는 근거로 남기고, 환경 graph·동기화
진단과 SDK/driver 경로 조사를 이어간다. 짧은 검사 통과만으로 수 시간 뒤의 재발을 배제하지 않는다.

실제 시작한 실행은 `2026-11-06_12-40-40_stability_model_eager`, PID298879다.
초기 iteration0–4의 PPO update·checkpoint0 저장까지 확인했다. 이 구간의 높은
`hand_velocity` 종료율(0.9989–1.0)은 이전104절 실행의0–5회에도 동일하게 관찰되어,
컴파일 해제로 새로 발생한 현상으로 단정하지 않는다. 학습 품질·초기 reset 빈도 문제는
별도 평가 대상이며, GPU 오류의 재발과 구분한다. 이번 장기 실행의 최종 결과는 아직 없다.

## 106. 모델 컴파일 기본값 복원 (2026-11-06)

사용자가 learning time 증가를 검토한 뒤 Actor·Critic의 `compile_model`을 다시 `true`로
설정하도록 요청했다. 원본 `ContactActorCfg` 기본값을 복원했고 `ContactCriticCfg`도 이를 상속한다.
환경 graph 스위치·`triton.cudagraphs=false`·네트워크·PPO·reward 설정은 유지했다.
105절의 PID298879 대조 실행은 앞선 사용자 중단 요청으로 종료·GPU 점유 해제를 확인했다.
이번 변경은 다음 실행부터 적용되며 이미 실행 중인 학습의 설정을 바꾸지 않는다.
사용자가 학습 중이라고 명시했으므로 테스트·벤치마크·학습 실행은 하지 않고 파일만 수정했다.
Xid69의 원인 확정이나 컴파일 ON/OFF의 새 성능 비교 결과를 뜻하지 않는다.

## 107. 2026-11-06 — URDF importer 2.4.31 손상 캐시 복구

`contact_manipulation/scripts/train.py --num_envs 4 --headless`의 Hand URDF 변환에서
`ModuleNotFoundError: No module named 'isaacsim.asset'`가 발생했다. 설치된 Isaac Sim은
5.1.0.0이고, pip의 `isaacsim-asset` 기본 importer는2.4.30이다. Isaac Lab은2.4.31을 요구한다.
초기에는2.4.30 호환 경로를 검토했으나 사용자가 실제2.4.31 설치를 요청해 해당 임시 코드와
검사 파일을 모두 제거하고 Isaac Lab 원본을 복원했다.

추가 확인에서 실제 원인은 **손상된2.4.31 사용자 캐시**였다. Kit의 설치 경로가
`~/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311`을
가리켰으며, 파일198개 중197개가0바이트였다. `extension.gen.toml`613바이트만 남아 있고
`extension.toml`, Python 모듈, native plugin·binding도 비어 있었다. Kit는 이를 의존성이 없는
extension으로 활성화했지만 실제 `isaacsim.asset` 모듈은 제공하지 못했다.

캐시의 NVIDIA 배포 메타데이터에 기록된 다음 원본을 내려받았다.

`https://d4i3qtqj3r0z5.cloudfront.net/isaacsim.asset.importer.urdf-2.4.31%2B107.3.3.lx64.r.cp311.zip`

17,480,710바이트이며 SHA-256은
`665cc13d48842ac6d0c203359819c1d5c767a1d920fb8a00179e147ec2fb5201`이다.
ZIP CRC와 내부 package version2.4.31·Linux/Python3.11용 두 native library의 실제 내용을
검사했다. 기존 캐시는 `~/.local/share/ov/data/exts/urdf-importer-backups/`에 보존하고
검증한 패키지로 원래 캐시 경로를 교체했다. Kit의 기존 symlink는 유지했다.
복구한 config는4,504바이트, Python native binding은680,112바이트다.

**검증:** 실제 Kit에서 Isaac Lab 원본 `UrdfConverter`로 RH56E2 Hand URDF를 변환했다.
활성 extension ID는 `isaacsim.asset.importer.urdf-2.4.31`이며, 생성한 USD에서 tactile
링크17개와 mimic 제약6개를 확인했다. 기존 `HandUrdfBoundsTests` CPU4개도 통과했다.
단독 변환 probe는 위 검증을 완료한 뒤 normal shutdown 대기에서180초 제한으로 종료됐다.
이후 원본 학습 스크립트의 기본 shutdown 경로로 다음 실행을 완료해 별도로 확인했다.

```bash
/home/ssu/miniforge3/envs/env_isaaclab/bin/python -B -u \
  src/contact_manipulation/scripts/train.py \
  --num_envs 4 --max_iterations 1 --headless --run_name urdf2431_smoke
```

환경 생성·Git-bundled catalog 복원·초기 reset·PPO `Learning iteration 0/1`을 완료하고
exit0으로 종료됐다. 학습 구간은35.38초이며 전체 초기화 시간과 구분한다. 실행 결과는
`logs/rsl_rl/contact_manipulation/2026-11-06_17-57-41_urdf2431_smoke/`에 있으며
`model_0.pt`와 `params/env.yaml`, `params/agent.yaml` 생성을 확인했다.
학습 성능이나 장기 접촉 안정성을 검증한 결과는 아니다. 임시2.4.30 호환 수정은 남기지 않았다.

## 108. 2026-11-06 — 동일 Actor 관측을 사용하는 독립 GNN 비교 프로젝트

사용자 요청으로 `src/contact_manipulation_symmetric`을 현재 `contact_manipulation`에서
독립 복사했다. 기존 MLP `contact_manipulation_parallel`과 별개다. 새 Gym task는
`Isaac-Contact-Manipulation-Symmetric-UR5e-v0`, experiment/log 경로는
`contact_manipulation_symmetric`이다. 원본 runtime·기존 Parallel·Isaac Lab은 수정하지 않았다.
Asset 경로와 `CONTACT_MANIPULATION_*` 환경변수는 공통으로 사용하며,
초기 카탈로그의 manifest와4개 archive part를 같은 내용으로 독립 복사했다.

두 관측 그룹의 기하 term을 모두 `initial_scene_geometry`에 연결하고 current·previous
term도 동일하게 유지했다. 직접 호출하는 `critic_observation`과 `critic_scene_geometry`도
Actor cache를 읽는다. Actor의232D 전체 값·가시성·padding·현재 EEF frame·reset/부분 reset·
주기 비전 갱신을 Critic과 일치시킨다. 모델에 현재 GT OBB를 별도로 입력하지 않는다.
물리·reward·termination의 GT 처리와 내부 상태 계산은 원본을 유지한다.
Actor와 Critic의 GNN·attention·MLP·정규화 인스턴스는 독립적이다.
Critic 관측 계약을 `actor_observation_v1`로 구분해 기존 `tracked_initial_objects_v1`
checkpoint의 전체 resume와 play를 거부한다. 입력 폭만 같은 checkpoint를 섞지 않는다.

모델과 하이퍼파라미터를 제외한 학습 실행은 **Isaac Lab 기본 RSL-RL 예제**를 따르라는
추가 요청을 반영했다. 원본의 `AppLauncher → Hydra → gym.make → ManagerBasedRLEnv →
RslRlVecEnvWrapper → OnPolicyRunner.learn()` 순서를 유지하고, 공식 PPO·RolloutStorage·
GAE/return·minibatch update·logger·checkpoint를 사용한다. 자체 환경 step/reset override,
PPO subclass나 학습 loop를 추가하지 않았다. 기존 Time schedule은 공식 CurriculumManager,
센서 모니터는 native logger 기록을 사용하고 checkpoint helper는 계약 검사 후 runner.load에
위임한다. 이 흐름을 실제 checkout의 `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`와
대조하고 native API를 직접 실행하는 기존 학습 통합 검사로 확인했다.

**원본 설정 보존:** 12D wrench/Hand·232D 입력, 물리100Hz/정책50Hz·8초,
Actor vision interval0, GNN2층·64D와 head256/128·ReLU, Gaussian std1,
rollout36·epochs8·minibatches4·gamma.95·lambda.95·LR1e-3 adaptive·entropy.01·KL.02,
90,000 iteration·50회 저장·Time50% curriculum을 유지했다. Reward·종료·clutter·센서·
로봇·초기화 코드도 이름공간을 제외하면 동일하다. 이름공간 변경을 정규화한 파일 비교와
원본 파일 hash를 확인했다. 새 프로젝트 테스트의 과거 성공250/실패10·속도3 기대값은 현재
원본의 성공10/실패0·속도1 설정에 맞추거나, 비영 비용 검사를 위한 명시적 fixture 값으로
분리했다. 이를 위해 production 설정을 과거 값으로 되돌리지 않았다.

**CPU 검증:** 전체 MDP474개와 learning76개, 총550개 중 **527개 통과·CUDA/Kit 전용23개 생략**.
동일232D 값·물체 이동의 GT 입력 차단·새 가시성/padding·현재 frame·부분 reset·terminal 보존,
동일 관측 rollout 저장과 독립 모델 update·공식 PPO 메서드·checkpoint 왕복 및 GT 계약 거부를
포함한다. 초기 CPU 전체 검사에서 Warp cache의 read-only 경로 오류가 발생해
`WARP_CACHE_PATH=/tmp/contact_symmetric_warp_cache`로 실행했다.

**실제 학습:** 새 production `scripts/train.py`로4환경·`--max_iterations 1`을 실행했다.
환경 생성·reset·`Learning iteration 0/1`을 완료하고 exit0으로 종료됐다.
학습 구간은22.54초이며 전체 초기화 시간과 구분한다. 결과는
`logs/rsl_rl/contact_manipulation_symmetric/2026-11-06_18-18-09_symmetric_smoke/`이며
`model_0.pt`와 native env/agent YAML을 저장했다. 이는 실행 검증이며 관측 통일에 따른
학습 성능 향상이나 장기 안정성의 증거는 아니다.

추가로 해당 checkpoint를 공식 runner에 inference 모드로 로드해 실제4환경에서16step을
재생했다.23회 auto-reset 전후를 포함해 매 관측의 `policy`·`critic` 전체 `[4,232]` 값이
`torch.equal`로 일치했고, Actor/Critic parameter storage가 공유되지 않음도 확인했다.

## 109. 2026-11-06 — 원격 병합 후 Symmetric의 최신 reward·action 대조와 동기화

사용자 요청으로 통합 브랜치 `work/integrate-20261106-183634`의 HEAD `617d4d6`에서
최신 `contact_manipulation`과 `contact_manipulation_symmetric`을 비교했다.
108절의 최초 복사본은 원격 `8ca3bae` 이전 원본을 기준으로 만들어졌으므로, Git 병합만으로
새 프로젝트의 reward·reset 코드에 원격 변경이 반영되지 않았던 상태다.

**이번 반영:**

- `blocker_distance` term을 episode 최소거리 함수에서 현재17pad OBB 거리 함수로 복원했다.
  weight5·decay10·Target 보호 gate는 최신 원본과 같다. 과거 최소거리 helper와 buffer는
  원본처럼 비교용으로 보존했다.
- `blocker_relocation`을 추가했다. 현재 Blocker OBB 중심에서 초기 빈 convex 영역들까지의
  최소 XY 거리 d에 대해 raw `exp(-10*d)`, weight3이다. 영역 내부·경계는 raw1이며,
  유효 영역이 없으면0이다. Native weight×dt를 적용해50Hz에서 최대0.06/step이다.
  별도 safety gate·신기록 이력·`/dt` 상쇄는 없다. 101절의 초기 weight1·decay ln2/.1이
  아니라 **현재 실제 원본 cfg의3·10**을 기준으로 했다.
- `blocker_contact_hold` weight를0.4→3으로 동기화했다. 유지 pad+0.5×새 pad·Target gate와
  canonical17pad·force threshold·부분 reset history는 기존 수식을 사용한다.
- 새 공유 scene에서 초기 재배치 영역을1회 준비하고 SceneBatch의 region/count/valid를
  zero-stride broadcast로 공유하는 경로를 동기화했다. 부분 reset은 이 metadata를 재사용한다.
  TaskState의 보정 Blocker 중심 갱신과 초기 영역 validation·binding도 함께 반영했다.
- 원본의 transition/support/robot-start CUDA graph 진단 스위치와 CPU/GPU 분기 회귀를
  동기화했다. 세 기본값과 모델 compile_model=True는 유지한다.

**Action 확인:** 이름공간을 정규화한 source 비교에서 `ur5e_env_cfg.py`·`scene.py`·`sensors.py`
및 action 관측 이력 계산은 이미 원본과 동일했다. Sensor-frame wrench6 + Hand6의12D,
Force10N/Torque1N·m scale, SI-200-8 범위 clipping·sensor→COM moment 보정,
Hand scale.5·native limit mapping과 clipped action-rate를 그대로 사용한다.
추가 action 변경을 만들어 넣지 않았다.

Reward·events·PPO cfg·train/play 등은 이름공간을 제외하면 현재 원본과 동일하다.
남은 runtime 차이는 Actor 관측을 Critic에도 제공하는 observation cfg·helper와 model/loader의
`actor_observation_v1` 계약뿐이다. 원본 패키지·기존 Parallel·Isaac Lab은 변경하지 않았고,
원본27개 파일의 hash 보존을 확인했다. 공식 ManagerBasedRLEnv·RslRlVecEnvWrapper·
OnPolicyRunner/PPO·rollout·GAE·update·checkpoint 경로를 유지한다.

**CPU 검증:** MDP491개·learning76개, 전체567개 중 **544개 통과·CUDA/Kit 전용23개 생략**.
현재 거리와 과거 최소거리의 구분, polygon 내부 plateau·경계·최근접 영역·invalid/빈 영역,
1024행 host 조회 금지, 공유 scene 생성당1회 준비·부분 reset metadata 재사용, graph 스위치,
기존 대칭 관측·독립 모델·native PPO·checkpoint 계약 회귀를 포함한다.
복사한 과거 테스트 중 relocation weight1/decay ln2/.1과 Hold.4 기대값만 현재 원본3/10·3에
맞췄으며 production 설정을 테스트의 과거 값으로 되돌리지 않았다.

**실제 실행:** GPU compute process가 없는 것을 확인한 뒤 새 production train에서
4환경·`--max_iterations 1 --run_name symmetric_sync_smoke`를 실행했다.
`Learning iteration 0/1`·새 checkpoint 저장을 완료하고 exit0으로 종료했다.
학습 구간15.80초는 전체 초기화 시간과 구분한다. 결과는
`logs/rsl_rl/contact_manipulation_symmetric/2026-11-06_18-47-59_symmetric_sync_smoke/`이며
`Episode_Reward/blocker_relocation`의 실제 양수 지급도 확인했다.
이는 변경된 경로의 실행 검증이며 학습 성능·장기 안정성이나 CUDA 오류 해결의 증거는 아니다.

새 checkpoint를 native runner로 로드한16step 재생에서도20회 auto-reset 전후의
`policy`·`critic` 전체 `[4,232]` 관측이 bit-exact하게 일치했다. 해당 장면은 공유 영역12개를
가졌고, 실제 relocation/Hold weight3·현재 거리 함수 연결과 독립 모델 parameter storage도
검사했다. 영역 개수는 이 검증 장면의 결과이며 전체 scene에 대한 고정 개수가 아니다.

## 110. 2026-11-06 — 물체 접촉 시 손가락 역꺾임 재현과 native hand 제약 수정

**현재 상태:** 아래 강체 mimic 후보는111절의 사용자 요청으로 제거했다.
이 절의 강체0/0·10rad/s·motor envelope와 측정 결과는 현재 설정의 설명이 아니다.

사용자가 지정한 문제는 물체를 밀 때 손가락이 반대로 꺾이는 현상이다. 이번 수정은
`contact_manipulation`과 `contact_manipulation_symmetric`의 손 설정에 동일하게 적용했다.
모델·관측 계약·reward·PPO·12D action 및 공식 학습 흐름은 변경하지 않았다.

**확인한 원인과 변경:**

- 원본 URDF의12회전관절 하한은 모두0이다. Importer2.4.31은 일반 손가락 child 하한을
  −17.8686°, thumb3를−5.9622°, thumb4를−12.7497°로 넓혔고 기존 후처리는 이를 유지했다.
  종속6관절의 하한을 원본0°로 복원했다. 종속 상한·속도·drive와 mimic reference·gearing·
  offset은 보존한다. Master 위치 범위는 기존 URDF 연동 범위의 교집합을 유지한다.
- 하한만 복원한 대조에서는 child 대신 master가−16.89°까지 밀렸다. 실제 backend master
  하한도0이었으므로 설정 누락이 아니라 제약 수렴 문제였다. URDF 모터 속도
  .922254rad/s·2rad/s를 관절 전체의 solver brake로 사용하면 접촉/한계 보정까지 막는다.
  해당 solver 상한을10rad/s로 완화하고, 기존 모터 속도는 native
  `PhysxDrivePerformanceEnvelopeAPI:angular/maxActuatorVelocity`에 deg/s로 전달한다.
  이 API는 모터가 속도 경계 밖으로 더 가속하는 힘을 제한한다. 외력·mimic으로 생긴
  모든 관절 속도를 정격 속도에 강제로 묶지는 않는다. API가 없으면 명확한 오류로 중단한다.
- Native mimic6개를0/0 강체 연동으로 설정한다. 실제 gearing 속성의 namespace를 사용해
  일반 손가락 rotZ와 엄지 rotX에 적용한다. Importer의 엄지 `rot:*` compliance 이름 오류를
  그대로 따르지 않는다. 설치 API에는 mimic.GetName()이 없어
  `GetGearingAttr().GetNamespace()`를 사용한다.

Implicit PD20/1·1N·m, 종속 gain0/0, 기존 self-collision filter·collision mesh,
physics100Hz·policy50Hz·solver32/1은 유지했다. Python joint pose 덮어쓰기,
종속 action/속도 brake, 물체/손 self-collision 비활성화는 추가하지 않았다.

**비교 과정:** 강체 mimic만 적용하고 기존의 좁은 속도 brake를 유지한 대조는 불안정했다.
Solver 속도를 사실상 무제한으로 푼 강체 구성은1/3N 접촉 시험을 통과했지만 완전 닫기에서
불안정했다.25/1·100/1·1000/1 compliance, 시간 간격.005, solver64/4 등도 분리 비교했으며,
이번 최종 설정으로 채택하지 않았다. 최종적으로 모터 envelope와 유한한10rad/s solver
상한을 함께 둔 강체 구성이 접촉과 완전 여닫기를 모두 완료했다.

**동일1N 자유 구체 접촉 비교:** 고정 베이스·무중력 손,0목표, 반지름12mm·질량.2kg의
동적 구체를 검지 distal pad의 음의 flexion 방향으로 가압했다.200step 무하중 후
10step ramp를 포함한200step 접촉 구간을 비교했다. 임시 동일 fixture/physics 조건에서
기존 설정과 최종 설정을 각각 새 프로세스로 실행했다.

| 측정 | 기존 | 수정 후 |
|---|---:|---:|
|12관절 중 URDF 하한 최대 이탈|−17.8686°|−.1245°|
|구체 접촉 frame /200|12|199|
|최대 관절 속도|5.5494rad/s|.4583rad/s|
|최대 mimic 오차|.29317rad|.002357rad|

기존 최대 하한 이탈은 index2, 수정 후 작은 잔여 이탈은 index1에서 발생했다.
원본 결과는 `/tmp/hand_contact_baseline.json`, 최종 동일 조건 결과는
`/tmp/hand_final_matched_1N.json`이다. 이는 한 fixture의 비교이며 실제 로봇의 강성 보정이나
임의 하중에서 완전한 한계 준수를 입증하지 않는다.

**현재 production 재현 도구:** `src/contact_manipulation/tests/probe_hand_contact.py`를 추가했다.
실제 프로젝트의 손·simulation cfg를 import하고12관절·6연동·17pad, native 속성,
단계별 접촉 frame 수와 궤적을 JSON에 저장한다. `--package contact_manipulation_symmetric`
으로 대칭 프로젝트도 검사한다. 탐색 중 사용한 물리 override는 이 도구에 넣지 않았다.
Guide는 기존 translate XformOp의 실제 precision으로 배치하고, physics 생성 전 위치 및
settle 후 guide 직선 이탈을 검증한다. 초기 임시 guide의 XformCommonAPI 배치 경고가 있었던
결과는 지속 접촉의 근거에서 제외한다.

수정된 production guide 검사의1N 구간은199/200frame 접촉·최대 하한 이탈−.1313°였다.
3N 구간은33/200frame만 접촉했고 최대 하한 이탈은−.6496°였다. 따라서3N을 끝까지
지지했다고 해석하지 않는다. 결과는 `/tmp/hand_final_index_corrected.json`이다.
강한 하중의 접촉 유지·미끄러짐/통과와 native solver의 잔여 오차는 추가 검증 대상이다.

실제 symmetric cfg의 무접촉 완전 닫기300step→열기300step도 측정했다.
하한 이탈은0이고 종료 시12관절 모두 절댓값1.2e−15rad 미만으로 열렸다. 닫기 중
원본 상한 초과.02027rad·mimic 오차.11820rad·종속관절 최고35.21rad/s는 남아 있었다.
결과는 `/tmp/hand_final_close_open.json`이다. 완료 상태가 모든 원본 한계의 엄격한 준수나
장기 안정성을 뜻하지 않으며, child 순간 속도는 모터 정격 속도로 제한하지 않는다.

**검증:** Original/Symmetric의 손 제약·모터 envelope·Arm actuator 회귀12개가 통과했다.
Envelope 미지원 오류, 실제 축의 mimic 속성, reference/gearing/offset과 passive 속성 보존을
검사했다. 최종 설정에서 Symmetric 전체 CPU 검사는546개 통과·환경 의존33개 생략,
597개 subtest 통과(18.82초)였다. 로그는 `/tmp/hand_fix_full_cpu.log`에 저장했다.
실제 원본4환경·1 iteration PPO 학습도 exit0과 checkpoint 저장을 완료했다.
학습 구간은16.52초이며 전체 초기화 시간과 구분한다. 결과는
`logs/rsl_rl/contact_manipulation/2026-11-06_19-45-39_hand_contact_fix_smoke/`이다.
이는 실행 검증이며 장기 학습 성능이나 기존 CUDA 오류 해결의 증거는 아니다.

## 111. 2026-11-06 — 강체 mimic 강제 제거와 원본 URDF 정의값 정합성 복원

사용자는 강체 mimic의 기존 문제를 지적하고 URDF와 동일한 모델을 목표로 요청했다.
110절 후보의 강체0/0·10rad/s 상한·motor performance envelope를 제거했다.
현재 기준은 **원본 URDF의 정의값 보존**이다. 이것을 접촉 안정성이나 이상적인 URDF
mimic 연동의 정확한 재현이 완료됐다는 뜻으로 사용하지 않는다. Original과 Symmetric에
동일한 손 설정을 적용했으며 모델·reward·PPO·학습 loop는 바꾸지 않았다.

**원본 확인:** ROAS vendor 원본과 `_relative.urdf`는 mesh 경로·공백을 정규화하면 동일하다.
32link·31joint 중12개 revolute·6개 mimic이다. 모든 revolute 하한은0, effort는1N·m다.
일반 손가락의 master/child 속도는1rad/s, 엄지4관절은2rad/s다. 모든 mimic offset은0이다.
`dynamics`·PD gain·mimic 강성/감쇠·motor envelope 정의는 없다.
[URDF mimic XSD](https://github.com/ros/urdfdom/blob/rolling/xsd/urdf.xsd)는 joint·multiplier·offset을
정의하며 PhysX의 compliance 수치를 지정하지 않는다.

| 구분 | 원본 native 위치 상한(rad) | 학습용 master 상한(rad) |
|---|---|---|
| 일반 손가락 master4개 |1.4381|1.36159181|
| 일반 손가락 child4개 |1.476374|독립 action 없음|
| thumb1 |1.658|1.658|
| thumb2 |.62|.609220762|
| thumb3 |.5410520681182421|독립 action 없음|
| thumb4 |.45553093477052004|독립 action 없음|

**구현:**

- `_configure_hand_joints()`는12관절의 native 하한·상한·속도·drive max force를 원문에서
  복원한다. Child의 importer 확장 상한과 무제한 속도도 원본으로 되돌린다. 종속관절의
  PD0/0·독립 action 없음은 유지하므로 effort 값1은 추가 구동을 만들지 않는다.
  이 값은 drive cap이며 mimic·충돌 제약의 모든 반력을1N·m로 제한하는 기능이 아니다.
- Native mimic6개의 reference·gearing·offset과 기존 양수 compliance를 보존한다.
  실제 namespace에 값이 없고 importer의 잘못된 `rot:*`에만 있으면 그 값을 그대로 복사한다.
  기존 active 값은 덮어쓰지 않는다. 강성/감쇠를 새 숫자로 정하지 않았고, 유한한 양수 값이
  없으면 강체 fallback 대신 오류를 낸다. 현재 설치 importer 값은25/약.005이며 URDF 값이 아니다.
- Native hard bounds를 좁혀 action 범위를 만들던 처리를 분리했다.
  `configure_hand_action_limits`를 공식 startup event에 등록해 독립6관절의
  `robot.data.soft_joint_pos_limits`만 연동 가능 교집합으로 설정한다. 공식
  `JointPositionToLimitsAction`이 이 배열을 읽으므로 기존 raw±2의 position mapping을
  유지하면서 물리 hard bound는 원문 그대로 보존한다. 부분 reset에서 물리 한계를 다시 쓰지 않는다.
- 질량·관성·joint frame/axis·원본 mesh는 바꾸지 않았다. 기존 convex decomposition과
  mounting overlap filter도 유지한다. Importer의 재회전된 native X축에 raw URDF 벡터를
  다시 덮어쓰지 않는다. Offline 비교에서32link 질량·COM 오차는 각각 최대3.58e−9kg·
  3.42e−9m,12관절 축 벡터 오차는4.27e−7 이하였다. 관성 재구성 최대 절대 오차는
  8.20e−10kg·m²였다. 이를 고친다는 명목으로 armature·질량을 추가하지 않았다.

**실제 backend 정합성:** `/tmp/hand_urdf_final_audit.json`에서 위치·속도·drive effort가 각각
12/12 일치하고 mimic reference·비율·offset도 각각6/6 일치했다(비교 허용 오차1e−6).
양수 compliance6/6, startup action 설정 전후 physical limits 동일, child soft limits 보존,
master soft limits 교집합 일치를 확인했다. Native gearing은 URDF multiplier의 음수인
PhysX 표현으로 비교한다. 현재 offset6개가 모두0이므로 이 값의 일치를 확인했으며,
향후 비영점 offset은 단위·부호를 추측하지 않고 감사 결과에서 비교 불가로 표시한다.

**남은 동작 문제:** 현재 source 설정 그대로의 실제 symmetric 접촉 결과는
`/tmp/hand_urdf_final_contact.json`이다.1N 구간에서200step 중57frame만 구체 접촉이 있었고,
최대 하한 이탈은−.267557rad(약−15.33°), 연동 오차는.290112rad였다. 따라서 강체를
제거하고 URDF 정의값을 맞췄다고 역꺾임이 해결된 것은 아니다.

Original의 닫기300→열기300step은 `/tmp/hand_urdf_final_close_open.json`에 기록했다.
종료 시12관절은 절댓값6.1e−9rad 미만으로 열렸지만, 닫는 동안 최대 mimic 오차는
1.012067rad(약58°), 상한 초과는.005261rad, 관절 최고 속도는2.347rad/s였다.
Compliant backend에서 URDF의 이상적인 동시 연동까지 정확하게 재현했다고 볼 수 없다.

URDF 값을 그대로 두고 solver 반복 수/TGS·PGS만 바꾼 임시 대조도 수행했다.
TGS128에서는1N 이탈이.000317rad까지 줄었지만3N에서−1.531028rad로 악화했다.
TGS64·velocity0·PGS128도 전 하중 안정성을 확인하지 못해 **production에 채택하지 않았다.**
현재 physics100Hz·policy50Hz·TGS solver32/1과 독립 PD20/1·종속0/0은 기존 값이다.
이 중 PD·solver·importer compliance는 URDF에 없는 제어/수치 설정이며 별도 검증 대상이다.

**검증:** 두 프로젝트의 관련 CPU22개와 native action mapping subtest가 통과했다.
Symmetric 전체 CPU는551개 통과·환경 의존33개 생략·596개 subtest 통과(27.77초)했다.
로그는 `/tmp/contact_hand_urdf_symmetric_cpu.log`다. Original 전체 검사는 기존의 누락된
예제 cfg 파일과 relocation/Hold 과거 weight 기대값3건으로 중단됐으며 이 요청에서 수정하지 않았다.
실제 Original4환경·PPO1 iteration은15.79초 학습 후 exit0과 checkpoint 저장을 완료했다.
결과는 `logs/rsl_rl/contact_manipulation/2026-11-06_20-35-12_hand_urdf_alignment_smoke/`다.
이는 startup 및 학습 실행 확인이며 역꺾임·장기 물리 안정성의 통과 판정이 아니다.

## 112. 2026-11-07 — 원본 장기 학습의 broadphase 오류·Xid32·CUDA719와 메인 프로젝트 전환

사용자가 원본 `contact_manipulation`의 새 오류를 제공하고, 이후 메인 프로젝트를
**`contact_manipulation_symmetric`**으로 지정했다. 이번에는 저장 로그·설정·시작 시 Git diff와
운영체제 커널 기록만 읽었다. 재현·학습·play·GPU 진단·테스트는 실행하지 않았고,
runtime·reward·제어·물리·driver 설정은 바꾸지 않았다. 메인 지정과 조사 결과만 문서에 반영했다.

### 실행 식별과 시간

- 첨부: `/home/hoon/.codex/attachments/f0056847-8194-404f-9174-f2787cb4cad9/Pasted text.txt`.
- 실행: `logs/rsl_rl/contact_manipulation/2026-11-06_13-53-02/`, PID **324627**.
  TensorBoard 파일명의 PID와 커널 Xid의 PID가 같다.
- Kit 로그: `/home/hoon/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261106_135246.log`.
- App 시작은 **11월6일13:52:46 KST**, 최종 실패는 **11월7일04:43:31 KST**다.
  첨부의 `2026-11-06T19:43:31Z`는 UTC이며, 경과53,444,897ms는 약14시간50분45초다.
- 저장 Git 기준은 `main`, commit `7f81f78474b5909a42a679e34f4465dc2e3c22f2`다.
  시작 당시 변경은 run의 `git/contact_rich_manipulation_task_learning.diff`에 보존되어 있다.
- 마지막 저장 checkpoint는 `model_10000.pt`(04:39 KST)다. 정확한 실패 iteration을 뜻하지 않는다.

### 최초 이상 징후와 최종 실패 순서

전체 Kit 로그에서는 첨부보다 먼저 다음 오류가 확인됐다.

```text
2026-11-06T12:04:11Z [25,884,910ms] [Error] [omni.physx.plugin]
PhysX error: PxgAABBManager::processFoundPairs: found null elements!
PxgAABBManager.cpp, LINE 921
```

Kit5379행이며 **11월6일21:04:11 KST**, 최종 장애보다 **7시간39분20초 전**이다.
GPU broadphase의 충돌 후보 pair 처리에서 내부 오류가 보고됐지만 즉시 프로세스가 끝나지는
않았다. 어떤 pair·환경·reset에서 발생했는지와 나중의 GPU 장애에 이어졌는지는 미확정이다.
이 오류가 이후에도 정상 물리 계산을 보장하는 경고였다고 해석하지 않는다.

최종 장애 직전 `/var/log/kern.log:1257–1258`와 `/var/log/syslog:108407–108408`에는
같은 Xid32가 두 번 기록됐다.

```text
Nov  7 04:43:30 hoon kernel: NVRM: Xid (PCI:0000:26:00): 32,
pid=324627, name=python, channel 0x00000014 intr 00200000
```

약1초 뒤 Kit5387행의 PhysX CUDA719가 발생하고, Articulation·RigidBody tensor 조회가
연달아 실패했다. Python은 `scene.update → joint_acc → joint_vel → get_dof_velocities()`에서
`Failed to get DOF velocities from backend`로 종료됐다. 종료 과정의 allocator68행 경고는
이미 CUDA context가 실패한 뒤의 기록이다.

### 의미와 미확정 원인

1. **이번 커널 오류는 Xid32이며103–104절의 Xid69가 아니다.** 99절의 Xid32와 같은 분류지만
   당시 intr/HCE 세부값이 다르므로 같은 근본 결함이라고 확정하지 않는다.
   [NVIDIA Xid32 설명](https://docs.nvidia.com/deploy/xid-errors/archive/index.html#xid-32-pbdma-error)은
   이를 driver–GPU 명령 전달의 PBDMA 오류로 분류한다. Driver·PCIe/시스템 경로가 중요한 조사
   대상이지만, Xid 숫자만으로 물리적 PCIe 고장이나 사용자 코드의 무관함을 입증할 수는 없다.
2. [CUDA719](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html)는
   device kernel 실행 중 예외를 뜻한다. 이후 CUDA 호출에도 같은 오류가 전파될 수 있으므로
   관절 속도를 읽는 Python 행을 최초 결함이나 관절 가속도 수식의 오류로 단정하지 않는다.
3. 앞선 `PxgAABBManager` 오류는 **PhysX GPU broadphase 경로를 별도로 조사할 단서**다.
   충돌 후보 자료·reset/kinematic 전환·SDK 내부 처리와 관련된 가능성을 남기되,
   실제 최초 실패 pair와 kernel이 없어 특정 전환 코드나 물체 메시를 원인으로 확정하지 않는다.
4. 읽은 Kit 로그에는 앞선 명시적 OOM·contact/patch buffer overflow·singular matrix 오류가
   없었다. 따라서 allocator 경고만으로 VRAM 부족을 단정하거나 buffer를 다시 늘릴 근거는 없다.
   오류 직전 메모리 사용량과 종속 관절을 포함한 Hand12관절의 상태는 저장되지 않아 정상이라고도
   보증할 수 없다. `intr 00200000` 역시 action·토크 값으로 해독하지 않는다.

실행 장비는 RTX3090, driver580.173.02 Open Kernel Module이며 Kit에 IOMMU 활성화 경고가
있었다. 근접한 커널 구간에 명시적 PCIe AER bus error나 GPU fallen-off 기록은 확인하지 못했다.
이는 driver/PCIe 원인을 배제하는 근거가 아니다. Driver·BIOS·IOMMU·clock은 변경하지 않았다.

### 실행 당시 설정과 최신 손 변경의 구분

저장 `params/env.yaml`·`params/agent.yaml`은 다음과 같다.

| 항목 | 실패 실행의 저장 값 |
|---|---|
| 환경 수 / 프로젝트 |1,024 / Original `contact_manipulation`|
| 물리 / 정책 |100Hz /50Hz (`dt=.01`, `decimation=2`)|
| Actor·Critic compile |둘 다 true|
| Transition / support / robot-start graph |모두 true|
| Actor vision interval |0, reset-only|
| Arm / Hand |IdealPD / Implicit, Hand master20/1·child0/0|
| 로봇 solver |Position32 /Velocity1|
| 활성 물체 수 |15–25|
| GPU contact / patch 용량 |8,388,608 /262,144|
| Found/lost pair 용량 / partitions |2,097,152 /1|

이 프로세스는 **107–111절의 오후·저녁 수정 전에 시작됐다.** 이미 import된 Python 모듈과
생성된 articulation에 파일 변경이 소급 적용되는 것은 아니다. 저장 diff에도 `scene.py` 변경이
없다. 따라서 최신111절의12관절 hard limit·thumb compliance 보정이 적용된 실행에서
동일 오류가 났다고 판단하지 않는다. 모델 compile과 환경 graph도 모두 켜진 실행이므로,
이번 결과만으로 그중 하나가 원인이라고 확정할 수 없다.

### Symmetric에서 이어갈 조사

현재 Original과 Symmetric의 `scene.py`는 동일하며 PhysX·로봇·센서·물체/초기화 경로를
공통 설계로 사용한다. Critic 관측을 Actor와 맞춘 것이 이 물리/GPU 위험을 제거하지는 않는다.
111절의 현재 손 접촉 시험에서도1N 유지57/200frame·하한 이탈약15.33°·연동 오차.290112rad가
남아 있다. 이는 별도의 실제 물리 문제지만 이번719의 직접 원인으로 증명된 것은 아니다.
상대 프로젝트의12관절 drive 방식은 이번 조사에서 적용하지 않았다.

후속 재현은 **Symmetric**을 기준으로 계획한다. 최초 broadphase 오류 시점의 전체 Kit 로그와
커널 Xid를 보존하고, reset 여부·선택 환경·활성/주차 물체·kinematic 전환·12개 Hand 관절의
실측 상태·목표·mimic 오차를 작은 진단 snapshot으로 연계하는 것이 우선이다. 모델 compile과
환경 graph 분리는 기존 스위치를 이용하되 별도 실험으로 비교하며, 이 기록에서 실행하거나
기본값을 바꾸지는 않았다. 다음 실행도 실제 source/diff·asset/importer·설정에 연결해서 판별한다.

## 113. 2026-11-07 — Symmetric의 관측·action·모델 입출력 정적 점검

사용자는 보상·학습 설계의 적절성과 별개로, 실제 observation과 action 경로가 의도대로
작동하는지 점검하도록 요청했다. 메인 프로젝트 `contact_manipulation_symmetric`을 기준으로
source HEAD `c73c5ff`와 공식 Isaac Lab action·OSC·actuator·Manager step 코드를 읽었다.
이번에는 시뮬레이션·학습·play·CPU/GPU 테스트를 실행하지 않았으며 runtime은 수정하지 않았다.
아래의 **코드 계약 확인**을 실제 물리 응답의 통과 판정으로 사용하지 않는다.

### 현재 관측 계약

| 전체 observation 구간 | 내용 | 표현 |
|---|---|---|
| `[0:100]` |10개 OBB의 위치3·quaternion4·전체 변 길이3|현재 sensor EEF frame, quaternion wxyz|
| `[100:166]` |현재 robot/sensor/action sample66|아래 구성|
| `[166:232]` |직전 sample66|동일 구성, reset한 행만0으로 초기화|

Sample66의 순서는 `[q12/π, qdot12/2, sensor_pose_in_base7, compensated_FT6,
tactile17, accepted_action12]`다. q/qdot는 Arm6과 독립 Hand6이고 mimic child6는 제외한다.
F/T는 sensor 중심·축 기준이며 distal half와 Hand 전체의 자중·관성을 뺀 뒤 SI-200-8 범위로
나누어±5로 제한한다. 측정 모델은 보상 전에 축별 saturation·해상도를 적용하므로 포화로 잃은
정보는 복구되지 않는다. 물체와 달리 EEF pose7은 sensor 자체의 **base 기준 pose**다.
이를 sensor 자신의 좌표계로 표현하면 항등 pose가 되므로 두 좌표계의 용도를 구분한다.
Accepted action은 sensor-origin/hand-angle 변환 전의 clipped raw policy 단위다.

Actor·Critic은 같은 세 관측 함수와 snapshot을 읽는다. 역할은 Target0·Blocker1 슬롯으로
고정하고 role ID·mask를 feature로 추가하지 않는다. Model은 extent·quaternion·tactile 값에서
mask를 만든다. 나머지 슬롯은 보이는 OBB 중 두 역할에 가까운 물체를 선택한다.
History는 control transition당1회 current→previous로 진행하며 부분 reset은 선택 행만 비운다.
Retained terminal observation을 제자리 수정하지 않도록 packed tensor를 교체한다.

현재 `actor_vision_update_interval=0`은 **reset-only**다. Cached OBB를 매 step 현재 sensor frame으로
변환하는 것과 물체의 관측 pose 자체를 새 ray로 갱신하는 것은 다르다. 주기를 켜면 가시성에 따라
일반 물체의 membership/order도 바뀔 수 있다. 물리100Hz·정책50Hz이므로 interval10은5Hz다.
이 값들은 현재 README와 일치하며 과거100Hz/10step 결정만으로 버그나 복원 대상으로 판단하지 않는다.

### 현재 action과 모델 계약

- Native Gaussian12D: sensor-axis `[Fx,Fy,Fz,Tx,Ty,Tz]`와 독립 Hand6이다. Tanh가 없고
  wrapper `clip_actions=None`이며 PPO는 원래 표본/log-prob를 보존한다.
- Arm raw bounds는±`[20,20,36,8,8,8]`이다. Scale `[10,10,10,1,1,1]` 뒤 각각
  ±`[200,200,360] N`, ±8N·m의 sensor-origin wrench가 된다.
- 이 설치 버전의 native OSC는 root-origin moment 변환과 COM Jacobian을 함께 사용한다.
  Project adapter의 원점 보정은 이를 sensor-origin wrench와 일치시킨다.
  보정 결과는 `tau_COM = R*tau_sensor + (p_sensor-p_COM) x (R*F_sensor)`이며,
  sensor 범위 clipping은 보정 전에 수행한다. 보정 뒤 moment를 다시±8로 자르면 다른 명령이 된다.
- `contact_wrench_stiffness_task=None`이므로 **OSC 내부**에서 F/T 오차를 보정하지 않는다.
  보상된 F/T를 정책 observation으로 전달하는 피드백은 이미 존재한다.
  공식 OSC가 `J_COM^T*w_COM + g`를 만들고 IdealPD가 합산 토크를150/28N·m로 제한한다.
  명령10N과 실측10N의 일치를 보장하지 않는다. Arm wrench0도 위치 유지·정지 명령이 아니다.
- Sensor frame은 policy 시작 시 읽어 base wrench로 변환하고 두10ms physics substep 동안
  유지한다. 두 번째 substep에서 회전한 sensor 축으로 명령을 다시 변환하지 않는다.
  이는 sampled-frame 명령이며 계속 회전하는 body-fixed wrench가 필요하면 별도 계약이다.
- Hand는 공식 absolute `JointPositionToLimitsAction`, scale.5다.
  `q*=lo+(clip(.5*a,-1,1)+1)*(hi-lo)/2`이므로 raw−2는하한, **0은중간각**, +2는상한이다.
  0은현재각유지나완전열림이 아니다. 현재 구성은 master6 PD20/1과 child6 mimic0/0이며
  논의한 전체12관절 drive 대안은 아직 적용하지 않았다.
- 실제 forward는 scene GNN·role attention → scene320D와 raw current/previous132D concat →
  native MLP256/128 → Actor Gaussian12D 또는 Critic value1D다. 별도 history encoder가 없고
  두 모델의 파라미터는 독립이다. 출력되는 module 등록 순서는 실행 순서를 뜻하지 않는다.

관절·슬롯 순서, 현재/직전 이력 배치, clipped previous action, sensor-origin 변환에서
source로 입증되는 산술·인덱스 오류는 이번 정적 점검으로 발견하지 못했다.
F/T inertia는 native107.3 API의 COM 기준·link-prim 축 tensor를 column-major로 읽고
transpose하는 방식과 맞는다. Incoming wrench는 native child-joint frame이며 sensor의 child
joint frame을 sensor 중심·동일 축으로 작성하므로 이 조건에서 직접 읽는 경로가 일관된다.
Local Isaac Lab property의 parent-frame 설명 및 `time_stamp` 표기는 native SDK 설명/버퍼
필드와 다르다. 이를 근거로 sensor 부호·축을 임의로 뒤집거나 이번에 framework를 수정하지 않았다.
대조한 원문은 [Tensor API107.3](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_inertias)와
[PhysX incoming joint force](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force)다.

### 확인된 검증 공백과 관측 한계

1. **Physics 이전 nonfinite 차단은 별도 구현하지 않았다.** Arm/Hand clamp는NaN을 없애지 않는다.
   `_transition_tensors`의 finite 검사·history `nan_to_num`·invalid_state 종료는
   physics가 끝난 뒤 적용된다. 따라서 그것이 잘못된 입력의 PhysX 유입을 예방한다고
   해석하면 안 된다. 실제NaN action이 발생했다거나112절의 GPU 장애를 일으켰다는 증거는 없다.
   사용자 후속 확인에 따라 현재 invalid-state 종료를 방어 목적으로 유지하며 확장 우선순위에서 제외한다.
2. **저해상도 초기 role 관측의 최종 보장이 없다.** 초기 가시성 certificate는160×120,
   실제 observation fit은64×48·최소3점이다. Constructive 경로의 `observe_at_viewpoint`는
   항상 snapshot을 반환하고, reset은 연관 ID를 검사하지만 Target/Blocker 슬롯의
   유효성을 둘 다 요구하지 않는다. Certificate를 통과해도 저해상도에서3점 미만이면 역할
   슬롯이0일 수 있는 구조다. 현재 실제 발생률은 미측정이며 발생했다고 단정하지 않는다.
   Reset-only에서는 누락된 역할을 episode 동안 다시 인식하지 않는다.
3. **Play가 학습 당시 환경 의미를 복원·대조하지 않는다.** `play.py`는 checkpoint의
   모델 설정을 복원하지만 환경은 현재 cfg로 만든다. 고정 action contract 문자열과 차원
   검사는 실제 wrench/hand scale·vision interval·frame·F/T 보상·물리 설정 비교를 대신하지
   못한다. 같은 checkpoint가 다른 의미의 명령/관측으로 재생될 수 있다.
4. **종속 마디의 상태를 직접 관측하지 않는 것은 의도된 설계다.** Mimic이 이상적으로 맞으면
   master로 추정할 수 있지만111절의 실제 연동 오차가 큰 상황에서는 그 가정이 무너진다.
   현재 policy는F/T·tactile로 간접 반응할 수 있으나 child6의 실제각·속도를 받지는 않는다.
   실제 관측 가능한 독립 관절 상태만 사용한다는 사용자 기준을 따르며, 관측 누락 버그로 분류하지 않는다.
5. **Tactile binary는접촉 event의 완전한 기록이 아니다.** Canonical17채널은 올바르게
   매핑되지만 값은 마지막 physics sample의 `norm(net_forces_w)>.01N`이다.
   Filter는Blocker reward 구분에만 쓰며 observation은 전체 net contact force다.
   따라서 self/shelf 접촉도 포함할 수 있고, 같은 pad의 반대 방향 접촉이 상쇄되면 접촉이
   있어도0이 될 수 있다. History0·마지막 substep 읽기이므로20ms control 구간 중간의 짧은
   접촉을 누적해 제공하는 센서는 아니다. 구체적인 놓침 빈도는 물리 시험 전이다.
6. **`Action/Applied`는실현된힘·토크가 아니다.** TensorBoard는 accepted input을
   Arm SI·Hand raw 단위로 보여준다. Post-actuator joint torque나 실제 측정 wrench와 구분해야 한다.

기존 테스트는 행렬 변환·mask·부분 reset·native PPO 연결 등을 다루지만 이번에는 실행하지
않았다. Current code가 실제 sensor/actuator에서 동일한 수치 응답을 보이는지는 별도 검증이다.
Native Gaussian의 직접 학습 std가 양수로 제한되지 않는 기존 위험도 남아 있으나,
이는 GNN 입출력 순서 오류나 이번 관측 mapping의 실패로 분류하지 않는다.

### 후속 확인 순서

먼저 소규모 고정 action에서 raw→accepted→SI/Hand각→OSC raw torque→제한 torque→실측
12개 Hand 관절·sensor 상태를 같은 timestamp로 대조한다. Arm0의 중력 보상, 힘/토크 각 축의
부호·회전/원점 변환, Hand−2/0/+2의 의미를 독립적으로 확인한다. 다음으로 reset/부분 reset의
232D 값, role 슬롯 유효율, 실제 무접촉 자중·관성 제거, 첫/둘째 substep의 짧은 pad 접촉을
검사한다. Checkpoint의 저장 env.yaml과 현재 play cfg를 함께 비교해야 정책 실패와 설정 차이를
구분할 수 있다. 보상 weight·PPO·관측 차원을 임의로 바꾸어 통과시키는 방식은 사용하지 않는다.

불필요한 GT Critic tracking은 Symmetric에서도 매 step 계산되지만 실제 관측에는 연결되지
않는다. 이는 정리할 중복 연산이며 GT 유출로 판정하지 않았다. Physics/policy 모두100Hz라는
소스 주석 역시 현재100/50Hz 값과 다른 문서 오류로 기록하고 설정은 바꾸지 않았다.

## 114. 2026-11-07 — Symmetric 점검에 대한 사용자 기준과 F/T 이중 피드백 요구

113절의 정적 점검에 대해 사용자가 의도를 보충했다. 아래 결정이 이전 이슈 분류보다 우선한다.
이번에는 문서만 갱신했다. 실행 코드·설정은 수정하지 않았으며 시뮬레이션·학습·play·테스트도
실행하지 않았다. 아래 구현 방향을 적용 완료로 해석하지 않는다.

### 유지할 연구 전제와 관측

- `invalid_state`는 드문 비정상 상태를 위한 기존 방어 종료로 유지한다. NaN action이 현재
  반복 장애의 원인이라는 증거가 없으므로 physics 이전 차단을 새 우선 과제로 삼지 않는다.
- Target·Blocker는 상위 판단기가 제공하며 반드시 존재하고 인식된다는 연구 전제를 유지한다.
  113절의 저해상도 역할 슬롯 누락 가능성은 이 전제와 구현 사이의 계약 문제다. 발생률은
  미측정이며 GT로 누락을 채우거나 episode 재추첨으로 해결하지 않는다. 새 공유 scene의 실제
  snapshot 계약 확인과, 실제 observation ray grid까지 포함한 사전 인증의 일치가 수정 방향이다.
- Hand의 독립 6관절만 policy에 제공하는 것은 의도다. 현실에서 관측하지 못하는 mimic child의
  실제 상태를 observation에 추가하지 않는다. 물리적 mimic 오차의 진단과 입력 설계는 구분한다.
- Arm의 0 wrench가 정지를 보장할 필요는 없다. 정책이 센서 피드백으로 정지에 필요한 명령도
  학습한다. Sensor frame은 그 목적을 위해 선택했으며 pose 제어로 임의 복원하지 않는다.

### 확정한 F/T 피드백 구조와 현재 구현 차이

사용자가 선택한 기준은 **정책 피드백 + OSC 내부 F/T 오차 보정**이다.

현재 정책 피드백은 구현되어 있다. `mdp/events.py`의 transition은 incoming joint의 6축
측정값을 SI-200-8 모델로 보정하고 Hand·sensor distal 부분의 자중·관성을 빼서 observation에
넣는다. Actor는 이를 읽어 다음 wrench를 결정한다. 반면 `contact_wrench_stiffness_task=None`
이므로 현재 OSC는 목표 wrench의 feed-forward 경로만 사용한다. 따라서 부족한 것은 **내부 루프**다.

설치된 공식 `OperationalSpaceController.compute()`의 내부 피드백은 힘 3축만 지원한다.
측정 torque를 목표 torque와 같게 대입하므로 torque 오차는 0이다. 또한 공식 ActionCfg에서
gain을 켜면 제어 body 자체에 ContactSensor를 만들고 그 body의 `net_forces_w`를 사용한다.
이는 손가락 전체에 전달되는 하중을 읽는 현재 articulation-joint F/T와 다르다. 따라서 단순히
gain을 설정하는 것으로 사용자가 의도한 실제 6축 load-cell 피드백이 구현되지는 않는다.

확장 방향은 기존 `SensorFrameWrenchAction`에서 실제 보상된 6축 F/T로 wrench를 보정한 뒤
공식 OSC의 Jacobian·중력 보상·effort 제출과 IdealPD 상한을 계속 사용하는 것이다.
Isaac Lab controller core를 수정하거나 별도의 joint torque controller를 만들지 않는다.

구현 전 검증할 계약:

1. 목표와 측정값을 같은 sensor 중심·축·부호로 맞춘다. Incoming joint reaction과 외부 접촉력의
   부호를 동일하다고 가정하지 않는다. Force·torque 각 축에 대한 고정 명령 검증이 필요하다.
2. 물리 substep마다 최신 센서 값으로 보정한다. Policy transition 캐시의 직전 값을 단순 재사용하거나
   object geometry·vision까지 전체 갱신하지 않고, F/T 보상에 필요한 tensor만 일괄 취득한다.
3. Force·torque gain은 따로 설정하고, sensor-origin clipping과 COM moment 변환의 순서를 유지한다.
   현재 action/history는 제한된 **정책 목표값**이며 내부 보정된 wrench와 구분한다.
4. 새 gain, 보정 wrench의 상한, sampled-frame 명령과 센서 회전의 관계는 구현·검증 시 명시한다.
   목표 wrench가 측정값과 반드시 일치한다는 보장은 actuator 한계·접촉 조건 때문에 성립하지 않는다.
5. 센서 전용 캐시는 physics timestamp와 부분 reset 행의 유효성을 관리한다. Reset 직전 reaction을
   새 episode의 내부 루프에 재사용하지 않고 새 측정 전에는 feed-forward만 사용한다. 질량·관성·
   COM·link ID는 고정 캐시를 재사용하고, 마지막 substep의 센서 계산을 policy와 공유한다.
   Tactile 전체 조회·ray casting·reward·history 갱신은 내부 루프에서 수행하지 않는다.

정적 힘 평형에서 parent→child incoming wrench는 distal 외력의 반대 부호를 포함한다.
따라서 현재 `measured - self_load`를 외력이 Hand에 작용하는 방향이라고 가정해 임의로
반전하면 안 된다. 목표 wrench가 Hand에서 환경에 가하는 방향일 때에는 같은 방향의 residual이
될 수 있다. 이 유도는 payload 합산과 frame 작성이 맞다는 전제이므로 실제 축별 접촉 검증을
대체하지 않는다. 자유 공간에서는 보상된 측정값이 거의 0이어도 내부 보정이 비영 목표를
증폭할 수 있다. 접촉 상태에서만 보정을 켜는 조건은 별도 제어 설계이며 임의 추가하지 않는다.

### Hand 0rad와 raw action 0의 구분

URDF 하한 0rad는 현재도 물리·action limit에 구현되어 있다. 공식
`JointPositionToLimitsAction`과 `scale=0.5`가 normalized raw action을 관절 범위에 매핑한다.

```text
q_target = lo + (clip(0.5 * action, -1, 1) + 1) * (hi - lo) / 2
raw -2 -> lo(0rad), raw 0 -> 중간각, raw +2 -> hi
```

따라서 raw 0이 중간각인 것은 0rad 구현 누락이 아니다. Raw 0을 관절각 0rad로 쓰려면 action
의미를 별도로 변경해야 하며 기존 checkpoint 재생 의미도 달라진다. 이번에는 변경하지 않았다.

### Actor·Critic 독립 가중치와 play 설정

공식 Isaac Lab 예제와 로컬 Sweeping 예제 모두 Actor·Critic 파라미터가 독립이다. 현재 설치된
RSL-RL도 두 모델을 각각 생성하며, Symmetric의 독립 GNN·attention·MLP는 이 원칙과 같다.
공식 PPO optimizer 하나에 양쪽 파라미터를 등록하는 것과 파라미터 공유는 다른 개념이다.
각 파라미터의 Adam 상태는 따로 유지되고 Critic loss가 Actor에 직접 역전파되지 않는다.

현재 play는 모델 설정만 checkpoint에서 복원하고 환경은 현재 cfg로 만든다. 공식 Isaac Lab
play도 환경 YAML을 자동 복원하지 않는다. 필요하면 기존 `play.py`에서 학습 당시
`params/env.yaml`의 action·observation·물리·종료 계약을 복원·대조하고, num_envs/device/viewer
등 실행 옵션은 마지막에 적용할 수 있다. Resolve된 joint/body ID는 재사용하지 않고 이름으로
다시 찾는다. 과거 YAML의 slice 태그·함수 namespace도 처리해야 하므로 무조건 전체 덮어쓰기나
unsafe YAML load는 사용하지 않는다. 설정 복원만으로 과거 코드·자산 버전까지 복원되지는 않는다.

환경 기초가 고정된 뒤 reward만 바뀌면 동일 observation에 대한 deterministic 정책 출력은
바뀌지 않는다. 다만 termination·reset 변경은 episode 자체를 바꾸므로 재생 비교에 반영해야 한다.
이번에는 play 복원 기능을 추가하지 않았다.

### 정책 100Hz 검토 — 설정 변경 없음

| 설정 | Physics | Decimation | Policy | rollout 36의 physics 호출 / 모의 시간 |
|---|---|---|---|---|
| 현재 | 100Hz | 2 | 50Hz | 72회 / 0.72초 |
| Physics 유지 | 100Hz | 1 | 100Hz | 36회 / 0.36초 |
| Decimation 유지 | 200Hz | 2 | 100Hz | 72회 / 0.36초 |

Decimation 2를 유지할 때만 Physics 200Hz가 필요하다. 마지막 안도 rollout 길이가 36이면
iteration당 physics 호출 수와 PPO batch 크기는 현재와 같다. 따라서 iteration 시간이 단순히
두 배가 된다고 단정할 수 없다. 같은 모의 1초의 물리 계산은 늘고, 8초 episode의 정책 step은
400에서 800으로 늘어난다. 같은 iteration 수로 경험하는 모의 시간은 줄어든다. dt에 따른
접촉·센서 응답과 step 단위 할인·증가량 비용도 재검토 대상이며, 현재는 주기를 바꾸지 않았다.

## 115. 2026-11-07 — 최종 MLP의 ELU 비교 설정과 GNN 활성화 함수 조사

사용자가 Actor·Critic의 최종 MLP부터 ELU로 비교하는 방향에 동의하고, 일반적인 GNN에서
사용하는 활성화 함수도 확인하도록 요청했다. 메인 `contact_manipulation_symmetric`만 변경했다.
학습·시뮬레이션·play·GPU 실행은 하지 않았다.

### 변경 범위

- `config/ur5e/agents/rsl_rl_ppo_cfg.py`의 `ContactActorCfg.activation`을 `relu`에서 `elu`로
  변경했다. `ContactCriticCfg`가 상속하므로 독립 Actor·Critic 모두 최종 native MLP 은닉층에 ELU를 쓴다.
- GNN의 `ModelCfg.activation=relu`, node/message/update ReLU 및 Triton edge forward/backward는
  유지한다. Role attention의 Softmax와 action mean/value의 선형 출력도 유지한다.
- 물리 주기·action·observation·F/T 피드백·reward·termination·PPO hyperparameter는 수정하지 않았다.
  원본 및 parallel 프로젝트는 변경하지 않았다.
- 활성화 함수는 각 모델의 checkpoint metadata에 이미 포함된다. 과거 ReLU checkpoint는
  기존 play helper가 저장된 Actor·Critic head activation을 복원하므로 ReLU로 재생한다.
  반면 새 ELU 설정에 ReLU checkpoint를 그대로 resume하면 기존 계약 검사가 거부한다.
  기존 checkpoint의 의미를 바꾸거나 architecture/action contract 버전을 임의로 올리지 않았다.

### GNN 조사 근거와 해석

| 대표 모델 | 은닉 표현의 활성화 | 근거 |
|---|---|---|
| GCN | ReLU | [저자 구현의 첫 graph convolution](https://github.com/tkipf/gcn/blob/master/gcn/models.py#L145-L161), [PyG GCN 기본값](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html) |
| GraphSAGE | node update 및 pooling 내부 MLP에서 ReLU | [저자 aggregator 구현](https://github.com/williamleif/GraphSAGE/blob/master/graphsage/aggregators.py), [PyG GraphSAGE 기본값](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html) |
| GIN | 내부 MLP 은닉층 및 graph update에서 ReLU | [저자 MLP](https://github.com/weihua916/powerful-gnns/blob/master/models/mlp.py), [PyG GIN 기본값](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GIN.html) |
| GAT 원본 | hidden head 출력에 ELU, attention 점수에는 LeakyReLU 후 Softmax | [저자 모델](https://github.com/PetarV-/GAT/blob/master/models/gat.py), [attention 구현](https://github.com/PetarV-/GAT/blob/master/utils/layers.py) |

GCN의 마지막 분류 logits, GraphSAGE 마지막 aggregator의 바깥 활성화, GIN 내부 MLP의
마지막 linear 등은 은닉층 ReLU와 구분한다. 위 표는 모든 레이어 출력이 ReLU라는 뜻이 아니다.
대표적인 message-passing GNN과 공식 라이브러리 기본값에서는 ReLU가 흔한 선택임을 확인했다.
이는 전체 연구의 사용 빈도를 정량 조사한 결과나, 우리 강화학습 문제에서 ReLU가 더 낫다는
성능 근거는 아니다. 또한 우리 attention은 role-conditioned dot-product attention이므로 GAT의
LeakyReLU를 그대로 추가할 근거로 해석하지 않는다.

이번 비교는 GNN을 ReLU로 유지하고 Actor·Critic 최종 MLP만 Isaac Lab/Sweeping 예제의 ELU에
맞추는 것이다. 학습 효과와 실제 learning time 차이는 아직 측정하지 않았다.

### 검증

기존 설정 비교 테스트의 ELU 예외를 갱신했다. 해당 테스트 메서드만 AST로 추출하여 표준
라이브러리로 실행한 **정적 설정 검사 1개가 통과**했다. Torch·Isaac Sim을 import하거나
학습·GPU context를 생성하지 않았다. 전체 learning test suite는 실행하지 않았다.

## 116. 2026-11-07 — 학습 재개 전에 재확인할 attention-only128D 비교안

사용자가 현재320D readout과 attention-only128D를 동일 조건에서 비교하는 제안을 기록하고,
**검토를 마친 뒤 학습을 시작할 때 이 내용을 다시 알려달라**고 요청했다. 시간 예약이 아닌
학습 재개 전 확인 항목으로 CONTEXT의 최신 진입점에 기록했다.

현재 readout은 `[Target64, Blocker64, context1_64, context2_64, global_mean64]`의320D다.
주요 역할의 GNN 특징과 전체 평균이 최종 MLP로 직접 전달되므로, 정책이 attention context를
거의 사용하지 않아도 행동을 만들 수 있다. 이것이 현재 실제로 일어난다는 증거는 없으며,
attention이 선택을 담당한다는 설계 의도를 얼마나 활용하는지 확인할 비교 과제다.

비교 후보는 역할 조건부 query 두 개와 GNN을 유지하되 scene 출력에 context 두 개128D만
사용하는 것이다. 현재·직전 센서/로봇 이력132D를 붙이면 최종 MLP 입력은452D에서260D가 된다.
Target·Blocker는 여전히 query 생성에 사용되지만 직접 특징 전달 경로와 global mean은 없어진다.
균일 attention이나 요약 과정의 역할/기하 정보 손실 가능성도 함께 확인해야 한다.

비교 시 물리·관측·action·reward·termination·PPO 조건을 맞춘다. 최근 변경한 최종 MLP의 ELU는
양쪽에 동일하게 적용해야 비교 목적이 분명하다. Readout 변경은 새 학습과 checkpoint 계약
검토가 필요하며, 기존 checkpoint에 조용히 적용하지 않는다.

**상태: 비교안 기록만 완료. 모델 구조·학습 설정은 변경하지 않았고 실행하지 않았다.**

## 117. 2026-11-07 — Symmetric Hand의 URDF import·구동 설정 정적 재점검

사용자는 로봇 제어 검토를 시작하며 우선 Hand를 URDF에서 불러와 설정하는 경로를 요청했다.
Source·URDF·공식 converter를 읽고 표준 라이브러리 XML parser로 원문을 비교했다.
이번에는 USD 변환·Kit·시뮬레이션·학습·play·GPU context·테스트를 실행하지 않았으며
runtime 코드와 설정도 변경하지 않았다. 아래의 값은 설정 경로이며 접촉 동작의 통과 판정이 아니다.

### 입력과 공식 import 경로

- 기본 입력은 `assets/robots/Roas_provided_inspire_urdf/RH56E2 URDF/` 아래
  `RH56E2_R_2025_9_11/urdf/RH56E2_R_2025_9_11_relative.urdf`다.
  `CONTACT_MANIPULATION_HAND_URDF_PATH`로 변경할 수 있다.
- Vendor 원문과 relative 파일은 mesh filename의 directory 부분을 제외한 XML 내용이 같았다.
  32link·31joint(고정19/회전12), tactile17개, mimic6개, 명시 질량 합0.776662673kg이다.
- 검증 함수는 URDF/mesh 존재·Git LFS pointer·17개 tactile 이름·pad collision mesh를 확인한다.
- 공식 `UrdfFileCfg`와 `UrdfConverter`로 변환한다. `merge_fixed_joints=False`로 tactile fixed
  link를 보존하고 `fix_base=False`로 손 자체를 월드에 고정하지 않는다. 이후 손의 중복
  articulation root를 제거하고 sensor mount를 거쳐 UR5e의 articulation으로 결합한다.
- `convert_mimic_joints_to_normal_joints=True`라는 이름과 달리 설치된 converter는 이 값을
  그대로 `set_parse_mimic(True)`에 전달한다. 현재는 native mimic을 보존하는 설정이다.
- `joint_drive=None`은 converter의 일괄 drive 재설정을 건너뛴다. 이후 공식 actuator에서
  gain을 설정하므로 최종 drive가 없다는 뜻은 아니다.
- 현재 converter는 Sim5.1 이상에서 importer2.4.31을 활성화한다. 고정 `usd_dir`를 지정하지
  않아 기본 임시 변환 directory를 사용한다. 변환된 타 PC USD를 항상 재사용하는 경로가 아니다.

### 관절·drive·mimic의 실제 구분

URDF에는 revolute12개의 위치·속도·effort와 mimic 관계가 있다. PD gain·joint dynamics·
transmission·mimic compliance 값은 없다. 모든 하한은0rad, effort는1N·m다.

| 관절 | 원문 위치 상한(rad) | 원문 속도(rad/s) | 연동 |
|---|---:|---:|---|
| thumb1 |1.658|2|독립|
| thumb2 |.62|2|독립|
| thumb3 |.5410520681182421|2|thumb2×.8392|
| thumb4 |.45553093477052004|2|thumb3×.891|
| 일반 손가락 master4개 |1.4381|1|독립|
| 일반 손가락 child4개 |1.476374|1|각 master×1.0843|

모든 mimic offset은0이다. Post-import `_configure_hand_joints()`는12관절 hard 위치·속도·
drive effort를 원문 값으로 복원한다. 독립 관절 drive type은 force로 지정하고, 종속 관절의
reference/gearing/offset/compliance는 유지한다. 엄지의 `rot`/`rotX` namespace 문제는 기존
값을 복사해 보정하며, compliance가 없으면 임의 강체 값으로 대체하지 않고 오류를 낸다.

공식 `ImplicitActuatorCfg`는12개 회전 관절을 그룹으로 선택하지만, 독립6개에만 Kp20/Kd1을
설정하고 종속6개는0/0이다. 종속 관절은 native compliant mimic으로 움직인다. 따라서
**mimic 제거 후 전체12관절에 multiplier 기반 drive를 넣는 방식은 아직 적용하지 않았다.**
Drive effort1N·m는 모든 mimic/contact constraint 반력의 상한을 뜻하지 않는다.

Startup event는 독립6개의 action용 soft 위치 범위만 child 제약의 교집합으로 좁힌다.
Thumb2 상한은 약.609221, 일반 master 상한은 약1.361592rad다. Physical hard 범위는 원문을
유지한다. Helper가 계산하는 coupled velocity envelope는 현재 적용하지 않는다.
일반 master1rad/s와 child1rad/s를 동시에 유지하므로 exact mimic 관계에서 master 최고속도는
child의 속도 제한과 간섭할 수 있다. 이는 관측한 속도 장애의 확정 원인이라는 뜻은 아니다.

### URDF에 추가되는 물리 설정과 센서

- Hand 질량·COM·관성을 별도로 덮어쓰지 않는다. 공식5.1 URDF ImportConfig의
  [importInertiaTensor 기본값은 True](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/structisaacsim_1_1asset_1_1importer_1_1urdf_1_1_import_config.html)이며
  local converter는 이를 변경하지 않는다.111절의 실제32link 비교 기록과도 일치한다.
  이번에는 실제 USD/PhysX 값을 다시 측정하지 않았다.
- 원본 mesh를 collider로 그대로 쓰는 것은 아니다. Importer는 convex hull을 만들고,
  palm1개는 최대16hull, 일반 손가락 부모 mesh4개는 각각 최대4hull로 추가 분해한다.
  이5개 collider의 contact offset만0.5mm로 명시한다. Tactile17개는 독립 rigid body로 보존한다.
- Importer 단계 Hand self-collision은False이지만, 결합 articulation 설정은True다.
  Palm housing2개×finger root5개의 장착부10pair는 별도로 collision filtering한다.
- `activate_contact_sensors=True`는 공식 subtree helper를 통해 Hand 전체32개 rigid link에
  ContactReportAPI를 설정한다. 실제 tactile reader의 대상은17개다. 추가 mount2half에도
  ContactReportAPI를 직접 붙인다. Reporting이 tactile에만 제한된다는 일부 주석과 차이가 있다.
  연산 비용 차이는 측정하지 않았다.
- 마운트는 URDF 외부에서 만든 길이5cm·반경2.5cm·총0.20kg proxy다. 두 half 사이 중앙
  fixed joint가 F/T 계측 기준이며 Hand base도 fixed joint로 연결한다.
- Hand에 별도의 physics material·탄성 pad 모델을 설정하지 않는다. URDF의 material은
  시각 색상이며, 실제 pad의 마찰·탄성·motor/기구부 특성을 보장하지 않는다.

중요한 구분은 **URDF의 관절·질량 정의 보존**과 **실제 손의 하중–변형 재현**이다.
PD20/1·importer mimic compliance·collision 근사·solver·마운트 질량은 별도 simulation 설정이다.
111절의 접촉 시 역꺾임·mimic 오차는 여전히 미해결 기록이며 원본 hard limit 복원만으로
해결됐다고 판단하지 않는다. 이번 검토는 이력과 현재 source의 일치 여부만 확인했다.

## 118. 2026-11-07 — Compliant mimic의 접촉 유지력과 강성·감쇠 미적용 의심 검토

사용자는 무하중 구동은 되지만 외부 접촉에서 자세 유지가 어려운 이유가 compliant mimic의
stiffness/damping 미적용 또는 버그인지 확인하도록 요청했다. Symmetric source·기존111절
시험 기록·공식 PhysX5.5.1/Omni Physics107.3 문서를 읽었다. 설정 변경·USD 변환·simulation·
학습·play·GPU 진단은 실행하지 않았다. 아래는 원인 후보의 분류이며 새 접촉 시험 결과가 아니다.

### Drive PD와 mimic compliance는 별도이다

`config/ur5e/scene.py`의 `ImplicitActuatorCfg`는 독립6관절에 PD20/1, 종속6관절에0/0을
설정한다. 종속 관절은 별도 모터 목표 대신 native mimic constraint로 연동된다. 따라서
Drive PD0/0이 mimic의 복원 강성·감쇠까지0이라는 뜻은 아니다.

Mimic은 `naturalFrequency`와 `dampingRatio`를 사용하는 별도의 spring–damper 모델이다.
현재 후처리는 실제 API namespace의 두 값이 유한한 양수인지 검사하고 importer 값을 보존한다.
기존97/111절의 backend 조사 값은25/약.005였다. 이번에는 runtime 값을 다시 조회하지 않았다.
기존 `/tmp/hand_urdf_final_*.json`은 현재 경로에 없어 원자료 재검증 없이 durable 기록을 사용했다.

자연주파수의 단위는 각주파수rad/s이므로25는 약3.98Hz이며, 감쇠비.005는 임계 감쇠의
0.5%다. 이 수치를 master Drive stiffness20과 직접 비교할 수는 없다. 실제 등가 강성은
해당 연동 방향의 articulation 유효 관성에 따라 달라진다. 감쇠를 높이면 진동을 줄일 수
있지만 정지 상태의 상대 속도가0일 때 damping만으로 정적 하중 변형이 사라지지는 않는다.
[공식 mimic 설명](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#mimic-joints),
[주파수 단위 설명](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html#drive-stiffness-damping)

### 접촉·한계·연동 제약이 경쟁하는 공식 사례와 일치한다

Omni Physics107.3은 작은 finger inertia·강한 master Drive·수동 mimic·손끝 contact의
경쟁을 직접 설명한다. Compliance는 이를 완화하지만 연동 오차를 허용하며, 조정 안내는
감쇠비1 미만을 피하도록 권한다. 현재.005는 이 관점에서 점검할 값이지만 원인의 확정은 아니다.
[공식 compliant mimic 가이드](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joint-compliance)

PhysX5.5.1의 Drive Stability 절은 손가락이 상자를 누를 때 proximal Drive·distal 하한0·
contact가 경쟁해 하한을 위반하는 사례를 설명한다. 반복 solver에서 뒤의 제약이 앞의 보정을
교란할 수 있어 모든 조건의 만족을 보장하지 않으며, 반복 수만 늘려도 해결되지 않을 수 있다.
Mimic보다 limit/contact가 나중에 처리되므로 연동이 양보할 수 있다. 따라서 URDF 하한0이
backend에 존재해도 접촉 중 음의 각도가 발생할 수 있다. 다만 이를 현실적인 유연성이나
허용 가능한 결과로 받아들이지는 않는다.
[공식 Drive 안정성 설명](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#articulation-drive-stability)

### 현재 구성에서 구체적으로 확인한 경쟁 조건

일반 손가락은 master/child 속도 상한이 각각1rad/s이고 multiplier는1.0843이다.
Master1rad/s를 정확히 연동하려면 child1.0843rad/s가 필요해 child 상한과 충돌한다.
두 상한과 연동을 동시에 지키는 master 속도는1/1.0843≈.922254rad/s 이하다.
`hand_joint_limits()`는 이 호환 속도를 계산하지만 startup의 `configure_hand_action_limits()`는
`positions, _`로 위치만 적용한다. 현재 `_configure_hand_joints()`는12개 모두 URDF 속도를 복원한다.

공식 gripper 가이드는 특히 hard mimic에서 child의 낮은 최대 속도가 불안정을 만들 수 있어
보통 master 상한이 gearing을 통해 전파되도록 하고 child 속도 상한을 별도로 좁히지 말라고
설명한다. Literal URDF 보존과 수치적으로 호환되는 연동 속도 설정은 구분해야 한다.
이 사실은 빠른 운동 중 제약 경쟁의 근거이며 정적 접촉 유지 실패의 단독 원인 증명은 아니다.
이번에는 상한을 변경하지 않았다.
[Maximum Joint Velocity](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html#maximum-joint-velocity)

### Importer 결함과 남은 불확실성

공식 Isaac Sim v5.1.0 공개 소스는 비기본축에서 `getJointAxis()`가 빈 문자열을 반환하고,
`configureMimicAPI()`가 rotX instance를 선택하면서 compliance 이름은 rot:*로 작성하는
경로를 갖는다. 공개 태그 importer2.4.30과 설치2.4.31은 동일 바이너리가 아니지만, 기존
엄지 USD에서 관찰한 namespace 불일치와 같은 기작이다. 현재 scene 후처리가 기존 값을
active namespace에 복사해 보정하므로 이 결함만으로 현재 접촉 문제를 설명할 수 없다.
[공식 importer 소스](https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/source/extensions/isaacsim.asset.importer.urdf/plugins/isaacsim.asset.importer.urdf/UrdfImporter.cpp#L772)

111절의 현재 구성 시험은1N 접촉57/200frame·하한−.267557rad·mimic 오차.290112rad였다.
무하중 개폐도 최종 복귀는 정상이나 과정의 최대 mimic 오차는1.012067rad였다. 따라서
무하중에서 완벽한 연동이 입증됐다고 표현하지 않는다.110절에서는25/1·100/1·1000/1 등도
비교했으므로 감쇠비만1로 바꾸면 해결된다고 주장하지 않는다. 과거 강체 후보도 속도 상한·
motor envelope 등을 함께 바꿔 개선했기에 강체화 하나의 효과로 분리할 수 없다.

다음 물리 검증에서는 같은 손·하중·목표에서 master 목표 추종 오차/Drive 포화와 child
연동 오차/물리 한계 위반을 따로 측정해야 한다. 호환 속도·compliance·Drive·dt/solver는
한 번에 하나씩 비교한다. 현재 판정은 **강성·감쇠 기능 누락은 확인되지 않았고, 낮은 감쇠와
제약 경쟁이 유력한 점검 대상이며, 새로운 엔진 버그나 단일 원인은 확정하지 못했다**이다.

## 119. 2026-11-07 — Symmetric의 Hand action 유지·mimic 제거·URDF 연동 Drive 적용

사용자는 action 출력을 유지하면서 Hand의 mimic을 풀고 multiplier를 고려한 관절별 Drive로
구동하도록 승인했다. 적용 대상은 메인 `contact_manipulation_symmetric`다. 원본·Parallel의
production 설정은 변경하지 않았다. 새 파일 없이 기존 scene·UR5e cfg·테스트·문서를 수정했다.

### 구동 경로

- 정책 출력은 기존 Arm wrench6+Hand6의12D다. GNN/attention·Actor/Critic·PPO·관측·
  accepted-action history·clipping·reward·termination은 이번 변경 대상이 아니다.
- 공식 importer에 `set_parse_mimic(False)`를 전달한다. 설치된 converter에서는 이름이
  혼동될 수 있는 `convert_mimic_joints_to_normal_joints=False`가 이 값에 해당한다.
  원본 URDF의 `<mimic>`을 삭제하지 않고 명령각 매핑에 계속 사용한다.
- `_configure_hand_joints()`는 PhysX mimic API가 남아 있으면 오류를 내고,12개 관절 모두
  force-type Drive를 사용한다. 위치·속도·effort 상한은 URDF 원문 그대로다.
- 공식 `ImplicitActuatorCfg`의 PD는 전12관절에20/1을 설정한다. 기존 master PD는 유지하고
  이전0/0이던6개 child Drive를 활성화했다. URDF effort는 각 관절1N·m다.
- `UrdfCoupledHandAction`은 공식 `JointPositionToLimitsAction`을 상속한다. Native6개
  입력의 scale.5·clipping·live soft-limit rescale·raw/processed action 처리를 재사용한다.
  그 뒤 `q_i_target = a_i * q_master_target + b_i`로12개 목표각을 펼치고 공식
  `set_joint_position_target()`으로 제출한다. PD·torque 계산을 Python으로 재구현하지 않는다.
- `hand_joint_mapping()`이 URDF chain을 처음에 한 번 해석한다. 엄지4의 목표는
  `.891 * .8392 * thumb2_target = .7477272 * thumb2_target`이다. Signed multiplier와
  offset도 누적하고, 순환·없는 reference·비유한 계수는 거부한다. 명령 목표각을 기준으로
  하며 접촉으로 밀린 master 실측각을 따라 목표를 물러나게 하지 않는다.
- Rollout에서는 미리 저장한 index·coefficient와12관절 target buffer를 사용한다.
  환경 전체에 대해 GPU index-select·multiply·add를 policy step마다 수행하고, physics
  substep에서는 target 제출만 한다. 환경별 Python 반복문·CPU 전송·새 IK는 없다.

### Reset·센서·관측과 물리적 한계

Hand action cfg의 `joint_names`는 기존 master6을 유지한다. 확장12관절 ID는 action 내부에서만
관리하므로 실제 관측 가능한 joint state12(Arm6+Hand6)와 action history의 차원·순서가 같다.
Startup의 master soft-limit 교집합도 유지한다. 기존 reset writer가 이미 전체 관절 pose·
position target·velocity/effort target을 함께 복구하며 현재 URDF의 offset은 모두0이다.
초기 Hand12개0rad와도 일치하여 reset·IK 코드를 변경하지 않았다.

`merge_fixed_joints=False`,17개 tactile 이름·메시·ContactSensor·충돌 근사·self-collision
필터·mount·F/T·UR5e actuator는 유지한다. Physics100Hz·policy50Hz·solver32/1도 그대로다.
공유 `probe_hand_contact.py`는 Symmetric의 새 방식에서는12개 목표를 실제 mapper로 제출하고,
원본을 선택하면 기존 native mimic·master6 구동을 유지한다. 새 모드 JSON에는 제어 방식·
12관절 목표·추종 오차·명령각 연동 오차를 기록하며, native mimic의 원문 일치 판정은
적용 불가(null)로 표시해 비어 있는 검사 결과를 통과로 오해하지 않도록 했다.

이 모델은 실제6모터 기구와 역학적으로 같다는 주장이 아니다.12개의 독립 Drive는 추가
구동 용량을 만들고, 실제 각도의 연동 등식이나 물리적 기어 반력 전달을 강제하지 않는다.
이전보다 잘 버틴다면 mimic 제거와 추가 Drive의 효과를 분리해야 한다. 새 PD20/1은 기존
master 수치를 사용하는 초기 설정이며, 실제 손의 하중–변형으로 검증한 값은 아니다.
Checkpoint tensor/action 규격은 같지만 구동 동역학은 달라 이전과 동일 조건의 재생은 아니다.
접촉 유지·역꺾임·물체에 전달되는 힘·연산 시간은 이번에 새 물리 실행으로 확인하지 않았다.

### 검증

관련 CPU16개가 먼저 통과했다. 이어 Symmetric의 `test_mdp.py`와 `test_learning.py` 전체는
**556개 통과·환경 의존33개 생략·596개 subtest 통과(44.41초)**였다. CUDA 장치를 숨기고
CPU thread1로 실행했으며, Kit·PhysX·학습·play는 실행하지 않았다.

검사는 원문 물리 한계·전관절 force Drive·잔존 mimic 거부·17개 tactile 보존, 원문 재귀식과
목표각 일치·엄지 chain·signed multiplier·offset·순환/없는 reference 거부, 실제 native
6D 변환 재사용과12개 target 제출, 측정각 변화와 명령각 분리, partial reset·다른 환경 보존,
host scalar/CPU 전송 없는 rollout, native actuator의12개 position target 전달을 포함한다.
Scene·action·공유 probe의 AST 구문검사와 `git diff --check`도 통과했다.

검사 로그는 `/tmp/contact_symmetric_urdf_drives_cpu.log`다. 최초 pytest 실행은 무관한 ROS
자동 로드 plugin의 `lark` 누락으로 수집 전에 중단됐으며, plugin 자동 로드를 끈 뒤
동일 두 파일 전체를 통과했다. 추가 설치나 production 설정 변경으로 우회하지 않았다.

## 120. 2026-11-07 — 고정 박스 접촉 시 Symmetric 손가락 역꺾임 실제 검증

사용자는 고정된 물체를 손가락이 누를 때 꺾이는지 실제 테스트하도록 요청했다. 기존
`src/contact_manipulation/tests/probe_hand_contact.py`에 `--fixed-obstacle` 모드를 추가했다.
새 production 파일·PD/물리 수치·reward·학습 코드는 변경하지 않았다.

처음에는 원본1024환경 학습PID324627이 GPU 약11.7GiB를 사용하고 있어 동시 실행이 자동 승인
검토에서 거부됐다. 사용자가 학습을 직접 종료한 뒤 진행하도록 답변했고, 해당PID 소멸·GPU618MiB·
compute process 없음이 확인된 후 실행했다. 기존 학습을 에이전트가 종료하거나 재개하지 않았다.
초기 launcher의 TERM=dumb 오류는 TERM=xterm으로 지정해 해결했다.

### 실험 구성과 판정 범위

- RTX3090·Isaac Sim5.1·GPU PhysX를 사용했다. 손만 fixed base·무중력으로 분리하고
  production의12관절 implicit Drive20/1·각 effort1N·m·URDF 위치/속도 한계·collision 근사·
  self-collision 필터·solver32/1·physics100Hz를 재사용했다. 실제 UR5e/센서 mount의 관성·
  중력·정책·보상·OSC는 이 격리 시험에 포함하지 않는다.
- 고정 상대는24×24×96mm kinematic 박스다. Native static actor와 구현 경로까지 같다고
  주장하지 않는다. Physics 전 초기 pose만 설정하고 이후 teleport·외력·pose overwrite가 없다.
- 선택한 motor와 URDF로 연결된 전체 체인의 양의 굽힘 방향을 합해 박스 방향을 정한다.
  선택 손가락 전체 충돌 mesh에서1mm 간격을 확보한다. 다른 손가락은 열린 상태를 유지한다.
  엄지는 opposition인 thumb1을0으로 유지하고 thumb2 flexion만 명령한다.
- 열린 상태100step→press400step(200step ramp+200step hold)→reopen200step→release100step,
  총800step/8초다. 목표는 해당 motor의 feasible soft 상한까지 올리며,12개 목표각은
  production `hand_joint_mapping()`으로 생성한다. 느린 ramp의 최대 명령 속도는 일반
  손가락에서 URDF 상한의.7382배, 엄지에서.1523배였다. 정책 rollout 주기를 재현하는 시험이
  아니라 physics step마다 부드러운 목표를 직접 제출하는 접촉·고정 명령 시험이다.
- 기존17개 tactile 외에 진단용 전체32body의 박스별 접촉 조회를 추가했다. 실제 관절각·
  속도·12개 명령각·URDF 연동 오차·body별 접촉·박스 위치를 매step 기록한다. 목표 추종
  오차 자체는 고정 장애물의 정상 저항일 수 있어 실패로 취급하지 않고, 실제 관절 하한 위반을 본다.

### 초기 fixture 보완과 제외한 결과

처음에는 선택 distal pad mesh의 support만 사용했다. 검지의 부모 `right_index_2` collider가
pad보다 돌출되어 settle부터 접촉이100/100frame 발생했고 약3.79° 하한 위반이 관측됐다.
이 값은 초기 간섭의 영향을 배제하지 못해 최종 결과에서 제외한다.
`index_initial.json`·`index_before_initial_clearance.json`에 당시 기록을 보존했다.

이후 선택 손가락 전체 충돌 mesh의 support로 박스를 옮겼으며, 최종5개 시험 모두 settle의
실제 접촉0/100frame·하한 위반0을 확인했다. 이 변경은 테스트 초기 배치만 보완한 것이며
학습 환경의 충돌 형상·contact offset·Drive 수치를 조정한 것이 아니다.

### 최종 실측 결과

아래는 **press 구간의 URDF 하한0 대비 최대 위반**이다. 마지막2초 hold의 접촉은 해당
손가락 body와 박스 사이의 filtered force가0.1N을 넘는 step으로 계산했다.

| 손가락 | 위반 관절 | 누르는 중 최저각 | Hold 접촉 | Hold 중 최대 하한 위반 |
|---|---|---:|---:|---:|
| 검지 | right_index_2_joint |−23.8043°|200/200step|23.8043°|
| 중지 | 없음(press 기준) |0° 미만 없음|200/200step|0°|
| 약지 | right_ring_2_joint |−23.8041°|200/200step|23.8041°|
| 소지 | right_little_2_joint |−12.2018°|200/200step|12.2018°|
| 엄지 | right_thumb_3_joint |−34.0133°|200/200step|34.0094°|

네 손가락의 위반은 접촉 순간만의 이상치가 아니라 hold 구간에도 남았다. 중지도 재개방 중
master에서 약−.29992°가 관측돼 전 과정에서 정확한 hard-limit 준수를 주장하지 않는다.
엄지는 재개방까지 포함하면 최대 약−34.07474°다. 마지막 열린 상태에서 모든 선택 관절은
0으로 복귀했다(최대 약4e−9° 수준).

최종5개 시험은 모두 complete·800step이며 native mimic0개·12관절 PD20/1·effort1N·m,
원문 URDF 물리 한계 일치를 backend에서 확인했다. 명령 multiplier/offset 잔차는 최대
1.04e−7rad였고, 박스 위치 drift는 최대5.32e−9m였다. Nonfinite state나 CUDA/PhysX 오류는
없었다. 따라서 명령 매핑 누락이나 박스 이동 없이도 역꺾임이 재현됐다.

일반4손가락의 접촉은 주로 각 `*_2` 본체 collider에서 발생했다(소지는`*_1`도 접촉).
이때 tactile17개가0인 것은 센서가 손가락 본체까지 감시한다는 의미가 아니기 때문이다.
엄지에서는 thumb3/4 본체와 force_sensor2/3도 접촉했다. 이 실험은 손가락 충돌 접촉 시험이며,
선택한 distal pad 표면만으로 누르는 시험이 검증됐다고 해석하지 않는다.
보고된 contact force는 body별 resultant이므로 단순히1N·m/레버암과 비교하거나 임의로 dt를
다시 곱해 실제 전달 힘을 추정하지 않는다.

**판정: 전관절 Drive로 전환했지만 접촉 시 역꺾임은 해결되지 않았다.** 정책·reward 없이
고정 목표만으로 재현되므로, 이 재현의 원인을 정책 action이나 학습 손실에 돌릴 수 없다.
Drive·관절 한계/속도 제한·접촉 solver의 경쟁을 별도 인자로 분리하는 후속 실험이 필요하다.
이번에는 production 수치를 더 조정하거나 손가락 pose를 강제로 덮어쓰지 않았다.

### 원자료·재현·회귀 검사

최종 원자료는 `logs/diagnostics/hand_fixed_contact_2026-11-07/`의
`index.json`, `middle.json`, `ring.json`, `little.json`, `thumb.json`과 각 console log다.
`summary.json`은 hold 접촉·위반 지속 step·명령 속도·원문 한계 일치를 요약한다.
[관절각 그래프](figures/hand_fixed_obstacle_2026-11-07.png)는 초기 간섭을 제거한5개 시험이다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/tests/probe_hand_contact.py \
  --package contact_manipulation_symmetric --fixed-obstacle --finger index \
  --target-flexion 1 --ramp-steps 200 --settle-steps 100 --plateau-steps 400 \
  --return-to-open-steps 200 --release-steps 100 --headless --device cuda:0 \
  --output /tmp/hand_fixed_index.json
```

새 fixture의 CLI·step 합산·선택 motor만의 ramp·짧은 ramp 뒤 연속적인 reopen·전체 chain
접선·돌출된 같은 손가락 housing과의 초기 간격·다른 손가락 제외·고정 상대에 외력/pose 쓰기
없음을 CPU5개 검사로 확인했다. Symmetric 전체 두 파일은 **561개 통과·33개 생략·
612개 subtest 통과(34.36초)**였다. 로그는 `/tmp/contact_symmetric_fixed_obstacle_cpu.log`다.

## 121. 2026-11-07 — 고정 물체 손가락 접촉 영상

사용자가120절의 시험을 영상으로 요청했다. GPU에 학습 process가 없는 것을 확인한 뒤
Symmetric의 같은800step 시험을 각 손가락별로 다시 실행하며 실제 RGB를 녹화했다.
저장된 관절각을 강제로 재생하거나 학습 checkpoint를 사용한 영상이 아니다.
Stiffness20·damping1·URDF effort1N·m·solver·dt·고정 박스·목표각 순서는 변경하지 않았다.

공유 `probe_hand_contact.py`에 선택적 `--video-output`과 `--video-fps`를 추가했다.
녹화를 지정하면 AppLauncher 전에 camera 기능을 활성화하고, 공식 `CameraCfg`/`Camera`
두 대를 이용해 전체 손과 선택 손가락의 접촉 확대 화면을 만든다. 내부 손가락은 옆
손가락의 가림을 줄이도록 확대 시점만 조정했다. 선택 손가락 청록색·박스 주황색의
visual material과 DomeLight를 추가했으며 collision·rigid body·drive 설정은 바꾸지 않았다.

Physics100Hz에서4step마다 physics를 진행하지 않는 `sim.render()`를 호출하고 RGB를
25fps MP4로 저장한다. Pillow overlay에 finger·close/hold/reopen 단계·simulation 시간·
실제 관절각·접촉 여부·URDF 하한 위반각을 넣는다. 하한을1° 넘게 위반하면 빨간색이다.
모든800step의 관절각은 기존 비녹화 시험과 **각 손가락 모두 정확히 일치했다(최대 차이0rad)**.
기록 시작 전 접촉도 모두0/100step이며, 박스 drift는 최대5.32e−9m였다.

산출물은 `docs/videos/hand_fixed_contact_2026-11-07/`에 있다.

- `all_fingers.mp4`: 검지·중지·약지·소지·엄지 순서,40초·1000frame·1280×800·25fps·약6.38MB.
- `index.mp4`, `middle.mp4`, `ring.mp4`, `little.mp4`, `thumb.mp4`: 각각8초·200frame.
- 각 clip과 합친 영상의 모든 frame을 실제 decode해 frame 수·duration을 확인했다.
  확대 화면도 각 손가락별로 확인했다. 실제 속도1×이며 느린 재생으로 바꾸지 않았다.

새 원자료와 검증 요약은 `logs/diagnostics/hand_fixed_contact_video_2026-11-07/`의
각 JSON과 `summary.json`이다. Console은 `/tmp/hand_fixed_contact_video_<finger>.log`다.
재촬영 중 한 wrapper가 종료 신호143으로 중단됐으나 CUDA/PhysX 오류는 기록되지 않았고,
남은 GPU process가 없는 것을 확인한 뒤 단일 시험으로 다시 완료했다. 원인은 확정하지 않는다.
최종5개 시험은 모두 complete이며 GPU process가 남지 않은 것을 확인했다.

재현은120절의 명령에 다음 옵션을 추가한다. 카메라는 이 진단 실행에만 생성된다.

```bash
--video-output /tmp/hand_fixed_index.mp4 --video-fps 25
```

CPU의 기존 고정 박스 회귀검사5개와16개 subtest가 통과했고 syntax·`git diff --check`도
통과했다. 학습·policy·reward·production Hand 설정은 이번 녹화 작업에서 수정하지 않았다.


## 122. 2026-11-07 — 접촉 영상에 프레임별 법선 접촉 힘 추가

사용자가121절 영상에 접촉 당시 힘을 표시하도록 요청했다. 기존 RGB와 같은 실행의
800step JSON을 이용해 CPU에서 주석을 추가했다. Isaac Sim·GPU·학습·물리 시험은
실행하지 않았고 기존 영상도 보존했다.

표시량은 공식 ContactSensor의 법선 접촉 합력 크기[N]다. `sphere_contact_force_N`은
박스에 작용한 전체 법선 합력의 norm, `hand_body_contact_force_N`은 Hand body와 박스
사이 filtered 법선 합력의 body별 norm이다. 마찰에 의한 접선힘·개별 접촉점 force·F/T
wrench가 아니다. 서로 다른 링크의 norm을 더해 박스 합력으로 해석하지 않는다.

영상 frame i(0부터)는 `samples[4*(i+1)-1]`에 대응한다. 첫 frame은0.04초·sample3,
마지막은8초·sample799다.4step 구간의 평균이나 최댓값 대신 촬영한 마지막 physics
step의 값을 표시했다. 박스 전체 합력은 공통0–12N 막대와 소수3자리 수치로,
링크별 합력은0.001N을 넘는 링크를 크기순으로 표시한다. 마찰·개별점이 제외됨도 화면에 명시한다.

기존 영상 아래160px 패널만 추가했고, encoding 전 원래800px의 모든 RGB 값이 그대로
유지되는 것을1000frame 모두 확인했다. 향후 녹화도 같은 `annotate_contact_forces()`를
호출한다. 기존 물리·제어 코드는 변경하지 않았다.

산출물은 `docs/videos/hand_fixed_contact_2026-11-07/all_fingers_with_force.mp4`다.
검지·중지·약지·소지·엄지 순서40초·25fps·1000frame·1280×960·약7.40MB이며,
개별 `*_with_force.mp4`도 각각8초로 보존했다. 합친 영상1000frame을 decode해
길이·해상도를 확인했고 검지·엄지의 실제 frame에서 힘 표시도 시각적으로 확인했다.
검증 요약은 `logs/diagnostics/hand_fixed_contact_video_2026-11-07/force_overlay_summary.json`이다.
syntax와 `git diff --check`가 통과했다.


## 123. 2026-11-07 — 300g 물체 밀기: 기본 마찰에서 실제 녹화

사용자가 움직일 수 있는 물체를 미는 테스트와 영상을 요청했고, 질량300g·기본 마찰을
지정했다. 공유 `probe_hand_contact.py`에 `--push-object` 모드를 추가했으며 학습이나
production Hand 설정은 바꾸지 않았다. 공식 Articulation·RigidObject·ContactSensor·Camera와
native material·rigid body API를 사용했다.

### 시험 구성

- Symmetric Hand의12개 mapped implicit drive·stiffness20/damping1·URDF effort1N·m·
  위치/속도 한계·solver32/1·physics100Hz·17개 tactile을 재사용한다. 실제 backend의12관절
  설정과 mimic0개·원문 한계 일치를 최종5개 실행 모두 확인했다.
- 박스는80×24×60mm·dynamic·중력−9.81m/s²이며 native mass readback은
  0.3000000119kg이었다. 정지·동마찰0.5/0.5·반발0·average combine은
  `RigidBodyMaterialCfg()`의 기본값을 그대로 썼다. 지지면에도 같은 기본 재질을 사용했다.
  기존24×24×96mm 박스의 전도와 미끄러짐을 혼동하지 않도록 밀기 방향 폭을 늘렸다.
- Hand 기부는 고정하고 자중을 제외했다. 각 손가락 chain의 초기 closing tangent를+X로,
  palm-to-tip 방향을 수직으로 맞추는 root orientation을 physics 시작 전에1회 설정했다.
  선택 손가락만 구동하며 실제 관절각·Box pose를 실행 중 덮어쓰거나 외력을 가하지 않는다.
  UR5e·OSC·학습 policy를 포함한 밀기 시험이 아니다.
- 박스는 수평 지지면 위0.5mm에서 시작한다. 초기 Hand–지지면 간격은 각 실행의 native
  authored mesh 기준26.23–27.62mm였다. 선택 손가락–박스 초기 간격은1mm다.
- 열린 상태1초→2초 ramp+2초 hold→재개방2초→열린 상태1초의800step이다.
  공식 Camera 두 대로25fps·200frame·8초씩 녹화했다.

### 지지면 필터 보완과 제외한 잠정 결과

최초 collision-only static 지지면은 GPU에서 `GPU contact filter for collider '/World/Support'
is not supported` 경고가 발생했다. 따라서 이때의 Hand–Table 접촉0을 비접촉 근거로 쓸 수
없음을 사용자에게 정정했고, 잠정 검지·중지 JSON/영상과 관련 로그를
`unsupported_static_support_filter/`에 보존했다. 최종 비교에는 포함하지 않는다.

지지면을 공식 `RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)`의
움직이지 않는 rigid actor로 만들었다. Physics 전 초기 pose만 설정하며 이후 이동 명령은
없다. 물체 질량·마찰·Hand 구동 설정은 유지했다. 이 구성으로5개를 모두 다시 실행했고,
최종 로그에서 unsupported filter 경고·CUDA/PhysX error가 없었다. 검지는 이전 잠정 실행과
관절각·물체 변위·접촉 힘800sample이 정확히 같았다.
연속 실행 wrapper가 한번 종료 신호143으로 중단됐으나 CUDA 오류는 기록되지 않았고
원인은 확정하지 않는다. 최종 실행은 clip별 process session을 분리해 순차 완료했다.

### 최종 결과

변위는 초기 Box pose 대비 XY 거리이며 누적 경로 길이가 아니다. 하한 위반은 누르는
4초 동안 선택 손가락의 실제 URDF 하한 대비 최대 위반각이다. 힘은100Hz 측정의 Hand↔Box
법선 합력 최대값이며,25fps 영상이 모든 짧은 peak를 보여주는 것은 아니다.

| 손가락 | 누르는 중 최대 XY변위(mm) | 8초 후 XY변위(mm) | 최대 하한 위반(°) | 최대 법선 힘(N) | 마지막2초 접촉 |
|---|---:|---:|---:|---:|---:|
| 검지 | 0.19 | 0.19 | 42.05 | 1.304 | 200/200 |
| 중지 | 22.99 | 22.99 | 35.53 | 2.182 | 135/200 |
| 약지 | 0.25 | 0.27 | 41.97 | 1.344 | 200/200 |
| 소지 | 21.60 | 38.74 | 31.70 | 1.698 | 200/200 |
| 엄지 | 3.70 | 3.71 | 37.82 | 1.988 | 200/200 |

최종5개 모두 settle100step의 Hand↔Box·Hand↔Table 접촉이0이었고, 전800step에서
Hand↔Table 접촉도0이었다(0.1N 기준). 박스 최대 기울기는0.011° 미만이었다. 따라서
이 결과를 박스 전도나 손의 지지면 접촉으로 설명할 근거는 없으며, 동적 물체를 접촉해
누르는 상황에서도 관절 하한 위반이 남았다. 물체가 움직인 것만으로 정상 구동이라고
판정하지 않는다. 원인이 특정 gain·effort·solver·bug 중 하나라고 확정한 시험도 아니다.

### 힘·영상·검증

박스의 native net force에는 약2.94N의 중량 지지력이 포함된다. 영상에는 이 값을 쓰지
않고 Hand body→Box filter0의 힘을 벡터로 합한 뒤 norm을 취한
`hand_box_normal_force_N`를 표시한다. 링크별 norm도 함께 보이되 서로 더하지 않는다.
마찰력·개별 접촉점 force는 포함하지 않는다. Hand↔Support는 filter1로 별도 기록한다.
기존 `sphere_contact_frames`는 지지면도 포함하므로 손가락 접촉 통계에 쓰지 않는다.

영상은 `docs/videos/hand_push_300g_2026-11-07/all_fingers.mp4`이며, 검지·중지·약지·소지·
엄지 순서40초·1000frame·1280×960·25fps·약15.06MB다. 각 손가락8초 clip도 보존했다.
물체 변위·기울기·실제 관절각·URDF 하한 위반·법선 접촉 힘을 함께 표시한다.
전체1000frame을 decode해 길이와 해상도를 검증하고 각 손가락의 실제 frame도 확인했다.
원자료·최종 console log·요약은 `logs/diagnostics/hand_push_300g_2026-11-07/`에 있다.

CPU 회귀검사6개·20개 subtest가 통과했다. 새 검사는300g push CLI·fixture 충돌 옵션과
push 중 외력·pose overwrite가 없는 것을 확인한다. syntax·`git diff --check`도 통과했고,
모든 테스트 종료 후 GPU compute process가 남지 않은 것을 확인했다.
재현 명령은 Symmetric README의 `--push-object --mass .3` 예제를 따른다.


## 124. 2026-11-07 — 관절 목표를 유지하고 Hand 전체를 이동해300g 박스 밀기

사용자가 손가락 joint state를 유지한 채 손 전체를 이동시키는 시험을 요청했다.
관절 상태를 강제로 덮어쓰면 접촉 변형이 감춰지므로 **12개 관절의 목표각을0rad로
유지하고 기존 PD로 자세를 유지**하는 시험으로 수행했으며 이 해석을 사용자에게 설명했다.
Production Hand·UR5e·OSC·reward·학습 설정은 수정하지 않았다.

### 시험 장치와 제어

공유 `probe_hand_contact.py`에 `--translate-hand`를 추가했다. `--push-object`와
`--target-flexion 0`이 필요하고 기본 이동량은100mm다. 기존 고정 물체/손가락 굽힘 밀기는
유지한다. Hand의 fixed root를 해제하고 floating articulation의 원래12관절을 그대로 둔다.
URDF에서 실제 root link를 찾아 world와 연결한 공식 `UsdPhysics.PrismaticJoint`를 만들고
`excludeFromArticulation=True`로 설정한다. Axis는 world+X이며 joint frame은 초기 root
pose와 맞춘다. Simulation reset 후 실제 floating base 여부를 검사한다.

기부는 공식 `UsdPhysics.DriveAPI`의 linear force drive로 이동한다. 장치 stiffness20,000N/m·
damping500N·s/m·maxForce200N은 기부를 움직이는 시험 장치의 값이며, Hand 각 관절의
PD20/1·effort1N·m를 바꾼 것이 아니다. Box에 외력을 직접 가하지 않는다. 매 step에는
이 장치의 위치 목표와 동일한12개 손가락 위치 목표만 제출하며, root pose·관절 위치/속도는
초기 설정 이후 덮어쓰지 않는다. Hand 자중 제외·Box 중력−9.81m/s² 조건을 유지한다.

열린 자세1초→cubic smooth profile로100mm 전진2초→유지2초→복귀2초→정지1초다.
기존 JSON phase 이름 `press`는 전진+유지, `reopen`은 이 모드에서 기부 복귀를 뜻한다.
Physics100Hz·800sample, 영상25fps·200frame·8초다. Box80×24×60mm·질량300g,
정지/동마찰0.5/0.5·반발0·평균 combine, 움직이지 않는 kinematic 지지면은123절과 같다.
UR5e·OSC·학습 정책을 포함한 시험이 아니다.

### 목표와 실제 상태 확인

최종5개 실행 모두800sample×12관절의 제출 목표가 정확히0rad였고 URDF 연동 목표 오차도0이다.
실제 backend에서12관절의 PD20/1·effort1N·m·원문 위치/속도 한계 일치와 native mimic0개를
확인했다. 기부 실제 최대 X 이동은99.819–99.992mm, 전체 궤적의 최대 X 추종 오차는
1.701mm 미만이다. 최대 YZ 이탈0.01782mm·회전 변화0.01279° 미만으로 직선 안내를 따랐다.
기부 추종과 손가락 자세 유지를 구분해야 하며, 아래 하한 위반은 실제 측정각이다.

| 초기 Box 위치 | 전진·유지 중 최대 XY변위(mm) | 복귀 후 최종 XY변위(mm) | 선택 손가락 최대 하한 위반(°) | 최대 Hand↔Box 법선 합력(N) |
|---|---:|---:|---:|---:|
| 검지 앞 | 7.47 | 7.46 | 51.51 | 1.452 |
| 중지 앞 | 14.02 | 13.92 | 50.66 | 1.662 |
| 약지 앞 | 82.41 | 82.41 | 5.11 | 1.962 |
| 소지 앞 | 86.37 | 86.37 | 3.85 | 2.094 |
| 엄지 앞 | 69.82 | 21.17 | 6.05 | 4.527 |

선택 손가락의 최악 관절은 검지/중지/약지/소지의1번 joint와 엄지3번 joint다.
이 표는 손가락 단독 성능 비교가 아니다. 손 전체 이동으로 중지 배치에는 검지,
약지 배치에는 중지, 소지 배치에는 약지도 접촉한다. 엄지 배치는 복귀 중 검지 접촉으로
Box가 되돌아오며, 전체12관절의 최대 목표 오차12.06°는 엄지가 아니라 복귀 중 검지1번에서
발생했다. 다른4개 배치의 전체 관절 최대 오차는 표의 선택 손가락 하한 위반과 같았다.

모두 초기 settle의 Hand↔Box 접촉0·전체 Hand↔Table 접촉0이며0.1N 기준이다.
마지막 유지2초의 Hand↔Box 접촉은 모두200/200step이다. Box 전체 최대 기울기는
중지 배치1.815° 미만, 나머지0.047° 미만이었다. 목표각을 바꾸지 않고 기부를 이동해도
검지·중지가 크게 뒤로 꺾이는 현상은 재현됐다. 특정 gain·effort·solver·bug 중 하나를
이 시험만으로 원인 확정하지 않는다.

### 영상과 검증

영상은 `docs/videos/hand_translate_300g_2026-11-07/all_fingers.mp4`다. 검지·중지·약지·소지·
엄지 앞 배치 순서40초·1000frame·1280×960·25fps·약19.40MB이며 개별8초 clip도 보존했다.
전체/접촉 확대 영상에 실제 관절각·URDF 하한 위반·물체 변위/기울기·법선 접촉 힘,
손가락 목표0rad·기부 목표/실제 이동량·전체 관절 최대 목표 오차를 표시한다.
힘은Hand→Box filtered force의 벡터 합 norm이며, 바닥 중량 지지력·마찰력·개별점 force는
포함하지 않는다.100Hz force peak가25fps 영상에 모두 나타나는 것은 아니다.

전체1000frame을 decode해 길이/해상도를 검증했고 각 배치의 실제 hold frame을 확인했다.
5개 console log에 unsupported contact filter·CUDA/PhysX error가 없었다. 테스트 종료 후
GPU compute process도 없었다. 원자료800sample·최종 console log·요약은
`logs/diagnostics/hand_translate_300g_2026-11-07/`에 보존한다.

CPU 회귀7개·24개 subtest가 통과했다. 새 검사는 CLI 조건과0rad 관절 목표 유지,
전진·복귀 장치 목표, motion 중 외력/root pose/joint state overwrite가 없는 경로를 확인한다.
Syntax·`git diff --check`도 통과했다. 재현 명령은 Symmetric README의
`--push-object --translate-hand --target-flexion 0`을 따른다.


## 125. 2026-11-07 — Hand stiffness20·50·100 동기화 비교 영상

사용자가 stiffness 비교 영상을 요청했다.124절의 손 전체 이동 시험을 기준으로 Hand의
12관절 Kp만20·50·100N·m/rad로 비교했다. Damping1N·m·s/rad·effort1N·m·URDF 위치/속도
한계·mimic 제거·충돌 형상·17개 tactile·physics100Hz·solver32/1은 유지한다.
300g Box·기본 마찰0.5/0.5·움직이지 않는 지지면·초기 배치·기부 drive20000/500/200N과
목표 궤적도 유지했다.12관절 목표는 계속0rad이고 기부만100mm 전진2초·유지2초·복귀2초,
앞뒤 정지1초씩 총8초다. 실제 관절각·root pose를 덮어쓰지 않는다.

공유 진단 `probe_hand_contact.py`에 `--hand-stiffness`를 추가했다. 옵션이 없으면 production
actuator 값을 상속하고, 지정하면 진단용으로 복사한 actuator의 stiffness만 변경한다.
유한 양수만 허용하며 `diagnostic_overrides.hand_stiffness`와 실제 native gain을 JSON에
기록한다. 학습용 actuator·PPO·reward·스폰 설정은 수정하지 않았다.

Kp20은124절의 완료된5개 JSON/영상을 원본 그대로 재사용하고,50/100은 각 배치마다
새로 실행해10개를 녹화했다. 모든15개에서 native12관절 gain이 지정값과 정확히 같고,
damping/effort/속도·위치 한계·초기 fixture·프로토콜·관절 목표·기부 명령이 일치함을 확인했다.
각 실행은800sample·200frame으로 완료했고 최초 Hand↔Box 접촉은 없었다.

### 측정 결과

전진·유지4초 동안 선택 손가락에서 발생한 실제 URDF 하한 위반의 최대값이다.

| 초기 Box 위치 | Kp20(°) | Kp50(°) | Kp100(°) |
|---|---:|---:|---:|
| 검지 앞 | 51.506 | 51.500 | 51.500 |
| 중지 앞 | 50.656 | 50.598 | 50.661 |
| 약지 앞 | 5.106 | 5.107 | 5.106 |
| 소지 앞 | 3.850 | 3.857 | 3.871 |
| 엄지 앞 | 6.053 | 6.171 | 6.139 |

같은 순서로1° 넘는 하한 위반이 지속된 시간은3.55·3.54·3.55·3.56·3.32초이며,
각 배치의 세 Kp 조건에서 동일했다. 최대각 하나만 큰 짧은 transient였다고 해석할 수 없다.

다음은 Box의 전진·유지 중 최대 XY변위 /8초 후 최종 XY변위다. 단위mm이며 누적 경로가 아니다.

| 초기 Box 위치 | Kp20 최대/최종 | Kp50 최대/최종 | Kp100 최대/최종 |
|---|---:|---:|---:|
| 검지 앞 | 7.47 / 7.46 | 7.46 / 7.46 | 7.46 / 7.46 |
| 중지 앞 | 14.02 / 13.92 | 14.16 / 14.06 | 14.01 / 13.91 |
| 약지 앞 | 82.41 / 82.41 | 82.41 / 82.41 | 82.41 / 82.41 |
| 소지 앞 | 86.37 / 86.37 | 86.37 / 86.37 | 86.38 / 86.38 |
| 엄지 앞 | 69.82 / 21.17 | 69.75 / 20.31 | 69.78 / 20.44 |

전15개에서 Hand↔Table 접촉은0이었다. 기부 최대 X 추종 오차는1.6746–1.7026mm이고,
마지막 유지2초의 Hand↔Box 접촉은 모두200/200step였다. 엄지 배치는 복귀 중 검지도
접촉해 물체가 되돌아오며, 전체12관절 최대 목표 오차는 약12°로 위 선택 엄지 표와 다르다.
Hand 전체12관절의 Kp를 변경했으므로 이웃 손가락의 반응도 함께 바뀐다. 배치 간 결과를
손가락 단독 유지력 비교로 해석하지 않는다.

**각 배치에서 damping1·effort1N·m를 유지하고 Kp만20→50/100으로 높여도 큰 하한 위반은
해결되지 않았다.** 현재 조건의 각1회 실행 결과이며 stiffness가 모든 상황에서 무효라는
주장이나 실제 torque 포화가 원인이라는 확정은 아니다. Implicit drive 토크를 측정한 시험도
아니다. 구동 토크 상한과 기계적 관절 위치 한계는 별도이므로 큰 하한 위반 조사를 계속해야 한다.

### 산출물과 검증

`docs/videos/hand_stiffness_2026-11-07/comparison.mp4`는 왼쪽부터 Kp20/50/100인
1920×960·25fps·40초·1000frame·약36.55MB 비교 영상이다. 검지·중지·약지·소지·엄지 배치를 각8초씩
동기화했고 frame i는 각 실행의 `samples[4*(i+1)-1]`에 대응한다. 실제 접촉 확대 RGB에
선택 손가락 관절각·하한 위반·Box XY/기울기·Hand↔Box 법선 합력·기부 실제/목표 이동량·
전체 관절 목표 오차를 표시한다. 힘은 마찰·바닥 지지력·개별 접촉점 힘을 포함하지 않는다.
Kp20 원본 재사용과 이웃 손가락 접촉 가능성도 영상에 표시한다.

원본50/100 영상은 같은 폴더의 `kp_50/`, `kp_100/`에 있고, 데이터·console log·전체 요약은
`logs/diagnostics/hand_stiffness_2026-11-07/`에 보존한다. 신규10개 console log에
CUDA/PhysX error·overflow·unsupported contact filter 경고는 없었다. 첫 순차 wrapper가
종료 신호143을 받았으나 별도 session의 검지100 시험은 정상 완료했다. 완료 자료를 덮어쓰지
않고 별도 recording driver로 나머지를 이어갔으며 `All 10 new recordings completed`를 확인했다.
이 종료 신호를 CUDA 오류로 분류하지 않는다. 녹화 종료 후 GPU compute process는 없었다.

CPU 회귀9개·36개 subtest와 syntax·`git diff --check`가 통과했다. 추가 검사는 CLI 유효성,
진단 actuator 복사 독립성과 stiffness 외의 설정 보존을 확인한다. 비교 영상 전체1000frame을
재디코딩해 길이·해상도를 확인하고 검지·약지·엄지의 실제 합성 frame도 시각적으로 검증했다.
재현은 README의 손 전체 이동 명령에 `--hand-stiffness 50` 또는100을 추가한다.


## 126. 2026-11-07 — 비교 repo gain1.2/0.06에서 Hand effort1·10 비교

사용자가 비교 repo와 stiffness·damping을 맞춘 뒤 effort 상한1/10N·m 비교를 요청했다.
공유 진단 스크립트에 `--hand-damping`·`--hand-effort-limit`을 추가해 복사한 actuator에만
적용한다. Kp1.2·Kd0.06을 고정하고12관절 effort 상한을1 또는10N·m로 설정한다.
학습용 actuator 설정은 바꾸지 않으며, URDF 원문 effort1N·m도 보존한다. 따라서10N·m
실행의 source effort 일치0/12·전체 일치False는 의도한 override다. 위치·속도 한계는 유지한다.
실제 native gain·상한과 요청값을 별도로 확인하며, 상한을 실제 출력 토크로 해석하지 않는다.

### 준비 구간의 CUDA719 재발과 기록 보완

최초 effort10 검지 실행은 `run_phase("settle") → advance → hand.update`에서
2026-11-07T12:15:22Z PhysX CUDA719(NpScene.cpp3000)가 발생했고, DOF velocity 조회 실패로
전파됐다. Process 종료 코드는134였고 cleanup에서도 abort가 있었다. 밀기 명령 전의
준비 구간이며 영상 frame은 생성되지 않았다. 오류만 기록된 JSON이므로 최초 시도의 native
상한·정상 sample을 사후 입증할 수 없다. 이 실행을 완료된 비교 결과에 포함하지 않는다.

이전에 학습에서 관찰한 것과 같은 오류 코드이지만 동일 근본 원인임을 입증한 것은 아니다.
이번에는 학습·PPO·GNN·OSC·대규모 scene reset 없이 Hand 단독 진단에서도 발생했다.
정책의 큰 action이나 큰 접촉력 때문에 발생했다고 단정할 근거가 없다. 최초 console/JSON과
driver log는 `logs/diagnostics/hand_effort_2026-11-07/failed_attempt_1/`에 보존했다.

진단에 첫 물리 step 전의 관절 위치·속도·관절 목표·native gain/상한·source alignment를
`.startup.json`으로 저장하고, 예외 시 이미 CPU로 복사된 sample·phase summary·시도 중인
phase/step을 error JSON의 `partial`에 보존하도록 보완했다. 오류 후 GPU 값을 다시 읽지 않는다.
정상 실행의 구동·스폰·물리 설정은 바꾸지 않는다. CPU 회귀11개·52개 subtest가 통과했다.

새 프로세스에서 effort10 검지 재확인은800sample·200frame으로 정상 완료됐고 오류 로그가
없었다. Native12관절 상한10N·m를 확인했으며 초기 최대 절댓값은 관절각2.51e−10rad·
관절속도4.40e−7rad/s였다. 재시도 완료가 CUDA719 원인을 해결했다는 뜻은 아니다.
검지의 effort1/10은 native 상한이 달라도800sample 전체가 같은 값이었고 최대 하한 위반은
51.505463°, Box 최종 XY변위는7.467810mm였다. 실제 implicit drive 토크 측정은 아니므로
이 결과를 특정 포화·solver 문제의 원인 확정으로 해석하지 않는다.

### 완료된 두 상한 비교

검지·중지·약지·소지·엄지 앞의5개 배치에서 각 상한1/10N·m를 순차 녹화해 정상10개
실행을 완료했다. 각 실행은300g Box·마찰0.5/0.5·손 전체100mm 이동의124절 프로토콜이며,
12개 관절 목표를0rad로 유지한다. Kp1.2·Kd0.06·위치/속도 한계·solver32/1·physics100Hz·
초기 배치·기부 drive는 양쪽에서 같고 effort 상한만 다르다. 이는 비교 repo의 gain을 적용한
현재 Hand 단독 진단이며, UR5e·OSC·학습을 포함한 비교나 그 repo 전체 물리 설정의 복제는 아니다.

**다섯 배치 모두 effort1/10의800개 sample 전체와 phase summary가 정확히 일치했다.**
별도 프로세스로 각각 실행했고 native 상한은 모든12관절에서 요청한1/10N·m로 구분됐다.
설정 값의 차이를 확인한 것과 실제 출력 토크가 달라졌음을 입증하는 것은 다르다.

| 초기 Box 위치 | 상한1 최대 하한 위반(°) | 상한10 최대 하한 위반(°) | 양쪽 Box 최대/최종 XY변위(mm) |
|---|---:|---:|---:|
| 검지 앞 | 51.505 | 51.505 | 7.469 / 7.468 |
| 중지 앞 | 50.597 | 50.597 | 14.159 / 14.057 |
| 약지 앞 | 5.105 | 5.105 | 82.409 / 82.407 |
| 소지 앞 | 3.856 | 3.856 | 86.385 / 86.383 |
| 엄지 앞 | 6.474 | 6.474 | 69.416 / 21.759 |

하한 위반은 전진·유지4초 중 선택 손가락의 최대값이며, XY변위는 초기 Box pose 대비
거리로 누적 경로가 아니다.1° 넘는 하한 위반 지속시간은 같은 순서로3.55·3.54·3.55·
3.56·3.32초였다. 두 상한에서 같았으며 마지막 유지2초의 Hand↔Box 접촉도 모두200/200step다.
Hand↔Table 접촉은 전실행0이고 최대 기부 X 추종 오차는1.675–1.704mm였다.
약지·소지 등에는 이웃 손가락 접촉이 개입한다. 엄지 배치는 복귀 중 검지도 접촉해 물체가
다시 돌아왔으며, 전체12관절 최대 목표 오차12.330°는 이때의 검지에서 발생했다.

따라서 **이번 Kp1.2·Kd0.06과 각 초기 배치에서는 effort 상한을1→10N·m로 올려도 역꺾임이
개선되지 않았다.** 이 결과만으로 effort 설정이 무시됐다거나 항상 의미 없다고 단정하지 않는다.
실제 implicit drive 토크는 측정하지 않았으므로 토크 포화가 원인인지도 확정할 수 없다.
큰 관절 하한 위반은 단순 목표 추종 오차와 구분해 계속 조사해야 한다.

### 영상·원자료·검증

`docs/videos/hand_effort_2026-11-07/comparison.mp4`는 왼쪽1·오른쪽10N·m,
1280×960·25fps·40초·1000frame·약25.09MB다. 다섯 배치를 각8초씩 같은 시점으로 맞췄다.
실제 접촉 확대 RGB와 관절각·하한 위반·Box XY/기울기·법선 합력·기부 목표/실제 이동·
전체12관절 목표 오차를 표시하며, 상한과 실제 힘을 구분한다. 법선 합력은 마찰·바닥 지지력·
개별 접촉점 힘을 포함하지 않는다. Frame i는 해당 실행의 `samples[4*(i+1)-1]`이다.

원본 영상은 같은 폴더의 `effort_1/`, `effort_10/`에, JSON·startup·console log·요약은
`logs/diagnostics/hand_effort_2026-11-07/`에 있다. 최초 실패는 별도 디렉터리에 보존하고
summary의 `excluded_failed_attempts`로 명시했다. 최종10개 정상 실행 로그에는
CUDA/PhysX error·unsupported contact filter 경고가 없었다. 녹화 후 GPU compute process는
없었고 합성은 CPU에서만 수행했다. 영상 전체1000frame을 재디코딩해 길이·해상도를 확인하고
검지·약지·엄지의 실제 합성 화면을 확인했다. CPU 회귀11개·52개 subtest, syntax와
`git diff --check`도 통과했다. 학습용 gain·effort 설정은 변경하지 않았다.


## 127. 2026-11-07 — 비교 repo와 Hand 접촉 역꺾임 원인 분리

사용자가 비교 repo와의 차이를 실제 대조해 원인을 규명하도록 요청했다. 대상 repo는
`7cmdehdrb/Contact-Rich-Manipulation`의 당시 main commit
`79b5abf8748da5f92e0dfc61613ede2ad1ba1478`다. 비교 repo의 Hand는12개 implicit drive,
Kp1.2·Kd0.06·armature0.0001이며 Relative task에서 effort10N·m를 사용한다.
Self collision은False다. 별도 오른손 USD는 총질량790g으로 보정하며17개 tactile body의
질량은 각각약0.899g이다. 우리 source URDF는 총약776.7g·링크별 질량과 관절 frame이
다르고, 네 손가락 distal 상한도약84.59°로 peer약179.91°와 다르다. 이 상한 차이만으로
음의 하한 위반을 설명하지 않는다. 두 repo의 전체 환경을 동일하게 복제한 시험은 아니다.

### Native drive와 목표 전달 확인

진단 스크립트의 native readback을 추가했다. 실제 backend `get_drive_types()`는
None0·Force1·Acceleration2를 구분하고 `get_dof_armatures()`도 읽는다. 매 sample에서
`get_dof_position_targets()`를 제출한12개 목표각과 비교한다. Joint enabled와 live USD의
angular drive type도 기록한다. Gain·상한의 숫자를 읽은 것만으로 활성 drive를 입증했던
이전 해석을 보완한다. 새로운 옵션은 복사한 진단 설정에만 적용하며 production은 유지한다.

새 baseline에서12관절 모두 실제 Force1·joint enabledTrue·live USD typeforce였고,
800step의 native 목표와 제출 목표 차이는0이었다. Armature는12개 모두0이었다.
따라서 이번 재현에서 drive 누락·목표 전달 누락·runtime acceleration 오설정 가설은 배제된다.
Importer cache의 초기 acceleration625와 live force override를 혼동하지 않는다.
Peer의 실제 binary physics layer도 CPU로 읽었으며12개 Hand drive는 force·초기gain0/0·
maxForce10으로 작성되어 있었다. 이것은 그 layer의 authoring 확인이며 peer runtime 측정은 아니다.

### 단일 변수 비교: 같은 검지·300g Box·100Hz

Kp1.2·Kd0.06·effort10N·m를 고정하고, 기존124절의 동일 검지 배치·마찰0.5/0.5·
손 전체100mm 이동·solver32/1·12관절 목표0을 사용했다. 각 실행8초·800sample,
전진2초와 유지2초를 press로 평가한다. 영상 렌더링은 생략했다.

| 조건 | Native armature | Self collision | Joint 속도 상한(rad/s) | Press 최대 하한 위반(°) | Box 최대 XY변위(mm) |
|---|---:|---|---|---:|---:|
| 현재 baseline | 0 | On | 원본1/2 | 51.505463 | 7.469000 |
| Self collision만 Off | 0 | Off | 원본1/2 | 51.505463 | 7.469000 |
| Peer armature만 추가 | 0.0001 | On | 원본1/2 | 0.001564 | 89.152191 |
| Armature추가+Self collision Off | 0.0001 | Off | 원본1/2 | 0.001564 | 89.152191 |
| 속도 상한만 진단용 완화 | 0 | On | 1000 | 0.509115 | 72.493131 |

Armature의 실제 FP32 readback은0.00009999999747이다. Native gain·effort·drive type·
목표·원본 위치 한계·fixture·기부 목표 이력은 대조 조건에서 확인했다. Self collision Off는
baseline과800sample 전체가 정확히 같고, armature+Off도 armature만 추가한 결과와 같다.
따라서 이 검지 배치에서는 self collision이 큰 역꺾임의 설명이 되지 않는다.
속도1000rad/s는 제약 경쟁을 분리하기 위한 진단값이며 학습의 권장값으로 적용하지 않는다.

큰 위반 관절은 proximal `right_index_1_joint`다. 최초 접촉은1.43초·약0.490N이고,
1.46초−1.29°·1.66초−10.24°·2.23초−40.07°·2.59초−51.505°로 진행한다.
Distal 최대 하한 위반은0.002624°다. 따라서 distal 상한의 peer/우리 차이를 주원인으로
삼는 해석은 이 재현과 맞지 않는다. Sample의 최종 qdot도 press 중 상한99%를 넘지 않아
속도 제한에 계속 포화됐다고 단정하지 않는다. 속도 상한만 바꾼 대조 결과로 영향을 판단한다.

### 확인된 원인과 해석의 범위

이 시험에서 **armature0인 Hand의 작은 관성과 drive·관절 한계·접촉의 제약 경쟁**이
주된 원인으로 좁혀졌다. Peer의 armature0.0001만 추가해 물리 위치 한계·속도 한계와
토크 상한을 유지하면서 역꺾임이 크게 줄었다. 관성 모델이 바뀌어 제약 계산이 개선된
것으로 해석하며, 단순히 solver 설정만 보정했다고 표현하지 않는다. 실제 solver 내부의
개별 impulse를 측정한 것은 아니다. Armature는 joint-space 관성에 추가되는 값이며
body 질량 변경과 구분한다.

공식 [Isaac Lab armature 설명](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.armature)과
[PhysX Articulation Drive Stability](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#articulation-drive-stability)는
각각 관성 추가에 의한 안정성 개선과 drive·위치/속도 한계·접촉 제약의 경쟁을 설명한다.
Gain·effort 증가만으로 해결되지 않았던125–126절과도 일관되지만 실제 drive torque는
여전히 측정하지 않았다. TGS의 마지막 substep qdot와 fullstep Δq/dt가 다를 수 있으므로
마지막 q/qdot로 계산한 단순 PD 근사값을 실제 torque·포화 증거로 쓰지 않는다.

### 기존 학습용 gain·effort에서의 재확인

Kp20·Kd1·effort1N·m·원본 속도 한계·self collisionOn을 유지하고 armature0.0001만
추가해 검지·중지·엄지를 재실행했다. 앞선124절의 production baseline과 같은 fixture·
protocol·관절 위치/속도 한계·기부 명령임을 확인했다.

| 배치 | 이전 최대 하한 위반(°) | Armature추가 후(°) | 이전 Box 최대 XY변위(mm) | 추가 후(mm) |
|---|---:|---:|---:|---:|
| 검지 앞 | 51.506248 | 0.001633 | 7.466173 | 89.139722 |
| 중지 앞 | 50.656091 | 0.006304 | 14.022949 | 88.588376 |
| 엄지 앞 | 6.052789 | 0.004616 | 69.818149 | 75.155775 |

하한 위반은 선택 손가락의 press 최대값이다. 새3개 실행은 nativeForce1·Kp20·Kd1·
effort1N·m·armature0.0001·12개 목표0을 실제로 확인했다. 이전3개 baseline은124절
저장 자료를 재사용했으며 당시 armature·runtime drive type은 기록하지 않았다. 당시 설정은
armature미지정이었다는 사실과 이번 새 baseline의 native0 확인을 구분한다. 이전 sample에서
측정하지 않은 armature 값을 사후 측정한 것처럼 표현하지 않는다. 약지·소지의 추가 armature
시험과 UR5e·OSC를 포함한 전체 학습 scene의 접촉 검증은 이번 범위에 포함하지 않는다.

### 결과·재현·검증

[원인 비교 그래프](figures/hand_cause_2026-11-07.png)의 위쪽은 새 검지 단일 변수 실험의
관절각·Box변위, 아래쪽은 production gain의3개 배치다. 하한 위반 bar는 log축이다.
원자료8개·각 startup/console·matrix/confirm driver log·summary는
`logs/diagnostics/hand_cause_2026-11-07/`에 있다. Summary는 모든 입력 불변 조건·native
설정·과거 자료의 기록 한계를 포함한다. 전체새8개가complete·800sample이고 CUDA/PhysX
error·unsupported filter 오류가 없었다. 종료 후 GPU compute process가 없음을 확인했다.

기존 README의 손 전체 이동 명령에서 `--video-output`을 생략하고
`--hand-stiffness 1.2 --hand-damping .06 --hand-effort-limit 10`을 공통으로 사용한 뒤,
`--hand-armature .0001` 또는 `--hand-self-collision off` 또는
`--hand-velocity-limit 1000`을 별도 실행에 지정해 matrix를 재현한다. Production 재확인은
gain·effort override를 모두 생략하고 `--hand-armature .0001`만 추가한다.

CLI·복사 설정 독립성·다른 물리 설정 보존의 CPU검증14개·83subtest가 통과했고,
분석 스크립트는8개 새 report와3개 과거 report의 불변 조건을 검증한 뒤 그래프·summary를
생성했다. Production Hand 설정에는 아직 armature를 추가하지 않았으며 UR5e·reward·모델도
이번 조사에서 바꾸지 않았다. 공식 `ImplicitActuatorCfg.armature`로 추가 관성을 설정하는
방향이 이번 재현에서 유효했다. 실제 회전자 관성과의 정합은 추가 검증 대상이다.
이 원인 분리를 과거 CUDA719의 원인 규명이나 해결로 확대하지 않는다.


## 128. 2026-11-07 — Hand 임시 물리 설정과 실물 비교 필요 항목

사용자는 공식 자료와 제공 URDF로 결정할 수 있는 값을 우선 따르고, 아직 확정할 수 없는
구동 파라미터는 합리적인 임시값으로 정한 뒤 추후 실제 Hand와 비교 실험이 필요하다는 점을
기록하도록 요청했다. 메인 프로젝트 `contact_manipulation_symmetric`의 첫 기준은 다음과 같다.

| 항목 | 임시 기준 | 근거와 한계 |
|---|---|---|
| 관절 위치 한계·multiplier·offset | 제공 RH56E2 URDF 유지 | 종속 관절 한계까지 만족하는 master 목표 범위를 사용 |
| 관절 effort 상한 | 12관절 각각1N·m | URDF 명시값이며 실제6개 구동기의 토크 배분을 확인한 값은 아님 |
| 관절 속도 상한 | 엄지2rad/s·나머지1rad/s | URDF 유지. 제품의 손가락 속도·닫힘 시간과 좌표·stroke 정의를 먼저 대조해야 함 |
| 링크 질량·COM·관성 | URDF 유지 | 제품 명목 총질량만으로 개별 링크 물리값을 재분배하지 않음 |
| Stiffness·damping | 20N·m/rad·1N·m·s/rad | 현재 직접 시험한 구성의 기준값이며 실물 동특성에 정합된 값은 아님 |
| Armature | 0.0001kg·m² | 127절의 접촉 비교 근거로 공식 `ImplicitActuatorCfg.armature`에 추가. 실제 회전자 관성으로 확인한 값은 아님 |

이번 적용은 Symmetric Hand actuator에 armature 한 항목을 추가하는 것으로 제한했다.
UR5e·reward·policy·action 변환은 변경하지 않았다. 127절의 "학습용 armature 미적용"은
당시 상태의 기록이며, 현재 적용 여부는 이128절을 따른다. 원본·parallel 프로젝트는 유지한다.

공식 [RH56E2 제품 사양](https://en.inspire-robots.com/product/rh56e2/)의 손끝 힘28–30N은
관절별 토크 상한으로 직접 치환할 수 없다. 17개 tactile 모델의 명목 질량790±10g과 제공
URDF 합계약776.7g도 모델 개정·부속품 포함 조건을 확인한 뒤 비교한다. 공식 손가락 속도와
URDF 관절별 속도 한계도 같은 좌표량이라고 가정하지 않는다.
[NVIDIA 구동 안정성 가이드](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#drive-stability)는
관성·고유주파수·감쇠비를 이용한 gain 설정 방법을 제공하지만 RH56E2의 권장 gain을 제시하지 않는다.

**추후 실제 Hand와 비교 실험이 필요하다.** 정확한 Hand 개정·제어 모드·입력값·접촉 조건을
맞추고, 자유공간의 목표/실제 관절각과 이동 시간, 중간 관절각에서의 접촉 하중에 따른 변형,
접촉 해제 후 복귀 응답을 비교한다. 이를 바탕으로 stiffness·damping·armature와 구동 한계를
다시 결정한다. 실제6개 구동기와12개 독립 drive의 힘 전달 차이, 하드웨어의 force protection도
함께 확인해야 한다. 지금 임시 모델을 실물 동등 모델로 표현하지 않는다.

127절 시험은 목표0rad가 물리 하한인 hard-stop 시험이므로 gain 최적화를 입증하지 않는다.
공식 근거가 없는 접촉력5N·변형5°·추가0.2초·overshoot5%·한계오차0.1°를 확정된
합격 기준으로 사용하지 않는다. 현재 damping1도 자유 추종을 지나치게 둔하게 만들 가능성이
있으므로 중간 자세 시험을 거쳐 검토한다. 이번에는 새로운 GPU·시뮬레이션·학습을 실행하지 않는다.


## 129. 2026-11-07 — Symmetric Hand action에서 native position drive까지의 경로 점검

사용자는 정책 출력이 실제 Hand 제어로 변환되는 경로가 설계 의도와 일치하는지 확인하도록
요청했다. `ur5e_env_cfg.py`의 action 선언·`UrdfCoupledHandAction`, startup soft-limit event,
체크아웃의 공식 `JointPositionToLimitsAction`·`ImplicitActuator`·`Articulation`을 대조했다.
이번 점검에서 action 의미·scale·범위·controller 구현은 수정하지 않았다.

### 12D 정책 출력의 Hand6D와 절대 목표각

전체 action의 앞6개는 sensor-frame Arm wrench이고 뒤6개는 순서대로 thumb1·thumb2·
index1·middle1·ring1·little1의 master position 명령이다. Gaussian 출력은 wrapper에서
별도 clipping하지 않는다. Hand의 `JointPositionToLimitsActionCfg`는 `scale=0.5`,
`rescale_to_limits=True`, `preserve_order=True`다. 입력은 delta·속도·토크가 아니라
절대 목표 관절각을 선택하는 정규화 좌표이며 실제 각도 변환은 공식 action이 수행한다.

Master의 허용 목표 범위를 [l,u], 정책 raw Hand action을 a라 하면 다음과 같다.

```text
s = clamp(0.5 * a, -1, +1)
q_target = (l + u) / 2 + (u - l) / 2 * s
```

| Raw Hand action | 허용 목표 범위에서의 위치 |
|---:|---:|
| -2 이하 | 하한0% |
| -1 | 25% |
| 0 | 중간50% |
| +1 | 75% |
| +2 이상 | 상한100% |

즉 `scale=0.5`는0.5rad 증분이 아니다. URDF 관절각0rad와 raw action0도 구분해야 한다.
초기/reset Hand 관절각과 native position target은0rad지만, 첫 policy action0은 그 자세를
유지하는 명령이 아니라 중간 자세로 이동하라는 명령이다. 열린 하한 자세를 요구하려면
현재 mapping에서 raw action -2 이하가 필요하다. 양의 action은 일반 손가락의 굽힘 목표를
키우고, thumb1은 별도 엄지 회전 축이다. 큰 값은 native mapping에서 포화한다.

Startup의 `configure_hand_action_limits`는 child까지 URDF hard limits를 만족하도록 master
soft limits만 교집합으로 설정한다. 실제 hard limits는 원본대로 유지한다. 유효 범위는
thumb1=[0,1.658]rad, thumb2=[0,0.6092207623]rad, 나머지4개=[0,1.3615918104]rad다.
예를 들어 index1의 원본 상한1.4381rad보다 index2 상한1.476374rad/1.0843이 작으므로
master action 상한을1.3615918104rad로 사용한다.

### 6개 목표각을12개로 확장하고 PhysX에 전달

`UrdfCoupledHandAction.process_actions`는 공식 superclass로6개 목표각을 만든 다음,
GPU tensor `index_select`와 multiplier·offset으로12개 목표각을 한 번에 계산한다.
현재 URDF의 offset은0이고 다음 관계를 적용한다.

```text
thumb3_target = 0.8392 * thumb2_target
thumb4_target = 0.891 * thumb3_target = 0.7477272 * thumb2_target
finger2_target = 1.0843 * finger1_target  # index / middle / ring / little
```

확장은 측정된 master 각도가 아니라 명령된 목표각을 기준으로 한다. PhysX mimic 제약을
제거했으므로 접촉 중 실제 관절각이 multiplier 관계에서 벗어날 수 있다. 목표 관계를 맞춘
것만으로 실제6개 actuator의 기계적 힘 전달까지 복원한 것은 아니다.

`apply_actions`는 공식 `set_joint_position_target`에12개 목표를 제출한다. Native
`Articulation.write_data_to_sim`이 PhysX position target buffer로 전달하고, 공식
`ImplicitActuatorCfg`의 stiffness20·damping1·URDF effort/velocity와128절 armature를
사용해 실제 drive를 실행한다. 위치 오차와 속도 오차로 작용하는 PD를 PhysX가 implicit하게
계산하며 별도 수제 Hand PD 또는 Hand OSC를 사용하지 않는다. 목표 속도·feed-forward
effort는 reset에서0으로 설정하며 이 Hand action은 position target만 갱신한다.

현재 별도의 목표각 보간이나 목표각 변화율 제한은 없다. 매 policy step에 새 목표를 정하고
그 사이 physics step에는 같은 목표를 유지한다. 물리 관절 속도·토크 제한은 목표 변화율
제한과 구분한다. Observation의 previous action과 action-rate reward는 실제 포화 의미와
같도록 raw Hand action을[-2,2]로 제한해 사용하며, radian 목표각을 넣는 것은 아니다.
PPO/ActionManager의 원래 Gaussian 표본 이력은 유지한다.

### 검증 범위

CPU에서 기존 `HandUrdfBoundsTests`·`HandActionSoftLimitsTests`·
`UrdfCoupledHandActionTests`·`NativeArmEffortLimitTests` 총16개가 통과했다.
실제 체크아웃의 native 변환을 사용해 endpoint·포화·12개 child 목표·부분 reset·원래
action tensor 보존·위치 목표의 backend 전달을 확인한다. 별도 CPU 계산으로 위 실효 각도
범위와 thumb chain도 재확인했다. 이번 검증은 GPU·Kit·학습을 실행하지 않았으며128절
armature 추가 이후 전체 학습 scene의 동역학을 새로 검증한 것은 아니다.


## 130. 2026-11-07 — Symmetric OSC action·좌표·피드백·제어 주기 점검

사용자가 Hand 다음으로 OSC 점검을 요청했다. 이번에는 action/controller를 수정하지 않고
현재 체크아웃의 공식 OSC·Action·actuator와 프로젝트 adapter를 대조하고 CPU에서 검증했다.
시뮬레이션·GPU·학습은 실행하지 않았다. 사용자 합의인 **정책 피드백+OSC 내부6축 F/T 오차
보정**은114절을 따르며, 내부 루프는 아래와 같이 여전히 미구현이다.

### 현재 실제 명령 경로

`ur5e_env_cfg.py`에서 정책 앞6개 action은 sensor-origin 축의
[Fx,Fy,Fz,Tx,Ty,Tz]다. `wrench_scale=(10,10,10,1,1,1)`이며 raw1은 각각10N·1N·m다.
Raw를 먼저[-20,20]·[-20,20]·[-36,36]·[-8,8]·[-8,8]·[-8,8]로 제한하므로 sensor-origin
물리 목표 범위는±200/200/360N·±8/8/8N·m다. 이 범위는 센서 명목 범위와 맞춘 **명령의
상한**이지 실제 접촉 하중을 그 범위로 보장하는 물리적 제한은 아니다.

`SensorFrameWrenchAction`은 센서 원점의 wrench를 native COM Jacobian과 맞도록 보정한 뒤
공식 `OperationalSpaceControllerAction`·`OperationalSpaceController`에 맡긴다.
현재 구성은 순수 `wrench_abs`, motion axes0·wrench axes1, gravity compensationOn,
inertial decouplingOff, nullspaceNone, `contact_wrench_stiffness_task=None`이다.

```text
tau_raw = J_COM^T * W_COM + gravity
tau_submitted = clamp(tau_raw, -joint_effort_limit, +joint_effort_limit)
joint_effort_limit = [150, 150, 150, 28, 28, 28] N*m
```

팔의 공식 IdealPD stiffness/damping은0이므로 추가 position/velocity PD는 없다. Pure wrench
분기는 pose 오차·속도 damping·operational-space inertia 역행렬을 사용하지 않는다.
`impedance_mode=fixed`라는 이름만으로 motion damping이 작동한다고 해석하지 않는다.
따라서 과거 pose OSC의 singular matrix inversion 실패 경로와 현재 경로를 구분한다.
이는 과거CUDA719의 원인 해결을 뜻하지 않는다. Torque saturation은 중력 보상까지 포함한
합산 토크에 적용되므로 Cartesian 목표 wrench를 항상 달성할 수 있는 것도 아니다.

### Sensor 원점과 COM의 변환

R을 sensor→base 회전, p를 base에서 sensor 원점 위치, c를 sensor축으로 표현한
해당 link의 local COM offset이라 하면 다음 COM wrench가 필요하다.

```text
F_base = R * F_sensor
M_COM_base = R * (M_sensor - c cross F_sensor)
```

Sensor distal link의 c는0이 아니라(0,0,0.0125)m다. `body_com_pos_b`는 robot base에서의
COM 위치가 아니라 각 link frame에서의 local COM 위치다. Native OSC가 원래 더하는
`p cross F_base`를 중복 적용하지 않도록 adapter는 먼저 sensor moment에서
`(R^T*p+c) cross F_sensor`를 뺀다. Native 변환 후 위 식과 정확히 일치한다.
PhysX Jacobian의 기준점이 link COM이라는 근거는
[공식 Jacobian 설명](https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/Articulations.html#jacobian)에 있다.
이동 기준점을 바꾼 뒤 moment가±8을 넘는 것만으로 clipping 오류라고 판단하지 않는다.
±8N·m는 원래 sensor 원점의 한계다.

### 미구현: OSC 내부 F/T 피드백

`mdp/sensors.py`의 incoming joint wrench는 sensor joint의 parent→child reaction을 읽는다.
Transition은 SI-200-8 clipping/resolution 모델 뒤 distal self-load를 빼서 정책 관측에 넣는다.
이 값은 현재 action/controller에 전달되지 않는다. Observation에서의 자중·관성 제거와
controller의 joint-space 중력 보상은 목적과 계산 경로가 다르다.

공식 내부 gain을 켜는 것만으로는 원하는6축 센서 피드백이 구현되지 않는다.

- Native controller의 `current_ee_force_b`는[N,3]이며 force feedback만 받는다.
- Native controller는 measured moment에 desired moment를 대입하므로 torque 오차는0이다.
- Native Action이 생성하는 ContactSensor는 제어 link인 sensor cylinder 자체의 접촉력이다.
  Hand 전체가 sensor fixed joint를 통해 전달하는 하중이 아니다. Cylinder의 ContactReportAPI
  누락 문제가 아니라 측정 대상이 다르다는 문제다.

현재 incoming reaction 규약은 `reaction=self_load-external_on_hand`이므로 보상된 측정값의
부호를 일반 ContactSensor의 force-on-body와 같다고 가정하지 않는다. 이후6축 보정 구현 시
목표 wrench와 센서 측정의 원점·축·힘의 작용 방향을 먼저 맞춰야 한다. 114절에 기록한 대로
기존 adapter에서 최신 보상된 F/T로 wrench를 보정하고 공식 OSC의J^T·중력·actuator는
유지하는 방향이 남아 있다. 내부gain·보정 wrench 상한은 이번에 임의로 정하지 않았다.

### 실제 주기와 sample/hold 의미

현재 `sim.dt=0.01`, `decimation=2`이므로 physics·OSC100Hz, policy50Hz다.
`shelf_manipulation_env_cfg.py`의 "Policy, physics and OSC all run at100Hz" 주석은 실제
설정과 다르다. 이번에는 주기와 코드 주석을 수정하지 않고 차이를 기록했다.

Native `process_actions`가 policy step에 sensor/task pose를 읽고 wrench를 base로 변환해
`set_command`한다. 이후 physics substep에는 최신J·gravity를 사용하지만 원하는 wrench와
축·COM moment 변환은 새로 하지 않는다. 따라서 policy 출력 시점의 sensor축으로 해석한
명령을 base에서 유지하며, substep마다 변하는 현재 sensor축에 연속 추종하는 방식은 아니다.
현재 sensor-frame 요구에 대한 시간적 의미를 내부 피드백 구현과 함께 명확히 해야 한다.
Pure wrench0은 자세·속도 유지 명령이 아니며, 이를 정책이 학습하도록 한다는114절의 의도는
유지한다. 이번에는 임의 pose 제어·감쇠 제어를 추가하지 않았다.

### CPU 검증과 한계

기존 `SensorFrameWrenchControllerTests`·`NativeArmEffortLimitTests`·`WrenchTests`·
`SelfLoadCompensationTests` 총10개가 통과했다. 실제 체크아웃 controller/action을 실행하며
1024개 회전·COM offset·환경 원점 변환, sensor clipping 순서, gravityOn/Off, 중복 moment
반례와 effort 제출을 검증한다. 이는 새 physics 실험이 아니라 tensor 수준의 검증이다.

추가 CPU 확인에서 J=identity·gravityOff·desired wrench=[1,2,3,4,5,6]일 때 현재None gain은
measured force를[0,0,0]→[10,20,30]으로 바꾸어도 출력이 그대로였다. 공식gain을6축 모두2로
준 진단에서는 force0→[3,6,9,4,5,6], force[0.5,1,1.5]→[2,4,6,4,5,6]으로 torque3개는
측정 오차 보정을 받지 않음을 확인했다. `set_command` 없이 measured sensor pose만90°
회전해도 desired base wrench가 유지됨을 확인했다. 실물/전체physics에서6축 힘 방향과
추종 성능을 검증한 것으로 확대하지 않는다.


## 131. 2026-11-07 — 설치된 OSC 옵션의 의미·기본값·현재 활성 경로

사용자는 OSC 옵션을 구체적으로 분석하도록 요청했다. 현재 설치된
`controllers/operational_space_cfg.py`·`operational_space.py`와
`envs/mdp/actions/actions_cfg.py`·`task_space_actions.py`를 기준으로 확인했다.
Controller·action·학습 설정은 변경하지 않았다. 새 GPU·시뮬레이션·학습도 실행하지 않았다.

### ControllerCfg: 실제 제어 수식 선택

| 옵션 | 설치된 기본값 | Symmetric 현재 값·효과 |
|---|---|---|
| `class_type` | `OperationalSpaceController` | 공식 controller |
| `target_types` | 필수 지정 | `["wrench_abs"]`: 목표force3+moment3, pose target 없음 |
| `motion_control_axes_task` | `(1,1,1,1,1,1)` | 전부0. task축의 병진/회전 pose 제어 선택 |
| `contact_wrench_control_axes_task` | `(0,0,0,0,0,0)` | 전부1. task축의 force/moment 명령을 모두 적용하며 feedback 활성화 여부와는 별개 |
| `inertial_dynamics_decoupling` | `False` | False. Pose motion 분기에서만 operational-space inertia를 곱함 |
| `partial_inertial_dynamics_decoupling` | `False` | False. 위 motion 분기에서 병진/회전 inertia coupling을 무시하는 선택 |
| `gravity_compensation` | `False` | True. 관절 중력 보상 토크를 합산 |
| `impedance_mode` | `"fixed"` | fixed. Motion gain을 cfg값으로 고정. 정지/감쇠/중력보상의 동의어가 아님 |
| `motion_stiffness_task` | 6축각100 | 기본값. Pose 목표가 없고 motion mask도0이므로 비활성 |
| `motion_damping_ratio_task` | 6축각1 | 기본값. Motion에서 `Kd=2*sqrt(Kp)*zeta`. 현재 비활성 |
| `motion_stiffness_limits_task` | `(0,1000)` | variable_kp/variable 입력 clipping용. 현재 비활성 |
| `motion_damping_ratio_limits_task` | `(0,100)` | variable 입력 clipping용. 현재 비활성 |
| `contact_wrench_stiffness_task` | `None` | None. Wrench P feedback이 꺼져 있음 |
| `nullspace_control` | `"none"` | none. 공식구현은 position을 켜면 제어관절수>6을 요구 |
| `nullspace_stiffness` | `10` | 기본값. 현재 비활성 |
| `nullspace_damping_ratio` | `1` | 기본값. 현재 비활성 |

`target_types`의 abs는 선택한 task frame에서 지정하는 절대 목표를 의미하며 world축 고정을
의미하지 않는다. Pose relative만 지원하고 wrench relative target은 지원하지 않는다.
Pose_abs는xyz+quaternion(wxyz)7D, pose_rel은delta_xyz+axis-angle3의6D,
wrench_abs는force_xyz+moment_xyz6D다. 여러 target을 쓸 때는 list 순서로 연결한다.
축 mask를0으로 해도 명령 채널 자체는 없어지지 않는다.

`fixed`는 target 차원 그대로, `variable_kp`는 motion stiffness6개 추가,
`variable`은 motion stiffness6개+damping ratio6개 추가다. Pure wrench에서는 각각
Arm6/12/18D가 되지만 pose 목표가 없으므로 추가 motion gain은 torque에 효과가 없다.
현재 프로젝트 adapter도 fixed+pure wrench만 허용하므로 mode 한 줄 변경으로 pose/hybrid
제어 실험이 가능하지 않다. Sensor 원점 보정의 계약까지 재검토해야 한다.

Pose motion이 있을 때의 제어 원리는 다음과 같다.

```text
u_motion = Kp * pose_error - Kd * current_ee_velocity
W_motion = Lambda * u_motion   # inertia decoupling on
W_motion = u_motion            # off
tau_motion = J^T * S_motion * W_motion
Lambda = inverse(J * inverse(M) * J^T)
```

Partial mode는 linear3x3와angular3x3를 따로 역행렬 계산하고 두 coupling을 무시한다.
Full/partial 모두 damped inverse가 아니라 `torch.inverse`를 사용한다. 현재 wrench-only는
`desired_ee_pose_b=None`이므로 이 분기에 들어가지 않는다. Decoupling을 켜는 것만으로
force 제어에 관성 보상이 생기지 않는다. Controller에는 Coriolis/원심항과Jdot*qdot의 별도
보상이 보이지 않으므로 이를 완전한 inverse-dynamics 보상이라고 설명하지 않는다.
Motion gain의 물리 단위도 Lambda 사용에 따라 달라지며 Hand actuator stiffness와 혼동하지 않는다.

Motion과wrench 축을 겹쳐 켜면 두 branch의 torque가 합산된다. 자동으로 한쪽을 우선하거나
접촉 시 mode를 전환하는 기능이 아니다. 현재 순수 wrench 의도를 유지하면서 임의 pose
목표나 속도 damping을 추가하지 않는다.

### Force-feedback gain의 별도 의미

`contact_wrench_stiffness_task`는 위치 오차에 곱하는 기계 stiffness가 아니라
`W_cmd=W_desired+Kf*(W_desired-W_measured)`의 비례 gain이다. 같은 힘끼리 또는 같은
moment끼리 비교하므로 무차원이다. `variable_kp`가 조절하는 motion gain과도 별개다.
Native API는 measured force3만 받고 moment error를0으로 만들므로6개 gain을 주어도
measured torque feedback은 생기지 않는다. Native Action의 direct-cylinder contact와
우리 articulation-joint F/T가 다른 문제는130절을 따른다.

Nullspace position은 controller에서6관절 이하를 명시적으로 거부한다. UR5e6관절에서 일부
task축만 선택해도 이 검사를 통과하지 못하므로 singularity/self-collision 회피 옵션처럼
켜서는 안 된다. Action의 nullspace target은none/zero/default/center이며 position mode와
target 선택의 조합도 검사한다. 현재 none/none은 일관된다.

### ActionCfg: 대상 frame과 policy 입력 해석

| 옵션 | 현재 값·의미 |
|---|---|
| `class_type` | SensorFrameWrenchAction: sensor-origin clipping·COM 기준점 보정 |
| `asset_name` | robot |
| `joint_names` | torque를 받을 UR5e6관절 |
| `body_name` | ToolForceTorqueLink: Jacobian/EEF 상태 대상 body |
| `body_offset` | None. 가상 EEF offset을 쓰지 않으며 현재 adapter는 non-None을 거부 |
| `task_frame_rel_path` | Robot/ToolForceTorqueLink. 명령과 axis mask 기준. 기본None은root/base |
| `position_scale` | 기본1, 현재 pose가 없어 미사용 |
| `orientation_scale` | 기본1, 현재 미사용. Pose_rel rotation-vector와pose_abs quaternion에 적용 |
| `wrench_scale` | (10,10,10,1,1,1), raw1당10N/1N·m |
| `stiffness_scale` | 기본1, variable_kp/variable에서만 입력 motion gain에 적용. 현재 미사용 |
| `damping_ratio_scale` | 기본1, variable에서만 입력 damping ratio에 적용. 현재 미사용 |
| `nullspace_joint_pos_target` | none |
| `clip` | 상속필드 기본None. 설치된 native OSC Action은 wrench에 이 필드를 적용하지 않음 |
| `debug_vis` | 상속필드 기본False. 해당 native Action의 별도 시각화 구현은 없음 |

설치된 ActionCfg에는 force_scale/torque_scale 별도 필드가 없으며6축 `wrench_scale`을
사용한다. Native wrench 입력에는 자동tanh·workspace clipping·wrench clipping이 없어서
현재 adapter가 먼저 sensor-range clipping을 한다. Stiffness와damping ratio는 scale 후
해당limit으로 clamp하고 controller에서도 다시 clamp한다.
현재 adapter의 processed_actions 마지막3개는 native 변환에 필요한 기준점 보정을 포함한
중간 moment이므로 이를 sensor-origin 목표 torque로 그대로 표시해서는 안 된다.

### 공식 예제와 CPU 확인

체크아웃 `scripts/tutorials/05_controllers/run_osc.py:127`은7관절Franka에서 pose_abs+wrench_abs,
variable_kp, inertia decouplingOn, gravityOff를 사용한다. Motion은XY+회전, force는Z만
선택하고 contact gainZ=0.1이다. TiltedWall에서 읽은 contact force를 전달하며 nullspace
position도 사용한다. 이 값들을 그대로 UR5e6관절 순수wrench에 복사할 근거는 없다.

추가 CPU 확인은 실제 controller class를 사용했다. J=identity, wrench=[1,2,3,4,5,6],
gravityOff에서 motion Kp/zeta/mask 변경, full/partial inertia flag 활성화, variable_kp와
variable 모두 같은 torque를 냈다. `torch.inverse` 호출을 금지한 상태로도 통과했다.
Arm action dimension만 각각6/12/18로 달라졌다. Gravity vector를ones로 넣으면
[2,3,4,5,6,7]이 되었고 wrench Z축만 선택하면[0,0,3,0,0,0]이었다. Nullspace position을
6관절에서 켜면 공식 예외가 발생했다. 이는 현재 분기의 옵션 효과 확인이며 전체physics
tracking이나 contact-force 제어를 새로 검증한 것은 아니다.

### 후속 확인 — 최신3.0의6축 측정 피드백 지원 범위

사용자가 버전 업그레이드로6축 피드백을 사용할 수 있는지 질문했다. GitHub API로 공식
최신 GA가2026-10-20공개 `v3.0.0`임을 확인하고, 해당tag의 source와공식API 문서를 확인했다.
기존 `isaaclab.controllers.OperationalSpaceController`는v3.0.0에서도 measured force3축을
입력받고 measured moment를 desired moment로 채우므로 torque 오차는0이다.

별도 공식 `isaaclab_newton.controllers.NewtonOperationalSpaceController`는
`ee_wrench_des`·`ee_wrench`를각[N,6] force+moment로 받고 `use_wrench_feedback`·
`wrench_stiffness`를 제공한다. 이controller 자체는 caller가Jacobian·dynamics를 제공하는
model-free 방식이므로 문서상physics backend와 독립적이며 PhysX를 반드시Newton으로
바꿔야 한다는 의미는 아니다.

다만v3.0.0의 공식 `NewtonOperationalSpaceControllerAction`도 기본 측정 경로는
ContactSensor의normal force3축이다. `newton_task_space_actions.py`에서 measured moment를
desired moment로 채우므로 이Action을 그대로 쓰면 torque feedback은 여전히0이다.
따라서 **새controller는6축을지원하지만버전또는ActionCfg교체만으로손목6축F/T가자동연결되지는
않는다.** 실제 articulation-joint F/T를6D입력에 연결하고축·기준점·부호·보상을 맞춰야 한다.
Newton Action의기본pose명령계약도현재purewrench action과다르므로drop-in교체로보지않는다.

공식3.0release는IsaacSim6.2·Python3.12등을대상으로한다. 현재5.1환경에서이controller만
가져오는호환성은시험하지않았다. 이번에는업그레이드·설치·코드변경·GPU실행을하지않았다.

- [공식3.0릴리스](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [6축NewtonOSC문서](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_newton/isaaclab_newton.controllers.html)
- [기존OSC의3.0고정소스](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/controllers/operational_space.py)
- [NewtonAction의측정연결](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_newton/isaaclab_newton/envs/mdp/actions/newton_task_space_actions.py)


## 132. 2026-11-07 — 현재 버전 유지와6축 wrench 보정 구현 방향

사용자는Newton적용의부담을고려해현재버전의공식OSC를활용하는방향을선택했다.
이선택은프레임워크업그레이드나controller교체승인이아니다. 현재버전을유지하며기존
`SensorFrameWrenchAction`에서6축오차보정만추가하고공식OSC의J^T·중력보상·IdealPD
토크제한을계속사용하는안을검토했다. 아직실행코드는변경하지않았다.

센서중심·축·부호를맞춘목표Wd와보상측정Wm에대해아래P보정을제안한다.

```text
Wm = calibrated_joint_reaction - distal_self_load
Wc = clamp(Wd + K * (Wd - Wm), -sensor_command_limit, +sensor_command_limit)
tau = native_OSC_Jt_mapping_and_gravity(Wc)
```

Force/torquegain은독립적으로설정한다. 공식 `contact_wrench_stiffness_task=None`을
유지해native3축보정과중복적용하지않는다. 정책목표와내부보정명령을분리하고previous
action·action-rate·PPO에는기존clipping된정책목표의의미를유지한다. Scale은한번만
적용하고sensor원점의clipping뒤COM기준으로변환한다. 측정이무효인reset직후행에서는
feedback만끄며무효측정을0으로대입해오차를증폭시키지않는다.

현재보상측정은Hand가환경에가하는방향의reaction규약이다. 고정접촉의축별부호와
negativefeedback여부를검증하기전gain을확정하지않는다. 자유공간에서보상측정0이면
Wc=(I+K)Wd가되므로이보정이자동자세/속도유지나안정적인접근을보장하지는않는다.
처음에는P만검토하며포화시적분누적이발생하는적분기는추가하지않는다.

검증은현재sampled-basehold를유지한gain0기준선→force한축→force3축→torque한축→
전체6축순서로구분한다. 현재sensorframe을매substep따라가는변경은gain0에서도기존
동작과달라지므로별도비교해야한다. Target/측정을같은frame·원점으로환산하는것은필수다.
공식예제의forceZ gain0.1은비교후보의근거일뿐torque권장값이나6축확정값이아니다.

Physics주기의F/T루프는MDP전체transition/history나tactile17개조회경로를호출하지않는다.
기존순수calibration/self-load함수를재사용하고질량·COM·관성은cache하며최신완료physics
timestamp별로한번계산한측정값을controller와observation이공유하는방향이다. 추가비용은
6축P수식보다SDK wrench/가속도취득과Hand self-load계산을실제측정해판단해야한다.
설치된ArticulationData의incomingwrenchgetter는검사하는`timestamp`대신`time_stamp`를
대입해같은시각의중복SDK조회가능성이있음을소스에서확인했다. 이문제는오래된측정값을
반환한다는증거가아니며공식소스는이번에수정하지않았다. 자체공유timestampcache로
중복조회를피하는방향을검토한다.

이번에는문서기록과읽기전용설계검토만진행했다. Controller·gain·주기·frame동작을변경하거나
GPU·시뮬레이션·학습을실행하지않았다.


## 133. 2026-11-08 — 고정 wrench-feedback gain 선정안 기록과 현 상태 유지

사용자는 고정 K의 결정 기준을 논의한 뒤 "일단 이건 기록해두고 현재 상태를 유지하자"고
결정했다. 아래 내용은 후속 검토안이며 구현·gain 확정·실험 실행의 승인이 아니다.

논의한 P 보정은 다음과 같다. 목표와 측정은 같은 sensor frame·원점·부호 규약을 사용한다.

```text
Wc = Wd + Kf * (Wd - Wm)
Kf = diag(kFx, kFy, kFz, kTx, kTy, kTz)
```

`Kf`는 무차원 wrench 오차 보정 gain이다. 예를 들어 목표10N·측정8N에서 gain0.1이면
명령은10.2N이다. Pose 변위에 대한 Cartesian stiffness(N/m, N·m/rad)나 motion Kp와는
다르며, 정책이 Kf를 출력하는 것만으로 독립적인 stiffness·damping 제어가 되는 것은 아니다.
고정 Kf라도 정책은 F/T 관측을 이용해 목표 wrench를 조절할 수 있지만, 내부 보정 gain 자체를
선택하지는 못한다. 명시적인 compliance 학습의 정의와 action 확장은 아직 결정하지 않았다.

Kf는 URDF의 관성만으로 유일하게 결정할 수 없다. 접촉 강성, 팔 자세별 유효 관성, Hand
변형, 측정 잡음·보상 잔차·지연, 제어 주기와 관절 토크 포화가 함께 영향을 준다. 후속 선정은
다음 기준으로 진행하는 안을 기록한다.

1. Gain0 기준선에서 자유 공간의 보상 잔차·잡음과 접촉 시 축별 부호·기준점을 먼저 확인한다.
2. 정책 대신 고정된 목표 wrench를 사용해 한 축씩 비교한다. 0·0.05·0.1·0.2는 탐색 후보이며
   검증된 권장 범위나 확정값이 아니다. 힘과 토크 gain은 별도로 선정한다.
3. 목표 도달 시간·정상상태 오차·overshoot·진동·관절 토크 포화를 함께 비교한다. 여러 팔
   자세 및 고정 물체와 동적 물체 접촉에서 필요한 추종 성능을 만족하는 작은 gain을 고른다.
4. 자유 공간에서 보상 측정이0이면 Wc=(I+Kf)Wd가 되어 가속 명령도 커진다. Gain 증가를
   자동 자세 유지·정지 보장으로 해석하지 않으며, 포화 중 오차를 gain만으로 해결하지 않는다.

공식 `scripts/tutorials/05_controllers/run_osc.py`는 hybrid pose+wrench 구성에서 force-Z에만
gain0.1을 사용한다. 이는 초기 비교값의 근거이며 UR5e pure-wrench6축이나 torque gain의
공식 적정값이 아니다. [공식 OSC 예제](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/tutorials/05_controllers/run_osc.html)

현재 실행 설정은 **Arm wrench6+Hand6, sensor frame, gravity compensation 활성,
`contact_wrench_stiffness_task=None`**을 유지한다. 정책에는 자중·관성을 보상한 F/T가
들어가지만 OSC 내부6축 오차 보정은 아직 미구현이다. 이번에는 문서만 갱신했으며 controller,
gain, action 차원, 주기와 학습 설정을 변경하지 않고 GPU·시뮬레이션·학습도 실행하지 않았다.


## 134. 2026-11-08 — F/T 사양과 자중·관성 보상 감사

사용자는 메인 `contact_manipulation_symmetric`의 F/T가 SI-200-8 사양에 맞는지, 자중·관성
보상이 의도대로 구현됐는지 검토하도록 요청했다. 실행 코드는 변경하지 않고 소스·공식 문서와
기존 CPU 테스트를 확인했다. 실제 물리 실행에서 잔차를 측정한 검증과는 구분한다.

### 측정 및 observation 경로

`mdp/sensors.py:160`은 sensor distal link의 `body_incoming_joint_wrench_b`를 읽는다.
Native Omni Physics107.3 문서는 child joint frame이라고 명시하며, 현재 scene은 그 원점과
축을 sensor link 원점·축에 맞춘다. 로컬 Isaac Lab docstring의 parent-body-frame 문구만
보고 추가 회전·모멘트 이동을 적용하면 안 된다. Known-load 물리 시험으로 부호도 확인해야 한다.

```text
ideal = incoming parent-on-child joint reaction, at sensor origin, sensor axes
measured = round(clamp(ideal, -ranges, ranges) / resolution) * resolution
observation_wrench = measured - modeled_distal_self_load
observation = clamp(observation_wrench / ranges, -5, +5)
```

범위는force±(200,200,360)N, torque±(8,8,8)N·m, 분해능은force0.1N·torque0.005N·m로
공식 SI-200-8 표와 일치한다. 포화 전 ideal 값으로 축별 overrange를 판정한다. 보상은 센서
포화·양자화 뒤 적용하고 다시 양자화하지 않는다. 실제 센서 범위를 넘어 잃은 정보는 보상으로
복원되지 않는다. 현재 finite한 overrange 표본도 observation에 들어가므로, 향후 내부 feedback
도입 전 별도 유효성 정책을 정해야 한다.

ATI 표의 분해능은 잡음을 고려한 유효 분해능이다. 현재 결정론적 반올림은 단순 근사이며
잡음·bias·drift·축간 오차·필터·지연·복합하중은 모델링하지 않는다. 측정 범위와 기계적 파손/
허용 과부하 한계도 다른 값이므로 현재 clamp를 물리적 파손 모델로 설명하지 않는다.

### 자중·관성 보상 수식 확인

`mdp/events.py:1165`는 sensor distal half와 모든 `/ROASRightHand/` body의 mass·COM·inertia를
캐시한다. Proximal half는 sensing joint의 앞쪽이므로 제외한다. Native inertia는 COM 기준,
link-prim 축의 column-major tensor이므로 현재 reshape·transpose와 link rotation이 맞다.
COM principal rotation을 다시 곱하지 않는다.

`mdp/observations.py:179`의 계산은 다음과 같다. 아래 F와 M은 world axes이며 마지막에
sensor axes로 회전한다. Body COM 선형 가속도에는 구심 가속도가 이미 포함된다.

```text
Fi = mi * (ai_com - gravity)
Mi_sensor = Ii_world * alpha_i + omega_i x (Ii_world * omega_i)
            + (pi_com - p_sensor) x Fi
Wself = sensor_rotation_inverse * [sum(Fi), sum(Mi_sensor)]
```

자중·병진/회전 관성·gyroscopic 항·모멘트 팔이 모두 포함돼 있으며 수식상 확인된 오류는
없다. 입력 가속도는 native physics COM acceleration이고 reset pose의 차분이 아니다.
보상 후 값은 이상적인 경우 Hand에 가해지는 외력의 반대 부호인 기존 reaction 규약을 유지한다.

### 실제 보상 성능이 아직 입증되지 않은 지점

- PhysX 반력은 한 physics step의 평균이며 보상은 현재 COM 가속도·angular velocity 표본을
  사용한다. 급격한 운동·충격에서 시간 이산화 잔차가 남을 수 있다. 현재 policy50Hz는
  physics100Hz의 마지막 표본을 사용하며 모든 중간 표본이나 실제 센서 대역폭을 재현하지 않는다.
- PhysX는 armature를 외력과 유사한 항으로 취급한다. 새 Hand armature0.0001과 rigid-body
  damping/clamping의 전달 하중을 rigid-link Newton–Euler만으로 완전히 제거한다고 단정할 수
  없다. 실제 damping·gyro flag를 읽고 손가락만 가속하는 시험으로 잔차를 측정해야 한다.
  임의의 armature*qddot 항을6축에 단순 합산하지 않는다.
- 현재 센서는 length50mm·diameter50mm·mass0.20kg의 sensor/adapter proxy다. ATI 실물은
  표준 plate 포함diameter82mm·height25.4mm·mass0.30kg이다. 기존5cm 연결부 요구에 따라
  만든 근사이며 실물 datum·축·탄성·공진을 확인한 모델이 아니다. 이번에 형상·질량은 바꾸지 않았다.
- Observation만 self-load를 제거한다. GUI와 F/T-growth reward, raw TensorBoard 측정은
  보상 전 전체 반력이다. TensorBoard에는 보상 후 observation도 별도 항목으로 존재한다.
  `previous_wrench`가 reset에서0이므로 첫 유효 표본의 자중도 growth로 계산될 수 있다.
- 설치된 `ArticulationData`는 `.timestamp` 검사 뒤 `.time_stamp`를 대입한다. 동일 시각의
  중복 SDK 조회 가능성이 있지만 자체 SensorReader cache가 정상 경로를 제한한다. 이는
  보상 오차나 오래된 측정의 증거가 아니며 공식 소스도 변경하지 않았다.

### 완료한 CPU 검증과 후속 물리 검증

기존 검증31개가 통과했다: saturation/보상 수식·SensorReader·transition/cache13개,
TensorBoard8개, GUI10개. 보상 수식은 별도 SciPy/NumPy oracle, 회전·이동 불변성과
1024행 batch·subset·empty 및 hot path의 host 복사 부재를 포함한다. 실제 반력을 가상 입력한
검증이므로 backend 잔차가 작다는 증거는 아니다. 중복 실행한 기본10개도 통과했으며31개에
다시 더하지 않는다. `CUDA_VISIBLE_DEVICES=''`로 실행했고 Kit·GPU·시뮬레이션·학습은 실행하지 않았다.

후속 물리 시험은 (1) 여러 sensor/Hand 자세에서 정지 중 중력과 모멘트, (2) 비접촉 arm
병진·회전과 손가락 단독 가감속, (3) 알려진 외력 및 모멘트 팔을 가한 경우, (4) sensor 포화와
reset 직후를 분리한다. 각 경우 ideal 반력·양자화 측정·예측 self-load·보상 잔차·overrange를
같은 physics 시각으로 기록해야 한다. 현재 설정과 controller gain은 유지한다.

- [ATI Axia80-M20 사양](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Axia80-M20)
- [ATI 매뉴얼](https://www.ati-ia.com/app_content/Documents/9620-05-B-Axia80.pdf)
- [Native incoming-joint-force API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force)
- [PhysX 반력의 포함 항과 시간 평균](https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force)


## 135. 2026-11-08 — Axia80-M20 형상·질량과 임시 연결 플랜지

사용자는 실제 센서에 맞추되 길이가 미정인 연결 플랜지를 고려하도록 요청했다. 메인
Symmetric만 수정하며 원본·parallel 프로젝트는 변경하지 않는다. 센서를 단순히 길게 늘리지
않고 제조사 sensor assembly와 추가 custom flange를 구분한다.

| 항목 | 설정 | 근거 |
|---|---|---|
| sensor assembly | Ø82mm·높이25.4mm·0.30kg | ATI 공식 사양, 표준 interface plate 포함 |
| 추가 custom flange | 양쪽12.3mm·Ø50mm·총0.04kg | 기존 총50mm 간격 유지용 임시값, 실측 필요 |
| 전체 tool–Hand 거리 | 50mm | 기존 장착 위치 유지 |
| F/T reference | Tool Side 면 중심, tool 장착 시작점+37.7mm | 공식9230-05-1507 도면 |

`scene.py` cfg의 `sensor_body_length=.0254`, `sensor_mount_radius=.041`,
`sensor_mount_mass=.30`은 제조사 assembly의 치수·질량이다. `sensor_mount_length=.05`는
전체 거리이며 플랜지의 한쪽 길이는 `(mount_length-body_length)/2`다.
`sensor_flange_radius=.025`, `sensor_flange_mass=.04`는 추가 플랜지의 반경·총질량으로
hardware 확인 후 바꿔야 한다. 내부 질량 분포는 공개되지 않아 균일 원통 근사를 유지한다.

기존 proximal/distal 두 rigid link를 유지하고 각 링크에 센서 반쪽과 플랜지 한 개를 배치한다.
질량은각0.17kg, COM과 COM 기준 관성은 질량 가중 평균·평행축 정리로 계산한다. 실제 Hand
장착점은 tool 시작점+50mm 그대로다. COM·관성을 읽는 기존 보상과 OSC가 native 변경값을
자동 사용하며 별도 hardcoded payload mass나 COM 보정은 추가하지 않는다.

공식 도면을 이미지로 확인한 결과 sensing reference origin은 센서 중앙이 아니라 Tool Side
면이다. 두 link actor frame과 equivalent incoming joint frame을 이 원점에 두고, geometry와
COM offset을 함께−12.7mm 옮겨 실제 형상·질량 분포는 보존한다. Distal COM은sensor frame
기준Z=−4.8794118mm, proximal은−20.5205882mm다. 센서 내부 strain-gauge 위치를 가정하는
것이 아니라 제조사 기준점에서 wrench를 표현하는 fixed joint다. Sensor X/Y의 실물 장착 yaw는
추후 확인해야 한다. 기준점 변경으로 기존 checkpoint의 action/observation 의미도 달라지므로
그대로 play한 결과를 변경 전과 동일 조건으로 취급하지 않는다.

기존 native API를 사용하며 F/T 범위·분해능·gravity compensation·IdealPD·Hand 구동,
Arm6+Hand6 action, reward와 주기는 유지한다. OSC 내부 F/T feedback은 계속 OFF다.
물리 mount·질량·좌표계 변경 관련14개 CPU 검증이 통과했다. 회전된5개 world frame에서
reference·COM·형상·Hand 장착점의 일관성도 검사했다. 이후 영상 시험 결과는 별도 기록한다.

- [ATI 물리 사양](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Axia80-M20)
- [공식 reference 도면](https://www.ati-ia.com/app_content/Documents/9230-05-1507.auto.pdf)


## 136. 2026-11-08 — Armature 적용 후 Hand 전체 이동300g 밀기 영상

사용자는 armature 변경 뒤 물체 밀기 영상도 남기도록 요청했다. 124절과 같은 protocol을
현재 Symmetric Hand 설정으로 다시 실행했다. 상자300g·기본마찰0.5/0.5, 12관절 목표0rad
유지, 기부 native prismatic drive로100mm 전진2초·유지2초·복귀2초와 앞뒤 정지1초씩이다.
각 검지·중지·약지·소지·엄지 앞 배치를800step·8초씩 촬영했고 전체40초로 연결했다.

Native backend12관절에서 Force drive·Kp20·Kd1·effort limit1N·m·armature0.0001kg·m²와
URDF 속도 상한을 확인했다. 상태/root pose를 매 step 덮어쓰지 않으며 기존 진단 기부 drive의
설정과 Hand 자중 제외 조건도 유지했다. 이 시험은 UR5e·OSC로 기부를 구동한 시험이 아니다.
공유 probe의 영상 표시만 native 설정·armature가 항상 보이도록 보완했다.

| 배치 | 최대 상자 XY 이동(mm) | 전체 Hand 최대 하한 위반(°) | 최대 Hand–상자 법선 합력(N) | 전진/유지400step 접촉 |
|---|---:|---:|---:|---:|
| Index | 89.140 | 0.001633 | 3.991 | 358 |
| Middle | 88.588 | 0.006304 | 2.850 | 357 |
| Ring | 88.973 | 0.000897 | 3.527 | 358 |
| Little | 89.422 | 0.002113 | 2.670 | 359 |
| Thumb | 75.156 | 0.004616 | 5.828 | 334 |

모든 실행 status는complete이고 상한 위반은 없었다. Console에서 CUDA/PhysX failure나
overflow는 발견되지 않았다. 영상에 native PD·effort·armature, 관절각·하한 위반, 기부
실제/목표 위치, 상자 변위/기울기, 링크별 법선 접촉 합력을 표시한다. 접선 마찰력·개별 접촉점
힘·측정 모터 torque로 해석하지 않는다.

이전124절의 Index 이동 약7.47mm·하한 위반 약51.51°에 비해 현재89.14mm·0.00163°로
개선됐다. 이는 해당0rad hard-stop·기부 이동 조건의 비교이며, 중간 flexion에서의 접촉 변형,
실물 Hand의 강성/감쇠 일치, 학습 정책의 밀기 성능이나 CUDA719 해결을 입증하지 않는다.

- 영상: `docs/videos/hand_armature_push_2026-11-08/all_fingers.mp4` (40s·25fps·1000frames).
- 미리보기: `docs/videos/hand_armature_push_2026-11-08/preview.png`.
- 원자료·요약·재현 명령: `logs/diagnostics/hand_armature_push_2026-11-08/`.
- 모든 GPU 프로세스가 종료된 후 F/T 시험에 GPU를 넘겼다. 학습은 실행하지 않았다.


## 137. 2026-11-08 — Production OSC로 흔들며 F/T 자중·관성 보상 촬영

사용자는 센서 frame gizmo와 F/T 값을 동시에 보여주는 흔들기 영상, 특히 OSC로 명령하는
시험을 요청했다. `src/contact_manipulation_symmetric/tests/probe_ft_compensation.py` 한 파일로
재현용 진단을 추가했다. 학습용 robot assembly·질량·관성·armature·중력·공식 controller와
생산용 `SensorFrameWrenchAction`·`UrdfCoupledHandAction`을 사용한다. Shelf·물체·policy·
reward·termination은 없는 단일 articulation 시험이며 controller 설정은 바꾸지 않는다.

Native OSC action은policy50Hz에 명령을 갱신하고 physics100Hz에 적용한다. SensorReader와
기존 calibration/self_motion_wrench 함수로 실제 반력을 매 physics step 기록한다. 목표
wrench와 측정 wrench는 서로 다른 의미다. 비접촉에서 목표 wrench는 운동을 유발하고,
보상한 F/T는 외부 접촉이 없으므로0에 가까운 것이 기대된다. OSC 내부 F/T feedback은 OFF다.

초기에는 native `write_joint_state_to_sim`으로 학습용 default pose/velocity를 한 번 적용한다.
이후 모든 운동은 실제 제출 torque와 native dynamics에 의해 발생하며 q/root를 매 step
덮어쓰지 않는다. 초기 bare Articulation에 default state를 쓰지 않은 예비 실행은 authored
zero-q에서 시작한 진단 설정 오류였으며 `smoke_authored_zero_pose*`로 따로 보존했다.
본 결과는 수정된 초기화와 실제 production 초기 관절각을 확인한 실행만 사용한다.

### 명령·표시·실행 조건

- 2초 zero-wrench → forceX/Y/Z 각2초 → torqueX/Y/Z 각2초 → 손가락 open/close5초 →
  final zero-wrench2초, 합계21초·2100physics step.
- 축별 명령은 `A*(cos(2*pi*f*t)-cos(4*pi*f*t))`, f=1Hz다. Force A=0.6N(절댓값≤1.2N),
  torque A=0.012N·m(절댓값≤0.024N·m). Raw policy 단위로 환산한 뒤 생산용 action을 통과한다.
- 2rad/s는 진단 중 과도한 arm 속도를 발견해 중단할 기준일 뿐 생산용 종료 조건 변경이 아니다.
  실제 최대 arm 속도는1.50227rad/s였고 안전 중단은 없었다.
- 0.17kg distal sensor body와 COM z=−0.0048794118m의 native readback을 확인했다.
  실제41body USD 속성은linear damping0·angular damping0.05·gyroscopic forcesTrue였으며
  어느 값도 진단을 위해 변경하지 않았다.
- 영상은25fps·525frames·1280×1088·1배속이다. 전체 robot과 Hand/sensor 확대를 보여주고,
  Tool Side reference 원점의 native frame gizmo(X/Y/Z=빨강/초록/파랑)와6축 그래프를 표시한다.
  명령, 전체 센서 측정, 예측 self-load, 보상 후 observation 값을 함께 볼 수 있다.
- Marker에 RigidBodyAPI/CollisionAPI가 없음을 runtime 검사했다. Camera와 gizmo는 시각화용이다.

### 정량 결과

아래 RMS는 초기1초 이후 모든 표본을 합산한 축별 값이다. 아래 최대 잔차는 전체21초에서도
동일했다. 모든2100표본이 유한하고 overrange0·tactile 접촉0이었다.

| 축 | 보상 전 측정 RMS | 보상 후 RMS | 보상 후 최대 절댓값 |
|---|---:|---:|---:|
| Fx (N) | 8.77908 | 0.027661 | 0.049942 |
| Fy (N) | 0.85609 | 0.027226 | 0.049981 |
| Fz (N) | 2.91843 | 0.027999 | 0.049940 |
| Tx (N·m) | 0.10588 | 0.001501 | 0.002845 |
| Ty (N·m) | 0.83014 | 0.001347 | 0.002860 |
| Tz (N·m) | 0.10693 | 0.001834 | 0.004840 |

양자화 전 ideal reaction에서 self-load를 뺀 힘 잔차 최대는6.20e−6N이었다. 힘은 현재
0.1N 반올림의 절반 이내 잔차와 일치한다. Ideal torque 잔차 최대는0.002818N·m로 작지만
0은 아니었다. 따라서 실제 링크 관성 보상이 이 운동 범위에서 작동함은 확인했으나
armature·rigid-body damping·평균/현재 표본 차이의 각 기여를 개별 규명한 것은 아니다.

고정된 default pose의 zero-wrench 구간(t=1–2s) sensor translation 범위는축별약20–27μm였다.
Force 구간에서는 축별 수mm–약30mm 이동하고 rotation 약6–9°, torque-Z 약19°,
손가락 운동 구간 약22°의 자세 변화가 기록됐다. Zero-wrench는 pose-hold controller를
추가한 것이 아니며 운동 후 잔여 속도가 즉시 사라지는 조건도 아니다.

### 검증 범위와 산출물

초기 영상은 Hand가 팔 뒤로 가려져 카메라 각도와 marker scale만 조정해 다시 촬영했다.
두 실행의 모든2100step에서 joint position·raw wrench·self-load·명령이 bit-exact하게
동일했다. 최종 MP4의 프레임은 같은 시각의 최종 JSON 표본과 대응한다. 초기 영상은
`first_view*`로 보존했다. 실제 외력을 알고 가한 교정 시험, 센서 full-scale/과부하, 물체 접촉
추종, 실물 sensor 노이즈·지연·내부 탄성 검증으로 확대하지 않는다.

- 최종 영상: `docs/videos/ft_compensation_2026-11-08/ft_shake.mp4`.
- 최종 원자료: `logs/diagnostics/ft_compensation_2026-11-08/report.json`.
- 실행 console/예비 결과: 같은 diagnostic directory. Sandbox에서 GPU가 보이지 않아 실패한
  초기 실행은 `sandbox_no_device.log`이며 실제 GPU 고장으로 해석하지 않는다.
- 기존 mount·wrench·보상 관련14개 CPU 검증과 진단 syntax/overlay 검증도 통과했다.
  실제 학습이나 PPO update는 실행하지 않았다.

재현:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py \
  --headless --device cuda:0 --force-amplitude .6 --torque-amplitude .012
```


## 138. 2026-11-08 — 17개 Tactile의 실제 접촉·분리 영상

사용자는 예전에 확인했던 tactile 시험도 영상으로 남기기를 요청했다. 기존
`src/contact_manipulation_symmetric/tests/probe_ft_compensation.py`에 `--tactile-scan` mode를
추가했다. 기본 F/T 흔들기 mode는 유지하며 새 진단 파일이나 학습용 sensor/controller 변경은 없다.
전체 UR5e·센서 연결부·Hand, 현재 PD20/1·armature0.0001·URDF 한계와 production action을 사용한다.

### 실제 측정 경로와 시험 방법

- 순서는 Palm1, Thumb4, Index/Middle/Ring/Little 각3개로 총17개다. URDF의 Palm 이름
  `right_plam_force_sensor` 철자는 원본 그대로 사용하며 canonical mapping을 runtime 확인한다.
- Production grouped `ContactSensor`→`SensorReader`의 native `net_forces_w`를 canonical
  순서로 정렬하고, 유한하고 유효한 값에서 `norm(force)>0.01N`이면 ON이다. 모든 물체에 대한
  binary이며 Blocker 필터 reward를 시험한 것은 아니다. 강제로 특정 채널을 켜지 않는다.
- 표시할 binary를 기존 `dynamic_sample()`에 전달하고 observation의 tactile slot37:54와
  같은지 검사한다. 이는 slot/encoding 검사이며 전체 ManagerBased MDP transition 실행은 아니다.
- 별도의 공식 rigid contact view를17sensor×1probe 필터로 생성한다. Sensor path 순서와
  크기·유한성을 확인하며, 같은 시점의 해당 채널 binary ON과 probe-only 반력>0.01N을
  동시에 요구해 자체 접촉에 의한 거짓 통과를 방지한다.
- 외부 probe는 지름3mm·질량10g·중력 OFF인 동적 sphere다. World-anchored native
  PrismaticJoint/DriveAPI(Kp200N/m·Kd2N·s/m·최대1N)로 실제 접근·복귀한다. 초기 표면 간격6mm,
  추가 목표 stroke1.5mm이며 실제 접촉 변형·변위는 PhysX가 계산한다.
- 각 pad는 baseline0.4초→approach0.7초→drive target hold0.8초→release0.7초→clear0.4초,
  합계3초다. 각 시험 사이에만 robot/probe 초기 상태와 guide 기준을 native API로 복구한다.
  접촉 중 q/root를 반복해서 덮어쓰지 않는다. Arm은 기존 zero-wrench OSC·gravity compensation,
  Hand는 열린 목표를 유지하므로 접촉력으로 팔이 밀려나며 contact가 일찍 끝날 수 있다.
  `hold`는 guide 목표 유지이며 지속 접촉 제어를 추가한 것이 아니다.
- Sensor 기록은 physics100Hz, policy 표본은50Hz로 구분한다. 두 시점 모두 실제 ON이
  있음을 확인했으며 영상은25fps·1배속이다. GUI용 camera와 색 marker/frame gizmo에는
  RigidBodyAPI·CollisionAPI가 없음을 검사했다. 시험 구 자체에는 의도한 실제 물리가 있다.

### 최종 결과

동일 지름3mm 구를 사용한17개가 모두 baseline OFF→실제 probe 접촉 ON→분리 후 OFF를
통과했다. 총5100physics 표본·51초다. 최종 MP4는1280×1040·25fps·1275frames이며
실제 영상 metadata와 프레임 수를 별도로 확인했다. 모든 표본의 valid가True이고 힘이 유한했다.
17개 모든 net force와 probe-only force의 norm 차이는 전체 표본에서0이었다. 각 시험에서
ON이었던 채널은 선택한 pad 하나뿐이었으며, clear 마지막10표본은 모두 OFF였다.

| 채널 | Pad | 최대 법선 합력(N) | 접촉 ON 표본(100Hz) | 정책 시점 ON 표본(50Hz) |
|---|---|---:|---:|---:|
| 0 | Palm | 0.155396 | 18 | 9 |
| 1 | Thumb1 | 0.142506 | 21 | 11 |
| 2 | Thumb2 | 0.145309 | 17 | 9 |
| 3 | Thumb3 | 0.116685 | 13 | 7 |
| 4 | Thumb4 | 0.187300 | 29 | 14 |
| 5 | Index1 | 0.125213 | 16 | 8 |
| 6 | Index2 | 0.102209 | 13 | 7 |
| 7 | Index3 | 0.190025 | 22 | 11 |
| 8 | Middle1 | 0.145386 | 17 | 8 |
| 9 | Middle2 | 0.098908 | 13 | 6 |
| 10 | Middle3 | 0.183358 | 24 | 12 |
| 11 | Ring1 | 0.133027 | 15 | 7 |
| 12 | Ring2 | 0.100454 | 12 | 6 |
| 13 | Ring3 | 0.209241 | 23 | 11 |
| 14 | Little1 | 0.110856 | 12 | 6 |
| 15 | Little2 | 0.082529 | 10 | 5 |
| 16 | Little3 | 0.092199 | 9 | 4 |

영상은 손 전체와 접촉부 확대,17채널 ON/OFF/invalid, 선택 pad의 법선 합력과0.01N 기준선,
sensor frame gizmo를 함께 표시한다. Green은 실제 ON, gray는 OFF, amber는 invalid이고
cyan 테두리는 시험 대상으로서 접촉 표시와 구분한다. 값은 법선 합력이며 압력·접선 마찰력·
개별 접촉점 힘으로 해석하지 않는다. PNG는 실제 ON인 렌더링 표본을 사용한다.

### 시험 중 발견과 검증 범위

초기 kinematic anchor 시험은 pose write 뒤 이전 주차 target으로 복귀하여 접촉이 생기지
않았다. 별도의 `set_kinematic_targets()` 시도는 이 GPU backend에서 미구현 오류가 났다.
최종 fixture는 kinematic anchor를 제거하고 world joint의 기준을 시험 사이에만 옮긴다.
이 예비 실패를 학습 sensor의 고장이나 GPU launch failure로 해석하지 않는다.

앞선 지름6mm 구의51초 시험은16/17만 통과하고 Little3는 반력이0이었다. 같은 표면 대표점과
접근 방향에서 지름3mm로 바꾸자 Little3가 통과했고, 최종본은 모든 pad를 같은3mm로 다시
촬영했다. Little3 pad의 원본 크기는약6.47×4.66mm다. CPU STL 분석에서6mm 구와 부모
`right_little_2`의 여유가약0.395mm여서 주변 형상 선접촉 가능성을 뒷받침하나, cooked shape별
접촉을 분리한 검증은 아니다. 단일 원인을 확정하지 않으며 production collider는 변경하지 않는다.
큰 물체·넓은 접촉면·모든 pad 위치에서 감지된다는 보장이나 실물 tactile 교정 결과가 아니다.
특히 접촉 coverage와 지속 접촉 제어는 후속 검토 대상으로 남긴다.

- 최종 영상/PNG: `docs/videos/tactile_contact_2026-11-08/all_pads.mp4`, `preview.png`, `pad_00.png`–`pad_16.png`.
- 최종 원자료·명령·console: `logs/diagnostics/tactile_contact_2026-11-08/`의 `report.json`, `commands.json`, `console.log`.
- 예비 실패는 `smoke*`, 지름6mm 결과는 `first_pass_6mm_probe.json/.mp4`로 보존한다.
- 기존 sensor/GUI 관련 CPU16개 검증과 진단 syntax·채널·실제 접촉 대조를 확인했다.
  최종 console에 Error·CUDA error·overflow는 없었고 GPU 진단 프로세스도 모두 종료했다.
  실제 학습이나 PPO update는 실행하지 않았다.

재현:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py \
  --tactile-scan --headless --device cuda:0
```


## 139. 2026-11-08 — UR5e·F/T·Hand 결합 시스템의 OSC 물체 밀기 영상

사용자는 Hand 단독 기부 이동 시험을 넘어, 실제 UR5e·F/T·Hand 결합 시스템을 OSC로
구동해 물체를 미는 영상을 요청했다. 기존 `probe_ft_compensation.py`에 `--osc-push`를
추가했다. **학습된 policy의 성능 시험이 아니라 scripted 6D wrench를 실제 production
`SensorFrameWrenchAction`·공식 OSC·IdealPD에 전달하는 통합 구동 시험**이다.
학습용 robot·collider·질량·관성·Hand actuator·reward·termination·OSC 설정은 바꾸지 않았다.

### 초기 기하와 명령 경로

- Production default의 Palm은 아래를 향하므로, 초기화 때만 `wrist_3_joint`에 +π/2를
  더해 pad 법선을 수평으로 맞췄다. Shoulder/elbow 초기각은 유지한다. Hand는 native
  joint-limit mapping의 fraction0.15를6개 action으로 입력하고 URDF multiplier로12개
  목표각을 만든다. 초기 joint state와 box pose/velocity만 한 번 배치하며 이후 덮어쓰지 않는다.
- Box는220×180×200mm·0.300kg dynamic cuboid이며 중력 ON, 공식 기본 마찰계수
  static/dynamic0.5/0.5다. 지지면은 높이1.36m의 never-moved kinematic cuboid다.
  물체·Hand에 guide를 연결하거나 외부 body force로 이동시키지 않는다.
- 원본 Hand collision container 아래 mesh를 실제 native body pose로 변환해 box의 초기
  앞면을 배치했다. 초기 Hand 최저점은1.445879m로 지지면보다약85.9mm 높다. 초기 box
  중심은 `(-0.641022, -0.154719, 1.460500)m`, 수평 밀기 방향은약
  `(-0.247404, -0.968912, 0)`이다. 실제 물리 접촉 형상은 수정하지 않았다.
- Baseline1.2초→approach1.6초→push1.6초→brake0.8초→settle2.4초, 총7.6초다.
  최대 수평 힘1.8N을 부드럽게 명령하고 접촉 중 `r×F` 모멘트를 더한다. 물체가8cm
  이동하면 이후 force schedule을 중단하고 −1N half-sine0.6초의 제동 명령으로 전환한다.
- 자유 회전을 억제하기 위해 **진단용 명령 생성기에만** 초기 센서 자세에 대한 angular PD를
  사용했다. 공식 `compute_pose_error()`의 회전 오차에 Kp0.5N·m/rad·Kd0.02N·m·s/rad를
  적용하고 보정 torque norm을1.5N·m로 제한한다. 이는 목표6D wrench의 torque 성분을
  만드는 시험 script이며 production OSC의 내부 pose/힘 controller를 바꾼 것이 아니다.
  OSC 내부6축 F/T feedback은 계속 OFF이고, 이번 시험은 F/T 오차 추종 검증이 아니다.
- 실제 제어는 physics/OSC100Hz·명령 갱신50Hz다. Sensor-origin wrench scale·clip·COM
  변환과 gravity compensation·IdealPD150/28N·m 상한, Hand PD20/1·effort1N·m·
  armature0.0001을 그대로 사용한다. Camera·frame gizmo·영상 overlay는 물리에 참여하지 않는다.

### 최종 촬영의 실제 결과

최종 `report.json`의760physics 표본을 기준으로 하며, 같은 설정의 예비 `smoke5.json`
수치를 최종 촬영 결과로 대신 사용하지 않는다.
최종 MP4는1280×1264·25fps·190frames·7.6초이며 실제 영상 metadata와 프레임 수를 확인했다.

| 측정 항목 | 최종 촬영 결과 |
|---|---:|
| 최종 밀기 방향 변위 | 85.8038mm |
| 최대 밀기 방향 변위 | 85.8045mm |
| 최대 box 기울기 | 0.01978° |
| 최저 OBB corner의 최소 높이 | 1.359997m |
| 지지면 XY 경계 밖으로 나온 최대 거리 | 0m |
| Hand–box 법선 합력 최대 | 2.09775N |
| 12개 Hand 관절의 최대 목표 오차 | 0.43421° |
| Hand URDF 위치 한계 최대 위반 | 0rad |
| Arm 최대 관절 속도 | 1.34787rad/s |
| SI-200-8 overrange 표본 | 0/760 |
| 17개 tactile ON 표본 | 0/760 |

**물체가 이동했지만17개 tactile은 모두 OFF였다.** 별도의 native Hand-body×box contact
view에서 실제 접촉은 `right_middle_2`(최대1.59175N), `right_thumb_2`(1.73439N),
`right_thumb_3`(2.09775N)에 기록됐고 모든 tactile body의 box 반력은0이었다. 각 body의
최대치는 서로 다른 시점일 수 있으므로 합산하지 않는다. 이 결과는 Hand 본체로 힘을
전달해 밀었음을 보여주며, pad면 접촉 유지·17채널 감지 coverage를 확인한 결과는 아니다.
주변 collision 형상과 pad 노출 관계는138절의 소지 끝 문제와 함께 후속 검토 대상으로 남긴다.

실제 raw OSC torque와 IdealPD 제출 torque는 모든 표본에서 같았고, 각 축 최대 절댓값은
약 `[1.046, 13.823, 18.593, 2.621, 0.641, 0.087]N·m`로 상한에 도달하지 않았다.
F/T 보상값은 이 접촉 중0이어야 하는 값이 아니다. 영상에는 명령·전체 측정·예측 self-load·
보상 후6축, sensor frame gizmo, box 변위·기울기, Hand 오차·한계 위반,17개 실제 binary를
동시에 표시한다. 명령 wrench·측정 F/T·법선 접촉 합력은 서로 다른 측정량으로 구분한다.

### 예비 실행과 해석의 범위

- `smoke2.json`: 열린 Hand·최대2.2N에서 물체가약335.4mm 이동해 진단 경계로 중단됐다.
- `smoke3.json`: 제동으로 물체 이동은약109.3mm였지만 Hand가 크게 회전했다. Pure wrench의
  zero command는 자세 유지 명령이 아니므로 이를 물체 밀기 성공의 충분한 기준으로 삼지 않았다.
- `smoke4.json`: 진단 angular PD3/0.6에서 baseline11step 중 arm 속도3.004rad/s로 중단됐다.
  Gain을 키우는 것만으로 자세가 안정된다고 판단하지 않는다. Production 설정은 변경하지 않았다.
- `smoke5.json`: 진단 angular PD0.5/0.02·fraction0.15·최대1.8N에서86.2363mm 이동했다.
  최종 영상은 같은 조건을 별도 실행한85.8038mm 결과다. 두 실행을 bit-exact하다고 주장하지 않는다.

통합 시스템이 실제 OSC torque로 지지된 물체를 밀 수 있음을 확인한 시험이다. 학습 policy의
rapid manipulation, pad로 지속 접촉, 모든 자세의 안정성, 센서 외력 교정, 실물 일치나
과거 CUDA719의 원인 해결까지 검증한 것은 아니다. 학습이나 PPO update는 실행하지 않았다.

- 영상: `docs/videos/osc_hand_push_2026-11-08/osc_push.mp4`.
- 최종 원자료: `logs/diagnostics/osc_hand_push_2026-11-08/report.json`.
- 예비 원자료는 같은 directory의 `smoke2.json`–`smoke5.json`으로 보존한다.
- `path_order_check.json`의1step 추가 확인에서32개 Hand body와1개 box의 native contact
  view 순서가 기록 이름과 일치함을 검사했다. 최종 영상과 실제 측정 표본을 대조하고
  `summary.json`·`commands.json`·접촉 중 미리보기도 저장했다. Console에 Error·CUDA error·
  overflow는 없었고 모든 GPU 진단 프로세스가 종료됐음을 확인했다.

재현:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py \
  --osc-push --headless --device cuda:0 --push-force 1.8 --push-hand-fraction .15 \
  --push-orientation-kp .5 --push-orientation-kd .02
```


## 140. 2026-11-08 — 사전 카탈로그 없는 9단계 순차 스폰 예시

### 요청과 구현 범위

사용자는 기존 배치를 교체할 후보로 Target 종류·yaw 선택 → Target 배치 → 정면 폭에
margin을 더한 Clearance 설정 → Blocker 선택 → 정면 가로폭10–50% 겹침 배치 → 재배치 영역
예약 → 남은 공간 채우기 → 정면 ray cast → 로봇 자세 randomization의9단계를 지정했다.
가림은 실제 메시 실루엣 면적이 아닌 **정면 가로폭의 겹침 비율**로 확정했다. 추가로 재배치
영역은 Clearance 좌우 경계에 붙어야 하며, 그 영역 앞에서 선반 정면까지 같은 폭을 비운다.

`src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py` 단일 진단 파일을 추가했다.
학습용 환경 설정·reset·reward·termination·모델·제어 코드는 변경하지 않았다. 진단 cfg에서
기존 reset 이벤트를 연결하지 않고 `generation_mode=sequential_probe`로 고정 자원을 초기화한다.
카탈로그 초기화·기존 FFT·MILP 진입점은 호출하면 예외를 내도록 진단 프로세스에서만 막았다.
USD 고정 형상, ray mesh BVH와 RobotPoseModel은 처음 한 번 준비하며 장면의 사전 도메인
카탈로그는 읽거나 만들지 않는다. 이 고정 자원 준비와 생성9단계의 시간을 분리했다.

### 실제 한 장면

- Seed0, Clearance 좌우 margin은 각각10mm의 임시 예시 값.
- Target: YCB003 cracker box1개, Blocker: cup_1.
- 총25개: Target1, cup_1·bottle_1·can_1·mustard 각각5, mug_1은4개.
- 최종 카메라의 정면 원근 투영에서 지정 Blocker와 Target의 가로폭 겹침은10.50963196%.
  이는 실제 가려진 실루엣 면적이10.51%라는 뜻이 아니다.
- 재배치 영역2개는 각각 Clearance 좌우 Y 경계와 정확히 접한다. Blocker 전체가 들어가는
  공간이며 goal yaw는 초기 Blocker와 같다. 두 영역 앞의 같은 Y 폭 통로는 실제 선반
  정면까지 비어 있다. 초기 Blocker도 이 영역·통로를 침범하지 않는다.
- 실제 collision footprint의 모든 쌍에 초기 겹침이 없고, support 경계 밖 물체·주변 물체의
  Clearance/예약 영역/앞 통로 침범이 없다. 독립 CPU SAT/거리 검증의 최소 물체 간격은4.069mm.
- object-only64×48 ray cast에서 Target·Blocker를 포함해10개 유효 observation OBB를 얻었다.
- 기존 batched IK의 검증된 anchor 이후 연속 위치·회전 후보를 새로 뽑았다. Anchor 후보1,
  선택 후보5였고 anchor fallback은 없었다. 관절 한계·elbow-up·로봇 충돌 검사에 더해 실제
  articulation FK 일치를 확인했다. 실제 physics integration이나 policy/PPO update는 실행하지 않았다.

### 시간 측정

RTX3090, CUDA0,1환경에서 `torch.cuda.synchronize()` 전후 wall time을 측정했다.
배치는 CPU numpy 기하이며 ray cast와 batched IK는 CUDA다. 같은 seed·동일 장면을3회 더
계산하고 좌표가 첫 장면과 같음을 assert했다. 아래는 첫 호출과 반복3회의 단계별 중앙값이다.
렌더링을 단계 사이에 넣어 계산 시간을 부풀리지 않고, 먼저1–7단계를 저장한 후 각 중간 상태를 촬영했다.

| 단계 | 첫 호출 ms | 반복 중앙값 ms |
|---|---:|---:|
| 1 Target 선택 | 0.167 | 0.154 |
| 2 Target 배치 | 1.724 | 1.676 |
| 3 Clearance | 0.215 | 0.206 |
| 4 Blocker 선택 | 0.143 | 0.129 |
| 5 Blocker 배치 | 0.623 | 0.595 |
| 6 재배치 영역·앞 통로 | 0.444 | 0.420 |
| 7 주변 물체 채우기 | 55.660 | 55.487 |
| 8 ray observation | 86.369 | 12.092 |
| 9 anchor·랜덤 IK·native FK 검증 | 1223.252 | 25.314 |

첫 호출 합1368.598ms, 단계별 반복 중앙값 합96.073ms다. 후자는 한 번의 end-to-end
reset 측정값이나1024환경 처리량이 아니다. 실제3회 전체 시간은 `report.json`에 남긴다.
첫 ray/IK 호출에는 상수·kernel/CUDA graph 준비 비용이 포함된다. 이 실행의 앱 시작부터
고정 자원 준비까지는40.544초(그중 native 환경21.783초, 형상/BVH/RobotPoseModel2.187초),
9장 이미지의 임시 상태 반영·렌더링·PNG 저장 합은4.528초이며 모두 위 계산 시간과 별도다.
Native 물체 state/활성 flag를 실제 학습 reset으로 쓰는 비용까지96ms에 포함됐다고 주장하지 않는다.

### 빈 공간 표현과 예시의 제한

초기 진단에서 비어 있는 stage1을 기존 `_episode_proposals`에 넘기는 보조 코드 오류를
고쳤다. 이후 disjoint rectangle 분할은 큰 자유 공간을 조각내4개에서 채우기가 멈췄다.
빈 공간을 서로 겹칠 수 있는 최대 빈 직사각형으로 유지하고, 배치 후 모든 구획에서 점유
부분을 빼고 포함 관계를 정리하도록 고쳤다. 각 배치에서 들어가는 종류 중 개수가 적은
종류를 선택하며, 한 종류가 안 들어간다고 즉시 전체를 중단하거나 작은 한 종류로만
채우지 않는다. 장면 재생성·무작위 자세 재제안으로 결과를 고른 것은 아니다.
최종 실행은 사용자 추가 요구인 인접 영역·앞 통로까지 적용한25개 장면이다. 예비4개
그림과 보조 렌더링의 CUDA tensor→numpy 변환 오류 자료는 logs의 preliminary directory에 보존한다.

예시는 완성된 production generator가 아니다. Target 배치가 뒤의 모든 종류를 위한 공간을
보수적으로 예약하며, 각 미래 Blocker 종류의 임시 yaw를2단계에서 먼저 뽑는다. 종류는4단계에서
균등하게 고른다. 주변 물체 yaw는0/90/180/270°만 사용하고, 구획·종류 선택은 CPU의 유한 반복이다.
모든 종류 조합·독립적인360°·모든 seed·최소15개를 증명하지 않았으며 최대 수용량도 보장하지 않는다.
목표 영역과 앞 통로가 비어 있어도 전체 밀기/pivot 경로가 실행 가능하다는 인증은 아니다.
기존 Blocker95% 실루엣 가시성 gate나 시야 원뿔은 몰래 복원하지 않았다.

### 산출물과 재현

- `docs/figures/sequential_spawn_2026-11-08/01_step.png`–`09_step.png`, `overview.png`.
- 같은 directory의 `timings.csv`, `report.json`, `layout_stages.json`, `resources.json`.
- `logs/diagnostics/sequential_spawn_2026-11-08/console.log`, `independent_cpu_audit.json`.
- 전체 collision support vertices를 포함한 최초 원자료는 같은 logs의
  `full_report_with_collision_support.json.gz`에 보존하고 figure의 report는 resource 파일을 참조한다.
- 시각화용 선·지지면에는 RigidBodyAPI/CollisionAPI가 없음을 검사했다. 촬영 프로세스 종료와
  GPU compute process가 남지 않음을 확인했다. Prototype Python 구문 검사와 diff 검사를 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py \
  --headless --device cuda:0 --seed 0
```

## 141. 2026-11-08 — 서로 다른30개 순차 배치와 회전 목적지 검증

### 규칙과 구현 범위

이번 요청은 물체30개가 아닌 **서로 다른 배치30개**다. 같은 진단 파일에 `--num-layouts`를
추가하고 고정 USD·BVH·로봇 모델을 한 번 준비한 뒤 seed0–29를 순서대로 생성했다.
각 장면의 물체 수는 기존15–25개 범위다. 배치·관측·IK를 측정했으며 학습/PPO나 실제 물리
integration은 실행하지 않았다. Production scene/reset/reward/termination/제어 설정은 변경하지 않았다.

- 주변 물체는 각 종류/배치 순서에서 연속 `[-π,π)` base yaw를 뽑고 그 yaw와+90°의
  들어가는 구획을 계산한다. 최종 yaw가90° 단위로 제한되는 방식이 아니다.
- 좌우 목적지는 각각1개씩이며, 목적지 yaw는 초기 Blocker yaw와 별도로 연속 각도에서 뽑는다.
  두 목적지의 방향·종횡비·X 위치는 서로 다를 수 있다. 목표 방향으로 회전해서 들어간
  Blocker 전체가 맞는 영역을 예약하며, 현재 자세를 유지한 옆 밀기만을 가정하지 않는다.
- 초기 검사에서는 실제 collision·visual footprint만으로 목적지를 만들었다. 이때 일부
  회전 OBB의 빈 모서리가 Clearance와 겹쳤다. 기존 MDP가 OBB로 Clearance/success를
  계산하므로, 최종 규칙은 **회전된 전체 OBB와 실제 형상을 모두 감싸는 경계+4방향3mm**로 고쳤다.
  이 축 정렬 예약 영역이 Clearance 좌우 경계에 접하되 면적은 겹치지 않는다.
- 각 목적지에서 실제 선반 정면까지 같은 폭의 전방 띠도 비운다. 초기 Blocker와 주변
  물체 모두 목적지·전방 띠를 침범하지 않는다. 전체 pushing/pivot 경로를 미리 비우지는 않는다.
- Target 배치에서 가능한 미래 Blocker별 위치 구간을 계산한다. 공집합이 아닌 구간의
  면적 가중 혼합에서 Target을 뽑고,4단계에서 그 위치에 양쪽 목적지가 성립하는 종류 중
  Blocker를 선택한다. 모든 종류에 대한 최대 크기를 공통 예약하여 공간이 사라지는 문제를 피한다.
- 주변 물체는 최대 빈 직사각형의 적합 구간에 직접 배치한다. 현재 최소 개수인 종류가
  더 들어가지 않으면 특정 작은 종류로만 남은 공간을 채우지 않고 중단한다. 목표 개수는
  수용량에 따라 줄 수 있지만 이번30개는 모두15–25개, non-Target 종류별 개수 차이≤1이었다.
- 사전 배치 카탈로그·FFT·MILP·전체 장면 재생성은 없다. 위치/종류/yaw는 공간 조건에 따라
  연동된다. 선반 전체의 균등 분포나 모든 종류 조합의 모든 각도, 모든 seed에서 최소15개를
  일반적으로 보장했다는 뜻은 아니다.30개 장면의 검증 결과와 일반 보장을 구분한다.

### 검증과 이미지

최종 seed0–29 모두 물체15–25개·실제 종류4–10종·Target 종류1개를 유지했다. 물체 쌍의
실제 collision footprint 겹침·선반 이탈·주변 물체의 예약 영역 침범이 없다. 독립 CPU 검사의
최소 실제 footprint 간격은3.0000546mm였다. 목적지60개 모두 회전 OBB 전체가3mm 이상
안쪽에 들어가고 Clearance와 면적 겹침이0이며, 영역 앞도 비어 있다. 표시 선·면은
RigidBodyAPI/CollisionAPI가 없는 시각화다.

30개 모두64×48 object-only ray에서 Target·Blocker의 유효 관측을 얻었다. 로봇은 기존
batched IK로 anchor와 새로운 위치·회전 후보를 구해 elbow/관절/충돌 조건 및 실제 articulation
FK를 확인했다.30개 모두 통과했고 anchor fallback은0이었다. 전도·settling·조작 성공·전체
회전 경로의 충돌 여부를 physics로 검증한 결과는 아니다.

- `docs/figures/sequential_spawn_layouts30_2026-11-08/all_top_views.png`:30개 정사영 top-view 모음.
- 같은 directory의 `all_perspective_views.png`:30개 사선 시점 모음.
- `layout_01`–`layout_30`의 `final_top_view.png`, `final_perspective.png`, `report.json`.
  첫 장면에는1–9단계 그림도 남겼다. 초록 Target·빨강 Blocker·파랑 Clearance·노랑 목적지·주황 앞 통로다.
- 전체 시간은 `timings.csv`, 통계·장면별 검증은 `summary.json`, 고정 기하는 `resources.json`.
- 원자료: `logs/diagnostics/sequential_spawn_layouts30_2026-11-08/console.log`,
  `independent_cpu_audit.json`, `goal_clearance_audit.json`.
- 실제 footprint만 사용했던 첫30개 결과와 OBB 불일치 감사는 같은 logs의
  `first_pass_actual_footprint/`에 보존한다. 최종 figures는 OBB까지 반영한 두 번째 결과다.

### 시간

RTX3090·CUDA0, native 환경1개 재사용, 각 단계 전후 CUDA synchronize를 포함한 wall time이다.
배치1–7단계는 CPU numpy, ray/IK kernel은 CUDA를 사용한다. 같은 장면 반복 benchmark가
아니라 서로 다른30개 장면을 각각 한 번 실행했으며, 초기 kernel/graph 준비도 포함한다.

| 단계 | 평균 ms | 중앙값 ms |
|---|---:|---:|
| 1 Target 선택 | 0.166 | 0.158 |
| 2 Target 위치·미래 공간 구간 | 5.182 | 5.144 |
| 3 Clearance | 0.396 | 0.357 |
| 4 Blocker 선택 | 0.245 | 0.238 |
| 5 Blocker 배치 | 1.152 | 1.115 |
| 6 회전 목적지·앞 통로 | 0.969 | 0.839 |
| 7 주변 물체 채우기 | 36.849 | 33.800 |
| 8 ray observation | 10.282 | 7.510 |
| 9 로봇 검색·상태 적용·FK | 568.788 | 871.752 |

기하1–7 합은 평균44.959ms, 전체1–9 합은 평균624.028ms·중앙923.624ms·p951094.803ms다.
30장면의 전체 계산 합은18.721초다.9단계의 범위는22.444–1293.487ms로 넓다. anchor검색,
연속 후보 검색, native joint/root 쓰기·forward·FK 검증까지 포함하므로 **순수 IK kernel 시간**으로
해석하면 안 된다. 그 안에서 무엇이 변동을 만드는지는 이번 측정만으로 분리하지 않았다.

고정 자원 준비는 앱 시작부터41.465초(그중 native 환경21.421초, 기하/BVH/로봇 모델3.005초).
상태 반영·진단 RGB 렌더링·PNG 저장 합은25.554초로 위 계산과 별도다. 첫 장면의9단계 촬영
4.963초를 포함하므로, 전체 렌더 평균851.813ms보다 최종2시점 기준 중앙716.571ms를 함께 본다.
독립 audit·JSON 저장·모음 그림 작성은 별도이며 이 합을 전체 앱 wall time이라고 부르지 않는다.
1024환경 reset/학습 iteration 성능으로 환산하지 않는다. Console에 CUDA/PhysX error가 없고
진단 GPU 프로세스가 종료됐음을 확인했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py \
  --headless --device cuda:0 --seed 0 --num-layouts 30 --timing-repeats 0 \
  --output docs/figures/sequential_spawn_layouts30_2026-11-08
```


## 142. 2026-11-08 — 목적지 최소 한쪽과 Target–Blocker 간격 랜덤화

사용자는 양쪽 목적지를 필수로 만들 필요가 없으며 최소 한쪽이면 충분하다고 수정했다.
또한 Target–Blocker 간격도 랜덤이어야 한다고 요청했다. 기존 순차 스폰 진단 파일만
수정했고, 학습용 reset·reward·termination·모델·제어 설정은 변경하지 않았다.

### 생성 규칙 변경

- Target 위치를 정할 때 미래 Blocker 종류마다 **왼쪽만·오른쪽만·양쪽**의 유효 위치
  구간을 각각 계산한다. 그 합집합에서 Target을 배치하고, Blocker를 고를 때 해당 위치에서
  가능한 종류와 목적지 조합을 선택한다. 양쪽을 먼저 예약한 뒤 표시만 하나 지우는 방식이 아니다.
- 선택하지 않은 쪽에는 목적지나 전방 띠를 예약하지 않는다. 주변 물체는 그 공간에도
  배치될 수 있다. Target의 좌우 위치 구간에는10–50% 가림을 만드는 초기 Blocker가 들어갈
  폭을 보수적으로 남기지만, 그 폭 자체를 빈 공간으로 빼지는 않는다.
- 초기 Blocker X는 기존 `3–10mm` 간격 구간 대신, 선택한 목적지의 전방 공간이 성립하는
  **계산된 연속 X 구간 전체**에서 균등 샘플링한다. 물체 실제/visual footprint를 감싼
  축 정렬 외곽의 X 최저·최고를 사용하면 다음 범위다.

```text
B_x_min = Target_front + gap - Blocker_local_x_min
B_x_max = shelf_front - Blocker_local_x_max - deepest_selected_pocket_depth - gap
gap = 0.003 m
pocket_depth = rotated_goal_envelope_depth + 2 * gap
```

이 상한은 초기 Blocker가 목적지의 Y 띠와 겹칠 때에도 그 앞에 목적지와 빈 전방 띠를 둘 수
있도록 잡은 보수적인 예약 계획 범위다. 모든 가능한 XY 배치의 최대 범위와 같다는 뜻은 아니다.
Y는 정면 투영 폭10–50% 가림과 Clearance 겹침을 동시에 만족하는 구간에서 직접 샘플링한다.
장면 전체를 재생성하는 재시도나 사전 배치 카탈로그를 추가하지 않았다.

목적지 yaw는 초기 Blocker와 별도의 연속 각도이며, 회전된 전체 OBB+실제 형상+4방향3mm
여유를 포함한다. 목적지와 Clearance의 면적 겹침은0이고 경계만 접한다. 선택한 각 목적지 앞은
실제 선반 정면까지 비우며, 주변 물체의 연속 yaw·Target 종류 단일성·종류별 개수 균형도 유지한다.

### 새30개 장면과 독립 검증

같은 고정 자원을 한 번 준비하고 seed0–29를 생성했다. 물체는 모두15–25개,
non-Target 종류별 개수 차이는 최대1개였다.

- 왼쪽만8개, 오른쪽만11개, 양쪽11개로 총 목적지41개.
- 모든 장면에서 실제 충돌 footprint 겹침·선반 범위 이탈·주변 물체의 예약 공간 침범 없음.
- 목적지41개 모두 실제 collision/visual 형상과 회전 OBB가 Clearance 밖이며, 영역 안쪽과
  Clearance 측면에 최소3mm 여유가 있다. 초기 Blocker를 포함한 모든 물체가 목적지·앞 통로 밖이다.
- 최소 물체 footprint 간격3.0000185mm. Native 결과와 별도 CPU 생성 결과가30/30 정확히 일치했다.
- 30개 모두 Target·Blocker의 유효 ray 관측, batched IK의 새로운 시작 자세, 실제 articulation
  FK를 통과했다. Anchor fallback은0개였다.

| Target–Blocker 거리 | 최소 mm | 중앙 mm | 최대 mm |
|---|---:|---:|---:|
| X 방향 외곽 지지면 간격 | 3.327 | 26.220 | 104.436 |
| 실제 collision footprint 사이의 최단 유클리드 거리 | 6.079 | 48.896 | 127.096 |

두 거리는 다르다. X 구간을 랜덤 샘플링하며, 형상과 yaw 및 Y 위치가 함께 정해진 뒤 실제
최단거리가 결정된다. 위 실제 최단거리가 균등분포가 되도록 샘플링했다는 뜻은 아니다.
30개 중25개는 X 간격도10mm보다 컸다. 가능한 조합/각도/위치의 조건부 샘플링이며
모든 seed·종류 조합에 대해 같은 수용량을 일반적으로 증명한 것은 아니다.

### 시간과 산출물

RTX3090·CUDA0·단일 native 환경 재사용, 단계 전후 CUDA 동기화 wall time이다.

| 계산 범위 | 배치당 평균 ms | 중앙 ms |
|---|---:|---:|
| 1–7 배치 기하 생성 | 48.558 | 48.019 |
| 8 ray observation | 9.607 | 6.678 |
| 9 로봇 검색·상태 적용·FK | 514.474 | 907.618 |
| 1–9 합 | 572.639 | 949.055 |

30개 계산 합17.179초, 별도 촬영·상태 반영·PNG 저장 합24.550초, 앱 시작부터 고정 자원
준비까지40.717초였다. 첫 장면은9단계를 촬영하고 나머지는 최종2시점을 촬영한다.
로봇 단계는 순수 IK kernel뿐 아니라 anchor·랜덤 후보 검색과 native state/forward/FK를 포함한다.
기하 계산은 CPU numpy, ray/IK kernel은 CUDA다.1024환경 학습 iteration 시간으로 환산하지 않는다.
물리 integration·정책/PPO 실행·실제 밀기 경로 검증은 하지 않았다.

- 최종 이미지/원자료: `docs/figures/sequential_spawn_layouts30_variable_2026-11-08/`.
  `all_top_views.png`, `all_perspective_views.png`, `layout_01`–`layout_30`의 개별 이미지·report,
  전체 `timings.csv`, `summary.json`, `resources.json`을 저장했다.
- CSV와 summary에 목적지 쪽과 Target–Blocker X 간격도 남겼다.
- 독립 감사: `logs/diagnostics/sequential_spawn_layouts30_variable_2026-11-08/`의
  `independent_cpu_audit.json`, `independent_geometry_audit.json`, `goal_gap_audit.json`.
- Native 실행 로그는 같은 directory의 `console.log`, 생성 코드 snapshot은 `probe_sequential_spawn.py`다.
- 이전 양쪽 필수 예시와141절 결과는 덮어쓰지 않았다. Overlay에는 물리 API가 없고,
  최종 console에 CUDA/PhysX error가 없으며 진단 GPU 프로세스가 종료됐음을 확인했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py \
  --headless --device cuda:0 --seed 0 --num-layouts 30 --timing-repeats 0 \
  --output docs/figures/sequential_spawn_layouts30_variable_2026-11-08
```


## 143. 2026-11-08 — 초기 Blocker 앞의 접근 통로 예약

사용자 요청으로142절 규칙에 **초기 Blocker 앞도 비우는 조건**을 추가했다.
`tests/probe_sequential_spawn.py`의 순차 생성 예시만 수정했으며 학습용 reset에는 연결하지 않았다.

- 5단계에서 초기 Blocker yaw의 전체 OBB와 실제 collision/visual 형상을 감싼 XY 외곽을 구한다.
- 통로 X는 그 외곽의 앞면부터 실제 선반 정면까지다. Y는 전체 외곽 폭에 좌우3mm를 더하고,
  선반 실제 Y 경계 밖으로 나간 부분만 자른다. 손 전체 폭이나 전체 재배치 경로를 예약한 것은 아니다.
- 7단계의 빈 구획에서 이 통로를 추가로 제외한다. 목적지·목적지 앞 통로와 겹치는 것은 허용한다.
  모두 비워두는 공간이므로 이를 이유로 장면을 다시 생성하거나 다른 목적지를 고르지 않는다.
- Marker는 **분홍색**, 기존 목적지 앞은 주황색이다. 선과 바닥 표시에는 물리 API가 없다.
- 난수 호출을 추가하지 않았다. seed0–29 모두 기존142절과 Target·Blocker 종류/yaw/pose,
  Clearance·목적지·카메라가 정확히 같고, 새 예약 공간에 따라 주변 물체 배치만 달라진다.

### 30개 장면 검증과 측정

30개 모두 물체15–25개와 non-Target 종류별 개수 차이≤1을 유지했다. 목적지는 이전과 같은
왼쪽8개·오른쪽11개·양쪽11개(합41개)다. Blocker 전방 띠에 초기 Blocker·Target을 포함한
어떤 활성 물체의 collision/visual footprint도 면적으로 겹치지 않는다. 초기 Blocker의
외곽과 통로의 접촉 경계는 허용한다. 기존 목적지41개의 회전 OBB와 Clearance 비중복·각 방향
3mm 여유·목적지 앞 공백, 전체 물체 충돌/선반 범위 검사를 통과했다. 최소 실제 footprint
간격은3.00000075mm이며 Target–Blocker 간격 분포는142절과 같다.

Native object-only ray에서30개 모두 Target·Blocker 유효 관측을 얻었고, 로봇 시작 자세의
batched IK·충돌·실제 articulation FK도30/30 통과했다. Anchor fallback은0개였다.
실제 physics integration·정책/PPO 실행·밀기 경로의 실행 가능성을 검증한 것은 아니다.

RTX3090·CUDA0·단일 native 환경의 서로 다른30개 장면을 단계별 CUDA 동기화 wall time으로 측정했다.

| 계산 범위 | 배치당 평균 ms | 중앙 ms |
|---|---:|---:|
| 1–7 배치 기하 생성 | 48.693 | 48.889 |
| 8 ray observation | 9.944 | 7.325 |
| 9 로봇 검색·상태 적용·FK | 384.213 | 25.360 |
| 1–9 합 | 442.850 | 96.847 |

30개 계산 합13.286초, 촬영·상태 반영·PNG 저장 합24.899초, 앱 시작부터 고정 자원 준비까지
39.535초였다. 기하는 CPU numpy, ray/IK kernel은 CUDA다. 로봇 단계는21.581–1143.256ms로
변동이 크며 순수 IK kernel 시간만이 아니다. 이전 실행과 평균 차이를 이 예약 공간 추가의
최적화 효과로 해석하지 않는다.1024환경 reset/학습 iteration benchmark로 환산하지 않는다.

- 최종 이미지/원자료: `docs/figures/sequential_spawn_layouts30_blocker_front_2026-11-08/`.
  `all_top_views.png`, `all_perspective_views.png`, `layout_01`–`layout_30`의 개별 이미지·report,
  `timings.csv`, `summary.json`, `resources.json`을 남겼다.
- 독립 감사 및 실행 로그: `logs/diagnostics/sequential_spawn_layouts30_blocker_front_2026-11-08/`의
  `independent_cpu_audit.json`, `independent_geometry_audit.json`, `blocker_front_audit.json`, `console.log`.
  생성 코드 snapshot은 같은 directory의 `probe_sequential_spawn.py`다.
- 구문·산출물·독립 기하 검사와 diff 검사를 통과했다. Console에 CUDA/PhysX error가 없으며
  진단 GPU 프로세스가 종료됐음을 확인했다. 이전140–142절 이미지와 원자료는 보존했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py \
  --headless --device cuda:0 --seed 0 --num-layouts 30 --timing-repeats 0 \
  --output docs/figures/sequential_spawn_layouts30_blocker_front_2026-11-08
```


## 144. 2026-11-08 — 순차 스폰 규칙을 Symmetric 학습 환경에 적용

사용자 요청으로143절의 규칙을 **`contact_manipulation_symmetric`의 실제 reset**에 연결했다.
`shelf_manipulation_env_cfg.py`의 기본 모드는 `sequential`이다. 배치 생성 구현을
`mdp/events.py`의 `SequentialSceneGenerator`로 옮겨 학습과 이미지 진단이 함께 호출한다.
생성 규칙을 위한 별도 production 파일은 추가하지 않았다. 기존 두 프로젝트는 변경하지 않았다.

### 실제 reset 흐름과 관측·보상 연결

1. Target 종류/yaw와 배치 가능한 연속 위치 구간에서 위치를 정한다. 후속 Blocker·목적지가
   들어갈 수 있는 공간을 미리 계산하되 완성 장면이나 역할 조합을 사전 저장하지 않는다.
2. Target의 정면 실제 visual 폭에 좌우10mm를 더한 Clearance를 만들고, 다른 종류의
   Blocker를 정면 투영 가로폭10–50% 겹침과 Clearance 겹침이 성립하는 구간에 직접 배치한다.
   앞뒤 간격은3mm부터 목적지 깊이를 남길 수 있는 연속 상한까지 랜덤화한다.
3. Clearance 옆에 붙은 목적지를 왼쪽만·오른쪽만·양쪽 중 최소1개 생성한다. 목적지 yaw는
   초기 Blocker와 별도로 연속 샘플링하며, 실제 형상과 전체 회전 OBB 및 각 방향3mm 여유를
   포함한다. Clearance와 면적 겹침은 없고 목적지 앞은 선반 정면까지 비운다.
4. 초기 Blocker의 전체 회전 외곽 앞면부터 선반 정면까지도 같은 폭+좌우3mm로 비운다.
   Target·Clearance·Blocker·목적지·두 종류 전방 통로를 제외한 빈 구획에 주변 물체를 채운다.
   Target 종류는 한 개, non-Target 종류별 개수 차이는 최대1개다. 전체 push/pivot 경로는
   예약하지 않는다. 기존143절의30개 배치에서 역할·목적지·주변 pose가 유지됨을 확인했다.
5. Target 정면의 생성된 가상 시점에서 물체 메시만 ray cast한다. 기존64×48 해상도와
   관측 최대10물체·역할 우선 슬롯·모델 입력 구조는 유지한다. 시점·배치 후보를 재평가하는
   기존 `_capture_constructive_views`나 사전 visibility catalog 경로는 호출하지 않는다.
6. 기존 batched IK·elbow-up·충돌 검사와 실제 articulation FK 검증으로 로봇 시작 자세를
   정하고 native 상태 writer로 반영한다. 잘못된 규칙을 숨기기 위한 장면 재추첨은 없다.

새 가림 기준은 사용자 선택인 **정면 가로폭 겹침**이다. 단독 실루엣 면적 대비 가시성 비율은
이 경로에서 계산하지 않으며, 그 진단 필드는0이나 추정치를 넣는 대신 `(-1,-1)`로 표시한다.
실제 ray 관측에서 Target·Blocker의 유효 표면 정보가 없으면 오류를 보고하고 장면을 재생성하지
않는다. 생성 결과의 실제 역할 ID를 배치 tensor에 기록해 물리·관측·접촉 보상과 일치시켰다.
Blocker만 종류를 고정한 경우에는 Target 후보에서 그 종류를 제외하며, pool의 첫 번째가 아닌
Blocker ID도 배경에서 중복 사용하지 않는다.

재배치 reward의 영역은 **새 생성기의 목적지 사각형만** 사용한다. 전방 접근 통로는 reward의
목적지에 넣지 않는다. Blocker 중심과 영역은 모두 환경 원점 기준 XY다. 보상 수식·weight·
종료 조건·모델·PPO·로봇 제어 및 물리 주기는 변경하지 않았다. Actor vision 주기도 현재 설정인
`0`(reset-only)을 유지한다. 과거10Hz 실험 설정으로 자동 변경하지 않았다.

### 병렬 reset과 물체 수

새 장면은 env0에서 배치1회·ray query1회 준비한 뒤 전체 환경에 공유한다. 중간 종료 환경의
부분 reset은 같은 배치·관측을 복원하고 로봇 시작 자세만 새로 뽑는다. 기존 미사용 body의
kinematic 전환·native manager/writer·반복 난수 스트림·IK 실패 행 복구 경로를 유지한다.

기본 요청 수는15–25개이며, 빈 구획과 종류 균형의 수용량에 따라 요청 수보다 적을 수 있다.
추가 검사에서 seed189는 연속 base yaw와 그+90°만으로14개에서 멈췄다. 이 경우에 한해
현재 최소15개보다 적고 최소 개수인 종류가 들어가지 못할 때, 남은 빈 구획에 맞는 **연속 yaw
허용 구간을 해석적으로 계산**하도록 보완했다. 각도 grid나 기존 장면 재생성은 사용하지 않는다.
기존14개의 ID·pose·yaw와 역할·목적지·카메라를 그대로 두고 Tomato 한 개를 추가해15개가 됐다.

최초30개 배치의 기하 상태는 동일하며 진단 metadata4개만 추가됐다. 추가256개 seed30–285는
모두15–25개, 생성 예외0이었다. 이 표본으로 모든 seed의 하한을 증명하지는 않는다.
`minimum_count_shortfall`, `minimum_count_remaining_capacity_reason`,
`minimum_count_yaw_recoveries`에 부족 수·사유·보완 내역을 남긴다. 별도 self-contained 회귀
장면에서도29→30개를 채우며 기존29개·카메라·목적지를 보존하는지 검사했다.

### 실제 Isaac Lab 검증

RTX3090·headless·기존 physics10ms/decimation2 설정에서 **4환경과1,024환경 각각3장면**의
native reset을 실행했다. 매 장면 절반 환경을 부분 reset하고, zero wrench/손 중간 목표의
진단 action으로8control step씩 총24step을 진행했다. PPO 학습·정책 밀기 성능 검사는 아니다.
장기 episode를 기다리는 대신 진단 코드에서 shared-scene cache 만료 시점만 당겨 장면 교체를
유발했으며, common_step_counter나 학습 소스의 종료 조건은 바꾸지 않았다.

- 두 실행 모두 실제 역할 ID·초기 유효 ray 관측·새 목적지 map·관측 `[N,232]`를 확인했다.
  Actor/Critic 관측 전체는 bit-exact이고 step의 관측·reward는 유한했다.
- 각 실행의3장면에서 배치 생성3회·ray query3회, 매 query 요청 수는1이었다.
  부분 reset과 이후 auto-reset에서는 layout/ray 호출 수가 증가하지 않았다.
- 1,024환경의 매 장면512행 모두 새 로봇 자세를 얻었으며 anchor fallback0이었다.
  미선택512행의 관절 상태·reset count와 전체 물체 상태·공유 비전이 유지됐다.
- 장면별 물체 수22·19·21, 목적지 수1·2·1, 정면 폭 겹침11.34%·40.33%·25.12%였다.
- 최종 console에서 CUDA/PhysX 오류가 없었다. 이 검사는 장기 CUDA 장애 재발이나 실제 조작
  성공률을 보증하지 않는다.

1,024환경 실행의 동기화 wall time이다. 아래 full reset은 기하 생성뿐 아니라 ray·IK·전체 상태
반영을 포함하며, PPO collection/iteration 시간이 아니다.

| 장면 | 전체 reset ms | 512행 부분 reset ms | 배치 생성 ms | ray query ms |
|---|---:|---:|---:|---:|
| 최초 준비 포함 | 6183.280 | 54.453 | 68.721 | 112.359 |
| 다음 장면1 | 1482.597 | 85.515 | 48.530 | 5.283 |
| 다음 장면2 | 1348.039 | 44.415 | 42.203 | 4.657 |

원자료는 `logs/diagnostics/sequential_training_reset_2026-11-08/`의 `native4.json`,
`native1024.json`, 각 `*_console.log`, `native_reset_smoke.py`, `geometry_scan.json`이다.
최초 진단의 counter 직접 변경으로 transition guard가 거부한 자료는 `native4_attempt1*`에
별도 보존했다. 진단 방식을 cache 만료로 고친 후 최종 두 실행을 완료했으며 production의
transition 검사를 제거하거나 완화하지 않았다.

최종 CPU 회귀는 MDP533개 중512통과·CUDA/Kit전용21생략, learning76개 중74통과·
CUDA전용2생략이다. 전체609개 중586통과·23생략이며 구문·`git diff --check`도 통과했다.
검사 범위는 배치 불변조건·연속 yaw 해석·역할 ID·한 번의 ray query·실패 시 무재추첨·
goal-only map·부분 reset/rollback과 기존 관측/학습 계약이다. 실행한 진단 GPU 프로세스는
모두 종료했다. 학습 시작 전 검토할320D readout 대 attention-only128D 비교 과제는 기존대로
미결이며 이번 스폰 적용에서 모델 구조를 바꾸지 않았다.


## 145. 2026-11-11 — 현재 Blocker 접촉 pad 수와 step 개선량을 곱하는 보상

사용자 요청으로 `contact_manipulation_symmetric`의 clearance·relocation을 변경했다.
기존 `mdp.clearance(env)`와 `mdp.blocker_relocation(env, decay)` 함수 본문은 그대로 보존했다.
`shelf_manipulation_env_cfg.py`의 기존 두 선언을 주석으로 남기고, 활성 term 이름은
`clearance`·`blocker_relocation`을 유지하면서 각각 `mdp.clearance_contact_progress`와
`mdp.blocker_relocation_contact_progress`에 연결했다. 기존에 주석이던 clearance도 새 함수로
활성화했다. Weight는 각각12·3, relocation decay는10이다.

### 수식과 현재 접촉의 의미

- `C(t) = 1 - 현재 Blocker 가림 면적 / episode 초기 Blocker 가림 면적`:
  기존 signed clearance quality를 사용한다. 초기 유효 가림이 없으면 품질0이다.
- `Q(t) = exp(-10 * d(t))`: 기존 relocation quality이며 `d`는 현재 Blocker 중심에서
  초기 생성된 유효 목적지 polygon까지의 최소 XY 거리다. 영역 내부·경계는 품질1,
  유효 영역이 없으면 품질0이다.
- `n(t)`는 현재 유효한 Blocker 전용 접촉 합력 norm이0.01N을 초과하는 pad의 개수다.
  Canonical17pad를 각각 한 번씩 세며 새 접촉과 기존 접촉을 동일하게1개로 계산한다.
- 새 raw clearance는 `n(t) * max(C(t) - C(t-1), 0)`이다.
- 새 raw relocation은 `n(t) * max(Q(t) - Q(t-1), 0)`이다.

Native RewardManager가 raw 값에 해당 weight와 control dt를 한 번씩 곱한다. Reward 함수가
dt로 나누거나 weight를 중복 적용하지 않는다. 두 새 term에는 이전 step 접촉 여부나
Target 보호 gate를 추가하지 않았다. 현재 무접촉·정체·악화는0이며, 목적지 안에 정지한
상태도 반복 보상을 주지 않는다. Invalid force·센서 validity가 false인 pad는 세지 않으며,
유효하지 않은 현재/이전 기하 비교에서도 보상을 지급하지 않는다.

### 이력·센서·reset 연결

`TaskState.advance()`가 매 완료 control transition마다 직전 clearance 품질과 Blocker
중심을 먼저 저장한다. 접촉이 없어도 갱신하므로 비접촉 개선량을 재접촉 때 소급 지급하지
않는다. 같은 step의 reward·termination 재조회는 기준을 다시 진행시키지 않는다. 부분
reset은 선택 행의 이전 clearance 품질을0, 이전 Blocker 중심을 실제 초기 중심으로 맞추고
다른 행을 유지한다. 따라서 첫 접촉 transition의 실제 개선량도 계산할 수 있다.

접촉은 observation의 any-object flags 대신 기존 native `blocker_force_w`의 단일 Blocker
filter column을 사용한다. Canonical 순서로 pad force를 모으고 동일 순서의 센서 validity를
적용한다. Shared scene의 host Blocker ID로 필터를 idempotent하게 연결하므로 기존
`BlockerContactHold`의 weight가0이거나 term 자체가 없어도 두 새 term을 사용할 수 있다.
Timeline stop/play로 native view가 재생성되면, 등록된 기존 hold term의 재연결·이력 초기화를
먼저 수행한다. 기존 hold의 새 접촉0.5·유지1 수식과 나머지 reward는 유지했다.

### 검증

새 CPU 테스트9개로 양의 step 개선량·현재 pad 수 비례·신규 접촉 포함·비접촉 개선분
소급 방지·최초 transition·같은 step 반복·부분 reset 격리·invalid geometry 회복·native
weight와 dt 적용·timeline 재연결 순서를 확인했다. 접촉 집계는 tensor의 `item`·`__bool__`·
`cpu`·`numpy` 호출을 금지한 상태에서도 검사했다. 기존 두 함수의 AST가 변경 전과 같은지도
확인했고, 기존 함수 동작 테스트는 그대로 유지했다.

- 관련 집중 검사:107통과·1생략·subtest129통과.
- Symmetric 전체 CPU 회귀:596통과·33생략·subtest777통과,29.38초.
- 수정 Python 구문과 `git diff --check` 통과.

전체 검사 로그는 `/tmp/contact_progress_symmetric_cpu.log`다. GPU 시뮬레이션·PPO 학습은
실행하지 않았으며 실제 정책 성능 변화는 측정하지 않았다. 모델·관측·action·PPO와 native
학습 루프는 변경하지 않았다.
