# Sweeping policy 학습 명세

이 문서는 `src/sweeping_policy/sweeping_policy/shelf_sweep`의 **현재 코드가 실제로 계산하는 값**을 기준으로 한다. 핵심 과제는 작업 선반 위에서 무작위로 고른 물체 하나를 EEF로 밀어, 월드 (y)축 방향으로 0.18 m 떨어진 목표점에 보내고 성공 뒤 로봇을 기본 자세로 되돌리는 것이다.

## 한눈에 보는 학습 문제

| 항목 | 현재 구현 |
|---|---|
| 병렬 환경 | 4,096개, 환경 원점 간격 2.5 m |
| 물리 / 제어 주기 | `sim.dt=0.01 s` (100 Hz), `decimation=2`이므로 정책은 50 Hz |
| 최대 episode | 10 s = 500 control step |
| action | 7차원: 상대 EEF pose 6 + binary gripper 1 |
| observation | 35차원: joint 8+6, 이전 action 7, target 3, width 1, EEF pose 7, goal 3 |
| sweep 목표 | 선택 물체를 월드 (+y) 또는 (-y) 방향으로 0.18 m 이동 |
| 명시적 성공 종료 | 없음. 목표 도달 뒤에도 homing reward를 받으며 episode가 계속됨 |
| 알고리즘 | RSL-RL PPO, rollout 36 step/env, 최대 30,000 iteration |

## 기준 좌표계와 표기

아래첨자 `w`, `b`, `e`는 각각 world, robot-root(base), EEF frame을 뜻한다. Isaac Lab quaternion 순서는 `(w, x, y, z)`이다.

- World frame (W): 시뮬레이터의 전역 좌표계. 물체 spawn, sweep 방향, reward와 termination의 위치·속도는 대부분 이 좌표계다. 각 복제 환경은 `env_origin`만큼 평행 이동되어 있다.
- Robot-root frame (B): UR5e articulation root 좌표계. observation의 `ee_pose`만 이 좌표계로 변환된다.
- EEF frame (E): `robotiq_base_link`에 로컬 offset `(0.13, 0, 0) m`를 적용한 제어점. target과 goal observation은 이 움직이는 좌표계로 표현된다. orientation은 `robotiq_base_link`의 orientation을 그대로 사용한다.
- Shelf frame: shelf root는 환경 로컬 기준 `(-0.7, 0, 0)`에 놓인다. 정렬 reward는 shelf의 기본 orientation에서 얻은 (z)축과 EEF의 (y)축을 비교한다.

점 (p_w)를 EEF 좌표로 바꾸는 계산은 다음과 같다.

\[
p_e = R_{we}^{T}(p_w-p_{e,w}), \qquad
p_{e,w}=p_{body,w}+R_{we}[0.13,0,0]^T
\]

코드의 `subtract_frame_transforms(eef_pose_w, point_w)`가 이 변환을 수행한다. 복제 환경의 `env_origin`이 월드 위치에 포함되지만, 같은 환경 안에서 상대좌표나 위치 차를 계산하면 상쇄된다.

## Reset과 과제 생성

매 reset마다 6개 물체 중 target 하나를 균등 표본 추출한다. target의 환경 로컬 위치는

\[
x\sim U(-0.78,-0.57),\quad y\sim U(-0.22,0.22),\quad z=1.05\ \mathrm{m}
\]

이고 yaw도 무작위다. 나머지 5개 물체는 서로 분리된 고정 (x,y) slot의 `z=1.80 m` 대기 선반에 놓인다. 물체별 width는 코드 순서대로 `(0.06, 0.06, 0.05, 0.09, 0.09, 0.06) m`이다.

Sweep 방향 (s\in\{-1,+1\})은 목표가 spawn 범위 안에 남도록 선택한다. 양쪽 모두 가능한 경우에만 50:50으로 고르고, 한쪽만 가능하면 가능한 쪽을 강제한다.

\[
\mathbf d_w=[0,\;0.18s,\;0],\qquad
p_{goal,w}=p_{target,w}^{reset}+\mathbf d_w
\]

정책 시작 전 EEF는 target의 진행 반대편·위쪽으로 IK 배치된다.

\[
p_{reach,w}=p_{target,w}^{reset}+[-0.02,\;-s\,width,\;+0.09]+\epsilon,
\quad \epsilon_i\sim U(-0.01,0.01)\ \mathrm{m}
\]

EEF (y)축이 shelf (z)축과 같은 방향이 되도록 orientation을 정하고, DLS IK를 최대 40회 수행한다. 위치 오차 `<0.003 m`, axis-angle orientation 오차 norm `<0.001 rad`가 성공 기준이다. 실패하면 동일한 object/goal/noise를 유지한 채 세 joint seed를 순서대로 시험하며, 끝까지 실패하면 episode를 시작하지 않고 `RuntimeError`를 낸다.

## Observation

모든 term은 아래 순서로 concatenate된다. 별도 scale이나 normalization은 없고 actor/critic observation normalization도 꺼져 있다.

| slice | term | 차원 | 계산 및 좌표계 |
|---:|---|---:|---|
| `0:8` | `joint_pos` | 8 | 첫 8 joint의 (q-q_{default}), rad. 즉 arm 6개와 articulation 순서상 뒤따르는 gripper joint 2개 |
| `8:14` | `joint_vel` | 6 | arm 6 joint의 \(\dot q-\dot q_{default}\), rad/s. 기본 속도는 0 |
| `14:21` | `actions` | 7 | Action manager가 보관한 직전 policy action |
| `21:24` | `target_obs_state` | 3 | (R_{we}^T(p_{target,w}-p_{eef,w})), EEF frame, 각 성분에 (U(-0.01,0.01)) m noise |
| `24:25` | `target_obj_width` | 1 | 선택 물체 width에 (U(-0.01,0.01)) m noise |
| `25:32` | `ee_pose` | 7 | EEF의 robot-root 상대 position 3 + unique quaternion 4 |
| `32:35` | `goal_pos` | 3 | (R_{we}^T(p_{goal,w}-p_{eef,w})), EEF frame, noise 없음 |

`enable_corruption=True`이므로 위 두 noise가 학습 중 적용된다. target/goal 자체가 아니라 관측값에만 noise가 더해진다. `joint_pos`가 “arm 6개”가 아니라 단순히 articulation의 **첫 8개 joint**를 자른다는 점은 asset joint ordering 변경 시 특히 주의해야 한다.

## Action

Policy 출력은 다음 7개 성분이다.

\[
a=[\Delta x,\Delta y,\Delta z,\Delta r_x,\Delta r_y,\Delta r_z,g]
\]

앞 6개는 relative-pose differential IK command다. 현재 `scale=0.5`가 scalar이므로 translation과 axis-angle rotation 성분 모두에 동일하게 적용된다.

\[
\Delta p_e=0.5[a_0,a_1,a_2]\ \mathrm{m},\qquad
\Delta\theta_e=0.5[a_3,a_4,a_5]\ \mathrm{rad}
\]

이 상대 pose를 `robotiq_base_link + (0.13,0,0) m` 제어점에 적용하고, damped least-squares IK로 UR5e 6개 joint target을 구한다. 매 control step(0.02 s)마다 새 상대 명령을 낸다.

마지막 `g`는 binary gripper command이며, Isaac Lab `BinaryJointPositionAction`의 부호 규칙은 `g < 0`이면 close, `g >= 0`이면 open이다. 따라서 정확히 0도 open이다. Open target은 8개 gripper joint 모두 0 rad이고 close target은 설정된 joint별 `0.5/0/-0.5 rad` 값이다.

> 주의: 코드 주석에는 “normalized policy action”이라고 적혀 있지만 이 runner config에는 `clip_actions=1.0`이 명시되어 있지 않다. 따라서 현재 파일만으로 `[-1,1]` clipping을 보장한다고 해석하면 안 된다. 또한 이전의 작은 scale `(0.02 m, 0.10 rad)`은 주석 처리되어 있고 실제 값은 전 성분 `0.5`다.

## Reward

한 control step의 최종 reward는 아래 가중합이다.

\[
r=-0.03r_{action\_rate}-0.03r_{joint\_vel}-0.5r_{shelf}
-0.5r_{other\_objects}+2r_{align}+6r_{push}+9r_{home}
\]

`reaching` term은 계산되지만 weight가 0이므로 학습 reward에는 기여하지 않는다.

### Regularization과 collision

- `action_rate`: Isaac Lab 기본 term으로 
  \(r_{action\_rate}=\|a_t-a_{t-1}\|_2^2\).
- `joint_vel`: arm 6축에 대해 \(r_{joint\_vel}=\sum_{j=1}^{6}\dot q_j^2\).
- `object_collision`: 모든 물체의 world linear velocity를 0.01 m/s 단위로 반올림하고 target velocity를 0으로 만든 뒤, 전체 environment batch까지 포함한 `sum(abs(v))`에 `tanh`를 적용한다. 현재 구현은 dimension을 지정하지 않은 `torch.sum`이므로 **환경별 벡터가 아니라 batch 전체가 공유하는 scalar penalty**가 될 수 있다. 의도는 non-target motion의 환경별 penalty지만 실제 코드는 병렬 환경 간 결합 가능성이 있다.
- `shelf_collision`: shelf가 초기점에서 이동한 거리와 world velocity norm의 합이 `>0.005`이면 1. 여기에 EEF가 shelf 기준점 `(shelf root + [0,0,1.06])`의 0.2 m 안일 때 finger/wrist 높이 proximity를 최대 3까지 더한다.

Shelf proximity의 각 항은 다음과 같다.

\[
c_L=clip(1-(z_L-z_{ref})/0.02,0,1),\quad
c_R=clip(1-(z_R-z_{ref})/0.02,0,1),\quad
c_W=clip(1-(z_W-z_{ref})/0.08,0,1)
\]

### 정렬, pushing, homing

정렬은 world에서 EEF (y)축과 shelf 기본 (z)축의 내적 \(c\)를 구해

\[
r_{align}=sign(c)c^2
\]

로 계산한다. 같은 방향이면 최대 +1, 반대면 최소 -1이다.

Pushing에서 접촉 유도점은 reset pose와 동일한 형태로 매 step 현재 target 위치에서 계산한다.

\[
p_{contact}=p_{target,w}+[-0.02,-s\,width,+0.09]
\]

`EEF-contact <0.04 m`이면서 `|contact_y-wrist_y|<0.04 m`이면 \(\zeta_m=1\), 아니면 0이다. 목표 거리 \(d=\|p_{goal,w}-p_{target,w}\|_2\)와 target의 world (y) 속도 \(v_y\)에 대해

\[
r_{speed}=\begin{cases}
+0.5 & 0.05<|v_y|<0.1\\
-0.5 & |v_y|\ge0.1\\
0 & |v_y|\le0.05
\end{cases}
\]

\[
r_{push}=\begin{cases}
2e^{-5d} & d<0.03\\
\zeta_m(1-d/0.18+r_{speed}) & d\ge0.03
\end{cases}
\]

이다. 속도의 **방향은 검사하지 않고 절댓값만 검사**한다. 목표 반대 방향으로 움직여도 속도 구간만 맞으면 같은 shaping을 받을 수 있다.

Homing은 (y,z)만 사용한 목표 거리

\[
d_{yz}=\|(p_{goal,w}-p_{target,w})_{y,z}\|_2
\]

와 첫 5개 arm joint의 기본 자세 오차 \(e_q=\sum_{j=1}^{5}|q_j-q_{default,j}|\)로

\[
r_{home}=e^{-0.5e_q}\cdot\frac{1-\tanh(100(d_{yz}-0.03))}{2}
\]

를 계산한다. 즉 target이 목표의 3 cm 안에 들어오면 gate가 켜지고, 이후 default pose로 돌아갈수록 보상이 커진다. wrist 3번 joint(6번째 arm joint)는 homing 오차에서 제외된다.

참고로 weight 0인 reaching term은 \(r_{reach}=e^{-10\|p_{contact}-p_{eef}\|}\)이다.

## Curriculum

등록된 curriculum은 하나뿐이다.

| 시점 | `object_collision` weight |
|---|---:|
| 초기 | `-0.5` |
| global environment step 250,000 이후 설정값 | `-0.5` |

`modify_reward_weight`는 `env.common_step_counter > 250_000`일 때 reward term의 weight를 바꾸는 방식이며 점진적 ramp가 아니다. 즉 정확히 250,000이 아니라 그 다음 common step부터 적용된다. 현재 초기값과 목표값이 같으므로 **실효적인 curriculum이 없다**. Curriculum을 의도했다면 초기 weight 또는 250,000-step 목표 weight 중 하나가 다른 값이어야 한다.

PPO의 rollout은 iteration당 `36 × 4096 = 147,456` transition이다. 다만 curriculum의 `num_steps`는 PPO transition 총수나 iteration 수가 아니라 Isaac Lab environment의 global step counter 기준으로 해석해야 한다.

## Termination과 truncation

종료 조건은 환경별 boolean OR로 결합된다.

| term | 조건 | 분류 |
|---|---|---|
| `time_out` | 10 s, 즉 500 control step 도달 | truncation |
| `object_drop` | 6개 중 하나라도 world `z < 1.04 m`, 또는 `|roll| > 0.9 rad` 또는 `|pitch| > 0.9 rad` | failure termination |
| `push_fast` | target의 world linear speed norm `>0.3 m/s` | failure termination |
| `shelf_collision` | shelf world velocity norm `>0.1`, 또는 finger 높이가 `shelf root z + 1.05`보다 0.01 m 미만 위, 또는 wrist가 0.07 m 미만 위 | failure termination |
| `hand_velocity` | arm 6 joint 중 하나라도 `|qdot| >1.0 rad/s` | failure termination |

Roll/pitch는 object world quaternion에서 XYZ Euler angle로 바꾼 뒤 `[-pi, pi)`로 정규화한다. `object_drop`은 target만 보는 것이 아니라 위 대기 선반의 5개 물체까지 모두 검사한다.

중요하게도 `d<0.03 m` 같은 **성공 termination은 없다**. 성공하면 pushing reward의 가까운-목표 branch와 homing reward를 받으며 남은 시간 동안 계속 제어한다. 따라서 학습 목표는 “빨리 성공하고 종료”가 아니라 “목표에 물체를 유지하면서 기본 joint pose로 복귀하고, 실패 종료를 피하며 episode를 지속”하는 형태다.

## PPO 설정

- Actor/Critic MLP: `[256, 128, 64]`, ELU, observation normalization 없음
- 초기 action noise std: 1.0
- rollout: 36 step/env, minibatch 4개, epoch 8회
- learning rate `1e-3`, adaptive schedule, desired KL `0.02`
- `gamma=0.95`, `lambda=0.95`, clip `0.2`, entropy coefficient `0.01`
- gradient norm 최대 1.0, checkpoint 매 50 iteration, 최대 30,000 iteration

## 구현을 읽을 때의 핵심 점검 사항

1. 실제 action scale은 translation/rotation 모두 `0.5`이며 action clipping은 이 config에 명시되지 않았다.
2. Observation은 35차원이고 target/goal은 world가 아니라 현재 EEF frame이다.
3. Reward와 termination의 물체 위치·속도는 대부분 world frame이다.
4. 성공 종료가 없고, 성공 뒤 homing이 과제의 일부다.
5. Curriculum은 현재 초기/목표 weight가 같아 동작상 변화가 없다.
6. `object_collision`의 무차원 `torch.sum`은 환경별 reward가 아닌 batch 결합 scalar가 될 가능성이 있어 수정 전 확인이 필요하다.

## 근거 코드

- 환경 구성, 항목 weight와 threshold: `src/sweeping_policy/sweeping_policy/shelf_sweep/env_cfg.py`
- observation 좌표 변환: `src/sweeping_policy/sweeping_policy/shelf_sweep/mdp/observations.py`
- reward 계산: `src/sweeping_policy/sweeping_policy/shelf_sweep/mdp/rewards.py`
- termination 계산: `src/sweeping_policy/sweeping_policy/shelf_sweep/mdp/terminations.py`
- reset, target/goal 생성과 IK: `src/sweeping_policy/sweeping_policy/shelf_sweep/mdp/events.py`
- PPO hyperparameter: `src/sweeping_policy/sweeping_policy/shelf_sweep/rsl_rl_ppo_cfg.py`
