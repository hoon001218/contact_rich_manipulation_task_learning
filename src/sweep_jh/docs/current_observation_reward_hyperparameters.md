# Sweep JH 현재 학습 설정

이 문서는 `Isaac-Sweep-JH-v0`의 현재 policy observation, reward, 학습
hyperparameter를 정리한다. 환경, manager term, MDP 함수 및 PPO 설정은
`sweep_jh` 내부에서 직접 정의한다. Synthetic contact pad body와 filtered
contact sensor 없이 Robotiq USD의 원래 collision geometry를 사용한다.

## Observation

Policy observation은 아래 순서로 concatenate되는 **41차원 task-space 벡터**다.

| 순서 | Term | 차원 | 내용 및 좌표계 | Noise |
|---:|---|---:|---|---|
| 1 | `eef_pose` | 6 | robot base 기준 EEF `xyz + RPY` `[m, rad]` | 없음 |
| 2 | `eef_twist` | 6 | robot base 기준 EEF `[vx, vy, vz, wx, wy, wz]` | 없음 |
| 3 | `ft_sensor` | 6 | robot base 축 기준 `[Fx, Fy, Fz, Tx, Ty, Tz]` `[N, N·m]` | 없음 |
| 4 | `initial_target_pose` | 6 | reset 시 target pose, robot base 기준 `xyz + RPY` | 없음 |
| 5 | `desired_motion` | 5 | 방향 2 + 거리 + 목표 힘 + 힘 tolerance | 없음 |
| 6 | `last_action` | 12 | 직전 stiffness 6 + relative pose 6 action | 없음 |
| | **합계** | **41** | | |

별도의 contact point나 target-filtered contact force는 관측하지 않는다.
F/T force와 torque는 virtual sensor 원점에서 측정하며, 벡터의 표현 축만
sensor frame에서 robot base frame으로 회전한다.

정책에는 joint position, joint velocity, joint effort 및 현재 target pose가
제공되지 않는다. Joint velocity 대신 EEF의 선속도와 각속도를 task space에서
제공한다. Reward와 termination은 학습 중 simulator의 `target_object` 상태를
계속 사용할 수 있다. Actor와 critic 모두 같은 41-D `policy` observation
group을 사용하므로 critic에도 별도의 joint-space privileged state는 없다.

### Desired motion command

```text
[direction_x, direction_y, distance_m, desired_force_N, force_tolerance_N]
```

| Parameter | Sampling range |
|---|---:|
| XY direction angle | `[-π, π]` |
| Sweep distance | `[0.10, 0.22] m` |
| Desired wrist F/T force norm | `[8, 25] N` |
| Wrist-force tolerance | `[3, 6] N` |
| Command resampling time | 사실상 episode당 고정 (`1e9 s`) |

## Reward

Isaac Lab Reward Manager는 매 policy step에 각 항을 다음과 같이 합산한다.

```text
total_reward += raw_term_value × weight × step_dt
```

현재 policy 주기는 `30 Hz`이므로 `step_dt = 1/30 s`다.

| Reward term | Weight | 주요 parameter | 의미 |
|---|---:|---|---|
| `push_pose` | `-1.0` | stand-off `0.065 m`, scale `0.10 m` | 접촉 전, 움직이는 Cartesian push pose 오차 |
| `push_axis_alignment` | `-1.0` | proximity `0.08~0.18 m` | gripper push axis와 목표 방향 불일치 |
| `normal_force_tracking` | `+2.0` | force gate `1~4 N` | base-frame 목표 축 방향 force 추종 |
| `tangential_force` | `-0.75` | force scale `25 N` | sweep 축에 수직인 wrist force |
| `delta_progress` | `+6.0` | normalized rate scale `1.0` | 새로 발생한 종방향 이동량/시간 |
| `endpoint_error` | `-2.0` | command distance로 정규화 | 목표점 거리/명령 거리의 연속 비용 |
| `lateral_error` | `-2.0` | command distance로 정규화 | 목표 방향에 수직인 변위 |
| `overshoot` | `-4.0` | command distance로 정규화 | 목표 이동 거리를 넘은 종방향 변위 |
| `goal_speed` | `-1.0` | goal scale `0.06 m`, speed scale `0.05 m/s` | 목표점 근처 물체 속도 |
| `stopped_at_goal` | `+5.0` | position `0.025 m`, speed `0.02 m/s` | 목표 위치와 정지를 동시에 만족하는 연속 보상 |
| `pose_action_rate` | `-0.5` | action `6:12` | Cartesian relative-pose action 변화량 |
| `stiffness_action_rate` | `-0.25` | action `0:6` | task-space stiffness action 변화량 |
| `force_excess` | `-0.5` | soft/hard limit `75/100 N` | hard termination 전 force soft barrier |
| `ft_torque` | `-0.5` | deadband/hard limit `1.5/15 N·m` | base-frame F/T torque soft barrier |
| `torque_saturation` | `-0.5` | `arm_action` | action 이상 또는 torque saturation |
| `success` | `+600.0` | success termination | `step_dt` 적용 후 실제 terminal bonus `+20` |
| `failure` | `-600.0` | 세 safety termination | `step_dt` 적용 후 실제 terminal penalty `-20` |

### Raw term 정규화

Weight가 물리 단위나 outlier의 크기에 종속되지 않도록 모든 active shaping term의
raw value를 bounded 범위로 제한한다. 여기서 정규화는 관측 정규화와 별개이며,
reward 함수 자체의 출력 범위를 정하는 것을 뜻한다.

| Raw term | 정규화 | 출력 범위 |
|---|---|---:|
| `push_pose` | `(1-contact_gate) * tanh(position_error / 0.10)` | `[0, 1)` |
| `push_axis_alignment` | `proximity_gate * (1-cos_alignment)/2` | `[0, 1]` |
| `normal_force_tracking` | `contact_gate * exp(-(force_error/tolerance)^2)` | `[0, 1]` |
| `tangential_force` | `proximity_gate * tanh(F_tangential / 25)` | `[0, 1)` |
| `delta_progress` | `tanh(normalized_progress_rate / 1.0)` | `(-1, 1)` |
| `endpoint_error` | `tanh(endpoint_error / command_distance)` | `[0, 1)` |
| `lateral_error` | `tanh(lateral_error / command_distance)` | `[0, 1)` |
| `overshoot` | `tanh(overshoot / command_distance)` | `[0, 1)` |
| `goal_speed` | `goal_gate * tanh(object_speed / 0.05)` | `[0, 1)` |
| `stopped_at_goal` | position Gaussian × speed Gaussian | `(0, 1]` |
| 두 action-rate 항 | action delta를 `[-2,2]`로 제한 후 `mean(delta^2)/4` | `[0, 1]` |
| `force_excess`, `ft_torque` | soft/deadband부터 hard limit까지 선형 정규화 후 clamp | `[0, 1]` |
| `torque_saturation`, terminal 항 | binary indicator | `{0, 1}` |

`tanh`는 작은 오차 부근에서는 거의 선형이어서 gradient를 유지하고, 큰 오차나
순간적인 force spike가 전체 reward를 지배하지 않도록 포화시킨다. 반면 목표
도달처럼 중심값 주변에서 급격히 보상을 집중할 항은 Gaussian을 사용한다.

### Gate와 주요 수식

Contact sensor가 없으므로 EEF-object 거리와 wrist force로 smooth gate를 만든다.
Planar sweep과 무관한 gripper 중력의 base-Z 성분이 contact로 인식되지 않도록
gate와 force tracking에는 base XY force만 사용한다.

```text
proximity_gate = smoothstep((0.14 - eef_object_distance) / (0.14 - 0.08))
force_gate     = smoothstep((wrist_force_xy_norm - 1) / (4 - 1))
contact_gate   = proximity_gate * force_gate
```

Force는 robot base frame에서 desired direction으로 분해한다. Incoming joint
wrench의 parent/child 부호가 아직 calibration되지 않았으므로 첫 버전은 축 방향
성분의 절댓값을 사용한다.

```text
F_base_xy          = [Fx_base, Fy_base, 0]
signed_axial_force = dot(F_base_xy, desired_direction_base)
axial_force        = abs(signed_axial_force)
tangential_force   = norm(F_base_xy - signed_axial_force * desired_direction_base)

normal_force_tracking =
  contact_gate
  * exp(-((axial_force - desired_force) / force_tolerance)^2)
```

Progress는 현재 위치를 매 step 반복 보상하지 않고 새로 발생한 이동량만 보상한다.

```text
progress_t = dot(object_position_t - initial_position, desired_direction)

delta_progress =
  tanh(
    ((progress_t - progress_(t-1)) / (desired_distance * step_dt))
    / progress_rate_scale
  )
```

Episode reset 직후 첫 step의 `delta_progress`는 0으로 초기화해 command resampling
때문에 잘못된 progress spike가 생기지 않게 한다.

### 성공과 실패

성공은 다음 조건을 `0.25 s` 연속 유지할 때 발생한다.

```text
endpoint error < 0.025 m
normalized lateral error < 0.12
object speed < 0.02 m/s
```

성공 weight `600`은 `600 * 1/30 = 20`의 일회성 reward가 된다. 물체의 잘못된
pose, F/T limit 초과, arm speed limit 초과에는 weight `-600`, 즉 실제 `-20`의
terminal penalty를 적용한다. Timeout에는 failure penalty를 적용하지 않는다.

Joint velocity 및 commanded joint effort는 reward에서 제거했다. Joint speed는
정책 shaping이 아니라 safety termination으로만 남아 있다.

## Hyperparameters

### Environment와 simulation

| Hyperparameter | 값 |
|---|---:|
| Parallel environments | `2048` |
| Environment spacing | `2.0 m` |
| Episode length | `8.0 s` |
| Physics timestep | `1/120 s` (`120 Hz`) |
| Decimation | `4` |
| Policy timestep | `1/30 s` (`30 Hz`) |
| Replicate physics | `True` |
| Target cube mass | `0.35 kg` |

### Action과 OSC

Policy action은 **12차원**이며 순서는 다음과 같다.

```text
[normalized_stiffness(6), relative_pose(6)]
```

| Hyperparameter | 값 |
|---|---:|
| Task-space stiffness range | `[20, 300]` |
| Default translational stiffness | `[120, 120, 120]` |
| Default rotational stiffness | `[35, 35, 35]` |
| Damping ratio | 모든 축 `1.0` |
| Position action scale | `0.025 m/step` |
| Orientation action scale | `0.12 rad/step` |
| Effort-limit scale | `0.9` |
| Gravity compensation | `True` |
| Inertial dynamics decoupling | `True` |

### RSL-RL PPO runner

| Hyperparameter | 값 |
|---|---:|
| Seed | `42` |
| Device | `cuda:0` |
| Steps per environment | `32` |
| Maximum iterations | `12,000` |
| Save interval | `100` iterations |
| Action clipping | `1.0` |
| Logger | `tensorboard` |
| Actor observation group | `policy` |
| Critic observation group | `policy` |
| Experiment name | `sweep_jh` |

### Actor와 critic network

| Hyperparameter | Actor | Critic |
|---|---:|---:|
| Hidden dimensions | `[512, 256, 128]` | `[512, 256, 128]` |
| Activation | `ELU` | `ELU` |
| Observation normalization | `True` | `True` |
| Initial Gaussian action std | `0.8` | 해당 없음 |

### PPO algorithm

| Hyperparameter | 값 |
|---|---:|
| Learning rate | `5e-4` |
| Learning-rate schedule | `adaptive` |
| Desired KL | `0.01` |
| Discount factor (`gamma`) | `0.99` |
| GAE (`lambda`) | `0.95` |
| Clip parameter | `0.2` |
| Value-loss coefficient | `1.0` |
| Clipped value loss | `True` |
| Entropy coefficient | `0.004` |
| Learning epochs | `8` |
| Mini-batches | `4` |
| Maximum gradient norm | `1.0` |

## TensorBoard metrics

RSL-RL logger는 `tensorboard`로 명시한다. 저장소 루트에서 다음 명령으로 현재와
이전 학습 run을 함께 확인할 수 있다.

```bash
tensorboard --logdir logs/rsl_rl/sweep_jh
```

| TensorBoard tag | 의미 |
|---|---|
| `Train/mean_reward` | 최근 완료 episode들의 total reward score 평균 |
| `Train/mean_episode_length` | 평균 episode 길이 |
| `Episode_Termination/success` | success termination 비율 |
| `Episode_Termination/time_out` | timeout 비율 |
| `Metrics/desired_motion/endpoint_error_m` | episode 종료 시 목표점 거리 오차 `[m]` |
| `Metrics/desired_motion/lateral_error_m` | 종료 시 sweep 축 수직 오차 `[m]` |
| `Metrics/desired_motion/normalized_lateral_error` | lateral error / command distance |
| `Metrics/desired_motion/progress_ratio` | 종방향 이동량 / command distance; 목표는 `1.0` |
| `Metrics/desired_motion/object_speed_mps` | 종료 시 물체 속력 `[m/s]` |
| `Episode_Reward/<term>` | 각 weighted reward term의 episode 기여도 |

`Train/mean_reward`는 한 step의 reward가 아니라 episode 동안 누적된 total reward의
이동 평균이다. `Episode_Reward/<term>`은 Isaac Lab Reward Manager가 비교하기
쉽도록 episode 누적값을 최대 episode 시간으로 나눈 값이므로, 두 값의 단순 합이
`Train/mean_reward`와 같지는 않다.

## Source of truth

- 환경과 Observation/Reward term 연결: `src/sweep_jh/sweep_jh/osc_sweep/env_cfg.py`
- Robot/virtual F/T asset assembly: `src/sweep_jh/sweep_jh/osc_sweep/assets.py`
- Action 계산: `src/sweep_jh/sweep_jh/osc_sweep/mdp/actions.py`
- Command 계산: `src/sweep_jh/sweep_jh/osc_sweep/mdp/commands.py`
- Observation 계산: `src/sweep_jh/sweep_jh/osc_sweep/mdp/observations.py`
- Cartesian reward 계산: `src/sweep_jh/sweep_jh/osc_sweep/mdp/rewards.py`
- Reset/termination: `src/sweep_jh/sweep_jh/osc_sweep/mdp/events.py`,
  `src/sweep_jh/sweep_jh/osc_sweep/mdp/terminations.py`
- PPO 설정: `src/sweep_jh/sweep_jh/osc_sweep/rsl_rl_ppo_cfg.py`

이 문서는 2026-07-21 현재 코드 기준이다. 이후 JH 환경을 변경할 때 이 문서도
함께 갱신해야 한다.
