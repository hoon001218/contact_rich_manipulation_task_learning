# Hand RL 환경 명세

## 목적

`Isaac-Hand-RH56-Grasp-v0`는 UR5e 끝단에 Inspire RH56 오른손을 고정하고,
primitive shelf 위에 놓인 cube를 집어서 shelf 표면으로부터 12 cm 이상 들어
올리는 manager-based 강화학습 환경이다.

## Scene과 asset

| 구성요소 | 구현 | 주요 값 |
|---|---|---|
| Robot arm | `sweep_jh`와 동일한 UR5e USD | 기본 Nucleus 경로: Isaac 5.0 UR5e |
| Hand | `assets/robots/inspire_rh56/inspire_rh56_right.urdf` | 실행 시 URDF→USD 변환 후 UR5e tool frame에 fixed joint로 조립 |
| Shelf | collision cuboid 4개 | board, back, left, right panel |
| Object | dynamic cuboid | 5 cm, 80 g |
| Ground/light | plane, dome light | 환경 공통 |

UR5e와 RH56 경로는 각각 `HAND_RL_UR5E_USD_PATH`,
`HAND_RL_RH56_URDF_PATH` 환경변수로 덮어쓸 수 있다. RH56 장착 pose는
`Ur5eRh56SpawnerCfg.mount_translation`과 `mount_rotation_deg`에서 조정한다.
초기 장착 회전은 sweep 예제의 tool mounting convention에 맞춘 `(0, 90, 0)` deg이다.

Shelf board 상면은 각 environment origin 기준 `z=0.495 m`이다. Reset마다
object의 `x`, `y`, `yaw`가 각각 ±7 cm, ±18 cm, ±π 범위에서 무작위화된다.

## Action

Policy action은 총 12차원이며 최종적으로 `[-1, 1]`로 clip된다.

| 순서 | Term | 차원 | Joint | Mapping |
|---:|---|---:|---|---|
| 0–5 | `arm` | 6 | UR5e 6축 | default joint pose + `0.35 × action` rad |
| 6–11 | `hand` | 6 | thumb yaw, thumb pitch, index, middle, ring, pinky proximal | `[-1,1]`을 각 URDF joint limit으로 선형 변환 |

RH56의 intermediate/distal mimic joint는 URDF mimic 관계를 유지한다. 따라서
policy는 여섯 개의 독립 actuator만 명령한다.

## Observation

Actor와 critic은 동일한 59차원 `policy` observation을 사용한다. Quaternion은
`(w, x, y, z)` 순서이며 TCP/object pose와 velocity는 UR5e root frame 기준이다.

| Term | 차원 | Scale | 설명 |
|---|---:|---:|---|
| `arm_joint_pos` | 6 | 1.0 | UR5e default pose 대비 joint position |
| `arm_joint_vel` | 6 | 0.1 | UR5e joint velocity |
| `hand_joint_pos` | 6 | 1.0 | RH56 독립 joint position을 joint limit으로 정규화 |
| `hand_joint_vel` | 6 | 0.2 | RH56 독립 joint velocity |
| `tcp_pose` | 7 | 1.0 | robot-root frame의 TCP position + quaternion |
| `object_pose` | 7 | 1.0 | robot-root frame의 object position + quaternion |
| `object_to_tcp` | 3 | 1.0 | TCP에서 object로 향하는 vector |
| `object_velocity` | 6 | 0.2 | robot-root 축으로 표현한 object linear/angular velocity |
| `last_action` | 12 | 1.0 | 직전 policy action |
| **합계** | **59** | | |

Observation corruption은 첫 baseline에서는 비활성화했다.

## Reward

아래 term의 가중합이 매 control step reward다. `tanh` proximity term은
거리가 0에 가까울수록 1에 접근한다.

| Term | Weight | 정의/의도 |
|---|---:|---|
| `reach_object` | +2.0 | `1 - tanh(TCP-object distance / 0.12)` |
| `enclose_object` | +3.0 | 다섯 fingertip-object 거리 평균에 `std=0.10`의 tanh kernel 적용 |
| `lift_progress` | +12.0 | object 바닥과 shelf 사이 clearance를 0–12 cm 구간에서 0–1로 정규화 |
| `held_grasp` | +6.0 | 3 cm 이상 들었고 TCP-object 거리가 18 cm 미만이면 1 |
| `success` | +25.0 | clearance 12 cm, TCP 거리 18 cm 미만, object 근처 fingertip 2개 이상이면 terminal bonus |
| `action_rate` | −0.01 | 연속 action 차이의 squared L2 penalty |
| `joint_velocity` | −1e-4 | 전체 articulation joint velocity squared L2 penalty |

초기 curriculum 없이 reach → enclosure → lift의 dense shaping을 동시에 사용한다.
실제 학습 curve에서 arm reach만 최적화하거나 손을 일찍 닫는 local optimum이
관찰되면 각 stage gate 또는 curriculum을 추가해야 한다.

## Reset과 termination

Episode 길이는 8초다. Simulation은 120 Hz, action/control은 decimation 4에
따라 30 Hz로 실행된다.

| 조건 | 종류 | 설명 |
|---|---|---|
| `time_out` | truncation | 8초 경과 |
| `success` | termination | 12 cm clearance를 달성하고 TCP/object가 가깝고 fingertip 2개 이상이 object를 둘러싼 상태 |
| `object_dropped` | termination | environment-relative object 높이가 0.25 m 미만 |

Robot joint는 default pose 주변 ±0.04 rad로 reset되며 object linear/angular
velocity는 0으로 초기화된다.

## PPO 기본값

| 항목 | 값 |
|---|---:|
| Environments | 1024 |
| Rollout steps/env | 32 |
| Max iterations | 10,000 |
| Actor/Critic hidden dims | 512, 256, 128 |
| Activation | ELU |
| Learning rate | 5e-4 adaptive |
| PPO epochs / mini-batches | 8 / 4 |
| Gamma / lambda | 0.99 / 0.95 |
| Clip parameter | 0.2 |
| Entropy coefficient | 0.006 |
| Checkpoint interval | 100 iterations |
| Log directory | `logs/rsl_rl/hand_rh56_grasp/` |

## 실행 전 asset 조건

RH56 URDF가 참조하는 GLB/OBJ가 실제 binary여야 한다. Git LFS pointer만 있는
상태에서는 URDF converter가 정상 mesh를 생성할 수 없다. 또한 기본 UR5e 경로를
사용하려면 `192.168.0.13` Nucleus 인증이 유효해야 한다.
환경은 importer 실행 전에 모든 URDF mesh를 검사하며, 누락 파일이나 Git LFS
pointer를 발견하면 복구 방법을 포함한 명시적인 오류로 중단한다.
