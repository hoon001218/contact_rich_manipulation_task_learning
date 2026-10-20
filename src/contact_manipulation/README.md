# Contact Manipulation

Sweeping-Policy의 **`min` branch 구조**를 기준으로 구성한 Isaac Lab 연구 프로젝트다.
**장면·센서 생성, episode reset, MDP, GNN·attention 정책과 native PPO를 연결한 프로젝트다.**
기존 `src/contact_rearrangement`와 별도 패키지이고, 해당 패키지를 import하지 않는다.

본 학습의 설정·결과·종료 원인과 미해결 수정 과제는
[학습 기록](../../docs/TRAINING_LOG.md)에 정리한다. 연구·개발 전체 이력은
[CONTEXT.md](../../docs/CONTEXT.md)를 참고한다.

주요 구조 기준은 [Sweeping-Policy/min](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/tree/min)이다.
로컬 [Sweeping-Policy 예제](../../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/)와
[Isaac Lab manager-based 템플릿](../../IsaacLab/tools/template/templates/tasks/manager-based_single-agent/env_cfg)도 함께 참고한다.

## 디렉터리 구조

```text
contact_manipulation/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── train.py                     # Isaac Lab 예제의 native 학습 흐름
│   └── play.py                      # 저장된 checkpoint의 native 추론·GUI 재생
└── contact_manipulation/
    ├── __init__.py                  # 패키지 import 시 Gym 등록
    ├── shelf_manipulation_env_cfg.py # 공통 world·MDP term·시뮬레이션 설정
    ├── config/
    │   ├── __init__.py
    │   └── ur5e/
    │       ├── __init__.py          # 환경 ID, 환경/PPO 설정 진입점
    │       ├── ur5e_env_cfg.py      # 공통 설정을 상속하고 UR5e 설정 구체화
    │       ├── scene.py            # 정적 asset 구성·catalog·USD 기하·로봇 FK/IK
    │       └── agents/
    │           ├── __init__.py
    │           ├── rsl_rl_ppo_cfg.py # native runner 및 연구 PPO 설정
    │           ├── model.py         # 모델 설정·mask·GNN·attention·history·heads
    │           └── ppo.py           # native actor/critic 연결·PPO microbatch
    └── mdp/
        ├── __init__.py
        ├── events.py               # reset, 배치 상태, 공통 기하 계산
        ├── sensors.py              # 실제 F/T·contact reader와 기하 기반 초기 vision
        ├── observations.py         # observation 계산
        ├── rewards.py              # reward 계산
        └── terminations.py         # 성공·실패 조건
```

## 예제와의 대응

| Sweeping-Policy/min | 현재 프로젝트 | 역할 |
|---|---|---|
| `shelf_sweep_random_env_cfg.py` | `shelf_manipulation_env_cfg.py` | 공통 world, 각 manager의 term, physics/policy 주기·episode 길이 |
| `config/ur5e/osc_random_sweep_env_cfg.py` 및 hand 변형 cfg | `config/ur5e/ur5e_env_cfg.py` | 공통 cfg 상속, robot·shelf·object·sensor·공식 action 구성 |
| `config/ur5e/__init__.py` | 동일 위치 | Gym 등록과 설정 파일 연결 |
| `config/ur5e/agents/rsl_rl_ppo_cfg.py` | 동일 위치 | 정책 및 PPO 설정 |
| `mdp/*.py` | 동일 위치 | 연구에 필요한 MDP 함수 |
| `src/ur5e*.py`, `src/shelf_utils.py` | `config/ur5e/scene.py` | Asset 생성과 기하 준비; 기존 파일에 모아 관리 |
| `train.py`, `play.py` → Isaac Lab 실행 파일 | `scripts/train.py`, `scripts/play.py` | AppLauncher·Hydra·Gym·native wrapper/runner 흐름 |

로봇별 설정은 `ur5e_env_cfg.py`에서 구체화한다. Arm에는 연구 설계에 따라
가변 stiffness의 OSC torque 제어를, hand에는 독립 6관절 position 제어를 사용한다.
에셋 로딩은 예제의 `src/ur5e.py`, `src/shelf_utils.py`를 참고해 `scene.py`에 모았다.
공통 cfg의 observation은 `initial_scene` 100D, `current` 72D, `previous` 72D 순서로
각각 `ObservationTermCfg`에 선언한다. Native ObservationManager가 총 244D로 연결한다.
세 함수는 같은 step의 캐시를 조회하므로 센서 읽기·이력 갱신·기하 계산을 반복하지 않는다.

새 기능도 **cfg에서 선언 → native manager가 호출 → 해당 `mdp` 함수에서 연구 계산**하는
흐름을 따른다. `min`의 환경별 파생 cfg와 asset 설정 방식은 참고하되, 이 과제에 없는
command/curriculum이나 별도의 environment subclass는 추가하지 않는다.

## Native 기능과 연구 확장의 경계

새 기능은 설치된 Isaac Lab의 public class/config와 공식 예제를 먼저 확인한다.
직접 구현은 연구 규칙 또는 측정된 병목 때문에 필요한 부분에 한정한다.
새 파일을 늘리지 않고 기존 `mdp/`, `config/ur5e/`, `agents/`에서 확장한다.

| 영역 | 현재 사용하는 native 기능 | 남겨 둔 연구 확장 |
|---|---|---|
| 환경·학습 흐름 | `ManagerBasedRLEnv`, 각 Manager, `InteractiveScene`, `OnPolicyRunner` | Scene/MDP 설정과 함수 |
| 제어 | `OperationalSpaceControllerAction`, `JointPositionToLimitsAction`, 팔 `IdealPDActuator`·Hand `ImplicitActuator` | UR5e env cfg에서 대상 링크·관절 및 18D 입력 설정 |
| 좌표 연산 | `isaaclab.utils.math`의 quaternion 연산 | 배치 차원의 broadcasting adapter |
| EEF·접촉 센서 | `FrameTransformer`, `ContactSensor` | 17개 tactile 링크 통합, 전체 행 갱신 최적화 |
| 물체 주차 | `RigidObjectCollection.permanent_wrench_composer` | parked mask와 중력 상쇄 force 계산; transient force는 instantaneous composer 사용 |
| PPO | 기본 `PPO.update()`, storage·GAE·logger·checkpoint | 공유 GNN/attention actor–critic, terminal 관측 bootstrap, 큰 minibatch의 gradient accumulation |
| 초기 vision | `convert_to_warp_mesh`, Warp mesh query | 후보 pose의 object-only raycast, 두 역할의 무가림 실루엣과 가시성, 관측 표면 OBB |

직접 구현을 유지한 이유:

- 기본 reset의 균등 pose/joint sampling은 Target·Blocker 조건, 밀집도, pivot 재배치 공간,
  elbow/자기충돌 조건을 정의하지 않는다. `EventTermCfg` 안에서 연구 sampler를 사용한다.
- `DifferentialIKController`는 주어진 pose/Jacobian에서 관절 증분을 계산하며 후보의 FK나
  충돌 없는 초기 자세 탐색을 제공하지 않는다. 현재 GPU FK/IK는 물리 scene을 바꾸지 않고
  여러 후보를 함께 풀며 수십 번의 DLS를 fused kernel에서 처리하므로 유지한다.
- `MultiMeshRayCasterCamera`와 `raycast_dynamic_meshes`는 동적 mesh를 지원한다.
  따라서 "native raycast는 동적 물체를 지원하지 않는다"는 이유로 custom을 유지하는 것이 아니다.
  현재 구현은 **미적용 후보 pose**를 평가하고, Target·Blocker의 무가림 분모 실루엣을 동시에
  계산하며, projected bounds로 ray query를 줄이고 depth/label을 ray당 한 thread에서 결정한다.
  Native multi-mesh kernel의 depth/label 경쟁 쓰기 관련 FIXME도 있어 동일 대체로 취급하지 않는다.
- Native observation history는 reset 후 첫 sample을 과거 슬롯에도 반복한다. 우리 규약은
  reset 시 previous=0, current/previous 순서와 reset 직전 terminal 관측 보존이므로
  `TaskState`의 두 sample 저장을 유지한다.
- F/T 원시 반력은 public `ArticulationData.body_incoming_joint_wrench_b`에서 읽는다.
  Quaternion 부호 정규화와 물리 FK 검사의 회전 변환도 `isaaclab.utils.math`를 사용한다.
  SI-200-8 양자화·범위와 hand 링크의
  자중/운동 관성 보상, OBB 조작량·clearance 등은 연구 수식이다.
- 출력 분포는 RSL-RL의 `GaussianDistribution`을 사용한다. Sampling·log-probability·
  analytic entropy·KL은 native 구현에 맡기고 GNN/attention과 공유 actor–critic 연결을 유지한다.

비교한 공식 구현은 [OSC 예제](../../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/osc_env_cfg.py),
[관절 범위 매핑 예제](../../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/inhand_env_cfg.py),
[FrameTransformer 예제](../../IsaacLab/scripts/tutorials/04_sensors/run_frame_transformer.py),
[기본 MDP events](../../IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py),
[WrenchComposer](../../IsaacLab/source/isaaclab/isaaclab/utils/wrench_composer.py),
[MultiMeshRayCasterCamera](../../IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_camera.py),
[observation history](../../IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py),
[native 학습 스크립트](../../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py)다.

## 코드 흐름

```text
scripts/train.py
  → AppLauncher → 패키지 import / Gym 등록
  → Hydra 환경/PPO 설정 로딩
  → gym.make("Isaac-Contact-Manipulation-UR5e-v0", cfg=env_cfg)
  → ManagerBasedRLEnv
      → InteractiveScene: robot / shelf / object_collection / ee_frame
      → Action / Observation / Event / Reward / Termination managers
  → RslRlVecEnvWrapper → OnPolicyRunner.learn()
```

환경의 `step/reset`, runner, rollout storage, GAE, 로그와 checkpoint 파일은 표준 구현을 사용한다.
`PPO`는 공유 네트워크 연결, reset 이전 관측의 timeout bootstrap, microbatch 업데이트만 확장한다.
초기 vision·이력·경과 시간과 timeout을 일치시키기 위해 `init_at_random_ep_len=False`로 시작한다.
`commands`와 `curriculum`은 Isaac Lab 기본값인 `None`으로 두었다.

`min`의 실행 스크립트는 task를 등록하고 `runpy`로 Isaac Lab 스크립트를 호출한다.
우리 스크립트는 같은 실행 순서를 직접 사용하며 다음 차이만 유지한다.

- Train: 위 episode clock 규칙 때문에 stock 학습의 무작위 초기 episode counter를 사용하지 않는다.
- Play: 현재 graph 모델에서 지원하지 않는 JIT/ONNX export를 실행하지 않고 native inference로 재생한다.
- 모델 연결: 공유 GNN·attention, native Gaussian 분포 연결과 checkpoint 계약을 adapter에서 처리한다.

Reset은 `EventManager`가 `reset_scene → reset_task` 순서로 호출한다. 첫 함수가 물체·로봇의
초기 상태와 vision을 준비하고, 두 번째가 초기 snapshot·이력·누적량을 초기화한다.
이후 reward·termination·observation은 한 control step당 한 번 갱신되는 `TaskState`를 공유한다.
이 부분은 initial vision 고정, previous=0 reset 규칙, 누적 조작량과 terminal observation을
일치시키기 위한 연구 구현이다. 별도 `env.step()`이나 `_reset_idx()` override는 없다.

## 모델과 PPO

```text
초기 vision 기하 [N,10,10] ── 2-layer GNN (64D)
                             ├─ Target / Blocker embedding
                             ├─ Target / Blocker 조건부 attention → scene 320D
                             └─ valid object mean pooling
현재·직전 sample [N,2,72] ── 결측 masking → flatten 144D (별도 encoder 없음)
                                   ↓
                      concat 464 → MLP 256 → latent 128
                                   ├─ Gaussian actor: 18D
                                   └─ critic: 1D
```

GNN은 관측된 물체 사이의 위치·크기 차이를 사용해 두 번 message passing한다.
Target/Blocker embedding에서 두 attention query를 만들어 다른 물체의 정보를 읽는다.
Attention은 `SceneGraphEncoder.forward()`의 query/key/value·masked softmax에 구현되어 있다.
GNN 및 최종 MLP의 활성화는 **ReLU**다. `fusion = nn.Sequential(...)`은 concat 이후의
최종 MLP `Linear(464,256) → ReLU → Linear(256,128) → ReLU`를 순서대로 묶은 것이며,
시간 이력을 처리하는 recurrent/sequential encoder를 추가한 것이 아니다.
현재·직전 값은 순서대로 그대로 연결하여 최종 MLP에서 처리한다. 별도 temporal encoder는
확정된 설계가 아니므로 이번 baseline에서 제거했다. 두 head는 scene encoder와 최종 MLP를
공유하며 critic에도 동일한 observation만 제공한다. 기본 모델은 **234,789개 파라미터**다.
이전 temporal encoder 모델의 checkpoint는 구조가 달라 resume할 수 없다.
Activation도 checkpoint metadata에 기록하므로 이전 SiLU checkpoint의 그대로 resume는 거절한다.

Actor는 학습 시 `a=μ+σε`, 일반 재생 시 `a=μ`를 사용한다. Tanh와 별도 action clipping은 없다.
RSL-RL `GaussianDistribution(std_type="log")`가 양수 표준편차를 학습하고 정확한 log-probability·
entropy·KL을 계산한다. 초기 std는 **0.3을 유지**해 tanh 제거 효과를 분리한다.
Min의 초기 std1.0까지 맞춘 변경은 아니다. Scale1.0의 pose action에는 이제 ±1 상한도 없다.
현재/직전 observation의 action 값도 ±1로 자르지 않고 실제 raw 값을 보존한다.
Hand 관절 한계와 OSC Kp·actuator 토크 제한은 각 native 클래스에서 적용한다.

출력 분포는 checkpoint의 `output_distribution`에 기록한다. 이전 tanh checkpoint는
새 Gaussian으로 resume/play하지 않도록 weights·optimizer 로드 전에 명시적으로 거절한다.
변경된 분포는 새 학습으로 평가해야 한다.

`build_policy_masks()`가 observation의 extent·quaternion·센서 유효값과 고정 슬롯에서
mask를 만든 다음 모델에 별도로 전달한다. Mask·role ID를 observation이나 학습 feature에
붙이지 않는다. 물체 padding, reset 시 직전 step 이력의 결측, tactile 결측은 해당 mask로 제외한다.
Mask 판정에 필요한 원래 값을 유지하므로 추가 running observation normalization은 사용하지 않는다.

설정은 `config/ur5e/agents/rsl_rl_ppo_cfg.py` 한 곳에서 조합한다.

| 항목 | 기본값 |
|---|---|
| Rollout / PPO epochs / minibatches | 32 step / 4 / 4 |
| Learning rate / schedule | 3e−4 / fixed |
| γ / GAE λ / PPO clip | 0.99 / 0.95 / 0.2 |
| Value loss / entropy coefficient | 0.25 / 0.01 |
| Max gradient norm / 초기 Gaussian std | 0.5 / 0.3 (학습 가능한 log std) |
| 추론 chunk / 학습 microbatch | 최대 8,192 / 8,192 sample |
| 학습 iteration / checkpoint 주기 | 1,000 / 50 |
| Physics / policy 주파수 / timeout | 100 Hz / 50 Hz / 8 s |
| 기본 환경 수 / 환경 간격 | 1 / 2.5 m |

환경 수는 CLI로 늘릴 수 있다. 모델 추론 chunk와 PPO microbatch는 전체 환경 수와 별개다.
Minibatch가 `max_microbatch_size` 이하이거나 제한이 `None`이면 **native `PPO.update()`**를 호출한다.
현재 기본 설정에서는 1,024환경 × rollout 32 ÷ minibatches 4 = 8,192이므로 native 경로다.
상한을 초과할 때만 각 microbatch loss에 표본 비율을 곱해 gradient를 누적하고,
gradient clipping과 optimizer step은 원래 minibatch당 한 번 수행한다.
Actor와 critic이 같은 파라미터를 optimizer에 두 번 등록하지 않으며 rollout의 공통 encoder 계산도 재사용한다.

RSL-RL 5.x의 모델 API를 사용한다. 콘솔·TensorBoard 출력은 native logger를 그대로 사용하고
기본 경로는 native value/surrogate/entropy loss를 기록하고, microbatch 경로만 추가 진단값을 기록한다. 현재 single-GPU 학습을 지원하며 RND·symmetry·
multi-GPU 및 JIT/ONNX export는 구현하지 않았다.

## MDP

기준은 [기존 연구 설계](../../docs/CONTEXT.md)다. MDP 구현은 기존 네 개 모듈과
`sensors.py`에 모았다. 제어는 UR5e env cfg의 공식 ActionCfg에 설정하며,
공통 상태·배치 sampler·reset을 별도 파일로 분리하지 않았다.

| 영역 | 구현 |
|---|---|
| Observation | 초기 vision 기하 10×10 + 현재·직전 sample 2×72 = **244D** |
| 물체 정보 | 현재 UR5e 말단 `wrist_3_link` 기준 위치·quaternion·OBB extent. 초기 추정값은 episode 내내 고정 |
| 슬롯 | Target 0, Blocker 1, 나머지 관측 물체와 zero padding. Role ID·mask 채널 없음 |
| 72D sample | 12관절 q/dq 24 + base에서 표현한 `wrist_3_link` pose 7 + sensor 기준 F/T 6 + tactile 17 + 직전 적용 action 18 |
| F/T | SI-200-8 축별 clipping·양자화 → distal 링크 자중·병진/회전 관성 보상 → full scale 정규화 |
| Tactile | 마지막 physics sample의 normal force norm > 0.01 N이면 1, 미접촉 0, 결측 −1 |
| Action | 현재 EEF 축 TCP 증분 6 + EEF 축 stiffness 6 + hand 독립 관절 위치 6 = **18D** |

제어는 [UR5e 환경 cfg](contact_manipulation/config/ur5e/ur5e_env_cfg.py)의 `__post_init__`에서
공식 `OperationalSpaceControllerActionCfg`와 `JointPositionToLimitsActionCfg`로 설정한다.
공통 `ActionsCfg`는 두 term을 `MISSING`으로 선언한다. 별도 custom action/controller 클래스는 없다.
`ActionManager.process_action → native process_actions/set_command → physics substep의
native apply_actions/compute → scene.write_data_to_sim` 순서로 실행한다.

OSC gain·scale 설정은 원격 `min` branch의
[`UR5eSweepKpOSCRelativeObsEnvCfg`](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py)를 따른다.
2026-10-20 사용자 교정에 따라 제어 body는 UR5e의 실제 말단 rigid body인
`wrist_3_link`로 정한다. `body_offset`은 사용하지 않으며 `tool0` 가상 frame과 구분한다.
Action 좌표계와 물체 observation의 기준도 **현재 `wrist_3_link` frame**으로 통일한다.
`task_frame_rel_path="Robot/wrist_3_link"`와 scene `ee_frame`의 같은 링크 target을 지정해
공식 ActionCfg·FrameTransformer가 좌표 변환을 수행한다. 확인한 min 커밋 `baa3740`은
task frame을 지정하지 않아 base 축 증분을 사용한다는 점에서 다르다.

| 입력/설정 | 현재 native 동작 |
|---|---|
| Arm 앞 6D | 현재 EEF 축 `pose_rel`; position/orientation scale 모두 1.0 (m, rad) |
| Arm 뒤 6D | EEF 축별 `Kp = clip(100 × action, 50, 200)`; 병진·회전 동일 규칙 |
| 현재 Gaussian 정책의 Kp 범위 | raw 출력에 ±1 상한이 없어 native 범위 50–200을 사용 가능 |
| OSC | Full inertia decoupling, damping ratio 1, gravity compensation=True, nullspace 없음 |
| Hand 6D | raw action을 native에서 [−1,1]로 clamp한 뒤 live soft limit 범위에 매핑 |

Scale 1.0은 위치 action 0.01이 1 cm, 회전 action 0.01이 0.01 rad라는 뜻이다.
정책 step 시작 시 측정한 EEF pose를 task frame으로 사용한다. Base 기준 EEF pose가
$(p_t,R_t)$이면 목표는 $p^*=p_t+R_t\Delta p_{\mathrm{EEF}}$,
$R^*=R_t\operatorname{Exp}(\Delta r_{\mathrm{EEF}})$다. 회전 입력은 Euler가 아닌 회전벡터다.
공식 OSC는 EEF 축 stiffness/damping도 base로 변환하며, 생성된 목표는 해당 정책 step의
physics substep 동안 유지한다. 이전 목표에 증분을 계속 누적하지 않는다.
이전의 ±15 mm/±0.08 rad 제한과 축별 gain 매핑, workspace 목표 clipping,
COM→TCP Jacobian/속도 보정, hand 측정 위치 기준 lead 제한, custom effort clipping은 제거했다.
Native task-frame 변환과 기존 관절별 물리 제한을 사용한다.
OSC의 gravity compensation은 관절 중력 토크를 feedforward로 더해 제어 출력을 만드는 기능이다.
F/T 자중·관성 보상은 측정값에서 distal 링크의 하중을 제거하는 별도 센서 처리로 유지한다.
둘은 서로 다른 경로이며 중복 보상을 뜻하지 않는다. 이 설정만으로 정책 학습 개선을 보장하지 않는다.
Action/적용 torque의 nonfinite는 MDP에서 검사한다.
Arm actuator PD gain은 0이고 mimic 관절에는 별도 action을 주지 않는다.
2026-10-20 실제 stage 조사에서는 native mimic/tendon 제약이0개였다. 설치된 importer가
현재 cfg의False를 `parse_mimic=False`로 넘겨 distal6관절이 passive인 상태이며,
URDF의 mimic 관계가 물리적으로 유지된다고 보장하지 않는다. 후속 교정 항목으로 기록했다.
예제처럼 OSC 환경의 `__post_init__`에서도 arm stiffness/damping=0을 명시한다.
팔은 `IdealPDActuatorCfg`를 사용해 OSC가 계산한 토크를 PhysX에 전달하기 전에 제한한다.
`effort_limit`과 `effort_limit_sim`은 모두 shoulder/elbow 150·wrist 28 N·m이며,
`velocity_limit_sim`은 shoulder/elbow 3.14·wrist 6.28 rad/s다. Hand는 기존 implicit
position PD를 유지한다. 별도 custom controller나 torque clamp는 추가하지 않는다.
이 출력 제한이 정책의 delta pose/stiffness 요구를 제한하거나 적절한 요구를 학습하도록
보장하지는 않는다. 측정 joint torque observation/reward는 아직 추가하지 않았다.
속도 초과 자체에 대한 episode 종료는 연구 MDP에 추가하지 않는다.

Action 18D와 observation 244D, GNN·attention 구조는 유지한다. 제어 의미가 달라졌으므로
이전 base-frame·손바닥 EEF·custom controller checkpoint는 action-contract metadata 검사에서 거절한다.
현재 규약은 `isaaclab_ur5e_wrist_osc_variable_kp_gravity_v3`이며 새 학습을 시작한다.
244D 모양은 같지만 물체 기준점·관측 EEF pose·action의 의미와 중력 보상이 달라졌다.

| Reward | Weight | Weight 적용 전 discrete step 값 |
|---|---:|---|
| Clearance | 1.8 | $\exp(-a_C x_C)$, $x_C=A_{\mathrm{blocked}}/A_{\mathrm{blocker}}$ |
| `blocker_distance` | 0.40 | $\exp(-a_d d)$, 17개 pad 표면 대표점에서 현재 Blocker OBB까지의 최소 거리 |
| `blocker_contact_hold` | 0.01 | $\sum_{i=1}^{17}(b^B_{i,k-1}\land b^B_{i,k})$, 같은 pad의 직전·현재 Blocker 접촉 |
| Rearrangement | 0.6 | $\exp(-a_R x_R)-1$, $x_R=\Delta L_{\mathrm{OBB}}$ (이번 step의 이동량, m) |
| `motion_action_rate` | 0.15 | $\exp(-a_A x_A)-1$ |
| `motion_joint_velocity` | 0.15 | $\exp(-a_V x_V)-1$ |
| `motion_force_increase` | 0.15 | $\exp(-a_F x_F)-1$ |
| `motion_torque_increase` | 0.15 | $\exp(-a_T x_T)-1$ (F/T 센서 모멘트) |
| Time (현재 비활성) | — | $-\int_t^{t+\Delta t}0.02\bigl(\exp(0.7s)-1\bigr)\,ds$ |
| Terminal (현재 비활성) | — | 성공 +200, 실패 −50, 실패 우선·episode당 1회 |

**Clearance는 양의 주 보상, Rearrangement·Motion은 penalty로 지급한다.**
품질 함수 `q=exp(−a·x)`와 비용 x의 정의는 유지하며, 뒤의 두 그룹은 `q−1`을 지급한다.
비용이 0이면 penalty도 0이고, 비용이 커지면 −1에 가까워진다. 품질 캐시는 0~1을 유지한다.
이전의 R/M 정지 보상 1.2/step을 제거한 변경이며, 작은 Clearance 개선이 항상 모든 비용을
이기도록 보장하는 우선순위 제약은 아니다. 접근·접촉 유지 보상은 별도 지원 항이며,
Clearance 증가량 보상이나 접촉 gate는 추가하지 않았다.

Clearance의 분모는 **현재 Blocker의 XY 투영 footprint 전체 면적**이다. x_C는 초기 접근영역
A₀와 Blocker의 교차 면적을 이 분모로 나눈 0~1 값이다. Target·주변 물체의 overlap은 제외한다.
GPU 다각형 clipping도 Blocker에 대해서만 수행한다.

거리 보상은 실제 tactile collision mesh의 노출된 local +Z 표면에서 대표점 하나씩을
초기화 때 선택하고, 이후 native 링크 pose로 GPU에서 변환한다. Wrist 원점이나 초기 vision의
고정 Blocker pose를 거리 기준으로 쓰지 않는다. $a_d=\ln(2)/(0.10\,\mathrm{m})$이므로
10 cm에서 품질 0.5, 20 cm에서 0.25이며, OBB 내부·표면은 거리 0이다. OBB는 컵의 빈 공간 등
실제 메시와 다를 수 있으므로 실제 접촉은 별도의 filtered force로 판정한다.

접촉 유지의 $b^B_{i,k}$는 해당 pad와 **현재 episode의 Blocker 사이 normal force norm이
0.01 N을 초과했는지**이다. 17개를 평균하지 않고 합산하므로 최대 기여는 0.17/step이다.
최초 접촉은 0, 같은 pad가 다음 policy sample에서도 접촉하면 지급한다. 다른 물체에서 Blocker로
바뀌거나 다른 pad로 접촉을 넘긴 첫 sample은 유지로 세지 않는다. 20 ms 간격의 두 sample을
확인하며 그 사이의 완전한 연속 접촉까지 증명하지는 않는다. 힘을 더 크게 가해도 추가 점수는 없다.

`mdp/sensors.py`의 native `BlockerContactHold(ManagerTermBase)`가 이력을 관리하며,
**공식 RewardManager가 부분 reset된 행만 초기화**한다. 같은 control step을 재조회해도
이력은 한 번만 갱신한다. 하나의 PhysX contact view에서 pad×전체 물체 pool을 읽고,
collection의 실제 `blocker_id` 열을 GPU에서 선택한다. 매 episode 센서를 재생성하지 않는다.
Actor는 기존 **상대 구분 없는 binary tactile 17개와 244D observation**을 유지하며,
Blocker 구분·현재 pose는 reward의 privileged 정보로만 사용한다.

두 항의 weight는 초기 실험값이며, 접촉한 채 정지하는 행동도 양의 점수를 받는다.
따라서 접촉률과 함께 **접촉 이후 Clearance 증가·성공률**을 비교해야 한다.
접촉 판별을 제외하려면 `env.rewards.blocker_contact_hold=null`로 실행한다. 이 경우
기존 grouped net-contact backend와 거리 항이 유지되며 다른 물체 접촉으로 보상을 대체하지 않는다.

Rearrangement의 x_R는 모든 active 물체 각각의 대응 OBB 꼭짓점 8개의 변위 norm을 평균한 뒤
물체끼리 합산한 이번 step의 이동량이다. 회전·pivot·왕복 이동, Target·Blocker·초기 vision에
보이지 않은 물체도 privileged pose로 반영한다. 누적 `path_length`는 진단용이며 보상에 넣지
않는다. **멈춘 step에는 x_R=0, penalty=0**이므로 과거 이동량에 반복 감점을 주지 않는다.
지수 점수이므로 누적 비용이 전체 이동거리와 정확히 선형 비례하지는 않는다.

Motion은 다음 네 비용을 **각각 exp에 넣고 1을 뺀 독립 term**으로 기록한다. 각 weight0.15의
합은 `0.6 × 네 penalty의 평균`과 같다. 가속도 항은 없으며 tactile 유지 보상은 별도 항이다.

$$
\begin{aligned}
x_A &= \sum_{j=1}^{18}(a_{j,k}-a_{j,k-1})^2, \\
x_V &= \sum_{j=1}^{12}\dot q_{j,k}^{\,2}, \\
g_F &= \frac{\max(\lVert F_k\rVert_2-\lVert F_{k-1}\rVert_2,\,0)}{\Delta t}, \\
g_T &= \frac{\max(\lVert \tau_k\rVert_2-\lVert \tau_{k-1}\rVert_2,\,0)}{\Delta t}, \\
x_F &= \max(g_F-10,\,0)^2, \\
x_T &= \max(g_T-0.5,\,0)^2.
\end{aligned}
$$

공식 `isaaclab.envs.mdp.rewards.action_rate_l2`와 `joint_vel_l2`를 사용한다. Action은 raw
18D 제곱합이며 `/dt`가 없다. Joint velocity는 팔6+hand 독립6관절의 제곱합이고 mimic/passive
관절은 제외한다. 기준값·관절 수로 나누거나 제곱근을 취하지 않는다. 속도는 마지막 physics
substep의 값이며 policy 구간 적분이 아니다. 기존 GPU 버퍼를 사용하고 센서 조회를 추가하지 않는다.

F/T는 자중·관성 보상 후 **현재 센서 중심**에서 비교한다. 축 회전은 norm을 바꾸지 않으므로
world 축으로 돌릴 필요가 없고, 초기 센서 원점으로의 `r×F` 이동도 하지 않는다. 일정 하중·하중
감소·norm이 같은 방향 변화는 F/T 점수1이다. `wrench_rate_cost`는 `(num_envs, 2)`로 힘·토크의
초과 증가율 제곱을 각각 저장한다. Tactile 유지 보상은 이 네 Motion 항에 합치지 않는다.

| 계수 | 설정값 | 의미 |
|---|---:|---|
| `a_C`, `a_R` | 1, 2 | 기존 Clearance·Rearrangement 지수 유지 |
| `motion_action_rate_decay` | 0.004 | Raw action 변화 제곱합의 민감도 |
| `motion_joint_velocity_decay` | 0.004 | 관절 속도 제곱합의 민감도 |
| `force_rate_deadband` | 10 N/s | 작은 힘 증가율의 무감점 구간 |
| `torque_rate_deadband` | 0.5 N·m/s | 작은 토크 증가율의 무감점 구간 |
| `motion_force_rate_decay` | `ln(2)/90²` ≈ 8.55737e−5 | 힘 증가율100 N/s에서 해당 점수0.5 |
| `motion_torque_rate_decay` | `ln(2)/4.5²` ≈ 0.0342295 | 토크 증가율5 N·m/s에서 해당 점수0.5 |

Action·velocity의 계수는 Motion 하위 weight0.15와 함께 작은 비용에서의 감점 기울기
`0.15×0.004=0.0006`을 만든다. 이는 `min`의 native penalty weight0.03×dt0.02와 같지만,
상수 baseline과 큰 비용의 포화는 다르다. F/T dead band는50Hz에서 양자화·작은 보상 잔차를
무시하기 위한 초기값이며, half-score 지점도 실제 성공 조작으로 검증한 허용 한계는 아니다.

Velocity 항은 빠른 움직임 자체도 감점하므로 약한 규제로 시작한다. 감속하면 이 비용은
줄지만 action-rate는 필요한 제동 명령도 감점한다. Raw Kp가 clamp되는 구간의 action 변화도
계산에 포함되므로 action-rate를 실제 가속도나 stiffness 변화의 정확한 측정으로 해석하지 않는다.
정지 시 Motion 기여는 0이며 현재는 Clearance와 접근·접촉 지원 항이 조작을 유도한다.

현재 `RewardsCfg`는 Clearance/거리/접촉 유지/Rearrangement/Motion4항의 **8개 활성 term**이다.
사용자가 비활성화한 Time·Terminal은 그대로 비활성이다. 함수가
discrete 값/`step_dt`를 반환해 native manager의 `weight × dt`와 상쇄한다. 거리 weight만0.10→0.40으로 조정했으며
C의 최대 기여는1.8, R의 최소 기여는−0.6, Motion 합의 최소 기여는−0.6이다.
Time을 활성화하면 위 적분값을 지급한다. Step별 지급이므로 전체 return의 제어주기 불변성을
주장하지 않는다. γ와 관측·제어는 유지한다.

Terminal은 T06 학습로그를 검토해2026-10-19에 성공500→200·실패100→50으로 조정했다.
당시 step당 non-terminal 상한1.8과 γ=0.99의 할인합 상한180을 기준으로 성공200을 두고,
반복 관측된 실패100의 충격을 절반으로 줄였다. **거리·접촉 항 추가 후 상한은
2.37/step, 할인합 상한은237**이므로 이전180을 현재 정당화 근거로 사용하면 안 된다.
이번에는 Terminal을 활성화하거나 scale을 재조정하지 않았다. 재활성화 시 다시 검토한다.
Timeout은 Terminal지급0이며 native timeout bootstrap은 유지한다.
근거와 한계는 [학습 기록9절](../../docs/TRAINING_LOG.md)에 둔다.

Console과 TensorBoard에서 `Episode_Reward/motion_action_rate`, `motion_joint_velocity`,
`motion_force_increase`, `motion_torque_increase`를 각각 확인한다. 기존 단일 `motion` 항은
중복 지급하지 않는다. R/M 로그는 이제 0 이하이며 이전 양의 점수 로그와 직접 비교하지 않는다.
이 값은 native manager가 **weight 적용 후 episode 누적값을 최대 episode 시간8초로 나눈 값**이다.
0~1의 순간 점수나 실제 episode 길이로 정규화한 평균이 아니므로 episode 길이도 함께 확인한다.
`Episode_Reward/blocker_distance`와 `Episode_Reward/blocker_contact_hold`도 같은 공식
로깅 방식으로 각각 기록한다. 적용 후1,024환경에서24-step 환경 수집은 약0.919초였으며,
이는 policy 추론·PPO update를 포함한 iteration 시간이 아니다. GPU 성능 비교와 판정 근거는
[학습 기록14절](../../docs/TRAINING_LOG.md#14-2026-10-20--접근접촉-보상과-1024환경-비용-검증)에 둔다.
F/T 두 점수는 state 갱신 때 각각 한 번 계산해 캐시한다. 보상 계산에는 추가 센서·PhysX 조회나
명시적인 CPU 변환을 넣지 않고, 로그 집계는 native manager에 맡긴다.

성공은 **지정 Blocker의 A₀ 교차 면적 ≤ 1e−5 m², Target 초기 pose 대비 이동 ≤ 2 cm·회전 ≤
0.15 rad**를 5 step(0.1초) 연속 만족할 때다. 두 조건의 지속만 확인하며, 물체 속도·다른 물체의
상태·F/T·선반 상태를 성공 gate에 넣지 않는다. 초기 Target이 정확히 하나 있어야 하며
유효하지 않은 Target pose나 교차 면적으로는 성공할 수 없다.
낙하 6 cm, **같은 물체가 80° 이상 기울어진 상태를 0.2초 연속 유지**, 선반의 초기 위치 대비
**1 cm 초과 이동** 또는 초기 회전 대비 **2° 초과 회전**, 유효하지 않은 물리·제어 상태는
별도의 활성 실패 조건이다. `hard_load_or_overrange`는 현재 비활성이므로 보상 F/T 80 N·4 N·m 초과와
raw 측정 범위 초과는 내부 진단에만 남으며, 성공·실패 지급·PPO 종료 mask에 반영하지 않는다.
선반은 `RigidObject`로 관리하며
Sweeping-Policy 예제와 같은 100 kg dynamic body다. 관성은 PhysX가 충돌 형상에서 계산한다.
선반 pose는 종료 조건에만 사용하며 observation에 추가하지 않는다.
부분 reset 시 해당 환경의 선반 pose·속도를 복원하고 초기 pose 기준도 해당 행만 갱신한다.
Timeout은 native truncation 항으로 유지한다.

Tilt는 모든 활성 물체에 대해 개별적으로 누적하며, 50 Hz policy에서 같은 물체가 10 step
연속 기준을 만족해야 종료한다. 80° 미만으로 돌아오면 그 물체의 누적값을 0으로 만들고,
부분 reset 시 해당 환경의 누적값만 초기화한다. 순간적인 기울어짐이나 서로 다른 물체의
교대 기울어짐을 하나의 넘어짐으로 합치지 않는다. 평면 yaw 회전은 tilt에 해당하지 않는다.
2026-10-20부터 `workspace_exit`·`object_lift`는 제거했다. 영역 이탈·들림 자체는 실패나
성공 제한 조건이 아니다. 낙하·전도 실패와 통로 확보·Target 보호의 성공 판정은 분리한다.

실패 원인은 공식 `TerminationManager`의 개별 term으로 기록한다. Console과 TensorBoard의
`Episode_Termination/<원인>`에서 확인할 수 있으며, 기존 통합 `failure` 로그는 아래 항목으로
대체했다. 모든 term은 같은 step의 캐시된 판정 결과를 읽으며 물리·센서 조회를 반복하지 않는다.

| 원인 이름 | 현재 실패 조건 |
|---|---|
| `object_drop` | 활성 물체 OBB 최저점이 지지면보다 6 cm 초과 내려감 |
| `object_tilt` | 같은 활성 물체의 기울기 ≥80°가 0.2초 연속 지속 |
| `hard_load_or_overrange` (현재 비활성) | 보상 F/T 과부하 또는 raw 센서 측정 범위 초과; 진단값만 유지 |
| `shelf_displacement` | 선반 이동 1 cm 또는 회전 2° 초과 |
| `invalid_state` | 유효하지 않은 물리·센서·제어 상태 |

한 episode에 여러 원인이 동시에 기록될 수 있어 원인별 비율의 합이 전체 실패율과 같지는
않다. 현재 활성 실패 4개와 성공·timeout을 합쳐 native 종료 term은 6개다.
성공과 활성 실패가 같은 step에 충족되면 각각 기록하며, 종료는 활성 원인들의 OR이다.
Terminal reward를 활성화할 경우에는 기존대로 실패 우선으로 한 번만 지급한다.
내부 실패와 PPO `terminated` mask도 초기화 때 읽은 native 활성 term 설정을 따른다.
적용 근거와 PhysX 오류의 충돌량 검사는 [학습 기록16절](../../docs/TRAINING_LOG.md#16-2026-10-20--성공-판정-분리거리-weight-적용과-physx-중단-조사)에 둔다.

## 병렬 계산과 reset 연결

```text
환경 생성 1회: robot / shelf / 물체 pool / tactile 17링크 통합 센서 / 공유 ray-casting mesh 준비
새 scene round: Target/Blocker·물체 배치 1개 생성
              → 한 장면의 GPU ray casting·가시성 검증·template 저장
reset(env_ids): 공통 초기 배치·vision 복사
              → 선택 환경마다 Blocker 앞 EEF 위치·회전을 새로 샘플링
              → GPU 배치 IK·elbow-up·전체 로봇 충돌 검사
              → 물체·선반·로봇 상태를 선택 행에 일괄 적용
              → reset_mdp: 초기 vision·역할·누적 reward·직전 이력 확정
400 step 경계: 전체 환경 timeout → 다음 scene round
```

공통 배치·역할·가상 시점은 8초 round마다 교체한다. 중간 auto-reset에서는 해당 환경만
저장된 초기 배치로 복원하며, 로봇 자세는 매번 새로 생성한다. 탐색을 소진한 예외 행만
같은 장면의 검증된 기준 자세로 복구한다. 일반 sampling용 고정 자세 bank는 없다.
배치 seed는 기본 seed와 공통 round 번호로, 로봇 seed는 기본 seed·환경 ID·reset 횟수로
정한다. 다른 환경의 reset 대상 순서는 로봇의 난수열을 바꾸지 않는다.

공통 장면의 proposal·충돌 OBB·로봇 시작 영역과 역할 mask·local corner·초기 vision 정렬은
캐시해 부분 reset에서 재사용한다. 새 template에서는 다시 검증·계산하며, 시작 영역은
robot/spawn 설정이 달라져도 갱신한다. Native `reset_scene → reset_task`의 같은 reset 대상에만
검증 재사용을 허용하고, 외부 `SceneBatch` reset은 기존 검증을 수행한다. Catalog와 template은
생성 후 수정하지 않는 입력이다. 로봇·선반의 실제 초기 pose 검사는 매 reset마다 유지한다.
CUDA IK 요청은 먼저8개 후보를 검사하고, 남은 후보는32환경 이하에서 최대32개씩,
더 큰 요청은8개씩 처리한다. GPU accepted mask로 이미 성공한 행의 IK·충돌 검사를 건너뛰고,
후보 탐색 종료 후 한 번만 결과를 CPU로 전달한다. Native reset은 관절값·TCP·시작 영역을
batched tensor로 직접 적용한다. 전체 동시 후보 수는 `ik_batch_size` 이하로 제한하고,
고정 크기 CUDA graph 캐시는 최대8개다. Graph buffer는 일반 tensor로 생성해 학습과
play의 InferenceMode 전환을 지원한다. 절대 후보 번호별 난수·branch와 첫 유효 후보 선택은 유지한다.

충돌 검사는 CUDA FP32 Warp kernel에서 collider 변환·ground/elbow·scene/self collision을
통합한다. IK 실패 후보, AABB로 분리된 pair, 첫 충돌 이후의 불필요한 검사를 생략한다.
Scene15mm·self2mm 여유는 유지한다. 거의 평행한 축은 원래 좌표의 FP64 projection으로
계산해 FP32 상쇄로 겹침을 놓치던 문제를 보정한다. CPU·다른 dtype은 eager 수식을 사용한다.

Reset snapshot은 초기 OBB·로봇/선반 pose·관절값만 읽는다. 초기화 시 버리는 F/T·tactile·action과
가속도·물체 속도는 계산하지 않는다. 정상 step의 F/T 자중·관성 보상은 유지하며, 고정 local COM은
한 번 GPU에 저장하고 현재 link pose로 world COM을 계산한다. COM·질량·관성을 randomize한다면
이 고정 property cache를 함께 갱신해야 한다.

2026-10-15 RTX3090·1,024환경·10물체의 전체 최적화 후30 iteration에서 초기 compile 회차와
profiler 적용 회차를 제외한28회 평균은 **1.84초/iteration**이었다(1.73~2.02초).
직전14회 측정3.21초 대비 같은 iteration14개 비교는3.21→1.85초, 약42.5% 감소다.
Reset은1.02→0.31초, PPO update는0.32→0.23초로 줄었다. Native PhysX partition은1이며,
100Hz physics/50Hz policy, solver16/2, rollout32·PPO4epochs/4minibatches는 유지했다.
짧은 새 학습의 비교이므로 trajectory와 reset 행 수는 완전히 같지 않다. 상세 계측은
`outputs/contact_manipulation/full_optimization_20261015/comparison.json`에 있다.

현재 실험은 **10종을 하나씩, 총 10개** 생성하고 매 episode 모두 배치한다.
배치 전에 Target/Blocker를 정하고, 나머지 8개를 주변 물체로 사용한다. 중복 물체는 없다.
물리 pool과 관측 capacity 모두 10개다.

선반 전체 점유율은 **≥20%**, 2×3 분할 영역별 점유율은 **≥10%**다.
Footprint 면적 합을 가장자리 여유를 제외한 선반 면적으로 나눈 기준이다.
실제 USD에서 측정한 10종의 합산 점유율은 약 21.85%여서 이전 35% 조건은 달성 불가능하다.
이번 10개 실험에 맞춰 전체 점유율 기준만 20%로 낮췄으며 선반 범위와 역할·가시성 조건은 유지한다.

이전 30/40-slot checkpoint는 모델 설정과 observation 크기가 달라 기본 resume/play와 호환되지 않는다.

배치 생성은 CPU의 역할·물체 구성·조작 경로 준비와 Warp의 주변 물체 배치로 나눈다.
Blocker는 footprint 차단율을 만족하는 Y 구간 안에서 샘플링한다. 물체 높이와 전체 점유율은
구성 확정 시 확인하며, 주변 물체 위치는 해당 yaw에서 선반 안에 들어가는 중심 범위에서 만든다.

CUDA 환경에서는 최대 512개 장면의 배치 상태를 GPU에 유지한다. 한 물체당 위치·yaw 후보
64개를 병렬로 생성·검사·채점하고, 성공한 위치를 저장한다. 두 번 연속 후보가 없으면
후보와 가장 많이 충돌한 주변 물체 하나만 빼고 현재 물체를 먼저 놓은 뒤 다시 채운다.
Target·Blocker와 조작 경로는 이 부분 수정 동안 유지한다. 장면당 최대 16회 부분 수정 후에도
해결되지 않으면 같은 물체 구성으로 역할 pose·경로를 다시 제안한다. 전체 탐색에도 상한이 있다.

정밀 충돌 검사는 AABB로 가까운 장애물을 추린 뒤 수행한다. 경로 다각형의 축·투영 구간은
한 번 계산해 재사용하고, 구역별 점유 면적·이웃 관계는 변경된 물체에 대해서만 갱신한다.
GPU 기하는 float64/비융합 연산이며, CPU 환경도 같은 Warp 생성기를 사용한다.
Pivot sweep/convex hull과 Blocker 허용 구간 계산은 Numba로 컴파일한다.
조작 경로 후보도 고정 크기 배열로 생성해 Python 객체를 후보마다 만들지 않고,
선택된 경로만 객체로 복원한다. GPU 배치 kernel의 인자는 batch마다 한 번 준비해 반복 실행한다.
선택한 경로의 회전 구간·footprint·sweep hull은 Numba에서 배열로 만들고, 정점 배열과 유효
정점 수를 배치 준비까지 유지한다. 구간/꼭짓점마다 Python tuple을 만드는 왕복 변환은 생략한다.
생성 알고리즘이 바뀌었으므로 이전 버전과 같은 seed의 위치는 달라진다. 같은 버전·device에서는
환경 순서와 batch 크기에 독립적인 seed 결과를 유지한다. CPU/GPU 사이의 완전한 좌표 일치는
보장하지 않으며, 두 경로 모두 독립 다각형 검사로 연구 조건을 검증한다.

Blocker의 이동·pivot·회전 후 이동을 포함하는 지지면 내 충돌 없는 경로를 하나 확보하고,
그 경로와 hand 시작 영역을 비워 둔다. 이 경로는 배치 가능성을 확인하는 메타데이터이며
policy에 제공하지 않는다. 로봇의 실제 collider OBB와 선반·물체를 검사하고,
비인접 arm 링크 및 **hand–arm** 사이의 자기 충돌 여유 2 mm를 확인한다.
Elbow-up(shoulder–wrist 선분 대비 높이 ≥5 cm)에 더해, 전완이 되접히지 않도록
손목 관절 중심이 팔꿈치보다 선반 방향(-X)으로 ≥5 cm 앞에 있어야 한다
(`robot_start.min_forearm_forward=0.05`). 손 방향과 Blocker 앞 TCP randomization은 유지한다.
고정 장착 관계인 hand–wrist_3/센서 및 열린 hand 내부는 IK 자기 충돌 쌍에서 제외한다.
Arm끼리는 기존 기계적 하우징의 두 관절 이내 예외를 유지한다. 이 검사는 **초기 자세 승인용**이며
PhysX articulation의 runtime self-collision 설정을 바꾸지 않는다.
Box는 평면 collision proxy를 사용하고 바닥 높이에 바로 놓으며
초기 속도를 0으로 만든다. Reset 안에서 물리 simulation을 진행하는 settling loop는 없다.

여기서 convex hull은 USD의 물리 collider를 재생성하는 작업이 아니다. 배치 단계에서는
이동/pivot 도중 물체가 지나가는 평면 영역을 보수적으로 감싸서 재배치 공간을 예약한다.
회전은 최대 5° 구간의 양 끝 footprint hull에 회전 궤적의 오차 여유를 더한다.
MDP에서는 기울어진 OBB의 8개 꼭짓점을 XY에 투영한 hull과 A₀의 교차 면적을 계산한다.
USD collider는 PhysX의 접촉 계산에 사용하며, 경로 여유·선반 점유율·A₀ 차단율을 자동으로
제공하지 않는다. 배치용 OBB/sweep은 실제 collider보다 보수적인 근사이고 물리 조작 성공을 보장하지 않는다.

초기 관측에는 **실제 카메라·render product·annotator·참조 물체를 생성하지 않는다.**
Isaac Lab `RayCaster`와 같은 Warp mesh query를 사용한다. 물체 10종의 시각 메시를 한 번
읽어 종류별 BVH를 공유하고, 활성 물체만 검사한다. 물체 복제본마다 메시를 다시 만들거나
움직일 때 BVH를 재구성하지 않는다. **로봇·선반에 의한 가림은 제외**한 관측 모델이다.
이 제외는 ray 관측에만 적용하며, 물리 시뮬레이션과 로봇 초기 자세의 충돌 검사는 유지한다.
시점 난수·카메라 자세 생성, ray query, 가시율과 후보 선택, OBB 추정은 환경과 같은 CUDA
device에서 배치 처리한다. 물체·시점별 좌표 변환은 한 번 계산해 모든 픽셀이 공유한다.
시야 밖 광선은 Target/Blocker 메시로 바로 연결한다.

Reset마다 선반 앞 가상 시점의 위치·방향을 기존 범위에서 새로 샘플링한다. 기본 160×120 광선
격자에서 가장 가까운 표면을 선택하고, Target·Blocker만 따로 조회한 고립 실루엣 대비
**Target ≥40%, Blocker ≥95%**를 확인한다. 분모는 같은 자세·광선 간격의 2배 넓은 격자로
계산하며, 실제 시야에서 잘린 role 물체는 거절한다. 이는 렌더링 픽셀 대신 **광선 표본 면적**으로
계산한 근사 가시율이다. `initial_vision.ray_width`, `ray_height`로 해상도를 조절한다.

Observation의 OBB는 **첫 교차점으로 보이는 표면만** 모아 upright PCA로 추정한다. 보이지 않는
물체는 padding으로 남기며, 물체의 정답 pose·전체 크기로 가려진 부분을 채우지 않는다.
이상적인 교차점에는 depth outlier가 없으므로 이전 depth의 1–99% quantile 대신 표면 extrema를
사용한다. CUDA에서는 배경 ray를 제외한 count·moment·bounds를 Warp atomic 연산으로 계산한다. 초기 기하는
기존처럼 episode 중 고정되고 현재 EEF frame으로만 재표현한다. Actor 244D 규약은 유지한다.

`initial_vision.ray_batch_size=32`개 episode 요청을 처리하고, 완료한 자리에 다음 요청을 넣는다.
`ray_candidates_per_env=4`개 시점을 환경마다 한꺼번에 평가하므로 기본 최대 128개 시점을
동시에 검사한다. GPU에서 첫 통과 후보를 선택하고 그 후보에만 OBB를 계산한다. 환경별
독립 seed/후보 번호로 난수를 생성하므로 요청 순서·묶음 크기에 따라 후보가 바뀌지 않는다.
이전 Python 난수열과 같은 시점을 생성하지는 않는다. CPU에는 요청 포장·완료 큐 관리와
결과 객체 조립이 남으며, 후보 묶음당 선택 인덱스 한 번과 큐 완료 후 통계만 CPU로 읽는다.
시점 상한 32회, 가시율 40%/95%, 최소 실루엣 32픽셀·관측 16점은 유지한다.
RGB는 만들지 않는다. 디버깅 시
`capture_batch(..., retain_images=True)`는 합성 ray depth와 role mask만 반환한다.
Headless 학습은 카메라 기능을 켜지 않고 reset에서도 render를 호출하지 않는다.
GUI·`--video`를 요청한 경우 화면 표시/녹화의 native 렌더링은 별도로 유지한다.

이 관측 모델은 표면을 불투명하게 취급하며 조명·투명/반사 재질·검출 실패·센서 노이즈를
재현하지 않는다. Triangle mesh는 원본을 사용하고 USD primitive는 tessellation한다.
물체 association도 이상적이므로 실물 vision과의 차이는 별도로 검증해야 한다.

50 mm 연결부 중앙 fixed joint의 실제 reaction을 F/T로 읽고, SI-200-8 변환·보상은 MDP에서
한 번 적용한다. Tactile은 공식 `ContactSensor` 하나가 17개 링크를 함께 읽는다.
`net_forces_w`의 `[N,17,3]`를 `TACTILE_BODY_NAMES` 순서로 정렬하고, 링크별 0.01 N
기준으로 `[N,17]` binary observation을 만든다. 힘을 링크끼리 합치지 않는다.
기존 arm·spacer 금지 접촉 센서는 제거했으며, 선반 이동 여부를 종료 조건으로 사용한다.
부분 reset 직후 해당 행의 측정값은 무효이며, 다음 물리 step의 새 측정부터 유효하다.

Step 계산은 `[N,10,...]`, `[N,12]`, `[N,17]`, `[N,18]` 텐서로 처리한다.
환경별 Python loop, `.item()`, `.cpu()`, `bool(tensor)`를 step 경로에서 사용하지 않는다.
OBB hull·clipping은 고정 크기로 병렬 처리하고, 6×6 제어 행렬 계산은 native OSC에 맡긴다.
관절/body ID·센서 scale·질량·관성·catalog corner는 초기화/reset 때 저장한다.
기존 custom implicit damping·Newton–Schulz·제어 CUDA graph는 제거했다.
Native OSC의 batched mass/task-inertia 역행렬과 PD 계산을 그대로 사용한다.
따라서 SDK 내부 선형대수 호출까지 host synchronization이 없다고 보장하지 않는다.
매 step의 기하 좌표 변환·F/T 보상·동적 sample 구성은 물리 데이터를 읽은 뒤 CUDA graph로 처리한다.
Action/effort의 nonfinite flag는 환경별 실패 판정으로 연결한다.
OBB footprint의 hull·clipping·면적은 활성 물체만 하나의 Warp kernel에서 계산한다.
CPU는 기존 PyTorch 식을 검증 기준으로 유지한다.

Tactile 통합 센서 1개는 전체 환경의 갱신 시점이 같은 특성을 사용해 native `nonzero()`
인덱스 탐색을 생략한다. Observation을 읽을 때 최신 native force buffer를 조회한다.
선반 위치·회전 비교도 GPU tensor로 처리하고, quaternion 내적의 절댓값을 사용해
`q`와 `-q`를 같은 회전으로 취급한다. 전체 scene sensor는 tactile과 EEF frame의 2개다.
비활성 물체의 중력 상쇄는 reset 시 native permanent wrench composer에 설정한다.
매 step force 적용과 transient wrench 합성은 native collection이 수행한다.
전부 활성 상태면 composer를 비활성으로 유지하여 매 physics step의 0-force 제출을 생략한다.
실행 중 simulation/gravity flag를 전환하지 않는다. Permanent composer는 주차용으로 예약한다.

YCB 4종은 native `CollisionPropertiesCfg(contact_offset=0.005)`를 사용한다.
기존 6종의 자동 offset은 유지한다(현재 PhysX 측정 약 1.962 mm). Rest offset은 0이다.
공유 USD는 최종 충돌 설정을 bake한 reference로 교체하여 원본의 20 mm 설정이
instance 생성 뒤 되살아나지 않도록 한다. 매 step이나 reset에서 offset을 변경하지 않는다.

Hand의 `right_plam_1`과 네 `right_*_2` 링크는 각각 **2개 저정점 convex collider**를
사용한다. 관절 쪽 돌출부와 센서 쪽 본체를 분리하고, 본체를 pad 표면 뒤로 물려
tactile 접촉면을 확보한다. 기존 5×128개 분해 형상을 5×2개로 대체하여 실제 PhysX의
로봇 전체 shape 수는 676→46개다. UR5e, 17개 tactile 링크와 collider, F/T 연결,
관절·질량·COM·관성은 유지한다. 이는 상세 CAD의 충돌 근사이며 미세한 홈까지 보존하지는 않는다.
형상 생성은 최초 template 구성 때만 수행하고 복제 환경에서 공유한다.

GNN edge MLP의 첫 선형 연산을 node별 projection으로 분해해 `[N,10,10,134]`
concatenation과 반복 projection을 제거했다. 두 번째 선형 연산은 message 평균 이후로 옮겨
edge마다가 아닌 node마다 한 번 계산한다. 이웃이 없는 node에는 선형 bias가 유입되지 않게
처리한다. 기존 dense 식과 출력·gradient가 일치하며 모델 파라미터와 checkpoint 구조도 동일하다.
CUDA FP32에서는 ReLU·edge mask·합산을 Triton kernel 하나로 처리하여
`[B,10,10,64]` pair activation을 저장하지 않는다. Backward에서도 node별 입력으로
ReLU의 부호를 재계산해 receiver/sender gradient를 구하며, 큰 pair gradient도 만들지 않는다.
CPU 및 다른 dtype은 같은 PyTorch 식을 사용한다. 파라미터 이름·shape와 checkpoint는 유지한다.
Forward batch 상한과 PPO microbatch는 8,192다. 기본 PPO minibatch를 다시 1,024개씩
나누지 않으며, 1,024환경 rollout은 실제 입력 1,024개를 한 번에 처리한다.
Rollout·epoch·minibatch 수와 optimizer 갱신 횟수는 그대로다.
기본 PPO 경로는 native의 gradient clipping과 loss 전송 방식을 따른다.
따라서 native update 내부의 minibatch별 `.item()`까지 제거한 구현은 아니다.

CUDA에서는 `agent.actor.compile_model=true`가 기본이며, GNN·attention·mask·마지막 MLP의
순수 tensor forward를 Inductor로 compile한다. 모델 구조와 checkpoint key는 유지한다.
최초 추론/학습 compile 시간이 추가된다. CUDA graph를 쓰는 `reduce-overhead`는 native 학습의
InferenceMode 전환에서 오류가 재현되어 사용하지 않는다. CPU는 eager이며,
`agent.actor.compile_model=false`로 CUDA에서도 eager 실행을 비교할 수 있다.

2026-10-15 RTX 3090·1,024환경·10물체·30 iteration 측정에서 forward 분할 상한만
1,024→8,192로 늘렸을 때 collection+learning 평균은 **2.961→2.280초**였다
(최초 iteration 제외, 공통 scene 교체와 부분 reset 포함). PPO update는 **0.926→0.313초**다.
모델 구조·학습 수식은 같지만 GPU batch shape에 따른 수치 차이로 학습 trajectory가
bitwise 동일하지는 않다. 위 값은 짧은 처리량 측정이며 장기 학습 성능 비교는 아니다.

콘솔·TensorBoard 기록 약 **7.3 ms**, episode 로그 집계 **7.2 ms**, NaN 검사 **4.6 ms**로
합계 약 19 ms/iteration이었다. Native logger와 기본 NaN 검사는 유지한다.
Checkpoint는 50 iteration마다 및 마지막에 저장한다. Native 출력의 `Iteration time`에는
콘솔·TensorBoard 기록과 checkpoint 저장이 빠지므로 실제 wall time과 구분해야 한다.
기본 학습에 별도 profiler·강제 CUDA 동기화·센서 debug visualization은 켜져 있지 않다.

`events.py`의 `TaskState`는 누적 조작량, 직전 속도·wrench, 두 step 이력을 저장한다.
Native `common_step_counter`가 바뀐 경우에만 갱신하여 observation/reward/termination의
반복 호출로 이력이 밀리거나 시간이 중복 누적되지 않는다. 부분 reset은 선택된 행만 초기화한다.
물체 순서 정렬과 producer 데이터 검증은 reset에서만 수행한다.

Reset event와 센서 reader가 다음 데이터를 제공한다.

- `env.task_scene_data`: `SceneBatch`. Active collection IDs, Target/Blocker IDs, 승인한 초기
  vision 기하와 association, catalog OBB, A₀·workspace, F/T distal body 이름을 담는다.
- `env.task_sensor_reader()`: 완료된 physics sample의 `SensorBatch`. 이상적 joint reaction,
  유효성과 tactile 17채널을 반환한다. 선반 pose는 native rigid-body data에서 별도로 읽는다.

`EventCfg.reset_scene`에서 배치·로봇 pose·vision을 확정한 뒤 `reset_mdp`가 이 데이터를
저장한다. 물체 pose는 reward·종료 조건에만 사용하며 초기 vision을 대신하지 않는다.
정적 USD 기하와 로봇 FK 모델은 최초 한 번만 검사한다.

전체 환경은 **8초(400 policy step)의 공통 scene round** 동안 같은 초기 물체 배치와
가상 카메라 시점·vision 정보를 공유한다. 움직이기 시작한 뒤 물리 상태는 환경별로 독립적이다.
중간 성공·실패 시 해당 환경의 물체·선반만 저장된 초기 상태로 되돌린다.
**로봇은 매 reset마다 Blocker 앞 허용 영역에서 hand 위치·회전을 연속적으로 샘플링**하며,
GPU IK·elbow-up/전완 방향·충돌 검사를 수행한다. 이 배치용 TCP는 손바닥
`right_plam_force_sensor`를 유지한다. OSC·물체/robot observation의 `wrist_3_link`와
역할이 다르며, 말단 제어 기준 변경으로 초기 hand가 Blocker에서 멀어지지 않게 한다.
위치는 기존 Blocker 앞 12–40 cm, 좌우±8 cm, 선반 위20–28 cm와 선반 앞 여유를 유지한다.
기준 방향 R₀의 축이 환경 축의 순열이므로 이 직사각형은 중심 p₀에 대해
`p*=p₀+R₀δp`인 손바닥 TCP-local 범위와 동등하다. 위치 범위를 임의 회전과 함께 돌리지는 않는다.
회전은 검증된 기준 관절값의 실제 손바닥 TCP 방향에 local rotation vector 각 성분±10°를 합성한다.
이는 Euler 각별 범위나 총 회전각10° 상한을 뜻하지 않는다. 기존 hand 장착−90°는 유지한다.

같은 목표 pose마다 기준 관절값과 다른3개 branch seed를 사용하며, 총80개 IK후보를 검사한다.
성공한 행은 유지하고 실패 행만 최대3회 일반 탐색, 이어2회 기준점 주변의 좁은 범위에서
재제안한다. 복구 위치는 환경 축별±(2,2,1.5)cm와 원래 영역의 교집합, 회전 범위는±2.5°다.
Robot 전용 budget은 `robot_start.reset_attempts/recovery_attempts`이며 vision budget과 분리했다.
새 공유 배치는 anchor를 찾은 뒤 독립seed의 위치·회전 목표8개를 각1회 검사해 최소4개가
유효할 때 채택한다. 이 유한 probe가 전체 영역의 유효성이나 모든 IK수렴을 보장하지는 않는다.

끝까지 실패한 행은 같은 장면에서 충돌·물리 FK를 검증한 기준 관절값·TCP·영역으로 복구한다.
그 행은 새 randomization을 포기하며, 다른 행이나 공유 배치를 변경하지 않는다.
Native 로그 `Reset/robot_start_fallback_count`, `fallback_fraction`, `fallback_total`,
`recovery_count`, `search_rounds`(뒤 네 이름에도 `robot_start_` prefix)를 확인한다.
예외 상세는 실행 로그 폴더의 `robot_start_fallbacks.jsonl`에 환경ID·seed·reset횟수·장면번호와
budget을 기록한다. Normal reset에는 파일 쓰기를 추가하지 않는다. Native runner의 평균 로그를
fallback 총횟수로 합산하지 말고 누적 `fallback_total`과 JSONL을 사용한다.
카메라 관측은 robot-base frame으로 캐시하고 각 환경의 현재 `wrist_3_link` frame으로 변환한다.
Robot은 ray casting 대상이 아니므로 새 로봇 시작 자세 때문에 재촬영하지 않는다.

공통 round 경계는 `scene_time_out` termination term으로 전체 환경을 truncate한다.
최근에 재시작한 episode도 이 경계에서 종료하며, 실패 벌점 대신 기존 timeout bootstrap을 쓴다.
`ManagerBasedRLEnv.step/_reset_idx`는 그대로 사용한다. 새 round의 배치·vision은 env_0에서
한 장면만 준비해 공유하고, 로봇 시작 자세는 모든 환경에 대해 새로 계산한다.
배치·시점 후보가 가시성/IK 검사를 통과하지 못하면 제한된 재시도는 여전히 필요하다.
공통 round 길이는 `episode_length_s`를 따른다.

Reset 준비 연산은 다음과 같이 나눈다.

- **물체 배치:** 역할·구성과 경로 후보 목록은 CPU에서 준비하고, 이동·pivot 경로의
  endpoint 및 최대 5° 간격의 보수적 sweep을 GPU에서 함께 검사해 첫 유효 후보를 선택한다.
  지원 영역·Target과의 충돌·회전 중 충돌 조건은 기존 scalar 기하 검사와 같다.
  선택한 경로의 sweep은 예약 공간으로 유지하며, 예약 다각형의 SAT 축·투영·AABB도 GPU에서 계산한다.
  환경×64개 후보의 위치·yaw 생성,
  충돌 판정·채점·부분 재배치는 환경 device에서 처리한다. 배치 상태와 누적 면적/이웃 관계를
  유지하며 16개 진행 단계마다 완료 여부만 CPU로 가져온다. 최종 전체 pairwise 기하 재검사는
  학습 경로에 없고, `validate_sample()`은 별도 검증용으로 유지한다.
- **로봇 자세:** 연속적인 TCP 후보와 IK branch를 텐서로 묶어 GPU에서 풀고, elbow 높이와
  전체 collider 충돌 조건도 배치로 검사한다. FP32 CUDA에서는 damped IK의 FK·Jacobian·6×6 solve를
  후보별 Warp kernel 안에서 최대 48회 실행하고, 원래 승인 오차보다 작은 오차에서 일찍 종료한다.
  후보 난수도 환경·episode seed와 후보 번호를 이용해 GPU에서 생성한다.
  매 episode의 목표·초기 관절값은 새로 넣는다. `robot_start.ik_batch_size=1024`는 한 번에
  계산하는 **후보 수**의 상한이다. 512로 낮추면 임시 메모리와 처리량을 함께 줄일 수 있다.
  Blocker 앞 시작 영역도 pose·half extent tensor에서 직접 계산하고, 공통 설정 검사는 batch당 한 번 한다.
- **Vision·물리 적용:** 물체-only ray casting은 제안 pose를 직접 사용하므로 IK/PhysX 쓰기 전에
  검사한다. 보수적인 물체 투영 영역 밖 ray는 mesh query를 건너뛰며, 실제 메시·해상도·가시성
  기준은 유지한다. 보이는 표면의 OBB는 배경 ray를 제외한 GPU atomic reduction으로 추정한다.
  공통 template 생성 시 PhysX link pose와 IK 모델의 일치를 검증한다. 이후 매 reset은
  충돌 검사를 통과한 새 관절값과 캐시된 물체 상태를 `[M,...]`로 묶어 asset별 한 번씩 쓴다.

배치의 XY/yaw tensor는 IK 충돌 OBB와 물리 초기 상태 생성까지 GPU에 유지한다.
독립 검증·debug용 CPU pose 사본은 실제로 조회할 때만 생성한다.
IK 결과의 관절값·TCP·시작 영역도 tensor로 전달한다. 종류/instance ID와 지지 높이,
역할·경로 후보 준비 및 후보 승인 인덱스의 CPU 처리는 남아 있다.
물리 pose 조립의 ID/slot은 평탄화한 배열로 준비하며 지지 높이는 catalog별 GPU tensor로 캐시한다.
GPU IK 난수 생성기로 바뀌어 이전 코드와 같은 seed의 구체적인 시작 자세는 달라진다.
로봇 자세의 환경/episode별 독립 난수와 동일 버전에서의 재현성은 유지한다.
2026-10-19 위치·회전/seed축 분리로 같은 seed의 구체적인 자세는 이전 버전과 달라진다.
물체 배치·역할·시점은 공통 round의 seed로 선택한다.

각 공통 layout에서 가상 시점은 최대 32회 제안하며, 최대 32개 layout까지 재시도한다.
재시도 중 Target·Blocker, 물체 종류·instance 집합과 가시율 기준은 유지한다.
부분 reset은 layout/vision 검사를 생략하고, 실패한 로봇 IK 행만 재시도한다.

장면 복제는 Isaac Lab의 기본 USD cloner(`clone_in_fabric=False`, `replicate_physics=True`)를
사용한다. 따라서 GUI Stage에서 모든 `env_*`를 확인할 수 있다. 마지막 scene spawner가
env_0의 완성된 rigid body 기하를 임시 USD에 저장하고 instance로 공유한 뒤 환경을 복제한다.
물리 body·joint·pose는 환경별로 독립적이며, 충돌체와 접촉 센서 설정을 적용한 결과를 공유한다.
기존 nested instance를 먼저 펼치고 외부 material binding을 함께 옮겨 로봇 메시·재질 누락을
방지한다. 임시 USD는 프로세스 종료 시 제거하며 원본 에셋은 수정하지 않는다.

배치의 종류·밀집도·충돌·가시성 기준은 유지하고, 승인된 장면을 공통 round 안에서 재사용한다.
새 장면의 staging은 전체 환경이 reset되는 경계에서만 수행한다. 부분 reset에서는
로봇 자세 탐색을 완료한 뒤 선택 행만 쓰며, 미선택 환경의 물리 상태·episode 이력을 유지한다.
물체 기하·F/T 보상·vision 정렬도 선택 행만 계산한다.
CPU 배치 탐색과 reset 시 metadata 전송은 남아 있다. 환경별 SciPy IK는 GPU 배치로 대체했고,
초기 관측의 렌더링 경로는 제거했다. 이 탐색·기하 조회·CPU/GPU 동기화는 매 step에 실행하지 않는다.
Ray 및 IK 임시 텐서 크기는 전체 환경 수와 별개로 제한한다. 배치/vision 생성은 공통 round당
한 장면에 대해서만 수행하며, IK 비용과 물리 상태 쓰기는 reset 환경 수에 따라 증가한다.
질량/관성은 고정 asset 값으로 캐시하므로 향후 dynamics randomization을 추가하면 이 캐시도 갱신해야 한다.

Auto-reset 전 마지막 관측은 `extras["terminal_observation"]`에 보존한다.
`ppo.py`가 이 관측으로 timeout value를 계산하며 같은 step의 물리적 성공·실패는 bootstrap에서 제외한다.
Native reset의 `terminal_env_ids`를 재사용해 종료 환경이 없는 step의 추가 critic 추론을 생략한다.
공통 round 경계 전에는 timeout이 불가능하다는 host metadata로 실패 종료 때의 불필요한
terminal critic 추론도 생략한다. 이는 `init_at_random_ep_len=False`와 native episode counter
증가·reset 규칙을 전제로 한다. 외부 코드에서 episode counter를 임의로 채우는 방식은 지원하지 않는다.
Reset 후의 새 episode 관측을 이전 episode의 value로 사용하지 않는다.

## 검증과 남은 구현

2026-10-19 R/M penalty·IK복구 적용 후 실제GPU4환경·32step·부분reset2회를 검증했다.
Native보상 수식의 최대오차는1.19e−7이며 다른환경 상태·공유template이 보존됐다.
같은장면의1,024행GPU자세sampler는1,024개모두통과했고 준비후3회29.5–29.8ms였다.
이는 전체reset·학습iteration시간이 아니다. 코드변경·검사결과·한계는
[학습 기록8절](../../docs/TRAINING_LOG.md)에, 실제결과는
[verification.json](../../outputs/contact_manipulation/reward_ik_20261019/verification.json)에 둔다.

아래는 이전변경의 검증이력이다. Motion 로그 분리 후 CPU 검사는 **205개 통과·14개 skip·142 subtest 통과**다.
실제 GPU4환경·32 policy step과 부분 reset2회에서 native8개 reward term 및 Motion4개
로그를 확인했다. 분리 전 합산 수식과 전체 reward의 최대 차이는3.58e−7이다.
자료는 `outputs/contact_manipulation/motion_split_20261017/`에 있다.

앞선 Motion·weight 변경 후에는
실제 GPU4환경·48 policy step과 선택2환경의 부분 reset에서 새 수식, F/T 비용 및 native
RewardManager의 weight·dt 적용을 확인했다. Motion 독립 계산과의 최대 차이는5.96e−8이다.
자료는 `outputs/contact_manipulation/motion_update_20261016/`에 있으며, 변경 후 학습 성능은
아직 평가하지 않았다.

앞선 Exp 보상 GPU 4환경·64 policy step에서는
5개 native reward term, 공식 joint_acc_l2 기반 RMS, score 범위와 dt 처리, 자동 reset 6회를
확인했다. 현재 Motion의 정의와는 다른 이전 검사이며 자료는
`outputs/contact_manipulation/reward_exp_20261015/`에 있다.

전체 최적화 후 GPU 회귀 검사는 **205개 통과·7개 skip**이다. 실제4환경·410 step에서
기존 F/T 보상 결과와의 일치, 정상 경로의 반복 COM 조회 제거, 부분 reset의 다른 행 보존,
400 step 장면 교체를 확인했다. 초기 compile을 제외한1,024환경30 iteration 학습도 완료했다.

앞선 구조 정리에서는 세 observation term의 호출 순서와 부분 reset 후에도 244D 값·이력·
terminal snapshot이 보존되는지 검증했다. GPU 4환경·410 policy step에서 기존 packing과의 정확한 일치 및 부분/common reset을,
1,024환경·4 iteration에서 구조 변경 전 native checkpoint의 이어 학습을 확인했다.
자료는 `outputs/contact_manipulation/min_structure_20261015/`에 있다.

Reset cache 변경은 실제 GPU IK84개 비교에서 첫 유효 후보와 성공/실패가 정확히 같았고
선택 자세는1e-5 허용 오차로 일치했다. 별도4환경·410 step에서 초기 vision packing 일치,
부분 reset의 다른 행 보존,400 step 장면 교체 시 cache 갱신을 확인했다.

앞선 native/min action 전환 때에는 Isaac Sim 내 controller 검사 3개,
1,024환경·16 iteration 학습과 새 action-contract checkpoint 저장을 확인했다.
그 초기 학습 평균은 최초 iteration 제외 약 3.81초였다.
아래의 이전 성능·checkpoint 기록은 당시 제어 구성에서 측정한 개발 이력이다.

MDP 테스트는 `tests/test_mdp.py`, 모델·PPO 테스트는 `tests/test_learning.py`에 모았다.
Isaac Lab Python 환경에서 다음과 같이 실행한다.

```bash
python -m unittest discover -s src/contact_manipulation/tests -v
```

1,024개 환경 배치와 개별 계산의 일치, CPU scalar 추출 없이 동작하는 step 경로,
3D OBB footprint의 독립 기하 oracle 비교를 포함한다. 이는 **MDP 텐서 계산 검증**이며,
1,024개 실제 물리 환경의 생성·학습 메모리 검증은 아니다.

**CPU 테스트 159개 통과, 별도 native USD 검사 2개 통과.** 카메라를 끈 실제
1,024개 환경에서 기본 rollout 32·PPO 4 epochs로 native 학습 1 iteration도 완료했다.
CPU 테스트에는 배치 sampler와 독립 scalar oracle의 일치, 배치 FK/Jacobian·IK·충돌 검사,
mesh ray의 가림·고립 분모·좌표변환·배치 일치·실패한 행 재시도를 포함한다. 실제 Isaac Lab 설정
import, Hydra override, RSL-RL 설정 migration을 거쳐 native runner의 학습·TensorBoard·
checkpoint 재개를 확인했다.
GNN dense 식의 출력·gradient 일치, attention gradient, padding·순열 불변성,
불균등 microbatch의 gradient·Adam 갱신 일치, timeout과 물리 종료 우선순위를 포함한다.

RTX 3090에서 다음 텐서 검증도 통과했다. 아래 메모리는 PyTorch allocated peak이며,
**물리·렌더링·센서·Isaac Sim 메모리는 포함하지 않는다.**

| 검증 | 결과 |
|---|---|
| MDP 갱신·관측, 1,024 / 2,048 × 40개 물체 | CPU/GPU 일치, batch당 약 11.7 / 20.4 ms |
| Native PPO, 2,048 × rollout 32, epochs 4, minibatches 4 | 전체 update 약 35.3 s, peak 약 560 MiB, 유한한 파라미터 갱신 |
| 같은 257개 입력, 전체 계산 vs chunk 128+128+1 | action/value/log-prob 최대 절대 차이 < 7e−7 |

축소 PPO 비교(rollout 4, epoch 1)에서 2,048개 batch의 chunk 32 / 64 / 128은
약 5.61 / 1.80 / 1.45 s, peak 151 / 243 / 398 MiB였다. 이는 초기 chunk 128 선택 당시의
측정이다. 이후 메시 공유로 장면 메모리가 줄어 추론 512·학습 1,024로 기본값을 늘렸다.

2026-10-13 RTX 3090에서 raw current/previous 모델의 동일한 1,024환경×rollout 32,
PPO 4 epochs·4 minibatches를 합성 native runner로 비교했다. 아래 처리량은 ReLU 전환 전
SiLU 모델 기준이며, 합성 runner 수치는 **물리·reset을 제외**한 값이다.
GNN 선형 연산 재배치 전 chunk 128은 18.34초, 재배치 후 chunk 1,024는 3.97초였다.
후자의 PyTorch allocated peak는 약 1,893 MiB다. 추론 chunk만 512로 낮춰 peak를 줄이며,
sample 수·epoch·optimizer step 수는 유지한다. 이는 PPO minibatch 자체를 늘리는 변경이 아니다.

같은 변경을 실제 1,024환경에서 3 iteration 검증했다. 추론 512·학습 microbatch 1,024에서
learning time은 3.40 / 3.32 / 3.33초, collection time은 23.42 / 12.78 / 13.48초였다.
첫 iteration 26.82초 이후 16.10 / 16.81초로, 기존 사용자 로그의 약 30초보다 줄었다.
Raw current/previous 모델의 기존 checkpoint는 그대로 로드되며 모델 구조·PPO 설정은 유지한다.
3 iteration의 reset 준비 합계는 26.37초였고, 그 안에서 layout 탐색 8.87초,
IK 3.79초, 초기 vision 13.20초가 측정됐다. 이는 임시 계측 스크립트의 host wall time이며
GPU kernel 단독 시간은 아니다. 후반의 대규모 timeout reset과 장시간 평균은 별도 측정이 필요하다.

1,024환경×rollout 32에서 90,000 iteration은 **2,949,120,000 transition**이다.
30초/iteration이면 31.25일, 16–17초면 약 16.7–17.7일이다. Iteration 수를 다른 실험과
비교할 때 환경 수·rollout 길이·총 transition도 함께 비교한다. 메모리 여유가 작은 경우에는
`agent.algorithm.max_microbatch_size=512`로 낮출 수 있다.

2026-10-13, 실제 USD에서 측정한 40개 물체의 geometry로 reset 준비 연산을 별도 측정했다.
아래 CPU 배치는 독립적인 seed를 사용하며, 생성된 모든 결과를 기존 scalar 검사로 다시 확인했다.
JIT 준비 시간과 추가 oracle 검증 시간은 생성 시간에 포함하지 않았다.

| Reset 준비 연산만 측정 | 결과 |
|---|---|
| CPU 물체 배치 128개 | 기존 31.262 s → 4.791 s, **6.52배 개선** |
| CPU 물체 배치 1,024개 | **31.224 s**, 평균 30.49 ms, p95 87.17 ms |
| GPU IK·로봇 초기 자세 1,024개 | 약 **1.23 s**, PyTorch peak 약 **612 MiB** |

1,024개 물체 배치는 모두 서로 달랐고 scalar 검사에 통과했다. 최소 전체 점유율 35.001%,
최소 지역 점유율 14.756%였으며, pivot 392개·회전 후 이동 404개·이동 228개의 경로를 확보했다.
용량상 불가능한 종류 조합은 총 215회 다시 골랐으며, 밀집도 기준은 유지했다. GPU IK 측정은
준비된 장면에 대한 계산만 포함하고 **물체 배치 탐색·PhysX·카메라 렌더링을 포함하지 않는다.**
위 시간과 메모리는 **1,024개 실제 물리 환경의 생성·reset·학습을 검증한 수치가 아니다.**

2026-10-13 RTX 3090, 실제 32개 환경에서 카메라 방식과 기하 관측 방식을 비교했다.
각 실행은 독립 장면을 새로 생성하며, 아래 값은 1회 측정이다. JIT·IK graph 초기화와
후보 재시도도 reset 시간에 포함되므로 순수 ray 연산 비교는 아니다.

| 실제 환경 32개 | Tiled camera | 기하 관측 |
|---|---:|---:|
| Scene 생성 | 44.77 s | 20.09 s |
| 초기 reset | 21.86 s | 9.34 s |
| 16개 환경 부분 reset | 9.73 s | 2.35 s |
| GPU 전체 사용량 (`nvidia-smi`, 다른 프로세스 포함) | 8,363 MiB | 3,990 MiB |
| 프로세스 최고 RSS | 39,223 MiB | 20,309 MiB |
| 초기 / 부분 reset 렌더 호출 | 260 / 190 | **0 / 0** |

기하 관측 실행에는 `/World` 아래 camera prim이 없었다. 32개 각각의 seed·초기 관측,
4 control step, 부분 reset 시 미선택 환경의 물리 상태·관측·physics 시간 보존을 확인했다.
초기 최소 가시율은 Target 40.27%, Blocker 97.69%였다. 위 수치는 아래의 메시 공유 적용 전 결과다.

사용자의 1,024개 실행 종료는 커널 OOM 기록에서 **host RAM 부족**으로 확인했다.
종료된 Python의 anonymous RSS는 약 38.37 GiB였으며 초기 reset 전 장면 복제 단계였다.
메시 공유 후 같은 장면 생성·reset 검증을 다시 수행했다.

| 메시 공유 후 실제 환경 | 32개 | 128개 |
|---|---:|---:|
| Scene 생성 | 12.50 s | 14.78 s |
| 초기 reset | 9.26 s | 25.97 s |
| 절반 환경 부분 reset | 2.30 s | 9.60 s |
| 프로세스 최고 RSS | 5,002 MiB | 5,342 MiB |
| GPU 전체 사용량 (`nvidia-smi`) | 4,009 MiB | 4,340 MiB |

32개 환경의 RAM 최고 사용량은 공유 전 20,309 MiB에서 5,002 MiB로 감소했다.
동일 seed의 가시율 결과가 공유 전과 일치했고, 두 규모에서 4 control step 및 부분 reset 시
미선택 환경 보존을 통과했다. 별도 4개 환경 렌더에서는 UR5e·hand가 모두 표시되며,
Stage의 네 환경 각각에서 로봇 메시 78개와 총 articulation 4개를 확인했다.
Native USD 회귀 검사 2개는 nested instance의 메시·재질·충돌체 보존과 기하 공유,
복제 body pose의 독립성 및 joint를 포함한 subtree 제외를 확인한다.

**1,024개 실제 환경 headless 학습도 1 iteration 완료했다.** 기본 rollout 32 step,
4 epochs, 4 minibatches를 사용했고 초기 독립 reset, 학습 중 개별 auto-reset,
32,768 transition 수집, PPO 갱신과 TensorBoard·checkpoint 저장까지 exit code 0이었다.
프로세스 최고 RSS는 **8,658 MiB(8.45 GiB)**, 5초 간격으로 관측한 GPU 전체 사용량의
최댓값은 **7,732 MiB**였다. Scene 생성 13.78초, 학습 iteration 45.21초
(수집 23.79초 + 갱신 21.42초), 초기화·초기 reset·종료를 포함한 전체 실행은 3분 56초였다.
초기 reset은 최대 17번째 layout 제안에서 모든 환경이 승인됐다. 이 결과는 1회 실행이며
장시간 학습 안정성이나 여러 seed의 처리량을 검증한 것은 아니다.

이후 current/previous 직접 concat 모델과 hand–arm·전완 방향 검사를 적용한 상태에서도
같은 기본 설정으로 **1,024개 환경 학습 1 iteration을 다시 완료**했다. 최고 RSS는
8,228 MiB(8.04 GiB), iteration 41.16초, 초기화 포함 전체 실행 3분 41초였다.
별도 128개 독립 장면의 IK 비교에서 기존 승인 127개 중 전완 방향 위반 26개,
hand–arm OBB 여유 위반 10개가 발견됐다(중복 가능). 수정 후 승인된 125개는
모두 두 조건을 통과했고, 나머지 3개는 배치 재시도를 위해 거절됐다.

CPU sampler의 4·8개 스레드 실행은 동일한 작업에서 단일 스레드보다 느려 적용하지 않았다.
주요 잔여 비용은 Python의 회전 경로 convex hull 계산과 후보 생성이며, 작업별 실제 비용을
측정하면서 개선해야 한다. 여러 seed에서의 처리량과 장시간 학습 안정성은 별도 검증 대상이다.
별도 play/OBB 시각화 도구와 모델 export도 남아 있다.

## 실행

2026-10-14 센서·IK·reset·step 추가 최적화 후 RTX 3090에서 1,024개 환경을 측정했다.
동일 설정의 첫 8 iteration 평균(native collection+learning)은 **11.15→9.07초**, 약 **19% 감소**다.
해당 16 iteration 실행은 정상 종료했으며 전체 평균은 **9.53초**였다.
이후 Python 데이터 처리를 배열로 정리한 변경은 동일 1024개 배치 생성에서 **1.72→1.57초**였다.
이 변경의 native 8 iteration 평균은 **9.18초**로, 전체 학습 시간의 추가 개선은 확인되지 않았다.
초기 환경 생성은 제외한 수치이며, **1–2초/iteration 목표에는 아직 도달하지 못했다.**
위 계측 당시에는 rollout 32·PPO 4 epochs×4 minibatches·물리 120 Hz·정책 30 Hz를 사용했다.
현재 기본값은 Sweeping-Policy 예제와 같은 물리 100 Hz·정책 50 Hz(`dt=0.01`, `decimation=2`)다.
Rollout 32는 환경별 0.64초에 해당하며, 위 성능 수치는 새 주기로 측정한 결과가 아니다.
GPU 난수 생성·수치 연산 순서 변경으로 reset/학습 궤적이 달라질 수 있어 장기 처리량 보장은 아니다.
주요 잔여 비용은 물리 step과 CPU reset 준비·큐 처리다. 자세한 계측과 검증은
[개발 맥락](../../docs/CONTEXT.md)에 기록했다.

Isaac Lab이 설치된 Python 환경을 사용한다. 저장소 루트에서 별도로 설치할 수 있다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install --no-deps -e ./src/contact_manipulation
```

저장소의 robot URDF/mesh와 기본 Omniverse 서버의 선반·물체 에셋을 사용한다.
다른 에셋 위치는 `CONTACT_MANIPULATION_NUCLEUS_ROOT`, `CONTACT_MANIPULATION_SHELF_USD_PATH`,
`CONTACT_MANIPULATION_UR5E_USD_PATH`, `CONTACT_MANIPULATION_HAND_URDF_PATH`로 지정한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 4 --headless --run_name gaussian
```

1,024개 병렬 환경 학습은 다음과 같이 실행한다. 에셋 복제·JIT·초기 IK 준비로
첫 학습 로그까지 수 분이 걸릴 수 있다. 한 iteration만 점검하려면 `--max_iterations 1`을 붙인다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 1024 --headless --run_name gaussian
```

기하 관측과 IK의 임시 메모리는 native Hydra 설정으로 조절할 수 있다. 다음 예시는
동시에 32개 요청의 시점 후보를 각각 4개씩 처리하고 한 번의 IK 후보 수를 512로 제한한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 4 --headless \
  env.initial_vision.ray_batch_size=32 env.initial_vision.ray_candidates_per_env=4 \
  env.robot_start.ik_batch_size=512
```

Launcher는 기본적으로 이 저장소의 `IsaacLab/`을 사용한다.
다른 checkout을 사용하면 `ISAACLAB_PATH`를 지정한다.

저장된 정책을 GUI에서 재생하려면 저장소 루트에서 다음과 같이 실행한다.
아래는 위 명령으로 학습한 Gaussian run의 마지막 checkpoint를 선택한다.
`--checkpoint`로 특정 Gaussian checkpoint 파일을 직접 지정할 수도 있다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/play.py \
  --load_run '.*_gaussian' \
  --num_envs 4 --real-time
```

`--checkpoint`를 생략하면 native `get_checkpoint_path()`로 최신 run의 마지막 checkpoint를
선택한다. 이전 tanh run은 분포 계약 검사에서 거절되므로 새 Gaussian run을 지정한다. 기본 환경 수는 1이고,
중간 episode 종료 시 공유 배치를 복원하고 로봇 자세를 다시 샘플링한다.
공통 round 경계에서는 학습과 동일하게 새 배치·초기 관측으로 교체한다.
`--real-time`은 실행이 빠를 때 환경의 정책 주기(기본 50 Hz)에 맞춰 기다리며 느린 reset을 가속하지 않는다.

Play는 `RslRlVecEnvWrapper → OnPolicyRunner.load() → get_inference_policy()` 흐름을
사용하며 탐색 noise 없이 정책의 `mean` action을 적용한다. 모델·F/T·action/output-distribution metadata와
state dict를 엄격하게 검사하고 optimizer는 불러오지 않는다. 학습 업데이트나 자동 모델
export는 수행하지 않는다. 모델 구조와 action 계약이 모두 같은 checkpoint를 재생할 수 있으며,
현재 환경 설정을 사용하므로 과거의 배치·관측 모델까지 복원하는 기능은 아니다.
필요한 설정 변경은 학습과 같은 Hydra `env.*`, `agent.*` 인자로 전달한다.

GUI 없이 지정한 step까지만 확인하려면 `--headless --max_steps 260`을 사용한다.
영상은 다음과 같이 저장하며 checkpoint 옆 `videos/play/`에 생성된다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/play.py \
  --load_run '.*_gaussian' \
  --num_envs 1 --headless --video --video_length 300
```

영상 저장에만 렌더링을 활성화하며 정책의 초기 vision은 계속 GPU mesh ray casting을 사용한다.
`--max_steps`와 `--video_length`를 함께 지정하면 더 짧은 길이에서 종료한다.
