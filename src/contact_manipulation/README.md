# Contact Manipulation

Sweeping-Policy의 **`min` branch 구조**를 기준으로 구성한 Isaac Lab 연구 프로젝트다.
**장면·센서 생성, episode reset, MDP, GNN·attention 정책과 native PPO를 연결한 프로젝트다.**
기존 `src/contact_rearrangement`와 별도 패키지이고, 해당 패키지를 import하지 않는다.

**현재 배치 생성은 15–25개 물체의 통합 clutter 규칙이다.** 물리 최대 개수는25개이며,
공개 관측은 Target·Blocker를 포함한10개 slot과244D를 유지한다. 기존 `parameterized`
경로의90 ordered pair는10종에서 서로 다른 Target·Blocker를 고르는 `10×9` 종류 조합이며,
미리 저장한90개 완성 장면을 의미하지 않는다. 기존의 두-row 주변 배치가 역할 물체와 분리되어
보인다는 문제를 수정하며, 이전10물체·364장면·1,024환경 검증을 새 규칙의 검증으로 사용하지 않는다.
Blocker의 빈 재배치 공간은 확보하되 이동·pivot 전체 경로를 비워 두지는 않는다.
이전 최대40개 설정의 `clutter`는98개 standalone 장면과1,024환경 reset·10step을 통과했다.
해당 검사 범위는20–38물체이며 현재 생성 범위는15–25개로 낮췄다. 정책 관측은244D를 유지한다. 장기 학습·정규 iteration 속도는
이번 짧은 실행 검사의 범위에 포함하지 않는다.
현재 경로와 이전 구현은
아래 [병렬 계산과 reset 연결](#병렬-계산과-reset-연결),
상세 규칙은 [SPAWN_RULES.md](../../docs/SPAWN_RULES.md)에서 구분한다.

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
    │           ├── model.py         # 독립 Actor/Critic·mask·GNN·attention·history·head
    │           └── ppo.py           # 기존 checkpoint play 호환·native loader 연결
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
각각 `ObservationTermCfg`에 선언한다. Native ObservationManager가 총244D로 연결한다.
Critic은 별도 `critic` 그룹에서 같은10개 슬롯의 현재 실제 물체 기하100D와 동일한
현재·직전144D를 받는다. 초기 인식 ID·순서·padding은 reset에서 고정한다.
새 규약에서는 물리 활성 개수15–25와 관측 용량10을 분리한다. 인식한 물체가10개를 넘으면
Target·Blocker를 slot0/1에 두고, 나머지는 관측된 중심에서 두 역할의 관측 중심까지 거리의
최솟값이 작은 순서로8개를 선택한다. 보이지 않은 물체를 정답 pose로 채우지 않으며,
관측 물체가 적으면 나머지는 padding한다. 물리 개수를 늘려도 모델 입력 폭은 늘리지 않는다.
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
| 미사용 물체 | Native `UsdPhysics.RigidBodyAPI`의 kinematic 설정 | 격리된 주차 위치에서 kinematic 유지, 활성 물체만 dynamic으로 전환하며 GPU actor·view 유지 |
| 바닥·환경 충돌 분리 | Native `GroundPlaneCfg`·cloner의 PhysX 환경 ID | 기존 전역 바닥·4bit bounds encoding 유지 |
| PPO | 공식 `PPO` 전체와 Actor/Critic의 `MLPModel`, storage·GAE·logger·checkpoint | 각각의 특징 추출에 독립 GNN/attention 연결 |
| 초기 vision | `convert_to_warp_mesh`, Warp mesh query | 실제 메시 가시성의 연속 구간 사전 계산, 정해진 pose의 object-only 관측과 표면 OBB |

직접 구현을 유지한 이유:

- 기본 reset의 균등 pose/joint sampling은 Target·Blocker 조건, 밀집도, pivot 재배치 공간,
  elbow/자기충돌 조건을 정의하지 않는다. `EventTermCfg` 안에서 연구 sampler를 사용한다.
- `DifferentialIKController`는 주어진 pose/Jacobian에서 관절 증분을 계산하며 후보의 FK나
  충돌 없는 초기 자세 탐색을 제공하지 않는다. 현재 GPU FK/IK는 물리 scene을 바꾸지 않고
  여러 후보를 함께 풀며 수십 번의 DLS를 fused kernel에서 처리하므로 유지한다.
- `MultiMeshRayCasterCamera`와 `raycast_dynamic_meshes`는 동적 mesh를 지원한다.
  따라서 "native raycast는 동적 물체를 지원하지 않는다"는 이유로 custom을 유지하는 것이 아니다.
  새 연속 영역 경로는 실제 메시의 무가림 실루엣과 가림 구간을 시작 전에 계산하고, reset에서는
  **정해진 pose**를 관측한다. Projected bounds로 ray query를 줄이고 depth/label을 ray당 한 thread에서 결정한다.
  Native multi-mesh kernel의 depth/label 경쟁 쓰기 관련 FIXME도 있어 동일 대체로 취급하지 않는다.
- Native observation history는 reset 후 첫 sample을 과거 슬롯에도 반복한다. 우리 규약은
  reset 시 previous=0, current/previous 순서와 reset 직전 terminal 관측 보존이므로
  `TaskState`의 두 sample 저장을 유지한다.
- F/T 원시 반력은 public `ArticulationData.body_incoming_joint_wrench_b`에서 읽는다.
  Quaternion 부호 정규화와 물리 FK 검사의 회전 변환도 `isaaclab.utils.math`를 사용한다.
  SI-200-8 양자화·범위, clearance·지지 기하 등은 연구 수식이다.
  F/T 측정값에서 hand 링크의 자중·관성 하중은 제거하지 않는다.
- 출력 분포는 RSL-RL의 `GaussianDistribution`을 사용한다. Sampling·log-probability·
  analytic entropy·KL은 native 구현에 맡긴다. Actor와 Critic은 각각 공식 `MLPModel`을
  확장하고, GNN·attention·MLP 전체의 가중치를 별도로 학습한다.

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
학습 설정은 `examples/Sweeping-Policy-DRL-Sim-Train`의 random 태스크가 등록한
`rsl_rl_ppo_cfg_02.py`를 따른다. 자체 PPO subclass·microbatch 업데이트·timeout bootstrap은
사용하지 않는다. `init_at_random_ep_len=True`도 표준 학습 스크립트와 같다.
`commands`와 `curriculum`은 Isaac Lab 기본값인 `None`으로 두었다.

`min`의 실행 스크립트는 task를 등록하고 `runpy`로 Isaac Lab 스크립트를 호출한다.
우리 스크립트는 같은 실행 순서를 직접 사용하며 다음 차이만 유지한다.

- Train: 공식 runner가 초기 episode counter를 무작위화한다. 실제 초기 vision·이력은 reset 시점에 생성된다.
- Play: 현재 graph 모델에서 지원하지 않는 JIT/ONNX export를 실행하지 않고 native inference로 재생한다.
- 모델 연결: `NativeContactActor`와 `NativeContactCritic`의 특징 추출에 각각 GNN·attention을
  연결하고, head·분포·PPO 인터페이스는 공식 `MLPModel`에 맡긴다.

Reset은 `EventManager`가 `reset_scene → reset_task` 순서로 호출한다. 첫 함수가 물체·로봇의
초기 상태와 vision을 준비하고, 두 번째가 초기 snapshot·이력·시계를 초기화한다.
이후 reward·termination·observation은 한 control step당 한 번 갱신되는 `TaskState`를 공유한다.
이 부분은 initial vision 고정, previous=0 reset 규칙, 센서 이력과 terminal observation을
일치시키기 위한 연구 구현이다. 별도 `env.step()`이나 `_reset_idx()` override는 없다.

## 모델과 PPO

```text
Actor / Critic 각각 별도 인스턴스로 실행 (가중치 공유 없음):

Actor: 초기 vision / Critic: 최초 인식 ID들의 현재 GT [N,10,10]
                          └─ 2-layer GNN (64D)
                             ├─ Target / Blocker embedding
                             ├─ Target / Blocker 조건부 attention → scene 320D
                             └─ valid object mean pooling
현재·직전 sample [N,2,72] ── 결측 masking → flatten 144D (별도 encoder 없음)
                                   ↓
                      concat 464 → 공식 MLPModel head 256 → 128
                                   ├─ Actor: 18 → 공식 Gaussian distribution
                                   └─ Critic: value 1D (분포 없음)
```

GNN은 관측된 물체 사이의 위치·크기 차이를 사용해 두 번 message passing한다.
Target/Blocker embedding에서 두 attention query를 만들어 다른 물체의 정보를 읽는다.
Attention은 `SceneGraphEncoder.forward()`의 query/key/value·masked softmax에 구현되어 있다.
GNN 및 최종 MLP의 활성화는 **ReLU**다. `ContactManipulationEncoder`는 scene 특징과
현재·직전 정보를 합치는 기능만 담당한다. 두 모델은 공식 `MLPModel`을 상속해
`get_latent()`를 확장하며, 최종 MLP와 forward·분포·확률·entropy·KL·
reset 등 공통 인터페이스를 상속한다. 공식 `CNNModel`과 같은 확장 방식이다.
현재·직전 값은 순서대로 그대로 연결하여 최종 MLP에서 처리한다. 별도 temporal encoder는
확정된 설계가 아니므로 이번 baseline에서 제거했다.

Actor는 `policy`, Critic은 `critic`이라는 별도244D 관측 그룹을 받는다. Critic의 물체 목록은
reset에서 Actor의 초기 인식 ID·슬롯 순서·padding과 정확히 맞춰 고정한다. 이후 동일 물체의
현재 실제 중심·회전·전체 OBB 크기를 제공하며, 처음 보이지 않았던 물체는 추가하지 않는다.
가려지거나 선반 밖으로 나가도 ID는 유지한다. 추가 ray casting이나 최근접 재선택은 없다.
Actor의 초기 추정 OBB는 유지하므로 초기에도 Critic의 실제 OBB와 값이 다를 수 있다.
양쪽 물체 기하는 현재 UR5e 말단 EEF frame으로 표현하고, 로봇·센서·행동144D는 동일하다.
Critic head는 `464→256→128→1`이며 Gaussian 분포는 없다. 두 모델 모두 ReLU를 사용한다.
공통 Python 클래스로 구조를 정의하지만, encoder·attention·
MLP·정규화 통계·compile cache는 각각 별도 인스턴스다. Critic의 value loss는 Actor에 직접
역전파되지 않으며, PPO advantage를 통한 영향은 남는다.

시작 로그의 모듈 목록은 PyTorch **등록 순서**라 `mlp`가 `encoder`보다 먼저 표시될 수 있다.
실제 native forward는 `get_latent()`에서 scene encoder와 concat을 끝낸 뒤 MLP를 호출한다.
로그의 `forward: scene encoder ... -> concat(current, previous) -> mlp`가 실행 흐름이다.

현재 baseline은 공식 PPO의 checkpoint 형식을 사용하고 각 모델의 계약을
native `state_dict`의 extra state에 기록한다. 새 graph Critic checkpoint의 resume는 native 방식이다.
현재 Critic은 `native_graph_critic_v2`·`tracked_initial_objects_v1` 규약을 저장한다.
이전 초기 vision Critic은 입력 폭이 같아도 의미가 달라 전체 resume를 거절하고 play만 지원한다.
기존 raw MLP Critic checkpoint는 Critic·optimizer 구조가 달라 전체 resume를 거절하고,
play에서는 저장된 이전 Critic 구성을 복원한다. Actor의 구조·가중치 규약은 그대로 유지한다.
기존 독립 Critic·2-sample GNN의 log-std checkpoint는 play에서 저장된 Actor 설정을 복원하고
필요한 weight key만 변환한다. 새 scalar-std baseline으로 이전 optimizer를 이어 학습하지 않는다.

2026-10-21 구조 변경 전에 저장한 공유 actor–critic checkpoint는 새 독립 critic 및 optimizer 구조와 다르다.
Architecture metadata를 검사하여 전체 resume를 거절하므로 **새 학습을 시작해야 한다**.
이전 공유 모델을 현재 play에서 호환 로드하는 기능도 구현하지 않았다.
이전 temporal encoder 모델의 checkpoint는 구조가 달라 resume할 수 없다.
Activation도 checkpoint metadata에 기록하므로 이전 SiLU checkpoint의 그대로 resume는 거절한다.

Actor는 학습 시 `a=μ+σε`, 일반 재생 시 `a=μ`를 사용한다. Tanh와 별도 action clipping은 없다.
예제와 동일한 RSL-RL `GaussianDistribution(std_type="scalar", init_std=1.0)`를 사용한다.
Sampling·log-probability·entropy·KL과 MLP의 기본 초기화는 공식 구현에 맡긴다.
기존 mean-head gain.01/bias0 재초기화는 제거했다. Scale1.0의 pose action에는 ±1 상한이 없다.
현재/직전 observation의 action 값도 ±1로 자르지 않고 실제 raw 값을 보존한다.
Hand 관절 한계와 OSC Kp·actuator 토크 제한은 각 native 클래스에서 적용한다.

출력 분포는 Actor checkpoint의 `actor_model.distribution_cfg`에 기록한다. 이전 tanh checkpoint는
새 Gaussian으로 resume/play하지 않도록 weights·optimizer 로드 전에 명시적으로 거절한다.
변경된 분포는 새 학습으로 평가해야 한다.

`build_policy_masks()`가 observation의 extent·quaternion·센서 유효값과 고정 슬롯에서
mask를 만든 다음 모델에 별도로 전달한다. Mask·role ID를 observation이나 학습 feature에
붙이지 않는다. 물체 padding, reset 시 직전 step 이력의 결측, tactile 결측은 해당 mask로 제외한다.
Mask는 항상 raw 관측에서 판정한다. 각 모델의 `obs_normalization=True`를 선택하면 공식 running
normalizer를 robot/sensor/action history 144D에만 적용하고 결측 mask를 다시 적용한다.
Scene 기하와 role/padding 판정은 raw 값을 유지한다. 현재 기본값은 Actor/Critic 모두
`obs_normalization=False`이며, 예제의 값을 따른다.

설정은 `config/ur5e/agents/rsl_rl_ppo_cfg.py` 한 곳에서 조합한다.

| 항목 | 기본값 |
|---|---|
| Seed | 42, native runner 기본값 |
| Rollout / PPO epochs / minibatches | 36 step / 8 / 4 |
| Learning rate / schedule / desired KL | 1e−3 / adaptive / 0.02 |
| γ / GAE λ / PPO clip | 0.98 / 0.95 / 0.2 |
| Value loss / entropy coefficient | 1.0 / 0.005 |
| Max gradient norm / 초기 Gaussian std | 1.0 / 1.0, scalar 직접 학습 |
| Actor/Critic 각각의 feature chunk | 최대 8,192 sample; PPO gradient accumulation 없음 |
| Actor/Critic hidden layers / activation / observation normalization | 각각 [256,128] / ReLU / False |
| 학습 iteration / checkpoint 주기 | 90,000 / 50 |
| Physics / policy 주파수 / timeout | 100 Hz / 100 Hz / 8 s |
| 기본 환경 수 / 환경 간격 | 1 / 2.5 m |

환경 수는 CLI로 늘릴 수 있다. 현재 1,024환경 × rollout36 ÷ minibatches4 = 9,216이며,
batch 크기와 관계없이 공식 `PPO.update()`를 실행한다. 각 모델의 feature chunking은 특징 추출의
메모리 제한이며 PPO의 minibatch·loss·optimizer step을 바꾸지 않는다.
하나의 native optimizer에 서로 겹치지 않는 두 네트워크의 파라미터를 등록하며,
rollout과 update 모두 Actor의 action과 독립 Critic의 value를 별도로 계산한다.

RSL-RL 5.x의 모델 API를 사용한다. 콘솔·TensorBoard 출력은 native logger를 그대로 사용하고
native value/surrogate/entropy loss를 기록한다. 현재 검증 범위는 single-GPU이며,
graph Actor의 JIT/ONNX export는 구현하지 않았다.

## MDP

기준은 [기존 연구 설계](../../docs/CONTEXT.md)다. MDP 구현은 기존 네 개 모듈과
`sensors.py`에 모았다. 제어는 UR5e env cfg의 공식 ActionCfg에 설정하며,
공통 상태·배치 sampler·reset을 별도 파일로 분리하지 않았다.

| 영역 | 구현 |
|---|---|
| Observation | Actor·Critic 각각 물체 기하 10×10 + 현재·직전 sample 2×72 = **244D** |
| 물체 정보 | 현재 `wrist_3_link` 기준. Actor는 초기 vision 추정값 고정, Critic은 같은 초기 인식 ID들의 현재 실제 OBB 갱신 |
| 슬롯 | Target 0, Blocker 1, 나머지 관측 물체와 zero padding. Role ID·mask 채널 없음 |
| 72D sample | 12관절 q/dq 24 + base에서 표현한 `wrist_3_link` pose 7 + sensor 기준 F/T 6 + tactile 17 + 직전 적용 action 18 |
| F/T | Sensor frame의 반력 → SI-200-8 축별 clipping·양자화 → full scale 정규화. 자중·관성 하중 포함 |
| Tactile | 마지막 physics sample의 normal force norm > 0.01 N이면 1, 미접촉 0, 결측 −1 |
| Action | 현재 EEF 축 TCP 증분 6 + EEF 축 stiffness 6 + hand 독립 관절 위치 6 = **18D** |

제어는 [UR5e 환경 cfg](contact_manipulation/config/ur5e/ur5e_env_cfg.py)의 `__post_init__`에서
공식 `OperationalSpaceControllerActionCfg`와 `JointPositionToLimitsActionCfg`로 설정한다.
공통 `ActionsCfg`는 두 term을 `MISSING`으로 선언한다. 별도 custom action/controller 클래스는 없다.
`ActionManager.process_action → native process_actions/set_command → physics substep의
native apply_actions/compute → scene.write_data_to_sim` 순서로 실행한다.

OSC 구조·gain 설정은 원격 `min` branch의
[`UR5eSweepKpOSCRelativeObsEnvCfg`](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py)를 따른다.
2026-10-20 사용자 교정에 따라 제어 body는 UR5e의 실제 말단 rigid body인
`wrist_3_link`로 정한다. `body_offset`은 사용하지 않으며 `tool0` 가상 frame과 구분한다.
Action 좌표계와 물체 observation의 기준도 **현재 `wrist_3_link` frame**으로 통일한다.
`task_frame_rel_path="Robot/wrist_3_link"`와 scene `ee_frame`의 같은 링크 target을 지정해
공식 ActionCfg·FrameTransformer가 좌표 변환을 수행한다. 확인한 min 커밋 `baa3740`은
task frame을 지정하지 않아 base 축 증분을 사용한다는 점에서 다르다.

| 입력/설정 | 현재 native 동작 |
|---|---|
| Arm 앞 6D | 현재 EEF 축 `pose_rel`; position/orientation scale 모두 0.5 (m, rad) |
| Arm 뒤 6D | EEF 축별 `Kp = clip(100 × action, 50, 200)`; 병진·회전 동일 규칙 |
| 현재 Gaussian 정책의 Kp 범위 | raw 출력에 ±1 상한이 없어 native 범위 50–200을 사용 가능 |
| OSC | Full inertia decoupling, damping ratio 1, gravity compensation=True, nullspace 없음 |
| Hand 6D | `0.5 × raw action`을 native에서 [−1,1]로 clamp한 뒤 live soft limit 범위에 매핑 |

사용자가 직접 조정한 현재 scale0.5에서는 위치 action0.01이5mm, 회전 action0.01이0.005rad다.
Hand scale도0.5이며, Hand를 속도 종료에서 제외하는 수정에서 이 세 scale은 변경하지 않았다.
정책 step 시작 시 측정한 EEF pose를 task frame으로 사용한다. Base 기준 EEF pose가
$(p_t,R_t)$이면 목표는 $p^*=p_t+R_t\Delta p_{\mathrm{EEF}}$,
$R^*=R_t\operatorname{Exp}(\Delta r_{\mathrm{EEF}})$다. 회전 입력은 Euler가 아닌 회전벡터다.
공식 OSC는 EEF 축 stiffness/damping도 base로 변환하며, 생성된 목표는 해당 정책 step의
physics substep 동안 유지한다. 이전 목표에 증분을 계속 누적하지 않는다.
이전의 ±15 mm/±0.08 rad 제한과 축별 gain 매핑, workspace 목표 clipping,
COM→TCP Jacobian/속도 보정, hand 측정 위치 기준 lead 제한, custom effort clipping은 제거했다.
Native task-frame 변환과 기존 관절별 물리 제한을 사용한다.
OSC의 gravity compensation은 관절 중력 토크를 feedforward로 더해 제어 출력을 만드는 기능이다.
F/T 측정값의 자중·관성 보상은 제거했다. Actor·Critic observation과 F/T 증가량 reward는
동일한 SI-200-8 측정값을 사용하며 hand 자중·관성에 의한 하중도 포함한다.
이 센서 처리 변경은 OSC의 gravity compensation=True와 별개다.
Action/적용 torque의 nonfinite는 MDP에서 검사한다.
Arm actuator PD gain은0이고 Hand는 공식 `ImplicitActuatorCfg`와
`JointPositionToLimitsActionCfg`로 독립6관절을 제어한다. 종속6관절에는 별도 action이나
Python 위치 갱신을 넣지 않고 **native PhysX Mimic API**로 URDF의 비율·offset을 연결한다.
손의 위치·속도 제한과 토크1N·m는 URDF에서 읽어 공식 API에 설정한다.
독립관절 gain20/1·종속관절0/0을 사용한다. 설치된 importer의 flag가 `parse_mimic`에
직접 전달되므로 `convert_mimic_joints_to_normal_joints=True`와 `joint_drive=None`을 사용한다.
Native mimic은 강체 제약으로 설정한다. Importer가 종속관절에 둔 확장 범위와 별도 속도
brake를 두지 않는 방식은 유지하고, URDF 전체 제약을 독립관절의 범위에 반영한다.

URDF의 일부 master 상한은 mimic child 상한을 초과한다. 초기화 때 전체 연동 범위의
교집합을 공식 physics limit에 전달한다. 네 손가락 master는0–1.36159rad·최대.922254rad/s,
thumb2는0–.609221rad·최대2rad/s다. 공식 action은 이 한계를 그대로 사용한다.
종속관절의 SDK limit 숫자는 원본보다 넓지만, native mimic과 master 한계를 합친 유효 범위는
원본 URDF 제약을 따른다. 실제12관절각을 원본 범위와 비교해 검증한다.
매 step coupling 계산이나 joint pose 덮어쓰기는 없다.
속도 상한은 독립관절의 native 제한이다. 접촉 시 종속관절의 순간 속도까지 원본 상한에
엄격히 묶는 것은 검증되지 않았다. 닫기 시험에서 index2는 순간9.25rad/s가 관측됐다.
종속관절에도 별도 native 속도 제한을 중복 적용한 대조는 불안정해 채택하지 않았다.
Native mimic과 접촉 제약의 수렴을 위해 결합된 로봇 articulation의 position solver를
32회로 설정한다. 2026-10-23 사용자 요청으로 예제용 로컬 USD에 맞춰 velocity solver를
1회로 변경했고, 물체도 예제 cfg와 같은 position16회·velocity1회다. 팔·손이 하나의
articulation이므로 로봇의 수치 설정은 전체 로봇에 적용된다. UR5e의 actuator·controller·
collision mesh는 유지한다. 아래 시험 수치는 변경 전 velocity2회에서 측정한 기록이다.
100Hz 여닫기 시험에서 최대 관절 범위 이탈은.000068rad, 최대 mimic 연동 오차는.030736rad였다.
1,024환경의 동일 시작 상태에서 native32step은 solver32회1.330초·64회1.955초였다.
이는 NN/PPO update가 빠진 환경 실행 시간이며, 64회는 연동 오차를 줄이는 대신 비용이47% 컸다.
예제처럼 OSC 환경의 `__post_init__`에서도 arm stiffness/damping=0을 명시한다.
팔은 `IdealPDActuatorCfg`를 사용해 OSC가 계산한 토크를 PhysX에 전달하기 전에 제한한다.
`effort_limit`과 `effort_limit_sim`은 모두 shoulder/elbow 150·wrist 28 N·m이며,
`velocity_limit_sim`은 shoulder/elbow 3.14·wrist 6.28 rad/s다. Hand는 기존 implicit
position PD를 유지한다. 별도 custom controller나 torque clamp는 추가하지 않는다.
이 출력 제한이 정책의 delta pose/stiffness 요구를 제한하거나 적절한 요구를 학습하도록
보장하지는 않는다. 측정 joint torque observation/reward는 아직 추가하지 않았다.
UR5e 팔6개 중 하나라도 절대 관절 속도가3rad/s를 초과하면 `hand_velocity`로 종료한다.
Hand는 이 종료 검사에서 제외하며, joint velocity reward는 현재 비활성이다.
이는 actuator의 물리 속도 상한과 별개인 episode 종료 조건이다.

Action 18D와 observation 244D, GNN·attention 구조는 유지한다. 제어 의미가 달라졌으므로
이전 base-frame·손바닥 EEF·custom controller checkpoint는 action-contract metadata 검사에서 거절한다.
현재 규약은 `isaaclab_ur5e_wrist_osc_variable_kp_gravity_v3`이며 새 학습을 시작한다.
244D 모양은 같지만 물체 기준점·관측 EEF pose·action의 의미와 중력 보상이 달라졌다.

### Reward

예제처럼 `RewardsCfg`에 함수·weight·파라미터를 선언하고 native RewardManager가 호출한다.
`mdp/rewards.py`는 각 수식의 raw score 또는 양의 cost를 환경별 `(num_envs,)` tensor로 반환한다.
Action-rate·joint-velocity 계산은 공식 MDP 함수에 맡기고, NaN/Inf 결과만0으로 처리한다.
`invalid_state` 종료 행도 reset 전에 reward가 평가되므로 필요한 보호다. 유한 비용에는 clipping을 하지 않는다. 초기화·이력이 필요한
Blocker 접촉은 `ManagerTermBase`를 사용하고, 공통 기하·F/T는 기존 GPU snapshot을 재사용한다.
별도의 합산 함수·환경 `step()`·reward component packed tensor는 두지 않는다.

**Dense reward는 native `weight × dt`를 그대로 적용한다.** 아래 표의 값은 dt 곱 전이다.
현재100Hz에서는 한 step 지급이 표의 weighted 값의0.01배다. 예를 들어 Distance 최대값은
3 reward/s, step당0.03이며, 같은 상태를1초 유지하면 할인 전 합은3이다.
Terminal은 성공+250·실패−3을 각각 일회 지급하므로 rate로 변환한다. 성공·실패의 기하 기준은
유지하며 성공 termination은 끈다. 아래는 parallel 프로젝트와 맞춘 현재 설정이다.

| Reward | Weight | 함수가 반환하는 raw score / cost |
|---|---:|---|
| Clearance | 12.0 | $\min(C_t,0)+V_pV_T\max(C_t,0)$, $C_t=1-O_t/O_0$ |
| `blocker_distance` | 3.0 | $\frac{V_T}{17}\sum_{i=1}^{17}\exp(-10d_i)$, 각 pad 표면 대표점에서 현재 Blocker OBB까지 거리 |
| `blocker_contact_hold` | 0.20 | $V_T\sum_{i=1}^{17}(b^B_{i,k-1}\land b^B_{i,k})$ |
| `shelf_collision` | −0.5 | 초기 대비 선반 pose 또는 현재 속도가 허용값을 넘으면1, 아니면0 |
| `object_collision` | −0.5 | $\sum_{i\ne B,\,i\text{ active}}\frac18\sum_{c=1}^{8}\lVert\mathbf v_{i,c}\rVert_2$ — 현재 OBB 코너 속도 |
| `motion_action_rate` | −0.03 | $\sum_{j=1}^{18}(a_{j,k}-a_{j,k-1})^2$ — 공식 `action_rate_l2` |
| `motion_force_growth` | −0.03 | $[\lVert F_k\rVert_2-\lVert F_{k-1}\rVert_2]_+^2$ |
| `motion_torque_growth` | −0.03 | $[\lVert\tau_k\rVert_2-\lVert\tau_{k-1}\rVert_2]_+^2$ |
| `object_stability` | 0.5 | $\exp[-\max_i(a_P[d_i^{\mathrm{stab}}-0.01]_+^2+a_\theta\theta_i)]-1$ |
| `time_penalty` | −1.0 | $\mathbf1[\neg\mathrm{terminated}]\mathbf1[\neg\mathrm{success}]$ |
| Terminal | 1.0 | 성공 $250/\Delta t$·실패 $-3/\Delta t$를 각각 episode당1회, 일반·timeout0 |

기존 누적 Rearrangement와 그 OBB 경로 이력은 제거했다. 예제의 선반·주변 물체 움직임 비용으로
대체했으며, **활성 Motion 세 항은 exp 포화 없이 양의 cost × 음의 weight**로 계산한다.
예전 모든 dense 항의 `/step_dt` 상쇄와 양의 reward/step 예시는 현재 설정에 적용하지 않는다.
`motion_joint_velocity`와 `motion_singularity`는 함수는 남기되 cfg에서 비활성화했다.
GNN·attention과 관측·action·PPO 설정은 유지했다. 새 정책의 성능 개선을 측정한 결과는 아니다.
아래 과거 학습·검증 수치는 당시 설정의 기록이며 새로운 reward 규모의 실측으로 읽지 않는다.

**Clearance.** 분모 $O_0$는 episode 초기 Blocker와 Target 접근 영역의 XY 교차 면적이며,
분자 $O_t$는 같은 영역에 대한 현재 Blocker의 교차 면적이다.

$$
C_t=1-\frac{O_t}{O_0},\qquad
r_{C,t}=12\Delta t\left[\min(C_t,0)+V_pV_T\max(C_t,0)\right].
$$

Target·주변 물체의 overlap은 제외하고 GPU 다각형 clipping도 Blocker만 평가한다.
초기 교차 면적은 reset 대상 행에서 한 번 계산해 고정한다. 직전 step 대비 증가량이 아니라
초기 상태 대비 차이이므로 개선 상태를 유지하면 양의 보상을 계속 받는다.
공중·영역 밖·전도·활성 실패에서는 양의 부분만0이고 음의 Clearance와 다른 비용은 남는다.
진행 gate는 현재 선반 frame의 signed support gap −5~10mm·중심XY·footprint overlap을
사용하되 속도·tactile 조건은 넣지 않아 정상적으로 미는 동안에도 보상한다.
Target이 초기 pose에서2cm 또는0.15rad를 넘으면 `V_T=0`으로 양의 Clearance·Distance·Contact만
차단한다. 복구하면 재개하며 추가 termination은 없다. Contact 이력은 실제 접촉대로 갱신한다.
초기0·완전 확보1·초기보다 두 배를 가리면−1이다. 비율의 상한 clip·exp·추가 초기 품질
차감은 없다. 예를 들어 현재 겹침이 초기의60%라면 원점수0.4, weighted rate4.8,
100Hz 지급0.048이다. 양의 rate 상한은12이며, 악화 구간은 음수로 계속 내려간다.
초기 교차 면적이 유한한 양수가 아니면 원점수0으로 처리한다. 작은 초기 면적에서는
같은 절대 면적 변화가 더 크게 평가되므로 새 학습의 실제 분포를 확인해야 한다.

**Distance·Contact.** 실제 tactile collision mesh의 노출된 local+Z 표면에서 대표점 하나씩을
초기화 때 선택하고, native 링크 pose로 GPU에서 변환한다. 17개 pad 각각의 OBB 거리 $d_i$에
대해 $\exp(-10d_i)$를 계산한 뒤 평균한다. 현재 사용자 실험 설정은 weight3·decay10이며
점수 범위는0–1이다. 한 pad만 접근해도 다른16개의 거리가 계속 반영된다. 모든 pad가10cm이면
품질약0.3679, 반감거리는6.93cm다. 78절에서 제안했던 weight1·decay15와 구분한다.
Invalid pad는0점으로 두고 분모17을 유지한다. OBB 내부·표면은 거리0이며, 컵 빈 공간 등
실제 접촉 여부는 별도 filtered force로 판정한다. 모든 환경·pad를 한 번의 batched tensor 연산으로 처리한다.

접촉 binary는 현재 Blocker와 pad 사이 normal force norm이0.01N을 초과하는지다.
같은 pad의 직전·현재 binary가 모두true이면1점이며17개를 평균하지 않고 합산한다.
최초 접촉·다른 pad로 전환한 첫 sample은0,3개 pad 유지 시 weighted rate+0.6·100Hz 지급+0.006,
이론적 최대 rate+3.4·step당+0.034이다. 직전·현재 policy sample은10ms 간격이며,
두 sample 사이의 완전한 연속 접촉을 증명하는 것은 아니다.
`BlockerContactHold(ManagerTermBase)`가 이력을 관리하고 native manager가 reset 행만 초기화한다.
현재 Blocker1열의 PhysX contact view를 재사용하며 Blocker 교체 때만 view를 바꾼다.
Actor의 binary tactile17개·244D는 유지한다. Observation은 전체 `net_forces_w`를 써서 다른 물체·선반
접촉도 인식하고, Blocker 구분은 reward만 사용한다. 접근·유지 자체도 양의 점수이므로 접촉률만
높아지는지와 접촉 후 Clearance·성공이 함께 증가하는지를 비교해야 한다.
`env.rewards.blocker_contact_hold=null`이면 filtered 접촉 보상을 제외하고 기존 binary 관측은 유지한다.

**Shelf·Object collision.** 두 이름은 예제를 따르지만 새 contact 센서나 접촉 impulse를 측정하는
항은 아니다. `shelf_collision`은 선반의 초기 위치 대비 이동>5mm, 초기 회전 대비>1°,
현재 선속도>0.005m/s, 각속도>0.05rad/s 중 하나이면1이다. 길이·각도·속도의 기준을 별도 cfg
파라미터로 선언하며 과거 gripper 전용 링크·고정 선반 높이에 의존하는 근접 비용은 복사하지 않는다.
Pose 변위 때문에 켜진 비용은 선반이 멈춰도 남고, 기존 선반1cm/2° 종료 기준은 유지한다.

`object_collision`은 Target과 초기 비인식 물체를 포함한 모든 활성 non-Blocker의
**OBB 코너 속도 크기**를 계산한다. OBB 중심을 $\mathbf o_i$, 해당 중심의 선속도를
$\mathbf v_i$, 코너 위치를 $\mathbf p_{i,c}$, 각속도를 $\boldsymbol\omega_i$라 하면 다음과 같다.

$$
\mathbf v_{i,c}=\mathbf v_i+\boldsymbol\omega_i\times(\mathbf p_{i,c}-\mathbf o_i),\qquad
c_O=\sum_{i\ne B,\,i\text{ active}}\frac18\sum_{c=1}^{8}\lVert\mathbf v_{i,c}\rVert_2,\qquad
r_{O,k}=-0.5c_O\Delta t.
$$

코너8개는 평균하여 순수 병진을8번 중복 계산하지 않고, 물체들은 합산하여 여러 물체의
조작량을 모두 반영한다. Blocker와 미사용 pool slot은 연산 전에 제외하며 비유한 물체의
기여만0으로 처리한다. 기존 `invalid_state` 종료는 유지한다. 환경축을 유지하므로 다른 환경의
물체 속도가 현재 환경의 비용에 섞이지 않는다.

**tanh 포화는 사용하지 않는다.** 기존 snapshot의 OBB 중심 속도·각속도·기하로 control
step마다 한 번 계산하고 환경별 scalar `object_motion_cost`를 캐시한다. 추가 physics 조회·
ray casting·환경별 반복문·코너 이력·속도의 `/dt` 변환은 없다. Native dt로 적분한 합은
병진·회전을 포함한 조작 경로 길이의 근사이며 빠른 움직임 자체의 벌점은 아니다. 동일10cm
병진은0.1초든1초든 할인 전−0.05이고 왕복은 양쪽 경로 모두 남는다. 속도는 policy sample이므로
sample 사이 충격 궤적까지 정확히 복원하지는 않는다. 정지 후에는 비용0이며 초기 pose 오차를
유지하는 항도 아니다. 급격한 조작·힘 증가·기울기와 이탈은 별도 Motion·Stability가 평가한다.

이전 선속도 L1합의 tanh는 예제를 옮기면서 추가했던 처리이며 사용자가 action 출력에 요청한
tanh가 아니다. 후속 수정 방향을 논의할 때는 **현재 선형 비용과 이전 tanh 비용을 비교 후보로
다시 설명**한다. tanh는 큰 속도에서 포화되어 빠른 burst의 경로 비용을 상대적으로 줄이는 반면,
현재 선형 비용은 큰 순간 속도·진동·측정 노이즈에도 그대로 반응한다. 자동으로 tanh를 복원하지 않는다.

**Motion.** Action-rate는 현재·직전 raw18D 차이의 제곱합이며 Kp가 clamp되는 구간의 raw 변화도 포함한다.
Joint velocity 함수는 팔6+hand 독립6관절의 제곱합을 계산하지만 현재 reward는 비활성이다.
Action-rate와 F/T 증가량에 가속도·관절 수 정규화·exp는 추가하지 않는다.
큰 cost의−1 포화가 없으므로 활성 Motion 비용의 유한 하한은 없다.

$$
\Delta F_k=[\lVert F_k\rVert_2-\lVert F_{k-1}\rVert_2]_+,\qquad
\Delta T_k=[\lVert\tau_k\rVert_2-\lVert\tau_{k-1}\rVert_2]_+,\qquad
r_{F,k}=-0.03(\Delta F_k)^2\Delta t,\qquad
r_{T,k}=-0.03(\Delta T_k)^2\Delta t.
$$

사용자 결정대로 F/T는 **매 step 증가량의 제곱**이다. 증가율로 나누거나 deadband·exp를
적용하지 않는다. 자중·관성 하중을 포함한 SI-200-8 측정값의 norm을 비교하므로 하중 감소·일정 하중·
norm이 같은 방향 변화는 비용0이다. 힘과 토크는 서로 상쇄하거나 한 norm에 섞지 않는다.
두 항의 입력 단위는 각각N²와(N·m)²다. 실제 weight는 둘 다−.03이므로 ΔF=1N은−.03 reward/s,
ΔT=.05N·m는−.000075 reward/s다. 100Hz에서는 각각−.0003/step과−.00000075/step이다.
같은 weight가 같은 물리적 민감도를 뜻하지 않는다. 과거−.15/−60 계수의 설명은 현재 값에
적용하지 않는다. 센서 해상도·자중·관성 하중을 포함한 실제 증가량 분포를 새 run에서 확인해야 한다.
매 step 증분에 대한 제곱이므로 제어주기를 바꾸면 같은 연속 하중 궤적의 비용까지 불변이지는 않다.

**Singularity.** 현재 reward는 비활성이며 종료 조건은 유지한다. 현재 팔 자세의 공개 OSC
Jacobian으로 다음 보수적 지표를 계산한다.

$$
\bar J=\begin{bmatrix}J_v/0.25\\J_\omega\end{bmatrix},\qquad
s=\frac{1}{\|\bar J^{-1}\|_F}
=\left(\sum_{i=1}^{6}\sigma_i^{-2}\right)^{-1/2},\qquad
\frac{\sigma_{\min}}{\sqrt6}\le s\le\sigma_{\min}.
$$

0.25m는 기존 IK의 위치·회전 weighting 규약이며 인증된 안전 상수는 아니다. 정확한 최소
singular value·condition ratio는 아니고 여러 방향이 약하면 더 보수적으로 평가한다.
`torch.linalg.inv_ex(check_errors=False)`로 GPU에서 계산하며 실패·비유한 행은0이다.
Controller의 역산·토크에는 개입하지 않는다. $a_S=\ln(2)/.03^2$, warning .07과 종료 .005를
유지한다. 비활성 reward는 이 지표로 비용을 지급하지 않는다.
부분 reset 행은 첫 완료 physics sample까지0으로 초기화하며 이 지표를 관측에 추가하지 않는다.

**Object stability.** 현재 선반 frame에서 전체 OBB의 가장 큰 XY 돌출과 실제 충돌형상의 최저Z
지지 오차를 거리로 결합하고, yaw를 제외한 기울기θ를 더한다. 거리 deadband10mm,
20mm에서 거리 품질0.5,45°에서 기울기 품질0.5인 계수를 유지한다.

$$
r_{P,k}=0.5\left\{\exp\!\left[-\max_{i\text{ active}}
\left(\frac{\ln2}{0.01^2}[d_i^{\mathrm{stab}}-0.01]_+^2+
\frac{\ln2}{\pi/4}\theta_i\right)\right]-1\right\}\Delta t.
$$

Blocker도 포함하며 속도 항은 아니다. 기울기만45°이면−.25/s, 기울기 없이 지지 오차20mm면
−.25/s가 된다. 기존 중심기반 지지 실패·전도 종료는 그대로 둔다.

**Time·Terminal.** `RewardsCfg.time_penalty`는 `mdp.time_penalty`가 공식 `is_alive`에
현재 유효 성공 mask를 곱한다. Weight−1은 유지한다.

$$
f_{T,k}=\mathbf1[\neg\mathrm{terminated}_k]\mathbf1[\neg\mathrm{success}_k],\qquad
r_{T,k}=-\Delta t\,f_{T,k}.
$$

일반 step의 비용은100Hz에서−.01,50Hz라면−.02로 같은 실제1초의 비용이−1이 된다.
현재 정책·물리 주기는100Hz 그대로다. 1/2/4/8초 동안 모든 step이 과금되면 할인 전 합은
−1/−2/−4/−8이다. 비용률은 처음부터끝까지−1/s이며 경과 시간에 따라 가팔라지지 않는다.
현재 유효 성공을 유지하는 동안 비용은0이고, 그 상태를 잃으면 비용을 재개한다. 성공을
한 번 달성했다는 기록으로 영구 면제하지 않는다. 실제 episode 시간과8초 timeout은 멈추지 않는다.

Native step은 **termination 계산 → reward 계산 → reset** 순서다. 실패의 non-timeout
종료step은 `terminated=True`여서 이 비용이0이다. Timeout만 발생한 step은 truncated이므로
현재 성공이 아니면−.01이며 성공을 유지 중이면0이다. 공통 장면 교체도 같은 규칙을 따른다.
Reset 자체에는 과금하지 않으며 새episode는 첫 정상step부터 다시 평가한다.

실패도 남은 시간 비용을 피할 수 있으므로 Time만으로 빠른 성공을 보장하지 않는다.
현재γ=.98·100Hz의 할인반감기는약.343초이며,8초 Time 보상열 자체의
할인합은약−.5다. 무할인−8과 구분해야 한다. Timeout에서는 PPO가 다음 가치도 bootstrap하므로
이 값이 전체 학습 return은 아니다. 이번 동기화에서 PPO·γ는 변경하지 않았다.

Terminal은 성공+250과 활성 실패−3을 **각각 episode당 한 번** 지급한다.
성공 보상은 성공 종료의 활성 여부와 독립적이다. 현재 success termination은 꺼져 있어
유효 성공 후에도 조작·관측을 이어간다. 두 지급 이력은 `success_paid`·`failure_paid`로 분리한다.

$$
P_k=\begin{cases}
-3,&\text{아직 비용을 지급하지 않은 활성 실패},\\
250,&\text{성공·실패 지급 이력이 없고 실패 없이 유효 성공 확정},\\
0,&\text{그 외 및 timeout},
\end{cases}
\qquad f_{\mathrm{terminal},k}=P_k/\Delta t.
$$

RewardManager가 weight1·dt를 적용하므로 실제 한 번 지급되는 금액은 정확히−3 또는+250이다.
여러 실패가 겹쳐도 기존 aggregate `state.failure`로 한 번만 감점하며 실패가 성공보다 우선한다.
`hand_velocity`도 포함한다. 성공을 잃었다가 회복해도+250을 반복 지급하지 않으며, 성공 보상을
받은 뒤 실패하면−3은 별도로 지급한다. Timeout 단독은0이고 실패가 겹치면 실패 비용을 지급한다.
부분 reset은 선택 행의 두 지급 flag만 비워 새episode에서 다시 평가한다.
실패별 비용 합산·환경 step override·추가 센서나 PhysX 조회는 없다.
기존 `Episode_Reward/terminal`에는 성공·실패 지급이 함께 누적되며 원인별 termination 로그는 유지한다.
Distance 최대값의8초 합은24, Hold는27.2, Clearance는96이라는 느슨한 할인 전 상한이다.
이들이 동시에 성립한다고 가정한 성공 선호 보장은 하지 않는다. 음의 L2 비용은 무한히 커질 수 있어
과거 유계 reward·value 상한 비교를 새 설정의 전체 범위로 사용하지 않는다.

Console·TensorBoard는 native `Episode_Reward/<term>`으로 활성11개 항을 각각 기록한다.
`rearrangement`는 없고 `shelf_collision`·`object_collision`이 추가된다. 로그는 weighted episode
누적값을 최대 episode 시간8초로 나눈 값이며 순간 점수나 실제 episode 길이 평균이 아니다.
과거 `/step_dt` 상쇄·exp Motion 로그와 새 값을 그대로 비교하지 않는다.
`TaskState`는 `clearance_quality`, `terminal_reward`, `force_growth`, `torque_growth`,
환경별 scalar `object_motion_cost`를 이름 있는 tensor로 캐시한다. Packed reward_components와 누적 OBB
이력은 제거하고 공통 GPU snapshot·부분 reset·한 step당1회 갱신 규칙은 유지한다.
Native dt 변경은 [학습 기록63절](../../docs/TRAINING_LOG.md#63-2026-10-24--예제-구조의-native-dt-reward와-움직임-비용),
OBB 코너 속도와 tanh 제거의 근거·검증 범위는 [64절](../../docs/TRAINING_LOG.md#64-2026-10-24--object-비용의-obb-코너-속도와-tanh-제거)에서 확인한다.

성공은 **지정 Blocker의 A₀ 교차 면적 ≤ 1e−5 m², Target 초기 pose 대비 이동 ≤ 2 cm·회전 ≤
0.15 rad**에 더해 다음을 **0.5초** 연속 만족할 때다. 현재100Hz에서는50 sample이며
`success_duration / step_dt`를 올림해 주기가 바뀌어도 같은 시간을 유지한다.

- 모든 활성 물체의 XY OBB가 작업판 경계 안에 있음(경계5 mm 허용).
- 현재 선반 frame에서 물체 collision 기하의 최저점과 지지면의 signed gap이 **−5~5 mm**.
- 현재80° 이상 넘어진 물체가 없고, 모든 활성 물체의 선속도 ≤ **0.05 m/s**, 각속도 ≤ **0.3 rad/s**.
- 활성 실패가 없고, 물체·센서·제어 상태가 유효함.

Blocker뿐 아니라 주변 물체의 잔류·안정도 검사한다. 속도 기준은 **완료 판정에만** 쓰며,
미는 중의 진행 Clearance를 막는 조건이 아니다. 비활성 padding은 모두 제외한다.
원래 통로·Target 조건인 `raw_goal`과 유효 성공을 분리해 진단하며, 활성 실패와 성공은 상호배타적이다.

지지면은 선반 전체 OBB가 아니라 실제 작업판과 기둥에서 측정한 사용 영역이다.
물체의 활성 collider 정점을 시작 시 asset별로 한 번 읽고 convex hull의 지지점을 GPU에 저장한다.
최저 높이는 현재 선반 frame의 평면 방향에 대한 정점 내적으로 구해, 기울어진 OBB의 빈 모서리가
지지면 아래로 내려갔다는 이유만으로 정상 pivot을 거절하는 일을 줄인다. Mesh hull의 평면 극값은
원래 정점과 같고, 곡면 primitive는 촘촘한 근사와 오차 상한을 사용한다. 이 cache는 **물리 collision
형상을 바꾸지 않으며 실제 접촉력의 증명도 아니다.** 정점을 환경별로 복제하지 않고 약3.39MB의
공통 catalog를 Warp block reduction으로 처리한다. 작은 tensor연산은 cached CUDA graph로 묶는다.
실제10종의 초기 gap은0, 1,024환경64holdstep후 −.182~1.711mm로 진행·완료 물리mask는 모두
정상이었다. Support helper는.414ms/step, 같은 고정hold의 native step비교는40.34→39.82ms로
뚜렷한 지연 증가는 없었다. PPO·새scene reset을 제외한 결과이므로 전체iteration 성능의 증명은 아니다.
상세 측정과 한계는 [학습 기록34절](../../docs/TRAINING_LOG.md#34-2026-10-21--유효-재배치-성공과-비산-차단-1단계-적용)을 따른다.

물체의 지지 실패는 낙하·XY 이탈·큰 들림의 세 term을 하나의 **유한 작업판까지의 거리**로 대체했다.
현재 선반 frame에서 물체 OBB 중심의XY와 실제 collider 최저 높이로 대표점
$p_i=(c_{ix},c_{iy},z_{i,\min})$를 만들고, 실제 작업판 위 최근접점 $\Pi_S(p_i)$까지의 거리를 쓴다.

$$
S=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]\times\{z_{\mathrm{panel}}\},\qquad
d_i^{\mathrm{support}}=\|p_i-\Pi_S(p_i)\|_2.
$$

작업판 내부의 지지된 평면 밀기·yaw pivot은 $d_i^{\mathrm{support}}=0$이다. 위/아래 이탈과 수평 이탈은 같은
유클리드 거리로 평가한다. 전체 OBB가 경계를 완전히 벗어나야 하던 이전 기준과 달리 중심의
수평 이탈부터 반영하므로 의미가 동일하지 않다. 대표점은 실제 무게중심·접촉 지지다각형의
안정성을 증명하지 않는다. 지지된 옆넘어짐도 거리0일 수 있어 tilt는 별도로 유지한다.
**기존 지지 실패와 그 지속시간은 이 중심 기반 거리를 그대로 사용한다.**

Object stability reward는 XY에서 **전체 OBB 중 가장 많이 돌출한 꼭짓점**을 사용한다.
작업판 XY 사각형을 $S_{XY}$라 하면, 현재 선반 frame의 거리는 다음과 같다.

$$
\delta_{i,XY}=\max_{m=1,\ldots,8}
\|p_{i,m,XY}-\Pi_{S_{XY}}(p_{i,m,XY})\|_2,\qquad
g_i=z_{i,\min}-z_{\mathrm{panel}},\qquad
d_i^{\mathrm{stab}}=\sqrt{\delta_{i,XY}^2+g_i^2}.
$$

Z는 기울어진 OBB의 빈 모서리가 아니라 기존의 실제 collision 기하 최저점이다. 아래로 뚫린
상태와 위로 들린 상태 모두 $g_i^2$로 반영한다. 중심이 선반 안에 있어도 OBB가 돌출하면
거리 비용을 받을 수 있다. 추가 기울기 위험은 물체의 upright 기준 위쪽 축 $u_i$와 선반
법선 $n_S$ 사이의 각도 $\theta_i=\arccos(\operatorname{clamp}(u_i\cdot n_S,-1,1))$다.
평면 yaw는 제외하며 사용자 선택에 따라 **각도 자체**를 지수에 넣고 제곱하지 않는다.

$$
x_i=\frac{\ln2}{0.01^2}[d_i^{\mathrm{stab}}-0.01]_+^2
+\frac{\ln2}{\pi/4}\theta_i,\qquad
\dot r_{\mathrm{stability}}=0.5\left[\exp\left(-\max_{i\in\mathcal O_{\mathrm{active}}}x_i\right)-1\right].
$$

위 식은 reward rate이며 native manager가 `dt`를 곱한다. 기울기가0이면 거리1cm까지0,
2cm에서−.25/s,3cm에서−.46875/s다. 거리 초과 비용이 없고 기울기만45°면 품질0.5·
실제 비용−.25/s다. `stability_tilt_decay=ln(2)/(π/4) rad⁻¹`이며 각도에는 별도 dead band가 없다.
같은 물체가 거리2cm·기울기45°면 두 위험이 결합해−.375/s다.
Blocker를 포함한 활성 물체별로 두 위험을 결합한 뒤 최대값을 취하므로 서로 다른 물체의
최악 거리와 최악 기울기를 별도로 더하지 않으며, 물체 수로 비용이 배가되지 않는다.
추가 속도 비용이나 Blocker 불안정 gate는 없다. 같은 물체의 기존 중심 기반
$d_i^{\mathrm{support}}\ge3$cm가0.1초 지속되면
`object_support_lost`로 종료한다. 각 물체 카운터는 회복·비활성화 시0이며 부분reset은 선택 환경만
초기화한다. 진행 Clearance gate와 완료 조건은 기존의 더 엄격한 기하를 유지하고, 실패의 유예
중에도 진행 유효성을 잃으면 양의 Clearance를 즉시 차단한다.

같은 물체의80° 이상 기울어짐0.2초 지속, 선반의 초기 pose 대비1cm/2° 초과,
invalid·singularity는 유지한다. 손가락 관통의 물리 설정 수정은 사용자의 요청대로 보류했다.
지원 거리 비용은 충돌 누락 방지나 종료 전 감속을 보장하지 않는다.

비활성 `hard_load_or_overrange`의 종료 경로와80N/4N·m hard threshold는 제거했다. SI-200-8의
raw 범위 초과 진단은 유지하되 성공의 숨은 gate로 적용하지 않는다. 선반은 공식 `RigidObject`로 관리하는
100 kg dynamic body이며, 관성은 PhysX가 collision 형상에서 계산한다. 선반 pose는 observation에
추가하지 않는다. 부분 reset은 해당 환경의 pose·속도·초기 기준과 지속시간 카운터만 초기화한다.
Timeout은 native truncation으로 유지한다.

Tilt는 모든 활성 물체에 대해 개별적으로 누적하며100Hz에서 같은 물체가20 sample 연속
기준을 만족해야 종료한다. 80° 미만 회복·비활성화 시 해당 카운터를0으로 만들고, 서로 다른 물체의
교대 기울어짐을 합치지 않는다. 평면 yaw는 tilt가 아니다. 예전 `workspace_exit`·`object_lift`를
복구하지 않고, 위의 단일 지지 거리와 지속시간 조건을 사용한다.

실패 원인은 공식 `TerminationManager`의 개별 term으로 기록한다. Console과 TensorBoard의
`Episode_Termination/<원인>`에서 확인할 수 있으며, 기존 통합 `failure` 로그는 아래 항목으로
대체했다. 연구 task의 term은 같은 step의 캐시된 판정 결과를 읽는다.
`hand_velocity`는 공식 `joint_vel_out_of_manual_limit`을 사용하며 native 관절 속도 buffer를 재사용한다.

| 원인 이름 | 현재 실패 조건 |
|---|---|
| `object_support_lost` | 같은 활성 물체의 유한 작업판까지의 지지점 거리 ≥3cm가0.1초 연속 지속 |
| `object_tilt` | 같은 활성 물체의 기울기 ≥80°가 0.2초 연속 지속 |
| `shelf_displacement` | 선반 이동 1 cm 또는 회전 2° 초과 |
| `invalid_state` | 유효하지 않은 물리·센서·제어 상태 |
| `singularity` | 현재 팔 자세의 보수적 이동능력 지표 s≤0.005 |
| `hand_velocity` | UR5e 팔6개 중 하나라도 절대 관절 속도 >3rad/s. 한 sample 초과 시 즉시 종료. Hand 제외 |

한 episode에 여러 원인이 동시에 기록될 수 있어 원인별 비율의 합이 전체 실패율과 같지는
않다. 현재 활성 실패6개와 timeout을 합쳐 native 종료 term은7개다. 성공 판정은 계속 계산하지만
success termination은 비활성이므로 성공만으로 episode를 종료하지 않는다.
활성 실패끼리는 동시에 기록될 수 있지만 성공은 활성 실패가 없을 때만 성립한다.
Terminal은 유효 성공+250·활성 실패−3을 각각 한 번 지급하며 일반 진행·timeout 단독은0이다.
동시 실패는 합산하지 않으며 성공과 겹치면 실패가 우선한다.
연구 task의 내부 실패는 초기화 때 읽은 native 활성 term 설정을 따르며, 속도 실패는
native manager가 해석한 현재 `hand_velocity` 설정으로 성공·terminal 지급 전에 반영한다.
따라서 속도 초과와 성공 보상이 동시에 발생하지 않는다. 관절은 팔 action과 동일한 이름으로
선택하고 Hand·mimic/passive 관절은 검사하지 않는다. 경계3rad/s 자체는 통과한다.
Joint velocity reward는 현재 비활성이고 팔 속도 종료 조건은 유지한다.
현재 임계값은 최신 원격 `min`의 InspireOpen→Hand/HandMass 계열의3rad/s를 확인한 뒤 사용자 승인으로
변경했다. 후속 사용자 요청으로 우리 환경도 팔6개만 검사하도록 Hand를 제외했다.
관절 선택 변경은 [학습 기록70절](../../docs/TRAINING_LOG.md)을 따른다.
원격 기본 cfg의1rad/s와 별도 HandMassLimit variant는 구분한다. 변경 근거는
[학습 기록68절](../../docs/TRAINING_LOG.md)을 따른다.
Action 변화량 자체의 종료·clipping은 추가하지 않았다. Native 순서상 종료 step에도 reward를
계산하므로 이 속도 종료만으로 큰 finite action/관절속도 비용의 PPO 유입이 차단되지는 않는다.

새로 완료된 episode의 실제 누적 집계도 기존 native logger로 제공한다. `Task_Count/completed`,
`valid_success`, `raw_goal_seen`, `unsupported_goal_seen`, `goal_and_failure`는 종료된episode를
reset 전GPU에서 한 번씩 세는 누적 개수다. `Task_Rate/valid_success`는 누적 유효성공/완료episode다.
`raw_goal_seen`은 episode 도중 통로·Target조건을 한 번이라도 충족했는지, `unsupported_goal_seen`은
그때 진행유효성이 없었는지, `goal_and_failure`는 raw목표와 활성실패가 겹쳤는지 센다. 서로배타적인
실패 원인 비율은 아니다. 기존 `Episode_Termination/*`의 마지막episode snapshot 평균과 구분한다.
Singularity는 MDP step당한번계산하며 해당 reward는 비활성이다. Native로그
`Episode_Termination/singularity`로분리한다. 임계값은초기실험기준이다.
Native실행순서가OSC→physics→termination이므로현재OSC호출에서발생하는역행렬예외까지
막는장치는아니다. OSC정칙화·action clipping·이력전처리·초기IK는이번변경에포함하지않았다.
적용 근거와 PhysX 오류의 충돌량 검사는 [학습 기록16절](../../docs/TRAINING_LOG.md#16-2026-10-20--성공-판정-분리거리-weight-적용과-physx-중단-조사)에 둔다.

## 병렬 계산과 reset 연결

현재 기본 `spawn_rules.generation_mode="clutter"`의 계산 순서는 다음과 같다.

```text
reference10 catalog: T/B 상대 pose·시점·초기 가림만 재사용
  → 종류·균형 중복 순서와15–25개 요청 구성
  → GPU FFT: 실제 collider의 비충돌 위치/방향 영역에서 partial seed 생성
  → 한 번의 CPU 공동 제약 계산: 필수 물체·추가 prefix·빈 Blocker 목적지를 함께 결정
  → 고정3회 X/Y 유효 구간 mixing
  → 초기 메시 관측1회 → T/B + 가까운 주변8개 → 정책244D
```

GPU seed에서 빠진 필수 물체는 최종 장면에서 제거하지 않는다. Partial graph를 받아 공동 제약에서
필수13개 BG 전체를 포함한다. 추가 물체는 균형 prefix만 사용할 수 있고 큰 종류를 바꾸지 않는다.
빈 목적지는 초기 시각 보호 공간과 겹쳐도 되지만 T/B·A₀·배경 물체와 충돌하지 않아야 한다.
이전 full-path witness는 재사용하지 않는다. 공동 제약의 내부 최적화와 무작위 장면의 승인 재시도는
구분하며 전체 생성이 GPU-only이거나 모든 공간 조건을 기존 cache가 인증한다고 설명하지 않는다.

다음은20–40개 통합 clutter 변경 전에 검증한 `spawn_rules.generation_mode="parameterized"`의
흐름이다. 기존 두-row 배경 배치·물리/관측 개수 결합을 설명하는 비교 기록이다.
기존 `constructive` 경로의 DP·compaction·37/74시점 승인은 아래 이전 구현 기록에 보존한다.

```text
환경 생성 1회: robot / shelf / 물체 pool / tactile 17링크 통합 센서 / 공유 ray-casting mesh 준비
             → initialize_certified_catalog: 검증된 연속 영역 cache 로드 / 없으면 사전 계산
새 scene round: 역할·균형 중복 구성 → 해당 구성의 인증된 family 선택
              → CDF의 연속 변수 추출 → roles / camera / guard / 두-row background 직접 구성
              → observe_at_viewpoint: 한 번의 초기 관측 → template 저장
reset(env_ids): 공통 초기 배치·vision 복사
              → 선택 환경마다 Blocker 앞 EEF 위치·회전을 새로 샘플링
              → GPU 배치 IK·elbow-up·전체 로봇 충돌 검사
              → 물체·선반·로봇 상태를 선택 행에 일괄 적용
              → reset_mdp: 초기 vision·역할·시계·직전 이력 확정
800 step 경계: 전체 환경 timeout → 다음 scene round
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
가속도·물체 속도는 계산하지 않는다. 정상 step도 F/T 자중·관성 보상을 수행하지 않으며,
이를 위한 hand COM·질량·관성 cache와 링크 가속도·각속도 조회를 제거했다.
Object 코너 속도 계산용 고정 local COM은 유지하며, object COM을 randomize한다면 함께 갱신해야 한다.

2026-10-15 RTX3090·1,024환경·10물체의 전체 최적화 후30 iteration에서 초기 compile 회차와
profiler 적용 회차를 제외한28회 평균은 **1.84초/iteration**이었다(1.73~2.02초).
직전14회 측정3.21초 대비 같은 iteration14개 비교는3.21→1.85초, 약42.5% 감소다.
Reset은1.02→0.31초, PPO update는0.32→0.23초로 줄었다. Native PhysX partition은1이며,
100Hz physics/50Hz policy, solver16/2, rollout32·PPO4epochs/4minibatches는 유지했다.
짧은 새 학습의 비교이므로 trajectory와 reset 행 수는 완전히 같지 않다. 상세 계측은
`outputs/contact_manipulation/full_optimization_20261015/comparison.json`에 있다.

최신 요구는 **4–10종을 유지하면서 총15–25개**를 하나의 clutter로 배치하는 것이다.
Target 종류는 정확히1개이며 선택된 non-Target 종류별 개수 차이는 최대1개다. 개수는 같은
선반의 수용 가능한 범위에 연동하되 큰 물체나 선택 종류를 몰래 제외하지 않는다. 이 생성기
변경은98개 standalone 장면과 native4·1,024환경 검사로 확인했으며 기존10물체 결과와 구분한다.

전체 이동·pivot 경로의 빈 공간 예약은 제거하고 Blocker의 빈 목적 영역만 확보한다.
이는 충돌 없는 연결 경로나 모든 scene의 조작 성공 가능성을 증명하는 규칙이 아니다.
정해진 기하 조건에서 직접 만든 seed를 고정3회 조건부 X·Y sampling으로 섞는 helper는
CPU 검증을 마쳤다. 새4·1,024환경 검사에서는 실제 메시 가시성Target68.85%·Blocker100%,
관측244D와17개 tactile 채널을 확인했다. 1,024환경의 부분 reset은 다른1,023환경을 그대로 유지하고
장면·관측을 추가하지 않았다. 초기 특이점·비유한 상태·종료·fallback은0이었다.

`CONTACT_MANIPULATION_NUM_OBJECTS`는 최초 topology 구성 전에 읽는 물리 최대 개수로 기본25이다.
최소4종에서 임의 종류의 균형 중복을 지원하려면 종류별8개, 총80개의 body를 미리 생성해야
한다. 활성15–25개를 선반에 놓고 나머지는 기존의 격리된 parking 위치에서 유지한다.
현재 SDK의 GPU 경로에서 simulation OFF→ON 뒤 새 pose가 다음 물리 step에 유지되지 않는
문제가 재현되어 live simulation flag 전환은 사용하지 않는다. 미사용 물체는 기존의 격리된
주차 위치에서 native kinematic body로 유지하고 활성 물체만 dynamic으로 전환한다.
Native actor·GPU view와 collider는 유지하며 gravity flag나 중력 상쇄 외력을 사용하지 않는다.
Kinematic 물체는 중력 적분과 dynamic solver 처리를 줄일 수 있지만 collider·broadphase·
transform 관리와 메모리 비용은 남으므로 PhysX 연산에서 완전히 제외하는 방식은 아니다.
활성 mask가 바뀌는 전체 공유 장면 commit에서만 USD kinematic 속성을 갱신하고 PhysX에
반영한 뒤 native writer로 pose·velocity를 기록한다. 같은 장면의 부분 reset은 동일 mask를
재사용하여 USD 속성 변경과 flush를 생략한다.
바닥은 기존 native 전역 GroundPlane, 환경 ID bounds는 native cloner의4bit 설정을 사용한다.
스폰 규칙·높이와100Hz 물리·정책 주기는 유지한다. 검증·미확정 범위는
[TRAINING_LOG](../../docs/TRAINING_LOG.md)에 기록한다.
이번 변경은 최대30개 설정의100-body pool에서 환경당20개를 줄인다.
이전130-body source 검사와4환경30step·1,024환경10step을 통과했다. 당시 GPU used는
10.84GB, host RSS는9.54GB였다. 이 값은 현재80-body 설정의 측정값이 아니다.
Pool 크기로 첫 body의 주차 변환이 달라지지
않도록 기존3-copy slot을 유지하고 추가 복사본만 뒤에 둔다. 주차 위치 집합은 그대로다.

관측 용량은 물리 개수와 분리한10개로 고정한다. Target·Blocker가 slot0/1이며 인식한 주변
물체는 두 역할까지의 최소 거리 순서로 최대8개를 선택한다. 공개 관측 폭은244D다.
이전30/40-slot checkpoint를 현재10-slot 설정으로 해석하지 않는다.

### 이전 10물체 연속 영역 생성 경로의 검증

학습 전 `initialize_certified_catalog`가 실제 메시의 유효 Blocker 이동 구간, 초기 A₀ 차단,
카메라·선반·조작 경로와 주변 물체의 공간 예산을 결합한다. 병진 경로로 수용되지 않는 관계에는
`_precompute_normal_pivot_family`의 pivot 후 이동 경로를 사용할 수 있다.
`_precompute_guarded_role_family`는 순수±Y 이동으로 A₀를 비울 수 없는 부분 family를 인증한다.
일반 XY 이동 전체의 불가능성을 주장하지는 않는다. Cache에는 관계식·연속 구간·CDF와 실제 메시
가시성의 fingerprint를 저장한다. 완성된 전체 장면 bank를 재생하는 방식이 아니다.

같은 주변 종류·개수를 고정해도 두 역할 각각360°를 지원하도록 모든 허용 구성의 **ordered-pair별
최악 공간 예산**을 적용한다. Guard를 뺀 주변 물체는 별도의 residual 예산을 사용한다.
최종 공간 인증까지 통과한 guard 비의존 family들의 yaw 합집합에 빈 구간이 없어야 catalog를
승인한다. 각도 빈도와 변수 간 연동은 균등·독립일 필요가 없으며, 두 yaw의 모든 독립 조합을
지원한다는 의미는 아니다. 주변 물체도 모든 slot이 자유 yaw slot으로 선택될 양의 확률을 갖는다.

Reset의 `sample_certified_episode → sample_certified_scene`은 먼저 균형 종류·개수를 정하고,
사용 가능한 관계 영역에서 한 tuple을 추출하여 역할·카메라·필요한 guard·주변 chain을 직접 만든다.
고정 두-row chain은 회전된 collision footprint의 Minkowski 경계에서 다음 위치를 계산한다.
`spread_gaps=True`는 가까운 간격을 선호하며 남는 공간 안에서 넓은 간격도 연속적으로 허용한다.
이 경로는 생성 후 배치 승인·교체·재생성, DP 분할이나37/74개 시점 평가를 하지 않는다.
이 작은 CPU 기하 계산은 공통 장면 하나에만 수행하고1,024개 clone마다 반복하지 않는다.

기본 cache 위치는 `outputs/contact_manipulation/parameter_domains`다. Geometry·pool copy 수·
생성 규칙·센서 설정이 바뀌면 cache identity도 바뀌며 checksum을 검증한다. 완료된 ordered pair의
checkpoint로 사전 계산을 재개한다. 이번90 pair·6,593 family·207,027 cell의 cold-cache 생성은
**33.68분**이었으며, 같은 입력의 후속 실행은 cache를 로드한다. 이는 reset/iteration 시간이 아니다.
실제360장면과 guarded4장면의 기하·광선 검증,1,024환경 native robot-start/reset 검증을 통과했다.
Native 초기 자세의 활성 singularity 기준 위반은0행, fallback은0회였다. Reset은 최초1.960초,
새 bucket을 준비한5환경 부분 reset0.265초, warm 전체 reset0.117초로 측정했다. 이 값은 전체
학습 iteration 시간이 아니다. 공식 학습 진입점의
`2026-10-22_17-19-09_parameterized_default_smoke`는26 iteration·936 step을 완료하고800-step의
새 장면 전환을 확인했다. 두 장면의 초기 자세 위반과 reset fallback은0이었으며
actor·critic·optimizer 값도 모두 유한했다. 학습 시간108.254초에는 cold capture가 포함된다.
이는 짧은 통합 시험으로 장기 안정성이나 iteration당1–2초 달성을 뜻하지 않는다.
최신 결과는 [학습 기록43절](../../docs/TRAINING_LOG.md#43-2026-10-22--연속-파라미터-구간과-직접-생성-경로)에 둔다.

초기 vision은 실제 카메라나 render product 없이 공유 시각 메시 BVH를 GPU로 조회한다.
물체만 ray casting하고 로봇·선반 가림은 제외한다. `observe_at_viewpoint`는 정해진 시점의
첫 교차 표면으로 OBB를 추정하며 가려진 부분을 정답 pose·전체 크기로 채우지 않는다.
Target40–90%·Blocker≥95%는 startup의 실제 메시 구간에서 인증하며 reset에서 다시 승인하지 않는다.
따라서 관측의 `visible_fractions=None`은 미측정이라는 뜻이고, cache 예측값은 별도 provenance다.
관측 기하는 episode 동안 고정하고 매 step 현재 `wrist_3_link` frame으로 재표현한다.
Headless 학습은 render를 호출하지 않으며 GUI·video의 native 렌더링은 별도다.

구현은 [events.py](contact_manipulation/mdp/events.py)의 catalog·직접 생성·reset,
[sensors.py](contact_manipulation/mdp/sensors.py)의 `precompute_translation_domain`과
`observe_at_viewpoint`, [scene.py](contact_manipulation/config/ur5e/scene.py)의
`inspect_geometry`·native asset 준비에 모았다. 상세 규칙은 [SPAWN_RULES.md](../../docs/SPAWN_RULES.md)를 따른다.

<details>
<summary>이전 constructive 비교 기록: DP·compaction·37/74시점 승인</summary>

아래는 `env.spawn_rules.generation_mode=constructive`로 비교할 수 있는 이전 생성 경로다.
아래 시점 후보 승인과3–10mm 이웃 상한은 현재 `parameterized` 규칙이 아니다.

이전 `spawn_rules.generation_mode="constructive"`는 공유 scene seed당 역할·종류 구성·초기 yaw를
한 번 정한다. Yaw는 각 물체의 전체360°에서 선택한다. Target 앞의 Blocker와 pivot→병진
witness의 좌표 extrema로 작업 구획을 만든다. ±Y 두 방향을 먼저 계산하고 수용 가능한 유한
영역의 합집합에서 방향·구획을 선택한다. 주변 물체를 넣을 양옆 영역의 slicing 분할을
유한 dynamic programming으로 계산한다. 폭·깊이 Pareto frontier에 임의 후보 수 제한을 두지 않는다.
이후 두 방향에서 각각 한 번 해석적 compaction을 수행한다. 이미 이동한 물체와 아직 이동하지
않은 물체를 모두 반영하고, convex configuration-space 경계에서 위치를 직접 계산한다.
충돌할 때까지 다시 난수를 뽑거나 배치를 부분 수정한 뒤 반복 승인하는 방식은 사용하지 않는다.

공유 장면 하나의 구성·구획 계산은 CPU에서 한 번 수행한다. 1,024개 환경이나 physics step마다
수행하지 않는다. 상세 collision footprint의 Minkowski 경계는 정점 edge-angle 병합으로 계산하고,
거리 계산은 compiled CPU kernel에 맡긴다. 공간 영역이 비거나 실제 메시 시점 조건이 실패하면
원인을 보고하며 같은 seed의 장면을 조용히 다시 뽑지 않는다. 이전 재시도 sampler는
`generation_mode="legacy"`를 지정한 회귀 비교용이다. 새 버전은 같은 과거 seed와 다른 배치를 만든다.

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

여기서 convex hull은 USD의 물리 collider를 재생성하는 작업이 아니다. 최초에 authored collider의
upright XY convex hull을 종류별로 캐시하고 선택한 초기 yaw로 회전한다. 가까운 이웃의 목표
간격은 이 envelope 기준3–10mm이며 Blocker 조작 공간은 별도로 예약한다. 미래의 전체 회전
여유를 물체마다 확보하는 외접원 구획은 사용하지 않는다. Concave/compound 형상의 빈 공간을
convex hull이 메우므로 정확한 PhysX 접촉 표면이나 contact offset과 동일하게 취급하지 않는다.
MDP에서는 기울어진 OBB의 8개 꼭짓점을 XY에 투영한 hull과 A₀의 교차 면적을 계산한다.
USD collider는 PhysX의 접촉 계산에 사용하며, 경로 여유·선반 점유율·A₀ 차단율을 자동으로
제공하지 않는다. 예약 경로가 있어도 물리 조작 가능성이나 회전의 필수성을 보장하지는 않는다.
기존 주변 물체1–2개를 해석적 guard 영역에 놓을 수 있으면 `guarded_pivot_translate` mode와
일반 `pivot_translate` mode를 함께 준비한다. Guarded는 초기 X·yaw를 고정한 순수±Y 양쪽
탈출이 margin0에서도 막히고 pivot→병진은 가능한 branch다. 일반 XY 병진 해법까지 배제한
판정은 아니다. 실제 메시 가시성까지 만족하는 guarded mode를 우선 선택하며 없으면 이미
평가한 일반 mode에서 고른다. Guard는128개 고정 법선의 보수적 이동 envelope 밖에 두며,
이전 packing용 큰 직사각형 안에 들어갈 수 있다. 이때 `reserved_regions=()`로 표시해
빈 직사각형 metadata와 실제 witness envelope를 구분한다. 추가 물체나 랜덤 재시도는 없다.
세부 이전 규칙은 [SPAWN_RULES.md](../../docs/SPAWN_RULES.md)의 `constructive` 기록을 따른다.

초기 관측에는 **실제 카메라·render product·annotator·참조 물체를 생성하지 않는다.**
Isaac Lab `RayCaster`와 같은 Warp mesh query를 사용한다. 물체 10종의 시각 메시를 한 번
읽어 종류별 BVH를 공유하고, 활성 물체만 검사한다. 물체 복제본마다 메시를 다시 만들거나
움직일 때 BVH를 재구성하지 않는다. **로봇·선반에 의한 가림은 제외**한 관측 모델이다.
이 제외는 ray 관측에만 적용하며, 물리 시뮬레이션과 로봇 초기 자세의 충돌 검사는 유지한다.
시점 영역·카메라 자세 생성, ray query, 가시율과 시점 선택, OBB 추정은 환경과 같은 CUDA
device에서 배치 처리한다. 물체·시점별 좌표 변환은 한 번 계산해 모든 픽셀이 공유한다.
시야 밖 광선은 Target/Blocker 메시로 바로 연결한다.

새 공유 장면마다 선반 앞 가상 시점을 배치에 결합한다. 배치 가시성 인증은 기존160×120 광선
격자와 고립 실루엣을 사용하며, Target·Blocker만 따로 조회한 고립 실루엣 대비
**Target40–90%, Blocker≥95%**를 확인한다. 분모는 같은 자세·광선 간격의2배 넓은 격자로
계산하며, 실제 시야에서 잘린 role 물체는 거절한다. 이는 렌더링 픽셀 대신 **광선 표본 면적**으로
계산한 근사 가시율이다. `initial_vision.ray_width`, `ray_height`는 이 사전 인증의 해상도다.
기본 clutter reset은 인증된 시점에서 **observation-only ray casting을 한 번** 수행한다.

실제 정책 관측은 **64×48=3,072 rays**다. 이전160×120=19,200 rays보다84% 적다.
`initial_vision.observation_ray_width`, `observation_ray_height`로 조절하며, 최소 OBB
관측점은 `observation_min_depth_pixels=3`이다. Low-resolution grid와 intrinsics는 한 번
만들어 재사용한다. 관측 전용 세 값은 배치 인증 cache key에서 제외하므로 기존 catalog를
재사용하며 관측 해상도만 바꿔90개 역할 조합을 재계산하지 않는다. 인증의32/16 pixel 조건은
유지한다. 더 낮은48×36은1,728 rays로 기존의약1/11이며 별도 관측 해상도 비교에 사용할 수 있다.
낮은 해상도에서 작은 물체의 추정 OBB나 유효 slot 수는 달라질 수 있다.

Observation의 OBB는 **첫 교차점으로 보이는 표면만** 모아 upright PCA로 추정한다. 보이지 않는
물체는 padding으로 남기며, 물체의 정답 pose·전체 크기로 가려진 부분을 채우지 않는다.
이상적인 교차점에는 depth outlier가 없으므로 이전 depth의 1–99% quantile 대신 표면 extrema를
사용한다. CUDA에서는 배경 ray를 제외한 count·moment·bounds를 Warp atomic 연산으로 계산한다. 초기 기하는
기존처럼 episode 중 고정되고 현재 EEF frame으로만 재표현한다. Actor 244D 규약은 유지한다.

이하 전체 영역 평가 설정은 사전 인증/legacy 검증 경로다. 기본 clutter reset은 새 후보
시점을 평가하거나 재시도하지 않는다.
`initial_vision.view_domain_shape=(2,9,2)`는 거리2개·좌우9개·높이2개, 고정36개 시점을 뜻한다.
배치 규칙의 시점1개를 더한다. Mode가 하나면37쌍, base/guarded가 있으면 최대74개의
mode·시점 쌍을 처음부터 모두 평가한다. `ray_batch_size=32`는 고정 메모리 chunk의
상한이며 실패 후 후보를 추가하는 재시도 횟수가 아니다. 거리0.55–0.95m, Blocker 기준 좌우
±0.35m, 선반 위0.18–0.48m의 grid를 사용하고 방향은 두 역할을 바라보도록 계산한다.
GPU에서 각 mode의 유효 시점 중 Target 가시율이65%에 가장 가까운 것을 선택한다.
동률이면 Blocker 가시율이 큰 시점, 다시 동률이면 앞선 index를 고른다. 최대2개 mode의 결과는
CPU에서 유효 guarded 우선으로 정리하며 픽셀·메시를 CPU로 가져와 재검사하지 않는다.
선택한 시점의 depth·label을 재사용해 OBB를 계산하고 선택된 mode의 물체 pose·충돌 기하·
초기 vision을 함께 확정한다. 이는 두 mode의 균등 선택이나 실패 후 새 장면 생성이 아니다.
최소 비가림 실루엣32픽셀·실제 관측16점과 전체 화각 포함 조건도 유지한다.
유효 시점이 없으면 조건별 가시율·픽셀 수·실패 항목을 보고하고 배치를 재생성하지 않는다.
고정 영역을 평가한다는 사실이 모든 종류·역할·yaw 조합의 메시 가시성을 보장하지는 않는다.
RGB는 만들지 않는다. 디버깅 시
`capture_view_domain(..., retain_images=True)`는 합성 ray depth와 role mask만 반환한다.
Headless 학습은 카메라 기능을 켜지 않고 reset에서도 render를 호출하지 않는다.
GUI·`--video`를 요청한 경우 화면 표시/녹화의 native 렌더링은 별도로 유지한다.

이 관측 모델은 표면을 불투명하게 취급하며 조명·투명/반사 재질·검출 실패·센서 노이즈를
재현하지 않는다. Triangle mesh는 원본을 사용하고 USD primitive는 tessellation한다.
물체 association도 이상적이므로 실물 vision과의 차이는 별도로 검증해야 한다.

</details>

50 mm 연결부 중앙 fixed joint의 실제 reaction을 F/T로 읽고, SI-200-8 범위 제한·양자화를 MDP에서
한 번 적용한다. 자중·관성 보상은 수행하지 않는다. Tactile은 공식 `ContactSensor` 하나가 17개 링크를 함께 읽는다.
`net_forces_w`의 `[N,17,3]`를 `TACTILE_BODY_NAMES` 순서로 정렬하고, 링크별 0.01 N
기준으로 `[N,17]` binary observation을 만든다. 힘을 링크끼리 합치지 않는다.
기존 arm·spacer 금지 접촉 센서는 제거했으며, 선반 이동 여부를 종료 조건으로 사용한다.
부분 reset 직후 해당 행의 측정값은 무효이며, 다음 물리 step의 새 측정부터 유효하다.

Step 계산은 `[N,10,...]`, `[N,12]`, `[N,17]`, `[N,18]` 텐서로 처리한다.
환경별 Python loop, `.item()`, `.cpu()`, `bool(tensor)`를 step 경로에서 사용하지 않는다.
OBB hull·clipping은 고정 크기로 병렬 처리하고, 6×6 제어 행렬 계산은 native OSC에 맡긴다.
관절/body ID·센서 scale·object local COM·catalog corner는 초기화/reset 때 저장한다.
기존 custom implicit damping·Newton–Schulz·제어 CUDA graph는 제거했다.
Native OSC의 batched mass/task-inertia 역행렬과 PD 계산을 그대로 사용한다.
따라서 SDK 내부 선형대수 호출까지 host synchronization이 없다고 보장하지 않는다.
매 step의 기하 좌표 변환·F/T 범위 제한·양자화·동적 sample 구성은 물리 데이터를 읽은 뒤 CUDA graph로 처리한다.
Action/effort의 nonfinite flag는 환경별 실패 판정으로 연결한다.
OBB footprint의 hull·clipping·면적은 활성 물체만 하나의 Warp kernel에서 계산한다.
CPU는 기존 PyTorch 식을 검증 기준으로 유지한다.

Tactile 통합 센서 1개는 전체 환경의 갱신 시점이 같은 특성을 사용해 native `nonzero()`
인덱스 탐색을 생략한다. Observation을 읽을 때 최신 native force buffer를 조회한다.
선반 위치·회전 비교도 GPU tensor로 처리하고, quaternion 내적의 절댓값을 사용해
`q`와 `-q`를 같은 회전으로 취급한다. 전체 scene sensor는 tactile과 EEF frame의 2개다.
물체의 simulation/gravity flag와 collider는 실행 중 바꾸지 않는다. 전체 공유 장면에서 활성
mask가 바뀔 때만 `set_active_objects`가 native USD kinematic 속성을 갱신·flush한 뒤
native writer가 pose·velocity를 기록한다. 미사용 물체는 기존 주차 위치의 kinematic body,
활성 물체는 dynamic body다. 같은 mask를 사용하는 부분 reset은 USD 변경·flush 없이 선택
행의 상태만 복구한다. 후보 배치 검증은 object state·활성 mask를 건드리지 않고 로봇 FK만
임시 검증·복원하므로 준비 실패 시 다른 물체나 환경을 되돌리기 위한 flag 전환도 없다.
이 방식은 collider·broadphase·transform 관리와 메모리 비용을 제거하지 않는다.
이전 simulation OFF/ON 또는 중력 상쇄를 사용하던 프로세스에 hotpatch하지 말고 새 프로세스로 실행한다.
이번 전환의4환경 fixture 검사에서는17개 tactile pad와 Blocker 접촉 필터 구분을 확인했다.

YCB 4종은 native `CollisionPropertiesCfg(contact_offset=0.005)`를 사용한다.
기존 6종의 자동 offset은 유지한다(현재 PhysX 측정 약 1.962 mm). Rest offset은 0이다.
공유 USD는 최종 충돌 설정을 bake한 reference로 교체하여 원본의 20 mm 설정이
instance 생성 뒤 되살아나지 않도록 한다. 매 step이나 reset에서 offset을 변경하지 않는다.

Hand의 원본 URDF collision mesh를 사용한다. `right_plam_1`과 네 `right_*_2`의 오목한
형상만 공식 `ConvexDecompositionPropertiesCfg`로 분해하며 palm최대16hull·finger parent각4hull,
hull당64정점으로 제한한다. 기존 Python convex proxy 생성은 제거했다. 원본17개 tactile 링크·collision mesh와
ContactReportAPI를 유지하기 위해 `merge_fixed_joints=False`를 사용한다.
두 고정 palm housing과5개 proximal 연결부의10쌍만 공식 `FilteredPairsAPI`로 제외하고,
나머지 self-collision과 pad–물체 충돌은 유지한다. 링크 간 필터는 geometry prototype 밖의
live body root에 두어 instance 공유·환경 복제 후에도 대상이 유지되도록 한다.
Hand의 native max depenetration velocity는.1m/s다. UR5e·F/T 연결·원본 질량·COM·관성은
유지하며, convex cooking은 최초 구성 때만 수행한다.
4환경 실제 시험에서17개 pad 모두 구형 물체 접촉과 binary observation을 확인했다.
Blocker/다른 물체의 filtered force 구분,접촉 유지 이력,부분reset의 다른 환경 보존도 통과했다.

GNN edge MLP의 첫 선형 연산을 node별 projection으로 분해해 `[N,10,10,134]`
concatenation과 반복 projection을 제거했다. 두 번째 선형 연산은 message 평균 이후로 옮겨
edge마다가 아닌 node마다 한 번 계산한다. 이웃이 없는 node에는 선형 bias가 유입되지 않게
처리한다. 이 기하 연산 최적화 자체는 기존 dense 식과 출력·gradient가 일치하며
Actor 파라미터와 checkpoint 구조를 바꾸지 않는다.
CUDA FP32에서는 ReLU·edge mask·합산을 Triton kernel 하나로 처리하여
`[B,10,10,64]` pair activation을 저장하지 않는다. Backward에서도 node별 입력으로
ReLU의 부호를 재계산해 receiver/sender gradient를 구하며, 큰 pair gradient도 만들지 않는다.
CPU 및 다른 dtype은 같은 PyTorch 식을 사용한다. 파라미터 이름·shape와 checkpoint는 유지한다.
Actor feature forward의 chunk 상한은 8,192다. PPO는 공식 update를 그대로 사용하며
자체 microbatch나 gradient accumulation은 없다. 1,024환경 rollout은 실제 입력
1,024개를 한 번에 처리한다. Rollout36·8epochs·4minibatches는 예제 설정을 따른다.
기본 PPO 경로는 native의 gradient clipping과 loss 전송 방식을 따른다.
따라서 native update 내부의 minibatch별 `.item()`까지 제거한 구현은 아니다.

CUDA에서는 `agent.actor.compile_model=true`가 기본이며, Actor의 GNN·attention·mask·마지막 MLP의
순수 tensor forward를 Inductor로 compile한다. Compile 자체는 모델 구조나 checkpoint key를 바꾸지 않는다.
최초 추론/학습 compile 시간이 추가된다. CUDA graph를 쓰는 `reduce-overhead`는 native 학습의
InferenceMode 전환에서 오류가 재현되어 사용하지 않는다. CPU는 eager이며,
`agent.actor.compile_model=false`로 CUDA에서도 eager 실행을 비교할 수 있다.

아래 성능 수치는 독립 Critic으로 분리하기 전 공유 모델의 개발 이력이며,
새 Actor·Critic 구성의 처리량을 측정한 결과가 아니다.
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

`events.py`의 `TaskState`는 공통 물체 속도·기하, 직전 wrench, 두 step 이력을 저장한다.
Native `common_step_counter`가 바뀐 경우에만 갱신하여 observation/reward/termination의
반복 호출로 이력이 밀리거나 시간이 중복 누적되지 않는다. 부분 reset은 선택된 행만 초기화한다.
물체 순서 정렬과 producer 데이터 검증은 reset에서만 수행한다.

Reset event와 센서 reader가 다음 데이터를 제공한다.

- `env.task_scene_data`: `SceneBatch`. Active collection IDs, Target/Blocker IDs, 승인한 초기
  vision 기하와 association, catalog OBB, A₀·workspace, F/T sensor body 이름을 담는다.
- `env.task_sensor_reader()`: 완료된 physics sample의 `SensorBatch`. 이상적 joint reaction,
  유효성과 tactile 17채널을 반환한다. 선반 pose는 native rigid-body data에서 별도로 읽는다.

`EventCfg.reset_scene`에서 배치·로봇 pose·vision을 확정한 뒤 `reset_mdp`가 이 데이터를
저장한다. 물체 pose는 reward·종료 조건에만 사용하며 초기 vision을 대신하지 않는다.
정적 USD 기하와 로봇 FK 모델은 최초 한 번만 검사한다.

전체 환경은 **8초(800 policy step)의 공통 scene round** 동안 같은 초기 물체 배치와
가상 카메라 시점·vision 정보를 공유한다. 움직이기 시작한 뒤 물리 상태는 환경별로 독립적이다.
중간 실패로 종료되면 해당 환경의 물체·선반만 저장된 초기 상태로 되돌린다.
현재 success termination은 비활성이므로 성공만으로 reset하지 않는다.
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
구성적 새 공유 배치는 유효 anchor와 실제 PhysX FK·elbow·충돌 검사를 기준으로 확정한다.
임의6D목표8개의 단일 IK 판정 중4개 이상을 요구하던 추가 gate는 구성적 경로에 적용하지 않는다.
작은 임의 표본의 실패는 시작 영역의 부재를 증명하지 않기 때문이다. 이4/8 probe는 명시적인
legacy 회귀 비교 경로에만 남기며 실제 충돌·자세 제약과 환경별 연속 randomization은 유지한다.

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
새 `parameterized` 경로는 배치·시점을 다시 추출하지 않는다. 로봇 후보 탐색과 같은 장면의
검증 anchor 복구는 별도 budget으로 유지한다.
공통 round 길이는 `episode_length_s`를 따른다.

<details>
<summary>이전 legacy 배치·시점 탐색의 연산 기록</summary>

아래 배치 후보·시점 재시도는 이전 sampler의 기록이며 새 연속 영역 경로에는 적용하지 않는다.
로봇의 GPU IK·충돌 검사와 tensor 적용은 별도로 유지한다.

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

</details>

장면 복제는 Isaac Lab의 기본 USD cloner(`clone_in_fabric=False`, `replicate_physics=True`)를
사용한다. 따라서 GUI Stage에서 모든 `env_*`를 확인할 수 있다. 마지막 scene spawner가
env_0의 완성된 rigid body 기하를 임시 USD에 저장하고 instance로 공유한 뒤 환경을 복제한다.
물리 body·joint·pose는 환경별로 독립적이며, 충돌체와 접촉 센서 설정을 적용한 결과를 공유한다.
기존 nested instance를 먼저 펼치고 외부 material binding을 함께 옮겨 로봇 메시·재질 누락을
방지한다. 임시 USD는 프로세스 종료 시 제거하며 원본 에셋은 수정하지 않는다.

구성한 초기 장면을 공통 round 안에서 재사용한다.
새 장면의 staging은 전체 환경이 reset되는 경계에서만 수행한다. 부분 reset에서는
로봇 자세 탐색을 완료한 뒤 선택 행만 쓰며, 미선택 환경의 물리 상태·episode 이력을 유지한다.
물체 기하·vision 정렬도 선택 행만 계산하며 reset snapshot에서 F/T는 초기값0을 사용한다.
공통 장면의 작은 CPU 기하 계산과 reset 시 metadata 전송은 남아 있다. 환경별 SciPy IK는 GPU 배치로 대체했고,
초기 관측의 렌더링 경로는 제거했다. 이 탐색·기하 조회·CPU/GPU 동기화는 매 step에 실행하지 않는다.
Ray 및 IK 임시 텐서 크기는 전체 환경 수와 별개로 제한한다. 배치/vision 생성은 공통 round당
한 장면에 대해서만 수행하며, IK 비용과 물리 상태 쓰기는 reset 환경 수에 따라 증가한다.
물체 local COM은 고정 asset 값으로 캐시하므로 향후 COM randomization을 추가하면 이 캐시도 갱신해야 한다.

Auto-reset 전 마지막 관측은 진단용 `extras["terminal_observation"]`에 보존한다.
학습은 예제와 같은 native PPO timeout bootstrap·GAE를 사용한다. Timeout 시 저장된
해당 step의 V를 reward에 가산하며, 이 진단 관측으로 별도 critic 추론을 하지 않는다.
`init_at_random_ep_len=True`이며 공통 장면 교체와 부분 reset은 연구 MDP에서 관리한다.

## 검증과 남은 구현

2026-10-25 `hand_velocity`에서 Hand를 제외한 뒤 CPU 집중 검사 **41개가 통과**했다.
팔3rad/s 경계와 초과·Hand/mimic 제외·12관절 velocity reward 유지·성공 차단·native 로그·부분 reset을
검사했다. 실제 GPU·simulation·학습은 실행하지 않았으며 상세는 [학습 기록70절](../../docs/TRAINING_LOG.md)을 따른다.

2026-10-24 관절 속도 종료 임계값을3rad/s로 변경한 뒤 CPU 집중 검사 **41개가 통과**했다.
±3rad/s 경계·±3.0001rad/s 초과·±2rad/s 허용, 팔/Hand 선택·성공 차단·native 로그·부분 reset을
검사했다. 실제 GPU·simulation·학습은 실행하지 않았으며 상세는 [학습 기록68절](../../docs/TRAINING_LOG.md)을 따른다.

2026-10-24 예제 기준 팔·Hand 관절 속도 종료 추가 후 CPU 전체 suite는 **415개 통과·23개 skip**이다.
임계값 경계·12관절 선택·성공 보상 차단·native 로그·부분 reset을 확인했다.
실제 GPU·simulation·학습은 실행하지 않았으며 상세는 [학습 기록67절](../../docs/TRAINING_LOG.md)을 따른다.

2026-10-24 Hand F/T 자중·관성 보상을 제거한 뒤 CPU 전체 suite는 **411개 통과·23개 skip**이다.
SI-200-8 측정값의 관측/reward 공유·이력·부분 reset 및 보상용 dynamics 조회 제거를 검사했다.
실제 GPU·simulation·학습은 실행하지 않았으며 상세는 [학습 기록66절](../../docs/TRAINING_LOG.md)을 따른다.

이 절의 과거 학습·checkpoint 검증은 2026-10-21 독립 Critic 전환 이전 구성의 기록이다.
새 구성의 검증 결과는 [학습 기록](../../docs/TRAINING_LOG.md)에 별도로 기록한다.

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
비교할 때 환경 수·rollout 길이·총 transition도 함께 비교한다. 위 수치는 과거 설정의 기록이며,
현재는 자체 PPO microbatch 설정을 제공하지 않는다.

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
Play와 GUI OBB 표시는 아래 실행 절에서 제공하며, 모델 export는 별도 구현 대상이다.

## 실행

2026-10-14 센서·IK·reset·step 추가 최적화 후 RTX 3090에서 1,024개 환경을 측정했다.
동일 설정의 첫 8 iteration 평균(native collection+learning)은 **11.15→9.07초**, 약 **19% 감소**다.
해당 16 iteration 실행은 정상 종료했으며 전체 평균은 **9.53초**였다.
이후 Python 데이터 처리를 배열로 정리한 변경은 동일 1024개 배치 생성에서 **1.72→1.57초**였다.
이 변경의 native 8 iteration 평균은 **9.18초**로, 전체 학습 시간의 추가 개선은 확인되지 않았다.
초기 환경 생성은 제외한 수치이며, **1–2초/iteration 목표에는 아직 도달하지 못했다.**
위 계측 당시에는 rollout 32·PPO 4 epochs×4 minibatches·물리 120 Hz·정책 30 Hz를 사용했다.
현재 기본값은 사용자 요청에 따라 물리100Hz·정책100Hz(`dt=0.01`, `decimation=1`)다.
현재 rollout36은 환경별0.36초, 최대8초 episode는800step이다.
OSC 토크는 물리100Hz마다 재계산하고 상대 목표는10ms 동안 유지한다. γ=.98과 action scale은
유지한다. Action-rate·F/T 증가량 제곱은 주기 불변 비용이 아니므로50Hz 결과와 구분한다.
위 성능 수치는 현재 PPO 설정과 주기로 측정한 결과가 아니다.
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
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 4 --headless --run_name independent_critic
```

1,024개 병렬 환경 학습은 다음과 같이 실행한다. 에셋 복제·JIT·초기 IK 준비가 필요하다.
기본 설정은 Git에 포함한 초기 카탈로그를 복원해 사용한다. 입력 설정이 번들과 다르거나
별도 cache 경로를 지정한 경우에는 최초 연속 영역 계산 시간이 추가될 수 있다.
한 iteration만 점검하려면 `--max_iterations 1`을 붙인다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 1024 --headless --run_name independent_critic
```

### 다른 PC의 초기 카탈로그와 IK

다른 PC의 GNN 프로젝트에서 첫 reset에 `No collision-free elbow-up pose in the bounded
batched IK search`가 발생한 사례를 확인했다. 이 시점은 학습 시작 전이며, 오류는 정해진
후보에서 모든 초기 자세 조건을 함께 만족한 해를 찾지 못했다는 뜻이다. 이 메시지만으로
GPU 모델이나 설치 오류를 원인으로 확정할 수는 없다. 앞서 parallel에만 적용했던 초기화
이식성 보완을 이 GNN 프로젝트에도 적용했다.

현재 코드를 Git으로 전달한 뒤 기존 GNN 명령을 사용한다.

```bash
git pull --ff-only
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 1024 --headless
```

`contact_manipulation/data/initial_scene_catalog/`에 기본 입력용 카탈로그를 일반 Git 파일
네 조각과 manifest로 포함한다. Git LFS나 별도 수동 cache 복사는 필요 없다. 현재 물체 형상·
배치·vision 입력 지문이 일치하면 기존 로컬 생성 cache보다 번들을 우선한다. 최초 한 번
SHA-256·경로·파일 개수·크기를 검사하고
`outputs/contact_manipulation/parameter_domains/bundled_<archive SHA>/`에 원자적으로 복원한다.
이후에는 복원본을 재사용하며 `[Scene grammar] using Git-bundled catalog` 메시지로
선택 여부를 확인한다. setuptools 설치에도 번들을 포함하고 parallel package를 import하지 않는다.

입력이 달라지면 기존 생성 경로를 사용한다. 입력 일치를 강제 확인하려면
`env.initial_catalog_fingerprint=b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`를
명령에 추가한다. `env.spawn_rules.parameter_cache_directory=...`를 명시하면 해당 경로를
사용한다. 번들은 초기 입력을 통일하며, 이후 GPU 계산의 완전한 동일성이나 IK 성공을
보장하는 장치는 아니다.

초기 자세 생성·기하 검사는 해당 범위에서만 TF32를 끄고 완료 후 원래 설정을 복원한다.
첫 장면의 기준 자세 탐색은 기존 위치27개·방향9개·branch4개의 고정 후보를 유지한다.
정상 탐색이 실패할 때만 같은 후보·같은48회 IK iteration을 tensor eager 경로로 재검사한다.
후보 수나 재시도 budget을 늘리거나 충돌·elbow-up·forearm·시작 영역·특이점 조건을
완화하지 않는다. 실제 articulation은 공식 `robot.data.body_link_pose_w`로 갱신된 링크
자세를 읽고 Float64 기준 FK와 비교한다. 위치0.2mm·회전 행렬 오차0.0003 기준을 유지한다.

실패 전용 `[Initial IK replay]` 메시지와 다음 파일에서 조건별 통과 수·입력·버전을 확인한다.
일반 reset에는 이 진단 파일 쓰기를 추가하지 않는다.

```text
logs/rsl_rl/contact_manipulation/<실행 시각>/initial_robot_start_diagnostics.jsonl
```

이 보완은 GNN·attention·관측 순서·reward·termination을 변경하지 않는다. 이번 반영은
정적 코드 검토만 수행했으며 다른 PC의 실행 성공은 아직 확인하지 않았다.

위 명령은 `parameterized` 경로와 기본 cache를 사용한다. Cache 경로와 한 번의 IK 후보 수 상한은
native Hydra 설정으로 명시할 수 있다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 4 --headless \
  env.spawn_rules.parameter_cache_directory=outputs/contact_manipulation/parameter_domains \
  env.robot_start.ik_batch_size=512
```

Launcher는 기본적으로 이 저장소의 `IsaacLab/`을 사용한다.
다른 checkout을 사용하면 `ISAACLAB_PATH`를 지정한다.

저장된 정책을 GUI에서 재생하려면 저장소 루트에서 다음과 같이 실행한다.
아래는 위 명령으로 새로 학습한 독립 Critic run의 마지막 checkpoint를 선택한다.
`--checkpoint`로 같은 구조의 특정 checkpoint 파일을 직접 지정할 수도 있다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/play.py \
  --load_run '.*_independent_critic' \
  --num_envs 4 --real-time
```

GUI의 **IsaacLab → Scene Debug Visualization**에서 각 표시를 별도로 켜고 끈다.

- **Target / Blocker OBB:** 현재 실제 물체의 Target 초록색·Blocker 빨간색 bounds.
- **Observed objects OBB:** 정책의 유효 observation slot(최대10개)에 해당하는 청록색 OBB.
  실제 전체 bounds가 아닌 초기 visible-surface 추정값이며, 물체가 움직여도 초기 snapshot을
  유지한다. 현재 EEF frame의 관측을 world로 되돌린 것과 같은 위치에 표시한다.
- **Tactile / F/T monitor:** 기본 OFF. 기존 **Environment Index**로 선택한 환경의
  17개 tactile pad와 F/T를 표시한다. GUI의 환경1은 `env_0`이다.
  손 위 pad 표면의 점과 손가락별 타일은 접촉 시 초록색, 미접촉 시 회색,
  아직 유효하지 않은 측정은 주황색이다. 접촉은 observation과 동일한 threshold의
  **모든 물체에 대한 binary**이며, Blocker-only 접촉 표시는 아니다.
  F/T 위치에는 센서 XYZ축(빨강·초록·파랑), 힘 방향 화살표(청록), 오른손 법칙의
  토크 원호(주황)를 그린다. 힘은1cm/N·최대25cm, 토크는반경6cm·1N·m당288°·최대288°다.
  패널에는0중심의 축별 양방향 막대와 Force/Torque 이력 그래프도 표시한다.
  막대는 SI-200-8 전체 범위(Fx/Fy±200N,Fz±360N,Tx/Ty/Tz±8N·m)를 사용하며,
  그래프는 처음에±20N/±1N·m로 확대하고 native **Autoscale**로 조절할 수 있다.
  값은 센서 frame의 보정된 측정 반력으로 **Hand 자중·관성 성분을 포함**한다.
  패널은 최대20Hz로 갱신하고 환경 변경·episode reset 때 그래프를 비운다.

OBB 두 표시는 기본 ON이다. 초기 OFF는 각각 `env.role_obb_debug_vis=False`,
`env.observation_obb_debug_vis=False`로 지정하며, 센서 모니터의 초기 ON은
`env.sensor_monitor_debug_vis=True`다. 세 표시는 하나의 DebugDraw callback을 공유하므로
한 표시의 OFF가 다른 표시를 지우지 않는다. 모두 OFF일 때 callback을 해제한다.
`isaacsim.util.debug_draw` 선·점만 사용하고 USD prim·rigid body·collision shape를 만들지 않는다.
모니터는 이미 계산한 센서 cache만 읽고 센서를 추가 갱신하거나 reward·observation의 이력을
변경하지 않는다. Monitor OFF에서는 데이터 조회·GPU 연산·CPU 전송이 없으며,
Headless에서는 visualizer·UI·callback을 생성하지 않는다.

PhysX의 whole-scene Patch 용량은 `gpu_max_rigid_patch_count=2**18`(262,144)로 설정했다.
기본163,840에서 관측 요구량184,896을 초과한 오류에 대응한 값이며, contact count·solver
반복·정책 및 물리100Hz는 유지한다. 이 설정은 새 시뮬레이션 생성 시 적용된다.

`--checkpoint`를 생략하면 native `get_checkpoint_path()`로 최신 run의 마지막 checkpoint를
선택한다. 이전 tanh 및 공유 actor–critic run은 계약 검사에서 거절되므로
독립 Critic 구조로 새로 학습한 run을 지정한다. 기본 환경 수는 1이고,
중간 episode 종료 시 공유 배치를 복원하고 로봇 자세를 다시 샘플링한다.
공통 round 경계에서는 학습과 동일하게 새 배치·초기 관측으로 교체한다.
`--real-time`은 실행이 빠를 때 환경의 정책 주기(기본100Hz)에 맞춰 기다리며 느린 reset을 가속하지 않는다.

Play는 `RslRlVecEnvWrapper → OnPolicyRunner.load() → get_inference_policy()` 흐름을
사용하며 탐색 noise 없이 정책의 `mean` action을 적용한다. 모델·F/T·action/output-distribution metadata와
state dict를 엄격하게 검사하고 optimizer는 불러오지 않는다. 학습 업데이트나 자동 모델
export는 수행하지 않는다. 모델 구조와 action 계약이 모두 같은 checkpoint를 재생할 수 있으며,
현재 환경 설정을 사용하므로 과거의 배치·관측 모델까지 복원하는 기능은 아니다.
필요한 설정 변경은 학습과 같은 Hydra `env.*`, `agent.*` 인자로 전달한다.

Play 기본값도100Hz다. 동일한 독립 Critic 구조로50Hz에서 학습한 checkpoint의 정책 주기를
맞추려면 `env.decimation=2 env.sim.render_interval=2`를 추가한다. 성공 유지시간은 어느 주기든
0.5초이며 기본100Hz에서는50 sample다. 과거 성공 판정까지 그대로 복원하는 것은 아니며,
Play는 checkpoint의 env.yaml에서 주기를 자동 복원하지 않는다. 마찰은 현재값(정지·동마찰0.5,
반발0)을 유지하며, 추후 물성 조정은 Hold weight·정책 주기 변경과 분리해 비교한다.

GUI 없이 지정한 step까지만 확인하려면 `--headless --max_steps 260`을 사용한다.
영상은 다음과 같이 저장하며 checkpoint 옆 `videos/play/`에 생성된다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/play.py \
  --load_run '.*_gaussian' \
  --num_envs 1 --headless --video --video_length 300
```

영상 저장에만 렌더링을 활성화하며 정책의 초기 vision은 계속 GPU mesh ray casting을 사용한다.
`--max_steps`와 `--video_length`를 함께 지정하면 더 짧은 길이에서 종료한다.
