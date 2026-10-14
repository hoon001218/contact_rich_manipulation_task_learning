# Contact Manipulation

Sweeping-Policy 예제의 구조를 기준으로 새로 시작하는 Isaac Lab 프로젝트다.
**장면·센서 생성, episode reset, MDP, GNN·attention 정책과 native PPO를 연결한 프로젝트다.**
기존 `src/contact_rearrangement`와 별도 패키지이고, 해당 패키지를 import하지 않는다.

참조한 코드는 [Sweeping-Policy](../../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/)와
[Isaac Lab manager-based 템플릿](../../IsaacLab/tools/template/templates/tasks/manager-based_single-agent/env_cfg)이다.

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
    ├── shelf_manipulation_env_cfg.py # 공통 Scene 및 MDP manager 설정
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
        ├── actions.py              # native OSC·joint-position action의 연구 입력 변환
        ├── sensors.py              # 실제 F/T·contact reader와 기하 기반 초기 vision
        ├── observations.py         # observation 계산
        ├── rewards.py              # reward 계산
        └── terminations.py         # 성공·실패 조건
```

## 예제와의 대응

| Sweeping-Policy | 새 프로젝트 | 역할 |
|---|---|---|
| `shelf_sweep_env_cfg.py` | `shelf_manipulation_env_cfg.py` | 장면과 각 manager 설정 조합 |
| `config/ur5e/joint_pos_env_cfg.py` | `config/ur5e/ur5e_env_cfg.py` | 로봇·센서·제어 설정 구체화 |
| `config/ur5e/__init__.py` | 동일 위치 | Gym 등록과 설정 파일 연결 |
| `config/ur5e/agents/rsl_rl_ppo_cfg.py` | 동일 위치 | 정책 및 PPO 설정 |
| `mdp/*.py` | 동일 위치 | 연구에 필요한 MDP 함수 |

로봇별 설정은 `ur5e_env_cfg.py`에서 구체화한다. Arm에는 연구 설계에 따라
가변 stiffness의 OSC torque 제어를, hand에는 독립 6관절 position 제어를 사용한다.
에셋 로딩은 예제의 `src/ur5e.py`, `src/shelf_utils.py`를 참고해 `scene.py`에 모았다.

## Native 기능과 연구 확장의 경계

새 기능은 설치된 Isaac Lab의 public class/config와 공식 예제를 먼저 확인한다.
직접 구현은 연구 규칙 또는 측정된 병목 때문에 필요한 부분에 한정한다.
새 파일을 늘리지 않고 기존 `mdp/`, `config/ur5e/`, `agents/`에서 확장한다.

| 영역 | 현재 사용하는 native 기능 | 남겨 둔 연구 확장 |
|---|---|---|
| 환경·학습 흐름 | `ManagerBasedRLEnv`, 각 Manager, `InteractiveScene`, `OnPolicyRunner` | Scene/MDP 설정과 함수 |
| 제어 | `OperationalSpaceControllerAction`, `JointPositionToLimitsAction`, `ImplicitActuator` | 18D 입력 변환, TCP 보정, 제한 처리 |
| 좌표 연산 | `isaaclab.utils.math`의 quaternion 연산 | 배치 차원의 broadcasting adapter |
| EEF·접촉 센서 | `FrameTransformer`, `ContactSensor` | 매 physics tick 전체 행 갱신 최적화, 금지 접촉 상대 필터 |
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
- F/T 원시 반력은 native articulation API에서 읽는다. SI-200-8 양자화·범위와 hand 링크의
  자중/운동 관성 보상, OBB 조작량·clearance 등은 연구 수식이다.
- 설치된 RSL-RL은 기본 Gaussian을 제공한다. 기존 bounded tanh-Gaussian과 그 log-prob,
  GNN/attention은 연구 모델로 유지한다. PPO 알고리즘 자체의 기본 실행은 native에 위임한다.

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

## 모델과 PPO

```text
초기 vision 기하 [N,30,10] ── 2-layer GNN (64D)
                             ├─ Target / Blocker embedding
                             ├─ Target / Blocker 조건부 attention → scene 320D
                             └─ valid object mean pooling
현재·직전 sample [N,2,72] ── 결측 masking → flatten 144D (별도 encoder 없음)
                                   ↓
                      concat 464 → MLP 256 → latent 128
                                   ├─ tanh-Gaussian actor: 18D
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
| Max gradient norm / 초기 action std | 0.5 / 0.3 (tanh 이전) |
| 추론 chunk / 학습 microbatch | 최대 1,024 / 8,192 sample |
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
`actions.py`, `sensors.py`에 모았다. 공통 상태·배치 sampler·reset을 별도 파일로 분리하지 않았다.

| 영역 | 구현 |
|---|---|
| Observation | 초기 vision 기하 30×10 + 현재·직전 sample 2×72 = **444D** |
| 물체 정보 | 현재 EEF 기준 위치·quaternion·OBB extent. 초기 추정값은 episode 내내 고정 |
| 슬롯 | Target 0, Blocker 1, 나머지 관측 물체와 zero padding. Role ID·mask 채널 없음 |
| 72D sample | 12관절 q/dq 24 + base 기준 EEF pose 7 + sensor 기준 F/T 6 + tactile 17 + 직전 적용 action 18 |
| F/T | SI-200-8 축별 clipping·양자화 → distal 링크 자중·병진/회전 관성 보상 → full scale 정규화 |
| Tactile | 마지막 physics sample의 normal force norm > 0.01 N이면 1, 미접촉 0, 결측 −1 |
| Action | Base 축 TCP 증분 6 + stiffness 6 + hand 독립 관절 위치 6 = **18D** |

Action은 축별 위치 ±15 mm, rotation vector ±0.08 rad,
`arm_action` 12D와 `hand_action` 6D를 native ActionManager가 순서대로 처리한다.
- Arm: `ArmOscAction(OperationalSpaceControllerAction)` → Isaac Lab `OperationalSpaceController`.
- Hand: `HandPositionAction(JointPositionToLimitsAction)` → native `ImplicitActuator`의 position PD.

OSC는 `pose_rel`, `variable_kp`, full inertia decoupling, gravity compensation,
damping ratio 1을 사용한다. Gain 범위는 병진 100–800, 회전 5–40이며,
**관성으로 정규화된 gain(s⁻²)**이다. 이전 custom controller의 물리 강성
N/m·N·m/rad와 같은 의미가 아니다. Native 계산은
`τ = Jᵀ Λ [Kp e − 2√Kp v] + G`, `Λ = (J M⁻¹ Jᵀ)⁻¹`이고 custom damping은 없다.

TCP 목표는 native OSC가 매 policy step의 측정 pose에서 생성하고 physics substep 동안 유지한다.
연구 adapter는 입력 정규화·축별 gain 매핑·workspace/effort 제한과 TCP의 COM→link 좌표 보정만 수행한다.
Hand는 soft limit 범위의 절대 관절 목표를 사용하며, 측정 위치에서 `1 rad/s × physics_dt` 이하로
PD target의 lead를 제한한다(실제 관절 속도의 보장은 아니다).
Arm PD gain은 asset 설정에서 0이며 mimic 관절에는 별도 action을 주지 않는다.
Action 18D와 observation 444D의 채널 순서는 유지하지만 제어 응답이 바뀌므로,
이전 controller로 학습한 checkpoint를 같은 동역학의 continuation으로 해석하면 안 된다.

| Reward | Discrete step 값 |
|---|---|
| Clearance | `exp(−blocked_area / blocker_area)`; 초기 접근영역 A₀와 Blocker의 실제 footprint 교차 면적 |
| Rearrangement | `exp(−2·L_OBB)`; 모든 active 물체의 평균 8-corner 이동거리 합을 누적 |
| Motion | `(exp(−C_acc) + exp(−C_increase)) / 2`; 12관절 가속도 및 힘·토크 norm 증가율 |
| Time | `−∫[t,t+dt] 0.02(exp(0.7s)−1) ds` |
| Terminal | 성공 +500, 실패 −100, 실패 우선·episode당 1회 |

Motion의 정규화 기준은 arm 5 / hand 10 rad/s², 힘 증가율 200 N/s,
토크 증가율 10 N·m/s다. 일정한 속도·하중과 하중 감소에는 이 비용이 없다.
F/T 증가율은 world 축·episode 초기 sensor 원점으로 변환한 보상값을 비교한다.
각 reward weight는 `RewardsCfg`에서 설정하며 기본 1이다. 함수는 `step_dt`로 나눈
값을 반환하므로 native RewardManager의 `weight × dt` 후 위 discrete 값이 지급된다.

성공은 모든 non-target의 A₀ 교차 면적 합 ≤ 1e−5 m², Target 이동 ≤ 2 cm·회전 ≤
0.15 rad, 물체 선속도 ≤ 0.015 m/s·각속도 ≤ 0.1 rad/s를 5 step 연속 만족할 때다.
낙하 6 cm, lift 4 cm, tilt 50°, workspace 이탈, 보상 F/T 80 N·4 N·m 초과,
raw F/T 측정 범위 초과, arm/spacer 금지 접촉, 유효하지 않은 물리·제어 상태는 실패다.
Timeout은 native truncation 항으로 유지한다.

## 병렬 계산과 reset 연결

```text
환경 생성 1회: robot / shelf / 물체 pool / 17 tactile / 금지 접촉 / 공유 ray-casting mesh 준비
reset(env_ids): 선택 환경마다 새 Target/Blocker·물체 배치 생성
              → 제안한 물체 pose로 GPU ray casting·가시성 검증
              → 통과한 행만 GPU 배치 IK·elbow-up·전체 로봇 충돌 검사
              → 선택 행의 물리 상태를 tensor로 일괄 적용·실제 FK 확인
              → 실패한 행만 같은 역할·구성으로 재시도
              → reset_mdp: 초기 vision·역할·누적 reward·직전 이력 확정
```

전체 초기화와 일부 환경의 auto-reset은 **같은 생성 경로**를 사용한다. 매 episode마다 해당
환경의 물체 배치·역할·로봇 자세·가상 시점을 새로 샘플링한다. 기존 공통 배치와 4개 초기 상태
모음은 제거했다. Seed는 기본 seed, 환경 ID, 해당 환경의 reset 횟수로 정하므로 다른 환경의
종료 시점이나 reset 대상의 순서가 그 환경의 난수열을 바꾸지 않는다.

배치 전에 Target/Blocker를 정하고, Target과 같은 종류는 추가 spawn하지 않는다.
10종 중 4–10종을 골라 **선반 전체 점유율 ≥35%**, 2×3 분할 영역별 ≥10%를 만족하도록
개수를 늘린다. Footprint 면적을 가장자리 여유를 제외한 선반 면적으로 나눈 기준이다.
복제 개수는 적게 사용된 종류부터 균형 있게 추가한다. 10종×3개인 **30개 고정 물리 pool**을
생성하며, 불가능한 종류 조합은 점유율을 낮추지 않고 제한된 횟수 안에서 다시 제안한다.
Target 종류를 하나만 쓰므로 이 pool의 active 개수 상한은 28개이고 관측 capacity는 30개다.
이전 40-slot checkpoint는 모델 설정과 observation 크기가 달라 기본 resume/play와 호환되지 않는다.

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
기존처럼 episode 중 고정되고 현재 EEF frame으로만 재표현한다. Actor 444D 규약은 유지한다.

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
한 번 적용한다. 17개 tactile sensor는 각 링크에 독립적으로 연결한다. Arm·spacer 금지 접촉은
9개 sensor가 물체별 partner force와 최근 4개 physics substep을 검사한다.
부분 reset 직후 해당 행의 측정값은 무효이며, 다음 물리 step의 새 측정부터 유효하다.

Step 계산은 `[N,30,...]`, `[N,12]`, `[N,17]`, `[N,18]` 텐서로 처리한다.
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

17 tactile·9 forbidden 센서는 매 physics tick 전체 환경을 갱신하는 특성을 사용해 native
`nonzero()` 인덱스 탐색을 생략한다. Native force/history buffer와 접촉 상대별 판정은 유지한다.
비활성 물체의 중력 상쇄는 reset 시 native permanent wrench composer에 설정한다.
매 step force 적용과 transient wrench 합성은 native collection이 수행한다.
실행 중 simulation/gravity flag를 전환하지 않는다. Permanent composer는 주차용으로 예약한다.

GNN edge MLP의 첫 선형 연산을 node별 projection으로 분해해 `[N,30,30,134]`
concatenation과 반복 projection을 제거했다. 두 번째 선형 연산은 message 평균 이후로 옮겨
edge마다가 아닌 node마다 한 번 계산한다. 이웃이 없는 node에는 선형 bias가 유입되지 않게
처리한다. 기존 dense 식과 출력·gradient가 일치하며 모델 파라미터와 checkpoint 구조도 동일하다.
CUDA FP32에서는 ReLU·edge mask·합산을 Triton kernel 하나로 처리하여
`[B,30,30,64]` pair activation을 저장하지 않는다. Backward에서도 node별 입력으로
ReLU의 부호를 재계산해 receiver/sender gradient를 구하며, 큰 pair gradient도 만들지 않는다.
CPU 및 다른 dtype은 같은 PyTorch 식을 사용한다. 파라미터 이름·shape와 checkpoint는 유지한다.
늘어난 메모리 여유를 사용해 추론 chunk는 1,024, PPO microbatch는 8,192로 설정했다.
Rollout·epoch·minibatch 수와 optimizer 갱신 횟수는 그대로다.
기본 PPO 경로는 native의 gradient clipping과 loss 전송 방식을 따른다.
따라서 native update 내부의 minibatch별 `.item()`까지 제거한 구현은 아니다.

`events.py`의 `TaskState`는 누적 조작량, 직전 속도·wrench, 두 step 이력을 저장한다.
Native `common_step_counter`가 바뀐 경우에만 갱신하여 observation/reward/termination의
반복 호출로 이력이 밀리거나 시간이 중복 누적되지 않는다. 부분 reset은 선택된 행만 초기화한다.
물체 순서 정렬과 producer 데이터 검증은 reset에서만 수행한다.

Reset event와 센서 reader가 다음 데이터를 제공한다.

- `env.task_scene_data`: `SceneBatch`. Active collection IDs, Target/Blocker IDs, 승인한 초기
  vision 기하와 association, catalog OBB, A₀·workspace, F/T distal body 이름을 담는다.
- `env.task_sensor_reader()`: 완료된 physics sample의 `SensorBatch`. 이상적 joint reaction,
  유효성, tactile 17채널, partner별 force history에서 판정한 금지 접촉을 반환한다.

`EventCfg.reset_scene`에서 배치·로봇 pose·vision을 확정한 뒤 `reset_mdp`가 이 데이터를
저장한다. 물체 pose는 reward·종료 조건에만 사용하며 초기 vision을 대신하지 않는다.
정적 USD 기하와 로봇 FK 모델은 최초 한 번만 검사한다. Reset 준비 연산은 다음과 같이 나눈다.

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
  가시성·IK를 통과한 `[M,...]` 행을 asset별 한 번씩 쓰고 PhysX link pose를 검증한다.

배치의 XY/yaw tensor는 IK 충돌 OBB와 물리 초기 상태 생성까지 GPU에 유지한다.
독립 검증·debug용 CPU pose 사본은 실제로 조회할 때만 생성한다.
IK 결과의 관절값·TCP·시작 영역도 tensor로 전달한다. 종류/instance ID와 지지 높이,
역할·경로 후보 준비 및 후보 승인 인덱스의 CPU 처리는 남아 있다.
물리 pose 조립의 ID/slot은 평탄화한 배열로 준비하며 지지 높이는 catalog별 GPU tensor로 캐시한다.
GPU IK 난수 생성기로 바뀌어 이전 코드와 같은 seed의 구체적인 시작 자세는 달라진다.
각 환경/episode의 독립성과 동일 버전에서의 재현성은 유지한다.

각 layout에서 가상 시점은 최대 32회 제안하며, 실패한 환경만 최대 32개 layout까지
재시도한다. 재시도 중 Target·Blocker, 물체 종류·instance 집합과 가시율 기준은 유지한다.
1024개 초기 reset에서는 기존 layout 8회 한도로 드문 실패가 남아 한도를 늘렸다.

장면 복제는 Isaac Lab의 기본 USD cloner(`clone_in_fabric=False`, `replicate_physics=True`)를
사용한다. 따라서 GUI Stage에서 모든 `env_*`를 확인할 수 있다. 마지막 scene spawner가
env_0의 완성된 rigid body 기하를 임시 USD에 저장하고 instance로 공유한 뒤 환경을 복제한다.
물리 body·joint·pose는 환경별로 독립적이며, 충돌체와 접촉 센서 설정을 적용한 결과를 공유한다.
기존 nested instance를 먼저 펼치고 외부 material binding을 함께 옮겨 로봇 메시·재질 누락을
방지한다. 임시 USD는 프로세스 종료 시 제거하며 원본 에셋은 수정하지 않는다.

배치의 종류·밀집도·충돌·가시성 기준을 낮추거나 승인된 장면을 다른 episode에 재사용하지 않는다.
Reset하지 않는 환경을 임시 촬영 장면으로 쓰거나 전체 scene의 force buffer를 flush하지 않으며,
준비 실패 시 선택 행의 물리 상태를 복구한다. 물체 기하·F/T 보상·vision 정렬도 선택 행만 계산한다.
CPU 배치 탐색과 reset 시 metadata 전송은 남아 있다. 환경별 SciPy IK는 GPU 배치로 대체했고,
초기 관측의 렌더링 경로는 제거했다. 이 탐색·기하 조회·CPU/GPU 동기화는 매 step에 실행하지 않는다.
Ray 및 IK 임시 텐서 크기는 전체 환경 수와 별개로 제한한다. 총 reset 비용에는 생성할 episode 수와
배치·IK·가시성 재시도 횟수가 영향을 준다.
질량/관성은 고정 asset 값으로 캐시하므로 향후 dynamics randomization을 추가하면 이 캐시도 갱신해야 한다.

Auto-reset 전 마지막 관측은 `extras["terminal_observation"]`에 보존한다.
`ppo.py`가 이 관측으로 timeout value를 계산하며 같은 step의 물리적 성공·실패는 bootstrap에서 제외한다.
Native reset의 `terminal_env_ids`를 재사용해 종료 환경이 없는 step의 추가 critic 추론을 생략한다.
Reset 후의 새 episode 관측을 이전 episode의 value로 사용하지 않는다.

## 검증과 남은 구현

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
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 4 --headless
```

1,024개 병렬 환경 학습은 다음과 같이 실행한다. 초기 독립 배치·가시성 탐색 때문에
첫 학습 로그까지 수 분이 걸릴 수 있다. 한 iteration만 점검하려면 `--max_iterations 1`을 붙인다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py --num_envs 1024 --headless
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
`--checkpoint`에 재생할 파일을 지정한다. 아래는 저장된 iteration 2,050 모델의 예시다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/play.py \
  --checkpoint logs/rsl_rl/contact_manipulation/2026-10-13_23-18-25/model_2050.pt \
  --num_envs 4 --real-time
```

`--checkpoint`를 생략하면 native `get_checkpoint_path()`로 최신 run의 마지막 checkpoint를
선택한다. 특정 run은 `--load_run 2026-10-13_23-18-25`로 지정한다. 기본 환경 수는 1이고,
episode 종료 시 학습과 같은 환경의 auto-reset으로 새 배치·초기 관측을 생성한다.
`--real-time`은 실행이 빠를 때 환경의 정책 주기(기본 50 Hz)에 맞춰 기다리며 느린 reset을 가속하지 않는다.

Play는 `RslRlVecEnvWrapper → OnPolicyRunner.load() → get_inference_policy()` 흐름을
사용하며 탐색 noise 없이 정책의 `tanh(mean)` action을 적용한다. 모델·F/T metadata와
state dict를 엄격하게 검사하고 optimizer는 불러오지 않는다. 학습 업데이트나 자동 모델
export는 수행하지 않는다. 최적화 전후 모델 구조가 동일한 checkpoint는 재생할 수 있으나,
현재 환경 설정을 사용하므로 과거의 배치·관측 모델까지 복원하는 기능은 아니다.
필요한 설정 변경은 학습과 같은 Hydra `env.*`, `agent.*` 인자로 전달한다.

GUI 없이 지정한 step까지만 확인하려면 `--headless --max_steps 260`을 사용한다.
영상은 다음과 같이 저장하며 checkpoint 옆 `videos/play/`에 생성된다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/play.py \
  --checkpoint logs/rsl_rl/contact_manipulation/2026-10-13_23-18-25/model_2050.pt \
  --num_envs 1 --headless --video --video_length 300
```

영상 저장에만 렌더링을 활성화하며 정책의 초기 vision은 계속 GPU mesh ray casting을 사용한다.
`--max_steps`와 `--video_length`를 함께 지정하면 더 짧은 길이에서 종료한다.
