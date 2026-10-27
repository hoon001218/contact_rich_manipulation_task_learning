# Contact Manipulation Parallel

`src/contact_manipulation`에서 분리한 독립 실험 프로젝트다. Actor와 Critic 모두
Isaac Lab의 공식 `RslRlMLPModelCfg`와 RSL-RL의 기본 `MLPModel`을 사용한다.
GNN, Attention, 별도의 observation encoder, 사용자 모델 subclass는 없다.

원본의 MDP·물체 자세 갱신·제어·물리·배치 생성·PPO 설정을 유지하고,
**모델을 기본 MLP로 교체하고 초기 물체 슬롯 순서만 변경한 비교 실험**이다.
원본 패키지를 import하지 않는다.

## 모델

| 항목 | Actor | Critic |
|---|---|---|
| 관측 그룹 | `policy` | `critic` |
| 입력 | 244D | 244D |
| 네트워크 | 244 → 256 → 128 → 18 | 244 → 256 → 128 → 1 |
| 활성화 | ReLU | ReLU |
| 가중치 | 독립 | 독립 |
| 관측 정규화 | OFF | OFF |
| 출력 | 공식 Gaussian distribution | Value |

Actor의 Gaussian은 `std_type="scalar"`, 초기 표준편차 1.0이다.
Sampling·log probability·entropy·KL·PPO·GAE·logger·checkpoint는 공식 구현에 맡긴다.
추가 tanh나 action clipping은 없다. Critic에는 출력 분포가 없다.

현재·직전 상태를 별도 인코딩하지 않고, 물체 100D와 그대로 연결해 기본 MLP에 입력한다.
Actor·Critic 사이에 encoder·MLP·파라미터·optimizer parameter를 공유하지 않는다.

## Observation과 물체 순서

두 그룹 모두 `물체 100D + 현재 72D + 직전 72D = 244D`다.
물체 하나는 `중심 3 + quaternion 4 + OBB 크기 3`의 10D다.
Mask나 역할 ID를 observation에 추가하지 않는다.

| 슬롯 | 내용 |
|---|---|
| 0 | Target |
| 1 | Blocker |
| 2–9 | 초기 ray cast에서 인식된 나머지 물체, 화면 좌→우 순서 |

순서는 다음과 같이 결정한다.

1. 실제 ray first-hit label과 보이는 표면으로 초기 OBB를 만든다. 보이지 않은 물체를
   실제 pose로 채우지 않는다.
2. 10개보다 많은 물체가 인식되면 기존처럼 Target·Blocker를 포함하고, 두 역할 물체의
   관측 중심 중 더 가까운 쪽까지의 거리가 작은 주변 물체 8개를 고른다.
3. 선택한 주변 물체를 **ray 이미지에서 실제로 보이는 픽셀의 평균 X 좌표**로 정렬한다.
   World X, EEF X, 물체 ID, 가려지지 않은 전체 실루엣을 정렬 기준으로 사용하지 않는다.
4. 각 episode의 초기 관측에서 인접한 주변 물체끼리 확률적으로 순서를 교환한다.
   각 유효 이웃쌍의 실제 교환 확률은 20%이며, 한 물체는 최대 한 슬롯만 이동한다.
   Target·Blocker와 padding은 교환하지 않는다.
5. 같은 permutation을 Actor의 초기 OBB와 Critic의 추적 물체 ID에 적용한다.
   이후 해당 episode 동안 슬롯 순서를 다시 정렬하거나 교환하지 않는다.

겹치는 이웃쌍을 동시에 교환하지 않도록 매 reset에서 짝수/홀수 이웃쌍 그룹을 무작위로
선택하고, 선택된 그룹에서 40% 확률로 교환한다. 따라서 각 유효 이웃쌍의 최종 확률은
`0.5 × 0.4 = 0.2`다. 물리 장면을 공유하더라도 교환은 reset되는 환경별로 수행한다.
누락 슬롯은 계속 0으로 유지한다.

교환 확률은 env cfg의 `observation_adjacent_swap_probability=0.2`로 설정한다.
0이면 완전한 좌→우 정렬을 유지한다. 인접 교환의 비중복 조건 때문에 허용 범위는 0–0.5다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_parallel/scripts/train.py \
  --num_envs 1024 --headless env.observation_adjacent_swap_probability=0.0
```

좌우 정렬은 GPU tensor의 scatter reduction·argsort·gather로 수행한다.
픽셀×물체 크기의 비교 tensor나 환경·물체별 Python 반복문을 추가하지 않는다.
이웃 교환도 reset 행만 tensor로 처리하며 추가 ray cast를 실행하지 않는다.

### 물체 자세 갱신은 기존과 동일

- **Actor:** 초기 visible-surface OBB의 장면 좌표를 고정한다. 매 step 현재 UR5e 말단
  EEF frame으로 변환하며, 물체가 움직였다는 이유로 초기 vision을 갱신하지 않는다.
- **Critic:** 초기 Actor가 인식한 동일 물체 ID의 현재 실제 중심·회전·전체 OBB 정보를
  갱신하고, 현재 EEF frame으로 표현한다. 처음 보이지 않았던 물체를 새로 추가하거나
  사라진 물체를 제거하지 않는다.
- **공통:** 물체 ID·slot·padding은 episode 동안 고정이다. 현재·직전 로봇·센서·행동
  144D는 기존과 동일하다. 초기 reset의 직전 sample은 0이다. F/T는 센서 frame을 사용한다.

## MDP와 실행 주기

| 항목 | 설정 |
|---|---|
| 물리·정책 주기 | 100 Hz / 100 Hz (`dt=0.01`, `decimation=1`) |
| Episode | 최대 8초 |
| Action | UR5e 상대 pose 6D + 가변 stiffness 6D + Hand 6관절 = 18D |
| Arm 제어 | 공식 OSC, UR5e 말단 EEF/task frame, gravity compensation |
| Hand 제어 | 공식 joint-position-to-limits action, URDF 범위·mimic 반영 |
| 물체 수 | 15–25개, 관측 용량 10개 |
| Reset | 기존 공통 clutter 장면과 환경별 로봇 초기 자세 randomization |
| Tactile | 17개 binary, 기존 전체 접촉 관측과 Blocker 접촉 유지 보상 |
| Reward·종료 조건 | 복사 시점 원본과 동일 |
| PPO | Rollout 36, epochs 8, minibatches 4, γ=0.98, λ=0.95 |

원본의 거리·Clearance·접촉 유지·안정성·움직임 비용·Time·성공/실패 수식과
weight는 변경하지 않았다. 상세 설계의 기준 기록은
[원본 README](../contact_manipulation/README.md), [개발 맥락](../../docs/CONTEXT.md),
[배치 규칙](../../docs/SPAWN_RULES.md), [학습 기록](../../docs/TRAINING_LOG.md)에 있다.
이 기록들은 이번 MLP 변형의 새 학습 결과를 의미하지 않는다.

## 코드 구조

```text
contact_manipulation_parallel/
├── pyproject.toml
├── scripts/
│   ├── train.py
│   └── play.py
├── tests/
│   ├── test_learning.py
│   └── test_mdp.py
└── contact_manipulation_parallel/
    ├── shelf_manipulation_env_cfg.py
    ├── config/ur5e/
    │   ├── scene.py
    │   ├── ur5e_env_cfg.py
    │   └── agents/rsl_rl_ppo_cfg.py
    └── mdp/
        ├── events.py
        ├── sensors.py
        ├── observations.py
        ├── rewards.py
        └── terminations.py
```

Cfg가 공식 모델과 manager term을 선언하고, native runner·manager가 실행한다.
기존 graph 전용 `model.py`와 checkpoint 변환용 `ppo.py`는 제거했다.
Train resume와 Play는 공식 `runner.load()`를 사용한다.
GNN/Attention checkpoint는 입력 구조와 state dict가 다르므로 이 MLP 프로젝트에서
이어 학습하지 않고 새로 시작한다.

## 다른 PC 설치와 실행

| 구분 | 값 |
|---|---|
| Python 패키지 | `contact_manipulation_parallel` |
| 설치 패키지 | `contact-manipulation-parallel` |
| Gym 환경 ID | `Isaac-Contact-Manipulation-Parallel-UR5e-v0` |
| 로그 | `logs/rsl_rl/contact_manipulation_parallel/<실행 시각>/` |
| 기본 배치 캐시 | `src/outputs/contact_manipulation_parallel/parameter_domains/` |
| 환경변수 접두어 | `CONTACT_MANIPULATION_PARALLEL_` |

저장소 루트에 `IsaacLab/`, `assets/robots/Roas_provided_inspire_urdf/`,
`src/contact_manipulation_parallel/` 구조를 유지한다. 다른 Isaac Lab 설치 경로는
`ISAACLAB_PATH`로 지정할 수 있다.

복사 시점의 실행 환경은 Python 3.11, Isaac Sim 5.1.0, RSL-RL 5.0.1,
PyTorch 2.7.0+cu128, Warp 1.15.0이다.
Isaac Lab checkout은 `d6a18c1fb4f317f765788cebea6b4513d1e32906`이다.
이 환경을 준비한 뒤 저장소 루트에서 설치하고 실행한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e ./src/contact_manipulation_parallel

./IsaacLab/isaaclab.sh -p src/contact_manipulation_parallel/scripts/train.py \
  --num_envs 1024 --headless
```

기본 선반·UR5e·물체 USD는 `omniverse://192.168.0.13`에서 읽는다.
다른 PC에서도 서버에 접근하거나 `CONTACT_MANIPULATION_PARALLEL_NUCLEUS_ROOT` 및
개별 `CONTACT_MANIPULATION_PARALLEL_*_USD_PATH`로 위치를 지정해야 한다.
Hand 경로는 `CONTACT_MANIPULATION_PARALLEL_HAND_URDF_PATH`로 바꿀 수 있다.
원본 로그·checkpoint·배치/Python/JIT 캐시는 복사하지 않았으며, 첫 실행에서 필요한
캐시를 생성한다. `assets/`의 URDF·mesh는 저장소 공통 자산이다.

새 MLP checkpoint를 GUI에서 재생하려면 다음과 같이 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_parallel/scripts/play.py \
  --checkpoint logs/rsl_rl/contact_manipulation_parallel/<실행 시각>/model_1000.pt \
  --num_envs 4 --real-time
```

GUI의 기존 Target·Blocker OBB, observation OBB, tactile/F/T monitor 토글도 유지한다.
