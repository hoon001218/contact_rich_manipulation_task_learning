# Contact Manipulation Parallel

`src/contact_manipulation`에서 분리한 독립 실험 프로젝트다. Actor와 Critic 모두
Isaac Lab의 공식 `RslRlMLPModelCfg`와 RSL-RL의 기본 `MLPModel`을 사용한다.
GNN, Attention, 별도의 observation encoder, 사용자 모델 subclass는 없다.

2026-11-04 현재 `contact_manipulation`의 학습·제어·물리·배치·MDP 설정을 동기화했다.
차이는 기본 MLP 네트워크, MLP용 물체 좌우 정렬·초기 인접 교환과 프로젝트별 이름뿐이다.
원본 패키지를 import하지 않는 독립 프로젝트다. 아래 현재값은 실행 시 CLI/Hydra로 덮어쓸 수 있다.
기존 18D action/244D observation checkpoint는 새 12D/232D 환경에서 사용할 수 없으므로
새 학습을 시작한다. 실행 중인 학습에는 소급 적용되지 않는다.

## 모델

| 항목 | Actor | Critic |
|---|---|---|
| 관측 그룹 | `policy` | `critic` |
| 입력 | 232D | 232D |
| 네트워크 | 232 → 256 → 128 → 12 | 232 → 256 → 128 → 1 |
| 활성화 | ReLU | ReLU |
| 가중치 | 독립 | 독립 |
| 관측 정규화 | OFF | OFF |
| 출력 | 공식 Gaussian distribution | Value |

Actor의 Gaussian은 `std_type="scalar"`, 초기 표준편차 1.0이다.
Sampling·log probability·entropy·KL·PPO·GAE·logger·checkpoint는 공식 구현에 맡긴다.
분포에 tanh나 ±1 clipping을 추가하지 않는다. 환경에서 sensor wrench 범위와 native Hand
포화를 적용한다. Critic에는 출력 분포가 없다.

현재·직전 상태를 별도 인코딩하지 않고, 물체 100D와 그대로 연결해 기본 MLP에 입력한다.
Actor·Critic 사이에 encoder·MLP·파라미터·optimizer parameter를 공유하지 않는다.

## Observation과 물체 순서

두 그룹 모두 `물체 100D + 현재 66D + 직전 66D = 232D`다.
물체 하나는 `중심 3 + quaternion 4 + OBB 크기 3`의 10D다.
Mask나 역할 ID를 observation에 추가하지 않는다.

| 슬롯 | 내용 |
|---|---|
| 0 | Target |
| 1 | Blocker |
| 2–9 | Actor는 최근 ray cast의 주변 물체, Critic은 초기 인식 물체의 고정 ID |

순서는 다음과 같이 결정한다.

1. 실제 ray first-hit label과 보이는 표면으로 OBB를 만든다. 보이지 않은 물체를
   실제 pose로 채우지 않는다.
2. 10개보다 많은 물체가 인식되면 기존처럼 Target·Blocker를 포함하고, 두 역할 물체의
   관측 중심 중 더 가까운 쪽까지의 거리가 작은 주변 물체 8개를 고른다.
3. 선택한 주변 물체를 **ray 이미지에서 실제로 보이는 픽셀의 평균 X 좌표**로 정렬한다.
   World X, EEF X, 물체 ID, 가려지지 않은 전체 실루엣을 정렬 기준으로 사용하지 않는다.
4. 각 episode의 초기 관측에서 인접한 주변 물체끼리 확률적으로 순서를 교환한다.
   각 유효 이웃쌍의 실제 교환 확률은 20%이며, 한 물체는 최대 한 슬롯만 이동한다.
   Target·Blocker와 padding은 교환하지 않는다.
5. 같은 permutation을 Actor의 초기 OBB와 Critic의 추적 물체 ID에 적용한다.
   Actor의 후속 프레임은 현재 보이는 물체를 다시 선택·정렬하고 같은 swap 패턴을 재사용한다.
   새 난수는 뽑지 않으며, 두 슬롯이 모두 보일 때만 교환하여 padding 이동을 막는다.
   Critic의 물체 ID·순서는 reset 때 정한 그대로 유지한다.

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
교환 패턴의 추첨은 reset 행만 tensor로 처리하며, 교환 때문에 ray cast를 추가하지 않는다.

### Actor 비전 갱신과 Critic GT

- **Actor:** 기본 `actor_vision_update_interval=0`으로 reset 때만 메시 ray casting을 수행한다.
  보이는 표면의 OBB를 캐시하고 매 step 현재 sensor EEF frame으로 표현한다.
  주기를 양수로 설정하면 고정된 episode 시점에서 현재 보이는 물체를 다시 추정한다.
- **Critic:** 초기 Actor가 인식한 동일 물체 ID의 현재 실제 중심·회전·전체 OBB 정보를
  갱신하고, 현재 EEF frame으로 표현한다. 처음 보이지 않았던 물체를 새로 추가하거나
  사라진 물체를 제거하지 않는다. Actor와 별도 valid mask를 유지한다.
- **공통:** Target·Blocker는 0·1번 슬롯을 유지하고 관측 차원은 232D다. 현재·직전 로봇·센서·행동은
  132D다. 초기 reset의 직전 sample은0이다. F/T는 sensor frame에서 자중·관성을 보상한다.
  Action 이력은 실제 포화를 반영한 raw 단위의12D 명령이다.

현재 정책50Hz에서 주기5는10Hz, 주기10은5Hz 비전이다. 양수 주기에서는 env ID별 phase를
분산하고 reset 직후·종료 행·같은 step 중복 조회의 추가 ray를 건너뛴다. Critic GT만 읽으면
Actor ray를 실행하지 않는다.

기존 64×48 observation grid·메시 BVH를 재사용하고, 32개 view씩 나눠 GPU에서 처리한다.
Online 경로에는 실루엣 분모·가시성 조건 검사·시점 탐색·재시도·CPU 요청 dict·이미지 저장을
추가하지 않는다. 카메라 prim이나 render product도 생성하지 않는다. Phase 분산은 총 ray 수를
줄이는 방식은 아니며, 실제 GPU iteration 시간 증가는 아직 측정하지 않았다.
Critic·로봇 history 접근만으로는 Actor ray를 실행하지 않는다. 갱신된 관측 OBB GUI는 같은
Actor cache를 표시하고, 종료 전 관측 tensor는 갱신·부분 reset으로 수정되지 않는다.

기본값은 원본과 같은 `env.actor_vision_update_interval=0`이다. 10Hz 갱신을 선택하는 예시는 다음과 같다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_parallel/scripts/train.py \
  --num_envs 1024 --headless env.actor_vision_update_interval=5
```

## MDP와 실행 주기

| 항목 | 설정 |
|---|---|
| 물리·정책 주기 | 100 Hz / 50 Hz (`dt=0.01`, `decimation=2`) |
| Actor 비전 주기 | 0: reset 때만 |
| Episode | 최대 8초 |
| Action | Sensor-frame wrench6D + Hand6D = 12D |
| Arm 제어 | 공식 OSC `wrench_abs`/`fixed`, sensor 중심 frame, gravity compensation |
| Hand 제어 | 공식 joint-position-to-limits action, URDF master 한계·importer 기본 mimic |
| 물체 수 | 15–25개, 관측 용량 10개 |
| Reset | 기존 공통 clutter 장면과 환경별 로봇 초기 자세 randomization |
| Tactile | 17개 binary, 기존 전체 접촉 관측과 Blocker 접촉 유지 보상 |
| Reward | 선형 Clearance, 현재 성공 중 Time 면제, 성공 보상과 종료 분리 |
| 성공 종료 | 원본과 같은 유효 성공 판정 즉시 +10을 한 번 지급하고 종료 |
| 실패 | 원본과 같은 원인별 종료, terminal 실패 비용0 |
| Arm 과속 종료 | 팔6관절 중 하나라도 절댓값1rad/s 초과 |
| PPO | Rollout36, epochs8, minibatches4, γ=0.95, λ=0.95, entropy=0.01 |
| Learning rate | 초기 `1e-3`, `adaptive` 유지 |

### 현재 Reward 설정

일반 reward는 raw score를 반환하고 공식 `RewardManager`가 weight와 `dt`를 곱한다.
함수와 모든 coefficient·weight는 원본과 같다. 공식 manager가 일반 항에 `weight * step_dt`를 적용한다.

| 항목 | 계산 | Weight |
|---|---|---:|
| Clearance | `max(0, 1 - current_overlap / initial_overlap)` | 12 |
| Distance | 17개 pad별 episode 최소거리의 `exp(-10 * distance_m)` 평균 | 5 |
| Palmar contact-hold | 유지 접촉 수 + 새 접촉 수의 절반, Target 보호 gate | 0.4 |
| Shelf / Target / Object collision | 원본의 선반·Target 교란 및 주변 물체 코너 속도 비용 | 각각 −0.5 |
| Action-rate | Clipping 후12D 입력 변화 제곱합, raw 단위 | −0.03 |
| Joint velocity | 실제 팔6+Hand 독립6관절 속도 제곱합 | −0.03 |
| Force / Torque growth / Singularity reward | 비활성 | — |
| Object stability | Blocker 기울기5° 초과의 비양수 expm1 품질 | 0.5 |
| Time | 유효 성공 중 면제, 학습 절반부터 적용 | −0.3 |
| Terminal | 성공 +10 / 실패0, 일회 이벤트 | 1 |

Raw1당 Force10N·Torque1N·m이며 sensor 원점의 목표 wrench를 ±[200,200,360,8,8,8] SI로 제한한다.
Hand는 native `clamp(0.5*raw,-1,1)`을 관절 범위에 매핑한다. 관측 action 이력과 action-rate는
동등한 raw 한계 `[20,20,36,8,8,8,2,2,2,2,2,2]`를 사용한다. PPO의 원본 sample은 수정하지 않는다.
Sensor→COM moment 기준점만 기존 native OSC adapter에서 보정한다. 0 wrench는 gravity 보상만
남으며 pose hold가 아니다. F/T observation은 distal 자중·관성을 제거한 sensor-frame 반력이다.

Clearance의 분모는 초기 Blocker 전체 면적이 아니라 **초기 Blocker와 Target 접근 영역의
겹침 면적**이다. 초기에는 0, 완전 확보 시 1, 초기의 두 배를 가리면 −1이다.
진행률 상한 clip·exponential·추가 초기 점수 차감은 없다. 원점수는 악화 시 음수가 되지만
현재 지급 함수는 양수만 남기고 별도 상태 gate를 적용하지 않는다. 개선 상태를 유지하면
같은 점수를 계속 지급한다. 유효한 초기 겹침이 없으면0이다. 부분 reset은 선택 행만 갱신한다.

Time 함수의 기존 `is_alive * (~state.success)` mask를 유지한다. 현재 성공 여부를 사용하며
과거 성공 보너스 지급 이력으로 면제하지 않는다. 성공·실패의 non-timeout 종료 step에는
비용이0이다. `time_penalty_start_fraction=0.5`로 기본90000 iteration의45000부터−0.3을 적용한다.
50Hz에서 활성화된 정상 step 비용은−0.006이다. 시간 제한은8초다.

성공·실패의 구체적인 조건은 원본과 동일한 `mdp/terminations.py`를 사용한다. `success_duration=0`,
`success_reward=10`, `failure_penalty=0`이며 terminal 금액/`dt` 변환으로 성공 시 실제10을 지급한다.
Target displacement 종료와 원인별 로그도 포함한다. 순수 timeout은 truncation이다.

성공·실패 latch, F/T 이전 값, contact history는 기존 부분 reset 경로로 선택 행만 비운다.
팔6개 중 abs(qdot)>1rad/s를 포함한 모든 hard failure 기준, 기본 PPO·래퍼·timeout
bootstrap·로깅·reset 분포·Actor/Critic 관측은 유지한다. Original 프로젝트는 수정하지 않는다.

현재50Hz에서 Clearance step 보상은 `0.24 * max(0, score)`이며 완전 확보 시0.24다.
이전 수식과의 비교 자료는
[`분모·초기 기준값`](../../docs/figures/clearance_reward_comparison.png),
[`초기 겹침 분모`](../../docs/figures/clearance_overlap_normalization.png),
[`선형·지수식의 개선/악화`](../../docs/figures/clearance_linear_comparison.png)에 있다.
마지막 그림의 선형 계수0.8647은 최대값을 맞춘 비교용이며 실제 구현에는 적용하지 않는다.

상세 설계의 이전 기준 기록은
[원본 README](../contact_manipulation/README.md), [개발 맥락](../../docs/CONTEXT.md),
[배치 규칙](../../docs/SPAWN_RULES.md), [학습 기록](../../docs/TRAINING_LOG.md)에 있다.
이 기록들은 이번 MLP 변형의 새 학습 결과를 의미하지 않는다.

TensorBoard의 관절 속도·raw/보상 F/T·원본/적용 action 모니터도 원본과 동일하게 제공한다.
이번 동기화는 학습 중인 사용자의 지시에 따라 파일 수정·텍스트 검토만 수행했다.
테스트·모듈 import·학습·PhysX simulation·성능 측정은 실행하지 않았다.
이전 [83절](../../docs/TRAINING_LOG.md#83-2026-10-31--parallel-다음-학습-최종-명세-반영)의 검증 수치는
현재 설정의 실행 검증으로 사용하지 않는다.

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

### 첫 장면의 IK 실패 진단

초기 장면의 로봇 자세는 기존 위치 27개·방향 9개·IK branch 4개의 고정 후보에서 찾는다.
정상 경로에서 유효 후보가 없을 때만 **같은 장면·같은 후보·같은 48회 IK iteration**을
기존 tensor 방식의 eager 경로로 한 번 더 검사한다. 사용자 정의 CUDA graph와 통합
IK/clearance kernel을 우회하고, 재검사 중에만 TF32를 끈 뒤 원래 설정을 복원한다.
충돌·elbow-up·forearm·특이점·시작 영역 기준은 낮추지 않는다.
재검사에서 찾은 후보도 기존의 실제 articulation FK 검사를 통과해야 초기화에 사용한다.
정상 초기화나 일반 부분 reset에서는 이 진단을 실행하지 않는다.

실패 전용 콘솔 메시지 `[Initial IK replay]`와 다음 JSONL 파일에 결과를 남긴다.

```text
logs/rsl_rl/contact_manipulation_parallel/<실행 시각>/initial_robot_start_diagnostics.jsonl
```

`independent_pass`는 각 조건을 독립적으로 통과한 후보 수이며,
`cumulative_pass`는 유한 값 → IK 수렴 → 특이점 → elbow → forearm → 바닥 → 장면 충돌 →
자기충돌 → TCP 영역 순서로 누적 통과한 수다. 이 통계는 **eager 재검사 결과**이며
실패한 CUDA graph 실행의 내부 통계를 의미하지 않는다.
`evaluated`는 실제 검사한 후보 수다. 통과 후보를 찾으면 해당 행의 검사를 마치므로
항상 전체 972개와 같지는 않다. 최소 위치·회전 오차는 서로 다른 후보에서 나왔을 수 있다.

파일에는 실제 배치, 로봇·충돌 형상과 joint frame, 코드 fingerprint, seed, 규칙,
PyTorch·CUDA·Warp·Isaac Sim·SciPy 버전도 저장한다. 모델이 아직 생성되지 않은 첫 reset
오류에서도 두 PC의 입력을 비교할 수 있다. 재검사도 실패하면 초기화를 중단하며,
검사되지 않은 default 자세로 통과시키거나 물체 배치를 다시 생성하지 않는다.

## 다른 PC 설치와 실행

| 구분 | 값 |
|---|---|
| Python 패키지 | `contact_manipulation_parallel` |
| 설치 패키지 | `contact-manipulation-parallel` |
| Gym 환경 ID | `Isaac-Contact-Manipulation-Parallel-UR5e-v0` |
| 로그 | `logs/rsl_rl/contact_manipulation_parallel/<실행 시각>/` |
| 기본 배치 캐시 | `outputs/contact_manipulation_parallel/parameter_domains/` |
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
원본 로그·checkpoint·Python/JIT 캐시는 복사하지 않았다. `assets/`의 URDF·mesh는
저장소 공통 자산이다. 기존 PC의 초기 배치·가시성 카탈로그는 아래와 같이 Git에 포함한다.

### Git push·pull로 초기 카탈로그 전달

다른 PC에서는 저장소를 pull한 뒤 기존 명령으로 실행한다. 별도의 캐시 전송·압축 해제나
Git LFS 설치는 필요 없다.

```bash
git pull --ff-only
./IsaacLab/isaaclab.sh -p src/contact_manipulation_parallel/scripts/train.py \
  --num_envs 1024 --headless --seed 42
```

`contact_manipulation_parallel/data/initial_scene_catalog/`에 원본 PC의 완성된 카탈로그와
참조 가시성 JSON 6,592개를 포함한다. 재컴파일용 중간 파일은 제외했다.
총 6,593개 파일(약580MiB)을 약153MiB로 압축해 40MiB 이하의 네 조각과
`manifest.json`으로 일반 Git에서 관리한다. setuptools 설치에도 이 데이터를 포함한다.

처리는 학습 시작 시 한 번만 수행한다.

1. 현재 물체 형상·배치·vision 설정의 지문이 번들 지문과 일치하면 번들을 선택한다.
   같은 지문의 기존 로컬 생성 캐시보다 번들을 우선해 PC별 재계산 차이를 제외한다.
2. 최초 사용 시 조각·전체 SHA-256과 압축 내용의 경로·크기·개수를 검사한 뒤,
   `outputs/contact_manipulation_parallel/parameter_domains/bundled_<archive SHA>/`에
   원자적으로 복원한다. 중간에 실패한 내용을 완성된 캐시로 사용하지 않는다.
3. 이후에는 복원된 디렉터리를 재사용한다. `[Scene grammar] using Git-bundled catalog`로
   선택 여부를 확인할 수 있다. 복원이나 번들 카탈로그 검증 실패 시 재컴파일하지 않는다.

번들 출처는 2026-10-27에 읽은 원본 PC의 캐시다. 원본 캐시는 수정하지 않았다.
입력 지문은 `b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`,
전체 압축 SHA-256은 `18eba4c7e98625839d03dcc05776ccc3971313c702d3daa32e00e4f71afef3fc`다.

물체·설정을 바꿔 입력 지문이 달라지면 기존 생성 경로를 사용한다. 동일 입력인지 강제로
비교하려면 `env.initial_catalog_fingerprint=b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`를
실행 명령에 추가한다. 이때 지문이 다르면 첫 장면 생성 전에 명시적으로 중단한다.
`env.spawn_rules.parameter_cache_directory=...`를 직접 지정하면 그 디렉터리를 사용한다.

이 처리는 초기 카탈로그를 통일하며 전체 배치의 bitwise 동일성이나 IK 성공을 보장하지는
않는다. 이후 GPU 배치 계산과 IK의 차이는 실패 시 생성되는 진단 기록으로 구분한다.

초기 로봇 후보 생성·기하 검사는 TF32를 끈 FP32 행렬 연산을 사용한다. 완료 후 정책의
원래 TF32 설정을 복원한다. 실제 articulation 검증에서는 공식 `robot.data.body_link_pose_w`로
FK가 갱신된 링크 자세를 읽고, Float64 기준 FK와 비교한다. 이 조회는 Fabric 렌더링 갱신에
의존하지 않는다. 위치 0.2mm·회전 행렬 오차 0.0003 기준은 그대로 유지하며, 실패 시 가장
크게 어긋난 링크 이름·위치 오차·회전 행렬 오차·실제/예상 위치를 오류 메시지에 기록한다.

새 MLP checkpoint를 GUI에서 재생하려면 다음과 같이 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_parallel/scripts/play.py \
  --checkpoint logs/rsl_rl/contact_manipulation_parallel/<실행 시각>/model_1000.pt \
  --num_envs 4 --real-time
```

GUI의 기존 Target·Blocker OBB, observation OBB, tactile/F/T monitor 토글도 유지한다.
