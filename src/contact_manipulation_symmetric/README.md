# Contact Manipulation Symmetric

`src/contact_manipulation`의 현재 구현을 독립 복사한 GNN 비교 프로젝트다.
**Actor와 Critic이 모두 Actor의 동일한232D 관측값을 입력받는다.** 두 모델의
GNN·attention·MLP 가중치는 독립적이며, Critic의 출력은 value1D다.
기존 `contact_manipulation_parallel`은 별도의 MLP 실험이다.

## 원본과의 차이

| 항목 | `contact_manipulation` | 이 프로젝트 |
|---|---|---|
| Actor 물체100D | 보이는 표면에서 추정한 OBB | 동일 |
| Critic 물체100D | 최초 인식 ID의 현재 GT OBB | **Actor와 같은 OBB·가시성·padding** |
| 현재·직전 이력132D | 로봇·F/T·tactile·적용 action | 동일 |
| Actor/Critic 입력 | 각각232D, 기하 값이 다름 | 각각232D, 전체 값이 같음 |
| Critic observation contract | `tracked_initial_objects_v1` | `actor_observation_v1` |
| Task ID | `Isaac-Contact-Manipulation-UR5e-v0` | `Isaac-Contact-Manipulation-Symmetric-UR5e-v0` |
| 로그 | `logs/rsl_rl/contact_manipulation/` | `logs/rsl_rl/contact_manipulation_symmetric/` |

`policy`, `critic` 관측 그룹의 세 term은 같은 함수와 캐시를 사용한다. Actor의
주기 비전을 켜면 두 모델이 같은 시점의 갱신 결과를 읽는다. 현재 기본값0은
reset-only 비전이다. 물체 기하는 매 step 현재 sensor EEF frame으로 표현한다.
Reset·부분 reset·terminal snapshot·결측 mask도 두 입력에 동일하게 적용된다.
물리 GT를 이용하는 reward·termination은 기존 연구 규칙을 따른다.

현재 GT Critic checkpoint는 입력 의미가 달라 resume·play에서 거부한다.
이 실험은 새 학습으로 시작한다. 새 checkpoint는 자체 train/resume/play에서 사용한다.

## 유지한 모델과 설정

2026-11-06 통합 브랜치의 원본 코드 기준이며 namespace와 Critic 관측 계약 이외의 제어·물리·
배치·reward·termination·학습 설정은 동일하다. 원본 패키지를 import하지 않는다.
Asset과 Isaac Lab checkout은 저장소의 공통 경로를 사용하고, 초기 카탈로그는 독립 복사했다.

| 항목 | 기본값 |
|---|---|
| 모델 | 독립 Actor/Critic, GNN2층·64D, role attention, head256→128, ReLU |
| Action | sensor-frame wrench6 + Hand6 =12D |
| Observation | 물체100 + 현재66 + 직전66 =232D |
| 물리 / 정책 | 100Hz / 50Hz (`dt=.01`, `decimation=2`) |
| Episode / Actor 비전 | 8초 / reset-only (`actor_vision_update_interval=0`) |
| Clutter | 활성15–25개, 관측10slot |
| PPO | rollout36, epochs8, minibatches4, γ=.95, λ=.95 |
| LR / KL / entropy | 1e-3 adaptive / .02 / .01 |
| 학습 길이 / 저장 | 최대90,000 iteration / 50 iteration |
| 실행 경로 | 모델 compile=True, transition/support/robot-start CUDA graph=True |
| Time curriculum | 학습 budget의50%에서 weight0→−.3 |
| Terminal / Arm 과속 종료 | 성공+10·실패 비용0 / Arm6의abs(qdot)>1rad/s |

### 최신 reward·action 동기화

원격 변경을 병합한 원본 `contact_manipulation`과 reward·reset 상태 계산을 다시 맞췄다.

| 항목 | 현재 설정 |
|---|---|
| Clearance | 초기 Blocker 가림 대비 양의 개선, weight12 |
| Hand–Blocker distance | **현재**17pad 거리의 `mean(exp(-10*d))`, weight5·Target 보호 gate |
| Blocker relocation | 초기 빈 재배치 영역까지 현재 중심점 거리의 `exp(-10*d)`, weight3 |
| Contact hold | 새 접촉0.5·유지 접촉1의 pad 합, weight3·Target 보호 gate |
| Shelf / Target / 일반 물체 비용 | 각각−.5, 기존 수식 |
| Action-rate / Joint velocity | 각각−.03, 기존 clipping·12독립관절 선택 |
| Blocker stability / Time / Terminal | 위 원본 설정 유지 |

Distance는 이전 episode 최소거리 이력 대신 현재 거리를 평가한다. 초기 재배치 영역은
새 공유 scene을 생성할 때1회 준비하고 부분 reset에서 재사용한다. Blocker 중심이 유효 영역
내부나 경계에 있으면 relocation raw1이며, 영역이 없으면0이다. 접근 경로의 존재나 전체
Blocker footprint의 영역 진입을 보장하는 판정은 아니다. Native RewardManager가 weight와
control dt를 한 번 곱한다. 현재50Hz의 relocation 최대 지급은 step당0.06이다.

Action 관련 source는 원본과 동일함을 확인했다. Sensor-frame wrench6 + Hand6의12D,
Force10N/Torque1N·m scale, 센서 범위 clipping과 sensor→COM 원점 보정,
Hand scale.5·native joint-limit mapping, 적용 action 이력·action-rate를 그대로 사용한다.
Transition/support/robot-start의 진단용 CUDA graph 스위치도 원본과 동일하게 제공한다.

손 설정은 원본과 동일하게 URDF 정의값 보존을 기준으로 한다. 강체0/0 강제·10rad/s 상한·
motor envelope를 제거하고12관절의 원문 위치·속도·drive effort를 복원했다. Native mimic의
연동 관계와 importer compliance를 보존하며, 엄지 namespace 오류는 기존 값의 복사로 보정한다.
공식 startup event는 독립6관절의 학습용 soft position limit에만 연동 가능 범위를 적용한다.
기존 action 매핑·master PD20/1·child0/0·100Hz/solver32·1을 유지한다. 설정 일치가 실제 접촉
안정성의 보장은 아니며 역꺾임·연동 오차는 아직 미해결이다. 상세 내용은 원본 README와
`docs/TRAINING_LOG.md` 111절을 따른다.
공유 진단 스크립트 `src/contact_manipulation/tests/probe_hand_contact.py`에
`--package contact_manipulation_symmetric`를 추가하면 이 프로젝트의 실제 설정을 측정한다.

## Isaac Lab 기본 학습 흐름

```text
scripts/train.py
  → AppLauncher → Gym task 등록 → Hydra 설정
  → gym.make → ManagerBasedRLEnv와 native managers
  → RslRlVecEnvWrapper → OnPolicyRunner.learn()
  → 공식 PPO의 rollout 수집 → GAE/return → minibatch update → checkpoint
```

실행 흐름은 저장소의 `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`와 같다.
모델·하이퍼파라미터는 이 연구의 설정을 사용한다. 환경 step/reset override나 자체 PPO·
rollout storage·GAE·update loop는 추가하지 않는다. 기존 Time curriculum은 공식
CurriculumManager를 사용하고, 센서 모니터는 native logger의 기록 경로에 연결한다.
Checkpoint helper는 관측 계약을 검사한 뒤 공식 runner의 load를 호출한다.

RolloutStorage는 한 개이며 `policy`, `critic`을 각각 저장한다. 두 tensor의 값은 같고,
shape은 `[36, num_envs, 232]`다. Actor/Critic의 encoder·attention·MLP 파라미터는 공유하지 않는다.

## 실행

Isaac Lab Python 환경에서 저장소 루트를 기준으로 실행한다. 스크립트가 자체 패키지 경로를
등록하므로 editable install은 선택 사항이다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/scripts/train.py --num_envs 4 --headless
```

한 iteration 실행 검사:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/scripts/train.py \
  --num_envs 4 --max_iterations 1 --headless --run_name symmetric_smoke
```

본 학습과 재생:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/scripts/train.py --num_envs 1024 --headless

./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/scripts/play.py \
  --checkpoint logs/rsl_rl/contact_manipulation_symmetric/<run>/model_0.pt --num_envs 4
```

Asset 위치 등은 원본과 같은 `CONTACT_MANIPULATION_*` 환경변수를 사용한다.
실험별 geometry cache는 `outputs/contact_manipulation_symmetric/parameter_domains`에 저장한다.

## 검증

CPU 검사는 `tests/test_mdp.py`, `tests/test_learning.py`에 있다. 관측 전체 값의 일치,
GT 기하의 입력 차단, 부분 reset·terminal tensor 보존, 주기 비전·mask 일치, 단일 native
RolloutStorage와 PPO update, 독립 모델 가중치, checkpoint 계약을 검사한다.
실제 실행 결과와 범위는 [학습 기록](../../docs/TRAINING_LOG.md)을 참고한다.

최초 생성 시 검증(2026-11-06, 원격 변경 재동기화 이전): 전체550개 검사 중527개 통과·CUDA/Kit 전용23개 생략. 새 production
train에서4환경·1 iteration과 checkpoint 저장을 완료했다. 저장 checkpoint의 실제16step
재생에서도23회 auto-reset을 포함해 두 관측의 `[4,232]` 전체 값이 bit-exact하게 일치했다.

최신 reward 동기화 후 검증(2026-11-06): MDP491개·learning76개, 전체567개 중544개 통과·
CUDA/Kit 전용23개 생략.4환경·1 iteration의 native 학습과 새 checkpoint 저장을 완료했으며
`Episode_Reward/blocker_relocation`의 실제 지급을 확인했다.
새 checkpoint의16step 재생에서도20회 auto-reset 전후의 두 관측 전체 값이 일치했고,
실제 reward 함수 연결·weight와 초기 재배치 영역의 준비를 확인했다.
