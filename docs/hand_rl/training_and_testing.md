# Hand RL 학습 및 테스트 가이드

## 1. 환경 준비

모든 명령은 repository root에서 실행한다.

```bash
cd /home/hoon/workspace/contact_rich_manipulation_task_learning
conda activate env_isaaclab
```

현재 checkout의 `hand_rl`을 editable mode로 설치한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/hand_rl
```

기본 asset 경로는 다음과 같다.

- UR5e: `omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.0/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd`
- Hand: `assets/robots/Roas_provided_urdf/urdf/urdf_left_with_force_sensor.urdf`

다른 asset을 사용할 때만 환경변수를 지정한다.

```bash
export HAND_RL_UR5E_USD_PATH='omniverse://server/path/to/ur5e.usd'
export HAND_RL_HAND_URDF_PATH='/absolute/path/to/hand.urdf'
```

기본 Nucleus 경로를 사용하려면 `192.168.0.13` 서버에 로그인된 상태여야 한다.

## 2. 빠른 smoke test

본 학습 전에 environment 생성, UR5e-hand 조립, observation/action manager,
PPO update와 checkpoint 경로를 검증한다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/train.py \
  --num_envs 1 \
  --max_iterations 1 \
  --device cuda:0 \
  --headless
```

정상 실행 시 다음 항목을 확인한다.

- Task: `Isaac-Hand-RH56-Grasp-v0`
- Action shape: `12`
- Policy observation shape: `59`
- Reward terms: `7`
- `Learning iteration 0/1` 완료

## 3. 본 학습

RTX 3090 기준 시작 설정은 1024 environments다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/train.py \
  --num_envs 1024 \
  --device cuda:0 \
  --headless
```

환경 수가 많아 GPU memory가 부족하거나 simulation 속도가 불안정하면 먼저
512 또는 256으로 낮춘다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/train.py \
  --num_envs 512 \
  --device cuda:0 \
  --headless
```

기본 PPO 설정은 최대 10,000 iterations이고 100 iterations마다 checkpoint를
저장한다. 명령행에서 학습 길이를 제한할 수도 있다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/train.py \
  --num_envs 1024 \
  --max_iterations 10000 \
  --device cuda:0 \
  --headless
```

## 4. 로그와 checkpoint

학습 결과는 아래에 저장된다.

```text
logs/rsl_rl/hand_rh56_grasp/<YYYY-MM-DD_HH-MM-SS>/
```

실행별 checkpoint를 확인한다.

```bash
find logs/rsl_rl/hand_rh56_grasp -type f -name 'model_*.pt' | sort
```

가장 최근 실행 디렉터리를 확인한다.

```bash
ls -1dt logs/rsl_rl/hand_rh56_grasp/* | head -n 1
```

TensorBoard를 실행한다.

```bash
tensorboard --logdir logs/rsl_rl/hand_rh56_grasp --port 6006
```

주요 지표:

- `Train/mean_reward`: 전체 평균 episode reward
- `Episode_Reward/reach_object`: palm/TCP 접근 보상
- `Episode_Reward/enclose_object`: fingertip enclosure 보상
- `Episode_Reward/lift_progress`: shelf에서 들어 올린 높이
- `Episode_Reward/held_grasp`: 들어 올린 물체를 hand 가까이에 유지한 비율
- `Episode_Termination/success`: grasp/lift 성공률
- `Episode_Termination/object_dropped`: shelf 아래로 떨어뜨린 비율

## 5. 학습된 policy 테스트

Checkpoint 절대경로를 지정하여 GUI에서 재생한다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/play.py \
  --checkpoint /home/hoon/workspace/contact_rich_manipulation_task_learning/logs/rsl_rl/hand_rh56_grasp/2026-08-08_16-50-42/model_7700.pt \
  --num_envs 1 \
  --device cuda:0
```

화면 없이 policy loading과 inference만 검사할 때는 `--headless`를 추가한다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/play.py \
  --checkpoint /absolute/path/to/model_1000.pt \
  --num_envs 1 \
  --device cuda:0 \
  --headless
```

GUI 테스트에서는 다음을 확인한다.

1. Reset 직후 wrist와 palm이 shelf와 겹치지 않는가.
2. 손가락이 shelf 안쪽 `+X` 방향을 향하는가.
3. Palm이 object 위에서 내려오며 접근하는가.
4. 최소 두 fingertip이 object 주변에 남은 상태로 들어 올리는가.
5. Object를 쳐서 날리는 행동이 success로 판정되지 않는가.

## 6. 학습 재개

RSL-RL trainer의 resume 옵션으로 기존 실행을 이어서 학습한다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/train.py \
  --num_envs 1024 \
  --device cuda:0 \
  --headless \
  --resume \
  --load_run '<run-directory-name>' \
  --checkpoint 'model_1000.pt'
```

`load_run`에는 `logs/rsl_rl/hand_rh56_grasp/` 아래의 실행 디렉터리 이름을
지정한다. Robot mount, action 정의, observation 차원 또는 시작 자세가 변경된
checkpoint는 이어서 사용하지 않고 새 학습을 시작한다.

## 7. 문제 확인

### UR5e USD를 찾지 못하는 경우

Nucleus 로그인과 `HAND_RL_UR5E_USD_PATH`를 확인한다.

### Hand mesh를 찾지 못하는 경우

URDF의 `../meshes/*.STL` 경로와 다음 디렉터리를 확인한다.

```bash
find assets/robots/Roas_provided_urdf/meshes -type f -name '*.STL' | sort
```

### CUDA를 사용할 수 없는 경우

```bash
nvidia-smi
conda run -n env_isaaclab python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

