# Sweep JH

UR5e/Robotiq/virtual F-T sensor를 사용하는 독립적인 JH 실험용 Isaac Lab
패키지다. Asset assembly, 환경과 MDP term을 모두 `sweep_jh` 내부에서 직접
정의한다. Synthetic contact pad body와 contact sensor는 생성하지 않고 Robotiq
USD의 원래 collision geometry를 사용한다. 환경 변형을 추가하지 않고
`Isaac-Sweep-JH-v0` 단일 Gym 환경만 등록한다.

## 설치

저장소 루트에서 패키지를 Isaac Lab Python에 editable install 하는 것을 권장한다.
`scripts/train.py`와 `scripts/play.py`는 source checkout의 패키지 경로도 직접
등록하므로, 설치하지 않은 상태에서도 저장소 내부에서 실행할 수 있다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_jh
```

## 학습

```bash
./IsaacLab/isaaclab.sh -p src/sweep_jh/scripts/train.py \
  --num_envs 2048 --device cuda:0 --headless
```

빠른 smoke test는 환경 수와 iteration을 줄여 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_jh/scripts/train.py \
  --num_envs 4 --max_iterations 2 --device cuda:0 --headless
```

학습 로그와 checkpoint는 `logs/rsl_rl/sweep_jh/` 아래에 생성된다.

TensorBoard는 저장소 루트에서 다음과 같이 실행한다.

```bash
tensorboard --logdir logs/rsl_rl/sweep_jh
```

전체 episode reward는 `Train/mean_reward`, 성공률은
`Episode_Termination/success`, 최종 Cartesian 오차와 진행률은
`Metrics/desired_motion/*`에서 확인한다.

## 재생

```bash
./IsaacLab/isaaclab.sh -p src/sweep_jh/scripts/play.py \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

## 커스터마이징 위치

- 환경과 manager term 연결: `sweep_jh/osc_sweep/env_cfg.py`
- 로봇 및 F/T asset assembly: `sweep_jh/osc_sweep/assets.py`
- MDP 구현: `sweep_jh/osc_sweep/mdp/`
- PPO 설정: `sweep_jh/osc_sweep/rsl_rl_ppo_cfg.py`
- 학습 및 재생 환경 ID: `Isaac-Sweep-JH-v0`

현재 환경은 joint-space state, `current_target_pose`, contact point를 제외한
41-D task-space policy observation과 12-D variable-stiffness OSC action을
사용한다. 관절 속도 대신 robot-base frame의 6-D EEF twist를 관측한다.
목표 force는 filtered contact force 대신 virtual F/T의 전체 force norm으로
추종한다.

현재 observation, reward 및 학습 hyperparameter의 전체 표는
[`docs/current_observation_reward_hyperparameters.md`](docs/current_observation_reward_hyperparameters.md)를
참고한다.
