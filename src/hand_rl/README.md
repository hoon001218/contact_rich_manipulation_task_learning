# Hand RL

UR5e 끝단에 ROAS 제공 force-sensor 왼손을 부착하고 shelf 위의 primitive cube를
grasp/lift하도록 학습하는 Isaac Lab manager-based 환경이다.

저장소 루트에서 설치한다.

```bash
conda activate env_isaaclab
./IsaacLab/isaaclab.sh -p -m pip install -e src/hand_rl
```

학습 및 재생:

```bash
./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/train.py \
  --num_envs 1024 --device cuda:0 --headless

./IsaacLab/isaaclab.sh -p src/hand_rl/scripts/play.py \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

빠른 검증은 `--num_envs 1 --max_iterations 1 --headless`를 사용한다.
Nucleus 또는 로컬 USD 경로는 `HAND_RL_UR5E_USD_PATH`, hand URDF 경로는
`HAND_RL_HAND_URDF_PATH` 환경변수로 덮어쓸 수 있다.
