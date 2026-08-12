# Sweeping Policy

`Sweeping-Policy-DRL-Sim-Train`의 결합 UR5e/Robotiq USD와 shelf/object 배치를
기반으로 만든 Isaac Lab manager-based 환경 골격이다.

## 현재 범위

- UR5e와 Robotiq 2F-85가 결합된 Nucleus USD를 하나의 articulation으로 로드
- virtual F/T sensor는 생성하지 않음
- Nucleus shelf와 Bottle/Cup/Mug/Can 6개를 scene에 배치
- reset마다 6개 중 하나를 선택하여 작업층에 grid 없이 연속 좌표로 랜덤 배치
- 선택되지 않은 5개 물체는 최상단 대기층의 분리된 슬롯으로 이동
- 작업영역 안에서 0.18 m를 밀 수 있는 방향만 선택하고 EEF를 반대편에 배치
- reset-time IK로 reaching 완료 자세를 초기 joint state로 설정
- EEF orientation은 축 정렬 기준으로 고정하고 position에만 축별 ±1 cm noise 적용
- reference observation 순서를 유지하되 target/goal position은 EEF frame으로 제공
- 6D relative EEF pose action을 DLS Differential IK로 UR5e joint target에 변환
- binary gripper action 1차원을 결합하여 전체 policy action은 7차원
- scene observation, reset, 안전 termination 제공
- PPO 설정은 실행 파이프라인 검증용이며 sweep command와 task reward는 아직 미정

## 설치 및 실행

저장소 루트에서 다음을 실행한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweeping_policy

./IsaacLab/isaaclab.sh -p src/sweeping_policy/scripts/train.py \
  --num_envs 1 --max_iterations 1 --headless
```

등록된 task ID는 `Isaac-Shelf-Sweep-UR5e-v0`이다.

## Asset 경로 override

기본 경로는 `sweeping_policy/shelf_sweep/assets.py`에 있으며 다음 환경변수로
덮어쓸 수 있다.

- `SWEEPING_POLICY_ROBOT_USD_PATH`
- `SWEEPING_POLICY_SHELF_USD_PATH`
- `SWEEPING_POLICY_BOTTLE_1_USD_PATH`
- `SWEEPING_POLICY_CUP_1_USD_PATH`, `SWEEPING_POLICY_CUP_2_USD_PATH`
- `SWEEPING_POLICY_MUG_1_USD_PATH`, `SWEEPING_POLICY_MUG_2_USD_PATH`
- `SWEEPING_POLICY_CAN_1_USD_PATH`
