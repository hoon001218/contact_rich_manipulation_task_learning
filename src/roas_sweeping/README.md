# ROAS Sweeping

`sweeping_policy`의 shelf 및 6개 object 구성을 유지하면서 Robotiq 2F-85를
ROAS 제공 force-sensor 왼손으로 교체한 독립 Isaac Lab 프로젝트다.

## 환경 구성

- Robot: UR5e USD
- Hand: `assets/robots/Roas_provided_urdf/urdf/urdf_left_with_force_sensor.urdf`
- Mount: `hand_rl`과 같은 방식으로 UR5e tool frame에 fixed joint로 조립
- Scene: `sweeping_policy`와 같은 shelf, Bottle/Cup/Mug/Can 6개
- Task ID: `Isaac-Shelf-Sweep-UR5e-ROAS-Hand-v0`

URDF importer의 `merge_fixed_joints=False`를 사용하여 force-sensor link를
합치지 않는다. 다음 17개 mesh/link에는 각각 독립된 `ContactSensorCfg`가 있다.

- palm 1개
- thumb 4개
- index, middle, ring, little 각각 3개

각 센서는 world-frame 합력의 크기가 `0.1 N`을 초과하면 `1`, 아니면 `0`을
출력한다. 총 17개의 binary 값이 위 순서대로 `contact_states` observation에
연결된다. 현재 reward는 조립된 scene, action, observation 및 센서를 검증하기
위한 최소 baseline이며 최종 sweeping reward 설계는 포함하지 않는다.

## 설치 및 빠른 실행

저장소 루트에서 실행한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/roas_sweeping

./IsaacLab/isaaclab.sh -p src/roas_sweeping/scripts/train.py \
  --num_envs 1 --max_iterations 1 --headless
```

GUI에서 scene을 확인하려면 checkpoint 생성 후 다음을 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/roas_sweeping/scripts/play.py \
  --checkpoint /absolute/path/to/model_1.pt --num_envs 1
```

## Asset 경로 override

- `ROAS_SWEEPING_UR5E_USD_PATH`
- `ROAS_SWEEPING_HAND_URDF_PATH`
- `ROAS_SWEEPING_SHELF_USD_PATH`
- `ROAS_SWEEPING_BOTTLE_1_USD_PATH`
- `ROAS_SWEEPING_CUP_1_USD_PATH`, `ROAS_SWEEPING_CUP_2_USD_PATH`
- `ROAS_SWEEPING_MUG_1_USD_PATH`, `ROAS_SWEEPING_MUG_2_USD_PATH`
- `ROAS_SWEEPING_CAN_1_USD_PATH`

Shelf 및 object는 기존 `SWEEPING_POLICY_*_USD_PATH`도 fallback으로 인식한다.
기본 asset은 `192.168.0.13` Nucleus를 사용하므로 해당 서버의 인증이 유효해야
한다. 인증을 사용할 수 없는 머신에서는 위 환경변수에 로컬 USD 경로를 지정한다.
