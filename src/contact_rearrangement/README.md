# Contact Rearrangement

밀집 선반에서 Target 접근 공간을 확보하는 비파지 재배치 연구용 Isaac Lab
환경의 첫 구현 단계다. `ManagerBasedEnv`를 사용해 로봇·선반·물체를 스폰하고
물리 시뮬레이션과 reset을 확인한다.

## 초기 장면

- Robot: UR5e + RH56E2 오른손. `hand_rl`의 fixed-joint 조립과 손 방향을 참고했다.
- Hand: `RH56E2_R_2025_9_11_relative.urdf`. Fixed link와 mimic joint를 유지한다.
- Shelf: examples 및 기존 sweeping 환경의 speedrack USD. Kinematic으로 고정한다.
- Objects: 기존 Bottle/Cup/Mug/Can 6종 + YCB physics 4종에서 장면당 4–10개를 선택한다.
  Target 종류는 장면에 정확히 한 번만 사용하고 Blocker·주변 후보에서 제외한다.
  나머지 종류는 중복 가능하다. `--all_asset_types`이면 10종을 각각 하나씩 사용한다.
- Target·Blocker는 episode 시작에 먼저 외부 역할로 지정하고 배치 재시도 중 유지한다.
  +X 쪽 로봇에서 보아 Blocker가 Target 앞에 들어갈 수 있는 전체 X 범위에
  Target을 제안한다. 뒤쪽 10 cm에 제한하지 않는다. 나머지 2–8개는 주변 물체다.
- 로봇은 열린 손의 초기 관절 목표를 유지한다. Reset은 선택한 환경의 robot state·
  관절 목표를 복원하고, 물체 종류·역할·yaw·pose를 다시 샘플링한다.

## 밀집 장면 생성 규칙

다음 수치는 초기 환경 구현 기본값이며 연구의 최종 난이도 명세는 아니다.
계산식·생성 순서·현재 한계는 [상세 배치 기준](../../docs/SPAWN_RULES.md)에 정리했다.

- 실제 USD의 물체 bounds와 선반 collision을 읽어 지지 영역·바닥·천장을 구한다.
  기본 선반은 X `[-0.90, -0.50]`, Y `[-0.455, 0.455]`, 바닥 `1.05`, 천장 `1.48 m`다.
  물체는 가장자리에서 15 mm 이상 떨어지고, 서로 최소 3 mm 간격을 유지한다.
- 전체 선반의 가용 영역에서 물체를 제안한다. 초기 footprint span은 margin을
  제외한 support X의 55% 이상, Y의 65% 이상이어야 한다. Active 물체의 50% 이상은
  다른 footprint까지의 경계 거리가 4 cm 이하인 이웃을 가져야 한다. 좁은 cluster의
  최대 Y 폭 38 cm와 tight rectangle fill 35% 조건은 제거했다. 전체 가용 support의
  점유율은 보고하되 하한을 요구하지 않는다. 이 수치는 조정 가능한 구현 기본값이다.
- Upright 자세에 무작위 yaw를 적용한다. YCB 4종은 local −Y가 윗면이므로
  X축 −90°로 세운다. 원본 USD의 양쪽 렌더에서 003·004·005의 라벨과 006의
  뚜껑으로 상하를 확인했다. Bounds의 바닥을 선반 상면에 맞추고 pivot offset을
  보정한다. 초기 낙하를 줄이기 위해 이전 2 mm 여유는 모든 종류에서 제거했다.
- YCB 003 cracker box·004 sugar box는 visible scanned mesh를 유지하면서
  원본 collider를 끄고 동일 body-local AABB 크기·offset의 invisible Cube collider를
  추가한다. Clone spawn 시점의 override이며 원본 Nucleus 파일·authored mass·
  contact 속성·실제 physics material과 active gravity를 유지한다. 4개 환경·10종
  전체의 8개 layout을 각각 5초 유지한 검사와 전체/부분 reset이 통과했다.
- 초기 Target 앞에서 선반 앞쪽으로 이어지는 고정 접근영역 `A_0`를 만든다.
  통로 폭은 hand 10 cm + 양쪽 margin 1 cm다. 지정 Blocker footprint의 30% 이상이
  `A_0`와 겹쳐야 하고 주변 물체는 초기 통로 밖에 배치한다.
- Blocker의 병진·center rotation·모서리 pivot·회전 뒤 병진 등으로 통로를 비우는
  평면 경로 하나를 확보한다. 끝점뿐 아니라 회전 원호를 포함한 전체 swept footprint를
  보수적으로 검사하고 주변 물체를 그 공간 밖에 둔다. 이 검증용 경로는 actor에
  final goal pose로 전달하지 않는다. 로봇의 도달성·실제 pivot 성공은 후속 검증 범위다.
- 앞쪽 `+X` 카메라에서 Target visible fraction이 40% 이상이어야 한다는 조건을
  `InitialVisionCfg`에 기록했다. 분모는 같은 pose에서 가림 없는 전체 Target silhouette이며
  Target 전체가 frame 안에 들어와야 한다. **Task 카메라와 pixel visibility 판정은 아직
  구현 전**이다. 현재 `target_visibility_validated=False`, `target_visible_fraction=NaN`이며,
  A₀의 30% 차단 판정을 40% 시각 가시성 검증으로 사용하지 않는다.
- 같은 seed·환경 번호·reset 횟수는 같은 장면을 만든다. 실패한 배치를 재시도하며
  정해진 횟수 안에 생성하지 못하면 명시적으로 실패한다.

10종 × 10개 복사본의 100-body pool을 한 번 로드한다. 4–10개 active instance만
선반에 배치하고 나머지는 높이 4 m 이상의 격리 공간에 주차한다. 주차 물체는
매 physics substep에 world COM에서 중력을 상쇄해 정지시킨다. Active 물체는
기본 중력을 사용하고 추가 상쇄력을 받지 않는다. Task 카메라는 선반 작업 영역만 관측해야 한다.
GPU body 인덱스를 유지하기 위해 모든 pool
body의 simulation은 켜 둔다. Isaac Sim 5.1에서 simulation을 껐다 켜는 방식을
사용했을 때 반복 reset 후 다른 instance의 위치가 읽히는 문제가 재현되었다.
`active_object_mask`, `active_object_ids`, `target_object_id`, `blocker_object_id`와
`access_region`을 환경에 기록한다. Pool 전체를 actor 입력으로 사용하지 않으며,
후속 observation·reward는 active mask를 적용해야 한다. Actor observation은 아직 비어 있다.

## 실행

저장소 루트에서 Isaac Sim이 설치된 Python 환경을 활성화한다.
스크립트가 이 저장소의 패키지와 Isaac Lab source 경로를 등록하므로 editable
install 없이 실행할 수 있다.

```bash
conda activate env_isaaclab
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py --num_envs 1
```

GUI 없이 두 복제 환경에서 120 step 동안 스폰·상태·선반 지지·전체/부분 reset을
검사하는 명령은 다음과 같다. 기본 시간 설정에서는 4초의 시뮬레이션이다.
검사 명령이며 모든 seed와 asset 조합의 안정성을 보장하는 것은 아니다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 2 --headless --device cuda:0 --max_steps 120 --validate
```

10종을 모두 하나씩 사용하려면:

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 1
```

중복을 허용하는 일반 모드에서 active 개수만 10으로 고정하려면
`--min_objects 10 --max_objects 10`을 사용한다. 이 모드에서 10종 모두가 반드시
등장하는 것은 아니다. 기본 범위는 4–10개이고 `--seed 7` 등으로 seed를 지정한다.
`--target_asset ycb_003_cracker_box --blocker_asset ycb_006_mustard_bottle`로
두 역할의 종류를 지정할 수 있다. 생략하면 각 episode에서 먼저 뽑는다. 같은
종류를 양쪽에 지정하면 Target 유일성 조건에 어긋나므로 오류로 처리한다.
실행 스크립트는 기본적으로 **시뮬레이션 시간 5초마다** 전체 환경을 reset하고
새 장면을 생성한다. 현재 환경 주기에서는 150 step이며, 주기를 바꾸면 실제
`env.step_dt`에서 필요한 step 수를 계산한다. `--reset_interval 120`처럼 step 수를
직접 지정해 간격을 바꿀 수 있고, `--reset_interval 0`은 자동 reset을 끈다. `--max_steps 0`은
창을 닫을 때까지 실행한다. `--validate`는 reset 시 pose/velocity/관절 목표,
17개 tactile용 링크 및 독립 구동 관절의 존재, finite state, 초기 장면에서의
전체 분포·국소 근접·차단·SE(2) swept clearance 규칙, PhysX 직접 pose 읽기, gravity/simulation
flags 유지와 주차 정지 상태, 물체 낙하·과도한 이동·15° 이상 기울어짐을 확인한다.
접촉 센서·wrist F/T 신호와 camera visibility 검증은 후속 단계다. 카메라 추가 후에는
settle 뒤 Target mask/reference로 40% 조건을 검사하고, 실패 시 episode의 역할·개수를
유지한 채 배치를 다시 뽑아야 한다. 모든 초기 검사를 통과한 Target pose로 A₀를 확정한다.

## Asset 경로

기본 UR5e·선반·물체는 기존 프로젝트의 `omniverse://192.168.0.13` Nucleus
경로를 사용한다. 다른 서버나 로컬 USD는 다음 환경변수로 지정할 수 있다.

- `CONTACT_REARRANGEMENT_NUCLEUS_ROOT`: 공통 Nucleus root
- `CONTACT_REARRANGEMENT_UR5E_USD_PATH`
- `CONTACT_REARRANGEMENT_HAND_URDF_PATH`
- `CONTACT_REARRANGEMENT_SHELF_USD_PATH`
- `CONTACT_REARRANGEMENT_BOTTLE_1_USD_PATH`
- `CONTACT_REARRANGEMENT_CUP_1_USD_PATH`, `CONTACT_REARRANGEMENT_CUP_2_USD_PATH`
- `CONTACT_REARRANGEMENT_MUG_1_USD_PATH`, `CONTACT_REARRANGEMENT_MUG_2_USD_PATH`
- `CONTACT_REARRANGEMENT_CAN_1_USD_PATH`
- `CONTACT_REARRANGEMENT_YCB_003_CRACKER_BOX_USD_PATH`
- `CONTACT_REARRANGEMENT_YCB_004_SUGAR_BOX_USD_PATH`
- `CONTACT_REARRANGEMENT_YCB_005_TOMATO_SOUP_CAN_USD_PATH`
- `CONTACT_REARRANGEMENT_YCB_006_MUSTARD_BOTTLE_USD_PATH`

YCB 기본 경로는 `NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics/`다.
이 서버의 해당 폴더에 있는 003, 004, 005, 006 USD는 authored rigid body와 collision을
가진다. Body의 authored mass는 유지하고 005·006은 원본 collider를 사용한다.
003·004에만 `FlatBoxUsdCfg`로 스폰 시점의 평평한 Cube collider를 적용하며 원본
scanned visual과 Nucleus USD 파일은 유지한다. `Axis_Aligned/`의
visual-only 9종 사용 요청은 이 4종 physics asset 사용으로 대체했다.
동일 종류의 모든 복사본에는 해당 종류의 USD override가 함께 적용된다.

기본 hand URDF는 이 저장소의 `assets/robots/Roas_provided_inspire_urdf/`에서
가져온다. Spawner는 URDF의 mesh 존재 여부와 Git LFS pointer를 변환 전에 검사한다.

Nucleus에서 `Auth: Returned error: 5 (Expired)` 또는 `Authentication Failed`가
발생하면 해당 서버의 인증을 갱신한 뒤 다시 실행하거나 위 환경변수로 로컬
asset을 지정한다.

## 수정 위치와 후속 단계

- `contact_rearrangement/layout.py`: 10종 asset catalog·중복 pool·upright 방향
- `contact_rearrangement/geometry.py`: 실제 USD bounds·선반 지지 영역 측정
- `contact_rearrangement/sampling.py`: 물체 선택·밀집도·차단·clearance 규칙
- `contact_rearrangement/object_pool.py`: 주차 물체에만 world COM 중력 상쇄력 적용
- `contact_rearrangement/object_spawner.py`: YCB 003·004의 body-local AABB Cube collider
- `contact_rearrangement/assets.py`: UR5e/RH56E2 조립·초기 관절 자세·PD hold
- `contact_rearrangement/scene_cfg.py`: scene asset 설정
- `contact_rearrangement/env_cfg.py`: 환경 수·간격·주기·reset event, 후속 InitialVisionCfg
- `contact_rearrangement/mdp/events.py`: 역할 선지정·전체/부분 resampling·active mask·pool 관리
- `scripts/spawn_scene.py`: GUI 확인 및 bounded validation 실행

현재 action/observation manager는 비어 있으며 로봇은 초기 관절 목표를 유지한다.
학습용 18D OSC/hand action, 초기 vision/시간 이력 observation, Reward v2와
성공·실패 조건, PPO·GNN은 이후 이 장면 위에 구현한다. 이 단계에는 학습용
Gym task ID나 PPO 설정을 등록하지 않는다.

현재 바닥 여유 0에 맞춰 fixture를 수정한 unit test 20개, 문법 검사와 diff 검사가
통과했다. 명시적 Cube proxy의 초기 4개 환경·10종 전체·5초 진단에서 박스 8개의
최대 XY 이동은 **0.007894 mm**였고 계산 정밀도 안에서 거의 직립했다.

최종 공식 검사는 다음 명령으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

10초 동안 기본 150-step (5초) 전체 reset과 끝의 부분 reset을 검사했다.
8개 layout을 각각 5초 진행했고 5 cm drift·15° tilt 기준을 완화하지 않았다.
마지막 reset 뒤 생성한 장면에는 추가 물리 step을 진행하지 않았다. 로그는
`outputs/contact_rearrangement/box_stable_reset_validation.log`다. 이 검사를 다른
seed나 실제 로봇 밀기·camera 가시성의 검증 결과로 확대하지 않는다.

Flat-box 적용 전 전체 선반 규칙의 unit test 20개와 실제 측정 bounds의 일반 100 seed·all-types 30 seed가
통과했다. 4개 GPU 환경에서 일반/10종 전체 모드의 24 step·매 step reset과 마지막
부분 reset도 통과했다. 각 layout을 1/30초만 진행한 pool/reset 검사이며 장시간
정착 검사는 아니다. 별도로 10종 전체 장면을 4초간 유지한 검사는 003의 큰 이동으로
실패했다. 새 배치도는 [전체 선반·해소 경로 예시](../../docs/assets/full_shelf_clearance_examples.svg)에 있다.

낙하 여유만 0으로 바꾸고 원본 collider를 사용한 4개 환경·10종 전체·5초 시험은
003 최대 66.96°·XY 11.3 cm, 004 최대 3.80°·XY 5.79 mm의 흔들림을 보였다.
자동 fit하는 `boundingCube` approximation에도 0.83°·XY 1.54 mm 움직임이 남아
최종에 사용하지 않는다. 현재의 명시적 Cube proxy 측정 결과와 구분한다.

이전 4–6개 baseline에서는 YCB 4종의 상하 방향을 양쪽 렌더로 교정하고 기본
8개 환경·8초 반복 reset 검사를 통과했다. 별도 006 Target/003 Blocker 고정 장면은
003의 약 30° 기울어짐과 약 5.5 cm 이동으로 실패했다. 이 결과는 새 전체 선반·
100-body pool·pivot 규칙의 검증 완료를 뜻하지 않으며, 평면 배치만으로 물리 안정성을
보장할 수 없음을 보여준다. 상세 이력은 [CONTEXT.md](../../docs/CONTEXT.md)에 있다.

참고한 파일은 `examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/src/environment.yaml`,
`src/roas_sweeping/roas_sweeping/shelf_sweep/env_cfg.py`,
`src/hand_rl/hand_rl/grasp/assets.py`와 Isaac Lab의
`scripts/tutorials/03_envs/create_cartpole_base_env.py`다.
