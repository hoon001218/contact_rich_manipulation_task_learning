# Contact Rearrangement

밀집 선반에서 Target 접근 공간을 확보하는 비파지 재배치 연구용 Isaac Lab
환경의 첫 구현 단계다. `ManagerBasedEnv`를 사용해 로봇·선반·물체를 스폰하고
물리 시뮬레이션과 reset을 확인한다.

## 초기 장면

- Robot: UR5e + RH56E2 오른손. `hand_rl`의 fixed-joint 조립과 손 방향을 참고했다.
- Hand: `RH56E2_R_2025_9_11_relative.urdf`. Fixed link와 mimic joint를 유지한다.
- Shelf: examples 및 기존 sweeping 환경의 speedrack USD. Kinematic으로 고정한다.
- Objects: 기존 Bottle/Cup/Mug/Can 6종 + YCB physics 4종에서 장면당 **4–10종**을 선택한다.
  Target 종류는 장면에 정확히 한 번만 사용하고 Blocker·주변 후보에서 제외한다.
  선택한 나머지 종류의 active 개수가 균형을 이루도록 복사본을 추가해 선반
  밀도에 필요한 instance 수를 결정한다.
  `--all_asset_types`는 10종 모두 선택하며 정확히 10개 instance로 제한하지 않는다.
- Target·Blocker는 episode 시작에 먼저 외부 역할로 지정한다. 종류 집합·instance
  ID 전체·개수도 배치 전에 정하며 retry에서 유지한다.
  +X 쪽 로봇에서 보아 Blocker가 Target 앞에 들어갈 수 있는 전체 X 범위에
  Target을 제안한다. 뒤쪽 10 cm에 제한하지 않는다. 두 역할 외에는 주변 물체다.
- Reset에서 물체 장면을 먼저 정하고 Blocker 앞쪽의 충돌 없는 영역에 열린 손의
  시작 pose를 무작위로 정한다. Elbow-up 해만 허용하며
  episode의 관절 목표를 유지한다. 부분 reset은 선택한 환경에만 적용한다.

이전 설명은 사용자의 4–10종 기준을 active instance 4–10개로 잘못 해석했다.
현재 instance 수는 density로 산정하며 기본 최소 4·최대 40은 safety limit이다.
25–45% 요구를 비교해 전체 35%·각 구역 10%를 초기 밀도 기본값으로 선택했다.
이후 큰 물체부터 복사본을 추가하던 방식의 중복 편중을 수정해 종류별 개수를
균형 있게 늘리는 구성으로 변경했다. 현재 구성의 기하 40/40개와 일반 모드
4개 GPU 환경·151 step 검사가 통과했으며 이전 구성의 분포·통과 결과와 구분한다.
이후 지나치게 나뉜 구현을 책임별 6개 모듈로 정리했다. 스폰 규칙·물리 설정·
실행 명령을 유지했고 통합 후 23개 unit test와 4개 GPU 환경·5.03초 검사가 통과했다.
Hand 시작 randomization의 최종 24개 시작·반복 reset·5초 물리 검사와 unit test
29개도 통과했다. 이전 스폰·모듈 통합 검사와 새 기능의 결과는 구분한다.

## 밀집 장면 생성 규칙

다음 수치는 초기 환경 구현 기본값이며 연구의 최종 난이도 명세는 아니다.
계산식·생성 순서·현재 한계는 [상세 배치 기준](../../docs/SPAWN_RULES.md)에 정리했다.

- 실제 USD의 물체 bounds와 선반 collision을 읽어 지지 영역·바닥·천장을 구한다.
  기본 선반은 X `[-0.90, -0.50]`, Y `[-0.455, 0.455]`, 바닥 `1.05`, 천장 `1.48 m`다.
  물체는 가장자리에서 15 mm 이상 떨어지고, 서로 최소 3 mm 간격을 유지한다.
- Margin을 제외한 **전체 usable shelf 면적 대비 footprint 점유율 하한**과,
  **X 2 × Y 3의 6개 구역 모두의 점유율 하한**을 반드시 요구한다. 구역별 값은
  각 footprint와 구역의 교차 면적을 구역 면적으로 나눈 값이다. Global **35%**·
  각 구역 **10%**가 현재 기본값이다. 25–45%의 생성 성공률·시간을 비교해
  선택한 초기 구현값이며 연구의 최종 난이도 수치는 아니다.
- 전체 footprint span은 usable X의 55% 이상, Y의 65% 이상이어야 하고 active의
  75% 이상은 4 cm 이내 footprint 이웃을 가져야 한다. 이 판정은 occupancy와
  별개다. 옛 38 cm cluster 폭·tight-rectangle fill과 점유율을 보고만 하던
  방식은 사용하지 않는다. 수치는 변경할 수 있는 구현 기본값이다.
- Upright 자세에 무작위 yaw를 적용한다. YCB 4종은 local −Y가 윗면이므로
  X축 −90°로 세운다. 원본 USD의 양쪽 렌더에서 003·004·005의 라벨과 006의
  뚜껑으로 상하를 확인했다. Bounds의 바닥을 선반 상면에 맞추고 pivot offset을
  보정한다. 초기 낙하를 줄이기 위해 이전 2 mm 여유는 모든 종류에서 제거했다.
- YCB 003 cracker box·004 sugar box는 visible scanned mesh를 유지하면서
  원본 collider를 끄고 동일 body-local AABB 크기·offset의 invisible Cube collider를
  추가한다. Clone spawn 시점의 override이며 원본 Nucleus 파일·authored mass·
  contact 속성·실제 physics material과 active gravity를 유지한다. 이전 정확히
  10개 instance 버전의 8개 layout을 각각 5초 유지한 검사와 reset이 통과했다.
  이전 면적 우선 구성도 일반·all-types 각각 4개 환경·10초의 정착/reset 검사를
  통과했다. 현재 균형 구성은 일반 모드 4개 환경·5.03초 검사도 통과했으며
  범위와 로그는 아래에 기록한다.
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

10종 × 10개 복사본의 100-body pool을 한 번 로드한다. 선택한 4–10종을 최소
하나씩 사용하고 non-target 복사본을 추가해 요구 footprint 면적에 도달한다.
복사본을 추가할 때는 가능한 non-target 종류 중 active 개수가 가장 적은 종류를
고르고 동률은 무작위로 결정한다. Blocker도 해당 종류의 첫 instance로 센다.
현재 동일 pool capacity에서는 non-target 종류별 active 개수 차이가 최대 1이며
Target은 항상 하나다. 별도 종류별 고정 최대값은 없고 pool의 10-copy·전체 cap
40을 유지한다. 작은 물체 위주 종류 집합은 같은 면적을 채우는 데 더 많은
instance가 필요할 수 있다. 공간 배치는 packing을 위해 여전히 큰 footprint부터
진행한다. 복사본의 종류 선정과 위치 배치 순서를 구분한다. 정한 composition으로
geometry·구역 밀도·경로 조건을 배치할 수 없으면 cap이나 조건을 완화하지 않고
실패한다. Active instance만 선반에 놓고 나머지는 높이 4 m 이상의 격리 공간에 주차한다. 주차 물체는
매 physics substep에 world COM에서 중력을 상쇄해 정지시킨다. Active 물체는
기본 중력을 사용하고 추가 상쇄력을 받지 않는다. Task 카메라는 선반 작업 영역만 관측해야 한다.
`TaskObjectPool`은 link pose를 쓴 뒤 COM pose cache timestamp를 무효화해
같은 simulation timestamp의 연속 reset에서도 다음 COM 읽기를 새로 계산한다.
Countergravity 적용점만 갱신하며 추가 physics step이나 native flag 전환은 없다.
GPU body 인덱스를 유지하기 위해 모든 pool
body의 simulation은 켜 둔다. Isaac Sim 5.1에서 simulation을 껐다 켜는 방식을
사용했을 때 반복 reset 후 다른 instance의 위치가 읽히는 문제가 재현되었다.
`active_object_mask`, `active_object_ids`, `target_object_id`, `blocker_object_id`와
`access_region`을 환경에 기록한다. Pool 전체를 actor 입력으로 사용하지 않으며,
후속 observation·reward는 active mask를 적용해야 한다. Actor observation은 아직 비어 있다.

## 실행

손 시작 기준은 `cfg.robot_start`의 `RobotStartRules`로 조정한다. 현재 제안 영역은
Blocker의 회전 footprint 최전방 +X에서 **12–40 cm 앞**, 동시에 선반 앞 경계에서
**6 cm 이상 밖**, Blocker 중심 Y **±8 cm**, 선반 상면 위 **20–28 cm**다.
TCP는 `right_plam_force_sensor`이고 손가락은 world −Z, palm normal은 −X를
향한다. `assets.py`의 실제 USD 39-link FK와 SciPy IK로 열린 손의 finger q를
유지하며 shoulder–wrist chord보다 elbow가 **5 cm 이상 위**인 해만 고른다.
한 TCP의 IK 해가 충돌하면 `accept_candidate` 판정으로 거절하고 같은 TCP의
다른 seed/branch도 탐색한다. 첫 IK branch가 충돌했다는 이유로 바로 포기하지 않는다.
전체 로봇 collision OBB를 모든 active 물체·선반 개별 collider·ground와
**15 mm margin**으로 검사한다. UR arm 7개 link의 21쌍 중 두 arm joint 이내
11쌍은 제외하고 거리 3 이상인 10쌍의 self collision을 2 mm로 검사한다.
손 전체의 self collision 안전성을 보장하는 검사는 아니다.
최대 80번 안에 해를 찾지 못하면 실패하며 elbow-down fallback은 없다.
`cfg.robot_start.enabled=False`이면 이 randomization을 끈다. 설정 수치는
후속 제어·학습 분포에 맞춰 조정할 수 있으며 상세 기준은
[SPAWN_RULES.md](../../docs/SPAWN_RULES.md)에 기록한다.

저장소 루트에서 Isaac Sim이 설치된 Python 환경을 활성화한다.
스크립트가 이 저장소의 패키지와 Isaac Lab source 경로를 등록하므로 editable
install 없이 실행할 수 있다.

```bash
conda activate env_isaaclab
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py --num_envs 1
```

**역할 OBB 표시:** GUI·livestream에서는 Target을 초록색, Blocker를 빨간색
OBB로 표시한다. `RoleObbVisualizer`가 각 역할의 12개
모서리를 debug draw로 그리며, 현재 body link pose와 실제 local bounds를 사용해
YCB의 upright 방향·중심이 아닌 root·합성 scale을 반영한다. 매 step과 전체/부분
reset 뒤 갱신한다. USD prim·collider·material이나 physics state는 수정하지 않는다.
기본 표시를 끄려면 `--disable_role_obb`를 사용한다. 일반 headless 실행은
표시를 생략하고 `--headless --validate`에서는 viewport 없이 overlay 계산과
debug draw 검사를 명시적으로 실행한다. 2개 CUDA 환경·151 step의 headless 검사와
1개 GUI 환경·45 step 검사에서 표시와 reset 갱신이 통과했다. Draw 전후의 USD
prim/API·PhysX pose/velocity·native flag가 유지되는 것도 확인했다. 실제 GUI
렌더 `outputs/contact_rearrangement/role_obb_preview.png`에서 초록 cracker box와
빨간 mustard bottle의 OBB가 물체를 감싸는 것을 확인했다. 이는 역할 표시 검사이며
task camera의 40% 가시성 판정과 별개다. 별도 CPU 행렬 기준과 비교해 legacy/YCB
upright·yaw/tilt·scale/root offset·환경별 world offset·이동/기울어진 pose·부분 역할
교체와 48개 edge의 색상/3 px 두께·redraw/cleanup·body pose 무변경도 확인했다.

GUI 없이 두 복제 환경에서 120 step 동안 스폰·상태·선반 지지·전체/부분 reset을
검사하는 명령은 다음과 같다. 기본 시간 설정에서는 4초의 시뮬레이션이다.
검사 명령이며 모든 seed와 asset 조합의 안정성을 보장하는 것은 아니다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 2 --headless --device cuda:0 --max_steps 120 --validate
```

10종을 모두 선택하고 점유율에 필요한 복사본을 함께 사용하려면:

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 1
```

`--min_asset_types 6 --max_asset_types 6`으로 정확히 6종을 선택할 수 있다.
기본 종류 범위는 4–10종이다. `--min_objects`·`--max_objects`는 instance의 최소값·
safety cap이며 종류 수를 지정하는 옵션이 아니다. 기본 최소 4·cap 40이고, 낮은
cap으로 density를 채울 수 없으면 명시적으로 실패한다. `--min_shelf_occupancy`·
`--min_region_occupancy`로 global/구역 하한을 지정한다. 현재 기본값은 각각
0.35·0.10이며 다른 값에서의 생성 성공을 보장하지 않는다. `--seed 7` 등으로
seed를 지정한다.
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
settle 뒤 Target mask/reference로 40% 조건을 검사하고, 실패 시 episode의 역할·종류·instance 조합·개수를
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

## 구조와 수정 위치

```text
contact_rearrangement/
├── __init__.py             # 빈 package 초기화
├── catalog.py             # asset·pool metadata
├── assets.py              # 로봇·물체 생성과 pool 동역학
├── env_cfg.py             # scene·ManagerBasedEnv 설정
├── spawn.py               # USD 측정과 reset
├── sampling.py            # 순수 기하 샘플링
└── visualization.py       # 역할 OBB debug draw
scripts/spawn_scene.py     # 실행·검증 진입점
tests/test_sampling.py     # 순수 샘플러 검사
```

| 바꾸려는 것 | 구현 위치 |
|---|---|
| Asset 경로·10종 catalog·복사본·upright·parking 위치 | [catalog.py](contact_rearrangement/catalog.py) |
| UR5e/RH56E2 조립·`RobotPoseModel` USD FK/IK·PD hold, box collider·countergravity·물체 생성 | [assets.py](contact_rearrangement/assets.py) |
| Scene·환경 수·간격·주기·seed·SpawnRules·RobotStartRules·후속 vision 설정 | [env_cfg.py](contact_rearrangement/env_cfg.py) |
| USD bounds, 역할·전체/부분 reset·robot IK/충돌 후보 승인·시작 metadata·state 적용 | [spawn.py](contact_rearrangement/spawn.py) |
| 종류/instance·밀도·차단·clearance·retry, 순수 robot 영역/elbow/3D OBB helper | [sampling.py](contact_rearrangement/sampling.py) |
| 현재 Target/Blocker OBB 표시 | [visualization.py](contact_rearrangement/visualization.py) |
| CLI·5초 reset·GUI/headless·검증 흐름 | [spawn_scene.py](scripts/spawn_scene.py) |

실행 순서는 **catalog·CLI → AppLauncher → scene/env 생성 → USD 측정과 reset →
physics step·OBB 갱신**이다. `catalog.py`는 simulator 없이 읽는 metadata이며
`sampling.py`도 순수 계산을 유지한다. Simulator에 의존하는 모듈은 AppLauncher
뒤에 로드한다. `spawn.py`는 실제 geometry를 `sampling.py`에 전달하고 sampling
결과를 body state·active mask로 적용한다. `assets.py`의 `TaskObjectPool`은 매
physics substep에서 inactive COM countergravity를 적용하며 visualization은
현재 상태를 읽어 표시한다.
Robot 시작은 물체 sample 뒤 별도 seed로 후보를 검사하므로 기존 물체 배치
RNG를 소비하지 않는다. `RobotPoseModel`은 현재 실제 pose에서 FK를 대조하며
default 자세로 먼저 reset하지 않는다. 선택한 환경의 물체·robot 후보를 모두
CPU에서 승인한 뒤 default reset과 joint position·zero joint velocity·동일 PD
target/state 적용, TCP/영역/elbow/시도 수 저장을 진행한다. 후보 실패 시 기존
선택 환경의 physical pose와 metadata도 유지한다. 후보 검사·reset에 settling이나
추가 physics step은 삽입하지 않는다.

**Robot 시작의 최종 검사:** 4개 GPU 환경 × 6회 reset의 **24개 pose**(10종 전체
8개)가 통과했다. Physics 없는 6회 reset 뒤 150 step·5초의 parking 안정성과
전체/부분 reset, 불가능한 elbow 조건으로 강제 실패했을 때의 physical state·
PD target·reset count·역할·TCP metadata 유지도 확인했다. Robot collider 39개·
shelf collider 17개, FK 오차 2.8458e−7 m, elbow chord 높이 11.49–16.07 cm,
후보 1–3회였다. 최종 unit test **29개**가 24.673초에 통과했다. Stress 자료는
`outputs/contact_rearrangement/robot_randomization_benchmark.{json,log}`다.
최종 일반 모드 4환경·151 step·5.03초 `--validate`도 exit code 0으로 통과해
150-step 자동 reset·부분 reset 보존·COM cache 갱신·OBB 불변 조건을 확인했다.
로그는 `outputs/contact_rearrangement/robot_randomization_gpu.log`와
`robot_randomization_unit_tests.log`다.
4환경 reset batch는 물체 배치를 포함해 wall time 1.420–7.601초였으며 현재
CPU 후보 계산 비용은 환경 수를 늘릴 때 batch/vectorization 검토가 필요한 한계다.
GUI 45 step·1.5초와 전체/부분 reset·COM cache reset·OBB 불변 검사도 통과했다.
실제 손 시작 렌더는 `outputs/contact_rearrangement/robot_start_preview.png`다.
초기 151-step 통과·COM 회귀·수정 이력과 상세 GUI 검사 결과는
[CONTEXT.md](../../docs/CONTEXT.md)에 기록한다.

이전 `layout.py`는 `catalog.py`, `object_pool.py`·`object_spawner.py`와 물체
factory는 `assets.py`, `scene_cfg.py`는 `env_cfg.py`, `geometry.py`·`mdp/events.py`는
`spawn.py`로 통합했다. 이전 모듈·`mdp/`와 compatibility wrapper는 남기지 않았다.

**통합 후 검사:** 모든 모듈의 문법 검사와 `--help` 실행이 통과했다.
`catalog`·`sampling`을 simulator 없이 import했을 때 `isaaclab`·`isaacsim`·`pxr`·
`torch`가 로드되지 않았다. Unit test **23개**가 24.552초에 통과했다. 일반 모드
`--num_envs 4 --headless --device cuda:0 --max_steps 151 --validate`도 exit code 0으로
통과해 5.03초의 물리 실행, 150-step 자동 reset과 마지막 전체/부분 reset을 확인했다.
96개 역할 OBB line과 draw 전후 USD·PhysX 상태 유지도 검사했고 물리 guard를
완화하지 않았다. 최초·자동 reset 뒤 8개 장면 설명이 통합 전 균형 구성 실행과
정확히 같았다. Asset 18개 정의, USD/reset과 scene/env 설정의 AST도 통합 전과
일치했다. 새 시뮬레이션·camera 기능을 추가한 작업은 아니다.
로그는 `outputs/contact_rearrangement/refactor_cli_help.log`,
`refactor_unit_tests.log`, `refactor_gpu_validation.log`다.
아래 이전 실행 기록은 해당 시점의 결과로 유지한다.

## 후속 단계와 검증 기록

현재 action/observation manager는 비어 있으며 로봇은 초기 관절 목표를 유지한다.
학습용 18D OSC/hand action, 초기 vision/시간 이력 observation, Reward v2와
성공·실패 조건, PPO·GNN은 이후 이 장면 위에 구현한다. 이 단계에는 학습용
Gym task ID나 PPO 설정을 등록하지 않는다.

종류/instance 분리 전 바닥 여유 0에 맞춰 fixture를 수정한 unit test 20개, 문법 검사와 diff 검사가
통과했다. 명시적 Cube proxy의 초기 4개 환경·10종 전체·5초 진단에서 박스 8개의
최대 XY 이동은 **0.007894 mm**였고 계산 정밀도 안에서 거의 직립했다.

**이전 정확히 10개 instance 버전의 공식 검사**는 다음 명령으로 통과했다.
현재 같은 명령의 all-types는 density에 필요한 instance를 추가하므로 이전 결과와
같은 분포의 재현이라고 해석하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

10초 동안 기본 150-step (5초) 전체 reset과 끝의 부분 reset을 검사했다.
8개 layout을 각각 5초 진행했고 5 cm drift·15° tilt 기준을 완화하지 않았다.
마지막 reset 뒤 생성한 장면에는 추가 물리 step을 진행하지 않았다. 로그는
`outputs/contact_rearrangement/box_stable_reset_validation.log`다. 이 검사를 다른
seed나 실제 로봇 밀기·camera 가시성의 검증 결과로 확대하지 않는다.

**이전 면적 우선 구성의 4–10종·hard global/6구역 occupancy 검사:** 실제 USD bounds의 35% 기하
sample은 일반·all-types 각각 seed 0–19, **고유 40/40개**가 통과했다. 일반 모드는
4–10종·10–24 instance, all-types는 10종·14 instance였다. 40%는 고유 30/30개가
통과했지만 공통 seed 집합에서 평균 생성 시간이 약 0.99–1.00초로 35%의
약 0.206초보다 5배 길었다. 45% 초기 sample은 3/10개만 생성됐다. 35%를
초기 기본값으로 선택한 근거이며 benchmark 자료는
`outputs/contact_rearrangement/density_calibration.json`에 있다. 중복 seed는
합산하지 않는다. 당시 배치도는 4종·10개, 4종·24개, 10종·14개를 비교했다.
현재 배치도 파일은 균형 구성의 예시로 교체했으며 이 이전 분포를 현재 구성의
검증 결과로 사용하지 않는다.

당시 단위 검사 **21개**가 18.219초에 통과했고 문법·diff 검사도 통과했다.
당시 GPU에서는 아래 두 실행 모두 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

각 실행은 10초이며 기본 150-step 전체 reset과 마지막 부분 reset을 확인했다.
각 모드의 초기 4개·첫 reset 뒤 4개, **8개 layout을 각각 5초** 진행했고
5 cm drift·15° tilt 기준을 유지했다. 마지막 reset 뒤 새 장면은 물리 step을
진행하지 않아 정착 표본에 포함하지 않는다. 로그는
`outputs/contact_rearrangement/dense_shelf_default_validation.log`와
`outputs/contact_rearrangement/dense_shelf_all_types_validation.log`다.
기하 benchmark와 bounded settle/reset 결과를 미검사 seed·실제 로봇 밀기·
camera 가시성의 검증으로 확대하지 않는다.
**현재 균형 구성 검사:** 최종 source의 실제 USD bounds로 일반·all-types
각각 seed 0–19, **40/40개**가 생성됐고 non-target 종류별 개수 차이가 항상
최대 1이었다. 일반 모드는 12–25 instance·global 35.05–38.48%·검사 전체의
최소 구역 22.86%, all-types는 15–18 instance·global 35.14–38.71%·최소 구역
24.75%였다. Source SHA를 포함한 자료는
`outputs/contact_rearrangement/balanced_composition_benchmark.json`에 있다.
현재 unit test **23개**가 24.384초에 통과했으며 균형 선정·재고 소진 회귀를 포함한다.

현재 일반 모드의 다음 GPU 실행도 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 --max_steps 151 --validate
```

5.03초 동안 150-step 자동 reset과 마지막 전체·부분 reset, 정착 guard를 확인했다.
역할 OBB 96개 line과 draw 전후 USD·PhysX 상태 유지도 검사했다. Reset sample에
10종 장면이 포함됐지만 현재 구성의 별도 all-types GPU 실행으로 해석하지 않는다.
로그는 `outputs/contact_rearrangement/balanced_composition_gpu_validation.log`다.
이 결과를 미검사 seed·실제 robot push·camera visibility로 확대하지 않는다.

[현재 균형 구성 배치도](../../docs/assets/dense_shelf_examples.svg)는 일반 seed 0의
4종·12개, 일반 seed 2의 4종·25개, all-types seed 1의 10종·17개를 보여준다.
Target을 제외한 종류별 개수는 각각 3/4/4, 8/8/8, 1/1/2/2/2/2/2/2/2다.
Console의 `[SPAWN]`에는 반복 asset 목록 대신 종류별 `asset_counts`를 출력한다.

Flat-box 적용 전 전체 선반 규칙의 unit test 20개와 실제 측정 bounds의 일반 100 seed·all-types 30 seed가
통과했다. 4개 GPU 환경에서 일반/10종 전체 모드의 24 step·매 step reset과 마지막
부분 reset도 통과했다. 각 layout을 1/30초만 진행한 pool/reset 검사이며 장시간
정착 검사는 아니다. 별도로 10종 전체 장면을 4초간 유지한 검사는 003의 큰 이동으로
실패했다. [이전 전체 선반·해소 경로 예시](../../docs/assets/full_shelf_clearance_examples.svg)는
종류·instance 분리 및 hard occupancy 추가 전의 이력이다.

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
