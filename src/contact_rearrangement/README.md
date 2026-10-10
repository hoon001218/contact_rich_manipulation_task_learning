# Contact Rearrangement

밀집 선반에서 Target 접근 공간을 확보하는 비파지 재배치 연구용 Isaac Lab
환경의 첫 구현 단계다. `ManagerBasedEnv`를 사용해 로봇·선반·물체를 스폰하고
물리 시뮬레이션과 reset을 확인한다.

## 초기 장면

- Robot: UR5e + RH56E2 오른손. `hand_rl`의 fixed-joint 조립과 손 방향을 참고했다.
- Hand: `RH56E2_R_2025_9_11_relative.urdf`. Fixed link와 mimic joint를 유지한다.
- Connection: UR5e wrist-3와 hand 사이에 길이 5 cm·직경 5 cm의 visible collision
  cylinder를 두 25 mm rigid half로 만들고 중앙 fixed joint에서 hand load의 raw
  F/T를 읽는다. Hand fixed anchor는 tool +Z로 5 cm 이동한다. 현재 41 body·
  18 DOF·41 robot collider이며 nominal 원통 질량 합은 0.20 kg다.
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
- Initial vision: 앞쪽 camera pose를 무작위로 정해 실제 silhouette 기준
  Target ≥40%·Blocker 전체(2 pixel 면적 허용) 가시성을 승인한다. 초기 RGB-D와
  실제 role mask는 episode 중 고정한다.

이전 설명은 사용자의 4–10종 기준을 active instance 4–10개로 잘못 해석했다.
현재 instance 수는 density로 산정하며 기본 최소 4·최대 40은 safety limit이다.
25–45% 요구를 비교해 전체 35%·각 구역 10%를 초기 밀도 기본값으로 선택했다.
이후 큰 물체부터 복사본을 추가하던 방식의 중복 편중을 수정해 종류별 개수를
균형 있게 늘리는 구성으로 변경했다. Blocker 앞 reserve·camera gate 추가 전
균형 구성의 기하 40/40개와 일반 모드
4개 GPU 환경·151 step 검사가 통과했으며 이전 구성의 분포·통과 결과와 구분한다.
이후 지나치게 나뉜 구현을 책임별 6개 모듈로 정리했다. 스폰 규칙·물리 설정·
실행 명령을 유지했고 통합 후 23개 unit test와 4개 GPU 환경·5.03초 검사가 통과했다.
Hand 시작 randomization의 2026-10-09 검사에서 24개 시작·반복 reset·5초 물리
검사와 unit test 29개도 통과했다. 당시 tool은 아래를 향했고 현재는 사용자
요청에 따라 **tool 전방을 선반 방향인 world −X**로 교정했다. Palm normal이
위를 향하던 이 단계의 4개 GPU 환경·151 step 검사도 통과했다. 후속 요청으로
physical tool +Z 기준 hand mount에 **−90° roll**을 추가해 현재 palm normal은
world −Y를 향한다. 새 mount의 4개 GPU 환경·151 step·13개 시작 pose 검사도
통과했으며 앞 단계의 결과와 구분한다.
이후 센서·연결부 공간을 고려해 원통을 추가했다. 위 결과는 원통 추가 전 검증이며
새 조립의 4개 GPU 환경·151 step·13개 시작 pose 검사도 통과했다.
이전 결과는 중앙 F/T·초기 camera gate를 추가하기 전 검증이다. 현재 센서
dictionary observation을 구현하며 이번 센서 단계의 검증은 별도로 기록한다.

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
- 앞쪽 `+X` camera에서 실제 같은 pose의 비가림 silhouette 대비 Target ≥40%·
  Blocker 전체가 보여야 한다. Blocker는 reference 전체 면적에서 최대 2 pixel
  손실을 허용한다. 둘 다 전체 bounds가 frame 안에 있고 reference 최소 32 pixel을
  요구한다. Robot·shelf 가림도 포함하며 A₀의 30% 차단 조건과 별도로 검사한다.
- `SpawnRules.reserve_blocker_front=True`는 회전된 Blocker footprint의 전면부터
  shelf front까지, Y bounds 양쪽에 3 mm를 더한 직선 +X 통로를 주변 물체로부터
  비운다. Target A₀·clearance witness·전체 선반 밀도는 유지한다. 실제 가시성을
  보장하는 조건은 아니므로 pixel gate도 검사한다.
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
후속 물체 observation·reward는 active mask를 적용해야 한다. 현재 센서
observation은 raw F/T·valid mask와 승인한 초기 RGB-D·role mask·camera pose다.
여기에 각 hand link의 독립 ContactSensor에서 얻은 bool `tactile`·`tactile_valid`
17채널을 포함한다.

## 실행

손 시작 기준은 `cfg.robot_start`의 `RobotStartRules`로 조정한다. 현재 제안 영역은
Blocker의 회전 footprint 최전방 +X에서 **12–40 cm 앞**, 동시에 선반 앞 경계에서
**6 cm 이상 밖**, Blocker 중심 Y **±8 cm**, 선반 상면 위 **20–28 cm**다.
TCP는 `right_plam_force_sensor`이고 TCP +Y인 손가락/tool 전방은 **world −X**,
palm normal인 TCP +Z는 **world −Y**, TCP +X는 world +Z를 향한다. 이 조립에서 TCP +Y는 UR
`wrist_3_link`의 local +Z인 tool 전방과 일치한다. 목표 회전은
`sampling.ROBOT_START_TCP_ROTATION`이다. 기존 vendor adapter
`mount_rotation_deg=(0,-90,0)` deg는 유지하고 `mount_tool_roll_deg=-90` deg로
physical tool body +Z 기준 hand mount의 상대 회전과 translation을 함께
변환한다. `tool0` attachment의 local Z는 physical tool 전방과 다른 축이므로
추가 roll의 기준으로 쓰지 않는다.
센서·연결부 공간용 cylinder는 `sensor_mount_length=0.05 m`,
`sensor_mount_radius=0.025 m`이며 `wrist_3_link` local +Z의 z=0–50 mm에 둔다.
Hand fixed anchor를 같은 방향으로 50 mm 이동하며 −90° roll은 유지한다.
현재 원통은 두 25 mm rigid half와 중앙 fixed F/T joint로 구성해 hand load가
측정점을 통과하게 한다. 41 body·18 DOF·41 collider를 쓰며 scene·arm self
검사에 두 half를 포함한다. Nominal 질량 합 0.20 kg는 구현값이며 실제 센서
질량·inertia 교정과 구분한다. 원본 asset과 기존 wrist mass/inertia는 유지한다.
`--validate`는 실제 PhysX TCP 전체 회전 오차 0.01 rad 이하와 wrist-3 tool 축을
대조한다. `assets.py`의 실제 USD 41-link FK와 SciPy IK로 열린 손의 finger q를
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
17개 tactile ContactSensor의 정확한 body mapping·bool 관측·reset valid 처리와
독립 구동 관절의 존재, finite state, 초기 장면에서의
전체 분포·국소 근접·차단·SE(2) swept clearance 규칙, PhysX 직접 pose 읽기, gravity/simulation
flags 유지와 주차 정지 상태, 물체 낙하·과도한 이동·15° 이상 기울어짐을 확인한다.
현재 raw F/T와 실제 camera visibility도 검사한다. Camera 32개 후보가 실패하면
episode의 역할·종류·active ID·개수를 유지한 채 pose를 다시 제안하며 최대
8개 layout에서도 실패하면 기존 state·metadata로 rollback한다. 승인한 Target
pose에서 snapshot과 A₀를 함께 commit한다. Render는 physics step을 진행하지
않으므로 reset pose 가시성 검사와 후속 물리 정착 검사를 구분한다.
실패 시 물체 COM velocity·robot state/목표와 초기 image·camera·Target/Blocker
가시성 buffer를 모두 복구한다.

스크립트는 camera를 자동 활성화하므로 실행 명령에 별도 flag가 필요 없다.
환경을 다른 진입점에서 만들 때는 AppLauncher의 `enable_cameras=True`가 필요하다.

## 센서 출력

`ObservationsCfg.sensors`는 concatenate하지 않는 dictionary다.

| Key | 값 |
|---|---|
| `wrench` | Sensor 중앙 physical tool XYZ의 `[Fx,Fy,Fz,Tx,Ty,Tz]`, `[N,N,N,N·m,N·m,N·m]` |
| `wrench_valid` | 완료된 physics step 이후의 유효성, reset 직후 False |
| `tactile` | 독립 contact sensor 17개, bool `[N_env,17]` |
| `tactile_valid` | 채널별 유효성 bool `[N_env,17]`, reset 직후 False |
| `rgb` | 승인한 초기 RGB `480×640×3`, uint8 |
| `depth` | 초기 image-plane depth `480×640×1`, m |
| `role_masks` | 실제 segmentation의 Target·Blocker 순서 mask `480×640×2`, bool |
| `camera_pose` | 초기 world position + ROS camera quaternion(wxyz), 7D |

Wrench는 parent→child 방향이며 torque 기준점은 원통 중앙이다. 중력·관성·
접촉·외력을 포함한 raw 반력에 자동 tare·중력 보정을 적용하지 않는다.
`env.tool_force_torque_sensor.tare()`는 명시적 instantaneous bias 저장이며
중력 보정과 다르다. Reset은 선택한 환경의 cache·bias를 비우고 valid=False로
둔다. 다음 완료된 physics step 뒤 timestamp가 증가하면 raw 반력을 갱신한다.
RGB-D/mask/camera pose는 episode 동안 cache에서 반환하며 image를 계속
갱신하지 않는다. Reference silhouette·가시성 counts는 observation에 넣지 않는다.

Tactile 순서는 **palm → thumb 1–4 → index/middle/ring/little 각 1–3**이며
기존 `right_plam_force_sensor` 철자를 유지한다. Sensor 이름은 `contact_` + body
name, 경로는 `{ENV_REGEX_NS}/Robot/ROASRightHand/{body_name}`다.
`cfg.tactile.force_threshold=0.01 N`에 대해 world net normal contact force norm
**> threshold**를 binary로 계산한다. SDK의 air/contact-time threshold와 구분한다.
마지막 완료 120 Hz physics step의 값을 읽으며 render/reset의 같은 timestamp에서는
stale 값을 갱신하지 않는다. 선택 reset은 cache zero/invalid·다음 완료 step은
valid, 미선택 cache는 보존한다. Raw force는
`env.binary_tactile_sensor.net_forces_w`(shape `[N_env,17,3]`)에서 확인한다.
기존 부모 5개 convex hull이 pad CAD slot을 메우던 문제만 convex decomposition으로
교정했고 source URDF·visual·mass/inertia·17개 pad collider·41 body/18 DOF/41 collider
prim을 유지한다. 구체적인 설정은 [CONTEXT.md](../../docs/CONTEXT.md)에 있다.

실제 GPU [check_tactile.py](tests/check_tactile.py) `--num_envs 2`는 **151 env step·
5.03초**, reset 6회(부분 2회)·render 116회에서 통과했다. Test-only sphere로
17개 pad의 해당 채널 ON·분리 4 physics step 뒤 모두 OFF, 중력/PD hold의 OFF,
다른 환경 palm ON cache 보존·reset valid 처리·기존 camera/scene 검사를 확인했다.
로그는 `outputs/contact_rearrangement/tactile_gpu_validation.log`다. Binary 접촉
검증이며 하중 교정이 아니다. 기본 task에 probe geometry를 추가하지 않는다.

```bash
/home/hoon/miniconda3/envs/env_isaaclab/bin/python \
  src/contact_rearrangement/tests/check_tactile.py --num_envs 2
```

`cfg.initial_vision`은 camera 위치·aim·roll과 pixel acceptance를 조정한다.
기본 camera는 640×480, focal length 18 mm·horizontal aperture 24 mm로 수평
FOV 약 67°다. 선반 전면 +0.55–0.95 m, Blocker Y ±0.35 m(절대 ±0.50 m 제한),
선반 상면 **+0.18–0.48 m**에서 제안하고 두 역할 사이를 보며 aim Y ±0.06 m·
aim 높이 상면 +0.10±0.03 m·optical-axis roll ±3°를 적용한다.
두 역할의 전체 bounds는 실제 near/far clip `0.02–2.5 m`와 화면 안에 있어야
한다. Nonfinite/range 오류·음수 jitter를 거절하며 +X camera·전체 frame 포함·
같은 pose 비가림 reference는 필수다.
Physics를 제거한 10종 reference visual과 `SilhouetteCamera`는 작업층 밖에서
동일 offset으로 이동해 실제 물체·camera의 상대 pose를 유지하며 환경별
reference lane을 Z로 5 m 분리한다. Live body의
visibility·physics topology는 수정하지 않는다. 초기 5회·이후 2회 `sim.render()`로
texture를 준비하며 physics step은 없다. 상세식은 [SPAWN_RULES.md](../../docs/SPAWN_RULES.md)에 있다.
로컬 `InitialCamera`는 SDK 첫 Fabric pose 조회의 stage sync가 Kit update에서
physics를 진행시키지 않도록 pose 조회/설정·`camera.update`를 보호한다.
`/app/player/playSimulations=False`를 `try/finally`로 복구하며 SDK나 live body의
physics flag는 수정하지 않는다. 첫 camera data 읽기 전에 cache를 준비한다.

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
├── sensors.py             # 초기 vision·raw F/T·17채널 tactile 관측
└── visualization.py       # 역할 OBB debug draw
scripts/spawn_scene.py     # 실행·검증 진입점
tests/test_sampling.py     # 순수 샘플러 검사
tests/check_sensors.py     # 실제 camera/F/T·전체/부분 reset 재사용 검증
tests/check_tactile.py     # 17개 pad 실제 접촉·다중 환경/reset 검증
```

| 바꾸려는 것 | 구현 위치 |
|---|---|
| Asset 경로·10종 catalog·복사본·upright·parking 위치 | [catalog.py](contact_rearrangement/catalog.py) |
| UR5e/RH56E2·중앙 F/T 조립/raw reader·USD FK/IK·PD hold, box collider·countergravity | [assets.py](contact_rearrangement/assets.py) |
| Scene·카메라·센서 observation·주기·seed·SpawnRules·RobotStartRules | [env_cfg.py](contact_rearrangement/env_cfg.py) |
| USD bounds·역할/reset·robot 후보·camera acceptance retry·state/metadata commit/rollback | [spawn.py](contact_rearrangement/spawn.py) |
| 종류/instance·밀도·차단·clearance·retry, 순수 robot 영역/elbow/3D OBB helper | [sampling.py](contact_rearrangement/sampling.py) |
| 초기 camera/reference render·snapshot·raw F/T·17개 독립 tactile cache/관측 | [sensors.py](contact_rearrangement/sensors.py) |
| 현재 Target/Blocker OBB 표시 | [visualization.py](contact_rearrangement/visualization.py) |
| CLI·5초 reset·GUI/headless·검증 흐름 | [spawn_scene.py](scripts/spawn_scene.py) |

현재 센서 책임을 모은 `sensors.py`를 더해 package는 7개 모듈이다. 실행 순서는
**catalog·CLI → camera 활성화 AppLauncher → scene/reference/camera 생성 →
USD 측정과 reset·pixel acceptance → physics step·F/T 갱신·OBB 표시**다.
`catalog.py`는 simulator 없이 읽는 metadata이며
`sampling.py`도 순수 계산을 유지한다. Simulator에 의존하는 모듈은 AppLauncher
뒤에 로드한다. `spawn.py`는 실제 geometry를 `sampling.py`에 전달하고 sampling
결과를 body state·active mask로 적용한다. `assets.py`의 `TaskObjectPool`은 매
physics substep에서 inactive COM countergravity를 적용하며 visualization은
현재 상태를 읽어 표시한다.
Robot 시작은 물체 sample 뒤 별도 seed로 후보를 검사하므로 기존 물체 배치
RNG를 소비하지 않는다. `RobotPoseModel`은 현재 실제 pose에서 FK를 대조하며
default 자세로 먼저 reset하지 않는다. 선택한 환경의 물체·robot 후보를 모두
CPU에서 승인한 뒤 state와 joint/PD 목표를 적용해 camera gate를 검사한다.
초기 vision도 성공해야 snapshot·A₀·TCP/영역/elbow/시도 수를 commit한다.
Camera/layout retry 실패 시 기존 선택 환경의 physical pose·metadata로 복구하며
후보 검사·reset에 settling이나 추가 physics step은 삽입하지 않는다.

**Robot 시작의 이전 방향 검사 (2026-10-09):** 아래 결과는 tool 전방이 아래를
향하던 orientation의 검사이며 선반 방향 교정의 검사로 사용하지 않는다.
4개 GPU 환경 × 6회 reset의 **24개 pose**(10종 전체
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

**Tool 방향 교정, hand roll 추가 전 (2026-10-10):** 사용자의 요청에 따라 tool 전방을 world −X로
맞추고 실제 TCP 회전·wrist-3 tool 축 검사도 추가했다. 시작 영역·elbow-up·
15 mm scene 여유와 장면 생성 규칙은 유지한다. 일반 모드 **4개 GPU 환경·
151 step·5.03초** `--validate`가 exit code 0으로 통과했다. 초기·5초 자동
reset·부분 reset·전체 reset의 **13개 시작 pose**에서 실제 TCP 회전·tool 축·
elbow-up·scene 여유를 확인했다. 미선택 환경 유지·COM cache·역할 OBB 표시의
물리 상태 불변 검사도 통과했다. 로그는
`outputs/contact_rearrangement/tool_forward_gpu_validation.log`다.
1개 GUI 환경·45 step·1.50초와 전체/부분 reset·OBB 불변 검사도 통과했으며
tool·손가락이 선반 정면을 향하는 실제 렌더는
`outputs/contact_rearrangement/tool_forward_preview.png`다. 45-step GUI 실행 중
캡처 시점의 PD 유지 pose에서 tool 축은 world −X에서 약 1.85° 기울었다. 초기 회전 검사와
유지 pose의 상세 검사는 [CONTEXT.md](../../docs/CONTEXT.md)에 구분해 기록한다.
기존 29개 unit test는 이번 수정의 새 결과로 기록하지 않는다.

**Tool +Z 기준 hand −90° roll, 원통 추가 전 (2026-10-10):** 후속 요청으로 hand mount에
별도 roll을 추가했다. Tool 전방 −X·시작 영역·elbow-up·15 mm scene 여유와
물체 생성 규칙은 유지하며 palm normal은 world −Y를 향한다. 위
`tool_forward_*` 로그·렌더는 hand roll 추가 전의 검사로 남기고 새 mount의
실제 상대 transform과 GUI 렌더를 별도로 검사했다. 새 mount의 일반 모드
**4개 GPU 환경·151 step·5.03초** `--validate`가 exit code 0으로 통과했다.
초기·5초 자동 reset·부분 reset·전체 reset의 **13개 시작 pose**에서 실제
PhysX TCP 회전·tool 전방 −X·elbow-up·scene 여유를 확인했다. 미선택 환경·
COM cache·pool flag·역할 OBB 표시의 물리 상태 불변도 통과했다. 로그는
`outputs/contact_rearrangement/hand_tool_roll_gpu_validation.log`다.
첫 reset의 실제 PhysX tool body 기준 hand 상대 회전은 `Rz(-90°) @ R_old`와
3.23141e−7 rad 오차로 일치했으며 TCP offset도 같은 회전으로 이동한 것을
확인했다. 기존 tool 정면 목표의 전체 world 회전 오차는 1.15236e−5 rad였다.
1개 GUI 환경·45 step·1.50초와 전체/부분 reset·OBB 불변 검사도 통과했다.
실제 렌더는 `outputs/contact_rearrangement/hand_tool_roll_preview.png`다.
렌더는 약 20번째 step의 캡처이며 상대 transform 수치는 첫 update의 초기
pose에서 측정했다. 기존 unit test 결과는 이번 수정의 새 실행 결과로 기록하지 않는다.

**5 cm 원통 연결부, 중앙 F/T 추가 전 (2026-10-10):** 사용자 요청으로 wrist-3와 hand 사이에
길이 5 cm 원통을 추가했다. 직경 50 mm는 hand `base_link.STL` mount의 X/Y
bounds ±25 mm를 기준으로 선택한 구현 기본값이다. Hand anchor를 tool +Z로
50 mm 옮기고 roll·TCP 방향·39 body·18 DOF를 유지한다. 원통은 wrist compound
collider로 기존 외부 15 mm 여유·arm self 검사에 포함하며 robot collider는
40개다. 원본 USD·기존 mass/inertia를 유지하고 실제 센서 질량 보정·F/T 출력은
아직 구현하지 않았다. 앞 단계 로그와 새 조립의 검증은 구분한다.
새 조립의 일반 모드 **4개 GPU 환경·151 step·5.03초** `--validate`가 exit code
0으로 통과했다. 초기·5초 자동 reset·부분 reset·전체 reset의 **13개 시작
pose**에서 실제 PhysX TCP·tool 전방·elbow-up·15 mm scene 여유를 확인했다.
40개 robot collider·17개 shelf collider, 실제 link 대 FK position 오차는
5.15e−7 m였다. 미선택 환경·parking COM cache·pool flag·역할 OBB 표시의 물리
상태 불변도 통과했다. 로그는
`outputs/contact_rearrangement/sensor_mount_gpu_validation.log`다.
첫 reset의 실제 PhysX tool body 기준 wrist→hand base offset은 `[0,0,0.05] m`,
gap은 0.049999997 m였다. 원통 양끝이 wrist·hand base origin에 각각 1e−5 m
이내로 일치하고 길이 50 mm·반지름 25 mm를 확인했다. 기존 −90° 상대 hand
roll 오차는 4.51901e−7 rad였으며 TCP offset도 같은 50 mm 이동을 유지했다.
1개 GUI 환경·45 step·1.50초와 전체/부분 reset·OBB 상태 불변도 통과했다.
넓은 시야·연결부 근접 렌더는 `outputs/contact_rearrangement/sensor_mount_preview.png`,
`sensor_mount_detail.png`다. 각각 약 20번째·32번째 step의 캡처이며 조립 상대
transform은 첫 reset에서 측정했다. 상세 로그는 `sensor_mount_preview.log`다.

**중앙 F/T·초기 vision (2026-10-10):** 원통을 두 rigid half로 나누고 중앙
fixed joint에서 hand load의 raw 반력을 읽는다. 41 body·18 DOF·41 collider와
nominal 질량 합 0.20 kg를 사용한다. 실제 같은 pose silhouette로 Target ≥40%·
Blocker 전체 가시성을 승인한 RGB-D를 초기 snapshot으로 보존하고 raw F/T·
valid mask와 함께 sensor observation으로 제공한다. 이번 센서 단계의 검증은
원통 geometry만 검사한 이전 결과와 구분한다.
Blocker 앞 reserve·환경별 reference lane 분리 후 순수 sampling 검사 **32개**가
**27.993초**에 통과했다. 실제 렌더·F/T·GPU 검증과 별도로 기록한다.
Robot arm·hand 가림을 확인해 camera 높이를 상면 +0.18–0.48 m로 넓혔다.
1개 환경의 진단에서는 layout 1회 retry 뒤 2번째 camera 후보가 Target
56.370%·Blocker 100%로 승인됐다. 다중 환경 검증이나 전체 성공률과 구분한다.

**최종 센서 GPU 검사:** 4개 환경의 public **151 env step·5.03초**와 추가
2 env step·F/T bench 7 physics step이 exit code 0으로 통과했다. 6회 reset 중
2회 부분 reset, **18개 승인 시작**이었다. Successful capture call 6회·failed
layout 2개, 두 warmup call 포함 render 464회에서 capture 중 native state·
PD target·physics time/step 무변경을 확인했다. SDK 초기화의 별도 physics
step 2회는 env step·reset/capture와 분리한다. Target **47.1253–100%**, Blocker
17개 정확히 100%·1개 **99.97586%**(정확히 2 pixel 손실 허용), 승인 camera
시도 1–30회였다. Native **41 body·18 DOF·41 collider**, 153 env step 동안 초기
snapshot 고정·F/T valid/finite·선택 reset의 invalid/zero·미선택 cache 보존도 통과했다.
XYZ별 3 N force·0.10 N·m torque와 중력/관성을 포함한 Newton–Euler 대조에서
최대 norm 오차는 **0.0206053 N·0.00132915 N·m**(기준 0.10 N·0.015 N·m)이었다.
중앙점·frame·부호·성분 순서를 확인했고 downstream은 33 body·0.876663 kg였다.
로그는 `outputs/contact_rearrangement/initial_sensors_gpu_validation.log`다.
이 결과는 모든 seed·실제 robot push·하드웨어 교정 성공의 보장이 아니다.

**센서 GUI·실패 복구:** 1개 환경·45 env step·1.50초가 exit code 0으로 통과했다.
003 Target **7724/9286 pixel, 83.178979%**, 006 Blocker **3608/3608 pixel, 100%**,
camera 후보 3회였다. 이동 상태의 강제 capture/중간 commit 실패에서 root/DOF·
물체 COM velocity·position/velocity target·camera·모든 snapshot/역할/seed/count/sample·
F/T raw/valid/tare·clock/step 복구를 확인했다. Derived robot link velocity는
solver 출력과 최대 4.47e−5 차이가 있어 exact 복구로 기록하지 않는다.
로그는 `outputs/contact_rearrangement/initial_sensors_preview.log`이며 실제 초기
camera는 `initial_camera_preview.{png,json}`, GUI 전체·연결부 확대는
`initial_sensors_preview.png`·`initial_sensors_detail.png`다.

재사용 검증 진입점 [check_sensors.py](tests/check_sensors.py)은 다음처럼 실행한다.
`--num_envs 2`·`4`도 지원한다.

```bash
/home/hoon/miniconda3/envs/env_isaaclab/bin/python \
  src/contact_rearrangement/tests/check_sensors.py --num_envs 1
```

이 실제 repository 진입점도 **1개 환경·151+2 env step·bench 7 physics step**에서
exit code 0으로 통과했다. Reset/capture 각 6회(부분 reset 0회)·render 179회,
승인 6개의 Target **41.0632–100%**·Blocker 전부 100%였다. Reset/capture physics
step 증가 0·snapshot 보존·41 body/18 DOF/41 collider와 6축 bench 최대 norm
오차 **0.0122181 N·0.000850297 N·m**를 확인했다. 로그는
`outputs/contact_rearrangement/initial_sensors_reusable_check.log`다. 앞의 4환경·
GUI 결과와 별도 실행이며 `compileall`·`git diff --check`도 통과했다.

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

현재 action manager는 비어 있으며 로봇은 초기 관절 목표를 유지한다. 센서
dictionary observation은 raw F/T·binary tactile 17채널과 초기 snapshot을 반환한다.
학습용 18D OSC/hand action, 초기 물체 기하 추정/시간 이력 encoder, Reward v2와
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
