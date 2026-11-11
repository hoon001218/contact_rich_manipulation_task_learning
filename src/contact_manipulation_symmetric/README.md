# Contact Manipulation Symmetric

**2026-11-07부터 사용자 지정 메인 개발·학습 프로젝트다.** 기존 `contact_manipulation`은
비교 구현과 과거 실행 이력으로 보존한다. 공통 PhysX 경로의 오류 조사 기록은
[`docs/TRAINING_LOG.md` 112절](../../docs/TRAINING_LOG.md)을 따른다.

관측·action·모델의 현재 수치 계약과 검증 공백은 같은 기록의 **113–115절**을 따른다.
이는 정적 코드 점검이며 실제 물리 응답의 검증 완료를 의미하지 않는다.
사용자는 정책과 OSC 내부의 F/T 피드백을 모두 요구했다. 현재는 정책에만 보상된 F/T를
전달하며, 실제 6축 load-cell을 이용한 OSC 내부 오차 보정은 아직 구현하지 않았다.

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
| 최종 Actor/Critic MLP 은닉층 | ReLU | **ELU** (2026-11-07 비교 설정) |
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

2026-11-06 통합 브랜치의 원본 코드를 기준으로 복사했다. 이후 차이는 위 표를 따르며,
제어·물리·배치·termination·PPO의 후속 변경은 아래 기록을 따른다. Reward는 아래의
접촉·개선량 기준을 사용한다. 원본 패키지를 import하지 않는다.
Asset과 Isaac Lab checkout은 저장소의 공통 경로를 사용하고, 초기 카탈로그는 독립 복사했다.

| 항목 | 기본값 |
|---|---|
| 모델 | 독립 Actor/Critic, GNN2층·64D(ReLU), role attention, MLP256→128(ELU) |
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

2026-11-07부터 최종 native MLP의 은닉층만 공식 PPO 예제와 같은 ELU를 사용한다.
GNN node/message/update와 최적화된 edge 커널은 ReLU, attention은 Softmax를 유지한다.
Action mean과 value의 최종 출력은 선형이다. 이전 ReLU checkpoint는 play에서 저장된
activation을 복원하며, ELU 기본 설정으로 ReLU 학습을 그대로 resume하는 것은 계약 검사로 거부한다.

### 최신 reward·action 설정 (2026-11-11)

기존 clearance·relocation 함수와 cfg 선언은 보존하고, Symmetric의 활성 두 term을
현재 Blocker 접촉 pad 수와 직전 control step 대비 양의 개선량을 곱하는 함수로 교체했다.

| 항목 | 현재 설정 |
|---|---|
| Clearance | `현재 접촉 pad 수 * max(Q_clear(t)-Q_clear(t-1), 0)`, weight12 |
| Hand–Blocker distance | **현재**17pad 거리의 `mean(exp(-10*d))`, weight5·Target 보호 gate |
| Blocker relocation | `현재 접촉 pad 수 * max(Q_reloc(t)-Q_reloc(t-1), 0)`, weight3·decay10 |
| Contact hold | 새 접촉0.5·유지 접촉1의 pad 합, weight3·Target 보호 gate |
| Shelf / Target / 일반 물체 비용 | 각각−.5, 기존 수식 |
| Action-rate / Joint velocity | 각각−.03, 기존 clipping·12독립관절 선택 |
| Blocker stability / Time / Terminal | 위 원본 설정 유지 |

`Q_clear = 1 - 현재 Blocker 가림 면적 / 초기 가림 면적`이며 signed 값을 유지한다.
`Q_reloc = exp(-10*d)`의 `d`는 현재 Blocker 중심에서 초기 빈 재배치 영역까지의 XY 거리다.
영역 내부나 경계에서는 품질1이지만 정지하면 개선량0이므로 새 보상이 없다. 영역이 없거나
기하가 유효하지 않으면 보상0이다. 초기 영역은 새 공유 scene당1회 준비하고 부분 reset에서
재사용한다. 접근 경로나 전체 Blocker footprint의 영역 진입을 보장하는 판정은 아니다.

접촉 pad는 유효한 **Blocker 전용 법선 접촉 합력 norm > 0.01N**인17채널 중 개수다.
새로 접촉한 pad도 즉시1개로 세고, 직전 step 접촉 여부나 Target 보호 gate는 두 새 term에
추가하지 않는다. 무접촉·정체·악화는0이다. 비접촉 동안에도 비교 기준을 매 control step
갱신하므로 재접촉 시 과거 개선량을 소급 지급하지 않는다. 부분 reset은 해당 행의 초기
가림 품질0과 실제 Blocker 위치로 기준을 다시 잡는다. Native RewardManager가 weight와
control dt를 한 번 곱한다. Distance는 기존 현재 거리 평가를 유지한다.

기존 `mdp.clearance`·`mdp.blocker_relocation`은 그대로 남아 있고, cfg의 기존 선언은
주석으로 보존했다. 새 선언은 `mdp.clearance_contact_progress`와
`mdp.blocker_relocation_contact_progress`를 사용한다. 구현·검증은
[`TRAINING_LOG.md` 145절](../../docs/TRAINING_LOG.md)을 따른다.

Action 출력은 Sensor-frame wrench6 + Hand6의12D를 유지한다.
Force10N/Torque1N·m scale, 센서 범위 clipping과 sensor→COM 원점 보정,
Hand scale.5·native joint-limit mapping, 적용 action 이력·action-rate를 그대로 사용한다.
Transition/support/robot-start의 진단용 CUDA graph 스위치도 원본과 동일하게 제공한다.

F/T 형상은 Axia80-M20의 외경82mm·높이25.4mm·질량0.30kg(표준 인터페이스 plate 포함)을
사용한다. 미정인 추가 연결 플랜지는 양쪽12.3mm·외경50mm·합계0.04kg의 임시값이다.
전체 tool–hand 간격은50mm로 유지하며 `scene.py`의 `sensor_mount_length`로 조절한다.
`sensor_body_length`·`sensor_mount_radius`·`sensor_mount_mass`는 제조사 센서 assembly,
`sensor_flange_radius`·`sensor_flange_mass`는 추가 플랜지의 반경·총질량이다. 플랜지 길이는
전체 길이에서 센서 높이를 뺀 뒤 양쪽에 나눈다. 센서를 길게 늘려 관성을 바꾸지 않는다.
두 rigid link에 센서 반쪽과 플랜지의 질량·COM·관성을 합성하며 내부 질량 분포는 균일 원통
근사다. F/T·OSC·observation의 원점은 [공식 도면](https://www.ati-ia.com/app_content/Documents/9230-05-1507.auto.pdf)의
Tool Side 면 중심으로, wrist 시작점에서37.7mm다. 형상 중앙의 내부 strain-gauge 위치를
가정한 것이 아니라 이 기준점에서 equivalent joint wrench를 읽는다. 실물 플랜지 길이·질량과
센서 장착 yaw는 추후 측정해야 한다. SI-200-8 범위·분해능과 OSC 내부 피드백 OFF는 유지한다.

비접촉 F/T 검증 [OSC 흔들기 영상](../../docs/videos/ft_compensation_2026-11-08/ft_shake.mp4)은
센서 frame gizmo, 6축 명령·전체 측정·예측 self-load·보상값을 동시에 보여준다. 실제 production
OSC와 Hand action으로21초·2100physics step을 실행했고, 초기 자세 설정 이후 상태를
덮어쓰지 않았다. 보상 후 힘 RMS는약0.028N, 토크 RMS는0.00135–0.00183N·m였다.
이는 비접촉 운동 범위의 검증이며 외력을 가한 교정·실물 노이즈·과부하 검증은 아니다.
원자료는 `logs/diagnostics/ft_compensation_2026-11-08/report.json`, 해석은 학습 기록137절이다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py \
  --headless --device cuda:0 --force-amplitude .6 --torque-amplitude .012
```

같은 진단의 `--tactile-scan`은 Palm1·Thumb4·나머지 손가락 각3개, 총17개 pad를
차례로 접촉·분리한다. [Tactile 영상](../../docs/videos/tactile_contact_2026-11-08/all_pads.mp4)에
손 전체·접촉부 확대, 실제17채널 ON/OFF와 법선 합력 그래프를 표시한다. 학습과 같은
grouped `ContactSensor`·`SensorReader`를 사용하며 기준은 합력 크기 `>0.01N`이다.
별도의 probe-only contact view로 실제 시험 구 접촉도 대조한다. Sensor는 physics100Hz로
기록하고 정책50Hz 표본도 구분하며, 영상은25fps다. 외부 시험 구만 native prismatic drive로
움직이고 각 pad 시험 사이에만 초기 자세를 복구한다. 표시용 marker에는 물리 schema가 없다.
원자료는 `logs/diagnostics/tactile_contact_2026-11-08/report.json`, 결과는 학습 기록138절이다.
같은 지름3mm 구로17개 모두 OFF→ON→OFF를 확인했다. 총51초·5100physics 표본이며,
접촉 peak는0.0825–0.2092N이었다. 앞선 지름6mm 구에서는 Little3가 감지되지 않았으므로
모든 접촉 위치·물체 크기의 감지 보장으로 해석하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py \
  --tactile-scan --headless --device cuda:0
```

[UR5e·F/T·Hand 결합 OSC 밀기 영상](../../docs/videos/osc_hand_push_2026-11-08/osc_push.mp4)은
production `SensorFrameWrenchAction`과 실제 articulation으로300g box를85.80mm 민7.6초
시험이다. 기본 마찰0.5/0.5·최대 수평 힘1.8N·Hand fraction0.15를 사용했고, 초기 배치 이후
guide·외부 body force·상태 덮어쓰기는 없다. Scripted6D wrench에 진단 angular PD0.5/0.02로
만든 torque를 더했으며 production OSC 내부 F/T feedback은 계속 OFF다. 학습 policy가 아니다.
최대 물체 기울기0.01978°·Hand 목표 오차0.4342°·URDF 한계 위반0이었다.
**접촉은 중지·엄지의 부모 링크에서 발생했고17개 tactile은 모두 OFF였다.** 영상에도 이 실제
binary를 표시한다. Pad 접촉 coverage 검증으로 확대하지 않으며 collider는 바꾸지 않았다.
원자료는 `logs/diagnostics/osc_hand_push_2026-11-08/report.json`, 예비 실행과 한계는
학습 기록139절을 따른다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py \
  --osc-push --headless --device cuda:0 --push-force 1.8 --push-hand-fraction .15 \
  --push-orientation-kp .5 --push-orientation-kd .02
```

Hand는 PhysX mimic parsing을 끄고12개 회전 관절을 모두 공식 `ImplicitActuatorCfg`의
position drive로 구동한다. `UrdfCoupledHandAction`은 공식 `JointPositionToLimitsAction`의
6개 목표각을 재사용하고, 원본 URDF의 multiplier·offset을 체인 전체에 누적해
`q_i_target = multiplier_i * q_motor_target + offset_i`로12개 목표각을 생성한다.
기준은 접촉으로 밀린 측정각이 아니라 명령 목표각이다. URDF 해석·관절 ID 조회는 초기화 때
한 번 수행하며, rollout은 환경 전체의 GPU tensor 연산과 공식 position-target API만 사용한다.

모든 Hand drive의 초기 PD는20/1, 임시 armature는0.0001kg·m²이며, 원문 URDF의 위치·속도·
effort 상한(1N·m)은 보존한다.
Startup의 master soft position limit 교집합·관측 가능한 Hand6관절·17개 tactile 링크·충돌 형상·
센서 설정도 유지한다. Physics100Hz·policy50Hz·solver32/1은 변경하지 않는다.
이는 물리적인 기어 제약 대신 목표각 연동과12개 drive를 쓰는 대체 모델이다. 실제 각도의
연동 등식은 강제하지 않으며, 추가 drive로 기존6모터 모델보다 유지력이 커질 수 있다.
현재 구조의 checkpoint를 읽을 수 있어도 구동 동역학이 바뀌므로 이전과 같은 조건의 재생은 아니다.
접촉 유지력·관절 한계 위반·실제 전달 힘은 별도 물리 검증이 필요하다.
변경 전 compliant mimic의 접촉 문제는 `docs/TRAINING_LOG.md` 111·118절,
새 구동 경로와 검증은119절을 따른다.
공유 진단 스크립트 `src/contact_manipulation/tests/probe_hand_contact.py`에
`--package contact_manipulation_symmetric`를 추가하면 이 프로젝트의 실제 설정을 측정한다.

Armature 반영 후 [300g 전체 손 이동 밀기 영상](../../docs/videos/hand_armature_push_2026-11-08/all_fingers.mp4)은
현재 native PD20/1·effort1N·m·armature0.0001을 사용해 다섯 배치를 다시 촬영한40초 영상이다.
이 조건에서 최대 관절 하한 위반은0.00630°, 상자 이동은75.16–89.42mm였다. 이전0rad
hard-stop 시험과 비교하며 중간 flexion·실물 Hand 검증으로 확대하지 않는다. 원자료와 재현
명령은 `logs/diagnostics/hand_armature_push_2026-11-08/`, 해석은 `docs/TRAINING_LOG.md`136절이다.

고정 박스에 손가락을 누르는 검사는 다음과 같이 실행한다. `--finger`는
`index`, `middle`, `ring`, `little`, `thumb` 중 하나다. 선택 손가락만 닫고, 박스는 초기 배치
후 움직이지 않는 kinematic body다. 총8초: 열린 상태1초·누르기4초(2초 ramp+2초 hold)·
재개방2초·열린 상태1초다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/tests/probe_hand_contact.py \
  --package contact_manipulation_symmetric --fixed-obstacle --finger index \
  --target-flexion 1 --ramp-steps 200 --settle-steps 100 --plateau-steps 400 \
  --return-to-open-steps 200 --release-steps 100 --headless --device cuda:0 \
  --output /tmp/hand_fixed_index.json
```

명령에 `--video-output /tmp/hand_fixed_index.mp4`를 추가하면 공식 Camera 두 대로
전체 손과 접촉 부위 확대 화면을25fps·실시간 속도로 녹화한다. 선택 손가락은 청록색,
고정 박스는 주황색으로 표시하고, 관절각·접촉 여부·URDF 하한 위반각을 영상에 넣는다.
Camera 활성화는 자동이며 물리 설정과800step 제어 순서는 유지한다.
[다섯 손가락 접촉 영상](../../docs/videos/hand_fixed_contact_2026-11-07/all_fingers.mp4)은
검지·중지·약지·소지·엄지 순서로 각각8초씩 구성한다.
[접촉 힘 표시 영상](../../docs/videos/hand_fixed_contact_2026-11-07/all_fingers_with_force.mp4)은
프레임 시점의 박스 전체·접촉 링크별 법선 합력을N으로 표시한다. 마찰력과 개별 접촉점의
힘은 포함하지 않는다. 기존 RGB·측정 JSON으로 만들었으며 물리 시험을 다시 실행하지 않았다.

움직이는300g 물체 밀기는 `--push-object --mass .3`을 사용한다.80×24×60mm dynamic
박스를 수평 지지면에 놓고 기본 재질의 정지·동마찰계수0.5/0.5를 사용한다. 지지면은
GPU 접촉 필터가 지원하는 kinematic rigid body이며 움직이지 않는다. 손 기부를 고정하고
각 손가락의 초기 밀기 방향을 수평으로 맞춘다. Hand 자중은 제외하고 박스에만 중력을
적용하므로 UR5e·OSC·정책을 포함한 시험은 아니다. PD·effort·URDF 관절 한계는 유지한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/tests/probe_hand_contact.py \
  --package contact_manipulation_symmetric --push-object --mass .3 --finger index \
  --target-flexion 1 --ramp-steps 200 --settle-steps 100 --plateau-steps 400 \
  --return-to-open-steps 200 --release-steps 100 --headless --device cuda:0 \
  --output /tmp/hand_push_index.json --video-output /tmp/hand_push_index.mp4
```

영상의 힘은 손가락과 박스 사이 법선 합력이며 바닥 지지력을 제외한다. 물체의 초기
위치 대비 XY 변위·기울기와 손–지지면 접촉도 별도로 기록한다.
[300g 물체 밀기 영상](../../docs/videos/hand_push_300g_2026-11-07/all_fingers.mp4)은
다섯 손가락을 각각8초씩 촬영했다. 일부 물체 이동에도 관절 하한 위반이 남았으므로
정상 구동으로 판정하지 않는다. 최종 수치와 지지면 필터 보완 과정은 `docs/TRAINING_LOG.md`123절이다.

손가락 자세 목표를 유지하고 손 전체로 밀려면 `--translate-hand --target-flexion 0`을 쓴다.
12관절의 목표는 전 구간0rad이고, floating Hand 기부를 native 외부 prismatic joint/drive로
100mm 전진2초·유지2초·복귀2초시킨다. 앞뒤 정지1초씩을 포함해 총8초다. 실제 관절각이나
root pose를 매 step 덮어쓰지 않으므로 접촉에 의한 자세 변형이 그대로 나타난다.
Hand PD20/1·effort1N·m는 유지하며, 이동 장치의 drive20,000N/m·500N·s/m·200N과 구분한다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation/tests/probe_hand_contact.py \
  --package contact_manipulation_symmetric --push-object --translate-hand \
  --mass .3 --finger index --target-flexion 0 --translation-distance .1 \
  --ramp-steps 200 --settle-steps 100 --plateau-steps 400 \
  --return-to-open-steps 200 --release-steps 100 --headless --device cuda:0 \
  --output /tmp/hand_translate_index.json --video-output /tmp/hand_translate_index.mp4
```

[손 전체 이동 밀기 영상](../../docs/videos/hand_translate_300g_2026-11-07/all_fingers.mp4)은
물체300g·기본 마찰0.5/0.5에서 검지·중지·약지·소지·엄지 앞의 다섯 초기 배치를 촬영했다.
전체 손이 움직여 이웃 손가락도 접촉하므로 손가락 단독 성능 비교는 아니다. 영상에는
법선 접촉 합력(마찰력 제외), 실제 관절각, 목표·실제 기부 이동과 자세 오차를 표시한다.
목표각을 유지해도 검지·중지에서 약51°의 역꺾임이 재현됐다. 상세 기록은124절을 따른다.

위 명령에 `--hand-stiffness 50` 또는 `--hand-stiffness 100`을 추가하면 진단용으로
복사한 Hand actuator의12관절 stiffness만 변경한다. 옵션을 생략하면 production 값을
상속한다. Damping·effort·관절 한계·기부 이동 drive는 유지하고 실제 native gain을 JSON에
기록한다. 학습용 `scene.py`의 actuator 설정은 변경하지 않는다.
[Stiffness20·50·100 비교 영상](../../docs/videos/hand_stiffness_2026-11-07/comparison.mp4)은
다섯 배치의 같은 시점을 나란히 표시한다. Kp20은 기존 녹화를 재사용하고50/100을 새로
촬영했다. Damping1·effort1N·m를 유지한 이 조건에서는 역꺾임이 해결되지 않았다.
실제 적용값·정량 결과와 검증 범위는 `docs/TRAINING_LOG.md`125절을 따른다.

비교 repo의 gain으로 effort 상한을 비교하려면 같은 명령에
`--hand-stiffness 1.2 --hand-damping .06 --hand-effort-limit 1` 또는
`--hand-effort-limit 10`을 지정한다. 세 옵션 모두 진단용 복사 actuator에만 적용한다.
URDF 원본 effort1N·m는 기록에 유지하므로10N·m 실행의 source effort 일치 검사는
의도적으로False다. 실제 backend 상한이 요청한10N·m인지 별도로 확인한다.
[Effort1·10 비교 영상](../../docs/videos/hand_effort_2026-11-07/comparison.mp4)은 왼쪽1·오른쪽10,
Kp1.2·Kd0.06 고정이다. 다섯 배치에서 상한을 높여도 역꺾임이 개선되지 않았다.
최초 준비 중 CUDA719 실패와 정상 재실행 자료는 분리 보존했다. 상세는126절을 따른다.
진단은 `.startup.json`과 오류 발생 시 error JSON의 CPU `partial` 자료도 남긴다.

원인 분리에는 `--hand-armature .0001`, `--hand-self-collision off`,
`--hand-velocity-limit 1000`을 각각 별도 실행에 사용한다. 마지막 값은 속도 제약의 영향을
분리하는 진단용이며 학습 설정의 권장값이 아니다. 모든 옵션은 복사한 진단 설정에만
적용한다. JSON의 `native_joint_properties`에는 실제 backend의 `drive_type`
(0=None,1=Force,2=Acceleration)·armature도 기록하고, 매 sample의
`native_joint_targets_rad`를 제출 목표와 비교할 수 있다. 원인 비교 결과는
`docs/TRAINING_LOG.md`127절을 따른다.

2026-11-07 실제 GPU 검사에서 검지·약지·소지·엄지의 역꺾임이 재현됐다. 새 Drive 방식이
이를 해결했다고 판단하지 않는다. 수치·초기 배치 보완 과정은 `docs/TRAINING_LOG.md`120절,
[관절각 그래프](../../docs/figures/hand_fixed_obstacle_2026-11-07.png)를 따른다.

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
현재 스폰은 `spawn_rules.generation_mode="sequential"`이다. Target 배치 → 폭 기반 Clearance →
10–50% 정면 가로폭을 가리는 Blocker → Clearance 옆 빈 목적지 최소1개 → 주변 물체 순서로
직접 생성한다. 목적지와 초기 Blocker의 앞은 선반 정면까지 비우고, 목적지 yaw는 초기
Blocker와 독립적으로 뽑는다. 물체 수는 기본15–25개를 요청하며 빈 공간·종류 균형에 따라
수용 가능한 수에서 멈춘다. 최소 수 미달은 생성 결과 metadata에 기록한다.

한 장면의 배치·초기 ray 관측은 전체 환경이 공유한다. 부분 reset은 이를 재사용하며 로봇
자세만 새로 샘플링하고, 전체 장면을 교체할 때 배치1회·ray query1회를 수행한다. 사전 역할
카탈로그·FFT·MILP나 배치를 다시 뽑는 조건 반복은 사용하지 않는다. 배치 기하는 CPU,
물체 메시 ray cast와 batched IK는 CUDA다. Actor 관측 물체는 최대10개로 유지한다.
재배치 보상 영역도 새 생성기가 확보한 목적지를 사용한다. 적용·검증 기록은144절이다.

## 검증

새 순차 생성 규칙의 한 장면과 단계별 시간·이미지 기록:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py \
  --headless --device cuda:0 --seed 0
```

학습용 reset과 같은 배치 생성 코드를 호출하는 단계별 진단이다. Target부터 순서대로 생성하고,
Clearance 옆 재배치 영역과 그 앞 통로를 비운다. `docs/figures/sequential_spawn_2026-11-08/`에
9단계 PNG·모음 이미지·시간 CSV·검증 JSON을 저장한다. 배치는 CPU 기하 계산, ray cast·IK는
CUDA를 사용한다. 초기 구현의25물체 예시·실측은 학습 기록140절에 보존하며,
아래 최신 규칙의 변경점과 제한은141절을 따른다.

서로 다른 배치30개를 같은 고정 자원으로 연속 생성하는 최신 예시:

```bash
./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_sequential_spawn.py \
  --headless --device cuda:0 --seed 0 --num-layouts 30 --timing-repeats 0 \
  --output docs/figures/sequential_spawn_layouts30_blocker_front_2026-11-08
```

물체는 각 배치에서15–25개다. 치울 영역은 왼쪽만·오른쪽만·양쪽 중 가능한 조합을 선택하며
최소1개다. Target–Blocker 깊이 간격은 목적지 공간이 성립하는 연속 구간에서 랜덤화한다.
주변 물체는 연속 yaw를 사용하고, 목적지 yaw는 초기 Blocker yaw와 별도로 정한다.
Blocker의 초기 회전 OBB 앞면부터 선반 정면까지도 비운다. 폭은 전체 OBB/실제 형상 폭에
좌우3mm 여유를 더하며, 이 통로는 분홍색으로 표시한다.
`layout_01`–`layout_30`의 top-view·사선 이미지와
전체 모음, 단계별 시간 CSV를 저장한다. 이 규칙은 현재 학습 reset에도 연결되어 있다.
최신 변경점·검증·실측은 학습 기록143절, 목적지 수·간격 변경은142절,
이전 양쪽 목적지 필수 예시는141절을 따른다.

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
