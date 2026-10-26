# 연구와 Isaac Lab 개발 맥락

최종 갱신: **2026-10-26, Asia/Seoul**.

밀집 선반에서 Target 접근 공간을 확보하는 비파지 재배치 연구의 목적, 설계가
바뀐 이유, 현재 구현과 검증 결과, 미결 질문을 유지한다. 새 대화와 개발 작업은
이 문서를 먼저 읽고 관련 코드 및 세부 문서로 이어간다.

연구 기준은 Notion의 [연구 맥락 — 생각의 흐름과 현재 설계](https://app.notion.com/p/3f34968094ee81ca8eb6feb1f700e0f9)
이다. 해당 페이지의 연구 기준일은 2026-10-08이며, 2026-10-09에 다시 확인했다.
이 문서는 그 맥락과 이후 이 저장소에서의 개발을 통합한다.

## 최신 진입점 — 2026-10-26

현재 개발·학습 대상은 **`src/contact_manipulation`**,
`Isaac-Contact-Manipulation-UR5e-v0`다. 아래 1절은 별도로 보존한 이전
`contact_rearrangement` 프로젝트의 당시 상태이며 최신 구현 안내로 사용하지 않는다.
현재 코드 구조·MDP·모델은 [프로젝트 README](../src/contact_manipulation/README.md),
본 학습 기록·정책 행동의 문제·수정 과제는 **[TRAINING_LOG.md](TRAINING_LOG.md)**를 먼저 읽는다.

- 2026-10-26 사용자 승인으로 reward weight와 민감도를 함께 조정했다. Clearance12·decay2,
  Distance1·decay15, Shelf−2·Object−1, Action-rate/Joint-velocity각−.01, Stability1이다.
  Hold.2·F/T각−.03·Singularity.3·성공250과 기존 gate/종료는 유지한다.
  Time은 custom 양의 `time_reward`를 제거하고 공식 `is_alive`에 weight−2를 주는
  `time_penalty`로 교체했다. 100Hz 일반step−.02, 성공/실패 마지막step0, timeout과 장면 교체
  truncation step−.02다. 8초 timeout의 할인 전 Time합은−16이며 새 센서·history·파일은 없다.
  기존 증가형과 달리 처음부터 일정−2/s이고, 실패로도 잔여 비용을 피할 수 있다는 한계가 있다.
  CPU91개 통과·CUDA전용3개 skip이며 simulation·재학습은 미실행이다. TRAINING_LOG78절 참조.
- 2026-10-26 GUI의 Scene Debug Visualization에 기본 OFF인 **Tactile / F/T monitor**를
  추가했다. Environment Index로 선택한 환경의17개 pad를 접촉 초록/미접촉 회색/invalid 주황으로
  표시하고, 센서 frame의 F/T를 방향 화살표·토크 원호·축별 양방향 막대·native LiveLinePlot으로
  보여준다. 기존 OBB와 하나의 DebugDraw callback을 공유하며 물리 prim·sensor view는 추가하지
  않는다. 이미 계산된 any-object tactile binary와 SI-200-8 측정값(자중·관성 포함)만 읽는다.
  UI 최대20Hz, reset/선택 변경 시 이력 초기화, OFF/headless 추가 샘플링·연산 없음이다.
  관련 CPU22개 검사 통과이며 실제 Kit GUI/GPU 렌더링은 미검증이다. TRAINING_LOG77절 참조.
- 2026-10-26 최신 저장 실행 `2026-10-25_18-57-11`의17183회까지를 읽기 전용으로 분석했다.
  해당 실행은 **기존 음의 Time penalty**와17개 pad별 Distance exp 평균을 사용했다. 현재 코드의
  양의 `time_reward` 효과를 검증한 결과는 아니다. 최신1000회에서 weighted episode/8초 로그 평균은
  Clearance+.1081·Distance+.4952·Hold+.04975·Action-rate−.1313·Joint-velocity−.05496이며,
  C/D/H 중 Distance 기여는75.8%다. C/H는 초기보다 증가했으나 누적 성공은약.08198%로 드물다.
  Hold가 작게 실현된 것만으로 weight 부족을 단정하지 않으며 손등 밀기의 원인도 현재 로그로
  확정하지 않는다. 기존 Time은 초기2초의 비용률이 작았다. 행동·모델 코드와 설정은 변경하지
  않았고 GPU·simulation은 실행하지 않았다. 단위·종료 snapshot 주의사항은 TRAINING_LOG76절 참조.
- 2026-10-26 사용자 선택에 따라 Time을 매 step 초반에 크고 점차 작아지는 양의 보상으로
  전환했다. 기준 곡선은 `exp(-0.7*t)`, `RewardsCfg.time_reward`의 weight1·decay.7이며
  완료 control 구간 평균에 native dt를 곱한다. 8초 할인 전 누적은+1.423289다.
  성공 여부와 무관하게 지급하므로 episode가 길수록 누적 Time 보상 자체는 증가한다.
  기존 `time_penalty` 함수·term·MdpCfg scale/growth 필드는 제거했다. 기존 elapsed clock과
  부분 reset을 재사용하며 물리/정책100Hz·Terminal250·다른 항과 종료조건은 유지했다.
  CPU 관련검사32개 통과·simulation/학습 미실행이다. TRAINING_LOG75절 참조.
- 2026-10-25 장면 생성 FFT에서 명시적인 CUDA OOM이 발생했다. 첨부 로그는 별도 PID858212의
  11.98GiB와 학습 process 전체10.85GiB를 보고하며, GPU23.55GiB 중 free50.75MiB였다.
  배치 seed의 FP64 커널 캐시가 마지막 사용 후 즉시 해제되도록 하고 큰 중간 tensor도
  variant 평가 후 해제했다. 생성 규칙·0.5mm grid·정밀도·물리/정책100Hz는 유지한다.
  CPU 기존/수정4사례의 배치·기존진단값 일치,44개 고유 커널 입력에서 캐시 peak2개,
  관련검사12pass·CUDA5skip을 확인했다. GPU peak 감소·단독학습 재현은 미측정이다.
  다른 PID의 프로그램·현재 점유는 확인하지 못했으며 종료하지 않았다. TRAINING_LOG74절 참조.
- 2026-10-25 사용자 요청으로 정책을 **100Hz(decimation1)**로 복원했다. 물리·OSC100Hz,
  γ=.98과 action scale, Distance의17개 pad별 exp 평균은 유지한다. 성공.5초=50sample,
  episode8초=800step, rollout36=.36초다. 아래50Hz 설정은 직전 비교 설정의 기록이며
  현재 기본값은 아니다. 상세 변경은 TRAINING_LOG73절을 따른다.
- 2026-10-25 후속 실험 설정: Distance를 최소 pad 거리에서 **17개 pad별 exp 품질의 평균**으로
  변경했다. `V_T * mean_i(exp(-10*d_i))`이며 weight3·Target gate는 유지한다. Invalid pad는
  0점이고 분모는17이다. 접촉 유지 보상은 기존17개 binary 합산을 유지한다.
  정책은 **50Hz(decimation2)**, 물리·OSC는100Hz(dt.01), γ=.98과 action scale은 유지한다.
  성공.5초=25sample, episode8초=400step, rollout36=.72초다. Action-rate·F/T 증가량 제곱은
  주기 불변이 아니므로 이전100Hz 결과와 비교할 때 구분한다. CPU 검사46pass·CUDA2skip이며
  simulation·새 학습은 실행하지 않았다. 상세 변경은 TRAINING_LOG72절을 따른다.
- 2026-10-25 `02-40-51` run의 CUDA 오류와 느린 로봇 동작·적은 Hand 움직임을 분리해 분석했다.
  사용자 표현의 속도는 iteration 시간이 아니라 play의 실제 동작이다. 실제 설정은 OSC 위치.03m/
  회전.15rad·Hand scale.5·팔만3rad/s 종료, C8/D3(decay10)/Hold.2·F/T각−.03이다.
  마지막15497회까지 누적 성공3036/1,303,568≈.2329%이며 후반 양의 dense 기여의81.4%가
  Distance다. 접근 보상은17개 pad의 최소거리만 보므로 손가락 추가 접촉·Blocker 이동을 직접
  유도하지 않는다. Hand std는.710–.841로 소멸하지 않았지만 play는 deterministic mean을 쓴다.
  γ=.98·100Hz의 할인 반감기는.343초여서 접근 유지와 지연된 성공의 균형도 검토해야 한다.
  CUDA 직전 value/std와 checkpoint는 finite였고 최초 실패 kernel은 미확정이다. 다음 비교는
  진단 전용 단계별 sync·compile/graph 분리와 action→target→실제 관절속도 확인이다.
  Reward/PPO/제어는 수정하지 않았으며 제안과 scene별 편차·한계는 TRAINING_LOG71절을 따른다.
- 2026-10-25 사용자 요청으로 **Hand를 `hand_velocity` 종료에서 제외**했다. 공식
  `joint_vel_out_of_manual_limit`은 UR5e 팔6개만 검사하며 **|qdot|>3rad/s** 조건은 유지한다.
  Joint velocity reward는 팔6개+Hand 독립6개를 그대로 평가한다. Native 원인 로그·속도 실패의
  성공 보상 차단은 유지한다. 사용자가 직접 조정한 OSC position/orientation scale과 Hand scale
  **각0.5**, 초기 Gaussian std1은 변경하지 않는다. URDF·actuator·reward·PPO·reset/IK도
  변경하지 않는다. 아래67–69절 당시의12관절 종료 기록보다 이 결정이 우선하며 상세는
  TRAINING_LOG70절을 따른다.
- 2026-10-25 속도 종료 후 즉시 reset·연산 증가를 실제 로그로 확인했다. 임계값3rad/s의
  `2026-10-24_23-33-13`은 초기 episode1.4–1.7step,302회에도약4step이며 성공/raw goal은0이다.
  초기 collection 중앙값15.07초 대 이전2.65초, 마지막10회도6.53초다. 부분 reset마다 공통 배치·
  vision을 재사용하지만 연속 EEF 후보·IK·상태 쓰기는 다시 하므로 reset 폭증이 비용을 늘린다.
  IK fallback은0이며 최초 초과 관절은 현재 로그로 특정할 수 없다. Reset은 qdot0·현재 위치
  target으로 일치한다. 첫 정책 입력 후 arm/Hand 속도를 분리해 기록하고 정지 명령과 비교한 뒤
  OSC scale·속도 종료를 함께 검토한다. std1·상대 pose scale1은 위험 후보이며 원인 확정은 아니다.
  min Inspire의 joint-position 제어·팔만 검사하는3rad/s는 우리 OSC·12관절에 그대로 호환된다는
  뜻이 아니다. PPO std·물리·종료를 임의 변경하지 않았으며 상세는 TRAINING_LOG69절을 따른다.
- 2026-10-24 최신 원격 `min`을 다시 확인한 뒤 사용자 승인으로 `hand_velocity` 임계값을
  **1→3rad/s**로 변경했다. 팔6개+Hand 독립6개 선택과 공식 `joint_vel_out_of_manual_limit`은
  유지하며 어느 관절이든 **|qdot|>3rad/s**일 때 즉시 종료한다. 정확히±3rad/s는 통과한다.
  원격 HEAD `baa3740e38c1a693b2094dc4d141b72de0dd3142`의 기본 cfg는1이지만
  InspireOpen→Hand/HandMass·센서/질량 변형은3을 상속하고 HandMassLimit만1로 복원한다.
  이전의 원격 `min` 전체가1이라는 설명은 override를 놓친 불완전한 설명이었다.
  원격의 팔6개와 달리 Hand 포함은 사용자 승인한 연구 환경의 차이다. URDF·actuator 물리 속도
  상한·OSC·action·reward 수식/weight는 변경하지 않는다. 아래 최초1rad/s 기록보다 이번 결정이
  우선하며 상세는 TRAINING_LOG68절을 따른다.
- 2026-10-24 예제의 `hand_velocity`를 참고해 공식 `joint_vel_out_of_manual_limit` 종료를 처음 추가했다.
  당시 사용자 추가 요청으로 팔6개와 Hand 독립6개를 함께 검사하며 어느 관절이든 **|qdot|>1rad/s**면
  즉시 종료한다. 이름 기반 `SceneEntityCfg`를 사용하고 joint velocity reward와 같은 관절이며
  mimic/passive는 제외한다. `Episode_Termination/hand_velocity`에 독립 기록하고 native manager의
  해석된 cfg로 검사한 결과를 task 실패·성공 보상 차단에도 반영한다. 속도 초과와 성공 보상은
  동시에 발생하지 않는다. 사용자는 우선 관절 속도 종료만 선택했으므로 action-rate 종료·clipping,
  reward 수식·weight·최종 transition 격리는 추가하지 않았다. 따라서 큰 finite action/관절속도
  비용의 유입 문제 전체가 해결된 것은 아니다. 이전 속도 종료 미추가 기록보다 이 결정이 우선하며
  변경·검증은 TRAINING_LOG67절을 따른다.
- 2026-10-24 사용자 요청으로 Hand observation의 F/T 자중·관성 보상을 제거했다.
  공통 측정 경로는 sensor frame의 native 반력에 SI-200-8 clipping·분해능만 적용한다.
  Actor·Critic의 현재/직전 F/T 및 Force/Torque 증가량 reward가 같은 측정값을 사용하므로
  hand 자중·관성 하중이 포함된다. 손 링크 COM·질량·관성 cache와 가속도·각속도 조회,
  Newton–Euler 보상 함수·bias·distal body 목록을 제거했고 Object COM 보정은 유지한다.
  reset F/T=0, invalid sensor mask, 관측244D, OSC gravity_compensation=True 및 기존
  reward 계수·제어·termination은 유지했다. 이 제거만으로 큰 action·관절 속도 원인이나
  학습 발산이 해결됐다고 판단하지 않는다. 이전 F/T 보상 유지 기록보다 이 결정이 우선하며
  변경·검증은 TRAINING_LOG66절을 따른다.
- 2026-10-24 최신 `22-25-22` 학습이0–7회 후 Gaussian sample의 `std >= 0.0` 오류로 종료됐다.
  최초 이상은3회의 Action-rate−6.17e12·관절속도−1.15e18·F/T 증가−4.18e16/−3.87e30과
  value loss Inf다. 당시 std는1.004이며 invalid 종료는0으로, 이전 std≈5 증가와 구분한다.
  큰 finite 관절 상태가 관성 보상과 제곱 비용으로 증폭되는 경로는 확인했지만 최초 물리 원인은
  미확정이다. Torque 계수−60은 초기값이며 정상 구간2회에도 양의 항 합의약1,045배였다.
  다음 우선순위는 물리 한계 기반 크기 검사·이상 transition의 PPO 유입 차단·최초 이상 기록이다.
  종료만 추가해도 native 순서상 reset 전 reward는 지급되므로 충분하지 않다. 이번에는 분석과
  CPU 수치 확인만 했고 runtime 수정·학습 재현은 하지 않았다. Object 비용은−.0044로 폭발하지
  않았으며64절의 tanh 제거 결정을 자동 복원하지 않는다. 상세·근거·한계는 TRAINING_LOG65절이다.
- 2026-10-24 후속 사용자 승인으로 Object 비용의 tanh를 제거하고 **활성 non-Blocker의
  OBB 코너 속도**를 평가한다. 기존 snapshot의 중심 선속도와 각속도로
  `v_corner=v_OBB+ω×(corner−center)`를 구해8코너 norm 평균·물체 합을 환경별로 계산한다.
  Target·초기 비인식 활성 물체도 포함하고 Blocker·미사용 slot은 제외한다. Native weight−.5와
  dt를 유지하므로 병진·회전 조작 경로 길이의 근사이고, 10cm 병진은0.1초든1초든 할인 전−.05다.
  코너 이력·누적 cache·추가 SDK/ray casting/환경별 loop는 없으며 환경별 scalar
  `object_motion_cost`를 step당1회 계산한다. 비유한 물체의 기여만0이고 invalid 종료는 동일하다.
  **사용자 요청에 따라 향후 수정 방향을 물으면 이 선택을 반드시 다시 설명한다.** 이전 tanh는
  예제를 옮기며 추가한 Object cost 포화였고 policy action tanh가 아니다. 선형 비용은 빠른 burst도
  경로에 비례해 평가하지만 큰 순간 속도·진동·노이즈 기여가 커질 수 있다. 이전 tanh와의 비교를
  제안할 수 있으나 임의로 복원하지 않는다. Time weight.1·Shelf 연속비용은 논의만 했으며
  **미적용**이다. 현재 Time1과 다른 모든 weight·reward·termination·제어·모델은 그대로다.
  아래 Object 선속도L1/tanh 기록보다 이 변경이 우선한다. 세부사항은 TRAINING_LOG64절을 따른다.
- 2026-10-24 직전 사용자 승인으로 reward를 로컬 examples의 native 구조로 수정했다.
  Dense 함수의 `/step_dt` 상쇄를 제거하고 native `weight×dt`를 적용한다. 성공1회+250만
  `250/dt` 반환으로 유지하고 Time의 구간 적분도 그대로 둔다. 누적 Rearrangement를 제거하고
  Shelf/Object collision weight각−.5로 대체했다. 당시 Object는 활성 non-Blocker의 선속도 L1합에
  tanh를 적용했으며 이후 위 OBB 코너 속도 방식으로 대체했다. Shelf는 위치5mm/회전1°/
  선속도.005m/s/각속도.05rad/s 기준을 cfg에서 분리한다. Action-rate·12독립관절 velocity는
  공식 L2 함수각−.03을 사용하며 반환 NaN/Inf만0으로 처리한다. F/T는 사용자 확정인 **매 step norm 양의 증가량 제곱**,
  weight−.15N⁻²/−60(N·m)⁻²의 별도 초기 계수이며 exp·deadband·증가율 변환은 없다.
  C8/D1/H.2, Singularity.3, 수정 직전 사용자 Stability.5, Time1, 성공250을 유지한다.
  cfg에서 함수·weight·파라미터를 선언하고 mdp/rewards.py에서 수식이 직접 보이도록 정리했다.
  누적 path·OBB 이력·packed reward_components를 제거하고 기존 GPU snapshot은 재사용한다.
  현재13term이며 아래12term/누적R/exp Motion/dense dt상쇄 기록보다 이 결정이 우선한다.
  성공·termination·관측·action·모델·PPO는 동일하다. 구현·단위·Terminal 상대 강도 변화와
  검증 한계는 TRAINING_LOG63절을 따른다. 새 설정의 학습 성능·처리량은 아직 측정하지 않았다.
- 2026-10-24 사용자 승인으로 reward weight만 **Clearance4→8·Distance4→1·Hold/pad1→0.2**로
  조정했다. Rearrangement0.5·Action rate/Joint velocity/Force growth/Torque growth각0.15·
  Singularity0.3·Object stability3·Time1·Terminal weight1/성공250은 유지한다. 수식·민감도·
  gate·종료·action·PPO는 변경하지 않는다. 직전 `01-11-37` run 후반25%의 weighted
  Episode_Reward 중앙값은 Distance12.64/ Clearance0.411로, 접근 상태 체류보다 가림 개선을
  우선하는 비교 실험이다. `16-57-56`의0–5회만으로 효과를 판단하지 않는다.
  아래 과거 weight 기록보다 이 설정이 우선하며 변경·검증 기록은 TRAINING_LOG62절을 따른다.
- 2026-10-24 최신 run `01-11-37`의234 checkpoint를 CPU로 조사했다.18D 모든std가 약1에서
  최종4.685–5.229로 증가했지만 원래 step별 action/관측은 저장되지 않아 당시 최대값은 복구할 수
  없다. 유계 합성 관측의11650 정책 CPU 진단은 최대|μ|9.947·새 sample28.975였고,
  고정 물리 상태 action feedback은 최대61.025였으며 원래 궤적 재현은 아니다. 이 조건에서
  10^6 action/거대value는 재현하지 않았다. Pose scale1·무clip과raw18D history경로를 확인했고
  production을 수정하지 않았다. 근거·18축 std·해석 한계는 TRAINING_LOG61절을 따른다.
- 2026-10-24 후속 사용자 확정으로 Critic에 **최초 인식된 물체만 현재 GT로 갱신**하는
  별도 `critic` 관측 그룹을 연결했다. Actor의 초기 vision10개·244D는 유지하며 Critic도
  같은 인식 ID·슬롯·padding의10개·244D다. 초기 비인식 물체 추가·가림에 따른 삭제·재 ray casting은
  없다. 기존 physics snapshot의 실제 OBB 중심·회전과 reset에 캐시한 크기를 현재 EEF frame으로
  변환한다. 현재·직전144D는 Actor와 같다. 전체 물체GT·별도scalar상태·reward/PPO 변경은 없다.
  새 Critic 계약은 `native_graph_critic_v2`·`tracked_initial_objects_v1`이다. 같은폭의 이전
  초기 vision Critic은 의미가 달라 전체 resume를 거절하며 기존 play는 원래 관측으로 복원한다.
  구현·검증과 한계는 TRAINING_LOG60절을 따른다.
- 2026-10-24 사용자 요청으로 Critic에도 **별도 scene encoder·GNN·역할 attention**을 구성했다.
  Actor와 같은 공개244D에서 scene320D와 현재/직전144D를 concat하고, 각각 독립 ReLU MLP
  `464→256→128`을 사용한다. Actor는 Gaussian18D, Critic은 deterministic value1D다.
  두 모델은 encoder·head·정규화 통계를 공유하지 않으며 native `MLPModel.forward`·PPO는 유지한다.
  시작 로그의 MLP 우선 표시는 모듈 등록 순서이고 실제 forward는 scene→concat→MLP다.
  로그에 실행 흐름을 명시했다. 기존 Actor 규약은 유지하지만 raw MLP Critic의 전체 resume는
  호환되지 않아 새 학습이 필요하며, 이전 checkpoint play는 저장된 구성을 복원한다.
  아래 독립 raw MLP/ELU Critic 기록은 당시 상태다. 검증은 TRAINING_LOG59절을 따른다.
- 2026-10-23 추가 사용자 확인으로 Rearrangement를 **episode 누적 OBB 경로**에 대한
  `exp(-20P)-1`로 바꿨다. 왕복·정지 후에도 비용이 남으며 Blocker 이동은 제외한다.
  Object stability는 XY 전체OBB돌출·기존collision최저Z의 거리와 upright기울기각을
  같은 물체의 위험으로 결합한다. 각도는 제곱하지 않고 `aθ=ln2/(π/4)`를 적용한다.
  기존 중심 기반 지지 실패·tilt 종료·gate는 유지한다. 아래 직전step R/기울기미평가 기록보다
  마지막「누적 Rearrangement와 전체 OBB·기울기 안정성」의 최신 결정이 우선한다.
- 2026-10-23 후속 reward 정리: 성공과 같은 Target 보호2cm/0.15rad를 공통 gate로 써서
  위반 중 양의 Clearance·Distance·Contact만0으로 만들고, 회복하면 재개한다. 추가 종료는 없다.
  Clearance 분모는 초기 Blocker 투영면적으로 고정한다. F/T는 독립 native term 각각weight.15로
  분리하고, Time은 기존 비용률`.02*(exp(.7*t)-1)`의 구간 적분·weight1로 복원했다.
  현재 reward는12개이며 아래 과거10개/결합wrench/Time제거 기록보다 이 결정이 우선한다.
  Rearrangement의직전step이동량 수식·지지 거리 비용·물리/PPO는 유지한다. 자세한 의도와
  미결 논의는 문서 마지막「Target 보호 gate·초기 면적·F/T 분리와 Time 복원」을 따른다.
- 2026-10-23 학습 종료 후 Patch 용량262,144, observation ray64×48/min3,
  GUI의 청록색 관측 OBB 독립 toggle을 적용했다. 가시성 인증160×120과 기존 cache는
  유지한다. 시각화는 초기 관측 추정값이며 PhysX에 참여하지 않고 headless에서는 생성되지
  않는다. 검증 범위는 TRAINING_LOG48절을 따른다.
- 2026-10-22 사용자 승인으로 23:26:39 학습 분석의1차 weight 조정을 적용했다.
  Hold/pad **.05→.10**, Object stability **2→3**, 유효 성공 bonus **100→250**이며
  Clearance4·Distance.4·Rearrangement1·Motion weight와 모든 decay·threshold는 유지한다.
  Stability 경고는10mm, 실패는 같은 물체 지지 거리3cm가.1초 지속하는 기존 조건이다.
  Hold 상한1.7/step, 양의 dense 상한4.62848/step이고 γ=.98 할인합 상한은231.424다.
  실제 성공은.5초 유효 완료 후1회 지급하며 활성 실패가 우선이다. 새 학습은 실행하지 않았고
  value-loss/std 이상 해결을 확인한 변경도 아니다. 현재 설정은 TRAINING_LOG47절을 따른다.
- 2026-10-22 배치 생성 기본 경로를 규칙 기반 구성으로 변경했다. 한 seed에서 역할·균형 중복
  구성·전체360° 초기 yaw를 한 번 선택하고, 유한 slicing 계산과 두 방향의 해석적 compaction으로
  배치를 만든다. 기본 N=10, 4–10종, Target 종류1개, non-Target 개수 차이≤1이며 전체점유율
  20% 하한은 강제하지 않는다. Authored collider의 convex XY footprint로 가까운 이웃의
  3–10mm 간격과 Blocker 조작 공간을 다룬다. 외접원 회전 여유를 모든 물체에 예약하는 비교안은
  채택하지 않았다. 공유 장면 하나의 구획 계산은 CPU, 실제 메시 ray casting과 관측 생성은 GPU다.
  기하 mode마다 고정36개와 배치에 결합한1개를 모두 평가해 Target40–90%·Blocker≥95%를
  만족하는 mode별 시점을 GPU에서 계산한다. Base/guarded mode가 있으면 최대74쌍을 처음부터
  계산하고, 유효한 guarded mode를 우선한다. 실패 후 장면을 새로 뽑지 않는다. 회전 필요성·모든 메시 가시성의
  보장까지 완료됐다는 뜻은 아니다. 세부 규칙은 [SPAWN_RULES.md](SPAWN_RULES.md)의 현행 절과
  아래「2026-10-22 — 파라미터화된 배치 생성 설계와 사용자 확정 사항」의 후속 구현 절을 따른다.
- 2026-10-21 후속요청에따라Actor네트워크구조외학습은examples의random-sweep
  `rsl_rl_ppo_cfg_02.py`로통일했다. NativePPO를직접사용하고자체microbatch/KL/timeoutbootstrap을
  제거했다. Seed42,rollout36,90,000회,scalarstd1,8epoch/4batch,lr.001adaptive,γ.98,λ.95,
  KL.02,value1,entropy.005,gradnorm1,normalizationFalse다. ActorGNN/attention/headReLU는
  유지하고mean-head수동초기화는제거했다. 표준init_at_random_ep_len=True이며연구MDP의
  reset/vision/reward/controller는유지한다. 40절의설정유지·자체PPO방침은이번변경으로대체한다.
  구형logstdcheckpoint는원래설정으로play만지원하며새baselineresume로해석하지않는다.
  실제1024환경2iteration과기존정책4환경8stepplay를완료했다. 상세는TRAINING_LOG41절이다.
- 2026-10-21 사용자 동의에 따라 Actor를 공식 `MLPModel` 기반으로 정리했다. GNN/attention은
  `ContactManipulationEncoder`에서 특징만 만들고 native head464→256→128→18·forward·
  Gaussian·PPO 인터페이스를 상속한다. Raw mask 후 선택적history144D정규화·padding재적용을
  지원하지만 기본Actor/Critic정규화는False다. 기존logstd.3/entropy.01/fixedLR/KL없음/value.25와
  MDP·action·reward를유지해구조변경효과를분리한다. 동등한v1checkpoint의key와Adam순서를
  함께변환해play/resume를지원한다. CPUlearning70개(2CUDA제외)·MDP225개(16제외),GPU2검사와
  1024/8192배치정규화OFF/ON일치를확인했다. 실제Kit1024환경2iteration도완료했으며loss2.77/3.25,
  두번째iteration2.006초였다. 장기발산해결이나성공률개선을입증한검사는아니다.
  초기σ.3→1은명령noise분산11.11배이며 log/entropy/KL/value계수효과는단순배율관계가아니다.
  구현·설정별분석·검증한계는TRAINING_LOG40절을따른다.
- 2026-10-21 공식Actor/Critic/PPO와GNN연결구조를비교했다. 1,024환경의loss/GAE/optimizer는
  native이고Critic도공식독립MLP로FrankaLift와동일한폭/ELU/정규화OFF다. 따라서정규화OFF나
  raw previous action자체를공식흐름위반으로보지않는다. 우선검토할차이는logstd+entropy.01+
  fixedLR/KL없음, Actor의normalizationAPI거부, sceneLayerNorm을우회하는history144D,
  자체Triton/compile경로다. CPU회귀와GPU1024/8192배치의집계·Actor compile출력/gradient
  일치검사를통과했으나실제장기학습이상입력을재현한검사는아니다. 공식CNNModel처럼
  MLPModel의latent extractor만확장하는방향을제안했으나아직적용하지않았다. 앞서공식으로부른
  UR10OSC torque예제는로컬IsaacLab fork추가파일이므로upstream공식권장값의근거에서제외했다.
  비교표·근거·분리실험안은TRAINING_LOG39절을따른다.
- 2026-10-21 후속 traceback은 PPO update 중 Gaussian sample의 `std >= 0.0` 오류다.
  현재는exp(log_std)이므로 음수std보다NaN분포파라미터로해석한다. Critic loss의inf만으로
  독립Actor gradient오염을확정하지않으며 GAE·공통입력·log_prob/ratio경로를분리한다.
  제안순서는최초이상값보존/optimizer유한성검사→큰finite입력생성경로수정→공식Critic
  관측정규화→KL/entropy별도비교다. 아직production에적용하지않았고Actor정규화나새action
  제한도확정하지않았다. 상세와재현한CPU수치경로는TRAINING_LOG38절을따른다.
- 2026-10-21 반복되는 `Loss/value` 폭증의 의미를 설치된 native PPO·로그·CPU checkpoint로
  검토했다. 최신 `19-59-36`은284회42.49→285회4.276e11→287회39.10으로 초기에는
  batch별 극단값이고, 후반에는 지속되며2229회inf·2230회nan이다. 첫 spike 전후250/300회
  Critic weight는정상규모이고 실제입력·V·GAE target은저장되지않아 최초원인은미확정이다.
  Value loss는계수.25적용전 clipped제곱오차이며 실제reward가아니다. 입력정규화OFF·큰finite
  qdot/raw action이력과bootstrap target을다음진단에서분리해야한다. Std.315→56.79와
  return112.84→−29.15의동반변화는안정적인접촉회피수렴으로단정하지않는다.
  Production·MDP·PPO를변경하지않았고새학습을실행하지않았다.
  수치·한계·정확한수식은[TRAINING_LOG37절](TRAINING_LOG.md#37-2026-10-21--반복되는-value-loss-폭증의-의미와-최초-이상치-검토)을따른다.
- 2026-10-21 reward·termination 전체를 검토해 **유한 작업판까지의 단일 지지 거리**로 물체의
  낙하·이탈·큰 들림을 통합한다. 현재 선반 frame에서 `p=(OBB중심x,y,실제collider최저z)`와
  작업판 위 최근접점 사이 거리 `d`를 사용한다. 가장 큰 활성 물체 거리로 `object_stability`
  당시 weight2 비용을 계산하며1cm까지0·2cm에서−1·3cm에서−1.875였다. 현재 weight는 위
  2026-10-22 조정으로3이며 경고 거리와 곡선은 같다. 같은 물체의거리≥3cm가
  .1초 지속되면 `object_support_lost`다. 기존 `object_drop/ejected/off_support`는 제거하고
  tilt80°/.2초는 자세 목적이 달라 유지한다. 성공의 기존 잔류·지지·저속 **.5초** 조건과 실패 우선은
  유지하며 당시 bonus는 **+100**으로 변경했다(현재+250). F/T 두 증가 reward는 보정된 증가량 norm 하나와
  exp 하나인 `motion_wrench_growth` weight.30으로 통합한다. 작은 비용의 기울기는 유지하지만
  중간 영역의 곡선은 달라진다. 거리·접촉 유지·R·action-rate·joint-velocity·singularity는 목적이
  달라 각각 유지한다. Reward는10→10개, native 종료는9→7개다. 비활성Time 계산·설정,
  비활성hard-load종료경로와사용하지않는failure_penalty를제거하며센서범위진단은보존한다.
  **손가락 관통의 물리 설정은 이번 범위에서 변경하지 않는다.** 세부 통합·제거 목록과검증상태는
  [TRAINING_LOG36절](TRAINING_LOG.md#36-2026-10-21--rewardtermination-통합과-지지-위험-비용)을 따른다.
- 2026-10-21 비산을 성공과 양의 Clearance에서 제외하는1단계 수정을 적용했다. **아래는34절
  적용 당시 설정이며 bonus와 지지 실패 통합은 위36절 변경으로 대체됐다.** 사용자의
  annotation을 반영해 완료 유지시간은 **0.5초**다. 통로·Target 보호에 모든 활성 물체의
  작업판 잔류·signed 지지 gap −5~5mm·tilt<80°·선속도≤.05m/s·각속도≤.3rad/s와
  활성 실패 없음을 추가했다. 성공과 활성 실패는 상호배타적이며 성공만+50, 그외0이다.
  양의 Clearance는 현재 선반 frame의 지지 gap −5~10mm·중심 잔류·footprint overlap으로
  즉시 gate하되 속도 제한은 없고, 음의 Clearance와 R/M 비용은 유지한다. XY 완전 이탈
  2cm초과.05초·지지 gap3cm초과.1초를 물체별 native 실패로 추가했다. 기존 실패도 유지한다.
  활성 collider의 지지 hull은 asset별 시작 시 추출해GPU에 저장하며 **물리 collision 설정 변경이 아니다.**
  작은 지지 pivot의 OBB 하단 오판정을 줄이지만 실제 접촉력의 증명은 아니다. 다른weight·action·
  Blocker 일반 이동R제외는 유지했다. 실제10종 초기지지gap은0, 1,024환경64holdstep후
  −.182~1.711mm로 진행·완료 물리mask가모두정상이었고 부분reset도지역적이었다. 공통정점catalog와
  Warp reduction·cachedCUDAgraph로 helper.414ms/step, 고정holdnative비교40.34→39.82ms/step
  으로 뚜렷한지연증가는없었다. PPO·새scene reset제외측정이며 전체iteration성능을입증하지않는다.
  새Task_Count/Task_Rate는완료episode의누적실제개수·비율이며기존종료snapshot과구분한다.
  상세는 [TRAINING_LOG34절](TRAINING_LOG.md#34-2026-10-21--유효-재배치-성공과-비산-차단-1단계-적용)을 따른다.
  33절은 구현 전 계획이며 완료 유지시간.2초 제안은 이번.5초 요청으로 대체됐다.
- 2026-10-21 사용자가 play에서 관찰한 물체 튕겨내기·주변 물체 비산은 허용할 조작 성공으로
  취급하지 않는 요구를 확인했다. **32절 검토 당시** 성공은 XY통로와 Target보호만.1초 검사해 공중·영역밖·
  빠르게 움직이는 Blocker도 성공할 수 있고, success와 failure가 동시참이어도+50이다.
  CPU production판정 반례로 공중Blocker2.245m/s·10rad/s의성공과 낙하/전도/선반실패 동시성공을
  확인했지만 실제학습에서의오판정비율이나물리궤적재현은아니다. 기존checkpoint1350은
  scalar상평가후보일뿐 올바른밀기정책으로검증된것이아니다. 다음검토우선순위는 최종성공에
  실제선반지지·잔류·저속유지·활성실패없음을 추가하고, 양의Clearance만물리유효성으로gate하는것이다.
  미는도중의일시적pivot/lift/속도를일괄종료하지않으며 Blocker의일반이동R제외는유지하는방향이다.
  32절 검토에서는구현·weight·종료기준을바꾸지않았다. 근거·당시미적용수정안은
  [TRAINING_LOG32절](TRAINING_LOG.md#32-2026-10-21--물체-튕겨내기와-성공-판정의-허점-검토)을 따른다.
- 2026-10-21 후속 사용자 요청으로 Rearrangement와 singularity의 weight·곡선을 함께 조정했다.
  R은weight.6→1.0·a_R2→20이며1mm에−.019801,1cm에−.181269다. Singularity는
  weight.15→.30·warning.05→.07·half-deficit.025→.03으로, s=.06에서−.022238,
  s=.04에서−.15를 지급한다. 선반1cm/2°·singularity s≤.005 종료와 C4/D.4/H.05,
  다른 네 Motion각.15·성공+50은 유지했다. 기존 tensor 식의 상수만 조정해 추가 센서·물리 연산은 없다.
  아래30절은 이전 설정으로 수행한 검토이며, 이번 동시 조정은 그때 제안한 분리 실험과 구분한다.
  CPU259pass·18skip·381subtest(15.40초), 첫장면1,024환경32holdstep의 finite 보상·관측과
  nativeweight/dt·부분reset 검사를 통과했다. 그러나 세장면의 reset시점 SDK지표 검사 중
  한장면은 새warning36.23%(기존22.36%)·기존hard기준.1953%였으므로 초기상태 전체정상을
  확인한 결과는 아니다. 실제자세와 SDK갱신/기준점의 영향을 추가로 구분해야 한다. 적용·근거는
  [TRAINING_LOG31절](TRAINING_LOG.md#31-2026-10-21--rearrangementsingularity-weight와-경고-곡선-조정)에 기록한다.
- 독립 Critic의 본 학습 `2026-10-21_13-33-02`를1843회까지 재검토했다. 선반 종료 snapshot은
  1775–1799회7.37%→1800–1824회32.04%→1825–1843회4.30%였지만, 급증 전에 이미 return이
  −22.52이고 singularity69.05%·std56.99였다. Std는최종73.856, value loss는403회에1.827e11로
  커져 독립 Critic만으로 불안정이 해소되지는 않았다. 선반 penalty는 누락되지 않았지만 이번 step의
  corner 이동량에 `0.6(exp(−2ΔL)−1)`을 적용하므로1mm에−.001199,1cm에−.011881로 작고,
  초기 위치에서9mm 밀린 채 정지하면0이다. Weight 증폭만으로 원인이 해결된다고 판단하지 않는다.
  미적용 실험 후보는 Rweight.6을 유지한a_R2→20, 별도 실험으로 singularityweight.15를 유지하고
  warning.05→.07·half-deficit.025→.03으로 경고를 앞당기는 것이다. 후자는 reset 지표 검증이
  선행돼야 하며 weight.3은 이후 조건부 후보다. 제어·reward 코드는 이번 검토에서 바꾸지 않았다.
  로그 해석의 한계·분리 실험안은 [TRAINING_LOG30절](TRAINING_LOG.md#30-2026-10-21--독립-critic-학습의-선반-종료-급증과-reward-scale-재검토)을 따른다.
- 2026-10-21 사용자 요청으로 Critic을 공식 RSL-RL `MLPModel`로 분리했다.
  Actor의 GNN·역할 attention·ReLU fusion은 유지하며, Critic은 동일한 공개244D 관측에서
  `244→256→128→64→1`을 계산하는 독립 ELU MLP다. `obs_normalization=False`이며
  파라미터·latent·value cache를 공유하지 않는다. Value loss의 직접적인 Actor 역전파는 없어졌지만
  Critic이 GAE·advantage를 통해 정책 학습에 주는 간접 영향은 남는다.
  기존 공유 Critic 클래스·Actor 내부 value head를 제거하고 native 설정과 호출 방식을 사용한다.
  Reward·MDP·controller·PPO 하이퍼파라미터는 유지했다. 기존 공유 checkpoint는 전체 resume와
  현재 play에 호환되지 않으므로 새 학습을 시작한다.
  CPU259pass·18skip·381subtest, Actor CUDA compile1pass와 실제1,024환경2iteration을 통과했다.
  독립 checkpoint의 model·optimizer는 finite였고, 4환경8step play도 완료했다.
  첫29.96초에는 compile이 포함됐고 다음1.87초(collection1.662/learning.205)였다.
  두 iteration의 smoke 검사이므로 장기 처리량이나 정책 개선을 입증하지 않는다.
  상세 변경·검증 근거는 [TRAINING_LOG29절](TRAINING_LOG.md#29-2026-10-21--critic을-공식-독립-mlp로-분리)을 따른다.
- 2026-10-21 사용자가 보고한 OSC `linalg.inv` 오류는 full inertial decoupling의
  `inverse(J M⁻¹ Jᵀ)`에서 batch939가 singular로 판정돼 전체 호출이 중단된 경우다.
  현재 실패 행의 관절값·J·M은 저장되지 않아 기하학적 특이점과 상태/수치 문제를 구분하지 못했다.
  과거10월17일 replay에는 실제 near-singularity와 FP32 증폭이 확인됐다. IdealPD 제한은
  OSC 계산 이후이므로 이 예외를 막지 못한다. 최신 공식 develop에는 inertia conditioning이
  있으나 설치본에는 없다. 제어 설정은 이번 조사에서 바꾸지 않았으며 근거·후속 방향은
  TRAINING_LOG25절에 기록했다.
- 같은 날 최신 `2026-10-20_23-36-51`의 invalid 급증을 조사했다. Value loss가 먼저
  735회9.48e25→739회3.76e34로 커졌고740회Inf/invalid1.72%→741회invalid75.81%였다.
  모든 저장 checkpoint의 parameter·Adam state는finite였다. 고정된 합성 관절·센서 입력에서
  raw previous action만 되먹임한 CPU 검사에서는700은안정적이나750·950은자가증폭했다.
  이는 실제 실패장면 재현이 아닌 구조적 위험의 재현이다. Invalid는특이점/큰finite값검사가아니며,
  종료로그는현재불량률이아닌마지막episode원인의snapshot이다. 상세근거·개선방향은26절을따른다.
- 같은 날 사용자 요청으로 `motion_singularity` penalty(추가 당시weight0.15)와 `singularity` 종료를
  추가했다. 현재 wrist_3·팔6관절의 native public Jacobian을 step당한번 GPU에서 평가해
  보상·종료가 공유한다. `s=1/||inverse([Jv/.25;Jw])||F`는 최소 singular value의 보수적
  대리지표다. 추가 당시s≥.05이면0, s=.025이면−.075/step이었으며, 후속 조정은31절을 따른다.
  종료는s≤.005를 유지한다.
  SVD의 CPU동기화를 피하도록 공식 `torch.linalg.inv_ex(check_errors=False)`를 썼다.
  OSC·action·관측·기존weight·초기IK는변경하지않았다. 현재활성reward10개,실패5개다.
  CPU256pass와실제1,024환경32step을검증했다. 근거와한계는TRAINING_LOG27절을따른다.
- 후속학습 `2026-10-21_11-09-26`을2058회까지분석했다. 600–699회return124.0/success
  snapshot12.63%에서800–899회return66.7/success20.23%로갈라졌고,1600–1699회는
  return−13.6/success약0/singularity77.02%로붕괴했다. Invalid는0이나value loss가최대5.09e17,
  std가.309→2.762로증가해정상적인안정수렴으로볼수없다. Success종료는마지막보상+50을
  정상지급하지만C/D/H의반복수입때문에성공지연이유리할수있는구조는남아있다.
  Success종료제거보다반복수입과완료return의관계를먼저검토한다. 상세는TRAINING_LOG28절이다.
- 사용자관찰의정확한의미는학습초기정책은상호작용하지만후기정책은episode시작부터물체를
  건드리지않는다는것이다. 한episode에서조금민뒤후퇴한다는뜻으로해석하지않는다.
  최신1500–1574→1600–1699에서길이는57.38→56.67step인데D/H/C가지급로그상각각
  92.53%/99.987%/99.917%감소했다. 접근·지속접촉상실을뒷받침하지만후퇴의최적성은미확정이다.
  untouchedscene의C는0이며기존C이득유지로설명할수없다. 즉각비용/실패위험에의한접촉회피와
  정책불안정에의한능력붕괴를구분해야한다. 같은episode길이에서action-rate비용은오히려증가했다.

- 현재 observation은 **244D**(초기 물체 기하 100 + 현재/직전 각각 72)다.
  초기 기하만 사용하며 별도 temporal encoder 없이 GNN·역할 attention 결과와 concat한다.
- 10개 물체의 공통 배치·초기 object-only mesh raycast 정보를 공유하고,
  각 환경의 초기 로봇 자세는 reset마다 새로 샘플링한다. 실제 카메라는 생성하지 않는다.
- 공식 OSC·hand position ActionCfg와 arm IdealPD(P/D=0, 150/28 N·m)를 사용한다.
  Physics 100 Hz·policy 100 Hz(`dt=.01`, `decimation=1`)·공통 장면 최대8초다.
  2026-10-20 사용자 요청으로 정책50→100Hz를 적용했다. 이전 학습 로그는50Hz 당시 결과다.
- Hand는 URDF의 native mimic6개와 독립관절6개로 구성한다. 토크1N·m와 연동식을 반영한
  관절 범위를 공식 API로 설정하며17개 tactile 링크·binary observation·Blocker hold를 유지한다.
  결합 로봇 solver는position32/velocity2다. 접촉 시 종속관절의 순간 속도 초과는 남아 있다.
  적용·검증·수치 한계는 TRAINING_LOG21절을 따른다.
- 2026-10-20 후속 교정으로 OSC 제어 body·task frame·물체 observation의 기준을
  **UR5e 말단 `wrist_3_link`**로 통일했다. 공식 `task_frame_rel_path="Robot/wrist_3_link"`와
  scene `ee_frame`의 같은 링크 target을 지정하며 body offset은 없다. `tool0` 가상 frame이 아니다.
  Robot observation의 EEF pose도 같은 링크를 base에서 표현한다. Arm의 위치·회전 증분과
  stiffness 6축은 **정책 step 시작 시 현재 wrist 축**으로 해석한다.
  목표는 $p^*=p_t+R_t\Delta p_{\mathrm{EEF}}$, $R^*=R_t\operatorname{Exp}(\Delta r_{\mathrm{EEF}})$이며
  physics substep 동안 유지한다. 사용자가 켠 `gravity_compensation=True`를 유지해 native OSC가
  관절 중력 토크를 feedforward로 더한다. F/T의 sensor-frame 자중·관성 측정 보상과는 별개다.
  Gain·scale·hand 동작·244D observation 모양은 유지하며 학습 개선은 아직 확인하지 않았다.
  규약은 `isaaclab_ur5e_wrist_osc_variable_kp_gravity_v3`이며 이전 base-frame(v1)·손바닥
  EEF(v2) checkpoint는 로드 전에 거절한다. 아래 과거 기록과 10월19일 reward ablation은 base-frame
  당시 결과다. 손바닥 기준으로 변경했던 중간 단계는 TRAINING_LOG11절, 최신 변경은 13절을 따른다.
- 출력은 RSL-RL 공식 GaussianDistribution으로 전환했다. Tanh 없이 학습 시 Gaussian sample,
  play 시 mean을 사용하며 초기 std0.3은 유지했다. 이전 tanh checkpoint는 분포 검사에서 거절한다.
- 현재 활성 native reward는 **Clearance4.0·거리0.4·Hold0.10·Rearrangement1.0·Action rate/Joint velocity각0.15·Wrench growth/Singularity각0.30·Object stability3.0·Terminal의10개**다.
  Clearance는 `exp(−a·x(t))−exp(−a·x(0))`로 초기 가림 대비 개선을 지급한다.
  초기와 같으면0·개선양수·악화음수이며, 개선 상태를 유지하면 양의 지급이 계속된다.
  거리는 양의 exp 점수, Motion은 `exp(−ax)−1` penalty다. Clearance 기준은 부분reset행만 갱신한다.
  Rearrangement는 지정Blocker를 제외한 active물체(Target포함)와 선반의 OBB8corner
  직전step변위 합을 `exp(−20x_R)−1`로 지급한다. Time과 hard-load 종료는 제거했다.
  선반localOBB는공식USD API로한번측정하고기존pose로GPU변환하며부분reset복원은과금하지않는다.
  Terminal은 유효 성공+250·그 외0을 episode당한번 지급한다. 활성 failure가 우선이므로
  원래 목표와 failure가 동시에 참이어도 유효 성공과 성공 보너스는 없다.
  최신22-33-03의초기값차감후101–200회C/D지급비.753을근거로C1.8→4를적용했다.
  Hold는 같은 pad의 직전·현재 Blocker 접촉을17개 합산한다. .05→.10 변경으로 최대1.70/step이다.
  Object stability는 경고10mm를 유지하며2cm에서−1.5·3cm에서−2.8125다.
  현재 γ=.98·rollout36이며 이번 weight 조정에서 PPO·scale은 유지했다. Dense reward는 per-policy-step이므로
  같은8초 지급 횟수가50Hz의 두 배가 되고, rollout의 simulation 시간은0.36초다.
  마찰은 현재 μs=.5/μd=.5를 유지하며 추후 물성 조정은 별도 실험으로 비교한다.
- 2026-10-20 종료 조건을 완화했다. `workspace_exit`·`object_lift`를 제거하고,
  `object_tilt`는 **같은 물체가 80° 이상 기울어진 상태를 0.2초 연속 유지**할 때만 작동한다.
  현재100Hz에서 물체별20step이며, 80° 미만이면 해당 누적값을0으로 만든다. 해당 환경 reset에서도
  초기화한다. 밀기·pivoting 중의 순간 기울어짐을 허용하려는 변경이며 성공 판정의 안정화·Target
  보호·Blocker 통로 확보는 유지한다. 들림·작업 영역 이탈 자체는 실패나 성공 gate가 아니다.
  이후16절에서 성공은 Target 보호·지정 Blocker 통로 확보만10step(.1초) 연속 검사하도록 분리했다.
  Tilt pending·속도·F/T·선반 상태는 성공 gate가 아니다. 활성 실패는 같은 step에 따로 기록될 수 있다.
  `object_drop`의 지지면 아래6cm·선반 변위·invalid state 기준은 유지하며 hard-load는 진단만 남긴다.
  10월21일부터singularity 실패가추가됐다. 이전 학습의 실패 원인 7개 기록은 당시 조건이며,
  이번 변경 후 새 학습 성능은 아직 미확인이다.
- 10월19일 추가 학습 `15-43-41`과 네 reward 조합 실험을 분석했다. 설정 차이는 활성 reward다.
  공통 checkpoint750을 같은 4장면·8시작자세에서 mean/sample로 평가한320episode는
  통로확보·성공0회, 과부하/센서범위초과278회였다. 초기 raw Clearance0.462601에서 개선이 거의 없다.
  조합별 mean 재생 생존은65.8–324.3step으로 달라, 행동 차이가 작업 진전으로 연결되지 않는 상태다.
- 최신 장기 checkpoint11550의 std는 회전·강성·hand에서 수천으로 커졌고 모든 tensor는finite였다.
  평균 행동만 써도 위치목표증분 norm평균12.14m·성공0/32로 악화됐다. 실제 로봇이12m이동한뜻은아니다.
  Reward 함수의0.5상한이 아니라 큰 명령·반복과부하·분포 및 평균정책 악화를 우선 분리해야 한다.
  원인 확정이나 production 수정 없이 근거·한계를 **TRAINING_LOG10절**에 남겼다.
- 초기자세는 Blocker 앞 위치와 EEF-local 회전을 randomize하고 검증된 기준 관절값을
  우선 seed로 쓴다. 같은 목표에 다른 branch를 재시도하며, 실패 행만 제한적으로 복구한다.
  최종 실패 시 같은 장면의 검증된 기준 자세로 복구하고 횟수·환경 ID·seed를 남긴다.
  이는 해당 예외의 새 자세 randomization을 포기하는 정책이다. CUDA719의 원인 해결과 구분한다.
  이 배치용 IK의 TCP는 손바닥 `right_plam_force_sensor`로 유지해 초기 hand가 Blocker 앞에 오게 한다.
  제어·관측 wrist frame과 용도가 다르다. 물리 F/T는 기존 `ToolForceTorqueLink` sensor frame,
  tactile은 기존 17개 링크를 유지하며 공유 배치·reset·reward·종료 조건은 이번 frame 변경으로 바꾸지 않는다.

## 1. 이전 프로젝트의 당시 위치와 상태 구분

**스폰·센서 위에 학습용 MDP와 모델·PPO를 구현했으며, 시간 입력을 현재·직전
두 sample로 변경했다.**
사용자의 후속 교정에 따라 **mask는 observation 채널에 포함하지 않으며,
모델 호출 전에 observation 정보만으로 생성**한다. 이어 역할 ID 대신
**Target 슬롯 0·Blocker 슬롯 1을 고정**하도록 교정했다. 현재 policy observation은
초기 물체 기하와 두 72D sample을 합친 **544D**다.
장면 확인용 `ManagerBasedEnv`와 학습용 `ManagerBasedRLEnv`를 구분한다. 사용자가
초기에는 최근 16 step 이력을 선택했으나 후속 요청에 따라 **현재 t와 직전 t−1의
센서·로봇 상태·행동 두 sample**로 줄였다. 18D OSC/hand action,
Reward v2·종료 조건, 초기 장면 GNN·역할 조건부 attention·유한 이력 encoder,
공동학습 PPO를 구현했다. 물체 입력은 승인한 초기 RGB-D에서 추정한 기하만
사용한다. 초기 기하를 robot base에 저장하고 매 policy step에 **현재 EEF
좌표계로 재표현**하며 현재 실제 물체 pose를 actor에 넣지 않는다. 제어 기준
좌표계는 `examples`의 sweeping 프로젝트처럼 robot root/base다. 아래의
스폰·센서 GPU 검증을 새 MDP의 학습 성공이나 제어 검증으로 확대하지 않는다.
Reward·제어 수치는 provisional default이며 정책 성능은 아직 입증하지 않았다.
후속 Time 교정은 `T_ref`를 제거하고 초반부터 완만하게 증가하다가 후반에
가팔라지는 지수형 비용을 적용한다. 환경별 episode clock은 reward 내부에 두며
현재 544D observation에는 추가하지 않는다. 자세한 수식과 개형은 4절에 둔다.
Clearance·Rearrangement·Motion은 `exp(−a·x)` 품질 점수로 통일하고,
Rearrangement는 모든 물체의 평균 OBB corner 3D 경로를 누적해 병진·회전을
평가한다. 실제 물체 pose는 privileged reward·검증에만 사용한다.
F/T 측정 모델은 사진의 **ATI Axia80 SI-200-8** 범위·유효 분해능으로 설정하고,
측정 후 자중·관성을 보상해 축별 full scale로 정규화한다. 센서 전체 하중의
범위 초과와 기존 보정 하중의 작업 안전 한계는 구분한다.
코드 구조는 `examples`의 `sweeping_policy`를 따라 공통 환경 설정,
`config/ur5e`의 로봇·학습 설정, `mdp`의 manager 함수, `src`의 장면 지원 구현으로
정리했다. 학습은 `Isaac-Contact-Rearrangement-UR5e-v0`의 등록된 설정으로 실행한다.
다중 환경 학습은 공통 초기 장면을 사용한다. 전체 reset마다 물체 배치·역할·로봇
초기 자세를 새로 정하고 초기 vision을 한 번 승인해 모든 환경에서 재사용한다.
부분 reset은 현재 공통 초기 장면을 복원하며 물리 상태·행동·센서는 환경별로
독립적이다. 카메라는 실제 RGB-D와 비가림 기준 실루엣용 총 2개를 공유하고,
Headless 실행에서는 자동 렌더링을 생략하고 초기 촬영 때만 명시적으로 렌더링한다.
카메라 버퍼는 계속 유지하며 env 0만 시각적으로 렌더링한다. 물리 replica는
Fabric으로 복제한다. PPO는 원래 minibatch의 gradient를 최대 32 sample씩 누적하며,
학습 흐름은 예제의 **Hydra → gym.make → RslRlVecEnvWrapper → OnPolicyRunner → runner.learn**이다.
Rollout storage·GAE·학습 반복·콘솔·TensorBoard·checkpoint는 native RSL-RL이 담당하며
연구 network 공유·정확한 final-observation bootstrap·microbatch·profile 검사만 확장한다.
환경은 SceneCfg와 manager/event 구조를 유지하고 RewardTerm 5개로 항별 로그를 생성한다.
기본 출력 경로는
`logs/rsl_rl/contact_rearrangement/<시작 시각>/`이다.
새 패키지 `src/contact_rearrangement`는 **10종 asset 중 4–10종을 선택하고
전체 선반 밀도에 필요한 수의 instance를 배치**한다. 이전에 4–10종을
active instance 4–10개로 해석한 것은 잘못이며 종류 수와 개수를 분리했다.
`--all_asset_types`는 10종 모두 선택하고 non-target 복사본을 추가할 수 있다.
**Target 종류는 active 장면에 한 번만 등장**하고 Blocker·주변 후보에서 같은
종류의 모든 복사본을 제외한다. 종류마다 최소 한 instance를 선택한 뒤 footprint
면적에 따라 개수를 정한다. 추가 가능한 non-target 종류 중 active 개수가 가장
적은 종류를 고르고 동률은 무작위로 정해 중복 편중을 줄이는 구성으로 변경했다.
Blocker도 해당 종류의 첫 instance로 센다. Target은 항상 하나이며 현재 동일
pool capacity에서는 non-target 종류별 개수 차이가 최대 1이다. 별도 종류별
최대값은 없고 pool의 10-copy·전체 cap 40을 유지한다. 작은 물체 위주 종류
집합은 같은 면적을 채우려면 더 많은 instance가 필요할 수 있다.
기본 instance 최소 4·최대 40은 safety limit이다.
10종 × 10개 복사본의 100-body pool을 유지한다. Pool 전체의 rigid-body
simulation과 native gravity를 켜 두고,
나머지는 높은 격리 위치에 주차해 매 physics substep마다 COM에 `-m*g` 외력을
적용한다. Active 물체에는 추가 힘을 주지 않아 원래 중력과 접촉을 유지한다.
Episode의 Target·Blocker를 먼저 지정하고 **종류 수·집합, 전체 active instance
ID와 개수**를 pose sampling 전에 고정한다. Retry에서는 yaw와 위치만 바꾼다.
Target은 Blocker가 앞에 들어갈 수 있는 전체 X 범위에서 선택한다. 기존 뒤쪽
10 cm 제한과 38 cm cluster·35% tight-rectangle fill 조건은 폐기했다.
현재 구현은 전체 usable shelf 점유율 **35% 이상**, X 2 × Y 3의 **6개 구역
각각 10% 이상**, 4 cm footprint 이웃을 가진 active 비율 **75% 이상**을 hard
acceptance 조건으로 요구한다. X/Y coverage와 병진·평면 pivot·회전 후 이동의
다단계 clearance witness도 검사한다. 사용자는 구현 가능한 범위를 먼저 측정해
밀도를 정하도록 요청했고, 25–45% 비교에서 생성 속도와 성공률을 고려해 35%를
초기 기본값으로 선택했다. 사용자가 직접 정한 최종 연구 난이도 수치는 아니다.
앞쪽 +X camera의 실제 초기 영상에서 Target 가시 비율 **40% 이상**을 요구하기로
했다. 분모는 같은 camera·같은 pose에서 가림 없이 렌더링한 전체 Target
silhouette 픽셀 수다. 현재 앞쪽 RGB-D camera와 실제 silhouette acceptance를
구현하며 **Blocker 전체 가시성**도 함께 요구한다. Target·Blocker 둘 다 전체
bounds가 frame 안에 있어야 한다. 승인한 초기 snapshot은 episode 동안 고정한다.
YCB 003·004·005·006의 초기 회전을 **X축 −90°**로 교정하고, 원본 USD의
수정 전·후 각 8장 렌더에서 라벨·캔 고리·병뚜껑으로 상하를 확인했다.
이전 큰 footprint 우선 복사본 구성의 35% 규칙은 실제 USD 측정 bounds로 일반·all-types 각각 seed 0–19,
**총 40개 기하 sample**을 통과했다. 일반 모드는 4–10종·10–24 instance,
all-types는 10종·14 instance였다. 일반·all-types 각각 **4개 GPU 환경·10초**의
정착·전체/부분 reset 검사도 통과했다. 각 모드의 8개 layout을 각각 5초 진행한
범위이며 기하 benchmark와 물리 검사를 구분한다. 당시 unit test 21개도 통과했다.
이 분포와 통과 결과를 현재 균형 구성의 결과로 사용하지 않는다. 현재 균형
구성은 Blocker 앞 reserve·camera gate 추가 전 기하 sample **40/40개**와
일반 모드 **4개 GPU 환경·151 step·5.03초**의
정착·전체/부분 reset 검사를 통과했다. 당시 unit test **23개**도 통과했으며
현재 구성의 별도 all-types GPU 실행은 완료한 것으로 기록하지 않는다.
**전물체의 2 mm 낙하 여유를 제거**하고 003·004에만 visible mesh를 유지한
**명시적 Cube collision proxy**를 적용해 초기 흔들림을 줄였다. 원본
Nucleus USD·authored mass·contact 속성·physics material·active gravity는 유지한다.
이전 정확히 10개 instance 버전에서는 이 collider로 **4개 GPU 환경·10초**의
정착·전체/부분 reset 검사가 통과했다. 이 결과와 이전 실패·비채택 approximation,
새 종류/instance 분리 benchmark는 8절에서 구분한다.
판단·행동 모델과 PPO 코드는 추가했으며 학습 성능은 아직 확인하지 않았다.
사용자 요청으로 장면 확인 스크립트에 **Target 초록색·Blocker 빨간색 OBB**의
debug draw 표시를 추가했다. 물리·USD asset을 수정하지 않는 확인용 표시이며
2개 CUDA 환경의 headless reset 검사와 1개 GUI 환경의 실제 표시 검사가 통과했다.
2026-10-09에는 사용자가 파일이 지나치게 나뉜 구조를 정리하도록 요청해
구현을 책임별 **6개 모듈**로 통합했다. 당시 `catalog`·`assets`·`env_cfg`·`spawn`·
`sampling`·`visualization` 구조였으며 스폰 규칙·기본값·실행 명령을 유지했다. 통합 후 unit test **23개**와
일반 모드 **4개 GPU 환경·151 step·5.03초** 검사가 통과했으며 초기·자동 reset
뒤 8개 장면 설명이 통합 전과 같았다. 아래 기존 실행 결과와 구분해 기록한다.
사용자는 **Blocker 앞의 충돌 없는 hand 시작 영역 randomization과
elbow-down 금지**를 요청했다. RobotStartRules·실제 USD FK/IK·초기 collision
판정을 구현했고 2026-10-09의 24개 시작·반복 reset·5초 물리 검사와 unit test
**29개**가 통과했다. 당시 손가락/tool은 아래를 향했다. 사용자의 후속 요청에
따라 tool 전방을 **선반 방향인 world −X**로 교정했고 palm normal이 위를
향하던 이 단계의 4개 GPU 환경·151 step·5.03초 검사도 통과했다. 이어
사용자가 tool 기준 Z축으로 hand를 **−90° 회전**하도록 요청해 mount에 roll을
추가한다. 현재 tool 전방은 −X를 유지하고 palm normal은 world −Y를 향한다.
새 hand roll의 4개 GPU 환경·151 step·13개 시작 pose 검사도 통과했다.
앞 단계의 통과 결과와 새 hand roll의 검사를 구분한다. 후속 요청으로 실제
센서·연결부의 공간을 고려해 UR5e 말단과 hand 사이에 **길이 5 cm·직경
5 cm의 원통 연결부**를 추가한다. 연결부는 wrist-3의 compound collider이며
별도 body나 DOF를 만들지 않는다. 연결부 추가 전 검증과 새 조립 검사를 구분한다.
새 연결부의 4개 GPU 환경·151 step·13개 시작 pose 검사도 통과했다.
후속 센서 단계에서는 원통을 25 mm 두 rigid half로 나누고 중앙 fixed joint의
실제 F/T를 읽는다. 앞쪽 camera pose를 무작위로 제안해 Target ≥40%·Blocker
100%(면적 2 pixel 허용)를 실제 렌더로 승인하며 raw F/T와 초기 RGB-D/mask를
`sensors` observation group에 제공한다. 이번 센서 단계의 검증은 별도로 기록한다.
최종 4개 GPU 환경 검사에서 18개 시작 pose의 실제 가시성과 capture 중 native
state 무변경, 초기 snapshot 보존·F/T 유효성·알려진 force/torque 응답이 통과했다.
가시성 proposal을 개선하려고 `reserve_blocker_front=True`로 Blocker 앞의
직선 +X 통로를 주변 물체로부터 비운다. 기존 전체 선반 밀도와 Target A₀는
유지하며 이 통로가 실제 가시성을 보장하는 것으로 취급하지 않는다.
현재 hand의 17개 tactile link 각각에 독립 `ContactSensor`를 붙이고 world net
normal force norm **>0.01 N**을 binary contact로 제공한다. 2개 GPU 환경에서
17개 pad의 실제 contact ON/OFF·부분 reset·render cache 보존 검증도 통과했다.

| 상태 | 의미 |
|---|---|
| 결정 또는 방향 | 사용자가 선택한 연구 방향. 구현 완료나 성능 입증을 뜻하지 않는다. |
| 현재 작업안 | 현재 구체화 기준으로 삼는 설계. 세부 구조와 수치는 더 정해야 한다. |
| 구현 기본값 | 지금 코드 실행을 위한 설정. 연구 전체의 최종 명세로 간주하지 않는다. |
| 제안 및 미채택 | 검토 후보. 사용자의 질문이나 설명 요청을 채택으로 기록하지 않는다. |
| 검증 결과 | 명시한 조건과 검사에서 실제 확인한 결과. 검사 범위 밖으로 확대하지 않는다. |

우선순위는 **최신 사용자 결정 → 현재 연구 맥락 → 과거 설계·발표 자료**다.
코드의 실제 동작은 해당 코드와 실행 결과로 확인한다. 연구 설계와 코드가 다르면
어떤 부분이 구현 전인지, 이전 실험의 설정인지 명시한다.

현재 코드 구조는 사용자의 후속 요청에 따라
`examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy`를 기준으로 정리했다.
공통 환경 설정, `config/ur5e`의 로봇별 설정·Gym task 등록·agent 설정,
`mdp`의 manager 함수, `src`의 장면 지원 구현을 분리한다. Shared transition은
`mdp/state.py`가 한 번만 갱신하며 물리·관측·reward·18D action 명세를 유지한다.
현재 파일 위치와 task registry 기반 실행은 7절과 패키지 README에 정리한다.

## 2. 연구 목적과 범위

근사적인 초기 물체 기하와 촉각·힘/토크 피드백을 이용해, 명시적 Blocker 종착
pose 없이 Target 접근 공간을 확보하는 신속한 비파지 재배치 정책을 학습한다.

- **Target:** 접근 공간을 확보하려는 물체. 역할은 외부에서 지정한다.
- **Blocker:** 접근을 방해하며 주로 조작할 물체. 역할은 외부에서 지정한다.
- **주변 물체:** 추가 접촉·충돌·차단이 발생할 수 있는 물체.
- **조작:** 선반의 지지를 받으며 밀기·회전으로 재배치한다. 파지해서 들어 옮기는 것을 주된 해법으로 삼지 않는다.
- **정책 역할:** Blocker의 최종 goal pose를 사전에 입력하지 않고 재배치 판단과 로봇 행동을 함께 학습한다.
- **실행 목표:** 접촉 변화에 빠르게 대응하면서 성공·안정성을 유지하고 수행시간을 줄인다. Rapid는 아직 입증되지 않은 연구 목표다.
- **환경과 장비:** 고정 선반, UR5e + RH56E2 hand, wrist F/T, 17개 위치의 binary tactile. 선반 형상은 현재 observation 명세에 포함하지 않는다.

가림과 근사 형상 때문에 유효한 종착 pose를 사전에 안정적으로 특정하기 어려울
수 있다는 문제 설정이다. 상위 모델이 goal pose를 생성할 능력이 없다는 주장은 아니다.
실제 Target 파지·인출까지 학습할지, 여러 Blocker의 조작 순서를 다룰지는 미정이다.
발표용 제목은 **신속한 비파지 재배치를 위한 촉각 및 힘·토크 피드백 기반
판단·행동 통합 심층 강화학습 정책**이었다. 학위논문의 최종 제목 확정과 구분한다.

## 3. 현재 관측과 정책 작업안

### 관측과 시간 구조

최신 방향은 **초기 vision을 사용하고, 이후 robot state·F/T·tactile·행동 이력으로
이번 episode의 환경 변화를 해석하는 것**이다. 지속적인 vision 갱신을 기다리는
의존성을 줄여 촉각·힘 피드백의 빠른 대응을 살리려는 의도다. 초기에는 16-step
window를 구현했으나 사용자가 **previous step까지만 제공**하도록 변경했다.
현재 입력은 **현재 t와 직전 t−1의 두 sample**이며 newest-first로 저장한다.
Policy는 30 Hz, physics는 120 Hz이므로 sample 사이 간격은 약 33 ms다.
Reset 직후 현재 sample만 valid, 직전 sample은 zero/invalid로 두며 그보다 오래된
이력은 제공하지 않는다. Episode를 넘어 유지하는 GRU/LSTM은 도입하지 않았다.

| 입력 | 현재 구현 | 후속 교정 |
|---|---|---|
| 초기 물체 기하 | 물체별 위치 3D + quaternion(wxyz) 4D + OBB extent 3D = 10D, 현재 EEF 기준 | 단일 시점 visible depth의 근사 오차와 실기 detector |
| 물체 역할 | K=40 중 슬롯 0=Target·1=Blocker·2–39=주변 물체. 역할 ID 채널 없이 고정 슬롯과 유효 기하로 mask 생성 | 실기에서의 identity/role association과 초기 정렬 |
| Robot state | Arm 6개·hand 독립 6개 관절의 q/dq 24D + base 기준 EEF pose 7D = 31D | 실제 구동 관절 정규화·동기화 |
| Wrist F/T | SI-200-8 측정 범위·분해능 근사 후 sensor-frame 자중·병진/회전 관성 보상 force/torque 6D, 선택적 residual bias tare | Filtering·실제 질량/관성·가속도 추정과 bias 교정 |
| Tactile | 17D, 0=비접촉·1=접촉·−1=결측. 유효성 mask는 이 값으로 생성 | 0.01 N은 변경 가능한 초기 threshold |
| Previous action | 해당 관측 직전 적용한 normalized action 18D | Policy/control 주기 교정 |

K는 Target·Blocker를 포함한 전체 최대 슬롯 수다. 같은 물체 기하를 역할별로
중복 계산하지 않는다. 모든 가려진 물체를 초기 vision으로 알고 있다는 가정도 하지 않는다.
관절 관측 차원은 URDF의 전체 물리 관절 수와 독립 구동 관절 수를 구분해 구현해야 한다.

초기 기하는 승인한 camera snapshot의 instance mask 안에 있는 visible depth를
back-project하고 PCA OBB로 근사한다. Simulator instance ID는 물체별 분할과
외부 Target/Blocker 역할의 association에 사용하므로 이 부분은 oracle이다.
USD의 전체 bounds나 현재 실제 물체 pose를 geometry feature로 복사하지 않는다.
관측하지 못한 물체는 invalid padding이다. 가림으로 extent가 작게 추정될 수 있으며
완전한 물체 형상을 복원했다고 해석하지 않는다.

Camera extrinsic으로 변환한 초기 geometry의 canonical copy를 robot base에
저장한다. 매 policy step마다 현재 EEF pose의 역변환을 적용해 위치·orientation을
현재 EEF 기준으로 표현하고 extent는 유지한다. 이는 world에 고정한 **초기
추정 장면의 재표현**이다. 조작 후의 실제 물체 이동을 vision으로 갱신하거나
정답으로 대체하지 않는다. 제어·robot EEF pose는 base, 물체 입력은 현재 EEF,
F/T 입력은 sensor frame을 각각 사용한다.

한 history sample은 `31 + 6 + 17 + 18 = 72D`다. Actor flat observation은
`geometry 40×10 + history 2×72`
순서로 **544D**다. 현재 sample에는 해당 관측 직전 적용한 action aₜ₋₁,
직전 sample에는 aₜ₋₂를 포함한다. Reset 첫 현재 sample의 action은 0이다.
관측 생성 단계에서 외부 역할 association으로 Target·Blocker의 기하를 각각
슬롯 0·1에 배치한다. 관측된 주변 물체는 기존 상대 순서대로 슬롯 2부터 채우고
나머지는 0이다. 역할 물체가 미관측이면 예약 슬롯은 0으로 유지하며 다른 물체로
대체하지 않는다. 역할 ID 채널은 observation에서 제거했다.
`build_policy_masks(observation, cfg)`가 observation만으로 물체·역할·history·
tactile mask를 생성한 뒤 `model.act(obs, masks=...)`로 별도 전달한다.
물체 유효성은 finite 기하·양의 extent·0이 아닌 quaternion으로, history 유효성은
finite robot/F/T/action·0이 아닌 EEF quaternion으로 판단한다. 미관측 물체와
reset 직후의 직전 sample 슬롯은 zero padding, 결측 tactile은 −1이므로 추가 내부 정보가
필요하지 않다. Reset 현재 tactile도 −1이다. Boolean mask는 learned feature로
concat하지 않으며 결측 tactile은 encoder에 넣기 전에 0으로 필터링한다.
관절 q/dq는 π·2 rad/s, F/T는 축별 SI-200-8 full scale
`[200,200,360,8,8,8]`(힘 3축 N, 토크 3축 N·m)로 나누고 기존 `[-5,5]` clip을 유지한다.
과거의 scalar 50 N·5 N·m 정규화는 제거했다.
과거 `72 + 11K`의 current-only 입력안은 이력 구조로 대체했다.

Actor가 현재 실제 object pose·velocity를 받기로 한 것은 아니다. 시뮬레이터
정답을 reward·검증·학습 label에 쓰는 것과 actor 실행 입력에 쓰는 것은 구분한다.
History를 외부 상태 추정기로 해석한 뒤 추정 상태를 actor에 넣는 구조는 지양한다.

### 관계 처리와 학습

현재 구현 구조는 다음과 같다.

```text
Observation 기하의 고정 역할 슬롯·zero padding·tactile 결측값
  → 모델 호출 전 물체·역할·history·tactile mask 생성
초기 물체 기하(현재 EEF 재표현)
  → 공유 node encoder / dense GNN 2 rounds
  → Target·Blocker 조건부 masked attention readout
현재 t·직전 t−1의 robot state·F/T·tactile·action
  → 두 sample을 받는 causal convolution
두 encoder feature 결합
  → bounded Gaussian actor의 18D action + shared critic
```

- Target·Blocker 슬롯 0·1은 역할을 나타내므로 고정한다. 주변 물체 슬롯 2 이상은 순서를 바꿔도 같은 장면에 대해 같은 판단을 하도록 집합을 처리한다. 실제 공간 배치는 입력에 남아야 한다.
- GNN edge는 메시지 교환을 허용하는 연결이다. 접촉·방해 관계의 정답 label을 미리 제공하는 것으로 정하지 않았다.
- 로봇에 대한 물체 위치를 판별할 정보가 필요하다. 물체 간 상대 관계만 남기는 구조로 단순화하지 않는다.
- Role mask와 valid mask는 목적이 다르다. Zero padding에서 valid를 만들면 noise·정규화 전에 생성하고 message passing과 attention에서 padding을 제외한다.
- GNN의 같은 forward 내 message-passing 반복과 episode의 시간적 기억은 다르다.
- Node 64D, message passing 2 rounds의 dense valid edge를 사용하며 self/padding edge를 제외한다. Edge feature는 초기 물체 간 상대 위치·extent 차이다. Role-conditioned query는 Target·Blocker embedding에서 만들며 두 attention readout과 role/global feature를 결합한다.
- Node encoder는 기하 10D, history projection은 sample 72D를 입력으로 받는다. Mask는 learned feature로 concat하지 않고 선택·필터링에 사용하며 역할 ID 채널도 없다.
- History encoder는 64D causal convolution이며 기본 1개 layer로 현재·직전 두 sample을 처리한다. Padding을 제외하고 유효 sample만 pooling한다. Fused latent는 128D이며 기본 모델은 247,717개 parameter다. 더 오래된 sample이나 지속 hidden state는 제공하지 않는다.
- Native RSL-RL runner/PPO에 연구 모델 adapter를 연결해 scene/history encoder와 actor/critic을 공동학습한다. Actor와 critic은 같은 544D 입력과 encoder를 공유하며 privileged critic·현재 물체 GT 입력·보조 예측 head/loss는 추가하지 않았다. Rollout·PPO minibatch·timeout 마지막 관측에서 모두 observation만으로 mask를 다시 생성한다.
- Tanh-Gaussian action의 log probability에는 변환 Jacobian을 반영한다. GAE는 실제 terminal state에서 bootstrap하지 않고, timeout에서는 auto-reset 이전 실제 final observation의 value로 bootstrap하며 다음 episode의 advantage를 이어 붙이지 않는다.

이전 16-step 모델은 관측 1960D·286,053 parameter, mask 채널을 포함했던
2-sample 모델은 700D·248,997 parameter, 역할 ID를 사용하던 모델은
584D·247,717 parameter였다. 현재 모델은 고정 슬롯 관측 규약
`observation_layout="fixed_roles_v1"`을 checkpoint에 저장한다. 역할 ID 제거 전후
network weight shape가 같아도 관측 규약을 검증하며 이전 checkpoint를 직접
복원하지 않는다. 아래의 기존 검증 기록은
각 당시 모델의 결과이며 현재 규격의 검증과 구분한다.

### Action과 제어

| 출력 | 차원 | 제어 경로 |
|---|---:|---|
| EEF motion | 6 | 위치·회전 증분 → variable-stiffness OSC |
| Cartesian stiffness | 6 | 병진·회전 각 3축 강성 → OSC |
| Hand configuration | 6 | RH56E2 독립 구동 관절 목표 위치 → position controller |

총 **18D action**이며 arm의 12D 출력은 OSC를 통해 관절 토크로 연결한다.
목표를 이전 명령에 계속 누적하지 않는 방향을 유지한다. 직전 action을 관측하는
것과 목표 누적은 별개다. Examples의 robot-root reference 구현을 따라 증분은
**base 기준**이며 매 policy step의 실제 TCP pose에서 position과 orientation
목표를 새로 만든다. 회전 증분은 base 축 기준으로 현재 quaternion에 left-multiply한다.
기본 scale은 위치 축별 ±15 mm·회전 축별 ±0.08 rad다.
현재 제어는 사용자의 "예제들처럼 controller class 사용" 요청에 따라 native action 두 개로 나뉜다.

- `arm_action`: `ArmOscAction(OperationalSpaceControllerAction)`의 12D.
  `OperationalSpaceController`가 `pose_rel`·`variable_kp`를 받아 토크를 계산한다.
- `hand_action`: `HandPositionAction(JointPositionToLimitsAction)`의 6D.
  Native joint-position action이 PhysX `ImplicitActuator` PD의 목표 위치를 설정한다.

연구 adapter는 normalized input 매핑, workspace/effort 제한, PhysX COM Jacobian과
velocity를 TCP link origin/base 축으로 맞추는 변환만 추가한다. 목표 pose 생성과
spring/damping/관성 분리/중력 보상은 native OSC가 수행한다.
Custom backward-Euler damping, matrix geometric mean, 32회 Newton–Schulz,
별도 damping CUDA graph는 제거했다. 제어에서 실제 TCP velocity를 native OSC에 전달한다.

기본값은 full inertia decoupling·gravity compensation 활성화, damping ratio 1이다.

$$
\Lambda=(J M^{-1}J^T)^{-1},\qquad
\tau=J^T\Lambda\left[K_p e-2\sqrt{K_p}v\right]+G.
$$

병진 gain 100–800·회전 gain 5–40은 **관성 정규화 gain(s⁻²)**이다.
이전 controller의 물리 강성 N/m·N·m/rad와 구별해야 한다. 숫자 범위와 action/observation
채널 순서는 유지하지만 제어 응답이 달라졌으며, 기존 checkpoint와 같은 제어 동역학을
유지하는 변경은 아니다. 이 값은 native controller 실행 검증의 초기 설정이며 학습 성능에
대한 최종 gain 결정은 아니다. F/T 자중·관성 보상은 센서 처리이며 이 제어 변경과 별도로 유지한다.

Arm actuator PD gain은 0이고, hand의 authored mimic constraint는 유지한다.
Hand 6D는 soft limit 안의 원하는 절대 관절 목표다. 실제 PD target은 매 physics substep의
측정 q에서 `min(native velocity limit, target_rate)`×dt만큼 목표 방향으로 제한한다.
기본 `target_rate=1 rad/s`이며 이는 position lead 제한이지 실제 속도 상한의 보장은 아니다.
Native ActionManager가 action을 처리한 뒤 physics substep마다 두 term을 적용한다.
부분 reset은 해당 행의 action/effort/invalid flag만 초기화한다.

## 4. Reward 기준점과 사용자 교정

**Formulation v2를 새 학습 환경에 구현했으며 수치는 초기 실행 기본값이다.**
Transition `s_t → s_{t+1}`에 대해 계산한다. 상위 weight는 비음수, scale과
`Δt`는 양수다. 최종 weight·정규화 scale·종료 허용오차와 학습 충분성은 미정이다.
Native `rewards.{clearance,rearrangement,motion,time,terminal}.weight` 5개는 모두 1이다.
`MdpCfg.w_*`는 해당 설정을 반영하는 diagnostics 값이며 품질 점수별 민감도는 아래에 둔다.
성공 bonus 500·실패 penalty 100이다. Time의 `T_ref`는 제거하고 지수형 비용률의
강도 `a=0.02 reward/s`·증가율 `b=0.7 /s`를 초기값으로 둔다.
Episode timeout 8초는 별도 초기 설정이다.

$$
r_t=w_{\mathrm{clear}}r_t^{\mathrm{clear}}
+w_{\mathrm{rearr}}r_t^{\mathrm{rearr}}
+w_{\mathrm{motion}}r_t^{\mathrm{motion}}
+w_{\mathrm{time}}r_t^{\mathrm{time}}
+r_t^{\mathrm{terminal}}
$$

### 접근성

지정 Blocker의 차단 비율을 줄인다.

$$
r_t^{\mathrm{clear}}
=\exp\left(-a_{\mathrm{clear}}\frac{A_{\mathrm{blocked},t+1}}{A_{\mathrm{blocker},t+1}}\right)
$$

분자는 고정 접근영역 `A_0`와 지정 Blocker footprint의 겹침 면적, 분모는 같은
shelf 평면에 투영한 **Blocker의 전체 footprint 면적**이다. 접근영역 면적으로
나누지 않는다. `A_0`는 초기 Target pose·접근 방향·hand footprint·margin으로
고정해 Target 이동에 따라 성공 영역이 같이 움직이지 않게 한다. 차단 비율이
0이면 1, 1이면 `e^-1`이다. 이 점수만으로 전체 경로 확보를 판정하지 않는다.

### 최소 재배치와 공통 지수 점수

최신 사용자 선택에 따라 Clearance·Rearrangement·Motion의 품질 점수는
**`exp(−a·x)`** 형태로 통일한다. Time은 앞서 선택한 증가형 penalty를 유지하고
Terminal은 별도 종료 사건 보상이다. `x`는 평가 비용이며 `a`는 비용 단위의
역수를 가진 지수 민감도다. 상위 `w`는 항 사이 비중으로 역할이 다르다.

| 품질 점수 | x | a의 초기값 |
|---|---|---:|
| Clearance | Blocker 차단 비율 ρ_b | clearance_decay=1 |
| Rearrangement | 전체 누적 OBB corner 조작량 L (m) | rearrangement_decay=2 m⁻¹ |
| Motion의 관절 항 | 정규화한 관절 가속도 제곱합 | motion_acceleration_decay=1 |
| Motion의 F/T 항 | 정규화한 힘·토크 크기 증가율 제곱합 | motion_wrench_increase_decay=1 |

민감도는 모두 유한한 양수다. 같은 `[0,1]` 범위가 실제 영향의 균형까지 보장하지
않으며 비용 분포와 a·w를 함께 평가해야 한다. Clearance의 비율과 Motion의
기존 정규화 비용은 유지한다. Rearrangement의 `displacement_ref=d_0`는 제거하고
초기 `a_rearr=2 m⁻¹`는 이전 `1/0.5 m`의 민감도를 이어 받는다. 기준거리 표기를
없애도 같은 지수 곡선의 scale 선택이 사라지는 것은 아니다.

Rearrangement의 현재 x는 **직전 pose와 현재 pose에서 대응하는 OBB corner의
이동량을 episode 동안 누적**한 값이다:

$$
\Delta L_k=\sum_{i\in\mathcal O}\frac18\sum_{j=1}^{8}
\|c_{i,j,k+1}-c_{i,j,k}\|_2,\qquad
L_0=0,\quad L_{k+1}=L_k+\Delta L_k,\qquad
r_k^{\mathrm{rearr}}=\exp(-a_{\mathrm{rearr}}L_{k+1})
$$

`O`는 Target·Blocker·처음에 인식되지 않은 물체를 포함한 모든 active 물체이고
padding과 주차한 pool 물체는 제외한다. OBB 8개 corner는 USD 측정 bounds와
고정된 object-local 순서로 만들고 실제 pose로 갱신한다. 고정 environment frame에서
같은 corner끼리 비교해 로봇 EEF 운동을 물체 이동으로 계산하지 않는다.
매 step OBB를 다시 PCA로 추정하거나 가까운 점으로 재대응하지 않는다.

물체별 corner는 평균내고 물체 간·step 간에는 합산한다. 순수 10 cm 병진은
물체당 0.1 m다. 중심 이동 없이 한 변 10 cm인 정육면체를 90° yaw 회전하면
8개 corner가 각각 10 cm 움직여 동일한 0.1 m다. 병진·회전·pivot·왕복·Z 이동을
meter 단위로 평가한다. 실제 mesh 표면을 OBB로 근사하며 회전 비용은 실제
대표점의 이동거리에 따라 크기·형상의 영향을 받는다. 물체 크기나 개수로
정규화하는 것은 아니다. 이 형태는 두 pose에서 동일한 model vertex의 평균
거리를 정의한 [BOP의 ADD 설명(2.2절)](https://arxiv.org/pdf/1808.08319)과 연결된다.
이를 우리 RL reward에 사용하는 성능은 별도 검증 대상이다.

**선택 과정:** 사용자는 모호한 d_0와 XY 위치만의 평가를 지적하고 물체 pose의
병진·회전 평가를 요청했다. 초기–현재 편차 대신 **전체 조작량**을 선택해 왕복도
남긴다. 위치(m)·회전(rad)을 별도로 더하는 안은 단위별 가중치 문제 때문에 OBB
corner의 실제 이동거리로 구체화했다. 그 후 품질 항의 `exp(−a·x)` 공통 형태를
선택했다. 앞서 제안했던 step 조작량의 직접 음수 비용은 현재 채택된 품질 점수가
아니다. 현재 식은 누적량을 매 step 평가하므로 과거 이동이 이후 점수에도 영향을
준다. **최종 조작량이 같아도 이동 시점에 따라 episode return이 달라진다.**

**부분관측 조건:** 사용자는 observation은 초기 vision이고 물체 pose를 갱신하지
않지만, **reward에서는 privileged info로 전체 물체를 사용해도 된다**고 명시했다.
요구는 GT reward 금지가 아니라 observation 정보로 장면 변화의 영향을 어느
정도 인지하도록 유도하는 것이다. Actor의 544D 초기 vision+현재·직전 센서 규약은
유지하고 실제 corner·step 증가량·누적량은 reward와 검증용 extras에만 사용한다.
초기 관측 물체만 평가하거나 손 이동 proxy로 대체하는 것은 선택하지 않았다.
`initial_observed_object_ids`와 원본 `initial_object_valid`는 physical-slot 순서이고
actor 역할 재정렬 mask와 구분해야 한다는 구현 근거는 유지한다.

Reward만으로 숨은 물체 pose를 정확히 인식했다는 보장은 없으며 같은 센서 신호에
여러 장면 변화가 대응하는 모호성과 두 sample 밖의 기억 제한은 남는다.
**명시적인 인식 학습은 미채택 보조 목적 검토안**이다. Shared latent에서 직전
transition의 전체 ΔL 또는 주변 물체 변화 여부를 예측하고 GT는 학습 label로만
사용할 수 있다. `L_PPO+λ_aux L_pred`는 별도 loss이며 예측 정확도를 reward에
더하는 안과 구분한다. 보조 head·label·loss·λ·기억 구조 변경은 아직 채택하지 않았다.
보조 loss만으로 센서가 전달하지 않는 정보가 생기거나 모든 pose가 복원되는 것은 아니다.

`MdpState.previous_corners`는 승인된 scene reset의 실제 corner로 초기화하고
`rearrangement_increment`와 누적 `path_length`를 선택된 환경에만 갱신한다.
기존 timestamp guard가 반복 manager 조회의 중복 누적을 막는다. 종료 전이의
corner·증가량·누적량은 auto-reset 전에 복사하고 extras에 보존한다.
`train.py --validate`는 SDK pose 변환으로 corner를 독립 재구성해 실제 증가량과
지수 점수를 검사한다. Config·checkpoint에 수식·민감도를, metrics에 step 증가량과
최대 누적 조작량을 기록한다.

### 운동과 접촉 안정성

최신 사용자 교정에 따라 **빠른 속도·큰 힘 자체를 비용으로 보지 않고,
급격한 관절 가속도와 힘·토크 크기 증가**를 평가한다. 두 무차원 비용을
지수 점수로 바꾸어 동일 계수로 평균한다.

$$
r_t^{\mathrm{motion}}=\frac12
\left[\exp(-a_{\mathrm{acc}}x_{\mathrm{acc},t})+\exp(-a_{\mathrm{inc}}x_{\mathrm{inc},t})\right]
$$

| 비용 | 현재 계산 의미 |
|---|---|
| `x_acc` | 같은 12관절의 속도 변화 / control dt를 가속도 ref로 정규화·제곱해 합산 |
| `x_inc` | 힘·토크 norm의 증가분만 control dt와 rate ref로 나누고 제곱해 합산 |

두 그룹 사이의 내부 λ는 두지 않는다. 다른 reward와의 비중은 상위 `w_motion`으로
조절한다. Joint 가속도와 하중 증가율은 `Δt`로 차분한다. Acceleration 비용은
`Σ_j(((q̇_j−q̇_previous,j)/dt)/a_ref,j)^2`이며 mimic 관절은 제외한다. Object 수나
순서에 따라 Motion 비용이 바뀌지 않는다. Robot 속도는 실제 rad/s를 사용하고
actor의 정규화 q̇를 비용 입력으로 재사용하지 않는다. Acceleration ref는
arm 5·hand 10 rad/s²다. 증가율은 `max(||F_t||−||F_previous||,0)/dt` 및 torque의
같은 식을 force 200 N/s·torque 10 N·m/s로 정규화해 제곱한다. 사용자가
**크기의 급격한 증가만 비용 부여**를 선택했으므로 감소·동일 크기의 방향 변화는
이 비용에 포함하지 않는다. Sensor 반력의 음수 부호에도 norm 증가를 평가한다.
일정한 높은 q̇·일정한 큰 하중은 두 비용이 0이면 Motion 점수가 1이다.
기존 속도·절대 하중 soft/ref 설정과 두 항은 제거했다. 정규화 ref는 비용 문턱이
아니며 연구 최종 기준이나 hard limit가 아닌 실행 초기값이다. 두 항 평균이므로
기존 네 항 평균에서 한 항이 포화될 때와 Motion 점수 감소 폭이 달라진다.
Reset의 이전 q̇는 해당 환경의 실제
초기 속도로 설정하고 매 완료 전이에서 한 번만 갱신한다. 물체 이동은 최소
재배치 비용, 물체 속도는 안정성·종료 판정에 유지한다. Wrench는 자중·관성·bias 보정
후 같은 좌표축·토크 기준점에서 비교해야 한다. Actor의 sensor-frame 입력과
reward 계산 좌표계를 혼동하지 않는다. Actor에는 움직이는 sensor center의
sensor-frame 보정 wrench를 제공한다. 센서 뒤 33개 링크 각각의
`m(a_COM−g)`와 COM lever arm에 의한 moment, `I_world α + ω×(I_world ω)`를
합산해 raw에서 제거하고 명시적 tare에서 남은 bias만 보정한다. 외력·접촉 반력은
원래 parent→child 부호로 유지한다. Reward에는
이를 world 축으로 회전하고 **episode 시작 sensor center라는 고정 기준점**으로
moment를 옮긴 값을 사용하므로 sensor 회전·이동 자체가 wrench rate로 섞이지 않는다.

### 시간과 종료

$$
c(s)=a\left(e^{bs}-1\right),\qquad
r_t^{\mathrm{time}}=-\int_{t}^{t+\Delta t}c(s)\,ds
=-\frac{a}{b}\left[e^{bt}\left(e^{b\Delta t}-1\right)-b\Delta t\right]
$$

$$
r_t^{\mathrm{terminal}}=
R_{\mathrm{success}}\mathbf1_{\mathrm{success},t}
-R_{\mathrm{failure}}\mathbf1_{\mathrm{failure},t}
$$

앞선 설계는 `−dt/T_ref`였으나, 사용자는 **시간이 지날수록 penalty가 커지고
그 증가폭도 커져야 한다**고 교정했다. 이어 `T_ref`와 무관한 곡선을 요청했고,
초반부터 완만하게 증가하다가 후반에 가팔라지는 **지수형**을 명시적으로 선택했다.
이 선택이 기존 선형 비용과 성공 시에만 완료시간을 평가하는 제안을 대체한다.
구간별 문턱을 둔 제곱형과 누적 제곱 비용의 증분도 최종 식으로 채택하지 않았다.

`time_penalty_scale=a=0.02 reward/s`, `time_penalty_growth_rate=b=0.7 /s`는
개발 초기값이며 사용자가 이 숫자 자체를 지정한 것은 아니다. `a`는 전체 강도,
`b`는 곡선이 가팔라지는 속도다. 비용률은 0에서 시작하고 `c′(t)>0`, `c″(t)>0`다.
매-step reward는 구간 적분이므로 비용의 크기와 증가폭이 모두 점차 커진다.
`expm1`와 작은 `b·dt`의 급수 계산으로 첫 step의 상쇄 오차를 줄인다.
상위 Time weight는 1, terminal의 +500/−100은 유지하며 종료 보상은 한 번 지급한다.

| 경과 시간 | 직전 한 step Time reward, 30 Hz | 할인 전 누적 Time reward |
|---|---:|---:|
| 2초 | −0.002006 | −0.047291 |
| 4초 | −0.010170 | −0.361276 |
| 6초 | −0.043276 | −1.756752 |
| 8초 | −0.177531 | −7.537897 |

![지수형 시간 비용의 개형](figures/time_penalty.png)

첫 step은 약 −0.00000784다. 할인 전 누적 비용은
`−a/b · (exp(bT)−1−bT)`이며 같은 T에서는 control dt를 바꿔도 동일하다.
할인된 합은 이 식과 다르며 gamma·제어 주기에 의존한다. Clock은 승인된 scene
reset 후 0부터 control dt만큼 증가한다. Camera의 render-only warmup이나 같은
step의 반복 reward/termination/observation 조회는 시간을 증가시키지 않는다.
부분 reset은 선택 환경만 초기화하고 종료 전이의 clock·Time reward는 extras에
보존해 PPO 검증에 사용한다. SDK episode counter로 clock을 독립 검증한다.

현재 544D observation에는 경과 시간이 없으며 clock은 reward 내부 상태다.
시간에 따른 행동 변화를 명시적으로 조건화하려면 actor 시간 입력을 별도로
검토해야 한다. 이번 변경에 시간 feature를 추가한 것으로 해석하지 않는다.
기존 품질 항이 양의 점수를 매 step 누적하는 구조와 terminal의 상대 scale는
별도 검증 과제다. 지수형 선택만으로 +500의 영향이나 대기 유인이 해결됐다고
기록하지 않는다.

- 성공은 접근영역의 **전체 non-target 차단 면적**이 허용오차 이하이고 Target 보호·안정 조건을 만족하는 것이다. 지정 Blocker만 치우고 다른 물체가 막은 상태를 성공으로 처리하지 않는다.
- 성공의 초기 기준은 전체 non-target 겹침 합 ≤1e−5 m², Target 이동 ≤2 cm·회전 ≤0.15 rad이며 모든 active 물체가 선속도 ≤0.015 m/s·각속도 ≤0.10 rad/s로 5 policy step 연속 안정한 상태다.
- 실패의 초기 기준은 낙하 6 cm, 지지면에서 4 cm 초과 들어올림, shelf workspace 여유 1.5 cm 이탈, upright 대비 tilt 50° 초과, 보정 force 80 N·torque 4 N·m 초과, raw F/T의 SI-200-8 축별 측정 범위 초과 또는 금지 접촉이다. 성공과 실패를 동시에 검출하면 실패를 우선한다.
- 학습 scene에만 UR arm 7개 body와 원통 2개 half의 ContactSensor **9개**를 둔다. 선반·물체·ground의 각 partner와의 normal force norm **>1 N**을 금지 접촉으로 보는 provisional 기준이다. GPU contact filter는 동적/kinematic body의 자식 mesh가 아닌 RigidBodyAPI prim을 사용한다. 기본 pool 100개와 선반 1개의 **101 rigid-body filter**, rigid body가 없는 static ground의 collision-shape filter를 각각 지정한다. 최근 4 physics substep 중 하나라도 초과하면 실패하며 서로 다른 partner의 반대 방향 force를 합산해 상쇄하지 않는다. Hand 접촉이나 hand–Target 접촉 자체는 이 판정에서 금지하지 않는다. Target 이동/회전은 별도 보호 조건으로 판정한다.
- 8초 time limit는 PPO의 truncation으로 처리하며 성공/실패는 termination이다. 실기에서 초기 vision만으로 성공을 판별하는 방법은 미정이다.

### 구현과 검증에서 남은 문제

양의 매-step 점수는 대기하면서 보상을 누적하는 행동을 유리하게 만들 수 있다.
성공 bonus를 **할인된 누적 대기 return**과 비교해야 한다. 안정성 항이 rapid
동작을 과도하게 막는지, 물체 수에 따른 합산 scale 변화, 평면 면적이 실제 hand/arm
접근 가능성을 충분히 나타내는지도 검증해야 한다.

사용자는 성공 terminal **+500**이 다른 항보다 dominant해질 위험을 지적했다.
이는 검증으로 정한 값이 아닌 구현 초기값이다. 기존 선형 Time을 사용한 당시
30 Hz·8초(240 step)·γ=0.99에서 dense step 최대는 `3−1/300≈2.9967`,
240-step 할인 누적 최대는 **272.808**이었다. 현재 지수형 Time에서는 step
최대가 경과 시간에 따라 달라진다. 8초의 Time 할인 합은 **−1.111502**,
나머지 3항이 매 step 최대일 때 terminal 제외 할인 합은 **271.999912**다.
성공 bonus의 초기 상태 return 기여는 2초 성공 **276.342**,
4초 **151.202**, 8초 **45.267**이다. 따라서 단일 성공 전이에서는 크지만,
모든 성공 시점에서 전체 return을 반드시 지배한다고 해석하지 않는다.
같은 시점에 성공하는 두 경로에서는 terminal 항이 같아 dense 품질 차이는 남는다.

성공을 한 step 미루는 비교에서 즉시 성공은 `r_success+B`, 기다린 뒤 성공은
`r_wait+γ(r_success+B)`다. 기다림이 유리한 조건은
`r_wait>(1−γ)(r_success+B)`다. 매-step dense=2라는 예에서는 B가 **198보다
작으면** 한 step 지연이 유리할 수 있다. Bonus를 낮추기만 하면 양의 지속 reward가
대기/성공 지연을 유도할 수 있으므로 terminal과 dense baseline을 함께 설계해야 한다.

현재 PPO는 advantage만 batch 정규화하고 return/critic target은 정규화하지 않는다.
정규화가 terminal 대 다른 항의 상대적 비중을 제거하지 않으며, 큰 terminal TD
오차는 actor/critic 공유 encoder와 global gradient clipping에 영향을 줄 수 있다.
실제 dominant 학습 여부는 성공/실패별 discounted term return·완료 시간·actor/critic
gradient 크기로 검증해야 한다. 기존 smoke 검증에는 task 성공이 없으므로 이 균형을
입증한 결과로 사용하지 않는다. Quality 항을 zero 기준 비용으로 바꾸고 terminal
비중을 비교하는 것은 **미채택 검토안**이다. Terminal scale 질의만으로 +500·−100을
바꾸지 않았으며 후속 Time 선택에서 시간 수식만 변경했다. 전체 reward를 공통 배율로 나누는 것은 critic의 수치
scale만 바꾸며 상대적 목적 가중치 문제의 해결과 구분한다.

로컬 Isaac Lab [RewardManager](../IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py)는
각 term에 `weight * dt`를 곱한다. Policy-step 수식을 그대로 옮길 때 시간 벌점의
`Δt`와 one-shot 종료 보상을 포함해 시간 scale을 중복 적용하지 않도록,
구현은 완성한 discrete transition reward를 `dt`로 나눈 값을 manager에 반환한다.
위 reward를 잘 받는다는 사실만으로 내부 상태 이해의 정확성을 인증할 수는 없다.

## 5. 환경 변화 이해와 미채택 제안

사용자가 요구한 것은 초기 장면–행동–sensing–환경 변화의 관계를 정책이 배우도록
유도하는 학습 신호와, 제대로 배웠는지 판별할 근거다. 연산 지연은 구현 후 측정할
문제로 두었으며, 이 논의를 반복해서 latency 구조 논의로 되돌리지 않는다.

같은 초기 장면에서 Blocker를 서로 다른 만큼 밀고 손이 같은 pose로 돌아온
두 episode를 생각한다. 현재 q·dq·F/T·tactile이 같고 밀기 사건이 유한 history
window 밖이라면 기억 없는 정책은 두 상태를 구분하지 못할 수 있다. Reward를
바꾸어도 현재 입력에서 사라진 사건 정보가 생기지는 않는다.

Current-only, 유한 window, 지속 hidden state를 갖는 recurrent policy는 서로
다르다. 가중치는 episode 전반의 일반적인 관계를, hidden state는 이번 episode의
상호작용 정보를 담는다. GNN round 수와 history 길이, hidden state 유지 기간과
학습 sequence 길이도 구분한다. **LSTM/GRU 도입은 아직 승인되지 않았다.**

현재 task reward만 사용하는 finite-window baseline을 먼저 구현했다. 다음은
아직 미채택 후보이며 baseline에 포함하지 않는다.

- 정책의 공유 표현에서 물체 이동·회전·차단비율 등을 예측하는 학습용 head와 `L_train = L_PPO + λ L_pred` 추가. Label·정규화·loss·λ는 미정이며 실행 시 head를 제거할 수 있다는 제안이다.
- GT 상태 label 대신 action-conditioned latent forward dynamics 사용.
- 초기 scene encoder와 recurrent temporal encoder 결합, sensor-conditioned query와 복수 readout.

예측 결과나 simulator 정답을 actor에 직접 주기로 한 것은 아니다. 보조 loss도
사용자가 수용한 상태가 아니다. 예측 정확도를 scalar reward에 더하면 예측하기
쉬운 상태에 머무르는 행동을 선호할 수 있다는 쟁점이 남아 있다.

평가 후보는 고정 분포의 변화 예측 오차, no-change predictor 대비 개선,
history 제거·교란 시 성능 변화, 비슷한 현재 관측과 서로 다른 과거를 가진 조건의
다음 행동 비교다. 모두 확정 실험 계획과 구분한다.

## 6. 설계가 바뀐 흐름과 문헌 교정

| 시점 또는 단계 | 변경과 현재 의미 |
|---|---|
| 초기 BrainStorming | 초기 vision·상위 VLM·pull-forward/lateral sweeping skill 구상. 접근 공간 확보 문제는 유지되지만 두 skill/VLM 구조 전체를 현재 정책으로 채택하지 않았다. |
| Track B | 상위 판단기가 회전·이동 명령을 주고 순차 수행, vision 지속 갱신. 명시적 종착 목표와 고정 단계 순서는 현재 명세로 가져오지 않는다. |
| 2026-09-30 중간발표 | Target 접근성을 목표로 no-blocker-goal-pose 판단·행동 통합, 18D action, variable stiffness, 네 reward 목적을 구체화했다. |
| 2026-10-01∼04 | Planar 관측 후보에서 3D OBB10D·quaternion·단일 카메라·외부 역할 지정·집합 처리로 구체화. Map은 채택 방향에서 제외했다. |
| 2026-10-06 | GNN·mask·EEF 기준 정보 보존을 검토하고 message-passing 반복과 temporal memory를 구분했다. |
| 2026-10-08 | Reward v2, 공통 거리 scale, T_ref 유지. 지속 tracking에서 초기 vision + 상호작용 이력으로 전환하는 방향이 추가됐다. |
| 2026-10-09 초기 구현 | 사용자가 Isaac Lab 구현 시작과 새 환경의 스폰부터 진행하도록 요청. `contact_rearrangement`의 고정 6개 장면·reset 구현과 원본 USD 검증을 완료했다. |
| 2026-10-09 장면 구체화 | 사용자가 장면당 4–6개, 더 밀집한 배치, 같은 종류 중복을 선택했다. YCB visual 폴더의 9개 사용 요청은 실제 `Axis_Aligned_Physics` 폴더의 physics USD 4개 사용으로 대체됐다. 기존 6종과 합친 10종 pool, instance별 역할, geometry 기반 sampling으로 확장했다. |
| 2026-10-09 역할·pool 수명주기 교정 | 사용자가 episode Target·Blocker를 먼저 정한 뒤 배치 retry에서 유지하도록 선택했다. GPU 반복 reset의 simulation flag 전환과 live gravity flag 전환에서 문제가 재현돼 둘 다 폐기했다. Pool topology·native gravity를 유지하고 inactive COM에 countergravity를 주는 `TaskObjectPool`로 변경했다. 8개 환경·16초 반복 reset과 동일 종류 역할 선지정 검증이 통과했다. |
| 2026-10-09 YCB 상하 방향 교정 | 사용자가 006의 뒤집힘 수정 및 다른 YCB 확인을 요청했다. 실제 렌더에서 003·004·005도 뒤집혀 있어 4종 모두 X +90°에서 −90°로 교정했다. 수정 후 양쪽 렌더와 기본 8개 환경 물리·reset 검사가 통과했으며 별도 고정 pair의 5.5 cm drift 실패도 기록한다. 상세 배치 기준은 `SPAWN_RULES.md`에 분리했다. |
| 2026-10-09 선반 전체·Target 유일성·초기 가시성 구체화 | 사용자가 10종 전체 사용 허용, Target 종류 중복 금지, 앞쪽 Target, 선반 전반의 배치와 이동 여유, pivot/회전+밀기 경로를 요청했다. 4–10개·100-body pool·all-types 모드, coverage/국소 밀집도 분리, 다양한 평면 path로 변경한다. Camera의 40% 기준은 전체 비가림 silhouette 대비 실제 픽셀 수로 선택했으며 실제 camera 검사는 후속 구현이다. |
| 2026-10-09 YCB 박스 초기 정착 수정 | 사용자가 스폰 직후 낙하·흔들림 수정을 요청했다. 전물체의 2 mm 여유 제거만으로 003의 흔들림이 남아 003·004에 명시적 AABB Cube collider를 적용했다. 원본 visible mesh·Nucleus USD·mass/contact/material·active gravity는 유지한다. 4개 환경·10종 전체·10초 정착 및 reset 검사가 통과했다. |
| 2026-10-09 종류 수·instance 수와 선반 전체 밀도 교정 | 사용자가 4–10은 종류 선정 기준이었다고 교정했다. 이전 4–10 instance 제한·all-types 정확히 10개를 대체해 종류 4–10, 면적 기반 instance 수·cap 40으로 분리했다. 전체 점유율과 6개 구역 하한을 hard 조건으로 적용하며, 25–45% 생성 비교로 global 35%·구역 10%·이웃 75%를 현재 구현 기본값으로 선택했다. |
| 2026-10-09 역할 OBB 표시 | 사용자가 Target은 초록색, Blocker는 빨간색 OBB로 표시하도록 요청했다. 현재 pose의 debug draw 표시를 스크립트에 추가했고 USD·물리 상태 무변경, headless reset 갱신과 실제 GUI 표시를 제한된 실행에서 확인했다. |
| 2026-10-09 중복 종류 편중 교정 | 사용자가 한 종류가 공간을 과하게 채우는 문제를 지적했다. 큰 footprint 우선 복사본 선택을 폐기하고 선택된 non-target 종류의 최소 active 개수를 우선하는 균형 구성으로 변경했다. 종류·역할·density·clearance 조건을 유지하고 현재 source의 기하 40개와 일반 모드 4개 GPU 환경·5.03초 검사가 통과했다. |
| 2026-10-09 책임별 모듈 통합 | 사용자가 과하게 세분화된 패키지 구조를 정리하도록 요청했다. Metadata·asset 생성·환경 설정·spawn/reset·순수 sampling·visualization의 6개 모듈로 통합하고 기본값·동작을 유지했다. 이전 모듈은 wrapper 없이 제거했다. 통합 후 23개 unit test·simulator 없는 import·CLI와 4개 GPU 환경·5.03초 검사가 통과했다. |
| 2026-10-09 Blocker 앞 hand 시작 randomization | 사용자가 충돌 없는 hand 시작 영역 randomization과 elbow-down 금지를 요청했다. USD FK·SciPy IK·scene collision과 별도 robot seed를 구현했다. 초기 151-step 검사 뒤 callback·실패 시 state 유지·COM cache·arm 이웃 OBB 처리를 보강했고 최종 24개 시작·5초 stress와 29개 unit test가 통과했다. |
| 2026-10-10 Tool 전방 교정 | 사용자가 아래를 향하던 tool을 선반 정면으로 향하게 요청했다. Palm normal을 전방으로 맞췄던 초기 orientation을 교정해 TCP +Y와 wrist-3 +Z가 world −X를 향하게 한다. 시작 영역·elbow-up·15 mm scene 여유는 유지하고 실제 PhysX TCP 회전과 tool 축을 검증한다. |
| 2026-10-10 Tool 기준 hand −90° roll | 사용자가 tool Z축 기준 hand 회전을 요청했다. 기존 vendor adapter `(0,-90,0)` deg에 별도 `mount_tool_roll_deg=-90`을 추가해 physical tool body인 `wrist_3_link`의 +Z축으로 hand 조립 transform을 회전한다. Tool 전방 −X와 기존 시작 조건은 유지하며 TCP 목표는 X→+Z, Y→−X, Z→−Y다. |
| 2026-10-10 말단–hand 원통 연결부 | 사용자가 실제 센서·연결부 공간을 고려해 길이 5 cm 원통을 요청했다. Wrist-3 compound collider로 직경 5 cm의 visible `SensorMount`를 추가하고 hand fixed anchor를 physical tool +Z로 5 cm 옮긴다. Hand roll·TCP 방향·39 body·18 DOF를 유지하며 enabled robot collider는 40개가 된다. 원본 USD·기존 mass/inertia는 유지하고 실제 센서 질량·F/T 구현과 구분한다. |
| 2026-10-10 중앙 F/T·초기 vision | 원통 중앙 fixed-joint F/T와 앞쪽 camera를 추가한다. 원통의 두 25 mm rigid half를 통해 hand load가 중앙 joint를 지나게 하고 raw sensor-frame wrench를 읽는다. 실제 같은 pose의 비가림 silhouette로 Target ≥40%·Blocker 전체 가시성을 승인하고 초기 RGB-D를 보존한다. Nominal mount mass 0.20 kg는 구현값이며 실제 하드웨어 질량으로 확정하지 않는다. |
| 2026-10-10 독립 tactile 17채널 | 기존 URDF tactile link 17개에 독립 ContactSensor를 연결하고 net normal force norm >0.01 N을 binary로 관측한다. Palm·finger pad-2의 원래 CAD slot을 메우던 부모 5개 convex hull을 convex decomposition으로 바꿔 접촉이 pad까지 닿게 한다. Source URDF·visual·mass/inertia·41 body·18 DOF·41 collider prim은 유지한다. 실제 sphere contact로 모든 채널 ON/OFF와 reset/다중 환경 cache 보존을 확인했다. |
| 2026-10-10 시간 입력 축소 | 사용자가 16-step history를 previous step까지만 제공하도록 요청했다. 현재 t·직전 t−1의 두 sample로 줄여 기본 입력을 1960D에서 700D로 변경하고 temporal convolution을 1개 layer로 줄인다. GNN·역할 attention·18D action·좌표계·PPO rollout 길이는 유지한다. 기존 16-step checkpoint와 새 규격의 이어 학습은 호환되지 않는다. |
| 2026-10-10 mask 전달 교정 | 사용자가 mask는 observation이 아니라 모델 입력 전에 별도로 생성하며 observation 정보만으로 만들 수 있어야 한다고 교정했다. Mask feature 채널을 제거하고 역할 ID·zero padding·tactile 결측 −1로부터 Boolean selector를 생성한다. Policy 입력은 584D, 각 sample은 72D, 모델은 247,717 parameter다. 이전 700D checkpoint는 호환되지 않는다. |
| 2026-10-12 역할 슬롯 고정 | 사용자가 역할 ID 대신 Target·Blocker의 observation 위치를 고정하자고 요청했다. Target 슬롯 0·Blocker 슬롯 1·주변 물체 슬롯 2 이상으로 정렬하고 역할 ID 40D를 제거한다. Observation 544D이며 역할 mask는 고정 슬롯과 기하 유효성으로 생성한다. 두 sample·기하 encoder·247,717 parameter를 유지한다. |
| 2026-10-12 F/T 관성 보상 | 자중만 제거하고 운동 관성이 남던 F/T에 관성 보상도 필요하다고 사용자가 요청했다. 센서 손쪽 33개 링크의 COM 병진 가속도·회전 가속도·각속도와 질량·관성으로 Newton–Euler 자체 반력을 계산해 제거한다. Observation·하중 reward·F/T 실패 판정에 공통 적용한다. |
| 2026-10-12 Motion의 robot 기준 교정·tactile 검토 | 사용자가 Motion reward는 물체가 아닌 robot joint 속도 등의 기준이어야 한다고 교정하고 tactile의 역할을 검토하자고 요청했다. Motion의 속도·가속도 비용을 arm 6 + hand 독립 6관절로 바꾸며 F/T load/rate 네 항 구조를 유지한다. Tactile은 기존 observation으로 유지하고 신호 보존·reward 유인 문제를 검토하며 새 tactile reward는 채택하지 않았다. |
| 2026-10-12 Motion의 급격한 변화 기준 교정 | 사용자가 빠른 속도·큰 힘 자체보다 급격한 가속·힘 증가를 억제하는 목적이라고 교정했다. 속도·절대 하중 reward 항과 soft 기준을 제거하고 joint acceleration·F/T norm 증가율 두 항만 평균한다. 후속 선택은 힘·토크 크기의 증가만 비용 부여이며 빠른 감소와 같은 크기의 방향 변화는 제외한다. 하드 안전 한계는 별도 유지한다. |
| 2026-10-12 시간 비용의 지수형 교정 | 사용자가 시간이 지날수록 비용과 증가폭이 커지는 곡선을 요청하고 지수형을 선택했다. T_ref를 제거해 c(t)=a(exp(bt)−1)의 control 구간 적분을 차감한다. 초기 a=0.02 reward/s·b=0.7 /s, 환경별 clock·부분 reset·중복 조회 방지를 구현하며 544D observation과 terminal scale는 유지한다. |
| 2026-10-12 품질 점수의 공통 지수형·pose 조작량 | 사용자가 Clearance·Rearrangement·Motion을 exp(−a·x)로 통일하고 Time은 기존 증가형으로 유지하도록 선택했다. Rearrangement는 모든 물체의 평균 8-corner 3D 이동거리 누적으로 확장하며 d_0를 민감도 a_rearr로 대체한다. 초기 계수는 1·2 m⁻¹·1·1이며 actor 관측과 보조 head 미도입 상태를 유지한다. |
| 2026-10-12 SI-200-8 F/T 측정 모델 | 사용자가 사진의 ATI Axia80 SI-200-8 spec으로 센서를 설정하도록 요청했다. Raw sensor-frame 범위 Fx/Fy200 N·Fz360 N·토크8 N·m, 유효 분해능 힘0.1 N·토크0.005 N·m를 saturation·rounding으로 근사한 뒤 기존 자중·관성을 보상한다. 축별 normalization과 범위 초과 판정, reset/terminal cache·checkpoint profile을 연결하며 544D 관측과 기존 작업 안전·Motion 기준을 유지한다. |

Notion에 정리된 기존 문헌 논의에서 다음 교정을 유지한다. 논문의 성과를 우리
과업의 성공·속도·상태 이해 결과로 기록하지 않는다.

| 문헌 | 유지할 근거와 적용 한계 |
|---|---|
| [Rotating without Seeing](https://arxiv.org/html/2303.10880) | Binary tactile·짧은 history·PPO/task reward로 접촉 이력을 활용한 선례. 별도 이해 보상이 항상 필수라는 주장에는 반례지만, 우리 다물체 과업에서 충분하다는 증거는 아니다. |
| [Enhancing Tactile-based Reinforcement Learning for Robotic Control](https://arxiv.org/html/2510.21609v1) | Forward dynamics 등 보조 목적의 근거. Auxiliary replay buffer와 실행 recurrent memory를 구분하고, FD+확장 buffer의 개선을 loss 단독 효과로 해석하지 않는다. |
| [RoboPack](https://arxiv.org/html/2407.01418) | Dense packing의 초기 vision+tactile 선례. Recurrent GNN 추정·dynamics·지도학습·MPC 구조이며 우리 단일 정책을 검증한 것이 아니다. Box pushing은 push 사이 vision을 갱신한다. |
| [HORA](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf) | Privileged 정책 학습 후 고정하고 history adaptation module을 latent L2 오차로 별도 학습한다. Task reward만으로 history를 해석한 선례로 인용하지 않는다. |
| [ReNN](https://richardrl.github.io/relational-rl/) | 관계 처리와 RL 공동학습의 선례. 명시적 물체 goal/velocity, SAC+HER·curriculum의 stacking 실험이다. GNN 단독 효과나 rapid 실행 성능으로 확대하지 않고 64D·3 rounds 등도 우리 명세로 채택하지 않았다. |

## 7. 현재 Isaac Lab 구현

새 환경의 위치는 [src/contact_rearrangement](../src/contact_rearrangement/README.md)다.
장면 확인용 `ManagerBasedEnv`는 빈 action manager와 PD hold를 사용하며
`sensors` group에서 raw F/T·17채널 tactile/valid·초기 RGB-D·역할 mask·camera
pose를 반환한다. 학습용 `ContactRearrangementRLEnv`는 별도 설정으로 18D
OSC/hand action·Reward v2·종료 조건과 544D `policy` group을 사용한다.
초기 visible-depth 기하·현재 EEF 재표현·현재/직전 두 sample, GNN/attention/temporal
encoder와 shared actor/critic을 native RSL-RL runner에 연결한다. `train.py`는
`--task Isaac-Contact-Rearrangement-UR5e-v0`의 registry에서 환경·agent 설정을
읽으며 `env.py`가 auto-reset 이전 final observation을 PPO에 보존한다.

### 장면과 초기 설정

- UR5e + RH56E2 **오른손**을 fixed joint로 하나의 articulation에 조립한다. 손의 fixed link와 mimic 관계를 유지하고 contact-report API를 활성화한다. 기존 vendor adapter `mount_rotation_deg=(0,-90,0)` deg에 `mount_tool_roll_deg=-90` deg를 추가한다. 추가 roll은 physical tool body인 `wrist_3_link` local +Z축을 기준으로 hand 조립 transform에 적용한다. 같은 tool +Z로 hand anchor를 5 cm 옮기고 길이 5 cm·직경 5 cm 원통을 두 25 mm rigid half로 조립한다. 중앙 fixed joint를 F/T 측정점으로 삼아 hand load가 통과하게 한다. 현재 41 body·18 DOF·41 enabled robot collider이며 nominal 원통 질량 0.20 kg는 실제 센서 질량이 아닌 구현 기본값이다. 원본 asset과 기존 wrist mass/inertia는 유지한다.
- Kinematic speedrack shelf와 **기존 6종 + YCB 4종 = 10종** asset을 사용한다. YCB는 `NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics`의 `003_cracker_box`, `004_sugar_box`, `005_tomato_soup_can`, `006_mustard_bottle`이다. 해당 폴더에서 확인한 4개 USD는 rigid body와 collision이 이미 authoring되어 있다.
- 장면당 **4–10종**을 선택하고 종류별 최소 하나씩 사용한다. 요구 면적에 도달할 때까지 추가 가능한 non-target 종류 중 active 개수가 가장 적은 종류에 복사본을 추가하며 동률은 무작위로 정한다. Blocker도 해당 종류의 첫 instance로 센다. 현재 같은 pool capacity에서는 non-target 개수 차이가 최대 1이다. Target은 정확히 1개이며 별도 종류별 고정 최대값을 추가하지 않는다. `all_asset_types=True`는 catalog 10종을 모두 선택하고 같은 균형 원칙으로 중복을 추가한다. Instance 최소 4·cap 40, 종류당 10-copy·환경당 100-body pool은 별도 값이다. 공간 배치는 packing을 위해 큰 footprint부터 진행하며 복사본 종류 선택 순서와 구별한다. Inactive는 `z=4.0/4.4/4.8/5.2 m`에 주차하고 속도를 0으로 설정한다. Parking은 작업층 camera ROI 밖에 두며 USD visibility를 수정해 숨기지 않는다. **Pool의 모든 body는 simulation enabled와 native gravity를 유지**해 topology를 바꾸지 않는다. 후속 actor·reward·종료에는 active mask를 적용한다. Observation K를 100으로 확정하는 뜻은 아니다.
- `TaskObjectPool`은 매 physics write/substep에서 inactive body에만 world force `-m*g`를 추가해 중력을 상쇄한다. PhysX `apply_forces_and_torques_at_position`에 실제 **world COM 위치**를 명시해 asset root pivot과 COM의 차이로 불필요한 토크가 생기지 않도록 한다. Active body의 추가 힘은 0이며 native gravity와 접촉 동역학을 유지한다. Live CPU simulation/gravity flag나 USD visibility를 매 reset마다 바꾸지 않는다.
- `TaskObjectPool.write_object_link_pose_to_sim`은 상위 writer 호출 뒤 `_object_com_pose_w.timestamp=-1.0`으로 COM cache를 무효화한다. Physics step 없이 같은 timestamp에서 연속 reset해도 다음 COM 읽기를 갱신하며 countergravity의 적용점을 최신 COM으로 유지한다. 추가 physics step이나 simulation/native gravity flag 변경은 없다.
- **Target 종류는 active 장면에 정확히 한 번** 등장한다. 해당 종류는 Blocker·주변 물체에서 제외하고, 선택한 다른 종류의 중복은 허용한다. 초기 baseline은 Target 1개와 Blocker 1개이며 episode 시작에 쌍을 먼저 선택한다. 종류 수·집합, 전체 active instance ID와 개수를 pose retry 전에 고정하며 yaw·위치만 다시 뽑는다. Cap·복사본 수 안에서 면적을 충족하지 못하거나 300 layout 시도에서 배치하지 못하면 구성을 교체하거나 조건을 낮추지 않고 실패한다.
- `cfg.target_asset`·`cfg.blocker_asset`과 CLI의 `--target_asset`·`--blocker_asset`으로 역할의 종류를 고정할 수 있다. 기본 `None`은 episode 역할을 처음에 한 번 추출한다. 두 옵션에 같은 종류를 지정하면 Target 유일성에 위배되어 오류다. Blocker 종류만 고정한 경우 Target 후보에서도 그 종류를 제외한다. 이는 외부 역할 지정이며 Blocker의 최종 목표 pose를 제공하는 것이 아니다.
- CLI `--min_asset_types`·`--max_asset_types`는 종류 수, `--min_objects`·`--max_objects`는 instance 최소값·cap이다. `--min_shelf_occupancy`·`--min_region_occupancy`는 global/구역 하한을 지정한다. 종류와 instance 개수를 혼용하지 않는다.
- 로봇 base는 환경 로컬 `(0, 0, 0.79505)`, quaternion `(w,x,y,z)=(0,0,0,1)`, 선반은 `(-0.7,0,0)`에 놓인다. Quaternion 표기는 Isaac Lab 순서다.
- 기존 2×3 고정 배열과 `cup_1` Target/`mug_2` Blocker는 초기 구현 이력이다. 현재는 환경별 seed와 reset 횟수로 episode 역할을 먼저 정하고 그 역할에 맞는 upright yaw·배치를 뽑는다. 선택되지 않은 환경의 reset sequence는 유지한다. 종류·역할별 배치 실패율은 따로 확인하며, 역할을 placement 성공률에 따라 바꿔 분포를 감추지 않는다.
- 물체의 spawned rigid-body local bounds를 읽고 canonical upright rotation을 적용한다. YCB 4종은 local −Y를 world +Z로 옮기는 X축 −90° 회전 후 world Z yaw를 적용한다. 수정 전 양쪽 렌더에서 003·004·005의 라벨과 006의 뚜껑이 모두 아래를 향해 기존 +90°를 교정했다. Root pivot과 bounding center를 구분해 XY를 보정하고, 실제 lower Z bound로 선반 위 높이를 정한다.
- 모든 물체의 root Z는 `shelf_top - lower_z`로 정해 bounds 바닥을 선반 상면에 바로 맞춘다. 기존 2 mm 여유는 제거했다. 003·004만 `FlatBoxUsdCfg`로 visible scanned mesh의 collision을 끄고 동일 body-local AABB 크기·center offset의 invisible Cube collider를 추가한다. 기존 single body를 유지하고 clone spawn 시점에만 적용하며 Nucleus 원본을 수정하지 않는다. Authored mass·원본 contact 속성·physics material을 유지하고 active body의 gravity·simulation을 그대로 사용한다. 자동 fit `boundingCube` approximation은 최종 구현이 아니다.
- 실제 shelf collision에서 support와 상부 panel을 읽는다. 현재 기본 asset의 작업층은 환경 로컬 `x=[-0.9,-0.5]`, `y=[-0.455,0.455]`, 상면 `z=1.05 m`, 위쪽 panel 하단 `z=1.48 m`이며 물체 footprint에 edge margin과 상부 여유를 적용한다. Asset override에서는 해당 geometry를 다시 측정한다.
- 기본 물리 주기는 `1/120 s`, decimation은 4로 환경 step은 `1/30 s`다. 기본 환경 수 1, 간격 2.5 m다.
- 장면 확인용 `spawn_scene.py`는 기본적으로 시뮬레이션 시간 5초마다 전체 환경을 reset해 새 배치를 생성한다. 실제 `env.step_dt`로 간격을 계산하며 현재 설정에서는 150 step이다. `--reset_interval N`으로 step 간격을 지정하고 0이면 자동 reset을 끈다. 이 설정은 실행 스크립트의 확인 주기이며 학습 episode timeout을 확정하는 것은 아니다.
- `RoleObbVisualizer`는 Target 초록색·Blocker 빨간색 OBB를 표시한다. 각 역할의 12개 모서리를 현재 body link pose·local bounds에서 계산해 YCB upright·root offset·합성 scale을 반영하고 매 step과 전체/부분 reset 뒤 갱신한다. Debug draw만 사용하며 USD prim·collider·material·physics state를 쓰지 않는다. GUI·livestream에서 기본 표시하고 `--disable_role_obb`로 끈다. 일반 headless에서는 생략하며 `--headless --validate`에서 viewport 없이 overlay 검사를 실행한다. 제한된 headless·GUI 검사 결과는 8절에 기록한다.
- Reset은 물체 sample 뒤 Blocker 앞의 robot 시작 후보를 찾고 실제 앞쪽 camera 렌더로 초기 가시성을 승인한다. Camera 32개 후보가 실패하면 같은 역할·종류·active ID·개수로 pose를 다시 제안하며 최대 8개 layout을 검사한다. 승인된 pose·A₀·robot 시작 metadata와 initial snapshot을 해당 환경에 저장하고 후보 실패 시 기존 physical pose·metadata로 복구한다. Render는 physics/settling step을 추가하지 않으며 pool의 simulation/native gravity와 inactive parking/countergravity를 유지한다.
- 17개 tactile link에 각각 독립 ContactSensor를 붙인다. 장면 확인 dictionary에는 binary contact·valid를 제공하며, 학습 policy에는 0/1 또는 결측 −1의 17개 측정값만 넣는다. 중앙 fixed-joint raw F/T를 유지하고 학습 actor에는 sensor-frame 자중·관성·선택적 bias 보정값을 제공한다. Filtering·실제 하드웨어 threshold/질량·관성·가속도 추정 교정은 후속 단계다.
- 학습 `UR5eHandLearningSceneCfg`에만 arm/spacer의 9개 금지 접촉 sensor를 추가한다. 설정 factory는 `src/sensors.py`, 최근 4-substep/각 partner >1 N 판정은 `mdp/actions.py`, threshold는 `ContactRearrangementRLEnvCfg.forbidden_contact_threshold`에 둔다. Hand 조작 접촉을 이 판정에 포함하지 않으며 Target의 실제 이동 보호는 MDP가 맡는다. 이 접촉 분류·threshold는 초기 task 기본값이다.

### Blocker 앞의 robot 시작 randomization

`cfg.robot_start`에 `RobotStartRules`를 둔다. 기본 제안 영역은 Blocker의
**회전 footprint +X 최전방보다 12–40 cm 앞**, 동시에 **선반 앞 경계에서
6 cm 이상 밖**, Blocker bounding center의 **Y ±8 cm**, shelf top 위
**20–28 cm**다. 거리 기준은 palm TCP `right_plam_force_sensor`이며 손가락
표면이 아니다. 현재 TCP local +Y인 손가락/tool 전방은 **world −X, 선반 방향**,
local +Z인 palm normal은 **world −Y**, local +X는 world +Z를 향하게 고정한다.
조립된 USD에서 TCP +Y는 UR `wrist_3_link`의 local +Z인 tool 전방과 일치한다.
기존 vendor adapter `(0,-90,0)` deg는 유지하고 tool +Z 기준 **hand −90° roll**을
mount에 추가한다. Arm wrist 관절각만 바꾸는 것이 아니라 hand의 mount 회전과
translation을 함께 변환한다. `tool0` attachment의 축은 physical tool body와
다르므로 `tool0` 자체 local Z를 회전축으로 쓰지 않는다. 구체적인 영역 식과
회전 행렬은 [SPAWN_RULES.md](SPAWN_RULES.md)에 있다.

실제 센서·연결부 공간을 반영하는 길이 5 cm 원통을 wrist-3와 hand 사이에
둔다. Hand fixed anchor는 physical tool +Z로 5 cm 이동하고 현재 −90° roll은
유지한다. 현재 원통은 중앙 F/T joint를 둔 두 rigid half이므로 **41-link FK·
18 DOF**, 외부 scene·arm self 검사에는 **41개** robot collider를 쓴다.
Nominal mass/inertia는 구현값이며 실제 하드웨어 교정과 구분한다.

`assets.py`의 `RobotPoseModel`은 실제 조립 USD의 41-link joint frame·axis·scale로
FK를 구성하고 SciPy bounded IK로 arm 해를 찾는다. Finger q는 기존 열린 손의
기본값을 유지한다. Elbow-up은 관절각 부호로 결정하지 않고 elbow joint center가
shoulder–wrist-1 chord의 가장 가까운 점보다 **world Z로 5 cm 이상 위**인지
검사한다. Elbow-down/낮은 elbow 해는 채택하지 않는다.
현재 실제 pose에서 모델 FK를 대조하므로 default 자세로 선행 reset하지 않는다.
`accept_candidate` callback으로 한 TCP의 IK 해가 scene collision에서 탈락하면
같은 TCP의 다른 seed/branch도 탐색한다. 모든 branch의 완전 탐색을 의미하지 않는다.

모든 enabled arm/hand collision OBB를 모든 active object(역할 포함)·선반 개별
enabled collider·ground와 **15 mm 여유**로 검사한다. UR arm 7개 link의 21쌍 중
두 arm joint 이내의 11쌍은 제외하고 거리 3 이상인 10쌍을 2 mm 여유로 검사한다.
손 자체의 모든 self pair를 검사하지 않으므로 손 전체의 self collision 안전성을
검증했다고 기록하지 않는다.
이 판정은 초기 pose의 보수적인 OBB 조건이며 실제 mesh 거리·접촉·재배치 경로의
도달성이나 swept collision을 보장하지 않는다.

물체 sample을 고정한 뒤 `episode_seed XOR 0x5D2EA91C`의 독립 RNG로 최대
**80개** robot 시작 후보를 찾는다. 실패하면 이유를 포함해 명시적으로 중단하며
elbow-down fallback이나 물체 배치 재선정으로 조건을 완화하지 않는다.
`robot_initial_joint_pos`·`robot_start_tcp_pos_w`·`robot_start_region`·
`robot_start_elbow_height`·`robot_start_attempts`를 기록한다. 부분 reset에서는
선택하지 않은 환경의 state·목표·reset sequence를 유지한다.
선택된 모든 환경의 물체·robot 후보가 CPU에서 성공한 뒤 default reset과
실제 state/PD target·metadata 적용을 진행한다. 후보 실패 시에는 기존 선택
환경의 physical pose·metadata도 유지한다.
`cfg.robot_start.enabled=False`이면 기존 기본 자세를 사용한다. 수치는 후속
제어·학습 분포에 맞춰 조정할 수 있다. Reset 검증은 실제 PhysX TCP의 전체
회전 오차를 **0.01 rad 이하**로 확인하고 wrist-3 local +Z가 world −X인지도
대조한다. 2026-10-09의 24개 시작과 반복 reset·5초 물리 검사는 이전 아래를
향하던 tool 자세의 결과이며 조건·범위는 8절에 기록한다.

### 밀집 배치와 해소 가능성의 초기 조건

좌표·수치·샘플링 순서·계산식은 [상세 배치 기준](SPAWN_RULES.md)에 정리했다.

샘플러는 처음부터 Target 접근을 막되 간단한 재배치 해법이 존재하는 baseline을
만든다. Blocker는 Target의 로봇 쪽인 +X에 두고, 초기 Target과 +X 접근 방향으로
고정한 `A_0`에 대해 Blocker footprint의 30% 이상이 겹치도록 한다. 주변 물체는
초기 `A_0` 밖에 둔다. Footprint는 실제 body bounds에서 얻은 보수적 upright
box의 평면 투영이며 mesh의 실제 점유 면적과 동일하지 않다.

Target X는 Blocker가 앞에 들어갈 수 있는 전체 유효 범위에서 뽑는다.
Blocker의 Y도 ±30 mm로 제한하지 않고 footprint가 A₀와 겹칠 수 있는 범위에서
제안한 뒤 실제 겹침을 검사한다. 이 offset 제안은 부분 가림을 허용하지만 영상
가시성의 40% 조건을 증명하지 않는다.

현재 구현 기본값은 최소 footprint gap **3 mm**, edge margin **15 mm**,
margin을 뺀 shelf 범위 대비 footprint 집합의 **X span 55%·Y span 65% 이상**이다.
별도로 active 물체의 **75% 이상이 4 cm 이내 footprint 이웃**을 갖도록 한다.
전체 footprint 면적 합/전체 usable floor는 **35% 이상**이고, X 2 × Y 3의
각 구역은 footprint 교차 면적 합/구역 면적이 **10% 이상**이어야 한다. 면적
기준이 미달이면 넓은 X/Y span만으로 채택하지 않는다. 35%는 실제 측정 bounds의
25–45% 생성 비교에서 선택한 초기 구현값이며 최종 연구 명세가 아니다. Reserved
path는 점유 면적으로 계산하지 않는다. 정확한 mesh 점유율·동역학 안정성·영상
가시성을 보장하는 수치도 아니다.

**Clearance witness**는 병진·footprint 모서리의 평면 pivot·중심 회전 후 병진·
작은 앞쪽 이동 후 회전/병진 등의 후보에서 하나를 찾는다. Pivot은 고정점 주변의
원호로 검사하고 회전 구간을 최대 5°씩 나눠 arc sagitta로 확장한 보수적 swept
polygon을 사용한다. 전체 경로가 support 안에 있고 Target·주변 물체와 떨어져야
한다. 경로를 예약한 뒤 부족한 6개 구역을 우선 채우는 위치·이웃 proposal을
평가해 주변 물체를 놓는다. Episode 구성은 유지하고 최종 hard threshold를 모두 검사한다.

Witness는 생성·검증용 privileged metadata다. Actor에 Blocker의 최종 goal pose로
입력하는 것이 아니며 정책이 이 병진 해법만 수행하도록 결정한 것도 아니다.
이 조건은 보수적 평면 기하에서 경로 하나가 존재한다는 뜻이다. Bounding corner
pivot은 실제 hand contact point를 인증하는 것이 아니다. UR5e의 접근
가능성, 손의 접촉점과 충돌, OSC로 밀 수 있는지, 마찰·회전·전도를 포함한 동역학,
센서 기반 종료 판정과 실제 재배치 성공은 아직 입증하지 않았다.

### 중앙 fixed-joint F/T와 초기 센서 observation

원통 중앙의 fixed joint에서 PhysX incoming-joint reaction을 직접 읽는다.
측정 frame은 중앙점에 놓인 physical tool XYZ이며 parent→child 방향의
`[Fx,Fy,Fz,Tx,Ty,Tz]`, 단위는 `[N,N,N,N·m,N·m,N·m]`다. 힘·토크에는 중력·
관성·접촉·외력이 포함된다. Raw 관측에 자동 tare나 중력 보정을 적용하지 않는다.
이상적 PhysX 반력은 별도로 보존하고 raw 관측에는 SI-200-8 범위 제한·유효
분해능 rounding을 먼저 적용한다. 이후 학습의 `corrected_wrench`가 자중·관성을
보상한다. 보상 후 잔차가 다시 분해능의 정수배일 필요는 없다.
명시적인 `tare()`는 그 시점의 bias만 저장하며 중력 보정과 다르다. Reset은
선택한 환경의 ideal/raw cache·bias·범위 초과 flag를 비우고 `wrench_valid=False`로 둔다. 다음 완료된
physics step 뒤 timestamp가 증가했을 때 반력을 읽어 valid로 갱신한다.

두 환경의 `cfg.force_torque`는 기본 `SI_200_8_CALIBRATION`이다. 사진과
[ATI Axia80-M20 SI-200-8 공식 사양](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Axia80-M20)을 따라
센서 frame의 `[Fx,Fy,Fz,Tx,Ty,Tz]` 측정 범위는
`±[200,200,360,8,8,8]`(N/N·m), 유효 분해능은
`[0.1,0.1,0.1,0.005,0.005,0.005]`다. 전체 하중을 축별로 saturation하고 rounding한
측정값에서 자중·관성·선택적 residual bias를 제거한다. 유효 분해능에는 noise가
포함된다는 사양이므로 deterministic 양자화는 시뮬레이션 근사이며 정확도·ADC
bit depth·noise 분포와 동일시하지 않는다. 임의 noise·지연·sampling rate 변경을
추가하지 않았다. 기존 원통 조립·질량·41 body/18 DOF/41 collider는 유지한다.

`read()`는 측정 모델을 거친 raw 값이고 `read_ideal()`은 PhysX 반력 보존값이다.
`range_exceeded`는 saturation 전 raw의 축별 범위 초과이며 `valid`는 sample의
freshness다. 두 기준을 섞지 않는다. 범위 초과를 기존 hard-load 실패 channel에
OR하며 보정 하중 norm의 80 N/4 N·m 작업 기준은 별도로 유지한다. Motion ref
200 N/s·10 N·m/s도 측정 범위로 바꾸지 않는다. Auto-reset 전 raw·ideal·범위 초과
flag를 extras에 보존한다. Profile과 axis normalization은 config·checkpoint에
기록하고 이어 학습은 profile이 없거나 다르면 거절해 이전 50 N/5 N·m 관측을
조용히 새 scale로 해석하지 않는다. Native checkpoint에는 관측 규약과 F/T profile을 함께 보존하며 이전 custom trainer 형식은 직접 복원하지 않는다.

`ObservationsCfg.sensors`는 concatenate하지 않는 dictionary다. `wrench` 6D,
`wrench_valid` 1D, `wrench_range_exceeded` bool 6D,
`tactile`·`tactile_valid` 각 bool 17D, RGB `480×640×3`,
depth `480×640×1` m, Target/Blocker 순서의
실제 instance mask `480×640×2`, camera world position+ROS quaternion(wxyz) 7D를
반환한다. RGB/depth/mask/camera pose는 승인한 초기 snapshot의 cache를 읽으며
episode 중 image를 갱신하지 않는다. 비가림 reference·가시성 비율은 reset의
acceptance metadata이며 observation으로 제공하지 않는다. 학습용 `policy` group은
이 초기 snapshot의 visible depth에서 추정한 OBB10과 robot/sensor/action 이력을
별도로 구성한다. Raw sensor dictionary와 actor의 자중·관성·bias 보정 F/T를 구분한다.

### 기존 hand link의 독립 tactile 17채널

Scene에 `contact_` + canonical tactile body name으로 17개 독립 ContactSensor를
둔다. 각각 `{ENV_REGEX_NS}/Robot/ROASRightHand/{body_name}` 한 body만 검사한다.
순서는 **palm 1개 → thumb 1–4 → index 1–3 → middle 1–3 → ring 1–3 → little 1–3**다.
기존 URDF의 palm 이름 `right_plam_force_sensor`는 철자를 그대로 유지한다.
초기 URDF 검사에서 정확한 17개 link와 기존 collision mesh를 확인한다.

`cfg.tactile.force_threshold=0.01 N`은 finite positive 값이어야 한다.
`env.binary_tactile_sensor.net_forces_w`의 world net normal contact force
`[N_env,17,3]`에 대해 norm **> threshold**를 `contacts[N_env,17]` bool로 계산한다.
`valid`도 같은 shape의 bool이다. SDK `ContactSensorCfg.force_threshold`는
사용하지 않는 air/contact-time용 설정이므로 binary threshold로 대신 쓰지 않는다.
각 sensor는 `update_period=0`, `history_length=0`이며 120 Hz physics의 마지막
완료 step, 즉 30 Hz 관측의 마지막 substep 값을 읽는다. 자유 공간의 중력·
관절하중 자체는 normal contact가 아니다. Render/reset의 같은 timestamp에서는
stale report를 다시 읽지 않는다. 선택 reset은 cache를 zero·invalid로 비우고
다음 완료 physics step 뒤 valid로 갱신하며 미선택 환경 cache는 유지한다.

기존 단일 convex hull이 palm·finger pad-2의 CAD slot을 메워 pad collider까지
접촉이 닿지 않는 문제를 교정했다. `assets._preserve_tactile_parent_slots()`는
`right_plam_1`, `right_index_2`, `right_middle_2`, `right_ring_2`, `right_little_2`의
기존 부모 collider 5개만 convex decomposition으로 바꾼다. 기본값은 최대
128 hull·hull당 64 vertex·1M voxel·error 0.5%·shrink wrap·min thickness 0.1 mm다.
17개 pad collider·source URDF·visual·mass/inertia는 유지한다. Robot은
41 body·18 DOF·**41 collider prim**이며 decomposition의 cooked hull 수와
collider prim 수를 구분한다. Sensor 책임은 기존 `sensors.py`에 통합한다.

### 초기 vision의 실제 silhouette 판정

`InitialVisionCfg`는 앞쪽 **+X camera에서 −X를 보는 방향**,
`min_target_visible_fraction=0.40`, `min_blocker_visible_fraction=1.0`을 설정한다.
분모는 같은 camera intrinsics·상대 pose에서 해당 역할 물체만 외부 가림 없이
렌더링한 **전체 silhouette 픽셀 수**다. 실제 장면의 해당 instance mask 픽셀
수를 분자로 사용한다. 다른 물체뿐 아니라 shelf·robot 가림도 포함하며 물체
self-occlusion은 reference silhouette에 자연스럽게 반영한다. 둘 다 전체 bounds가
frame 안에 있어야 하고 reference 최소 32 pixel을 요구한다. Blocker는 전체
reference 면적에서 최대 2 pixel의 손실을 허용하며 contour 경계 band로 빼지 않는다.
가시 비율의 상한은 추가하지 않는다.

`InitialCamera`는 640×480 RGB·image-plane depth·instance segmentation을 읽는다.
카메라 X는 선반 전면보다 0.55–0.95 m 앞, Y는 Blocker 중심 ±0.35 m에서 제안해
±0.50 m로 제한하며 높이는 선반 상면 **+0.18–0.48 m**다. 두 역할 사이를 보되
aim Y ±0.06 m, aim 높이 상면 +0.10±0.03 m, optical-axis roll ±3°를 무작위로
정한다. Pinhole focal length 18 mm·horizontal aperture 24 mm로 수평 시야각은
약 67°다.
두 역할의 전체 bounds를 실제 camera near/far clip `0.02–2.5 m`와 화면 안에서
검사한다. Nonfinite/range 오류·음수 jitter는 거절하며 `camera_side="+X"`,
전체 frame 포함과 같은 pose 비가림 reference 규칙은 필수다.

동일 형상의 10종 visual reference를 physics 시작 전에 작업층 밖의 격리 lane
`z+(20+5×env_id) m, y+5 m×type index`에 만들고 physics API를 제거한다. 환경
사이도 Z로 5 m 분리한다. Reference 물체와
`SilhouetteCamera`를 같은 offset으로 평행 이동해 실제 물체·camera의 상대 pose를
보존한다. Live body의 visibility·physics topology는 바꾸지 않는다.
순서는 **역할/종류/active ID/개수 고정 → pose·elbow-up 제안 → native state 적용 →
camera·실제 mask·동일 pose reference → pixel 승인 → 초기 snapshot·A₀ commit**이다.
최초 5회·이후 2회 `sim.render()`로 texture를 준비하며 physics step은 진행하지
않는다. 따라서 settled pose가 아니라 승인된 reset pose의 가시성을 검사한다.
32개 camera 후보가 실패하면 같은 composition의 pose를 재시도하고 최대 8개
layout에서도 실패하면 기존 state·metadata로 복구해 명시적으로 실패한다.
실패 복구는 물체 COM velocity와 robot state/목표뿐 아니라 모든 초기 image·
camera·Target/Blocker 가시성 buffer도 보존한다.

Camera는 로컬 `InitialCamera` subclass를 쓴다. Isaac Lab 2.3.2의 첫 Fabric pose
조회는 `Camera.reset → _update_poses → XformPrimView`의 stage sync에서 Kit
`app.update()`를 호출하므로 physics가 진행될 수 있다. `_update_poses`·
`set_world_poses`·`camera.update` 동안 `/app/player/playSimulations=False`로
보호하고 `try/finally`로 이전 값을 복구한다. 첫 camera data 읽기 전에 cache를
준비하며 SDK·live body physics flag는 수정하지 않는다.

`SpawnRules.reserve_blocker_front=True`는 rotated Blocker footprint의 가장
큰 X부터 shelf front까지, Y bounds 양쪽에 `gap=3 mm`를 더한 직선 통로를
주변 물체로부터 비운다. Target A₀와 clearance witness는 유지한다. Global
35%·구역 10% 등 density 조건도 그대로 검사한다. 이는 camera proposal의
성공 가능성을 높이는 기하 조건이며 실제 silhouette 판정을 대체하지 않는다.

| 파일 | 역할 |
|---|---|
| [shelf_rearrangement_env_cfg.py](../src/contact_rearrangement/contact_rearrangement/shelf_rearrangement_env_cfg.py) | 공통 scene·camera·MDP manager term과 환경 기본값 |
| [config/ur5e/osc_env_cfg.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/osc_env_cfg.py) | UR5e/RH56E2 장면 확인·학습 설정, OSC용 actuator |
| [config/ur5e/__init__.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/__init__.py) | Gym task와 env/agent entry point 등록 |
| [env.py](../src/contact_rearrangement/contact_rearrangement/env.py) | Auto-reset 전 실제 terminal observation 보존 |
| [mdp/actions.py](../src/contact_rearrangement/contact_rearrangement/mdp/actions.py) | Base 기준 18D OSC/hand action과 arm/spacer 금지 접촉 이력 판정 |
| [mdp/state.py](../src/contact_rearrangement/contact_rearrangement/mdp/state.py) | MdpCfg·기하·부분 reset·한 transition에 한 번 갱신하는 이력/누적량 |
| [mdp/observations.py](../src/contact_rearrangement/contact_rearrangement/mdp/observations.py) | 초기 기하의 현재 EEF 재표현·현재/직전 두 sample·sensor-frame F/T 보정 |
| [mdp/rewards.py](../src/contact_rearrangement/contact_rearrangement/mdp/rewards.py) | Reward v2와 dt 보정 |
| [mdp/terminations.py](../src/contact_rearrangement/contact_rearrangement/mdp/terminations.py) | 성공·실패 판정 |
| [mdp/events.py](../src/contact_rearrangement/contact_rearrangement/mdp/events.py) | USD 측정·역할/reset·robot 후보·camera retry·commit/rollback·MDP reset |
| [src/catalog.py](../src/contact_rearrangement/contact_rearrangement/src/catalog.py) | Simulator 없이 읽는 10종 metadata·100개 pool·upright·parking·asset 경로 |
| [src/assets.py](../src/contact_rearrangement/contact_rearrangement/src/assets.py) | 로봇/F/T 조립·raw reader·tactile CAD slot·FK/IK·물체/pool |
| [src/sampling.py](../src/contact_rearrangement/contact_rearrangement/src/sampling.py) | 순수 density·SE(2) 경로·robot 시작 영역/elbow/3D OBB helpers |
| [src/sensors.py](../src/contact_rearrangement/contact_rearrangement/src/sensors.py) | 초기 RGB-D/PCA 기하·raw F/T·17채널 tactile·금지 접촉 sensor factory |
| [src/visualization.py](../src/contact_rearrangement/contact_rearrangement/src/visualization.py) | Target/Blocker OBB debug draw |
| [agents/model.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/model.py) | 공유 GNN·역할 attention·유한 temporal encoder·actor/critic |
| [agents/ppo.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/ppo.py) | Native PPO 확장: 공유 모델·final-observation timeout·microbatch·profile 검사 |
| [agents/ppo_cfg.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/ppo_cfg.py) | Simulator 독립 ModelCfg 차원 |
| [agents/rsl_rl_ppo_cfg.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/agents/rsl_rl_ppo_cfg.py) | 예제의 RslRlOnPolicyRunnerCfg/RslRlPpoAlgorithmCfg 기본값 |
| [spawn_scene.py](../src/contact_rearrangement/scripts/spawn_scene.py) | 장면 실행·5초 reset·역할 OBB·검증 |
| [train.py](../src/contact_rearrangement/scripts/train.py) | 예제의 Hydra·gym.make·RslRlVecEnvWrapper·OnPolicyRunner·learn 연결 |

구조는 [sweeping_policy](../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy)의
공통 `shelf_sweep_env_cfg.py`, `config/ur5e`의 concrete 환경과 등록,
`mdp`의 manager 함수, `config/ur5e/agents`의 runner 설정에 대응한다.
`Isaac-Contact-Rearrangement-UR5e-v0`의 `env_cfg_entry_point`와
`rsl_rl_cfg_entry_point`로 설정을 불러온다. `rsl_rl_ppo_cfg.py`의 native
`ContactRearrangementPPORunnerCfg`에 model·algorithm·iteration·저장 간격을 둔다.
`examples`의 sweeping 설정은 legacy actor/critic 형식이며 설치된 RSL-RL 5.0.1의
actor/critic 모델 인터페이스에 연구 network를 연결한다. 검증 함수는
`tests/check_learning.py`로 옮기고 `--validate`에서만 사용한다.
장면 지원의 asset·sensor·sampling 구현은 각각 하나의 모듈로 유지한다.
이전 flat 모듈을 다시 import하는 wrapper는 만들지 않는다.

스폰 실행은 **catalog·CLI → camera 활성화 AppLauncher → scene/reference/camera
생성 → USD 측정/reset·pixel acceptance → physics step·F/T 갱신·OBB 표시** 순서다.
Metadata와 순수 sampling은 simulator 없이 읽으며 simulator 의존 모듈은
AppLauncher 뒤에 로드한다. `mdp/events.py`는 실제 geometry를 sampler에 전달하고
그 결과를 state로 적용한다. Pool은 매 substep countergravity를 적용한다.
학습에서는 초기 pixel acceptance 후 visible-depth geometry를 저장하고
history/reward 누적량을 reset한 뒤 OSC→physics→센서→transition reward/종료→policy
관측 순서로 진행한다. `mdp/state.py`가 공통 transition을 한 번만 갱신해
manager 호출에 따른 누적량·history 중복 갱신을 막는다. Package tree와 실행·
수정 위치는 [README](../src/contact_rearrangement/README.md)에 있다.

2026-10-09의 6개 모듈 통합 기록과 해당 검사 결과는 8절에 보존한다. 이후
학습 코드가 추가되고 sweeping example 구조로 재배치되었으며 현재 위치는 위
표를 기준으로 읽는다.

Asset 기본 경로는 `omniverse://192.168.0.13`이며
`CONTACT_REARRANGEMENT_*` 환경변수로 override할 수 있다. 손은 저장소의
`RH56E2_R_2025_9_11_relative.urdf`를 사용한다. 세부 명령과 변수는 패키지 README에 둔다.

### 기존 환경에서 참고한 범위

Examples와 `roas_sweeping`에서 shelf·6개 물체 종류와 배치를,
`hand_rl`에서 RH56E2 오른손 조립을, Isaac Lab의
`scripts/tutorials/03_envs/create_cartpole_base_env.py`에서 base environment
생명주기를 참고했다. 새 제어의 기준 frame은 examples의 sweeping 프로젝트에서
`subtract_frame_transforms(root_pose_w, object_or_ee_pose_w)`와 `base_link` source
frame을 사용하는 방식을 참조한다. 구체적인 참조는
[Sweeping observations.py](../examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/mdp/observations.py)의
`object_position_in_robot_root_frame`와 EEF pose 변환이다. `sweep_jh`의 기존 goal/reward나 가상 F/T를
새 MDP의 실제 센서로 대체하지 않는다.
기존 환경의 목표 pose·reaching IK·sweep reward·단일 active-object reset을
새 연구 명세로 그대로 사용하지 않는다. 오래된 `docs/hand_rl`의 왼손 설명보다
현재 오른손 코드를 확인해야 한다.

## 8. 개발과 검증 기록

과거 기록의 모듈 이름은 당시 구조다. 현재 위치는 위 모듈 표를 따르며,
통합 전 검사 결과를 통합 후 runtime 검사의 통과로 사용하지 않는다.

### 2026-10-09 초기 고정 6개 장면 스폰과 reset — 이전 버전

**구현:** 독립 패키지 `contact_rearrangement`와 실행 스크립트를 추가했다.
Arm·hand는 초기 PD 목표를 유지하고 물체 6개는 중력과 접촉에 따라 시뮬레이션된다.

**검증 조건:** 이 저장소의 Isaac Lab source 2.3.2, 설치된 Isaac Sim 5.1,
RTX 3090에서 원본 Nucleus UR5e·선반·물체 USD 및 로컬 RH56E2 오른손 URDF를 사용했다.
2개 환경을 headless CUDA로 120 environment step, 4초 실행했다.

**통과한 검사:** 6개 물체 이름·순서, arm/hand 독립 관절과 17개 tactile용 링크,
fixed robot base, finite robot joint position·물체 state, 초기 높이보다 10 cm 이상
낙하하지 않음, XY 이동이 5 cm를 넘지 않음, 전체 reset 및 한 환경만 교란한 뒤
부분 reset의 물체 pose/velocity·robot 관절 위치/목표 복원. 부분 reset에서
다른 환경의 물체 state가 유지되는 것도 확인했다.

이 결과는 초기 장면과 reset의 제한된 smoke 검증이다. 재배치 성공, 센서 신호
품질, OSC 제어 성능, 파지·비파지 판별, 장기 안정성, 정책 학습과 rapid 성능을 입증하지 않는다.

아래는 **이전 고정 6개 버전 검증에 사용한 명령**이다. 같은 경로의 현재 코드가
확장되고 있으므로 지금 실행한 결과를 이전 결과와 동일하게 해석하지 않는다.

```bash
conda activate env_isaaclab
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 2 --headless --device cuda:0 --max_steps 120 --validate
```

이전 버전의 최종 로그는 `outputs/contact_rearrangement/spawn_validation.log`에 있다.
`outputs/`는 Git에서 제외되므로 다른 checkout에 로그가 없을 수 있다.

**유지할 개발 교훈:** 로컬 기본 `reset_scene_to_default`는 rigid-object collection을
복원하지 않아 별도 처리를 추가했다. Collection default state에는 link pose와
COM velocity가 들어가므로 대응 writer인 `write_object_state_to_sim`을 사용한다.
설치된 `isaaclab`이 다른 checkout을 가리켜 실행 스크립트에서 이 저장소의 source를
우선한다. 중간에 Nucleus 인증 만료와 `fast_shutdown=False` 종료 시 충돌이 있었으나,
최종 실행은 원본 asset 연결과 기본 AppLauncher 종료 설정으로 통과했다. 임시
테스트 물체 검증과 최종 원본 USD 검증을 혼동하지 않는다.

### 2026-10-09 물체 종류·중복·밀집도·역할 선지정 확장 — 검증 완료

**사용자 결정:** 장면당 4–6개 active 물체, 더 밀집한 배치, 같은 종류의 중복을
허용한다. Target과 Blocker의 종류가 같아도 서로 다른 instance이면 된다.
초기 YCB visual 폴더의 9개 사용 요청은 실제 `Axis_Aligned_Physics`에 존재하는
authored physics USD 4개를 기존 6종과 함께 사용하는 것으로 대체됐다.
Target·Blocker는 episode 시작에 먼저 지정하고 배치 재시도 동안 유지한다.
옵션으로 종류를 고정해도 같은 종류의 두 역할은 서로 다른 instance를 사용한다.

**구현 변경:** 10종 × 6-copy pool, active mask와 instance별 역할, 환경·reset별
seed, 실제 body/shelf geometry 기반 높이·support 검사, 고정 `A_0`의 초기
차단 조건, 가까운 이웃 배치와 밀집도 threshold, 전체 swept footprint의
lateral clearance witness를 추가했다. 수치와 단일 Blocker 가정은
bootstrap baseline이며 최종 curriculum·최대 K·일반 다중 Blocker 설계와 구분한다.

**중간 GPU 검증과 변경 이유:** 기존 simulation-disable 방식에서 8개 환경의
한 reset cycle은 통과했지만, active 개수를 6으로 고정한 반복 reset의 두 번째
cycle에서 body pose가 다른 환경의 정확한 parking 좌표로 바뀌었다. 보통의
낙하·밀림과 다른 현상이며, `set_disable_simulations` 전환으로 인한 GPU body
index/cache 대응 변화가 원인이라는 추론에 따라 해당 방식을 폐기한다. 엔진
내부 원인을 확정한 것은 아니다.

후속 live gravity flag 방식도 이 direct GPU 실행에서는 동작을 보장하지 못했다.
Inactive getter가 disable-gravity `1`을 반환해도 첫 4개 physics substep 뒤
`vel_z=-0.327 m/s`, `delta_z=-0.006812 m`로 정확히 `9.81 m/s²`의 낙하가
관측됐다. Nucleus asset·로봇 없이 480개 cube body만 사용하는 최소 재현에서도
active→parking 전환 뒤 gravity-disable이 반영되지 않는 현상을 확인했다. 이는
이 환경에서의 API/runtime 관측이며 Isaac Sim 전체의 일반적인 엔진 결함을
입증한 것은 아니다.

따라서 live simulation/gravity flag 전환을 피하고 **모든 pool body의 native
gravity와 simulation을 유지**한다. 당시 `object_pool.py`(이후 `assets.py`에 통합)의 `TaskObjectPool`이
inactive COM에만 매 substep world `-m*g` 외력을 적용한다. 높은 격리 parking으로
작업층과 분리하고 USD visibility는 수정하지 않는다. COM 적용은 pivot 기준
외력이 만드는 torque를 피하기 위한 선택이다.

**당시 최종 검증 (YCB 방향 교정 전):** 이 저장소의 Isaac Lab 2.3.2, Isaac Sim 5.1, RTX 3090 CUDA에서
원본 Nucleus USD와 기본 fast shutdown으로 다음 두 실행이 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 8 --min_objects 6 --max_objects 6 --headless --device cuda:0 \
  --max_steps 480 --reset_interval 120 --validate

./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 2 --seed 7 --target_asset ycb_003_cracker_box \
  --blocker_asset ycb_003_cracker_box --headless --device cuda:0 \
  --max_steps 240 --reset_interval 120 --validate
```

첫 실행은 8개 환경에서 6개 active 고정, 4초 간격 reset, 총 16초다. 두 번째는
Target·Blocker를 cracker box의 서로 다른 복사본으로 먼저 지정하고 4–6개를
생성했다. 로그에서 4·5·6개 모두 확인했으며 총 8초를 실행했다. 전체·부분 reset의
PhysX 직접 pose 읽기, quaternion·velocity·PD 목표, unselected 환경의 state·mask·
역할·고정 접근영역·reset 횟수 유지가 통과했다. Physics 후 active 낙하 10 cm 초과,
XY drift 5 cm 초과, 초기 upright 대비 tilt 15° 초과를 검사했고 inactive position·
velocity는 1e-5 허용오차 안에서 정지했다. 초기 배치의 지지·간격·밀집도·차단·
swept clearance와 native gravity/simulation flag 유지도 통과했다.

샘플러 unit test **14개**가 통과했다. 역할 선지정·rejection 뒤 역할 유지,
중복 instance·수 고정·seed 재현, clipping/SAT, 불가능한 배치의 명시적 실패,
끝점이 비어 있어도 중간 경로가 막히는 경우 등을 검사한다. 별도 기하 검사에서
측정된 10종 bounds의 60-slot pool, 6개 고정, 역할을 먼저 정한 500 seed가 모두
생성됐다. 로그는 ignored `outputs/contact_rearrangement/clutter_com_validation.log`와
`clutter_fixed_roles_validation.log`에 있다.

이는 스폰과 제한된 무동작 안정성 검증이다. Settle 이후 모든 기하 threshold의
유지, 손의 실제 접촉·도달성, 센서 신호, 재배치 성공이나 PPO 학습은 입증하지 않았다.

### 2026-10-09 YCB 상하 방향 교정과 배치 기준 문서화

**사용자 요청:** 006 mustard bottle의 뒤집힘을 수정하고 다른 YCB의 상하도
확인한다. 현재 Target·Blocker 중심 배치 기준을 자세히 설명한다.

**확인과 교정:** `Axis_Aligned_Physics` 원본 003·004·005·006을 기존 X축
+90° 상태로 각각 양쪽에서 렌더링했다. 003·004의 라벨, 005의 라벨과 캔 윗면,
006의 병뚜껑이 모두 뒤집혀 있었다. 모든 YCB의 canonical quaternion을
`(w,x,y,z)=(sqrt(0.5),-sqrt(0.5),0,0)`으로 통일해 local −Y를 world +Z로 옮긴다.
복사본도 같은 catalog 값을 사용하고 이후 world Z yaw만 추가한다.
당시 USD의 mass·collision approximation은 변경하지 않았다.

교정 후에도 같은 4종의 양쪽을 8장 렌더링했다. 003·004 라벨, 005 라벨과 여는
고리, 006 뚜껑이 정상 방향임을 직접 확인했다. 렌더는 각 asset을 바닥 위 2 mm에
kinematic으로 두고 640×640 RGB camera 8개로 촬영했다. 이 검사는 실제 라벨과
형상을 통한 상하 확인이며 무동작 물리 안정성 검사와 구분한다.
이미지와 사용한 quaternion·높이 manifest는 ignored
`outputs/contact_rearrangement/ycb_orientation/{before,corrected}/`에 있다.

**교정 후 통과한 물리 검증:** 같은 Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090,
기본 seed 0에서 아래 실행이 exit code 0으로 통과했다. 각 환경에 6개 active,
총 8초, 4초마다 전체 reset을 수행했고 마지막 부분 reset도 검사했다. 로그의
active 장면들에 YCB 4종이 모두 포함됐다. 초기 기하·PhysX 직접 pose·quaternion·
velocity·pool flag·COM 주차 정지·관절 목표·제한된 낙하/이동/기울기 검사를 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 8 --min_objects 6 --max_objects 6 --headless --device cuda:0 \
  --max_steps 240 --reset_interval 120 --validate
```

로그: `outputs/contact_rearrangement/ycb_upright_reference_validation.log`.
기존 sampling unit test 14개도 다시 통과했다.

**별도 고정 pair 검사의 실패:** `--seed 7 --target_asset ycb_006_mustard_bottle
--blocker_asset ycb_003_cracker_box`를 추가한 8개 환경·6개 active 검사에서는
첫 step의 검사를 통과했지만 120 step (4초) 시점의 drift 검사에 실패했다.
Env 3의 Blocker `ycb_003_cracker_box_copy_06`가
`Δx=-0.00868 m`, `Δy=+0.05434 m`, `Δz=-0.00515 m` 움직여 XY 5 cm 기준을
넘었다. Runtime error에 물체 이름·초기/현재 위치·속도를 포함하도록 진단을
보강했다. 해당 결과를 모든 초기 장면이 정적으로 안정하다는 근거로 사용하지 않는다.
로그: `ycb_upright_validation.log`, `ycb_upright_diagnostic.log`.

같은 scene seed `3000016`으로 1개 환경을 재현하고 Blocker의 접촉을 매
physics substep (1/120초) 기록했다. 1초 시점의 기울기는 6.79°, 2초 이후에는
29.91°이며 `Δx=-0.00863 m`, `Δy=+0.05409 m`에서 멈췄다. 처음에는 선반과만
접촉했고, step 175 (1.458초)부터 옆의 `ycb_004_sugar_box_copy_04`와 접촉했다.
480개 substep 동안 Blocker–robot 접촉은 관측되지 않았다. 따라서 이 실패를
로봇이 밀어서 생긴 drift로 설명하지 않는다. 원본 convexHull collider와 바닥의
실제 지지 형상이 기울기에 미치는 영향은 별도로 확인해야 한다. Mass는
0.411 kg이고 local COM 위치는 root와 같았다. 접촉 reporter는 임시 진단 환경의
spawn 시점에만 활성화했으며 본 환경의 asset mass/collision은 바꾸지 않았다.
진단 로그·JSON은 `outputs/contact_rearrangement/ycb_upright_contact_diagnostic.*`다.

**배치 기준:** [SPAWN_RULES.md](SPAWN_RULES.md)에 실제 좌표·role/count 선지정·
회전/pivot·pair gap·고정 A₀·차단 비율의 분모·swept witness·주변 물체 이웃 제안·
cluster 밀집도·retry/seed·inactive pool을 정리했다. 당시 baseline은 주변 물체가
초기 A₀와 한쪽 witness 경로를 비우므로 연쇄 재배치가 필수인 clutter를 생성하지
않는다. 10 cm hand width와 모든 threshold는 구현 기본값이며 손의 실제 형상과
제어·동역학·학습 성공의 보장은 아니다.
실제 측정 geometry와 기본 seed 0·6개 active의 샘플을 이용한 평면 배치도는
`docs/assets/spawn_rules_seed_0.svg`다. 통과한 기본 실행의 env 0과 같은
Target·Blocker·active 종류, Y span 32.3 cm, footprint fill 36.2%를 재현했다.

### 2026-10-09 선반 전체 분포·10종 허용·Target 유일성·회전 경로 확장

이 절은 당시 구현의 이력이다. **4–10종을 active instance 4–10개로 잘못 해석한
설정**은 뒤의 종류/instance 분리 교정으로 대체했다. 당시 all-types의 정확히
10개 검증을 현재 density 산정 개수의 검증으로 읽지 않는다.

**사용자 교정:** 4–6개 제한을 확장해 10종 전체 사용을 허용한다. Target 종류는
한 장면에 하나이며 중복하지 않는다. 앞에 Blocker가 들어갈 수 있으면 Target을
앞쪽에도 놓는다. 좁은 cluster 대신 선반 전체에 분포시키되 해소 공간을 남긴다.
Lateral push 외에 pivoting, 회전 후 밀기와 복합 동작을 고려한다.
초기 vision은 선반 앞 +X camera에서 Target의 40% 이상이 보여야 한다.
사용자는 분모를 **같은 pose의 가림 없는 전체 silhouette 픽셀 면적**으로 선택했다.

**구현:** 기본 4–10개·100-body pool, 10종 각각 하나씩 사용하는 all-types 옵션,
Target 종류 전체의 중복 배제, 전체 pair-feasible X와 넓은 Blocker Y 제안,
coverage/국소 근접 판정, 실제 고정점 pivot 원호와 회전/병진 다단계 witness를
추가했다. 기본 X coverage 55%·Y coverage 65%, 4 cm 이웃 비율 50%는 이번
요구를 구현하기 위한 수치이며 사용자가 정한 최종 난이도 명세는 아니다.
기존 38 cm cluster 최대 폭·tight rectangle fill 35% 하한을 제거했다.
당시 YCB upright 회전·authored mass/collider·active native gravity를 유지했다.

`InitialVisionCfg`와 reset metadata에 40% 요구와 **미검증 상태**를 기록했다.
실제 camera·segmentation·reference와 settle 이후 acceptance/rejection은
구현 전이다. 이 후속 단계에서 역할/개수를 유지하고 승인한 초기 pose와 A₀를
일치시켜야 한다. 현재 geometric reset을 camera gate의 통과로 해석하지 않는다.

**기하와 unit 검사:** unit test **20개**가 통과했다. Target 종류 유일성·10종
전체 사용·앞쪽 Target·역할/개수 retry 유지·true polygon 거리·중간 pivot
충돌·회전 구간 내부 corner의 보수적 envelope 포함 등을 검사한다.
기존 실제 측정 USD bounds로 일반 seed 0–99·all-types seed 0–29를 확인했고
모두 생성됐다. 일반 모드의 active 개수는 4–10 모두 나왔고, 경로는 병진 15개·
회전/병진 36개·pivot 49개였다. All-types는 병진 6개·회전/병진 8개·pivot 16개였다.
이 경로 분포는 이 seed 집합의 결과이며 균등 분포나 실제 로봇 성공률을 뜻하지 않는다.

**GPU pool/reset 검사 — 통과:** Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090에서
아래 두 실행이 exit code 0으로 통과했다. 각각 4개 환경·24 env step·0.8초다.
매 step reset하므로 각 layout은 1/30초만 진행한다. 이는 100-body pool의 상태
쓰기·gravity/주차·직접 PhysX readback·전체/부분 reset과 기하 metadata 검사다.
장시간 settle 안정성을 검증한 것으로 기록하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --all_asset_types --headless --device cuda:0 \
  --max_steps 24 --reset_interval 1 --validate

./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 \
  --max_steps 24 --reset_interval 1 --validate
```

로그는 `outputs/contact_rearrangement/full_shelf_pool_reset_validation.log`와
`full_shelf_mixed_reset_validation.log`다. 각 로그에서 초기+24회 reset의 100개
장면을 기술했고, 마지막 부분 reset도 통과했다. 일반 모드에서는 active 4–10,
Target 종류 1회 등장, 다른 종류 중복을 확인했다. All-types 모드에서는 매번
10종 각각 하나씩 사용했고, 경로 종류는 병진 15·회전/병진 35·pivot 50개였다.

**별도 settle 검사 — 실패 유지:** 같은 기본 seed 0·4개 환경·all-types에서
`--max_steps 240 --reset_interval 120 --validate`를 실행했다. 초기 규칙과 첫 step
검사는 통과했지만 120 step (4초) 시점에 env 0의 003이 XY 약 11.2 cm,
env 3의 003이 약 13.8 cm 움직여 실패했다. 로그는
`outputs/contact_rearrangement/full_shelf_all_types_validation.log`다.
당시 버전에는 003의 자연 정착 문제가 남았다. 그 실행에서는 collider/mass를
바꾸거나 검사의 5 cm·15° 기준을 완화해 통과시키지 않았다. 이후 사용자의
흔들림 수정 요청에 따라 collider를 바꾸는 현 구현과 구분한다.

**배치도:** `docs/assets/full_shelf_clearance_examples.svg`는 실제 측정 geometry의
10종 전체 seed 0·2·10으로 병진·pivot 후 이동·회전 후 이동을 보여준다.
이는 생성 기하와 경로 예약의 설명이며 실제 camera visibility나 조작 실행 결과는 아니다.

### 2026-10-09 실행 스크립트의 5초 자동 reset

사용자 요청으로 `spawn_scene.py`의 기본 동작을 시뮬레이션 시간 5초마다 전체
환경을 reset하도록 변경했다. `ceil(5 / env.step_dt)`로 계산하며 현재 150 step이다.
기존 `--reset_interval N`의 step 단위 지정과 0의 비활성 동작을 유지한다.

Headless CUDA 환경 1개를 `--max_steps 151`로 실행해 exit code 0으로 종료했다.
150 step·5.00초에서 reset하고 seed 0의 4개 장면이 seed 97409의 6개 장면으로
바뀌는 것을 확인했다. 로그는 `outputs/contact_rearrangement/five_second_reset_validation.log`다.
이 실행은 자동 reset 주기 확인이며 `--validate`를 사용한 settle 안정성 검사는 아니다.
스크립트 문법 검사와 `git diff --check`도 통과했다.

### 2026-10-09 YCB 박스의 초기 낙하·흔들림 수정 — 제한된 GPU 정착·reset 검증 통과

**사용자 요청:** 스폰 직후 YCB 박스가 낙하하며 흔들리는 현상을 해결한다.
먼저 모든 종류의 2 mm 낙하 여유를 제거해 `root_z=shelf_top-lower_z`로 바꿨다.
높이를 바꾸는 것만으로 원본 scanned mesh collider의 접촉면이 평평해지는 것은
아니므로 collision 형상을 별도로 비교했다.

| 중간 시험 | 확인한 결과와 상태 |
|---|---|
| 여유 0·원본 collider, 4개 환경·10종 전체·5초 | 003 최대 기울기 66.96°·XY 이동 11.3 cm, 004 최대 3.80°·XY 5.79 mm. 높이 수정만으로 안정화하지 못했다. |
| 원본 mesh의 `boundingCube` fitted approximation | 0.83°·XY 1.54 mm의 움직임이 남아 최종 방식으로 사용하지 않았다. |
| 명시적 body-local AABB Cube proxy의 초기 진단 | 4개 환경·10종 전체·5초 동안 박스 8개를 120 Hz로 기록. 최대 XY 이동 0.007894 mm, 계산 정밀도 내 거의 직립. |

**구현:** 당시 `object_spawner.py`(현재 `assets.py`에 통합)의 `FlatBoxUsdCfg`는 003·004의 visible scanned
mesh를 보존하고 원본 collider를 disabled로 바꾸며, 기존 single rigid body 아래에
같은 unscaled body-local AABB 크기·offset의 invisible `UsdGeom.Cube`를 추가한다.
Cube size 1·extent ±0.5를 명시하고 body·ancestor scale을 상속하며 spawn/clone
시점에만 override한다. 원본
Nucleus USD는 무수정이고 body의 authored mass는 유지한다. 원본 collider의
authored `physxCollision:*` contact 속성을 가져오고 실제 `UsdPhysics.MaterialAPI`가
있는 physics material binding만 proxy에 적용한다. Visual material을 physics
material로 오인하지 않도록 guard한다. Active native gravity·simulation과 inactive
COM countergravity도 유지한다.
003·004의 접촉 형상을 평평한 box로 근사하는 변경이며 원본 mesh collision과
동일하다고 표현하지 않는다. 005·006과 기존 6종은 원본 collider를 사용한다.

**Unit 검사:** 새 높이 0에 맞춰 fixture를 수정한 20개 sampling test가 통과했다.
이는 placement·역할·경로 명세 검사이며 새 collision의 물리 정착 성공을 증명하지
않는다. 실행 시간은 2.934초였고 `py_compile`·`git diff --check`도 통과했다.

**초기 정량 진단:** 4개 GPU 환경·10종 전체 장면을 5초 유지하면서 박스 8개를
매 physics substep (120 Hz) 기록했다. 최대 XY 이동은 **0.007894 mm**였다.
Float32 `acos` 기울기 값은 계산 정밀도 내 0이었으므로 **거의 직립**으로
기록하고 수학적으로 정확한 0° 보장으로 취급하지 않는다. 003 mass 0.411 kg,
004 mass 0.514 kg를 유지했고 진단한 박스 8개 각각에서 rigid body 1개·enabled Cube
collider 1개, 원본 mesh collision disabled를 확인했다. 로그·JSON은
`outputs/contact_rearrangement/box_supported_proxy_diagnostic.{log,json}`다.

**최종 공식 실행:** Cube extent·physics material guard를 보강한 버전에서
아래 명령이 exit code 0으로 통과했다. Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090,
기본 seed 0, 환경당 100-body pool에서 10종 각각 하나씩 active다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

총 300 environment step·10.00초다. Reset interval을 생략해 기본 150 step을
사용했고 150·300 step에서 전체 reset, 끝에서 부분 reset을 확인했다.
최초 4개 환경의 layout과 첫 reset 뒤 4개 layout, **총 8개 layout을 각각
5초** 진행했다. 300 step에서 새로 생성한 장면에는 물리 step을 진행하지
않았으므로 추가 정착 검증으로 세지 않는다. 5 cm drift·15° tilt 기준은
완화하지 않았다. 전체/부분 reset과 bounded settle 검사가 통과한 범위이며,
다른 seed·모든 배치·실제 robot push·camera visibility의 검증으로 확대하지 않는다.
로그는 `outputs/contact_rearrangement/box_stable_reset_validation.log`다.

### 2026-10-09 종류·instance 분리와 전체 선반 밀도 — 이전 면적 우선 구성의 검사

아래 수치·실행·배치도는 큰 footprint부터 복사본을 고르던 **이전 구성**의
기록이다. 현재 균형 구성의 instance 분포·생성 성능·정착 결과로 사용하지 않는다.

사용자가 **4–10은 종류 선정 기준**이라고 교정했고 선반 전체의 density를
충족하도록 요청했다. 이전 active 4–10개·all-types 정확히 10개 해석을 폐기했다.
당시는 종류 4–10개를 고른 뒤 선택한 non-target 종류의 큰 footprint 복사본부터 더해 요구 면적을
충족한다. Target 종류는 한 번만 사용하며 instance 최소 4·cap 40,
10종 × 10-copy pool을 유지한다. 역할·종류 집합·전체 instance ID·N은 배치
재시도 전에 고정한다. 구성에 필요한 면적 하한은
`max(min_shelf_occupancy, min_region_occupancy) × usable_area`다.
Global 면적과 X 2 × Y 3의 6개 구역 면적 하한을 모두 실제 배치에서 검사한다.

**기하 benchmark와 수치 선택:** 실제 USD 측정 bounds와 현재 clearance 조건으로
global 25–45%를 비교했다. 아래는 일반·all-types 두 모드의 고유 sample 수다.

| Global 요구 | Seed·표본 | 생성 통과 |
|---|---|---|
| 25% | 각 모드 seed 0–4, 총 10개 | 10/10 |
| 30% | 각 모드 seed 0–4, 총 10개 | 10/10 |
| 35% | 각 모드 seed 0–19, 총 40개 | 40/40 |
| 40% | 각 모드 seed 0–4·10–19, 총 30개 | 30/30 |
| 45% | 각 모드 seed 0–4, 총 10개 | 3/10 — 일반 1/5, all-types 2/5 |

35%·40%의 공통 seed 집합에서 평균 생성 시간은 각각 약 0.206초와
0.99–1.00초였다. 45% 실패는 일반 모드의 capacity 1개·packing 3개,
all-types의 packing 3개였다. 생성 성공률과 약 5배의 시간 차이를 고려해
**global 35%·각 구역 10%·4 cm 이웃 비율 75%**를 초기 구현 기본값으로 선택했다.
사용자가 직접 선택한 최종 연구 난이도 수치는 아니다.

35%의 일반 모드는 4–10종·10–24 instance, all-types는 10종·14 instance였다.
실제 global 점유율은 각각 35.39–38.68%·36.32–37.36%였고, 장면별 최소
구역 점유율은 전체 sample에서 12.98–35.22%였다. 자료는
`outputs/contact_rearrangement/density_calibration.json`의 초기 비교·독립
확인·확장 기본값 확인에 저장했다. 같은 seed의 중복 검사를 표본 수에 합산하지
않는다. 이는 geometry 생성 검사이며 물리 정착·camera·robot push 검사는 아니다.

**당시 GPU 검사:** 다음 두 실행이 모두 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --all_asset_types --num_envs 4 --headless --device cuda:0 --max_steps 300 --validate
```

각 실행은 기본 35%·seed 0·4개 GPU 환경·300 step·10.00초다. 기본 150-step
간격으로 150·300 step의 전체 reset과 끝부분의 partial reset을 확인했다.
각 모드에서 초기 4개·첫 reset 뒤 4개, **8개 layout을 각각 5초** 진행했고
5 cm drift·15° tilt 기준을 완화하지 않았다. 마지막 reset에서 생성한 장면은
추가 물리 step이 없어 정착 표본으로 세지 않는다. 로그는
`outputs/contact_rearrangement/dense_shelf_default_validation.log`와
`outputs/contact_rearrangement/dense_shelf_all_types_validation.log`다.
미검사 seed·실제 로봇 밀기·camera visibility의 성공으로 확대하지 않는다.

**당시 단위 검사:** 21개가 18.219초에 통과했고 `py_compile`·`git diff --check`도
통과했다. Region 요구가 global보다 큰 경우의 composition 면적 하한 검사도
포함하며 기본 35%/10% 동작은 같다.

**당시 배치도:** 실제 측정 geometry의 4종·10개, 4종·24개, 10종·14개를
비교했다. 해당 파일은 이후 아래의 현재 균형 구성 예시로 교체됐으며 이전
개수의 그림을 현재 배치도로 연결하지 않는다.

### 2026-10-09 Target·Blocker의 현재 OBB 표시

사용자 요청에 따라 `visualization.py`의 `RoleObbVisualizer`를 장면 확인
스크립트에 연결했다. Target은 초록색, Blocker는 빨간색으로 각 12개 edge를
debug draw한다. 현재 body link pose와 upright frame bounds를 사용해 YCB 방향,
중심과 다른 root 위치, 합성 scale을 반영하며 전체·부분 reset의 역할 ID도 갱신한다.
USD prim·API·collider·material·physics state를 추가하거나 변경하지 않는다.
GUI·livestream에서는 기본 표시하고 `--disable_role_obb`로 끈다. 일반 headless는
생략하고 `--headless --validate`는 viewport 없이 같은 debug draw 경로를 검사한다.

**제한된 실행 검사:** YCB 003 Target·006 Blocker를 지정한 2개 CUDA 환경에서
151 step·5.03초의 headless `--validate`가 exit code 0으로 통과했다. 총 48개
line과 draw 전후 USD prim/API 집합·PhysX pose/velocity·native flag 유지,
150-step 자동 reset과 마지막 전체·부분 reset 갱신을 확인했다. 별도의 1개 GUI
환경·45 step·1.5초 `--validate`도 exit code 0이었다. 실제 RTX viewport 렌더
`outputs/contact_rearrangement/role_obb_preview.png`에서 초록 cracker box·빨간
mustard bottle의 OBB가 물체를 감싸는 것을 확인했다. 이 검사는 역할 표시와
무변경 동작의 범위이며 task camera의 Target 40% 가시성이나 로봇 조작 검사가 아니다.

**독립 수치 검사:** 별도의 CPU 행렬 기준과 비교해 legacy·YCB upright, yaw/tilt,
scaled/noncentered bounds, 두 환경의 world offset, 이동·기울어진 pose와 부분
역할 교체를 확인했다. 48개 edge의 연결·초록/빨강 색상·3 px 두께, redraw/cleanup과
body pose의 정확한 유지도 통과했다. 검사 스크립트는
`/tmp/contact_rearrangement_obb_numerical_review.py`이며 `py_compile`과
`git diff --check`도 통과했다.

### 2026-10-09 non-target 복사본의 종류별 개수 균형

사용자가 한 asset의 복사본이 공간을 지나치게 채우는 문제를 지적했다.
이전 큰 footprint 우선 복사본 선택은 채울 면적에 필요한 instance 수를 줄이는
대신 큰 종류에 중복을 집중시켰다. 현재는 추가 가능한 non-target 종류 중
**active 개수가 가장 적은 종류**를 고르고 동률만 무작위로 정한다. 지정
Blocker도 해당 종류의 첫 instance로 세며 모든 종류의 개수가 같아지면
다음 회차를 시작한다. Target 종류는 항상 하나다.

현재 같은 pool capacity에서는 non-target 종류별 개수 차이가 최대 1이다.
종류별 새 고정 최대값은 두지 않고 기존 종류당 10-copy·전체 cap 40을 유지한다.
작은 종류 위주 집합은 같은 global 점유율에 더 많은 instance가 필요할 수 있다.
종류 4–10개, 역할·종류 집합·전체 instance ID·N의 retry 전 고정,
global 35%·6개 구역 각 10%·이웃 75%와 clearance 조건은 유지한다.
공간 배치는 packing을 위해 큰 footprint부터 진행하므로 **종류 선택과 위치
배치의 순서는 별개**다. 이전 calibration·instance 분포·배치도를 현재 결과로
사용하지 않으며 다음은 최종 source에서 별도로 확인한 결과다.

**현재 기하 검사:** 실제 USD bounds로 일반·all-types 각각 seed 0–19,
**40/40개**가 통과했다. 모든 장면에서 non-target 종류별 개수 차이가 최대 1이었다.
일반 모드는 12–25 instance·global 35.05–38.48%·검사 전체의 최소 구역 22.86%,
all-types는 15–18 instance·global 35.14–38.71%·최소 구역 24.75%였다.
Source SHA를 포함한 자료는
`outputs/contact_rearrangement/balanced_composition_benchmark.json`에 있다.
이는 geometry 생성 검사이며 camera·robot push 검사와 구분한다.

**현재 GPU 검사:** 다음 일반 모드 실행이 exit code 0으로 통과했다.

```bash
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py \
  --num_envs 4 --headless --device cuda:0 --max_steps 151 --validate
```

4개 CUDA 환경·151 step·5.03초에서 150-step 자동 reset과 마지막 전체·부분
reset, 기존 정착 guard를 확인했다. 총 96개 역할 OBB line과 draw 전후 USD·
PhysX 상태 유지도 검사했다. Reset sample 중 10종 장면이 있었지만 현재 구성의
별도 `--all_asset_types` GPU 실행을 완료한 것으로 해석하지 않는다. 로그는
`outputs/contact_rearrangement/balanced_composition_gpu_validation.log`다.
미검사 seed·실제 로봇 밀기·camera visibility의 성공으로 확대하지 않는다.
당시 unit test **23개**가 24.384초에 통과했으며 균형 선정·복사본 재고 소진
회귀 검사를 포함한다.

**현재 배치도:** [균형 구성 예시](assets/dense_shelf_examples.svg)는 일반 seed 0의
4종·12개, 일반 seed 2의 4종·25개, all-types seed 1의 10종·17개를 보여준다.
Target은 하나이며 non-target 종류별 개수는 각각 3/4/4, 8/8/8,
1/1/2/2/2/2/2/2/2다. Console의 `[SPAWN]`에도 반복 목록 대신 종류별
`asset_counts`를 출력해 편중을 확인할 수 있다.

### 2026-10-09 책임별 6개 모듈 통합 — 동작 유지 검사 통과

과하게 세분화된 파일을 `catalog.py`·`assets.py`·`env_cfg.py`·`spawn.py`·
`sampling.py`·`visualization.py`로 통합하고 빈 `__init__.py`를 유지했다.
이전 모듈·`mdp/`는 wrapper 없이 제거했다. 실행 진입점과 CLI, 스폰 규칙·
기본값은 유지했으며 새 시뮬레이션·camera 기능을 추가하지 않았다.

모든 모듈의 문법 검사가 통과했고 `catalog`·`sampling`을 simulator 없이
import해 `isaaclab`·`isaacsim`·`pxr`·`torch`가 로드되지 않는 것을 확인했다.
CLI `--help`는 exit code 0이었다. Unit test **23개**가 24.552초에 통과했다.
6개 모듈의 의존관계에는 cycle이 없고 상대 Markdown 링크도 유효했다.

일반 모드의 `--num_envs 4 --headless --device cuda:0 --max_steps 151 --validate`가
exit code 0으로 통과했다. 4개 CUDA 환경·5.03초에서 150-step 자동 reset과
마지막 전체/부분 reset, 역할 OBB 96개 line·draw 전후 USD/PhysX 상태 유지를
검사했다. 물리 guard를 완화하지 않았다. 초기·150-step reset 뒤의 **8개 장면
설명**은 이전 `balanced_composition_gpu_validation.log` 실행과 정확히 같았다.
로봇·물체 18개 정의, USD 측정/reset 정의와 scene/env 설정의 AST도 통합 전과
일치했다. 이 결과를 새 camera 가시성·로봇 조작의 검증으로 확대하지 않는다.
로그는 `outputs/contact_rearrangement/refactor_cli_help.log`,
`refactor_unit_tests.log`, `refactor_gpu_validation.log`에 남겼다.

### 2026-10-09 Blocker 앞 hand 시작과 elbow-down 금지 — 이전 tool 방향의 검증

사용자 요청으로 기존 고정 robot 시작 pose를 episode별 Blocker 앞 영역의
collision-checked elbow-up 시작으로 확장했다. 6개 모듈 구조를 유지하고
`assets.py`가 USD FK/IK·collision geometry, `sampling.py`가 순수 영역/elbow/OBB
계산, `spawn.py`가 독립 seed·후보 승인·state/PD target·metadata를 담당한다.
물체 선정·배치 RNG를 유지하고 추가 physics step 없이 선택한 환경에만 적용한다.
거리·높이·여유·80번 cap은 현재 구현 기본값이며 후속 제어·학습 분포에 맞춰
조정할 수 있다.

이 절의 수치와 렌더는 tool 전방이 world −Z였던 당시 orientation의 검사다.
2026-10-10의 선반 방향 교정 결과로 사용하지 않는다.

초기 구현은 4개 GPU 환경·151 step·5.03초의 검사에서 150-step·5초 자동 reset과
마지막 전체/부분 reset, 선택하지 않은 환경의 q/velocity/PD target과 robot metadata
유지를 통과했다. Robot collider 39개·shelf collider 17개와 모델 FK 대 실제 link
position 오차 5.13e−7 m도 확인했다. 이후 solver에 같은 TCP의 다른 IK branch를
scene 조건까지 검사하는 callback을 넣고 물체·robot 후보 모두 성공한 뒤
reset/state를 적용하도록 보강했다. 초기 검사와 아래의 최종 변경본 검사를 구분한다.

후속 stress에서 **4개 환경 × 6회 reset의 24개 robot 시작 pose**는 실제 PhysX
조건을 통과했다. `min_elbow_height=2 m`·1회 시도로 실패를 강제해 joint q·object
pose·PD target·reset count·역할·robot metadata 유지도 확인했다. 다만 physics 없이
6번 reset한 뒤 150 step에서 parked body angular velocity가 32–71 rad/s로 증가했다.
해당 경로의 상위 `RigidObjectCollection.write_object_link_pose_to_sim`이 별도 COM
pose cache를 무효화하지 않아 같은 timestamp의 이전 COM을 읽는 것을 확인했다.
Pool의 writer에서 그 cache를 무효화하도록 수정했다. 이후 UR arm의 가까운
joint housing OBB가 충돌로 잡히는 false positive도 수정했다. 직렬 7-link arm의
21쌍에서 두 arm joint 이내의 11쌍을 제외하고 거리 3 이상인 10쌍만 self 검사한다.
전체 39개 robot collider 대 물체·17개 shelf collider·ground의 **15 mm 기준은
완화하지 않았다.**

**최종 변경본 검사:** 4개 GPU 환경 × 6회 reset의 **24개 초기 pose**가 실제
PhysX 조건을 통과했고 8개는 10종 전체 장면이었다. 같은 timestamp에서 physics
없이 6번 reset한 뒤 **150 step·5초**의 parked 안정성, 전체/부분 reset도 통과했다.
`min_elbow_height=2 m`·1회 시도로 실패를 강제했을 때도 physical joint q·object
pose·PD target·reset count·역할·TCP metadata가 유지됐다. 최종 FK position 오차는
**2.8458e−7 m**, elbow chord 높이는 **11.49–16.07 cm**, 후보 시도는 **1–3회**였다.
현재 10종의 enabled collider mesh points·primitive OBB가 측정한 물체 bounds 안에
포함되는 것도 확인해 초기 scene obstacle OBB의 보수적인 형상을 대조했다.
최종 unit test **29개**(기존 23개 + 영역/SAT/elbow 6개)가 **24.673초**에 통과했다.
로그는 `outputs/contact_rearrangement/robot_randomization_unit_tests.log`다. Stress 자료는
`outputs/contact_rearrangement/robot_randomization_benchmark.{json,log}`다.

4환경 reset batch의 wall time은 물체 배치를 포함해 **1.420–7.601초**였다.
현재 CPU 후보 계산 비용은 환경 수를 확대할 때 batch/vectorization 검토가 필요한
한계다.

**GUI 검사:** 45 step·1.5초의 실제 GUI 실행과 전체/부분 reset, COM cache를
미리 읽은 뒤 reset하는 검사, OBB draw 전후 USD/PhysX 불변 조건도 통과했다.
실제 렌더에서 손이 Blocker 앞 공간에 있고 elbow가 높게 위치한 것을 확인했다.
당시 palm normal은 선반을 향했지만 tool 전방은 아래를 향했다. 자료는
`outputs/contact_rearrangement/robot_start_preview.{png,log}`다.
Elbow 높이 계산은 `assets.py`의 FK와 `sampling.elbow_above_chord`를 공유한다.

**최종 자동 reset 검사:** 현재 변경본의 일반 모드
`--num_envs 4 --headless --device cuda:0 --max_steps 151 --validate`도 exit code 0으로
통과했다. 151 step·5.03초에서 150 step·5.00초의 자동 reset, 부분 reset의
미선택 환경 보존, COM cache를 미리 읽고 pose write 뒤 새로 읽는 검사,
역할 OBB draw 전후 USD/PhysX 상태 유지를 확인했다. 로그는
`outputs/contact_rearrangement/robot_randomization_gpu.log`다.
위 통과를 손 전체 self collision·재배치 제어 성공·task camera 가시성의 보장으로
확대하지 않는다.

### 2026-10-10 Tool 전방을 선반 방향으로 교정 — hand roll 추가 전 검증

사용자가 시작 자세에서 tool이 바닥을 향한다고 지적하고 정면인 선반 방향으로
수정하도록 요청했다. 이전 목표 회전은 palm normal인 TCP +Z를 world −X로
맞춰 실제 tool 전방인 TCP +Y·wrist-3 +Z를 world −Z로 향하게 했다.
`sampling.ROBOT_START_TCP_ROTATION`을 TCP +X→world +Y, +Y→world −X,
+Z→world +Z로 교정했다. 손 mount·시작 영역·elbow-up·15 mm 외부 충돌 여유와
물체 배치 규칙은 유지한다. Reset 검증에 실제 PhysX TCP 전체 회전과 wrist-3
tool 축 검사를 추가했다.

**선반 방향 GPU 검사:** 일반 모드 4개 GPU 환경·151 step·5.03초의
`--validate` 실행이 exit code 0으로 통과했다. 초기 4개·5초 자동 reset 4개·
부분 reset 1개·마지막 전체 reset 4개, **총 13개 시작 pose**에서 실제 PhysX
TCP 회전 오차 ≤0.01001 rad와 wrist-3 tool 전방의 world −X 축 검사를 통과했다.
Elbow-up·초기 scene 충돌 여유·부분 reset의 미선택 환경 유지·COM cache 갱신·
OBB 표시 전후 물리 상태 불변도 확인했다. 로그는
`outputs/contact_rearrangement/tool_forward_gpu_validation.log`다.
이번 수정에서 순수 helper는 바꾸지 않았으므로 앞의 29개 unit test를 새 결과로
기록하지 않는다.

**선반 방향 GUI 검사:** 1개 환경·45 step·1.50초의 실제 GUI 실행과 전체/부분
reset, OBB 표시 전후 물리 상태 불변 검사가 exit code 0으로 통과했다. 렌더에서
tool·손가락이 선반 정면을 향하는 것을 확인했다. 45-step GUI 실행 중 캡처
시점의 PD 유지 pose에서
wrist-3 tool 전방은 world `(-0.99947666,-0.0005177,-0.03234399)`이며 정확한
world −X와의 각도는 **0.03235 rad, 약 1.85°**였다. 초기 reset의 0.01 rad 조건과
구분해 캡처 시점의 유지 pose는 별도 0.05 rad 검사를 통과했으며, 물리 진행 내내 정확한
수평 회전이 유지된다는 의미는 아니다. 자료는
`outputs/contact_rearrangement/tool_forward_preview.{png,log}`다. 이전
`robot_start_preview` 렌더는 과거 orientation의 자료로 남긴다.

### 2026-10-10 Tool +Z축 기준 hand −90° roll — 원통 연결부 추가 전 검증

사용자가 tool 기준 Z축으로 hand를 −90° 회전하도록 요청했다.
`Ur5eHandSpawnerCfg.mount_tool_roll_deg=-90.0`을 추가하고 기존 vendor adapter
`mount_rotation_deg=(0,-90,0)` deg는 유지한다. 실제 조립에서는 `tool0` attachment
frame과 physical tool body `wrist_3_link`의 축이 서로 달라, 두 frame 사이의
transform을 반영한 뒤 physical tool +Z를 기준으로 hand mount를 회전한다.
Hand의 상대 회전뿐 아니라 상대 translation도 함께 회전하며 UR wrist 관절의
기본값만 바꾸는 방식으로 처리하지 않는다.

TCP 목표는 X→world +Z, Y→world −X, Z→world −Y로 바꾼다. 손가락/tool 전방
−X, 시작 영역, elbow-up, 15 mm 외부 충돌 여유와 물체 생성 규칙은 유지한다.
앞 절의 `tool_forward_*` 검사·렌더는 hand roll 추가 전의 결과로 남긴다.

**Hand roll GPU 검사:** 일반 모드 **4개 GPU 환경·151 step·5.03초**의
`--validate`가 exit code 0으로 통과했다. 초기 4개·5초 자동 reset 4개·부분 reset
1개·마지막 전체 reset 4개, **총 13개 시작 pose**에서 실제 PhysX TCP의 새
회전과 wrist-3 tool 전방 −X, elbow-up·15 mm scene 여유를 확인했다. 미선택
환경 유지·COM cache·pool flag 유지·역할 OBB 표시의 물리 상태 불변도 통과했다.
로그는 `outputs/contact_rearrangement/hand_tool_roll_gpu_validation.log`다.

**Hand roll 조립·GUI 검사:** 첫 reset의 실제 PhysX pose에서 physical tool body
기준 새 hand 상대 회전이 **`R_new = Rz(-90°) @ R_old`**임을 확인했다. 회전
오차는 **3.23141e−7 rad**였고 TCP 상대 translation에도 같은 회전이 적용된
것을 1e−5 m 허용 오차로 확인했다. 기존 tool 정면 목표에 대한 전체 world
회전 오차는 **1.15236e−5 rad**였다. 이로써 hand mount의 roll과 tool 전방
유지를 실제 조립된 body pose로 대조했다.
1개 GUI 환경·45 step·1.50초와 마지막 전체/부분 reset·역할 OBB 불변 검사도
exit code 0으로 통과했다. 렌더는 실행 중 약 20번째 step의 캡처이며 앞의
상대 transform 수치는 첫 update의 초기 pose에서 측정했다. 자료는
`outputs/contact_rearrangement/hand_tool_roll_preview.{png,log}`다. 순수 helper를
바꾸지 않았고 이번 수정에서 새 unit test 실행 결과를 추가하지 않는다.

### 2026-10-10 UR5e 말단과 hand 사이 5 cm 원통 연결부 — 중앙 F/T 추가 전 검증

사용자가 실제 센서·연결부를 고려해 UR5e 말단과 hand 사이에 길이 5 cm의
원통을 요청했다. `Ur5eHandSpawnerCfg.sensor_mount_length=0.05 m`,
`sensor_mount_radius=0.025 m`로 설정한다. 직경 50 mm는 실제 hand
`base_link.STL`의 mount X/Y bounds ±25 mm를 기준으로 선택한 구현 기본값이다.
Wrist-3 아래 visible collision Cylinder `SensorMount`를 추가하며 physical tool
+Z 방향 z=0–50 mm, 중심 z=25 mm에 둔다. World rigid rotation을 반영해 원본
asset scale과 무관하게 실제 치수를 적용한다.

Hand fixed-joint anchor를 같은 tool +Z 방향으로 50 mm 옮기고 −90° hand roll과
TCP 목표 회전은 유지한다. 원통은 wrist-3 compound collider이므로 별도 rigid
body나 DOF를 추가하지 않는다. FK body 39개·DOF 18개는 유지하며 robot collider는
39개에서 **40개**로 늘어난다. 원통도 기존 15 mm scene 여유와 arm self
collision 판정에 포함한다. 원본 USD를 수정하지 않고 기존 wrist mass/inertia를
유지한다. 실제 센서 질량 보정·F/T 센서 출력은 아직 구현되지 않았다.
위 `hand_tool_roll_*` 결과는 원통 추가 전 검증이며 새 조립 검사는 별도로 기록한다.

**연결부 GPU 검사:** 일반 모드 **4개 GPU 환경·151 step·5.03초**의
`--validate`가 exit code 0으로 통과했다. 초기 4개·5초 자동 reset 4개·부분 reset
1개·마지막 전체 reset 4개, **총 13개 시작 pose**에서 실제 PhysX TCP 회전·tool
전방 −X·elbow-up·15 mm scene 여유를 확인했다. Robot collider **40개**·shelf
collider 17개, 실제 link 대 FK position 오차 **5.15e−7 m**였다. 미선택 환경
유지·parking COM cache·pool flag·역할 OBB 표시의 물리 상태 불변도 통과했다.
로그는 `outputs/contact_rearrangement/sensor_mount_gpu_validation.log`다.

**연결부 조립·GUI 검사:** 첫 reset의 실제 PhysX pose에서 wrist-3→hand base
offset은 tool body local `[0,0,0.05] m`, gap은 **0.049999997 m**였다. 원통의
near/far end가 wrist와 hand base origin에 각각 1e−5 m 이내로 일치했고 길이
50 mm·반지름 25 mm를 확인했다. 기존 −90° 상대 hand roll의 오차는
**4.51901e−7 rad**였으며 TCP 상대 offset도 원통 추가 전 값에 `[0,0,0.05] m`를
더한 값과 일치했다. 39 body·18 DOF·40 collider를 유지한 실제 조립을 대조했다.
1개 GUI 환경·45 step·1.50초와 마지막 전체/부분 reset·역할 OBB 상태 불변
검사도 exit code 0으로 통과했다. 실행 중 약 20번째 step의 넓은 시야 렌더와
약 32번째 step의 연결부 근접 렌더에서 원통 공간을 확인했다. 자료는
`outputs/contact_rearrangement/sensor_mount_preview.{png,log}`와
`sensor_mount_detail.png`다. 위 상대 transform 수치는 첫 reset에서 측정했으며
이번 수정에서 새 unit test 실행 결과를 추가하지 않는다.

### 2026-10-10 원통 중앙 F/T와 initial vision acceptance

사용자가 원통 중앙의 articulation fixed-joint F/T와 앞쪽 초기 vision camera를
요청했다. 원통을 두 25 mm rigid half로 분리하고 중앙 fixed joint를 통해 hand
load를 전달한다. Nominal 질량 합 0.20 kg와 원통 inertia를 명시하며 원본 wrist
질량은 유지한다. Body·collider는 각각 41개, DOF는 18개다. Sensor frame의 raw
반력을 읽고 reset 직후 stale sample은 valid=False로 제외한다. 자동 tare·중력
보정·filtering은 추가하지 않는다.

실제 같은 pose의 비가림 reference로 Target ≥40%·Blocker 전체(2 pixel tolerance)
가시성을 검사한다. 앞쪽 camera 위치·aim·roll을 무작위로 정하고 승인한 RGB-D와
실제 role mask를 초기 snapshot으로 보존한다. Physics step 없이 camera/layout
retry를 진행하며 역할·종류·active ID·개수는 유지한다. `sensors.py` 한 모듈에
render·snapshot·F/T observation 연결을 모으고 script는 camera를 자동 활성화한다.
검증 결과는 이전 원통 geometry 검사와 구분해 기록한다.

Blocker 앞 reserve 통로와 환경별 reference lane 분리를 추가했다. 변경 후
순수 sampling 검사 **32개**가 **27.993초**에 통과했다. 이 결과는 렌더 가시성·
F/T 정확도·GPU reset 검사의 통과가 아니며 실제 센서 실행 결과는 별도로 기록한다.

낮은 camera 후보에서는 실제 robot arm·hand가 역할 물체를 가리는 것을 확인해
기본 camera 높이를 상면 +0.18–0.48 m로 넓혔다. 1개 환경의 진단에서는 같은
composition의 layout을 한 번 재시도한 뒤 높은 camera 후보가 Target **56.370%**·
Blocker **100%**로 2번째 camera 시도에 승인됐다. 이는 한 진단 장면의 결과이며
다중 환경 GPU 검증이나 전체 생성 성공률로 해석하지 않는다.

Reset의 native physics counter 추적에서 첫 camera pose의 SDK Fabric sync가
Kit update를 통해 physics를 진행시키는 경로를 확인했다. 로컬 `InitialCamera`와
camera update guard로 이 경로를 보호한다. 이 수정의 reset/render/time/native
counter 검증은 아래 최종 GPU 실행에서 통과했다.

**최종 센서 GPU 검사:** `outputs/contact_rearrangement/initial_sensors_gpu_validation.log`
실행이 exit code 0으로 통과했다. 4개 환경의 public script **151 env step·
5.03초**, 추가 2 env step과 F/T bench 7 physics step을 검사했다. 전체 reset
6회 중 2회는 부분 reset이며 **18개 승인 시작 pose**였다. Successful capture
call은 **6회**, failed layout은 **2개**, render는 두 warmup call을 포함해
**464회**였다. Capture 중 object/robot native pose·velocity·joint q/v·PD target·
physics time/step은 정확히 유지됐으며 추가 physics step이 없었다. SDK 초기화의
별도 physics step 2회는 env step·reset/capture 검사와 분리해 센다.

18개 시작의 Target fraction은 **47.1253–100%**였다. Blocker는 17개가 정확히
100%, 1개가 **99.97586%**로 정확히 **2 pixel** 손실을 허용한 경우였다. 승인한
camera 후보 시도는 1–30회였다. Native robot은 **41 body·18 DOF·41 collider**다.
Reset 직후 선택된 F/T는 zero·invalid였고 다음 physics step 뒤 valid·finite로
갱신됐다. 미선택 F/T·snapshot cache를 보존했으며 153 env step 동안 승인한
initial snapshot은 고정됐다.

F/T bench는 distal link에 sensor XYZ별 **3 N force**와 **0.10 N·m torque**를
독립 적용해 baseline 및 Newton–Euler의 중력·관성 항과 대조했다. Downstream은
33개 body·질량 합 **0.876663 kg**였다. 최대 vector norm 오차는 force
**0.0206053 N**, torque **0.00132915 N·m**로 기준 0.10 N·0.015 N·m보다 작았다.
중앙 측정점·sensor frame·parent→child 부호·성분 순서를 실제 물리 반력으로
확인했다. 이는 해당 제한된 초기 장면과 bench의 검증이며 모든 seed·실제
재배치 성공·실제 하드웨어 교정·보정 F/T observation의 입증은 아니다.

**센서 GUI·실패 복구 검사:** 1개 GUI 환경·45 env step·1.50초가 exit code 0으로
통과했다. Target `ycb_003_cracker_box`는 **7724/9286 pixel, 83.178979%**, Blocker
`ycb_006_mustard_bottle`은 **3608/3608 pixel, 100%**, camera 후보 3회였다.
실제 원통은 25 mm half 두 개, 중앙점 wrist +25 mm·hand anchor +50 mm와
41 body·18 DOF·41 collider를 확인했다. Capture 실패와 중간 commit 오류를
강제한 이동 상태에서도 물체 COM velocity·robot root pose/velocity·q/qd·
position/velocity target·camera pose·모든 초기 snapshot/역할/seed/count/sample·
F/T raw/valid/tare·clock/step 복구가 통과했다. Authoritative root·DOF state는
정확히 복구됐으며 derived robot link velocity는 kinematic 재계산 전 solver
출력과 최대 **4.47e−5** 차이가 있어 exact 복구로 기록하지 않는다.
자료는 `outputs/contact_rearrangement/initial_sensors_preview.log`,
실제 초기 camera `initial_camera_preview.{png,json}`, GUI overview
`initial_sensors_preview.png`, 연결부 확대 `initial_sensors_detail.png`다.
Camera/F/T·전체/부분 reset의 재사용 검증 진입점은
[check_sensors.py](../src/contact_rearrangement/tests/check_sensors.py)이며
실행 명령은 [README](../src/contact_rearrangement/README.md)에 기록한다.

**Repository 진입점 재현:** `check_sensors.py --num_envs 1`도 exit code 0으로
통과했다. 151+2 env step·bench 7 physics step, reset/capture 각 6회(부분 reset
0회), render 179회였다. 승인 6개의 Target **41.0632–100%**·Blocker 전부 100%,
reset/capture의 physics step 증가 0·snapshot 보존·41 body/18 DOF/41 collider를
확인했다. 6축 bench의 최대 norm 오차는 **0.0122181 N·0.000850297 N·m**였다.
로그는 `outputs/contact_rearrangement/initial_sensors_reusable_check.log`다.
기존 4환경·GUI 결과와 별도 실행이며 `compileall`·`git diff --check`도 통과했다.

### 2026-10-10 Tactile 17채널 실제 접촉·reset 검증

[check_tactile.py](../src/contact_rearrangement/tests/check_tactile.py) `--num_envs 2`
GPU 실행이 exit code 0으로 통과했다. Public 151 env step·5.03초, reset 6회
(부분 2회), render 116회에서 tactile cache와 physics time/step의 보존을 확인했다.
반지름 3 mm의 test-only sphere로 모든 17개 pad의 해당 채널 ON, 분리 뒤
4 physics step에서 모두 OFF를 확인했다. Probe penetration은 16개 pad 1 mm·
`right_little_force_sensor_3`만 2 mm, 측정력은 0.108537–31.0263 N이었다. 이는
binary 실제 접촉 검증이며 하중 교정으로 해석하지 않는다. 중력·PD hold만으로
모두 OFF, 둘째 환경 palm ON을 유지한 env0 부분 reset의 contacts/valid/raw cache
보존과 reset zero/invalid·다음 완료 step valid를 확인했다. Native 독립 body·
reporter 17개, 41 body·18 DOF·41 collider prim, dense/역할/robot elbow-up·support·
camera visibility·OBB 무변경도 통과했으며 FK 오차는 4.96e−7 m였다. 로그는
`outputs/contact_rearrangement/tactile_gpu_validation.log`다. Probe는 검증 스크립트에만
있으며 기본 task geometry에는 추가하지 않는다. Compile·diff check도 통과했다.

### 2026-10-10 학습용 MDP·초기 장면/16-step 모델·PPO

이 절은 현재/직전 두 sample로 변경하기 전의 구현·검증 기록이다.

사용자가 설계했던 MDP/모델 구현을 요청했고 최근 16 step 이력부터 시작하도록
선택했다. 기준 frame은 examples를 참조하고 물체 기하는 현재 EEF 기준으로
입력한다. 초기 geometry의 base copy와 현재 EEF pose로 재표현하며 현재 object
GT를 actor에 제공하지 않는다. Initial RGB-D의 visible point cloud PCA OBB,
valid/role mask, sensor-frame 보정 F/T, 18D OSC/hand action과 Reward v2·성공/실패,
true-final-observation timeout bootstrap을 구현했다. GRU·privileged critic·
보조 prediction loss는 도입하지 않는다. Model/PPO의 **14개 CPU 검사**에서
슬롯 순열 불변성, NaN padding/all-invalid, 16-step 전체 반영, bounded/saturated
action log-probability, 양 encoder gradient, terminal/timeout GAE 분리와 inference-mode
rollout update, model/optimizer checkpoint 복원이 통과했다.

**통합 검증:** 전체 CPU 검사 **89개**(sampling 32, action/contact/implicit control 22,
MDP 21, model/PPO 14)와 compile/diff 검사가 통과했다. 실제 GPU에서
`train.py --headless --num_envs 2 --iterations 2 --rollout_steps 16
--episode_length_s 0.8 --validate`가 exit code 0으로 완료했다.
열린 손의 unloaded OSC hold 16 step에서 EEF 위치 drift는 최대 **0.0914 mm**,
orientation drift는 최대 **0.00195 rad**, 보정 F/T norm peak는 **0.0108 N /
0.00138 N·m**였다. 실제 Blocker를 proximal cylinder와 겹치게 만든 검증 fixture에서
filtered arm/spacer contact failure가 ON임을 확인했다. 이 fixture는 검증 옵션에서만
실행하며 학습 시작 전에 장면을 다시 reset한다.

학습 rollout은 **64 transition**, PPO optimizer update **2회**를 완료했다.
첫 rollout에서 양 환경의 history 16개가 모두 유효했고 다음 rollout에서는
timeout **2회**의 실제 final observation을 auto-reset 이전에 보존했다.
학습 rollout의 failure는 0회, 보정 wrench peak는 **1.407 N / 0.1885 N·m**였다.
Geometry가 episode 동안 고정된 canonical 초기 값임을 확인하고, 현재 EEF 입력을
SDK frame transform으로 다시 base에 옮겨 원본 초기 기하와 일치함을 검사했다.
Terminal reward의 dt 적용, reset history/action 분리, native arm/mimic PD 비활성,
finite effort, 실제 parameter update 및 GPU checkpoint 복원도 통과했다.
로그는 `outputs/contact_rearrangement/learning_gpu_validation.log`, checkpoint는
`outputs/contact_rearrangement/learning_smoke/model_latest.pt`다. 0.8초 time limit는
reset 검증용이며 기본 학습 limit는 8초다. 이 검증은 구현 연결과 짧은 제어 안정성을
보여주며 재배치 task 성공·정책 수렴·실기 성능을 입증하지 않는다.

초기 진단에서는 SDK unit-mass explicit damping이 손을 열린 채 유지해도 팔을
진동시켰다. Native damping mode의 `dt×eigenvalue`가 최대 **51.78**이었다.
Hard-load 기준을 높이지 않고 coupled mass-aware damping과 backward-Euler 계산으로
수정했다. 또한 GPU에서 child collision mesh filter가 지원되지 않아 rigid-body
partner 경로로 교체했고, manager 초기화 시 이름이 겹치는 hand base 90 g을 포함해
F/T 중력 보상 대상 **33개 body**를 정확히 선택했다. 이 진단의 실패 결과를 위의
최종 통합 검증과 혼합하지 않는다. 기본 학습 출력은
`outputs/contact_rearrangement/learning/<시작 시각>/`에 저장한다.

### 2026-10-10 sweeping_policy 기준 코드 구조 정리

이 절의 학습 검증은 당시 16-step 모델을 사용했다.

사용자가 `examples`의 `sweeping_policy` 구조를 참고하도록 요청했다.
공통 scene·MDP 설정은 `shelf_rearrangement_env_cfg.py`, 로봇별 설정과 Gym
등록은 `config/ur5e`, 모델·PPO·runner 설정은 `config/ur5e/agents`, manager
함수와 공통 transition 상태는 `mdp`, 장면 지원 구현은 `src`로 옮겼다.
장면·센서·샘플러를 작은 helper 파일로 추가 분해하지 않으며 이전 flat 경로의
wrapper도 남기지 않는다. `train.py`는 등록된 task의 환경·agent 설정을 읽는다.
물체의 현재 EEF 기준 재표현, F/T sensor frame, 최근 16-step 이력, 18D action과
Reward v2 명세는 유지한다. 이전 통합 검증 수치를 이번 구조 변경의 검증 결과로
사용하지 않는다.

**구조 변경 후 실제 검증:** Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090·기존
Nucleus asset에서 CPU 검사 **92개**가 **33.203초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 21, model 9, PPO/runner 6, Gym registration 2다.
새 Gym ID의 env/agent entry point를 실제로 읽어 환경을 생성하고 이전
`learning_smoke/model_latest.pt`를 이어서 학습했다. **2환경·64 transition·추가
PPO update 2회(누적 4회)**, failure 0회·timeout 2회, 16-step history·부분 reset,
EEF 초기 기하의 SDK transform 대조, 실제 arm/spacer 접촉 실패 판정,
reward dt·native actuator 설정·parameter 갱신·GPU checkpoint 복원이 통과했다.
Hold drift 최대 **0.0914 mm**, 학습 중 보정 wrench peak는
**1.3860 N / 0.1841 N·m**였다. 실행 옵션은
`train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless --num_envs 2
--iterations 2 --rollout_steps 16 --episode_length_s 0.8
--resume outputs/contact_rearrangement/learning_smoke/model_latest.pt
--log_dir outputs/contact_rearrangement/structure_learning_smoke --validate`이며
로그는 `outputs/contact_rearrangement/structure_learning_validation.log`다.

장면 확인 환경도 `spawn_scene.py --headless --num_envs 2 --max_steps 151
--validate`로 exit code 0을 확인했다. **151 step·5.03초** 동안 150-step 자동
reset과 정착, 역할·밀도·elbow-up 시작, 전체/부분 reset, raw F/T·17채널 tactile
observation, OBB 표시의 USD/PhysX 불변 검사를 통과했다. 로그는
`outputs/contact_rearrangement/structure_spawn_validation.log`다. 이 구조 변경
검증을 재배치 정책의 수렴이나 task 성공률로 해석하지 않는다.

### 2026-10-10 시간 입력을 현재·직전 관측으로 축소 — mask 전달 교정 전 기록

사용자가 previous step 정보까지만 제공하도록 요청해 관측 window를
**현재 t와 직전 t−1의 두 sample**로 변경했다. `MdpCfg.history_length`와
당시 `ModelCfg.history_steps`는 2, actor/critic 입력은 **700D**, 기본 모델은
**248,997 parameter**다. Temporal convolution은 두 sample에 맞춰 1개 layer로
구성하며 초기 장면 GNN·역할 attention·18D action·Reward v2·좌표계를 유지한다.
Reset 직후 현재 sample만 valid이고 직전 sample·현재 action은 0이다.
현재 sample은 그 관측 직전 적용한 aₜ₋₁, 직전 sample은 aₜ₋₂를 담는다.
기존 16-step checkpoint는 새 모델로 이어 학습하지 않으며 새 checkpoint를 쓴다.

**검증:** CPU 검사 **97개**가 **33.587초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 25, model 10, PPO/runner 6, registration 2다.
이력의 오래된 sample 제거·반복 manager 조회·행동 정렬·전체/부분 reset과
이전 episode padding 차단을 검사했다. Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090의
실제 GPU 실행은 **2환경·64 transition·PPO update 2회**, failure 0회·timeout
2회로 exit code 0을 확인했다. 매 전이에서 마지막 두 sample의 현재 action과
직전 관측을 대조하고, timeout의 true final observation·reset previous invalid,
initial geometry의 EEF/SDK transform, 실제 금지 접촉, actuator·reward dt,
parameter 갱신·GPU checkpoint 복원을 확인했다. Rollout history는 두 환경
모두 **2개**였고 보정 wrench peak는 **1.4059 N / 0.1883 N·m**였다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/previous_step_learning_smoke --validate`다.
`rollout_steps`는 PPO batch 수집 길이를 정한다. 로그는
`outputs/contact_rearrangement/previous_step_learning_validation.log`, checkpoint는
`outputs/contact_rearrangement/previous_step_learning_smoke/model_latest.pt`다.
짧은 통합 검증이며 정책 수렴이나 task 성공률을 입증하지 않는다.

### 2026-10-10 observation만으로 모델 입력 전 mask 생성 — 역할 슬롯 고정 전 기록

사용자가 mask를 observation에 포함한 구현을 교정하고 **observation 정보만으로
생성해야 함**을 명시했다. Actor 관측은 초기 기하 400D·categorical 역할 ID
40D·현재/직전 sample 각 72D를 합친 **584D**로 변경했다. 물체·history valid,
one-hot 역할 mask, tactile valid의 별도 feature 채널을 제거했다.

물체 zero padding의 extent·quaternion, history zero padding의 EEF quaternion,
tactile 0/1과 결측 −1, 외부 역할 ID로 모든 policy mask를 생성한다.
`build_policy_masks`는 simulator나 env를 인자로 받지 않는다. 내부 perception
유효성과 `MdpState.history_valid`는 observation producer의 padding·reset 관리에
사용하며 policy 호출의 mask는 packed observation에서 다시 계산한다.
Reset 직후 현재 sample은 유효하고 이전 sample은 0, 현재 tactile은 −1이다.
Model에 mask를 별도 keyword로 전달하며 node encoder는 기하 10D,
history encoder는 측정 72D만 projection한다. 새 모델은 **247,717 parameter**다.
PPO rollout·minibatch·timeout 마지막 관측 모두 같은 전처리를 사용한다.
이전 700D·1960D checkpoint는 새 규격과 호환되지 않는다.

**검증:** CPU **110개**가 **32.977초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 27, model 20, PPO/runner 7, registration 2다.
Observation만으로 mask 생성, 접촉 OFF와 결측 구분, zero/NaN padding,
reset 이력과 역할 선택, GNN 10D·history 72D projection, 순열 불변성·gradient,
새 checkpoint 복원과 이전 89D sample checkpoint 거부를 검사했다.

Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090의 실제 GPU에서 **2환경·64 transition·
PPO update 2회**, failure 0회·timeout 2회로 exit code 0을 확인했다.
매 전이의 packed observation으로 생성한 mask를 내부 producer 기록과
검증용으로 대조했다. 모델 호출은 같은 observation에서 생성한 mask만 받는다.
Reset 뒤 이전 sample 제거·현재 tactile 결측, timeout 전 마지막 관측의
history mask와 value bootstrap, EEF 좌표계·보상·실제 금지 접촉·parameter
갱신·GPU checkpoint 복원이 통과했다. Rollout wrench peak는 **1.4037 N /
0.1879 N·m**였다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/observation_mask_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/observation_mask_learning_validation.log`,
checkpoint는 `outputs/contact_rearrangement/observation_mask_learning_smoke/model_latest.pt`다.
이 결과는 짧은 통합 검증이며 정책 수렴이나 재배치 성공률을 입증하지 않는다.

### 2026-10-12 역할 ID를 제거하고 Target·Blocker 슬롯 고정

사용자가 물체 역할 ID를 관측에 넣는 대신 Target·Blocker의 관측 위치를
고정하자고 요청했다. Policy 관측에서 Target은 슬롯 0, Blocker는 슬롯 1,
관측된 주변 물체는 슬롯 2부터 기존 상대 순서대로 배치한다. 남는 슬롯과
관측되지 않은 역할 물체의 예약 슬롯은 0으로 채운다. 슬롯 0·1을 주변 물체로
채우거나 missing role을 다른 물체로 대체하지 않는다.

기하는 여전히 승인한 초기 RGB-D의 visible depth로 추정한 EEF-frame OBB이며
현재 물체 GT로 대체하지 않는다. 외부 역할 association은 관측 정렬에 사용하고
정책에는 별도 역할 ID를 전달하지 않는다. 초기 기하 400D와 두 72D sample의
합인 **544D**로 줄인다. `build_policy_masks`는 고정 슬롯과 기하 유효성만으로
Target·Blocker mask를 만들며, 모든 mask는 모델 호출 전에 observation만으로
생성한다. 물체·history padding과 tactile 결측 −1 규칙을 유지한다.

GNN node 입력 10D·시간 입력 72D·18D action·Reward v2·좌표계와
**247,717 parameter**를 유지한다. Policy의 순열 불변성은 역할 슬롯을 고정한
상태에서 주변 물체 슬롯의 순서에 적용한다. Network weight shape가 같아도
서로 다른 관측 규약을 혼동하지 않도록 ModelCfg와 checkpoint에
`observation_layout="fixed_roles_v1"`을 기록하고 이전 규약의 직접 복원을 거부한다.

**검증:** CPU **122개**가 **32.329초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 34, model 23, PPO/runner 9, registration 2다.
환경마다 다른 원본 role 위치의 고정 슬롯 정렬, 입력 메타데이터 무변경,
미관측 역할 슬롯 유지, 주변 물체 순열 불변성, Target·Blocker 교체의 의미 변화,
관측만으로 mask 생성, 새 checkpoint·optimizer 복원과 이전 규약 거부를 확인했다.

Isaac Lab 2.3.2·Isaac Sim 5.1·RTX 3090의 실제 GPU에서 **2환경·64 transition·
PPO update 2회**, failure 0회·timeout 2회로 exit code 0을 확인했다.
기존 canonical 초기 기하와 역할 기록에서 독립적으로 정렬한 base-frame 기준을
만들어 packed 관측의 slot·mask 및 EEF→base SDK 변환을 매 전이에서 대조했다.
Reset 결측·현재/직전 action 정렬, timeout의 true final observation,
실제 금지 접촉·보상·parameter 갱신·GPU checkpoint 복원이 통과했다.
Rollout wrench peak는 **1.4038 N / 0.1879 N·m**였다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/fixed_role_slots_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/fixed_role_slots_learning_validation.log`,
checkpoint는 `outputs/contact_rearrangement/fixed_role_slots_learning_smoke/model_latest.pt`다.
이 검증은 학습 코드 연결과 관측 규약의 확인이며 정책 성능 입증은 아니다.

### 2026-10-12 F/T 자중과 운동 관성 보상

사용자가 F/T의 자중·관성 보상 여부를 확인한 뒤 관성도 제거하도록 요청했다.
기존 policy F/T는 자중만 제거하고 운동 관성을 남겼다. 이제 센서 손쪽 원통
절반과 hand를 포함한 **33개 body**에 대해 자체 Newton–Euler 반력을 계산한다.
링크별 질량 `m_i`, world COM 가속도 `a_i`, 각가속도 `α_i`, 각속도 `ω_i`,
센서 중앙에서 COM까지의 world 벡터 `r_i`로 다음을 계산한다.

$$
F_{self}=\sum_i m_i(a_i-g),\qquad
\tau_{self}=\sum_i\left[r_i\times m_i(a_i-g)
    + I_i^w\alpha_i + \omega_i\times(I_i^w\omega_i)\right].
$$

World에서 합산한 힘·토크를 현재 sensor frame으로 변환해 raw에서 빼고
선택적으로 residual bias를 제거한다. 이로써 무접촉 운동의 자중·관성은 제거하고
외력·접촉의 반력을 raw parent→child 부호로 유지한다. 원통 distal 링크의
actor origin인 센서 중앙을 torque 기준점으로 삼으며 센서 COM과 구분한다.

PhysX `get_inertias()`는 COM 기준 actor/link-frame tensor를 column-major로
반환하므로 reshape·transpose해 저장하고 **link quaternion**으로 회전한다.
COM principal-axis quaternion으로 다시 회전하지 않는다. Native
`body_com_acc_w`와 `body_com_ang_vel_w`는 raw 반력과 동일한 마지막 완료
physics step의 world COM 운동을 읽는다. 손가락의 독립 운동과 COM 구심 가속도도
포함하며 policy 30 Hz 속도 차분으로 근사하지 않는다.

Invalid/reset F/T는 motion을 평가하지 않고 0을 반환한다. 관성용 temporal cache를
추가하지 않아 부분 reset의 이전 운동이 이어지지 않는다. `tare_compensated_wrench`
는 전체 자체 반력을 제거한 뒤의 residual만 기록하며 자동 tare는 하지 않는다.
Raw sensor·물리 force·contact·544D observation 구조·센서 좌표계는 유지한다.
보상·F/T 실패 판정에도 같은 `corrected_wrench`를 사용한다.
시뮬레이션에서는 native 질량·관성·가속도를 사용하며 실기는 calibrated 동역학
모델과 관절/IMU 기반 운동 추정·동기화가 필요하다.

**검증:** 현재 소스로 전체 CPU **129개**가 **32.058초**에 통과했다.
Sampling 32, action/contact/control 22, MDP 41, model 23, PPO/runner 9,
registration 2다. 신규 7개 회귀 검사에서 병진 관성과 센서-center lever moment,
비등방 회전 관성, 일정 각속도의 gyro, 독립적으로 움직이는 링크와 COM
principal-axis/actor-frame 구분, reset stale NaN·부분 reset 격리·residual tare를 확인했다.

별도 실제 GPU 센서 검사는 **2환경·151 environment step + 2 environment step +
19 bench physics step**으로 완료했다. Wrist ±0.02 rad와 index +0.03 rad의
무접촉 운동에서 자중만 제거한 피크 **1.2007 N / 0.140091 N·m**가 전체 보상 후
**4.44494e−6 N / 0.000312104 N·m**로 줄었다. 알려진 각 축 3 N 힘·0.10 N·m
토크의 외력 반력 보존 오차는 최대 **0.012221 N / 0.000850212 N·m**였다.
독립 SciPy Newton–Euler raw 반력 대조와 반복 읽기의 물리 상태 무변경,
초기 vision snapshot 보존·전체/부분 reset도 통과했다. 임시 bench의 force·joint
target·MDP 상태·sensor cache를 복구하고 최종 정상 reset을 확인했다.
로그는 `outputs/contact_rearrangement/inertia_sensor_validation.log`이며
`check_sensors.py --num_envs 2 --seed 0`으로 실행했다.

학습 환경 GPU에서는 **2환경·64 transition·PPO update 2회**, failure 0회·timeout
2회로 exit code 0과 `LEARNING_VALIDATED`를 확인했다. Observation은 **544D**,
모델은 **247,717 parameter**를 유지했다. Hold·실제 spacer 금지 접촉,
정렬된 EEF 물체 기하·두 sample 정렬·reset·timeout 마지막 관측·reward 일관성,
parameter 갱신과 GPU checkpoint 복원이 통과했다. Rollout의 보상된 wrench 피크는
**4.90412e−6 N / 0.00239641 N·m**였다. 마지막 자세·속도/가속도와 physics-step
평균 반력을 사용하는 이산화 차이 등의 작은 torque 잔차는 남는다.
실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/inertia_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/inertia_learning_validation.log`, checkpoint는
`outputs/contact_rearrangement/inertia_learning_smoke/model_latest.pt`다.
이는 시뮬레이션 보상·학습 연결 검증이며 실기 보상 정확도나 정책 성능의 입증은 아니다.

### 2026-10-12 Motion reward의 로봇 관절 기준 교정 — 속도·절대 하중 항 제거 전 기록

사용자가 Motion 항은 물체 운동보다 로봇 joint 운동을 평가해야 한다고 교정했다.
`reward_components`의 velocity/acceleration 입력을 object velocity에서 native
**UR5e 6 + hand 독립 6관절의 q̇**로 바꿨다. 물체가 정지해도 로봇이 움직이면
Motion 비용이 발생할 수 있으며, 로봇이 정지한 채 물체가 움직이면 이 비용이
바뀌지 않는다. 물체의 누적 XY 이동은 Rearrangement, 물체의 정착 속도는
성공/안정성 판정에서 계속 평가한다. Mimic 관절은 중복 비용에 포함하지 않는다.

Velocity 비용은 각 joint의 속도 절댓값에서 soft 기준을 뺀 양수 부분을 ref로
정규화·제곱한 12관절 합이다. Acceleration 비용은 직전 q̇와 현재 q̇를 control
dt로 차분한 값을 ref로 정규화·제곱한 합이다. 초기 soft/ref는 arm·hand 모두
**0.5/1 rad/s**, acceleration ref는 **arm 5·hand 10 rad/s²**다.
기존 자중·관성 보상 F/T load/rate 및 `mean(exp(−costs))`의 네 항을 유지한다.
선택 reset에서만 `previous_joint_velocity`를 실제 초기 q̇로 초기화해 이전
episode의 움직임을 acceleration 비용으로 이어 붙이지 않는다.

`extras.motion_costs`는 검증·진단용이며 policy observation에 추가하지 않는다.
Auto-reset된 전이에서도 reset 전 값을 보존한다. GPU 검증은 마지막 관측과
직전 관측의 q̇를 rad/s로 복원해 속도·가속도 비용을 독립 계산하고
`mean(exp(−costs))` 및 native continuing joint velocity를 대조한다.

**검증:** CPU **136개**가 **32.501초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 48, model 23, PPO/runner 9, registration 2다.
신규 검사는 로봇 운동/물체 정지와 로봇 정지/물체 이동의 비용 분리,
관절 부호·arm/hand scale, control dt에 따른 acceleration 제곱 비용,
mimic 제외·반복 manager 조회·부분 reset의 실제 초기 q̇를 포함한다.

실제 GPU의 **2환경·64 transition·PPO update 2회**, failure 0회·timeout 2회가
exit code 0과 `LEARNING_VALIDATED`로 통과했다. 매 전이의 native q̇와 관측 이력을
비교하고, auto-reset 전 속도·가속도 비용과 Motion 점수를 독립 복원했다.
544D observation·247,717 parameter·관성 보상 F/T·단계별 action/geometry,
hold·실제 금지 접촉·terminal bootstrap·checkpoint 복원도 통과했다.
Rollout별 평균 `[velocity,acceleration,load,rate]` 비용은 각각
`[0.006463,11.19205,0,0.00001095]`,
`[0.001345,13.84008,0,0.00001990]`였다. 이 무작위 초기 정책에서는 가속도 비용이
큰 편이며 현재 ref의 최종 적합성이나 학습 성능을 입증하지 않는다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/joint_motion_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/joint_motion_learning_validation.log`,
checkpoint는 `outputs/contact_rearrangement/joint_motion_learning_smoke/model_latest.pt`다.

### Tactile의 역할과 후속 검토 — 새 동작은 미채택

사용자는 tactile에 대해 고민하자고 요청했다. 현재 **17개 binary pad**는
observation으로 유지하며 tactile reward·새 observation 채널·sampling 규칙을
추가한 것으로 기록하지 않는다. 각 pad의 실제 net normal force norm >0.01 N을
접촉으로 판단하고 미측정은 −1로 구분한다. F/T는 손 전체의 합력·합모멘트,
tactile은 어느 손 부위가 닿았는지를 알려준다. 서로 다른 pad의 접촉 힘이
상쇄돼 F/T가 작아져도 tactile의 공간 패턴은 남을 수 있다. 한 pad 안의 여러
접촉이 net force에서 상쇄되면 binary 자체가 OFF가 될 수 있으므로 접촉 개수를
정확히 센다고 해석하지 않는다.

Binary tactile만으로 압력 크기·마찰·slip 또는 접촉 상대가 Blocker/Target/선반인지
판별할 수 없다. 현재·직전 sample에서 접촉 발생·해제·손가락별 패턴 변화는
유도할 수 있으며 별도 16-step history가 필요하다고 결정하지 않는다.
Current HistoryEncoder는 고정 pad 순서의 17채널을 입력으로 받으므로 pad identity를
이미 보존한다. 별도 tactile branch/공간 encoder는 실험 후보이지 현재 필수 변경이 아니다.

먼저 검토할 것은 **측정한 접촉을 정책이 놓치지 않는가**다. ContactSensor는
120 Hz이고 `history_length=0`, policy는 30 Hz로 마지막 substep만 읽는다.
앞의 세 substep에서 발생했다가 해제된 짧은 접촉은 현재/직전 관측 모두에서
사라질 수 있다. 실제 밀기·pivoting 로그에서 누락과 threshold 경계 ON/OFF를
측정한 뒤 구간 OR, onset/해제 event 보존 또는 hysteresis를 선택할 수 있다.
구간 OR는 ‘현재 접촉’에서 ‘지난 control 구간에 접촉했음’으로 의미가 바뀌며
관측 규약·reset·mask·학습에 같은 기준을 적용해야 한다. 아직 구현/채택하지 않았다.

접촉 수·새 접촉·접촉 유지 자체의 reward는 현재 연구 목표의 대리 지표로
단정하지 않는다. 많거나 잦은 접촉을 보상하면 불필요한 파지·주변 물체/선반
접촉·반복 접촉을 유도할 수 있다. 항상 동일 접촉을 유지하도록 하거나 접촉
변화에 벌점을 주면 필요한 pivoting·접촉점 전환을 방해할 수 있다. 우선 접근
공간 확보·재배치 최소화·robot 가감속·F/T 증가 안정성을 reward로 두고 tactile을
상호작용 피드백으로 활용하는 것을 권장한다. 새 tactile shaping은 실제 실패
유형과 task progress를 확인해 구체화할 미결 사항이다.

### 2026-10-12 급가감속과 F/T 크기 증가만 평가하는 Motion

사용자가 “빠른 속도·큰 힘을 막으려는 것이 아니라 급격한 가속·힘 증가를
막고 싶다”고 교정했다. 앞선 velocity/load 항은 목적과 달라 제거했으며,
속도 0.5 rad/s soft 문턱·velocity ref·force/torque soft와 절대 크기 ref도 삭제했다.
로봇 12개 독립 관절의 acceleration과 F/T 증가율만 비용으로 계산하고
`(exp(−C_acc)+exp(−C_inc))/2`로 평균한다. 삭제한 항을 비용 0으로 남겨 네 항의
평균을 취하지 않는다. Observation 544D·action 18D·모델·tactile은 유지한다.

힘을 빠르게 빼는 경우의 처리에 대해 사용자는 **힘·토크 크기의 급격한 증가만
비용 부여**를 선택했다. 증가율은 signed component 차이가 아니라
`max(||F_t||−||F_previous||,0)/dt`와 같은 torque 식이다. 음수 반력의 크기 증가도
비용에 포함하고 감소·동일 크기의 방향 변화는 제외한다. Frame·torque 기준점은
기존 world 축·episode 시작 sensor center를 유지한다. Acceleration은 관절 속도
차분이며 급가속과 급감속 모두 평가한다. Jerk를 추가한 것으로 해석하지 않는다.

일정한 높은 속도·일정한 큰 하중에서는 Motion 비용이 0일 수 있다. 80 N·4 N·m
F/T 실패와 actuator 제한·물체 안정 종료 기준은 별도의 물리/안전 조건으로 유지한다.
비용 정규화 scale은 arm acceleration 5·hand 10 rad/s², force 증가율 200 N/s,
torque 증가율 10 N·m/s의 기존 초기값이다. 정규화 scale은 soft 문턱이 아니다.

`motion_costs`는 `[joint_acceleration,wrench_increase_rate]`의 **2D** 진단값이며
policy 입력에 넣지 않는다. 비용 순서를 config·metrics에 함께 저장한다.
Reset 전 `reward_wrench`를 진단 extras로 보존하고 직전 world wrench와 비교해
timeout 자동 reset 후에도 해당 전이의 force/torque 증가 비용을 재검증한다.
기존 544D checkpoint의 network 구조는 호환되지만 critic의 학습 reward는
이전 규약일 수 있으므로 새 run의 reward metadata와 구분한다.

**검증:** CPU **141개**가 **32.031초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 53, model 23, PPO/runner 9, registration 2다.
일정한 고속·큰 하중에서 Motion 비용 0, 관절 가속도·힘/토크 증가율의 dt²
스케일링, 음수 반력의 norm 증가, 빠른 하중 감소·같은 크기의 방향 변화 제외,
두 항 평균·삭제된 soft/ref 설정·별도 하드 하중 실패와 reset 격리를 검사했다.

실제 GPU의 **2환경·64 transition·PPO update 2회**, failure 0회·timeout 2회가
exit code 0과 `LEARNING_VALIDATED`로 완료했다. 두 비용을 매 전이에서
이전 joint 관측·이전 world wrench·reset 전 진단값으로 독립 재계산했으며,
timeout에서도 phase가 섞이지 않는 것을 확인했다. Hold·실제 금지 접촉·EEF
기하/역할/이력·reward·parameter 갱신·GPU checkpoint 복원이 통과했다.
Observation 544D·action 18D·247,717 parameter를 유지한다.
평균 `[joint_acceleration,wrench_increase_rate]` 비용은 rollout별로
`[11.19205,0.0000032374]`, `[13.84075,0.0000031699]`였다.
정규화 ref의 적합성과 정책 수렴은 이 짧은 검증에서 입증하지 않는다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 16 --episode_length_s 0.8
--log_dir outputs/contact_rearrangement/smooth_motion_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/smooth_motion_learning_validation.log`,
checkpoint는 `outputs/contact_rearrangement/smooth_motion_learning_smoke/model_latest.pt`다.

### 2026-10-12 지수형 시간 비용과 환경별 clock

사용자가 Time을 기존 선형 비용에서 **시간이 지날수록 penalty와 증가폭이
함께 커지는 형태**로 바꾸도록 교정했다. `T_ref`와 무관한 함수 개형을 요청하고,
지수형과 초반 일정 비용 이후의 구간 제곱형 중 **지수형을 명시적으로 선택**했다.
최종 식은 `c(t)=a(exp(bt)−1)`의 해당 control 구간 적분을 차감하는 것이다.
4절의 수식·개형·수치는 이 구현을 가리킨다. `a=0.02 reward/s`, `b=0.7 /s`는
조절 가능한 개발 초기값이며 사용자가 정한 숫자는 아니다.

`MdpCfg.time_ref`를 제거하고 `time_penalty_scale`, `time_penalty_growth_rate`를
추가했다. Scale은 비음수이고 growth rate는 양수이며 둘 다 유한해야 한다.
Scale 0은 시간 비용을 끈다. Float64 `MdpState.elapsed_time`은 완료된 control
transition마다 한 번 증가하고 선택된 환경 reset 시에만 0으로 돌아간다.
기존 timestamp guard가 여러 manager의 반복 조회를 막는다. `reward_components`
는 transition 직전 경과 시간을 keyword 인자로 받으며 다른 reward 수식은 유지한다.

환경의 `episode_elapsed_time` extras는 종료 직전 clock을 보존한다. Training의
`--validate`는 step 전 SDK episode counter로 완료 시각을 독립 계산하고 Time
적분과 continuing/reset clock을 매 전이에서 검사한다. Config·checkpoint에
수식과 두 파라미터를 기록하고 metrics에 평균 Time component·최대 경과 시간을
남긴다. 시간은 actor·mask 입력으로 전달하지 않는다. Network 구조는 호환되지만
이전 checkpoint의 critic reward 규약과 새 run의 metadata를 구분한다.

**검증:** CPU **151개**가 **33.297초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 63, model 23, PPO/runner 9, registration 2다.
수치 적분과의 일치, 초반의 작은 비용·후반의 증가폭, 작은 dt의 수치 안정성,
15/30/60 Hz에서 8초 누적 비용 **−7.537897**의 일치, zero scale,
부분 reset·혼합 timestamp·중복 manager 조회·simulator time gap을 검사했다.

실제 GPU의 **2환경·160 transition·PPO update 2회**가 exit code 0과
`LEARNING_VALIDATED`로 완료했다. 학습 rollout의 failure 0회·timeout 2회,
최대 episode 경과 시간 **2.4초**였다. 평균 Time component는 rollout별로
**−0.000435456**, **−0.001496055**였다. Hold·실제 금지 접촉 fixture,
EEF 기하·역할·이력·reward 검증, parameter 갱신과 checkpoint 복원도 통과했다.
기존 544D observation·18D action·247,717 parameter·terminal +500/−100을 유지한다.
짧은 smoke는 8초까지의 실제 정책 실행이나 reward 균형·정책 성능을 입증한 것은 아니다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 40 --episode_length_s 2.4
--log_dir outputs/contact_rearrangement/exponential_time_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/exponential_time_learning_smoke/validation.log`,
checkpoint는 같은 디렉터리의 `model_latest.pt`, 개형은
[time_penalty.png](figures/time_penalty.png)다. `py_compile`도 통과했다.

### 2026-10-12 공통 지수형 품질 점수와 OBB 조작량

사용자가 Clearance·Rearrangement·Motion에 `exp(−a·x)` 공통 형태를 적용하고
Time은 앞서 선택한 증가형 penalty를 유지하도록 명시적으로 선택했다.
`clearance_decay=1`, `rearrangement_decay=2 m⁻¹`,
`motion_acceleration_decay=1`, `motion_wrench_increase_decay=1`을 추가했다.
`displacement_ref`를 제거했으며 Rearrangement의 초기 민감도는 이전 `1/0.5 m`와
같다. 다른 항과의 상위 weight 4개는 모두 1이다. 이 초기값의 학습 적합성을
공통 함수 형태만으로 검증한 것은 아니다.

Rearrangement의 x는 매 step 모든 active 물체의 **평균 8-corner 3D 이동거리**를
episode 동안 합산한 값이다. Target·Blocker·초기 미인식 물체를 포함하고 물체 간
평균이나 크기 정규화는 하지 않는다. Fixed USD bounds·corner 순서·environment
frame을 유지하며 병진·회전·pivot·왕복·Z 이동을 평가한다. 승인된 reset pose로
`previous_corners`를 초기화하고 선택된 환경만 누적량을 초기화한다.
`rearrangement_increment`, `path_length`, `previous_corners`를 같은 completed
transition에서 한 번 갱신한다. 현재 reward는 **누적 조작량의 품질 점수**라 과거
이동이 이후 step의 점수에도 영향을 준다. 앞선 직접 음수 증분 비용은 채택하지 않았다.

환경 extras에 종료 전이의 corner·step 증가량·누적량을 보존한다. Train 검증은
native pose와 SDK quaternion 변환으로 corner를 독립 재구성하고 증가량·누적량·
지수 점수와 reset 이후 기준점을 비교한다. 해당 GT는 모델 입력이나 mask 생성에
사용하지 않는다. Config·checkpoint에 공통 수식과 민감도, metrics에 평균 step
조작량과 최대 누적 조작량을 저장한다. Observation 544D·action 18D·247,717
parameter·기존 Time·terminal +500/−100을 유지한다. 보조 인식 head는 미도입이다.

**검증:** CPU **161개**가 **33.162초**에 통과했다. Sampling 32,
action/contact/control 22, MDP 73, model 23, PPO/runner 9, registration 2다.
민감도별 지수 계산·invalid 계수, 순수 병진·제자리 회전·회전 왕복·quaternion
부호·SciPy 변환을 이용한 pivot 교차 검증·Z 이동·미인식 물체 포함·비활성 slot
제외·부분 reset·반복 manager 조회를 검사했다.

실제 GPU **2환경·160 transition·PPO update 2회**가 exit code 0과
`LEARNING_VALIDATED`로 완료했다. 학습 rollout의 failure 0회·timeout 2회였다.
평균 step 조작량은 rollout별 **0.000201431 m**, **0.000134409 m**이며 최대
누적 조작량은 **0.00896450 m**, **0.00984458 m**였다. 매 전이의 corner·누적량·
Motion 비용·Time·reward 검증, hold·실제 금지 접촉 fixture·parameter 갱신과
checkpoint 복원이 통과했다. 이 짧은 검증은 reward 균형·정책 재배치 성공·숨은
물체 변화 인식 성능을 입증한 결과가 아니다.

실행은 `train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless
--num_envs 2 --iterations 2 --rollout_steps 40 --episode_length_s 2.4
--log_dir outputs/contact_rearrangement/common_exp_obb_learning_smoke --validate`다.
로그는 `outputs/contact_rearrangement/common_exp_obb_learning_smoke/validation.log`,
checkpoint는 같은 디렉터리의 `model_latest.pt`다. `py_compile`도 통과했다.

### 2026-10-12 SI-200-8 측정 범위·분해능과 보상 경로

사진의 calibration SI-200-8과 ATI 공식 사양을 확인해 범위·유효 분해능을
`ForceTorqueCalibration`에 설정했다. Simulator-independent profile을 기존
`src/catalog.py`에 두고 측정 reader·센서 생성·두 환경 설정에 연결한다.
생산 코드 모듈을 새로 세분화하지 않는다. ToolForceTorqueSensor는 이상적인
PhysX 반력을 보존하고 별도의 raw 측정 cache에 saturation·rounding을 적용한다.
Hardware axis range는 gravity/inertia 보상 전 전체 하중에 적용한다.

정책은 보상한 wrench를 `[200,200,360,8,8,8]`로 축별 정규화한다. 기존 scalar
50 N/5 N·m 설정은 제거하고 544D/72D sample·frame·tactile/mask 규약은 유지한다.
Fresh sample의 raw range 초과를 hard-load 실패에 추가하고 기존 compensated
norm80 N/4 N·m 및 Motion 증가율 ref200 N/s/10 N·m/s는 유지한다.
종료 이전 raw·ideal·range flag·보상값을 보존해 reset과 섞이지 않게 한다.
Checkpoint의 F/T profile 누락·불일치 시 training resume를 거절한다.

**CPU 검증:** **179개**가 **33.089초**에 통과했다. Sampling32,
action/contact/control22, MDP79, model23, PPO/runner12, registration2,
calibration9다. Axis별 범위·분해능·경계·invalid profile, Fz360 N 정규화,
quantization 후 compensation 순서, raw 범위 초과가 보상으로 숨지 않는 판정,
invalid cache 무시, 기존 exact Newton–Euler 보상 수학과 profile 호환성을 검사했다.

**GPU 센서 검증:** 실제 **2환경·151 public step + 2 env step + 19 bench physics
step**, reset6회·partial reset2회·capture6회·render98회가 exit code0과
`SENSOR AUDIT: PASS`로 완료했다. 33개 downstream link·0.876663 kg를 유지하며
실제 force3 N·torque0.10 N·m를 각 축에 적용했다. Ideal load balance 오차는
최대 **0.0122202 N/0.000850184 N·m**, 측정·보상 후 external 오차는
**0.0787559 N/0.00359748 N·m**였다. 양자화 오차를 분리한 보상 오차는
**0.012222 N/0.000850259 N·m**다. Axis별 최대 양자화 오차는 힘
**0.0471473/0.0484791/0.0490457 N**, 토크
**0.00247812/0.00243416/0.00240188 N·m**로 각각 half-resolution 이하였다.
무접촉 운동의 gravity-only peak **1.17404 N/0.14016 N·m**는 full compensation
후 **0.0787559 N/0.00359748 N·m**였다. 이전 이상적 sensor의 약1e−6 N 잔차와
비교할 때 새 유효 분해능 근사에 따른 잔차를 구분한다. Controlled GPU tensor
fixture로 per-axis saturation·range flag·tare·partial reset도 검사했다. 큰 물리
과부하를 실제 articulation에 가한 시험으로 기록하지 않는다.

**GPU 학습 검증:** **2환경·160 transition·PPO update2회**, failure0·timeout2회가
exit code0과 `LEARNING_VALIDATED`로 완료했다. 매 전이 axis별 raw 측정·범위
flag·보상값의 policy normalization, 기존 reward·OBB·Time·reset과 새 profile
checkpoint 복원을 검사했다. Rollout의 range-exceeded counts는 모두0이었다.
Compensated wrench peak는 rollout별 **0.0736205 N/0.00434070 N·m**,
**0.0748436 N/0.00485677 N·m**였다. 센서 측정 근사와 데이터 경로의 검증이며
실제 하드웨어 정확도·sampling bandwidth·정책 성공률을 입증한 결과는 아니다.

센서 명령은 `tests/check_sensors.py --num_envs 2 --seed 0`, 로그는
`outputs/contact_rearrangement/si2008_sensor_validation.log`다. 학습 명령은
`scripts/train.py --task Isaac-Contact-Rearrangement-UR5e-v0 --headless --num_envs 2
--iterations 2 --rollout_steps 40 --episode_length_s 2.4
--log_dir outputs/contact_rearrangement/si2008_learning_smoke --validate`이며 로그와
checkpoint는 같은 디렉터리의 `validation.log`·`model_latest.pt`다.
`py_compile`도 통과했다.

### 2026-10-13 예제 기준 native 학습·환경 흐름 정리

사용자는 학습·환경 생성 전반을 Isaac Lab과 `examples`의 예제 흐름으로 따르고,
그 안에 연구에 필요한 수정만 두도록 요청했다. `train.py`의 직접 rollout 수집,
GAE와 학습 반복을 제거하고 native `OnPolicyRunner.learn`에 위임했다.
등록은 `rsl_rl_cfg_entry_point`, 설정은 `RslRlOnPolicyRunnerCfg`와
`RslRlPpoAlgorithmCfg`를 사용하며 YAML·TensorBoard·`model_<iteration>.pt`를 저장한다.
Native value loss 계수 0.25는 이전 구현의 0.5×0.5 영향력을 유지한다.
Shared GNN 모델·현재/직전 관측·observation에서 mask 생성·18D tanh Gaussian은 그대로다.
Native PPO 확장은 shared parameter의 단일 등록, 실제 final observation timeout 값,
32-sample gradient accumulation 및 checkpoint 관측/F/T profile 검사로 한정한다.
Native storage와 GAE는 재구현하지 않는다. 기존 custom trainer checkpoint는 새
native 포맷으로 직접 읽지 않는다.

환경은 기존 SceneCfg→robot별 EnvCfg→ManagerBasedRLEnv→manager/reset event 흐름을
보존하고 보상을 5개 기본 RewardTerm으로 연결한다. Term weight가 권위 설정이며
MDP diagnostics는 이를 반영한다. 센서의 동적 class/globals 등록과 빈 manager config는
제거한다. Hydra가 수정할 parameter 설정은 mutable로 두고 실제 F/T calibration은
immutable 객체로 변환해 검증한다. 로봇·물체·스폰·카메라·F/T·tactile 조건은 유지한다.

검증은 CPU **194개**와 실제 GPU **2환경·160 transition·2 PPO update**가 통과했다.
Hold·실제 금지 접촉·EEF/frame·SI-200-8·history·reward 합·timeout·native checkpoint 복원을
확인했다. 로그는 `outputs/contact_rearrangement/native_flow_validation.log`,
checkpoint는 `outputs/contact_rearrangement/native_flow_validation/model_1.pt`다.
이는 학습 연결 검사이며 수렴/성공률을 입증하지 않는다. 이전 확장 실행에서는
128개 장면 구성의 높은 host RAM이 남아 메모리 guard로 중단했다. **1024개 이상
환경 지원은 미완료**이며 이번 구조 변경의 통과와 구분한다.

## 9. 다음 구체화와 구현에서 정할 사항

아래는 미결 사항이며 다음 작업의 순서가 확정된 것은 아니다.

1. 실제 학습 분포 구체화. 종류별 채택률·실패율, settle 이후 기하 조건 유지, 밀집도 수치와 sampler bias, 로봇의 실제 접촉·접근 가능성을 확인한다.
2. 구현한 tactile 17채널과 raw F/T의 실기 threshold·filtering·동기화 교정, F/T 자중·관성·bias 보상에 필요한 질량·관성·가속도 추정 교정.
   Tactile의 짧은 substep 접촉 누락·event 보존·hysteresis와 직접 reward의 행동 유인도 별도로 검토한다. 모두 미채택이며 17채널 binary 관측 명세를 바꾼 것으로 해석하지 않는다.
3. Base 기준 18D OSC/hand 제어의 gain·workspace·effort limit와 접촉 안정성 검증, 실기 제어 경로.
4. Visible-depth 초기 PCA OBB의 가림·부분 형상 오차, oracle instance-ID association을 대체할 실기 perception, 현재/직전 두 sample에 남지 않는 정보 손실 평가.
5. 구현한 전체 차단·Target 보호·비파지 조건·성공/실패·timeout과 arm/spacer 금지 접촉 기본값의 task 적합성.
6. Reward v2의 초기 수치 조정, 대기 보상 악용과 안정성/속도 trade-off 검증.
7. 구현한 GNN·attention·finite temporal encoder·shared actor/critic·PPO의 학습 수렴과 일반화, 필요 시 구조 변경.
8. 환경 변화 이해를 무엇으로 평가할지와 task reward만의 baseline, 추가 reward/보조 loss 비교의 채택 여부.

연구 논의를 이어갈 핵심 사례는 같은 초기 OBB와 같은 명령에 대해 마찰·접촉
차이로 Blocker 이동량이 달라진 두 episode다. 이력에 어떤 차이가 남는지,
그 차이가 다음 행동을 바꿔야 하는지, 이를 기억·평가 label·reward/loss 중
무엇이 담당하는지 구체화한다. 모든 숨은 상태의 정확한 복원이 목표로 확정된 것은 아니다.

## 10. 이 맥락을 갱신하는 방법

개발 진행, 중요한 연구 결정·사용자 교정·검증 결과가 생기면 이 문서에 반영한다.
대화 전문을 누적하거나 질문마다 별도 보고서로 분리하지 않는다.

- **현재 상태부터 수정:** 1절과 해당 설계·구현 절을 먼저 갱신한다. 과거 문장을 그대로 두고 로그만 추가하지 않는다.
- **이유와 이전 상태 보존:** 무엇을 선택했고 왜 바꿨는지, 이전 선택이 폐기·대체·보류 중 무엇인지 간결하게 남긴다.
- **제안의 상태 유지:** 승인 전에는 미채택으로 두고 사용자 확인 질문을 도입 승인으로 승격하지 않는다.
- **검증 근거 기록:** 날짜, 코드/명령, 버전·환경·asset, 검사 범위와 결과를 남긴다. 미실행 계획을 성공 결과로 기록하지 않는다.
- **주기와 상태 일관성 확인:** Scene, action/observation, reward, termination, reset 간 전제를 확인한다. 복제 환경의 부분 reset과 actor 입력의 GT 유출도 해당 구현에서 확인한다.
- **세부 문서 연결:** 실행·asset 설정은 패키지 README, 큰 MDP/실험 명세는 관련 세부 문서로 연결하고 여기에는 현재 판단을 복원할 핵심을 유지한다.
- **출처 충돌 처리:** 나중의 명시적 사용자 변경을 우선하고 연구 작업안·실제 코드·과거 자료의 차이를 적는다. Notion과 로컬 문서의 동기화는 별도 요청 범위에 따라 수행한다.

## 11. 기준 자료

- [Road to 졸업](https://app.notion.com/p/39e4968094ee8088b12af2849757fa73): 프로젝트 상위 페이지.
- [연구 맥락 — 생각의 흐름과 현재 설계](https://app.notion.com/p/3f34968094ee81ca8eb6feb1f700e0f9): 설계 변경 이유·사용자 교정·미채택 제안의 기준.
- [학습 정리](https://app.notion.com/p/3f24968094ee80c6a0d4f15c94c5e081): Framework·Observation·Reward v2·Action 정리. 초기 vision/history 전환 이전 문구는 최신 맥락으로 읽는다.
- [Reward 설계 근거](https://app.notion.com/p/3f24968094ee819fb1a4df6ff4bdc9a1): 원문 수식·우리 적용·선택 이유·대안. Potential-based shaping은 미채택 대안이다.
- [Track B](https://app.notion.com/p/3da4968094ee804d8f5afeeb17ba5279), [Task BrainStorming](https://app.notion.com/p/3d14968094ee80448d6fe2f3189b8b17): 이전 구상의 역사 자료.
- [새 환경 README](../src/contact_rearrangement/README.md): 현재 실행·asset 경로·구현 위치.

Notion에 언급된 이전 대화 전체·발표 원본·`docs/연구논의/` 등 모든 자료가
이 checkout에 존재하거나 이번 작업에서 직접 확인된 것은 아니다. 필요하면 원문을
확인하고, 이 문서에 보존된 요약을 전체 transcript 확인으로 확대하지 않는다.

### 2026-10-13 새 프로젝트: 렌더링 없는 초기 기하 관측

대상은 `src/contact_manipulation`이다. 기존 `src/contact_rearrangement`는 변경하지 않는다.
사용자는 카메라·렌더링 메모리 비용을 없애고, 각 reset 시점에 보일 것으로 예상되는 정보만
초기 observation으로 구성하는 방향을 제안했다. 기존 카메라는 reset마다 생성한 것이 아니라
고정된 tiled pool을 재사용했지만, 실제 측정에서 첫 렌더 초기화의 메모리 증가가 컸다.

새 프로젝트의 camera/render product/annotator/실루엣 참조 객체를 제거하고, Isaac Lab
RayCaster와 같은 Warp mesh query 방식으로 대체했다. 사용자는 컵 손잡이·빈 공간을 반영하고
보이는 표면에서만 observation을 추정하는 **실제 메시 기반 ray casting**을 선택해 확정했다.
물체 종류별·로봇 링크별·
선반의 표면 BVH를 공유하고 각 환경의 현재 pose만 적용한다. 매 episode의 독립 배치,
Target/Blocker 선정과 앞쪽 시점·방향 randomization은 유지한다.

- 기본 격자 320×240, 동시에 처리하는 가상 시점 상한 8. 환경 수만큼 카메라나 메시를 만들지 않는다.
- Target ≥40%, Blocker ≥95% 조건은 같은 자세의 비가림 실루엣에 대한 **광선 표본 비율**로 계산한다.
  넓힌 2배 격자의 고립 실루엣을 분모로 쓰고 시야 밖으로 잘린 role 물체는 거절한다.
- 물체·로봇·선반 중 가장 가까운 표면만 보인다. 보이지 않는 물체는 observation에서 제외하며,
  물체 전체 pose/크기를 직접 제공하지 않는다. 보이는 교차점의 PCA·extrema로 upright OBB를 추정한다.
- 기하 추정은 tensor scatter로 처리한다. 초기 OBB를 episode 동안 고정하고 EEF frame으로
  재표현하는 기존 544D observation, 고정 Target/Blocker 슬롯, mask 생성 규약은 유지한다.
- 광학적 렌더링과 동일하지 않다. 불투명 표면·이상적 물체 association을 가정하며 투명/반사 재질,
  조명, 검출 실패와 depth noise를 재현하지 않는다. 해상도 및 primitive tessellation 오차도 남는다.
- Headless 학습은 카메라 기능을 강제로 켜지 않으며 reset render 호출은 0이다. GUI/video는
  사용자가 요청한 표시/녹화용 렌더링을 별도로 사용한다.

실제 32개 환경에서 초기 reset은 21.86→9.34초, 16개 부분 reset은 9.73→2.35초였다.
프로세스 최고 RSS는 39,223→20,309 MiB, GPU 전체 사용량은 8,363→3,990 MiB였다.
이는 각각 1회 통합 측정이며 JIT·IK 준비·재시도까지 포함한다. 부분 reset의 미선택 상태와
physics 시간 보존, 새 초기 관측 및 가시율 조건을 통과했다. 장면·물리 복제 메모리는 여전히
남으므로 이를 1,024개 실제 물리 환경의 생성/학습 검증으로 해석하지 않는다.

### 2026-10-13 새 프로젝트: 장면 복제 메모리와 UR5e 표시 수정

사용자가 1,024개 headless 실행의 종료와 4개 GUI 환경에서 UR5e가 하나만 보이는 현상을
보고했다. 커널 OOM 기록상 15:02:12에 해당 Python이 host RAM 부족으로 종료됐으며,
anonymous RSS는 40,233,996 KiB(약 38.37 GiB)였다. 초기 reset 이전 장면 복제 단계로,
함께 출력된 Replicator semantic-label 경고가 종료 원인은 아니었다.

`src/contact_manipulation`의 scene 마지막 spawner가 완성된 rigid body subtree를
임시 USD에 저장하고 instance로 공유하도록 했다. 기존 nested geometry instance는
flatten 전에 펼치며 외부 재질 binding도 함께 보존한다. 환경별 물리 body·관절·pose는
독립적으로 유지하고 원본 에셋은 수정하지 않는다. Native USD cloner와 physics replication을
사용하며 `clone_in_fabric=False`로 모든 환경을 Stage tree에서도 확인할 수 있다.
필요 없는 semantic label 작성은 제거했다. ManagerBasedRLEnv·학습 runner는 그대로 사용한다.

실제 32개 환경의 최고 RSS는 20,309→5,002 MiB로 감소했고 초기 가시율 결과가 기존과
일치했다. 128개 환경은 최고 RSS 5,342 MiB, GPU 전체 사용량 4,340 MiB였다.
32/128개에서 4 control step 및 절반 환경 reset 시 미선택 환경 보존이 통과했다.
4개 환경의 실제 렌더에서 UR5e·hand가 모두 표시되며 각 환경의 로봇 메시 78개,
총 articulation 4개를 확인했다. Native USD 검사 2개로 메시·재질·충돌체 보존과
geometry 공유, body pose 독립성을 검증했다.

1,024개 환경 생성은 통과했으나 초기 가시성 검사에서 기존 layout 8회 한도로는
2개 환경이 남았다. 실패한 환경만 최대 32개 layout까지 재시도하도록 한도를 늘렸다.
Target·Blocker·물체 구성 고정과 Target 40% / Blocker 95% 가시성 조건은 유지한다.

수정 후 실제 1,024개 환경에서 기본 rollout 32, PPO 4 epochs·4 minibatches로
native 학습 1 iteration을 완료했다(exit code 0). 초기 reset은 최대 17개 layout 제안으로
모든 환경이 승인됐고, 학습 중 개별 auto-reset·32,768 transition 수집·PPO 갱신·
TensorBoard와 checkpoint 저장까지 확인했다. 프로세스 최고 RSS 8,658 MiB(8.45 GiB),
5초 간격 GPU 전체 사용량 관측 최댓값 7,732 MiB였다. Scene 생성 13.78초,
학습 iteration 45.21초(수집 23.79초·갱신 21.42초), 초기화·초기 reset·종료 포함
전체 실행 3분 56초였다. CPU 검사 156개와 별도 native USD 검사 2개가 통과했다.
이로써 이전의 “1,024개 실제 환경 미검증” 상태를 해소했지만 장시간 학습·여러 seed는
아직 검증하지 않았다. 새 production 파일이나 별도 Env/runner 클래스는 추가하지 않았다.

### 2026-10-13 새 프로젝트: raw current/previous 입력과 초기 IK 교정

사용자는 current/previous observation의 별도 encoder 도입을 확정한 적 없음을 지적했고,
우선 scene encoder 결과에 두 sample을 직접 concat하여 최종 MLP에서 처리하도록 결정했다.
이에 `HistoryEncoder`의 causal convolution과 `temporal_dim` 설정을 제거했다.
Scene 320D + 현재 72D + 직전 72D = 464D를 MLP 256→128에 입력한다.
GNN·역할 조건부 attention, actor/critic 공유, 544D observation 및 외부 mask 규약은 유지한다.
Mask는 결측만 0으로 처리하며 feature로 붙이지 않는다. 파라미터 수는 234,789개다.
이전 temporal encoder checkpoint는 새 구조와 호환되지 않으며 metadata 검사에서 거절한다.

초기 자세 문제의 최종 사용자 설명은 “hand는 선반을 향하지만 elbow가 선반 쪽으로 나와
꼬인 듯한 자세”였다. 기존 elbow-up 검사만으로는 높은 팔꿈치 앞에서 전완이 뒤로 접히는
branch를 막지 못했고, 자기 충돌 검사는 arm끼리만 검사하여 hand–arm 겹침을 빠뜨렸다.
수정 후 손목 관절 중심이 팔꿈치보다 선반 방향(-X)으로 최소 5 cm 앞에 있어야 한다.
기존 elbow 높이 ≥5 cm·Blocker 앞 random TCP·hand 방향·scene 여유 15 mm도 유지한다.

모든 hand collider와 arm 사이의 OBB 여유 2 mm를 배치 GPU SAT에 추가했다.
고정 장착 관계인 hand–wrist_3/센서와 열린 hand 내부는 제외한다. Arm끼리의 두 관절 이내
기계적 하우징 예외는 유지한다. 쌍 목록은 한 번 캐시해 CPU 검증과 GPU 검증이 공유한다.
이는 초기 reset 후보의 검사이며 runtime PhysX self-collision 활성화 변경은 아니다.
실패한 후보는 채택하지 않고 기존 독립 배치 재시도 경로로 처리한다.

실제 USD 기하에서 독립 배치 128개로 비교했다. 기존 IK의 승인 127개 중 26개가 새 전완 방향
조건을 위반했고, 10개가 hand–arm OBB 여유를 위반했다(두 집합은 중복 가능).
수정 후 같은 배치 128개 중 승인된 125개는 모두 두 조건을 통과했고 최소 전완 전진 거리
5.17 cm였다. 3개는 IK 한도 내에 해를 찾지 못해 거절됐다. 이 측정에는 시점 가시성 검사는
포함하지 않는다. 4개 실제 환경의 초기 reset과 렌더 확인도 완료했다.

최종 CPU 검사 158개와 native USD 검사 2개가 통과했다. 새 모델·IK 검사로 실제
1,024개 환경에서 기본 rollout 32·PPO 4 epochs의 학습 1 iteration도 완료했다(exit code 0).
TensorBoard·checkpoint 저장까지 확인했으며 최고 RSS 8,228 MiB(8.04 GiB),
iteration 41.16초, 초기화·종료 포함 전체 실행 3분 41초였다. 새 production 파일은 없다.

### 2026-10-13 새 프로젝트: iteration 처리량 개선

사용자는 약 30초/iteration에서 90,000 iteration이 너무 오래 걸린다고 지적했다.
실제 사용자 TensorBoard 로그에서 collection 약 9–13초, learning 약 17–18초를 확인했다.
환경 수·rollout 32·PPO 4 epochs·4 minibatches·모델 설계를 유지하는 최적화를 적용했다.

- GNN message MLP의 두 번째 affine map을 edge 평균 이후로 옮겼다.
  `mean(Wx+b)=W mean(x)+b`를 사용하며, 이웃이 없는 node에서는 bias를 0으로 차단한다.
  기존 dense 식의 출력·입력/파라미터 gradient 및 고립 node의 bias gradient 검사가 통과했다.
- 추론 chunk 기본값 128→512, 학습 microbatch 128→1,024로 변경했다.
  각 PPO minibatch 안의 gradient 누적 횟수만 줄이며 전체 sample·optimizer step 수는 유지한다.
  모델 파라미터·state dict·metadata는 같아서 기존 raw current/previous checkpoint도 호환된다.
- 합성 native runner(1,024환경·rollout 32·epochs 4·minibatches 4)에서 기존 chunk 128의
  18.34초가 최적화 후 chunk 1,024에서 3.97초로 감소했다. PyTorch allocated peak는
  약 1,893 MiB이며 이는 시뮬레이터 제외 측정이다.
- 실제 1,024환경·3 iteration에서 learning 3.40/3.32/3.33초,
  collection 23.42/12.78/13.48초, iteration 26.82/16.10/16.81초를 확인했다.
  CPU 검사 159개가 통과했고 기존 사용자 checkpoint의 metadata·전체 모델 weight를
  strict load로 확인했다. Native runner와 production reset logging은 변경하지 않았다.

남은 큰 비용은 환경 실행과 독립 reset이다. 임시 계측에서 3 iteration reset 준비 합계
26.37초 중 layout 8.87초·IK 3.79초·vision 13.20초였다. 이는 host wall time이며
긴 학습에서 timeout reset이 늘어난 구간의 평균은 아직 확인하지 않았다.
90,000 iteration은 현재 설정에서 2,949,120,000 transition이므로 30초일 때 31.25일,
16–17초일 때도 약 16.7–17.7일이다. 이번 작업에서 epoch·rollout·환경 수를 줄이거나
학습을 90,000회 실행하지는 않았다. 새 production 파일은 추가하지 않았다.

### 2026-10-13 새 프로젝트: ReLU baseline과 attention/fusion 명확화

사용자는 SiLU 대신 ReLU를 선택했다. GNN node/message/update MLP와 concat 이후 최종
MLP의 SiLU를 모두 ReLU로 변경했다. Activation은 weight가 없으므로 tensor shape만으로
checkpoint 호환성을 판별할 수 없다. `ModelCfg.activation="relu"`를 metadata에 포함하여
이전 SiLU checkpoint를 동일 모델처럼 resume하는 것을 막는다. 위 처리량 수치는 ReLU 전환
전 SiLU 모델에서 측정한 결과다.

역할 조건부 attention은 제거되지 않았으며 `SceneGraphEncoder.forward()`에서 Target/Blocker
embedding으로 두 query를 만들고 전체 관측 node의 key/value와 masked softmax를 계산한다.
`fusion`의 `nn.Sequential`은 concat 후 최종 MLP의 layer container다. 시간 인코더가 아니며
`Linear(464,256) → ReLU → Linear(256,128) → ReLU`를 순서대로 실행한다.

### 2026-10-13 초기 ray perception의 중복 연산 축소

Ray casting은 CPU가 아니라 환경의 CUDA device에서 Warp mesh BVH query로 실행한다.
CPU에는 시점 난수 생성과 후보 재시도 제어가 남는다. ReLU/SiLU 교체와 별개로
reset의 초기 관측 생성이 큰 비용임을 확인하고 기존 sensors.py 안에서 최적화했다.

- 시야 밖의 고립 실루엣 광선은 전체 body 목록을 순회하지 않고 두 role mesh를 직접 조회한다.
- Target/Blocker 가시성 검사를 통과한 후보만 visible point cloud와 OBB를 계산한다.
- 실제 메시, 320×240 해상도, 40%/95% 가시율, 독립 episode seed와 재시도 규칙은 유지한다.

32환경 native 검사에서 16환경 부분 reset의 capture_batch 합계는 1.196→1.026초,
부분 reset 전체는 2.187→2.046초였다. 최초 reset은 7.770→7.794초로 유의한 개선이
없었다(초기 컴파일/준비 비용 포함). 단일 비교이며 1,024환경 전체 iteration의 개선율로
일반화하지 않는다. CPU 테스트 160개 실행 중 159개 통과, native 전용 1개 skip.

추가로 같은 GPU 실제 장면에서 기존/신규 capture를 11 batch 비교했다. Role mask,
가시율, pixel count, 승인 여부, 선택 시점의 attempt가 일치했다. Depth/OBB/시점 pose는
float32 연산 오차를 고려한 rtol=1e-4, atol=1e-5로 일치했다. 최초 준비 batch를 제외한
동일 입력 비교에서도 초기 관측 생성 시간이 감소했으며, 장시간 1,024환경 학습은
이번 변경 후 다시 측정하지 않았다.

### 2026-10-13 1,024환경 iteration 병목 실측

사용자는 512환경도 느려 중단했으며, 분석 기준을 1,024환경으로 지정했다.
학습 알고리즘·관측·보상·가시율·해상도·timeout을 변경하지 않고 native runner의
10 iteration을 임시 wrapper로 계측했다. 새 production 파일은 추가하지 않았다.

기존 장시간 실행 `logs/rsl_rl/contact_manipulation/2026-10-13_17-25-25`의
마지막 100 iteration은 collection 평균 37.563초, learning 평균 3.394초였다.
약 92%가 rollout 쪽이다. 초기 몇 iteration만 측정하면 reset 부담을 과소평가한다.

이번 계측은 `/tmp/contact_hotspots_train.py`, 결과는 `/tmp/contact_hotspots_1024.json`,
native 로그는 `/tmp/logs/rsl_rl/contact_manipulation/2026-10-13_19-36-13`에 있다.
초기화 비용은 제외했다. 10 iteration 전체 학습 시간은 284.07초였다.

8번째 iteration(index 7)의 collection 123.24초 + learning 3.45초를 분해하면:

| 구간 | 시간 | 비고 |
| --- | ---: | --- |
| CPU 배치 생성 `sample_scene` | 59.38초 | 1,323회 호출; reset 대상별 순차 실행 |
| 초기 vision 전체 | 51.86초 | 1,091장면 요청, 8,910시점 후보 |
| IK 초기 자세 검색 | 3.83초 | 이미 CUDA graph를 사용하는 batch IK |
| 나머지 reset | 0.88초 | reset 전체 115.94초에서 위 세 항을 제외 |
| reset 제외 env.step | 6.90초 | 물리·제어·상태 갱신 등 |
| PPO/return 계산 | 3.45초 | native logger learning_time |
| 나머지 rollout | 0.40초 | policy 추론·rollout bookkeeping 등 |

이 iteration의 reset 대상은 924개였다. 위 시간은 중첩 구간을 제외한 분해다.
별도 CUDA event에서 vision 내부 `_trace`는 41.94초, OBB fit은 0.99초였다.
GPU 시간은 vision 시간에 포함되므로 표에 다시 더하면 안 된다.
계측 wrapper와 GPU 동기화/CPU 대기 때문에 미세한 비용 귀속 오차가 있으며,
동일 seed라도 GPU 수치 오차와 학습 경로에 따라 기존 실행과 reset 대상 수는 달라질 수 있다.

Reset이 적은 index 1–6은 reset 제외 env.step 평균 6.61초, PPO update 평균 3.34초였다.
현재 구현에서 reset을 완전히 제거한다고 가정해도 약 10초 수준의 기본 비용이 남는다.
이는 하드웨어의 절대 하한이 아니라 이번 구현에서 측정한 기준이다.

원인과 다음 최적화 우선순위:

1. `events._build_episode_batch`가 Python `for row in pending` 안에서 전체 layout을
   순차 탐색한다. 일부 SAT/영역 계산만 Numba이며 경로 sweep·convex hull은 Python이다.
   실제 catalog를 재사용한 별도 CPU profile(50회)에서는 `sample_scene` 누적 2.244초 중
   `_path_sweeps` 0.871초, 그 내부 `_convex_hull` 0.764초였다(중첩 수치).
   성공 경로 sweep의 재계산을 줄이고 이 기하 루프를 컴파일/배치화하는 것이 우선이다.
   독립 seed를 유지하면서 환경별 준비를 병렬화할 여지도 크다.
2. Ray query 자체는 Warp CUDA다. 현재 이미지 내 광선은 모든 활성 object/robot/shelf
   mesh를 순회하고, 가시성에 실패한 시점에서도 이 전체 작업을 수행한다.
   실제 메시 기반 role-only 판정으로 필수 조건을 먼저 검사하고, 통과 후보만 전체 장면을
   조회하는 단계화와 보수적인 광선/메시 후보 제거가 다음 대상이다.
   Role-only 가시율은 다른 가림을 추가하기 전의 상한이므로 이미 기준 미달이면
   전체 장면 조회 없이 안전하게 거절할 수 있다. 최종 승인에는 전체 메시 판정이 필요하다.
3. Episode 8초 / control dt 1/30초 = 240step, rollout 32step이므로 동시에 시작한
   episode의 첫 timeout이 8번째 iteration에 몰린다. 부하 분산은 최대 지연을 줄이지만
   reset 총량을 없애지는 않는다. Native timeout counter만 무작위화하면 elapsed time,
   initial vision/history와 episode 의미가 어긋나므로 그 방식은 적용하지 않는다.
4. 그다음은 기본 물리/제어 비용이다. 이번 보통 iteration에서 physics_step 약 2.8초,
   apply_actions 약 1.3초, scene.update 약 0.62초였다(호스트 측 구간이며 CUDA 대기 포함).
   OBB fit, activation 교체, policy 추론은 현재 우선순위가 낮다.

이번 턴은 원인 분석이며 위 구조 변경의 속도 향상을 구현/보장한 것은 아니다.
모든 환경의 독립 장면과 실제 메시 가시성 조건을 유지하는 방향으로 후속 최적화를 진행한다.

### 2026-10-13 배치 생성 최적화 1단계: 동일 proposal stream + GPU SAT

사용자가 생성 규칙/배치 결과를 유지하는 첫 최적화를 승인했고, 가능한 연산의 GPU 사용을
선호했다. Ray casting 변경은 다음 논의로 남겼다. 이번 변경은 기존 events.py와 tests,
README/이 문서에만 있으며 production 파일을 추가하지 않았다.

- Scalar sample_scene의 내부를 proposal generator로 추출하고, sample_scenes가 환경별
  generator를 SAT 요청 지점에서 모은다. 난수 순서, 180개 위치 후보, 첫 6개 채점,
  큰 물체 우선 배치, 300회/32회 재시도와 처음 성공한 경로 선택은 같다.
- 최대 512개 환경 요청을 묶어 Warp CUDA에서 물체 위치 후보의 SAT를 계산한다.
  장애물별 축/projection은 한 번 계산해 같은 물체의 모든 후보에 공유한다.
  Float64, fuse_fp=False로 CPU gap/EPS 연산을 보존한다. 판정 결과는 묶음마다 반환한다.
  64개 미만의 요청은 Numba CPU 경로를 사용한다. 실측에서 8–16개 GPU 묶음은
  전송/동기화 비용으로 느렸고, 64개 이상 SAT 묶음은 CPU 대비 이득이 있었다.
- 회전 구간의 두 convex hull을 Numba 안에서 계산한다. 5도 이하 회전 구간과 padding,
  EPS 기준을 유지한다. Near-collinear 입력에서 꼭짓점을 과소 할당하지 않도록
  모든 확장 입력점(최대 32개)을 담을 공간을 사용한다.
- 성공한 경로의 sweep을 재계산하지 않고 사용한다. 불변 catalog 면적/종류 정보를 캐시한다.
- 환경별 CPU 난수 생성, 후보 제안/채점은 남아 있다. 배치 생성 전체가 GPU인 것은 아니다.

실제 catalog와 episode seed 1,024개, 초기 컴파일/CUDA 준비 제외 비교:
기존 28.442초 → 최적화 CPU 24.236초 → GPU 혼합 경로 25.374초.
기존 대비 혼합 경로 약 10.8% 단축이며, 현재는 CPU 전용 최적화보다 약 4.7% 느리다.
GPU SAT 일부의 개선을 전체 배치 생성의 추가 개선으로 해석하지 않는다.
전체 batch 처리에는 전송 및 generator 간 전환 비용이 남는다.
비교 스크립트 `/tmp/contact_layout_compare.py`, 결과 `/tmp/contact_layout_compare1024_final.log`.
1,024개 모두 물체 구성·역할·위치·yaw·clearance 경로 및 최종 random.Random state가
기존 구현과 정확히 일치했다. CPU/GPU 회귀 테스트는 총 163개 중 162개 통과,
native 전용 1개 skip. Ragged 다각형, gap/EPS 경계, 회전 sweep, 환경 순서 및 batch 크기
변경을 포함한다. GPU 시험은 실제 RTX 3090에서 실행했다.

Isaac Sim은 standalone Warp 1.15와 다른 Warp 1.8.2를 포함한다. 커널 옵션을 공통
`wp.set_module_options` API로 설정하여 양쪽을 지원한다. 내장 1.8.2의 실제 CUDA에서도
배치 관련 12개 테스트가 통과했고, 공통 옵션으로 바꾼 후 전체 CPU 검사 163개도 재통과했다.

최종 native 검증: 1,024환경 headless 8 iteration이 첫 대량 timeout까지 정상 완료했다.
임시 계측 스크립트 `/tmp/contact_layout_native_profile.py`, 결과
`/tmp/contact_hotspots_layout_1024.json`, 로그 `/tmp/contact_layout_native_1024.log`.
8번째 iteration에서 reset 대상 924개로 이전 계측과 같았으며, 배치 생성 구간은
59.38→42.10초(약 29.1% 감소), reset 전체는 115.94→101.15초였다.
Vision은 51.86→53.90초, 후보 수는 8,910→8,979개로 달라졌다. 학습/물리 궤적은
완전히 같은 실행이 아니므로 29.1%를 동일 입력의 엄밀한 속도 향상으로 취급하지 않는다.
정확히 같은 입력의 비교는 위 1,024배치 실험(28.44→25.37초)이다.
이번 작업에서 ray casting 구현이나 가시성 조건은 수정하지 않았다.

### 2026-10-13 배치 재시도 원인 집계와 다음 설계 검토

사용자는 위 개선 폭이 부족하며 조건 실패 시 전체 배치를 다시 만드는 구조를 지적했다.
다음 방법을 검토하기 위해 실제 저장된 catalog/rules와 독립 episode seed 1,024개로
현재 sampler의 분기 횟수를 집계했다. 학습 실행이나 IK/vision을 포함한 계측이 아니며,
분기 횟수를 시간 비율로 해석하지 않는다. Production sampler는 변경하지 않았다.
임시 도구 `/tmp/contact_layout_rejections.py`, 결과 `/tmp/contact_layout_rejections_1024.json`.
계측 유무에 따른 16개 배치 및 RNG state의 일치를 확인했다.

- 승인된 1,024개 배치를 만드는 데 위치/yaw 시도 6,706회(평균 6.55회)가 필요했다.
- Blocker footprint의 접근 영역 차단율 부족 2,121회, 주변 물체 배치 실패 3,561회였다.
  이는 탈락한 위치/yaw 시도의 각각 37.3%, 62.7%다. 후보가 없다는 것은 현재 yaw와
  유한한 위치 후보에서 찾지 못했다는 뜻이며, 기하학적으로 배치 불가능하다는 증명은 아니다.
- 배치 실패 때 버린 이미 배치된 주변 물체는 총 34,972개, 실패당 평균 9.82개였다.
  남은 물체가 3개 이하인 실패도 1,396회였다.
- 이 표본에서는 최종 구역 밀집도, 전체 점유율, 퍼짐, 이웃 비율, clearance 경로 검사로
  탈락한 시도가 없었다. 이후 reset이나 다른 seed/config에도 없다고 일반화하지 않는다.
- 위치 생성 이전의 물체 구성 실패도 별도로 187회 있었다. 전체 호출은 1,211회였으며
  위 위치/yaw 시도 횟수와 구분한다.

검토 중인 다음 방향(아직 구현/성능 검증하지 않음): 실패한 물체의 위치/yaw만 재제안하고,
막히면 충돌에 관여한 소수 주변 물체를 되돌려 부분 수정한다. 역할과 승인된 clearance 경로는
가능한 한 유지하되, 부분 수정 예산을 소진하면 경로/역할 pose 재탐색으로 단계적으로 확대한다.
Blocker Y는 footprint 차단율을 만족할 수 있는 구간에서 제안해 조기 탈락을 줄인다.
이는 카메라 가시율을 보장하지 않으며 실제 메시 가시성 검사는 그대로 필요하다.
GPU에서는 위치/yaw/활성 mask/구역별 면적/후보 점수를 유지하고 환경×후보 축으로 계산하여,
현재의 CPU generator와 SAT 결과 왕복을 줄이는 방향이다. 물체 간 순차 의존성은 남는다.
정확히 같은 과거 seed 결과보다 연구 조건과 분포 다양성을 검증해야 하는 알고리즘 변경이며,
구현 시 재시도 횟수·평균/p95/p99 생성 시간·배치 분포·독립 geometric validator로 평가한다.
Ray casting 최적화는 별도 후속 논의로 남긴다.

사용자는 다음 설계에서 조건 검사 자체를 가능한 한 줄일 것을 추가로 요구했다.
부분 재배치만 도입하여 같은 검사를 반복하는 것으로는 충분하지 않다. 불변 catalog/높이
조건은 설정 준비 시, 전체 점유율은 물체 구성 확정 시 처리하고, 위치는 해당 yaw에서
허용된 중심 영역 안에서 생성하는 방향이다. 비싼 충돌 판정은 공간상 관련된 장애물과
변경된 물체에 한정하고, 구역별 면적·이웃 관계·범위는 누적 갱신 결과를 재사용한다.
구역별 면적은 이미 누적되고 있으므로 새 최적화로 중복 주장하지 않는다. 구성만으로
보장하지 못한 전역 조건의 최종 scalar 판정과 필요한 정확한 충돌 판정은 유지한다.
허용 영역 생성도 비용이 있으므로 후보마다 재구축하지 않고 유효한 동안 재사용해야 한다.
현재 전체 validate_sample 재검사는 이미 학습 경로에서 제외되어 있다. 다음 최적화는
검사 함수 삭제 자체보다 후보 수·정밀 기하 판정 수·전체 재탐색 횟수 감소로 평가한다.
위 항목은 설계 요구이며 이 문서 갱신에서 생성 알고리즘을 변경하지 않았다.

### 2026-10-13 배치 생성 최적화 2단계: 조건부 생성과 GPU 부분 재배치

사용자가 위 개선안의 적용을 승인했다. 기존 events.py에서 생성기를 교체했고,
새 production 파일을 만들지 않았다. Ray casting·IK·MDP·policy 구현은 변경하지 않았다.

- 물체 구성의 점유율과 높이는 구성 확정 시 처리한다. Blocker footprint와 접근 영역의
  겹침 면적은 Y 이동에 대해 구간별 이차식이므로 경계와 근을 계산해 허용 Y 구간을 구한다.
  그 안에서 샘플링하며, 차단율 미달 때문에 역할 pose 전체를 버리는 단계를 없앴다.
- CPU는 역할·물체 구성·clearance witness를 준비한다. 주변 물체 배치는 Warp device에
  위치/yaw·배치 mask·구역 면적·이웃 관계를 유지하며 실행한다. 후보마다 CPU로 돌아오지 않는다.
  한 물체의 위치/yaw 후보 64개를 환경과 후보 축으로 병렬 처리한다. 선반 내 중심 범위에서
  직접 생성하고, 미달 구역 및 가까운 이웃을 고려해 선택한다. 이전의 yaw 고정/180개 위치
  후보/첫 6개 채점 방식과 달라졌으며, 과거 버전과 동일 좌표/RNG state 보존은 하지 않는다.
- 두 번 후보를 찾지 못하면 후보를 가장 많이 막은 주변 물체 하나를 빼고 현재 물체를
  먼저 넣는다. 이후 빠진 물체를 다시 채운다. Target·Blocker와 조작 경로는 유지한다.
  최대 16회 부분 수정 및 제한된 진행 단계 이후에도 실패하면 해당 역할 pose/경로만 다시
  제안한다. 성공 환경은 유지하며, 기존 composition/outer reset 재시도 상한도 유지한다.
- AABB broad phase로 정밀 SAT 대상만 남긴다. SAT의 축별 gap 조건을 유지하도록
  broad phase에 sqrt(2)×gap을 사용한다. 예약 다각형의 축·투영값은 한 번 계산하고,
  구역 면적·이웃 관계는 삽입/제거한 물체만 갱신한다. 최종 조건은 누적 결과로 확인한다.
  Float64/fuse_fp=False이며 CPU도 같은 Warp 구현을 사용한다. 16개 진행 단계마다 완료
  여부만 확인하고, 최대 512개 장면을 묶어 임시 메모리를 제한한다.

실제 catalog와 동일 역할·seed 1,024개로 초기 준비를 제외한 반복 비교:

| 방식 | 1회 | 2회 | 평균 |
| --- | ---: | ---: | ---: |
| 직전 1단계 GPU 혼합 생성기 | 23.949초 | 23.716초 | 23.833초 |
| 새 GPU 생성기 | 3.161초 | 3.126초 | 3.144초 |

배치 생성 시간 약 86.8% 감소(약 7.6배 처리 속도)다. 학습 iteration 전체의 개선율이 아니다.
스크립트 `/tmp/contact_layout_stage2_compare.py`, 결과 `/tmp/contact_layout_stage2_comparison.json`.
두 번 모두 1,346개 역할/경로 plan 중 1,024개를 승인했고, 부분 수정 8,048회였다.
64후보 진행 단계는 총 45,818회였다. 기존 재시도 방식 대신 일부 물체만 다시 배치한 결과다.
두 방식의 모든 결과를 독립 validate_sample로 검사했다. 1,024개 모두 역할/물체 구성은
같았고 각 방식 안에서 1,024개 장면이 서로 달랐다. 개수 범위 11–23개, 평균 16.39개,
평균 전체 점유율 36.25%였다. 위치 분포는 동일하지 않다. Target 평균 X가
-0.7276→-0.7503m로 약 2.27cm 뒤로 이동했으며, 전체 위치 범위는 계속 사용했다.
따라서 속도 개선을 기존과 같은 학습 데이터 분포의 보장으로 해석하지 않는다.

CPU 전체 검사 165개 중 164개 통과/native 전용 1개 skip. Isaac Sim 내장 Warp 1.8.2의
CPU/CUDA 배치 관련 14개 검사도 통과했다. 새 GPU 기하 함수의 독립 scalar oracle 비교,
gap 경계, Blocker 허용 구간, 부분 수정 후 전체 조건, 고정 구성, 2개 물체 설정,
환경 순서/batch 크기에 대한 device별 재현성을 포함한다. CPU/GPU 사이의 동일 좌표는
보장하지 않는다. 길이가 EPS 이하인 경로 edge는 기존 scalar SAT처럼 분리 축에서 제외하며,
중복/아주 짧은 edge를 추가해도 배치가 바뀌지 않는 검사도 포함한다.

실제 1,024환경 headless 학습은 첫 대량 timeout을 포함한 8 iteration을 정상 완료했다.
스크립트 `/tmp/contact_layout_stage2_native.py`, 계측 결과
`/tmp/contact_hotspots_layout_stage2_1024.json`, 로그 `/tmp/contact_layout_stage2_native_1024.log`.
8번째 iteration의 reset 대상 917개, layout 6.324초, vision 46.811초, reset 전체 57.515초,
env.step 64.299초, native iteration 68.08초였다. GPU trace 37.70초는 vision 시간에 포함된다.
직전 1단계 실행은 reset 924개, layout 42.10초, reset 전체 101.15초, iteration 112.17초였다.
배치 분포와 학습/물리 궤적, vision 후보 수(8,979→8,014)가 달라 같은 작업량의 엄밀한
속도 비교는 아니다. 같은 입력의 배치 생성 비교는 위 표를 사용한다. 새 실행의 8 iteration
전체 학습 시간은 181.21초였으며 초기 장면 생성 시간은 제외했다. 장시간 학습의 평균 또는
p95/p99 iteration 시간은 이번 짧은 검증으로 확정하지 않는다. 검증 실행은 종료했다.

### 2026-10-13 초기 ray perception의 다음 최적화 설계 검토

사용자는 ray casting을 어떻게 줄일지 물었다. sensors.py를 다시 확인했으며, 이번에는
구현을 변경하지 않았다. 현재 GPU와 종류별 공유 mesh BVH는 이미 사용한다. 시점 후보마다
640×480 광선 위치를 처리하며 중앙 320×240은 모든 활성 물체/robot link/shelf를 순회하고,
나머지는 두 역할의 고립 실루엣만 조회한다. OBB fit은 이미 가시성 통과 후에만 실행한다.

다음 설계의 핵심은 역할 가시성 검증과 전체 관측용 depth/label 생성을 분리하는 것이다.
각 역할의 실제 메시를 완전히 감싸는 투영 범위 안에서 고립 실루엣/깊이를 구한다. 두 역할만
고려한 가시율은 다른 물체를 추가하기 전의 상한이므로 기준 미달 시 바로 탈락시킬 수 있다.
상한을 통과한 시점은 역할 실루엣 픽셀에 대해서만 다른 메시의 앞쪽 가림을 검사하고,
실제 가시율까지 통과한 시점에 전체 depth/label과 visible-surface OBB를 계산한다.
분모는 같은 pose의 역할 단독 실루엣이며 Target 40%/Blocker 95%, 경계/최소 픽셀 조건을
유지한다. Bounding box는 메시 조회 범위를 줄이는 용도이며 관측 geometry를 대체하지 않는다.

정밀 query 전에는 보수적인 투영 tile 목록 또는 ray–bounds 검사로 교차 가능 메시만
남긴다. 카메라 근평면을 가로지르는 bound는 안전하게 처리하며, 애매한 경우 원래 조회로
돌려 실제 교차를 누락하지 않는다. 물체 inverse pose와 시점별 local ray origin도 공유한다.
8개 장면 묶음에서 마지막 어려운 시점을 기다리는 현재 방식은 완료 슬롯에 다음 독립 장면을
채우는 큐로 바꿀 수 있다. 요청별 seed/시도 순서는 보존한다. 기대 개선율은 아직 측정하지
않았으며, 동일 장면/시점의 실루엣 픽셀·가시율·승인 여부·최종 OBB와 1,024환경 비용으로
검증해야 한다. 최종 관측 해상도나 실제 메시 기준을 낮추는 설계는 제안하지 않았다.

### 2026-10-13 물체만의 가림 계산과 낮은 ray 해상도 검토

사용자는 robot/shelf를 ray 검사에서 제외하고 픽셀 해상도를 낮추는 방향을 제안했다.
위의 해상도 유지 설계보다 먼저 검토할 단순화다. 아직 sensors.py 구현이나 기본값은
변경하지 않았다. 다음 단계의 제안은 활성 물체만의 실제 메시 가림 계산과 160×120이다.

- Robot/shelf는 ray용 메시 등록과 body pose 수집에서 제외한다. Target/Blocker뿐 아니라
  모든 활성 물체를 포함하며, 비활성 pool slot은 활성 ID 목록으로 압축해 순회 비용도 줄인다.
  물리 시뮬레이션 및 초기 로봇 자세의 충돌 검사는 그대로 필요하다. 관측 좌표 변환에 쓰는
  robot root pose도 계속 사용한다.
- 관측 의미는 물체 간 가림으로 제한된다. 실제로 robot/shelf에 가려진 물체 표면이 관측에
  포함될 수 있으므로 원래 전체 장면 카메라와 동등한 최적화로 해석하지 않는다.
- 320×240→160×120이면 중앙 광선 위치는 76,800→19,200개이고 역할의 단독 실루엣을
  계산하는 확장 격자도 640×480→320×240으로 줄어든다. 각각 1/4이며 전체 실행 시간이
  반드시 1/4이 된다는 뜻은 아니다. 같은 종횡비와 FOV를 유지하도록 intrinsics도 축소한다.
  추가 비교 후보는 128×96(기존 광선 위치 수의 16%)이다.
- Target 40%/Blocker 95%와 단독 실루엣 대비 가시 면적의 정의는 유지한다. 픽셀 수가
  줄면 경계 근처 판정과 작은 물체의 검출이 달라질 수 있다. 현재 최소 실루엣 32픽셀,
  최소 관측 16점은 가시율과 별도의 조건이므로 재검토하되, OBB 안정성을 확인하지 않고
  단순히 1/4로 낮추지는 않는다. 관측은 계속 실제 보이는 표면에서만 추정한다.
- 같은 장면/시점으로 전체 메시 320×240, 물체만 320×240, 물체만 160×120을 비교해
  가림 대상 제외의 영향과 해상도 축소의 영향을 분리한다. 관측 누락, 역할 가시율/승인 여부,
  OBB 변화, 시점 재시도 수와 1,024환경 vision/reset/iteration 시간을 확인한다.
  더 복잡한 역할 우선 검사나 선택적 고해상도 재검사는 이 결과로 필요성을 판단한다.

### 2026-10-13 초기 관측의 반복 조건 검사 점검

사용자는 반복문으로 조건을 검사하는 과정도 검토하도록 요청했다. 센서/episode builder의
실행 경로와 기존 1,024환경 계측을 대조했다. 이번에도 실행 코드는 변경하지 않았다.

- `sensors.py:capture_batch`는 기본 8개 요청을 순차 처리한다. `_capture_pool`은 미통과
  환경마다 시점을 하나씩 제안하고 `_trace`로 전체 depth/label과 실루엣을 다시 계산한다.
  시점 재시도 상한은 32회이고, 각 회차 `accepted.tolist()`로 GPU 결과를 CPU에서 읽는다.
  성공한 환경은 pending에서 제외하지만 그 빈자리에 다음 요청을 채우지는 않는다.
- `events.py:_build_episode_batch`의 바깥 재시도 상한도 32회다. 시점 검색을 모두 실패한
  환경은 배치 생성→IK→물리 상태 기록/FK 확인→시점 검색을 반복한다. 이미 성공한 환경은
  반복하지 않으며, 실패 환경도 역할/한 번 정한 물체 구성을 유지하고 pose를 다시 제안한다.
  이론적으로 매 배치가 IK까지 통과하고 시점만 실패하면 한 episode 준비에서 최대
  32×32개 시점을 검사할 수 있다. 이는 상한이며 일반적인 시도 횟수가 아니다.
- `_initial_surface_rays`의 body 순회는 Python CPU 루프가 아니라 광선별 GPU 루프다.
  화면 내부는 전체 슬롯의 enabled 여부를 확인하고 활성 메시를 조회한다. 동일 물체의
  inverse pose 및 시점의 body-local origin도 광선마다 다시 계산한다. 화면 바깥 영역은
  이미 Target/Blocker 두 메시만 직접 조회한다.
- `_trace`가 확인한 역할별 `visible >= min_depth_pixels`는 `fit_visible_obbs`의
  `valid = counts >= min_points`와 같은 labels/임계값을 사용한다. 이후 역할 valid를
  `decisions.tolist()`로 다시 읽는 승인은 현재 구현에서 중복 최소 점 수 검사다.
  이 두 번째 승인은 유한한 OBB인지 등 별도 품질을 검사하지 않는다. 비역할 물체의
  valid mask 계산은 계속 필요하다.

기존 `/tmp/contact_hotspots_layout_stage2_1024.json`의 마지막 iteration은 reset 917개,
vision 요청 1,048개, 시점 후보 8,014개, `_trace` 호출 3,485회였다. 요청당 약 7.65개
시점이며 한 호출당 평균 약 2.30개 시점이다. 이는 최종 성공 장면당 평균이 아니며,
작은 묶음 자체가 GPU 이용률 저하를 증명하지는 않는다. Vision 46.81초 중 GPU trace
37.70초이며, Python 반복/동기화만 제거해 전체 문제가 해결된다고 해석하지 않는다.

앞서 제안한 활성 물체만 검사/160×120과 함께 고려할 수정:

1. 중복 역할 최소 점 수 승인과 GPU→CPU 읽기를 제거하고 결과 전송을 묶는다.
2. 소수(예: 4개) 시점 후보를 환경별로 묶어 GPU에서 평가하고 첫 통과 후보를 선택한다.
   한 후보씩 CPU에 반환하는 횟수는 줄지만, 일찍 성공할 때 불필요한 광선을 더 계산할 수
   있으므로 후보 수는 실측으로 정한다. OBB는 선택된 후보만 계산한다.
3. 완료 슬롯을 다음 요청으로 채워 작은 잔여 묶음의 반복 호출을 줄이고, 물체 inverse pose와
   시점별 local origin은 광선 밖에서 공유한다. 활성 ID만 순회하도록 준비한다.
4. Robot을 가림 대상에서 제외하면 vision은 로봇 관절 자세와 독립적이므로, 생성한 물체
   pose를 raycaster에 전달해 배치→vision→IK 순서로 바꾸는 방안을 검토한다. 이렇게 하면
   vision에서 탈락할 배치에 IK/물리 기록을 먼저 수행하는 낭비를 줄일 수 있다. 이후 IK가
   실패하는 비용도 있으므로 전체 reset 시간으로 판단한다. 관측의 robot base/EEF 좌표계와
   최종 물리 자세 검증은 계속 유지해야 한다.

### 2026-10-13 GPU tensor 기반 초기 관측 최적화 구현

사용자가 반복 조건 검사를 최소화하는 수정을 요청했고, CPU 연산을 줄이고 tensor 처리를
우선하도록 강조했다. 기존 `mdp/sensors.py` 안에서 구현했으며 새 프로젝트 파일은 없다.

- Ray 메시 등록과 pose 수집에서 robot/shelf를 제외했다. 공유 BVH는 물체 종류별로
  유지하고 활성 object ID만 압축해 순회한다. 초기 로봇 충돌 검사와 물리 설정은 변경하지
  않았다. 관측 의미는 물체 간 가림만 반영하는 것으로 바뀐다.
- 기본 해상도 160×120, 같은 FOV, 확장 실루엣 격자 320×240이다. Target 40%, Blocker 95%,
  최소 실루엣 32픽셀과 최소 관측 16점, 시점 후보 상한 32회는 유지했다.
- 시점 난수/위치/ROS 카메라 quaternion을 Warp CUDA 커널에서 한 번에 생성한다.
  `[요청, 후보, 위치/회전]` tensor를 사용하며 episode seed와 후보 번호가 RNG를 정한다.
  CPU에서 pose를 읽어 Python random으로 시점을 하나씩 만들던 경로를 제거했다.
  같은 seed라도 이전 Python 난수열과는 다른 시점을 생성한다. 요청 순서·묶음 크기에
  따른 후보/첫 통과 선택의 재현성은 유지한다.
- 기본 `ray_batch_size=32`개 요청에 `ray_candidates_per_env=4`개 후보씩, 최대 128개
  시점을 묶어 검사한다. 가시성 mask와 첫 통과 index는 GPU tensor reduction으로 계산한다.
  성공/소진 슬롯은 다음 요청으로 채운다. 후보 묶음당 작은 선택 index만 한 번 CPU로 읽으며,
  가시율/픽셀 통계는 전체 큐 완료 후 한 번 전송한다. 요청 포장, 완료 큐와 결과 객체 조립은
  여전히 CPU이며 완전한 GPU reset 파이프라인은 아니다.
- 물체·시점별 camera→body 변환을 GPU에서 먼저 계산해 픽셀마다 inverse pose/local origin을
  반복하지 않는다. 화면 밖에서는 두 역할만 검사한다. OBB는 첫 통과 시점만 tensor scatter로
  계산하고, 같은 역할 최소 점 수를 다시 검사하던 `decisions.tolist()`는 제거했다.
  관측은 계속 보이는 교차 표면만으로 추정하며 초기 vision 고정/EEF 재표현/544D 규약을 유지한다.
- Episode 바깥의 배치→IK→물리 자세 확인→vision 순서는 이번에 바꾸지 않았다. 실패 환경의
  배치 재시도 상한도 유지한다. GPU 묶음은 먼저 통과한 후보 뒤의 일부 후보까지 계산할 수
  있으므로 실제 광선 수와 호출 수를 모두 측정했다.

동일한 실제 128개 장면·512개 시점으로 비교했다. 워밍업 후 두 번의 평균이며
`/tmp/contact_ray_compare_native.py`, `/tmp/contact_ray_controlled_comparison.json`에 기록했다.

| 검사 구성 | ray/가시율 처리 시간 | 승인 시점 |
| --- | ---: | ---: |
| 기존 전체 메시, 320×240 | 1.0233초 | 195/512 |
| 새 구현 물체만, 320×240 | 0.4731초 | 373/512 |
| 새 구현 물체만, 160×120 | 0.1997초 | 375/512 |

같은 시점의 ray 처리 속도는 최종 구성에서 약 5.1배였다. 전체 메시 slot은 82개였고,
새 등록 목록은 물체 pool 40개이며 실제 query는 활성 ID만 순회한다. 같은 320×240에서
기존 커널의 robot/shelf만 비활성화한 기준과 새 커널을 비교하면 label 일치율
99.999982%, 역할 실루엣 픽셀 수 차이 최대 1, 승인 판정 차이 0이었다.
부동소수점 변환 결합 순서에 따른 메시 경계의 작은 차이는 존재한다.

Robot/shelf 제외로 178/512개 시점의 승인이 달라졌다. 이는 관측 모델 변경의 효과다.
물체만의 320×240→160×120 비교에서는 4/512개 승인 판정이 달라졌으며,
역할 가시율 절대 차이 평균은 Target 0.01278, Blocker 0.000224였다.
유효 물체 관측은 7,486→7,079개(407개, 약 5.4% 감소)였다. 작은 가시 영역이
최소 16점에 못 미칠 수 있다. 두 해상도에서 모두 관측된 물체의 OBB 중심 차이는
평균 4.07mm/p95 13.13mm, 축 교환을 제거한 정렬 크기의 최대 성분 차이는
평균 7.80mm/p95 24.04mm였다. 낮은 해상도를 동일한 관측 품질로 해석하지 않는다.

실제 1,024환경 headless 학습은 대량 timeout을 포함한 8 iteration을 완료하고 정상 종료했다.
스크립트 `/tmp/contact_ray_native.py`, 로그 `/tmp/contact_ray_native_1024.log`, 계측 결과
`/tmp/contact_hotspots_ray_opt_1024.json`. 비교 기준은 직전 layout 최적화 완료 실행이다.

| 대량 reset이 발생한 8번째 iteration | 직전 | 이번 |
| --- | ---: | ---: |
| Reset 환경 수 | 917 | 915 |
| Vision 요청 / 평가 시점 수 | 1,048 / 8,014 | 978 / 6,256 |
| `_trace` 호출 수 | 3,485 | 160 |
| 초기 관측 생성 | 46.811초 | 2.500초 |
| GPU trace (관측 시간에 포함) | 37.701초 | 2.272초 |
| 배치 생성 | 6.324초 | 5.760초 |
| IK 초기 자세 | 3.499초 | 2.930초 |
| Reset 전체 | 57.515초 | 11.989초 |
| Native iteration | 68.08초 | 22.45초 |

관측 의미·난수열·학습 궤적이 바뀌어 작업량이 완전히 동일하지 않다. 같은 입력의
비교는 위 512개 시점 결과를 사용한다. 새 실행의 8 iteration 학습 시간은 113.14초
(초기 환경 생성/별도 비교 제외)였다. 장시간 평균이나 최종 학습 성능을 확정하는 검증은 아니다.
검증 중 확인한 GPU 프로세스 메모리는 약 10.4GiB였으며 별도 비교의 이전 메시/버퍼도
만든 실행이다. 정밀한 peak 메모리 측정으로 해석하지 않는다.

전체 검사 168개 중 166개 통과, CUDA/native USD용 2개 skip. 별도 실제 GPU 실행에서
Isaac Sim 내장 Warp 1.8.2의 관측 검사 11개가 모두 통과했다. CPU/CUDA ray·시점 결과 일치,
GPU 시점 생성 중 host tensor 읽기 부재, 첫 통과 선택/큐 보충/시도 상한(4의 배수가 아닌 경우),
요청 재정렬·후보 묶음 크기에 대한 재현성, 가려진 물체 및 시야 밖 역할 처리를 포함한다.

### 2026-10-13 수정 후 이슈와 남은 최적화 검토

사용자는 수정으로 발생한 이슈와 추가 최적화 가능성을 검토하도록 요청했다.
실행 코드는 변경하지 않고 현재 코드, 직전 native 로그와 계측을 점검했다.
기존 8 iteration 로그에서 Traceback/NaN/CUDA 오류는 확인되지 않았으며, 이 결과를 장시간
학습이나 관측 모델의 실물 정확도 보장으로 해석하지 않는다.

확인한 이슈/제약:

- Checkpoint의 `research_metadata`는 model_cfg와 F/T profile만 포함한다
  (`agents/ppo.py:construct_algorithm/save/load`). 따라서 같은 모델·F/T 설정이면
  전체 메시/320×240/Python 시점 RNG로 만든 이전 checkpoint와 물체만/160×120/GPU RNG의
  현재 관측 모델 차이를 checkpoint 호환성 검사에서 구분하지 못한다. Run의 env 설정과
  별도로 perception 버전·가림 대상·해상도·RNG 방식 등을 checkpoint에 기록할 필요가 있다.
  이전 모델의 의도적인 fine-tuning까지 무조건 막아야 한다는 뜻은 아니다.
- 물체만의 가림 모델과 낮은 해상도는 의도한 근사다. 앞선 비교의 관측 5.4% 감소,
  OBB 중심 p95 13.13mm/정렬 크기 최대 성분 p95 24.04mm 변화와 승인 판정 변화는
  관측 품질의 실질적인 차이이며, 임계 픽셀 수를 무조건 낮춰 해결하지 않는다.
  이전 seed와 같은 장면/시점/학습 궤적도 보장하지 않는다.
- 조건 재시도가 전부 제거된 것은 아니다. 시점 실패 시 배치/IK부터 재시도하는 바깥
  episode 루프와 CPU 역할/Blocker 경로 탐색이 남는다. 고정 4개 시점은 첫 후보가
  통과해도 나머지 3개까지 검사할 수 있다. 단일/소수 환경 reset에서 1·2·4개 후보의
  비용을 같은 후보열로 비교할 여지가 있다.

추가로 실제 catalog의 1,024개 배치만 생성해 Python CPU 경로를 계측했다.
`/tmp/contact_layout_review_profile.py`, `/tmp/contact_layout_review_profile.json`,
`/tmp/contact_layout_review_cprofile.txt`. Isaac Sim 내장 Warp 1.8.2를 사용했으며
1,024개 모두 생성됐다. cProfile과 CUDA event 삽입이 있어 아래 시간은 이전 무계측
3.14초 또는 native 5.76초와 직접 비교하는 성능 수치가 아니다. 함수 호출 위치/빈도가 목적이다.

- 계측 전체 5.706초. `_scene_proposals` 누적 2.931초(역할/경로 준비 및 결과 조립),
  `_pack_layouts` 누적 2.755초(CPU 포장·GPU 실행·대기 포함).
- `_clearance_candidates` 1,344회/누적 1.542초, `_path_supported_clear` 4,455회/1.023초.
  `polygons_separated`는 두 경로 등을 합쳐 59,750회/누적 1.272초 호출됐다.
  이 시간들은 서로 포함되므로 합산하지 않는다.
- GPU layout kernel 호출 2,304회, CUDA event로 감싼 launch 구간 합 약 0.863초.
  Tensor.cpu 호출 88회/호스트 소요 0.443초(전송뿐 아니라 GPU 완료 대기 포함).
  CPU/GPU 시간은 중첩되며 GPU 구간 합을 전체 시간에서 빼서 순수 CPU 시간을 계산하지 않는다.

다음 우선순위는 다음과 같다.

1. Blocker의 translation/pivot/rotate-translate 경로 후보와 swept geometry를
   `[환경, 후보, 구간, vertex/axis]` 형태로 묶어 준비하고 GPU에서 조건을 함께 판정한다.
   물체 종류별 고정 footprint/회전 envelope를 재사용하고, 자세에 따라 달라지는 변환만
   갱신한다. 기존 지지/충돌/경로 보장과 독립 scalar 검증을 유지한다.
2. 생성 결과를 `_pack_layouts` GPU→NumPy→SceneSample dict→`pack_episode_states` GPU로
   왕복시키는 연결을 tensor로 유지한다. IK도 NumPy 난수/충돌 box 준비와 최종 CPU 변환 후
   재업로드가 남는다. Episode batch를 tensor로 계속 전달하고 debug 출력만 CPU로 가져온다.
   짧은 layout kernel들의 반복 launch 및 임시 버퍼도 재사용/고정 크기 묶음을 검토한다.
3. 대량 reset 이외의 step/PPO를 별도로 최적화한다. 직전 1024환경의 iteration 1–6은
   평균 12.325초였다. 8번째 iteration은 배치 5.760초, IK 2.930초, vision 2.500초,
   reset 제외 env.step 6.771초, PPO update 3.283초였다. Physics 3.051초와 action 적용
   1.282초는 env.step 안에 포함되므로 다시 합산하지 않는다. Vision만 비용 0으로 만들어도
   해당 22.45초 iteration에서 직접 줄일 수 있는 시간은 약 2.50초(11.1%)다.
   Epoch 수·물리 dt 등 실험 정의 변경에 앞서 기존 tensor 연산의 할당/launch와 PPO
   microbatch 크기를 확인한다. IK solve는 이미 CUDA graph를 사용하므로 그 재도입을
   새 최적화로 제안하지 않는다.

Vision→IK 순서 변경, 시점 후보 묶음 크기 변경, 추가 ray ROI/버퍼 최적화는 후순위다.
실패 이유별 비용을 먼저 비교해야 하며, 앞 단계에서 다른 단계가 실패할 후보를 더 계산하는
역효과도 있을 수 있다. 해상도를 더 낮추는 것은 현재 우선순위가 아니다.

### 2026-10-14 저장된 정책 재생

추가 최적화에 앞서 학습된 checkpoint를 확인할 수 있도록
`src/contact_manipulation/scripts/play.py`를 추가했다. 새 파일은 이 진입점 하나다.
Isaac Lab play 예제의 AppLauncher/Hydra → gym.make → RslRlVecEnvWrapper →
OnPolicyRunner.load → get_inference_policy → env.step 흐름을 사용한다.
공유 GNN/attention 모델의 기존 actor가 observation으로 mask를 만들고
`tanh(mean)`을 출력한다. 별도의 정책 구현이나 자동 JIT/ONNX export는 추가하지 않았다.

- `--checkpoint`는 파일 경로이며 생략하면 native get_checkpoint_path로 최신 run/model을
  선택한다. `--load_run`, `--num_envs`, `--seed`, `--device`, Hydra override를 지원한다.
- `--real-time`은 제어 주기에 맞춰 대기한다. `--max_steps`는 종료할 제어 step 수다.
  Episode 종료 후 새 장면/관측 생성은 환경의 기존 auto-reset을 그대로 사용한다.
- `--video --video_length N`은 checkpoint 옆 videos/play에 영상을 저장한다.
  렌더링은 영상 저장/GUI용이며 정책의 초기 관측은 GPU ray casting을 유지한다.
- Runner는 log_dir=None으로 생성하고 optimizer state는 불러오지 않는다.
  모델/F/T metadata와 state dict의 strict 검사를 유지하며 checkpoint를 수정하지 않는다.
  현재 환경 설정으로 재생하므로 이전 환경/관측 생성 방식의 완전한 복원은 아니다.

기존 tests/test_learning.py에 optimizer 없이 checkpoint를 불러오는 play 경로를 검증했다.
원본 정책과 출력 일치, 반복 추론의 결정성, 비어 있는 optimizer state, iteration 복원을
확인했고 관련 checkpoint/metadata/inference 테스트 3개가 통과했다.
실제 GPU에서 `2026-10-13_23-18-25/model_2050.pt`를 환경 4개로 260 control step
headless 재생하여 정상 종료(exit 0)를 확인했다. 실행 명령은 패키지 README에 기록했다.
이전 `2026-10-13_17-25-25/model_150.pt`도 환경 1개에서 영상 저장·실시간 pacing을 켜고
12 control step 재생하여 정상 종료했다. 첫 녹화 프레임이 검게 나오는 문제는 RGB annotator를
render-only 3회로 먼저 초기화해 수정했고, 저장 MP4의 첫 프레임과 장면 렌더링을 확인했다.

### 2026-10-14 CPU 경로 검사·GNN/PPO·제어 실행 최적화

사용자가 앞서 제안한 추가 최적화를 진행하도록 요청했다. 기존 파일 안에서 구현했으며
배치 조건, 관측/보상/action, GNN·attention의 파라미터 구조, PPO epoch/minibatch,
물리 120 Hz·제어 30 Hz는 유지했다. Ray casting 방식·해상도도 변경하지 않았다.

구현:

- `events.py`: 경로 후보의 A₀ 이탈 여부는 동일한 rectangle SAT 식을 Numba로 묶어 계산한다.
  이동·회전·pivot 후보의 endpoint/전체 sweep 지지·Target 충돌 검사는 환경×후보의 Warp
  kernel로 처리한다. 회전마다 최대 5° 간격, endpoint hull와 arc 오차 padding, 기존 EPS를
  유지하고 후보 순서상 첫 유효 경로를 선택한다. 선택한 경로만 CPU에서 예약 sweep으로 만든다.
  예약 다각형 SAT 축·투영 구간·AABB도 GPU에서 한 번에 계산한다.
- 생성된 XY/yaw tensor를 SceneSample에 보존하고 IK용 active OBB, 물리 pose 생성까지
  재사용한다. 전체 catalog가 아닌 각 batch의 최대 active 수까지 OBB를 모아 collision
  scratch 크기를 유지한다. IK 관절값/TCP/시작 영역은 tensor로 물리 쓰기와 FK 검사에 전달한다.
  독립 검증/debug용 CPU pose 사본, 역할·구성/ID metadata와 NumPy IK 난수 생성은 남는다.
  전체 reset 경로가 GPU-only가 되었다는 뜻은 아니다.
- `model.py`: CUDA FP32에서 edge add/ReLU/mask/reduction을 Triton kernel로 융합한다.
  Backward는 node 입력에서 ReLU 부호를 재계산하여 receiver/sender gradient를 모은다.
  `[B,40,40,64]` pair activation/gradient를 저장하지 않는다. CPU/다른 dtype은 기존 Torch 식이다.
  Node projection·role attention·최종 MLP, state_dict와 model_cfg는 동일하다.
- `rsl_rl_ppo_cfg.py`: 추론 chunk 512→1024, 학습 microbatch 1024→4096.
  1024 env×32 step, 4 epoch×4 minibatch에서 microbatch backward는 128→32회로 줄지만,
  원래 minibatch당 optimizer update 1회(총 16회), sample 가중 gradient 합산은 유지한다.
- `actions.py`: Newton–Schulz 제곱근의 기존 32회 연산을 CUDA graph로 재실행한다.
  순수 tensor 계산만 capture하며 PhysX/OSC 호출 순서·제어식·수렴 판정은 유지한다.
  반환 tensor는 clone하여 다음 replay가 이전 결과를 덮어쓰지 않는다.
  CPU/gradient 필요 입력은 기존 경로다. torch.compile 방식도 별도 측정했으나 근특이 자세의
  제어 출력 차이 때문에 채택하지 않았다. CUDA graph는 TF32 켜짐/꺼짐, broadcast identity,
  변경된 입력에서 유한 값과 invalid/NaN 위치가 기존과 정확히 일치했다.

검증 중 GPU hull의 초기 unique=0일 때 음수 scratch 인덱스에 접근하는 문제를 발견했다.
첫 정점을 미리 포함하고 두 번째 정점부터 처리하도록 수정한 뒤 CPU 및 Isaac Sim 내장
Warp 1.8.2 CUDA 검증과 1024환경 native 실행이 통과했다. 해결 전 커널은 최종 경로에 없다.

계측(RTX 3090, 동일한 장비):

| 구간 | 이전 | 최종 |
|---|---:|---:|
| 독립 배치 1024개, 같은 seed, warm 2회 평균 | 3.327 s | 2.254 s |
| 모델 FP32 forward/backward, batch 1024 | 22.18 ms | 8.50 ms |
| 위 모델의 추가 peak allocated memory | 1705 MiB | 250 MiB |
| Native PPO update, iteration 1–7 평균 | 3.301 s | 1.030 s |
| Native 학습 8 iteration 전체 | 113.14 s | 91.10 s |
| 8 iteration 기준 평균 | 14.143 s | 11.388 s |

독립 모델 비교는 TF32를 끈 FP32이며 물리/센서 메모리를 포함하지 않는다.
개선 모델의 batch 2048/4096 forward+backward는 16.42/31.35 ms,
추가 peak 500/999 MiB였다. Native 학습은 기존처럼 TF32를 켜서 실행했다.

최종 native iteration 시간은 14.39, 10.64, 9.96, 9.90, 9.32, 9.84, 8.58, 18.22초였다.
초기화 이후 iteration 1–6 평균은 9.707초(이전 12.325초)다. 마지막 iteration에서
918개 reset, layout 4.051초, IK 2.786초, vision 2.608초, reset 전체 10.318초,
reset 제외 env.step 6.672초, PPO 1.018초였다. Physics 3.188초와 action 적용 1.019초는
env.step 안에 포함되므로 합산하지 않는다. Damping 부분은 약 0.402→0.106초였다.

8 iteration 전체 시간은 약 19.5% 감소했다. 두 실행의 전체 reset 수는 1164/1166,
마지막 iteration은 915/918로 조금 다르며 부동소수 합산 순서와 물리/학습 궤적의 차이가 있다.
초기화 시간을 제외한 단기 비교이며 장기 처리량 보장은 아니다. 이 평균을 단순 적용하면
90000 iteration은 약 285시간(11.9일)로, 직전 추정 354시간보다 약 69시간 짧다.
남은 큰 비용은 물리 step, reset의 역할/구성 준비·packing과 IK이며 모든 CPU 작업을 제거한 것은 아니다.

CPU 전체 172개 검사 중 168개 통과, CUDA/native USD용 4개 skip.
별도 실제 GPU 실행에서는 Isaac Sim 내장 Warp 1.8.2를 사용한 관련 검사 6개가 모두 통과했다.
기존 파일의 검사에 GPU fused GNN의 출력/gradient, 빈 이웃·ReLU=0·비연속 upstream gradient,
경로 전체의 scalar oracle(중간 pivot 충돌 포함), device pose→충돌/물리 상태 일치 및 host
pose 읽기 부재, CUDA graph 제어의 결과/수명 검사를 추가했다.
실제 1024환경×8 iteration은 정상 종료(exit 0)했고 native logging/checkpoint를 유지했다.

재현 자료는 `/tmp/contact_gnn_opt_bench.{py,json,log}`, `/tmp/contact_layout_opt_bench.{py,json,log}`,
`/tmp/contact_compute_opt_final_native_1024.log`, `/tmp/contact_hotspots_compute_opt_final_1024.json`,
`/tmp/contact_opt_final_cpu_tests.log`, `/tmp/contact_opt_final_gpu_tests.log`에 있다.

### 2026-10-14 센서 동기화·reset·IK·step 연산 추가 최적화

사용자는 1024환경에서 iteration당 1–2초를 목표로, 앞서 검토한 GPU 연산 최적화를
적용하도록 요청했다. 새 파일 없이 기존 `contact_manipulation` 파일 안에서 수정했다.
Native manager/runner/logging, rollout 32, PPO 4 epochs×4 minibatches, 물리 120 Hz·제어
30 Hz, GNN·attention과 current/previous 직접 concat 구조, MDP와 배치 승인 조건은 유지했다.

반영 내용:

- `sensors.py`: 17 tactile·9 forbidden contact sensor가 매 physics tick 전체 환경을
  갱신한다는 조건에서, 갱신할 행을 찾는 native `nonzero()` 동기화를 생략한다.
  Native contact force/history 및 partner별 접촉 판정을 그대로 사용하고, 다른 sensor
  설정은 native 경로로 돌아간다. 부분 reset은 완료된 비선택 행의 history를 보존하고
  reset 행에 이전 PhysX 접촉을 다시 읽어 넣지 않는다.
- `scene.py`: 비활성 물체의 중력 상쇄 force·COM을 reset에서 갱신하고 PhysX에 넘기는
  GPU 메모리를 재사용한다. 매 substep의 COM 조회·force 재계산·reshape 복사를 제거했다.
  물체의 native gravity와 simulation flag는 유지한다. 비활성 물체의 물리 비용 전체가
  제거된 것은 아니다.
- `events.py`: 배치 pose를 GPU에 유지하고 CPU 위치/yaw dictionary는 debug 요청 때만
  생성한다. 정적 지지 높이는 pose 다운로드 없이 사용한다. 이동·회전·pivot 경로 후보는
  Numba 배열로 생성해 Python 객체/중첩 loop를 줄이고, 선택한 후보만 경로 객체로 만든다.
  후보 내용·shuffle 순서·후속 RNG 상태가 기존 scalar 경로와 같은지 검사했다.
  GPU 배치 kernel의 인자는 `record_cmd=True`로 batch마다 한 번 준비해 재사용한다.
  64개 위치 후보, 16단계마다 완료 확인, 실패 시 부분 재배치/재시도 조건은 유지했다.
- `scene.py/events.py`: FP32 CUDA IK의 FK·Jacobian·damped 6×6 solve를 후보별 Warp kernel로
  묶었다. 최대 48회이며 위치 오차 1e−6 m·회전 오차 1e−5 rad 안에서 조기 종료한다.
  기존 승인 오차 5e−4 m·0.01 rad와 elbow/전완 방향/hand–arm 충돌 검사는 유지한다.
  초기 자세 후보 난수도 GPU에서 생성한다. 이전 NumPy 방식과 같은 seed의 시작 자세는
  달라질 수 있다. CPU tensor 식은 검증 기준으로 유지한다.
- `events.py/sensors.py`: 제안한 물체 pose로 초기 가시성을 먼저 확인한 다음 통과한
  장면만 IK와 실제 물리 쓰기/FK 검사를 수행한다. Ray 관측이 물체만 검사하므로 가능한
  순서 변경이다. 관측 pose는 실제 writer와 같은 좌표 변환을 사용한다.
- `sensors.py`: 공유 메시 bounds를 시점에 투영하여 물체를 통과할 수 없는 ray의
  mesh query를 건너뛴다. Bounds가 eye plane을 가로지르면 생략하지 않는다.
  일반 물체는 현재 가장 가까운 hit까지만 검사하고, Target/Blocker의 고립 실루엣은
  기존 far 범위를 유지한다. 보이는 표면 OBB 추정은 배경 ray에 대한 atomic 연산을
  제거했다. 160×120 해상도, 실제 메시, 40%/95% 가시율, 관측된 표면만의 PCA는 유지한다.
- `events.py/actions.py`: 활성 물체의 투영 hull·clipping·면적을 하나의 Warp kernel로
  계산한다. 매 step의 기하 변환·F/T 보상·동적 sample, 전체 implicit damping solve를
  각각 CUDA graph로 재실행한다. PhysX 읽기/쓰기와 OSC 호출은 graph 밖에 둔다.
  결과는 clone하여 다음 replay가 이전 결과를 덮어쓰지 않게 한다.
- `rsl_rl_ppo_cfg.py`: 학습 microbatch 4096→8192. 1024환경의 기본 설정에서 backward
  chunk는 32→16회가 되며, rollout sample·epoch·minibatch·optimizer step 수는 같다.

검증 중 force buffer를 reset마다 교체하던 구현으로 한 번 CUDA 719 오류가 발생했다.
Force/position buffer를 고정 메모리에 복사하는 방식으로 수정했다. 오류의 단일 원인이
이것이었다고 확정하지 않으며, 이후 동일 1024환경의 16 iteration 정상 종료를 반복 확인했다.
초기화 시 모든 물체의 중력을 끄고 활성 물체에만 중력을 적용하는 대안도 별도 `/tmp`
복사본에서 시험했다. 물리 호출 시간은 약 2% 줄었으나 전체 8 iteration 평균은
9.24→9.75초로 늘어 채택하지 않았다. 이 대안은 저장소 코드에 포함하지 않았다.

전체 CPU 검사 181개 중 174개 통과, CUDA/native USD용 7개 skip.
별도 RTX 3090에서는 Isaac Sim 내장 Warp 1.8.2로 관련 검사 37개가 모두 통과했다.
제어 graph의 TF32 on/off·근특이 입력·결과 수명, F/T/기하 graph의 측정값 변경 및 부분 reset,
IK 목표 도달/불가능한 목표 거절, 가림 query 생략 전후 일치, 관측 표면 PCA와 CPU 비교,
배치 경로 scalar oracle·후보 순서·센서 history·고정 force buffer를 포함한다.

CPU 역할/구성 선정, 선택된 경로의 sweep 생성, reset/시점 후보 큐와 일부 완료 상태
GPU→CPU 읽기는 남는다. 전체 실행이 GPU-only이거나 모든 조건 재시도가 제거된 것은 아니다.

최종 native 검증은 RTX 3090에서 1024환경·seed 0·16 iteration으로 정상 종료(exit 0)했다.
이번 수정 전 코드를 별도 복사한 baseline도 같은 장비·설정으로 8 iteration 실행했다.
아래 iteration 평균은 native logger의 collection+learning 합이며 초기 환경 생성은 제외한다.

| 구간 | 이번 수정 전 | 최종 |
|---|---:|---:|
| 첫 8 iteration 평균 | 11.147 s | 9.065 s |
| Iteration 1–7 PPO 평균 | 1.027 s | 0.914 s |
| Iteration 1–7 MDP 상태 갱신 평균 | 0.237 s | 0.101 s |
| Iteration 1–7 기하·센서 읽기/변환 평균 | 0.805 s | 0.675 s |
| Iteration 1–7 IK 준비 전체 평균 | 0.854 s | 0.427 s |
| Iteration 1–7 배치 생성 평균 | 1.065 s | 0.970 s |
| Iteration 1–7 vision 전체 평균 | 0.588 s | 0.650 s |

첫 8 iteration 시간은 약 18.7% 감소했다. Vision 전체 시간은 이 짧은 실행에서 개선이
확인되지 않았다. 각 구간은 host wall time이며 GPU 완료 대기·후보 재시도와 결과 조립을
포함한다. 서로 포함된 구간이 있으므로 표를 합산하지 않는다. 같은 seed라도 GPU IK 난수와
수치 계산 차이로 승인 자세·접촉·reset·학습 궤적이 바뀌므로 동일 작업량의 microbenchmark는 아니다.

최종 16 iteration 평균은 **9.534초**(범위 7.356–14.006초)이며, logging/checkpoint를
포함한 전체 학습 시간은 153.08초였다. **1–2초 목표는 달성하지 못했다.**
최초 iteration을 제외한 15회에서 물리 step 호출은 평균 2.916초(128회/iteration),
action 적용 0.988초, reset 전체 3.121초, PPO 0.910초였다.
마지막 iteration은 reset 제외 env.step이 약 5.234초로, ray casting만 제거해도 목표에
도달할 수 있는 상태가 아니다. 물리 비용과 CPU reset 준비·시점 큐 처리의 추가 개선이 필요하다.

학습 종료 후 비활성 물체 최대 위치 오차는 2.38e−7 m, 6D 속도 norm 최대는 9.73e−6이었다.
PyTorch peak allocated/reserved는 약 2.30/2.79 GiB이며 PhysX·Isaac Sim 메모리는 포함하지 않는다.
16 iteration 검증은 장시간 학습 안정성이나 최종 정책 성능 검증을 대신하지 않는다.

재현 자료: `/tmp/contact_baseline_1024.{json,log}`, `/tmp/contact_release_1024.{json,log}`,
`/tmp/contact_perf_verified.py`, `/tmp/contact_release_cpu.log`, `/tmp/contact_release_gpu.log`.

### 2026-10-14 events.py의 Python 데이터 반복 처리 축소

사용자는 `events.py`의 많은 반복문을 tensor/적절한 자료형으로 정리할 필요를 지적하고,
USD에 physics 속성이 있는데 별도 convex hull 계산이 필요한 이유를 질문했다.
기존 파일 안에서 아래 변경을 추가했다. 환경별 독립 배치·밀집도·경로 여유·물리/학습 설정은 유지한다.

- 선택된 경로의 회전 구간별 pose·네 꼭짓점 생성은 `_motion_envelopes` Numba 함수로
  묶었다. 결과는 `_SweepPolygons(vertices, counts)` 배열로 `_pack_layouts`까지 전달한다.
  이전의 구간/꼭짓점별 Python tuple 생성→NumPy→tuple→NumPy 변환을 학습 경로에서 제거했다.
  경로 후보 검사 자체는 기존 GPU kernel을 사용한다. 이 변경이 sweep 준비까지 GPU로
  옮겼다는 뜻은 아니다. 작은 가변 길이 sweep 준비는 컴파일된 CPU 배열 연산이다.
- `_pack_layouts`는 물체별 half extent 조회를 catalog 배열의 일괄 인덱싱으로 바꾸고,
  예약 sweep을 polygon별 변환 없이 배열 블록으로 복사한다. Scalar/debug용 임의 다각형
  입력은 유지한다. 축·충돌·점유율과 후보 진행/부분 재배치는 기존 device kernel을 사용한다.
- `_episode_proposals`는 환경×물체의 `(row, id)` Python tuple 목록 대신 평탄화한 ID와
  길이/offset 배열을 사용한다. 정적 지지 높이는 catalog당 한 번 계산하여 GPU에 캐시하고
  인덱싱한다. 높이를 float32로 먼저 빼면 원래 Python double 식을 마지막에 cast할 때와
  약 1.2e−7 m 차이가 생겨, 기존 순서대로 한 번 계산한 후 캐시하도록 했다.
- 로봇 시작 영역은 blocker pose·half extent의 GPU tensor에서 회전된 box의 앞 경계를
  직접 계산한다. 공통 규칙 검사는 환경별 반복 대신 batch당 한 번 한다. 빈 시작 영역과
  비정상 수치는 계속 거절하며, IK·elbow/전완·hand–arm 충돌 승인 조건은 유지한다.
  FP32 tensor로 범위를 계산하므로 이전 double 계산 후 cast와 작은 반올림 차이는 가능하다.

Convex hull의 용도:

1. 배치 중 이동/pivot 경로가 지나가는 영역을 확보한다. 직선 이동은 두 끝 footprint의
   hull, 회전은 최대 5° 구간의 끝 footprint hull과 corner arc 오차 여유를 사용한다.
   시작/끝 pose만 겹치지 않더라도 회전 중 다른 물체에 닿는 경우를 제외하기 위한 계산이다.
2. Reward의 clearance 계산에서는 기울어진 3D OBB의 XY 투영 외곽과 A₀ 교차 면적을 구한다.
   기울어진 box의 평면 투영은 항상 네 꼭짓점 사각형이 아니므로 전체 투영 hull이 필요하다.
3. USD의 collider는 실제 PhysX 접촉/동역학 계산에 사용된다. 위 코드가 물리 collision
   mesh를 매 reset 다시 cooking하는 것은 아니다. USD physics 속성이 선반 점유율이나
   재배치 경로의 여유를 자동으로 정의하지 않는다. 현재 배치용 box/sweep은 실제 collider의
   보수적인 근사이며, 가용 공간을 실제보다 좁게 판단할 수 있고 조작의 물리적 성공 보장은 아니다.

Python의 환경별 generator/역할·구성 선정, 후보 완료 큐와 `.cpu()`/`.tolist()` 승인 확인은
아직 남아 있다. Warp의 GPU thread 내부 반복, Numba의 컴파일된 CPU 반복, 제어 시간축의
substep 반복은 Python으로 개별 물체 데이터를 반복 처리하는 비용과 구분해야 한다.
남은 generator/queue를 공통 batch state와 mask로 전환하는 것은 추가 구조 개선 지점이다.

CPU 전체 검사 184개 중 177개 통과, CUDA/native USD용 7개 skip.
Isaac Sim 내장 Warp 1.8.2 GPU 관련 검사 40개가 모두 통과했다. 새 검사는 이동·pivot·
다단계 sweep과 scalar hull의 일치, 배열/tuple 경로의 동일 packing, device pose의 host
읽기 없이 캐시 높이 사용, 회전 box의 GPU 시작 영역과 scalar footprint 범위의 일치를 포함한다.

RTX 3090·Warp 1.8.2에서 실제 catalog의 동일 seed 1024개 배치를 비교했다. 모든 역할·
active ID·생성 XY/yaw가 변경 전과 정확히 일치했다. JIT warm-up 후 다른 검증 작업 없이
실행 순서를 번갈아 3회씩 측정한 배치 생성 시간은 평균 **1.721→1.567초(약 9% 감소)**였다.
별도 cProfile에서 Python 수준 호출 수는 약 215만→163만 회로 줄었다. 이 수치는 배치 생성만의
비교이며 전체 iteration의 개선율이 아니다.

실제 1024환경·8 iteration도 exit 0으로 종료했다. Native collection+learning 평균은
**9.184초**, logging/checkpoint 포함 학습 시간은 74.03초였다. 직전 코드의 첫 8 iteration
평균 9.065초와 비슷하며, 이번 변경으로 전체 iteration 시간이 개선됐다고 결론내리지 않는다.
FP32 시작 영역 계산과 물리/학습 궤적 차이, 실행별 변동이 있는 단기 측정이다.
비활성 물체 위치 오차 최대 2.38e−7 m, 6D 속도 norm 최대 9.77e−6도 확인했다.
연구 조건·물리 주기·epoch를 완화하지 않았고, 1–2초/iteration 목표는 아직 달성하지 못했다.

재현 자료: `/tmp/contact_array_bench.py`, `/tmp/contact_array_bench_final.log`,
`/tmp/contact_array_bench.json`, `/tmp/contact_arrays_cpu.log`, `/tmp/contact_arrays_gpu.log`,
`/tmp/contact_arrays_native_1024.{json,log}`. 비교 원본은 `/tmp/contact_events_before_arrays.py`다.

### 2026-10-14 — Isaac Lab controller/action class로 제어 경로 통일

사용자가 UR5e는 OSC, hand는 joint position controller를 사용하고 예제의 controller
class를 따라야 한다고 명시했다. `ActionsCfg`를 `arm_action` 12D와 `hand_action` 6D로
분리했다. `ArmOscAction`은 native `OperationalSpaceControllerAction`을 상속하고,
`HandPositionAction`은 native `JointPositionAction`을 상속한다. 두 클래스 모두 부모의
`apply_actions()`를 호출한다. OSC 수식은 복제하지 않고, native 클래스가 실제 TCP 속도·
Jacobian·mass matrix·gravity를 이용해 계산한다. 커스텀 implicit damping과 32회
Newton–Schulz 계산은 삭제했다. 새 파일 없이 `mdp/actions.py`는 207줄로 정리했다.

Native OSC 예제처럼 full inertia decoupling을 사용한다. 이에 따라 기존 숫자 범위의
stiffness 채널은 물리 강성이 아닌 관성 정규화 gain이 된다. 현재 제어 규약은 위
"Action과 제어" 절을 따른다. 이전 custom controller 기록은 당시 실험 이력으로 남긴다.
관측의 관절 순서·18D action 순서와 F/T 보상은 유지한다. MDP는 native ActionManager의
전체 action tensor와 두 term의 환경별 invalid flag를 사용한다.

검증:

- CPU 검사 178개 중 171개 통과, native/CUDA 의존 검사 7개 skip.
- Isaac Sim에서 native OSC의 상대 목표 latch·관성/감쇠 계산과 부분 reset 후
  transition CUDA graph 일치 검사를 통과했다.
- 64개 독립 초기 자세에 최소·중간·최대 gain을 분배하여 목표를 고정하고 120 Hz에서
  240 step(2초)을 실행했다. 최대 위치 오차 0.0477 mm, 회전 오차 0.003323 rad(0.190°),
  최대 arm 관절 속도 0.009463 rad/s였다. 초기 workspace clipping·torque saturation·
  invalid control은 없었다. Native OSC 출력과 적용 토크 일치 및 선택 reset도 검사했다.
- RTX 3090에서 1024환경·8 iteration(seed 0) 학습을 exit 0으로 완료했다.
  collection+learning은 평균 7.621초/iteration(6.496–12.482초), 총 학습 61.31초였다.
  직전 9.184초와 비교하면 이번 실행의 전체 시간은 줄었지만 제어 응답과 reset 빈도도
  바뀌었으므로 동일 궤적의 연산 최적화 결과로 해석하지 않는다. 첫 iteration을 제외한
  action 적용 시간은 arm+hand 약 1.080초/iteration으로 직전 약 0.990초보다 짧지 않았다.
  Native inverse 등 제어 계산이 없어지는 변경은 아니다. 1–2초 목표는 아직 미달이다.

이는 짧은 실행 및 비접촉 hold 검증이며 장기 접촉 안정성·학습 수렴을 검증한 것은 아니다.
재현 자료: `/tmp/contact_native_controller_check.{py,json,log}`,
`/tmp/contact_native_action_unit.{py,log}`, `/tmp/contact_native_actions_cpu.log`,
`/tmp/contact_native_actions_profile.py`, `/tmp/contact_native_actions_1024.{json,log}`.

### 2026-10-14 — Native 기능 재사용 점검과 불필요한 재구현 제거

사용자가 controller 외에도 직접 만든 부분을 점검하고, 공식 예제의 흐름과 Isaac Lab
기능을 최대한 사용하도록 요청했다. 앞으로 일반 기능은 설치된 checkout의 public
class/config 및 예제를 먼저 확인하고, 연구 요구나 검증된 병목이 있는 부분만 확장한다.
검토 표와 구체적인 공식 소스 링크는 `src/contact_manipulation/README.md`의
"Native 기능과 연구 확장의 경계"에 기록했다. 기존 프로젝트와 새 파일 수는 건드리지 않았다.

이번 변경:

1. Hand의 수동 soft-limit affine mapping을 제거했다. 공식 in-hand 예제가 사용하는
   `JointPositionToLimitsAction`이 입력 clipping·관절 한계 변환·position target 적용을 맡는다.
   연구 adapter에는 nonfinite 처리·측정 q 기준 position lead 제한·선택 reset만 남긴다.
2. 관측·기하·F/T 보상에서 사용하는 tensor quaternion 곱/회전/켤레는
   `isaaclab.utils.math`를 호출한다. Native 함수의 동일-shape 요구를 맞추는 broadcasting만
   adapter로 남긴다. CPU 검사도 수식 복사본 대신 checkout의 실제 math 모듈을 로드한다.
3. `FrameTransformerCfg`가 native `FrameTransformer`를 직접 생성한다.
   현재 `clone_in_fabric=False`의 USD cloning에서는 `_ReplicaSensorBase`와
   `TaskFrameTransformer`가 필요 없어 삭제했다. `ContactSensor`는 native 초기화·force/history를
   사용하고, 측정된 hot path의 전체 행 update 최적화와 금지 접촉 필터 구성만 유지한다.
4. 비활성 물체의 중력 상쇄는 `permanent_wrench_composer.set_forces_and_torques()`에
   등록한다. 직접 PhysX force를 제출하던 `write_data_to_sim()` 재구현과 COM/force/index
   캐시를 제거했다. Force 적용과 instantaneous wrench 합성은 native collection에 맡긴다.
   Permanent composer는 주차용으로 예약한다. 같은 timestamp에서 pose를 다시 쓰면 composer의
   frame cache도 reset한다. SDK의 별도 COM-pose cache 무효화 누락에 대한 작은 보정은 유지한다.
5. Minibatch가 상한 이내이면 `super().update()`로 native RSL-RL PPO를 실행한다.
   기본 1024환경 × 32 step ÷ 4 minibatches = 8192이므로 native 경로다.
   상한을 넘을 때만 기존 microbatch gradient accumulation을 쓴다. 공유 GNN/attention,
   bounded tanh-Gaussian, reset 직전 상태의 timeout bootstrap은 연구 adapter로 남긴다.
6. Warp mesh 생성도 공식 `convert_to_warp_mesh()`를 사용한다.

대조 후 유지한 연구 코드:

- 배치/초기 자세: 기본 uniform reset 및 DifferentialIKController는 FK를 포함한 후보 탐색,
  Target·Blocker 조건, elbow·충돌·재배치 경로를 정의하지 않는다. 물리 scene에 후보를 쓰지
  않고 검증하는 GPU FK/IK·배치 sampler를 유지한다.
- Vision: 기본 RayCaster는 정적 단일 mesh 제약이 있지만 **MultiMeshRayCasterCamera와
  raycast_dynamic_meshes는 동적 mesh를 지원한다.** Custom 유지 이유를 동적 mesh 미지원으로
  설명하면 안 된다. 후보 pose 입력, 무가림 role 실루엣 분모, object-only projected culling,
  ray별 depth/label 일관성과 visible-surface OBB라는 차이 때문에 현재 fused 구현을 유지한다.
  Native dynamic kernel의 atomic depth와 label 쓰기 관련 FIXME도 확인했다.
- History: native CircularBuffer/ObservationManager는 첫 sample을 과거에도 반복한다.
  우리 규약의 previous=0, newest-first와 reset 이전 terminal observation 보존을 위해
  두 sample을 TaskState에서 관리한다. 보상/종료 시 공유하는 누적 상태도 연구 전용이다.
- F/T: 원시 반력은 native articulation에서 읽고 SI-200-8 측정 모델과 Newton–Euler 보상은
  연구 코드로 유지한다. GNN/attention과 연구 reward/termination 수식도 동일하다.

Native 전환 중 확인한 차이:

- Native PPO는 기존 별도 update의 추가 loss 지표 대신 value/surrogate/entropy를 기록한다.
  SDK 내부 minibatch별 `.item()`과 선형대수 동기화까지 제거한 구현이라고 주장하지 않는다.
- Native permanent wrench는 world force를 local axes로 바꿔 저장한다. 최초 1024환경
  8 iteration 학습은 완료했지만, 격리된 주차 물체의 최대 이동 0.2685 mm가 예전 임시
  profiler의 0.1 mm 기준을 넘어 후속 assertion이 실패했다. 이를 숨기지 않고 별도 진단했다.
  4환경·960 physics step(8초)에서 추가 torque·각속도는 0, 최대 이동 0.1335 mm,
  선속도 약 4.03e−5 m/s, 회전된 force의 차이는 약 1e−6 N 이하였다. FP32 local/world
  force 변환을 포함한 native 경로의 수치 오차로 판단한다. 검증 기준을 비활성 주차 물체에
  한해 위치 1 mm·속도 1e−3로 명시했다. 물체 간 주차 간격 0.32 m보다 충분히 작으며,
  각 episode reset에서 pose/속도가 다시 초기화된다. Active 물체의 중력이나 접촉 조건은
  완화하지 않았다. 이 미세 오차가 완전히 사라졌다고 주장하지 않는다.

최종 검증: CPU 검사 180개 중 172개 통과·native/CUDA 8개 skip. Isaac Sim 내에서
native OSC, hand limit mapping/lead/reset, transition CUDA graph, contact update 및
parking composer 위임 검사 5개 통과. Native FrameTransformer의 1024행 출력도 검사했다.
1024환경·8 iteration에서 microbatch fallback을 호출하면 실패하도록 계측하여 기본 PPO가
실제로 native update를 사용하는 것을 확인했고, 실행은 exit 0으로 완료했다.
평균 7.628초/iteration, 총 학습 61.35초였다. 기존 7.621초 수준과
유사하며 이번 구조 정리를 학습 속도의 획기적 개선으로 주장하지 않는다.
주차 물체 최대 이동 0.2685 mm·속도 norm 7.014e−5로 명시한 검증 범위 안이었다.

자료: `/tmp/contact_native_reuse_cpu.log`, `/tmp/contact_native_reuse_profile.py`,
`/tmp/contact_native_reuse_1024.{json,log}`, `/tmp/contact_native_parking_diagnostic.{py,log}`.
최초 주차 assertion 실패 기록은 `/tmp/contact_native_reuse_first_1024.{json,log}`에 보존했다.

### 2026-10-14 Sweeping-Policy 예제와 실행 주기 일치

사용자가 예제와 같은 주기를 요청하여 새 프로젝트의 `ur5e_env_cfg.py`를
`sim.dt=0.01`, `decimation=2`, `render_interval=2`로 변경했다.
물리 및 OSC/hand 적용은 100 Hz, 정책은 50 Hz다. 이전 120 Hz/30 Hz 설정을 대체한다.
금지 접촉 sensor의 history_length도 2로 맞춰 현재 정책 step의 모든 물리 substep만 검사한다.
Hand 목표 제한은 physics dt, MDP 시간·차분 및 play pacing은 env.step_dt를 사용한다.
Episode timeout은 기존 8초(이제 400 policy step), rollout은 기존 32 step(0.64초)이다.
제어 방식·solver·reward 계수·PPO 설정은 이번 주기 변경의 대상이 아니다.

두 Sweeping-Policy 예제의 dt/decimation/render_interval과 실제 설정 메서드의 결과를 비교했다.
기존 reward·state·sensor runtime·native adapter CPU 검사 35개가 통과했고 diff 공백 검사도 통과했다.
새 주기로 Isaac Sim 실행 및 처리량은 아직 측정하지 않았다. 앞선 7.628초/iteration 기록은
120 Hz/30 Hz 실행의 결과이며 새 설정의 성능으로 인용하지 않는다.

### 2026-10-14 물체 pool 및 관측 슬롯 30개로 축소

물리 step 비용을 줄이기 위한 사용자 요청에 따라 새 프로젝트의 물리 pool을
10종×4개=40개에서 10종×3개=30개로 축소했다. 1024환경에서 물체 rigid body는
40,960개에서 30,720개로 줄어든다. Target 종류의 중복 금지는 유지하므로 실제 선반의
active 물체 상한은 28개다. 매 episode를 정확히 30개로 채우는 설정은 아니다.

MDP·spawn rules·reset 버퍼·초기 vision·GNN의 기본 관측 슬롯을 모두 30개로 맞췄다.
Observation은 30×10+2×72=444D이며 Target/Blocker 슬롯 0/1과 action 18D는 유지한다.
기존 40-slot checkpoint와 model_cfg가 달라 기본 resume/play는 호환되지 않는다.
선반 전체 점유율 35%·지역 점유율 10%, 역할·가시성 조건은 변경하지 않았다.

검증: 기존 CPU 검사 180개 중 172개 통과, Isaac Sim/CUDA 필요 검사 8개 skip.
기존에 실제 USD에서 추출해 저장한 catalog의 네 번째 복제본을 제외한 30개 형상으로
seed 0–15의 배치 생성 및 validate_sample을 확인했다. 16/16개 배치가 통과했고
active 수는 15–18개, 최소 전체 점유율은 35.07%였다. 작성된 catalog도 10종×3개임을 확인했다.
실제 Isaac Sim의 새 pool 생성·학습 처리량은 아직 측정하지 않았다.
CPU 결과는 `/tmp/contact_30_objects_cpu.log`에 기록했다.

### 2026-10-14 물체 10개 실험

사용자가 10개로 줄여 실험하도록 요청했다. 물리 pool을 10종×1개로 만들고,
`SpawnRules`의 min/max_objects=10, all_asset_types=True로 매 episode 정확히 10개를
배치한다. Target 1개·Blocker 1개·주변 물체 8개이며 종류 중복은 없다.
1024환경의 물체 rigid body는 10,240개다. 물리 100 Hz·정책 50 Hz는 유지한다.

저장된 실제 USD 형상에서 10종 footprint 합은 0.0711358692 m², 유효 선반 면적은
0.3256 m²로 전체 점유율은 21.8476%다. 35% 조건을 유지하면 모든 reset이 실패하므로
이번 10개 실험의 전체 최소 점유율은 20%로 조정했다. 선반 영역, 지역별 10% 점유율,
Target/Blocker 조건, 재배치 공간, 가시율 기준은 유지했다. 이 변경은 물체 수와 함께
장면 밀집도를 낮추므로 단순한 동일 작업량 최적화로 해석하지 않는다.

MDP/reset/vision/GNN 슬롯은 10개로 통일했고 observation은 10×10+2×72=244D다.
이전 30/40-slot checkpoint는 기본 resume/play와 호환되지 않는다.
README와 기존 검사 fixture를 갱신했으며 파일은 추가하지 않았다.

검증: CPU 검사 180개 중 172개 통과, Isaac Sim/CUDA 검사 8개 skip
(`/tmp/contact_10_objects_cpu.log`). 실제 USD에서 추출해 저장한 catalog로 seed 0–31의
배치 생성 및 validate_sample을 실행해 32/32 통과했다. 모두 10종·10개이며 전체 점유율
21.85%, 지역별 점유율 최솟값 10.55%였다. 작성된 scene catalog도 10종×1개임을 확인했다.
이번 검증은 배치 생성과 CPU 모델/MDP에 대한 것이며 실제 Isaac Sim 실행, IK/vision을
포함한 전체 reset 및 학습 처리량은 아직 재측정하지 않았다.

### 2026-10-14 물체 축소 후 병목 재계측: 로봇 물리와 반복 COM 조회

사용자는 물체 30→10개 변경의 효과가 작다며 현재 느린 원인 분석을 요청했다.
Production 동작은 변경하지 않고 `/tmp`의 native train 복사본에 함수별 inclusive/exclusive
wall timer, cProfile, PyTorch CPU/CUDA trace 및 일시적인 COM-cache 비교를 추가했다.

최근 사용자 로그에서 첫 iteration을 제외한 30개 실행(19-18-24, 5회)은 평균 4.004초,
10개 실행(19-32-12, 6회)은 3.772초였다. 둘 다 1024환경·100 Hz physics·50 Hz policy다.
약 5.8% 차이지만 배치·밀집도·학습 궤적이 다르고 timeout 전 짧은 구간이므로 엄밀한
동일 작업량 비교나 장기 평균으로 해석하지 않는다.

RTX 3090에서 현재 10개 설정으로 16 iteration을 계측했다. 첫 warm-up iteration과
cProfile을 켠 index 2를 제외한 14회 평균은 3.995초(collection 3.116, PPO 0.879)였다.
대량 timeout이 발생한 index 12를 추가로 제외한 13회 평균은 3.748초다.
그 구간의 주요 inclusive 시간/iteration은 다음과 같다. 일부 중첩되므로 합산하지 않는다.

| 구간 | 시간 |
|---|---:|
| Physics step (64회) | 1.299 s |
| PhysX fetch_results (physics 내부) | 1.280 s |
| PPO update | 0.880 s |
| Arm apply / action process | 0.391 / 0.107 s |
| OSC Jacobian 준비 (arm apply 내부) | 0.236 s |
| OSC.compute (arm apply 내부) | 0.055 s |
| MDP 기하·상태·센서 읽기 | 0.422 s |
| Scene update | 0.190 s |
| Contact buffer 읽기/갱신 (위 센서/scene 구간에 포함) | 0.179 s |
| MDP.advance | 0.100 s |
| 소규모 reset 전체 | 0.102 s |

Index 12의 iteration은 7.209초, reset 3.579초였다. Reset 내부는 vision 1.937초,
배치 생성 0.567초, IK 0.864초다. 다른 iteration의 vision 비용은 평균 0.013초였다.
따라서 ray casting은 대량 reset spike의 주요 요소지만 평상시 step 병목 전체를 설명하지 않는다.

Native 자원 수: 로봇당 41 rigid bodies·18 joints, 환경당 27 sensors(17 tactile,
9 forbidden-contact, 1 frame transformer). 물체 10개로 줄여도 로봇 링크만 41,984개다.
Contact 실제 읽기/갱신은 정상 iteration당 1120회(17×32 + 9×64)이며 각 호출은
1024환경을 tensor로 처리한다. 환경별 Python loop가 아니라 센서별 SDK 호출·buffer 연산이다.

별도 GPU trace는 실제 env.step 4회(physics 8회)를 측정했다. kernel duration 합에서
가장 큰 단일 항목은 `artiSolveInternalConstraintsTGS1T` 39.818 ms/144회였다.
144=8×(position 16 + velocity 2)로 현재 solver 설정의 반복이 trace에 나타난다.
그 외 `convexPlaneNphase_Kernel` 21.835 ms, `solveBlockUnified` 11.339 ms,
`artiPropagateImpulses2TGS` 10.295 ms, `stepArticulation1TTGS` 9.469 ms였다.
Kernel 누적 시간은 overlap·계측 overhead가 있어 전체 wall-time 분해와 같지 않다.
Physics 직전 CUDA synchronize를 넣은 16 policy step 진단에서 선행 CUDA 대기는
합계 1.63 ms, physics.step은 669.15 ms였다. Fetch 비용의 대부분을 앞선 정책/제어의
미완료 GPU 작업 탓으로 돌릴 근거는 없었다.

구체적인 불필요 조회 후보도 확인했다. 현재 SDK의 `ArticulationData.body_com_pose_b`는
simulation timestamp마다 `root_physx_view.get_coms().to(device)`를 호출한다.
Runtime에서 get_coms 반환은 CPU, shape [1024,41,7]이었다. 우리 OSC의 TCP Jacobian
COM→link 보정과 MDP의 world COM 계산이 이 경로를 사용한다. 움직이는 world COM은
매번 갱신해야 하지만 link-local COM은 mass/COM randomization이 없는 현재 구성에서는
고정이다. 모든 링크의 고정 COM 정보를 CPU에서 읽고 GPU로 복사하는 비용이 반복된다.

임시 프로세스 안에서 native/cached/native/cached를 16 step씩 교대로 실행했다.
Reset은 모두 0회였고 local COM 값이 끝까지 동일함을 확인했다.
Native 1.423/1.468초, cached 1.248/1.305초로 평균 env.step 시간은 1.446→1.276초,
약 11.7% 감소했다. 32 step 환산 약 0.339초의 후보 절감이며 PPO·전체 학습 개선율이 아니다.
연속된 서로 다른 물리 구간의 짧은 비교이므로 같은 초기 상태 replay 및 관측/토크 동등성,
mass/COM 변경 시 invalidation 검증이 정식 적용 전에 필요하다. SDK/production에는
캐시 패치를 적용하지 않았다.

다음 우선순위는 고정 local COM 재조회 제거, articulation solver 16/2 비용과 접촉 정확도의
비교, sensor별 반복 SDK 호출/작은 kernel 집계, 대량 reset vision/IK 순이다.
PPO 자체도 약 0.88초이므로 물체 수 또는 ray casting만 줄여 1–2초를 보장할 수 없다.

자료: `/tmp/contact_detailed_profile.py`, `/tmp/contact_detailed_10_1024.{json,log}`,
`/tmp/contact_detailed_10_1024_cpu.txt`, `/tmp/contact_trace_profile.py`,
`/tmp/contact_trace10_1024_trace.json`, `/tmp/contact_trace10_1024_torch.txt`,
`/tmp/contact_trace10_1024_com_ab.json`, `/tmp/contact_trace10_1024_fenced.json`.
첫 실행은 16 iteration 수집 후 추가 trace를 inference_mode 없이 호출한 임시 계측 오류로
종료코드 1이었다. Trace/비교 함수를 inference_mode로 수정한 두 번째 실행은
3 iteration 및 trace·대기 분리·COM 비교까지 완료하고 종료코드 0이었다.

### 2026-10-14 — Tactile 17개 링크의 ContactSensor 통합

사용자 요청: 링크별 17D 접촉 observation을 유지하며 native ContactSensor를 묶고,
자세/금지 접촉용 센서도 통합 가능한지 확인한다.

`UR5eContactSceneCfg.tactile_contact` 하나가 정확한 17개 링크 이름의 regex로 body view를
생성한다. `SensorReader`는 초기화 때 `find_bodies(..., preserve_order=True)`로
`TACTILE_BODY_NAMES` 순서의 GPU index를 저장한다. 매 측정은 native `[N,17,3]`를 한 번
읽고 `index_select`로 정렬한다. 기존 0.01 N binary 판정, 유효성 처리, 17D observation,
부분 reset 규칙은 유지한다. Native body 순서가 observation 순서와 실제로 달랐으므로
이 명시적인 정렬이 필요하다. 새 production 파일은 추가하지 않았다.

금지 접촉 9개는 유지한다. 현재 checkout의 공식 ContactSensor 문서는 상대별 force
filter를 one-to-many로 제한한다. 각 arm/spacer body가 선반·물체·바닥과 접촉한 힘을
상대별/physics substep별로 검사하는 현재 의미를 보존하려면 개별 sensor가 필요하다.
Unfiltered `[N,9,3]`로 묶는 것은 가능하지만 상대별 힘의 상쇄와 접촉 상대 구분 때문에
현재 failure 판정과 같지 않다. 초기 IK의 기하 충돌 검사는 이 물리 contact history와 별개다.
근거: `IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`의
class docstring, `IsaacLab/scripts/tutorials/04_sensors/contact_sensor.py`의 grouped hind feet.

검증:
- CPU suite: 181 tests, 173 통과·8 skip. 링크 순서를 뒤집은 native buffer fixture에서
  canonical 17채널 복원, 일부 환경 reset, nonfinite 측정, 반대 방향 partner force 검사 통과.
- Native GPU 4환경: 통합 sensor와 기존 방식의 17개 reference sensor를 동일 scene에
  붙여 실제 물체 접촉을 비교했다. 17채널 모두 nonzero 접촉을 경험했고 34 physics samples의
  힘 차이 최댓값 0.0, binary 판정 동일. 환경 1·3 reset 후 0·2 측정값 유지 확인.
- 임시 검증 자료: `/tmp/contact_grouped_validation.py`,
  `/tmp/contact_grouped_validation.json`, `/tmp/contact_grouped_validation.log`.

1024환경·물체10개·100Hz physics/50Hz policy에서 16 iteration 학습도 종료코드 0으로
완료했다. 초기화/의도적 cProfile iteration(0·2)을 제외한 14회 평균은 3.965초였고,
이전 17개 센서 실행의 3.995초와 큰 차이는 없었다. 대량 reset(index12)은 7.167초였다.
대량 reset도 제외한 13회 평균은 3.719초(이전 3.748초)다. 짧은 별도 실행의 비교이므로
전체 iteration 개선율을 안정적인 수치로 단정하지 않는다.

변경의 직접적인 효과는 정상 iteration당 native contact buffer update가 1120→608회,
접촉 buffer 구간 평균 0.179→0.143초, 전체 SensorReader 구간 0.126→0.044초였다.
두 구간은 일부 중첩되므로 절감 시간을 합산하면 안 된다. Scene sensor 인스턴스는
27→11개(1 tactile, 9 forbidden, 1 frame transformer)이고 실제 물리 링크 수는 그대로다.
전체 학습 속도의 주요 병목인 물리·PPO·대량 reset은 이 변경으로 해결되지 않는다.
자료: `/tmp/contact_grouped_profile.py`, `/tmp/contact_grouped_1024.json`,
`/tmp/contact_grouped_1024.log`, `/tmp/contact_grouped_1024_meta.json`.

### 2026-10-14 — 금지 접촉 센서 제거 및 선반 pose 기반 실패 판정

사용자는 arm/spacer 접촉 자체를 실패로 볼 필요가 없으며, 선반이 움직인 결과로 종료하기로
결정했다. 기준은 초기 위치 대비 1 cm 또는 초기 회전 대비 2° **초과**다.

- `ForbiddenContactSensor`와 관련 9개 scene sensor, 상대 필터 생성, history 읽기/집계,
  SensorBatch의 forbidden flag, 상수 및 사용하지 않는 helper를 제거했다.
- 선반은 native `RigidObjectCfg`로 변경하고 kinematic을 해제했다. 중력과 원본 collision을
  사용한다. 질량은 `examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/shelf_sweep_env_cfg.py`
  의 설정과 같은 100 kg이며, 실측 질량은 아니다. 관성은 PhysX의 collider 기반 계산을 쓴다.
- 실제 USD는 `/Rack/rack`에 RigidBodyAPI 하나가 있고 MassAPI는 없었다. 현재 SDK의
  UsdFileCfg.mass_props는 기존 MassAPI만 수정하므로 단순 설정 시 경고와 함께 미적용되어
  PhysX 질량이 35.8 kg이었다. 기존 scene.py에 작은 spawn_shelf adapter를 두고 native
  spawn_from_usd + define_mass_properties를 호출해 명시적으로 적용했다. 원본 USD는 수정하지
  않으며, 최종 PhysX에서 100 kg을 확인했다.
- `TransitionBatch.shelf_pose` [N,7]와 `TaskState.initial_shelf_pose`를 사용한다.
  초기 pose는 reset에서 해당 환경 행만 저장한다. GPU에서 위치 차이 제곱합과 normalized
  quaternion 내적의 절댓값을 비교한다. 각도 계산의 acos 없이 2° 기준을 검사하고 q/-q를
  동일 회전으로 취급한다. Nonfinite/zero quaternion은 invalid-state 실패다.
- 기존 failure reason slot 5를 shelf displacement로 교체했다. 실패 보상 -100과 실패 우선
  terminal 처리는 그대로다. Shelf pose는 privileged 종료 판정에만 쓰며 observation에
  추가하지 않는다. OSC pose 제어와 F/T·17채널 tactile 입력도 그대로다.
- apply_episode_batch에서 선반 pose를 default_root_state + env_origin으로 복원하고 속도를
  0으로 만든다. 빌드 중에는 선반을 쓰지 않으므로 후보 생성 실패 시에도 원래 상태가 유지된다.
  초기 pose reference와 다른 환경의 상태는 부분 reset으로 변경되지 않는다.
- Scene sensor는 2개(tactile 통합 + EEF frame)이며, UR5e contact-report 활성화도 제거했다.
  Self-collision 설정/초기 IK의 기하 검사는 이 변경의 대상이 아니다.

검증:
- CPU suite 182 tests: 174 통과, 8 skip. 위치·회전 임계값 양쪽, q/-q 및 normalization,
  nonfinite pose, env별 실패/보상, 초기 pose 부분 갱신, 물리 reset의 선택 행/속도 복원 검증.
- CUDA transition graph와 eager 결과 및 부분 reset 일치 테스트 1개 통과.
- Native 4환경에서 8초간 초기 선반의 최대 이동 약 0.0132 mm로 1 cm 미만.
  위치 2 cm/회전 3° 주입 시 해당 두 환경만 shelf failure flag가 켜지고 -100 적용 확인.
  Native manager 부분 reset에서 다른 환경 pose/reference 유지와 선택 행 속도 0 확인.
- 동적 응답 추가 검사: 한 환경에 2 m/s 초기 속도를 주고 0.2초 후 약 6.11 cm 이동 확인.
  최초 짧은 1 m/s·0.05초 검사에서는 1 cm 이동을 넘는다는 임시 테스트 가정이 실패했으며,
  더 긴 검사에서 실제 pose와 velocity를 함께 확인했다. 초기 설정을 kinematic으로 남긴
  상태는 아니다. 마지막 native 검증은 종료코드 0.
- 자료: `/tmp/contact_shelf_inspect.log`, `/tmp/contact_shelf_validation.py`,
  `/tmp/contact_shelf_validation.json`, `/tmp/contact_shelf_validation.log`,
  `/tmp/shelf_cpu_tests.log`, `/tmp/shelf_cuda_tests.log`.

최종 1024환경·물체10개·100/50 Hz에서 16 iteration 학습이 종료코드 0으로 완료됐다.
Scene sensor 수는 2개다. 일반 iteration당 contact buffer 조회는 608→32회,
해당 구간 평균 0.143→0.0167초로 줄었다. Warmup과 cProfile 구간(0·2)을 제외한
14회 평균은 3.735초(직전 통합 tactile+9 forbidden 실행 3.965초)였고,
대량 reset(index12)도 제외한 13회 평균은 3.479초(이전 3.719초)였다.
대량 reset iteration은 7.056초, reset 자체 3.620초다. 서로 다른 짧은 실행의 비교이며
센서 제거뿐 아니라 선반 동역학/종료 조건이 달라졌으므로 엄밀한 동일 trajectory A/B는 아니다.
일반 collection 2.603초, PPO 0.876초, physics.step 누적 1.289초로 전체 1–2초 목표에는
추가 병목 개선이 필요하다. 자료: `/tmp/contact_shelf_profile.py`,
`/tmp/contact_shelf_1024.json`, `/tmp/contact_shelf_1024.log`.

### 2026-10-14 — 센서 제거 후 1024환경 단계별 상세 프로파일

사용자 요청에 따라 학습 로직 변경 없이 실행 단계별 inclusive/self wall time과 호출 수를 계측했다.

조건: RTX 3090, PyTorch 2.7.0+cu128, CPU torch threads=8, 1024환경, 환경당 물체10개,
로봇41 bodies, 센서2개, dynamic 선반100 kg, physics100 Hz/policy50 Hz,
rollout32 step, PPO4 epochs×4 minibatches. 한 iteration은 32,768 transitions와
64회의 병렬 physics step이다. 아래 초 단위 수치는 한 환경/step이 아니라 iteration 합계다.

첫 32회 실행에서는 iteration3에 cProfile, iteration5에 torch CPU/CUDA profiler를 사용했다.
프로파일 후 PPO 시간이 약1.1초로 달라져, 프로파일러를 전혀 켜지 않는 새 프로세스로
동일한 32회 실행을 별도로 수행했다. 최종 시간 표는 두 번째 실행의 perf_counter 계측만 사용한다.
두 실행 모두 정상 종료했다. CUDA 이벤트/커널 분석은 첫 실행의 진단 자료로만 사용한다.

두 번째 실행: warmup(0)을 제외한 31회 평균 **3.877초**. 대량 reset iteration
12·24(각 iteration 합계1006·993환경 reset)를 분리한 일반29회 평균 **3.646초**,
중앙값3.613초, 범위3.361–4.160초다. 일반 iteration도 평균2.9회의 소규모 reset 호출을 포함한다.
대량 reset 2회 평균7.218초(7.124–7.311초)다. Native collect_time+learn_time 기준으로
logger/checkpoint는 포함하지 않으며, logger 자체는 평균0.0079초였다.

아래 행은 서로 겹치지 않는다. 하위 단계 표는 상위 행에 포함되므로 다시 더하지 않는다.

| 일반 iteration 단계 | 평균 초 | 비중 |
|---|---:|---:|
| 물리 simulation / PhysX | 1.2882 | 35.3% |
| PPO update + GAE | 0.9090 | 24.9% |
| Action 처리 + OSC/hand 제어 | 0.5356 | 14.7% |
| MDP 상태 읽기·공통 보상/종료 계산 | 0.4590 | 12.6% |
| 소규모 reset | 0.1753 | 4.8% |
| Rollout 정책 추론 | 0.1193 | 3.3% |
| Scene write/update | 0.1008 | 2.8% |
| Reward/observation manager 반환 | 0.0171 | 0.5% |
| Wrapper·storage·NaN 검사·runner bookkeeping 등 | 0.0418 | 1.1% |

제어 상세(0.5356초에 포함):

| 단계 | 평균 초 | 호출/iteration |
|---|---:|---:|
| action.process | 0.1215 | 32 |
| action.apply/arm.apply/osc.ee_jacobian | 0.2381 | 64 |
| action.apply/arm.apply/osc.ee_jacobian/data.local_com | 0.1826 | 64 |
| action.apply/arm.apply/osc.compute | 0.0566 | 64 |
| action.apply/arm.apply/osc.dynamic_quantities | 0.0277 | 64 |
| action.apply/hand.apply | 0.0118 | 64 |

MDP 상세(0.4590초에 포함):

| 단계 | 평균 초 |
|---|---:|
| mdp.read_transition | 0.3137 |
| mdp.read_transition/data.world_com | 0.1919 |
| mdp.read_transition/data.world_com/data.local_com | 0.1821 |
| mdp.read_transition/data.object_com | 0.0593 |
| mdp.read_transition/sensors.read | 0.0277 |
| mdp.read_transition/sensors.read/tactile.read | 0.0165 |
| mdp.advance | 0.1197 |
| mdp.advance/mdp.conditions | 0.0605 |
| mdp.advance/mdp.reward_components | 0.0172 |
| mdp.advance/mdp.projected_areas | 0.0097 |
| mdp.eef_geometry | 0.0121 |

Native termination manager 호출이 공통 MDP 갱신을 처음 수행하므로, 이 구간에는
센서·물체·로봇 데이터 읽기와 reward/termination 공통 계산이 함께 들어 있다.
Reward manager에서 위 계산을 다시 수행하지 않는다.

PPO 상세(update0.9023초에 포함):

| 단계 | 평균 초 |
|---|---:|
| model.actor | 0.3787 |
| model.actor/model.scene | 0.2598 |
| model.actor/model.masks | 0.0591 |
| model.actor/model.fusion_head | 0.0149 |
| model.critic | 0.0027 |
| ppo.backward | 0.4560 |
| ppo.clip_grad | 0.0154 |
| ppo.optimizer | 0.0122 |

각 minibatch8192개를 max_forward_batch_size1024로 8분할하여 GNN/attention forward는
update당128회다. Rollout에서는32회, GAE/timeout value는 별도다. Actor와 critic은 shared
encoder 결과를 재사용하며, 위 model.actor 시간에는 shared value head도 포함된다.

대량 reset 2회의 reset 구간만 평균3.7022초. 같은 iteration 전체7.2176초와 구분한다.

| Reset 단계 | 평균 초 |
|---|---:|
| 배치 생성 | 0.7166 |
| 초기 vision 전체 | 1.9575 |
| IK·후보 자세 검사 전체 | 0.8758 |
| 나머지 실제 FK 확인·쓰기·commit·MDP reset·manager 등 | 0.1522 |

IK 내부: solve_batch 0.1654초, 후보 충돌 검사0.5889초. 각각의 상위 전체0.8758초에 포함된다.
Vision 내부 _trace 함수의 host-return 시간은 약0.10초지만 GPU ray kernel 완료를 기다리는
후속 tolist()/동기화는 바깥 capture_batch에 잡힌다. 따라서 나머지를 Python 순수 계산 시간으로
해석하거나 ray casting이0.10초뿐이라고 결론 내리면 안 된다.

확인된 비용과 다음 후보:


- Tactile native buffer는32회, 0.0165초로 전체 일반 iteration의 약0.45%.
  직전9개 forbidden 포함 실행의0.143초에서 크게 줄었지만 전체 시간 중 비중은 작다.

- PhysX step1.2882초 중 fetch_results 경로1.2682초. 이는 물리 완료 대기/동기화 포함 wall time이며
  순수 GPU kernel 시간과 같지 않다. 별도 CUDA trace의 실제 kernel 누적은 convexPlaneNphase
  185.2ms, articulation internal constraints177.2ms, solveBlockUnified106.2ms,
  articulation impulse propagation98.2ms, stepArticulation83.5ms였다.
  Internal constraints1152호출=64physics×(position16+velocity2). 센서 개수와 별도로 남는 비용이다.

- Robot link-local COM 경로는 OSC0.1826초 + MDP0.1821초, 약0.365초.
  SDK ArticulationData.body_com_pose_b가 timestamp마다 get_coms().to(device)를 실행한다.
  PhysX SDK get_coms는 CPU tensor(device -1)를 만들며, CPU profile에서 articulation
  get_coms65회(64 substep+reset) 누적0.339초였다. Object COM도 별도0.0593초다.
  고정 local COM 재조회 제거는 우선 후보지만, 이 wall time 전체가 절감된다고 보장하지 않는다.

- PPO는 forward0.3787초와 backward0.4560초가 대부분이다. Forward 조각 수를 줄이는 후보는
  활성화 메모리와 처리량을 함께 측정해야 한다. 학습 epoch/rollout을 줄이는 방식과 구분한다.

- 대량 reset은 vision 약1.96초 및 IK 충돌 검사약0.59초가 우선순위다.


주의: perf_counter 수치는 CPU 호출 시작→반환 경과시간이다. GPU 연산은 비동기이므로
완료 대기가 후속 호출에 귀속될 수 있다. GPU trace의 CUDA user-range(stage::...)는
실제 kernel과 중첩되어 합산하면 부풀려진다. gpu_kernels_filtered.json은 이 annotation
범위를 제외했으며, 커널 누적 시간 역시 iteration wall time에 더하면 안 된다.

원자료/재현 스크립트: `outputs/contact_manipulation/stage_profile_20261014/`.
최종 `timing_only/iterations.json`, `timing_only/summary.json`, `timing_only/summary.txt`,
`metadata.json`, `cpu.pstats`, `gpu_ops.json`, `gpu_kernels_filtered.json`,
`profile_timing.py`, `profile_with_trace.py`, `commands.json`, 각 run log를 보관했다.
프로파일 스크립트는 고정 output 경로를 사용하므로 재실행하면 해당 계측 자료를 덮어쓴다.

### 2026-10-14 — Sweeping-Policy `min` 브랜치 물리·제어 비교

- 검토 기준: 로컬 examples/Sweeping-Policy-DRL-Sim-Train의 origin/min,
  `57caa46bd3fe9f8fb00fc0160d1c31beb23b4c78` (2026-08-12).
  원격 ls-remote는 GitHub 인증 실패로 최신 여부를 확인하지 못했다.
  기존 main checkout을 변경하지 않고 git archive를 /tmp/sweeping_min_review에 추출했다.
- 가장 가까운 예제는 Isaac-Sweep-Object-UR5e-Random-v2:
  sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py.
  공통 UR5eSweepEnvCfg를 상속하고 arm implicit stiffness/damping=0 및 공식
  OperationalSpaceControllerActionCfg만 교체한다. 별도 physics step override가 없다.
- 예제는 pose_abs 7D + binary gripper 1D, fixed impedance 6축 stiffness=100,
  full inertial decoupling 및 gravity compensation, nullspace 없음이다.
  우리의 pose_rel + variable_kp + 독립 hand joint position 18D는 연구 요구 차이이며
  예제의 action 공간으로 바꿀 이유는 아니다. 두 환경 모두 pose 제어에 contact sensor가 필요 없다.
- 물리 dt=.01/decimation=2는 동일하다. 예제 물체 solver는 position16/velocity1,
  현재 물체와 articulation은 16/2다. 예제 robot의 articulation solver 및 self collision은
  Python config에서 override하지 않는다. 외부 UR5e_v4.usd가 로컬에 없어 해당 실제 값은 미확인이다.
  물체의 16/1을 robot 설정이라고 해석하면 안 된다.
- 예제 bounce threshold=.2, friction correlation=.00625, rigid patch capacity=5*2**17,
  found/lost aggregate pairs=2**28, total aggregate pairs=2**18이다.
  현재는 bounce=.05, 나머지는 로컬 Isaac Lab 기본값(.025, 5*2**15, 2**25, 2**21)이다.
  버퍼 capacity는 반복 횟수 설정이 아니므로 그대로 복사해서 속도 향상을 기대하지 않는다.
- 예제의 native OSC 역시 매 physics step Jacobian/mass/gravity/state를 조회한다.
  우리의 추가 COM→TCP Jacobian/속도 보정은 좌표계 일관성을 위해 존재한다.
  앞선 계측에서 비싼 body_com_pos_b 조회를 줄이려면 보정 제거 대신 고정 local COM
  캐시 및 질량/COM 변경 시 무효화를 검토한다. 이번 검토에서는 구현을 변경하지 않았다.
- 예제 reset은 2×3 slot 순열/XY jitter/yaw와 pose write이며, target·순열·XY jitter·방향은
  reset batch에서 공유된다. Python object loop도 있다. 독립 clutter 생성·IK·가시성 검증이
  필요한 현재 reset의 대체 구현이나 GPU 최적화 기준으로 그대로 사용하지 않는다.
- 예제 PPO는 일반 MLP [256,128,64], rollout36, epochs8/minibatches4다.
  현재 rollout32 및 GNN/attention 정책과 iteration 수치만으로 성능을 비교할 수 없다.
- 정적 검증: python -m unittest discover -s /tmp/sweeping_min_review/tests
  -p test_v2_osc_contract.py → 3개 중 2 pass/1 fail. 실제 stiffness는 전축100,
  테스트 기대값은 translation80/rotation40인 불일치다. 예제 문서에도 이 차이가 설명되어 있다.
  min 환경의 실제 simulation/성능 측정은 이번에 수행하지 않았다.


### 2026-10-14 — 공통 scene round·초기 vision 공유, 매 reset 연속 EEF randomization

사용자 승인: 전체 환경이 한 초기 물체 배치/가상 시점을 공유하고, 중간 종료는 그 배치로
복원한다. 이후 명시한 보정에 따라 로봇 자세 bank는 도입하지 않는다. 매 episode마다
Blocker 앞 허용 영역에서 EEF 위치를 새로 뽑고 GPU IK·elbow-up·전완 방향·충돌 검사를 유지한다.

구현:
- 기존 events.py/shelf_manipulation_env_cfg.py/terminations.py 안에서 수정했다. 생산 코드 파일 추가 없음.
- 공통 round는 episode_length_s와 같은 8초=400 policy step. scene_time_out은 native episode
  timeout OR 공통 round 경계다. 최근 재시작한 환경도 round 경계에서 truncate한다.
  물리 성공/실패가 동시에 발생하면 기존 PPO의 termination 우선순위로 bootstrap하지 않는다.
  나머지 timeout은 reset 전 terminal_observation으로 bootstrap한다. 환경 step/reset override 없음.
- 전체 reset 경계에서 env_0 한 개로 새 배치·기하 vision·물리 FK 검증을 수행한다.
  _build_episode_batch(scene_generation=...)가 singleton template과 SceneSample을 반환한다.
  가시성/IK 검사에 실패한 제안의 bounded retry는 유지하므로 모든 round에 capture 호출이
  반드시 정확히 한 번이라는 보장은 없다. 승인된 장면을 환경 수만큼 다시 관측하지 않는 구조다.
- _shared_reset_batch는 template의 물체 상태/역할/관측을 tensor broadcast하고, 선택 환경마다
  독립 seed로 연속 EEF 위치와 새 관절 자세를 계산한다. 실패한 IK 행만 재시도하고 승인 행은 유지한다.
  CPU metadata/seed 처리와 bounded IK batch loop는 남아 있다. 환경별 SciPy IK를 도입하지 않았다.
- Initial geometry는 robot-base frame으로 저장하고 기존 geometry_in_eef_frame으로 변환한다.
  Robot은 ray casting에서 제외되므로 로봇 자세 변경으로 물체 가시성 결과를 다시 계산하지 않는다.
  Camera pose cache는 env-local이며 reset 대상 origin을 더해 world pose로 기록한다.
- 중간 reset은 레이/배치 생성 없이 캐시를 복원한다. 선반 pose·속도, 물체/로봇 속도·target,
  F/T 처리 상태, previous history와 MDP 누적값도 기존 selected-row reset 경로로 초기화한다.
  배치 생성 seed는 round 번호, 로봇 seed는 환경 ID/reset 횟수에 따른다.
- 기존 independent batch helper는 단위 검증 경로를 유지하지만 학습 reset은 singleton template만 생성한다.

검증:
- CPU 전체 184개: 176 pass, CUDA/Isaac 런타임 관련 8 skip. 공유 cache 보존, 부분 reset의 새 IK,
  미선택 행 보존, round 경계의 전체 timeout/교체, IK 재시도/실패 시 미commit을 포함한다.
- 실제 CUDA/Isaac 4환경: 초기/다음 round의 capture request 크기는 [1,1]. 부분 reset에서
  capture 호출 추가 없음; 로봇 관절 자세는 서로 다르고 재reset 시 바뀜. 새 자세의 실제 PhysX FK,
  서로 다른 EEF 기준 관측, 카메라 원점 이동, 미선택 물리/이력 보존과 전체 truncation 확인. exit0.
- 별도 검증 스크립트에서 경계를 빨리 시험하기 위해 common counter를 옮길 때 MDP counter도
  같이 옮겨야 했다. 두 초기 검증 실패는 스크립트의 불일치였으며 생산 코드의 step guard는 유지했다.
- 실제 RTX 3090/1024환경/10물체/32 rollout/32 iteration 학습 exit0. 기존 계측과 같은
  host perf_counter 계측을 사용했고 초기 iteration0을 제외한 31개를 비교했다.

| 항목 | 직전 독립 장면 방식 | 공통 장면 방식 |
|---|---:|---:|
| 일반 iteration 평균 (29개) | 3.646 s | 3.320 s |
| 대규모 reset 포함 iteration 평균 (2개) | 7.218 s | 3.878 s |
| 전체 iteration 평균 (warmup 제외 31개) | 3.877 s | 3.356 s |
| 대규모 reset 내부 합계/iteration | 3.702 s | 0.510 s |
| 해당 구간 layout 생성 | 0.717 s | 0.0085 s |
| 해당 구간 vision 호출 | 1.958 s | 0.0064 s |
| 해당 구간 IK/후보 검사 | 0.876 s | 0.427 s |
| 일반 iteration의 부분 reset 합계 | 0.175 s | 0.0216 s |

대규모 reset 내부 약86% 감소, 전체 iteration 평균 약13% 감소다. IK는 제거/캐시하지 않았다.
공유 layout에 따라 IK 난이도/실패 빈도 및 물리 부하도 달라져 이전/이후가 동일 trajectory는 아니다.
새 실행에서 round 교체 iteration은12/24이며 각각1024환경을 reset했고, vision request는
각각[1]이었다. 다른 iteration에는 vision 호출이 없었다. 정상 구간 physics.step 평균1.197초,
PPO/GAE 평균0.883초는 남아 있어 목표1–2초/iteration 달성으로 해석하면 안 된다.
호출 시간은 GPU 비동기 대기가 후속 호출에 귀속될 수 있으며 순수 kernel 시간은 아니다.
두 round 교체만 포함한 짧은 실험으로 장기 성능/학습 품질까지 검증한 결과는 아니다.

자료: outputs/contact_manipulation/shared_scene_20261014/ 아래 validation.json/log, tests.log,
profile_timing.py, timing_run.log, timing_only/iterations.json 및 metadata.json, compare.py,
comparison.json, commands.json. 비교 기준은 stage_profile_20261014/timing_only의 기존 측정이다.


### 2026-10-14 — YCB 대 기존 물체 USD/collision 비교 및 1024환경 물리 비교

사용자 요청: 같은 scene으로 2048환경을 사용하는 사례가 있으므로 hand 링크 수만으로
병목을 단정하지 않고, YCB 메시가 기존 물체보다 복잡한지 실제 asset과 물리 비용을 비교한다.
생산 코드/원본 USD를 수정하지 않았다. 진단용 임시 공유 USD에서만 collider를 변경했다.

실제 source USD 및 env_0의 최종 runtime prim을 조사하고, PhysX cooking interface의
request_convex_collision_representation(run_asynchronously=False)로 생성된 hull을 직접 조회했다.
Mesh 삼각형 합은 source USD의 모든 Mesh 합계이며, collider/보조 mesh도 포함한다.
설정의 maxConvexHulls가 아니라 실제 cooking 결과 hull 수를 아래에 기록했다.

| 물체 | USD Mesh 삼각형 합 | 활성 runtime collision | cooked hull 정점 합 |
|---|---:|---|---:|
| bottle_1 | 727,598 | convex hull 2 | 96 |
| cup_1 | 140 | convex hull 1 | 48 |
| cup_2 | 125,340 | convex hull 2 | 96 |
| mug_1 | 5,466 | convex hull 2 | 97 |
| mug_2 | 284 | convex hull 1 | 45 |
| can_1 | 501,628 | convex hull 2 | 96 |
| YCB003 cracker box | 16,384 | convex hull 1 + Cube 1 | 34 (Cube 별도) |
| YCB004 sugar box | 16,384 | convex hull 1 + Cube 1 | 34 (Cube 별도) |
| YCB005 soup can | 16,382 | convex hull 1 | 34 |
| YCB006 mustard bottle | 16,382 | convex decomposition 32 hulls | 640 |

기존6종은 총10 collision shapes, YCB4종은 총37개(35 hulls+2 cubes)다.
Mustard 하나가32개를 차지한다. Source 메시 삼각형 수는 기존 병/유리컵/캔이 더 크다.
따라서 높은 render/source triangle count와 매 step PhysX 형상 복잡도를 구분해야 한다.

발견된 미수정 결함 — YCB box collider 비활성화 override 소실:
- spawn_flat_box_from_usd는 원본 Mesh collisionEnabled=False, FlatBoxCollision=True로 작성한다.
- share_rigid_body_geometry 전/후를 직접 기록하니 두 박스 모두 False/True → True/True가 된다.
- 현재 공유 함수는 기존 reference 뒤에 baked reference를 추가하고 body를 instanceable로 만든다.
  원본 reference의 collider 의견이 baked override보다 강하게 적용되어 비활성화가 소실된다.
- 실험에서 해당 body의 baked reference를 원본보다 우선하도록 정렬하면 원본 Mesh=False,
  Cube=True가 유지된다. 이전 문서의 “박스는 단일 평면 proxy만 사용한다”는 설명은 의도한
  동작이며 현재 런타임 상태와 달랐다. 이 문제는 별도 생산 수정/회귀 검증 대상이다.
  현재 synthetic geometry 공유 테스트는 외부 reference와 상충하는 disabled collider override를
  포함하지 않아 이 사례를 검출하지 못했다.

성능 비교: RTX3090/1024환경/10물체/동일41-link 로봇, dt=.01, solver16/2 유지.
기준 상태를 fixed_states.pt에 저장해 초기 물체/관절/선반 상태와 물체 질량·관성·COM을
비교 실행에 재사용했다. 각 조건을 별도 프로세스로 실행해5블록 측정했으며, 블록마다
64 physics steps warmup 후64 steps를 측정했다. OSC는 zero action으로 초기 pose를 유지한다.
매 측정 전후 CUDA synchronize로 앞서 제출된 제어 tensor 작업의 대기시간을 분리했다.
따라서 아래는 PPO iteration 시간이 아니라 PhysX 64회 호출의 누적 wall time이다.

| 조건 | 64 physics steps 평균 | 블록 범위 | 기준 대비 |
|---|---:|---:|---:|
| 현재 설정 | 1.188709 s | 1.178267–1.194988 s | 기준 |
| Mustard만 32 hull→1 convexHull | 1.169826 s | 1.160980–1.187903 s | 약1.6% 감소 |
| 위 변경 + YCB 박스2종 중복 collider 제거 | 1.147127 s | 1.135426–1.159629 s | 약3.5% 감소 |

모든 실행 exit0, 비교 조건의 runtime collider enable/approximation을 JSON에 저장했다.
형상 수상 mustard는 유난히 복잡하지만 이 정지 자세 유지 실험에서 단순화 효과는 작았다.
현재 약1.2초의 물리 시간을 YCB collision 복잡도만으로 설명할 근거는 부족하다.
접촉 조작 중의 효과와 정책 학습 품질은 이 실험에서 검증하지 않았다. 단일 hull은 오목한
부분을 채우므로 학습용 형상을 바꿀 때에는 기하 정확성 검토가 필요하다.
Hand 링크 수가 주원인이라고도 아직 단정할 수 없다. 동일 robot/solver/collider 조건의
실제 단계별 비교가 필요하다. 이번 요청 범위는 비교/진단이며 위 변경은 학습에 적용하지 않았다.

자료: outputs/contact_manipulation/object_collision_audit_20261014/ 아래 assets.json,
summary.json, baseline.json, mustard_hull.json, ycb_fixed.json, 각 log, fixed_states.pt,
audit_assets.py, benchmark.py, commands.json. 스크립트 재실행은 해당 결과를 덮어쓴다.


### 2026-10-14 — 물리 step 재검토: 손의 충돌 shape 676개와 TGS 반복 비용

사용자 요청: 링크 수만으로 병목을 단정하지 말고 물리 simulation step이 긴 이유를 다시 검토.
학습 코드/원본 USD는 변경하지 않았다. 진단 시에만 임시 geometry library 또는 native PhysxCfg를 변경했다.

측정 조건:
- RTX 3090, 1024환경, 10물체, 로봇 41링크/18DOF, physics dt=0.01.
- 이전 object_collision_audit_20261014/fixed_states.pt의 동일 초기 물체/로봇/선반 상태 사용.
- 물체 및 로봇 mass/COM/inertia도 기준값으로 복원해 collision 변경에 따른 자동 질량 변화를 배제.
- zero18D action: arm delta pose=0으로 OSC 목표 유지, hand는 정규화 action0에 해당하는 중간 목표로 움직임.
  완전히 정지한 장면 또는 학습 정책의 contact-rich rollout으로 해석하면 안 된다.
- 각 설정 5블록, 블록마다 상태 복원 후 64step warmup + 64step 측정.
  apply_action/scene.write 이후 CUDA synchronize, sim.step(False) 전후 wall time+CUDA synchronize.
  reset/IK/raycast/PPO 및 OSC 계산은 측정 구간 밖. 별도의 64step GPU trace도 기준/손 단순화에서 수집.
- 따라서 결과는 iteration 전체 시간이 아니라 rollout당 물리 64회에 해당하는 누적 시간이다.

핵심 발견:
- PhysX ArticulationView.max_shapes와 env_0의 링크별 RigidBodyView.max_shapes로 실제 shape 수 확인.
  robot=676 shapes, links=41. 단순히 max_convex_hulls 설정값만 보고 추정한 수치가 아니다.
- right_plam_1, right_index_2, right_middle_2, right_ring_2, right_little_2가 각각128 shapes.
  나머지36개 링크는 각각1 shape: 5*128+36=676.
  1024환경에서 위5개 링크만655,360 shapes, 전체 로봇692,224 shapes.
- 원인 설정은 config/ur5e/scene.py::_preserve_tactile_parent_slots.
  tactile pad 주변 홈을 convex hull이 메우지 않도록 max_convex_hulls=128, error_percentage=.5,
  voxel_resolution=1_000_000, shrink_wrap=True를 적용했던 부분이다.
- 손의 위5개 링크만 convexHull로 바꾼 진단에서는 robot=41 shapes로 감소.
  41링크/18DOF와17개 tactile 링크, 제어 방식은 유지되지만 collider가 홈을 메우므로
  tactile 접촉 의미까지 동등하게 유지되었다고 해석하면 안 된다.

| 단독 변경 | 물리64step 평균 | 5블록 범위 | 기준 대비 감소 |
|---|---:|---:|---:|
| 기준: 기존 형상, solver16/2 | 1.189156s | 1.174827–1.201106s | — |
| 손의5개 parent collider만 convexHull | 0.760691s | 0.753221–0.772928s | 36.03% |
| 모든 robot mesh collider를 convexHull | 0.756063s | 0.750266–0.763840s | 36.42% |
| 형상 유지, scene solver8/1 | 0.881642s | 0.878452–0.889159s | 25.86% |
| 형상 유지, scene solver4/1 | 0.729232s | 0.721431–0.737664s | 38.68% |
| 형상/solver 유지, ground plane→100×100×1m static box | 1.380843s | 1.368049–1.399931s | -16.12% (느려짐) |

solver 실험은 native scene min/max position/velocity iteration cap을 모두8/1 또는4/1로 설정했다.
로봇만의 반복 횟수를 낮춘 실험은 아니며 물리100Hz/정책50Hz 주기는 동일하다.
solver와 collision 단순화의 개선율은 별도 실험이므로 더하면 안 된다.
기준은 첫 실행1.182523s, 마지막 재실행1.189156s로 유사하게 재현됐다. 표는 마지막 기준을 사용.

별도 GPU trace(64physics steps, kernel duration 합; wall time의 배타적 분해가 아님):

| GPU kernel | 기준 | 손5개 collider 단순화 |
|---|---:|---:|
| convexPlaneNphase_Kernel | 189.120ms | 20.420ms |
| doAggPairCollisions | 75.519ms | 5.413ms |
| sortAndUpdateAggregateProjections | 48.644ms | 2.941ms |
| artiSolveInternalConstraintsTGS1T | 167.502ms | 177.318ms |
| stepArticulation1TTGS | 83.836ms | 79.613ms |

충돌 형상 단순화로 충돌 후보 처리/convex-plane 비용이 줄었고, articulation 내부 제약 비용은
그대로 남았다. artiSolveInternalConstraints는 두 trace 모두1152회=(16+2)*64,
stepArticulation은1024회=16*64. 링크 수를 제거하지 않아도36%가 감소했으므로
기존의 '링크가 많아서 느림' 설명보다 '우리 손 collider 세분화 + TGS 반복'이 구체적인 근거다.
YCB 단순화의 이전 측정1.6–3.5%보다 영향이 컸다.

제외한 실험/제약:
- no_robot_collision은 프로세스 exit0이지만 PhysX foundLostPairsCapacity 부족(요구67,699,200)
  및 miss interactions 오류가 있었고 선반이 약8m 자유 낙하했다. 속도 비교에서 제외했다.
- 최초 진단에서는 Xform collider에 Mesh 전용 cooking query를 호출해 query error가 났다.
  Xform에 대해 이 query를 하지 않도록 진단 스크립트를 수정하고 baseline을 재실행했다.
  손의 실제 shape 수는 SDK rigid-body view로 측정했으며 failed cooking query의0을 사용하지 않는다.
- 축소한 solver 및 단일 hull의 장기 접촉 안정성, F/T 정확도, tactile 가시 접촉, 학습 품질은 미검증.
  속도만으로 production 설정을 바꾸지 않았다. 기존 센서 홈을 보존하는 소수 compound collider를
  위5개 링크에 구성하는 것이 우선 후보이며, solver8/1은 접촉/제어 오차를 비교한 뒤 판단해야 한다.
- 남은 articulation 비용을 링크 수 자체의 불가피한 한계로 단정하지 않는다. fixed joint 유지 이유,
  solver/constraint 설정 및 필요한 정밀도를 분리해 추가 검토할 수 있다.

관련 공식 근거: IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py의 native solver bounds;
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html
(TGS position iteration 증가의 성능 비용),
https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Simulation.html
(simulate/fetch synchronization 및 solver iteration의 의미).

재현 자료: outputs/contact_manipulation/physics_reaudit_20261014/
benchmark.py, run_suite.py, confirm.py, commands.json, confirm_commands.json,
summary.json, 각 variant의 json/log, baseline/hand_parents_hull의 trace/kernel 집계.
production 코드 변경이나 추가 테스트 파일은 없다.

### 2026-10-14 — 원격 min 브랜치의 Inspire Hand 처리 재확인

이전 로컬 origin/min=57caa46 사본보다 원격 min이 새 버전임을 GitHub connector로 확인했다.
원격에는 Robotiq뿐 아니라 Inspire Open v0/v1/v2 및 mass 파생 환경이 추가되어 있다.
따라서 이전 로컬 OSC/Robotiq 분석을 원격 min 전체에 일반화하면 안 된다.
- src/ur5e_inspire_open.py: 외부 ur5e_inspire.usd 로드, inspire_left_* 손 관절12개;
  v0는 revolute를 fixed로 바꾸고 hand actuator/action 제거. self-collision=False.
- src/ur5e_inspire_hand.py: v1은 일반 UsdFileCfg로 원래12가동 관절 유지.
  hand action2D(공통 굽힘/엄지 opposition)를 tensor로12joint position targets에 전개.
  native ImplicitActuatorCfg K=1.2,D=.06,effort=.20Nm,velocity=1rad/s,armature=.0001.
  v1이 v0를 상속해도 fixed-open spawner는 교체되므로 손 관절이 고정된 것은 아니다.
- v2 sensor cfg:17pad를 하나의 ContactSensorCfg regex로 묶어 threshold=.05N,history1.
  v2-mass 및 그 파생에서는 palm1채널만 관측. v2와 v2-mass를 혼동하지 않는다.
- src/ur5e_inspire_sensors.py는 외부 src/trash/hand_manipulation_rl의
  assets.robot.Ur5eAxia80InspireSpawnerCfg 및 sensors.py를 import한다.
  기본 USD도 동일 외부 패키지의 assets/data/ur5e_inspire_usd/ur5e_inspire.usd.
  이 repo 자체에는 실제 USD/외부 spawner가 없고 이 로컬 workspace에도 찾지 못했다.
- 확인한 repo hand 설정에는 우리 _preserve_tactile_parent_slots의128hull 설정이나
  로봇 solver16/2 override가 없다. 하지만 외부 USD/spawner 내부 설정은 미확인이라
  min의 실제 collision shape 수, hull 방식, 최종 solver 값을 단정할 수 없다.
- 정책 hand action2D는 물리 손 관절2개를 의미하지 않는다. 가동형은12개를 유지한다.
근거: https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/src/ur5e_inspire_hand.py
https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/src/ur5e_inspire_open.py
https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/src/ur5e_inspire_sensors.py
https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/docs/11_inspire_hand_variants.md

### 2026-10-14 — 최소 collider 설계안 검토 (미구현)

사용자가5개 parent 링크의 collider 수를 정말 최소화할 방안을 요청했다.
ROAS right-hand URDF와 STL을 CPU로 읽어 고정 센서의 parent-relative pose를 확인했다.
right_plam_1에는 palm pad, 네 finger_2에는 force_sensor_2/3가 서로 다른 방향으로 붙는다.
따라서 단순 uniform shrink 또는128→임의 작은 decomposition budget으로 끝내기보다,
센서 접촉면을 침범하지 않는 소수 convex proxy를 명시적으로 구성하는 방향을 제안한다.
- 각 parent를1개 primitive/저정점 convex core에서 시작하고 센서면 쪽을 half-space로 절단.
- 센서 뒤쪽과 옆면/관절부를 얼마나 보존하는지 검사하고, 중요한 비센서 표면 누락에만
  별도 side/support shape 추가. 전체 링크를 센서면 한 평면으로 잘라 버리는 것은
  멀리 떨어진 외형까지 깎을 수 있으므로 최소안의 검증이 필요하다.
- 손가락 parent1–2개/개, 손바닥1–4개를 초기 탐색 budget으로 제안:
  해당5개 합5–12 shapes, 로봇 전체41–48 shapes가 목표이며 달성/최적성 보장은 아니다.
- 17개 센서 collider/링크 및 F/T 전달 경로 유지. 원래 질량/COM/관성 명시 유지.
- one-time authoring/cooking만 수행, episode/step별 재생성·분할 없음.
- 실제 cooked geometry, GPU 지원, contact/rest offset을 포함해 pad 중앙/가장자리/비스듬한
  접근 시 채널 반응, 인접 parent-only 접촉 시 오검출, 비센서 표면 관통을 검증해야 함.
- 단순한 parent clipping의 후보 계산은 pad local+Z와0.5mm retreat를 가정한 탐색용이었다.
  외부 접촉면 normal/최소여유를 확정한 값이 아니고, 실제 CAD 표면 보존이나 센서 접촉을
  검증한 결과가 아니므로 production 치수/최소 shape 수의 근거로 사용하지 않는다.
- 속도는 shape 수에 정비례하지 않음. solver 비용이 남고 기존 단일hull 실험의36%는
  이상화된 진단 비교이며 새 proxy의 학습 속도/정확도를 보장하지 않는다.

### 2026-10-14 — Hand만 최소 convex proxy 적용

사용자 요청: UR5e는 수정하지 않고 Hand에만 적용한다.
`config/ur5e/scene.py`의 기존 `_preserve_tactile_parent_slots`를 교체했다.
새 production 파일 없이 같은 파일에 순수 geometry helper를 추가했다.

- 변경 대상은 `right_plam_1`, `right_{index,middle,ring,little}_2`의 5개 parent collider뿐이다.
  각 128 convex decomposition → 각 2개의 저정점 convex mesh. 실제 PhysX shape count는
  해당 5개 합 640→10, 로봇 전체 676→46이다. 41 bodies/18 joints는 유지한다.
- 26개 방향의 support plane으로 단순 envelope를 만들고, 첫 pad의 tangent 방향으로
  관절부와 센서부를 분리한다. 센서부는 pad의 outward PCA normal 기준 표면에서
  1.2 mm 뒤로 자른다. 4개 분할 후보 중 원본 parent 점의 최대 half-space 오차가
  가장 작은 후보를 선택한다. 이 오차는 정확한 Euclidean 거리/Hausdorff 거리가 아니다.
- 2개로 나누는 이유: 센서면으로 parent 전체를 자르면 손바닥/관절 돌출부가 크게 사라진다.
  돌출부와 센서부를 분리해 유지한다. 2개가 수학적 최소임을 입증한 것은 아니다.
  세밀한 CAD 홈은 근사하며 정확한 원본 collision 표면을 유지한다는 의미는 아니다.
- Native ConvexHullPropertiesCfg(hull_vertex_limit=64), contact_offset=0.5 mm,
  rest_offset=0. 생성은 최초 template authoring에서만 수행하고 native PhysX가 cooking한다.
  reset/physics step에는 추가 geometry 계산이 없다.
- 원본 collision 전용 subtree를 inactive로 처리한다. collisionEnabled=false만으로는
  PhysX shape 배열에 잔류하여 51개였으므로 최종 구성은 subtree 비활성화로 46개다.
  원본 visual mesh는 유지한다. 표시된 5개 Hand body의 reference는 완전히 펼친 baked
  subtree 하나로 교체한다. Flatten은 inactive prim을 생략하므로 옛 body reference를
  함께 남겨 두면 제거한 collider가 instance 안에서 다시 나타날 수 있어 이를 제거했다.
- UR5e collider/제어, 17개 tactile body/collider, F/T 연결, joints, solver 설정은 변경하지 않았다.
  Native PhysX에서 mass/COM/inertia를 변경 전 snapshot과 비교해 일치 확인했다.

검증과 한계:
- 실제 ROAS STL/URDF 기반 회귀 테스트: 5개 parent 각각 2개, piece당 64개 이하 정점,
  pad local+Z 기준 중앙/주변 접근 시 parent가 센서 표면 뒤에 있음 확인.
- Native GPU 접촉 시험: 반지름 1.5 mm kinematic sphere로 17개 pad의 중앙/주변 5지점씩
  접근, 0.01 N 기준. 변경한 parent에 붙은 9개 pad의 45지점은 모두 감지된다.
  전체 85지점 중 84개 감지, 17개 채널 모두 감지 이력이 있다.
  `right_middle_force_sensor_1`의 한 주변 지점은 기존128-hull Hand에서도 같은 시험으로
  미감지됐다. 변경 대상 parent의 pad가 아니며 이번 변경의 새로운 누락은 아니다.
  변경하지 않은 thumb pad1의 한 지점은 기존과 새 구성 모두 parent가 먼저 접촉한다.
  이 시험은 모든 방향/가장자리/실제 물체 조작에 대한 완전한 검증은 아니다.
- CPU test_mdp 135 tests 완료(7 skipped). Native USD sharing 3개 회귀 테스트도 통과했다.
  접촉 비교는 변경 전후 joint pose가 동일한지 별도로 검사했으며 새로운 누락 없이 완료했다.
- RTX3090,1024환경,10물체,동일 저장 초기상태/질량,100 Hz physics/50 Hz policy,
  64 warmup+64 timed steps를 5회: 물리64step 평균 기존1.189156 s → 새0.750181 s,
  약36.9% 감소. 새 측정 범위0.7386–0.7604 s. 동기화된 sim.step만 측정했다.
  Zero action OSC pose hold에서 hand는 중간 굽힘 target으로 움직이는 진단이다.
  PPO/reset/raycast/OSC 계산은 측정 구간 밖이며 전체 iteration 1–2초 달성을 의미하지 않는다.
- Artifacts: `outputs/contact_manipulation/hand_proxy_20261014/`의
  `scene_before.py`, `benchmark.py`, `baseline.json`(이 폴더에서는 새 proxy 결과),
  `validation.json`, `validation_reference.json`, `validate.py`, `validate_reference.py`.
  비교 이전 물리 결과는 `physics_reaudit_20261014/baseline.json`이다.

### 2026-10-15 — 장시간 학습 중 PhysX CUDA 719 조사

사용자가 1024환경 학습 중 CUDA719 및 DOF velocity 읽기 실패를 보고했다.
현재는 근본 원인이 확정되지 않았으며, Hand collider rollback이나 GPU buffer 증설을
해결책으로 적용하지 않았다. UR5e/Hand/제어/MDP/PPO production 코드는 이 조사에서 변경하지 않았다.

확인한 증거:
- 원본 run: `logs/rsl_rl/contact_manipulation/2026-10-14_23-25-08`.
  Kit log `.../Isaac-Sim/5.1/kit_20261014_232454.log:5207`의 오류 시각은
  UTC2026-10-14 17:33:10 = KST2026-10-15 02:33:10. 약3시간8분 실행 후 발생했다.
- 같은 시각 `/var/log/syslog`에 `NVRM Xid13 / MISSING_INLINE_DATA`, PID68242가 기록됐다.
  이는 CUDA kernel/command 실행 실패 증거이며, DOF velocity 예외는 PhysX CUDA 실패 뒤의 후속 오류다.
  Xid13만으로 application kernel, PhysX, driver, hardware 중 하나를 확정할 수 없다.
- 직전 Kit 로그에 GPU capacity overflow/OOM/NaN 경고가 없다. 종료 중 allocator 경고는
  CUDA 실패 이후의 메시지이므로 선행 메모리 부족 증거로 해석하지 않는다.
- TensorBoard 마지막 완료 iteration3556. 모든 scalar가 finite였다.
  마지막 collection2.058s/learning0.883s로 지속적인 속도 저하도 관찰되지 않았다.
  단, scalar의 finite 여부가 개별 물리 상태의 finite 여부를 보장하지 않는다.
- 마지막 checkpoint `model_3550.pt`의 actor/critic/optimizer tensor를 CPU로 검사해
  NaN/Inf가 없음을 확인했다. Checkpoint에는 물리 state/실패 당시 scene/RNG 진행 상태가 저장되지 않는다.
- Custom Triton/Warp의 padding mask, 고정 크기 배열 인덱스, stream/메시·tensor lifetime,
  CUDA graph buffer 소유권을 검토했으나 명확한 범위 밖 접근이나 소유권 문제는 발견하지 못했다.
- 물체 pool과 Hand proxy는 최초 template에서 생성하며 reset은 pose/velocity만 다시 쓴다.
  reset을 반복할 때 collider가 누적 생성되는 경로는 확인되지 않았다.

재실행:
```bash
CUDA_LAUNCH_BLOCKING=1 ./IsaacLab/isaaclab.sh -p src/contact_manipulation/scripts/train.py \
  --headless --num_envs 1024 --resume \
  --load_run 2026-10-14_23-25-08 --checkpoint model_3550.pt \
  --max_iterations 20 --run_name cuda719_resume_probe
```
- 실제로는 같은 env_isaaclab Python을 직접 실행했다. Native runner/환경 흐름과 모델을 유지했다.
  3550–3569의20iteration을86.98초에 오류 없이 완료했고 정상 종료했다.
  생성 run: `logs/rsl_rl/contact_manipulation/2026-10-15_11-51-09_cuda719_resume_probe`.
- 새 초기 scene이므로 원래 실패한113824+policy step의 물리 상태를 재현한 시험이 아니다.
  짧은 재실행 통과는 장시간 안정성 검증 또는 근본 해결을 의미하지 않는다.
- CUDA_LAUNCH_BLOCKING은 최초 CUDA 실패 위치를 더 가깝게 확인하기 위한 진단 옵션이다.
  동기화 때문에 학습 속도가 감소하므로 정상 throughput 비교에는 사용하지 않는다.
- 원래 PID68242는 조사 시점에 stopped(Tl/do_signal_stop) 상태이며 VRAM5866MiB를 점유했다.
  해당 기존 프로세스를 종료하거나 재개하지 않았다. CUDA719가 난 프로세스의 context를
  재사용하여 계속 학습하는 것은 복구 방법이 아니며 새 프로세스로 checkpoint를 로드해야 한다.
- GPU smoke test는 새 프로세스에서 통과. 조사 시 driver580.173.02, RTX3090,
  Torch2.7.0+cu128. 이를 근거로 driver/hardware의 장시간 문제를 배제하지는 않는다.
- 증거 artifact: `outputs/contact_manipulation/cuda719_20261015/incident.json`,
  `original_error_context.log`, `kernel_xid.log`, `resume_probe.log`.
- NVIDIA Xid13 분류/진단 권고: https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
  다음 장시간 재현에서는 동기화 진단 로그의 최초 실패 지점을 우선 확인하고,
  불명확하면 Compute Sanitizer memcheck로 custom/native kernel 접근을 구분해야 한다.

### 2026-10-15 — PhysX 추가 최적화 후보 실측 (production 미적용)

사용자가 PhysX 연산을 더 줄일 부분을 요청했다. UR5e 모델·제어와 articulation16/2,
100Hz physics/50Hz policy를 유지한 후보를 우선 조사했다. Production Python/config는
변경하지 않고 `outputs/contact_manipulation/physx_candidates_20261015/benchmark.py`의
진단용 config/임시 USD library/native PhysX property에서만 변형했다.

측정 조건:
- RTX3090,1024환경,10물체,현재 Hand46shapes/41bodies/18DOFs.
- 기존 fixed_states.pt의 동일 물체·선반·로봇 시작 상태 및 동일 mass/COM/inertia 사용.
  이전 Hand 측정의 단순 timing과 달리 이번 비교는 로봇 질량 속성도 동일 snapshot으로 명시 복원.
- 각 조건5블록, 매블록64step warmup+64step timed. Zero action OSC pose hold이며
  Hand는 중간 굽힘 target으로 움직인다. 실제 policy의 pushing/pivoting 학습 rollout이 아니다.
- apply_actions 뒤 동기화, scene.write_data_to_sim 전후 동기화, sim.step 전후 동기화로
  데이터 전달과 순수 물리 호출 wall time을 분리했다. reset/vision/PPO는 측정 밖이다.
  이전0.750s와 절대값을 섞지 않고 이번 동일 protocol 기준0.798937s와 비교한다.

| 진단 조건 | 물리64step 평균 | 기준 대비 감소 | scene.write64회 평균 |
|---|---:|---:|---:|
| 현재 구성 |0.798937s|—|0.080530s|
| 접촉 보고34→17body |0.800178s|-0.16%|0.079678s|
| 전체 active일 때0인 permanent wrench 제출 생략 |0.798095s|0.11%|0.051415s|
| 물체 cfg만 solver16/2→8/1 |0.793508s|0.68%|0.078627s|
| Mustard32hull→1 + YCB박스2종 중복 collider 비활성화 보존 |0.745398s|6.70%|0.079632s|
| YCB 전체 collider contact offset20→5mm |0.689108s|13.75%|0.082024s|
| 위 YCB형상 정리+offset5mm |0.664220s|16.86%|0.080088s|
| GPU max partitions8→4 |0.737557s|7.68%|0.082572s|
| GPU max partitions8→1 |0.698532s|12.57%|0.079910s|
| YCB형상 정리+offset5mm+partitions1+0외력 제출 생략 |0.607671s|23.94%|0.051721s|

결합 조건 물리 시간 범위0.604829–0.614805s. 각 단독 개선율을 합산하지 않고 직접 측정했다.
모든 사용한 비교 실행 exit0이며 finite 상태 검사를 통과했다. 단기 pose-hold 시험이므로
contact-rich 밀기/pivoting, 관통·접촉력·F/T·17채널 tactile 및 장시간 안정성은 별도 검증 대상이다.
특히 mustard 단일 hull은 형상을 바꾸는 진단이며 그대로 production 채택한 것이 아니다.

실제 남은 비용과 해석:
- 현재 구성64step GPU trace: artiSolveInternalConstraints175.9ms/1152회,
  stepArticulation84.1ms/1024회, solveBlockUnified81.6ms/9216회,
  solveStaticBlock55.8ms/1152회. Convex-plane은19.0ms 수준이다.
- 결합 구성: 내부 articulation 제약179.9ms/1152회, 적분77.5ms/1024회는 남지만
  solveBlockUnified가29.3ms/1152회로 줄었다. Partition8→1의 batch/launch 감소가 확인된다.
  이 kernel duration들은 별도 CUDA trace의 합이며 물리 wall time의 배타적 분해가 아니다.
  Trace 전체에는 측정 physics 구간 밖의 OSC/data-transfer kernel도 있으므로 전부를
  PhysX 내부 비용으로 합산하지 않는다.
- Official Factory/Automate 조립 예제도 gpu_max_num_partitions=1을 사용한다.
  우리 설정은 기본8이었다. 이는 solver iteration 자체를 낮추지 않는 후보다.
  Factory의 다른 solver/마찰 설정까지 가져온 것은 아니다.
- Hand와 UR5e는 하나의 articulation이므로 articulation root16/2를 줄이면 UR5e도 영향을 받는다.
  Hand만 solver iteration을 줄이는 독립 설정으로 설명하면 안 된다. 이번에는 root16/2 유지.
- 일반 fixed-joint merge는17 tactile body channel을 없앤다. 비센서 palm 고정연결2개만
  선택 병합하면41→39body 후보지만 효과 미측정이고 우선순위는 낮다.

발견한 불필요한 작업/설정 주의점:
- Hand import는32body 전부, F/T spacer는2body에 PhysxContactReportAPI를 활성화한다.
  실제 reader는17 tactile만 사용하고 F/T는incoming joint reaction을 읽는다.
  Native runtime report34→17을 확인했지만 이번 시험에서 순수 물리 개선은 관찰되지 않았다.
- pool10개와 min/max active10개가 같아 주차 물체가 없다. refresh_parking은 그 경우에도
  0force를 set하여 native permanent WrenchComposer.active=True로 만든다.
  Native collection은 매step10,240objects의 force/torque를 제출한다. reset()으로 composer를
  inactive로 두면 데이터 전달 약29ms/64step를 줄일 수 있었다. Zero force가 모든 actor를
  깨운다고 단정하지 않으며, 순수 물리의 큰 개선은 이 시험에서 확인되지 않았다.
- 실제 native contact offsets: 기존6종 약1.962mm, YCB4종20mm. YCB끼리의 contact generation
  시작 거리는 두 offset의 합40mm다. Rest offset 또는 실제 정지 간격이40mm라는 뜻은 아니다.
- 처음 cfg.collision_props만5mm로 설정한 시험은 box proxy만5mm, 원본 YCB는20mm로 남았다.
  USD reference/instance 공유가 descendant override를 잃는 기존 문제와 연결된다.
  이 불완전한 시험은 objects_offset_partial로 보관하고 전체5mm 결과에서 제외했다.
  최종 offset 비교는 native RigidBodyView.set_contact_offsets()로 최초 한 번 적용하고
  get_contact_offsets()로 전체 YCB5mm, 기존6종1.962mm를 확인했다. 매step/episode 조작이 아니다.
- YCB box2종에는 원본 mesh collider와 FlatBoxCollision이 중복 활성화되는 문제가 남아 있다.
  강한 baked reference로 disabled override를 보존한 진단에서 정리 효과를 측정했다.
  Mustard의32hull도 별도 기하 정확도 검토 후 소수 compound proxy로 줄일 후보다.
- 선반은 이미17개 native Cube, self-collision/CCD/추가 stabilization은 꺼져 있고 Fabric은 켜져 있다.
  이전 giant ground box 교체는 느렸으므로 다시 우선 후보로 제안하지 않는다.

자료:
- summary.json, 각 variant.json/.log, baseline/combined_trace.json 및 _kernels.json,
  각 *_object_offsets.json. 진단 실행 스크립트는 같은 outputs 폴더에 있다.
- IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py:111
  및 direct/automate/{assembly,disassembly}_env_cfg.py의 partitions1 설정.
- IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py 및 native WrenchComposer/collection.
- https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html
- https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxSceneDesc.html

### 2026-10-15 — 전체 물체 contact offset2mm 대 YCB5mm 비교 (미적용)

사용자가 전체 물체의 offset을 약2mm로 맞추면 더 빨라질지 질문하여 진단 비교를 수행했다.
1024환경,RTX3090,물리100Hz/정책50Hz,robot41bodies18DOFs46shapes,solver16/2,
GPU partitions8을 유지했다. 동일 저장 초기상태와 mass/COM/inertia를 복원하고 각 조건
5블록×(64warmup+64timed physics steps)를 측정했다. Reset/vision/PPO/OSC 계산은
순수 물리 시간에서 제외했다. 단기 zero-action OSC hold이며 실제 밀기/pivoting rollout은 아니다.

| 설정 | 물리64step 평균 | 블록 범위 |
|---|---:|---:|
| 현재: 기존6종1.962mm/YCB20mm |0.757462s|아래 summary.json 참조|
| 기존6종유지/YCB5mm |0.647007s|아래 summary.json 참조|
| 조작물체10종 전체2mm |0.652493s|아래 summary.json 참조|

- Native get/set_contact_offsets()로 실제 전체10종의2mm 적용을 확인했다.
  Robot/Hand/Shelf/ground offset은 바꾸지 않았다. Rest offset은 기존0을 유지하고 getter로 확인했다.
- YCB5mm는 현재 대비14.58% 감소, 전체2mm는현재대비13.86% 감소했다.
  전체2mm는5mm보다평균약0.85%느렸고, 이번짧은시험에서 추가성능이득은 확인하지 못했다.
  이 작은 차이로2mm가항상느리다고일반화하지 않는다.
- 기존6종1.962→2mm는거의같은값이며, 주된변화는YCB5→2mm다.
  값을통일하는것자체가연산을줄이는것은아니고 실제생성되는접촉의수/범위가중요하다.
  20→5mm의큰개선율을5→2mm에비례외삽하면안된다.
- 모든실행exit0,finite상태검사통과. 전체2mm에서support이후물체최대이동은약1.73mm,
  물체최대선속도약0.0191m/s로진단동안큰발산은없었다. 빠른밀기·모서리pivot·관통과
  F/T/tactile17채널정확성또는장기안정성의검증으로해석하지않는다.
- 추가물체이동지표의초기baseline/5mm실행은world/local좌표를혼합해무효였다.
  해당max_object_displacement필드는사용하지않으며timed영역밖의지표여서성능비교에는영향없다.
  2mm실행은env origin을제거한올바른좌표차이를사용했다.
- Production 기본값은변경하지않았다. 자료:
  outputs/contact_manipulation/contact_offset_2mm_20261015/summary.json,
  benchmark.py,run_suite.py,각variant.json/.log및*_object_offsets.json.

### 2026-10-15 — YCB 5 mm 적용과 native API·iteration·로깅 재점검

사용자는 YCB offset 5 mm 적용, 직접 구현한 코드의 Isaac Lab 기능 재사용 여부,
전체 iteration 비용을 검토하도록 요청했다. 추가로 로깅·디버깅 지연도 포함하도록 했다.

적용 내용:

- YCB 4종의 `UsdFileCfg`/`FlatBoxUsdCfg`에 native
  `CollisionPropertiesCfg(contact_offset=0.005)`를 설정했다. 기존 6종은 `None`으로
  원래 자동 값을 유지하며 rest offset도 변경하지 않는다.
- `share_rigid_body_geometry`는 최종 flattened body library만 reference로 사용한다.
  원본 reference와 추가 reference를 함께 남기면 local descendant override가 instance
  생성 뒤 원본의 강한 설정에 가려질 수 있었다. 원본 USD를 수정하거나 매 reset/step에서
  `set_contact_offsets()`를 호출하지 않고 최초 native spawn에서 설정이 유지되도록 했다.
- 전부 활성인 pool은 permanent wrench composer를 reset한 뒤 비활성으로 둔다.
  Native composer는 0-force라도 set 호출을 하면 active가 되어 매 물리 step SDK 제출을
  실행하므로 이를 생략한다. 비활성 물체가 있는 경우에는 전체 parked mask의 보상력을
  다시 설정하여 부분 reset 뒤 다른 행의 보상도 보존한다.
- F/T 조회는 raw PhysX view 호출에서 public
  `ArticulationData.body_incoming_joint_wrench_b`로 변경했다. 기존 센서 축·부호·SI-200-8
  처리와 자중·관성 보상은 유지한다. 설치된 SDK의 해당 property에는 `time_stamp` 오타가
  있으나 SensorReader 자체의 timestamp cache가 같은 tick의 중복 조회를 막는다.
- Quaternion 부호 정규화는 native `quat_unique`, 물리 FK 검사의 xyzw→wxyz/회전행렬
  변환은 `convert_quat`/`matrix_from_quat`를 사용한다.
- Reset joint velocity 0 tensor를 재사용하고, 공통 template의 읽기 전용 배치 필드는
  broadcast한다. 수정하는 camera/robot 필드 4개만 clone한다.
- 한 forward에 들어가는 batch이면 singleton `torch.cat`을 생략한다. 독립 critic 조회는
  사용하지 않는 actor head를 계산하지 않는다. 파라미터 이름·크기와 checkpoint는 유지한다.

직접 구현을 유지한 근거:

- OSC/hand position controller, Scene/Manager, FrameTransformer/ContactSensor,
  pool force 제출, 기본 PPO update/storage/GAE/logger/checkpoint는 이미 native이다.
- Native reset은 Target·Blocker/밀집도/재배치 공간/충돌 없는 elbow-up 초기 자세와 공통
  초기 vision 규약을 제공하지 않으므로 event의 연구 sampler를 유지한다.
- Native DifferentialIKController는 미적용 후보 pose의 FK·충돌 탐색을 제공하지 않는다.
  Native raycast의 dynamic mesh 지원 여부와는 별개로 후보 pose·역할별 무가림 분모·보이는
  표면 OBB를 함께 계산하는 현재 fused query와 동등한 대체가 아니다.
- Native history의 reset padding, Gaussian의 bounded action log-prob 규약이 달라 연구
  history/tanh distribution을 유지한다. F/T 자중·관성 보상과 OBB reward도 연구 내용이다.
- Native OSC와 COM→TCP Jacobian·속도 보정은 함께 유지한다. Hand의 측정 위치 기준
  lead 제한은 EMA action과 의미가 달라 단순 대체하지 않는다.

검증:

- CPU 전체 suite 185개 중 177개 통과, CUDA/native USD 의존 8개 skip.
- Isaac Sim 안에서 geometry sharing 4개 검사 통과. 외부 reference의 collision enable/
  contact offset override가 hand marker 없이도 살아남는 회귀 검사를 추가했다.
- GPU 4환경에서 YCB 4종의 모든 native shape offset이 5 mm, 기존 6종은 약 1.962 mm,
  rest offset 0임을 확인했다. 로봇 41 body·46 shape와 mass/COM/inertia는 이전 snapshot과
  같았다. 8 policy step의 reward·244D observation이 finite이고 전부 활성일 때 composer가
  비활성임을 확인했다. `max_shapes`에는 비활성 slot도 포함될 수 있으므로 이 값만으로
  YCB box의 중복 활성 collider 수나 제거 효과를 주장하지 않는다.

자료는 `outputs/contact_manipulation/native_audit_20261015/`에 보존한다. 첫 진단 실행은
임시 파일 `profile.py`가 Python 표준 모듈을 가려 AppLauncher에 재진입하면서 startup
실패했다. `stage_timing.py`로 변경한 재실행이 위 검증이며 production 경로와 무관하다.

Forward 분할 최적화:

- 기본 1024환경×32step÷4minibatches=8192 샘플을 다시 1024개씩8회 forward하던
  상한을8192로늘렸다. PPO minibatch/4epochs/16optimizersteps/learning rate는그대로이며
  `max_microbatch_size=8192`이므로 native `PPO.update()`를 사용한다.
- 동일8192sample을 사용한 standalone native update 평균은 chunk1024 0.829초,
  2048 0.543초,4096 0.404초,8192 0.314초였다. Allocated peak는589→613MiB,
  reserved는610→790MiB였다. Physics/GAE/storage는제외한측정이다.
- 모델은 BatchNorm/Dropout/RNN/running state가없고 LayerNorm은 feature축만쓴다.
  고정entropy noise의CPU FP32/FP64검사에서 출력/loss는같고 gradient상대L2차이는
  최대2.62e−7/2.39e−15였다. GPU TF32에서16Adamstep뒤 최대parameter차이는
  0.00145–0.00199여서 bitwise 동일trajectory를보장하지않는다.
- 자료: `outputs/contact_manipulation/native_ppo_review_20261015/`의
  chunks.json/equivalence.json 및재현스크립트.

전체 iteration과logging 측정:

- RTX3090,1024환경,10물체,100Hz physics/50Hz policy,32step rollout,4epochs×4minibatches,
  seed0,headless/no video. YCB5mm/native API수정은두구성모두적용했다.
- 각30iteration을새학습으로실행했다. 최초iteration을제외한29개평균이며공통scene교체
  2회와부분reset을포함한다. 수집/학습trajectory가부동소수오차뒤달라져reset횟수도
  조금다르므로 전체차이를순수GNN속도차이로간주하지않는다. 단독같은batch측정도함께확인했다.

| 구간 | Forward 상한1024 | Forward 상한8192(적용) |
|---|---:|---:|
| Native collection |2.028s|1.960s|
| Native learning(GAE+update) |0.933s|0.320s|
| collection+learning |2.961s|2.280s|
| PPO update(learning내부) |0.926s|0.313s|
| Physics step 64회 |0.719s|0.697s|
| Action apply 64회 |0.409s|0.402s|
| Action process 32회 |0.127s|0.122s|
| Termination manager(공통MDP갱신포함) |0.441s|0.429s|
| Reset 전체(위manager내부포함) |0.081s|0.072s|
| 콘솔·TensorBoard Logger.log |7.61ms|7.26ms|
| Episode log bookkeeping 32회 |7.39ms|7.23ms|
| check_nan 32회 |4.72ms|4.55ms|
| 초기/마지막저장포함총native학습30회wall time |90.78s|70.44s|

위구간은inclusive이므로하위구간을더하여전체를재계산하면안된다. Python wall timing은
GPU비동기작업의대기가다음호출로넘어갈수있어kernel실행시간과도다르다. Profiler는
함수별timer만활성이고torch.profiler/cProfile/추가CUDA synchronize는꺼져있었다.
진단JSON은종료후한번저장하여iteration중계측파일쓰기비용을제거했다.

Logging/debug 검토결과:

- 기본 `check_for_nan=True`: 매step observation/reward/done의GPU scalar검사3개로
  96회/iteration의잠재동기화가있다. 측정한전체호출비용은4.6ms수준이라유지한다.
- Native Logger.process_env_step의dones.nonzero 및reward/length CPU복사도7.2ms수준이다.
- Logger.log는20개scalar기록/iteration을포함해7.3ms. 이중SummaryWriter.add_scalar
  누적은약0.92ms다. 콘솔과TensorBoard,bookkeeping,NaN검사합계약19ms로주병목이아니다.
- Native Logger.log와checkpoint저장은RSL-RL이출력하는Iteration time에포함되지않는다.
  초기checkpoint실제저장은1024/8192구성각약21.6/16.6ms였다. 50iteration주기+마지막
  저장을유지한다. TensorBoard flush는native기본10초주기이다.
- 모델/geometry/schema출력은시작때뿐이고reset반복print는비활성이다. Train기본경로에
  torch.profiler,cProfile,detect_anomaly,CUDA_LAUNCH_BLOCKING,명시적synchronize는없다.
  tactile/EEF debug_vis=False. NativePPO의minibatch별loss.item()은GPU결과완료를기다릴수
  있으므로그전체대기시간을로깅계산비용으로돌려서는안된다.

남은유력절감후보:

- Native `ArticulationData.body_com_pose_b`의고정local COM조회비용은약0.351s/iteration이다.
  여러property호출약194회중timestamp당최대1회실제조회하며현재64physics tick마다
  CPU get_coms→GPU복사를다시한다. 이는OSC/MDP시간내부에포함되며따로더하지않는다.
  mass/COM randomization이없는현재구성의캐시는유망하지만public정적cache설정은없다.
  추후적용시관측·torque동등성과COM변경invalidate를검증해야하므로SDK패치는추가하지않았다.
- Physics약0.697초와제어/MDP약0.5/0.43초가남는다. 앞서측정한native
  gpu_max_num_partitions1 및YCBmustard의collision근사는후보이나이번요청에서는
  solver/UR5e/physics주기/추가물체형상을변경하지않았다.

자료: `outputs/contact_manipulation/native_audit_20261015/summary.json`,
`timing*/{metadata,iterations,total}.json`, `chunk{1024,8192}.log` 및재현스크립트.

최종 기본흐름재개검증:

- 별도profiler/timer wrapper없는native train복사본에서기존`model_3550.pt`를strict로드하여
  1024환경·16iteration재개가exit0으로끝났다. Native collection+learning평균2.489초,
  최초iteration제외15개평균2.376초(collection2.043/learning0.334),전체wall40.19초였다.
  첫iteration은4.185초였다. 앞선fresh30회와policy/resettrajectory가달라같은benchmark로
  섞지않는다. 실제다른학습상태에서도처리량개선설정이동작하는호환성검증이다.
- 모든10240object의mass/COM/inertia를이전fixed_states.pt와비교하여최대차이0을확인했다.
  YCB두box의USD원본Mesh collision_enabled=False,FlatBoxCollision=True도확인했다.
- 위실행은장기밀기/pivot접촉안정성이나이전3시간후CUDA719의해결을증명하지않는다.
- 자료: `resume3550.log`, `resume_summary.json`, `object_properties.json`, `train_smoke.py`.
  `train_smoke.py`는시작시1회물성비교만추가했으며학습중계측은없다.
- Compileall과git diff공백검사도통과했다. Production파일을추가하지않고기존파일에서수정했다.

### 2026-10-15 — 시작 시 URDF·Fabric 경고 검토

사용자가 학습 시작 로그의 지연/문제 여부를 질문했다. 첨부 로그와 URDF·native converter·
현재 FK/센서 코드를 읽기 전용으로 대조했으며 production 코드는 변경하지 않았다.

- Axis 경고22개는 `axis="0 0 0"`인 fixed joint19개(17 tactile+2 palm),
  `axis="0.17365 0 -0.98481"`인 thumb revolute3개다. Fixed joint에는 회전 자유도가 없다.
- NVIDIA 공개 importer의 addJoint는 X축 표현에 필요한 보상 회전을 joint localRot0/1
  양쪽에 적용한다. 문구의 "reorienting bodies"만으로 실제 손이 잘못 돌아간 것으로
  해석하면 안 된다. 공개 소스와 설치된2.4.31 바이너리의 완전한 일치를 가정하지는 않는다.
- 현재 FK는 변환 후 USD의 joint frames/axis에서 구성하고 전체41body를 PhysX와 비교한다.
  F/T는 별도 authored mount frame이고 tactile binary는 world force norm을 사용한다.
  직전 native FK·17채널 접촉 검증에서 이 경고에 따른 오류 증거는 없었다.
- 경고 첫/마지막 timestamp16.077–16.843초의 간격0.766초는 import 작업이 섞인 구간이다.
  순수 로그 출력 비용 또는 전체 URDF 변환 시간으로 해석하지 않는다. Import는 template
  생성 때만 일어나며 reset/iteration에서 반복되지 않는다. 다만 hand UrdfFileCfg에 usd_dir가
  없어 native converter가 실행마다 새로운 /tmp 경로를 만들고 재변환한다. 고정 USD cache는
  프로세스 시작 시간의 개선 후보이며 iteration 비용과 구분한다. Mesh 변경 시 native
  converter hash가 자동 갱신되지 않는 제한도 캐시 적용 시 고려해야 한다.
- No Viewport Window는 headless viewport 부재 알림이다. Fabric removePath의 경로는
  Hand proxy 교체에서 의도적으로 비활성화한 right_index_2 원본 collision mesh와 일치한다.
  초기 USD/Fabric 삭제 동기화의 중복 알림으로 보이며, 이 한 줄로 native 내부 상태 전체를
  증명하지는 못한다. 직전 native46shape/대체proxy/17채널 검증은 정상이다.
- IOMMU는 URDF와 별개인 시스템 설정이다. Isaac Sim5.1 공식 문서는 bare-metal Linux의
  multi-GPU 성능을 위해 비활성화를 권고한다. 현재 단일GPU 지연의 원인이라는 측정은 없다.

근거: IsaacLab native asset_converter_base.py65–106/urdf_converter.py,
scene.py516–530/943–949/1014–1026, mdp/sensors.py145–149/198–203,
https://github.com/isaac-sim/urdf-importer-extension/blob/main/source/extensions/isaacsim.asset.importer.urdf/plugins/import/UrdfImporter.cpp#L976,
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html#disabling-iommu-on-linux .

### 2026-10-15 — local COM 조회 경로와 공식 OSC 비교

사용자가 고정COM 반복조회가 언제 발생하며 공식기능/예제는 어떻게 처리하는지 질문했다.
코드수정없이 설치된버전의공식소스와 현재호출경로를대조했다.

- 고정인것은 각link원점에대한localCOM이며 worldCOM이나전체로봇COM이아니다.
- actions.py93의TCP Jacobian보정이body_com_pos_b를읽고,101의native body_link_vel_w도
  내부에서동일값을쓴다. events.py610의F/T보상은native body_com_pose_w를통해같은값을쓴다.
- articulation_data.py713–727은sim timestamp변경후첫접근에서get_coms().to(device)를한다.
  PhysXTensors api.py2478은(count,max_links,7)을device=-1(CPU)에만들어반환한다.
  같은tick의반복접근은cache되므로세호출경로마다별도복사하는것은아니다.
- 현재32policy step/decimation2에서기본64physics tick동안반복된다. TCP한링크만선택해도
  property가전체1024×41×7을먼저조회한다. 약1.12MiB/회지만0.35초포함시간을순수
  memory copy비용이나보장된절감량으로해석하지않는다.
- 공식run_osc.py314–342와native OSCAction653–716은직접PhysX Jacobian/body_pos_w/
  body_vel_w를사용해이localCOM경로를조회하지않는다. 현재alias body_pos_w는linkpose,
  body_vel_w는COMvelocity이므로우리의COM→TCP기준점보정을삭제하는근거로삼지않는다.
- Native body_offset은지정TCP의고정변환을초기화때GPUtensor로만들어사용한다(325–330).
  localCOM영구cache스위치는아니다. Native randomize_rigid_body_com(events.py400)은
  CPUtensor를쓰며초기화때만사용하도록권장한다.
- examples/Sweeping-Policy-DRL-Sim-Train의joint_pos_env_cfg.py39는JointPositionActionCfg를
  사용하고있어OSC Jacobian/TCP보정과같은조회요구가없다.
- 향후방향은native OSC유지+localCOM초기GPUcache+현재linkpose로worldCOM갱신이다.
  COM/관성변경시cache를갱신해야하며현재publicAPI에는정적cache설정이없다.

### 2026-10-15 — 공식 ManagerBased ActionCfg로 제어 구조 전환

사용자는 action 코드가 공식 controller 예제와 달라 보인다며 Isaac Lab 예제 구조를
요청했다. 두 차례 선택을 통해 다음을 명시했다.

1. Env cfg에서 공식 ActionCfg를 설정하고 실행은 native 클래스에 맡긴다.
2. 기존 custom 동작을 보존하는 adapter를 남기지 않고, 공식 ActionCfg의 기본 동작을
   사용하며 차이를 문서화한다.
3. OSC는 원격 `min` branch 설정에 stiffness 입력을 사용하는 형태로 맞춘다.
   Joint-position은 examples의 구성을 참고한다.

참조한 원격 현재 `min`의 `osc_random_sweep_env_cfg.py`에는 fixed-impedance base와
`UR5eSweepKpOSCRelativeObsEnvCfg`의 variable_kp variant가 이미 있다. 후자를 기준으로
제어 설정을 맞췄다. 로컬 origin/min의 과거 pose_abs/gravity=True 분석과 혼동하지 않는다.
원격 경로:
https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py

적용 구조:

- 공통 ShelfManipulationEnvCfg의 ActionsCfg는 arm/hand ActionTermCfg를 MISSING으로 선언한다.
- config/ur5e/ur5e_env_cfg.py::__post_init__에서 native
  OperationalSpaceControllerActionCfg와 JointPositionToLimitsActionCfg를 생성한다.
- mdp/actions.py의 ArmOscAction/HandPositionAction과 helper/cfg subclass를 전부 제거했다.
  mdp/__init__.py의 custom action lazy export도 제거했다. Controller/math 구현을 복제하거나
  native 클래스 private hook을 override하지 않는다.
- Arm: native ActionManager.process_action → OSCAction.process_actions/set_command →
  physics substep의 OSCAction.apply_actions/OperationalSpaceController.compute →
  joint effort target → scene.write_data_to_sim. Hand: native soft-limit mapping →
  joint position target → 기존 ImplicitActuator/PhysX position drive.
- MDP는 action subclass의 joint_ids/tcp_body_id/invalid_control에 의존하지 않는다.
  Native robot.find_joints/find_bodies로 ID를 한 번 resolve하고 raw action 및 public
  robot.data.applied_torque의 finite 여부를 GPU에서 확인한다. CUDA graph 입력에도
  두 tensor를 명시적으로 포함했다. F/T 자중·관성 보상과 연구 MDP는 유지한다.

최종 설정과 이전 동작의 차이:

| 항목 | 현재 native/min 설정 |
|---|---|
| Arm action | pose_rel6 + stiffness6 |
| position/orientation scale | 모두1.0; 이전0.015/0.08 제거 |
| stiffness_scale |100.0|
| stiffness limit | 모든축50–200; 이전 병진100–800/회전5–40의 affine mapping 제거 |
| 실제정책kp범위 | bounded action[-1,1]이므로 clip(100a,50,200)=50–100;200은native상한 |
| inertial decoupling | full=True,partial=False |
| gravity compensation |False (원격현재min과동일)|
| damping ratio / nullspace |1 / none|
| motion/wrench control axes |전축motion / wrench제어없음|
| TCP |우리자산의right_plam_force_sensor;Robotiq용0.13m offset은사용하지않음|
| Hand |독립6관절,순서고정,[-1,1]→live soft limits의nativeposition target|

- Raw position0.01=1cm,rotation0.01=0.01rad다. ±1m/±1rad까지의 raw bounded 입력을
  예전 ±15mm/±0.08rad 제한과 혼동하지 않는다. Network/distribution의 tanh bounds는 그대로다.
- 추가 workspace 목표 clipping, COM→TCP Jacobian/속도 보정, custom effort clipping,
  hand measured-relative lead 제한, action sanitization은 제거했다. Native frame 처리와
  기존 asset의 actuator/PhysX 한계를 사용한다. UR5e/Hand geometry·질량·관성은 변경하지 않았다.
- Native OSC에서 localCOM조회가없어졌지만 F/T보상의worldCOM계산은여전히native COM API를쓴다.
  이번변경을localCOM완전cache최적화로해석하지않는다.
- Action18D/observation244D,initialvision·GNN·attention·PPO설정은유지한다. 입력의물리적의미가
  달라졌으므로 action_contract=isaaclab_min_osc_variable_kp_joint_position_to_limits_v1을
  checkpoint metadata에추가하여이전customcontroller checkpoint의자동resume/play를거절한다.
  모델parameter shape가같다는이유로같은제어로이어학습할수있다고표현하지않는다.

검증:

- test_mdp.py:133개중125통과,CUDA/native8skip. test_learning.py:51개중50통과,CUDA1skip.
- Isaac Sim안에서native controller 검사3개통과(native hand mapping/reset,
  nativeOSC action gain scaling/clamp,nativeOSC relative latch/inertia-scaledPD).
- GPU4환경에서stock action class의정확한type,12+6=18D,실제resolvedjoint/TCP ID,
  stiffness[-1,0,1,2]→[50,50,100,200],handsoftlimit중간값을확인했다.
  410policy/820physics step에서observation244D/reward/applied torque가finite였고
  선택행reset의다른환경상태보존,53회자동reset,공통scene교체1회를검증했다.
- 자료: outputs/contact_manipulation/native_actions_20261015/의validate.py,
  validation.json/log,actions_before.py,events_before.py. Production파일추가없이actions.py를삭제했다.

최종1024환경학습검증은16iteration을exit0으로완료했다. Native collection평균3.651초,
learning평균0.328초,합계3.979초/iteration(최초제외3.814초),전체학습wall64.06초였다.
현재min의scale1.0/gravityFalse/nativehand동작으로초기랜덤정책의행동·termination양상이
달라졌으므로앞선custom제어2.35초와동등조건성능비교로해석하지않는다. Episode length의
native최근100episode평균지표를16회평균하면77.8step이었다. 단순히구조정리로iteration이
단축됐다고주장하지않는다. 이후성능/학습안정성비교시동일물리action범위와policy상태를
고정한측정이필요하다.

새model_15.pt의action_contract와저장을확인했다. 학습위치:
logs/rsl_rl/contact_manipulation/2026-10-15_14-05-26_native_min_actions/.
자료: native_actions_20261015/train1024.log,training_summary.json.

현재native/min checkpoint를production play.py로4환경·64controlstep재생하여exit0으로
완료했다. play.log의Playing iteration15/Played64controlsteps를확인했다. README의재생예시도
구계약model2050에서새계약model15로갱신했다. Compileall/git diff공백검사통과.

## 2026-10-15 — min branch를 프로젝트 전반의 구조 기준으로 사용

사용자는 Sweeping-Policy의 `min` branch가 유사한 로봇·환경에서 다른 과제를 학습시키는
프로젝트임을 강조했다. 제어뿐 아니라 전체 코드 구조와 실행 흐름을 주요 참고 대상으로
삼는다. 이후 개발에서도 먼저 해당 구조와 Isaac Lab public 기능으로 표현할 수 있는지
확인하고, 이 연구에 필요한 계산을 그 안의 cfg/MDP/model 확장으로 넣는다.

원격의 현재 `min`을 다시 읽어 공통 환경 cfg, OSC cfg, hand·sensor 변형 cfg,
Gym 등록, train/play 및 PPO 설정을 비교했다. 로컬 examples checkout과 원격 min은
동일한 버전이 아니므로 혼동하지 않는다. 주요 참고 파일:

- https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/shelf_sweep_random_env_cfg.py
- https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py
- https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/config/ur5e/joint_pos_random_single_inspire_mass_sensors_env_cfg.py
- https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/train.py
- https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/play.py

적용한 구조 정리:

- 공통 `shelf_manipulation_env_cfg.py`: ground/light, manager term 선언, physics 100 Hz,
  policy 50 Hz, episode 8초 및 viewer/render 설정을 관리한다.
- `config/ur5e/ur5e_env_cfg.py`: 공통 cfg를 상속하고 robot/shelf/object/sensor와
  공식 OSC·joint-position action을 구체화한다. 값과 제어 의미는 앞선 native/min 구성과 같다.
- Observation을 단일 `measurements` term에서 `initial_scene` 100D, `current` 72D,
  `previous` 72D의 native `ObservationTermCfg`로 선언한다. 구성과 순서가 cfg에 드러난다.
  각 함수는 동일한 packed snapshot의 view이며 native manager가 순서대로 연결한다.
  관측 정보·244D 순서·모델 parameter·action 계약은 그대로여서 직전 native checkpoint를
  그대로 사용할 수 있다. 새 history encoder나 mask/role feature를 추가하지 않았다.
- 관측·reward·termination이 공유하는 `TaskState`는 한 step의 센서·이력·차분 값을
  일치시키고 reset 직전 terminal observation을 보존하므로 유지한다. 세 observation term이
  센서 취득/기하 계산/이력 갱신을 세 번 실행하지 않는다. Native manager의 작은 clone 호출이
  두 번 늘어나므로 이 변경을 속도 최적화라고 설명하지 않는다.
- Production 파일 수를 늘리지 않았다. Asset 준비와 GPU FK/IK는 기존 `scene.py`,
  reset/sampler/공유 상태는 기존 `events.py`, 모델은 기존 `agents`에서 관리한다.

학습 실행 계층 검토 결과:

- `min`은 task 등록 후 `runpy`로 stock train/play를 실행한다. 우리도 AppLauncher → Hydra →
  gym.make → RslRlVecEnvWrapper → OnPolicyRunner 순서다. 별도 runner나 환경 step/reset
  subclass를 추가하지 않고 native rollout/GAE/logger/checkpoint를 사용한다.
- 직접 launcher만 남기면 stock train의 `init_at_random_ep_len=True`가 초기 vision·history·
  경과 시간과 어긋난다. 따라서 기존 False를 유지한다. Stock play의 무조건 JIT/ONNX export는
  현재 graph adapter에서 지원하지 않으므로 native inference로 바로 재생하는 차이도 유지한다.
- 공유 GNN/attention과 tanh 분포, optimizer 중복 등록 방지, terminal 관측 bootstrap은
  연구 adapter에 유지한다. 기본 1024환경의 PPO update는 이미 native `super().update()`다.
  Minibatch 8192 초과일 때만 메모리 제한을 위한 microbatch 경로를 사용한다.
- `min`의 다른 과제용 reward, observation, MLP와 PPO 수치를 복사하지 않는다. 구조를
  참고하되 연구 수식·센서 의미·정책 설계의 변경은 별도로 판단한다.

README에 대응표, 전체 실행 흐름, native 예제와 달라지는 이유를 정리했다.
Train/play docstring에도 짧은 launcher로 대체하지 않은 구체적인 이유를 남겼다.

검증 결과:

- CPU MDP 127개 통과·8개 skip, 모델/PPO 50개 통과·1개 skip. 새 관측 term을 역순으로
  조회해도 이력이 한 번만 갱신되고, concat 결과와 terminal snapshot이 보존되는 회귀 검사 포함.
- GPU 4환경·410 policy step에서 native manager의 term 이름과 100/72/72 차원을 확인했다.
  모든 step에서 합쳐진 244D가 기존 packed snapshot과 tolerance 0으로 일치했다.
  반복 관측 조회의 이력 보존, 선택 행 reset의 다른 행 보존, 이전 terminal snapshot 보존,
  49회 자동 reset과 공통 장면 교체 1회를 확인했다. Exit 0.
- Production train.py로 직전 native `2026-10-15_14-05-26_native_min_actions/model_15.pt`를
  불러와 1024환경·4 iteration 이어 학습을 완료했다. Exit 0, 학습 wall 17.59초.
  단기 호환성 검사이며 처리량 개선이나 장시간 학습 안정성의 증거로 해석하지 않는다.
- Compileall과 git diff 공백 검사 통과. 검증 스크립트·로그·JSON은
  `outputs/contact_manipulation/min_structure_20261015/`에 있다.

## 2026-10-15 — 2026-10-14_23-25-08 reward 로그와 수식 재검토

사용자는 접근 보상을 따로 주지 않아도 물체와 접촉하는 행동이 나타난 점을 긍정적으로
보고 reward weight/scale을 검토하도록 요청했다. 추가로 수식 자체를 바꾸거나 같은 목적을
더 적절하고 저렴한 계산으로 달성하는 방안도 검토하도록 했다. 아래는 분석과 제안이며
새 reward 설계로 확정하거나 production 수식/weight를 변경한 것은 아니다.

해당 run의 저장 env/agent YAML, git diff의 기준 commit 7be67ac 및 TensorBoard event를
확인했다. rewards.py는 해당 기준 commit과 현재 코드가 같다. 반면 당시 action은 이전
custom controller의 position/orientation scale 0.015/0.08이었다. 현재 native/min action과
학습 결과를 직접 대조하면 controller 변경 효과가 섞이므로 후속 실험은 controller를 고정한다.

기록은 iteration 0~3556, 총 3557개이며 1024환경×32step 기준 116,555,776 transition이다.
`Episode_Termination/success`는 모든 기록에서 0이었다. 각 구간의 scalar 평균:

| 구간 | Train/mean_reward | Train/mean_episode_length × 0.02 |
|---|---:|---:|
| 0~99 | 416.04 | 5.04초 |
| 500~999 | 277.63 | 3.65초 |
| 3000~3556 | 490.06 | 5.61초 |

이는 일관된 단조 향상이 아니다. 사용자가 본 접촉은 유용한 행동 관찰이나, 저장된 로그에는
접촉률·raw overlap·path length·분리된 motion cost·원인별 failure가 없어 접촉 증가와 그 원인을
직접 검증할 수 없다. Reward 상승만으로 통로 확보가 개선됐다고 결론 내리지 않는다.

로그의 단위:

- `Episode_Reward/*`는 reset 대상들의 episode sum 평균 / 설정된 8초다. 실제 episode
  길이로 나눈 step quality가 아니다. Logger는 reset 크기로 가중하지 않고 extras를 평균하며
  reset 없는 step에는 이전 extras가 남을 수 있다.
- `Train/mean_reward`, `Train/mean_episode_length`는 최근 완료 episode 100개의 평균이다.
- `Episode_Termination/*`는 전체 환경의 가장 최근 종료 원인 flag 평균이다. 현재 완료한
  episode의 event 비율과 다르다. 따라서 terminal reward와 failure 지표의 분모가 다르다.

기존 수식의 관찰 가능한 구조 문제:

1. Clearance는 지정 Blocker만 보지만 success는 모든 non-target의 A0 overlap 합을 본다.
   다른 물체가 통로를 막거나 들어오면 clearance가 높아도 성공할 수 없다. 전체 overlap은
   이미 같은 GPU kernel에서 계산하므로 재사용하는 데 추가 geometry query는 필요 없다.
2. Rearrangement는 누적 L에 대한 exp(-2L)을 매 step 지급한다. 0.1m 이동 후에는
   이후 매 step 0.1813점씩 적게 받는다. 초기 blocker overlap ratio가 0.3이면 완전히 비워도
   clearance 개선은 0.2592인데, 누적 L=0.15m의 rearrangement 감소도 정확히 0.2592다.
   이는 전체 조작량 외에 조작 시점/남은 시간에 따른 추가 영향을 만든다.
3. Motion의 normalized acceleration 제곱합은 여러 joint가 움직이면 쉽게 지수 score를
   포화시킨다. 팔 6축이 각각 기준 5rad/s²일 때 exp(-6)=0.00248이다. 이는 예시이며
   실제 run에서 포화됐다는 증거는 없다. wrench가 안정적이면 합성 Motion은 약 0.5012다.
4. 앞의 세 항은 per-step score이고 Time만 초 기준 적분이다. 8초 dense 최대합 1200에
   비해 Time 총합은 -7.5379, 마지막 step은 -0.1070이다. 모든 weight=1이어도 균형이 아니다.
5. 성공 +500은 이 로그에서 경험한 기록이 없으므로 dominant했다는 근거가 없다.
   gamma=0.99에서 step400의 +500은 시작점 할인값 약9.066이다. 반대로 즉시 가능한
   성공을 무조건 미루는 것이 유리한 것도 아니다: dense 무한 할인 상한은 3/(1-.99)=300이다.

수식 변경 후보(검토 중):

- Clearance: 전체 non-target overlap B_t를 쓰고 초기 B_0 또는 A0 면적으로 정규화한
  exp 품질값을 고려한다. 분모 선택은 목적/scale 변경이며 단순 최적화가 아니다.
  품질값을 매 step 주는 대신 진행량을 보상하는 후보도 있다. 단순 Phi_t-Phi_prev는
  gamma<1에서 potential-based policy invariance를 보장하지 않는다. 그 성질이 필요하다면
  gamma*Phi(next)-Phi(current), true terminal의 potential=0, timeout의 pre-reset final
  state bootstrap 처리를 함께 맞춰야 한다. 이는 기존 state score와 같은 objective가 아니다.
  참고: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Rearrangement: 합의한 전체 조작량 최소화에 가장 직접적인 transition 비용은
  -w_r*DeltaL이다. 할인 전 합은 정확히 -w_r*L_total이며 왕복도 남는다.
  exp(-aL_t)-exp(-aL_prev)는 최종 누적 exp 비용으로 telescoping하지만 총 비용이 포화된다.
  exp(-a*DeltaL)-1은 작은 움직임에서 선형에 가깝지만 큰 한 번의 움직임을 상대적으로 덜
  벌주고 step 분할에 따라 합이 달라진다. 모든 항을 exp로 맞추는 기존 방침도 재검토할 수 있다.
- Motion: 가속도·힘/토크 크기 증가율이라는 목적은 유지하면서 joint 평균/Huber 등의
  비포화 비용을 검토한다. 합→평균, exp→Huber는 목적의 상대 weighting을 바꾸므로 동등한
  계산 최적화로 표현하지 않는다. 속도 자체·지속 하중·힘 감소에는 새 벌점을 주지 않는다.
- Time: 지수 증가 개형과 해석적 적분은 유지할 수 있다. 기존 수식 기준 scale0.1~0.2는
  8초 비용37.69~75.38의 비교 후보지만, 다른 수식을 바꾸면 이 수치도 재설정한다.
  현재 observation에는 elapsed time이 없으므로 매우 강한 시간 의존성도 별도로 검토한다.
- 기존 수식만 쓰는 비교군의 한 후보는 clearance/rearrangement/motion=2/0.5/0.5다.
  최대 dense합3을 유지하면서 task 신호 비중을 늘리지만, 위 수식 변경 후 그대로 쓸 수 있는
  최종 weight 제안은 아니다. Gamma와 terminal을 동시에 바꾸지 않는다.

Pose 기반 조작량 계산 최적화 후보:

OBB 중심 변화 dc, local half-extents h, q_rel=conj(q_prev)*q_now의 vector부 u에 대해
8-corner RMS 변위는 다음과 같이 직접 계산할 수 있다.

`d_RMS² = ||dc||² + 4*sum_i[u_i²*(sum_j h_j² - h_i²)]`

이 식은 8개 corner의 norm 대신 물체당 sqrt 하나를 쓰며 평행이동·회전·pivot을 함께 반영한다.
단 기존 산술평균 corner distance와는 다른 metric이며 RMS가 더 크거나 같다. OBB 중심과
half-extents의 좌표계가 정확히 일치해야 한다. Synthetic float64 pose 10,000개에서 explicit
8-corner RMS와 최대 오차2.01e-16을 확인했다. 기존 mean 대비 RMS 증가 중앙값4.33%,
95백분위11.32%는 합성 표본의 결과이며 실제 학습 물체의 분포나 성능 향상을 뜻하지 않는다.

계산 비용은 큰 병목과 구분한다. 이전 custom-action 1024환경×32step profiler에서
reward_components 약17.57ms, projected_areas9.85ms, corner_path3.51ms/iteration이었다.
중첩 측정이므로 합산하지 않으며 현재 native action의 절대 성능으로 해석하지 않는다.
Exp 자체를 바꿔 전체 iteration을 크게 줄일 근거는 없다. Full3D projected OBB hull은 이미
Warp GPU kernel이며, yaw-only/AABB 대체는 tilt/pivot에서 기하 의미가 달라지는 근사다.

분석 자료: `outputs/contact_manipulation/reward_review_20261015/`의 scalars.csv,
summary.json, training_trends.png/pdf, rms_geometry_check.json.


## 2026-10-15 — Blocker 기준 Clearance와 발생량 기반 비용 적용

**아래 선형/pseudo-Huber 보상 및 이에 맞춘 weight는 다음 사용자 정정에 따라 철회했다.**
검증 기록은 당시 실험의 이력이며 현재 보상 설계가 아니다. 현재 정의는 다음 절을 따른다.

사용자가 직전 reward 검토의 네 항목을 구체화했다. Clearance는 다른 non-target 물체를
포함하지 않고 **지정 Blocker가 A0를 가리는 면적 / Blocker 전체 footprint 면적**으로
정의한다. Rearrangement의 반복 누적 감점을 발생한 조작량의 비용으로 바꾸는 데 동의했다.
Motion은 examples/의 구현을 참고하며, Time·Terminal scale은 이 수정 이후 검토한다.
직전 절의 전체 non-target Clearance 제안은 채택하지 않는다.

적용한 수식과 native manager 연결:

- Clearance: `exp(-clamp(B/A_blocker,0,1))` 유지. B와 분모는 현재 Blocker의 XY 투영 OBB
  footprint에서 계산한다. 성공 overlap gate도 Blocker의 B<=1e-5 m²로 맞춘다.
  주변 물체 overlap은 이 gate에서 제외하되 Target 보호, 모든 active 물체의 안정화·안전
  조건은 유지한다. GPU projected hull/clipping은 blocker mask 밖의 object를 건너뛴다.
  CPU fallback이 mask를 무시하던 차이도 회귀 검사에서 확인하고 수정했다.
- Rearrangement: `r_R=-2*DeltaL`. DeltaL은 모든 active 물체 각각의 대응 OBB corner
  8개 변위 norm을 평균하고 물체 간 합산한 m 단위 거리다. Target·Blocker·초기 vision에
  없는 물체도 포함하며 privileged physical pose를 사용한다. 회전·pivot·왕복도 비용이
  남는다. 누적 path_length는 진단용으로만 유지하고 멈춘 후 과거 이동을 다시 과금하지
  않는다. 할인 전 총 비용은 -2*L_total이다. RMS metric 제안은 적용하지 않았다.
- Motion: 예제의 음수 RewardTermCfg 비용 구조를 따르되 예제의 joint velocity/action-rate
  목적은 그대로 복사하지 않는다. 공식 joint_acc_l2를 arm 독립6관절/hand 독립6관절에
  각각 적용한다. Native tensor의 가속도를 사용하므로 기존 policy velocity-history와
  custom acceleration 계산을 제거한다. Mimic/passive 관절은 중복 계산하지 않는다.
- 공식 제곱합으로 `x_arm=sum(arm_acc²)/5²`, `x_hand=sum(hand_acc²)/10²`를 계산한다.
  큰 가속도의 극단적인 quadratic 비용을 완화하되 포화시키지 않기 위해
  `rho(x)=2*(sqrt(1+x)-1)`를 적용한다. 이는 정규화한 vector 크기에 대한 pseudo-Huber다.
  작은 변화에는 제곱 비용, 큰 변화에는 변화량 크기에 비례한 비용이며 상한이 없다.
  작은 x의 소거 오차를 피하는 동등식 `2*x/(sqrt(1+x)+1)`을 GPU tensor로 계산한다.
  Wrapper가 공식 함수의 결과를 변환하는 연구 수식이며 native 함수와 동등하다고 주장하지 않는다.
- F/T는 기존의 보상된 wrench를 world 축·episode 초기 sensor 원점에 고정하여 비교한다.
  `x_FT=(max(norm(F_t)-norm(F_prev),0)/(dt*200))²
  +(max(norm(tau_t)-norm(tau_prev),0)/(dt*10))²`에도 같은 rho를 적용한다.
  일정한 하중·힘 감소·같은 크기의 방향 변화는 이 비용이 없다. 일정한 관절 속도도 가속도
  비용을 만들지 않는다. raw x_FT는 wrench_rate_cost에 진단용으로 유지한다.
- 최종 초기 Motion weight는 arm/hand/F/T 각각 -0.05이다. 따라서
  `r_M=-0.05*dt*(rho(x_arm)+rho(x_hand)+rho(x_FT))`다.
  기존 양의 exp score를 음수 비용으로 바꿨으므로 단순한 계산 최적화나 동등 scale이 아니다.
- Native joint_acc는 매 physics step에서 갱신된다. Reward 시점 값은 마지막10ms substep의
  가속도이며 종전20ms policy 구간 평균과 다르다. 두 physics substep의 제곱 가속도 평균
  또는 정확한 시간적분이 아니다. Native joint-velocity reset은 해당 행의 이전 속도 cache와
  가속도를 함께 초기화하므로 reset 점프를 별도 history로 보정하지 않는다.
- Time은 `-integral[elapsed,elapsed+dt]0.02*(exp(0.7s)-1)ds`, Terminal은 성공+500/
  실패-100, gamma0.99를 그대로 둔다. 새 수식의 분포를 보고 별도 조정하며 지금 정한
  비용 weight를 최종 균형값으로 해석하지 않는다. 이전 reward로 학습한 critic의 return
  scale은 달라지므로 수식 비교는 새 run에서 수행한다.

설정은 기존 RewardsCfg의7개 native term(clearance, rearrangement, arm_acceleration,
hand_acceleration, wrench_increase, time, terminal)에 모았다. Arm/hand 관절 선택은
UR5e cfg가 공식 ActionCfg의 동일 joint_names를 SceneEntityCfg에 연결한다. Native logger가
Motion을 세 항으로 기록한다. Production 파일은 추가하지 않았고 observation/action/model
계약은 변경하지 않았다. Geometry·Time·Terminal은 discrete 값/dt를 반환하며 가속도·F/T는
비용률을 반환한다. 따라서 native weight*dt가 중복 적분을 만들지 않는다.

초기 scale 선택 근거:

- 과거 exp 식의 normalization 기준을 그대로 quadratic 비용에 적용한4환경 검증에서
  Motion 평균 크기30.684/step, Clearance0.369였다. 1024환경4iteration에서는 드문 hand
  가속도 값이 cost와 value target을 크게 만들었다. 해당 run은 수식 후보의 검증이며 채택하지 않는다.
- 단순 weight축소 후보에서도 131,072 transition의 hand 비용 중앙값은 약5.0e-6이지만
  최대값은12603.65였다. 평균만 맞추기 위해 scale을 더 내리면 일반적인 동작 신호도 약해져
  큰 변화에서 선형 증가하는 rho를 선택했다. 물리 상태의 큰 가속도를 해결한 것은 아니다.
- 위 동일 표본의 raw 제곱 비용을 최종 rho와 weight -0.05로 재계산하면 Motion 평균0.0881,
  Clearance평균0.3705다. 첫 reset후 step평균0.0754/이후0.0885, failure step평균0.5903/
  nonfailure0.0763이다. 모든 raw 표본은 finite였다. Motion p50/p90/p99는0.0598/0.1545/0.5630,
  최대72.43으로 여전히 큰 급격한 변화에 비용이 남는다. 작은 값으로 clipping한 것이 아니다.
- 이 수치는 같은 표본의 비용 재계산이다. 새 reward로 학습한 성능 개선이나 장시간 안정성의
  증거가 아니며, controller와 초기화는 그대로 둔 초기 coefficient 선택 근거다.

검증 자료는 outputs/contact_manipulation/reward_update_20261015/에 저장한다.
CPU 회귀 검사는 Blocker/주변 물체의 overlap 분리, 새 이동량만 과금, 왕복·비관측 물체 포함,
실제 native joint_acc_l2 호출/관절 선택, rho의 작은 값·큰 값 특성, 일정 속도/하중/힘 감소
무과금, FT와 Rearrangement의 native weight·dt 계약을 포함한다. 기본 pytest는 ROS 자동
플러그인의 lark 의존성 때문에 수집 전에 실패하여 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1로
프로젝트 테스트를 실행한다.

최종 코드 검증 결과:


- CPU MDP/모델/PPO 185개 통과, 12개 skip, subtest112개 통과. Compileall·공백 검사 통과.
- GPU 4환경 410 policy step, 자동 reset 49회, 공통 장면 교체 1회를 완료했다. Native 7 reward term,
  arm/hand 독립 관절 선택, 실제 공식 가속도 값에 rho/ref/weight/dt를 적용한 결과, FT/조작량
  accounting, 244D 관측·선택 행 reset·native 가속도 reset을 검증했다. Exit 0.
- 최종 수식으로 1024환경 4 iteration, 131072 transition의 native PPO를 실행했다. 모든 reward와
  value loss가 finite이고 Exit 0. 기록용 계측은 검증 스크립트에만 추가하고 production에는 넣지 않았다.
  Per-transition 평균 Clearance +0.3712, Motion 합 -0.08476. 학습 wall 19.49초는 짧은 실행 결과이며
  장시간 수렴·안정성이나 성능 향상을 보장하지 않는다. Time·Terminal을 이 수치로 자동 조정하지 않았다.
- 최종검증의 validation.json/log, train1024_robust.log, robust_motion_distribution.json,
  final_training_summary.json과 후보 분석 자료를 위 artifact 경로에 보존했다.


## 2026-10-15 — exp(-a·x) 유지: 수식 검토 범위 정정

사용자가 “수식 변경 검토”는 **exp(-a·x)를 유지하면서 x에 넣을 것을 고민하자는 의미**라고
명확히 정정했다. 바깥 함수를 선형 비용/pseudo-Huber 비용으로 바꾼 것은 assistant의 잘못된
해석이었다. 해당 변환과 -2/-0.05의 음수 외부 weight를 철회했다. exp-1 변환도 하지 않는다.

현재 정의:

- Clearance: x_C=Blocker의 A0 교차 면적/Blocker의 현재 XY 투영 footprint 전체 면적.
  exp(-x_C)이며 주변 물체 overlap은 제외한다. Blocker 기준 성공 overlap gate와 Target
  보호·모든 active 물체의 안정화 및 안전 조건은 직전 수정대로 유지한다.
- Rearrangement: x_R=이번 transition의 ΔL_OBB. 대응하는 corner8개 변위 norm의 물체별
  평균을 모든 active 물체에 대해 합산한 m 단위 값이다. 누적 L은 보상 입력에 넣지 않는다.
  exp(-2·ΔL)을 지급하므로 멈춘 다음 step에는1로 복귀한다. 왕복·회전·비관측 물체도 포함한다.
  누적 보상 감소가 -2·L_total과 정확히 같지는 않으며, step 분할에 따른 비선형 영향은 남는다.
- Motion은0.5 exp(-a_acc·x_acc)+0.5 exp(-a_FT·x_FT).
  x_acc=sqrt((joint_acc_l2(arm)/5²+joint_acc_l2(hand)/10²)/(n_arm+n_hand)).
  공식 joint_acc_l2를 사용하며 n_arm=n_hand=6, mimic 제외. Native SceneEntityCfg로 관절을
  선택하고 count는 CPU metadata에서 얻는다. Tensor 값을 CPU로 복사하거나 환경별로 순회하지 않는다.
  관절 수 때문에 지수가 커지는 효과와 제곱 증폭을 줄이기 위해 정규화 RMS를 x로 사용한다.
- x_FT=sqrt(((Δ⁺norm(F)/(dt·200))²+(Δ⁺norm(tau)/(dt·10))²)/2).
  기존 고정 world 축·episode 초기 sensor 원점의 보상 wrench를 사용한다. 증가만 입력하며
  일정 하중·힘 감소는 x=0이다. 일정 속도도 x_acc=0이다. Raw FT 제곱합은 진단용으로 유지한다.
- Native 가속도는 마지막10ms physics substep 값이며20ms policy 평균이 아니다. 이전 custom
  속도 history를 복구하지 않는다. Reset 때 native 가속도 cache가 초기화되는 동작을 사용한다.
- a_C=1,a_R=2,a_acc=1,a_FT=1은 기존 계수 기준으로 유지한다. 이전 음수 비용에 맞춘 새
  weight는 이식하지 않는다. RewardsCfg를5개 term(clearance,rearrangement,motion,time,terminal),
  외부 weight 모두1로 복구했다. Time의 기존 적분식, Terminal+500/-100, gamma0.99는 유지한다.
- 모든 term adapter는 discrete score/payment를 step_dt로 나눠 native RewardManager의 dt와
  상쇄한다. Clearance/Rearrangement/Motion은 step당 최대1이다. Time은 적분값이다.
  전체 return의 제어주기 불변성은 보장하지 않으며 현재50Hz 기준으로 비교한다.

남은 x/scale 검토 사항:

이전 motion_samples.pt의131072 transition을 새 x로 재계산하면 x_acc p50/p90/p99는
약8.72/21.83/61.98이다. a_acc=1에서 exp score의83.0%가0.01미만이다. RMS만 적용해 포화가
해결됐다고 해석하지 않는다. 동일 표본의 a_acc=.1/.05 평균 점수는 .410/.610으로 후속 비교
후보일 뿐 이번 코드에 적용하지 않는다. x_R의 표본 중앙값은약0.000127m이므로 a_R=2에서는
score가1에 매우 가깝다는 점도 검토 대상이다. 바깥 exp 형태를 바꾸지 않고 x 정의·a·단위·
항 간 비중을 따로 검토한다. 앞 절의 선형/pseudo-Huber 검증은 이 수식의 학습 검증이 아니다.

이번 변경 검증: CPU MDP/모델/PPO 186개 통과, 12개 skip, subtest114개 통과.
GPU 4환경 64 policy step과 자동 reset 6회에서 native 5개 term의 weight=1, 0~1 score,
공식 joint_acc_l2와 독립 계산한 RMS/exp의 일치, RewardManager dt 상쇄, Rearrangement의
현재 step 이동량, 244D 관측·부분 reset 보존을 확인했다. Exit 0. Compileall·공백 검사 통과.
장기 학습이나 새 수식의 1024환경 PPO를 실행한 결과는 아니며, 기존 131072개 표본의 지수
입력 재계산과 구분한다. 검증 및 표본 재계산 자료는 outputs/contact_manipulation/reward_exp_20261015/.


## 2026-10-15 — 최근 iteration 속도 저하 원인 계측

사용자가 최근 수정 과정에서 연산 속도가 느려졌다고 보고하여 실제1024환경 학습16iteration을
별도 계측 스크립트로 실행했다. Production의 제어·보상·reset 설정과 실행코드는 변경하지 않았다.
현재 RTX3090에는 중지된 기존 사용자 학습 PID68242가 약5866MiB만 점유했고 실행 중인 다른
GPU학습은 없었다. 해당 프로세스는 종료/재개하지 않았다.

기존 로그를 먼저 비교했다. 최신15-23-21은1024환경3iteration,15-17-51은2048환경62iteration,
14-22-35는4환경GUI라 직접비교할수없다. 최신1024 첫회제외2회평균은collection3.899+
learning0.285=4.185초. 직전native_min_actions의같은iteration1~2는4.537초이므로 최근exp/RMS
수식만으로추가회귀가발생했다고결론낼수없다. Native action전환 전후에는collection증가가일관된다.

이번 계측은1024환경·물체10개·robot41body/46shape·센서2개·physics100Hz/policy50Hz·
rollout32·PPO4epochs/4minibatches·forward8192·seed0·headless/no video로실행했다.
첫iteration0과cProfile를켠iteration4를제외한1~15의14개를동일index의이전측정과비교했다.
이전값은native_audit_20261015/timing_chunk8192(이전customaction)의보존자료다.

| 단계 | 이전 custom action | 현재 native action |
|---|---:|---:|
| Collection |1.901s|3.251s|
| PPO learning |0.322s|0.313s|
| 합계 |2.223s|3.564s|
| Reset 전체 |0.050s|1.406s|
| Reset의로봇시작자세탐색(상위reset에포함) |0.035s|0.808s|
| Physics step64회 |0.665s|0.875s|
| Action process+apply |0.522s|0.268s|
| Termination/common MDP manager |0.428s|0.406s|
| Reward manager |0.0148s|0.0231s|
| Console/TensorBoard Logger.log |0.0070s|0.0085s|

전체증가약1.341s에대해reset증가약1.356s가가장크다. 현재는iteration당reset호출31.36회/
선택된환경375.43개이고이전은0.571회/73.79개였다. 전체1024환경을매step리셋한다는뜻은아니며,
32policy step중거의매step에서일부환경이종료되어다른subset으로작은reset을반복한다는뜻이다.
작은batch의고정launch/검증/동기화비용이누적된다. 초기자세IKsolve/collision은현재139.14회/
iteration,이전3.57회였다. 현재전체16회학습wall60.55초,마지막iteration2.92초로초기구간보다
낮아졌으므로짧은3iteration로그만으로정상상태평균을추정하지않는다.

원인과해석:

- 공식nativeaction클래스의계산시간자체는이번측정에서감소했다. 하지만의미도바뀌었다:
  position scale0.015→1.0,orientation0.08→1.0,hand목표lead제한제거,workspace/customeffort
  제한제거,gravitycompensationTrue→False. 예를들어위치action0.1은종전1.5mm에서현재10cm다.
  같은정책출력분포라도접촉·실패·episode길이와reset부하가달라진다. 이원인분리는서로다른
  행동trajectory를비교한것이므로각설정의단독인과효과까지증명한것은아니다.
- Reset은공통물체배치와초기vision을재사용한다. 이번raycapture시간평균은약0.0004s/iteration
  (공통장면교체1회포함)으로주병목이아니다. 매번새EEF자세를찾는GPU IK/전체로봇충돌검사,
  선택행상태쓰기와metadata/초기센서검증이남아있다. CPUprofile에서는IK/충돌재시도가반복됐다.
- 새exp/RMS reward의manager증가는약8.3ms다. 전체초단위회귀를설명하지못한다.
  가속도는nativecache를읽으며새PhysXread가아니다. 관측3term도동일snapshot의view다.
- Logging은약8.5ms로주병목이아니다. 별도디버그출력이나productionprofiler를추가하지않았다.
- 고정COM의nativeget_coms→GPU조회는여전히약0.333s/iteration이지만이전도약0.348s로
  이번증가원인과구분한다. 향후고정metadatacache는절감후보다.

가장우선인개선대상은정상동작을만드는공식ActionCfg의연구용scale검토와작은partialresetbatch의
IK후보일괄처리/충돌검사launch절감이다. 관측·제어주기를낮추거나초기자세randomization을없애는
변경은하지않았다. 이번요청은원인확인이며production최적화는적용하지않았다.

계측은호스트wall시간이며CUDA비동기작업의대기위치에따라귀속이달라질수있다. 중첩단계를
합산하지않는다. 서로다른제어매핑·reward·policy상태라동일trajectory의알고리즘A/B도아니다.
진단JSON은종료시저장했고추가cProfile는한iteration만켰다. 최초계측스크립트의nativeaction
import경로오류두번은시뮬레이션진입전실패로,위정상계측자료와분리한다.

자료: outputs/contact_manipulation/slowdown_20261015/의stage_comparison.json,
log_audit_summary.json,current_timing.log,current_timing/{iterations.json,cpu.txt,metadata.json}.


## 2026-10-15 — 구조 변경 중 reset/물리 최적화의 보존 여부 확인

사용자가 구조 변경 때문에 앞서 줄인 reset/physics 연산이 이전 상태로 되돌아갔는지 질문했다.
단순 HEAD 비교 대신 실행 시 저장한 git diff를 공통 commit7be67ac에서 복원해 코드와 설정을
대조했다. 보존된 빠른 실행13-30native_audit_chunk8192,13-32resume3550,14-05native_min_actions,
14-20min_structure,15-23현재 실행을 확인했다.

확인 결과 reset/physics의 주요 연산 최적화가 되돌아간 증거는 없다.

- events.py의 Shared layout preparation 이후2694줄(현재739행~끝)은 위 실행본과 현재가
  byte-for-byte 동일하다. SHA256은 a066fce3b5b052372e245a17e7ef1cd5f1e8543d7f79b8487ee5bd2e826e1bc2.
  배치 생성, GPU IK·충돌 검사, 공통 template, 부분 reset 알고리즘이 포함된다.
- sensors.py 전체755줄도13-30 이후 현재까지 동일하다. 실제 카메라/render product를 만들지
  않는 물체-only GPU raycast, 공유 BVH, tactile17채널 통합 센서 및 전체 batch 갱신 fast path가 유지된다.
- reset_task는 동일한 선택 행 처리이며 전체 환경 초기화로 돌아가지 않았다. 정상 step의
  transition CUDA graph capture/replay도 유지된다. 관측3term은 같은 cache의 view다.
- 공통 scene round에서 singleton env0만 배치/vision을 만들고, 중간 reset에서는 template를
  재사용한다. 물체 재spawn·raycast 재생성이 없다. 선택 행의 물체/선반/로봇 tensor state를
  쓰며, reset 중 settling simulation을 추가하지 않았다. sim.forward는 기존부터 있던 갱신이다.
- 저장 YAML의 sim/scene 설정은13-30,13-32,14-05,14-20,15-23 실행에서 동일하다.
  physics100Hz/policy50Hz,headless render 비활성,10개 물체 pool,robot solver16/2,
  Hand 충돌 proxy 단순화, YCB5mm와 box proxy,geometry sharing,17tactile 통합,금지접촉센서
  제거 설정이 보존됐다. 별도 새1024물리 생성을 반복한 검증은 아니며 저장 설정/소스의 동등성이다.
- 고정 mass/inertia cache와 전체 활성 시 parking force composer 비활성 최적화도 유지된다.
  Native COM 조회는 이전에도 매tick 발생하던 남은 비용으로, 새로운 cache 삭제가 아니다.

실제로 제거/변경된 것은 native ActionCfg 전환 시의 custom 제어 매핑·workspace/effort/hand lead
제약과 gravity/gain 설정이다. 이는 당시 공식 기본 동작 사용 및 min branch 설정 요청에 따라
변경·문서화한 항목이며, reset이나 collider 최적화를 복구 전 상태로 되돌린 변경은 아니다.
다만 제어량·접촉/종료 양상을 바꾸어 기존 reset 경로를 더 자주 호출하게 만들 수 있다.

기존8개씩 IK 후보 처리와 first.cpu().tolist 동기화/pending 반복은23-25 실행본에도 있었다.
현재 계측의 작은 reset 평균12환경/호출에서는 이 고정비가 두드러진다. iteration당 reset
0.571→31.36회,전체reset비용0.050→1.406초라는 이전 절의 계측과 일치한다. 향후 해당 작은
batch에서 후보 폭을 늘리는 최적화는 남아 있지만 이번 확인에서 production은 변경하지 않았다.

증거는 outputs/contact_manipulation/slowdown_20261015/reset_optimization_forensics.json 및
동일 폴더의 로그/설정 비교 자료에 보존한다. 기존 HEAD는 공유 배치 도입 이전 코드도 포함하므로
HEAD와의 차이만으로 최근 구조 정리에서 회귀했다고 판단하지 않는다.


## 2026-10-15 — 예제와 robot velocity 제한 비교

사용자가 현재 로봇 속도 제한과 examples/의 차이를 질문했다. 설정과 설치된 actuator 구현을
읽기 전용으로 확인했으며 production 값은 바꾸지 않았다.

- 현재 scene.py의 arm velocity_limit_sim은 shoulder_pan/lift/elbow 각각3.14rad/s,
  wrist1/2/3 각각6.28rad/s다. 로컬 examples/src/ur5e.py의 명시 숫자 및 보존된 min의
  ur5e.py 숫자와 같다. 보존된 min은 velocity_limit_sim을 사용한다.
- 로컬 examples는 구형 ImplicitActuatorCfg.velocity_limit만 지정한다. 설치된 Isaac Lab
  actuator_pd.py81~91은 이 경우 해당 값을 무시하고 USD에서 읽은 제한을 사용한다. 따라서
  로컬 설정 숫자가 같다고 실제 solver 값까지 같다고 단정하지 않는다. 우리 코드는
  velocity_limit_sim이므로 위 숫자를 PhysX joint max velocity로 명시한다.
- 우리 Inspire Hand의 모든 가동 관절은 velocity_limit_sim=1rad/s다. 원본 URDF thumb는
 2rad/s, 다른 손가락은1rad/s지만 actuator 설정이 최종 override한다. 로컬 예제 gripper의
  지정값2rad/s는 Robotiq이며 동일한 손 모델의 비교가 아니다.
- 로컬 예제에는 추가로 hand_velocity termination threshold1rad/s가 있다. 이름과 달리
  terminations.py160은 UR5e 앞6관절의 abs(joint_vel)을 검사한다. 이는 속도 clamp가 아니라
  초과 시 episode 종료다. 현재 연구 환경에는 로봇 속도 자체에 의한 종료 조건이 없다.
- 예제 joint_vel_l2 reward weight-.03도 현재 연구 목적과 차이가 있다. 우리는 일정한 고속
  자체보다 가속도와 F/T 크기의 급격한 증가에 비용을 주는 설계다.
- 앞서 native action 전환에서 제거했다고 설명한 hand 제약은 **측정 관절 위치 기준 목표
  위치의 lead 제한**이다. 물리 관절 속도 상한1rad/s는 제거되지 않았다. 또한 OSC의 pose
  action scale은 관절 속도 상한과 별개의 설정이다. 큰 목표 오차가 물리 속도 제한 안에서도
  큰 torque/급격한 가속을 만들 수 있으므로 속도 상한이 같다고 제어 동작까지 같지는 않다.

근거: current config/ur5e/scene.py642~656, 로컬예제sweeping_policy/src/ur5e.py80~116,
예제shelf_sweep_ramdom_env_cfg.py185 및 mdp/terminations.py150~160,
IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py80~101.


## 2026-10-15 — UR5e 제어 설정을 예제 기준으로 명시

사용자가 UR5e 설정을 examples/를 참고해 구성하라고 요청했다. 추가 범위 질문에는
**제어·물리 설정만 예제 기준으로 유지**를 선택했다. 예제의1rad/s초과 episode 종료 조건을
연구 MDP에 추가하지 않는다.

현재 UR5e의 velocity_limit_sim(shoulder/elbow3.14,wrist6.28rad/s),effort_limit_sim
(shoulder/elbow150,wrist28Nm)은 예제와 이미 같아 수치를 유지했다. 구형 velocity_limit의
무시 동작을 따라하지 않고 현재 Isaac Lab의 *_sim 필드로 PhysX에 적용한다.
OSC는 보존된 min의 UR5eSweepKpOSCRelativeObsEnvCfg 기준 pose_rel/variable_kp,
position/orientation_scale1,stiffness_scale100,stiffnesslimits50~200,dampingratio1,
full inertia decoupling,gravity compensation=False,nullspace none을 유지했다.

코드는 예제처럼 UR5eContactManipulationEnvCfg.__post_init__에서도 arm actuator의
stiffness와 damping을0으로 명시하도록 정리했다. Asset factory에서도 이미0이었으므로
이번 명시는 현재 실행의 제어값을 바꾸지 않는다. Joint position 예제의 높은 arm PD gain을
OSC에 복사하여 torque가 중복되는 일을 막는 설정 위치다. Hand 설정과 초기자세/센서,
충돌 형상 최적화·기존접촉해소속도1m/s·solver16/2는 보존했다. 접촉해소속도는 관절의일반
velocity 상한이 아니며, 예제5m/s를 단순복사하면 조립된 hand에도 적용되므로 올리지 않았다.

Production 변경은 기존 UR5e cfg의2개 대입과참조주석,README 설명에 한정한다. 새파일이나
controller subclass는 추가하지 않았다. Compileall·git diff공백 검사를 통과했다.

GPU1환경에서실제PhysX max-velocity와ArticulationData effort/gain을확인했다. Arm6관절의
속도3.14/6.28rad/s,토크150/28Nm,stiffness/damping0이모두기대값과일치했다. Hand6독립관절의
속도1rad/s및기존3개termination term도확인했다. Native OperationalSpaceControllerAction으로
4policy step의관측·보상이finite였다. 자료는outputs/contact_manipulation/ur5e_example_20261015/
validation.json/log이다. 초기검증스크립트는PhysX최대속도조회가CPU tensor를반환하는데GPU
기대값과비교하여중단됐고,비교device만수정한재실행에서위검증을완료했다. Production문제가아니다.


## 2026-10-15 — Scene reset 연산 절감 검토와 후보 batch 실험

사용자가 현재 구조에서 scene reset 시간을 더 줄일 수 있는지 검토하도록 요청했다. 이번에는
production을 변경하지 않고 기존 계측, 소스의 데이터 사용 여부, 실제 GPU sampler의 임시
복사본을 비교했다. 현재의 공유 배치/초기 vision, 매 reset 새로운 연속 EEF pose, elbow-up/
forearm/충돌 조건, 선택 행만 복원하는 규약을 유지하는 방안이다.

우선순위가 높은 개선 후보:

1. Reset용 선택 행 snapshot: 현재 reset_task는 _read_transition을 호출해 정상 step과 같은
   물체 COM/속도, 모든 로봇 링크의 COM/가속도/각속도, F/T 자중·관성 보상까지 계산한다.
   그러나 TaskState.reset은 초기물체 pose/corners, sensor위치, TCP pose, joint q/dq를 사용하고
   sample의FT/tactile/action은 즉시0/-1/0으로 덮어쓴다. 초기물체속도/보상wrench/overrange/
   state_valid도 reset에서 소비하지 않는다. 필요한 native pose·joint만 선택 행으로 읽으면
   이 불필요한 연산을 생략할 수 있다. 정상 step의 센서 보상은 유지한다.
   기존 계측 reset 전체1.406s/iteration 중 reset MDP는0.411s, 그 안의 _read_transition은
   0.274s이고, 그 안의 world COM 조회·변환은0.176s다. 중첩 시간이며 합산하지 않는다.
2. 작은 reset batch의 IK 후보 폭 조정: 현재 후보8개 고정 폭과 CPU acceptance 왕복을 반복한다.
   현재 평균약12환경/호출에서 이를 더 크게 묶으면 GPU launch와 CPU동기화 횟수를 줄인다.
   단순히 가능한 최대80개를 쓰는 것은 과도한 후보 연산을 늘릴 수 있으므로 아래 실험 결과를
   바탕으로 batch 크기별 폭을 정해야 한다. 총 동시후보 상한1024는 유지한다.
3. 공통 template 검증·전처리 cache: catalog lower/upper/upright 유효성은 초기1회, role관계와
   초기 vision정렬·local corners·장면 obstacle OBB·로봇시작영역은 scene generation당1회
   계산한다. 부분 reset에서는 template tensor를 재사용하며 실제 새 robot/shelf pose 검사만
   수행한다. 현재 _validate_reset_data의 별도 CPUprofile는 약0.036s/iteration이었다.
4. SAT GPU kernel 합치기: 현재 world AABB로 분리가 확인된 pair나 IK실패 후보도 뒤의15축
   tensor SAT를 계산한다. Device kernel에서 저렴한 pose/IK조건 실패 시 후보를 건너뛰고,
   AABB분리 pair의 세부 SAT를 생략하며 첫 충돌에서 종료하면 launch/임시tensor/불필요한
   세부연산을 줄일 수 있다. 검사 조건·거리 여유를 삭제하는 변경이 아니다. 기존 평균
   collision검사0.466s는 위 IK부모0.808s의 일부다.

실제 GPU microbenchmark:

RTX3090에서 실제 RobotPoseModel/GPU IK/충돌검사를 사용했다. 물리환경1개에서 공통장면을
초기화한 뒤1/4/12/32개 reset요청을 만들고, sampler의 후보 폭만8/32/64/80으로 바꾼 임시
함수들을 비교했다. Warmup 후각5회, 호출전후 CUDA synchronize로 sampler 전체wall시간을
쟀다. 아래는 한 shared scene·동일 seed 표본의 평균이며 전체 reset/학습 시간이 아니다.

| 선택 환경 수 | 기존8후보 | 32후보 | 64후보 | 80후보 |
|---:|---:|---:|---:|---:|
|1|6.17ms|6.02ms|6.36ms|7.53ms|
|4|34.29ms|13.39ms|11.14ms|11.76ms|
|12|33.17ms|19.79ms|23.06ms|26.30ms|
|32|56.26ms|40.61ms|59.86ms|70.10ms|

12환경에서는8→32후보로약40.3%감소했고, solver호출은6→2회였다. 32환경에서80후보는
오히려기존보다느렸다. 따라서 “작은batch는항상80개”를확정하지않는다. 1환경의첫8개에서
해가발견되는표본에는확대의이득이작았다. 폭에따른early-exit와불필요한후보수의tradeoff다.
12환경의peak allocated는8후보약50.8MiB,32후보174.7MiB,80후보423.0MiB였다.

비교한모든16case에서최초유효후보index가정확히같았고,선택jointpose는rtol1e-4/atol1e-5
내에서일치했다. RNG가절대후보번호에기반하고branch=%4,strength=absolute_attempt//8을
유지했기때문이다. 다른scene/경계값의동등성과실제1024환경전체iteration개선은적용후별도
검증해야한다. 이실험에서physics simulation이나PPO학습을반복하지않았다.

낮은우선순위는sim.forward제거(현재약0.0036s/iteration)다. 초기FK갱신역할에비해절감이작다.
센서중복reset이나Python tuple재조립도후보지만위큰항목뒤에처리한다. 고정localCOMcache는
정상step에도효과가있을수있으나reset snapshot절감과중복계산하지않는다.

이번production최적화는미적용이다. 검토/재현자료는
outputs/contact_manipulation/reset_review_20261015/{review_summary.json,candidate_widths.json,
candidate_widths.log,benchmark_candidates.py}에보존한다.


## 2026-10-15 — 작은 reset의 IK batch와 공유 장면 cache 적용

사용자는 COM 비용이 OSC 정리 때 해결되지 않았는지, 충돌 검사가 이미 GPU가 아니었는지
확인했다. 제어 경로의 COM→TCP 보정 조회는 native OSC 전환으로 제거됐다. 그러나
`_read_transition → body_com_pos_w → body_com_pose_b`는 F/T 자중·관성 보상을 위해 남아 있고,
현재 Isaac Lab은 physics timestamp마다 `get_coms().to(device)`로 고정 local COM도 다시 읽는다.
이 두 경로를 구분해야 한다. 충돌 검사는 기존부터 GPU tensor 연산이었다. Kernel 합치기란
CPU 검사를 GPU로 옮기는 뜻이 아니라 launch·임시 tensor·불필요한 세부 SAT를 줄이자는 뜻이다.
이번에 승인·구현한 범위는 아래 두 가지다. F/T 보상이나 collision kernel 자체는 변경하지 않았다.

- 작은 reset의 IK 후보 폭: 32환경 이하 요청은 최대32후보, 나머지는8후보씩 검사한다.
  동시 후보 총량은 기존 `ik_batch_size` 이하이며 절대 후보 번호별 RNG, branch `%4`,
  perturbation strength `absolute_attempt//8`, 첫 유효 후보 선택, 충돌 margin/조건을 유지한다.
- 공유 장면의 입력 재사용: singleton proposal·obstacle OBB·robot start region을 cache한다.
  역할 mask·local center/corners/upright·역할 순서의 initial vision과 producer 검증도 장면당1회다.
  새 template은 cache를 교체하고, robot/spawn 규칙 변경은 robot scene cache를 갱신한다.
  실제 새 로봇 자세의 IK·충돌·허용영역 검사 및 로봇/선반 finite pose 검사는 매번 유지한다.

Native reset_scene이 정상 적용을 마친 뒤 같은 env_ids에만1회 사용 가능한 marker를 전달한다.
reset_task는 marker를 소비하고 template identity로 검증된 공통 geometry를 재사용한다.
외부 SceneBatch나 다른 reset 대상은 기존 검증을 유지한다. Catalog/template은 생성 후
불변이라는 native producer 계약이다. 공통 vision의 robot-base frame은 유지하고 EEF frame
변환은 기존 observation 경로에서 계속 수행한다. Native EventManager/asset writes와
sim.forward를 유지했고 production 파일은 추가하지 않았다.

검증: CPU 기존/추가 테스트와 임시 후보 선택 oracle을 합쳐283passed,12skipped,145subtests.
그 중 임시 후보 폭 oracle은92case다. 실제 CUDA IK는3장면×7요청크기(1/4/12/32/64/128/1024)
×4변형(cache유무,8후보/변경폭),총84case에서 성공/실패와 첫 유효 후보 번호 및 시작영역이
정확히 같았고, 관절값/TCP는rtol1e-4/atol1e-5내에서 같았다. 규칙 변경 cache invalidation도
3장면에서 통과했다. 미세 수치 차이 이후의 전체 물리/학습 궤적이 동일하다는 의미는 아니다.

실제 RTX3090,1024환경,10물체,seed0,dt0.01/decimation2,native OSC/PPO로 변경 전후 각각
16iteration을 새로 실행했다. 첫 iteration0과 cProfile을 켠 iteration4를 제외한14개 평균이다.
변경 전은 보존한 이전 reset 함수만 임시 계측 프로세스에서 복원했고 production은 수정하지 않았다.

| 항목 | 변경 전 | 변경 후 |
|---|---:|---:|
| 전체 iteration(collection+learn) |3.5667s|3.2081s|
| reset 전체/iteration |1.4112s|1.0180s|
| 로봇 시작 탐색/iteration |0.8458s|0.5214s|
| reset MDP/iteration |0.4088s|0.3358s|
| IK solve/충돌 검사 호출/iteration |142.21회|56.64회|
| physics64step/iteration |0.8666s|0.8711s|
| PPO update/iteration |0.3167s|0.3224s|

전체 약10.1%,reset 약27.9% 감소했다. 위 하위 구간은 중첩된 host wall time이며 합산하지 않는다.
두 새 학습의 reset 대상 수는 평균336.4→367.4행/iteration으로 달랐고 reset 호출 수는
31.07→31.36회였다. 완전히 같은 궤적의 replay나 장기 처리량 보장이 아닌 짧은 실행 비교다.
공유 데이터 검증은 평균31회/iteration에서 scene 교체 때만으로 줄었다.

후보 확대는 모든 장면에 이득이 아니다.3장면 sampler 별도 평균에서12요청은19.21→14.94ms였지만
32요청은29.77→31.80ms였다. 쉬운 장면은 첫8개에서 모두 해를 찾아32개 계산이 낭비된다.
현재 학습의 작은 reset 분포에서 얻은 이득을 전체 장면/크기에 일반화하지 않는다. 쉬운 장면의
후보 폭 자동 조정은 후속 검토 후보이며 현재 결과는1~2초/iteration 목표에 아직 도달하지 않았다.

자료: outputs/contact_manipulation/reset_optimization_20261015/의 comparison.json,
before/after/iterations.json,cuda_equivalence.json과 계측·검증 스크립트.

추가 실제 rollout 검증:4환경410policy step에서 자동reset49개와 수동부분reset1회를 확인했다.
초기geometry/producer검증은 generation0(step0),generation1(step400)에서 각singleton1회뿐이었다.
부분reset시 미선택환경의nativepose·state·metadata·관측과terminal snapshot이 유지됐으며,
244D observation packing도 전step에서 정확히 일치했다. cache_rollout_validation.json에 저장했다.


## 2026-10-15 — CPU 한정이 아닌 전체 연산 절감 추가 검토

사용자는 CPU→GPU 전환뿐 아니라 연산을 빠르게 할 수 있는 요소 전반을 검토하도록 명확히 했다.
이번 범위는 소스·기존 계측 검토이며 production 변경은 하지 않았다. 추가 native 기능 재작성이나
정책/물리 주기·reward/모델 구조 변경을 속도 최적화에 섞지 않는다.

현재 unprofiled14회 평균에서 관련 비용은 reset MDP0.3358s(그 안의 transition0.2802s,
worldCOM0.1801s), 정상step worldCOM0.1802s/objectCOM0.0547s, IK 후보검사0.5024s
(충돌0.3523s,IK solve0.0818s),physics64step0.8711s,action처리+적용0.2775s,
정책 rollout0.1009s,PPO0.3161s,step저장/bootstrap0.0889s다. 중첩·GPU대기 포함 시간이며
합산하거나 전부 절감 가능한 시간으로 보지 않는다. broader_audit.json에 근거를 저장했다.

우선 후보:

1. Reset 전용 snapshot. TaskState.reset이 쓰는 center/corners/rotation/shelfpose/sensorposition/
   TCP/jointsample[:31]만 읽는다. 현재 불필요한 object COM/velocity·bodyCOM/가속도·F/T 보상을
   계산한 뒤 F/T/tactile/action을0/−1/0으로 덮는다. 실제 초기 observation을 유지하며 해당
   연산과 native 조회 자체를 생략할 수 있다.
2. 정상step의 고정 local COM을1회 GPU cache. Native get_coms는 CPU tensor(device−1)를 만들고
   body_com_pose_b는 physics timestamp마다 이를GPU로 복사한다. Reset joint state write도
   localCOMtimestamp를−1로 만든다. 현재 COM/mass/inertia randomization이 없으므로 task reader에서
   localCOM캐시+GPU link→COM 변환으로 대체 가능하다. 공식 OSC/native asset writes는 유지한다.
3. 충돌 검사. 현재 IK실패·ground/elbow실패 후보도 전체 OBB SAT를 계산하고, AABB로분리된 pair도
   세부15축을 계산한다. 조건/margin은 유지하고 Warp kernel에서 조기 탈락·분리 pair 건너뛰기,
   첫 충돌 종료 및 임시 tensor/launch 감소를 검토한다. IK는 이미 Warp CUDA다.
4. Native PhysxCfg.gpu_max_num_partitions. 현재 기본8. 기존 별도실험은8→1로64physics가
   0.799→0.699s(12.6%),8→4는0.738s였다. 현재nativeOSC/동적접촉에서1/2/4/8 A/B가 필요하다.
   기존combinedtrace에서는 solveBlockUnified launch9216→1152였지만 다른 최적화도 함께있어
   현재 개선량으로 단정하지 않는다. 물리/정책Hz를 바꾸지 않는 native설정 후보다.
5. Timeout이 없는 step의 terminal critic 생략. process_env_step은 terminal_env_ids가비지않으면
   time_outs가모두False인실패종료때도final GNN/critic을계산하고 이후0을곱한다. 실제 공통round
   boundary의 timeout metadata로필요성만전달하면 불필요추론을생략할수있다. 기존 nativePPO와
   실패bootstrap0,실제timeoutfinalstate bootstrap은유지해야한다. 무작정torch.any→Python판단은
   새GPU동기화를만드므로피한다. 저장/bootstrap전체0.089s가상한성관련구간이지보장절감은아니다.
6. 후보폭조정. 현재count≤32이면무조건32후보인데쉬운장면은12요청6.44→13.46ms,
   32요청10.69→27.50ms로손해다. 초기1024공통reset의이미host로받는acceptedindex분포로
   장면별8/16/32폭을정하면추가GPU읽기없이과잉계산을줄일수있다. firstvalid/RNG는보존가능하다.
7. Candidate accepted/pending을GPUmask와batchedtensor로유지. first.cpu().tolist왕복과
   Pythonresultlist→stack을줄인다. Warp IK/충돌kernel이이미성공한행을조기건너뛰어야하며,
   host분기만없애고80후보를모두돌리는방법은오히려느릴수있다. Seed CPU생성만옮기는이득은작다.
8. 순수tensor단위 compile/CUDA graph. 정상transition은이미graph,IK는이미fused다. 부분reset의
   새snapshot/충돌검사·모델mask/MLP등에적용여지. nativeasset조회/PhysX전체를capture하려고
   ManagerBased흐름을재작성하는것은피한다. 모델AMP는현재FP32전용Tritonedgekernel을우회하여
   densefallback으로가므로단순활성화가오히려느릴수있다. 지원dtype확장후비교해야한다.

낮은우선순위: groupedtactile0.0067s,logger0.0091s,NaN검사0.0044s,vision평균0.00044s,
CPUseed metadata(sampler바깥약0.019s),sim.forward약0.0036s. 이미카메라렌더없음·Fabric·
selfcollision/CCD비활성·handproxy/YCBbox·무효물체0wrench최적화가있다. 물체solver16/2→8/1은
기존물리실험이득0.7%라후순위다. Hand의19fixed중17tactile과F/T연결은보존해야하며,
비센서palm2fixed병합은제한적후보다. 다른6종contactoffset은기존USD약1.962mm여서5mm로
통일하면오히려커진다. GNN scene 결과는EEF frame에따라매step달라져episode전체cache는부적절하다.

Reset1.018s를완전히없애도기준3.208s−1.018s=2.190s이므로1~2초목표에는정상step/physics/
policy경로개선도필요하다. 사용자승인없이제어주기·rollout길이·PPOepoch·termination을변경하지않는다.


## 2026-10-15 — 전체 연산 최적화 적용,1024환경 평균1.84초/iteration

사용자가 앞선 전체 개선안을 진행하도록 승인했다. Production파일을 추가하지 않고 기존
nativeManagerBased환경·공식OSC/handActionCfg·nativePPO 흐름 안에서 작업했다. 제어주기100Hz,
정책50Hz,solver16/2,18Daction/244Dobs,initialvision/role/mask/이력·reward/termination 수식,
PPO4epochs/4minibatches/rollout32는 유지했다.

구현:

- `_read_reset_snapshot`: reset이 실제소비하는 OBB·jointq/dq·TCP·sensor위치·shelfpose만읽는다.
  초기화때버리는F/T/tactile/action, COM/가속도/물체속도/자중관성보상을계산하지않는다.
- `_bind_scene`: nativepublic body_com_pos_b/object_com_pos_b를1회GPUcache한다. 정상transition은
  rawlink/objectquaternion으로localCOM을회전해worldCOM을구하고distalbody만FT보상한다.
  COM/mass/inertiarandomization은없으며추가시cache갱신필요. 공식OSC/nativeassetwrite는보존한다.
- FP32CUDA충돌검사는Warpkernel로collidertransform/ground/elbow/scene/self조건통합.
  IKinvalid·AABB분리·첫충돌조기종료. CPU/다른dtype는eageroracle. Scene15mm/self2mm여유유지.
  거의평행한축에서기존relativeFP32식이겹침을놓치던오류를발견해해당축만directworldFP64
  projection으로계산한다. 수정후20kCUDA판정은수정eager와일치,doubleoracle19999/20000일치.
  남은1개는경계11.1nm FP32차이. 영구회귀fixture도tests/test_mdp.py에넣었다.
- CUDA후보탐색은첫8개,이후작은reset은최대32개/큰reset은8개씩고정GPUrow로처리한다.
  acceptedmask를GPU에유지해성공행의비선형IK/충돌을건너뛰고절대firstvalid를선택한다.
  동시후보상한1024와최대8개power-of-two CUDAgraphcache. 장면inputs는매replay복사한다.
  Graphcapture/warm/buffer를inference_mode(False)로생성해Native학습/일반reset간전환시
  PyTorchPhiloxinferenceTensor inplace오류를막았다. 정상transitiongraph도같이정리했다.
  첫validindexCPU왕복은후보chunk마다에서탐색종료1회로줄이고nativewriter는batchedtensor직접사용.
  CPUgenericlistAPI·episodeSeed문자열MT난수·Warp절대candidateRNG/branch%4/strength//8은유지.
- `scene_time_out`은reset전에공통round의hostcounter로time_outs_possible을제공한다.
  PPO는False면실패종료의무의미한terminalGNN/critic을생략하고진짜timeoutfinalstatebootstrap은
  유지한다. Generic환경은기존fallback. nativecounter와init_at_random_ep_len=False가전제이며
  외부에서episodecounter를임의로채우는방식은지원하지않는다.
- `agent.actor.compile_model=True`: CUDApuremodel+maskforward를Inductorfullgraph로compile.
  모델/파라미터/checkpointkey와FP32유지,CPUeager,Falseoverride지원. Reduce-overheadCUDAgraph는
  inference→training전환replay오류가재현돼사용하지않는다. 기본Inductor는통합검증통과했다.
- Native PhysxCfg.gpu_max_num_partitions=1. 실제1024환경410nativeenv.step에서1/2/4/8을비교:
  physics64hostwall은각0.7572/0.7634/0.7889/0.8579s였다. 처음의자동reset없는긴pose-hold진단은
  비정상drift를만들어discarded_raw_*로분리하고채택근거에서제외했다. 선택된비교는native
  termination/reset을포함하고같은초기상태/고정난수작은행동으로수행했다.

주요분리benchmark:

- 실제RobotPoseModel충돌1024후보:원본eager24.093ms→fused0.413ms,
  임시GPU메모리440.37MiB→0.0508MiB. 수정eager26.239ms는별도기록해비교과장을피했다.
- 같은fused충돌을쓰는sampler12환경:어려운장면5.729→2.353ms,쉬운장면3.061→1.544ms.
  1024환경은213.227→60.724ms/51.695→43.063ms였다. 일부쉬운64환경은0.3ms느려지는경우가
  있어모든scene·size에보편이득이라고하지않는다. 2장면×7크기×반복/모드비교28case통과.
- 모델1024inference1.982→0.776ms,8192forward+backward16.423→9.986ms.
  Puremodel수치이며Gaussian/loss/optimizer전체절감으로오해하지않는다. 초기compile약14s별도.

통합학습계측:RTX3090,1024physicalenv,10objects,seed0,nativeOSC/PPO로30iteration실행.
0(초기compile/warm),4(cProfile)를제외한28평균1.8370s,median1.8209s,범위1.7280~2.0216s.
같은index의직전14개평균3.2081s와새14개1.8460s비교는약42.46%감소다.

| 관련구간/iteration | 직전14회 | 새14회 | 새28회 |
|---|---:|---:|---:|
| 전체collection+learn |3.2081s|1.8460s|1.8370s|
| reset전체 |1.0180s|0.3188s|0.3133s|
| robotstart탐색 |0.5214s|0.0794s|0.0745s|
| resetMDP |0.3358s|0.0858s|0.0858s|
| 정상 transition + reset transition 읽기 |0.5634s|0.0429s|0.0428s|
| physics64step |0.8711s|0.6823s|0.6793s|
| PPOupdate |0.3161s|0.2295s|0.2275s|
| rolloutpolicy |0.1009s|0.0640s|0.0631s|
| storage/bootstrap |0.0889s|0.0062s|0.0061s|

중첩hostwall시간이므로합산하지않는다. 새학습trajectory와reset행수는같지않고,이전innerGNN
계측wrapper는torch.compilefullgraph를방해하므로새계측은puremodel내부wrapper를제거했다.
학습기본실행에profiler/추가CUDA강제동기화/디버그로그는추가하지않았다. 90000iteration성능이나
장기수렴/안정성을검증한것이아니며현재28회속도단순환산은약45.9시간이다.

검증:기존FT보상readeroracle와비교하는실제4env410step을통과했다. 정상hotpath COM조회0회,
8회의명시적oracle만robot/objectCOM을각8회조회했다. 초기관측앞31D/사용posefield와packing이
일치하고부분reset의다른행보존/commonround400cache갱신을통과했다. 독립SciPy13env임의3D
quat/localCOM/36morigin검사최대오차1.49e−8m,속도9.31e−10m/s였다. 모델실제NativeActor의
inference→gradient→parameterupdate→inference/checkpoint키검증도통과했다.

CPU198passed/14skipped. GPU전체회귀중기존testfixture가Warp초기cache=None을복원해다음IK
테스트를 깨뜨리는 문제를 발견했다. Fixture 초기화 순서를 수정한 최종 GPU 전체 검사에서
205 passed,7 skipped,145 subtests passed였다. Compileall과 git diff 공백 검사도 통과했다.
자료는outputs/contact_manipulation/full_optimization_20261015/의comparison.json,final/,partition_*,
clearance_*,candidate_gpu.*,model_compile_benchmark.*,validate_rollout.*,review_com_snapshot.*,tests_gpu.log.

### 2026-10-16 — 장시간 CUDA 719 재발: Xid69와 OSC 입력 토크 조사

이번 사고는 `logs/rsl_rl/contact_manipulation/2026-10-15_17-57-45`, PID246920이다.
Kit log `kit_20261015_175732.log:5224`의 UTC2026-10-15 18:14:15는
KST2026-10-16 03:14:15이며 uptime33,403,465ms(약9시간16분43초) 후 발생했다.
DOF velocity getter는 PhysX CUDA 실패 뒤에 실패하므로 getter 자체를 근본 원인으로
보지 않는다. 현재까지 CUDA719를 일으킨 단일 근본 원인은 확정하지 못했다.

- 같은 시각의 kernel journal에는 Xid69(ClassError), Class0xc7c0, Offset0x1b0,
  Data0x41, ErrorCode0x53이 있다. NVIDIA 공개 header에서 Class는 AMPERE_COMPUTE_B,
  Offset은 LAUNCH_DMA이다. GPU compute 명령/DMA 처리 중 포착된 오류라는 단서이며,
  드라이버 결함이나 특정 custom kernel의 범위 밖 접근을 확정하는 정보는 아니다.
  ErrorCode0x53의 공식 공개 정의는 찾지 못했다. 이전 사고는 Xid13/MISSING_INLINE_DATA로
  이번과 다르며, CUDA719라는 표면 증상만으로 두 사고의 원인을 동일시하지 않는다.
- 마지막 완료 iteration19099(총19100), 최신 checkpoint model_19050.pt이다.
  TensorBoard20개 scalar 전구간과 0/1000/10000/19000/19050의 모델·optimizer tensor가
  모두 finite였다. 직전100회 collection평균1.388s/learning0.230s로 처리량 붕괴도 없다.
  이 통계는 각 환경의 순간 관절 상태나 제어 입력의 정상성을 보장하지 않는다.
- 초기화 이후 최초 오류 전까지 Kit의 buffer overflow/OOM/NaN 경고는 없다.
  allocator 경고는 CUDA 실패와 종료 처리 이후 발생했다. 따라서 이 메시지만으로
  VRAM 부족을 원인으로 보지 않는다. 해당 시간대 kernel journal에도 추가 OOM/PCIe
  오류는 없었다. 사고 당시 VRAM peak/온도 이력은 저장돼 있지 않아 배제까지는 못 한다.
- 19100×32=611200=400×1528이므로 마지막 완료 rollout은 공통 장면 교체 경계와
  맞는다. 새 장면 첫 physics와의 연관성을 조사할 단서지만, 다음 iteration의 실제
  실패 substep과 개별 환경 상태가 저장되지 않아 reset 원인으로 단정하지 않는다.
- transition/reset CUDA graph의 입력·출력 소유권/eviction, Warp IK padding/indices,
  충돌/raycast indices를 검토했으나 명확한 OOB/dangling buffer는 발견하지 못했다.
  이는 정적 검토이며 Compute Sanitizer 검증이나 비동기 race 부재 증명이 아니다.

새 프로세스에서 최신 checkpoint를 불러 native env.step/reset/OSC를 사용하는 진단:

- 128환경800policy/1600physics step,466reset,28.97초를 오류 없이 완료했다.
  OSC raw토크와 PhysX get_dof_actuation_forces 입력값이 정확히 같았고 최대492512Nm였다.
  반면 data.applied_torque는150Nm 이하로 보였다. Jacobian 최소 singular value7.65e-7,
  최대 condition2.53e6, 전체 articulation 최대 관절 속도156.46rad/s였다.
- seed0의 scene_generation1528을 다시 생성한128환경400policy/800physics step도
  CUDA719 없이15.44초에 완료했다. 여기서 OSC/PhysX 입력 최대800880.94Nm,
  configured limit 대비 최대5339.2배, Jacobian condition1.44e7을 확인했다.
  checkpoint에는 물리/RNG/환경별 reset 횟수가 없어 원래 실패 trajectory 재현은 아니다.
- 진단에서는 순수 정책 compile만 startup 절감을 위해 껐고, 정책 출력은 stochastic으로
  사용했다. production 제어/모델/MDP/물리 설정은 이 조사에서 변경하지 않았다.
- 설치된 native ImplicitActuator.compute는 computed/applied_effort 통계만 clip하고
  원래 control_action을 반환한다. 따라서 gain0의 arm에 입력한 feed-forward OSC effort가
  통계에 보이는150/28Nm 상한으로 제한되어 제출된다고 가정하면 안 된다. 실제 SDK 입력
  조회로 초과를 확인했다. 이 값은 solver에서 적분 후 측정한 모터 토크와는 구별한다.
  native OSC의 full inertial inverse가 near-singular Jacobian에서 큰 유한 토크를 만들고,
  native implicit actuator 경로가 그 입력을 그대로 전달하는 위험이 실재한다.
- 상세 재실행에서는 최대800880.94Nm가 새 scene의 첫 physics step에 나왔다.
  해당 초기 arm q의 wrist_2=-0.00010765rad이고 관절 속도는 모두0이었다. 즉 외부 접촉이
  누적된 후에만 생기는 현상이 아니라 충돌 없는 elbow-up IK도 손목 특이점에 가까울 수
  있다. 초기 자세 검사에는 Jacobian condition/singular value 기준이 없다. 이 실행의
  전체 최대 arm/독립hand/mimic 속도는110.18/970.08/889.49rad/s였으나 모두 finite라
  기존 invalid-state 검사로는 구별되지 않았다. 이 값은 장시간 사고 당시 상태가 아닌
  동일 정책/장면의 새128환경 진단 실행에서 측정한 것이다.
- min branch도 gain0 ImplicitActuator와 native OSC를 조합하고 별도 effort clamp가 없다.
  예제를 따른 사실 자체가 이 구성에서 feed-forward 상한의 실효성을 보장하지 않는다.
  또한 min 예제는 arm 속도1rad/s 초과 종료가 있지만 사용자는 제어/물리 설정만 예제에
  맞추기로 하여 우리에는 그 종료 조건이 없다. 이를 이번 조사에서 무단 추가하지 않았다.
  native IdealPDActuator(gain0)의 명시적 effort clipping을 진단 대안으로 비교했다.
  이러한 토크 급증은 우선 해결할 물리 안정성 문제이나 CUDA719의 직접 원인으로 아직
  입증되지 않았다. 유한성 검사만으로는 이를 검출하지 못하고 종료 검사는 physics 이후다.

같은128환경/scene1528/19050policy/400step에서 arm actuator만 공식 IdealPD로 바꾼
진단 A/B(Hand actuator는 유지):

| 측정 | 현재 Implicit | 진단 IdealPD(gain0,effort_limit150/28) |
|---|---:|---:|
| 최대 raw OSC 명령 |800880.94Nm|800880.94Nm|
| 최대 PhysX 입력 명령 |800880.94Nm|150Nm|
| 최대 arm 속도 |110.18rad/s|6.28263rad/s|
| 최대 독립hand 속도 |970.08rad/s|3.68736rad/s|
| 최대 mimic 속도 |889.49rad/s|2.26674rad/s|

첫 step의 q/dq/action/rawOSC는 정확히 같고, 첫 물리 적분 직후 arm/hand 속도는
A10.2828/151.8687rad/s, B6.28134/1.41771rad/s였다. 따라서 잘못된 초과 입력 경로가
물리 불안정에 실제 영향을 주는 것을 확인했다. 양쪽 모두 CUDA719는 재현되지 않았다.
이후 trajectory/reset횟수(241vs560)가 달라 학습 성능이나 정책 품질의 동등성 비교는
아니며, IdealPD만으로 특이점 inverse의 문제까지 해결했다는 의미도 아니다.
Production 변경은 하지 않았고 다음 수정 후보는 arm의 native explicit effort 제한과
초기 IK의 near-singular 해 제외다. 물리 timestep/solver를 낮추거나 UR5e mesh를 변경할
근거로 이 결과를 사용하지 않는다.

조사 시점 RTX3090/driver580.173.02이고 이전 PID68242(stopped,5866MiB) 및
현재 PID246920(stopped,6340MiB)가 남아 있었다. 기존 사용자 프로세스는 종료/재개하지
않았다. 자료는 `outputs/contact_manipulation/cuda719_20261016/`의 log_analysis.json,
system_analysis.json,kernel_xid.log,physics_probe*.py/json/pt/log에 보존한다.

공식 근거:
- https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
- https://raw.githubusercontent.com/NVIDIA/open-gpu-doc/master/classes/compute/clc7c0.h
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html

### 2026-10-16 — 사용자 결정: min의 native implicit OSC 구성 유지

아래는 당시 결정 기록이다. 이후 같은 날짜의 "팔 IdealPD 적용 및 측정 torque 연구"에서
사용자가 explicit actuator 전환을 승인하여 현재 구성은 변경되었다.

사용자는 CUDA719 조사 이후에도 다음 구성을 `min`과 동일하게 사용하도록 명시했다.

- 팔 actuator는 `ImplicitActuatorCfg`를 유지한다.
- `effort_limit_sim`은 shoulder_pan/shoulder_lift/elbow 150Nm,
  wrist_1/wrist_2/wrist_3 28Nm이다.
- 팔의 actuator stiffness/damping은 모두0이며 공식
  `OperationalSpaceControllerActionCfg`가 OSC 명령을 계산/전달한다.
- 별도의 OSC 출력 토크 clamp나 `IdealPDActuatorCfg` 전환은 적용하지 않는다.
  IdealPD A/B 결과는 진단 근거로 보존하며 구현 계획으로 취급하지 않는다.

현재 production은 이미 위 설정과 일치하여 제어 동작은 변경하지 않았다. scene.py의
설명과 README에 이 결정을 반영했다. 앞선 조사에서 제안한 explicit actuator 전환은
이번 사용자 결정으로 채택하지 않는다. 초기 IK의 특이점 처리도 이번 작업에서는
변경하지 않았다. CUDA719의 직접 원인은 여전히 확정되지 않았다.

### 2026-10-16 — 실패 원인별 native 종료 로그 및 manipulability 검토

사용자는 실패 원인별 종료 로그 구현과 Motion의 manipulability 항 검토를 요청했다.

- `TerminationsCfg`의 통합 failure term을 object_drop/object_tilt/workspace_exit/
  object_lift/hard_load_or_overrange/shelf_displacement/invalid_state의7개 native term으로
  대체했다. timeout/success는 유지한다. Console/TensorBoard에서
  `Episode_Termination/<원인>`으로 구분하며 별도 logger나 디버그 출력은 추가하지 않았다.
- `mdp.failure(env, reason=...)`는 기존 TaskState의 failure_reasons 열 view만 반환한다.
  `failure(env)`의 aggregate API는 유지한다. 첫 호출만 완료 transition을 평가하고 이후
  원인별 term/observation/reward는 같은 캐시를 공유한다. 물리·센서 조회는 늘리지 않았다.
- 기존 임계값/OR 종료/실패 보상−100 단일 지급은 그대로다. 여러 원인이 동시 발생하면
  모두 기록한다. Native 로그는 환경별 마지막 완료 episode의 원인 비율이며 원인별 비율을
  단순 합산해 전체 실패율로 해석하지 않는다. 선반은 접촉 자체가 아닌1cm/2° 변위 기준이다.
- 기존 test_mdp.py에7개 단독 원인·복합 원인·단일 payment·step당 계산1회·부분reset 및
  checkout의 실제 TerminationManager.compute/reset 로그 동작 검사를 추가했다.
  CPU 전체 검사201 passed,14 skipped,145 subtests passed였다.

Manipulability는 아직 reward에 추가하지 않았다. 검토 목적은 속도/힘 크기 제한이 아니라
제어성이 급락하는 자세 회피다. 후보는 단위를 정규화한 full6D Jacobian의
mu=sigma_min/sigma_max에 대해 안전 수준 이상은 포화하고 이하만 비용을 주는
`x_m=max(0,1-mu/mu_safe)^2`, `r_m=exp(-a_m*x_m)`이다. mu_safe와 정규화 규칙은
확정되지 않았고 실제 reset/학습 자세 분포를 보고 결정해야 한다. Raw determinant를
무한히 키우는 보상이나 linear/angular block을 따로만 검사하는 방식은 권장하지 않는다.
현재 q 관측으로 기구학 지표가 결정되므로 observation 추가는 필요하지 않다.

SVD/eigvalsh는 CUDA 연산이어도 host synchronization이 있어 기본 hotpath에 바로 넣지
않았다. Native OSC의 public jacobian_b도 새로운 PhysX 조회이며 내부 buffer를 재사용하면
마지막 physics substep 직전 값이라는 시간 차이를 고려해야 한다. GPU batch determinant
등 대안은 정확도/척도/처리량을 대조한 후 선택한다. Motion 결합식/weight 및 초기 IK,
actuator/OSC 설정은 이번 작업에서 변경하지 않았다. Reward/종료만으로 첫 substep의
특이점 토크 또는 CUDA fault를 예방할 수 있다고 주장하지 않는다.

참고: https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-4-manipulability/
및 https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html .

### 2026-10-16 — Manipulability 방향성과 Motion 결합식에 대한 사용자 정정

사용자는 sigma_min/sigma_max가 약한 방향으로의 움직임 자체를 구분하지 못한다고
지적했다. 이 비율은 자세의 등방성/조건수 지표이며, 같은 자세의 서로 다른 행동에
동일한 값을 준다. Jacobian의 선형/각 성분 단위 정규화와 이 축 비율은 별개다.
약한 방향의 이동 요구를 억제하려면 명령한 EEF 이동·회전 방향을 함께 평가하는
directional manipulability를 검토해야 한다. 실제 속도만 쓰면 명령했지만 움직이지
못하는 경우를 놓칠 수 있다. 명령의 단위 방향을 쓰면 크기 자체의 비용과 구분할 수
있으며 무명령은 방향 비용0으로 정의한다. 지표/정규화/안전 기준은 아직 미확정이다.

사용자가 명시한 Motion 결합식은 각 요소를 exp(-a*x)로 계산한 뒤 전체 Motion weight를
요소 수N으로 나누는 산술평균이다. 현재 구현은 N=2로 acceleration/F-T score의
0.5*(score_acc+score_FT)이며 이 설계와 일치한다. Manipulability를 추가한다면 N=3으로
W_motion/3*(exp(-a_A*x_A)+exp(-a_F*x_F)+exp(-a_M*x_M)) 형태를 유지한다.
앞서 제안한 exp(-(a_A*x_A+a_F*x_F+a_M*x_M)) 곱형 결합은 구현하지 않았으며 채택하지
않는다. 한 요소 점수가 낮아도 다른 요소 점수가 남는 것은 평균 설계의 trade-off이며
이를 구현 오류로 취급하지 않는다. 이번 정정에서 reward/제어 코드는 변경하지 않았다.

### 2026-10-16 — Joint torque observation은 명령값이 아닌 측정값

사용자는 joint torque를 observation에 추가하는 방안을 검토하면서, OSC의 토크 명령이
아닌 측정 joint torque를 사용해야 한다고 정정했다. 정책에 관절 부하 피드백을 주는
목적과 OSC 명령 폭주를 진단하는 목적을 구분한다. `joint_effort_target`을 측정값으로
사용하지 않으며, implicit actuator의 `applied_torque` 추정값도 측정값으로 취급하지 않는다.

Isaac Sim5.1의 공식 `get_measured_joint_efforts()`는 내부에서 PhysX Tensor API의
`get_dof_projected_joint_forces()`를 사용한다. 이는 solver가 계산한 incoming joint force를
DOF 운동 방향에 투영한 값이며, revolute joint에서는 부호 있는 torque(Nm)다. 실제 로봇의
모터 전류 기반 토크/축 토크 센서/외력 토크 추정값과 자동으로 동일하다고 가정하지 않는다.
구현 시 기존 articulation view에서 환경 전체를 batch로 읽고 observation/reward가 결과를
공유하는 방향을 검토한다. 현재 CUDA backend의 실제 지원·측정 의미·비용 검증은 남아 있다.

측정 토크 observation과 토크 reward는 아직 구현하지 않았다. 앞서 제안한 명령 토크의
상한 초과 reward를 측정 토크에 그대로 적용하는 것도 확정하지 않는다. 물리 부하 또는
급격한 토크 증가 억제와 명령 폭주 억제는 평가 대상이 다르며, reward의 x와 scale은
이를 구분하여 결정해야 한다. 기존 Motion의 요소별 exp(-a*x) 산술평균 원칙은 유지한다.

참고: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physics_articulation_force.html

### 2026-10-16 — Actuator 상한과 실현 가능한 OSC 명령 학습은 별개

사용자는 IdealPDActuatorCfg로 출력 토크를 제한해도 정책이 상한 안의 토크를 요구하는
delta pose를 학습한다고 보장할 수 없으며, 추종 부족에 대해 더 큰 명령을 선택할 수도
있다고 지적했다. 이 구분을 유지한다. 팔 gains0인 explicit actuator의 clipping은
OSC 계산 이후의 출력 제한이며, raw OSC 요구량이나 정책의 pose/stiffness를 제한하지 않는다.
같은 상한으로 포화되는 여러 명령은 측정 응답/진행 보상만으로 초과 정도를 구별하지
못할 수 있다. 더 큰 명령을 선택하는 것은 가능한 학습 결과이며 필연적인 증가는 아니다.

현재 native OSC에는 적분항이 없고, pose_rel 목표는 매 정책 step의 측정 EEF pose에서
다시 계산한다. 이를 PID integral windup이나 목표 pose의 무한 누적으로 설명하지 않는다.
측정 joint torque observation이라는 사용자 결정은 유지한다. 물리 부하/증가량 reward와
raw OSC 과대 요구 억제는 별개이며, 후자가 필요하면 raw 명령 초과량/포화 차이를 reward나
진단 지표로 검토할 수 있다. 이는 명령값을 observation에 넣자는 뜻이 아니다. 측정된
projected joint torque가 actuator 명령 상한과 반드시 같은 범위라고도 가정하지 않는다.
이번 논의에서 actuator/clipping/observation/reward 구현은 변경하지 않았다.

### 2026-10-16 — 팔 IdealPD 적용 및 측정 torque 연구

사용자는 먼저 팔을 IdealPDActuatorCfg로 변경하고, 측정 joint torque observation/reward는
관련 자료와 설계를 검토한 뒤 진행하도록 요청했다. 앞선 implicit 유지 결정을 대체한다.

**적용한 변경**

- `config/ur5e/scene.py`의 arm만 공식 IdealPDActuatorCfg로 전환했다. explicit actuator의
  `effort_limit`과 PhysX의 `effort_limit_sim`을 모두 shoulder/elbow150, wrist28Nm로 지정했다.
  `effort_limit_sim`만 유지하면 explicit 모델의 clipping 값은 USD에서 별도로 가져오므로,
  `effort_limit`도 반드시 명시한다. 별도 controller/clamp 구현은 추가하지 않았다.
- 팔 actuator gains0, 속도 제한3.14/6.28rad/s, USD armature/friction, native OSC의
  pose_rel·variable_kp와 scale, Hand implicit position PD를 유지했다. Action18D와
  observation244D 및 기존 reward는 그대로다. 측정 torque 채널은 아직 추가하지 않았다.
- 이 변경은 PhysX에 전달하는 유한한 토크를 제한한다. 특이점 inverse/NaN이나 CUDA719의
  원인을 해결했다고 주장하지 않으며, 정책의 과도한 pose/stiffness 요구 학습과 구분한다.

**검증**

- 기존 test_mdp.py의 회귀 검사에서 production config와 checkout의 실제 IdealPD/Implicit/
  Articulation 메서드를 실행한다.3환경·비연속 관절 인덱스에서 양/음 과대 토크·정상 토크,
  raw/computed 보존, 실제 set_dof_actuation_forces 입력 clipping, Hand 목표 전달을 검사한다.
- CPU 전체 검사202 passed,14 skipped,145 subtests passed.
- 실제 CUDA8환경에서 시험 명령 최대800880Nm를150Nm로 제한하고,64policy/128physics step의
  native OSC 출력 최대721328.625Nm가 관절별150/28Nm 이내로 전달됨을 확인했다.
  PhysX getter와 기대 clipping 값의 최대 차이0, 최대 상한 비율1, 최종 joint state finite.
  장시간 학습이나1024환경 안정성 검증을 대신하지 않는다.
- 같은 진단에서 공식 get_dof_projected_joint_forces()가 cuda:0의[8,18] finite tensor를
  반환했다. 워밍업 후100회 연속 조회의 평균 wall time은 약0.0148ms였다. 고정 상태8환경의
  getter 미세 측정이며1024환경이나 매 step 전체 overhead로 일반화하지 않는다.
- 재현 코드·결과: `outputs/contact_manipulation/ideal_pd_20261016/verification.py`,
  `verification.json`, `verification.log`. 추가 진단 코드는 학습 경로에 연결하지 않았다.

**측정 torque observation/reward 검토: 아직 미구현**

1. Isaac Sim5.1 measured effort는 incoming joint wrench의 DOF 운동축 투영이다. 기존
   articulation view의 GPU batch getter를 정책 sample마다 한 번 읽고 cache를 공유하는
   방향이 적절하다. 매 관절 sensor 객체·환경별 Python loop는 필요하지 않지만 SDK 비용은 있다.
2. 실기 채널을 먼저 맞춰야 한다. URScript get_joint_torques()는 로봇 자체 운동에 필요한
   중력·마찰 등을 보정한6D torque이다. ROS2 driver의 기본 effort는 actual_current_as_torque
   (전류 환산)이며, use_currents_as_efforts=True는 토크가 아닌 전류다. raw simulator
   projection과 이 값들이 동일하다고 가정하지 않는다. 보정·부호·단위를 확인해야 한다.
3. Closing the Reality Gap(2026)의 force-adaptive grasping은 actor에12D joint torque와
   tactile을 제공하고(TableI), 접촉 중 목표 torque 오차를 Gaussian exponential reward로
   평가한다(Eq3). torque feedback의 실제 사용 사례이지만 목표 부하 추종 과제이므로 우리
   비파지 재배치의 수식/scale을 그대로 옮기는 근거로 삼지 않는다.
4. 우리 목적은 큰 유효 부하 자체를 무조건 억제하는 것이 아니다. 측정 torque 증가량이
   단순 torque L2보다 목적에 가깝다. positive magnitude growth는 unloading을 허용하지만
   +torque→-torque 반전을 놓치고, signed difference 크기는 반전을 잡지만 unloading도
   비용으로 본다. 어떤 물리 행동을 억제할지와 signal filtering/scale은 아직 미정이다.
5. clipping 이후의 측정 부하만으로 raw OSC 과대 요구를 구분할 수는 없다. actuator의
   computed/applied torque 차이는 별도의 포화 진단 후보이며 측정값으로 부르지 않는다.
   측정 torque가 반드시150/28Nm 이내라고 가정하지도 않는다. observation은 측정값이라는
   사용자 결정을 유지한다. Reward가 필요하면 각 exp(-a*x)를 구하고 요소 수로 평균한다.

자료:
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physics_articulation_force.html
- https://www.universal-robots.com/manuals/EN/HTML/SW5_23/Content/prod-scriptmanual/all_scripts/get_joint_torques.htm
- https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_robot_driver/ur_robot_driver/doc/hardware_interface_parameters.html
- https://arxiv.org/html/2607.04940v1

### 2026-10-16 — 측정 torque reward 설계 전 최신 학습 로그 검토

사용자는 새 torque reward를 고민하기 전에 가장 최근 학습 로그에서 기존 reward의
이슈를 먼저 검토하도록 요청했다. Reward/MDP/학습 설정은 변경하지 않았다.

**분석 기준**

- 최신 run은 `logs/rsl_rl/contact_manipulation/2026-10-15_17-57-45`이며1024환경,
  50Hz policy,8초 episode,32step rollout,19100iteration(index0–19099)이다.
  마지막 scalar 기록은2026-10-16 03:14:14 KST다.
- 당시 commit7be67ac와 저장된 dirty diff를 확인했다. 당시 rewards.py/events.py의
  postimage는 현재 코드와 일치한다. 당시 arm은Implicit이며 오늘 적용한IdealPD와
  원인별failure logging은 이 로그에 반영되지 않았다.

**실제 로그에서 확인한 추세: 처음1000 vs 마지막1000 iteration 평균**

| 지표 | 처음1000 | 마지막1000 |
|---|---:|---:|
| Train/mean_reward | 80.913 | 193.561 |
| Train/mean_episode_length(step) | 90.291 | 144.515 |
| 위 길이를 초로 환산 | 1.806 | 2.890 |
| 마지막 종료 상태의timeout 지표 | 28.15% | 75.29% |
| 마지막 종료 상태의failure 지표 | 71.91% | 24.81% |
| Episode_Reward/clearance | 5.572 | 10.102 |
| Episode_Reward/rearrangement | 11.687 | 20.708 |
| Episode_Reward/motion | 5.616 | 9.865 |
| Episode_Reward/time | -0.0336 | -0.1149 |
| Episode_Reward/terminal | -12.427 | -12.444 |

Success가 양수인iteration 기록은 전체에서10개뿐이며 마지막은8099다. 이후11000iteration은
success 기록이모두0이다. 이는 성공10회라는 뜻이 아니다. Mean reward와episode length의
상관은0.9877이다. 안전하게 더 오래 유지하는 방향의 개선과 양립하지만 작업 달성의
개선으로 해석할 수 없다. Clearance/Rearrangement 누적 로그의 합 비율은0.4768→0.4878,
Motion/Rearrangement는0.4805→0.4764로, 단순한 누적 항 증가와 구분해서 보아야 한다.
이 비율을 정확한 per-step score 또는 가림 비율로 역산하지 않는다.

**로그 집계의 한계**

- Native Episode_Reward는 reset batch의episode 누적합 평균을 고정 최대길이8초로 나눈다.
  실제episode 길이 또는step수로 나눈quality 평균이 아니다. 길게 유지하면 점수가 커진다.
- RSL logger는 각step의 extras log dict를 동일 가중 평균한다. batch크기로 가중하지 않고,
  reset 없는step에는 마지막dict가 재사용될 수도 있다.
- Episode_Termination은 전체환경 각각의가장최근종료 상태 평균이다. 완료episode 전체의
  성공률/실패율이 아니다. timeout과failure가 동시에 참일수도 있다.
- Train reward/length는 최근100개완료episode의deque 평균이며,1024개공유reset에서는
  환경인덱스 순서 영향도 있다. 공유장면은400common step마다교체되어 부분reset episode도
  경계에서잘린다. 다른모집단의지표들을 조합해정확한 성공률/평균step quality를 만들지 않는다.

**수식상 우선 점검 요소**

1. Time 누적비용은1/2/4/6/8초에0.00896/0.04729/0.36128/1.75675/7.53790이다.
   8초직전한step비용은0.1070으로, valid Clearance 최저점e^-1=0.3679보다작다.
   완전히가려지고정지해도 Rearrangement=Motion=1이면 매step합2.3679를얻는다.
   양의quality를유지하는보상이시간효율보다강하며, 높은return을작업진전으로 볼수없다.
2. gamma=.99는50Hz에서반감기약1.38초다. 성공500의시작시점할인가치는
   2/4/8초후184.86/67.67/9.07이다(terminal은N번째step,지수N-1).
   8초Time전체할인비용은0.3530이다. Terminal숫자만비교해서dominant하다고판정하지않는다.
   단, 최대densecontinuationvalue300보다즉시성공500이크므로이사실만으로성공지연이
   최적이라고단정하지도않는다. 목표까지의행동quality와도달시간을함께평가해야한다.
3. Rearrangement=exp(-2*이번step의총corner이동량)는1/5/10mm에서
   0.9980/0.9900/0.9802다. 주파수50Hz의작은step변위에대해거의1일수있으므로
   실제x분포와a를먼저검토해야한다. baseline누적점수상승은재배치가줄었다는증거가아니다.
4. Motion은각exp점수의평균으로사용자설계와일치한다. 문제후보는x의scale/포화다.
   팔6관절가속도50→100rad/s²,hand정지·F/T증가0인예시에서합성점수는
   0.500425→0.500000으로거의구별되지않는다. 실제run은두구성점수/raw x를저장하지
   않았으므로이포화가실제분포에서발생했다고확정하지않는다. 마지막physics substep만
   보는가속도측정도앞substep급변을놓칠수있다.
5. Clearance가높아도success는별도다. Blocker면적0.01m²,가림2e-5m²면
   score≈0.998이지만성공가림상한1e-5m²는넘는다. Targetpose보호·전체물체정착5step도
   별도gate다. 가림제거/Target보호/정착중어디서실패했는지현재로그로는구별할수없다.

다음검토는새torque항추가보다기존x분포,Motion구성점수,성공gate의진행상태와
Time/할인율의상대척도를먼저확인하는방향이다. 기존exp형태와요소별평균원칙은유지한다.
새reward나weight를확정하거나적용하지않았다. 물체접근항이없다는사실만으로항을추가하지않는다.

재현자료: `outputs/contact_manipulation/reward_log_review_20261016/`의
`analyze.py`, `summary.json`, `scalars.csv`, `reward_trends.png`.

### 2026-10-16 — 사용자의 최신 checkpoint play 관찰과 비조작 보상 검토

사용자는 약19100 iteration checkpoint를 play했을 때 선반 앞에서 이상한 팔 동작만
반복하고 timeout까지 물체를 조작하지 않았다고 보고했다. 이는 영상/새 replay를 확인한
결과가 아닌 사용자 관찰이다. 보상이 늘어도 성공 기록이 없는 최신 로그와 일치한다.
사용자는 후속 메시지에서 이 play가 IdealPD 변경 전이었다고 확인했다. 따라서 당시
학습과 같은 implicit 구성에서 비조작 행동을 관찰한 것이다. 기본 play.py는
deterministic inference다. 이 관찰을 IdealPD 변경의 결과로 해석하지 않는다.

현재 수식에서 물체가 움직이지 않으면 Rearrangement=1이다. F/T 증가가 없으면
wrench quality=1이므로, 가속도 score가0이어도 Motion은 평균 규칙에 따라0.5를 준다.
완전 가림(Clearance=e^-1), Rearrangement=1, Motion=0.5가400 step 유지된 예시의
할인 전 누적 return은400*(e^-1+1+0.5)-7.537897=739.614다. 완전히 안정된
Motion=1이면939.614다. 성공하거나 물체를 조작하지 않아도 이 양의 누적값을 얻을 수 있다.
실제 play의 측정 return이라고 주장하거나 이를 terminal500과 직접 비교하지 않는다.

문제 후보는 Rearrangement/Motion만이 아니다. 변하지 않는 Clearance도 매 step 양의
점수를 준다. 같은 경과 시간에서 Time은 조작/비조작 행동에 똑같이 적용되므로 scale을
키우는 것만으로 유효한 접촉의 방향을 알려주지 않는다. 물체에 접근하는 새 항이나 torque
항을 바로 추가하기보다 작업 진전과 품질 항 사이의 상대 유인을 먼저 검토한다.

기존 합의상 exp(-a*x)와 요소별 산술평균을 유지하며 exp(-a*x)-1 이동은 채택하지 않았다.
계수 a를 키워도 비조작의 x=0에서 점수1은 그대로이며, Rearrangement의 a만 키우면
조작 중의 점수가 더 내려가 비조작 선호를 강화할 수 있다. 현 형태를 유지하는 조정은
Clearance 대비 품질 항의 weight, Time scale, 할인율의 상대 효과부터 비교하는 방향이다.
Progress/potential 보상이나 baseline 변경은 별도의 설계 변경으로 명시하여 검토해야 한다.
양의 baseline 자체가 원인 전부이거나 비조작이 항상 전역 최적이라고 단정하지 않는다.
이번 검토에서도 reward 수식/weight/학습 설정은 변경하지 않았다.

후속 검토에서는 Clearance의 상대 weight를 높이는 초기 후보로 C:R:M=3:1:1을 검토했다.
기존 최대 dense 합3을 유지하도록 정규화하면1.8/0.6/0.6이다. 이는 제안이며 적용/확정하지
않았다. 단순히 C만3으로 올리는 설정(3/1/1)은 최대 dense 합을5로 키워 survival baseline과
Time/Terminal의 상대 scale도 바꾸므로 구분한다. 가림비율1→0.5의 개선은1.8가중 시
약0.430점이며, 같은 step의 Motion1→0.5와2mm 조작량 손실은 약0.302점이다.
이 순간 비교는 후보의 설명이며 작은 진행량, 접근 지연, 장기 return의 검증을 대신하지 않는다.

측정 torque observation은 관절 부하 피드백으로 검토를 유지한다. Torque quality를
Motion 평균의 세 번째 항으로 추가하면 각 요소 weight는W_M/3이다. 예를 들어 가속도
점수0·F/T점수1인 상태의 Motion0.5에 torque점수1을 추가하면2/3이 된다. 평균 자체가
오류는 아니지만 기존 항의 영향이 줄어드므로 단순 추가가 비조작 유인을 해결하지 않는다.
Torque observation/항 추가와 weight 변경은 아직 구현하지 않았다.

### 2026-10-16 — Torque 보류, F/T·가속도 점수의 의미와 허용 구간 검토

사용자는 joint torque 추가를 추후로 미루고, 현재 F/T와 가속도가 작은 변화에도 점수를
낮추어 정지를 선호하게 만드는지, 현재 수식·문제·해결 방향을 검토하도록 요청했다.
이번 작업은 분석이며 production reward/weight/관측은 변경하지 않았다.

현재 a_acc=a_FT=1, dt_policy=0.02초다.

```
x_acc = sqrt((sum_arm((qddot/5)^2) + sum_hand((qddot/10)^2)) / 12)
g_F = max(0, norm(F_k)-norm(F_previous)) / 0.02
g_M = max(0, norm(M_k)-norm(M_previous)) / 0.02
x_FT = sqrt(((g_F/200)^2 + (g_M/10)^2) / 2)
r_motion = (exp(-x_acc) + exp(-x_FT)) / 2
```

- 가속도는 증가량/jerk가 아닌 크기다. 선택된 관절에 nonzero 가속도가 있으면 점수가
  낮아진다. 감속도 포함한다. 일정 joint velocity는 가속도0이므로 이상적으로 최고점이다.
- F/T는 크기의 양의 증가만 평가한다. 일정 크기 하중·하중 감소·크기 유지 방향 변화는
  최고점이다. 이상적인 정지는 Motion=1이지만 정지만이 최고점인 것은 아니다.
- 현재5/10rad/s²와200N/s·10Nm/s는 정규화 scale이며 허용 구간이 아니다. 어떠한 작은
  양의 입력도 exp에서 점수를 낮춘다. 큰 x에서는 서로 다른 심각한 동작들의 점수가
  모두0에 가까워져 구별력이 약해진다. PPO가 reward를 직접 미분하지 않으므로 이를
  reward gradient의 backprop vanishing으로 설명하지 않는다.
- 예: 팔 한 관절만1rad/s²이면 acceleration score0.94390, 팔6개 모두5이면0.49307.
  F norm이20ms에0.1/1/4N 증가하면 F/T score0.98248/0.83797/0.49307이다.
  센서 토크 norm0.005Nm 증가도0.98248이다. 이것은 구성 점수이며 합성 Motion은 평균이다.

**새로 확인한 F/T 평가 원점 문제**

현재 events.advance는 자중·관성 보상된 센서 모멘트를 episode 초기 sensor 위치로 옮긴다.
`M_fixed=R*M_sensor+(p-p0)×(R*F_sensor)`는 센서 중심 부하와 다른 물리량이다.
Norm을 비교하므로 축 회전만으로 생기는 차이는 없으며, 고정 원점으로 옮길 필요가 없다.
센서 중심의 부하 증가 억제가 목적이면 보상된 sensor-frame 힘/토크 norm을 직접 비교하는
방향이 적합하다. 기존 구현·테스트는 고정 원점 합력 모멘트 평가를 의도적으로 보장하므로
단순 수치 버그가 아닌 보상 의미의 수정으로 취급해야 한다.

현재 순수 reward_components 함수로 계산한 구성 예시: F=(0,50,0)N, 센서 토크0이 계속
일정하고20ms 동안X로2mm 이동하면, 초기 원점 모멘트는0→0.1Nm가 된다. 현재 F/T score는
0.70219, sensor 중심 norm을 직접 비교하면1이다. 실제 rollout 관측이라고 주장하지 않는다.
반대로 센서 토크 증가가 원점 이동 항과 상쇄되어 비용이 줄어들 수도 있다.

**권장 검토 방향: exp와 요소별 평균은 유지**

1. 먼저 F/T를 센서 중심의 보상된 load norm으로 평가하도록 의미를 맞춘다.
2. x에 원래 크기 전체가 아닌 허용 범위 초과량을 넣는다.
   `u_acc_j=max(0, abs(qddot_j)-acc_allow_j)/acc_scale_j`,
   `u_F=max(0,g_F-rate_allow_F)/rate_scale_F`, M도 같은 방식이다.
   x_acc=RMS(u_acc), x_FT=RMS(u_F,u_M), r_motion은 기존 두 exp의 평균을 유지한다.
   정상 조작과 정지가 동일 최고점을 얻고 과도한 가속/하중 증가만 점수를 낮추게 한다.
   허용 구간과 정규화 scale은 서로 다른 파라미터이며 기존 scale을 한계로 재해석하지 않는다.
3. 임계값은 무접촉 정지·이동의 보상 잔차와 정상 조작의 필요한 증가율을 보고 정한다.
   현재 원시 축별0.1N/0.005Nm 양자화 후 연속 self-load를 빼므로 작은 잔차가 생긴다.
   양의 증가만 평가하면 왕복 오차도 일부 비용이 된다. 양자화 한 tick만으로 큰 Motion
   저하가 설명되지는 않는다. 실제 tare는 없고 wrench_bias는 현재0이다.
4. 가속도/F-T 개별 점수·원시 x·포화 비율을 분리해 scale을 재조정한다. 단순히 제곱하면
   작은 값에는 완만해져도 큰 값에서 포화가 빨라질 수 있으므로 자동 해결로 보지 않는다.
5. native 가속도는 각10ms physics step에서 갱신되지만 reward는 마지막 것만 읽는다.
   두 substep RMS 집계를 검토한다. F/T substep peak도 필요하다면 추가 반력 SDK 조회
   비용을 측정해야 한다. 단순 policy 끝점 차분이나 강한 필터는 짧은 충격을 숨길 수 있다.
6. 허용 구간을 두어도 양의 품질 baseline은 유지된다. 작업 진전은 Clearance가 구분하도록
   상대 weight를 함께 검토한다. 움직임 자체에 보상을 추가해 허공 동작을 유도하지 않는다.

구성 예시·곡선 재현: `outputs/contact_manipulation/motion_score_review_20261016/`의
`analyze.py`, `equations.json`, `motion_quality.png`. 우측 곡선의 허용값1은 입력을 허용값으로
나눈 일반적인 형태의 예시이며 실제 threshold를 결정/적용한 것이 아니다.

### 2026-10-16 — Motion 재설계를 위한 예제·논문 검토

사용자는 Motion을 다시 설계하기 전에 기존 예제와 관련 논문을 확인하도록 요청했다.
이번에는 자료 조사와 비교만 수행했다. Reward 수식·weight·PPO·observation은 변경하지
않았고, joint torque 추가도 계속 보류한다. 목표는 지속적인 빠른 속도나 필요한 힘을
일괄적으로 억제하는 것이 아니라 과도한 가속과 급격한 하중 증가를 줄이는 것이다.

**예제 구현에서 확인한 것**

- 로컬 `examples/Sweeping-Policy-DRL-Sim-Train`와 현재 원격 `min`의 기본 reward는
  `action_rate_l2=sum((a_t-a_previous)^2)`, arm `joint_vel_l2=sum(qdot^2)`에 각각
  weight -0.03을 사용한다. OSC 변형도 이를 상속한다. `/dt`가 없는 입력 차이이며
  실제 joint acceleration이나 F/T 증가율을 평가하는 항이 아니다. Native RewardManager의
  dt 곱이 적용되므로 우리 품질 항의 weight와 직접 비교하지 않는다.
  [원격 min RewardsCfg](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/min/sweeping_policy/shelf_sweep_random_env_cfg.py#L164)
- 공식 Isaac Lab v2.3.0(commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`)의
  Reach/Franka OSC는 action-rate weight를 -0.0001에서 -0.005로, joint-velocity는
  -0.0001에서 -0.001로 강화하는 curriculum을 상속한다. 공통 policy step counter가
  4500을 넘은 뒤 첫 reset에서 변경되며, PPO iteration이나 환경 수를 곱한 transition
  수가 아니다. 성공률을 확인하는 schedule은 아니다. 초반 규제를 약하게 두는 구조는
  참고할 수 있지만 속도 비용 자체는 우리 목표와 다르다.
  [공식 reward·curriculum](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py#L155)
- 공식 FORGE는 `relu(norm(F_smooth)-force_threshold)` 비용을 사용한다. 기본 threshold는
  episode마다 5–10N에서 선택하며 정책에도 제공한다. Factory의 action difference norm
  비용도 상속하고 weight 0.1로 활성화한다. 이는 힘 **크기** 초과 비용이며 증가율 비용이
  아니다. 제어기 내부 dead-zone randomization과 reward threshold도 구분해야 한다.
  [공식 FORGE reward](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py#L232),
  [설정](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_tasks_cfg.py#L11)

**관련 1차 논문과 적용 한계**

1. [Learning Gentle Object Manipulation with Curiosity-Driven Deep Reinforcement
   Learning (2019)](https://arxiv.org/html/1903.08542), §III-C/IV-A:
   힘의 양의 증가 `m=max(0,f_next-f)`와 비선형 비용
   `m*sigmoid(lambda1*(m-lambda2))`를 사용한다. 작은 증가를 약하게 평가해도 curiosity가
   없으면 접촉을 피하는 local optimum을 보고했다. 실험은 fingertip force channel별
   증가를 합하며 `/dt`가 없다. 우리의 보상된 wrist 합력 norm·증가율·토크 확장과 다르다.
2. [FORGE (2024/2025)](https://arxiv.org/abs/2408.04587), §III-A:
   필요한 접촉 하중을 허용하면서 설정된 힘 threshold의 초과량에 비용을 부여한다.
   Force-threshold conditioning까지 포함한 방법이다. 우리에게 옮기는 교훈은 필요한
   접촉과 과도한 접촉의 구분이며, 5–10N이나 힘 크기 비용을 그대로 복사하는 것은 아니다.
3. [CAPS, Regularizing Action Policies for Smooth Control with Reinforcement
   Learning (ICRA 2021)](https://arxiv.org/html/2012.06644v2), 식1–3:
   시간상 인접한 상태의 정책 출력 차이와 유사한 상태의 출력 차이를 정책 학습 목적에
   regularization으로 넣는다. 환경 reward가 아니며 실제 가속도·F/T smoothness를
   직접 보장하지 않는다. 도입하면 PPO update와 추가 forward 계산을 검토해야 한다.
4. [Grad-CAPS (2024)](https://arxiv.org/html/2407.04315v1), 식11–18:
   `norm((a_t-a_previous)-(a_next-a_t))`와 displacement normalization을 사용해 일정한
   명령 변화와 zigzag를 구별한다. 과도한 smoothing이 응답성을 떨어뜨리는 문제를
   다룬다. 단순한 action 2차 차분이 전체 방법은 아니며 physical joint jerk와도 다르다.
5. [Variable Impedance Control in End-Effector Space: An Action Space for
   Reinforcement Learning in Contact-Rich Tasks (IROS 2019)](https://arxiv.org/pdf/1906.08880),
   §V-B/C: 문 열기·닦기의 작업 진행과 필요한 접촉을 허용하고 과도한 하중에 비용을 준다.
   정책20Hz·제어500Hz 사이에 명령 보간도 사용하므로 reward만의 결과로 해석하지 않는다.

**우리 Motion에 대한 연구 기반 검토안 — 미확정·미구현**

- 기존 `r_motion=(exp(-a_acc*x_acc)+exp(-a_FT*x_FT))/2`와 요소별 평균 원칙은 유지한다.
  논문의 음의 비용을 그대로 추가하거나 `exp(-x)-1`로 이동하는 변경은 하지 않았다.
- F/T의 목적에는 양의 부하 증가가 직접 맞는다. 다만 episode 초기 원점으로 이동한
  moment 대신 sensor 중심의 자중·관성 보상된 F/T norm을 비교하는 의미 수정이 먼저다.
  Wrist 합력의 norm은 서로 상쇄되는 국소 접촉 증가를 모두 표현하지 못한다.
- 앞서 검토한 정확한 허용 구간 `relu(u-b)` 외에, Gentle Manipulation의 비용 형태를
  참고한 `phi(u)=u*sigmoid(k*(u-b))`를 **exp 안의 x**에 넣는 비교 후보가 생겼다.
  `u`는 단위 없는 양의 증가율, `b`는 전이 위치, `k`는 전이의 가파름이다. 작은 증가의
  비용을 낮추지만 정확한 zero-cost band는 아니다. 큰 값에서는 선형에 접근하고 exp의
  포화는 여전히 남는다. 이 변환과 토크 적용은 우리의 제안이며 원 논문 수식 그대로가 아니다.
  Gate 효과만 비교할 때는 F/T 내부의 기존 RMS 집계도 유지해 별도 변경과 섞지 않는다.
- 가속도는 현재 RMS 크기, 정규화한 제곱 평균, 허용 범위 초과량을 비교할 수 있다.
  제곱 평균은 작은 값에서 완만하지만 큰 값에서는 exp가 더 빨리 포화한다. Jerk만으로
  교체하면 큰 일정 가속을 충분히 구별하지 못하므로 자동 대체하지 않는다.
- Raw18D action 차이에는 relative EEF pose·clipped stiffness·absolute hand target의
  의미가 섞인다. 같은 raw 변화가 실제 제어에 같은 영향을 주지 않으므로 예제의
  action_rate를 그대로 Motion으로 바꾸지 않는다.
- 허용 구간이나 완만한 비용만으로 비조작 문제 해결을 보장할 수 없다. Clearance의
  실제 진전과 Motion 손실의 상대 크기, 접촉률·성공률, 초반 낮은 Motion weight 후
  강화하는 curriculum을 함께 비교할 후보로 둔다. Curriculum은 native 기능을 우선한다.
  새 curiosity predictor·접촉 보너스·temporal encoder를 자동으로 추가하지 않는다.
- 먼저 가속도/F-T 구성 점수와 원시 물리량, 정상 접촉의 범위를 분리해 측정해야 한다.
  임계값은 논문 수치나 현재 정규화 scale을 허용 한계로 재해석하여 결정하지 않는다.
  Tensor 기반 차분·norm·elementwise shaping은 기존 GPU 경로에 얹을 수 있으나 추가
  physics-substep F/T SDK 조회나 정책 regularization은 별도 비용 측정이 필요하다.

### 2026-10-16 — 구체적인 첫 reward 수정 실험안 제안

사용자는 문헌 검토 이후 구체적인 수정 방향을 물었다. 아래는 첫 실험을 위한 제안이며
구현하거나 확정하지 않았다. 논문 수식을 그대로 복사한 설계라고 주장하지 않는다.

1. Sensor 중심의 자중·관성 보상된 F/T norm을 비교하도록 원점을 정리한다.
   초기 sensor 위치로 moment를 옮기는 `(p-p0)×F`를 reward 경로에서 제거하는 제안이다.
2. 현재 Motion의 지수 안 RMS에서 sqrt를 제거해 정규화한 제곱 평균으로 바꾼다.

   ```
   x_acc = (sum_arm((qddot/5)^2) + sum_hand((qddot/10)^2)) / 12
   g_F = max(0, norm(F_sensor_k)-norm(F_sensor_previous)) / dt
   g_M = max(0, norm(M_sensor_k)-norm(M_sensor_previous)) / dt
   x_FT = ((g_F/200)^2 + (g_M/10)^2) / 2
   r_motion = (exp(-x_acc) + exp(-x_FT)) / 2
   ```

   독립 관절12개, a_acc=a_FT=1, 기존 scale5/10rad/s²·200N/s·10Nm/s를 유지한다.
   기존 normalized RMS를 z라 하면 exp(-z)→exp(-z²)다. z<1은 감점이 약해지고
   z>1은 강해진다. 작은 nonzero에도 감점은 남으므로 정확한 허용 구간이 생기는 것은 아니다.
   정상 분포를 측정하기 전 새 deadband·sigmoid 파라미터를 추가하지 않는 최소 초기안이다.
3. Dense 최대합3을 유지하며 C/R/M weight를 1.8/0.6/0.6으로 조정하는 제안이다.
   Motion 내부 가속도와 F/T의 실제 가중치는 각각0.3이다. Clearance/Rearrangement의
   기존 수식, Time의 증가형 적분 penalty, Terminal+500/-100은 이 첫 실험에서 유지한다.
   Curriculum은 동시에 도입하지 않고 고정 weight의 효과부터 비교한다.

계산 예시는 실제 rollout이 아닌 동일 물리량에 대한 수식 비교다.

| 조건 | 현재 구성 점수 | 제안 구성 점수 |
| --- | ---: | ---: |
| 팔6관절 각각1rad/s², hand0 | 0.868123 | 0.980199 |
| 팔6관절 각각10rad/s², hand0 | 0.243117 | 0.135335 |
| 힘 norm +0.1N/20ms, moment 일정 | 0.982478 | 0.999688 |
| 힘 norm +1N/20ms, moment 일정 | 0.837967 | 0.969233 |
| 힘 norm +8N/20ms, moment 일정 | 0.243117 | 0.135335 |

제곱 평균은 큰 입력의 점수 포화를 더 빠르게 만든다. 기존에 acceleration score가
이미0에 붙은 표본의 구별력 문제를 해결하지 않는다. 따라서 r_acc/r_FT를 분리하고
raw normalized cost 분포·score<0.01 비율·실제 Clearance 진전·성공률을 함께 확인해야 한다.
큰 입력 대부분이 포화된다면 scale을 다시 정해야 하며 곡선 변경만으로 해결했다고 보지 않는다.

정지 최대점은 여전히1이고 완전 가림 정지 dense 합도2.367879→1.862183으로 줄 뿐 양수다.
비조작 방지를 보장하는 제안은 아니다. 새 IdealPD와 이전 Implicit의 차이를 reward 효과로
섞지 않도록 비교 실험에서는 같은 controller·scene 설정을 사용한다. Torque 항 추가는 보류다.

### 2026-10-16 — 사용자의 기준값 차감 방식 제안

사용자는 기준값으로 나누는 대신 빼는 방식을 제안하며 의견을 물었다. 허용 범위 이하는
비용0, 초과량만 평가하는 해석이 정상 조작을 허용하려는 목적에 맞는다. 아직 파라미터나
구현을 확정한 것은 아니며 production reward는 변경하지 않았다.

```
e_acc_j = max(0, abs(qddot_j) - acceleration_allow_j)
e_F = max(0, positive_force_norm_rate - force_rate_allow)
e_M = max(0, positive_moment_norm_rate - moment_rate_allow)
quality = exp(-a * excess^2)
```

단순한 차감만 사용하면 음의 cost로 exp 점수가1을 넘으므로 반드시 양의 초과량을 쓴다.
가속도는 절댓값을 사용해 가속과 감속을 모두 평가한다. F/T는 기존 양의 norm 증가율을
사용하여 일정 부하와 하중 해제는 감점하지 않는다. 정확한 허용 구간이 생기므로 앞서
제안한 threshold 없는 MSE와 구분해야 한다.

허용 기준b와 감점 강도a는 역할이 다르다. 분모를 없애더라도 a가 단위와 민감도를 조절한다.
Force와 moment의 초과량을 raw 값으로 더하지 않고 각각의 단위에 맞는 계수를 사용해야
한다. 기존 Motion의 가속도/F-T 두 구성 점수와 산술평균을 자동으로 세 요소로 바꾸지 않는다.
예를 들어 기존 grouping을 유지하면 r_acc=exp(-a_acc*mean(e_acc²)),
r_FT=exp(-(a_F*e_F²+a_M*e_M²)/2), r_motion=(r_acc+r_FT)/2로 표현할 수 있다.

허용 가속도5rad/s²라는 **예시 가정**에서는 가속도3과5의 초과량이0,7의 초과량은2다.
기존 scale5/10rad/s²·200N/s·10Nm/s는 측정된 허용 한계가 아니므로 새 기준으로 그대로
재해석하지 않는다. 기준은 정상 조작과 무접촉 센서 보상 잔차의 분포를 보고 정해야 한다.
허용 구간을 두어도 비조작 최대점과 task reward의 상대 유인을 따로 검토해야 한다.

### 2026-10-16 — Motion 초과량 수식과 실제 무접촉 측정 기반 초기 기준

사용자는 초과량 방식의 식을 다시 세우고 기준을 합리적으로 정하도록 요청했다.
정상 움직임의 참고값을 얻기 위해 현재 native OSC+IdealPD/hand implicit PD 구성으로
8개 초기 자세, Kp50/100/200에서 무접촉 정지·작은 팔 주기동작·작은 hand 주기동작을
측정했다. 학습/reward production 코드는 바꾸지 않았다. 아래는 초기 실험용 제안이다.

**측정 조건과 한계**

- Physics10ms, policy20ms 유지. 로봇 base를 world Z로3m 옮겨 선반·물체 접촉을 배제하고
  native ActionManager의 process/apply와 Scene write/step/update를 그대로 호출했다.
  Robot reset은 IK가 검증한8개 자세를 사용한다. 정책 추론이나 학습은 실행하지 않았다.
  현재 tanh actor와 stiffness_scale100 조합에서 정책이 도달하는 Kp범위는50–100이다.
  Kp200은 controller cfg가 허용하는 상단에 대한 추가 진단이며 정책의 통상 출력은 아니다.
- Kp마다200policy step 안정화 후 각 모드200step을 수행하고 첫25step을 제외했다.
  모드당1400 env-transition, 총12600개를 기록했다. 가속도는 두 physics substep 모두
  저장했으며 아래 기준 계산은 현재 reward와 같은 마지막 substep 표본을 사용했다.
- 팔 목표는0.5Hz로 Y진폭2cm·Z범위0–2cm의 작은 동작, hand는 각 독립관절0–0.2rad의
 0.5Hz 목표였다. 모든 tactile force는0, 모든 joint acceleration/wrench는 finite였다.
- Gravity compensation=False인 기존 OSC를 유지했으므로 고정 목표에 대한 TCP 위치
  오차가 Kp에 따라 약4.3–16.5cm 남았다. 이 진단은 실제 성공 조작이나 정확한 위치 추종의
  검증이 아니며, 안정된 무접촉 상태와 작은 변화의 참고 분포다. 이를 정상적인 작업 전체의
  upper bound로 주장하지 않는다. 첫 안정화 충격을 허용해야 한다는 결론도 아니다.
- 기존 `reward_update_20261015/motion_samples.pt`는 Implicit 초기 탐색 자료다.
  팔 RMS 가속도 median61rad/s² 같은 값을 허용 기준으로 채택하지 않는다. 당시 기존
  가속도 score의 약83%가0.01미만이었으므로 threshold 없는 MSE만 적용하는 이전 제안은
  큰 입력 포화를 심화할 수 있다. 이번 측정 기반의 초과량 설계가 최신 검토안이다.

조건별 smooth-motion99.9백분위 중 최댓값은 다음과 같다.

| 물리량 | 측정값 | 허용 기준 b 제안 |
| --- | ---: | ---: |
| 팔 관절별 abs 가속도 | 2.5822rad/s² | 3rad/s² |
| Hand 관절별 abs 가속도 | 6.5142rad/s² | 8rad/s² |
| 보상된 힘 norm 양의 증가율 | 1.4401N/s | 10N/s |
| 보상된 센서 moment norm 양의 증가율 | 0.1043Nm/s | 0.5Nm/s |

팔 관절별 최대는2.6752rad/s²였다. Hand에는 최대20.438rad/s²의 드문 표본이 있어
99.9백분위 기준과 분리한다. 그 spike까지 모두 무비용으로 허용하려고 b를 올리지는 않는다.
F/T 기준은 관측된 잔차뿐 아니라 축별해상도0.1N/.005Nm의 반올림 오차를 고려한다.
실제 self-load 모델이 정확하고 overrange가 없을 때 두 norm 샘플 간 반올림 영향의 보수적
rate 상한은 sqrt(3)*resolution/dt, 즉8.6603N/s·0.4330Nm/s다. 따라서10/.5를 채택한다.
이는 보상 모델 오차 전체에 대한 보장은 아니며 센서 accuracy를 noise 표준편차로 간주하지 않는다.
[ATI 해상도 자료](https://www.ati-ia.com/es-MX/Library/documents/Axia80_Flyer.pdf)

**새 식 — 기존 가속도/F-T 두 구성 점수와 평균 유지**

```
E_arm  = mean_6(relu(abs(qddot_arm)  - 3)^2)
E_hand = mean_6(relu(abs(qddot_hand) - 8)^2)
g_F = (norm(F_sensor_k) - norm(F_sensor_previous)) / dt
g_M = (norm(M_sensor_k) - norm(M_sensor_previous)) / dt
e_F = relu(g_F - 10)
e_M = relu(g_M - 0.5)
r_acc = exp(-(0.00481352208722 * E_arm + 0.00143212227388 * E_hand))
r_FT = exp(-(0.00008557372600 * e_F^2 + 0.03422949039802 * e_M^2))
r_motion = (r_acc + r_FT) / 2
```

F_sensor/M_sensor는 센서 중심에서 자중·관성 보상된 값이며 초기 위치로 moment를 옮기는
`r×F`를 사용하지 않는 제안이다. 음의 rate는 최종relu에서0이 되므로 일정 부하·하중 해제는
무비용이다. 나눗셈 mean_6·/2는 개수 평균이며 기준값으로의 입력 정규화가 아니다.
서로 다른 단위의 force/moment 비용은 각각 단위를 상쇄하는 계수를 사용한다.
팔/hand·force/moment를 각자 exp로 만든 뒤4항 평균하는 새 grouping은 도입하지 않는다.
그렇게 하면 나쁜 팔 움직임을 정지 hand 점수가 더 쉽게 보상하는 변경까지 섞이기 때문이다.

**감점 계수의 해석**

허용 기준 b와 별개로 score가0.5가 되는 지점을 명시해 a=ln(2)/(L-b)^2로 정한다.
이 L들은 측정된 안전 한계가 아니라 초과 후 감점 강도를 정하기 위한 초기 설계 선택이다.

| 항 | b | L: 해당 구성 score0.5 기준 | a |
| --- | ---: | ---: | ---: |
| 팔 가속도 | 3rad/s² | 6관절 모두15rad/s² | 0.0048135221 |
| Hand 가속도 | 8rad/s² | 6관절 모두30rad/s² | 0.0014321223 |
| 힘 증가율 | 10N/s | 100N/s | 0.0000855737 |
| Moment 증가율 | 0.5Nm/s | 5Nm/s | 0.0342294904 |

각0.5 해석은 같은 구성 점수의 다른 비용이0일 때다. 한 팔 관절만15이고 나머지가 허용
이하면 r_acc=2^(-1/6)≈0.8909다. 두 팔/hand 그룹이 모두 L이면 r_acc=.25다.
힘100N/s와 moment5Nm/s가 동시에 발생하면 r_FT=.25다. Force100N/s는20ms에2N,
moment5Nm/s는20ms에0.1Nm 증가다. 힘20N/s는 r_FT≈0.99148이고, 같은20N을
0.2s 동안 선형으로 올리는100N/s는.5다(다른 moment 비용0). Peak나 종료 한계와 다르다.
팔15rad/s²는0.1s에 관절 속도를1.5rad/s 바꾸는 가속이고, hand30rad/s²는 현재
1rad/s 속도 상한까지 약33ms에 도달하는 가속이다. 이 해석으로 급격한 동작에 중간 점수를
배정하되 실제 task 성공·충격 통계를 보고 L과a를 조정할 초기 설계값으로 둔다.

전체 reward weight는 앞서 제안한 C/R/M1.8/.6/.6과 Time/Terminal 유지안을 함께 검토한다.
이번에 이를 production 설정에 적용하지 않았다. 과도한 하중·선반 변위 등의 종료 조건도
완화하지 않았다. 정지와 허용 범위의 동작은 같은 Motion최고점을 가지므로 성공/가림 감소와
각구성점수 분포를 확인할 짧은 재학습이 필요하며, 이 진단으로 비조작 해결을 주장하지 않는다.

재현 자료: `outputs/contact_manipulation/motion_thresholds_20261016/`의
`probe.py`, `probe.json`, `samples.pt`, `proposal.json`, `replay.json`, `probe.log`.
Offline replay에서 현재 정책이 도달하는 Kp50/100 표본의 제안 Motion 최소값은0.9998168,
추가 Kp200 진단까지 포함한 최소값은0.9818738이었다. 측정한 작은 무접촉 동작을 거의
감점하지 않는지는 확인했다. 현재 probe가 resolved joint IDs를 따로 저장하지 않아 이전
live PhysX joint metadata를 상속했으며, 해당 선택의 가속도 통계54개를9모드 probe 요약과
대조해 모두 일치함을 확인했다. 이 provenance와 입력 hash는 replay.json에 기록했다.

### 2026-10-16 — 빠른 접근 후 필요한 급감속을 구분해야 한다는 사용자 보완

사용자는 빠르게 접근한 뒤 물체 접촉 직전·접촉 중 감속하여 정밀 제어해야 하므로,
가속도 크기를 대칭적으로 벌하는 Motion이 필요한 급감속까지 억제한다고 지적했다.
기존 코드의 joint_acc_l2와 앞선 abs(qddot) 초과량 제안 모두 가속·제동을 구분하지 않는다.
작은 주기 동작으로 얻은 b_arm3/b_hand8만으로 이 요구를 검증하지 못했으므로 이 기준을
실제 조작의 확정값으로 취급하지 않는다. Production reward/weight/controller는 변경하지 않았다.

검토 방향은 속력 증가와 제동을 구분한 비대칭 가속도 비용이다. 물리 구간별로
`c_j=relu(abs(qddot_j)-b_j)^2 * (beta + (1-beta)*p_j)`를 후보로 둔다.
`p_j`는 해당 구간 중 관절 속력이 증가한 비율이며, 순수 가속1·순수 제동0이다.
0<beta<1이면 필요한 제동에 작은 배경 비용만 남긴다. Beta값은 아직 선택/적용하지 않았다.
이는 초기 임계값을 단순히 올려 가속/제동 모두 면제하는 변경과 다르다.
갑작스러운 제동이 항상 좋은 것은 아니므로 모든 제동을 무조건 무료로 간주하지 않는다.

Native physics interval h=.01에서 v0=v1-h*qddot로 이전 속도를 복원할 수 있다.
구간 중 일정 가속도 근사에서는 v0*v1>=0이면 p=1[abs(v1)>abs(v0)],
부호가 바뀌면 p=abs(v1)/(abs(v0)+abs(v1))이다. 단순한 양의 속력 끝점 차이는
동일 속력의 반전(+.1→-.1)을0으로 평가하므로 그 식으로 바로 대체하지 않는다.
현재 reward는 마지막 physics substep만 읽어 첫10ms의 피크를 놓칠 수 있다는 한계도 별도다.

접촉에 의한 급정지는 joint velocity가 감소해도 F/T 증가 비용으로 평가한다.
그러나 보상된 wrist net wrench가 모든 국소 충격을 관측한다는 보장은 없다.
Jerk/일반 action-rate 항도 필요한 제동 시작을 벌할 수 있어 자동 대체하지 않는다.
감속 허용은 접촉 직전부터 필요하므로 tactile 접촉 후에만 면제하는 gate로 해결하지 않는다.

관절 속력의 감속과 EEF의 접근 속력 감속은 Jacobian과 관절 결합 때문에 일치하지 않는다.
따라서 이 비대칭식이 필요한 EEF 제동을 전부 면제한다고 주장하지 않는다. 다음 기준 검증은
빠른 접근→접촉 전 제동→정밀 조작의 실제 동작을 포함해야 한다. 기존 무접촉 작은 주기 동작
측정은 센서 잔차·저주파 기준선으로 남기되 정상 조작 전체의 허용 범위를 대신하지 않는다.

사용자의 plot 요청에 대해 `motion_thresholds_20261016/acceleration_braking_plot.png`와
동명의SVG를 생성했다. b3rad/s², beta0.1, a=ln2/12²의 예시이며 적용/확정한 설정이 아니다.
좌측은 순간 비용 c_j(시간 적분 아님), 우측은 exp(-a*c_j) 커널(전체 Motion 아님)이다.
15rad/s²에서 순수 가속 p1의 커널은0.5, 순수 제동 p0은0.93303이다. 한 관절만 움직이면
실제 그룹 평균의 지수에는1/6이 적용되며, 팔6관절이 같은 크기이고 hand 비용0일 때
그림 우측이 가속도 그룹 점수와 일치한다. Production 코드는 변경하지 않았다.

### 2026-10-16 — Native action-rate·joint-velocity와 F/T 급증의 조합 제안

사용자는 속도·운동 측면에 action_rate_l2와 joint_vel_l2를 쓰고 큰 힘 변화에 별도 보상을
두는 것이 효과적인지 물었다. 다음 비교 실험의 기준선으로 타당하다고 검토했으며,
production reward/weight는 변경하지 않았다. 기존 abs(qddot)·비대칭 제동 비용을 이 항들에
자동으로 중복 추가하지 않는다. 지금 단계에서는 이전 가속도 threshold/제동 beta 확정도 보류다.

- 공식 action_rate_l2는 sum((raw_action-previous_raw_action)^2)이며 /dt가 없다.
  명령 변경을 억제하지만 급제동 명령 변경도 비용이다. 일정한 큰 pose_rel action은
  빠른 이동을 만들더라도 rate 비용0일 수 있다.
- 공식 joint_vel_l2는 선택된 관절의 실제 joint_vel 제곱합이다. 감속하면 비용이 줄며
  감속도 자체를 벌하지 않는 장점이 있으나 지속적인 빠른 운동에도 비용을 준다.
  따라서 빠른 접근을 살리려면 이 항의 영향은 약하게 시작해야 한다. SceneEntityCfg로
  팔6관절 또는 독립12관절을 명시하고 mimic 관절을 중복 집계하지 않는다.
- 우리 raw18D action은 모두 bounded 수치지만 pose_rel6/Kp6/hand target6의 물리 효과는
  다르다. Kp가 같은50으로 clamp되는 구간 내 raw 변화도 native action-rate는 벌한다.
  이 한계를 기록한 채 공식 기본 항을 먼저 쓸 수 있으며 실제 가속/충격 제어 보장은 아니다.
- F/T는 센서 중심의 자중·관성 보상된 norm의 큰 양의 증가를 평가하는 항으로 유지하는
  제안이다. 하중 감소는 비용을 주지 않으며, 모든 국소 충격의 완전한 관측은 아니다.
- 기존 exp(-a*x) 원칙을 유지하면 native 두 함수의 반환 cost를 exp의 x로 사용한다.
  음의 weight로 native raw cost를 직접 더하는 방식과 구분한다. 기존 quality wrappers는
  manager의 dt 곱을 상쇄하므로 예제의 raw penalty weight를 그대로 복사하지 않는다.
- 세부 항을 늘리면서 F/T의 실제 비중을 자동으로 낮추거나 움직임 규제를 강화해 비조작을
  악화시키지 않도록 구성 점수/실제 기여량·가림 감소·접촉률·성공률을 분리해서 확인한다.
  이 항들은 GPU tensor 차이·제곱합이며 추가 sensor/PhysX SDK 조회는 필요하지 않는다.

공식 근거:
[native rewards](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/mdp/rewards.py),
[Reach/OSC reward 구성](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py#L155).

### 2026-10-16 — Tactile 관측·Blocker 접촉 보상 관련 연구와 공식 예제 검토

사용자는 tactile을 어떻게 활용할지, Blocker 접촉 중이면 보상을 줄지, 관련 연구와
Isaac Lab 예제를 조사하도록 요청했다. 이번에는 조사·설계 비교만 수행했고
production 센서·observation·reward·PPO는 변경하지 않았다.

**현재 신호에서 확인한 제약**

- 하나의 grouped ContactSensor가17개 링크의 net normal contact force를 읽고,
  norm>0.01N을 접촉1로 만든다. 비접촉0, 무효/reset은-1이다. 현재·이전 관측에 있으며
  현재 모델은 이를 최종MLP에 직접 전달한다. 별도 tactile reward는 없다.
- 이17bits는 접촉 부위만 구분하고 상대가 Blocker·Target·선반·다른 물체인지는 알 수
  없다. Binary에서 Blocker identity를 복원했다고 주장하지 않는다. Net force 벡터의
  상쇄 가능성이 있으며 접촉점·접선력·slip을 직접 측정하는 신호도 아니다.
- 공식v2.3 ContactSensor는 한 환경 안에서 여러 sensing body를 묶은 센서의 partner
  filtering을 보장하지 않는다. 현재17body 센서에 filter만 추가하면 된다고 설명하지 않는다.
  정확한 Blocker 접촉은 별도 partner-resolved privileged 정보가 필요하며 view/버퍼/SDK
  비용을 검토해야 한다. 예전17sensor 구성을 성능 측정 없이 복구하지 않는다.
  [공식 센서 필터 제약](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py#L49)

**논문 — 각 연구의 보상을 구분한다**

1. [Improved Learning of Robot Manipulation Tasks via Tactile Intrinsic
   Motivation (RA-L2021)](https://arxiv.org/html/2102.11051v1), §IV-B:
   누적 force가 임계값을 넘으면 episode 끝까지 intrinsic reward1을 주는 latch다.
   task/intrinsic weight는.75/.25다. 단일 물체 Push/Slide/PickPlace의 탐색을 개선했지만,
   센서 배치에 따라 테이블을 누르는 악용도 저자들이 명시한다. 우리의 episode당1회
   보너스와 다른 식이며 DDPG/HER·contact-prioritized replay를 PPO에 바로 복사하지 않는다.
2. [Tac2Motion (2025)](https://arxiv.org/html/2509.17812v1), §II-B:
   뚜껑 회전을 위해 접촉압력의 기하 proxy·contact release·접촉 품질과 회전 진전의
   결합을 보상한다. 접촉 유지뿐 아니라 finger gaiting과 실제 작업 진전이 함께 필요하다.
   Binary17 입력만 사용하는 우리 과제와 달리 tactile embedding·추가 접촉 모델을 사용한다.
   Grasp/friction을 늘리는 목적을 그대로 pushing·pivoting에 적용하지 않는다.
3. [TaRL: Learning General and Physical Rewards from Tactile Demonstrations
   (2026 preprint)](https://arxiv.org/html/2609.36785v1), §III:
   성공·실패 tactile deformation-map 시퀀스로 task progress reward 모델을 학습한다.
   단순한 접촉 지속/개수 보상이 아니다. 별도 시연과 reward model이 필요하고 현재의
   binary17·현재/이전 관측만으로 같은 방법을 바로 구현할 수 없다. 새 encoder/16step
   history/reward network를 자동 추가하는 제안은 아니다.
4. 앞서 검토한 Gentle Manipulation2019의 force-growth penalty와 contact-focused
   surprise는 접촉 회피 문제의 근거다. 다만 그 predictor ensemble·추가 critic을
   단순 contact bonus라고 설명하거나 현재 PPO에 자동 도입하지 않는다.

**공식 Isaac Lab v2.3.0 — 로컬 fork와 구분해 확인**

- Dexsuite Kuka–Allegro는 각 fingertip에 object-filtered ContactSensor를 두며,
  `thumb_force>1N AND (index>1N OR middle>1N OR ring>1N)`이면 good-contact1이다.
  Native reward weight는+0.5다. 동일 접촉 조건으로 object pose tracking reward를 gate한다.
  이는 파지용 접촉 조건이며 많은 pad를 붙일수록 점수를 더 주는 식은 아니다.
  [공식 설정](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py#L24),
  [reward 구현](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/mdp/rewards.py#L49).
- Direct InHand ShadowHand/Allegro는 일부 asymmetric 모드에서 fingertip incoming joint
  wrench를 critic state에 넣지만 기본 actor tactile 입력과 같지 않다. Reward에는
  위치·회전·action²·success/fall이 있고 contact-count bonus는 없다.
  [공식 InHand](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py#L400).
- FORGE는 force observation과 `relu(norm(F)-threshold)` penalty를 사용한다.
  큰 force의 초과 비용이며 접촉 자체에 양의 보상을 주는 binary tactile 사례가 아니다.
  [공식 FORGE](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py#L232).

**우리 적용 제안 — 미확정·미구현**

Tactile17D는 접촉 위치별 feedback observation으로 유지한다. 접촉 개수나 지속시간에
매step 양의 보상을 주면 Blocker를 쥐고 정지하거나 반복 접촉을 수확할 유인이 생긴다.
이는 현재 비조작을 해결한다고 보장할 수 없다. 단순 sum17이나 임의 물체 접촉을 Blocker
접촉으로 취급하지 않는다. Pivoting/pushing에 특정 파지 형태를 강제하지 않는다.

접촉 탐색이 병목이라면 정확한 partner 정보가 확보된 뒤 작은 첫 유효 Blocker 접촉
milestone을 비교할 수 있다. `C_B`를 정확한 Blocker 접촉, `H_B`를 이전 episode 누적
접촉 여부로 두고 `r_touch=eta*C_B*(1-H_B_previous)`, `H_B=H_B_previous OR C_B`다.
Episode당1회로 제한하여 반복 tapping·유지만으로 추가 보상을 얻지 못한다. 이는 논문
그대로가 아니라 우리 탐색 보조안이며 기존C/R/M의 연속 exp 품질 항과 별개의 이벤트다.
실제 치우기는 Clearance가 계속 담당하고 F/T 급증 비용도 유지한다. 성공률·가림 감소를
개선하는지 관측 전용 baseline과 비교해야 하며, partner 조회의 추가 비용도 검토해야 한다.

### 2026-10-16 — Tactile 보상 보류 후 Motion 재정의안

사용자는 일단 tactile 없이 진행하고 Motion을 다시 정의하도록 요청했다. 직전 논의의
범위에 따라 tactile 기반 reward를 제외하는 안을 정리하며, 이 요청으로 tactile observation
삭제나 모델 차원 변경을 자동 수행하지 않았다. 이번에는 수식 설계만 했으며 production
reward·weight·센서 코드는 변경하지 않았다.

기존 joint_acc 기반 점수는 제외하고 다음 네 비용을 각각 exp에 넣은 뒤 평균한다.

- x_action = native action_rate_l2 = sum18((a_k-a_previous)^2). /dt나 기준값 나눗셈 없음.
- x_velocity = native joint_vel_l2를 팔6+독립 hand6에 적용한 제곱합. Mimic은 제외한다.
- g_F = relu(norm(F_k)-norm(F_previous))/dt,
  x_F = relu(g_F-b_F)^2.
- g_T = relu(norm(T_k)-norm(T_previous))/dt,
  x_T = relu(g_T-b_T)^2.
- M = (exp(-alpha_action*x_action) + exp(-alpha_velocity*x_velocity)
       + exp(-alpha_F*x_F) + exp(-alpha_T*x_T))/4.
  전체 Motion weight w_M를 네 구성 항이 w_M/4씩 나누며 최대는 w_M로 유지한다.
  기존 두 그룹 구조와 달리 F/T 각각의 비중은1/4이다. 모든 raw cost를 더한 뒤 exp를
  적용하는 곱셈형 점수와 구분한다. 기존 reward wrapper의 /dt 상쇄 규칙도 유지하는 안이다.

F/T는 자중·관성 보상 후 센서 중심의 wrench norm을 사용한다. Norm은 좌표축 회전에는
불변이지만 torque의 기준점에는 의존하므로 초기 센서 위치로 옮긴 moment를 쓰지 않는다.
현재 production advance의 r×F 원점 이동은 이 재정의를 구현할 때 함께 수정해야 한다.
하중 유지·하중 감소는 x_F/x_T=0이다. b_F/b_T는 센서 양자화·보상 잔차의 작은
무감점 구간이며 실제 접촉에서 허용할 충격을 측정했다고 해석하지 않는다. 작은 무접촉
probe만으로 얻은 이전 기준을 실제 조작의 확정 수치로 재사용하지 않는다.

alpha는 각 비용의 단위와 민감도를 담당한다. 기준값으로 나누지 않아도 스케일 선택은
남는다. alpha_velocity는 빠른 접근을 과도하게 억제하지 않도록 작게 시작할 필요가 있다.
속도 감소 시 velocity 비용은 감소하지만 native action-rate는 제동 명령 변경에도 비용을
주므로 급감속을 전부 면제하는 수식은 아니다. 원래 가속도 항과 중복 부과하지 않는다.

Raw18D는 수치 범위가 같아도 pose/Kp/hand의 물리 효과가 다르며, Kp clamp 내부의
raw 변경도 native action-rate에 포함된다. 이 항을 물리 가속도나 실제 Kp 변화의 정확한
측정이라고 부르지 않는다. Joint velocity는 팔·hand의 속도 범위 차이로 팔이 지배할 수
있어 구성별 통계를 확인해야 한다. 계수·dead band의 수치는 아직 선택/적용하지 않았다.

정지는 여전히 Motion 최대다. 이 항의 목적은 조작 유도가 아니라 조작 중 부드러움이며,
Clearance·성공·Time과 함께 조정해야 한다. Tactile gate, 접촉 유지 보너스, torque reward,
새 temporal encoder는 추가하지 않는다. Native 두 함수는 이미 있는 GPU tensor를 쓰므로
추가 ContactSensor/PhysX SDK 조회 없이 구현할 수 있다.

### 2026-10-16 — Motion 네 항과 Clearance 중심 weight 적용 완료

사용자가 직전 Motion 재정의안과 앞서 제안한 weight 수정을 적용하도록 요청했다.
기존 파일만 수정했고 새로운 production 모듈·센서·모델은 추가하지 않았다.

- Native RewardsCfg: Clearance1.8, Rearrangement0.6, Motion0.6. Time/Terminal은1.0 유지.
  세 quality weight의 합은3이고, Motion의 네 하위 exp 점수는 각각 유효 weight0.15다.
- motion은 공식 action_rate_l2 한 번과 joint_vel_l2 두 번(팔6/독립 hand6)을 사용한다.
  Raw18D action 제곱 차이합과12joint 속도 제곱합을 그대로 사용하며 /dt·관절수 정규화·
  sqrt가 없다. 기존 joint_acc_l2 Motion은 제거했다. 기준관절 선정은 기존 env cfg를 재사용한다.
- action/velocity alpha는 각각0.004다. 작은 cost에서의 weighted 감점 기울기0.15×0.004는
  min의 native weight0.03×dt0.02와 같도록 정했다. Exp 포화·baseline은 달라 전체 reward가
  min과 같다는 의미는 아니다. 약한 초기 규제이며 빠른 접근의 성능 검증은 별도다.
- F/T는 norm 증가율에서 각각10N/s·0.5Nm/s를 뺀 양의 초과량의 제곱이다.
  alpha_F=ln2/90²=8.5573726e−5, alpha_T=ln2/4.5²=0.03422949로 각각100N/s·5Nm/s에서
  해당 하위 점수0.5다. Dead band는 SI-200-8 양자화·작은 보상 잔차를 고려한 초기값이고,
  반품질 지점은 성공 접촉 동작에서 추정한 한계가 아닌 설계값이다.
- reward_components의 components[:,2]는 독립적인 F/T exp 점수의 평균이다. Motion은
  (q_action+q_velocity+2*components[:,2])/4로 네 점수를 동일하게 평균한다.
  wrench_rate_cost는(N,2)의 힘/토크 초과율 제곱으로 분리했다.
- TaskState.advance는 보상된 현재 sensor-center wrench를 이전 값과 직접 비교한다.
  초기 sensor 원점으로의 r×F 이동과 불필요한 축 회전을 제거했다. Norm의 회전 불변성을
  사용하며 실제 local torque 증가를 원점 이동으로 상쇄하지 않는다.
- Tactile observation17D·현재/이전 입력·18Daction·GNN/attention은 유지하고 tactile
  reward/gate는 넣지 않았다. Joint torque 추가도 계속 보류다. 새로운 PhysX 조회·CPU tensor
  동기화·환경별 반복문은 추가하지 않았다. 실제 iteration 시간 개선을 측정한 변경은 아니다.
- Time 적분식·성공+500/실패−100·종료 조건·gamma·제어주기는 유지했다.

검증:

- CPU 전체:204 passed,14 skipped,142 subtests passed(15.84s).
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, env_isaaclab python으로 tests 전체 실행.
- 실제 GPU4환경·48 policy step(96physics step): 네 점수 수식 및 전체 reward의 native weight·dt
  적용 일치. 명시적 부분reset2환경 후에도 일치했고 Motion 최대 oracle 오차5.9604645e−8.
  관측 Motion 범위0.81229657~0.99421406, 비용shape(4,2), 완료exit0. 이 동작에서는 자연
  termination은0회였다. 테스트에서 센서 이동·회전,실제localmoment성장,load유지/감소,
  dead band·반품질·물리ramp의dt일관성,12joint선택/mimic제외,1024tensor/host조회금지를 검증했다.
- GPU자료: outputs/contact_manipulation/motion_update_20261016/verify.py,
  verification.json, verification.log. 장기 학습의 성공률·비조작 개선은 아직 검증하지 않았다.

### 2026-10-17 — IdealPD 적용 후 CUDA719 재발 조사

사용자가 제공한 UTC2026-10-16 13:06:01(한국시간10월16일22:06:01) 사고는
`logs/rsl_rl/contact_manipulation/2026-10-16_21-36-16`, PID4340,1024환경이다.
Kit `kit_20261016_213601.log:5204`와 정확히 일치하며 uptime1,800,676ms(약30분)다.
직전 `kit_20261016_213547.log`는 시작 중 KeyboardInterrupt로 중단된 별도 실행이다.

확인한 사실:

- 저장 env.yaml에 arm IdealPD, P/D0, effort_limit/effort_limit_sim150/28Nm 및 새로운
  Motion·C/R/M weights1.8/.6/.6이 적용돼 있다. 실행 git snapshot은 clean
  `42b2a7bd312ccf37b8cf0ea19a0bf4484186e37c`이며 조사 당시 HEAD와 같다.
  따라서 이전 수정 누락으로 설명하지 않는다. 실제 실패 순간의 제출토크는 저장돼 있지 않다.
- 같은 시각 kernel journal은 Xid69,Class0xc7c0,Offset0x1b0,Data0x41,ErrorCode0x53이다.
  이전 PID246920 사고와 Xid/Class/Offset/Data/ErrorCode가 모두 같다. 공개 NVIDIA header는
  해당 class를 AMPERE_COMPUTE_B,offset을 LAUNCH_DMA로 정의한다. 어느 CUDA kernel이나
  stream/PhysX/driver가 발단인지는 이 정보만으로 알 수 없다. ErrorCode0x53의 의미는 미확정이다.
- 부팅은 같은 날21:35:09KST, driver580.173.02다. 재부팅 직후 새 프로세스에서 발생했으므로
  이전 사고의 중지된 프로세스/context만을 이번 원인으로 삼지 않는다.
- Physics resume 이후 첫719까지 Kit error/warning은0개다. 명시적 OOM·contact buffer overflow
  경고도 없다. Allocator line68은719후 종료 처리 경고다. Peak VRAM/온도 기록이 없어
  메모리/하드웨어 문제를 완전히 배제하는 근거는 아니다.
- TB 마지막 완료 iteration939,총940회다. 마지막 scalar는13:05:59.904UTC이며 다음
  rollout iteration940에서 실패한 것으로 보인다. 기본 common counter 흐름에서
  940×32=30080,mod400=80이고 실패 rollout30080~30112는 공통 장면 경계를 넘지 않는다.
  이전 사고와 공통배치reset 직후라는 원인을 묶을 수 없다. 개별 부분reset은 여전히 가능하다.
- 마지막10iteration930~939의 hard_load_or_overrange 로그 평균0.91814,
  Train/mean_episode_length30.957step이다. 800~899의0.59179/111.591step보다 과부하 종료가
  우세하고 episode가 짧아지는 정황이다. Native/RSL의 종료 요약 snapshot 평균이며
  편향 없는 전체 episode 빈도나 실제 힘 크기의 측정으로 해석하지 않는다.
- Loss/entropy는800~899평균8.59058에서930~939평균−53.26935로 급감한다. Iteration928부터
  −20미만이다. Gaussian latent mean std는0.58534→0.61859로 오히려 증가했다.
  Tanh action이 경계 쪽에 집중되거나 입력 분포가 변했을 가능성을 조사할 신호지만,
  실제 action 포화율·mean·관측·토크가 없으므로 정책 포화나 CUDA 인과를 확정하지 않는다.
- 모든 TB scalar는finite,invalid_state 로그는전구간0이다. Checkpoint0/500/850/900의 모델 및
  optimizer도finite다. 최신900은사고보다약40iteration 전이며 실패 순간 정상성을 보장하지 않는다.

미해결 위험과 다음 진단:

공식 OSC는 full inertial decoupling에서(J M^-1 J^T)의 damping 없는 역행렬을 사용한다.
IK에는 conditioning 기준이 없다. IdealPD는 큰 finite torque를 제한하지만 NaN을 복구하거나
특이점 계산 자체를 안정화하지 않는다. 또한 native termination은두 physics substep후이므로
해당step이이미CUDA fault로 실패하면 과부하/invalid 검사가 실행될 수 없다. 이들은 실제 구조상
제약이며 이번 순간 singularity/NaN이 있었다는 증거는 아니다. Hand의 nonzero implicit PD를
이전 arm gain0 feedforward 초과경로와 동일시하지 않는다.

우선 physics직전 action/OSC raw·submitted torque/유한성·관절상태·reset 여부를 분리해
첫 이상을 보존하는 진단이 필요하다. 여전히 finite·bounded이면 CUDA graph/Warp/PhysX/driver
경로를 독립 A/B·Compute Sanitizer로 좁혀야 한다. 단순 reward변경·토크limit만으로719를
예방한다고 주장하지 않는다. 이번에는 production변경·새 시뮬레이션·학습을 하지 않았다.

근거 artifact는 `outputs/contact_manipulation/cuda719_20261017/`의 incident.json,
log_analysis.json,kernel_xid.log,kit_failure_excerpt.log다. 단일 근본 원인은 아직 미확정이다.
공식 근거는 NVIDIA CUDA Runtime의 cudaErrorLaunchFailure 설명, Xid catalog와
open-gpu-doc/clc7c0.h,PhysX5.4.1 GPU-memory 설명을 확인했다.

### 2026-10-17 — 최신 저장 정책의 action→OSC→PhysX 입력 실측 재실행

사용자는 실제 action과 그로 인한 OSC 출력을 계산하도록 요청했다. 원본 사고 rollout의
obs/action/state/RNG/reset count는 checkpoint에 없어 복구할 수 없으며, 최신 model_900은
사고 rollout940의 정책보다39회 PPO update 이전이다. 아래는 원본 사고 순간의 값이 아닌
별도 재실행의 실측 결과다.

`cuda719_20261017/replay_actions.py`로 동일 seed0·사고당시공통배치generation75
(layout seed6136465260097156590),1024환경400policy/800physics step을 재생했다.
학습과 같은 stochastic_output=True를 사용하고 optimizer update는 하지 않았다.
Startup을 줄이기 위해 정책 compile만False로 두고 동일FP32/TF32수학을 사용했다.
제어/물리/reward/센서 및 production코드는 변경하지 않았다. 원래 개별reset횟수와
로봇상태는 미저장이라 복원하지 못한다. 공통배치75의 시작부터 새 trajectory를 측정했다.

매physics직전(native action apply와 scene.write후)에 raw18D/processed명령,pose오차,
J/M/Minv/task inertia/Kp/Kd,OSC torque,IdealPD후 SDKcommand를 동시 저장했다.
기존행렬로 독립 계산한OSC와 실제 출력의 오차는800substep모두0이고,
raw를150/28Nm로clip한 값과SDK제출값의 오차도0이다. SDKcommand는 측정 관절반력이 아니다.
전체raw·submitted출력은finite였고 CUDA719없이완료했다. Reset4693회,진단loop26.22s이며
계측부하가 포함되므로 학습 iteration성능으로 해석하지 않는다.

409600개action /819200개환경physics sample의 분포:

| 측정 | 중앙값 | 99percentile | 최대 |
|---|---:|---:|---:|
| 상대 위치 목표 norm(m) |0.668620|1.140431|1.471160|
| 상대 회전 목표 norm(rad) |0.902465|1.404281|1.672646|
| Kp(6축 전체) |50|91.43885|99.70876|
| raw OSC 관절 torque abs(Nm) |8.20045|837.74092|16358637|

Position scale1이므로 첫세raw값이 그대로m단위상대목표다. 이는20ms마다평균적으로수십cm
떨어진 목표를 만들었다는 뜻이며 실제EEF이동거리나속도가 아니다. 전체18D성분의abs>.9는
8.6316%,pose6D만은2.2356%,전체abs>.99는0.26666%다. 따라서 checkpoint900재실행에서
모든action이끝값포화됐다고 설명하지 않는다. 끝값까지 가지 않아도 scale1에서요구변위가크다.
한 관절이라도raw토크상한을초과한환경·substep은33.2070%,관절sample단위초과는7.31205%다.
제출명령의관절별최대는정확히[150,150,150,28,28,28]Nm였다.

최대rawtorque case는env179,zero-basedpolicy78/physics156(첫substep)이다.

- delta position=[−.76739472,.00472651,.58198482]m,norm.96313204m.
- delta rotation=[.69596422,−.92632860,−.75442189]rad,norm1.38260741rad(79.21757deg).
- rawstiffness=[−.751207,.357124,.383675,.112954,−.871663,−.798456]→전축Kp50,Kd14.142136.
- handraw=[.0219398,.370575,.184531,−.585129,.802944,.767651]→absoluteqtarget
  [.847188,.424878,.851737,.298313,1.296407,1.271029]rad(설정된독립handjoint순서).
- elbowq=.000530952rad(.030421deg),wrist2=.987133rad이다. 이전probe의wrist특이점과달리
  이번peak는팔꿈치가거의펴진자세다. 저장J의sigma_min8.6639564e−5,condition24919.65이다.
  J의병진/회전혼합단위condition은수치진단이며무차원물리manipulability한계로사용하지않는다.
- acc_des=Kp*pose_error−Kd*eef_velocity=
  [−34.814003,4.377518,24.081465,69.341919,−54.054489,−47.202793]
  (앞3m/s²,뒤3rad/s²).
- taskwrench=Lambda*acc_des≈[4.79954e10,−3.52411e10,1.35886e10,−8.27166e8,8.91858e8,−5.12566e9].
  이는계산된OSCcommandwrench이며실제F/T측정값이아니다.
- tau=J^T*taskwrench=[−1013760,−16358637,2158563,−1130003,−3228160,158208]Nm.
- 실제SDK제출명령=[−150,−150,150,−28,−28,28]Nm. 따라서IdealPD제한은실제로작동한다.

별도CPU FP64 재계산에서 M은양의정부호이고(J M^-1 J^T)의condition은9.4891e9이다.
Production cached FP32 Lambda는대칭화최소고윳값−2.75997e9로양의정부호성을잃었다.
같은J/M의FP64재계산Lambda는모든고윳값양수이며최대1.33359e7이다. 극심한ill-conditioning의
수치문제가실제중간행렬에확인됐다. FP64에서안정적인6×6square-J동등식M*solve(J,acc_des)도
최대78126.96Nm여서정밀도만높여서정상토크가되는상황은아니다. 실제FP32최대는이값의약209배다.
TF32가켜져있고큰중간값의상쇄에도민감하지만TF32off비교실행은하지않았으므로전체오차를
TF32하나로단정하지않는다. CPU64로cached행렬을곱한결과와GPUFP32결과의차이는이
수치민감도의일부이며계측오차와구분한다(GPUbatchoracle은정확일치).

확인한제어문제는큰상대목표·near-singular자세·fullinverse의수치증폭과빈번한토크포화다.
원본719를재현하지않았으므로위제어문제가CUDAfault의직접원인이라는결론은여전히내리지않는다.
Artifact: action_replay_stochastic.json/.pt/.log,replay_actions.py,
peak_independent_analysis.json(동시snapshot의독립검산),독립계산helper. 전부
outputs/contact_manipulation/cuda719_20261017/아래에있다.

### 2026-10-17 — Motion 네 항을 native reward/log로 분리

사용자가 Motion을 나눠 각각 확인하도록 요청했다. 단일motion RewardTerm을 제거하고
아래4개 native RewardTermCfg를 각각weight0.15로 등록했다.

- motion_action_rate: 공식 action_rate_l2의 raw18D 차이제곱합을exp에적용.
- motion_joint_velocity: 공식 joint_vel_l2를 팔6+독립hand6에적용한합을exp에적용.
- motion_force_increase: 보상된F/T힘norm의양의증가율초과량에대한exp점수.
- motion_torque_increase: F/T센서모멘트norm의양의증가율초과량에대한exp점수.
  Joint torque reward를추가한것이아니다.

정상유한입력에서0.15*(qA+qV+qF+qT)는이전0.6*mean(qA,qV,qF,qT)와같다.
계수/dead band·Clearance1.8·Rearrangement0.6·Time/Terminal·제어·관측은유지한다.
중복지급할기존motion항은남기지않았다. Native8개term으로Console/TensorBoard의
Episode_Reward/motion_action_rate,motion_joint_velocity,motion_force_increase,
motion_torque_increase가분리된다. 기존checkpoint의입력차원/모델계약도바꾸지않았다.

불필요한FTexp재계산/평균cache를없애고reward_components는(C,R,F,T,Time)의(N,5)를
반환한다. TaskState.components는(C,R,F,T,Time,Terminal)의(N,6)이며Time4/Terminal5다.
F/T두함수는각각캐시된column을읽는다. jointselection은UR5eEnvCfg에서
rewards.motion_joint_velocity.params에설정한다. 새production파일·센서·물리조회·명시적
CPUtensor변환·customlogger를추가하지않았다. Native로그집계는8개term기준으로수행된다.

Native Episode_Reward값은각term의weighted episode누적값을max_episode_length_s=8로
나눈수치다. 0~1순간quality나실제episode길이정규화평균으로읽지않는다. Episode길이도
함께봐야하며4개로그의합으로이전Motion기여량을비교할수있다.

검증은CPU205passed/14skipped/142subtestpassed(15.92s),GPU4환경32policy step및
선택2환경부분reset2회완료다. Native8개term의이름/weight·4개Motion로그각각과
sum기여량을확인했다. 분리수식오차최대2.98e−8,전체reward의이전합산식대비차이최대
3.58e−7(float32합산순서차이)다. 원본연구수식·학습성능을바꾸려는변경은아니다.
자료는outputs/contact_manipulation/motion_split_20261017/verify.py,
verification.json,verification.log다. README도새term/로그와검증결과로갱신했다.

### 2026-10-19 — Shared scene robot-start search 재시도 소진 오류 조사

사용자가 _shared_reset_batch의 `exhausted retries for 1 environments` 예외 원인을 물었다.
이번 예외는 CUDA719가 아니라 초기자세 후보가 조건을 통과하지 못해 명시적으로 발생시킨
RuntimeError다. 외부재시도(initial_vision.max_layout_attempts)32회×내부후보(robot_start.max_attempts)
80개로 실패행마다최대2560후보를 검사한다. 성공행은유지하고 실패행만새seed로 재시도하며,
외부seed는base+7919*(attempt+1)이라 같은후보만반복하는구조는아니다. 물체배치/TCP영역은
외부재시도동안고정이다. 재시도소진후복구경로없이단일실패행도전체학습예외로전파된다.

조건은Blocker앞12~40cm(선반front최소6cm),Y±8cm,선반위20~28cm의TCP영역과고정정면방향,
IK오차0.5mm/0.01rad,elbow높이5cm/forearmforward5cm,바닥·선반·물체와15mm,
검사대상selfpair와2mm여유다. 공통layout준비는env0의유효자세한개를확인할뿐 충분한
유효샘플영역을보장하지않는다. 유효해가존재해도허용영역이좁거나IK수렴/충돌검사에서
반복탈락하면유한검색이실패할수있다. `1 environments`는최종pending개수이며envID나
원래resetbatch크기가아니다. 현재는탈락사유별집계/envID/resetcount/layout정보가없어
실제원인이IK도달성·elbow·충돌중어느것인지확정하지못한다.

최신trainingrun2026-10-17_15-52-37은29574iteration을기록했지만 해당RuntimeError가
저장stdout에없어이번예외와직접연결하지않았다. 그run이라면실패다음rollout의끝이946400
commonstep경계이므로공통reset(generation2366)과부분reset둘다가능하다. 이를확정재현으로
기록하지않는다. 조사만수행했고production변경·시뮬레이션실행은하지않았다.
개선방향은탈락원인집계와유효한TCP영역확보/검증된자세주변샘플링및소진시복구정책이며,
조건무시·무검증자세fallback·무한재시도는도입하지않았다.

### 2026-10-19 — 본 학습 기록과 수정 과제 정리

사용자가 테스트를 제외한 학습 기록과 수정 방향을 정리하도록 요청했다.
`logs/rsl_rl/contact_manipulation`의 72개 실행을 검토해 본 학습 근거가 있는 6개를
[TRAINING_LOG.md](TRAINING_LOG.md)에 기록했다. 본 학습 후보 3개·목적 불명 단기 실행 36개·
초기화만 남은 2개는 테스트로 단정하지 않고 보류했다. 명시적 진단·단기 검증 25개는
학습 성능 비교에서 제외했다. Run별 iteration/checkpoint·제어/weight 변화·종료 원인과
근거 자료를 연결하고 native reward/termination 로그의 집계 한계를 명시했다.

최신 `2026-10-17_15-52-37`은 29574회 기록, 마지막 완료 iteration 29573,
최신 checkpoint 29550이다. 첫1000회→마지막1000회 return59.15→227.18,
episode길이75.64→152.08step이나 weight를 제거한 C/R aggregate비율은0.4865→0.4995로
거의 변하지 않았다. Success로그 양수 iteration은7개, 마지막16037이며 실제 성공 episode
7개라는 뜻은 아니다. 사용자의 “초기 물체 접촉 후 hand가 선반 반대쪽으로 이동” 관찰을
기록했으며, scalar만으로 접촉 상대·진행량·후퇴의 인과를 확정하지 않았다.

남은 작업은 초기자세 탐색 소진의 원인 기록/유효영역/검증된 복구, 작업 진행과 성공 gate의
episode단위 진단, 큰 상대pose명령과 OSC수치조건, exp내부비용·weight 및 Time/Terminal/γ
검토다. 통로 확보 후 안정화를 기다리는 올바른 후퇴와 미완료 후퇴를 먼저 구분한다.
Joint torque observation/reward·tactile reward·추가 temporal encoder는 여전히 보류다.
이번 작업은 문서·기존 로그 분석 artifact만 갱신했고 production코드 수정이나 새 학습/재생은 하지 않았다.

### 2026-10-19 — Clearance 우선 보상·OSC scale·초기 pose sampling 후속 교정

사용자는 Clearance를 main reward로 두고 R/M은 penalty 또는 조건부 누적으로 쓰는 방향을
검토하도록 했다. 우선 제안은 exp를 quality로 유지하면서 R/M을 `−w(1−q)`로 바꾸는 대조다.
이는 비활동 R/M 보너스를 없애지만 양의 Clearance 상태 보상은 남으며, 코드에는 아직 적용하지 않았다.
Clearance 증가 시에만 비용을 부과하는 gate는 productive contact/pivot 준비를 불리하게 할 수 있어
즉시 비용 구성을 먼저 검토한다. Weight·Time·Terminal·γ의 최종 수치는 확정하지 않았다.

공식 v2.3.2 Franka OSC Reach의 scale은 position1/orientation1/stiffness100이나 pose_abs의
절대 위치·quaternion 입력이다. 우리 pose_rel 증분과 직접 동등 비교하지 않는다. 이전0.015/.08은
우리 과거 대조값이며 공식 상대OSC 기본값으로 설명하지 않는다. 출처는 TRAINING_LOG 6절에 연결했다.

사용자가 원하는 시작 상태는 Blocker 정면의 기준 EEF pose에 position·orientation randomization을
주고 IK를 푸는 순서다. 현재 위치만 random이고 orientation은 고정이므로 이 차이를 수정 설계에 반영한다.
기준 손바닥EEF frame에서 p=p0+R0δp, R=R0δR로 합성하고 기존 GPU batched IK/충돌/elbow 검사를
활용한다. 기존 위치 분포 재사용과 작은 회전 범위를 검토하며 최종 범위·코드 적용은 미확정이다.
CUDA719는 사용자의 의견대로 이후 실행에서 더 확인한다. 이번에는 문서 조사·설계 검토만 수행했다.

### 2026-10-19 — min과 유사한 표준 Gaussian 출력으로 전환

사용자는 tanh 압축 때문에 더 큰 action을 지속했을 가능성을 제시하며 min과 유사한 출력
방향을 원했다. 포화 현상은 가능하지만 원인은 확정하지 않았다. RSL-RL의 GaussianDistribution을
policy가 단일 소유하도록 바꾸고 학습 sample/일반 play mean·logprob·analytic entropy·KL을
native 구현에 위임했다. Tanh 관련 확률 보정과 Monte Carlo entropy를 삭제했다.
초기 std0.3과 log parameterization은 유지해 tanh 제거 효과를 분리한다. Min의 초기1.0이나
다른 PPO hyperparameter까지 모두 맞춘 것은 아니다. 새 production 파일은 만들지 않았다.

GNN/attention·244D/18D·native OSC/hand·scale1.0·Kp/토크/관절 제한·reward/reset은 유지했다.
이전 action을 observation에서 ±1로 자르던 두 처리는 제거했다. 이제 Gaussian raw action의
크기가 current/previous에 보존된다. Checkpoint output_distribution metadata를 추가해 기존
tanh 분포를 새 Gaussian으로 조용히 실행하지 않도록 weights/optimizer 전에 거절한다.
README의 현재 모델/제어 범위·새 Gaussian run의 train/play 안내도 갱신했다.

CPU learning57pass2skip/20subtest, MDP152pass12skip/125subtest, 실제 RTX3090 learning59pass/
20subtest를 확인했다. GPU 컴파일 추론/학습 전환·native PPO/microbatch·정확한 분포 계산·
구 checkpoint 거절·action>1 이력 보존을 포함한다. 실제 PhysX 학습·CUDA719 재현이나
정책 성공 개선을 검증한 것은 아니다. 본 학습 기록과 전환 내용은 TRAINING_LOG 7절에 구분했다.

### 2026-10-19 — R/M penalty와 IK 실패 복구 적용

사용자 승인으로 Clearance는 양의 exp점수, Rearrangement·Motion4항은 exp점수−1로 지급한다.
Weight1.8/0.6/각0.15·비용x·Time·Terminal은 유지한다. R/M정지보너스1.2/step을 없애지만
Clearance의 양의상태보상은 남으며 성공률개선을 입증한 것은 아니다.

Blocker앞 기존위치영역과 anchorEEF-local 회전벡터 각성분±10°를 샘플링하고 같은목표에
검증된관절값 우선/다른3branch IK를 적용한다. 당시 새장면은8개단일6Dprobe중4개이상 유효해야
채택했다. 이 추가 gate는2026-10-22 구성적 경로에서 제외했고 legacy 비교에만 남긴다.
실패행만 일반3회/근방2회 탐색후 같은장면의검증된anchor로 복구한다. 예외적으로
그행의새randomization을 포기하는 정책이며 nativeReset로그와 JSONL에 기록한다.

실제RTX3090 IsaacSim4환경32step·부분reset2회에서 수식오차1.19e−7과 neighbor/template
보존을 확인했다. 같은실제장면의1,024행 GPU sampler는모두성공, warm-up뒤약29.5–29.8ms였다.
이는전체reset/학습iteration시간이아니다. CUDAgraph내부host상수전송 오류도 캐시로수정했다.
상세수식·파일·회귀검사·한계는 TRAINING_LOG8절과 README에 기록했다. 장기학습은하지 않았다.

### 2026-10-19 — 최근 학습에 근거한 Terminal 스케일 조정

최신본학습T06의 마지막1,000iteration은 성공snapshot0, nativeTerminal평균−12.426,
Clearance평균19.151이었다. 성공500이 지배했다는 증거보다 반복된 실패100의 기여가 컸다.
성공+200·실패−50을 초기조정값으로 적용했다. 현재non-terminal상한1.8/step과γ=.99의
할인합상한180보다 성공200을 크게 두어, 성공가능상태에서 단순대기로더버는구조는 피한다.
실패비용은절반으로낮추며 종료조건·timeout bootstrap·Time·γ·다른weight는 유지한다.
T06는옛양의R/M·tanh정책이므로 변경후성능은새run에서확인해야한다. 학습은실행하지않았다.
근거·할인표·집계한계는 TRAINING_LOG9절과 outputs/contact_manipulation/terminal_scale_20261019/에 둔다.

### 2026-10-20 — 접근 거리·Blocker 접촉 유지 보상과 비용 우선 검증

사용자 승인으로17개 tactile pad의 실제 접촉면 대표점과 현재 Blocker OBB 사이의 최소 거리
`d`에 `0.10*exp(−ln(2)/0.10*d)`를 추가했다. 대표점은 collision mesh의 노출된 local+Z
표면에서 초기화 때만 구하고 이후 GPU 링크 pose로 변환한다. 현재 privileged 물체 pose를
쓰며 initial-vision 관측의 pose 갱신이나 actor 입력 추가는 하지 않는다.

접촉 유지는 **같은 pad가 직전·현재 모두 episode Blocker와 접촉했을 때만** 0.01씩 합산한다.
0.01 N 초과 normal force를 binary로 만들며17개를 평균하지 않는다. 최초 접촉0, 최대0.17/step,
다른 물체에서 Blocker로 바뀐 첫 sample은0이다. 상대 구분 없는 기존 tactile observation의
previous bit를 사용하지 않고 reward 전용 Blocker binary를 저장한다. 50 Hz 두 sample 조건이며
그 사이의 완전한 연속 접촉을 증명하지 않는다. 강한 힘·유지 시간에 비례하는 추가 배율은 없다.

Native `BlockerContactHold(ManagerTermBase)`와 공식 RewardManager.reset으로 이력을 관리한다.
부분 reset은 해당 행의 previous/count만 지우고, 같은 step 재조회는 이력을 다시 갱신하지 않는다.
필터는 공식 PhysX 한 view에서 pad×물체 pool을 고정 연결하고 GPU로 Blocker 열을 선택한다.
기존17 pad의 grouped net-force 관측과244D/18D·OSC·MDP/reset CUDA graph는 유지한다.
새 production 파일은 없으며 Time/Terminal은 사용자 설정대로 계속 비활성이다.

먼저 실제1,024환경에서 기존/거리만/거리+접촉 판별을 비교했다. 정지 명령은
37.435/38.467/39.145 ms/step(+4.57%), 작은 접근 역순은
36.743/37.791/38.940(+5.98%), 실제 접촉·자연 부분 reset을 포함한 접근 역순은
47.492/48.572/48.035(+1.14%)였다. 초기 상태/reset 순서가 같았고 후반 두 비교는 최종
상태 hash도 같았다. 앞의 두 비교는 접촉0이므로 실제 접촉 성능 근거로 사용하지 않는다.
후반 비교는 Blocker 접촉·직전부터의 유지·다른 물체 접촉 제외를 실제 sample에서 확인했다.
큰 새 병목이 없어 유지하며, `env.rewards.blocker_contact_hold=null`이면 필터와 유지 보상을
끄고 기존 net-contact backend와 거리 항으로 돌아간다.

실제8개 native reward를 활성화하고 임시 hook을 끈 최종1,024환경·1,008step 측정은
38.277 ms/step,24-step 수집0.91865초였다. 첫 정지 baseline0.89843초보다 약2.25% 증가지만
별도 단독 실행의 비교다. 초기 hash와 reset 순서는 일치했다.

이 값은24-step 환경 수집 비용이며 PPO update/logging 포함 iteration 시간이 아니다.
실제CUDA4환경16step에서 native reward/reset/관측 유지 검증을 통과했고 CPU 검사235pass/
18skip/150subtest였다. 이전 Time/Terminal 활성 가정의 cfg 검사도 현재 비활성 설정에 맞췄다.
두 지원 항도 접촉한 채 정지하면 양의 보상을 주므로 본 학습에서 접촉 이후 Clearance 증가와
성공률을 비교해야 한다. Dense 상한은2.07/step, γ=.99의207이 되어 과거 Terminal200의
180 상한 근거는 더 이상 현재값이 아니다. Terminal을 재활성화하거나 재조정하지 않았다.
상세 근거와 artifact는 TRAINING_LOG14절·outputs/contact_manipulation/contact_reward_cost_20261020/에 둔다.

### 2026-10-20 — 후속 학습 분석: 약한 접근 신호·latent hard gate·분산 증가

사용자는 D/H 도입 후 Clearance·success 개선, hard-load의 잦은 종료, 거리 weight/a의
재검토를 요청했다. **분석만 수행했고 production reward·종료 설정은 수정하지 않았다.**
10월20일 새7run 중 앞4개는4환경 검증으로 제외했다. 비교 가능한1,024환경 실행은
01:54:22(C/R/M,last421),02:49:46(+D.1/H.01,last755),03:16:36(같은보상,hardnativeOFF,last3250)다.
세run의 agent/seed0/wrist/gravityTrue는같고, 첫차이는D/H추가, 다음차이는hardnative제거뿐이다.

공통321–420iteration에서 C native로그11.440/13.048/24.092, episode길이112.34/125.72/
298.71step이었다. C로그는reset그룹누적값/8초이고 Train길이는최근100episode 버퍼라
서로 나누어품질을복원하면 이론하한도위반한다. 따라서 C로그상승을가림개선배수로해석하지않는다.
HardOFF학습은190–2612사이86개iteration에서successsnapshot이양수였으나 마지막100은0이다.
최대0.5524%는최근종료원인snapshot이며 누적성공률이나86회성공이아니다.

최신학습최초→최종100회는 D/C지급비율2.235→0.900%, 유지로그8.29e−5→3.92e−7,
평균episode303.52→373.23step, timeoutsnapshot85.63→99.08%였다. 접근신호는작고 유지접촉은
거의없으며 후반계속개선중이라고보기어렵다. 유지기회가희박한원인을weight만으로단정하지않는다.

중요한구현불일치: 사용자가hard_load_or_overrange를nativeTerminationsCfg에서꺼도
task_conditions.failed에는hard가남아success/stability를차단한다. extras.terminated에도남아
공통scene timeout의bootstrap을잘못막을수있다. 일반step의native done을바꾸는것은아니다.
Terminal은off지만추후그대로켜면disabledhard가−50및terminal_paid잠금을일으킬수있다.
진단용hardflag와활성failuremask를분리해native done/success/terminal/bootstrap을일치시키는
교정이우선이다. 보상force80N·torque4Nm·rawSI축overrange를분리집계한후지속시간/허용기준을정한다.

또Gaussianmeanstd최초.306→최종100회48.64였다. model3250의그룹std는위치.373m,
회전63.26rad,rawKp54.19,rawhand74.93이다. 따라서위치std48m라고설명하지않는다.
Kp/hand의범위clamp와큰raw분산이겹치고action-rate exp비용은포화하기쉬워,후반정책분산문제를
weight만으로해결할수있다고보지않는다. 현재entropy_coef=.01·unboundedGaussian을유지했으며
분산안정화/entropy대조는별도검토한다.

예제거리식은선형거리의exp(−10d), reaching/mainweight2/6또는3/6이다. 예제는dt를곱하고
우리는상쇄하므로rawweight를직접복사하지않는다. 최초대조후보는C1.8,D.4,H.01,R.6,M각.15로
Dweight만4배, aD=ln2/.1과aC1은유지한다. 다음aD5/10을분리비교한다. 큰a는원거리신호를약화할수
있고현재palm시작거리.12–.40m는최소pad거리와같지않으므로실제dquantile계측이필요하다.
이것은미적용후보다. Time/Terminaloff의양의C/D/H지급은접촉정지/성공직전대기의이득을남기므로
본학습전완료인센티브도검토해야한다. D.4후dense상한2.37/step(γ.99상한237)은그자체로terminal을
자동결정하는수치가아니다. 상세표·그래프·설정diff·CPU판정재현·checkpoint분석은TRAINING_LOG15절과
outputs/contact_manipulation/reward_followup_20261020/에보존했다.

### 2026-10-20 — 승인 weight 적용·성공 판정 분리·PhysX fault 조사

이전 절의 weight 후보를 사용자가 승인했다. 현재 C1.8/D0.4/H0.01/R0.6/Motion4항각0.15이며,
실제 변경은 거리 weight0.1→0.4뿐이다. aD=ln2/.1, aC=1, Time/Terminal/hard-load OFF는 유지한다.
Dense 상한2.37/step과 γ.99 상한237은 Terminal scale 자동 변경의 근거가 아니다.

Success는 Target 초기 pose 대비2cm/.15rad 이내 + 지정 Blocker/A₀ 교차≤1e−5m²만
5sample(.1초) 연속 검사한다. 전체 물체 속도, pending tilt, F/T, 선반, 다른 실패를 성공 gate에서
제거했다. 활성 실패는 별도로 종료하며 성공과 동시에 기록될 수 있다. Terminal 활성 시 지급은
실패 우선이다. Native 활성 실패 mask와 success 활성 여부를 초기화 때 캐시해 내부 failure,
Terminal 지급, extras.terminated를 설정에 맞췄다. Disabled hard는 진단만 유지하며 bootstrap을
막지 않는다. 관측244D·action18D·제어·물체 형상·PhysX buffer는 변경하지 않았다.

첨부2026-10-19 19:51:55UTC 오류의 최초 표시 kernel은 artiSolveInternalTendonAndMimicJointConstraints1T다.
같은시각·같은PID1432859의 kernel journal에서 Xid31 FAULT_PDE ACCESS_TYPE_VIRT_WRITE가 확인됐다.
GPU 주소 접근 오류의 증거이며 단순 계산 부하/OOM 증거는 아니다. PID와 TensorBoard 마지막3250이
03-16-36 실행과 일치한다. 첨부626줄에는 overflow/OOM 경고가 없으나 이전경고까지 배제할 수 없다.

실제 환경당 collision shape는 robot46+shelf17+pool45, 그중 Mustard32다. 과거 hand5×128 hull은
재도입되지 않았고5개 parent각2proxy를 유지한다. 전체1,024환경은 convex91,136/box19,456/plane1,
trimesh0이다. Native 통계를 읽은 단기 hold/접근/checkpoint3250재생과 별도 희박한 tactile접촉 검사에서
contact sample최대92,637/8,388,608(1.10%), patch24,889/163,840(15.19%)였고719/overflow는 재현되지 않았다.
이는 장기 peak도 조밀한 hand접촉 stress도 아니므로 충돌량 원인을 배제하지 않는다. Heap64MiB는
초기크기이며 확장된2.50GiB를 overflow라 부르지 않는다. Stack0통계는 미확인으로 취급한다.
버퍼 일괄증설·충돌 단순화·joint변경 없이 먼저 최초 실패kernel을 동기진단/memcheck로 좁혀야 한다.

별도 Hand정확성 문제: URDF mimic6개와 달리 실제stage mimicAPI/tendon은0개다. 설치converter는
convert_mimic_joints_to_normal_joints=False를 parse_mimic=False로 직접넘긴다. 공식 importer test도
False이면 mimicAPI가 없음을 확인한다. Distal6관절은PD0인passive DOF이며 native mimic보존을
가정하면 안 된다. 이번 오류kernel 이름만으로 mimic원인을 주장하지 않고 후속 교정으로 남겼다.
CPU 검사는241pass/18skip/152subtest다. 실제GPU4환경×8step은 종료0·유한관측/reward,
8reward/6termination/mask·부분reset neighbor보존을 통과했다. 복사입력의 별도 nativepredicate
검사에서도 disabledhard와 높은 속도가 task success/bootstrapping을 막지 않음을 확인했다.
상세는 TRAINING_LOG16절과
outputs/contact_manipulation/physx_failure_20261020/에 보존한다.

### 2026-10-20 — 최신13:08 학습의 후반 정책 불안정 분석

분석 snapshot은13-08-06의 scalar4934(15:44:38KST), checkpoint4900이다. 새12-50-52는191회,
checkpoint150까지며 설정상test로 단정할 근거가 없다. 둘 다1,024환경/seed0/fresh/목표90000,
C1.8/D.4/H.01/R.6/M4×.15, 새success·disabled-hard mask가 실제 저장 소스에 적용됐다.
Time/TerminalOFF, entropy.01/unboundedGaussian/OSC/IdealPD/hand 설정은 이전과 같다.

최신long first100→last100: reward253.66→5.69, episode294.84→9.42step(.188s), 낙하snapshot
.448→78.112%, timeout84.91→14.84%, 전체meanstd.306→148.235. 최신model의 위치σ는축별
16.018/14.427/32.082m이며 평균20.842m다(실제이동량X). 이전동일3250은posσ.373m,새것5.251m.
Value loss는4292부터폭증,4309에3e20,4555최대3.284e32; invalid_state는더늦은4412처음기록.
마지막value loss28대로작아져도짧은episode·낙하우세는회복되지않았다. Scalar와169checkpoint의
actor/optimizer는finite이며 critichead weight폭증도없어, 큰유한입력/activation/V/target을확인해야한다.

성공신호는중간일부있지만새predicate는failure와동시true가능하며native scalar는최근원인snapshot이므로
실패없는실제성공률을복원못한다. C/D로그는reset그룹eps합/8초라거리·가림개선값이아니며 Dweight4배효과도
섞인다. 지속적clearance/접촉개선을입증하지못했다. Exp action-rate penalty포화와logσentropy증가,
rawpreviousaction의무정규화history입력은우선추적후보다. Entropy축소/0대조와GPU집계기반의action·obs·V·
bootstrap/target/gradient로그가다음진단방향이다. 이번에는분석·문서만추가하고production설정은유지했다.
수집+update시간1.735→3.096초이며update는.20→.19로일정하다. 잦은reset이후보지만세부비용분해는없다.
TRAINING_LOG17절과 outputs/contact_manipulation/recent_training_20261020/에수치·그래프·재현코드보존.

### 2026-10-20 — Hold 영향과 실제 object physics 정리

16-54-31의3772까지분석에서H/C지급비중은900근처.001894%,1550근처.001919%,마지막100.000040%다.
H는samepad의previous/current Blocker force>.01N일때pad당.01/step이고첫접촉·pad외링크접촉은0이다.
실제접촉률은이nativeeps로그에서복원못한다. Weight.03/.05 대조를제안하되실제반복접촉기회와
접촉이후Clearance진행을계측해야한다. 이번에는weight를변경하지않았다. 현재source의R주석은
사용자변경으로보존했으며16:54학습YAML의R=.6과구분한다. Robotselfcollision은현재True다.

환경1개native조회로질량kg확인: bottle=0.549279/cup_1=0.447285/cup_2=0.271988/mug_1=0.641424/
mug_2=0.534807/can=0.368075/YCB003=0.411/004=0.514/005=0.349/006=0.603. 기존6종은질량미지정+기본밀도1000kg/m³로
충돌체적에서자동계산되며실측값아니다. YCB만명시질량이다. Mass/COM/inertia는Oct14snapshot과동일.
물체10종·선반·17tactilepad는모두nativeμs=.5/μd=.5/e=0,combineaverage이며재질별차이가없다.
관성자동계산,선반100kg,contactoffset기존6종+선반1.962mm/YCB5mm,rest0,linear/angular damping0/.05,
solver16/2,maxdepenetration1m/s,중력활성,CCD비활성,질량/마찰resetrandomization없음.
각속도USD단위주의: 기존default5729.58deg/s≈100rad/s,YCB100deg/s≈1.745rad/s로상한이다르다.
Full3×3inertia/COM/원본과composedUSD/native값은TRAINING_LOG18절과
outputs/contact_manipulation/contact_hold_physics_20261020/에보존. 물성은변경하지않았다.

### 2026-10-20 — Hold0.05·정책100Hz와 self-collision slowdown 확인

사용자 승인으로 Hold weight.01→.05를 적용했다. Same-pad Blocker의 previous/current
binary 합 수식은 유지하며 최대기여.85/step이다. 마찰은 μs=.5/μd=.5/e=0을 유지하고
추후 조정할 수 있는 항목으로 남겼다. 사용자 R주석과 Time/Terminal/hard-load OFF도 유지했다.

추가 요청대로 physicsdt.01·decimation1로 정책100Hz를 적용했다. 성공확인10step으로.1초를
유지하고, tilt.2초는20step·8초공통장면은800step으로 자동 반영된다. 관측244D/action18D는
유지하되 current/previous 간격은10ms다. γ=.99/rollout32/다른weight/scale은 그대로다.
Dense는step별지급이므로8초지급기회가두배이며 rollout시간은.32초다. 이전50Hz모델play에는
`env.decimation=2 env.mdp.stable_steps=5`를 추가한다. GUI렌더까지 맞추려면
`env.sim.render_interval=2`도 추가한다. 이미 실행 중인 학습은 소스 설정을 자동 반영하지 않는다.

과거13:08/16:54/20:18의동일101–359iteration 비교에서 collection1.630/2.809/2.825초,
PPOupdate.205/.201/.204초다.13:08→16:54의저장설정차이는selfcollisionFalse→True뿐이며
16:54→20:18은Roff뿐이다. 최신episode302step은baseline301step과유사하고 일반logging등
overhead약.0235초여서 이번약1.19초증가를설명하지못한다. 물리/reset최적화rollback은없다.

실제RTX3090/1024환경/동일초기tensor해시/nativehold로재현:32step환경실행은
selfOFF50Hz1.1380초→selfON50Hz2.3873초,physics호출8.825→22.817ms였다.
각경우공통장면교체1회·earlytermination0으로reset빈도차이가아니다. SelfON100Hz는1.3069초/32step,
800stephorizon·success10/H.05/7reward를확인했다. NN추론/PPOupdate없는환경계측이며
100Hz의같은32stepsimulation시간은절반이어서 새학습iteration성능보장이아니다.

Env0nativecontactmatrix에서palm `right_plam_1`↔proximal5개의 큰 내부 접촉력(약2.9–9.7kN)이
측정됐다. 서로같은base_link의자식이라parent-child자동filter에서빠지는sibling쌍이다.
다음개선은handhousing의의도된중첩을형상/관절범위로확인하고필요한pair만공식FilteredPairsAPI로
제외하는것이다. 아직filter/UR5e/selfcollisionTrue는변경하지않았으며719원인해결도미확정이다.
CPU241pass/18skip/152subtest,실제100Hz832step+부분reset종료0.
상세·한계·재현산출물은TRAINING_LOG19절과outputs/contact_manipulation/slowdown_20261020/에보존한다.

### 2026-10-20 — Hand URDF 범위 일치와 실제 역방향 굽힘 검증

원본/relativeURDF의31joint kinematic/limit/mimic 정의는같다. 손12revolute의native
hard/softpositionlimit도원본과최대1.389e−7rad차이로일치한다. 모두lower0이고
공식JointPositionToLimitsAction도[-1,1]clip후같은범위에매핑하므로음수목표명령문제가아니다.

하지만실제1환경/100Hz/열린손목표0/200physicsstep에서selfcollisionON은little2−30.21°,
thumb2−29.69°,middle2−26.53°,ring2−27.34°까지이탈했다. 첫step부터큰housing–proximal
접촉과한계위반이있고,동일조건OFF에서는최대하한이탈.000017°/상한.000039°수준이었다.
현재hand내부충돌제약아래solver가관절한계를크게위반하는상태를재현한것이다.
따라서positionlimit을확대하거나실제q를매step덮어쓸문제가아니다. 우선handhousing의
의도된중첩과proxy형상을검토해필요한pair만공식filter/형상교정으로처리한다.
ProductionselfcollisionTrue/UR5e/hand형상은아직변경하지않았고719원인도미확정이다.

별도문제로URDFmimic6개대신실제stageMimic/TendonAPI0/nativefixed·spatialtendon0,
distal6개PD0passive를재확인했다. OFF에서도thumb3약31°대thumb2약.153°이하로연동이
보장되지않는다. Mimic공식import복구시개별upper와선형mimic비율충돌도검토해야한다.
Handeffort상한은원본1Nm대현재10Nm이며이차이는이번대조에서유지했다.
상세는TRAINING_LOG20절과outputs/contact_manipulation/hand_limits_20261020/에보존한다.

### 2026-10-20 — 공식 Hand mimic 복원과17 tactile 보존

Hand는 공식URDF importer·ImplicitActuator·JointPositionToLimitsAction과 native PhysX
mimic6개로 연동한다. 설치converter의flag를 실제parse_mimic=True로 전달하고
joint_drive=None·force drive를 사용한다. Python의매step coupling/관절각 덮어쓰기는 없다.
Handeffort는URDF1Nm를계승하며 gain은독립6개20/1·종속6개0/0이다.

URDF연동식과전체position/velocity제약을구성시교차해 master한계에반영한다.
손가락master0–1.36159181rad·.922253966rad/s,thumb2는0–.609220762rad·2rad/s다.
종속관절은importer의확장limit·별도velocitybrake없는구성을유지한다.
중복된childstop/brake는native mimic과충돌했다. 실제12관절각을원본URDF범위로검증한다.

Pythonconvexproxy를제거하고원본mesh에공식convexdecomposition을사용한다.
Palm최대16hull/fingerparent4개각4hull·hull당64정점,17padmesh/body/report는보존한다.
Housing2×proximal5의연결부10쌍만공식FilteredPairsAPI로제외하고나머지selfcollision은True다.
공유geometry에서crossbody필터대상을livebodyroot에보존해복제후유실도수정했다.
Handmaxdepenetration.1m/s,결합robotposition solver32/velocity2다. 팔·손이같은articulation이라
solver반복수는전체로봇에적용되며UR5econtroller/actuator/collisionmesh는유지한다.

1환경100Hz800stepopen/half/close/reopen에서32회solver의최대mimic오차.030736rad,
actualURDF하한/상한최대이탈.0000465/.0000677rad,유한관절/F/T를확인했다.
속도는독립관절native제한이며contact중child순간속도초과는남는다(index2최대9.25rad/s).
64회solver에서도child별도velocitybrake대조는불안정해채택하지않았다.
4환경에서17개pad모두실제접촉·binary관측,Blocker구분,hold이력,
실제부분reset의다른환경보존을통과했다. 관측244D/action18D와reward는유지한다.
실제1024환경832policy step에서도유한상태·부분reset[0,512]·800step장면교체0→1을통과했다.
Native32step평균1.33313초(전체reset포함)이며NN추론/PPOupdate는제외한다.
동일초기상태의짧은비교는solver32회1.330초/64회1.955초로,비용47%차이와관절범위를
고려해32회를채택했다. .02rad mimic진단기준은32회전체개폐시험에서false로보존했다.
기존checkpoint와Handdynamics가다르고장기719원인해결은아직입증하지않았다.
상세·비교는TRAINING_LOG21절과outputs/contact_manipulation/hand_urdf_native_20261020/에보존한다.

### 2026-10-20 — 초기 Clearance 기준 보정 적용

사용자 요청으로Clearance를 `1.8(exp(−x_t)−exp(−x_0))`로변경했다(a_C=1).
초기와같으면0·개선양수·악화음수이며,초기로복귀하면0이다.
직전step차이가아니므로개선된상태를유지하면보상은계속된다.
초기기준은실제초기BlockerOBB에서GPU로계산하며부분reset행만갱신한다.
Rawquality캐시는유지하고공식manager어댑터에서차이를/dt로반환한다.
관측·weight·termination은유지한다. 장면별Clearance최대는1.8(1−exp(−x_0)),
현재전체dense상한은약2.388/step이다. 이후Clearancelog는과거절대점수와직접비교하지않는다.
상수보정이라개선기울기는같으며고정길이episode에서는return상수이동이다.
조기종료시생존유인이달라지므로이변경만으로정체해결을보장하지않는다.
CPU244pass/18skip/332subtest,RTX3090GPU1024행의부호·정지0·부분reset·반복지급을통과했다.
초기품질CPU/GPU오차2.47e−8,기준query핵심.182ms로전체reset/iteration비용은아니다.
상세는TRAINING_LOG22절과outputs/contact_manipulation/clearance_initial_20261020/에보존한다.

### 2026-10-20 — Terminal 성공50·Clearance weight4 적용

사용자가활성화한Terminal을성공+50·그외0으로변경했다(weight1).
기존failure우선감점을제거해success/failure동시참도+50이며episode당한번이다.
실패판정·종료원인로그는유지한다. 성공없이실패/timeout만있으면0이다.
Legacy failure_penalty는0으로보존하지만계산에는사용하지않는다.

새비test학습22-33-03(1024환경/100Hz/H.05/TerminalOFF/Roff)의savedgitdiff에
초기Clearance차감이반영된것을확인했다.101–200회native C5.95921/D7.91652/H.12512/
M합−1.34249,success원인snapshot7.34094%였다.201–300회C6.70647/D5.23216/H.06193/
M합−1.35254,successsnapshot3.30649%다.이는episode누적/8초로그이며step보상/누적성공률이아니다.
이근거로Cweight1.8→4.0(2.22배),a_C1유지다.동일101–200궤적환산C/D는.753→1.673으로,
새정책성과예측이나최적weight라는근거는아니다.새실행부터반영된다.

같은run의value loss는324에서1.558e10,327(22:45:44KST)에처음Inf이며invalidsnapshot도
327첫발생했다.이후구간은weight근거에서제외했고원인미확정이다.새weight로해결됐다고하지않는다.
CPU244pass/18skip/334subtest.상세는TRAINING_LOG23절과
outputs/contact_manipulation/reward_update_20261020/comparison.json에보존한다.

### 2026-10-20 — Rearrangement 활성화·Blocker 제외·선반 포함

사용자가활성화한Rweight.6/a2를유지하고active&~blocker물체의평균8corner직전step변위에
선반전체OBB8corner평균변위를더해 `0.6(exp(−2x_R)−1)`로지급한다.
Target·비가시다른물체는포함하고지정Blocker의병진/회전은모두제외한다.
정지0·왕복비용은유지하며과거누적path_length는진단용이다.현재reward9개/C4/Terminal50이다.

공식USD BBox API로실제Shelf/rack의전체bodylocalOBB를한번측정해GPU에캐시한다.
실측크기(.4085,1.01,1.8)m이며작업층크기로대체하지않는다.
SceneBatch.shelf_local_corners[N,8,3]와TaskState이력으로기존nativeworldpose를GPU변환한다.
고정episode원점을사용해큰clone좌표의회전변위상쇄를줄이고부분reset복원은과금하지않는다.
추가PhysXpose/CPU조회·관측채널·production파일은없다.선반1cm/2°종료는유지한다.
CPU248pass/18skip/374subtest,GPU1024행,실제IsaacLab4환경8step+부분reset을통과했다.
GPU변위계산핵심.280ms는전체physics/reset/PPO시간이아니다.
상세는TRAINING_LOG24절과outputs/contact_manipulation/rearrangement_20261020/에보존한다.

## 2026-10-22 — 파라미터화된 배치 생성 설계와 사용자 확정 사항

### 상태와 문제

이 절은 사용자 확정 사항과 설계·구현의 순서를 보존한다. 규칙 기반 생성 경로를 구현했으며
현재 처리 방식은 이 절 마지막의 「후속 구현」을 따른다. 아래의 초기 실패 설명과 공간 비교 계산은
설계 당시의 기록이며, 현재 기본 경로에 재시도 생성이나 외접원 구획을 사용한다는 뜻이 아니다.
사용자는 구체화 과정에서 연구 조건·배치 분포를 바꾸는 선택을 질문하며 진행하도록 요청했다.
이전 생성기는 고정한 역할·물체 구성에 대해 pair/경로를 제안하고, initial vision이 실패하면
배치를 다시 만든다. `Independent episode preparation exhausted retries: [(0,
'Initial geometric visibility/depth requirements were not met.')]`는 새 공유 장면을 준비하는
env_0의 마지막 실패 사유다. 전체32회가 모두 같은 조건에서 실패했다는 뜻은 아니다.
현재 메시지만으로 Target비율·Blocker비율·화각·픽셀수 중 원인을 구분할 수 없다.

목표는 후보를 반복 승인하는 대신 `scene = G(u; catalog, N, shelf, rules)`로 생성하는 것이다.
`u`는 난수 tensor이고, `G`가 사용할 유효 파라미터 범위를 먼저 구성한다.
물체 스폰은 최초 한 번, 이후 reset은 기존 물리 인스턴스의 pose/state를 복원하는 흐름을 유지한다.
공통 장면·초기 vision 공유와 환경별 로봇 시작 자세 randomization도 유지한다.

### 사용자 확정 사항

| 항목 | 확정 내용 |
|---|---|
| 활성 인스턴스 수 N | 현재10개. 추후 증가를 고려하여 생성 규칙에10을 고정하지 않는다. |
| 물체 종류 수 K | 현재 후보10종 중4–10종. 개수가 달라지면 K≤N이다. |
| Target | 해당 asset 종류를 장면에 정확히1개 사용한다. |
| Blocker와 주변 물체 | Target 종류와 다르게 선택하며, non-Target 종류의 중복을 허용한다. |
| 중복 균형 | 선택된 non-Target 종류별 개수 차이는 최대1개다. |
| Clutter 이웃 간격 | 초기 yaw에서 회전된 충돌 footprint 사이3–10mm. Blocker의 예약 조작 공간에는 이 상한을 강제하지 않는다. |
| 전체 밀집도 | 기존20% 점유율 하한을 강제하지 않고 현재 총10개를 우선한다. |
| Target 가시성 | 동일 자세의 가림 없는 실제 실루엣 대비 관측 면적40–90%. 최소10%는 가려진다. |
| 가림 원인 | Blocker 단독으로 제한하지 않으며 주변 물체에 의한 추가 가림도 허용한다. |
| Blocker 가시성 | 기존 최소95%를 유지한다. |
| 초기 yaw | 각 물체가 전체360° 방향을 가질 수 있도록 우선 설계한다. 임의의 좁은 구간으로 제한하지 않는다. |
| 조작 난이도 | 회전·pivot이 필요한 장면도 포함한다. 모든 장면에 단순한 옆 밀기 해법을 열어두지 않는다. |

K종을 선택한 뒤 non-Target K−1종의 기본 개수는 `q=(N−1)//(K−1)`, 추가1개를 받을
종류 수는 `r=(N−1)%(K−1)`이다. 따라서 N=10,K=4이면Target1개+나머지3종각3개다.
K=10이면각각1개다. 종류 선택과 여분 배정의 확률분포는 공간·관측 가능 영역과 함께 구체화한다.
특정 종류나 역할 조합을 제외해야 한다면 그 목록과 근거를 사용자에게 먼저 제시한다.

### 실제 크기를 이용한 사전 계산

기존 runtime 실측 자료 `outputs/contact_rearrangement/ycb_upright_contact_diagnostic.json`의
`spawn_geometries/spawn_rules`를 사용했다. 동일한10종 source asset의 upright body bounds이며
새 production stage를 이번에 측정한 것은 아니다. 구현 시 현재 합성된 stage에서 한 번 재측정한다.
자료의 선반 지지면은0.40×0.91m, 가장자리15mm를 제외한 면적은0.3256m²다.
cfg에 기재된 nominal X폭0.38m와 구분한다.10종을각각1개쓰면footprint합0.07113587m²,
전체점유율21.8476%다.

확정된 중복 균형에 해당하는 종류 구성을 열거한 면적 계산은 다음과 같다.

| N | 열거한 구성 수 | 전체 footprint 점유율 최소–최대 |
|---:|---:|---:|
| 10 | 37,270 | 13.86–34.39% |
| 15 | 33,120 | 20.62–52.69% |
| 20 | 31,590 | 27.39–70.69% |

이는 가능한 면적 합의 범위다. 실제 packing·시야·빈 조작 공간·로봇 도달성을 검증한 수치나
해당 구성의 학습 중 선택 확률이 아니다. 기존20%하한을유지하면N=10구성의약25.4%를
면적만으로제외하지만,사용자는그하한을강제하지않기로했다. 선반 전체에 분포시키는 목적은
남기되 기존 coverage/neighbor/지역별 점유율 조건을 그대로 중첩 적용할지는 별도 설계한다.

원본·계산 방법·한계는
`outputs/contact_manipulation/constructive_layout_20261022/geometry_feasibility.json`에 저장했다.

### 생성 규칙의 구체화 방향

1. 역할·종류 구성을 결정하고 실제 크기에서 필요한 공간을 산출한다. 임의 구성 생성 후
   packing 실패로 다시 뽑는 경로는 목표에 포함하지 않는다.
2. Target·Blocker·가상 시점을 결합하고 접근 영역의 초기 차단 관계를 직접 구성한다.
   Target 깊이는 구획 전체가 선반 안에 들어오는 구간에서 움직이며 뒤쪽에 고정하지 않는다.
3. Blocker가 도달할 빈 공간과 그 사이의 이동·회전 swept 영역을 먼저 예약한다.
   단순 push뿐 아니라 pivot 및 pivot 후 push를 표현할 수 있어야 한다. 이 영역은 가능한
   기하 경로의 증거이며 정책에 제공할 목표 경로나 실제 접촉 조작 성공 보증이 아니다.
4. 각 물체의 초기yaw를 먼저 전체360° 범위에서 결정하고, 그 자세의 footprint로 필요한 공간을
   계산한다. 인접 물체의 크기와 작은 간격에서 위치를 직접 결정하며 물체별 큰 jitter 구획을
   선반 전체에 균등 분배하지 않는다. 특정 N·구성·yaw의 전체 수용 가능성은 별도 확인이 필요하다.
5. 최종 ray casting은 실제 관측 정보를 만드는 단계로 사용한다. 재시도 기반 승인 검색과
   분리하되, 실제 메시 가시성의 보장 방법을 먼저 정해야 한다.

주변 물체의 추가 가림을 허용하므로 Target의 보이는 영역만 확보해서는 충분하지 않다.
40%이상 남길 시야와10%이상 가릴 영역을 동시에 구성해야 한다. 가림은실제메시실루엣으로
평가하며 OBB의 투영 겹침 비율로 대체하지 않는다. Blocker95%·전체실루엣화각포함·최소픽셀수도
함께 고려한다. 일부 자세 표본이 통과했다는 이유로 연속 파라미터 구간 전체가 보장된다고 하지 않는다.

### 전체360° yaw를 위한 평면 구획의 1차 공간 계산 — 보수적 비교안

**아래 비교안은 초기 yaw에서의 collision footprint를 사용하는 후속 구현으로 대체했다.**
아래 외접원 구획은 초기 공간 비용을 비교한 계산이다. 사용자의후속clutter요구에따라
최종밀집생성방식으로채택하지않는다. 전체360°초기yaw허용과모든물체주변에미래회전공간을
예약하는것은다르다. 이비교안의폭부족을실제수용불가능이나역할제외근거로사용하지않는다.

위 실측 형상의 외접원 직경은약95.18–179.06mm다. 실측usable깊이370mm에서
가장큰물체2개의직경과3mm간격합은361.12mm여서앞뒤2행을구성할수있다.
Target/Blocker를같은열에두고옆열의앞칸을Blocker이동공간으로예약하는6열구성에서는
10종을각각1개사용할때90개모든역할쌍이폭757.02–834.13mm로들어간다(가용폭880mm).
남는여유는위치jitter·접근영역과의상대오프셋에배분해야하며무제한이동범위가아니다.
회전중심은OBB중심으로계산하고중심이USDroot와다른경우root위치를보정해야한다.

이는평면의보수적크기계산일뿐메시가시성·로봇도달성·접촉조작을검증한것은아니다.
특히일부균형중복구성은이6열구획에들어가지않으며,N증가시고정2행은금방제약이된다.
이는물리적으로배치불가능하다는증명이아니므로해당구성을임의로제외하지않는다.
행수·열수·구획폭을N과실제물체크기에서정하는규칙을구체화해야한다.
이옆칸예약안은순수밀기경로를확보하는계산이므로,사용자가추가로확정한회전필요장면의
최종생성규칙으로그대로채택할수없다.

재현용CPU분석은`outputs/contact_manipulation/constructive_layout_20261022/`의
`full_rotation_planar_feasibility.py/.json`에보존했다. N=10의균형구성·역할종류사례213,750개중
4,885개가이보수적구획에들어가지않았다. 이비율은실제샘플링실패확률이아니다.
구획폭을최소화하는pairing계산은독립완전탐색160사례와최대오차1.11e−16m로일치했다.
행렬·모델·PhysX를실행한성능측정은아니며production코드는변경하지않았다.

회전경로를하나예약했다고해서회전이필요한장면이되는것은아니다. 초기yaw를유지한채
다른방향으로통로를비우는해법이남는지별도로다뤄야한다. 원형에가까운물체에서회전이
footprint를바꾸지않는경우도구분한다. 회전필요구성의비율과그필요성을판정하는정확한기준은
아직확정하지않았으며,구체적기하구조와함께사용자에게제시한다.

### Clutter를 위한 수정 — 초기 yaw에 맞춘 인접 배치

사용자가물체끼리너무떨어지는설계를우려하며clutter를다시강조했다. 외접원고정구획은
Cracker의실측XY크기164.04×71.80mm에179.06mm폭을양방향으로예약하는등과도한빈공간을
만든다. 따라서주변물체의회전여유는일괄예약하지않고Blocker의필요조작공간만별도로확보한다.

직사각형footprint의회전된반폭은
`h_x=(s_x*abs(cos(yaw))+s_y*abs(sin(yaw)))/2`,
`h_y=(s_x*abs(sin(yaw))+s_y*abs(cos(yaw)))/2`로계산한다.
동일행의이웃을`x_next=x+h_x+h_x_next+g`로생성하면임의위치후충돌재시도없이
선택한yaw에서투영상자비중첩을구성할수있다. 행별prefixsum과batch연산으로처리가능하다.
다만이식의g는투영상자간격이며실제메시표면간최단거리나원하는clutter를보장하지않는다.
최종간격기준은회전된충돌footprint로명시하고실측가까움과평면구획계산을구분해야한다.

선반전역분포와국소clutter도별개다. 총10개·작은물체위주의구성에서는전체선반을고밀도로
채울수없으므로물체들을넓게균등분산해개별물체주변여유를늘리는방향으로해결하지않는다.
인접관계를가깝게구성하면서조작영역과선반내분포를함께설계한다. 전체점유율20%하한을
다시강제하거나사용자가확정한N을임의로늘리지않는다.
사용자는이웃간초기간격을3–10mm로확정했다. 생성식의간격파라미터유효범위는
`g∈[0.003,0.010]m`다. 이범위는가까이배치하도록선택한이웃관계에적용하며모든물체쌍의
거리가10mm이하여야한다는뜻이아니다. 거리기준은초기yaw에서회전된충돌footprint이고
AABB간격3–10mm를실제footprint거리3–10mm로대체해주장하지않는다.
Blocker의이동·pivot을위한예약공간은별도로유지한다. 이 설계 당시에는 production에 미적용했고,
후속 구현에서 초기 yaw의 collision footprint와 해석적 compaction으로 연결했다.

### 개수 확장과 분리할 용량

- 종류 catalog 크기 K_catalog, 활성 개수 N, 스폰된 후보 인스턴스 수 P, 관측 슬롯 용량 M을
  서로 구분한다. Target/Blocker는 활성 슬롯0/1로 유지하고 candidate pool의 global ID와 분리한다.
- 이전 `OBJECT_COPIES_PER_ASSET=1`을 종류별 `ceil((N−1)/3)`개로 바꿨다.
  기본 N=10, 최소K=4에서 종류별3개, P=30을 최초 생성하고 미사용 물체는 기존 native
  parking으로 분리한다. Reset에서 prim이나 PhysX body를 새로 만들지 않는다.
- `CONTACT_MANIPULATION_NUM_OBJECTS`를 topology 구성 전에 설정하며 기본값은10이다.
  Scene pool·`MdpCfg`·reset/vision slot·모델 설정에 같은 N을 사용한다. Pool global ID를
  관측 slot으로 압축하며 비가시 물체는 padding한다. 물체 feature 폭10과 이력2×72는 유지해
  observation 폭은 `10N+144`다. 용량 확장과 모든 구성의 기하 수용 가능성은 별도 문제다.
- GNN은집계방식상노드수변화를표현할수있지만,공개관측폭과독립MLPCritic입력은M에따라
  달라진다. 기존checkpoint의호환성을자동으로보장하지않으며N증가시새실험설정으로관리한다.

### 다음 검토와 확인 항목

전체360°yaw를수용하는구획의실제공간비용과역할별관측가능영역을먼저계산한다.
구획분할만으로불가능한구성이나역할이확인되면원인과대안을제시하고사용자확인을받는다.
고정구획재생이나확인되지않은jitter를새장면생성의보증으로사용하지않는다.
사용자는회전·pivot이필요한장면도포함하도록확정했다. 그비율이나일부역할조합의제외등
추가적인난이도·분포선택은구체적공간계산후확인한다. 기존20%하한을제거하는결정과예약공간을밀집도분모에서
빼는결정은다르며후자는확정하지않았다. 전체점유율의분모는기존대로유지한다.

### 후속 구현 — 규칙 기반 장면과 고정 시점 영역

기본 `SpawnRules.generation_mode="constructive"`는 공유 scene seed당 물체 구성을 한 번만
정한다. N·K와 균형 개수, Target/Blocker, 각 초기 yaw를 선택한 뒤 그 자세의 충돌 footprint에서
작업 구획과 주변 물체 영역을 산출한다. 물체를 임의 위치에 놓고 실패한 전체 장면을 다시 뽑는
반복은 없다. 이전 sampler는 명시적인 `generation_mode="legacy"` 비교 경로로만 남긴다.

Target 앞의 Blocker와 pivot→병진 경로의 좌표 extrema로 작업 구획을 예약한다. ±Y 두 방향의
작업 구획을 먼저 모두 계산하고 수용 가능한 영역의 합집합에서 하나를 선택한다. 한 방향의
배치를 실제로 만든 뒤 실패하면 반대 방향으로 재생성하는 흐름이 아니다. 주변 물체는
이 구획 양옆에서 slicing dynamic programming으로 수용 가능한 구획 형태를 계산한다.
이는 제한된 종류의 직사각형 분할을 계산하는 유한 과정이며 임의 배치의 반복 승인이 아니다.
넓이·깊이의 Pareto frontier에 임의 후보 수 제한을 두지 않는다. 이후 두 방향에서 각각 한 번,
configuration-space 경계를 해석적으로 계산해 물체를 가까이 옮긴다. 이미 옮긴 물체와 아직
옮기지 않은 물체를 모두 장애물로 반영하므로 중간 이동이 다음 물체를 침범하지 않는다.

충돌 footprint는 합성된 USD의 활성 authored collider에서 한 번 측정한 upright XY convex hull이다.
미래의 전체 회전 여유를 물체마다 확보하지 않는다. 가까운 이웃의 목표 간격은3–10mm이고,
모든 물체 쌍을10mm 이내로 만드는 조건은 아니다. Concave/compound collider의 빈 공간은
convex hull이 메우므로 정확한 PhysX 접촉 표면과 같은 형상으로 취급하지 않는다.
Minkowski 경계는 정점 edge-angle 병합으로 계산하고, 다각형 거리는 compiled CPU kernel을 쓴다.
이 구획 계산은 모든 환경이 공유하는 장면 하나에만 수행하고 per-environment physics step에는 없다.

추가로 기존 주변 물체 중1–2개의 위치를 해석적으로 결정할 수 있으면 lateral guard로 사용한다.
고정한 초기 X·yaw에서 순수±Y 양쪽 탈출이 모두 막히고 pivot→병진 witness는 유지되는 유한
영역을 계산한다. 새로운 물체를 추가하거나 통과할 때까지 배치를 다시 뽑지 않는다.
해당 영역이 있으면 `clearance_kind="guarded_pivot_translate"` mode와 일반 `pivot_translate`
mode를 함께 준비한다. 영역이 없으면 일반 mode만 남긴다. 이후 실제 메시 가시성까지 만족하는
mode·시점 쌍에서 guarded를 우선 선택한다. 이는 일반 XY 병진 해법까지 불가능하다는 판정이 아니다.
Guard는 정점 원호의 해석적 support를128개 고정 법선으로 감싼 보수적 이동 envelope 밖에 둔다.
이때 앞서 packing에 쓴 큰 직사각형에는 들어갈 수 있으므로 `reserved_regions=()`로 표시한다.
빈 metadata가 조작 경로를 예약하지 않았다는 뜻은 아니며, 일반 branch의 직사각형 reserve와
guard branch의 실제 이동 envelope를 구분한다. 생성 비율과 실제 메시 검증 수치는 별도 기록한다.

실제 시각 메시의 GPU ray casting에는 각 기하 mode에 선반 앞의 고정 `2×9×2=36`개 시점과
배치 규칙에서 계산한 시점1개를 연결한다. Base/guarded 두 mode가 있으면 최대74쌍의 고정
공동 영역이다. 처음부터 전 영역을 고정 메모리 chunk로 모두 평가하고 유효한 guarded mode를
우선 선택하며, 그 mode에 유효 쌍이 없으면 이미 평가한 base mode에서 고른다. 한 mode 안에서는
Target 가시율이65%에 가장 가까운 시점, Blocker 가시율, index 순서로 선택한다.
이는 실패 후 다른 장면을 생성하는 fallback이나 적응적 재시도가 아니다. 선택한 mode의 물체 pose·
충돌 기하·초기 관측을 함께 commit한다. 광선·가시율·mode별 시점 선택·관측 표면 OBB는 CUDA에서
계산하고, 최대2개 mode의 우선순위 적용과 결과 metadata 조립은 CPU에서 수행한다.

유효성은 같은 자세의 비가림 실루엣 광선 수 대비 Target40–90%, Blocker≥95%, 두 역할의
전체 화각 포함, 비가림32픽셀 이상·관측16픽셀 이상으로 계산한다. 역할 외 물체도 가림에 포함하고
로봇·선반은 기존 object-only 관측 범위대로 제외한다. OBB 투영 비율로 메시 가시성을 대신하지 않는다.
실패 시 조건별 비율·픽셀 수·실패 항목을 보고하며 같은 seed의 배치를 조용히 다시 뽑지 않는다.
이 구조가 모든 역할·yaw·종류 구성에서 유효 시점을 보장하거나, witness에 회전이 있다는 것만으로
회전이 반드시 필요한 장면을 보장하지는 않는다. 해당 보장과 실제 메시 검증 결과는 별도로 관리한다.

공통8초 scene round와 부분 reset의 초기 배치·vision 재사용은 유지한다. 각 환경의 hand 시작
위치·방향, GPU IK·elbow-up·충돌 검사, native state tensor write, 센서·MDP·PPO 규약은 유지한다.
자세 randomization의 IK 후보 검사는 배치 재생성 루프와 별개다. 세부 구현과 최종 검증 기록은
[현행 배치 규칙](SPAWN_RULES.md)과 [학습 기록](TRAINING_LOG.md)을 함께 확인한다.

미사용 후보의 parking은 기존 native permanent wrench composer의 COM 중력 상쇄를 유지한다.
설치된 `RigidBodyView.get/set_disable_gravities`를 확인했지만, 위 2026-10-09 기록처럼 direct GPU에서
getter가1이어도 실제 물체가 낙하한 근거가 있으므로 flag 전환 방식은 채택하지 않았다.
API 존재나 mock 호출 검사만으로 실제 중력 적용이 검증됐다고 판단하지 않는다. 후보 pool을30개로
늘린 물리 비용과 배치 계산 최적화는 구분해서 측정해야 한다.

### 후속 검증 — 2026-10-22 공동 영역 버전

CPU MDP249개 실행·16개 skip, learning56개 실행·2개 skip에서 실패가 없었다.
실제 collider hull의 raw seed1,024개에서는 빈 생성 영역이 없었고 생성 평균75.215ms,
p95137.187ms였다. Cold/concurrent outlier를 포함한 최대3.704초와 전체 iteration 시간은 구분한다.
최종 실제 메시 검사는256장면·11,470개의 mode·시점 쌍을 평가해256개 모두 통과했다.
54개 geometric guarded/base 장면 중24개가 guarded를 선택하고30개는 함께 평가한 base를
선택했으며 나머지202개는 singleton이었다. Target43.650794–79.436153%, Blocker95.454544–100%,
두 역할의 관측 OBB가 모두 유효했다. 생성22.891초·공동 영역 처리4.427초는256장면 합계이며
후자에는 state packing·OBB·host metadata가 포함된다. 순수 ray kernel이나 학습 iteration 시간이 아니다.
Guard 확장 전4환경·2iteration smoke와 최신1,024환경·26iteration 검사는 구분한다.
공동 영역 버전의 첫 native 실행은iteration0–21의22회 update 뒤800step scene 교체에서 로봇의 random domain
probe `2/8<4`로 중단됐다. 물체 기하·가시성 실패는 아니었다. 소표본의 단일 목표·IK 판정은
유효 시작 영역의 부재를 증명하지 않으므로 이4/8 추가 gate를 구성적 경로에서 제외했다.
검증된 anchor·실제 PhysX FK/elbow/충돌 검사·연속적 GPU IK randomization·실패 행 복구는 유지한다.
수정 후 `2026-10-22_13-41-29_constructive_verified_anchor_smoke`는1,024환경·26iteration,
936개 정책 step과800step 공통 장면 교체를 통과하고 exit0으로 종료했다. Compile 이후 평균3.3208초,
중앙값3.31초이며 전체 학습 구간105.05초다. Anchor fallback·좁은 영역 recovery·invalid-state 종료는0,
`model_25.pt`의 Actor37개·Critic8개·optimizer135개 텐서는 모두 finite였다.
본 학습 성능 비교에 포함하지 않는 실행 검증이며 장기 수렴을 입증하지 않는다.
상세와 원본 링크는
[TRAINING_LOG.md 42절](TRAINING_LOG.md#42-2026-10-22--규칙-기반-밀집-배치와-고정-기하시점-공동-영역)에 둔다.

## 2026-10-22 — 요구 재확인: 후보 선택 없는 순차 생성

사용자는 "하나의 규칙으로 순차적으로 생성된 scene이 당연히 설정한 조건을 만족"해야 한다고
재확인했다. 이는 유한 후보를 모두 검사하는37/74개 공동 영역 방식보다 강한 요구다.
현재 production의 `generation_mode="constructive"`라는 이름은 이 요구를 충족했다는 뜻이 아니다.
앞 절의 통과 수치는 현재 후보 검색 구현의 검증이며, 아래 새 생성기의 검증으로 해석하지 않는다.

사용자는 **학습 시작 전에 유효한 연속 파라미터 범위를 사전 계산·캐시하고 reset에서 직접
샘플링하는 방식**을 선택했다. 완성된 scene을 저장해 재생하는 bank는 승인한 방식이 아니다.

새 생성기의 실행 계약은 다음과 같다.

1. 사전 계산한 비어 있지 않은 유효 공동 영역에서 파라미터를 한 번 샘플링한다.
2. 정해진 순서로 역할·카메라·조작 공간·주변 물체를 직접 구성한다. 앞 단계가 다음 단계의
   허용 범위를 정하며, 종류·yaw·시점의 모든 조합을 서로 독립적으로 추출한다고 가정하지 않는다.
3. 물체를 배치한 뒤 승인/거절하거나 다른 base/guarded/view를 골라 교체하지 않는다.
4. 최종 실제 메시 ray casting은 observation 생성에만 쓴다. 생성기의 조건 검사는 개발·검증 경로로
   분리하며, 계약 위반을 새 후보로 숨기는 재시도는 두지 않는다.

실제 메시 가시성의 사전 계산에는 Blocker의 한 축 이동량을 매개변수로 하는 역문제를 검토한다.
Camera·Target·Blocker yaw가 고정되면, Target ray와 이동하는 triangle의 교차 depth·barycentric
좌표가 이동량의 일차함수다. 선형 부등식으로 triangle별 가림 구간을 구하고 pixel별 합집합 뒤
구간 끝점을 sweep하면 Target40–90%의 유효 이동 구간을 직접 얻는다. 후보 pose 검사나 이분 탐색과
다르며 cup 손잡이 등으로 가림이 단조롭지 않아도 여러 구간의 합집합으로 표현한다.

합성 두-bar mesh의 CPU prototype에서441개 Target ray·24개 triangle로 이 원리를 확인했다.
계산한 유효 구간에서 추출한1,000개 값이 독립 Möller–Trumbore 검사에서 모두40–90%를 만족했고,
추가513개 점의 구간 분류 비교에도 불일치가 없었다. 이1,513회 검사는 prototype의 회귀 검증이며
생성 알고리즘에서 후보를 승인하는 과정이 아니다. 원본은
`outputs/contact_manipulation/constructive_layout_20261022/inverse_visibility_prototype.py/.json`이다.

이는 **실제 YCB용 캐시나 새 production 생성기를 완성한 결과가 아니다.** Blocker95%·화각·최소
pixel 수·물리 간격·선반 수용·pivot 경로를 동시에 만족하는 비어 있지 않은 영역의 구성은 남아 있다.
주변 물체는 Blocker와 보장할 Target pixel의 관측 cone 밖에서 고정된 chain으로 생성해야 한다.
소수 샘플 통과나 보간을 연속 영역의 보장으로 바꾸어 부르지 않는다. 전체360° yaw 등의 기존
조건을 바꾸거나 특정 역할·구성을 제외할 필요가 확인되면 근거를 제시하고 사용자 확인을 받는다.

사용자에게 종류·yaw·camera의 연동 샘플링을 질문했으며, 답변은 **"그 샘플링 규칙에 대한 설명이
필요함"**이었다. 따라서 연동 분포 자체가 승인됐다고 기록하지 않는다. 먼저 역할/구성 선택,
유효 관계 영역 선택, 연속 변수 샘플링, 주변 chain 직접 계산의 순서와 각 단계의 분포를 설명한다.

검토 중인 연속 관계의 한 예는 Target·Blocker·camera를 하나의 역할 assembly로 보고 함께
수평 회전/평행이동하는 것이다. 이 rigid transform은 camera 기준의 실제 메시 ray 관계를 보존하므로
사전 계산한 Blocker 이동 구간의 가시율을 그대로 유지한다. 다만 world shelf·접근 방향·camera의
선반 정면 조건을 만족하는 회전/이동 범위는 별도로 제한해야 한다. 여러 관계 영역의 yaw 합집합이
전체360°를 덮는지 확인하기 전에는 yaw coverage를 보장했다고 하지 않는다. 이는 제안 단계다.

실제 메시 역문제의 첫 확인은 Target `mug_2`, Blocker `ycb_004_sugar_box`의 고정 yaw·camera에서
수행했다.93개 Target pixel과16,384개 Blocker triangle의 affine 교차를 GPU에서 계산했고
solver 안의 후보 pose query는0회다. 가시성 및 shelf Y 지지 조건에 해당하는 Blocker +Y 이동량은
`[0.01558769,0.05082184] ∪ [0.11475542,0.15068152]m`였다. 구간 길이에 비례한 CDF로 뽑은
100개 값은 독립 Warp query와 pixel 예측 불일치0건이었다. Target43.01–82.80%, Blocker100%가
측정됐다. Affine 계산55.23ms·구간 합집합/sweep26.60ms는 한 조합의 사전 계산이며 전체 캐시
구축 비용·reset 시간으로 일반화하지 않는다.

원본은 `outputs/contact_manipulation/constructive_layout_20261022/actual_inverse_visibility_probe.py/.json`이다.
이 구간은 전체 scene의 유효 범위가 아니다. 초기 접근 영역의 차단율·조작 경로·주변 물체와
Blocker95%의 연속 영역 보장, 모든 yaw·시점 조합은 아직 포함하지 않는다. 실제100개 위치의
검사는 새 캐시 원리의 독립 검증이며 reset에서 승인 후보를 찾는 루프로 사용하지 않는다.

고정 chain의 공간을 보장하는 제안으로, scene마다 주변 물체 slot을 골라 yaw를 전체360°에서
뽑고 나머지는 각 형상의 작은 Y폭을 만드는 yaw 주변의 허용 구간에서 뽑는 방식을 검토했다.
모든 slot이 자유 slot으로 선택될 양의 확률을 가지면 각 물체의360° support를 유지할 수 있지만,
각 물체의 yaw 주변 확률은 균등하지 않다. 사용자는 후속 질문에 **"모든 방향이 가능하면 되고,
각도 빈도는 균등하지 않아도 됨"**으로 답했다. 따라서 자유 slot을 바꾸어 가며 나머지 yaw를
공간과 연동하는 분포를 사용할 수 있다. 각 역할/slot의 전체360° support는 여전히 검증 대상이다.
전체360° support와 균등 marginal, 물체 간 독립성은 서로 다른 요구이므로 혼동하지 않는다.

추가 확인에서 사용자는 **"각 종류 조합에서도 두 역할 모두360° 가능해야 함"**을 선택했다.
따라서 각 ordered Target/Blocker asset pair마다 Target yaw와 Blocker yaw의 주변 support가 각각
전체 원을 덮어야 한다. 전체 catalog를 합쳤을 때만360°가 되는 분포는 허용되지 않는다.
각 조합 안에서 두 yaw·camera·위치 사이의 상관관계와 비균등 빈도는 허용된다. 두 yaw의 모든
독립 조합을 보장한다는 의미는 아니다. 새 캐시의 승인 기준은 각 ordered pair의 연속 yaw 구간
합집합에 빈 구간이 없는지 확인하는 것이다. 소수 각도의 표본 통과로 이 기준을 대신하지 않는다.

추가 확정 사항:

- 간격은3–10mm로 고정하지 않는다. 사용자는 **"모든 물체는 넓은 간격도 허용되지만, 짧은 간격도
  허용되어야 함"**이라고 설명했다. 짧은 간격을 선호하면서 남는 공간 안에서 넓은 간격도 연속적으로
  지원하는 `spread_gaps`를 새 직접 생성 경로에서 사용한다. 지지/시야/조작 공간 인증은 유지한다.
- 사용자는 **"같은 주변 구성에서도 두 역할 모두360° 가능해야 함"**을 선택했다. 주변 물체의
  종류·개수를 특정 각도에 유리하게 바꾸어 역할 yaw coverage를 채우지 않는다. 모든 주변 구성에
  유효한 공간 예산 또는 해당 ordered pair의 최악 예산으로 인증한다.
- 현재 부분 catalog 검사에서10개 ordered pair 중9개는 연속360°를 지원했으나,
  `bottle_1 → ycb_003_cracker_box`의 Blocker 지원은 초기90°였다. Camera lateral±8cm 보완 후에도
  100°에 그쳤다. 이는 선택한 상대 관계/보수적인 공간 인증의 한계이며 해당 각도 자체가 불가능하다는
  증거가 아니다. 임계값 완화·각도 제외·주변 구성 변경으로 처리하지 않는다.
- `generation_mode="parameterized"`의 startup catalog와 reset/관측 연결은 개발 중이며,
  전체360°와 pivot subset 인증이 불충분하면 시작 단계에서 거절한다. 기본값은 여전히 기존
  `"constructive"`다. 단일 family의 실제10물체/GPU 검증과 전체 catalog/1024 reset 검증을 구분한다.

후속 보완에서 `bottle_1 → cracker_box`의 초기 A0 조건 자체는 누락된27개 중심각 모두에서
공통 회전0°로 성립했다. 이전의90°/100° coverage는 불가능성의 증거가 아니라 넓은 공통 회전 범위와
순수Y 이동 경로를 함께 묶은 충분조건의 한계였다. 초기 배치는 유지하고 pivot 후 이동의 연속
경로 bounds를 사용한 결과, 기본45 family와 추가133 family의 합집합이 해당 pair의 두 역할에서
각각360°를 덮었다. 주변 구성은 바꾸지 않았고 모든 허용 구성의 최악 공간 예산0.389202927m를
사용했다. 이는 첫 문제 pair의 검증이며 전체90 pair 완료를 뜻하지 않는다.

새 경로의 첫1,024 reset smoke는 초기 anchor IK에서 실패했다. 같은 실패 장면을1환경에서
재현했을 때 기존80개 후보 중26개가 IK에 수렴했으나 elbow-up과 forearm 방향 조건을 동시에
통과한 것은0개였다. 외부 물체·선반 충돌이 원인은 아니었다. 같은 영역의 고정 grid를 진단한
결과58,212개 후보 중2,704개가 모든 native 조건을 통과했다. 따라서 장면을 다시 생성하거나
충돌/팔 자세 기준을 풀지 않고, 초기 anchor의 pose·IK branch 구성을 보완한다. Anchor 주변729개
표본 통과는 국소 수치 검증이며 연속 전체의 수학적 보증으로 기록하지 않는다.

초기 template의 pose/branch를 고정 strata로 덮는 보완 후 같은 family의1,024환경 통합 reset과
부분 reset, 다른32장면의 native producer가 통과했다. 최종 환경들의 robot start는 연속적으로
randomize하며 장면을 재생성하지 않는다. 다만 zero-action 첫 step의2개 singularity 종료를 확인해,
초기 pose에도 기존 활성 singularity 기준을 적용하는 추가 보완 중이다. 전체90 pair catalog는
별도로 구축/검증 중이며 기본 생성 모드는 아직 바꾸지 않았다.

후속 전체 catalog 검증은 완료됐다.90 ordered pair 모두에서 두 역할 각각의 연속 yaw 합집합이
360°이고, pair별 모든 허용 주변 구성의 최악 공간 예산을 유지한다. 총6,593 family·207,027 cell,
normal-pivot101개·pivot-required guarded1개다. Cold 구축은 약33.68분이며90 checkpoint 모두
checksum reload를 통과했다. 직접 생성한360장면과 guarded 끝점4장면의 독립 기하/실제 메시
ray 검사도 모두 통과했다. Target 실측41.279–87.952%, Blocker100%다.
일부 pixel count에1–2pixel 오차가 있었으므로 연속 기하 인증과 float32 표본 검증을 구분한다.
이는 robot-start의 전체 종류 검증/실제 학습 테스트를 대신하지 않으며 기본 전환은 해당 검증 뒤다.

환경 원점의 world 변환 상쇄가 float64 hull/일부 float32 mesh hash를 바꿀 수 있어
`ComputeRelativeTransform(prim, env_0)` 기반 추출로 보완했다. Body-local 연산 순서를 새로
정의하거나 hash를 반올림하지 않고 기존 rigid/stretch 계산을 보존한다. 실제10종 asset의
동일 archived root에 대한 CPU 검사에서는 원점0의 기존 결과와 일치하고±46.5m에서도 불변이었다.
Shelf는 `ComputeUntransformedBound`에 상대 변환을 합성한다. 현재 native1/128/1024환경 간
source 일치 및 캐시 재사용은 별도로 검증한다. Archive가 현재 compiler의 모든 mesh hash와
일치한다는 뜻은 아니다.

Native 대조에서 남은 원점 의존성은 bottle/can의 `ComputeUntransformedBound(body)`에도 있었다.
환경 ancestor를 제외한 composed body subtree를 임시 in-memory USD stage에 reference하고,
기존 BBox/stretch 계산을 그대로 수행하도록 수정했다. Live stage나 물리 형상을 수정하지 않는다.
이후1·128·1,024환경 source fingerprint가 모두 기존
`b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`과 일치했다.
완성 캐시는 기본 경로 `outputs/contact_manipulation/parameter_domains`에도 배치했다.

90 ordered pair에서 직접 생성한90장면의 초기 anchor는 재생성/제외 없이 native IK/FK와
기존 singularity 조건을 모두 통과했다(최소 native margin0.0146109).
1,024환경 비교에서는 초기 자세용 Jacobian의 기준점 오류를 추가로 찾았다. Native Jacobian의
선속도 기준은 link COM인데 초기 helper가 link 원점을 사용했다. 현재 wrist_3의 COM은 local
`(0,0,−0.02289999835)m`이며 COM으로 옮기자 native와의 Jacobian 최대 오차가 약8.05e−7로 줄었다.
초기 feasibility만 동일 COM 기준으로 수정하고 공유 metric·기존 threshold·공식 OSC는 유지한다.
Native COM은 최초1회 snapshot 후 캐시하며 후보/매 step마다 CPU 조회하지 않는다.

후속 1,024환경 smoke에서 두 전체 reset 모두 초기 singularity 판정 실패는 0개였다.
Native 최소 margin은 각각 0.005014884와 0.054135174로 기존 임계값 0.005를 통과했다.
첫 물리 step 뒤 발생한 singularity 종료 1개는 유효한 초기 자세에서 경계 아래로 이동한 경우이며,
초기 invalid state와 구분한다. 나머지 실패 원인은 0개였다. 부분 reset은 다른 1,019행을 그대로
유지하고 추가 장면 생성·관측·ray query 없이 완료했으며, 별도 32장면의 native anchor도 통과했다.
첫 reset 1.960초, 새 graph bucket의 5행 reset 0.265초, warm 전체 reset 0.117초였으며
이는 reset 측정값으로 PPO iteration 시간과 다르다. 검증 자료는
`parameterized_reset_1024_singularity_probe.json`이다.

이 검증 뒤 실제 `ShelfManipulationEnvCfg`의 기본 생성 모드를 `parameterized`로 전환했다.
별도 mode/cache override 없이 공식 학습 진입점으로 26 iteration·936 policy step을 완료했다.
실제 800-step 경계에서 장면이 교체됐고 두 전체 reset의 초기 native singularity 위반은 모두 0개,
fallback은 0회였다. Actor·Critic·optimizer 값도 모두 유한했다. 기본 cache 재사용 초기화는
8.754초, cold capture를 포함한 학습 구간은 108.254초였으며 장기 학습 성능과 구분한다.
로그는 `2026-10-22_17-19-09_parameterized_default_smoke`, 증거는
`parameterized_train_native_probe.json`이다. 상세 생성 규칙은 `docs/SPAWN_RULES.md`에 정리했다.
첫 iteration을 제외한 25 iteration은 평균 3.421초·중앙값 3.35초였고 평균 collection 2.899초,
learning 0.522초였다. `invalid_state`는 전 구간 0, 최종 checkpoint는 `model_25.pt`다.
짧은 시험의 success는 0이므로 정책 성능 개선과 생성·학습 경로의 실행 검증을 구분한다.

## 2026-10-22 — 통합 clutter와 물리·관측 용량 분리

사용자는 Target·Blocker와 나머지 물체가 별도 그룹으로 배치되는 느낌을 지적했다. 역할 관계를
만든 뒤 주변 물체를 잔여 두-row 영역에만 넣는 규칙은 충분히 랜덤한 clutter가 아니라는 피드백이다.
새 목표는 전체 물체의 위치·yaw가 하나의 clutter를 이루면서 역할 물체의 가시성·조작 조건을
만족하는 것이다. 이전90 ordered pair는10종에서 서로 다른 두 역할을 고르는 `10×9` 종류 조합이며,
미리 정해진90개의 완성 장면이 아니다. 각 조합의 연속 파라미터 범위를 사용하는 원칙은 유지한다.

물체40개를 검토한 뒤 사용자는 **같은 선반에서 활성20–40개를 공간 수용 가능성에 연동**하도록
확인했다. 기존4–10종, Target 종류1개, non-Target 종류별 중복 개수 차이≤1을 유지한다. 개수만
채우려고 큰 종류를 제외하거나 구성·선반 크기를 몰래 바꾸지 않는다. 같은 주변 종류·개수 구성의
각 ordered pair에서 두 역할 각각360° 가능성과 실제 메시 가시성 조건도 유지할 요구다.

공개 observation은 Target·Blocker를 포함한 **10개 slot과244D**로 고정한다. Slot0/1은 기존 역할
순서를 유지하고, 인식한 나머지 물체가8개를 넘으면 관측 OBB 중심에서 두 역할의 관측 중심까지
유클리드 거리 중 작은 값을 기준으로 가까운8개를 고른다. 보이지 않은 물체를 정답 pose로 채우지
않고 부족한 slot은 padding한다. 초기 vision만 제공하고 현재 EEF frame으로 변환하는 계약은 같다.
Reward·termination·물리 충돌은 선택된 관측10개가 아니라 전체 활성 물체를 계속 고려해야 한다.

`scene.py`의 `CONTACT_MANIPULATION_NUM_OBJECTS` 기본값은 물리 최대40으로 변경했다. 종류 선택을
episode마다 바꾸면서 최소4종의 균형 중복을 허용하려면 종류별 `ceil(39/3)=13`개, 총130개의
재사용 rigid body가 필요하다. 이는 이전30개보다1,024환경에서102,400개 추가 body다. 공식
`MultiAssetSpawner`는 초기 asset 선택 기능이므로 topology를 유지한 episode별 종류 변경을
대신하지 않는다. 미사용 body는 기존 주차와 native `permanent_wrench_composer`로 관리한다.
이전 direct-GPU 검사에서 gravity flag만 바꾸어서는 낙하가 멈추지 않아 검증된 composer를 유지한다.

이 절은 변경 요구와 구현 진행 기록이다. 기존10물체의364장면·1,024환경26 iteration 결과는
새20–40물체 배치나130-body pool의 실행·성능 검증이 아니다. 새 구성의 공간 수용 범위, 통합 배치,
고정10-slot 선택과 대규모 실행 결과는 구현 후 별도로 기록한다.

후속 확인에서 사용자는 **이동·pivot 전체 경로를 비워 두지 않고 빈 재배치 목적 영역만
확보**하는 방향을 승인했다. 주변 물체가 과거 full-path envelope 안에 놓일 수 있으므로 기존
collision-free witness를 새 scene의 보증으로 주장하지 않는다. 빈 목적지는 실제로 연결된
조작 경로의 존재와 다르며 접촉·pivot·여러 단계 조작은 정책이 학습할 대상이다.

유효한 seed의 주변 위치를 섞는 `_mix_clutter_positions`는 고정3회 sweep으로 물체별 X·Y의
정확한 free-interval union에서 직접 샘플링한다. Minkowski 경계는 최초1회 준비하고 Numba에서
유클리드 gap disk와의 교차·구간 차집합을 계산한다. 후보 위치의 승인·재생성이나 수렴 검사는
없고 yaw·종류 구성은 유지한다. 회귀6개 및 기하140장면·해석식171개 비교는 통과했으며,
40물체 CPU 비용은4/64/500정점에서 약5/14/79ms였다. 이 독립 helper 검증은 새20–40물체
생성기의 전체 가시성·공간 용량·native 학습 검증을 대신하지 않는다.

130-body1환경 source-only native 검사에서는 실제 geometry/resources 초기화와
`_certified_catalog_source(reference_count=10)`만 실행하고 episode 생성·reset·캐시 재계산을
차단했다. 최초에는 pool 확대가 각 종류 첫 body의 주차 위치를 바꾸어 mug bounds·cracker
collision 값이 약1e−16 달라졌다. 첫3개 복사본은 기존3-copy catalog의 주차 slot에 유지하고
나머지는 뒤로 배정하는 순열로 수정했다. 130개의 주차 위치 집합과 reset state writer는 같고
hash 반올림이나 geometry 근사는 없다. 1/2/3/4/13/33-copy 용량의 CPU 유일성·기준 위치 검사도
통과했다.

수정 후130-body native source가 기존 reference10 catalog의 `b29c25582cca0b352970065b46ecd3f341f56c72159c6ad94f0abdaa71c10e08`과
정확히 일치했다. 물리 ID 배열은 `[1,40]`, 관측 geometry는 `[1,10,10]`이며166.99MB 역할 cache의
source·payload checksum을 읽기 전용으로 검증·로드했다(7.961초). Scene 초기화23.492초,
geometry/resources2.138초였다. 이 검사는 새20–40물체 장면 생성·가시성·학습 시험이 아니며
역할 cache의 재사용이 새 전체 packing 용량을 인증한다는 의미도 아니다. 자료는
`outputs/contact_manipulation/clutter_mixing_20261022/native_130_source_probe.json`이다.

새 기본 `generation_mode="clutter"`의 runtime 연결을 추가했다. Reference10 catalog는 역할의
상대 pose·시점·초기 가림 정보만 공급한다. 주변 물체는 실제 collider의 GPU FFT free-grid seed,
단일 CPU 공동 제약 계산, 고정3회 조건부 X·Y mixing 순서로 만든다. 이전 두-row 예산·전체 이동
경로는 사용하지 않는다. GPU seed의 필수 행이 빠져도 NaN인 부분 graph로 남겨 공동 제약에서
복구하며, 필수20개·종류 집합을 줄이는 승인 fallback으로 처리하지 않는다. 추가 활성 물체는
같은 균형 순서의 prefix다. 공동 제약은 solver 내부 탐색을 포함할 수 있으며 전체 생성이
GPU-only인 방식은 아니다.

Production GPU helper는 mandatory head의 큰 형상 우선 탐색과 원래 row ID 보존을 지원한다.
뒤의 optional tail은 입력 순서와 첫 실패 중단을 유지한다. Collider는15mm inset support,
canonical OBB는 별도의 panel±5mm 경계로 검사한다. 자유 yaw1개·나머지π/±.001rad 변화가 있는
어려운8장면에서 필수18개 BG가 모두 seed됐고 실제 최소 간격은약3.20mm였다. CUDA 회귀6개도
통과했다. Warm 약110–120ms, CUDA 최대480MB는 seed helper 측정이다.

카메라 렌더 없이 기본 경로를 실행하고 초기 관측/독립 실루엣 검사·partial reset·전체 활성
물체·VRAM을 기록할 outputs-only probe를 준비했다. 새4환경 실행 결과는 아래에 구분한다.

후속 CPU 전체 검사331개는33.084초에 통과했고21개는 skip이었다. 학습 관련56개도 통과했고
2개를 skip했다. Camera lateral 경계 수정의 표적 회귀3개도 통과했다. 별도90 ordered pair
initial-domain 검사에서는 normal6,592 family·103,513 dedup cell의 두 역할360° coverage와
camera lateral±.35m를37.14초에 확인했다. 자료는
`clutter_initial_angle_coverage_with_lateral_audit.json`이다. 이는 역할 영역의 검사이며 주변
20–40개 packing의 전체 용량 인증은 아니다.

새 기본값4환경 native reset·30step은 통과했다. Seed42의24개·10종 장면이며 Target 종류는1개,
non-Target은2–3개씩 균형을 유지했다. 독립 full-silhouette query1회에서 Target68.85246%·
Blocker100%, framing·최소 pixel 조건을 통과했다. 초기 singularity0, 최소 margin0.14111;
정책 `[4,244]`, 물리 ID `[4,40]`, geometry `[4,10,10]`, tactile force `[4,17,3]`를 확인했다.
부분 reset은 다른3환경의 관절·물체·초기 geometry를 bit-exact하게 유지하고 새 layout·ray query를
만들지 않았다. 30step 상태는 모두 유한했고 종료 원인·fallback은0이었다.

Scene 초기화23.76초, 첫 reset20.48초에는 catalog9.49초·첫 Numba mixing4.84초 등이 포함됐다.
이 cold 검사와30step3.59초는 steady 학습 iteration 측정이 아니다. 종료 시 GPU used5.39GB·
free19.90GB, host RSS5.56GB였다. 자료는 `native_clutter_reset_4_probe.json`이다.

최종 standalone 검사는90 ordered pair와 어려운 큰 물체4종 구성8개, 총98장면을 통과했다.
90개 표본의 활성 개수는20–38개·종류4–10종이며 warm 생성 중앙값0.343초·95백분위0.574초였다.
이는 전체 종류·각도의 보편적인 수용 가능성이나40개 실증을 뜻하지 않는다.

새 기본1,024환경 native reset·10step도 통과했다. 같은24물체 장면에서 독립 메시 가시성
Target68.85246%·Blocker100%, 초기 singularity0·최소 margin0.028813을 확인했다.
정책 `[1024,244]`, 물리 ID `[1024,40]`, geometry `[1024,10,10]`와 모든 환경의 유효10-slot을
유지했다. 부분 reset은 다른1,023행을 bit-exact하게 유지하며 장면·ray query를 만들지 않았다.
10step의 상태는 유한했고 종료·fallback은0이었다. GPU used10.84GB/free14.45GB,
host RSS9.54GB로 OOM·PhysX 오류가 없었다.

초기화51.70초·첫 reset15.84초에는 catalog9.75초 등이 포함됐다. 공유 장면 생성0.583초는
GPU seed0.379초·CPU 공동 solve0.158초·mixing0.024초를 포함한다. 10step4.318초는 cold 진단이며
정규 PPO iteration 처리량이나 장기 안정성 검사가 아니다. 자료는
`native_clutter_reset_1024_probe.json`; 실제 collision footprint 그림은 같은 출력 폴더의
`clutter_layout_native1024.png`와 `.svg`다.

## 2026-10-22 — GUI 역할 OBB와 기본 debug-panel 토글

사용자는 Target·Blocker의 OBB를 GUI에 표시하되 PhysX 계산에는 포함하지 말 것을 요청했고,
EEF gizmo처럼 실행 중 ON/OFF할 수 있어야 한다고 추가했다. 기존 파일 안에서
`RoleObbVisualizer`와 작은 `ManagerBasedRLEnvWindow` subclass를 연결한다. 공식
`_create_debug_vis_ui_element`로 Scene Debug Visualization에 `Target / Blocker OBB` 체크박스를
추가하며, Target은 초록색·Blocker는 빨간색12-edge wireframe이다.

표시는 정답 catalog bounds와 현재 body-link pose로 계산하며, 정책의 초기 vision이나 물리
상태를 수정하지 않는다. Debug-draw API만 사용하고 USD·rigid body·collision shape를 만들지
않는다. OFF는 선과 render callback을 제거하고, headless는 visualizer 자체를 만들지 않는다.
학습·play의 기본 실행 흐름은 유지하며 `role_obb_debug_vis`는 GUI 초기 표시 여부만 설정한다.

실제4환경에서 공식 `SimpleCheckBox`의 모델을 OFF→ON으로 바꾸어96선→0선→96선과 callback
제거·복원을 확인했다. 등록·토글·UI 조작 전후 물리 상태와 USD prim·RigidBodyAPI·CollisionAPI
집합은 동일했다. 위치·기울기·역할 변경 및 부분 reset을 따라가며 종료 시 선과 callback을
정리한다. Headless startup에서는 visualizer와 callback이 생성되지 않았다. 실제 UI 위젯/API를
검증한 headless 진단이며 GUI 화면의 육안 검사는 아니다. 기록은
`outputs/contact_manipulation/clutter_mixing_20261022/native_role_obb_probe.json`이다.

## 2026-10-22 — 활성 물체 상한30개로 조정

사용자가 최대40개는 많다고 판단하여 기본 물체 수 범위를 **20–30개**로 낮췄다.
`scene.py`와 환경 cfg의 기본 상한을 함께30으로 변경한다. 종류별 재사용 body는 자동 계산으로
13개에서10개, 전체 pool은130개에서100개로 줄어든다. Target 종류1개·균형 중복·빈 재배치 공간
규칙과10-slot/244D 관측, GUI OBB 토글은 유지한다. 이전130-body의98장면·native1,024환경
측정 기록은 이력으로 보존하며 현재100-body의 성능 측정값으로 해석하지 않는다.

## 2026-10-22 — 활성 물체 범위15–25개로 조정

사용자의 추가 요청에 따라 생성 범위를 **15–25개**로 변경했다. 필수 주변 물체는13개이며,
재사용 pool은 종류별8개·총80개다. 관측10-slot/244D와 기존 배치·가시성·GUI OBB 규칙은 유지한다.

## 2026-10-22 — 부분 reset의 parking wrench 갱신 범위 축소

사용자는 headless1,024환경 학습의 지연 증가에 대해 부분 reset의 불필요한 전체 pool 갱신을
확인한 뒤 수정하도록 승인했다. `TaskObjectPool.refresh_parking(env_ids)`에 공식
`WrenchComposer.reset(env_ids)`와 `set_forces_and_torques(env_ids=...)`를 사용하고, 물리 reset과
예외 복구 양쪽에서 대상 ID를 전달한다. Setter force는 선택한 ID 순서의 행만 구성하며 다른
환경의 force·torque를 보존한다. 부분 경로의 전체 mask `any()` host 동기화도 없앴다.

전체 refresh에서는 모든 물체가 활성인 경우 composer를 비활성화하는 기존 처리를 유지한다.
부분 refresh가0힘만 갱신한 경우에도 global composer의 활성 상태를 바꾸지 않으며, 현재
80-pool/최대25-active 구성에서는 항상 주차 물체가 존재한다. Native composer 내부의 전체
pose clone과 매 physics step의 전체 pool force 제출은 이 변경으로 제거되지 않는다.

Blocker-only contact filter는 이번에 변경하지 않았다. 이 필터는 reward용 pair-force에만
관여하며,17채널 binary observation은 필터와 무관한 전체 `net_forces_w`로 다른 물체 접촉도
계속 감지할 수 있다. 사용자가 원하지 않은 장면 전환 세부 계측도 추가하지 않았다.

실제 production class와 공식 composer·Warp kernel의 CPU 검사에서 순서가 뒤섞인 부분 ID,
반복된 mask·자세 변경6회가 이전 전체 refresh 결과와 일치했다. 선택하지 않은 행의 force·
torque는 bit-exact하게 보존됐고 빈 subset 무동작·전체0힘 비활성화·부분 경로 host-bool 없음도
확인했다. 관련 reset/contact 회귀33개 중32개 통과·1개 skip이다. 재현 자료는
`outputs/contact_manipulation/constructive_layout_20261022/production_parking_subset_cpu_verify.json`이다.
이번 검증은 CPU 단위 검사이며 GPU 학습 iteration의 개선폭은 측정하지 않았다.

## 2026-10-22 — Blocker 전용 접촉 필터 적용

사용자 승인으로 reward용 접촉 필터를 전체 물체 pool에서 **공유 장면의 현재 Blocker 하나**로
축소했다. `TaskContactSensor.enable_blocker_filtering`은 공식 PhysX
`create_rigid_contact_view`에 pad별로 해당 환경의 Blocker 경로 하나만 전달한다.
기존 `[N,17,80,3]` 대신 `[N,17,3]`을 읽으므로 step마다 전체 행렬을 복사하고 Blocker 열을
gather하는 과정도 제거된다. 물리 body 수·충돌 형상·17개 tactile 채널·관측 차원은 유지한다.

`reset_scene`은 배치를 적용한 뒤 `sample.blocker_id` Python 정수로 필터를 연결한다.
같은 ID의 부분 reset에서는 native view를 재사용하고 다른 환경의 접촉 이력은 보존한다.
공유 Blocker가 달라질 때만 view를 교체하며 과거 역할의 view는 보관하지 않는다. 최초의
net-only view 하나는 유지한다. Stop/Play로 native view가 재생성된 경우 다음 reset에서
다시 연결한다. Reset 없이 재개한 첫 reward에서도 보관한 host ID로 재연결하며 이전 접촉
이력은 제거한다. 접촉 유지 이력은 공식 RewardManager의 부분 reset 경로를 계속 사용한다.

Binary observation은 여전히 전체 `net_forces_w`로 계산하므로 **다른 물체·선반 접촉도 감지**한다.
Blocker 필터는 reward의 접촉 상대 구분에만 사용한다. 접촉 유지 term을 제거하거나 시작부터
weight0으로 설정하면 필터를 생성하지 않는다. CUDA의 역할 ID를 매 step CPU로 읽지 않는다.

1,024환경 기준 native pair buffer와 task cache의 합계는 약31.88MiB에서0.40MiB로 줄어든다.
이는 접촉 행렬만의 크기이며 물리 충돌·전체 simulation·PPO 시간의80배 감소를 뜻하지 않는다.
검증 결과와 비용 측정은 TRAINING_LOG45절에 기록한다.

## 2026-10-22 — 23:26:39 학습의 보상 재검토

2026-10-21_23-26-39의0–5,221iteration을 분석했다. 당시10개 물체이며 reward 공식·
weight·민감도는 분석 당시 설정과 같았다(후속 weight 적용은 아래 절). 누적 성공률 약2.018%, 최상250구간1136–1385의4.923%에서
마지막250구간2.345%로 감소했다. 동시에 native Clearance 지급은29.4% 증가했다.
안전한 재배치가 아닌 진행·유효 성공 간 관계를 검토하되 로그만으로 원인은 확정하지 않는다.

분석상의1차 후보는 Hold .05→.10, Stability2→3, Success100→250이며 Clearance4,
Distance .4, Rearrangement1과 Motion weight·각 decay는 유지하는 것이었다. 분석 시점에는 적용하지 않았다.
Stability 경고10mm→5mm와 상승 폭 확대(a6931→3081)는 정지 오차·실패 원인 확인 후 별도
실험 후보로 분리한다. Unsupported goal은 들림뿐 아니라 XY 이탈·전도 등도 포함한다.
현재15–25개 장면에는 all-object 조건과 합산 비용이 더 자주 발생할 수 있다.

Value loss는 초반부터 간헐적으로1e12이상이고 std는 약1→6.81로 증가했다. 검사한
checkpoint의 Critic weight 자체 폭발은 없었으며 큰 finite 관측·bootstrap target 등이
원인인지 아직 미확정이다. Weight 조정으로 이 문제가 해결됐다고 판단하지 않는다.
원본 로그·설정·학습 코드 변경 없이 근거와 후보를 TRAINING_LOG46절 및
outputs/contact_manipulation/reward_review_20261021_232639/에 기록했다.

## 2026-10-22 — 보상 weight 1차 조정 적용

사용자 승인으로 앞 절의1차 후보를 적용했다. `blocker_contact_hold.weight=.10`,
`object_stability.weight=3.0`, `MdpCfg.success_reward=250.0`이며 Terminal weight는1.0이다.
17pad Hold의 최대 기여는1.70/step, Stability는 지지 거리10mm까지0·20mm에서−1.5·30mm에서
−2.8125다. 경고10mm, 지지 실패30mm/.1초와 나머지 성공·종료 조건은 유지한다.

모든 exp 민감도와 나머지 reward weight·공식 PPO 설정·action은 바꾸지 않았다. 경고5mm와
넓은 Stability 상승 구간은 별도 미적용 후보다. 양의 dense 상한은4.62848/step, γ=.98의
할인합 상한은231.424다. 성공+250은 지연 할인·실제 궤적까지 고려한 최적값의 증명이 아니다.
새 학습은 실행하지 않았으며 value-loss/std 이상 해결이나 현재15–25개 장면의 성공률 개선을
확인한 결과도 아니다. 관련 회귀109개 중106개 통과·CUDA3개 skip과 문법 검사를 완료했다.
설정 변경과 검증 범위는 TRAINING_LOG47절에 기록한다.

## 2026-10-23 — Patch 용량·낮은 관측 해상도·관측 OBB

학습 종료 후 사용자 요청으로 Patch 용량을163,840→262,144로 늘렸다. 실제 observation
ray는160×120→64×48(3,072개,84% 감소), 최소 표면점은3개다. 배치 인증160×120/32·16pixel과
기존b29c...90-pair cache는 유지하며 observation 전용3개 설정만 source hash에서 제외한다.
추가 시점 탐색·재시도·메시 복제는 없고 부분 reset은 초기 관측을 재사용한다.

GUI에 청록색 Observed objects OBB를 별도 표준 checkbox로 추가했다. 정책의 유효한 초기
geometry_b/geometry_valid만 표시하며 실제 물체가 움직여도 초기 추정값을 유지한다.
역할 green/red와 하나의 DebugDraw callback을 공유하되 toggle은 독립적이다. USD·PhysX
형상/상태를 추가하거나 수정하지 않으며 headless에서는 visualizer를 생성하지 않는다.

실제4환경에서 기존 cache 재사용·관측[4,244]와10개 유효 slot·2step·부분 reset을 통과했다.
Native DebugDraw의 두 레이어·독립 toggle을 검사했고 USD 구성·물리 상태는 변하지 않았다.
CPU 전체348개 중327개 통과·CUDA21개 skip이다. 육안 GUI 확인·1,024환경 장기 학습·
overflow 재발률·전체 iteration 향상은 측정하지 않았다. 상세 결과는 TRAINING_LOG48절과
outputs/contact_manipulation/ray_obb_20261023/에 보존했다.

## 2026-10-23 — 1,024환경 collection 비용 실측

사용자 요청으로 production 코드 변경 없이 native 학습 경로를 32 iteration 진단 실행했다.
Warmup 10회·baseline 5회·wall 계측 14회·동기화 진단 3회이며 물리·정책 100 Hz,
rollout 36, headless, seed 42, 활성 17→19물체다. 공유 장면 교체를 제외한 wall 13회의
collection은 평균 3.460초 / 중앙값 3.352초였다. 물리 step 1.923초(55.6%), 부분 reset
0.746초(21.5%), 종료 판정과 공통 MDP·센서 계산 0.257초(7.4%)가 큰 항목이다.
Actor·Critic 추론은 0.095초였고 console/TB 출력 0.036초는 collection 밖이다.

800-step 공유 장면 교체 iteration은 collection 7.066초 / reset 합계 4.487초였다.
Template 배치 생성 0.262초와 ray·관측 생성 0.00435초보다 로봇 시작 자세 준비가 컸다.
일반 iteration에도 평균 35.85 batch의 부분 reset이 있었다. 동기화 진단에서도 물리 step이
최대 항목이지만 fetch_results 대기를 순수 충돌 kernel 시간으로 단정할 수는 없다.
새 장면 직후 graph/cache 준비 비용과 새 정책의 종료 분포가 포함된 짧은 측정이다.
상세 계측·한계·중복 없는 표는 TRAINING_LOG49절과
outputs/contact_manipulation/collection_profile_20261023/summary.json/txt에 보존했다.

## 2026-10-23 — 미사용 body 비활성화 실험: 미채택·기존 방식 복원

Pool과 생성 규칙을 유지하며 미사용 body의 simulation 처리를 줄이는 방법을 실제 SDK에서
검사했지만 안전성과 속도를 함께 통과한 방식은 없었다. **미사용 body 비활성화는 아직
미해결**이며 production은 기존 공중 격리+native permanent COM countergravity로 복원했다.
공식 set_disable_simulations/wake_up은 object-major mask가 맞아도 physics 뒤 GPU pose
readback이 주차 pose에서 최대 2.82m 달라져 채택하지 않았다. 실제 actor 이동인지 SDK
readback/index 문제인지는 미확정이다. 공식 explicit sleep도 공유 장면 교체에서 미사용
body의 5.886mm 낙하가 재현돼, CPU sleep flag만으로 성공으로 처리하지 않는다.

기존 Ground 위 주차+countergravity 제거는 4환경 반복 reset·17 pad binary·Blocker 필터
검사를 통과했지만 1,024환경의 모드별 192-step 대조에서 physics 34.662→36.783ms(+6.12%),
action/write/physics/update 합계 42.477→43.872ms(+3.28%)로 악화됐다. Scene write는 줄었지만
3개 paired block 모두 전체 시간이 늘었으므로 해당 대안도 복원했다. PPO/reset을 포함한
전체 iteration 비교는 아니며, 주차 방식만 바꿔 물리 최적화가 완료됐다고 설명하면 안 된다.

Mustard는 원본 Axis_Aligned_Physics USD부터 maxConvexHulls32·실제 cooking32 hull이다.
우리가 늘린 값이 아니다. 예제 robot cfg는 solver를 USD에서 가져오며 직접 확인한 로컬
UR5e_v4.usd는 Position32/Velocity1이다. 예제 물체는 명시적으로16/1, 현재 task는 robot32/2·
물체16/2다. 예제에 하드코딩된 다른 사용자 경로의 파일 동일성까지 확인한 것은 아니다.
이번에는 solver 설정을 바꾸지 않았다. 복원된 production 관련 CPU28개 검사 통과.
상세 실패·대조·원본 근거는 TRAINING_LOG50절과
outputs/contact_manipulation/inactive_pool_20261023/에 보존했다.

## 2026-10-23 — 예제 solver 적용과 미사용 물체 옵션

사용자 승인으로 현재 solver를 로봇 Position32/Velocity1·물체16/1로 맞췄다. Position 반복 수,
물리·정책100Hz, Mustard의 원본32 hull은 유지한다. README도 갱신했으며 과거 velocity2에서
측정한 native 손 제약/성능 수치는 새 설정의 검증으로 사용하지 않는다. 관련 CPU36개 중
30개 통과·CUDA6개 skip, 설정 AST·문법·diff 검사 통과다. GPU 성능 재측정은 하지 않았다.

kinematic_enabled=False는 이미 현재 설정이고 dynamic body이므로 물리 연산을 끄지 않는다.
미사용 body를True로 전환하는 방법은 별도 후보이나 collider는 남는다. USD 타입 변경은 가능하지만
설치된 tensor API에 일괄 타입 setter가 없고 현재 GPU collection 반복 reset은 미검증이다.
실제 simulation ON/OFF는 공식 set_disable_simulations(1=비활성,0=활성)과 재활성 후 wake_up으로
설정할 수 있다. 앞 절의 미채택 이유는 API 부재가 아니라 해당 경로의 native pose/reset 검사
실패다. 이번에는 solver만 적용했고 미사용 body는 기존 활성·격리·countergravity를 유지한다.
상세 변경과 확인 범위는 TRAINING_LOG51절에 기록했다.

## 2026-10-23 — Simulation ON/OFF 적용, 실제 실행은 사용자 확인

사용자 요청으로 미사용 body 처리를 공식 set_disable_simulations로 변경했다. Reset 대상
환경의 pool을 활성화한 뒤 native pose·velocity를 기록하고 unused를OFF·active를wake한다.
전체 CPU uint8 flag buffer·object-major 인덱스를 재사용하며 부분 reset의 다른 환경은
보존한다. CPU ID·mask 전송은 reset에만 있고 기존 중력 상쇄·매 step parking force 제출은
제거했다. Native collection update/reset/write_data_to_sim은 override하지 않는다.

실패 rollback도 같은 순서를 사용한다. Inactive GPU readback은 복구 상태로 신뢰하지 않고
authored parking state+env origin으로 치환하며 active native state는 보존한다. 이전 pose 조회
차이가 해결됐다는 확인은 아니다. 사용자 본인이 실행 검증하기로 했으므로 이번 작업에서는
GPU simulation·학습을 실행하지 않았다. 관련 CPU mock/reset29개 검사는 통과했고 실제 native
재활성화·접촉·성능은 미검증이다. 현재 production은 더 이상 countergravity 방식이 아니다.
물체 pool/배치·센서·관측·solver32/1 및16/1·Mustard32 hull은 유지한다. TRAINING_LOG52절 참고.

## 2026-10-23 — PhysX GPU 내부 비용 실측

1024환경·headless·100Hz에서 짧은 native 학습5회와 정상 자세 유지 대조1회를 측정했다.
정상 유지72step은 손 limit 최대.000369rad·mimic.000435rad, q/qd finite·mass 보존·
unused64512개 유지로 통과했다. Physics26.12ms/step, GPUkernel합22.83ms 중 내부constraint+
mimic51.90%, contact/jointsolver15.47%, articulation dynamics/integration13.96%,
bookkeeping10.89%, broadphase+narrowphase6.48%다. 주요constraintkernel은33회/step이고
32blocks×32threads로 실행됐다. 병목은 native articulation 제약 반복이며 Python 환경별
반복문 제거와 별개다. Kernel합·GPU점유·CPUwall·계측 오버헤드는 구분해야 한다.

움직이는 정책에서는44–45ms/physics step이었으나 수만rad 이상의 큰 유한 손 관절값이
나타나 정상 동작 benchmark라고 볼 수 없다. TGS·PGS·flag미사용countergravity 모두에서
발생했으므로 특정solver나새API 하나의 원인으로 단정하지 않는다. 원인은 미해결이다.
정상 유지와 정책의 차이도 명령·접촉·조회·계측 차이가 있어 상태 이상 한 원인의 비용이 아니다.

Contactreport32→17 진단은physics−.28%로 이득 근거가 없고, PGS도 정상성 문제가 있어
채택하지 않았다. Env-ID bounds기본4는 이미 사용한다. 선택적fixedmerge는17pad를 보존하면
housing2개만 줄일 수 있어 큰 개선을 약속할 수 없다. Solver 반복 축소는 손 정확도 재검증이
필요하다. GPUbuffer용량·Mustard32·UR5e·100Hz·현재source는 변경하지 않았다.
자료는 outputs/contact_manipulation/physics_step_profile_20261023/와 TRAINING_LOG53절이다.


## 2026-10-23 — 장면 교체 후 선반 통과 수정

Live `set_disable_simulations` OFF/ON은8-Cube/16환경에서 GPU actor 좌표 대응 불일치와
CUDA700을 재현해 제거했다. 모든 actor를 유지하고 unused를 공식 permanent COM wrench로
중력 상쇄하는 방식으로 돌아갔다. Active 외력0, selected reset/rollback, 고정 mass/gravity
GPU cache를 사용한다. 미사용 body의 simulation 비용은 남는다. 과거 모든 CUDA719의 원인이
같다고 단정하지 않는다. 이미 OFF/ON을 수행한 앱은 새 프로세스로 시작해야 한다.

이 변경만으로는1,024환경의 충돌 누락이 해결되지 않았다. 기본4bit 환경 bounds에서 scene2의
4물체가 env%16!=0인960환경에서 낙하했다. 독립 native view와 실제 Shelf GPU pose·native
scene query로 표시 오류/잘못된 좌표를 배제했고,2mm 여유/200Hz/필터·외력·geometry 공유 제거는
효과가 없었다. 현재는 **8bit 환경 ID bounds + 환경별 native static Cuboid 바닥**을 함께 쓴다.
바닥2.49×2.49×.10m, 윗면Z=0,collision_group0이며 전역 무한 Ground는 제거한다.
8bit만 쓰면 global encoded bounds 경계 때문에 env%256=0/1/254/255에서 선반이 떨어져
바닥 교체가 필수다. 4bit 물체 누락의 세부 engine 오류는 미확정인 검증된 회피 구성이다.

Native cloner가 root layer에 쓰는4보다 강한 session layer에8을 physics attach 전에 기록한다.
실제USD root/session 합성과 edit-target 복원을 CPU로 검증했다. Reset 중 scene 설정/topology를
바꾸지 않고, 스폰 높이/upright/100Hz/solver/UR5e/센서17개/Mustard는 유지한다.

실제 production seed42·1,024환경에서8장면×80step과512행 부분 reset8회 모두 표본 낙하0,
최저 지지면 차이−1.009mm, reset 위치 오차2.384e−7m, Shelf 위치 오차0.0706mm,
독립 view 오차0, 미선택 행 bit-exact를 확인했다. 다른seed43의 native `env.step()`도4장면
검사에서 낙하0이었다. CPU44개 통과/CUDA1개 skip. 고정 제어 회귀이며 장기 정책 학습 안정성을
보증하는 검사는 아니다. 자료: outputs/contact_manipulation/reset_support_20261023/,
TRAINING_LOG54절. 앞선52절의 simulation OFF/ON 적용 상태는 이 수정으로 대체된다.


## 2026-10-23 — 스폰 변경 없는 미사용 물체 제외 대안 확인(아직 미적용)

후속 적용과 실제 collider 동작 정정은 바로 다음 항목 및 TRAINING_LOG56절을 우선한다.
이 조사의 shared prototype collider OFF는 native instance의 충돌 제외로 검증되지 않았다.

사용자 요청으로 simulation OFF/ON 대안을 output-only 진단에서 조사했다. 미사용 body를
공식 USD kinematic=True, 원래 enabled collider만 collisionEnabled=False로 전환하고,
다시 사용할 때 flag 복원/flush 후 pose·velocity를 기록하는 방식이다. Body와GPU island index를
유지하므로 기존 actor 제거형 OFF/ON과 다르다. Native tensor에 일괄 타입 setter가 없어
현재 공통 배치의 env0/프로세스 전용 shared library 속성 전파를 사용한다. Actor/buffer 메모리와
일부 transform관리 비용은 남는다. 원본 asset 파일·geometry·mass/friction은 수정하지 않았다.

**원래 전역 Ground·4bit·같은 스폰/100Hz**로 실제4환경4장면 및1,024환경6장면/부분 reset을
통과했다. 표본 낙하0, unused pose 오차2.384e−7m, 독립 view오차0, 미선택512행bit-exact,
실제kinematic 수=unused 수, simulation disable0, tactile17body 유지다. 실제pad접촉fixture와
장기 정책 안정성 검증은 아직이다. 8Cube 재사용 중력/지지 검사도 통과했다.

정상2쌍의직렬화microbenchmark에서 physics31.128→26.342ms(−15.38%),
action/write/physics/update합38.867→34.290ms(−11.78%)였다. 대조군 unused drift12.56cm가
나온1쌍은양쪽모두제외했고원시자료를남겼다. PPO iteration개선율이아니며, 전체unused속성을
전환하는준비에는CPU약.9초가소요됐다. 변화없는부분reset은속성변경/flush를피해야한다.

Production은이번턴에변경하지않았다. 통합시전역flag변경을최종전체commit으로옮기고,
단일env0준비실패rollback과의정합성·원래disabledcollider보존·공유layer소유권·실제접촉fixture를
확인해야한다. 채택하면54절의바닥/8bit우회대신원래바닥/4bit를유지할수있는실행근거가있다.
상세는TRAINING_LOG55절과outputs/contact_manipulation/inactive_alternatives_20261023/summary.json.


## 2026-10-23 — 미사용 pool의 kinematic 전환 적용

현재 production은 미사용 body에만 공식 USD kinematic=True를 적용하고, 재사용 시False와
native flush 후 pose/velocity를 기록한다. Simulation OFF/ON·gravity toggle·countergravity는
사용하지 않는다. Actor/GPU index와 기존 격리 주차 위치를 유지하며, 스폰 규칙·질량/마찰·
물체 수·로봇·100Hz·MDP·PPO는 변경하지 않는다. 이전 환경별 Cuboid 바닥/8bit 설정은 제거해
원래 전역 GroundPlane/native4bit로 돌아갔다. 앞선 바닥 우회 적용 기록은 현재 설정이 아니다.

실제 접촉 검사를 추가하자 prototype collider OFF가 live instance shape에 적용되지 않는
경로가 드러났다. 무효한 collider 속성 변경은 제거했으며 기존 collision flag는 전부 유지한다.
미사용 동역학 solve/integration은 제외하지만 collider·broadphase·transform·메모리는 남는다.
55절 microbenchmark의 약15% 물리 시간 차이는 collider 제외나 전체 PPO 개선율의 근거가 아니다.

공통 active mask 변경은 전체 장면의 최종 commit에만 허용한다. 같은 mask의 부분 reset은
USD/flush/CPU flag 복사를 생략한다. 후보 준비는 물체를 건드리지 않고 로봇만 임시 배치한 뒤
성공/실패 모두 복구한다. Flag commit 실패 시 이전 flag와 전체 물체 velocity/pose를 복구한다.
Native collection의 COM-pose cache 보정은 유지하며 별도 Python physics step override는 없다.

CPU47개 통과/CUDA1개 skip. 실제4환경에서 reset12회, 주차 오차0, native pose 오차2.384e−7m,
다른 행 오차0,17pad 각각 binary 접촉/분리 및 Blocker 재활성화3회 filtered contact를 통과했다.
모든simulation disable flag0이며 재활성 물체의 정상 중력 응답과 native env.step finite도 확인했다.
최종 production1,024환경6장면×80step과 매 장면512행 부분 reset도 표본 낙하0, 다른 행
bit-exact였다. Native kinematic 수가 unused 수와 일치하며 주차/reset 오차2.384e−7m,
실제 bounds4bit·전역 plane1개를 확인했다. 결과는 production1024/support.json과applied_summary.json이다.
장기 학습 안정성을 보증하는 결과는 아니다. 이미 OFF/ON을 사용한 앱은 새 프로세스로 시작한다.
상세: TRAINING_LOG56절, outputs/contact_manipulation/inactive_applied_20261023/.


## 2026-10-23 — Target 보호 gate·초기 면적·F/T 분리와 Time 복원

사용자는10,000 iteration을 초기 학습으로 보고, 접촉·통로 확보 행동의 발전과 수치 이상을
구분해서 평가하도록 교정했다. 낮은 초기 성공률이나 원시reward 진동만으로 학습 실패를
단정하지 않는다. 최신 후속 검토에서는 모든 penalty를 일괄 증액하지 않고 수식의 불일치를
먼저 정리하기로 했다. 최종 사용자 선택은 **추가 Target 종료 없이 회복 가능한 gate**다.

- Target 초기 pose 대비 이동≤2cm·회전≤0.15rad라는 기존 성공 조건을 공통 함수로 사용한다.
  이를 벗어난 동안 양의 Clearance·Distance·Contact 유지 보상을0으로 만든다. 음의 Clearance와
  Rearrangement·Motion·지지·Time 비용은 유지한다. Target이 범위로 돌아오면 보상을 재개하며
  실제 접촉 이력도 gate와 독립적으로 갱신한다. Object stability와 중복되는 새 Blocker 불안정
  gate는 추가하지 않는다. 기존 물리 진행 gate는 양의 Clearance에 계속 적용한다.
- Clearance는 현재 Blocker와 초기 접근영역의 교차면적을 **초기 Blocker의 XY 투영면적**으로
  나누고0~1로 제한한다. 분모와 초기 품질은 episode 동안 고정하며 선택된 reset행만 갱신한다.
  기울어짐·회전으로 현재 투영면적이 달라져 분모가 바뀌는 효과를 제거한다. 지급은 여전히
  초기 대비 품질 차이를 반복 지급하는 방식이며 직전step개선량으로 변경하지 않았다.
- F/T는 `motion_force_growth`·`motion_torque_growth` 각각native weight.15로 분리한다.
  보정 후 현재 센서 중심의 힘·토크 norm 증가율에서10N/s·0.5N·m/s를 각각 뺀 양수부 제곱을
  독립exp에 넣는다. 계수는`ln(2)/90²`·`ln(2)/4.5²`이므로100N/s·5N·m/s에서 각각품질.5,
  비용−.075다. 하중 감소는0이며 힘과 토크의 단위를 직접 합치지 않는다. 결합wrench함수는
  제거하고 components는`[C,R,F,T,Terminal]`5열로 바꾼다. 작은 비용 기울기와 F/T총하한−.30은
  유지하지만 중간 곡선은 이전 결합식과 다르므로 과거로그를 동일한 점수로 비교하지 않는다.
- Time은 사용자가 앞서 정한 지수형 비용률`c(t)=.02*(exp(.7*t)-1)`의 완료 control 구간
  적분을 음수로 지급한다. `T_ref`는 없고 native weight1이다. 함수는 구간 평균 비용률을 반환해
  공식 RewardManager의dt가 한 번만 적용된다. 경과시간은 완료전이당1회 증가하고 재조회로
  늘지 않으며, 부분reset은 선택행만0으로 초기화한다. 할인 전8초합−7.53789735503,
  100Hz 마지막step−.0536964239다. 이는 과거초기값 복원이며 새 최적weight를 측정한 값은 아니다.

Rearrangement는 **직전step**의 non-Blocker 물체·선반 평균8-corner 이동량 합에
`exp(-20*x)-1`을 적용하는 기존 방식이다. Native manager가dt를 곱하지만 adapter가dt로
나눠 상쇄하므로 초기위치와의변위나누적경로를 매step반복과금하는 방식이 아니다.
동일10cm를1step에 움직이면약−.865,10step에 균등분할하면총−1.813이라는 시간분할편향은
현재남아있다. 누적경로는진단용이며 수식변경은 이번에 확정하지 않았다.

Object stability도 현재선반frame의`(OBB중심XY,실제collider최저Z)`에서 유한작업판 최근접점까지의
거리만 사용한다. 이거리는지지·회전안정성이나접촉지지다각형의증명이 아니며, 지지된 전도나
빠른평면이동은거리0일수있다. weight3·10mm무감점·같은물체3cm/.1초종료·기존tilt80°/.2초는
그대로두고 구체적기하의의미를 사용자와 더 논의한다. 임의의속도·기울기 penalty는 추가하지 않았다.

활성reward는12개, native종료는기존7개이며 성공bonus250·PPO·모델·action·스폰·물리주기는
유지한다. 이 변경으로 학습 개선이나 기존 value-loss 이상 해결을 입증한 것은 아니다.
상세 코드·회귀 검증 결과는 TRAINING_LOG의 최신 적용 기록을 따른다.


## 2026-10-23 — 누적 Rearrangement와 전체 OBB·기울기 안정성

사용자가 Rearrangement의 의도를 **초기부터 발생한 전체 조작량의 누적**으로 다시 확인했다.
이에 직전step 증분만 평가하던 방식을 episode 누적 경로`P_k=sum_j ell_j`로 변경한다.
`ell_j`는 지정Blocker를 제외한 활성물체와 선반 각각의 대응8-corner변위norm평균의 합이다.
이 값은 초기·현재pose의 순변위가 아니므로 왕복도 남는다. 기존 `path_length`를 보상 입력에
사용하며 `rearrangement_increment`는 진단용이다. 품질`exp(-20*P_k)`에서1을 빼고 weight1로
매step지급하므로 이동후멈춰도 비용이남는다. 누적1cm이면 매step약−.181269,10cm이면−.864665다.
완전히 미교란인상태만0이며 정지 자체의 양의보상은 없다. Native adapter의 `/step_dt`와
manager의dt는 기존처럼 상쇄하며, 전체이동량에 대한 일회성 선형비용이나 주기불변비용이라고
해석하지 않는다. 선택된행만 reset 시 누적0·이전corner현재값으로초기화한다.

Object stability는 사용자승인으로 경계돌출과기울기를 기존term에 반영한다. 선반frame에서
물체i의 OBB8개 XYcorner가 작업판사각형바깥으로 벗어난 거리의 최대값을`delta_xy`로 구한다.
Z는 기존과같이 실제collision기하의최저점과 작업판높이 사이 signed gap `g`다.
새reward거리 `d_stab=sqrt(delta_xy²+g²)`는 중심이안에있어도 물체가돌출하면 커지며,
OBB의 빈Z모서리를 collider최저점 대신 사용하지 않는다.

물체upright기준위쪽축과 선반법선의 내적에서 `theta=acos(clamp(dot,-1,1))`를 구한다.
평면yaw는제외한다. 사용자는 **theta 자체를 exp에 넣고 제곱하지 않도록** 선택했다.
물체별무차원위험은 `x_i=(ln2/.01²)*[d_stab-.01]_+²+(ln2/(π/4))*theta_i`다.
기존weight3으로 `3*(exp(-max_active(x_i))-1)`을 지급한다. 같은물체에서거리와각도를합친뒤
가장위험한활성물체를선택하므로 서로다른물체의최악거리·최악기울기를따로더하지않는다.
Blocker도포함하며 센서·물리query·환경별Pythonloop는추가하지 않는다.

거리쪽10mmdeadband와20mm단독half-quality는유지한다. 각도는별도deadband없이0부터
평가하며 `stability_tilt_decay=ln2/(π/4) rad^-1`이다. 단독45°에서품질.5·비용−1.5이고,
같은물체가거리20mm와45°를함께보이면품질.25·비용−2.25다. 정상pivot도각도가생기면비용을
받는설계이므로 이값은실제성공조작의허용범위를측정한결과가아닌초기민감도다.

종료에쓰는`support_distance`는 기존 **OBB중심XY+collision최저Z**의 유한작업판거리다.
같은물체3cm/.1초지속 종료와tilt80°/.2초·성공/진행gate는변경하지않는다. 새reward기하로
종료기준을몰래강화하지않으며 속도penalty나추가Blocker gate도없다. Helper는기존출력에
`stability_distance`·`tilt_angle`두필드를추가하고 TaskState에한번복사한다. 같은step의
반복rewardquery는재계산하지않고 부분reset은선택행만초기화한다.

앞선Target보호gate·초기Blocker분모·F/T분리·Time복원과12개native reward·7개종료·
성공bonus250·물리/PPO/관측은유지한다. 이수정으로장기학습개선을확인한것은아니며
실행/회귀검증의범위는TRAINING_LOG최신적용기록을따른다.
