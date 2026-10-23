# 물체 배치 생성 규칙

최종 갱신: **2026-10-22, Asia/Seoul**.

## 현재 변경 요구와 구현 상태 — 15–25개 통합 clutter

사용자는 역할 물체와 나머지 물체가 별개 구획으로 보이는 이전 두-row 배치를 지적했다.
목표는 **전체 물체가 하나의 랜덤 clutter를 이루면서 Target·Blocker의 가시성과 조작 조건을
만족하는 배치**다. 역할 물체의 위치·yaw와 주변 물체의 위치·yaw를 공동으로 정해야 하며,
주변 물체를 한쪽 잔여 영역에만 몰아넣는 규칙은 새 요구를 충족한 것으로 보지 않는다.
기본 모드는 `generation_mode="clutter"`이며 현재 생성 범위는15–25개다. 이전 최대40개 설정에서
90개 종류 조합과 어려운8구성의 standalone 생성,1,024환경 native reset·10step을 확인했다.
해당 검사에서 실제 활성 개수는20–38개였다.
이는 모든 종류·각도 조합의 수용 가능성 증명이나 정규 학습 iteration 속도 측정은 아니다.
아래10물체 이력의 검증 수치를 새 규칙의 증거로 사용하지 않는다.

| 항목 | 최신 확정 요구 |
|---|---|
| 활성 물체 수 | 같은 선반에서15–25개 범위, 구성과 공간 수용 가능성에 연동 |
| 종류 구성 | 기존10종 중4–10종 유지; Target 종류는1개, 선택된 non-Target 종류별 개수 차이≤1 |
| 구성 보존 | 큰 물체나 특정 종류를 제외해 개수를 채우지 않음; 공간에 맞는 허용 개수를 결정 |
| 관측 용량 | 실제 물체 수와 독립적으로10개 slot, 전체244D 유지 |
| 관측 선택 | Target·Blocker는 slot0/1; 인식된 나머지는 두 역할까지의 최소 유클리드 거리 순서로 최대8개 |
| 거리 기준 | 초기 vision이 추정한 OBB 중심 사용; 보이지 않은 물체의 정답 중심으로 선택하지 않음 |
| 가시성 | Target40–90%, Blocker≥95%와 기존 화각·최소 픽셀 조건 유지 |
| yaw | 같은 주변 종류·개수 구성에서도 각 ordered pair의 두 역할 각각360° 가능성 유지 |
| 재배치 공간 | Blocker를 옮길 수 있는 빈 목적 영역을 확보; 이동·pivot 전체 경로는 예약하지 않음 |
| 생성 방식 | 역할·가시성 연속 영역 → GPU seed → 단일 공동 제약 계산 → 고정3회 위치 mixing; 무작위 장면 승인 재시도 없음 |

90개 ordered pair는 종류10개에서 서로 다른 Target·Blocker를 고르는 `10×9` 조합이다.
완성된90개 장면을 저장한 목록이 아니며 각 조합 아래 연속적인 위치·yaw·시점 변수를 갖는다.
새 통합 배치가 사용할 연속 영역과 공간 제약은 다시 검증해야 한다.

사용자는 전체 이동 경로를 비워 두는 대신 **빈 재배치 공간 하나만 확보**하는 방향을 확인했다.
이 공간은 Blocker가 도착할 수 있는 목적지이며, 초기 위치부터 그곳까지 충돌 없는 연결 경로가
있다는 증명은 아니다. 주변 물체는 기존 경로 envelope 안에도 배치될 수 있고 조작 중 추가 접촉·
pivot·여러 단계의 재배치가 필요할 수 있다. 아래 이전 구현의 full-path witness와 guard 인증은
이력이며 새 통합 clutter의 성공 가능성을 보증하는 근거로 사용하지 않는다.

### 현재 구현된 계산 흐름

1. 기존 reference10 catalog에서는 **역할 두 물체의 상대 pose·시점·초기 가림 관계**만 재사용한다.
   같은 구성에서 두 역할의360° support 요구는 유지하지만, 이전 두-row 배경 예산·전체 경로·
   10물체 공간 인증을15–25개 장면에 적용하지 않는다.
2. 종류 집합과 균형 중복 순서를 먼저 정한다. Target 종류는1개이며 선택된 non-Target 종류는
   개수 차이≤1인 prefix를 이룬다. 주변 물체 하나의 yaw는 전체360°에서 뽑고 나머지는 얕은
   선반에 맞는 caliper 방향·180° 뒤집기·작은 연속 변화 및90° 대안을 함께 사용한다. 모든 방향의
   최종 공간 가능성이 인증됐다는 주장은 아직 하지 않는다.
3. `_clutter_gpu_seed`가 실제 collider의 보수적인 cell mask와 float64 FFT로 각 물체의 전체
   비충돌 위치·방향 영역을 계산해 한 번 선택한다. 필수13개 BG는 큰 형상부터 처리하되 원래
   ID를 유지하며, 넣지 못한 필수 행은 NaN인 **부분 graph**로 남긴다. 선택 가능한 추가 물체는
   균형 순서를 유지하고 첫 실패에서 멈춘다. 이 seed 자체가 승인된 장면은 아니다.
4. `_solve_clutter_joint_domain`의 **한 번의 CPU 공동 제약 계산**으로 필수 BG 전체, 수용 가능한
   추가 prefix와 빈 Blocker 목적지를 함께 결정한다. 배경의 XY 위치를 움직일 수 있으며 목적지는
   초기 시각 보호 영역과 겹쳐도 된다. 실제 T/B·A₀·선반 지지 및 배경과의 비충돌은 유지한다.
   Solver 내부의 최적화 탐색이 없는 계산이라는 뜻은 아니며, 무작위 장면을 생성·승인·재생성하는
   loop와 구분한다. 실패 시 역할·큰 종류·필수15개를 조용히 바꿔 통과시키지 않는다.
5. 확정한 형상을 `_mix_clutter_positions`의 고정3회 조건부 X·Y 추출로 섞는다. Collider의
   inset 선반 경계와 전체 OBB의 허용 panel 경계를 모두 반영하며 yaw·종류 구성·빈 목적지는
   유지한다. 마지막에 실제 메시의 보이는 표면을 한 번 관측하고 Target·Blocker와 가까운8개를
   골라10-slot 정책 관측을 만든다. Reset의 시점 승인·장면 재생성은 없다.

이 계산은 공유 scene 하나에만 수행한다. Clone별 물리·로봇 초기 IK는 계속 분리되며, 중간 종료는
같은 scene·초기 vision으로 복구한다. 실제 카메라·render product는 만들지 않는다. 위 공동 제약은
GPU seed 뒤 CPU에서 처리하므로 전체 생성이 GPU-only라고 설명하지 않는다.

구성한 유효 seed의 주변 위치를 섞는 `_mix_clutter_positions`는 별도 CPU 검증을 마쳤다.
고정3회 sweep에서 물체별 X·Y 좌표의 collision-free 구간 합집합을 정확히 계산한 뒤 직접
샘플링한다. 유효성 승인 재시도나 반복 수렴 검사는 없으며 yaw·종류 구성을 유지한다. 6개 회귀
검사와140개 기하 장면·171개 기존 해석식 비교를 통과했다. 40물체 전체 준비·240좌표 draw의
CPU 평균은4정점5ms·64정점14ms·500정점79ms였다. 이 값은 공유 scene helper의 비용이며 새
배치 생성기 전체, 실제 메시 가시성 또는 native 학습 성능 검증이 아니다.

Production GPU seed는 CUDA 회귀6개와 어려운8장면의 필수18개 BG를 통과했다. 이8장면에는
180°·±.001rad 변화와 한 주변 물체의 자유 yaw를 넣고 collider/전체 OBB 지지를 별도로 검사했다.
실제 최소 pair·guard 간격은약3.20mm, warm 비용은약110–120ms, 최대 CUDA 할당은약480MB였다.
이 결과는 seed 생성의 검증이며 최종 장면 검증과 구분한다.
역할 초기 영역의90개 ordered pair 검사에서는 normal6,592 family와 중복 제거한103,513 cell의
두 역할360° coverage와 camera lateral±0.35m 경계를 확인했다. 이것도 주변15–25개 전체의
수용 가능성을 증명하는 검사는 아니다.

새 sampler의90개 ordered pair와 어려운 큰 물체4종 구성8개, 총98개 standalone 장면이 독립
기하 검사를 통과했다. 활성 개수20–38개·종류4–10종이며90개 표본의 warm 생성 시간은
중앙값0.343초·95백분위0.574초였다. 이 수치는 공유 장면 하나의 생성 비용이다.

이전 최대40개 설정의 native 검사는4환경·30step과1,024환경·10step을 완료했다. 두 검사 모두24개 물체·10종 장면이며
실제 메시의 독립 가시성은 Target68.85%·Blocker100%로 framing·최소 pixel 조건도 통과했다.
1,024환경의 정책 관측은 `[1024,244]`, 물리 ID는 `[1024,40]`, geometry는 `[1024,10,10]`이다.
부분 reset은 다른1,023환경을 bit-exact하게 유지하며 장면·관측을 추가하지 않았다. 초기 특이점·
비유한 상태·종료·fallback은0이었다. 종료 시 GPU used10.84GB/free14.45GB, host RSS9.54GB였다.
이는 짧은 실행 검증이며 장기 학습 안정성이나 정규 iteration 처리량 검증은 아니다.

`CONTACT_MANIPULATION_NUM_OBJECTS`는 물리 최대 개수이며 기본25이다. 기존 최소4종·균형 중복과
임의 종류 선택을 유지하려면 종류별 `ceil((25−1)/3)=8`개, 전체80개의 rigid-body pool을 처음에
준비해야 한다. 활성15–25개만 선반에 놓고 나머지는 주차하며 reset에서 새 USD를 spawn하지 않는다.
이 값은 observation 용량이나 모델 `max_objects=10`을 바꾸는 설정이 아니다.

현재80개 pool은 최대30개 설정의100개 pool보다 환경당20개 줄어든다.
이 비용을25개의 실제 활성 물체만 생성하는 비용과 혼동하지 않는다. 공식 `MultiAssetSpawner`는
초기 생성 시 asset을 고르므로 episode마다 종류 구성을 바꾸는 재사용 pool을 대신하지 않는다.
2026-10-23 `set_disable_simulations` 적용 후 장면 변경 회귀에서 활성 물체의 GPU pose 불일치가
재현되어 live OFF/ON을 제거했다. 미사용 물체는 기존 격리 주차 위치에서 native kinematic
body로 유지하고 활성 물체만 dynamic으로 전환한다. Native actor·GPU view·collider와
simulation/gravity flag는 유지하며, 중력 상쇄 외력은 사용하지 않는다. Kinematic 상태는
중력 적분과 dynamic solver 처리를 줄이지만 collider·broadphase·transform 관리와 메모리
비용까지 제거하지 않는다. 전체 공유 장면 commit에서 active mask가 바뀔 때만 native USD
kinematic 속성을 갱신·flush하고 그 뒤 pose·velocity를 기록한다. 같은 mask를 쓰는 부분
reset은 USD 변경·flush 없이 선택 행의 상태만 복구한다. 후보 준비는 로봇 FK만 임시 검증·
복원하므로 물체 상태·활성 mask나 이웃 환경을 변경하지 않는다.
과거130-body source와4·1,024환경 reset·step 검사는 중력 상쇄를 사용하던 시점의 기록이다.

바닥은 기존 native 전역 GroundPlane을 사용하고 환경 ID bounds encoding은 native cloner의
4bit 설정으로 유지한다. 앞서 시험한8bit override와 환경별 Cuboid 바닥은 사용하지 않는다.
Reset에서 actor·shape·physics scene의 topology는 바꾸지 않는다. 배치 규칙·스폰 높이와
100Hz 물리·정책 주기도 유지한다. 검증·미확정 범위는 TRAINING_LOG의 최신 기록을 참고한다.

Pool 확대 시 각 종류 첫 body의 주차 높이가 달라져 geometry 값이 약1e−16 변하고 cache identity가
달라지는 문제가 있었다. 첫3개 복사본을 기존30-body pool의 주차 slot에 두고 추가 복사본을 뒤에
놓는 순열로 고쳤다. 130개의 주차 위치 집합 자체는 같고 실제 활성 물체·state writer는 변경하지
않는다. Hash 반올림·형상 근사 없이 native source가 기존 `b29c255…`와 정확히 일치했으며 역할
cache를 읽기 전용으로 로드했다. 이것은 새 clutter의 공간 용량 인증을 재사용한다는 뜻이 아니다.

## 이전 10물체 연속 영역 직접 생성 경로 — `parameterized`

**변경 전 검증 경로:** `ShelfManipulationEnvCfg.spawn_rules.generation_mode="parameterized"`다.
전체90 ordered pair의 연속 yaw 검사, 실제364장면의 기하·광선 검증과1,024환경 native
robot-start/reset 검증을 통과했다. 공식 학습 진입점에서도26 iteration·936 step을 완료했고
800-step 장면 전환을 확인했다. 이 절은20–40개 통합 clutter 변경 전의 실행 계약을 설명한다.
연구 요구와 승인 이력은 [CONTEXT.md](CONTEXT.md#2026-10-22--요구-재확인-후보-선택-없는-순차-생성)에 둔다.

### 물체와 분포

| 항목 | 새 경로의 기준 |
|---|---|
| 활성 물체 수 N | 기본10개, 설정 증가 시 공간·pool 용량을 startup에서 다시 인증 |
| 종류 수 K | 10종 중4–10종, K≤N |
| Target | 해당 종류를 정확히1개 사용 |
| non-Target | 중복 허용, 선택된 종류별 개수 차이≤1 |
| 밀도 | 20% 점유율 하한 대신 총 개수와 가능한 가까운·넓은 간격을 함께 지원 |
| 초기 yaw | 같은 주변 종류·개수 구성에서도 두 역할 각각360° support; 빈도·변수 간 독립성은 요구하지 않음 |
| 간격 | collision footprint 사이 최소3mm, 가까운 간격을 선호하며 남는 공간의 넓은 간격도 허용 |
| Target 가시성 | 같은 자세의 비가림 실루엣 대비40–90% |
| Blocker 가시성 | 같은 기준으로95% 이상 |

종류별 `ceil((N−1)/3)`개의 물리 body를 최초 생성하므로 N=10의 pool은30개다.
Reset에서는 이미 있는 body의 좌표·자세·속도를 갱신하고 미사용 body를 주차한다.
관측은 활성 pool ID를 N개 slot으로 압축하며 Target/Blocker가 slot0/1이다. 비가시 물체는
padding하고 공개 관측 폭은 `10N+144`다. N 변경은 모델과 checkpoint 입력 계약도 바꾼다.

### 학습 시작 전: 유효 연속 영역의 계산과 캐시

1. `inspect_geometry`와 `visual_footprints`가 실제 authored collider와 시각 메시의 upright XY
   convex hull을 측정한다. 이 평면 envelope는 배치 계산용이며 PhysX collider를 교체하지 않는다.
2. `precompute_certified_role_family`가 Target/Blocker의 상대 yaw·시점 관계를 정하고,
   `precompute_translation_domain`이 실제 메시의 ray/triangle 교차 부등식에서 유효 Blocker 이동
   구간을 역산한다. 부분 가림을 만족하는 위치를 뽑아 승인할 때까지 반복하는 과정이 아니다.
3. 초기 A₀ 차단율, 선반 지지, 카메라 범위, 시각 보호 영역과 조작 경로를 연속 cell 전체에 대해
   인증한다. 병진 family 외에 `_precompute_normal_pivot_family`의 pivot 후 이동 경로와
   `_precompute_guarded_role_family`의 guard 관계를 사용할 수 있다.
4. `certified_catalog_capacity`가 모든 허용 종류·개수 구성의 공간 예산을 계산한다. 각 ordered pair의
   최악 예산을 normal family에 적용하며, guard를 뺀 주변 물체는 별도의 최악 residual 예산을 쓴다.
   주변 구성을 바꾸어 부족한 각도를 채우지 않는다.
5. `certified_role_yaw_gaps`가 **최종 공간 조건까지 통과한** guard 비의존 family들의 각도 합집합을
   검사한다. 모든 ordered pair에서 Target/Blocker 각각 빈 구간이 없어야 catalog를 승인한다.
   이는 각 yaw의360° support이며 두 yaw의 모든 독립 조합을 보장하는 것은 아니다.

`initialize_certified_catalog`는 관계식·연속 구간·CDF·가시성 fingerprint를 저장한다. 완성된 전체
scene을 저장해 재생하는 bank가 아니다. Asset 메시·bounds·pool copy 수·upright·생성 규칙·센서
설정이 달라지면 content identity가 달라진다. JSON payload checksum도 확인하며, 완료된 ordered
pair checkpoint부터 계산을 재개할 수 있다. 기본 cache 위치는
`outputs/contact_manipulation/parameter_domains`이며 `spawn_rules.parameter_cache_directory`로 바꾼다.

현재 전체 catalog의 최초 계산은 **33.68분**이 걸렸다. 이는 cache가 없을 때 한 번 수행한 측정이고
매 reset·iteration의 비용이 아니다. 같은 입력의 후속 실행은 저장된 catalog를 로드한다.

### Reset 시: 한 번 추출하고 순서대로 구성

1. `sample_certified_episode`가 Target·Blocker와 균형 중복 구성을 정한다. 필요한 guard 종류가 있는
   family를 포함해 해당 구성에서 사용할 수 있는 인증된 관계를 선택한다.
2. `sample_certified_scene`이 CDF에서 cell을 정하고 Blocker 이동량·공통 yaw·XY 이동을 한 번 추출한다.
   Target·Blocker·카메라와 해당 family의 guard/pivot 관계를 직접 구성한다.
3. 주변 물체는 고정된 두-row chain으로 구성한다. 각 row의 자유 slot은 전체360°에서 yaw를 뽑고
   나머지는 남은 공간이 보장하는 구간에서 뽑는다. 모든 slot이 자유 slot으로 선택될 양의 확률을
   가져 전체360° support를 유지하며, 각도의 빈도와 물체 간 각도는 서로 연동된다.
4. 회전된 실제 collision footprint의 Minkowski 경계에서 다음 물체의 위치를 직접 계산하고 row
   전체를 남은 영역으로 옮긴다. `spread_gaps=True`는3–10mm 기본 간격과 남은 폭 안의 넓은 간격을
   혼합한다. 넓은 간격 예산은 가까운 값에 치우친 분포와 Dirichlet 배분을 사용하며, row 사이 X
   간격도 남은 깊이 안에서 조절한다. 반환한 이웃 관계의 실제 간격을 certificate에 기록한다.
5. `observe_at_viewpoint`가 정해진 시점에서 한 번 관측한다. **37/74개 시점 승인, DP 분할,
   compaction, 배치 승인/교체/재생성은 이 reset 경로에 없다.**

실제 camera·render product·참조 물체는 만들지 않는다. GPU mesh ray casting은 활성 물체만
검사하고 로봇·선반 가림은 제외한다. 초기 관측 OBB는 첫 교차점으로 보이는 표면만으로 추정하며
가려진 부분을 정답 pose·전체 크기로 채우지 않는다. 새 경로의 `visible_fractions=None`은 reset에서
가시율을 재측정하지 않았다는 뜻이다. Cache의 예측 가시율은 별도 provenance이며 정책 입력이 아니다.

모든 환경은8초/800 policy step의 초기 장면·vision을 공유한 뒤 독립적으로 물리를 진행한다.
중간 종료한 환경은 같은 초기 장면으로 복원하고 hand 시작 자세만 연속적으로 randomize한다.
GPU IK·elbow-up·충돌·native FK·활성 singularity 조건 검사는 장면 생성과 별개로 유지한다.
부분 reset은 새 배치나 관측을 만들지 않는다.

Normal-pivot family의 회전 경로가 유효하다는 것과 회전이 반드시 필요하다는 것은 구분한다.
Guarded 부분 family는 초기 X/yaw의 순수±Y 이동으로는 A₀를 비울 수 없고 지지면 내 pivot→이동
경로는 유효하도록 인증한다. 일반 XY 병진 해법 전체가 불가능하다는 주장은 하지 않는다.

### 검증 범위와 구현 위치

현재 cache는90 ordered pair·6,593 family·207,027 cell이며 두 역할의 yaw 빈 구간은 모두0이다.
별도 실제 메시 검사에서360장면(그중 normal-pivot12개)과 guarded4장면이 기하·가시성 조건을
통과했다. 이 유한 표본 검증과 연속 구간 인증은 서로 다른 근거다.

1,024환경 native 검증에서 초기 자세의 활성 singularity 기준 위반은0행이었고 fallback도0회였다.
Reset 측정은 최초1.960초, 새 bucket을 준비한5환경 부분 reset0.265초, warm 전체 reset0.117초였다.
이 수치는 reset 통합 시험이며 전체 학습 iteration 시간은 아니다. 공식 학습 진입점의
`2026-10-22_17-19-09_parameterized_default_smoke`는26 iteration·936 step과800-step 장면 전환을
완료했다. 두 장면의 초기 자세 위반과 reset fallback은0이었고 actor·critic·optimizer 값은 모두
유한했다. Cold capture를 포함한 학습 시간은108.254초였으며 장기 안정성 검증과는 구분한다.
학습 처리량·장기 안정성 결과는
[TRAINING_LOG.md §43](TRAINING_LOG.md#43-2026-10-22--연속-파라미터-구간과-직접-생성-경로)에 구분해 기록한다.

| 구현 | 관련 API |
|---|---|
| [events.py](../src/contact_manipulation/contact_manipulation/mdp/events.py) | `initialize_certified_catalog`, `certified_catalog_capacity`, `sample_certified_episode`, `sample_certified_scene`, `reset_scene` |
| [sensors.py](../src/contact_manipulation/contact_manipulation/mdp/sensors.py) | `precompute_translation_domain`, `visual_footprints`, `observe_at_viewpoint` |
| [scene.py](../src/contact_manipulation/contact_manipulation/config/ur5e/scene.py) | `inspect_geometry`, pool·native asset·FK/IK 준비 |
| [환경 README](../src/contact_manipulation/README.md) | 모델·MDP·학습 흐름·실행 |

## 이전 Contact Manipulation `constructive` 경로 — 비교 기록

아래는 DP·compaction·고정 시점 공동 영역을 평가하던 구현의 설명이다. 비교할 때
`env.spawn_rules.generation_mode=constructive`로 명시한다. 아래37/74개 시점 승인과3–10mm 이웃 상한은
현재 `parameterized` 경로의 규칙이 아니다. 이후 Contact Rearrangement의 2026-10-10 절은
더 이전 프로젝트의 기록이며 렌더 카메라·35% 밀도·재시도 수치도 현행 연구 기준이 아니다.

### 물체 구성과 용량

| 항목 | 이전 `constructive` 기준 |
|---|---|
| 활성 물체 수 N | 기본10개, 추후 증가 가능 |
| 종류 수 K | 현재10종 중4–10종, K≤N |
| Target | 해당 종류를 정확히1개 사용 |
| non-Target | 중복 허용, 선택된 종류별 개수 차이≤1 |
| 전체 점유율 | 20% 하한을 강제하지 않음 |
| 초기 yaw | 물체마다 전체 `[-π, π]` 범위에서 한 번 선택 |
| 가까운 이웃 간격 | 초기 yaw의 convex collision footprint 사이3–10mm |
| Target 가시성 | 같은 자세의 비가림 실루엣 대비40–90% |
| Blocker 가시성 | 같은 기준으로95% 이상 |

non-Target K−1종에 `q=(N−1)//(K−1)`개를 배정하고,
`r=(N−1)%(K−1)`종에 한 개씩 추가한다. Target 종류는 나머지 슬롯에 다시 배정하지 않는다.
`CONTACT_MANIPULATION_NUM_OBJECTS`는 최초 topology 구성 전에 읽으며 기본값은10이다.
최소K=4를 지원하도록 종류별 `ceil((N−1)/3)`개를 최초 생성한다. 따라서 기본 물리 후보 pool은
P=30, 활성·관측 슬롯은 N=10이다. 미사용 후보는 기존 parking과 native 중력 상쇄를 사용한다.
Reset은 기존 body의 state tensor를 쓰며 매번 새 물체를 spawn하지 않는다.
중력 상쇄는 기존 `permanent_wrench_composer` 경로를 유지한다. 설치된 tensor API에
`get/set_disable_gravities`가 있지만, 이전 direct GPU 검사에서 getter가1을 반환해도 물체가
실제 중력으로 낙하한 기록이 있어 live gravity flag 전환으로 대체하지 않았다.

물리 pool ID와 관측 slot은 구분한다. 활성 pool ID를 관측 슬롯으로 압축하고 Target/Blocker를
slot0/1로 정렬하며 비가시 물체는 padding한다. 공개 observation 폭은 `10N+144`다.
N 변경은 모델·Critic 입력과 checkpoint 계약도 바꾸며 모든 종류 구성의 공간 수용 가능성을
자동으로 보장하지 않는다.

### 한 seed에서 배치를 직접 구성하는 순서

1. Target·Blocker와 균형 중복 구성을 한 번 정하고 각 초기 yaw를 선택한다.
2. 그 yaw의 collision footprint에서 Target 앞의 Blocker와 pivot→병진 witness를 계산한다.
   ±Y 두 방향의 작업 구획을 모두 계산한 뒤 수용 가능한 유한 영역의 합집합에서 선택한다.
   정점 궤적의 좌표 extrema로 작업 구획을 예약한다.
3. 작업 구획 양옆의 주변 물체 영역은 ordered slicing dynamic programming으로 계산한다.
   가능한 분할의 폭·깊이 Pareto frontier를 유한하게 산출하며 임의 후보 수 제한을 두지 않는다.
4. 수용 가능한 구획의 위치 여유를 랜덤화한 뒤 두 방향에서 각각 한 번 compaction한다.
   Convex Minkowski 경계와 원호·선분 교차식을 사용해 위치를 직접 계산한다.
   이미 배치한 물체와 아직 이동하지 않은 물체를 모두 장애물로 반영한다.
5. 기존 주변 물체1–2개로 초기 X·yaw를 유지한 순수±Y 탈출을 모두 막을 수 있는 guard 영역을
   해석적으로 계산한다. 영역이 있으면 base/guarded mode를 함께 준비하고 없으면 base만 준비한다.
6. 각 mode와 고정 시점의 공동 영역을 실제 메시 ray casting으로 한 번 평가하고 유효 쌍을 선택한다.
   Guarded의 유효 쌍을 우선하며 선택한 mode의 물리 pose와 초기 관측을 함께 확정한다.

조건을 통과할 때까지 임의 위치·구성·yaw를 다시 뽑는 과정은 없다. 공간 영역이 비어 있으면
해당 구성과 필요한 공간을 보고한다. 실패한 seed를 조용히 다른 장면으로 교체하지 않는다.
구획 계산은 모든 환경이 공유하는 장면 하나에만 CPU에서 수행한다. Physics step이나
1,024개 clone마다 독립 실행하는 계산이 아니다. 기존 sampler는 `generation_mode="legacy"`로
명시한 회귀 비교에만 사용한다.

### Clutter의 형상과 간격

최초 초기화에서 합성된 USD의 활성 authored collider를 upright 좌표계로 측정하고 XY convex
hull을 종류별로 캐시한다. 초기 yaw를 먼저 정하므로 각 물체의 미래360° 회전 공간을 일괄
예약하지 않는다. 3–10mm는 가깝게 만든 이웃의 목표 간격이며 모든 물체 쌍을10mm 안에
넣는 조건은 아니다. Blocker의 이동·pivot 공간은 별도로 비워 둔다.

이 hull은 PhysX collider를 교체하지 않는다. Concave/compound collider의 구멍·빈 공간을
메우는 보수적 평면 envelope이며 PhysX가 cook한 접촉 표면이나 contact offset과 같은 개념이
아니다. 상세 정점 간 Cartesian product 대신 edge-angle 병합으로 Minkowski 경계를 만들고,
거리 계산의 정점 반복은 compiled CPU kernel에서 수행한다. 초기 비중첩·예약 경로와 실제 접촉
조작 가능성, 그리고 경로에 회전이 있다는 것과 회전이 반드시 필요하다는 것은 구분한다.

Guard branch는 `clearance_kind="guarded_pivot_translate"`로 표시한다. 이 branch는 초기 X·yaw에서
순수±Y 양쪽 출구가 clearance margin0에서도 막히도록 하고 pivot→병진 경로는 유지한다.
일반 XY 병진이나 다른 비회전 경로 전체가 불가능하다고 주장하지 않는다. Guard가 없는
`pivot_translate` mode도 같은 유한 영역에 남긴다. Guarded가 실제 메시 가시성을 만족하면
우선 선택하므로 두 mode를 균등하게 뽑는 분포는 아니다. 생성 비율과 검증 수치는 별도로 관리한다.

Guard의 경계는 정점 원호의 해석적 support와128개 고정 법선으로 만든 보수적 이동 envelope다.
따라서 guard 물체가 이전 packing용 직사각형 안에 들어가더라도 실제 witness를 침범하지 않을 수 있다.
이 경우 `reserved_regions=()`는 그 큰 빈 직사각형을 주장하지 않는 metadata이며 이동 경로의
예약을 제거한다는 뜻은 아니다. 일반 branch는 기존 직사각형 reserve를 기록한다.

### GPU에서 한 번 평가하는 가상 시점 영역

실제 Camera·render product·reference 물체는 만들지 않는다. 기존 실제 시각 메시 BVH를
공유하고 물체들만 ray casting한다. 선반·로봇의 가림을 제외하는 관측 모델은 유지한다.

각 기하 mode의 시점 영역은 앞쪽 거리2개×좌우9개×높이2개, 총36개이며 배치 규칙의 시점1개를
더한다. 일반 mode만 있으면37쌍, base/guarded가 있으면 최대74개의 mode·시점 쌍을 모두 평가한다.
거리0.55–0.95m, Blocker 기준 좌우±0.35m, 선반 위0.18–0.48m를
사용하고 방향은 두 역할을 바라보도록 계산한다. 고정 chunk는 메모리 상한을 위한 분할이며
실패 후 후보를 추가하는 재시도 단계가 아니다. GPU에서 모든 결과를 계산한 후 유효한 guarded
mode를 우선하고, 그 mode에 유효 쌍이 없으면 함께 평가한 base mode를 고른다. 한 mode 안에서는
Target 가시율이65%에 가장 가까운 시점, Blocker 가시율, index 순서로 정한다.
선택한 depth·label에서 visible-surface OBB를 추정하고 그 mode의 pose·기하와 함께 commit한다.
다른 mode의 관측을 현재 물리 배치에 붙이거나 실패 후 새 배치를 생성하지 않는다.
광선·가시율·mode별 시점 선택·OBB는 GPU에서 계산하며 최대2개 mode의 우선순위와 metadata는
CPU에서 정리한다. CPU로 픽셀·메시를 가져와 가시성을 다시 계산하는 경로는 아니다.

160×120 인증 광선과 같은 간격의 확장 격자에서 두 역할의 비가림 실루엣을 계산한다.
유효 조건은 Target40–90%, Blocker≥95%, 전체 실루엣 화각 포함, 역할별 비가림32픽셀 이상·
실제 관측16픽셀 이상이다. Target의 추가 가림에는 주변 물체도 포함한다. OBB 겹침 비율로
실제 메시 가시성을 대체하지 않는다. 가시율과 픽셀 수는 이상적인 ray 표본 기준의 근사다.
실패 시 조건별 counts·도달 가시율을 기록하며 배치를 다시 생성하지 않는다.

고정 영역 평가가 모든 역할·구성·yaw에서 유효한 시점을 보장한다는 뜻은 아니다. 회전이
반드시 필요한 장면의 생성 비율과 실제 메시 가시성의 전체 검증은 별도로 관리한다.
현재 기하·센서 검증 결과는 [TRAINING_LOG.md](TRAINING_LOG.md)의 후속 기록을 따른다.

2026-10-23 관측 해상도 조정: 기본 clutter reset에서 정책 observation을 생성하는 광선은
별도64×48(3,072개), 최소 유효 표면점3개를 사용한다. 위160×120·32/16pixel은 사전 배치
가시성 인증용으로 유지한다. Observation 전용 해상도·점 수는 인증 cache key에 포함하지
않으므로 기존90-pair catalog를 다시 만들지 않는다. Runtime은 정해진 시점에서 관측1회만
생성하고 낮은 해상도에서 인식되지 않은 물체는 invalid/padding으로 남긴다. 물체·배치·가시성
규칙과10-slot 관측 선택은 바꾸지 않는다.

### Reset과 구현 위치

전체 환경이8초 scene round의 초기 배치·vision을 공유한다. 중간 종료 시 해당 환경만 초기
상태로 복원하고 hand의 위치·방향은 새로 샘플링한다. GPU IK·elbow-up·충돌 검사와 예외 행의
동일 장면 기준 자세 복구는 배치 생성과 별개이며 이번 변경으로 제거하지 않는다.
구성적 새 scene은 실제 물리 FK·elbow·충돌을 검증한 anchor를 기준으로 확정한다.
임의6D목표8개 중4개를 요구하던 추가 domain probe gate는 구성적 경로에 적용하지 않고
legacy 회귀 비교에만 유지한다. 연속적인 초기 pose randomization과 실패 행 복구는 유지한다.

| 구현 | 역할 |
|---|---|
| [scene.py](../src/contact_manipulation/contact_manipulation/config/ur5e/scene.py) | 가변 pool·authored collision hull 캐시·native 생성 |
| [events.py](../src/contact_manipulation/contact_manipulation/mdp/events.py) | 역할·균형 구성·slicing·해석적 compaction·reset 연결 |
| [sensors.py](../src/contact_manipulation/contact_manipulation/mdp/sensors.py) | GPU 고정 시점 영역·실루엣·보이는 표면 OBB |
| [환경 README](../src/contact_manipulation/README.md) | 모델·MDP·학습 흐름·실행 |

## 이전 Contact Rearrangement 기준 — 2026-10-10 기록

최종 갱신: **2026-10-10, Asia/Seoul**.

현재 샘플러는 **episode의 Target·Blocker를 먼저 정한 뒤, 선반 전체의 가용
영역에 4–10종을 선택하고 선반 점유율에 필요한 수의 instance를 배치하는 초기
장면**을 만든다. **종류 수와 물체 개수는 별개**다. Target 종류는 장면에 정확히
한 번만 등장한다. 나머지 선택된 종류는 밀도를 충족하기 위해 중복할 수 있고,
별도 옵션으로 10종을 모두 선택할 수 있다. 해소 경로는
병진·회전·모서리 pivot 및 여러 단계의 조합을 고려한다.

이전 구현은 사용자의 4–10종 기준을 active instance 4–10개로 잘못 해석했다.
현재는 전체 선반과 각 구역의 점유율을 반드시 만족하도록 개수를 산정한다.
25–45% global 요구를 비교해 **global 35%·각 구역 10%**를 초기 구현 기본값으로
선택했다. 이후 같은 종류의 중복이 편중되는 문제를 수정해 **선택한 non-target
종류의 active 개수를 균형 있게 늘리는 구성 방식**으로 변경했다. 이전 면적
우선 구성의 검사 결과를 새 구성의 성공률·instance 분포로 사용하지 않는다.
Blocker 앞 reserve·camera gate 추가 전 균형 구성의 기하 sample 40개와 일반
모드 4개 GPU 환경·151 step 검사가
통과했다. 연구의 최종 난이도 수치는 별도이며 검증 범위는 11절에 기록한다.
2026-10-09에 책임별 6개 모듈로 통합한 뒤에도 아래 규칙과 기본값을 유지했다. 통합 후
23개 unit test와 일반 모드 4개 GPU 환경·151 step 검사가 통과했으며
기존 균형 구성 실행의 초기·reset 뒤 8개 장면 설명과 정확히 같았다.

앞쪽 camera를 무작위로 제안해 실제 같은 pose의 비가림 silhouette 대비 Target
**40% 이상**, Blocker **전체**가 보이는 초기 snapshot을 승인한다. Blocker에는
최대 2 pixel의 면적 손실만 허용한다. 현재 Blocker 앞의
충돌 없는 elbow-up hand 시작을 randomize하고 tool 전방은 선반 방향인 world
−X로 정한다. 2026-10-09의 24개 시작 pose와 반복 reset·5초 물리 검사는 이전
아래를 향하던 tool orientation의 결과다. 이번 방향 교정의 검사와 구분한다.
선반 방향 수정 뒤 palm normal이 위를 향하던 자세의 4개 GPU 환경·151 step·
13개 시작 pose 검사도 통과했다. 후속 요청으로 physical tool +Z 기준 hand
mount에 −90° roll을 추가해 현재 palm normal은 world −Y를 향한다.
새 mount도 4개 GPU 환경·151 step·13개 시작 pose 검사를 통과했다.
이후 센서·연결부 공간을 고려해 wrist-3와 hand 사이에 길이 5 cm·직경 5 cm의
원통을 추가하고 hand anchor를 tool +Z로 5 cm 옮긴다. 원통 추가 전 결과와 새
조립의 검사를 구분한다.
새 연결부도 4개 GPU 환경·151 step·13개 시작 pose 검사를 통과했다.
이후 중앙 F/T를 위해 원통을 두 rigid half로 나누고 raw sensor-frame wrench를
읽는다. 현재 41 body·18 DOF·41 robot collider이며 센서 단계의 검증은 이전
원통 geometry 단계와 구분한다.
현재 17개 tactile link에 독립 ContactSensor를 붙이고 world net normal force
norm >0.01 N을 bool 17채널로 관측한다. 해당 실제 GPU 접촉·reset 검사도 통과했다.
로봇은 episode에 선택한 관절 목표를 유지한다.
아래 길이·비율은 스폰 구현 기본값이며 연구의 최종 난이도
명세나 물리 제어 성공의 보장이 아니다.

연구 배경과 변경 이유는 [CONTEXT.md](CONTEXT.md), 실행 방법은
[환경 README](../src/contact_rearrangement/README.md)에 있다. 이 문서는 현재
[src/sampling.py](../src/contact_rearrangement/contact_rearrangement/src/sampling.py),
[mdp/events.py](../src/contact_rearrangement/contact_rearrangement/mdp/events.py)의 USD 측정·reset,
[src/assets.py](../src/contact_rearrangement/contact_rearrangement/src/assets.py)의 pool 처리를 바탕으로
판정과 제안 분포를 구체적으로 설명한다.

![35% 전체 선반 점유율의 종류 수와 instance 수 예시](assets/dense_shelf_examples.svg)

그림은 **Blocker 앞 reserve·camera gate 추가 전 균형 구성**으로 만든 일반 seed 0의 **4종·12개**, 일반 seed 2의
**4종·25개**, all-types seed 1의 **10종·17개** 초기 장면이다. Target은 하나이며
non-target 종류별 개수는 각각 **3/4/4**, **8/8/8**, **1/1/2/2/2/2/2/2/2**다.
세 장면 모두 현재 전체·구역 점유율과 clearance 조건을 통과했다.
파랑은 Target, 주황은 Blocker, 회색은 주변 물체, 하늘색은 A₀다.
보라 영역은 예약한 경로이며 주황 점선은 검증용 마지막 footprint다.
로봇이 경로를 실제 실행한 결과나 가시성 40%가 확인된 장면은 아니다.
[이전 전체 선반 경로 예시](assets/full_shelf_clearance_examples.svg)는
종류·instance 분리와 hard occupancy를 추가하기 전의 이력이다.

이전 [seed 0 배치도](assets/spawn_rules_seed_0.svg)는 4–6개를 좁은 cluster에
놓던 baseline의 이력이다. 현재 전체 선반 분포·회전 경로의 예시로 사용하지 않는다.

## 1. 물체 종류, instance, 역할

Catalog에는 기존 shelf 물체 6종과 YCB physics 물체 4종, 총 **10종**이 있다.

| 구분 | Asset 종류 이름 |
|---|---|
| 기존 물체 | `bottle_1`, `cup_1`, `cup_2`, `mug_1`, `mug_2`, `can_1` |
| YCB | `ycb_003_cracker_box`, `ycb_004_sugar_box`, `ycb_005_tomato_soup_can`, `ycb_006_mustard_bottle` |

YCB는 `NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics`의
authored rigid-body·collision USD를 사용한다. Visual-only 폴더의 다른 종류가
자동으로 포함되는 것은 아니다. 현재 003 cracker box와 004 sugar box에는
스폰 시점의 flat-box collision override를 적용한다. 원본 파일은 수정하지 않으며
구체적인 처리와 검증 상태는 3절에 설명한다.

각 종류를 10번 복제해 **환경당 100개의 rigid-body pool**을 미리 로드한다.
첫 복사본은 종류 이름, 이후 복사본은 `_copy_02`부터 `_copy_10`까지의 이름을
갖는다. 선택하는 **종류 수는 기본 4–10종**이며 active instance 개수는 그 종류들의
footprint와 요구하는 선반 점유율로 산정한다. Instance 최소값은 4, 현재 safety
cap은 40이다. `MAX_ACTIVE_OBJECTS=40`이 종류당 복사본 수를 40으로 늘리는 뜻은
아니며 pool은 계속 10종 × 10-copy다.
**Target의 asset 종류는 active 장면에 정확히 한 번만 등장**하며, 같은 종류의
다른 복사본을 Blocker나 주변 물체로 사용하지 않는다. Target 이외의 종류는
중복 가능하다. 역할은 개별 active instance ID에 지정한다.

환경 reset event가 Target instance를 먼저 선택하고 그 **종류 전체를 제외한**
후 Blocker instance를 선택한다. `cfg.target_asset`·`cfg.blocker_asset`, 또는
CLI `--target_asset`·`--blocker_asset`에 종류 이름을 주면 해당 종류의 복사본 중에서
선택한다. 두 옵션에 같은 종류를 지정하면 Target 종류의 유일성 조건과 충돌하므로
오류다. 기본 `None`이면 유효 후보에서 episode마다 처음에 한 번 선택한다.
Target 종류를 전체 학습 동안 하나로 고정하려면 `--target_asset`을 사용한다.
존재하지 않는 종류도 오류로 처리한다.

역할을 먼저 정한 다음 `min_asset_types=4`, `max_asset_types=10` 범위에서 종류 수를
한 번 뽑고, 두 역할의 종류를 포함한 종류 집합을 선택한다. 선택한 모든 종류를
최소 한 번씩 사용하도록 instance를 정한다. 그 다음 Target 종류를 제외한 선택
종류의 복사본을 추가해 footprint 면적 합이 전체 선반의 요구 면적에 도달하도록
한다. **추가 가능한 non-target 종류 중 active 개수가 가장 적은 종류**를 먼저
고르고 동률은 무작위로 결정한다. 지정 Blocker도 해당 종류의 첫 instance로
센다. 모든 non-target 종류가 같은 개수에 도달하면 다음 회차를 시작한다.
현재 catalog는 종류별 pool capacity가 같으므로 선택된 non-target 종류의
active 개수 차이는 최대 1이다. Target 종류는 항상 1개다.

종류별 별도 고정 최대 개수를 새로 두지는 않는다. Pool의 종류당 10-copy와
전체 safety cap 40이 한계이며 작은 물체 위주 종류 집합은 같은 점유율에 더 많은
instance가 필요할 수 있다. 이전의 **큰 footprint 우선 복사본 선택**은 중복을
한 종류에 집중시키므로 폐기했다. 배치 단계에서는 packing을 위해 큰 footprint
instance부터 위치를 찾는다. **중복 종류 선택 순서와 공간 배치 순서는 별개**다.

```text
usable_area = (shelf_width_x - 2*edge_margin) * (shelf_width_y - 2*edge_margin)
required_occupancy = max(min_shelf_occupancy, min_region_occupancy)
required_area = required_occupancy * usable_area
Σ upright footprint area >= required_area 이고 N >= min_objects가 될 때까지 추가
```

각 구역의 최소 점유율이 global 요구보다 높아도 그 하한을 전체 면적 합에
반영한다. 기본 35%/10%에서는 요구 면적이 전체 usable 면적의 35%다.
Footprint 면적은 yaw로 바뀌지 않으므로 배치 전에 종류와 개수를 정할 수 있다.
복사본이 부족하거나 40개 cap 안에서 면적 요구를 충족하지 못하면 배치 시도 전
명시적으로 실패한다. 지역별 점유율·충돌·경로·이웃 조건은 실제 위치를 정한 후
별도로 검사한다. 면적 합만으로 적절한 배치가 존재한다고 확정하지 않는다.

**Target·Blocker 쌍, 종류 수·집합, active instance ID 전체와 N은 layout retry
전에 고정한다.** Retry에서는 모든 active yaw와 위치만 바꾼다. 역할을 바꾸거나
다른 종류·instance를 뽑아 난이도를 낮추거나 N을 줄이는 fallback은 없다.

`SpawnRules.all_asset_types=True`, CLI `--all_asset_types`이면 **catalog 10종을
모두 선택**한다. Target은 하나이고 나머지 종류는 점유율에 필요한 만큼 중복한다.
**10종 선택은 정확히 10개 instance를 뜻하지 않는다.** CLI는 종류 범위를 10으로
고정하며 instance cap은 최소 10 이상이어야 한다. 일반 모드와 동일하게 density로
N을 산정하며 기본 cap 40을 유지한다.

저장소 루트에서 실행하는 두 옵션의 예시는 다음과 같다.

```bash
# 10종을 모두 선택하고 density에 필요한 복사본을 함께 배치
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py --all_asset_types
# 정확히 6종을 선택하고 density에 필요한 instance 수를 산정
./IsaacLab/isaaclab.sh -p src/contact_rearrangement/scripts/spawn_scene.py --min_asset_types 6 --max_asset_types 6
```

## 2. 좌표계와 실제 선반 범위

모든 sampling 좌표는 개별 environment의 로컬 좌표다. 물리 상태를 쓸 때
각 환경의 `env_origins`를 더한다. 로봇은 선반의 +X 쪽에 있고 초기 접근 방향은
**−X**, 즉 Target에서 로봇 쪽을 향하는 접근영역의 길이 방향은 **+X**다.
X가 더 작으면 선반 뒤쪽, 더 크면 로봇 쪽이다.

선반에 아래 숫자를 무조건 하드코딩해 사용하는 대신, spawn 후 실제 collision
panel과 기둥의 bounds를 읽어 작업층을 정한다. 현재 기본 speedrack의 측정값은
다음과 같다.

| 항목 | 현재 기본 asset의 환경 로컬 범위 |
|---|---|
| Support X | `[-0.900, -0.500] m` |
| Support Y | `[-0.455, +0.455] m` |
| 작업층 상면 | `z = 1.050 m` |
| 바로 위 panel의 하단 | `z = 1.480 m` |
| 초기 footprint의 edge margin | 사방 **15 mm** |

따라서 footprint의 모든 꼭짓점이 들어가야 할 영역은 현재 기본값에서
`X=[-0.885,-0.515]`, `Y=[-0.440,+0.440] m`이다. 물체 중심만 선반 안에 있다고
통과시키지 않는다. 회전된 footprint 전체가 이 안에 있어야 한다.

`mdp/events.py`는 X 폭 25 cm 초과·Y 폭 50 cm 초과·두께 6 cm 미만인 collision
부분을 수평 panel 후보로 보고, 설정한 작업층 높이에 가장 가까운 상면을 고른다.
그 위의 겹치는 panel 중 가장 낮은 하단을 ceiling으로 쓴다. Y 범위는 작업층과
겹치는 가는 수직 기둥의 안쪽 경계로 줄인다. 해당 기둥을 찾지 못한 custom shelf는
그쪽 panel 경계에서 5 cm 안쪽을 사용한다. 적절한 panel이나 양의 지지 영역이
없으면 오류다.

`SpawnRules` 클래스의 X·ceiling 초기값과 spawn 후 측정값은 구별해야 한다.
Runtime에서는 측정한 shelf support와 ceiling이 해당 항목을 대체한다.

## 3. 물체 크기, 방향, pivot 보정

Spawn된 각 asset에 dynamic rigid body가 정확히 하나 있고, 그 아래에 enabled
collision shape가 있는지 검사한다. 크기는 body subtree의 USD bounds를 읽는다.
이때 body와 ancestor에 적용된 **scale을 포함**하고 world translation·rotation을
제외한 다음, catalog의 upright 회전을 적용한다. 양수인 finite extent가 없는
asset은 사용하지 않는다.

Bounds는 default/render/proxy geometry를 포함한 보수적인 box다. 실제 mesh의
빈 공간, 컵 내부, 손잡이 사이를 통과하는 경로까지 정확히 판정하는 collision
mesh 검사는 아니다. 작게 만든 asset을 원본 크기로 판정하거나, root pivot을
물체 중심으로 취급하지 않도록 scale과 offset을 함께 다룬다.

Upright 상태에서 bounds를 `(lower, upper)`라고 하면:

```text
hx = (upper.x - lower.x) / 2
hy = (upper.y - lower.y) / 2
cx = (upper.x + lower.x) / 2
cy = (upper.y + lower.y) / 2
root_z = shelf_top - lower.z
```

마지막 식은 물체의 가장 낮은 bound를 **선반 상면에 바로 맞춘다.** 이전의
2 mm 낙하 여유는 모든 종류에서 제거했다. Root Z를 모든 물체에 일괄 `1.05 m`로
지정하지 않는다. 또한
`root_z + upper.z <= ceiling - 0.003`이어야 하므로 위 panel과 최소 3 mm의
높이 여유를 두고, 높이가 맞지 않으면 해당 배치 시도를 버린다.

### YCB 003·004의 flat-box collision

두 scanned box의 초기 낙하·흔들림을 줄이기 위해
[src/assets.py](../src/contact_rearrangement/contact_rearrangement/src/assets.py)의
`FlatBoxUsdCfg`를 **003 cracker box와 004 sugar box에만** 사용한다. 기존의
visible scanned mesh는 유지하고 그 mesh의 collision을 끈 다음, 같은 single
rigid body 아래에 invisible `UsdGeom.Cube` collider를 추가한다.

Cube는 원본의 **unscaled body-local AABB와 같은 크기·중심 offset**으로 만든다.
Unit Cube의 size는 1, extent는 `[-0.5,+0.5]`로 명시한다.
Body와 ancestor의 scale은 상속하고 runtime geometry 검사에서 upright 회전과
함께 한 번 적용한다. 따라서 sampling footprint의 bounding 규모를 바꾸는
것이 아니라 박스 바닥의 실제 접촉면을 평평하게 만드는 override다. 이것은
원본 scanned mesh와 정확히 같은 collision 형상은 아니며, 두 박스의 동역학을
box proxy로 근사하는 선택이다.

처리는 `@clone` spawn 시점에만 수행해 PhysX view를 만들기 전에 모든 복제
환경에 동일하게 적용한다. 원본 Nucleus USD를 저장하거나 수정하지 않는다.
Body의 authored mass·관성 등 설정은 그대로 두고 원본 collider의 authored
`physxCollision:*` contact 속성을 가져오고, `UsdPhysics.MaterialAPI`를 가진
실제 physics material이 있을 때 그 binding을 proxy에 적용한다. Visual material을
physics material로 잘못 재사용하지 않는다.
Active body의 native gravity·simulation도 유지한다. 원본 mesh에
`boundingCube` approximation을 지정해 자동 fit하는 방식은 최종 구현으로
사용하지 않는다.

**현재 검증 상태:** 2 mm 여유만 제거하고 원본 collider를 유지한 4개 환경·
10종 전체·5초 진단에서 003은 최대 **66.96°·XY 11.3 cm**, 004는 최대
**3.80°·XY 5.79 mm**를 보였다. `boundingCube` fitted approximation 시험에도
**0.83°·XY 1.54 mm**의 움직임이 남아 최종 collider로 채택하지 않았다.
명시적 Cube proxy의 초기 GPU 진단은 4개 환경·10종 전체·5초 동안 8개 박스를
매 physics substep (120 Hz) 기록했다. 최대 XY 이동은 **0.007894 mm**였고,
float32 `acos` 계산 정밀도 안에서 거의 직립했다. 이를 수학적으로 정확한
0° 기울기 보장으로 읽지 않는다. 003 mass는 0.411 kg, 004는 0.514 kg이며,
진단한 박스 8개 각각에서 rigid body 1개·enabled Cube collider 1개와 원본 mesh collider off를
확인했다. 로그는 `outputs/contact_rearrangement/box_supported_proxy_diagnostic.{json,log}`다.

종류/instance 분리 전 Cube extent와 physics material guard를 보강한 뒤 공식 `spawn_scene.py` 실행도
**4개 환경·10종 전체·10초**에서 통과했다. 기본 150-step reset으로 8개 layout을
각각 5초 유지하고 전체/부분 reset을 검사했다. 5 cm drift·15° tilt guard는
완화하지 않았다. 마지막 300-step reset 뒤 생성한 새 layout은 물리 진행 시간이
0초이므로 추가 5초 정착 검증으로 세지 않는다. 새 높이 fixture를 적용한 unit
test 20개도 통과했다. 상세 명령과 조건은 [CONTEXT.md](CONTEXT.md)에 있으며,
이 결과를 다른 seed·실제 로봇 밀기·camera 가시성 검증으로 확대하지 않는다.

각 layout 시도에서 모든 active 물체의 yaw를 `uniform(-π,+π)`로 뽑는다.
Roll/pitch를 무작위로 기울이는 것은 아니며 upright 회전 뒤 world Z yaw를
적용한다. Quaternion은 `(w,x,y,z)`이고 최종 회전은 `yaw_Z * upright`다.

| 종류 | Upright 회전과 현재 확인 상태 |
|---|---|
| 기존 6종 | Catalog의 identity upright 회전, 이후 world Z yaw |
| YCB 003·004·005·006 | 4종 모두 X축 **−90°**. Local **−Y → world +Z**가 되며 이후 world Z yaw를 적용. 각 물체의 양쪽 렌더로 올바른 상하 방향을 확인 |

수정 전 X축 +90° 스폰을 양쪽에서 촬영한 8장의 실제 렌더를 확인한 결과,
YCB 003·004·005·006 모두 상하가 뒤집혀 있었다. 상자·캔의 라벨 방향과
머스터드 병의 뚜껑이 아래를 향한 것이 확인되어 4종 모두 −90°로 바꿨다.
형상 bounds가 수직이라는 사실만으로 의미적 상하 방향을 확인할 수는 없다.
수정 후에도 같은 원본 USD를 양쪽에서 8장 렌더링해 003·004의 라벨,
005의 라벨과 여는 고리, 006의 뚜껑이 정상 방향임을 확인했다. 이미지와 quaternion
manifest는 `outputs/contact_rearrangement/ycb_orientation/{before,corrected}/`에 있다.
현재 회전값은 [src/catalog.py](../src/contact_rearrangement/contact_rearrangement/src/catalog.py)에 있다.

Sampler의 XY 위치는 **회전된 footprint 중심**이다. 이를 root pose로 쓸 때는
회전된 local bounding center를 빼서 보정한다.

```text
root_x = center_x - cos(yaw)*cx + sin(yaw)*cy
root_y = center_y - sin(yaw)*cx - cos(yaw)*cy
```

## 4. Blocker가 앞에 들어갈 수 있는 전체 Target 범위

Upright box를 yaw로 회전한 polygon이 실제 판정 footprint다. 위치를 제안할
때는 그 회전된 polygon을 감싸는 world XY AABB의 half extent를 쓴다.
Upright half extent가 `(hx,hy)`, yaw가 `θ`라면:

```text
HX = |cos θ|*hx + |sin θ|*hy
HY = |sin θ|*hx + |cos θ|*hy
```

Target과 Blocker 사이의 X 방향 AABB gap은 **5–15 mm**에서 뽑는다.
Target 중심을 놓을 수 있는 범위는 두 물체가 모두 선반 안에 들어가도록 계산한다.

```text
target_x_min = shelf_x_min + 0.015 + target_HX
target_x_max = shelf_x_max - 0.015 - target_HX - 2*blocker_HX - pair_gap
target_x ~ Uniform(target_x_min, target_x_max)
blocker_x = target_x + target_HX + blocker_HX + pair_gap
```

Target은 앞에 Blocker가 들어갈 수 있는 **전체 유효 X 범위**에 제안한다.
뒤쪽 첫 10 cm 제한은 제거했다. 앞쪽 끝은 선반의 15 mm margin, Target·Blocker의
회전된 크기와 pair gap으로 정하며, 이 조건이 허용하는 만큼 앞으로 올 수 있다.
유효한 범위가 없으면 배치를 다시 시도한다. 제안이 uniform이어도 이후 coverage·
경로·충돌 rejection으로 채택된 위치 분포는 달라질 수 있다.

Target Y는 Target footprint와 접근영역 폭이 선반 안에 드는 범위에서 뽑는다.
Blocker Y의 ±30 mm 제한도 제거했다. 회전된 Blocker가 support 안에 있으면서
초기 접근영역의 Y 구간과 겹칠 수 있는 전체 범위에서 제안한다.

```text
blocker_y_min = max(shelf_y_min + 0.015 + blocker_HY,
                    target_y - 0.06 - blocker_HY)
blocker_y_max = min(shelf_y_max - 0.015 - blocker_HY,
                    target_y + 0.06 + blocker_HY)
blocker_y ~ Uniform(blocker_y_min, blocker_y_max)
```

구간의 단순 겹침만으로 채택하지 않고 아래의 실제 footprint 30% 차단 판정을
적용한다. 앞·뒤 관계는 최종적으로
`min_x(Blocker footprint) >= max_x(Target footprint) + 3 mm`인지 검사한다.

5–15 mm는 회전 후 AABB로 제안한 간격이며 두 mesh 표면 사이의 실제 최소
거리라고 단정하지 않는다. 모든 active footprint 쌍은 별도로 SAT 검사를 통과해야
하며, 적어도 한 normalized edge axis의 projection에서 **3 mm 이상의 gap**을
요구한다. 이는 mesh 간격의 정확한 계산이 아니라 보수적인 box 분리 판정이다.

## 5. 고정 접근영역 A₀와 초기 차단

현재 `hand_width=0.10 m`, `approach_margin=0.01 m`를 사용한다.
**10 cm hand width는 임시 설정값이며 RH56E2의 실제 형상에서 측정한 값이 아니다.**
고정 접근영역 `A₀`는 초기 Target footprint의 앞쪽과 중심 Y로 만든 직사각형이다.

```text
A₀.x_min = max_x(initial Target footprint) + 0.01
A₀.x_max = measured shelf_x_max = -0.50
A₀.y_min = initial Target center_y - 0.06
A₀.y_max = initial Target center_y + 0.06
```

따라서 기본 폭은 **12 cm**다. Target 앞에서 1 cm 떨어진 지점부터 선반 앞
경계까지 뻗는다. 접근영역 X 끝은 물체의 15 mm edge margin을 적용한 `-0.515`가
아니라 측정 support의 앞 경계 `-0.500 m`다. X 길이가 양수가 아니면 거부한다.
Episode 초기값을 저장하므로 Target이 나중에 움직였다는 이유로 영역이 같이
따라 움직이는 구조가 아니다.

초기에 지정 Blocker가 실제로 이 영역을 막도록 다음 조건을 요구한다.

```text
Area(Blocker footprint ∩ A₀) / Area(Blocker footprint) >= 0.30
```

분모는 **Blocker 전체 footprint 면적**이다. A₀ 면적이나 선반 면적으로 나누지
않는다. Convex polygon clipping으로 겹침을 구한다. 주변 물체는 처음부터 A₀
밖에 있고 SAT 기준으로 3 mm 이상 떨어져 있어야 한다. 초기부터 여러 물체가
접근영역을 함께 막는 장면을 허용하는 baseline은 아니다.

## 6. 병진·회전·pivot 조합을 허용하는 재배치 공간

Sampler는 초기 Blocker pose에서 A₀ 밖까지 이어지는 **평면 경로 하나**를
찾고 그 공간을 남긴다. 경로 후보의 순서를 무작위로 섞은 뒤 지원 가능한 경로를
선택한다. 다음 종류를 포함한다.

- ±Y lateral translation.
- Bounding center에서 ±45° 또는 ±90° 회전, 필요하면 이어서 lateral translation.
- 초기 footprint의 네 모서리 중 하나를 고정한 ±45° 또는 ±90° **실제 pivot arc**,
  필요하면 이어서 lateral translation.
- +X로 15 mm 또는 30 mm 병진한 뒤 ±90° 회전하고 lateral translation하는 다단계 경로.

Pivot은 yaw만 바꾸는 center rotation과 다르다. 고정 world XY pivot을 중심으로
footprint 중심도 원호를 따라 움직인다. 이 후보들은 가능한 모든 조작을 탐색하는
완전한 planner가 아니며, 채택 경로 종류가 균등하게 나온다고 보장하지 않는다.

각 경로는 연속된 `PlanarMotion(start, end, pivot)` 배열로 저장한다. Start/end는
footprint 중심 X·Y와 yaw의 `PlanarPose`다. `pivot=None`이면 중심 병진과 yaw가
선형 보간되고, pivot이 있으면 그 고정점을 중심으로 회전한다. 최종 pose와 초기
pose의 연결, 각 segment의 연속성과 pivot arc의 끝점을 함께 검사한다.

**끝점만 비어 있어서는 통과하지 않는다.** 회전 없는 병진은 두 endpoint footprint의
convex hull로 전체 swept footprint를 정확히 표현한다. 회전은 최대 5°씩 구분한
endpoint hull을 만들고, 모서리 원호가 chord 밖으로 나갈 수 있는 거리
`R*(1-cos(Δθ/2))` 이상을 사각형으로 확장한다. `R`은 pivot 경로에서는 pivot부터
가장 먼 모서리까지, 일반 회전에서는 bounding center부터 가장 먼 모서리까지의
거리다. 이렇게 회전 중간의 footprint를 보수적으로 포함한다.

모든 segment의 swept envelope는 15 mm margin 안에서 지지되고 Target·모든
주변 footprint와 SAT 기준 3 mm 이상 분리되어야 한다. 마지막 footprint는 A₀와
3 mm 이상 분리되어야 한다. Lateral endpoint 제안은 회전 후 half extent를 써서
A₀의 Y 경계로부터 4 mm 여유를 둔다. 주변 물체도 선택된 경로의 모든 envelope
밖에 배치하므로 전체 선반 분포와 재배치 여유를 함께 검사한다.

`SceneSample.clearance_path`와 `clearance_kind`는 생성 검증용 metadata다.
`clearance_witness`는 마지막 중심 XY를 보존한다. 이 경로를 actor의 final goal
pose나 명령으로 제공하지 않는다. 경로가 있다는 것은 보수적 평면 기하에서
여유가 있다는 뜻이며, 접촉을 통해 실제 pivot을 수행할 수 있다는 보장은 아니다.

## 7. 전체 선반과 각 구역의 hard occupancy, coverage와 국소 이웃

현재는 전체 가용 support를 분모로 하는 **global 점유율 하한**과, X 2분할 ×
Y 3분할의 **6개 구역 모두의 점유율 하한**을 acceptance 조건으로 요구한다.
양쪽에 작은 물체 몇 개를 두어 전체 span만 넓어지는 장면은 충분하지 않다.
이전 38 cm cluster·tight rectangle fill 조건과 점유율을 보고만 하던 방식은
현재 조건에서 제외한다.

```text
usable_x = shelf_x_max - shelf_x_min - 2*edge_margin
usable_y = shelf_y_max - shelf_y_min - 2*edge_margin
usable_area = usable_x * usable_y
shelf_occupancy = Σ Area(active footprint) / usable_area
region_occupancy[j] = Σ Area(active footprint ∩ region[j]) / Area(region[j])
```

기본 선반의 usable 크기는 **37 cm × 88 cm**, 면적은 **0.3256 m²**다.
각 구역은 18.5 cm × 약 29.33 cm이다. 물체 중심이 구역에 들어왔는지로 세지
않고 oriented footprint와 각 구역의 실제 교차 면적을 합한다. 한 물체가 경계를
넘으면 해당 면적만큼 여러 구역에 기여한다. 모든 footprint가 분리되어 있으므로
면적을 중복 계산하지 않는다. Target·Blocker·주변 물체 모두 포함하며 inactive
pool과 reserved 재배치 경로 면적은 포함하지 않는다.

| 판정 | 현재 구현 상태 |
|---|---|
| Global footprint occupancy | `>= min_shelf_occupancy` — 현재 초기 기본값 **0.35** |
| 각 구역의 footprint occupancy | 6개 모두 `>= min_region_occupancy` — 현재 초기 기본값 **0.10** |
| 전체 가용 X 범위 대비 footprint span | `x_coverage >= 0.55` |
| 전체 가용 Y 범위 대비 footprint span | `y_coverage >= 0.65` |
| 4 cm 이내 polygon 경계 이웃이 있는 물체 비율 | `local_neighbor_fraction >= 0.75` |

Global 35%와 옛 tight-rectangle fill 35%는 분모와 의미가 다르다. 전자는
**전체 usable shelf** 면적, 후자는 물체를 감싼 작은 rectangle 면적이었다.
Region 조건은 전체 평균을 맞추면서 특정 구역을 비우는 배치를 막는다.
Coverage는 footprint 꼭짓점의 X/Y span을 usable X/Y로 나눈 값으로 별도 검사한다.
4 cm 이웃 판정은 polygon 경계의 Euclidean 거리이며 N으로 나눈 비율이다.

**이전 면적 우선 구성의 측정으로 35%/10%를 초기 기본값으로 선택했다.** 사용자는 먼저 구현 가능한
범위를 측정하도록 요청했다. 35%는 고유 기하 sample 40/40개, 40%는 30/30개가
생성됐지만 공통 seed 비교에서 40% 생성 시간이 약 5배 길었다. 45%는 초기
10개 중 3개만 생성됐다. 이를 근거로 35%를 선택했으며 사용자에게 직접
확정받은 최종 연구 난이도로 기록하지 않는다. 이 비교의 생성 시간·성공률·개수
분포는 균형 구성으로 변경한 현재 버전에 그대로 적용하지 않는다. 조건·표본은 11절과
[CONTEXT.md](CONTEXT.md)에 정리했다.
모든 occupancy는 보수적인 upright box footprint 기준이며 실제 scanned mesh의
점유율이나 카메라 가시성 percentage가 아니다.

### 부족한 구역을 채우는 placement proposal

현재 코드에서는 주변 instance를 큰 footprint부터 배치한다. 지금까지의 구역별
점유 면적을 누적하고 하한보다 부족한 구역을 우선 후보로 고른다. 모두 하한에
도달했다면 점유 면적이 최소값 근처인 구역을 고른다.

Proposal의 60%는 선택 구역 안의 유효 중심 위치를 uniform으로 뽑는다. 나머지
40%는 그 구역에 가까운 기존 물체를 anchor로 택해 X/Y 이웃 위치를 제안한다.
Neighbor gap은 yaw-expanded AABB half extent 합에 **4–28 mm**를 더한 값이며
정확한 mesh 표면 간격은 아니다. 모든 proposal은 footprint support·3 mm 물체
간격·초기 A₀ 외부·reserved 경로 외부 조건을 통과해야 한다.

현재 proposal score는 부족한 구역의 면적을 얼마나 채우는지, 4 cm 이내 이웃을
만드는지, 구역 점유 균형을 개선하는지를 조합한다. 유효 후보를 최대 6개 모아
가장 높은 score를 택한다. 이 점수는 proposal 선택에 쓰는 값이고 최종 hard
threshold를 대체하지 않는다. 배치 후 global·6개 구역·coverage·이웃을 모두
검사하며 하나라도 부족하면 layout을 다시 시도한다. 이 proposal 구성은 현재
성능 측정과 함께 바뀔 수 있는 구현 방식이다.

## 8. 시도 횟수, 실패, 재현성

각 주변 물체의 placement proposal은 한 layout 안에서 최대 **180번** 시도한다.
하나라도 놓지 못하거나 전체 coverage·국소 근접 등 최종 조건이 맞지 않으면 새 layout을
시도한다. 전체 layout은 최대 **300번**이다.

전체 retry 동안 Target·Blocker instance 쌍, 종류 수·집합, **전체 active instance
ID 조합과 개수**는 고정이다. 모든 active yaw와 Target·Blocker·주변 위치만 다시
뽑는다. 300번 뒤에도 맞는 장면이 없으면 `RuntimeError`를 낸다. 역할을 바꾸거나
gap·밀집도를 자동 완화한 fallback은 없다. 종류·역할 조합별 성공률과 retry 수를
보고 sampler의 어려운 조합을 확인해야 하며, 모든 가능한 조합이 균등하게
성공한다고 가정하지 않는다.

환경별 배치를 사용하는 spawn 환경(`shared_initial_scene=False`)의 reset seed는 다음과 같다.

```text
episode_seed = cfg.seed + 1_000_003*env_id + 97_409*reset_count[env_id]
```

`reset_count`는 첫 episode에서 0이며 해당 환경이 성공적으로 reset될 때
증가한다. 부분 reset은 선택한 환경의 count만 바꾸므로 다른 환경의 다음
장면 sequence를 진행시키지 않는다. 같은 catalog geometry·설정·seed와 동일
reset 순서는 동일 sampling을 재현한다.

학습 기본값 `shared_initial_scene=True`에서는 전체 환경에 공통 초기 물체 배치·역할·
로봇 자세를 적용한다. 전체 reset마다 `cfg.seed + 97_409*shared_generation`으로
새 공통 장면을 만들고 env 0에서 초기 RGB-D와 가시성을 한 번 승인한다. 부분 reset은
현재 공통 초기 장면·vision을 재사용하고 선택된 환경의 reset count만 증가시킨다.
모든 물체·로봇은 환경별 PhysX 인스턴스를 유지하므로 이후 동작과 접촉은 독립적이다.
기하·카메라 자세는 환경 원점 차이를 반영해 각 환경의 좌표로 옮긴다.

카메라 2개와 render-only 기준 물체 10개는 전체 환경이 공유한다. 실제 카메라는
RGB-D를, 기준 카메라는 같은 시점·물체 자세의 비가림 실루엣을 제공한다. Headless
실행은 step 중 자동 렌더링을 생략하고 초기 촬영 때만 명시적으로 렌더링한다.
카메라 버퍼는 유지하며 부분 reset은 새 이미지를 촬영하지 않는다. Headless 학습은
env 0의 visual만 렌더링하고 다른 환경의 physics는 Fabric replica로 유지한다.

Event는 선택된 모든 환경의 물체 sample·robot 시작 후보를 CPU에서 승인한 뒤
native state에 적용하고 camera pixel acceptance를 검사한다. Camera 32개
후보가 실패하면 같은 역할·종류·active ID·개수로 pose를 다시 제안한다. 최대
8개 layout에서도 실패하면 기존 physical pose·metadata로 복구하고 오류를 낸다.
Render는 physics step을 진행하지 않는다. 초기 snapshot·A₀는 같은 승인 pose에서
commit한다.
Object velocity와 robot joint velocity를 0으로 설정하고,
active ID/mask·Target/Blocker ID·초기 state·A₀·sample metadata를 저장한다.
유효 ID 배열은 현재 safety cap인 최대 40개이며 사용하지 않는 항목은 `-1`이다.

### Blocker 앞의 hand 시작 영역과 elbow-up 조건

사용자는 Blocker 앞쪽 영역에서 다른 물체와 충돌 없이 hand 시작을 무작위로
정하고 elbow-down은 허용하지 않도록 요청했다. 구현 설정은
`cfg.robot_start`의 `RobotStartRules`이며 기본 `enabled=True`다.
다음 수치는 현재 구현 기본값이며 이후 제어·학습 분포에 맞춰 조정할 수 있다.

| 기준 | 현재 값 |
|---|---|
| Blocker 회전 footprint의 가장 큰 X에서 TCP까지의 앞쪽 거리 | 0.12–0.40 m |
| 선반 앞쪽 X 경계에서 TCP까지의 최소 거리 | 0.06 m 밖 |
| Blocker bounding center Y 대비 TCP | ±0.08 m |
| 선반 상면 대비 TCP 높이 | +0.20–0.28 m |
| Shoulder–wrist chord 대비 elbow 높이 | 최소 +0.05 m |
| Robot 대 scene/ground OBB 여유 | 0.015 m |
| Arm joint 거리 3 이상인 UR arm 10쌍의 self collision 여유 | 0.002 m |
| Robot 시작 후보 시도 | 최대 80번 |

이 거리의 기준은 **palm TCP**이며 가장 가까운 손가락 표면이 아니다. 로봇을
나타내는 환경 로컬 영역은 다음과 같다.

```text
blocker_front_x = max_X(rotated_Blocker_footprint)
xmin = max(blocker_front_x + 0.12, shelf_xmax + 0.06)
xmax = blocker_front_x + 0.40
y = Blocker_center_y ± 0.08
z = shelf_top + [0.20, 0.28]
```

영역 안 TCP의 X/Y/Z를 uniform으로 제안한다. 두 X 조건의 교집합이 비면
명시적으로 실패하며 Blocker를 옮기거나 거리 조건을 완화하지 않는다.
TCP body는 **`right_plam_force_sensor`**다. 손가락/tool 전방인 TCP local +Y가
**world −X, 선반 정면**, palm normal인 local +Z가 **world −Y**, local +X가
world +Z를 향하는 orientation을 고정한다. 실제 조립 USD에서 TCP +Y는
`wrist_3_link` local +Z인 tool 전방과 일치한다. 기존 vendor adapter
`mount_rotation_deg=(0,-90,0)` deg는 유지하고 `mount_tool_roll_deg=-90` deg를
추가한다. 추가 roll은 physical tool body `wrist_3_link` local +Z축을 기준으로
hand의 상대 mount 회전과 translation에 적용한다. `tool0` attachment 축은
physical tool body와 다르므로 `tool0` 자체 local Z로 회전하지 않는다. 목표
회전은 `sampling.ROBOT_START_TCP_ROTATION`이며 각 열이
world에서 본 TCP의 X/Y/Z 축이다.

센서·연결부 공간용 `SensorMount`는 `wrist_3_link`에 부착한 visible collision
Cylinder 형상의 연결부다. 길이는 `sensor_mount_length=0.05 m`, 반지름은
`sensor_mount_radius=0.025 m`다. Tool body local +Z의 z=0–50 mm에서 두 25 mm
rigid half를 중앙 fixed F/T joint로 연결하고 같은 +Z로 hand fixed anchor를
50 mm 이동한다. −90° hand roll과 TCP 목표 회전은 유지한다. Nominal 질량 합
0.20 kg는 구현 기본값이며 실제 하드웨어 질량으로 확정하지 않는다. 원본 asset과
기존 wrist mass/inertia를 유지하고 총 41 body·18 DOF·41 collider를 쓴다.

```text
R_world_tcp = [[0, -1, 0],
               [0,  0, -1],
               [1,  0, 0]]
```

Reset 검증에서는 실제 PhysX TCP의 전체 회전과 이 목표 사이 angular error가
0.01 rad 이하인지 확인한다. 같은 PhysX pose에서 wrist-3 local +Z의 world
방향을 구해 `(−1,0,0)`과의 vector norm 오차도 0.01 이하인지 대조한다.
두 비교의 수치 허용 오차는 1e−5다. TCP 위치·영역 검증만으로 tool 방향을
판정하지 않는다.

`src/assets.py`의 `RobotPoseModel`은 실제 조립 USD의 joint frame·axis·scale과
41개 link의 fixed/revolute 연결을 읽어 FK를 구성하고 SciPy의 bounded IK로
UR arm 해를 찾는다. Finger q는 기존 열린 손의 기본값을 유지한다. Elbow-up은
관절각 부호로 분류하지 않는다. Shoulder-lift·elbow·wrist-1 joint center를 S/E/W로
삼고 E에서 S–W 선분의 가장 가까운 점 P를 구해 **`E.z - P.z >= 0.05 m`**를
요구한다. Elbow-down/낮은 elbow 해는 채택하지 않는다.

`RobotPoseModel.from_env`는 현재 실제 관절/link pose에서 FK를 대조한다. 이
측정을 위해 robot을 default 자세로 먼저 reset하지 않는다. Solver의
`accept_candidate` callback은 IK가 만족해도 TCP 영역·충돌 조건을 거절할 수 있다.
한 seed/branch가 충돌하면 같은 TCP의 다른 IK seed/branch를 계속 탐색한다.
이는 여러 seed의 탐색이며 가능한 모든 IK branch의 완전 탐색을 증명하지 않는다.

IK 후 전체 arm/hand의 enabled collision OBB를 **모든 active 물체**(Target·
Blocker 포함), 선반의 개별 enabled collider OBB, ground와 검사한다. OBB는
보수적인 bounds이며 정확한 mesh 거리 판정은 아니다. UR arm 7개 link의
21쌍 중 **두 arm joint 이내의 11쌍**(direct parent·grandparent)은 제외하고
거리 3 이상인 10쌍을 2 mm margin으로 검사한다. 가까운 joint housing을 감싼
OBB의 false positive를 피하는 처리이며 원통을 포함한 전체 41개 robot collider 대 외부 물체·
17개 shelf collider·ground의 15 mm 판정은 유지한다.
**손 전체 self collision 안전성을 확인했다고 기록하지 않는다.**

물체 sample을 먼저 고정하고 `episode_seed XOR 0x5D2EA91C`의 별도 RNG로
robot 후보를 제안하므로 기존 물체 배치 RNG sequence를 소비하지 않는다.
최대 80번 안에 영역·IK·elbow·충돌 조건을 만족하지 못하면 실패 이유별 개수를
담은 `RuntimeError`를 낸다. Elbow-down fallback은 없다.

채택하면 선택한 환경의 joint position·zero joint velocity와 동일한 PD target을
쓴다. `robot_initial_joint_pos`, `robot_start_tcp_pos_w`, `robot_start_region`,
`robot_start_elbow_height`, `robot_start_attempts`를 기록하며 다른 환경의 state는
유지한다. Candidate 평가·reset에 settling이나 추가 physics step을 넣지 않는다.
`cfg.robot_start.enabled=False`로 끄면 기존 기본 robot 자세를 사용한다.
**이전 tool 방향의 검증 (2026-10-09):** 아래 수치와 로그는 손가락/tool 전방이
world −Z였던 orientation의 결과다. 선반 방향 수정의 검증으로 사용하지 않는다.
초기 구현은 4개 GPU 환경·151 step·5.03초에서 150-step 자동 reset과 마지막
전체/부분 reset, 선택하지 않은 환경의 q/velocity/PD target/robot metadata 유지
검사를 통과했다. 기본 USD의 robot collider 39개·shelf collider 17개,
FK 대 실제 link position 오차 5.13e−7 m도 확인했다. 이후 callback·후보 승인 후
state 적용·COM cache·arm 이웃 OBB 처리를 보강한 **최종 4환경 × 6회, 24개
시작 pose**도 통과했다. 8개는 10종 전체 장면이었다. Physics 없는 6회 reset 뒤
150 step·5초의 parking 안정성과 전체/부분 reset, 불가능한 elbow 조건으로
강제 실패한 경우의 state·metadata 유지도 통과했다. 최종 FK 오차는 2.8458e−7 m,
elbow chord 높이는 11.49–16.07 cm, 후보 시도는 1–3회였다.
최종 unit test 29개도 24.673초에 통과했다. Stress 자료는
`outputs/contact_rearrangement/robot_randomization_benchmark.{json,log}`다.
현재 변경본의 일반 모드 4환경·151 step·5.03초 자동/부분 reset·COM cache·
OBB 불변 검사도 exit code 0으로 통과했다. 로그는
`outputs/contact_rearrangement/robot_randomization_gpu.log`다.
GUI 45-step·1.5초의 실제 표시·전체/부분 reset 검사도 통과했다.
상세 변경 이력과 GUI 검사 결과는 [CONTEXT.md](CONTEXT.md)에 기록한다.

**Tool 방향 교정, hand roll 추가 전 (2026-10-10):** 사용자의 요청에 따라 palm normal을 선반으로
향하던 회전 대신 실제 tool 전방을 world −X로 맞췄다. 시작 영역·elbow-up·외부
충돌 여유를 유지하고 실제 TCP 회전·wrist-3 tool 축 검사를 추가했다. 이번
방향의 일반 모드 **4개 GPU 환경·151 step·5.03초** `--validate`가 exit code
0으로 통과했다. 초기·5초 자동 reset·부분 reset·마지막 전체 reset의 **13개
시작 pose**에서 실제 PhysX TCP 회전·tool 축·elbow-up·scene 여유를 확인했고
미선택 환경 유지·COM cache·역할 OBB 표시의 물리 상태 불변도 통과했다.
로그는 `outputs/contact_rearrangement/tool_forward_gpu_validation.log`다.
1개 GUI 환경·45 step·1.50초와 전체/부분 reset·OBB 불변 검사도 통과했으며
실제 렌더에서 tool 전방이 선반 정면을 향하는 것을 확인했다. 자료는
`outputs/contact_rearrangement/tool_forward_preview.{png,log}`다.
45-step GUI 실행 중 캡처 시점의 PD 유지 pose에서 tool 축은 world −X에서
약 1.85° 기울었으며
초기 회전 검사와 유지 pose 검사는 [CONTEXT.md](CONTEXT.md)에 구분해 기록한다.
2026-10-09의 29개 unit test는 이번 수정의 새 테스트 결과로 기록하지 않는다.

**Tool +Z 기준 hand −90° roll, 원통 추가 전 (2026-10-10):** 사용자의 후속 요청으로 hand의
mount에 별도 −90° roll을 추가했다. Tool 전방 −X와 기존 시작 조건을 유지하고
palm normal은 world −Y를 향하게 한다. 위 `tool_forward_*` 결과는 palm normal이
위를 향하던 이전 mount의 검사다. 새 mount의 일반 모드 **4개 GPU 환경·151 step·
5.03초** `--validate`가 exit code 0으로 통과했다. 초기·5초 자동 reset·부분
reset·마지막 전체 reset의 **13개 시작 pose**에서 실제 PhysX TCP 회전·tool
전방 −X·elbow-up·scene 여유를 확인했고 미선택 환경·COM cache·pool flag·역할
OBB 표시의 물리 상태 불변도 통과했다. 로그는
`outputs/contact_rearrangement/hand_tool_roll_gpu_validation.log`다.
첫 reset의 실제 PhysX tool body 기준 상대 회전은 `Rz(-90°) @ R_old`와
3.23141e−7 rad 오차로 일치했고 TCP 상대 translation도 같은 회전을 적용한
값과 1e−5 m 이내로 일치했다. 기존 tool 정면 목표의 전체 world 회전 오차는
1.15236e−5 rad였다. 1개 GUI 환경·45 step·1.50초와 전체/부분 reset·OBB 불변
검사도 통과했다. 자료는 `outputs/contact_rearrangement/hand_tool_roll_preview.{png,log}`다.
렌더는 실행 중 약 20번째 step의 캡처이며 상대 transform 검사는 첫 update의
초기 pose다. 기존 unit test 결과를 이번 수정의 새 실행 결과로 기록하지 않는다.

**5 cm 원통 연결부, 중앙 F/T 추가 전 (2026-10-10):** 사용자 요청으로 센서·연결부 공간을
반영하는 원통을 추가한다. 직경 50 mm는 실제 hand `base_link.STL` mount의
X/Y bounds ±25 mm를 기준으로 선택했다. Asset scale과 무관하게 physical tool
방향의 실제 치수를 적용하고 hand anchor를 +Z로 50 mm 옮긴다. 39 body·18 DOF와
−90° hand roll을 유지하며 원통을 포함한 40 collider가 scene·arm self 검사에
참여한다. 원본 USD와 기존 mass/inertia를 유지하며 센서 질량 보정·F/T 출력은
아직 구현하지 않았다. 원통 추가 전 로그와 새 조립의 검증을 구분한다.
새 조립의 일반 모드 **4개 GPU 환경·151 step·5.03초** `--validate`가 exit code
0으로 통과했다. 초기·5초 자동 reset·부분 reset·전체 reset의 **13개 시작
pose**에서 실제 PhysX TCP·tool 전방·elbow-up·15 mm scene 여유를 확인했다.
40개 robot collider·17개 shelf collider, 실제 link 대 FK position 오차는
5.15e−7 m였으며 미선택 환경·parking COM cache·pool flag·역할 OBB의 물리
상태 불변도 통과했다. 로그는
`outputs/contact_rearrangement/sensor_mount_gpu_validation.log`다.
첫 reset의 실제 PhysX tool body 기준 wrist→hand base offset은 `[0,0,0.05] m`,
gap은 0.049999997 m였다. 원통 양끝이 wrist·hand base origin에 각각 1e−5 m
이내로 일치했고 길이 50 mm·반지름 25 mm를 확인했다. 기존 −90° 상대 hand
roll 오차는 4.51901e−7 rad였으며 TCP offset에도 같은 50 mm 이동이 적용됐다.
1개 GUI 환경·45 step·1.50초와 전체/부분 reset·OBB 상태 불변도 통과했다.
자료는 `outputs/contact_rearrangement/sensor_mount_preview.{png,log}`와
`sensor_mount_detail.png`다. 렌더는 각각 약 20번째·32번째 step의 캡처이며
상대 transform 수치는 첫 reset에서 측정했다.

### 원통 중앙의 raw F/T

현재 두 원통 half 사이 fixed joint의 반력을 PhysX에서 직접 읽는다. Torque
기준점은 원통 중앙이며 축은 physical tool XYZ, 부호는 parent→child 방향이다.
`[Fx,Fy,Fz,Tx,Ty,Tz]`의 단위는 `[N,N,N,N·m,N·m,N·m]`다. 중력·관성·접촉·외력이
포함된 raw 값이며 자동 tare·중력 보정은 없다. 명시적 `tare()`는 현재 bias를
저장할 뿐 중력 보정이 아니다. 선택된 환경의 reset은 cache·bias를 비우고
valid=False로 둔다. 다음 완료된 physics step 뒤 timestamp가 증가하면 새
반력을 읽고 valid=True로 갱신한다. 초기 camera render는 이 timestamp를
진행시키지 않으므로 stale reaction을 valid로 취급하지 않는다.

### 독립 tactile 17채널

Scene sensor 이름은 `contact_` + canonical body name이며 각각
`{ENV_REGEX_NS}/Robot/ROASRightHand/{body_name}` 한 body에 연결한다. 순서는
palm → thumb 1–4 → index/middle/ring/little 각 1–3이다. Palm의 기존 철자
`right_plam_force_sensor`를 유지하고 URDF의 정확한 17개 link·collision mesh를
초기에 검사한다. `cfg.tactile.force_threshold=0.01 N`은 finite positive 값이다.
World net normal contact force norm **> threshold**가 binary이며 SDK의
air/contact-time용 `force_threshold`는 이 계산에 쓰지 않는다. `update_period=0`,
`history_length=0`으로 마지막 완료 120 Hz physics step(30 Hz 관측의 마지막
substep)을 읽는다. `tactile`·`tactile_valid`는 bool `[N_env,17]`, raw debug cache
`env.binary_tactile_sensor.net_forces_w`는 `[N_env,17,3]`다. 같은 timestamp의
render/reset은 stale report를 재조회하지 않으며 선택 reset은 zero/invalid,
다음 완료 step은 valid, 미선택 cache는 보존한다. 자유 공간의 중력·관절하중
자체는 contact가 아니다.

Palm·finger pad-2 CAD slot을 메우던 기존 부모 5개 collider에만 convex
decomposition을 적용한다. 17개 pad collider·source URDF·visual·mass/inertia는
유지하며 41 body·18 DOF·41 collider **prim**을 유지한다. Cooked hull 수는 prim
수와 다르다. 대상·설정은 [CONTEXT.md](CONTEXT.md)에 기록한다.

**실제 GPU 검사:** `check_tactile.py --num_envs 2`가 exit code 0으로 통과했다.
151 env step·5.03초, reset 6회(부분 2회)·render 116회에서 tactile cache와
physics time/step을 보존했고 반지름 3 mm test-only sphere로 17개 pad의 해당 채널 ON·분리 4 physics
step 뒤 모두 OFF를 확인했다. 중력/PD hold만으로 OFF, reset zero/invalid·다음
step valid, 다른 환경 palm ON cache 보존과 기존 scene·camera·OBB 검사도
통과했다. 로그는 `outputs/contact_rearrangement/tactile_gpu_validation.log`다.
이는 binary 접촉 검증이며 하중 교정 결과가 아니다.

## 9. 앞쪽 camera와 실제 초기 silhouette 판정

`sensors.InitialVisionCfg`에 다음 기본값을 둔다.

| 항목 | 설정과 의미 |
|---|---|
| Camera side | 선반 앞쪽인 `+X` |
| 최소 Target visible fraction | `0.40` |
| 최소 Blocker visible fraction | `1.0`, 면적 손실 최대 2 pixel 허용 |
| 분모 reference | 같은 역할 물체 pose·같은 camera intrinsics/상대 pose의 **비가림 전체 silhouette** |
| 화면 포함 | Target·Blocker 전체 OBB corners가 frame 안에 있어야 함 |
| Reference 최소 크기 | 각 역할 32 pixel |
| 해상도·intrinsics | 640×480, focal length 18 mm·horizontal aperture 24 mm, horizontal FOV 약 67° |
| Camera X | 선반 전면 +0.55–0.95 m |
| Camera Y | Blocker center ±0.35 m에서 제안, 선반 중심 ±0.50 m로 제한 |
| Camera Z | 선반 상면 **+0.18–0.48 m** |
| Aim | 두 역할 중간, Y ±0.06 m·높이 상면 +0.10±0.03 m |
| Optical-axis roll | ±3° |
| Retry | Camera 최대 32개 × 같은 composition의 layout 최대 8개 |

판정은 `실제 Target instance mask pixel 수 / 비가림 reference pixel 수 >= 0.40`,
Blocker는 `실제 pixel 수 + 2 >= reference pixel 수`다. 2 pixel 허용은 전체
면적의 정수 차이에 대한 허용이며 contour boundary band를 제외하는 방식이
아니다. 다른 물체·선반·로봇 가림을 포함하고 self-occlusion은 reference에도
포함된다. 전체 bounds의 frame 포함과 reference 최소 pixel 조건을 별도로
확인해 화면 밖으로 자르거나 작은 물체로 가시 비율을 높이지 않는다.
전체 bounds는 실제 camera near/far clip `0.02–2.5 m` 안에 있어야 한다.
설정의 nonfinite 값·잘못된 range·음수 jitter는 오류다. `camera_side="+X"`,
`require_full_target_in_frame=True`와 같은 pose 비가림 reference 규칙만 지원하며
이를 끄거나 다른 방식으로 바꾸는 설정은 거절한다.

`InitialCamera`는 RGB·image-plane depth·instance segmentation을 읽고
`SilhouetteCamera`는 reference segmentation을 읽는다. 같은 형상의 10종
render-only reference를 physics 시작 전에 작업층 밖 `z+20 m, y+5 m×type index`
lane에 만들며 환경별로 Z offset에 `5 m×env_id`를 더해 분리하고 physics API를
제거한다. Reference 물체와 camera를 같은 offset으로
옮겨 실제 물체·camera의 상대 pose를 보존한다. Live body visibility나 physics
topology는 바꾸지 않는다. 최초 5회·이후 2회 `sim.render()`로 texture를 준비하고
physics step은 진행하지 않는다. 검사 대상은 settled pose가 아니라 승인된
reset pose다. A₀의 30% Blocker overlap은 별도의 평면 접근 차단 조건이다.

로컬 `InitialCamera`는 SDK 첫 Fabric pose 조회의 stage sync가 Kit update로
physics를 진행시키지 않도록 pose 조회/설정·`camera.update` 동안
`/app/player/playSimulations=False`로 보호하고 `try/finally`로 복구한다.
첫 camera data 읽기 전에 cache를 준비하며 SDK·live body physics flag는 바꾸지 않는다.

순서는 **역할·종류·active ID·개수 고정 → pose·robot 후보 승인 → native state
적용 → camera·실제 mask·같은 pose reference → pixel acceptance → snapshot·A₀
commit**이다. Camera 32개가 실패하면 composition을 유지한 채 pose만 다시
제안한다. 최대 8개 layout이 실패하면 기존 state·metadata로 rollback한다.
Rollback은 물체 COM velocity·robot state/목표와 초기 snapshot·camera·두 역할의
가시성 buffer를 모두 복구한다.
성공한 reset은 `target_visibility_validated`·`blocker_visibility_validated=True`,
실제 `target_visible_fraction`·`blocker_visible_fraction`과 pixel counts를 저장한다.
승인한 초기 Target pose의 A₀와 RGB-D/mask/camera pose snapshot은 episode 동안
고정한다. `ObservationsCfg.sensors`는 raw F/T와 valid mask·실제 initial snapshot을
dictionary로 제공하며 17채널 `tactile`·`tactile_valid`도 포함한다. Reference·가시성
counts는 actor 관측에 넣지 않는다.

### Blocker 앞의 기하 reserve 통로

`SpawnRules.reserve_blocker_front=True`가 기본이다. 회전된 Blocker footprint를
기준으로 주변 물체가 다음 직선 +X 통로에 들어오지 않도록 pose proposal과
최종 validation 모두에서 검사한다.

```text
x = [max_X(Blocker footprint), shelf_front_x]
y = [min_Y(Blocker footprint) - gap, max_Y(Blocker footprint) + gap]
gap = 0.003 m
```

Target·Blocker 이외의 주변 footprint를 통로 밖에 둔다. 기존 Target A₀와
clearance witness는 유지하고 전체 선반 35%·각 구역 10% 등 density 조건도
완화하지 않는다. 이 기하 reserve는 앞쪽 camera proposal의 성공 가능성을
높이려는 기준이며 실제 Blocker 전체 가시성을 보장하지 않는다. Pixel gate는
계속 별도로 검사한다.

**센서 단계 (2026-10-10):** 중앙 fixed-joint F/T와 실제 초기 vision gate를
추가했다. 원통 geometry만 검사하던 위 결과와 이번 센서·render 검증을 구분한다.
Reserve 통로와 환경별 reference lane 분리 후 순수 sampling 검사 **32개**가
**27.993초**에 통과했다. 이를 렌더·F/T·GPU 실행 검증으로 확대하지 않는다.
실제 robot arm·hand 가림을 확인해 camera 높이를 상면 +0.18–0.48 m로 넓혔다.
1개 환경의 진단에서는 layout 1회 retry 뒤 2번째 camera 후보가 Target
56.370%·Blocker 100%로 승인됐다. 전체 생성 성공률이나 GPU 검증 결과는 아니다.

**최종 센서 GPU 검사:** 4개 환경의 public **151 env step·5.03초**와 추가
2 env step·F/T bench 7 physics step이 exit code 0으로 통과했다. 6회 reset 중
2회 부분 reset, **18개 승인 시작**을 검사했다. Successful capture call 6회·
failed layout 2개, 두 warmup call 포함 render 464회였으며 capture 중 native
object/robot pose·velocity·joint state·PD target·time/step이 정확히 유지됐다.
Target **47.1253–100%**, Blocker 17개 정확히 100%·1개 **99.97586%**(정확히
2 pixel 손실 허용), 승인 camera 1–30회였다. 153 env step 동안 initial snapshot을
유지했으며 reset 직후 F/T zero·invalid, 다음 physics 뒤 valid·finite와 미선택
cache 보존을 확인했다. Native **41 body·18 DOF·41 collider**였다.
F/T의 XYZ별 3 N force·0.10 N·m torque와 Newton–Euler 대조에서 최대 norm
오차는 **0.0206053 N·0.00132915 N·m**(기준 0.10 N·0.015 N·m)이었다. 중앙점·
frame·부호·성분을 확인했으며 downstream은 33 body·0.876663 kg였다. 로그는
`outputs/contact_rearrangement/initial_sensors_gpu_validation.log`다. 이 검증을
모든 seed·robot push·하드웨어 교정 성공으로 확대하지 않는다. SDK 초기화의
별도 physics step 2회는 env step·reset/capture 검사와 분리한다.

**GUI·실패 복구:** 1개 환경·45 env step·1.50초도 exit code 0으로 통과했다.
003 Target **7724/9286 pixel, 83.178979%**, 006 Blocker **3608/3608 pixel, 100%**,
camera 후보 3회였다. 이동 상태의 강제 capture/중간 commit 실패에서도 물체
COM velocity·robot root/DOF·목표·camera·snapshot/역할/seed/count/sample·
F/T raw/valid/tare·clock/step 복구를 확인했다. Root/DOF는 정확히 복구됐지만
derived robot link velocity의 solver 출력 차이 최대 4.47e−5를 exact 복구로
취급하지 않는다. 자료는 `outputs/contact_rearrangement/initial_sensors_preview.log`,
`initial_camera_preview.{png,json}`, `initial_sensors_preview.png`,
`initial_sensors_detail.png`다. Camera/F/T·전체/부분 reset을 재사용 검증하는
[check_sensors.py](../src/contact_rearrangement/tests/check_sensors.py)의 재현 명령은
[README](../src/contact_rearrangement/README.md)에 있다.

**Repository 진입점:** `check_sensors.py --num_envs 1`도 exit code 0으로 통과했다.
151+2 env step·bench 7 physics step, reset/capture 각 6회·render 179회였다.
부분 reset은 0회이며 승인 6개의 Target **41.0632–100%**, Blocker 전부 100%였다.
Reset/capture physics step 증가 0·snapshot 보존과 6축 bench 최대 norm 오차
**0.0122181 N·0.000850297 N·m**를 확인했다. 로그는
`outputs/contact_rearrangement/initial_sensors_reusable_check.log`다. 앞의 4환경·
GUI 검사와 별도 실행이며 `compileall`·`git diff --check`도 통과했다.

## 10. 선택하지 않은 pool body의 처리

Inactive instance는 작업층 밖의 격리 parking으로 옮긴다. Pool index를 `i`라
하면 현재 parking 좌표는 다음과 같고 마지막에 environment origin을 더한다.

```text
x = -0.64 + 0.32*(i % 5)
y = -0.64 + 0.32*((i // 5) % 5)
z = 4.0 + 0.4*(i // 25)
```

높은 layer에 분리하고 시작 속도를 0으로 설정한다. All-body simulation과
native gravity는 유지하며 live disable-simulation/disable-gravity flag를
전환하지 않는다. USD visibility를 수정해 숨기는 방식도 사용하지 않는다.
현재 작업층 camera ROI 밖에 두는 공간 분리와 active mask가 역할을 구분한다.

[TaskObjectPool](../src/contact_rearrangement/contact_rearrangement/src/assets.py)이
매 physics write/substep에서 inactive body의 실제 world COM에 `-m*g` 힘을
적용해 중력을 상쇄한다. Root pivot에 힘을 가해 불필요한 torque가 생기지 않도록
COM 위치를 명시한다. Active body에는 이 추가 힘이 0이므로 native gravity와
원래의 접촉 동역학을 사용한다. Link pose write 뒤 `_object_com_pose_w` cache의
timestamp를 `-1.0`으로 무효화해 같은 timestamp에서 여러 reset을 해도 다음
world COM을 새로 읽는다. 추가 physics step·simulation/gravity flag 전환은 없다.
연속 reset에서 드러난 회귀와 수정 후 통과 결과는 [CONTEXT.md](CONTEXT.md)에 기록한다.
Task·향후 actor 관측·reward·종료 계산은 inactive
instance를 active mask로 제외해야 한다. 센서 dictionary observation은 구현하며
학습용 물체 기하·history·policy encoder는 아직 구현 전이다.

## 11. 현재 생성되는 과업의 범위와 검증

**현재 균형 구성의 기하 검사:** 최종 source의 실제 USD bounds로 일반·all-types
각각 seed 0–19를 생성해 **40/40개**가 통과했다. 모든 장면에서 non-target
종류별 active 개수 차이가 최대 1이었다. 일반 모드는 12–25 instance,
global 35.05–38.48%, 검사 전체의 최소 구역 점유율 22.86%였다.
All-types는 15–18 instance, global 35.14–38.71%, 최소 구역 24.75%였다.
Source SHA를 포함한 자료는
`outputs/contact_rearrangement/balanced_composition_benchmark.json`에 있다.
이 수치와 이전 면적 우선 구성의 calibration을 구분한다.

**현재 균형 구성의 GPU 검사:** 일반 모드를
`--num_envs 4 --headless --device cuda:0 --max_steps 151 --validate`로 실행해
exit code 0으로 통과했다. 5.03초 동안 기본 150-step 자동 reset과 마지막
전체·부분 reset, 기존 정착 guard를 검사했다. 역할 OBB 총 96개 line과 draw 전후
USD·PhysX 상태 유지도 확인했다. Reset sample 중 10종 장면도 포함됐지만
현재 구성의 별도 `--all_asset_types` GPU 실행을 완료한 것으로 기록하지 않는다.
로그는 `outputs/contact_rearrangement/balanced_composition_gpu_validation.log`다.
단위 검사 **23개**가 24.384초에 통과했고 균형 선정·복사본 재고 소진 회귀를
포함한다. 미검사 seed·robot push·camera visibility의 성공으로 확대하지 않는다.

**이전 면적 우선 구성의 기하 검사:** 실제 USD 측정 bounds에서 35% 요구는 일반·
all-types 각각 seed 0–19, **고유 40개 sample 모두** 통과했다. 일반 모드는
4–10종·10–24 instance, all-types는 10종·14 instance였다. Global 점유율은
각각 35.39–38.68%·36.32–37.36%였고, 장면별 최소 구역 점유율은 전체 sample에서
12.98–35.22%였다. 25%·30%는 초기 각 10/10개, 40%는 고유 30/30개,
45%는 초기 3/10개가 통과했다. 35%와 40%의 공통 seed 집합에서 평균 생성 시간은
약 0.206초와 0.99–1.00초였다. 초기·독립·확장 검사의 중복 seed를 합산하지 않는다.
자료는 `outputs/contact_rearrangement/density_calibration.json`에 있다.
이는 geometry 생성 검사이며 물리 정착·robot push·camera 검사는 아니다.

**이전 면적 우선 구성의 GPU 검사:** 기본 35%의 일반 모드와 `--all_asset_types` 모드를 각각
`--num_envs 4 --headless --device cuda:0 --max_steps 300 --validate`로 실행해
둘 다 exit code 0으로 통과했다. 각 실행은 10초이며 기본 150-step 간격의
전체 reset과 마지막 부분 reset을 검사했다. 각 모드의 초기 4개와 첫 reset 뒤
4개, **8개 layout을 각각 5초** 진행했다. 마지막 전체 reset 뒤 생성한 장면은
추가 물리 step이 없어 정착 표본으로 세지 않는다. 기존 5 cm drift·15° tilt
기준을 완화하지 않았다. 로그는
`outputs/contact_rearrangement/dense_shelf_default_validation.log`와
`outputs/contact_rearrangement/dense_shelf_all_types_validation.log`다.
단위 검사 **21개**가 18.219초에 통과했고 `py_compile`·`git diff --check`도
통과했다. 지역 요구가 global보다 큰 경우의 composition 면적 산정도 검사한다.
이 결과를 미검사 seed·실제 robot push·camera 가시성으로 확대하지 않는다.
현재 균형 구성은 density·지역·이웃·clearance 조건을 유지하면서 복사본 선정만
바꿨다. 위 이전 기하·GPU 결과와 21개 단위 검사 기록을 현재 구성의 통과
결과로 승격하지 않으며 현재 결과는 이 절 앞에 별도로 기록했다.

**Flat-box 적용 전 전체 선반 규칙의 검사:** unit test 20개와 실제 USD 측정 bounds의 일반 100 seed·
10종 전체 30 seed가 통과했다. 새 100-body pool의 4개 환경·10종 전체 장면은
매 step reset하는 24 step 실행과 마지막 부분 reset을 통과했다. 이 실행은
총 0.8초이고 각 layout을 1/30초만 진행시켜 pool 수명주기를 확인한 것이다.
같은 조건의 일반 4–10개 모드에서도 모든 개수·중복·Target 유일성과 reset을 검사했다.
같은 10종 전체 장면을 4초간 유지한 검사는 env 0·3의 003이 5 cm 이상 움직여
실패했다. 초기 기하 조건과 reset의 통과를 장시간 settle 안정성으로 확대하지 않는다.
자세한 명령과 로그는 [CONTEXT.md](CONTEXT.md)에 기록한다.

종류·instance 분리 및 hard occupancy 추가 전 버전은 전물체의 2 mm 여유를 제거하고 003·004의 collision을 명시적 Cube proxy로
바꿨다. 별도 4개 환경·10종 전체·10초 실행에서 8개 layout의 각 5초 정착과
전체/부분 reset을 통과했다. 당시 all-types는 정확히 10개 instance였으며 현재의
density 산정 instance 수와 다른 분포다. 이를 새 density 버전의 검증 결과로
승격하지 않는다. 이후 면적 우선 4–10종/hard occupancy 구성의 결과는 위 별도
검사로 기록하며 현재 균형 구성의 검증과 구분한다.

현재는 **단일 지정 Blocker와 주변 물체가 있는 밀집 baseline**이다. 주변 물체는
초기 A₀와 Blocker의 witness 경로를 비우므로, 주변 물체를 먼저 연쇄적으로
재배치해야만 Target 접근을 확보할 수 있는 장면을 요구하지 않는다. 밀집도
조건과 단일 free path의 존재를 동시에 만족시키도록 장면을 단순화하고 있다.
경로는 lateral translation에 한정하지 않으며 실제 모서리 pivot·회전·병진 조합을
포함한다. 복수 Blocker 조작 순서를 결정하는 과업은 아직 구현하지 않았다.

Robot 시작 판정은 해당 초기 pose의 IK·scene collision·elbow 조건을
검사한다. 재배치 경로 전체의 joint reachability·hand/arm swept collision, 실제
접촉점, friction으로 밀 수 있는지, OSC 추종·전도·낙하의 동역학 조건은 없다.
수평 footprint 경로가 있다는 사실과 로봇이 수행할 수 있다는 사실은 구별한다.
또한 A₀의 10 cm hand width, margin, 30% 차단, 전체/구역 occupancy·coverage·국소 근접은 변경할
수 있는 구현값이다. 이후 실제 hand geometry와 제어·센서 검증에 맞춰 구체화해야
한다. Spawn criteria의 충족을 재배치 정책의 성공 판정이나 연구 성능의 증명으로
사용하지 않는다.

**이전 4–6개 baseline의 검증 기록:** YCB 방향 교정 후 기본 seed 0·8개 환경·
6개 active·8초 반복 reset 검사는
통과했지만, seed 7·006 Target/003 Blocker 고정 검사에서는 한 장면의 003이
약 30° 기울어지고 XY 약 5.5 cm 움직여 실패했다. 해당 장면을 1/120초 접촉
기록으로 재현했을 때 선반에만 닿은 상태에서 먼저 기울었고, 1.46초부터 옆의
004에 닿았다. Blocker–robot 접촉은 없었다. 따라서 현재 planar spawn 판정만으로
settle 이후 물체의 자세·위치 안정성을 보장할 수 없다. 상세 조건과 로그는
[CONTEXT.md의 검증 기록](CONTEXT.md)에 남긴다. 이 이전 실행을 새 100-body pool·
전체 선반·pivot 경로 규칙의 검증 완료로 해석하지 않는다.

## 12. 관련 구현 위치

현재 구조는 `examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy`를 따라
공통 환경 설정·로봇별 설정, `mdp`의 manager 함수, `src`의 장면 지원 구현을
분리한다. 모델과 학습 설정을 포함한 전체 구조는
[README](../src/contact_rearrangement/README.md)에 있다.

| 파일 | 이 문서와 연결되는 역할 |
|---|---|
| [src/catalog.py](../src/contact_rearrangement/contact_rearrangement/src/catalog.py) | 10종 catalog·upright·10-copy pool·parking 위치 |
| [src/assets.py](../src/contact_rearrangement/contact_rearrangement/src/assets.py) | 로봇/F/T 조립·USD FK/IK·물체 생성·countergravity·Cube collider |
| [shelf_rearrangement_env_cfg.py](../src/contact_rearrangement/contact_rearrangement/shelf_rearrangement_env_cfg.py) | 공통 shelf/camera·manager 연결·SpawnRules·RobotStartRules·seed·주기 |
| [config/ur5e/osc_env_cfg.py](../src/contact_rearrangement/contact_rearrangement/config/ur5e/osc_env_cfg.py) | UR5e/RH56E2·object pool·17채널 tactile·장면 확인/학습 설정 |
| [mdp/events.py](../src/contact_rearrangement/contact_rearrangement/mdp/events.py) | USD bounds·역할/reset·robot 후보·camera retry·commit/rollback |
| [src/sampling.py](../src/contact_rearrangement/contact_rearrangement/src/sampling.py) | 순수 density·SAT·SE(2) 경로·robot 영역/elbow/3D OBB helpers |
| [src/sensors.py](../src/contact_rearrangement/contact_rearrangement/src/sensors.py) | Camera/reference render·pixel acceptance·snapshot·raw F/T·tactile |
| [src/visualization.py](../src/contact_rearrangement/contact_rearrangement/src/visualization.py) | 현재 pose의 역할 OBB debug draw |
| [mdp/state.py](../src/contact_rearrangement/contact_rearrangement/mdp/state.py) | 학습 환경의 history·reward 누적량 reset과 transition 상태 |
| [spawn_scene.py](../src/contact_rearrangement/scripts/spawn_scene.py) | CLI·5초 reset·실제 스폰/settle·전체/부분 reset 검증 |

Simulator 없이 읽는 catalog와 순수 sampler를 바탕으로 AppLauncher 이후
공통 환경 설정과 UR5e concrete 설정이 asset·sensor scene을 만든다. Reset에서는
`mdp/events.py`가 실제 USD geometry를 측정하고 역할·종류·active ID·개수를 정한
뒤 sampler로 배치와 robot 후보를 승인한다. 선택한 환경의 native state를 적용한
후 `src/sensors.py`가 실제 렌더의 pixel acceptance를 검사한다. Camera 후보가
실패하면 같은 composition으로 layout을 재시도하며 성공한 pose·metadata·초기
snapshot을 함께 commit한다. 정해진 retry가 모두 실패하면 이전 선택 환경의
physical state·metadata·sensor snapshot을 복구한다. 이 과정에 settling이나
추가 physics step을 삽입하지 않는다. 학습 환경은 장면 reset 성공 뒤
`mdp/state.py`의 history·reward 누적량도 reset한다.

Step에서는 raw F/T·tactile·pool countergravity·OBB 표시를 갱신한다.
밀도·선정·clearance 수치는 `SpawnRules`와 `src/sampling.py`, 실제 USD geometry와
reset 적용은 `mdp/events.py`에서 찾는다. 이전 flat 경로의 wrapper는 남기지 않는다.
2026-10-09의 6개 모듈 통합 때는 문법·CLI help, simulator 없는 catalog/sampling
import, 23개 unit test와 4개 GPU 환경·5.03초의 전체/부분 reset·OBB 불변 검사가
통과했다. 물리 guard는 유지했다. 이 결과를 현재 구조 변경의 검증으로 사용하지
않으며 이전 조건·로그와 후속 검증은 [CONTEXT.md](CONTEXT.md)에서 구분한다.
