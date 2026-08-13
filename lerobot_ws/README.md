# LeRobot Workspace

`Manipulator/` 본체(ROS2/MoveIt/Isaac Sim 파이프라인)와는 완전히 분리된 별도 워크스페이스.
LeRobot + SmolVLA(언어모델 결합 VLA) 실험을 위한 공간이며, 기존 `rb5_binpicking` / `rb5_isaac` /
`rbpodo_ros2` 는 건드리지 않는다.

```
lerobot_ws/
├── lerobot/               huggingface/lerobot 클론 (v0.6.2, 2026-08-05 기준 main)
├── lerobot_robot_rb5/     pip 패키지. RB5-850E용 커스텀 LeRobot Robot 플러그인 (--robot.type=rb5_850e)
├── ros_bridge/            rb5_lerobot_bridge.py — ROS2(Humble, py3.10) 쪽에서 도는 ZMQ 브릿지, 위 플러그인의 상대편
├── tools/                 convert_isaac_episodes_to_lerobot.py — 데이터셋 변환 스크립트 스캐폴드
└── dual_robot/            매니퓰레이터 2대 릴레이 pick-and-place (아래 참조)
```

- `lerobot/` 은 자체 git 저장소(자체 `.git`)이므로 Manipulator 저장소와 이력이 섞이지 않는다.
- LeRobot 요구 Python **3.12+** — 기존 conda 환경(`ground`, `isaaclab`, `physical` 등, 전부 3.10~3.11)과
  호환 안 됨 → 새 conda env(예: `lerobot`, python=3.12) 또는 `uv` 기반 venv를 이 폴더 안에서 별도로 구성 필요.
  (`lerobot/CLAUDE.md` 는 `uv sync --locked` 사용을 권장.)
- GPU: RTX 4090 24GB — SmolVLA(450M) fine-tuning에 충분 (공식 문서 기준 A100 1장으로 20k step ~4h).

## 분석 요약 (2026-08-05)

### 1. Robot 추상 인터페이스 (`src/lerobot/robots/robot.py`, `config.py`)

`Robot(abc.ABC)` 가 요구하는 것:
- 프로퍼티: `observation_features`, `action_features`, `is_connected`, `is_calibrated`
- 메서드: `connect()`, `calibrate()`, `configure()`, `get_observation()`, `send_action()`, `disconnect()`

시리얼/Dynamixel 전제가 아니다 — `get_observation`/`send_action` 내부 구현은 자유이므로 ROS2
토픽/액션(MoveIt `FollowJointTrajectory` 등)으로 채워도 무방. `RobotConfig` 는
`draccus.ChoiceRegistry` 라 `@RobotConfig.register_subclass("rb5_850e")` 형태로 등록하면
`--robot.type=rb5_850e` 로 모든 CLI에서 바로 사용 가능.

**참고할 기존 구현체**
- `robots/reachy2/` — SDK/네트워크 기반 로봇의 표준 템플릿 (조인트 이름 매핑 dict 패턴). RB5를
  감쌀 때 가장 가까운 참조.
- `robots/unitree_g1/` — 별도 소켓/브릿지 프로세스(`run_g1_server.py`)로 SDK를 격리하는 패턴.
  ROS2 Humble(Python 3.10 ABI)과 LeRobot(Python 3.12)의 인터프리터 충돌을 피하려면 이 구조가
  유력함 — RB5 제어는 기존 `physical` conda env(ROS2 Humble)에서 별도 프로세스로 돌리고,
  LeRobot 쪽 `Robot` 클래스는 소켓/ZMQ로만 통신.
- `robots/lekiwi/` — client/host 분리 패턴, 네트워크 너머 로봇 제어 시 참고.

### 2. 데이터셋 변환 (`src/lerobot/datasets/`)

`lerobot-record` 로 실제 텔레옵할 필요 없이, `LeRobotDataset.create(repo_id, fps, features, ...)`
→ 에피소드 루프에서 `add_frame(frame)` (frame에 `"task"` = 언어 지시문 문자열 포함) →
`save_episode()` → 마지막 `finalize()` 로 임의 소스 데이터를 LeRobotDataset으로 변환 가능.
템플릿: `examples/port_datasets/port_droid.py` (하드웨어 없이 외부 RLDS 데이터셋을 통째로 변환).

→ Isaac Sim 로그(조인트 상태, EEF pose, RGB, 언어 지시문)나 ROS2 bag을 이 패턴으로 그대로
LeRobotDataset repo로 포팅할 수 있음. `rb5_binpicking/scripts/binpicking_scene.py` 가 이미
퍼블리시하는 `/binpicking/object_pose` + 카메라 토픽을 재사용하면 시뮬레이션 기반 데이터 수집
스크립트를 새로 짤 필요 없이 로깅만 추가하면 될 수도 있음.

### 3. SmolVLA (`src/lerobot/policies/smolvla/`)

- `chunk_size=50`, `n_action_steps=50` (액션 청크 호라이즌), `n_obs_steps=1`
- **`max_state_dim=32` / `max_action_dim=32`** — 상태/액션 벡터는 32차원까지 자동 zero-padding
  됨 (`pad_vector`). 즉 RB5의 6/7-DoF 조인트 + 그리퍼, 혹은 6D EEF pose + 그리퍼 어느 쪽이든
  차원 문제 없이 그대로 fine-tuning 가능 — pretrained(`smolvla_base`, SO-100 기준)와 정확히
  차원을 맞출 필요 없음.
- 카메라: 키/개수 자유, `(512,512)` 로 리사이즈+패딩. 언어 지시문은 토크나이저로 인코딩
  (`tokenizer_max_length=48`).

### 4. 주요 CLI (`pyproject.toml [project.scripts]`)

`lerobot-record`(텔레옵 데이터 수집) / `lerobot-train`(정책 학습) / `lerobot-eval`(sim 평가) /
`lerobot-rollout`(실물/시뮬 배포 실행) / `lerobot-calibrate` / `lerobot-teleoperate` /
`lerobot-dataset-viz` / `lerobot-edit-dataset` 등.

## 진행 상황 (2026-08-05, 두 번째 세션)

작업 시점에 `rb5_isaaclab/scripts/train.py` RL 학습(PID, `isaaclab` conda env, GPU ~53%/6GB)이 돌고 있어서
**GPU를 쓰지 않는 작업까지만** 진행함 (conda env 생성, 패키지 설치, 코드 스캐폴딩, 실제 접속 없이
설정 해석만 검증). 실제 fine-tuning/추론(GPU 사용)은 보류.

1. ✅ conda env `lerobot` (python=3.12) 생성 + `uv pip install -e ".[dataset,training,smolvla]"` 완료.
   `conda activate lerobot` 로 사용.
2. ✅ `lerobot_robot_rb5/` — RB5-850E용 커스텀 `Robot` 서브클래스(`RB5850E`, `--robot.type=rb5_850e`).
   ZMQ REQ/REP로 `ros_bridge/rb5_lerobot_bridge.py` 와 통신 (rclpy는 Python 3.10/Humble ABI라 `lerobot`
   env(3.12)에서 직접 import 불가 → 프로세스 분리, `unitree_g1` 패턴 참고). 패키지명이
   `lerobot_robot_rb5` 라 lerobot의 `register_third_party_plugins()` 가 자동 인식 —
   lerobot 소스 자체는 한 줄도 수정하지 않음. `pip install -e .` 후 `--robot.type=rb5_850e` 로
   config/factory 해석까지 확인함 (연결 자체는 브릿지 미기동 상태라 실패하는 게 정상).
3. ✅ `ros_bridge/rb5_lerobot_bridge.py` — ROS2 Humble 쪽에서 도는 독립 rclpy 스크립트(colcon 패키지 아님).
   `/joint_states`, `/camera/color/image_raw` 구독, `/isaac_joint_commands` + `/gripper_joint_commands`
   로 직접 퍼블리시 (trajectory_bridge.py와 동일 토픽 — **MoveIt 휴리스틱 데모와 동시 실행 금지**,
   같은 커맨드 토픽을 두 발행자가 놓고 싸우게 됨). 그리퍼 mimic 관절 상수는 trajectory_bridge.py에서
   그대로 복사 — 수정 시 양쪽 다 갱신 필요. 아직 실제 Isaac Sim/실물 로봇 대상으로 실행해보지 않음.
4. ✅ `tools/convert_isaac_episodes_to_lerobot.py` — 완성. `iter_episodes()`는 `ros_bridge/episode_logger.py`
   가 남기는 `episode_XXXX/{state.npz, images_color.npy, task.txt}` 포맷을 읽음.
5. ✅ `ros_bridge/episode_logger.py` — **기존 파일은 하나도 건드리지 않고**, `binpicking_scene.py` +
   `trajectory_bridge.py`(MoveIt 휴리스틱 데모)가 이미 퍼블리시하는 토픽(`/joint_states`,
   `/isaac_joint_commands`, `/gripper_joint_commands`, `/camera/color/image_raw`)을 그냥 구독만
   해서 에피소드로 저장하는 수동적 로거. 즉 지금 돌아가는 Phase-1 휴리스틱 pick-place를 그대로
   시연 데이터 수집원으로 쓸 수 있음 — Enter 키로 녹화 시작/종료 구간을 표시.
6. ✅ **더미 데이터로 변환 파이프라인 전체를 스모크 테스트함** (GPU 불필요, CPU만 사용): 가짜
   에피소드 2개(20프레임씩, 랜덤 상태/액션/이미지) → `convert_isaac_episodes_to_lerobot.py` 실행 →
   AV1 비디오 인코딩까지 정상 완료 → `LeRobotDataset`으로 다시 로드해서 shape까지 확인
   (`observation.state`: (7,), `observation.images.color`: (3,480,640)). 실데이터만 꽂으면 되는 상태.

## 실제 Isaac Sim으로 end-to-end 실행 완료 (2026-08-06)

RL 학습이 끝난 뒤 사용자 확인을 받고, headless Isaac Sim으로 1~3단계를 전부 실제 데이터로
검증함. (headless 실행은 `binpicking_scene.py`의 `CONFIG["headless"]`를 `sed`로 바꾼 임시
사본으로 했음 — 원본 파일은 그대로. `~/isaacsim/python.sh <임시복사본>`.)

1. ✅ **브릿지 왕복 테스트**: `binpicking_scene.py`(headless) + `rb5_lerobot_bridge.py` 띄운 채로
   `RB5850E.connect()` → `get_observation()`(조인트 6개 + `color` 480x640x3 이미지) →
   `send_action()`으로 `wrist3`를 +0.3rad 이동 → 재조회해서 실제로 `wrist3: 0.0 → 0.3` 반영된 것까지
   확인. 전체 경로(lerobot_robot_rb5 → ZMQ → rb5_lerobot_bridge → `/isaac_joint_commands` → Isaac Sim
   → `/joint_states` → 역방향)가 실제로 동작함.
2. ✅ **실제 에피소드 녹화**: 기존 파이프라인(`binpicking.launch.py` + `moveit_pick_place.py`, 아무것도
   수정 안 함) 그대로 돌리고 그 옆에 `episode_logger.py`를 비대화형 모드(`--duration 150 --count 2`,
   신규 추가)로 붙여서 실제 에피소드 2개 녹화 (150초씩, ~24fps, 각 3629/3743 프레임). 녹화 도중
   trial 1은 grasp 성공(test-lift로 32mm 상승 확인), trial 2/3은 물체가 옆으로 넘어져 있어 grasp
   거부(`tilt 170.8deg > max_object_tilt`)로 실패 — 3연속 실패 시 자동 정지하는 기존 안전장치가 그대로
   작동함(내가 만든 코드가 아니라 moveit_pick_place.py 자체 로직). 즉 녹화된 2개 에피소드는 성공/실패가
   섞여 있음 — 다음 녹화부턴 trial 성공/실패에 맞춰 에피소드 경계를 끊는 게 나을 듯.
   (`episode_logger.py` 버그 하나 고침: 원래 `/joint_states`마다도 프레임을 찍어서 카메라(~24fps)보다
   훨씬 빠른 physics-rate로 중복 프레임이 쌓이던 것을, 이미지 콜백에서만 프레임을 찍도록 수정.)
3. ✅ **실제 LeRobotDataset 생성**: 위 2개 에피소드 → `convert_isaac_episodes_to_lerobot.py` →
   2 episodes / 7372 frames, AV1 인코딩 후 6.7GB(raw) → 62MB. `LeRobotDataset`으로 재로딩해서 shape
   확인 완료.
4. ✅ **SmolVLA fine-tuning 스모크 테스트**: `lerobot-train --policy.path=lerobot/smolvla_base
   --policy.push_to_hub=false --dataset.repo_id=local/rb5_pick_place_v0 --dataset.root=<위 데이터셋>
   --rename_map='{"observation.images.color":"observation.images.camera1"}' --batch_size=4 --steps=20`
   — `smolvla_base`(전체 450M / 학습 대상 100M) 다운로드 → 카메라 키 이름 불일치는 `--rename_map`으로
   해결 (에러 메시지가 정확히 안내해줌) → 20 스텝 정상 진행, loss 0.2~3 사이에서 노이즈 있지만
   전반적으로 하강, GPU 메모리 1.8GB만 사용, 체크포인트 저장까지 확인.

결과물은 전부 세션 scratchpad(`/tmp/claude-.../scratchpad/`)에 있음 — **세션이 끝나면 정리될 수 있는
임시 위치**라 계속 보존하려면 옮겨야 함: `episodes/`(원본 녹화), `lerobot_rb5_dataset/`(LeRobotDataset,
62MB), `smolvla_smoketest/checkpoints/000020/`(체크포인트, 1.3GB).

작업 중 실행했던 Isaac Sim/ROS2 프로세스(headless Isaac Sim, move_group, rviz2, trajectory_bridge,
depth_to_pointcloud, scene_setup, moveit_pick_place.py)는 전부 정리해서 GPU는 다시 유휴 상태.

## 매니퓰레이터 2대 릴레이 pick-and-place (2026-08-10)

블럭 하나를 로봇 A가 소스 빈 → 중앙 핸드오프 트레이로 옮기고, 로봇 B가 그걸 집어서 자기
목적지 빈으로 옮기는 릴레이. `--cycles N` 이면 되돌리기(B→핸드오프→A→소스)까지 해서 반복.
`dual_robot/` 안에만 있고 `rb5_binpicking` / `rb5_isaac` / `rbpodo_ros2` 는 전혀 건드리지 않음.

```
dual_robot/
├── layout.py                        모든 좌표의 단일 소스 (씬과 컨트롤러가 공유 → 어긋날 수 없음)
├── dual_binpicking_scene.py         Isaac Sim 씬: URDF를 두 번 임포트, 트레이 3개, 블럭 1개
├── dual_scene_setup.py              각 로봇 planning scene에 실제 트레이/상대로봇 발행
├── namespaced_trajectory_bridge.py  trajectory_bridge 포크 (액션 이름만 상대경로화)
├── dual_binpicking.launch.py        robot_a / robot_b 네임스페이스로 MoveIt 스택 2벌
└── relay_pick_place.py              릴레이 컨트롤러
```

실행:
```bash
~/isaacsim/python.sh dual_robot/dual_binpicking_scene.py          # 터미널 1
ros2 launch dual_robot/dual_binpicking.launch.py                  # 터미널 2
python3 dual_robot/relay_pick_place.py --cycles 1                 # 터미널 3
```
블럭 리셋 `/binpicking/reset_object` (Empty), 임의 위치 이동 `/binpicking/place_object` (Point)
— 한 단계만 반복 테스트할 때 Isaac 재시작(~1.5분) 없이 쓰라고 넣음.

### 실제로 문제였던 것들 (전부 증거 기반으로 잡음)

싱글로봇 데모를 그대로 두 벌 돌리면 될 줄 알았는데, 아래가 전부 실제로 걸렸음:

1. **IK 솔버 타임아웃 5ms** — `rbpodo_moveit_config/config/kinematics.yaml` 의 MoveIt Setup
   Assistant 기본값. 같은 4cm 직선 이동에 대해 `compute_cartesian_path` 가 fraction 0.00 /
   0.50 / 0.25 / 0.88 을 무작위로 뱉고, 충돌검사를 꺼도 더 나빠지는 게 증거였음(기하 문제가
   아니라 시간 초과). Pilz LIN 의 `-31 NO_IK_SOLUTION` 도 대부분 이것. **런치 파일에서만
   0.05s 로 오버라이드** (공용 config는 안 건드림). → 플래너 실패가 크게 줄어듦.
2. **관절 리미트가 시뮬레이터에서 강제되지 않음** — 세션이 길어지자 base 관절이 누적 회전해서
   `/joint_states` 가 **-4.4049rad** 을 보고. URDF 한계(±3.14) 밖이라 그 순간부터 그 팔의 모든
   플래닝이 이유 없는 `FAILURE` 로 즉시 실패. 세션 내내 "팔이 이상하게 멈춤"으로 보이던 것의
   정체. 씬에서 각 관절에 ±178° 스톱을 명시하고, 컨트롤러에도 범위 밖이면 바로 알려주는 체크 추가.
3. **collision object 가 애초에 도착하지 않았음** — `scene_setup.py` 는 절대 토픽
   `/collision_object` 로 발행하는데 move_group 은 상대 토픽(=`/robot_a/collision_object`)을
   구독. 즉 두 플래너 모두 **빈 세계**를 상대로 계획하고 있었음. 게다가 좌표도 싱글로봇 것이라
   핸드오프 트레이는 존재 자체를 몰랐음. → `dual_scene_setup.py` 로 교체하고
   `get_planning_scene` 으로 16개 오브젝트가 실제로 들어간 것 확인.
4. **놓는 높이가 7mm 낮았음** — 트레이 바닥판(7mm) 두께를 빼먹어서, 블럭을 잡은 채로 바닥에
   박아넣고 놓는 순간 PhysX가 밀어내며 **5m 날려버림**. 트레이 바닥면 기준으로 계산하도록 수정.
5. **그리퍼 열림이 거짓 실패** — 열기 램프가 0.5s인데 실제 드라이브는 ~1.8s 필요. 측정해서
   `gripper_open_duration:=1.5`, settle 0.8s로. 전에 "release 실패"로 보이던 게 이거였음.
6. **잡는 힘의 딜레마 → 2단 클로즈** — 32mm로 한 번에 조이면 42mm 큐브가 패드 사이에서
   튀어나가고(4회 프로브 중 2회, 양쪽 패드 접촉은 매번 확인됨), 36mm로 살살 잡으면 0.5m 옮기는
   도중에 떨어뜨림. **36mm로 먼저 닿게 한 뒤 24mm로 조이는** 2단으로 둘 다 해결.
7. **블럭에 마찰 재질이 없었음** — 그리퍼 링크에만 high-friction을 걸고 정작 블럭은 기본값이라
   접촉면 마찰이 낮았음. 블럭에도 적용(1.5/1.2).
8. **큐브 대칭을 무시한 yaw 누적** — 블럭은 축정렬 상태로 시작하므로 A는 좋은 손목 자세로 잡는데,
   A가 놓으면 블럭 yaw가 그리퍼 yaw(~89°)가 됨. B는 그걸 읽어 **또 89°를 더해서** 완전히 다른 IK
   가지로 감 → B가 3번 연속 허공을 잡음(그리퍼가 stall 없이 목표까지 닫힘 = 사이에 아무것도 없음).
   큐브는 90° 대칭이므로 yaw를 [-45°,45°)로 접도록 수정 → -175.8° → +4.2°.
9. **`compute_fk` 에 그리퍼 관절을 보내면 move_group 이 죽음** — 런치에서 그리퍼 관절을 fixed로
   바꿔 MoveIt 모델엔 없는 변수라, 이름을 넣으면 `moveit::Exception` 이 **잡히지 않고** 프로세스가
   terminate. 6개 팔 관절만 보내도록.

이 외에 검증 장치를 넣어서 조용한 실패가 없게 함: FK로 실제 TCP 확인(플래너의 SUCCESS는 "허용
오차 안"이지 "정확히 거기"가 아님), 테스트 리프트로 grasp 확인, 운반 중 블럭이 아직 손에 있는지
확인, 놓기 직전 타겟 위에 있는지 확인, 실패 시 블럭 위치를 다시 보고 재시도(최대 4회), 팔이 낀
경우 관절공간 홈으로 복구.

### IK 솔버 교체 (KDL → LMA) 와 그 뒤에 드러난 것들

grasp 성패가 TCP 오차와 강하게 상관(3.4mm 이내면 성공, 5.4mm 이상이면 실패)한다는 게 보여서
근본 원인인 IK를 손댐. `KDLKinematicsPlugin` 은 수렴이 막히면 **랜덤 시드로 재시작**해서, 같은
목표 포즈에 매번 다른 팔 자세를 돌려줌 → 그게 오차 변동의 정체였음.

`lma_kinematics_plugin/LMAKinematicsPlugin` (Levenberg-Marquardt, Humble에 이미 설치돼 있음)으로
교체 — 공용 config 대신 런치 파일에서만 오버라이드. 효과가 명확했음:

| | KDL | LMA |
|---|---|---|
| 같은 grasp 목표 4회 반복 시 TCP 오차 | 1.6 / 3.2 / 6.1 / 12.0mm | **3.2 / 3.2 / 3.2 / 3.2mm** |
| `compute_cartesian_path` 관절 이동량(18cm 하강) | 37~50 rad (특이점 순회) | **0.39~0.66 rad** |

> 더 좋은 선택지는 `pick_ik` (`sudo apt install ros-humble-pick-ik` → 플러그인 이름
> `pick_ik/PickIkPlugin`). MoveIt2의 현재 권장 솔버인데 설치에 sudo 비밀번호가 필요해서 못 넣었음.
> 넣으면 런치 파일의 `kinematics_solver` 한 줄만 바꾸면 됨.

**부수 효과 — 직선 하강이 되살아남.** 원래 `compute_cartesian_path` 로 수직 하강을 직선화하려다
뺐었음(위 표의 37~50 rad 경로 때문에 200초짜리 궤적이 나오고 컨트롤러가 중단). 그 원인이 KDL이었던
거라 LMA에선 fraction=1.00 / 0.4~0.7 rad 의 멀쩡한 직선이 나옴 → 다시 넣음. 단
`CARTESIAN_MAX_JOINT_TRAVEL` 검사는 남겨둠 (fraction=1.00 이 경로가 멀쩡하다는 보증이 아니라는 걸
겪었으므로).

**그리고 남은 변수를 하나씩 측정으로 지움.** IK가 결정론적이 되자(3.2mm 고정) B가 *같은 포즈에서*
성공/실패를 반복하는 게 드러나서, 하나씩 배제:
- 접근 경로가 블럭을 치는 것 → 직선 하강으로 제거. 그래도 실패.
- 툴 기울어짐(핑거팁이 툴 원점 150mm 아래라 2.9° 기울면 7.6mm 어긋남) → 검증 코드를 넣어
  측정하니 **0.00°**. 배제.
- 그리퍼 명령이 안 유지됨 → 65초간 knuckle 0.5854rad 고정 확인. 배제.

남은 차이는 **도달 거리** 하나였음. A는 0.51m에서 안정적으로 잡는데 B는 0.67m — 팔이 많이 펴진
영역. 베이스 간격을 1.0m → 0.8m로 줄여서 핸드오프를 양쪽 0.60m로 당김("핸드오프는 두 트레이의
중앙" 이라는 요구는 그대로 유지됨). 그 뒤 B는 grasp 포즈에 도달한 시도마다 전부 성공(+35.3mm,
+36.7mm).

### 결과

**왕복 반복까지 동작함.** `--cycles 2` 실행에서 사이클 1 정방향과 복귀가 모두 완주:

| | 단계 | grasp TCP 오차 | grasp 확인 | 놓은 위치 |
|---|---|---|---|---|
| 1 정방향 | A: 소스 → 핸드오프 | 1.6mm, 0.00° | +36.1mm | 4mm |
| | B: 핸드오프 → 목적지 | 1.3mm, 0.00° | +38.6mm | 3mm |
| 1 복귀 | B: 목적지 → 핸드오프 | 1.1mm, 0.00° | +36.9mm | 7mm |
| | A: 핸드오프 → 소스 | 1.5mm, 0.00° | +37.1mm | 5mm |
| 2 정방향 | A: 소스 → 핸드오프 | 2.3mm, 0.00° | +38.7mm | 운반 중 놓침 |

IK 교체 뒤 grasp가 전부 1.1~3.2mm / 0.00° 로 들어옴 (KDL 시절 1.6~12mm 변동 대비). 4번 연속
transfer 성공 후 5번째에서 실패 — 실패 지점은 아래 하나로 좁혀짐.

**남은 문제 하나: 운반 중 미끄러짐.** grasp 확인됐고(+38.7mm), 궤적도 중단 없이 깨끗하게
실행되는데(pos_err ~0.003rad) 이동 도중 블럭이 빠짐. 다음은 전부 측정으로 배제했음:
- 그리퍼 명령이 안 유지됨 → 65초간 knuckle 0.5854rad 고정 확인. 아님.
- 컨트롤러 중단으로 인한 급정거 → 해당 구간에 abort 없음 확인. 아님.
- 접근 경로가 블럭을 침 / 툴이 기울어짐 → 직선 하강 + 0.00° 측정으로 배제.
- 블럭 마찰 → high-friction 재질 적용함.

남은 후보는 스윙 중 관성 하나. 속도 스케일 0.03 + 운반 높이 0.18m 로 낮춰서 빈도가 크게 줄었고
(그 전엔 첫 transfer에서도 자주 떨어뜨림), 재시도 루프가 대부분 흡수하지만 완전히 없어지진 않음.
다음에 손댈 곳: 그리퍼 핑거팁 조인트 damping (씬 파일에 근거와 함께 주석으로 남겨둠 — 한 번
900으로 올려봤으나 효과를 입증하지 못해 원래 120으로 되돌림), 또는 운반 구간만 더 느리게.

**실행 사이에 시뮬레이터를 재시작할 것.** 실패로 끝난 실행은 팔을 이상한 자세에 남겨두고, 다음
실행은 거기서 시작하기 때문에 이유 없이 또 실패함 (한 번은 이것 때문에 코드 변경의 효과를 잘못
판단할 뻔했음). ROS 스택만 재시작하면 팔 자세는 그대로 남으니 씬까지 같이 내렸다 올려야 함.
컨트롤러에 복구 경로(수직으로 20cm 들어올리기 → 관절공간 홈)를 넣어두긴 했지만 만능은 아님.

기타 알려진 제약:
- 한 transfer 당 ~2분. 이동 하나마다 플래너 폴백을 최대 3번 시도하는 구조라 느림.
- `/tf` 는 두 로봇이 같은 프레임 이름(link0..tcp)을 전역 `/tf` 에 같이 발행해서 사실상 못 씀.
  플래닝은 각자 자기 link0 기준이라 영향 없지만, RViz나 tf 기반 측정은 불가 (그래서 이 코드는
  TCP 위치를 tf 대신 `compute_fk` 로 읽음).

## 안정화 완료 — 2사이클 무중단 완주 (2026-08-11)

`--cycles 2` 가 **6/6 transfer 전부 성공**하고 `Relay complete` 로 끝남. 총 277초(이송당 ~46초).
재시도는 1회(B `close` 실패 → 재관측 후 성공). 앞 절에 남아 있던 "운반 중 미끄러짐"과, 그 뒤에
드러난 더 큰 원인 두 개를 잡은 결과.

### 원인 1: IK 브랜치가 매번 달랐음 (제일 컸음)

`pre_grasp` 를 **TCP 포즈 목표**로만 줬기 때문에, 6축 팔이 같은 툴 자세를 만드는 여러 관절 해
중 OMPL이 그때그때 아무거나 골랐음. 도착한 자세에 따라 그 다음 18cm 수직 하강 비용이 완전히
달라짐. `ik_seed_probe.py` 로 측정:

| IK 시드 | elbow(3번 조인트) | 하강 관절 이동량 (A소스/A핸드오프/B핸드오프/B목적지) |
|---|---|---|
| zeros / elbow_up / folded | **+1.6 ~ +1.8** | **0.49 / 0.43 / 0.43 / 0.49 rad** |
| elbow_dn | −1.2 ~ −1.6 | 0.37 / **3.45 / 3.18 / 5.27 rad** |

즉 **elbow 부호가 브랜치를 가름.** 음수 브랜치로 도착하면 하강이 3~5rad짜리 곡예가 되고, 이게
로그에 찍히던 3.5 / 3.6 / 5.5 / 6.6 / 6.8 rad 의 정체. 가드가 그 직선을 거부 → 플래너 폴백이
팔을 엉뚱한 데로 끌고 감 → 최악의 경우 로봇 B가 로봇 A 베이스에 박혀서, `robot_a_base` 충돌
오브젝트 안에 시작 상태가 들어가는 바람에 **관절공간 홈 복귀까지 포함해 모든 플래닝이 −2로 거부**,
팔이 바닥을 갈면서 시뮬레이터를 재시작할 때까지 못 빠져나옴.

고친 방법: `Arm.solve_ik()` + `move_to_seeded()` — 목표 포즈를 **고정 시드(`IK_SEED`, 전부 0)** 로
IK를 풀어서 **관절 목표**로 명령. LMA는 시드에 가장 가까운 해로 수렴하므로 브랜치가 고정됨.
`pre_grasp` / `pre_place` / `watch` 에 적용. 적용 후 모든 하강이 0.33~0.48 rad.

### 원인 2: 플래닝 씬에 바닥이 없었음

`dual_scene_setup.py` 가 발행하던 16개는 트레이 3개(각 5조각) + 상대 로봇 베이스뿐이고 **ground
plane이 없었음.** Isaac에는 실제 바닥이 있으니, MoveIt은 팔꿈치가 바닥을 뚫는 경로를 자유롭게
계획하고 → 실제로는 걸려서 추종 오차 → `CONTROL_FAILED(-4)` / `TIMED_OUT(-6)`. 눈으로는 "링크가
바닥에 닿은 채 끌린다"로 보임. 부수 효과로 Cartesian 솔버가 `avoid_collisions=True` 일 때
아래쪽을 자유공간으로 알고 이상한 브랜치로 우회하기도 했음.

→ `ground` 콜리전 오브젝트 추가 (4×4×0.02m, 윗면을 z=0 보다 1cm 아래에 둠 — z=0에 딱 맞추면
link0가 항상 충돌로 잡혀서 move_group이 아무것도 계획하지 못함). 로봇당 17개로 늘어남.

### 원인 3: 마찰 계수 (운반 중 미끄러짐)

그리퍼 패드 재질을 **μs 1.2 → 3.0, μd 1.0 → 2.6** 으로 올림. 재질 결합 모드가 양쪽 다 `max`라
쌍(pair) 마찰은 둘 중 큰 값이 되므로, 그리퍼 값이 블럭(1.5/1.2)보다 커야 의미가 있음. 적용 링크도
핑거팁 2개 → **접촉 가능한 6개**(finger_tip / finger / inner_knuckle 좌우)로 확대.

같이 넣은 검증: 재질을 링크 프림에만 바인딩하면 USD 상속으로만 전달되는데, URDF 임포터가 콜리전
프림에 자기 기본 재질을 직접 바인딩해두면 **더 가까운 바인딩이 이겨서 조용히 무효화됨**. 그래서
콜리전 프림마다 직접 바인딩하고 `ComputeBoundMaterial` 로 되읽어 `[CHECK] ... mu_s=3.0 ... on N
collider(s)` 로 출력함. (이 파일에서 이미 겪은 "Set()이 조용히 무시되는" 부류의 버그와 같은 형태)

### 그 외 같이 고친 것

- **속도**: `max_velocity_scaling_factor` 0.03 → 0.15, accel 0.10, Cartesian 0.15 → 0.35 rad/s.
  0.03은 미끄러짐을 속도로 억누르려던 값이었고, 마찰에 여유가 생겨서 되돌림. (이송 ~2분 → ~46초)
- **바닥 눌림**: 집을 때 `GRASP_FLOOR_MARGIN` 2 → 6mm, 놓을 때 `PLACE_CLEARANCE` 5 → 20mm.
  블럭/손끝이 트레이 바닥에 닿았는데도 목표 z가 더 아래라 강성 10000짜리 드라이브가 계속 밀어서
  팔이 떨리던 문제. 이제 살짝 위에서 놓아 떨어뜨림 (블럭 재질 restitution 0이라 안 튐).
- **하강에는 플래너 폴백 금지**: `move_linear(..., fallback=False)` 를 grasp 하강에 사용. 자유공간
  플래너에게 "트레이 위에서 트레이 안으로" 를 시키면 배회하는 경로가 나오고, 컨트롤러가 그 도중에
  abort하면 팔이 그 배회 지점에 남음 — 위의 "A 베이스에 박힘"이 정확히 이 경로였음.
- **막혔을 때 탈출**: `Arm.escape_to_home()` — move_group을 우회해 컨트롤러의
  FollowJointTrajectory 로 직접 홈까지 후퇴. move_group이 "시작 상태 충돌"로 아무것도 안 해줄 때
  쓸 수 있는 유일한 수단.

### 카메라 추가 (VLA 데이터 수집 준비)

씬에 **정면 전체 관측 카메라** 추가: `/scene_camera/rgb` (640×480, rgb8, ~57Hz).
`UsdGeom.Camera` + `rep.create.render_product` + `ROS2CameraHelper` 조합.
카메라 거리는 **HOME 자세(팔이 수직)의 그리퍼가 프레임에 남는지**로 정함 — 더 당기면 트레이가
커지지만 그리퍼가 잘려서 정책 학습에 못 씀. 현재 위치에서 두 팔 + 트레이 3개 + 블럭이 모두 보임.

### 그리퍼: 조임 세기와 wall-clock 램프

카메라를 붙인 뒤 `close` 가 연속 실패했고, 두 가지가 겹쳐 있었음.

**(a) 조임이 과했음.** 2단 클로즈의 조임 목표가 24mm인데 큐브는 42mm라 과도한 오버슈트였고,
멈추는 이유가 "큐브가 손가락을 막아서"뿐이라 남은 구동력이 큐브를 튕겨냄. 실패 로그에서 조임
단계가 6.7mm에서 stall(= 패드끼리 만남)했고 블럭은 10cm 밖에서 발견됨. 성공 사례는 전부
28~35mm에서 stall. → `GRASP_CLOSE_TARGET_M` 24 → 30mm, `GRASP_FLOOR_MARGIN` 6 → 4mm.
마찰이 3.0이 된 뒤로 클램핑 힘은 부족한 자원이 아님 (0.1kg 큐브 유지에 필요한 수직력 ~0.2N).

**(b) 그리퍼 램프가 wall-clock 타이밍임.** `namespaced_trajectory_bridge.py` 의 그리퍼 램프는
`for step in ...: time.sleep(dt)` 루프이고 settle도 `time.monotonic()` 기준인데, 위에서 측정한
"드라이브 응답 ~1.8초"는 **시뮬레이션 시간**임. 둘은 real-time factor가 1일 때만 같은 값.
측정된 RTF는 0.77이라 2.0초 close 램프가 실제로는 1.52 시뮬초 → 손가락이 다 닫히기 전에 액션이
끝나고 `reached_goal=True stalled=False` 로 "잡았다"고 보고함(실제론 빈손).
→ `gripper_close_duration:=3.5`, `gripper_open_duration:=2.0`, `gripper_settle_duration:=1.2`.
카메라 발행도 `frameSkipCount=3` 으로 ~12Hz로 낮춤(다만 이것만으로 RTF는 안 돌아옴 — 비용은
발행이 아니라 렌더링 자체. 카메라 이전 RTF를 재둔 적이 없어 "카메라가 RTF를 낮췄다"는 인과는
미증명이고, 램프를 늘린 것이 실제 수정임).

**카메라 켠 상태 재검증:** `--cycles 2` 6/6 성공, 놓은 위치 오차 0 / 4 / 2 / 2 / 2 / 3 mm.
남은 산발적 실패는 A가 핸드오프에서 `close` 2회 실패(3번째 성공) — 재시도 루프가 흡수함.

### 현재 상태 / 다음

- 안정성: 2사이클 6/6 성공. 재시도 루프가 남은 산발적 실패를 흡수함.
- 아직 안 한 것: 손목 카메라, 블럭 위치/색 랜덤화, 이미지까지 저장하는 에피소드 로깅.
  → 이게 VLA fine-tuning 데이터셋의 나머지 전부. 블럭 위치 랜덤화는 선택이 아니라 필수 (고정
  위치만 모으면 정책이 좌표를 외움). 씬에 `/binpicking/place_object` (geometry_msgs/Point) 와
  `/binpicking/reset_object` (std_msgs/Empty) 가 이미 있어서 위치 랜덤화는 이걸로 붙이면 됨.

## VLA 데이터셋 수집 파이프라인 (2026-08-11)

릴레이가 안정화된 뒤 곧바로 붙인 수집 경로. 구성 요소 넷:

### 1. 카메라 3대

| 토픽 | 해상도 | 위치 |
|---|---|---|
| `/scene_camera/rgb` | 640×480 | 정면 전체 관측 (두 팔 + 트레이 3개 + 블럭) |
| `/robot_a/wrist_camera/rgb` | 320×240 | A 손목 |
| `/robot_b/wrist_camera/rgb` | 320×240 | B 손목 |

손목 카메라는 URDF에 이미 있는 RealSense 마운트
(`camera_joint → camera_link → … → camera_color_optical_frame`)에 그대로 붙였음 — 실제 로봇에
달릴 위치와 같음. ROS 광학 규약(+Z 전방, +Y 하)과 USD 카메라 규약(−Z 전방, +Y 상)이 달라서
X축 180° 회전을 넣음. 손목 뷰는 파지 순간 블럭이 패드 사이에 잡힌 게 또렷이 보임.

### 2. 도메인 랜덤화

`/binpicking/randomize_object` (std_msgs/Empty) → 씬이 블럭의 **위치·yaw·색**을 새로 뽑음.
- 위치: 소스 빈 내부 균일 분포, 단 **로봇 A 기준 반경 0.62m 이내로 rejection sampling**.
  빈 전체(0.34×0.29m)의 먼 구석은 0.70m라, 0.67m에서 반복 실패가 측정된 영역이라 그대로 두면
  집을 수 없는 에피소드를 양산함. 사각형으로 자르지 않고 rejection으로 뽑아서 도달 가능한
  영역은 균일하게 덮음.
- 색: 8색 팔레트(노랑/빨강/파랑/초록/주황/보라/청록/분홍).
  **주의**: `DynamicCuboid(color=...)`는 displayColor primvar가 아니라 PreviewSurface
  **비주얼 머티리얼**을 만들어 바인딩함. displayColor를 써봐야 바인딩된 머티리얼이 이겨서
  에러 없이 색이 그대로임(마찰 바인딩 때와 같은 조용한 무효화). 머티리얼의 `set_color()`를
  쓰고 `get_color()`로 되읽어 확인함.
- 물리는 안 건드림(질량/마찰/충돌 동일) — 색만 변해야 시연의 물리 조건이 유지됨.

### 3. 에피소드 경계

`relay_pick_place.py` 가 `/relay/episode` (std_msgs/String, JSON)로 발행:
`{"event": "start"|"end", "index", "task", "arm", "attempt", "success"}`.
**경계를 컨트롤러가 아는 게 핵심**: `/joint_states` 만 보는 수동 로거는 성공한 이송과
실패 후 재시도를 구분할 수 없고, 그 둘을 섞은 데이터셋은 "블럭 떨어뜨리기"를 가르침.
1 에피소드 = pick-and-place 시도 1회. 성공/실패는 `success/` 와 `failure/` 트리로 분리 저장.

task 문자열(VLA 조건부 입력)은 내부 id가 아니라 **눈에 보이는 것**으로 씀:
`"pick up the block and place it on the blue tray in the middle"` 등.

### 4. 로거 `dual_dual_robot/dual_episode_logger.py`

`ros_bridge/episode_logger.py` 는 단일 로봇 · 네임스페이스 없음 · 카메라 1대 · 시간 기반
경계라 재사용이 안 돼서 새로 씀. 프레임 페이싱은 손목 카메라 기준(관절 상태는 물리 레이트로
와서 그대로 쓰면 같은 이미지에 붙은 중복 프레임만 쌓임).

저장 형식은 **프레임별 JPEG**(q=92) + `state.npz`. raw uint8로 쌓으면 카메라 2대 합쳐
프레임당 ~1.15MB, 12Hz × 52초면 **에피소드 하나가 ~600MB** — 천 에피소드가 어디에도 안 들어감.
JPEG는 이 화면 내용에선 사실상 무손실이고 ~40배 작음.

```
episode_0001/
  state.npz        state (N,7)  action (N,7)  timestamps (N,)   # 관절6 + 그리퍼(0=닫힘,1=열림)
  images_scene/000000.jpg ...
  images_wrist/000000.jpg ...
  task.txt  meta.json
```

### 검증된 실측치

에피소드 1개 = **~470 프레임 / ~52초 / 13.5MB** (씬 9.6MB + 손목 3.8MB). 프레임 간격 0.108초.

### 저장 위치

메인 디스크 공간이 부족해서 **HDD(`/dev/sda1` → `/mnt/hdd`, 166GB 여유)** 에 저장:
`/mnt/hdd/relay_datasets/`, `lerobot_ws/datasets` 는 거기로 가는 심볼릭 링크.
13.5MB/에피소드면 여유 공간으로 ~12,000 에피소드 분량.

### 실행 방법

```bash
# 1) 씬            ~/isaacsim/python.sh dual_binpicking_scene.py
# 2) MoveIt 스택   ros2 launch dual_binpicking.launch.py
# 3) 로거          python3 dual_episode_logger.py --out-dir /mnt/hdd/relay_datasets/relay_v1
# 4) 수집          python3 relay_pick_place.py --cycles 3 --randomize
```
4번은 배치로 반복하는 게 좋음 — 한 이송이 재시도 4회를 소진하면 relay 프로세스 전체가 끝나므로,
무인 수집에서는 짧은 배치를 반복해서 다음 배치가 새로 시작하게 함
(스크래치패드의 `collect.sh`).

### 그리퍼 타이밍을 sim 시계로 옮김 (구조적 수정)

`namespaced_trajectory_bridge.py` 의 그리퍼 램프/settle만 wall-clock(`time.sleep` 루프,
`time.monotonic`)이었고, 궤적 실행부는 이미 `self.get_clock()`(=sim 시계)를 쓰고 있었음.
그런데 튜닝 근거인 "드라이브 응답 ~1.8초"는 **시뮬레이션 초**라, 렌더링 부하가 바뀔 때마다
램프 길이가 조용히 변함:

| 씬 | 측정 RTF | wall 2.0초 램프의 실제 시뮬 시간 |
|---|---|---|
| 카메라 없음 | (미측정) | — |
| 씬 카메라 1대 | 0.77 | 1.54초 |
| 씬 + 손목 2대 | 0.58 | 1.16초 |

→ 램프와 settle을 ROS 시계 기준으로 다시 씀(궤적 실행부와 동일한 방식). sim 시계가 멈추는
경우를 대비해 wall-clock backstop을 별도로 둠. 이제 launch의 세 duration은 **시뮬레이션 초**이고
RTF와 무관 (1.5 / 2.5 / 1.0).

### 파지 신뢰도 — 측정치와 기각된 가설들

파지 첫 시도 성공률(재시도 전):

| 조건 | 성공 : 실패 |
|---|---|
| 씬 카메라만, 랜덤화 없음 (grip_v3) | 6 : 2 |
| 카메라 3대, 랜덤화 없음 | 10 : 6 (3사이클 완주) |
| 카메라 3대, 랜덤화 | 3 : 6 |

**기각된 가설(전부 동일 조건 비교 실험으로 확인):**
- *손목 카메라가 RTF를 떨어뜨려 파지가 깨진다* → `WRIST_CAMERAS=0` 으로 끄고 동일 실행:
  1 성공 / 4 실패로 **똑같이 실패**. 카메라 아님.
- *조임 목표가 문제* → 24mm(1:5), 30mm(3:6) 둘 다 시험. 어느 쪽도 결정적이지 않음.
- *랜덤화가 원인* → 랜덤화를 끄면 10:6으로 나아지지만 여전히 실패함. 단독 원인 아님.

즉 **파지는 원래부터 이 시스템에서 가장 불안정한 부분**이고(첫 시도 60~75%), 릴레이가 완주하는
이유는 pick+place 전체를 단위로 하는 재시도 루프임. 위에서 "6/6 성공"으로 기록된 실행들도
내부적으로는 파지 실패를 재시도로 흡수하고 있었음. 조기 종료로 보였던 케이스는 B가 재시도 4회를
소진한 경우.

데이터 수집 관점에서는 이 상태로 진행 가능 — 실패 에피소드는 `failure/` 트리로 분리 저장되므로
성공 시연과 섞이지 않고, 오히려 실패 사례로 따로 쓸 수 있음. 파지 신뢰도 자체를 더 올리려면
다음이 남아 있음(아직 안 해봄): 접근 전 블럭 자세(기울어짐) 검사, 파지 실패 시 블럭을 밀어내지
않는 후퇴 경로, 패드 형상/컴플라이언스 조정.

### 파지 신뢰도를 지배하는 것은 도달거리였음 (2026-08-11, 추가)

위 표의 "파지가 원래 불안정하다"는 결론은 반쪽이었음. 랜덤화된 스폰 위치별로 보면 패턴이 명확함:

| 블럭까지 도달거리 | 결과 |
|---|---|
| 0.377 / 0.440 / 0.498 m | 파지 성공 |
| 0.559 m | **4회 연속 실패** |
| 0.596 m (구 핸드오프, 로봇 B) | 오늘 내내 반복 실패 |

툴은 어느 경우에도 결정적으로 ~1~3mm / 0.00° 로 도착함 — 정확도 문제가 아니라 **팔이 뻗을수록
말단 강성이 떨어지는** 문제. RB5-850e의 공칭 850mm 안에 있다는 것과, 42mm 큐브를 물 수 있을 만큼
툴이 안정적인 영역에 있다는 것은 다름.

**고친 것 두 가지:**

1. 핸드오프 위치를 "두 빈의 중점"에서 **"두 베이스에서 등거리 0.510m인 점"** 으로 바꿈
   (`layout.py`의 `_equidistant_point`). (0.210, 0.575) → **(0.094, 0.501)**.
   A 0.612 / B 0.596 → **A 0.510 / B 0.510**, 즉 A가 자기 빈을 집을 때의 검증된 거리와 동일.
2. 랜덤 스폰 반경 상한 0.62 → **0.53m**. 집을 수 없는 위치의 에피소드는 데이터가 아니라 노이즈임.

**결과:** B가 새 핸드오프에서 파지 성공(툴 1.6mm, 두 단계 모두 `stalled=True` = 큐브가 손가락을
막음). 즉 오늘 내내 B를 괴롭히던 파지 실패는 도달거리 문제였음.

**남은 문제는 다시 하나:** 파지 직후 들어올릴 때의 미끄러짐 (`aborted at 'carrying?'`).
세션 처음에 잡으려던 바로 그 문제이고, 패드 마찰 1.2 → 3.0 으로도 완전히 없어지지 않음.
파지 실패가 사라지면서 이것만 남았다는 게 오늘의 실제 진전.

## 운반 중 낙하 해결 + 병렬 수집 (2026-08-11)

### 운반 중 낙하: 잡은 구간만 느리게

세션 내내 남아 있던 `aborted at 'carrying?'`(파지 확인 직후 18cm 들어올리다 놓침)의 원인은
**내가 올린 속도**였음. "너무 느리다"는 지적에 `max_velocity_scaling_factor` 를 0.03 → 0.15,
Cartesian 을 0.15 → 0.35 rad/s 로 올렸는데, 그 0.03은 원래 미끄러짐을 속도로 억누르던 값이었음.

마찰은 범인이 아님 — μ=3.0에서 0.1kg 큐브를 유지하는 데 필요한 패드 수직력은 ~0.16N에 불과함.
문제는 스윙의 **가속도**이므로, 속도를 "아무것도 안 든 구간"에 몰아주고 "든 구간"에서 돌려줌:

```
빈손:  velocity 0.15  accel 0.10  cartesian 0.35 rad/s
운반중: velocity 0.04  accel 0.03  cartesian 0.12 rad/s
```

`RobotArm.carrying` 플래그로 전환(파지 확인 시 set, 놓거나 실패하면 clear).
**결과: 검증 실행과 병렬 3인스턴스 전체에서 운반 중 낙하 0회.**

### 병렬 수집: ROS_DOMAIN_ID로 격리

`collect_instance.sh <id> <batches> <cycles>` 하나가 Isaac 씬 + MoveIt 스택 + 로거 + 배치 루프를
통째로 띄움. 인스턴스 간 격리는 **토픽 이름 변경이 아니라 `ROS_DOMAIN_ID`** — 씬/런치/컨트롤러
어느 것도 자기가 여럿 중 하나라는 걸 알 필요가 없음.

**함정**: Isaac의 `ROS2Context` 노드는 `useDomainIDEnvVar` 가 기본 False, `domain_id` 기본 0이라
`ROS_DOMAIN_ID` 를 **무시함**. 그대로 두면 모든 인스턴스가 도메인 0에 발행해서 서로의 로봇을
조종함. 씬의 모든 그래프(로봇 A/B, 클럭, 카메라)에 `useDomainIDEnvVar=True` 를 설정함.
확인: 도메인 0에 토픽 0개, 도메인 1·2·3에 각각 독립 토픽.

병렬 인스턴스는 `ISAAC_HEADLESS=1` 로 GUI 없이 실행(카메라 render product는 뷰포트와 별개라
헤드리스에서도 정상 동작). `gui` 인자로 하나만 화면에 띄울 수 있음.

**이게 가능해진 전제**는 앞 절의 그리퍼 sim-시계 수정임. wall-clock 램프였다면 인스턴스가 늘어
RTF가 떨어질 때마다 그리퍼가 조용히 깨졌을 것.

### 측정된 병렬 성능 (RTX 4090)

| | 1 인스턴스 | 3 인스턴스 |
|---|---|---|
| RTF | 0.58 | 0.36 ~ 0.38 (각각) |
| GPU 메모리 | 3.1 GB | 10.8 GB |
| GPU 사용률 | 42% | 76% |

합산 처리량은 약 2배. GPU 사용률로 보면 3개가 현실적인 상한.

```bash
# 인스턴스 하나 = 씬 + MoveIt + 로거 + 60배치
setsid nohup bash collect_instance.sh 1 60 3 &   # 도메인 1 -> relay_inst1/
setsid nohup bash collect_instance.sh 2 60 3 &   # 도메인 2 -> relay_inst2/
setsid nohup bash collect_instance.sh 3 60 3 &   # 도메인 3 -> relay_inst3/
```

### 에피소드 번호 충돌 버그 (데이터가 조용히 섞이고 있었음)

`meta.json` 은 263프레임인데 `images_scene/` 에는 766장이 있었음. 에피소드 번호를 컨트롤러가
매기는데 수집 드라이버가 relay를 **배치마다 새 프로세스로** 띄우므로 번호가 매번 1부터 다시
시작했고, 같은 디렉터리에 덮어쓰면서 새 에피소드가 도달하지 못한 인덱스의 JPEG가 이전
에피소드 것으로 남았음. 에러 없음, 로그 없음.
→ 번호 매기기를 **로거 소유**로 옮기고(시작 시 디스크의 최대 번호를 스캔해 이어감), 쓰기 전에
디렉터리를 비움. 컨트롤러 번호는 `controller_index` 로 meta에 보존(로그 대조용).
검증: 모든 에피소드에서 JPEG 수 == `meta.json` 프레임 수.

### 남은 실패: 파지 (~40%)

운반 낙하가 사라진 뒤 남은 실패는 전부 `aborted at 'close'` 한 종류. 그리퍼가 허공에서 닫히고
(`stalled=False`) 블럭은 안 딸려옴. 병렬화가 악화시킨 건 아님 — 단일 인스턴스와 같은 비율.
아직 확인 안 한 유력 후보: **블럭의 기울기(roll/pitch)**. 컨트롤러는 yaw만 읽고 큐브 대칭으로
접는데, 블럭이 모서리로 기울어 쉬고 있으면 그 yaw가 무의미해지고 패드가 모서리를 침.
실패한 파지가 블럭을 1m 밖으로 날려버리는 것도 같이 봐야 함.

데이터셋 관점에서는 이 실패들이 `failure/` 로 정확히 라벨되어 저장되므로 손실이 아님.

## 무인 수집이 밤새 헛돈 건 (2026-08-12) — 헬스체크 설계 오류

8시간 무인 수집 결과: 성공 140개(목표 500의 28%), 실패 3,253개 중 **2,806개가 10프레임 미만인
빈 껍데기**. 배치 진행 로그가 원인을 그대로 보여줌 (inst3):

```
batch   1  22:24
batch  50  01:27   <- 여기까지 정상 (배치당 ~3.6분)
batch 100  01:34   <- 50배치를 8분에, 즉 배치당 10초
batch 400  02:24   collection finished
```

**자동 복구는 한 번도 발동하지 않았음(`rebuilds: 0`).** 앞 절에서 넣은 `scene_alive()` 는
*프로세스가 존재하는지*만 봤는데, 그건 그 전에 관측된 고장(세그폴트로 프로세스가 사라짐) 하나만
막는 검사였음. 실제로 밤새 일어난 고장은 **프로세스는 살아 있는데 시뮬레이션이 멈춘** 형태라
검사를 그대로 통과했고, 루프는 죽은 시뮬레이터를 상대로 400배치를 태우며 매번 빈 에피소드를
하나씩 남겼음. inst1은 또 다른 방식으로 죽었음 — relay 하나가 DDS goal response를 못 받고
**24시간 동안 한 배치 안에 걸려 있어서** 어떤 검사에도 도달하지 못했음.

### 고친 것: 산출물 기준 검사 세 가지

1. **`sim_healthy()` — /clock이 실제로 진행하는지 확인.** 프로세스 존재는 건강의 증거가 아님.
   시계가 도는지는 프로세스가 죽은 경우(시계 없음)와 얼어붙은 경우(시계 정지)를 모두 잡음.
2. **배치 타임아웃** (`timeout --signal=KILL 1800`). 무한 대기하는 relay를 끊음.
3. **짧은 배치 연속 감지.** 정상 3사이클 배치는 분 단위이므로, 60초 미만이 3회 연속이면
   업스트림이 고장난 것으로 보고 인스턴스를 통째로 재구성.

검증: 시뮬레이터 없는 도메인에서 exit=1, 살아있는 도메인에서 exit=0 (31샘플, 시계 진행 확인).

### 정리

50프레임 미만 실패 에피소드 2,816개 삭제 (5초도 안 되는 기록은 시연이 아님).
남은 것: **성공 140 + 유의미한 실패 437**. 9.5GB -> 9.2GB.

## 다음 단계 후보

0. 이 릴레이 씬에 카메라를 붙여서 (VLA 학습에는 이미지가 필수) 2-로봇 릴레이 에피소드 녹화 →
   "A가 집어서 넘기고 B가 받아서 옮긴다" 같은 언어 지시가 붙는 데이터셋. 지금은 카메라 없는
   휴리스틱 단계라 렌더링을 끄고 돌렸음.
1. 위 임시 산출물(데이터셋/체크포인트) 중 남길 것을 `lerobot_ws/` 안의 영구 위치로 옮기기.
2. 더 많은 에피소드 녹화 (trial 성공/실패 기준으로 에피소드 경계 분리, grasp 실패 케이스는 별도
   task 문자열로 남길지 등 결정 필요) → 진짜 규모 있는 fine-tuning.
3. `smolvla_base` 로 제대로 된 fine-tuning (수천~2만 스텝) 및 `lerobot-rollout` 으로 평가.
