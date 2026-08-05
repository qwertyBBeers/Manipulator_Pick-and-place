# LeRobot Workspace

`Manipulator/` 본체(ROS2/MoveIt/Isaac Sim 파이프라인)와는 완전히 분리된 별도 워크스페이스.
LeRobot + SmolVLA(언어모델 결합 VLA) 실험을 위한 공간이며, 기존 `rb5_binpicking` / `rb5_isaac` /
`rbpodo_ros2` 는 건드리지 않는다.

```
lerobot_ws/
├── lerobot/               huggingface/lerobot 클론 (v0.6.2, 2026-08-05 기준 main)
├── lerobot_robot_rb5/     pip 패키지. RB5-850E용 커스텀 LeRobot Robot 플러그인 (--robot.type=rb5_850e)
├── ros_bridge/            rb5_lerobot_bridge.py — ROS2(Humble, py3.10) 쪽에서 도는 ZMQ 브릿지, 위 플러그인의 상대편
└── tools/                 convert_isaac_episodes_to_lerobot.py — 데이터셋 변환 스크립트 스캐폴드
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

## 다음 단계 (Isaac Sim 재실행 필요 — 대기 중)

**2026-08-05 기준: RL 학습(rb5_isaaclab/scripts/train.py)이 Isaac Sim에서 GPU를 쓰고 있어서, 사용자
요청으로 Isaac Sim을 새로 띄우는 작업은 보류 중. 학습이 끝나면(또는 사용자가 신호를 주면) 아래
순서로 진행.**

1. Isaac Sim에서 `binpicking_scene.py` 띄운 채로 `rb5_lerobot_bridge.py` 실행 → `lerobot_robot_rb5.RB5850E`
   로 `connect()` → `get_observation()`(관절/카메라) → `send_action()`(관절 하나 살짝 이동) 왕복 테스트.
2. 기존 `moveit_pick_place.py` 휴리스틱 데모를 몇 번 돌리면서 `episode_logger.py`로 실제 에피소드 수집
   → `convert_isaac_episodes_to_lerobot.py`로 실제 LeRobotDataset 생성 (스모크 테스트는 이미 통과).
3. `smolvla_base` 로 소규모 fine-tuning 시험 (`lerobot-train --policy.path=lerobot/smolvla_base ...`).
