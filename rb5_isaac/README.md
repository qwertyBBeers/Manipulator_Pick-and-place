# RB5-850E Isaac Sim + MoveIt2 개발 환경

> **참고 (정리, `../README2.md` 참조):** 아래 문서에 언급된
> `scripts/01_convert_urdf_to_usd.py`, `scripts/02_isaac_rb5_scene.py`,
> `launch/moveit_isaac.launch.py`는 이후 `rb5_binpicking/scripts/binpicking_scene.py`
> + `launch/binpicking.launch.py`로 대체되어 `../deprecated/`로 옮겨졌습니다.
> 실제 실행 중인 파이프라인은 `rb5_binpicking` 패키지를 참고하세요. 새로 추가된
> IsaacLab 기반 RL 학습 패키지는 `../rb5_isaaclab/`입니다.

Rainbow Robotics RB5-850E 매니퓰레이터를 Isaac Sim 4.1 + MoveIt2 Humble로 모션 플래닝 테스트하는 개발 환경입니다.

---

## 시스템 환경

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 22.04 LTS (Jammy) |
| ROS2 | Humble |
| Isaac Sim | 4.1.0 (`~/isaacsim/`) |
| IsaacLab | `~/asl_ws/vla_project/IsaacLab/` |
| GPU | NVIDIA GeForce RTX 4090 |
| Python (Isaac Sim) | 3.10 (`~/isaacsim/kit/python/bin/python3`) |

---

## 패키지 구조

```
~/asl_ws/
├── Manipulator/
│   ├── rbpodo_ros2/              # Rainbow Robotics 공식 ROS2 패키지
│   │   ├── rbpodo_description/   # URDF, mesh (rb5_850e.urdf 포함)
│   │   ├── rbpodo_hardware/      # ros2_control hardware interface
│   │   ├── rbpodo_moveit_config/ # MoveIt2 config (SRDF, kinematics, controllers)
│   │   ├── rbpodo_bringup/       # controllers.yaml
│   │   └── rbpodo_msgs/          # custom messages
│   └── rb5_isaac/                # ← 이 패키지 (개발 환경)
│       ├── scripts/
│       │   ├── 01_convert_urdf_to_usd.py   # URDF → USD 변환 (최초 1회)
│       │   ├── 02_isaac_rb5_scene.py        # Isaac Sim 씬 + ROS2 bridge
│       │   └── 03_trajectory_bridge.py     # MoveIt2 ↔ Isaac Sim 브릿지 노드
│       ├── launch/
│       │   └── moveit_isaac.launch.py       # MoveIt2 통합 launch
│       └── rb5_isaac/
│           └── trajectory_bridge.py         # ROS2 실행 진입점
└── install/                      # colcon 빌드 결과

~/rb5_isaac_assets/
├── rb5_850e.usd                  # 변환된 로봇 USD (defaultPrim: /rb5_850e)
└── instanceable_meshes.usd       # 메시 데이터 (~21MB)
```

### 2. ROS2 의존 패키지
```bash
sudo apt install -y \
  ros-humble-ament-cmake \
  ros-humble-joint-state-publisher \
  ros-humble-moveit \
  ros-humble-pluginlib \
  ros-humble-robot-state-publisher \
  ros-humble-ros2-controllers \
  ros-humble-ros2-control \
  ros-humble-rviz2 \
  ros-humble-urdf-launch \
  ros-humble-xacro
```

### 3. PyTorch (Isaac Sim Python 환경)
```bash
# Isaac Sim의 Python에 torch 설치 (omni.isaac.core 의존)
/home/hh/isaacsim/kit/python/bin/python3 -m ensurepip
/home/hh/isaacsim/kit/python/bin/python3 -m pip install torch \
  --index-url https://download.pytorch.org/whl/cu121
```

### 4. ROS2 워크스페이스 빌드
```bash
cd ~/asl_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select \
  rbpodo_msgs rbpodo_description rbpodo_bringup \
  rbpodo_hardware rbpodo_moveit_config rb5_isaac
```

### 5. ~/.bashrc 설정
```bash
source /opt/ros/humble/setup.bash
source ~/asl_ws/install/setup.bash
```

---

## 실행 방법

### Step 1: URDF → USD 변환 (최초 1회만)

```bash
source /opt/ros/humble/setup.bash
source ~/asl_ws/install/setup.bash
~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_isaac/scripts/01_convert_urdf_to_usd.py
```

**출력 확인:**
```
[OK] defaultPrim set to /rb5_850e and saved
```

결과물: `~/rb5_isaac_assets/rb5_850e.usd`

---

### Step 2: Isaac Sim 실행 (Terminal 1)

```bash
source /opt/ros/humble/setup.bash
source ~/asl_ws/install/setup.bash
export DISPLAY=:1
~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_isaac/scripts/02_isaac_rb5_scene.py
```

**정상 동작 확인:**
```
[OK] ROS2 OmniGraph built.
[OK] Isaac Sim running.
     Publishing:  /joint_states  /tf  /clock
     Subscribing: /isaac_joint_commands
```

ROS2 topic 확인:
```bash
ros2 topic echo /joint_states --once
# name: [base, shoulder, elbow, wrist1, wrist2, wrist3]
# position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

---

### Step 3: MoveIt2 실행 (Terminal 2)

```bash
source /opt/ros/humble/setup.bash
source ~/asl_ws/install/setup.bash
ros2 launch rb5_isaac moveit_isaac.launch.py
```

**정상 동작 확인:**
```
[move_group-3] You can start planning now!
[trajectory_bridge-5] TrajectoryBridge ready.
  Action: /joint_trajectory_controller/follow_joint_trajectory
```

---

### Step 4: RViz2에서 모션 플래닝

1. RViz2 창의 `Motion Planning` 패널 → `Planning` 탭
2. `Goal State` 드롭다운 → 원하는 포즈 선택 (또는 인터랙티브 마커로 직접 지정)
3. `Plan` 버튼 → 경로 확인
4. `Execute` 버튼 → Isaac Sim에서 로봇 움직임 확인

---

## 알려진 이슈 및 해결 방법

### 1. `Prim is not an articulation` 경고
- **원인**: OmniGraph 첫 번째 tick에서 physics 초기화가 완전히 끝나지 않은 경우
- **해결**: 무시해도 됨 (joint_states 데이터가 정상 publish되면 OK)
- **확인**: `ros2 topic echo /joint_states --once`

### 2. Isaac Sim 창이 뜨자마자 꺼지는 경우
- **원인**: 스크립트 Python import 에러
- **확인**: `/tmp/isaac_scene.log` 에서 `Traceback` 검색

### 3. RViz2에서 로봇 모델이 안 보이는 경우
- **원인**: `robot_state_publisher`가 `/joint_states`를 못 받는 경우
- **확인**: `ros2 topic hz /joint_states` → Isaac Sim이 먼저 실행됐는지 확인

### 4. `colcon build` 시 `{{ name }}` 패키지 오류
- **원인**: `~/asl_ws/vla_project/IsaacLab/tools/template/` 내 템플릿 파일
- **해결**: `--packages-select`로 rbpodo 패키지만 지정해서 빌드

---

## 핵심 파일 설명

### `02_isaac_rb5_scene.py`
Isaac Sim에서 URDF를 직접 import해서 로봇을 로드하고 ROS2 bridge를 구성합니다.
- `fix_base=True`: 로봇 베이스를 ground에 고정
- `make_instanceable=False`: 씬에 메시 인라인 포함 (USD 참조 문제 회피)
- `initialize_physics()` + `play()` 후 OmniGraph 빌드: articulation 인식 타이밍 문제 해결

### `trajectory_bridge.py`
MoveIt2의 `FollowJointTrajectory` action을 Isaac Sim의 `JointState` 토픽으로 변환합니다.
- Action: `/joint_trajectory_controller/follow_joint_trajectory`
- 출력: `/isaac_joint_commands` (JointState)

### `moveit_isaac.launch.py`
MoveIt2 전체 스택 + trajectory_bridge를 함께 실행합니다.
- `move_group` (OMPL/CHOMP/Pilz 플래너)
- `robot_state_publisher`
- `rviz2`
- `trajectory_bridge`
- `ros2_control_node` 없음 (Isaac Sim이 hardware 역할)

---

## 다음 작업 방향 (TODO)

- [ ] 카메라/depth sensor 추가 (Isaac Sim ROS2 bridge)
- [ ] 물체 pick & place 태스크 환경 구성
- [ ] IsaacLab `ArticulationCfg` 등록 (`rb5.py` in isaaclab_assets)
- [ ] RL 학습 환경 (`ManagerBasedRLEnv`) 구성
- [ ] Sim-to-Real: 실제 로봇 IP 연결 테스트 (`robot_ip:=10.0.2.7`)
