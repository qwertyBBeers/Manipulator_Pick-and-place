# Manipulator — 진행 상황 정리 (최종 갱신 2026-07-22, §7 대규모 신뢰성 개선 반영)

`README.md`가 3-phase 로드맵(휴리스틱 → GR00T → DUNE) 자체를 담고 있다면, 이 문서는
**"지금 뭐가 실제로 동작하고, 어떤 알고리즘으로 동작하며, 어디가 아직 하드코딩/미해결
상태인지"** 를 코드 기준으로 정리한 현황판입니다.

---

## 1. 패키지 구성

```
Manipulator/                       (colcon workspace)
├── rbpodo_ros2/                   Rainbow Robotics 벤더 스택
│   ├── rbpodo_description/        RB5-850E URDF/mesh
│   ├── rbpodo_moveit_config/      MoveIt2 SRDF, kinematics, RViz 설정
│   ├── rbpodo_bringup/            ros2_control 컨트롤러 설정 (실물 로봇용)
│   ├── rbpodo_hardware/           실물 로봇 하드웨어 인터페이스
│   └── rbpodo_msgs/                RB 전용 msg/action/srv
├── rb5_isaac/                     Isaac Sim 연동 공통 브릿지
│   ├── urdf/rb5_with_tools.urdf   RB5 + Robotiq 2F-85 + RealSense D435i
│   └── rb5_isaac/trajectory_bridge.py   MoveIt 궤적 → Isaac Sim 조인트 커맨드
└── rb5_binpicking/                실제 pick-and-place 구현체
    ├── scripts/binpicking_scene.py     Isaac Sim 씬 (로봇/그리퍼/카메라/bin/오브젝트)
    ├── scripts/moveit_pick_place.py    Phase 1 pick-and-place expert (§7: stage 기반, 검증 포함)
    ├── scripts/scene_setup.py          bin collision object → RViz 표시
    ├── scripts/depth_to_pointcloud.py  depth 이미지 → pointcloud 변환
    ├── scripts/action_adapter.py       Phase 2용 미배선 스캐폴딩 (4D action ↔ EEF pose)
    ├── rb5_binpicking/bin_geometry.py  bin_geometry.yaml 로더 (§7: YAML 기반으로 교체)
    ├── config/bin_geometry.yaml        bin 치수/위치 단일 출처 (신규, §7)
    ├── config/rb5_action_space.yaml    Phase 2용 미배선 action-space 설정
    ├── lab_envs/rb5_binpicking_env.py  Phase 2용 Isaac Lab 환경 스텁
    └── launch/binpicking.launch.py     move_group + rviz2 + trajectory_bridge 기동
```

빌드는 시스템 ROS 2 Humble(`/opt/ros/humble`) 기준. Isaac Sim은 `~/isaacsim/python.sh`로
별도 파이썬 환경에서 실행.

---

## 2. 현재 파이프라인 (Phase 1) — 동작 알고리즘

### 2.1 실행 순서

```bash
# 터미널 1 — Isaac Sim 씬
source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_binpicking/scripts/binpicking_scene.py

# 터미널 2 — MoveIt2 + RViz + trajectory bridge + scene_setup(bin 표시)
source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
ros2 launch rb5_binpicking binpicking.launch.py

# 터미널 3 — pick-and-place expert 실행
source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
ros2 run rb5_binpicking moveit_pick_place.py
```

### 2.2 제어 방식 — End-Effector 기반

로봇은 **EE(TCP) pose를 목표로 주면 MoveIt이 IK/경로계획을 하고, 그 결과 조인트
궤적을 trajectory_bridge가 실행**하는 구조입니다. 조인트 각도를 직접 계산해서 주는
방식이 아닙니다.

```
moveit_pick_place.py            trajectory_bridge.py           Isaac Sim
  EE Pose(x,y,z,quat)              (MoveIt→Isaac 브릿지)
      │  /move_action                    │
      ▼  (MoveGroup action)              │
  move_group (IK + 경로계획)              │
      │  FollowJointTrajectory action    │
      ▼ ──────────────────────────────►  │
                              Cubic Hermite spline 보간
                              (100Hz 제어 루프)
                                          │  /isaac_joint_commands
                                          ▼ ─────────────────────► 관절 구동
```

- Planner: **Pilz `LIN`(직선 보간) 우선 시도 → 실패 시 OMPL로 폴백** (`_move_to_pose_once` /
  `move_to_pose`, `rb5_binpicking/scripts/moveit_pick_place.py`).
- `trajectory_bridge.py`는 waypoint에 velocity가 있으면 **cubic Hermite spline**
  (C1 연속, ros2_controllers와 동일 방식), 없으면 선형 보간으로 100Hz 커맨드를 생성.
- 그리퍼는 별도 `GripperCommand` 액션 → Robotiq 2F-85의 6개 조인트에 mimic 비율
  적용해서 동시 구동.

### 2.3 Pick-and-Place 상태 흐름 (`moveit_pick_place.py`, §7에서 stage 기반으로 재작성)

```
run()
 ├─ /move_action, gripper 액션 서버 대기
 ├─ publish_bin_collision_objects()  — source/dest bin을 planning scene에 등록 (YAML 기반)
 ├─ /binpicking/object_pose 3초 대기
 │
 ├─(못 받음)→ run_saved_pose_sequence()  — SAVED_POSES 그대로 (완전 고정 fallback)
 │
 └─(받음)  → run_dynamic_object_sequence(object_pose)
                initial pose를 TF로 PLANNING_FRAME 변환
                → generate_grasp_candidates() (최대 4개, §3 한계 참고)
                → candidate별로 _attempt_grasp():
                     pre_grasp(FREE_SPACE) → 재관측(refine) → grasp(CONTACT_SENSITIVE)
                     → gripper close → 그리퍼 stall로 grasp 검증 → test-lift로 재검증
                     → 실패 시 다음 candidate, 성공 시 planning scene에 attach
                → lift → pre_place(bin 기하 기반) → place(bin 기하 기반, jitter 재시도)
                → release → 배치 검증(재관측 object pose가 dest bin 안에서 정지했는지)
                → detach → retreat → watch
```

물건 **위치(XYZ)만 카메라 detection(`/binpicking/object_pose`, 지금은 Isaac Sim
ground-truth) 기반으로 동적**입니다. §7 이후로 grasp 목표는 pre-grasp 도달 후
**재관측**되고, place 목표는 **dest bin 기하에서 계산**되지만, 파지
orientation은 여전히 고정값입니다(perception이 object 자세를 안 주므로). 자세한
내용은 §3, §7 참고.

### 2.4 RViz에 bin(박스) 표시 (오늘 추가)

기존에는 `moveit_pick_place.py`를 실행해야만 bin collision object가
`/collision_object`에 publish돼서 RViz에 보였습니다. `binpicking.launch.py`만
실행하면 아무것도 안 보이는 문제가 있어서:

- `rb5_binpicking/bin_geometry.py` — bin/dest-bin 치수·위치 상수 + `CollisionObject`
  빌더를 공용 모듈로 분리.
- `rb5_binpicking/scripts/scene_setup.py` — 이 모듈로 2초마다 `/collision_object`를
  계속 publish하는 노드. `binpicking.launch.py`에 상시 등록.
- `moveit_pick_place.py`는 이 공용 모듈을 import하도록 리팩터링(중복 제거).

이제 `ros2 launch rb5_binpicking binpicking.launch.py`만 실행해도 source/destination
bin이 RViz Motion Planning scene에 자동으로 나타납니다.

---

## 3. 하드코딩 현황

| 항목 | 위치 | 동적/하드코딩 | 비고 |
|---|---|---|---|
| Pick 목표 XYZ | `object_pose` 토픽 + `grasp_offset_*` | **동적** (위치만) + pre-grasp 후 **재관측**(§7) | offset 자체는 하드코딩 기본값 |
| 파지 orientation | `SAVED_POSES["gripping"]["orientation"]` | 하드코딩 (명시적 fallback) | perception이 object 자세를 안 줌 — §7 `GraspCandidate` docstring 참고 |
| Place 목적지 | `generate_place_pose()`가 `dest_bin` 기하에서 계산 (§7) | **하드코딩 아님** | TCP↔object 오프셋(`grasp_offset_z`) 보정 포함 |
| Watch 포즈 | `SAVED_POSES["watch"]` | 하드코딩 | 관측 위치라 파생시킬 기하가 없음 |
| grasp/lift 오프셋 (`grasp_offset_z=0.16` 등) | `DEFAULT_DYNAMIC_PICK_PARAMS`, `moveit_pick_place.py` | 하드코딩 기본값 (ROS param 오버라이드 가능) | launch에서 세팅해주는 곳 없음 |
| Bin/dest-bin 치수·위치 | `config/bin_geometry.yaml` | **단일 출처로 통합 완료 (§7)** | MoveIt·Isaac Sim 씬 모두 이 YAML만 읽음 |
| 씬 오브젝트(구/큐브/실린더) 크기·색·mass·랜덤시드 | `binpicking_scene.py` | 하드코딩 (`random.seed(7)`) | |
| D435i 카메라 intrinsics | `binpicking_scene.py` | 하드코딩 | |
| `GROUP_NAME = "mainpulation"` | SRDF + 3개 파이썬 파일 + `kinematics.yaml` | 하드코딩 유지 (의도적 결정, §7) | 4곳 전부 rename 검증할 방법이 없어 보존 |
| `pcd_offset_x/y/z` (카메라 포인트클라우드 보정) | `binpicking.launch.py` launch arg | **기본값 0.0으로 변경 (§7)**, 0이 아니면 launch 경고 출력 | 근본 원인(TF/frame 불일치)은 여전히 미해결 |
| Object 크기 (`object_half_height=0.021`) | `moveit_pick_place.py` place-pose 계산 | 하드코딩 (단일 큐브 크기 가정) | perception이 치수를 안 줌 — §7 한계 |

---

## 4. 알려진 이슈 (2026-07-21 진단)

1. **~~갑작스런 고속 회전~~ — FIXED (2026-07-21)**
   `trajectory_bridge.py`의 세그먼트 선택 로직이 `max(1, ...)`로 세그먼트 0을 강제로
   건너뛰어서, MoveIt 재계획(replan) 후 첫 waypoint의 `time_from_start`가 0이 아닐 때
   Hermite spline을 `t < t0`(음수 s)에서 평가 → 극단적으로 잘못된 pos/vel 산출.
   `max(0, ...)`로 수정, `rb5_isaac` 재빌드 완료.

2. **"궤적 시간 = 성공"으로 오판 — FIXED (2026-07-22, §7)**
   `trajectory_bridge.py`가 궤적 시간이 다 지나면 실제 도달 여부와 무관하게 무조건
   `goal_handle.succeed()`하던 문제. 이제 `/joint_states` 실제 피드백으로 목표
   도달 + 정지(settling)를 확인한 뒤에만 성공 처리. 단, **Isaac Sim
   contact/force sensor는 여전히 연결 안 돼 있음** — grasp/충돌 검증은 그리퍼
   stall 신호 + test-lift(§7 §16)라는 대체 신호를 쓰는 것이지 진짜 접촉 센서가
   아님. 진짜 contact 모니터링은 미해결.

3. **카메라 포인트클라우드 오프셋 — 부분 해결 (2026-07-22, §7)**
   `pcd_offset_x/y/z` 기본값을 0.0으로 변경했고, 0이 아닌 값을 쓰면 launch가
   경고를 출력하도록 했음. 하지만 **근본 원인(Isaac Sim RTX depth 카메라 prim과
   URDF `camera_depth_frame`/`camera_depth_optical_frame`의 실제 불일치)은 여전히
   미조사** — 지금은 그냥 보정값을 안 쓰는 상태로 되돌린 것뿐, TF diagnostic 도구는
   아직 없음.

4. **하드코딩 중복 — 해결 (2026-07-22, §7)**
   `config/bin_geometry.yaml`을 단일 출처로 만들어 `bin_geometry.py`(MoveIt 쪽)와
   `binpicking_scene.py`(Isaac Sim 씬)가 모두 이 파일만 읽도록 통합. Place
   목적지도 이제 하드코딩이 아니라 `dest_bin` 기하에서 계산됨. Grasp orientation은
   perception이 object 자세를 안 줘서 여전히 하드코딩(§3, §7 한계로 명시).

### 환경 관련 주의사항 (버그 아님, 함정)

`~/.bashrc`가 `/home/hh/asl_ws/install/setup.bash`(이 저장소와 무관한 **오래된 별도
workspace**, 2026-06-05 빌드)를 자동으로 source함. 이 워크스페이스가 우연히
`rb5_isaac`, `rb5_binpicking` 등 동일 패키지명을 담고 있어서, 새 터미널을 열면
이 저장소를 리빌드해도 옛날 빌드가 조용히 실행될 수 있음. 항상 아래처럼 **현재
저장소 setup.bash를 나중에 다시 source**해야 함:

```bash
source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
```

---

## 5. 로드맵 (요약 — 상세는 `README.md` 참고)

- **Phase 1 (현재)**: 위 §2의 하드코딩 상태머신. 동작 확인됨.
- **Phase 2**: `moveit_pick_place.py`를 NVIDIA Isaac GR00T(VLA) 정책으로 교체.
  `action_adapter.py`(`[dx,dy,dz,gripper]` ↔ EEF pose)와 `rb5_action_space.yaml`이
  이미 이 용도로 미리 만들어진 미배선 스캐폴딩으로 존재.
- **Phase 3**: NAVER LABS Europe의 **DUNE**(범용 2D/3D 인코더)을 spatial feature
  소스로 GR00T에 추가해 fine-tuning. 착수 보류 상태, 방향만 정해짐.

---

## 6. 남은 작업 (2026-07-22 갱신 — §7 반영 후 기준)

- [ ] Isaac Sim 실제 contact/force sensor 연결 (지금은 gripper stall + test-lift가
      대리 신호일 뿐, 진짜 접촉 센서 아님)
- [ ] 카메라 포인트클라우드 오프셋 근본 원인(TF/frame 불일치) 조사 — TF diagnostic
      도구는 아직 없음
- [ ] Grasp orientation을 object 자세 기반으로 바꾸려면 `/binpicking/object_pose`에
      방향(quaternion) 정보 자체를 추가해야 함 (현재 ground-truth position만 옴)
- [ ] `object_half_height` 같은 object 치수 가정을 perception이 실제로 주는 값으로
      교체 (지금은 알려진 큐브 하나 가정)
- [ ] MoveIt Task Constructor / MoveIt Servo 전환 검토 (설계는 §7 코드 구조로 준비됐지만
      실제 마이그레이션은 안 함)
- [ ] `mainpulation` 오타 전체 rename 여부 재검토 (§7에서 "보존"으로 결정, 사유는 §3 표 참고)
- [ ] JSONL 트라이얼 로그(`~/.ros/rb5_binpicking_trials.jsonl`)에 joint RMS
      tracking error, perception latency 등 세부 지표 추가

---

## 7. Phase 1 신뢰성 대개편 (2026-07-22)

"박스가 RViz에 안 보인다"는 사소한 요청에서 시작해, pick-and-place가 실제로
얼마나 신뢰할 수 있는지 코드 레벨에서 감사(audit)하고 고친 큰 세션. **GR00T/DUNE/
학습 정책은 아직 손 안 댔고, 순수 Phase 1(휴리스틱) 신뢰성만 다룸.**

### 7.1 근본 원인 (심각도 순)

1. **`trajectory_bridge.py`가 궤적 "시간 경과"를 "성공"으로 취급** — 실제
   `/joint_states`를 전혀 확인하지 않고 나열된 시간이 지나면 무조건
   `goal_handle.succeed()`. 로봇이 막히거나, 충돌하거나, 명령이 씹혀도 MoveIt은
   성공으로 앎. **가장 심각한 원인이었음.**
2. **grasp 검증이 구조적으로 불가능했음** — `moveit_pick_place.py`가 그리퍼를
   `/gripper_joint_commands` 토픽에 직접 publish해서 움직였는데, 이건
   `trajectory_bridge.py`의 `GripperCommand` 액션 서버를 완전히 우회하는 경로라
   애초에 피드백을 받을 수 있는 통로가 없었음.
3. **Bin 기하가 두 군데(MoveIt/Isaac Sim)에 독립적으로 하드코딩** — 값이 갈라져도
   아무도 모름.
4. **object pose를 frame 변환 없이 그대로 사용** — TF 변환 없이 좌표를 바로
   PLANNING_FRAME으로 취급. 지금은 `world`와 `link0`가 identity라 우연히 맞았지만,
   구조적으로는 틀린 가정.
5. **재관측 없음** — object pose를 한 번만 받아서 grasp 끝까지 그대로 사용.
6. **LIN 실패 시 항상 OMPL로 폴백** — grasp/place처럼 접촉이 민감한 구간에서도
   OMPL이 예측 불가능한 경로로 접근할 수 있었음.
7. **Place 목적지가 bin과 무관한 저장된 좌표** — bin 크기를 바꿔도 place 위치는
   안 바뀜.

### 7.2 변경 파일

| 파일 | 무엇을 | 왜 |
|---|---|---|
| `rb5_isaac/rb5_isaac/trajectory_bridge.py` | 전면 재작성: 궤적 유효성 검사, start-state 체크, mid-execution tracking-error/command-jump 체크, 실제 정지(settling) 확인 후에만 성공, 그리퍼 stall 신호 추가 | §7.1-1, 2 |
| `rb5_binpicking/config/bin_geometry.yaml` | 신규 — bin 기하 단일 출처 | §7.1-3 |
| `rb5_binpicking/rb5_binpicking/bin_geometry.py` | 하드코딩 상수 → YAML 로더로 교체 | §7.1-3 |
| `rb5_binpicking/scripts/binpicking_scene.py` | bin 상수 블록을 plain YAML 리더로 교체 (Isaac Sim은 별도 인터프리터라 ROS 패키지 import 대신) | §7.1-3 |
| `rb5_binpicking/scripts/moveit_pick_place.py` | 전면 재작성: TF 기반 pose 변환, pre-grasp 후 재관측, stage 기반 흐름(StageResult), 접촉-민감 구간엔 LIN-only 정책(MotionClass), 그리퍼 액션 기반 grasp 검증 + test-lift, dest-bin 기하 기반 place, attach/detach, JSONL 트라이얼 로깅 | §7.1-2,4,5,6,7 |
| `rb5_binpicking/launch/binpicking.launch.py` | `pcd_offset_*` 기본값 0.0으로, 0이 아니면 경고 로그 | §4-3 |
| `rb5_binpicking/package.xml` | `tf2_geometry_msgs`, `control_msgs` 의존성 추가 | 위 변경들이 실제로 씀 |

### 7.3 새 ROS 파라미터

`trajectory_bridge.py`: `path_tolerance`(0.08rad), `goal_tolerance`(0.02rad),
`stopped_velocity_tolerance`(0.02rad/s), `goal_time_tolerance`(1.0s),
`joint_state_timeout`(0.5s), `settling_time`(0.2s), `allowed_start_tolerance`(0.05rad),
`max_command_jump`(0.3rad), `path_tolerance_grace_period`(0.2s).

`moveit_pick_place.py`: `max_object_pose_age`(1.0s), `refine_pose_timeout`(2.0s),
`max_object_pose_jump`(0.08m), `test_lift_height`(0.04m), `test_lift_z_threshold`(0.015m),
`place_settle_wait`(0.4s), `place_settle_move_threshold`(0.01m), `grasp_max_attempts`(4),
`place_max_attempts`(2), `lift_max_attempts`(3, §7.6), `lift_step_count`(3, §7.6),
`object_half_height`(0.021m), `destination_wall_clearance`(0.03m),
`place_approach_lift`(0.15m), `trial_log_path`(`~/.ros/rb5_binpicking_trials.jsonl`).

모두 초기 디버깅 기본값 — 실측 튜닝 안 됨.

### 7.4 검증 수준 (정직하게)

- **코드 작성**: 완료.
- **문법/import 체크**: 완료 (`ast.parse`, 실제 모듈 import 테스트, ROS 환경에서).
- **colcon build**: 완료, `rb5_isaac` + `rb5_binpicking` 모두 성공.
- **Isaac Sim 런타임 테스트**: 이 섹션 작성 시점엔 안 했었는데, 바로 이어서 사용자가
  실제로 돌려봤고 진짜 버그를 하나 잡음 → §7.6 참고. 그 버그를 고친 뒤의 재검증은
  아직 안 됨 (다음 실행에서 확인 필요).

### 7.5 의도적으로 안 한 것

- Isaac Sim 실제 contact/force 센서 연동 (API 존재 여부부터 미확인)
- MoveIt Task Constructor / MoveIt Servo 실제 마이그레이션 (코드는 나눠놨지만 전환 안 함)
- 커널MPC
- Object 방향 인식 기반 grasp candidate 생성 (perception이 방향 정보를 안 줌)
- `mainpulation` 오타 전체 rename (SRDF 2곳 + kinematics.yaml + 파이썬 — 검증 없이
  건드리면 planning group이 깨질 위험이 더 큼)

### 7.6 실제 실행에서 발견된 버그 + 재시도 로직 추가 (2026-07-22, 후속)

§7 코드를 실제로 돌려본 첫 실행에서 바로 문제가 재현됨. `run_dynamic_object_sequence`가
grasp까지는 성공했는데 LIFT에서 실패하고 멈췄고, 재실행하니 이번엔 첫 `watch` 이동부터
실패함. `move_group` 로그를 직접 확인해서 원인을 특정함:

```
Found a contact between 'robotiq_85_base_link' (Robot link) and 'target_object' (Robot attached)
Start state appears to be in collision with respect to group mainpulation
Motion planning start tree could not be initialized!
```

**원인**: `_attach_object()`에서 `GRIPPER_TOUCH_LINKS`에 손가락 링크만 넣고
`robotiq_85_base_link`/knuckle 링크를 빠뜨림 → attach하자마자 그리퍼 몸체와 attach된
박스가 충돌로 잡혀서 **현재 상태 자체가 영구히 "충돌 중"** 이 됨 → 이후 모든
planning(LIN이든 OMPL이든)이 즉시 실패. 게다가 그 시점까지 실패 경로에서 detach를
안 하고 있어서, `move_group`이 살아있는 한(재실행해도) 계속 이어짐.

**수정**:
- `GRIPPER_TOUCH_LINKS`에 그리퍼 전체 링크(`robotiq_85_base_link` + knuckle 4개) 추가.
- LIFT/PRE_PLACE/PLACE 실패 경로 전부에서 `_detach_object()` 호출 (물리적으로 그리퍼를
  여는 것과는 별개 — planning scene에서만 해제).
- `run()` 시작 시 무조건 한 번 `_detach_object()` 시도 — 이전 실행이 죽어서 attach가
  남아있어도 새 실행에서 자동 청소되도록.

**추가로 요청받은 재시도 로직** ("LIFT/PLACE도 자동 재시도"):
- `_attempt_lift()` 신규 — 한 번에 `post_grasp_lift`까지 큰 직선으로 올리는 대신,
  `lift_step_count`(기본 3)개의 작은 단계로 나눠서 올림. bin 벽 바로 옆에서 긴 직선
  경로 하나보다 짧은 경로 여러 개가 IK/충돌회피 여유가 더 있다는 판단. 전체 단계
  시퀀스를 `lift_max_attempts`(기본 3)번까지 재시도.
- `dynamic_pre_place` 이동도 `place_max_attempts`만큼 재시도하도록 변경 (기존엔
  1번 실패하면 바로 포기).
- place 자체는 이미 있던 XY jitter 재시도(§7 원안)를 그대로 유지.
- 모든 재시도는 grasp_attempts/lift_attempts/place_attempts로 JSONL 트라이얼 로그에
  기록됨.

### 7.7 상위 trial 루프 추가 (2026-07-22, 계속 진행 요청 반영)

`run()`을 "서버 대기 + 반복 루프"로, 기존 단발성 로직은 `_run_one_trial()`로 분리함.

```
run()
 ├─ /move_action, gripper 서버 대기 (1회)
 └─ while True:
       trial += 1
       if trial > max_trials: break          # max_trials=0이면 무제한
       success, stage, reason = _run_one_trial()
       if success: continue
       if stage in {LIFT, PRE_PLACE, PLACE}:  # 아직 물체를 쥔 채로 실패
           break                              # 물리 상태 불확실 → 루프 중단, 사람 확인 필요
       consecutive_failures += 1
       if consecutive_failures >= max_consecutive_failures: break
```

**안전 규칙**: 그리퍼가 물체를 쥔 채로 실패(LIFT/PRE_PLACE/PLACE)하면 무조건 루프를
멈춤 — 뭘 쥐고 있는지 모르는 채로 다음 pick을 또 시도하는 게 더 위험하다고 판단.
반대로 아직 아무것도 안 쥔 단계(WATCH, GRASP, 재관측 등)에서 실패하면 그리퍼가 이미
열려있는 상태이므로 다음 trial로 안전하게 넘어감 — 단, 그런 실패가
`max_consecutive_failures`(기본 3)번 연속되면 "구조적으로 뭔가 잘못됐다"고 보고 멈춤.

**중요한 한계**: `/binpicking/object_pose`는 `binpicking_scene.py`의
`/World/Objects/Cube0` 딱 하나만 계속 추적합니다. 즉 이 루프는 **"bin 안의 여러
물체를 순서대로 다 치운다"가 아니라, 그 시점에 토픽이 알려주는 위치(성공하면 이제
destination bin 안, 실패하면 여전히 source bin 안)로 반복해서 pick-place를 재도전하는
것**입니다. 여러 개의 서로 다른 물체를 자동으로 순회하려면 `binpicking_scene.py`
쪽에 "다음으로 집을 물체를 고르는" 진짜 perception 로직이 따로 필요합니다 — 지금은
없음.

새 파라미터: `max_trials`(기본 10, 0=무제한), `max_consecutive_failures`(기본 3).

### 7.8 카메라 transform 근본 수정 시도 (2026-07-22)

**진단**: `pcd_offset_*`(§4-3)가 필요했던 이유를 URDF와 `binpicking_scene.py`를 직접
대조해서 찾음. URDF는 `camera_depth_optical_frame`의 방향을
`camera_depth_optical_joint`의 `rpy`로 해석적으로 정확히 정의하는데,
`binpicking_scene.py`는 실제 렌더링에 쓰는 Isaac 카메라 prim의 방향을
`camera_link` 밑에서 `RotateY(-90)+RotateZ(90)`로 **눈대중 튜닝**했었음(주석에도
"makes that view vector match..."라고 명시). 시야 방향은 맞아도 광축 기준 roll까지
정확히 일치한다는 보장이 없어서, 회전 오차를 XYZ 평행이동(`pcd_offset_*`)으로
땜질하면 특정 거리에서만 맞는 patch가 됨 — 겪었던 증상과 일치.

**수정**: `camera_link` 밑에 새 prim을 만드는 대신, URDF에서 이미 정확히 정의된
`camera_depth_optical_frame`/`camera_color_optical_frame` 링크를 그대로 사용.
`import_config.merge_fixed_joints=False`라 이 fixed-joint 링크들도 독립 prim으로
보존돼 있음. 여기에 필요한 회전은 "ROS 광학 좌표계(+Z 정면) → Isaac/USD 카메라
좌표계(-Z 정면)" 변환 하나뿐인데, 이건 축 하나짜리 180도 회전(`Rx(180°)`)이라 회전
합성 순서 모호성이 없음(직접 손으로 URDF 체인 전체를 재유도하는 것보다 훨씬 안전).
이 180도 회전은 `camera_depth_optical_frame` 자체가 아니라 위치 오프셋 0인 자식
prim(`.../camera_depth_optical_frame/isaac_view`)에만 걸어서, TF로 나가는
`camera_depth_optical_frame` 값 자체는 URDF 그대로 유지되도록 함.

**자체 검증 장치**: 시작할 때 optical_frame(TF 기준 위치)과 실제 렌더링 prim의
위치를 비교해서 `[CHECK] depth: ... diff=X.XXmm` 로그를 출력함 — 회전만 다르고
위치는 동일해야 하므로 diff는 항상 ~0mm이어야 함. 링크를 못 찾으면(예상과 다르게
merge된 경우) 예전 방식(camera_link + 수동 회전)으로 폴백하고 `[WARN]` 로그를 남김.

**검증 수준**: 코드 작성 + 문법 체크만 완료. **Isaac Sim 런타임 검증은 안 됨**
(`binpicking_scene.py`는 `~/isaacsim/python.sh`라는 별도 인터프리터가 필요해서 이
세션에서 import조차 테스트 불가). 다음 실행 때 `[OK] Camera sensors attached
directly...` 로그가 뜨는지, `[CHECK]` diff가 0에 가까운지, RViz에서 pointcloud가
이제 `pcd_offset` 없이도 맞게 보이는지 확인 필요.

**Delay(지연) 문제는 별개 원인**— 아직 손 안 댐. 실제로 측정부터 필요:
```bash
ros2 topic hz /camera/depth/image_rect_raw
ros2 topic hz /camera/color/image_raw
ros2 topic delay /camera/depth/image_rect_raw   # timestamp vs 수신 시각 차이
ros2 topic hz /clock
```
`RayTracedLighting` 렌더러 + 매 `simulation_context.step()`마다 수동 OmniGraph
impulse를 쏘는 구조라, GPU가 못 따라가면 publish 주기가 밀릴 수 있음 — 위 측정
결과를 보고 렌더러/해상도 조정이 필요한지 판단해야 함.

### 7.9 실제 실행 피드백 반영 — 들어올리는 높이 / 그립 힘 / transfer 중 회전 (2026-07-22)

실제로 돌려보고 받은 피드백: (1) 들어올릴 때 좀 더 위로, (2) 옮기는(돌리는) 중에
물체가 떨어짐 → 더 꽉 쥐어야 함, (3) bin 벽에 박음.

**원인**: 세 증상이 서로 얽혀 있었음.
- `post_grasp_lift=0.22m`가 source bin 벽 높이(0.22m)와 거의 같아서, 바닥 근처에서
  집은 물체는 들어올려도 벽 위로 몇 cm밖에 안 남았음 → `dynamic_pre_place`로 옆으로
  이동할 때 벽에 닿을 여유가 거의 없었음.
- `use_path_orientation_constraint=False`였어서, LIN이 실패해 OMPL로 폴백하는
  경우(예: 벽 근처에서 흔한 상황) 시작/끝 orientation만 맞으면 되고 **경로 중간에는
  손목이 자유롭게 돌아갈 수 있었음** — "돌리는 과정에서 떨어진다"는 증상과 정확히
  일치.
- 그리퍼 관절 stiffness(5000)가 낮아서, 물체를 문 상태에서 회전/가속 중 관성력을
  버틸 그립 힘이 부족했을 가능성.

**수정**:
- `post_grasp_lift` 기본값 0.22m → 0.30m.
- `_attempt_lift()`에 **bin 기하 기반 최소 들어올림 높이**를 추가: `(source_bin
  floor_z + height + lift_clearance_margin(기본 0.05m)) - object_z` 를 계산해서,
  파라미터 기본값보다 이 값이 크면 그만큼 더 들어올림. 바닥 어디서 집든 벽을
  일정 여유로 넘도록 물체 위치 기준으로 동적 계산.
- `use_path_orientation_constraint` 기본값 False → **True**. LIN이 실패해서 OMPL로
  넘어가도 경로 내내 orientation_tolerance 안에서 유지하도록 강제. Trade-off: OMPL이
  풀기 더 어려워져서 transfer 중 planning 실패가 늘 수 있음 — 그건 (이미 있는
  bounded retry로) 눈에 보이는 실패로 처리되니, 조용히 물체를 떨어뜨리는 것보다
  낫다고 판단.
- `binpicking_scene.py`의 그리퍼 관절 drive: stiffness 5000→9000, damping 200→350
  (maxForce는 이미 1e6으로 충분히 커서 그대로 둠). 같은 finger gap에서 더 센 그립
  힘이 나오도록.

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build`만 완료. Isaac Sim 런타임
검증은 안 됨 — 특히 `use_path_orientation_constraint=True`로 인해 transfer 단계
planning 실패율이 실제로 얼마나 늘어나는지는 직접 돌려봐야 알 수 있음. 만약 planning
실패가 너무 잦아지면 `orientation_tolerance`(현재 0.20rad)를 먼저 늘려보는 게
우선순위 — 원래 코드 주석에도 있던 조언 그대로 유효함.

### 7.10 detach가 "삭제"가 아니라 "그 자리에 world object로 되돌림"이었던 버그 (2026-07-22)

RViz에서 "춤추는" 것처럼 보인다는 피드백 → `move_group` 로그 확인 →
```
Found a contact between 'target_object' (type 'Object') and 'robotiq_85_base_link' (type 'Robot link')
Start state appears to be in collision with respect to group mainpulation
```
**원인**: `_detach_object()`가 `AttachedCollisionObject` REMOVE만 보냈는데, 이건
"링크에서 떼어낸다"는 뜻이지 "삭제한다"는 뜻이 아님. 떼어낸 물체는 **그 자리에
고정된 world collision object로 남음** — 놓는 위치가 그리퍼 바로 옆이라 유령
충돌체가 그리퍼 몸체와 계속 겹침. 이후 모든 planning이 이 유령 물체를 피하려고
뒤틀린 경로를 짜내는 게 RViz의 "춤추는" 동작으로 보였을 가능성이 높고, 같은 물체를
반복해서 집었다 놓았다 하는 구조라 "다른 박스로 옮기다 떨어진다"는 증상에도
영향을 줬을 것으로 추정.

**수정**: `_detach_object()`가 이제 `AttachedCollisionObject` REMOVE(그리퍼에서
떼기) + `/collision_object` REMOVE(월드에서 완전히 삭제)를 둘 다 보냄.

### 7.11 실제 실행 피드백 2차 — "위에서 조금 내렸다 다시 올라오는" 반복 + 안 집었는데 닫힘 (2026-07-22)

트라이얼 로그를 다시 확인해서 서로 다른 두 개의 원인을 찾음.

**증상 A — "위에서 조금 내렸다 다시 올라왔다 반복"**: 로그 상 4개 grasp candidate
전부가 `dynamic_grasp`(CONTACT_SENSITIVE, 원래 LIN 전용) 단계에서 `PLANNING_FAILED`.
Pilz LIN이 `NO_IK_SOLUTION`을 내면 그 후보는 영원히 도달 불가능한 채로 남고(OMPL
폴백이 없었으므로), pre_grasp 높이만 후보마다 살짝(2cm 정도) 다르게 오르내리는 게
반복됨 — 실제로 내려가서 집는 시도 자체가 한 번도 성공 못 함.

- **수정**: CONTACT_SENSITIVE 모션도 이제 **OMPL 폴백을 허용**함. 처음에 LIN 전용으로
  만든 이유(OMPL이 예측 불가능한 방향으로 접근할 위험)는, `use_path_orientation_constraint`가
  이제 기본 True라서 OMPL 폴백도 **항상 경로 전체 orientation 제약을 강제로 걸도록**
  바꿔서 상쇄시킴. `move_to_pose()` 독스트링에 트레이드오프 정리해둠.

**증상 B — "집지도 않았는데 닫혀 있고 다른 박스로 놓으려 함"**: 트라이얼 로그에서
grasp는 검증(그리퍼 stall + test-lift 둘 다) 통과했는데 `lift_attempts=3`(즉 staged
lift 1·2차 시도가 실패하고 3차에 성공)인 trial을 발견. **원인**: lift 재시도 도중
실패한 시도가 급정지하면서 이미 검증됐던 그립이 흔들려 빠질 수 있는데, `_attempt_lift()`가
"결국 목표 높이에 도달했다"는 것만 성공 조건으로 봐서 **그 사이에 물체를 놓쳤는지는
전혀 재확인을 안 했음**. 그러니 빈 그리퍼로 계속 place까지 진행.

- **수정**: `_attempt_lift()` 성공 직후 `_confirm_still_holding()`을 새로 호출 —
  `/binpicking/object_pose`를 다시 관측해서 물체가 "그리퍼가 들고 있어야 할 높이"에
  실제로 있는지(±5cm) 확인. 어긋나면 `LIFT_FAILED`로 처리하고 detach 후 중단 —
  빈 그리퍼를 들고 목적지까지 가서 아무것도 안 내려놓는 상황을 막음.

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build`만 완료, Isaac Sim 재실행 검증
필요.

### 7.12 Pick 성공 후 절대좌표 Z 방향으로만 들어올리기 (2026-07-22)

`move_to_pose()`/`move_to_xyz()`에 `linear_only` 옵션 추가. `True`면 OMPL 폴백을
아예 안 붙이고 **Pilz LIN만** 시도함 — OMPL은 (§7.11에서 강제한 orientation 제약이
있어도) **position은 전혀 제약을 안 하므로**, X/Y가 흔들리며 우회하는 경로가 나올 수
있음. Lift처럼 "정확히 한 방향으로만 움직여야 하는" 모션은 애초에 Pilz LIN(직선
보간, PLANNING_FRAME=link0 기준 — world와 identity라 절대좌표와 동일)만 써야
의미가 있어서, 실패하면 그냥 실패(→ 기존 `_attempt_lift`의 bounded retry)로 처리하고
OMPL이 대신 구불구불한 경로를 만들어내지 않게 함.

적용 위치: `_test_lift_verify()`의 test-lift, `_attempt_lift()`의 staged lift 전부
`linear_only=True`로 호출하도록 변경. 둘 다 X/Y는 grasp 시점 값 그대로 두고 Z만
바꾸는 waypoint라 원래도 "직선"을 의도했는데, 지금까지는 LIN이 실패하면 OMPL이
그 의도를 깨고 우회할 수 있었음.

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build`만 완료, Isaac Sim 재실행 검증
필요.

### 7.13 물체 orientation을 실제로 사용 + quaternion 정규화 버그 (2026-07-22)

**발견 1 — quaternion 정규화 버그**: `/binpicking/object_pose`의 orientation 값이
전부 norm≈0.53으로 찍힘(정상 회전 quaternion은 norm=1이어야 함). 원인:
`binpicking_scene.py`의 `_target_object_pose_msg()`가 `ComputeLocalToWorldTransform()`
결과(스케일 포함된 4x4 행렬)에서 `ExtractRotationQuat()`으로 바로 뽑아 쓰는데, 이
prim이 `DynamicCuboid(scale=...)`로 스케일이 걸려 있어서 순수 회전이 아니게 됨.
**수정**: publish 직전에 명시적으로 정규화.

**발견 2 — orientation을 아예 안 쓰고 있었음**: `generate_grasp_candidates()`가
지금까지 물체가 어떻게 놓여있든 항상 고정된 `SAVED_POSES["gripping"]` orientation만
썼음. 예전 docstring에 "position-only ground truth라 orientation 정보가 없다"고
적어놨었는데, 이건 **잘못된 가정**이었음 — 토픽엔 실제 orientation이 들어있음(정규화만
안 됐을 뿐). **수정**: 물체의 실제 yaw(수평 회전)를 추출해서 고정 orientation에
합성 — `object_yaw`만큼 world Z축으로 추가 회전시켜서 그리퍼가 물체의 실제 방향에
맞춰 접근하도록 함. 또한 물체가 25도(`max_object_tilt`) 이상 기울어져 있으면(옆으로
쓰러진 경우) top-down 방식으로는 애초에 못 집으니, 억지로 시도해서 물체를 더 나쁜
자세로 쳐내는 대신 **grasp 자체를 깔끔하게 거부**하도록 함.

**실제 로그로 검증해본 결과 (정직하게 보고)**:

| Trial | tilt | yaw | 예상 |
|---|---|---|---|
| c08bd97e (성공) | 0.0° | 0.0° | 통과, 변화 없음 — 맞음 |
| ae7c8fa6 (실패) | **171.6°** | -3.9° | 이제 즉시 거부됨 — 이 케이스는 해결 |
| 08372308 (실패) | **8.7°** | -3.2° | 25° 미만이라 거부 안 됨, yaw 보정도 미미함 — **이 케이스는 orientation 문제가 아니었음** |

즉 심하게 기울어진 물체(거의 뒤집힌 경우)는 이제 확실히 걸러지지만, **거의 flat인데도
실패했던 케이스(08372308)는 이 수정으로 설명이 안 됨** — orientation/tilt가 원인이
아니라 다른 이유(예: §7.9에서 올린 그리퍼 stiffness가 너무 세서 닫히는 순간 작고 가벼운
큐브를 쳐내버리는 것, 혹은 접근 깊이 자체의 문제)일 가능성이 있음. **다음으로 의심할
곳**: 그리퍼 닫는 속도/힘 — 지금 `trajectory_bridge.py`의 `_execute_gripper`는
여는 것과 닫는 것 모두 동일하게 0.5초 고정 램프를 씀. 작고 가벼운 물체에 더 부드럽게
접근하도록 닫는 속도를 늦추거나 damping 비율을 높이는 걸 다음 후보로 검토 필요.

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build` + 실제 로그 데이터로 수식
검증(위 표)까지 완료. Isaac Sim 재실행 검증은 아직 안 됨.

### 7.14 그리퍼 닫는 속도 분리 (2026-07-22)

§7.13에서 orientation/tilt로는 설명 안 되는 실패 케이스(거의 flat한데도 grasp
실패)가 남아있었음. 유력 후보로 지목했던 것: `trajectory_bridge.py`의
`_execute_gripper`가 **열기/닫기 둘 다 똑같이 0.5초 고정 램프**를 씀 — §7.9에서
그립 힘(stiffness)을 5000→9000으로 올렸는데, 빠른 속도로 닫으면 작고 가벼운 큐브가
접촉 순간 튕겨나가버릴 수 있음(momentum 문제 — 쥐는 힘과는 별개).

**수정**: `gripper_close_duration`(기본 1.2s, 기존 0.5s보다 느림)과
`gripper_open_duration`(기본 0.5s, 그대로) 파라미터로 분리. 여는 건 섬세할 필요가
없어서 빠른 그대로 두고, **닫을 때만 느리게** 해서 접촉 시 momentum을 줄임 — 일단
잡힌 뒤의 쥐는 힘(stiffness)은 그대로라 transfer 중 안 놓치는 효과는 유지됨.
`gripper_settle_duration`(0.3s)도 파라미터화.

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build`만 완료. 이 가설이 실제
원인이었는지는 Isaac Sim 재실행 후 확인 필요 — 안 되면 damping 비율 조정이나
stiffness 자체를 낮추는 것도 다음 후보.

### 7.15 Place/retreat도 linear_only 누락 + "기울어진 채 영구 고착" 패턴 (2026-07-23)

트라이얼 로그를 넓게(30개) 봐서 패턴을 확인: `VERIFY_PLACE` 실패("목적지 bin에
제대로 안착 안 함")가 발생할 때마다, 그 직후부터 같은 물체가 **똑같은(기울어진)
orientation으로 계속 즉시 거부**됨 — grasp+lift+transfer는 되는데 마지막에 놓는
동작이 물체를 깔끔하게 눕히지 못하고, 한 번 이렇게 되면 아무도 물체를 다시
안 건드리니 영구히 그 자세로 고착됨. `max_consecutive_failures=3`은 정상 작동
중이었음(로그가 정확히 3개씩 끊겨 있었음) — 사용자가 그때마다 수동 재시작해서
계속 반복된 것.

**원인**: §7.12에서 lift에만 `linear_only=True`를 적용했고, **place와 retreat엔
빠뜨렸음.** place도 X/Y 고정, Z만 바뀌는 직선 하강이 의도인데, LIN이 실패하면
OMPL이 대신 (orientation만 지키고 position은 안 지키는) 우회 경로로 내려갈 수
있어서, 목적지 바닥에 똑바로 안착 안 하고 튀거나 비스듬히 놓일 수 있었음.

**수정**: `dynamic_place_{i}`, `dynamic_retreat` 둘 다 `linear_only=True` 추가.

**한계 (솔직히)**: 이걸로 기울어진 착지 빈도는 줄어들 수 있지만, 실제 물리
시뮬레이션이라 완전히 없앨 순 없음. 만약 다시 기울어진 채로 고착되면, 지금은
**사람이 직접 Isaac Sim에서 물체를 다시 눕혀주거나 씬을 재시작하는 것 외엔 자동
복구 수단이 없음** — 자동으로 "슬쩍 밀어서 바로 세우기" 같은 복구 동작은 아직 구현
안 함(잘못 밀면 bin 밖으로 나갈 위험이 있어서 신중하게 설계해야 함).

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build`만 완료, Isaac Sim 재실행
검증 필요. 참고로 URDF 자체(그리퍼 9개 링크 구조)는 이번 세션에서 한 번도 안
건드렸음 — 그리퍼 모델 로딩 문제는 아님, 동작 품질 문제.

### 7.16 그리퍼 fingertip 마찰력 — URDF 설정이 Isaac Sim에서 무시되고 있었음 (2026-07-23)

`rb5_isaac/urdf/rb5_with_tools.urdf`의 `robotiq_85_left/right_finger_tip_link`
collision에 이미 `<surface><friction><ode><mu1>100000.0</mu1><mu2>100000.0</mu2></ode></surface>`
가 박혀 있어서 마찰을 최대로 의도했던 것으로 보임. **문제**: 이건 Gazebo/ODE
전용 `<gazebo>` 확장 태그라 Isaac Sim의 URDF importer는 표준 URDF 요소만 읽고
이 블록을 완전히 무시함 — 지금까지 fingertip은 PhysX 기본 마찰 계수(~0.5)로만
시뮬레이션되고 있었고, 원래 URDF가 의도했던 "최대 마찰"은 시뮬레이션에 전혀
반영이 안 되고 있었음. "옮기는 중 미끄러진다"는 피드백과 정확히 맞아떨어짐.

**수정**: `binpicking_scene.py`에 `_apply_high_friction()` 추가 — PhysX
`PhysicsMaterial`(staticFriction=1.2, dynamicFriction=1.0, restitution=0.0)을
만들어서 양쪽 fingertip 링크에 `materialPurpose="physics"`로 바인딩.
`frictionCombineMode="max"`로 설정해서, 집는 물체 쪽 마찰 계수가 뭐든(기본값
그대로 둠) 이 fingertip 마찰이 항상 우선 적용되도록 함.

**한계**: URDF의 mu=100000 같은 극단적인 값은 PhysX Coulomb 마찰 모델에 그대로
포팅하면 수치적으로 불안정해질 수 있어서, 물리적으로 그럴듯한 범위(고무 정도,
~1.0~1.2)로 대신 설정함 — "무한 마찰"이 아니라 "현실적으로 꽤 높은 마찰"임.
그래도 여전히 미끄러지면 값을 더 올리거나(예: 1.5~2.0), 물체 쪽에도 별도
PhysicsMaterial을 붙여서 조합 방식을 재검토해야 함.

**검증 수준**: 코드 작성 + 문법 체크만 완료, Isaac Sim 재실행 검증 필요
(`binpicking_scene.py`는 Isaac Sim 전용 인터프리터라 이 세션에서 import/build
테스트 자체가 불가능 — `colcon build`도 이 파일엔 해당 없음, 그냥 소스를 그대로
읽어서 실행하는 스크립트).

### 7.17 "떨어짐"과 "두 번째 박스 collision" — 사실 같은 원인이었음: 팔 전체 여유 공간 부족 (2026-07-23)

마찰력 수정(§7.16)이 실제로 반영된 상태(Isaac Sim 시작 14:10:08 > 파일 수정
14:08:19로 확인)에서도 계속 떨어진다는 보고 → `move_group` 로그를 다시 넓게
분석. 어떤 로봇 링크가 충돌에 관여했는지 집계해보니:

```
link3(팔뚝)             224회
link2(팔꿈치)            199회
camera_link             87회
robotiq_85_base_link    85회
(그 외 그리퍼 손가락들)   합쳐서 ~90회
```

**그리퍼보다 팔뚝/팔꿈치가 훨씬 더 많이 충돌하고 있었음.** source_bin과
dest_bin 벽 양쪽 다에서, 그리고 `camera_link`와 `link3`끼리의 자체충돌(self
-collision)도 상당수 발생. 즉 "두 번째 박스에서 collision 난다"와 "잡았는데
떨어진다"는 **별개 문제가 아니라 같은 근본 원인** — bin 벽 주변에서 팔 전체가
거의 여유 공간 없이 스치고 있고, 이 과정에서 흔들리는 움직임이 마찰력을 아무리
올려도 극복 못 할 정도로 물체를 흔들었을 가능성이 높음.

**수정**: `bin_geometry.py`의 `make_bin_collision_objects()`에 `safety_margin`
(기본 0.02m) 추가 — 각 벽을 **바깥쪽으로만** 부풀림. 안쪽(내부 공간)은 정확히
그대로 두고 바깥쪽 면만 넓혀서, bin 내부에서 물체 집는 데는 지장 없이 팔이 벽을
피해 지나갈 때 여유 공간을 줌. 좌표 계산으로 안쪽 면이 안 움직이는 것도 직접
검증함.

**검증 수준**: 코드 작성 + 문법 체크 + `colcon build` + 좌표 계산 검증까지
완료. 마진 0.02m이 충분한지, 혹은 근본적으로 dest bin 위치/팔 도달 범위 자체를
재검토해야 하는지는 Isaac Sim 재실행 후 collision 로그가 실제로 줄었는지
확인해야 함.

### 7.18 그리퍼가 "닫히면서 살짝 들어올리듯" 움직이는 문제 — 손가락 링크들이 물리적으로 연결돼있지 않았음 (2026-07-23)

사용자 보고: 물체를 잡는 순간 그리퍼가 살짝 들어올리는 듯한 동작을 보이고,
그 결과 물체의 끝부분만 걸쳐 잡아서 이후 떨어짐/불안정으로 이어진다는 문제.

**원인**: Robotiq 2F-85는 실제로는 액추에이터가 달린 축이 `left_knuckle_joint`
하나뿐이고, 나머지 5개 관절(`right_knuckle`, `left/right_inner_knuckle`,
`left/right_finger_tip`)은 4-bar 링크 구조로 그 하나의 축에 기구적으로
종속되어 움직이는 수동(passive) 관절이다. 그런데 이 URDF는 닫힌 링크 구조를
표현할 수 없는 트리 구조라서, 지금까지 `binpicking_scene.py`는 이 6개 관절을
**각자 독립적인 PD 드라이브**로, `trajectory_bridge.py`가 보내는 **시간 기반
스케줄**(실제 knuckle 각도가 아니라 "이만큼 시간이 지났으니 이 각도"라는
계획값)을 향해 따로따로 구동하고 있었다.

빈 공간에서는 문제가 없다 — 아무 저항이 없으니 6개 관절이 모두 계획대로
동시에 목표에 도달한다. 문제는 손가락 끝이 물체에 닿는 순간 발생한다:
주 관절(`left_knuckle`)은 물체에 막혀 멈추는 게 정상이고 의도된 동작이다
(이 "멈춤"을 감지해서 grasp 성공 여부를 판단함). 하지만 나머지 5개 관절은
주 관절이 막혔다는 걸 전혀 모른 채 자기 스케줄대로 계속 움직이려 하고,
각자 물체와 접촉하는 위치/저항이 다르므로 서로 다른 지점에서 멈추게 된다.
그 결과 손가락 전체 형상이 원래 의도한 "평행하게 오므라드는" 모양에서
벗어나면서 fingertip pad가 기울어지고, 그 기울어짐이 마치 "닫으면서 살짝
들어올리는" 것처럼 보이고 실제로 물체 끝부분만 걸치게 되는 것으로 판단됨.
마찰 계수(§7.16)를 아무리 올려도,애초에 접촉면이 기울어져서 제대로 안
걸리면 소용이 없다는 점과도 일치.

**수정**: `left_knuckle_joint`만 실제 PD 드라이브로 구동하고, 나머지 5개는
PhysX의 `PhysxMimicJointAPI`(기어 구속조건 — 매 물리 스텝마다 주 관절의
*실제 측정 각도*에 대수적으로 묶이는 진짜 물리 제약)로 전환. 스케줄이 아니라
실제 상태에 종속되므로, 주 관절이 물체에 막혀 멈추면 나머지 5개도 그 순간의
실제 각도에 맞춰 함께 멈춘다 — 진짜 4-bar 링크가 하는 일과 동일하게 동작.
독립 드라이브는 제거(stiffness=0, damping만 약간 남겨 수치 안정화용).
Isaac Sim/PhysX 확장 트리의 `MimicJointDemo.py` 예제를 참고해 기어비를
유도했고(공식: `jointPosition + gearing·referenceJointPosition + offset = 0`),
기존 `trajectory_bridge.py`의 `GRIPPER_MIMIC` 부호와 URDF의 joint axis/limit
방향을 대조해서 검증함. 스키마 적용이 실패하는 경우(버전 차이 등)에 대비해
try/except로 감싸 실패 시 기존 독립 드라이브 방식으로 자동 폴백하도록 처리.

**검증 수준 / 결과**: Isaac Sim에 반영해서 실제로 테스트한 결과, 그리퍼뿐
아니라 **팔(manipulator) 쪽 동작(action)까지 이상해짐**을 사용자가 확인함.
`PhysxMimicJointAPI` 문서에 "mimic joint는 양방향 상호작용이며 reference
joint에도 반작용 힘이 가해진다"고 명시돼 있는데, 그리퍼와 팔이 하나의
articulation으로 묶여 있으므로 기어 구속조건 설정(축 지정/기어비 등)에
문제가 있었다면 그 힘이 체인을 타고 팔 관절 솔버까지 전달되어 전체가
불안정해질 수 있다 — 실제로 그런 것으로 보임. **원인 진단(수동 관절들이
독립 PD 드라이브라서 접촉 시 형상이 틀어진다는 분석) 자체는 유효할 가능성이
높지만, mimic joint를 이용한 이번 구현은 안전하지 않아 즉시 원복함**
(6개 관절 모두 독립 PD 드라이브로 되돌림, §7.16 상태와 동일). 이 접근은
Isaac Sim 재시작으로만 반영/원복되므로(코드가 Isaac Sim 전용 인터프리터에서
실행됨), 폴백 이후 팔 동작이 정상으로 돌아왔는지 재시작 후 확인 필요.
근본적인 "닫으면서 살짝 들어올리는" 문제는 아직 미해결 상태 — 더 안전한
접근(예: 관절 계층 구조 재검토, gearing 부호 재검증, 또는 mimic 대신 다른
방법)이 필요함.

### 7.19 mimic joint 대신 — 수동 관절 5개의 힘을 약하게만 (팔에 영향 없는 안전한 버전) (2026-07-23)

§7.18의 mimic joint 시도가 팔까지 불안정하게 만들어서 원복한 뒤, 같은 원인
진단(수동 관절 5개가 독립 PD로 각자 스케줄만 보고 움직이다가, 접촉 저항을
만나면 서로 다른 지점에서 멈춰서 fingertip pad가 기울어짐)을 유지한 채,
articulation 간 새 constraint를 만들지 않는 훨씬 안전한 완화책 적용:
주 관절(`left_knuckle_joint`)은 기존 그대로(stiffness 7000/damping 350/
maxForce 1e6)로 두고, 나머지 5개 수동 관절만 훨씬 약하게
(stiffness 1200/damping 150/maxForce 3e4) 낮춤. 빈 공간에서 가벼운 링크
자체를 움직이는 데는 충분한 힘이지만, 물체 접촉 저항을 만나면 억지로 뚫고
들어가지 않고 순순히 멈추도록 하는 목적. mimic joint와 달리 관절 간
구속조건을 새로 만들지 않으므로(각 관절이 여전히 독립적인 PD 드라이브일
뿐, 서로/팔에 반작용력을 주고받지 않음) 팔에 영향을 줄 수 없음.

**검증 수준**: 문법 체크만 완료. Isaac Sim 재시작 후 실제로 "닫으면서
살짝 들어올리는" 현상이 줄었는지 확인 필요.

### 7.20 진짜 근본 원인 발견 — `grasp_offset_z`가 처음부터 ~5cm 높게 잘못 설정되어 있었음 (2026-07-23)

사용자 보고: pick은 되는데 place가 안 됨, 이유는 정육면체 박스의 옆면이
아니라 위쪽 모서리를 집는 것 같다는 것. §7.18/7.19에서 다루던 "손가락
링크 간 미세한 형상 어긋남"과는 별개로, **애초에 grasp 목표 높이 자체가
크게 잘못 계산되고 있었던** 훨씬 큰 원인을 발견함.

`grasp_offset_z`(기존 0.16m)는 "TCP를 물체 중심보다 이만큼 위에 두면,
그리퍼가 top-down orientation으로 접근했을 때 손가락 pad가 물체 중심
높이에 오도록" 만드는 상수인데, 이 값이 실제 기하와 맞는지 한 번도
검증된 적이 없었음. 확인 방법 (README2.md 관례대로 코드로 직접 검증,
손으로 3D 회전 계산 안 함):

1. Isaac Sim이 켜져 있는 동안 실측: `ros2 run tf2_ros tf2_echo tcp
   robotiq_85_left_finger_tip_link` → tcp 로컬 프레임 기준
   `[0.068, -0.098, 0.000]` (그리퍼 열림 상태).
2. URDF의 관절 원점/축 값으로 직접 순운동학(forward kinematics)을
   Python으로 계산 → 열림 상태 `[0.0678, -0.0983, 0.0]`로 실측치와
   소수점 단위까지 일치 (계산 스크립트 검증됨).
3. 같은 계산을 "gripping" 저장 orientation(`[0.532, 0.528, 0.458,
   0.478]`)으로 world frame으로 회전시켜, tcp 기준 fingertip이 world Z로
   얼마나 아래에 있는지 계산: 열림 상태 ~98mm, 물체를 실제로 쥘 만큼
   닫힌 상태(knuckle 0.3~0.7rad 범위, 즉 대략 6~10cm 폭 물체)에서는
   ~105~111mm로 이 범위 전체에서 거의 일정함.

즉 실제로 필요한 값은 ~0.105~0.111m인데 코드는 0.16m을 쓰고 있었음 —
**약 5cm(!) 너무 큼**. TCP를 물체 중심+16cm에 놓으면 실제 손가락은
물체 중심보다 약 5cm 위에서 닫히게 되고, 물체 높이가 4.2cm
(`object_half_height`=0.021m → 전체 높이)밖에 안 되니 이 정도면 물체
윗면 위쪽에서 닫히려는 것과 마찬가지 — "위쪽 모서리를 집는다"는 사용자
관찰과 정확히 일치함. §7.14/7.16/7.18/7.19에서 계속 다루던 "손가락이
미끄러진다/기울어진다"는 사실 부차적인 문제였고, 이게 훨씬 큰 주된
원인이었을 가능성이 높음.

**수정**: `grasp_offset_z` 기본값을 0.16 → 0.108로 변경
(moveit_pick_place.py `DEFAULT_DYNAMIC_PICK_PARAMS`). fingertip pad의
실제 접촉면이 `robotiq_85_left_finger_tip_link` 원점과 정확히 일치하는지는
(그 링크는 primitive가 아니라 mesh collision이라 URDF만으로는 pad 표면
오프셋을 알 수 없음) 확인 못 했으므로 수 mm 단위 잔여 오차 가능성은 있음.
그 정도 오차가 남아있다면 재빌드 없이
`ros2 run rb5_binpicking moveit_pick_place.py --ros-args -p
grasp_offset_z:=<값>`으로 바로 미세 조정 가능.

**검증 수준**: 문법 체크 + `colcon build` 통과. 라이브 TF 실측 + 순운동학
계산 교차검증까지 완료(계산 신뢰도 높음). 다만 **Isaac Sim에서 실제로
옆면을 잡고 place가 성공하는지는 아직 미검증** — moveit_pick_place.py
재시작(Isaac Sim 재시작은 불필요, 이 파일은 일반 ROS 노드라 재빌드 후
재실행만 하면 반영됨) 후 확인 필요.

### 7.21 §7.17의 벽 안전 여유가 역효과 — "링크가 벽에 끼임"으로 이어짐, 원복 (2026-07-23)

사용자 보고: "매니퓰레이터가 계속 벽에 끼인다, 링크 부분이." 실행 중이던
move_group 로그를 직접 확인:

```
Found a contact between 'source_bin_wall_xn' (Object) and 'link2' (Robot link)
Start state appears to be in collision with respect to group mainpulation
```

같은 로그 파일에서 141번 전부 `source_bin_wall_xn` ↔ `link2` 조합으로만
발생 — 매우 일관된 패턴. §7.17에서 "팔이 벽을 스친다"는 문제를 완화하려고
bin 벽에 바깥쪽으로만 2cm 안전 여유를 추가했었는데, 이게 **정반대 효과**를
낸 것으로 확인됨: 물체를 집으러 bin 안쪽으로 팔을 뻗는 동작 자체가
`link2`(팔꿈치)가 `source_bin_wall_xn` 벽 근처를 반드시 지나가야 하는
구조라서, 원래는 (빠듯하지만) 유효했던 자세가 벽을 2cm 부풀리자 아예
"start state가 collision 상태"로 바뀌어버림 — 이건 이동 중 스치는 것보다
훨씬 나쁜 상태로, planning 자체가 막힘(MoveIt이 `fix_start_state_collision`
으로 억지로 빠져나오려 시도하지만 매번 성공한다는 보장이 없음).

즉 "벽에 여유를 더 준다"는 접근 자체가, 애초에 팔이 벽에 바짝 붙어서
지나가야만 하는 동작(bin 안으로 손을 뻗는 것)과 근본적으로 상충함 —
장애물을 부풀리는 것보다는 접근 경로 자체를 제어하는 방향이 맞았을 것으로
보임.

**수정**: `WALL_SAFETY_MARGIN`을 0.02 → 0.0으로 원복 (원래의, 마진 없는
벽 크기로 복귀). `colcon build` 통과.

**검증 수준**: move_group 로그로 직접 확인된, 재현 가능한 회귀였음
(추측이 아니라 실측 로그 기반 진단). 원복 자체는 §7.17 이전 상태로 되돌리는
것이므로 그 이전까지의 (이미 알려진) 동작으로 돌아갈 것으로 기대되지만,
"링크가 벽에 끼이는" 현상이 실제로 사라졌는지는 재시작 후 재확인 필요.

### 7.22 §7.20이 드러낸 진짜 도달 범위 한계 — source_bin이 로봇에 너무 가까움, bin 위치 이동으로 해결 (2026-07-23)

§7.21 원복 후에도 사용자 보고: "근처에 가서 내려가야 하는데 뭔가 막힌 듯
내려가지 않고, 다시 올라가고 반복 → 결국 실패로 체크됨." 최근 12개
트라이얼을 전수 조사(`rb5_binpicking_trials.jsonl`):

```
429595dd  x=0.297 y=0.002  GRASP 실패 (POSE_TIMEOUT)
40c56da9  x=0.301 y=0.093  GRASP 실패 (PLANNING_FAILED)
269528c1  x=0.307 y=0.116  GRASP 실패 (PLANNING_FAILED)
b1cdf27c  x=0.297 y=0.002  VERIFY_PLACE 실패
24e6443d  x=0.317 y=-0.033 GRASP 거부 (tilt)
982b5bcc  x=0.317 y=-0.033 GRASP 거부 (tilt)
5bb8693e  x=0.297 y=0.002  WATCH 단계 실패
818fba88  x=0.297 y=0.002  GRASP 실패 (PLANNING_FAILED)
424fa077  x=0.322 y=0.051  GRASP 실패 (PLANNING_FAILED)
3da9c1a5  x=0.297 y=0.002  GRASP 실패 (POSE_TIMEOUT)
1f641d07  x=0.297 y=0.002  GRASP 실패 (PLANNING_FAILED)
8e211a5c  x=0.297 y=0.002  GRASP 실패 (PLANNING_FAILED)
```

12개 전부 실패, 전부 x∈[0.297, 0.322] — source_bin의 "가까운 쪽"(로봇
베이스에 가까운) 벽 바로 근처(당시 벽은 x=0.24, 겨우 5~8cm 안쪽). move_group
로그도 이 구간에서 `source_bin_wall_xn` ↔ `link2`/`link4` 충돌과
`Unable to sample any valid states for goal tree`(그 근방에 유효한 자세가
하나도 없음)를 반복.

**원인**: §7.20에서 `grasp_offset_z`를 실측대로 0.16→0.108로 낮춘 게
정확한 수정이었지만, 그 결과 팔이 실제로 벽 높이(0.22m)까지 내려가야 하는
상황이 됨. 로봇 베이스에 가까운 물체를 집으려면 팔꿈치가 접힌(folded)
자세가 되는데, 하필 그 자세가 가까운 쪽 벽과 거의 같은 공간을 차지함 —
이전엔 grasp 높이가 5cm 높아서 이 문제를 우연히 피해가고 있었을 뿐,
원래부터 있던 도달 범위 한계였음.

**수정**: 코드가 아니라 **씬 레이아웃 변경** — `bin_geometry.yaml`의
`source_bin.center.x`를 0.45 → 0.51로, bin 전체를 로봇에서 6cm 더 멀리
배치. 이러면 가까운 쪽 벽에 붙어 있는 물체라도 팔이 덜 접힌 자세로 접근할
여유가 생김. (다른 옵션: grasp 시도 전에 "가까운 벽 근처는 애초에 거부"도
고려했으나, 그러면 그 구역의 물체는 영영 못 집게 되므로 사용자가 이 옵션을
선택함.) `colcon build` 통과, install된 yaml에 반영 확인.

**검증 수준**: 실패 위치 데이터(12/12 근접 벽 클러스터) + move_group 로그
기반 원인 진단은 신뢰도 높음. bin 위치 이동이 실제로 문제를 해결하는지는
**Isaac Sim 재시작 필요**(물리적 bin 위치가 USD 스테이지에 있으므로) —
재시작 후 같은 근처 벽 구역에서 다시 grasp 실패가 나는지 재확인 필요.

### 7.23 §7.22도 효과 없었음 — 진짜 원인은 물체 스폰 위치가 고정 시드로 항상 벽에 붙어 있었던 것 (2026-07-23)

§7.22 적용(bin을 6cm 멀리 이동) 후에도 사용자 보고: "지금도 그냥 밑으로
내려가면 집을 수 있는데 장애물에 걸리지도 않아. 근데 왜 안 내려가지?
RViz에서는 내려가는 모션이 발생하는데 실제로는 안 집혀." (Isaac Sim
화면에는 장애물이 안 보이는데 안 내려간다는 것 — 실제로는 MoveIt 전용
충돌 박스라 Isaac Sim 화면에 안 보일 뿐, 여전히 벽 충돌임을 로그로 확인.)

실행 중이던 move_group 로그 재확인: 여전히 `source_bin_wall_xn`과의
충돌(이번엔 link4/손목) + `Unable to sample any valid states for goal tree`.
bin을 옮겼는데도 왜 똑같은지 추적: 트라이얼 로그 79개 전체를 봤더니
**성공이 단 한 건도 없었음** — 즉 이 세션 내내 싸워온 물체(Cube0, 추적
대상)가 애초부터 벽에 너무 가깝게 스폰되고 있었던 것.

`binpicking_scene.py`의 스폰 로직 확인: `random.seed(7)`로 난수 시드가
고정돼 있고, `_rpos()`가 `BIN_X + random.uniform(-ix, ix)`로 위치를 정함.
같은 시드로 실제 계산해보니 Cube0의 오프셋은 `ix`(허용 범위)의 **88%
지점**(거의 끝) — bin을 통째로 옮기면 `BIN_X`와 이 오프셋이 같이
이동하므로, **벽으로부터의 거리는 전혀 안 바뀜** (§7.22가 효과 없었던
정확한 이유). 벽 여유 buffer가 0.03m이라 실제 벽까지 거리 ~5cm였음.

**수정**: `_rpos()`의 여유 buffer를 0.03 → 0.10으로 올림. 같은 시드
시퀀스로 재계산해서 Cube0가 근처 벽에서 ~11cm 떨어지도록 확인(다른
물체들도 전부 10.8~23cm 여유, 검증 완료). 시드를 바꾸는 대신 buffer를
올린 이유: 시드를 바꾸면 "운 좋은 값"을 고르는 것뿐이라 재현성/원칙이
없고, buffer는 전체 스폰 영역 자체를 벽에서 밀어내므로 Cube0뿐 아니라
다른 모든 물체에도 일반적으로 적용됨.

**검증 수준**: 문법 체크 완료 + 같은 seed(7)로 오프셋 재계산해서 새
buffer에서의 벽-거리 수치까지 확인(계산 신뢰도 높음). **Isaac Sim 전용
스크립트라 재시작 필요** — 재시작하면 물체들이 새 위치에 스폰되므로,
grasp가 실제로 성공하는지 재확인 필요. 지금까지 이 세션의 트라이얼 로그
79개 전부가 이 스폰 문제 하나로 인해 실패했을 가능성이 높음 — 다른
수정들(마찰, grasp_offset_z, gripper 등)이 실제로 충분한지는 이 스폰
수정이 반영된 뒤에야 제대로 검증 가능.

### 7.24 §7.23도 부족 — bin 벽 자체가 구조적 문제였음, table(벽 거의 없음)으로 전환 (2026-07-24)

§7.23 이후에도 사용자 보고: "충돌이라기보다는, 애초에 내려가서 잡지를
않는 게 문제야." move_group 로그로 확인: `Computed path is not valid.
Invalid states at index locations: [11 12 13 14] out of 15` — 15개 중
마지막 4개, 즉 경로의 끝(최종 자세) 자체가 무효라서 MoveIt이 실행을
아예 시작조차 안 함. 정확히 사용자 관찰과 일치.

**결정적 진단**: `/compute_ik`와 `/check_state_validity`를 직접 호출해서
확인. 물체가 벽에서 12cm 떨어진 상태에서도, 그 자리를 잡기 위한 IK 해
하나를 순수 기구학적으로(충돌 무시) 구해서 충돌 검사를 돌려보니:

```
robotiq_85_left_finger_tip_link  <-> source_bin_floor,     depth=0.094 (9.4cm)
robotiq_85_right_finger_tip_link <-> source_bin_floor,     depth=0.094
link3 <-> source_bin_wall_xn, depth=0.0034
link4 <-> source_bin_wall_xn, depth=0.0019
```

그리퍼 손끝이 바닥을 9.4cm나 뚫는 해가 나옴 — 위치를 아무리 조정해도
피할 수 없는, 자세 자체의 구조적 문제라는 뜻. 원인: bin 벽 높이(22cm)가
물체 크기(4.2cm)에 비해 지나치게 커서, top-down 접근으로 벽 높이만큼
깊이 파고들어야 하는 게 애초에 이 로봇의 팔 형상(팔꿈치/손목/그리퍼)과
근본적으로 안 맞음. §7.17/7.21/7.22/7.23에서 계속 위치만 바꿔가며
같은 벽에 부딪힌 이유가 이것 — 위치 문제가 아니라 벽 자체의 문제였음.

**사용자와 논의 후 결정**: bin 벽을 낮추는 것과 완전히 벽 없는 open
table로 바꾸는 것 중, **완전히 벽 없는 open table로 전환**하기로 결정
(79회 트라이얼 중 성공 0회였던 걸 감안하면, bin picking 시나리오 자체를
잠시 미루고 기본 pick-place 파이프라인부터 검증하는 게 우선이라고 판단).

**수정**: `bin_geometry.yaml`의 `source_bin`/`destination_bin`
`inner_size` height를 각각 0.22→0.01, 0.17→0.01로 낮춤 (사실상 벽 없는
평평한 table, 물체가 미끄러져 나가지 않을 정도의 아주 작은 턱만 남김).
`center`/`wall_thickness`/x,y 크기는 그대로 유지. 이 값들은 YAML 하나로
MoveIt 충돌 지오메트리(`bin_geometry.py`)와 Isaac Sim 물리 벽
(`binpicking_scene.py`) 양쪽에 다 반영됨.

**부수적으로 발견한 버그**: `moveit_pick_place.py`의 `_verify_placement()`
가 "물체가 그럴듯한 높이에 놓였는지" 판정할 때 `dest_bin.inner_size`의
height(기존 0.17)를 상한선으로 썼는데, 이제 height가 0.01이라 **물체
자체 높이(4.2cm)보다 낮은 상한**이 되어 정상적으로 놓인 물체도 무조건
FAILED로 판정될 뻔함. `object_half_height` 기반(`floor_z + 3×half_height`)
으로 수정. `_place_targets()`의 `transit_z`(접근 높이)는 원래부터
`height`에서 동적으로 유도되던 값이라 자동으로 낮아짐 — 오히려 의도에
맞게 개선됨(벽이 낮아졌으니 그 위로 넘어갈 필요도 없어짐).

**검증 수준**: 원인 진단은 `/compute_ik`+`/check_state_validity` 직접
호출로 확인된 실측 데이터 기반(신뢰도 높음). 코드 수정은 문법 체크 +
`colcon build` 통과, YAML 파싱 확인 완료. **Isaac Sim 재시작 필요**
(물리적 벽 크기가 USD 스테이지에 있음) — 재시작 후 실제로 grasp가
성공하는지가 이 세션 79회 연속 실패를 깨는 첫 번째 실질적 검증이 될 것.

### 7.25 §7.24 이후에도 여전히 안 내려감 — grasp_offset_z가 또 틀렸음 (fingertip 메쉬가 관절 원점보다 훨씬 김), 실측 이진탐색으로 재보정 (2026-07-24)

§7.24(open table 전환) 반영 후에도 사용자 보고: "충돌이라기보다는, 애초에
내려가서 잡지를 않는 게 문제야." 벽은 이제 거의 안 걸림(1회만 등장)을
로그로 확인했지만, **새로운 지배적 패턴** 발견:

```
source_bin_floor <-> robotiq_85_left_finger_tip_link   56회, depth 최대 9.4cm
source_bin_floor <-> robotiq_85_right_finger_tip_link   (동일)
link3 <-> robotiq_85_right_finger_tip_link/finger_link  (자체충돌 다수)
```

move_group이 켜져 있는 상태에서 `/compute_ik`, `/check_state_validity`를
직접 호출해 확인 — 서로 다른 IK 해(팔꿈치 방향이 전혀 다른 여러 branch)
전부에서 **손끝이 바닥을 9.4cm 뚫는 동일한 침투 깊이**가 재현됨. IK
branch가 달라도 같은 깊이가 나온다는 건, 팔 형상 문제가 아니라 TCP
목표 자체와 그리퍼 형상 사이의 고정된 기하 오차라는 뜻.

**진짜 원인**: §7.20에서 `grasp_offset_z`를 계산할 때
`robotiq_85_left_finger_tip_link`의 **관절 원점(origin)만** 순운동학으로
추적했음 (라이브 TF로 원점 위치까지는 정확히 검증했었음). 그런데
`left_finger_tip.stl` 충돌 메쉬 파일을 직접 열어서 정점(vertex) 범위를
읽어보니, 메쉬가 그 원점 기준 로컬 z 방향으로 **최대 5.1cm 더 뻗어나가
있었음** — 즉 실제 손끝 패드는 원점보다 5cm 넘게 더 아래까지 닿는데,
원점만 계산해서 그만큼 못 미치는 높이로 grasp_offset_z를 잡았던 것.

**수정 방법**: 손으로 메쉬 오프셋까지 다시 계산하는 대신(원점 계산도
이미 한 번 실수했으므로), **`/check_state_validity`로 직접 이진 탐색** —
TCP z를 0.136부터 조금씩 올려가며 실제 충돌 검사 결과(침투 깊이)를
관찰. z=0.136→0.158 구간에서 침투 깊이가 9.4cm→1.55cm→...→0으로
거의 선형으로 줄어드는 걸 확인, z=0.158에서 침투 0(바닥에 딱 닿음).
여기에 물체 높이(object_half_height=0.021m) 보정 + 5mm 여유를 더해
`grasp_offset_z = 0.135`로 재설정. 재검증: TCP z=0.163(=0.028+0.135)에서
`valid=True, floor_penetration=0.0` 확인.

**수정**: `moveit_pick_place.py`의 `grasp_offset_z` 기본값 0.108 → 0.135.

**검증 수준**: 이번엔 손 계산이 아니라 **실행 중인 move_group에 대한
실측 이진 탐색**으로 얻은 값이라 이전 두 번(0.16, 0.108)보다 신뢰도가
훨씬 높음 — 그래도 이 파라미터가 두 번 연속 손으로 계산했다가 틀린
전례가 있으므로, README2.md 코드 주석에도 "다음에 또 틀리면 반드시
`/check_state_validity` 실측으로 재검증할 것"이라고 명시해둠. `colcon
build` 통과. **moveit_pick_place.py만 재시작하면 반영됨** (Isaac Sim
재시작 불필요 — 이건 순수 MoveIt 목표 높이 계산이라 물리 지오메트리와
무관). 재시작 후 실제로 물체 옆면을 잡고 들어올리는지 확인 필요.

### 7.26/7.27 pick-and-place 최초 성공 — 남은 문제 2건: 가끔 바닥 박음 + 그리퍼 mesh가 물체를 밀어냄 (2026-07-25)

사용자 확인: **pick and place가 이제 작동함** — §7.17부터 이어진 벽/바닥/
grasp 높이 문제들이 정리됨. 남은 문제 2건 보고:

**(§7.26) 가끔 바닥에 쳐박음.** 원인: §7.25에서 찾은 "바닥에 딱 닿는"
정확한 높이(grasp_offset_z=0.130)에서 여유를 5mm(0.135)만 뒀는데, 이건
재관측된 물체 pose의 흔들림/실행 오차/TF 타이밍 같은 정상적인 시스템
노이즈만으로도 쉽게 잡아먹히는 수준이었음. **수정**: 여유를 5mm→15mm로
확대(`grasp_offset_z`: 0.135 → 0.145). fingertip pad 길이가 5.7cm(§7.25
에서 메쉬로 실측)나 되므로, 이 정도 여유를 더 둬도 물체 옆면 안에 충분히
들어감.

**(§7.27) 그리퍼가 닫힐 때 안쪽 mesh가 가운데로 튀어나와 물체를 밈.**
§7.18/7.19에서 진단했던 것과 같은 근본 원인일 가능성이 높음: Robotiq
2F-85는 실제 구동축이 `left_knuckle_joint` 하나뿐이고 나머지 5개는 4-bar
링크로 종속되어야 하는데, 이 URDF는 닫힌 링크 구조를 표현 못 해 6개
관절을 각자 독립된 힘으로 구동 중. 접촉 시 각 관절이 서로 다른 지점에서
멈추면서 pad가 완전히 평행을 유지하지 못하고, 한쪽 모서리가 안쪽으로
먼저 튀어나와 물체를 밀어내는 것으로 보임. §7.18에서 PhysX mimic joint로
"제대로" 고치려다 팔 전체가 불안정해져서 원복했고, §7.19에서 수동 관절
5개의 힘만 낮추는 안전한 완화책을 적용했었는데, 그것만으론 부족했음.

**사용자와 논의 후 결정**: mimic joint 재시도(위험) 대신, **안전한
점진적 조정**을 한 번 더 강화하기로 함:
- `binpicking_scene.py`의 `GRIPPER_FOLLOWER_JOINTS` 힘을 한 번 더 낮춤
  (stiffness 1200→600, damping 150→120, maxForce 3e4→1.5e4) — 접촉 저항을
  만나면 더 쉽게 순응하도록.
- `trajectory_bridge.py`의 `gripper_close_duration`을 1.2s→2.0s로 늘림 —
  닫는 속도 자체를 늦춰서 접촉 시 모멘텀으로 인한 오버슈트/기울어짐을
  줄임.

**검증 수준**: 바닥 마진 확대는 §7.25와 동일한 실측 이진탐색 논리의
연장이라 신뢰도 높음(문법 체크 + `colcon build` 통과). 그리퍼 mesh 밀림
완화는 §7.19와 같은 종류의 점진적/안전한 조정이라 팔 안정성에 위험은
없지만, 문제를 완전히 없애는지는 **Isaac Sim + moveit_pick_place 재시작
후** 실제로 여러 번 pick 해보며 확인 필요. 여전히 밀어낸다면 mimic
joint를 다시 시도하되 이번엔 관절 하나씩 순차 적용 + 즉시 테스트하는
방식을 사용자가 원하는지 물어볼 것.

### 7.28 §7.27 완화가 부작용 — inner_knuckle이 처져서 "연결부가 빠진" 것처럼 보임 (2026-07-25)

§7.27 반영 후 사용자가 스크린샷으로 보고: 그리퍼 중앙 부분의 연결부가
빠진 것처럼 보인다는 것. URDF 확인: `robotiq_85_left/right_inner_knuckle_link`
는 **다른 어떤 joint의 parent로도 등장하지 않는 막다른 링크** — 실제
하드웨어에서는 핀으로 finger_tip에 연결되어 닫힌 4-bar 루프를 이루지만,
이 URDF엔 그 연결 joint 자체가 없음. 즉 이 링크가 손가락 조립부와
정렬된 것처럼 보이는 건 오직 자기 자신의 PD 드라이브가 정확한 각도를
정밀하게 유지해주기 때문이고, 다른 어떤 물리적 힘도 그 정렬을 붙잡아주지
않음.

§7.27에서 그리퍼가 물체를 밀어내는 문제를 완화하려고 수동 관절 5개
전부(오른쪽 knuckle + 안쪽 knuckle 2개 + fingertip 2개)의 힘을
600/120/1.5e4로 낮췄는데, inner_knuckle은 애초에 물체와 절대 닿지 않는
링크라 접촉에 순응할 필요가 없었고 — 오히려 그 힘만으로 버티던 것이
약해지면서 처지고, 손가락 조립부에서 시각적으로 떨어져 나가 "연결부가
빠진" 것처럼 보이게 됨.

**수정**: `GRIPPER_FOLLOWER_JOINTS`를 두 그룹으로 분리 —
`GRIPPER_SOFT_FOLLOWER_JOINTS`(fingertip 2개, 실제로 물체에 닿는 것만)는
600/120/1.5e4 유지, 나머지(오른쪽 knuckle + 안쪽 knuckle 2개)는 원래대로
primary와 동일한 7000/350/1e6로 되돌림 — 어차피 아무것도 접촉하지 않으니
순응할 이유가 없고, 강하게 유지해야 제자리를 정확히 지킴.

**검증 수준**: 원인 진단(URDF의 dead-end 링크 구조)은 코드로 직접
확인됨 — 신뢰도 높음. 문법 체크 완료. **Isaac Sim 재시작 필요** —
재시작 후 연결부가 빠진 것처럼 보이는 현상이 사라졌는지, 그리고 §7.27의
원래 목적(그리퍼가 물체를 밀어내는 문제)이 fingertip만 부드럽게 해도
여전히 완화되는지 둘 다 확인 필요.

## 8. IsaacLab 기반 RL 파이프라인 추가 + 죽은 코드 정리 (2026-07-27)

기존 MoveIt2 휴리스틱 파이프라인(§7 전체)과는 별개로, **IsaacLab 기반
강화학습(PPO 메인, SAC 소규모 비교)** pick-and-place를 새로 만듦. 새
패키지는 `Manipulator/rb5_isaaclab/` — ROS colcon 패키지가 아니라 순수
Python 패키지(`isaaclab` conda 환경에 `pip install -e`)이며, 기존
`rb5_binpicking`/`rb5_isaac`는 전혀 건드리지 않음. 상세 설계/실행법은
`rb5_isaaclab/README.md` 참고, 여기서는 이 세션에서 실제로 부딪히고
고친 것들만 기록.

### 8.1 URDF에 `<mimic>` 태그 추가 — §7.18/7.19/7.27/7.28의 근본 해법

기존 ROS용 URDF(`rb5_isaac/urdf/rb5_with_tools.urdf`)에는 그리퍼 6관절에
`<mimic>` 태그가 하나도 없어서, 이번 세션 내내(§7.18~7.28) 수동 관절들이
물리적으로 안 이어진 문제와 씨름했었음. `rb5_isaaclab/assets/urdf/`에
**원본은 그대로 두고 사본만** 만들어 5개 수동 관절에 `<mimic>` 태그
추가(`trajectory_bridge.py`의 이미 검증된 `GRIPPER_MIMIC` 배열 값 재사용).
USD 변환 시 이 태그가 진짜 PhysX mimic-joint 제약으로 구워지므로, RL
쪽 그리퍼는 §7.19/7.27/7.28에서 했던 "수동 관절 힘 3단계 조정" 같은
보완이 아예 필요 없어짐 — ROS 쪽 라이브 씬보다 오히려 더 정확한 물리.

### 8.2 URDF 변환 중 발견한 Isaac Sim/IsaacLab 버그 3건 (전부 재현 확인 + 수정)

1. **IsaacLab의 `convert_urdf.py` CLI가 mimic 태그를 조용히 버림.**
   `UrdfConverterCfg.convert_mimic_joints_to_normal_joints`(기본 False)가
   내부적으로 `import_config.parse_mimic = (그 값)`으로 그대로 전달되는데,
   Isaac Sim 임포터 자체 테스트 코드(`test_urdf.py::test_urdf_parse_mimic`)
   로 직접 확인해보니 `parse_mimic=True`가 "mimic 보존", `False`가
   "mimic 무시"임 — 즉 IsaacLab 기본값이 실제로는 mimic을 전부 버리는
   방향으로 배선되어 있음(반대로 문서화된 것으로 보임). 우회책: CLI 대신
   `binpicking_scene.py`가 이미 쓰는 것과 같은 `omni.kit.commands` 직접
   호출 방식으로 변환하고 `parse_mimic=True` 명시
   (`rb5_isaaclab/scripts/convert_robot_to_usd.py`).
2. **`robotiq_85_left_inner_knuckle_joint`의 mimic `referenceJoint`
   관계가 비어있음.** gearing/offset은 정상 authored인데 참조 대상
   relationship만 비어있어서 제약이 무의미했음 — 2번 재현. 변환 스크립트에
   "USD 다시 열어서 5개 전부 확인 후 비어있으면 주 관절로 재타겟"하는
   복구+검증 pass 추가, 내보낸 파일을 다시 열어 재확인까지 함.
3. **같은 조인트에 joint limit 자체가 없음(`-inf`/`inf`).** PhysX가
   "mimic joint 기능을 쓰려면 revolute joint에 finite limit이 있어야
   한다"는 에러를 내며 시뮬레이션이 죽음. URDF의 원래 `<limit>` 값(라디안)
   을 degree로 변환해 누락된 경우 직접 authored — 같은 스크립트에서 처리.

세 버그 다 재현 가능했고, 수정 후 최종 변환 스크립트를 다시 돌려서
5/5 mimic joint가 (내보낸 파일을 새로 열어) 완전하게(gearing+offset+참조
전부) 존재함을 확인함.

### 8.3 그 외 실제로 막혔던 것들

- **PYTHONPATH 함정**: 이 머신의 셸 프로필이 `install/isaaclab*`(ROS
  colcon으로 빌드된, 이름만 같은 별개 패키지)를 PYTHONPATH에 넣어서
  `isaaclab` conda 환경의 진짜 IsaacLab을 가려버림 — ROS 쪽 dual-workspace
  문제와 같은 종류. `conda activate isaaclab && unset PYTHONPATH`로 해결,
  `rb5_isaaclab/README.md`에 명시.
- **Isaac Sim/Kit 프로세스가 표준출력을 삼킴**: headless로 리다이렉트한
  로그에 `print()` 출력이 하나도 안 남는 경우가 반복됨 — 결과를 별도
  파일에 `flush()+os.fsync()`로 직접 쓰는 방식으로 우회(검증 스크립트,
  `smoke_test.py`의 `--result_file` 옵션).
- **`UsdFileCfg`에 `physics_material_prim_path` 없음**: 그 필드는
  `UsdFileWithCompliantContactCfg` 서브클래스 전용이었음(오독) — fingertip
  고마찰 재질은 `binpicking_scene.py`와 같은 방식으로 변환 스크립트가
  USD에 직접 바인딩하도록 변경.
- **`GroundPlaneCfg`/디버그 마커 기본값이 Nucleus(원격 자산 서버) 의존**:
  이 머신에 Nucleus 연결이 없어서 최대 300초 서버 응답 대기 후 실패함
  — ground plane은 procedural `CuboidCfg`로, `UniformPoseCommandCfg`의
  `debug_vis`는 False로 바꿔서 완전히 오프라인으로 동작하게 함.
  (`--gui`로 Nucleus 있는 머신에서 디버그할 땐 다시 켜도 됨.)
- **`SceneEntityCfg(joint_names=...)`를 mdp 함수의 기본 인자 값으로만
  두면 조용히 무시됨**: IsaacLab 매니저가 `joint_names→joint_ids` 자동
  해석을 하는 건 `RewTerm`/`ObsTerm`/`DoneTerm`의 `params={...}` 딕셔너리에
  명시적으로 들어간 `SceneEntityCfg`뿐(`manager_base.py`의
  `_prepare_terms`가 `term_cfg.params`만 순회) — 함수 시그니처 기본값은
  절대 안 건드림. 처음엔 이걸 놓쳐서 `gripper_opening` 관측값과
  `place_and_release`/`object_placed`가 전부 12관절(필터 안 걸림)을
  반환하다가 텐서 shape 불일치로 크래시 — `pick_place_env_cfg.py`에서
  전부 `params`에 명시해서 해결.

전부 `scripts/smoke_test.py --num_envs 4 --headless`로 실제 실행해서
잡은 것들 — 코드만 보고 짐작한 게 아니라, 각 수정 후 매번 재실행해서
다음 에러로 넘어가는 식으로 반복 검증함. 최종적으로 스모크 테스트
통과: env 생성, reset, 랜덤 액션 20스텝 동안 NaN/Inf/크래시 없음 확인.
**PPO/SAC 실제 학습은 이 세션에서 돌리지 않음** — 학습이 되는지, 정책이
수렴하는지는 아직 완전 미검증.

### 8.4 기존 코드 정리 (죽은 파일 이동)

`rb5_binpicking`/`rb5_isaac`에서 아무 launch 파일/`setup.py` entry
point에서도 참조되지 않는 걸 확인한 파일들을 `Manipulator/deprecated/`
로 옮김(이 저장소의 `.git`이 실제로는 빈 디렉토리라 git 이력으로 복구가
안 되는 걸 뒤늦게 확인해서, 삭제 대신 이동으로 처리 — `deprecated/README.md`
에 사유 기록):

- `rb5_binpicking/lab_envs/`(§8의 `rb5_isaaclab/`로 대체)
- `rb5_binpicking/scripts/action_adapter.py`,
  `rb5_binpicking/config/rb5_action_space.yaml`(미배선 Phase-2 스캐폴딩)
- `rb5_isaac/scripts/01_convert_urdf_to_usd.py`,
  `rb5_isaac/scripts/02_isaac_rb5_scene.py`,
  `rb5_isaac/launch/moveit_isaac.launch.py`(`binpicking_scene.py`/
  `binpicking.launch.py`로 대체된 예전 버전)

`setup.py`에서 `action_adapter.py` 스크립트 엔트리 제거. 이동 후
`colcon build --packages-select rb5_binpicking rb5_isaac`로 재빌드
성공 확인 — 정상 동작 중인 `moveit_pick_place.py`/`trajectory_bridge.py`
등 나머지 코드는 전혀 건드리지 않음.

## 9. IsaacLab RL: ReachGrasp/Place 스테이지 + 트레이닝 안정화 (2026-08-03 ~ 08-05)

§8 이후 `rb5_isaaclab`에서 실제로 4-스테이지 커리큘럼(Reach/GraspLift/
Transport/Curriculum) 학습을 여러 차례 돌리면서 벌어진 일들. 세부 구현
기록(리워드 테이블, 파일 목록 등)은 `rb5_isaaclab/CURRICULUM_REPORT.md`
(~2026-07-29 시점까지)에 있고, 여기서는 그 이후 이 세션들에서 실제로
부딪히고 고친 것만 정리. **아직 `rb5_isaaclab/*.md` 쪽 문서는 갱신 안 됨**
— 다음에 손댈 때 이 절 내용을 반영할 것.

### 9.1 GraspLift → ReachGrasp: 그리퍼 닫기를 학습 대상에서 제거

GraspLift/Curriculum처럼 그리퍼 open/close를 RL이 직접 결정하게 하면,
`bilateral_fingertip_contact_reward`(희소/이진 보상)가 exploration이
붕괴되기 전에 한 번도 안 터지는 문제가 반복됐음(여러 라운드의 리워드
재조정으로도 근본 해결 안 됨 — `grasp_env_cfg.py`/`grasp_lift_env_cfg.py`
자체 docstring에 그 이력 있음). **구조적으로** 문제를 없애기로 함:
`mdp.AutoGraspActionTermCfg`(정책 액션 0-dim, `action_dim=0`)를 추가해서
그리퍼는 `grasp_state.is_near_pregrasp`(pre-grasp 타겟 근처) 이거나 이미
쥐고 있으면 자동으로 닫히게 스크립트로 처리. RL은 이제 6-dim 팔 액션만
제어하고, "정밀하게 접근하면 스크립트가 실제로 쥐어준다"는 실제 물리적
결과로만 평가받음 — grasp 타이밍을 따로 학습할 필요가 없어짐.
`reach_grasp_env_cfg.py`(`RB5-PickPlace-ReachGrasp-JointPos-v0`)로 신규
등록.

같은 시기에 `KLAdaptiveLR` 스케줄러가 학습률을 자기강화적으로 0에 가깝게
붕괴시키는 문제를 진단(policy std가 줄어들수록 같은 크기의 평균 변화가
더 큰 KL로 읽혀서 lr을 계속 깎는 악순환)하고, 스케줄러를 없애고
`learning_rate: 1.0e-4` 고정으로 교체 — 이 픽스 이후 첫 real grasp 달성
(`2026-08-03_20-33-44` run, bilateral_contact ~0.78까지 상승).

### 9.2 "떨림"/바닥 근처 낙하 조사 — 5개 변경 동시 적용 → 오리엔테이션 붕괴

위 체크포인트를 GUI로 재생해보니 팔이 눈에 띄게 떨리고, 바닥 근처에서
가끔 물체를 놓치는 현상 확인. 원인 후보 5개를 한 번에 적용해서 재학습
(damping 1000→2000, EMA 액션 스무딩, jerk penalty, floor-contact penalty,
PD gain 랜덤화) — 결과는 orientation reward가 거의 0으로 붕괴
(~51° 오차), `bilateral_contact`/`stable_grasp` 전 구간 0. 후속 수정
시도(관절군별 damping/EMA alpha 분리) → 학습이 발산(reward가 수천 스텝
만에 -1,000,000 이하로 폭주), 원인은 이전 세션에서 KLAdaptiveLR 제거할
때 같이 없어진 `agent.kl_threshold`(전혀 다른 메커니즘 — PPO
업데이트-내 KL 조기중단 안전장치)가 0(비활성)으로 남아있었던 것.
`kl_threshold=0.01`로 재시도 → 발산은 막았지만 이번엔 매 업데이트가
거의 즉시 조기중단되면서 학습이 통째로 멈춤(value loss가 0 근처에 고정).

세 번의 재학습이 전부 실패한 뒤, 하나씩 원인을 격리하지 않고 계속
쌓아올린 게 문제라고 판단 — known-good 베이스라인으로 되돌리고 변경
5개를 **하나씩** 따로 학습·비교하기로 함.

### 9.3 One-at-a-time ablation (2026-08-04, 각 100000 step)

| # | 변경 | 결과 |
|---|---|---|
| 1 | damping 1000→2000 단독 | 양호 — total reward 21.1 |
| 2 | EMA 액션 스무딩 단독 | **확인된 주범** — orientation reward 0.0098로 붕괴, contact/stable_grasp 전 구간 0 (wrist 관절 응답성 저하 가설과 일치) |
| 3 | jerk penalty 단독 | **문제 있음** — position reward가 랜덤 액션 기준선(0.27)보다 낮은 0.18, total reward 1.4(시리즈 중 최악) — orientation뿐 아니라 움직임 자체를 과도하게 억제 |
| 4 | floor-contact penalty 단독 | **문제 있음** — step ~29000에서 급붕괴(바닥 충돌 추정), 이후 부분 회복해도 total reward 3.4~4.8 그침, 실제 grasp 성공 없음 |
| 5 | PD gain 랜덤화 단독 (stiffness×0.9~1.1, damping×0.8~1.2, `mode="startup"`) | **세션 최고** — total reward 33.1(peak 50.7), bilateral_contact peak 1.70, stable_grasp peak 3.34 |

EMA/jerk penalty/floor-contact penalty 세 개는 폐기. damping 증가와 PD
랜덤화 두 개만 개별적으로 검증된 상태로 남음 — GUI로 PD 랜덤화 단독
체크포인트를 직접 재생해서 확인("잘 되네").

### 9.4 "검증된 두 개를 합치면 더 좋겠지" → 오히려 더 나빴음 (2026-08-05)

damping=2000 + PD 랜덤화를 합쳐서 150000 step으로 재학습
(`2026-08-05_13-37-31`). 결과: total reward 6.5로 오히려 **개별 결과
둘 다보다 나쁨** — orientation reward가 step 54000까지 0.35로 오르다가
0.001까지 재붕괴, bilateral_contact/stable_grasp 학습 내내 거의 0.
§9.2의 EMA 단독 실패와 같은 "orientation 붕괴 → grasp 실패" 패턴이
전혀 다른 원인으로 재현된 셈.

**원인 분석**: `randomize_actuator_gains`는 `operation="scale"`로
곱연산 — base damping이 1000일 때(#5)는 각 env가 800~1200 범위 중
하나로 고정 배정되지만, base가 2000이면 같은 배율이 1600~2400 범위로
그대로 밀려 올라감. 정책은 자신이 배정된 env의 게인 값을 관측하지
못하므로(observation에 없음), 8192개 env 전체에 걸쳐 통하는 **하나의**
제어 전략을 찾아야 함. #1(damping=2000 고정, 변동 없음)은 "모든 env가
똑같이 뻑뻑한" 상황이라 그 하나의 동역학에 특화하면 됐지만, 이번엔
"env마다 뻑뻑한 정도가 제각각인" 상황 + 그 range 자체가 상향된 것이
겹쳐서, damping에 민감한 것으로 이미 확인된(§9.3 EMA 실험) wrist
orientation 제어가 range 전체에서 robust하게 안 되고, 정책이 "range
전반에서 안전한" position tracking만 남기고 orientation을 포기하는
쪽으로 수렴한 것으로 추정.

### 9.5 최종 확정: PD 랜덤화 단독, damping 원복 (2026-08-05)

`robots/rb5_850e.py`의 `ARM_DAMPING`을 2000→1000(원래 값)으로 복원,
`reach_grasp_env_cfg.py`는 `USE_PD_GAIN_RANDOMIZATION=True`만 유지한
채로 150000 step 재학습(`2026-08-05_16-28-09`). 결과: **total reward
43.98(peak 50.08)**, bilateral_contact 1.59, stable_grasp 2.83,
orientation reward 0.25에서 안정(붕괴 없음), object_drop_penalty 0 —
세션 전체 최고 기록. 이걸 ReachGrasp 스테이지의 최종 설정으로 확정.

### 9.6 Place 스테이지 신규 구현 — 그리퍼 열기도 스크립트로

§9.1의 "그리퍼 닫기를 스크립트로 분리"와 대칭되는 구조를 놓는 쪽(release)
에도 적용. `mdp.AutoReleaseActionTermCfg`(역시 `action_dim=0`) 추가 —
물체가 목적지 안착 조건(`grasp_state.is_at_place_target`: destination
bin 실제 footprint 안 + 바닥 근처 + 정지, `full_place_success_condition`에서
그리퍼 상태만 뺀 버전 — 그리퍼 열지 말지를 결정하는 조건 안에 그리퍼
상태를 넣으면 순환참조가 됨)을 만족하면 자동으로 열리고, 이후 계속
열린 채 유지(sticky, 살짝 흔들려도 다시 쥐지 않도록).

`place_env_cfg.py`(`RB5-PickPlace-Place-JointPos-v0`)는 Transport
스테이지를 상속 — 이미 검증된 "물체를 쥔 채로 시작"(`reset_robot_holding_object`)
리셋을 그대로 재사용하고, 목표 지점만 Transport의 hover 지점(목적지
바닥 위 12cm)에서 실제 착지 높이로 낮춤. 실수로 놓치는 것(`premature_drop_penalty`,
신규)과 스크립트가 의도적으로 놓는 것을 구분해서 정상적인 place가
페널티를 받지 않게 처리. 성공 보상/종료 조건은 이미 Stage-4 Curriculum에
있던 `released_and_stable_reward`/`full_place_success`를 그대로 재사용
(새 물리적 성공 기준을 새로 만들지 않음). smoke test 통과 후 §9.5의
최종 ReachGrasp 설정(damping=1000, PD 랜덤화)을 공유 로봇 설정으로 물려받은
채 150000 step 학습 시작 — 결과는 아직 미확인(다음 세션에서 이어서 볼 것).

### 9.7 남은 일 / 알려진 미해결 사항

- Place 스테이지 학습 결과 아직 미확인.
- GraspLift/Transport/Curriculum(Stage 4) 체크포인트는 전부 §9.2~9.5의
  damping 변경(1000→2000→1000) 이전에 학습된 것이라, 최신 `ARM_DAMPING`
  기준으로는 재검증 안 됨 — 재생하거나 재사용하기 전에 재학습 필요할 수
  있음.
- `rb5_isaaclab/CURRICULUM_REPORT.md`/`PICK_PLACE_COMPLETION_REPORT.md`는
  §9 내용 반영 안 된 채로 2026-07-29 시점에 멈춰 있음.
- 사용자 요청으로 예정된 작업: `Manipulator/RB5/` 폴더를 만들어
  `rb5_binpicking`/`rb5_isaac`/`rb5_isaaclab`을 그 안으로 이동(현재
  세션의 IsaacLab 학습이 완전히 정리된 후 진행하기로 함, 아직 미착수).
