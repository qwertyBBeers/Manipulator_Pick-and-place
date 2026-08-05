# Manipulator — RB5-850E Pick-and-Place

Rainbow Robotics RB5-850E + Robotiq 2F-85 그리퍼로 pick-and-place를
구현하는 저장소. 서로 독립적인 세 갈래로 진행 중:

1. **휴리스틱 파이프라인** (`rb5_binpicking`/`rb5_isaac`, ROS2/MoveIt2) — 완료, 동작함.
2. **IsaacLab 강화학습** (`rb5_isaaclab`, 순수 Python/PyTorch) — 현재 가장 활발히 개발 중.
3. **VLA 계열 실험** (GR00T 로드맵 / `lerobot_ws` SmolVLA) — 탐색 단계.

각 갈래는 서로의 코드를 건드리지 않는다. 자세한 설계 이력/버그
진단은 `README2.md`(작업 로그, § 단위로 누적) 참고.

## 저장소 구조

```
rbpodo_ros2/     Rainbow Robotics 벤더 스택 (description/hardware/bringup/moveit_config)
                 실물 로봇 bringup 담당. 시스템 ROS 2 Humble(/opt/ros/humble)로 빌드됨.
rb5_isaac/       Isaac Sim 연동 + trajectory_bridge.py (MoveIt 궤적 → Isaac Sim 조인트 커맨드)
rb5_binpicking/  RB5-850E bin-picking 데모 (Isaac Sim + MoveIt2 휴리스틱 상태머신). Phase 1 실구현체.
rb5_isaaclab/    IsaacLab 기반 PPO 강화학습 (ROS 아님, isaaclab conda env + pip install -e).
                 상세 실행법: rb5_isaaclab/README.md
                 구현 이력: rb5_isaaclab/CURRICULUM_REPORT.md, PICK_PLACE_COMPLETION_REPORT.md
lerobot_ws/      LeRobot + SmolVLA 실험용 별도 워크스페이스 (자체 git, python 3.12 필요).
                 상세: lerobot_ws/README.md
deprecated/      더 이상 배선 안 된 죽은 코드 (삭제 대신 이동 — 사유는 deprecated/README.md)
README2.md       작업 로그 (버그 진단/의사결정 이력, § 번호로 누적, 날짜별)
```

`build/`, `install/`, `log/`는 colcon 산출물(gitignore 대상).

---

## Track 1 — 휴리스틱 Pick-and-Place (완료)

Isaac Sim 안에서 스크립트 상태머신으로 pick → place 수행. 실물 로봇
bringup도 `rbpodo_ros2`로 별도 지원.

**핵심 파일**
- `rb5_binpicking/scripts/binpicking_scene.py` — Isaac Sim 씬(로봇/그리퍼/카메라/bin/오브젝트), `/binpicking/object_pose` 퍼블리시
- `rb5_binpicking/scripts/moveit_pick_place.py` — watch→pre-grasp→grasp→lift→pre-place→place→retreat 상태머신 (MoveIt `/move_action`)
- `rb5_isaac/rb5_isaac/trajectory_bridge.py` — MoveIt 궤적 → Isaac Sim 조인트 커맨드 변환 브릿지

**실행**
```bash
~/isaacsim/python.sh rb5_binpicking/scripts/binpicking_scene.py
ros2 launch rb5_binpicking binpicking.launch.py
ros2 run rb5_binpicking moveit_pick_place.py
```

세부 버그 이력: `README2.md` §7 (Phase 1 신뢰성 대개편).

---

## Track 2 — IsaacLab 강화학습 (진행 중)

`rb5_isaaclab/`에 Track 1과 완전히 독립된 PPO 커리큘럼 학습 파이프라인.
같은 로봇/그리퍼 USD를 쓰지만 물리 시뮬레이션도, 리워드도, 정책도 전부
IsaacLab(skrl PPO) 기준으로 새로 설계.

**커리큘럼 스테이지** (각각 독립적으로 학습/재생 가능, `RB5-PickPlace-<Stage>-JointPos-v0`):

| 스테이지 | 목표 | 그리퍼 제어 |
|---|---|---|
| Reach | pre-grasp 타겟까지 접근 | 없음 (팔만) |
| ReachGrasp | 접근 + 실제 grasp | **스크립트** (근접 시 자동으로 닫힘) |
| Transport | 이미 쥔 물체를 목적지 상공까지 운반 | 학습 (계속 쥐고 있기) |
| Place | 이미 쥔 물체를 목적지에 내려놓기 | **스크립트** (안착 조건 만족 시 자동으로 열림) |
| GraspLift / Curriculum | 초기 4-스테이지 설계의 잔존 버전 (grasp/전체를 RL이 직접 결정) | 학습 |

ReachGrasp/Place는 "언제 쥘지/놓을지"를 정책이 직접 탐색하게 하는 대신
env 상태 기반으로 스크립트 처리 — grasp/release 타이밍 탐색 문제를
구조적으로 제거한 최신 설계(GraspLift/Curriculum의 학습된 그리퍼 제어가
반복적으로 겪은 문제의 해결책). 자세한 경위는 `README2.md` §9 참고.

**현재 최고 성능 (2026-08-05 기준, ReachGrasp)**: total reward 43.98
(peak 50.08), bilateral contact / stable grasp 안정적으로 발생,
orientation 붕괴 없음. `robots/rb5_850e.py`의 `ARM_DAMPING=1000` +
PD gain 랜덤화(`randomize_actuator_gains`, startup-only)가 검증된 최종
설정 — damping을 올리는 변경은 단독으로는 괜찮았지만 PD 랜덤화와
합치면 오히려 더 나빠짐(§9.4)이 확인된 것도 기록해 둠.

**실행법**: `rb5_isaaclab/README.md` 참고 (USD 변환 → smoke test → 학습
→ TensorBoard → 재생, 전부 명령어 포함).

```bash
conda activate isaaclab && unset PYTHONPATH
cd <IsaacLab repo>
./isaaclab.sh -p <path>/rb5_isaaclab/scripts/train.py \
  --task RB5-PickPlace-ReachGrasp-JointPos-v0 --num_envs 8192 --headless \
  --agent skrl_ppo_cfg_entry_point
```

---

## Track 3 — VLA 계열 실험 (탐색 단계, 미착수/초기)

### 3a. GR00T 로드맵

`moveit_pick_place.py`의 하드코딩 상태머신을 NVIDIA Isaac GR00T 정책으로
교체하고, 이후 DUNE 기반 spatial encoder로 확장하는 3단계 계획. **아직
착수 전** — Track 2(IsaacLab RL)를 먼저 진행 중이라 우선순위가 밀려있음.
계획 상세는 이 파일 하단의 "GR00T/DUNE 로드맵" 절 참고.

### 3b. LeRobot + SmolVLA

`lerobot_ws/`에 완전히 분리된 워크스페이스로 SmolVLA(450M) fine-tuning
탐색 중. `lerobot` 요구 Python 3.12+라 기존 conda 환경(전부 3.10~3.11)과
호환 안 됨 — 별도 env 필요. 현재는 RB5용 커스텀 LeRobot `Robot` 플러그인
설계 분석 단계. 상세: `lerobot_ws/README.md`.

---

## GR00T/DUNE 로드맵 (Track 3a 상세)

### Phase 2 — VLA(GR00T)로 대체

**2.1 환경 구축**
- `NVIDIA/Isaac-GR00T` 저장소 clone, 전용 conda 환경(`groot`, python 3.10) 생성
- Fine-tuning 기본 설정 기준 VRAM ~25GB 필요
- HuggingFace에서 GR00T N1.7 pretrained checkpoint 다운로드

**2.2 데모 데이터 수집** (Track 1을 "expert demonstrator"로 활용)
- `moveit_pick_place.py` 실행 중 RGB(-D)/로봇 state/action을 동기화해 기록하는 로거 노드
- LeRobot v2 스키마 + `meta/modality.json`으로 변환
- `binpicking_scene.py`의 object/bin pose 랜덤화를 켜서 여러 episode 수집

**2.3 Fine-tuning**
- `scripts/gr00t_finetune.py --dataset-path <경로>`로 bin-picking 단일 태스크 fine-tune
- 출력 action 포맷을 `action_adapter.py`/`rb5_action_space.yaml`(현재 `deprecated/`)에 맞춰 재검토

**2.4 추론 노드 개발**
- 신규 노드: 카메라+로봇 state 구독 → GR00T 추론 → EEF pose 변환 → MoveIt 실행
- `moveit_pick_place.py`를 대체하되 bin collision object 등록 등 나머지 인프라는 재사용

**2.5 평가**
- Isaac Sim closed-loop 실행 → Track 1 baseline 대비 성공률/정밀도 비교
- 학습 시 보지 못한 배치로 일반화 성능 확인

### Phase 3 — DUNE 기반 spatial input 추가

NAVER LABS Europe의 **DUNE**(DINOv2 + Multi-HMR + MASt3R distillation
기반 universal 2D/3D encoder)을 spatial feature 소스로 GR00T에 추가.

- GR00T 기본 vision encoder 교체 vs 별도 spatial branch concat — GR00T 코드 확인 후 결정 (보류 상태)
- `binpicking_scene.py`의 depth 퍼블리시를 데이터셋에 추가 채널로 저장
- DUNE 인코더는 freeze/LoRA, GR00T 나머지는 Phase 2 체크포인트에서 이어서 fine-tune
- Phase 2(GR00T-only) 대비 성공률/spatial 정확도 비교, 특히 가림(occlusion)/새 bin 배치 케이스 중심

### 참고 자료
- [GR00T N1.5 Explained](https://learnopencv.com/gr00t-n1_5-explained/)
- [NVIDIA/Isaac-GR00T (GitHub)](https://github.com/Nvidia/Isaac-GR00T)
- [Post-Training Isaac GR00T N1.5 for LeRobot SO-101 Arm](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [NVIDIA Isaac Teleop and GR00T 1.7 in LeRobot](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot)
- [DUNE: Distilling a Universal Encoder from heterogeneous 2D and 3D teachers — NAVER LABS Europe](https://europe.naverlabs.com/research/publications/dune/)
- [A Universal Encoder for Embodied Perception — NAVER LABS Europe](https://europe.naverlabs.com/research/a-universal-encoder-for-computer-vision/)

> API/스크립트 경로/데이터 포맷 등 GR00T/DUNE 세부 사항은 빠르게 바뀌는
> 영역이라, 실제 착수 시점에 최신 문서로 재확인 필요.
