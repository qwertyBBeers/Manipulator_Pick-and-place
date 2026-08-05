# 학습 설정 참고 문서

`RB5-PickPlace-JointPos-v0`의 RL 학습에 관여하는 모든 파라미터/함수를
2026-07-28 `num_envs=2048` / `timesteps=100000` 실행 기준으로 정리한 스냅샷입니다.
태스크 설계(관측/보상/하이퍼파라미터)를 반복 수정할 때 참고하는 작업용 문서이며
자동 생성되지 않으므로, 학습 결과에 영향을 주는 설정을 바꿀 때마다 이 파일도
손으로 같이 갱신해야 합니다.

> **2026-07-28 업데이트**: 이 태스크(`RB5-PickPlace-JointPos-v0`, 한 번에
> 전체 pick-and-place를 학습)는 그리퍼를 벌리고 팔을 눕히는 degenerate 정책만
> 나와서, 4단계 PPO 커리큘럼(`Reach`→`GraspLift`→`Transport`→`Curriculum`)으로
> 대체 개발 중입니다. 새 커리큘럼의 설계/진단/검증 결과는
> [CURRICULUM_REPORT.md](CURRICULUM_REPORT.md) 참고. 이 문서(레거시 단일
> 태스크)는 여전히 정확하지만, 실제 다음 학습은 커리큘럼 쪽에서 진행됩니다.

## 1. 씬 / 물리
[`pick_place_env_cfg.py`](rb5_isaaclab/tasks/pick_place/pick_place_env_cfg.py) `RB5ObjectSceneCfg`

- `num_envs=2048`, `env_spacing=2.5m`
- 물체: 0.042m 큐브, 0.10kg, source bin 중심 위쪽에 스폰
- source bin: center=(0.51, 0.0, 0.0), inner_size=(0.42, 0.36, 0.01) — `../rb5_binpicking/config/bin_geometry.yaml`에서 읽어옴
- dest bin: center=(0.28, 0.44, 0.0), inner_size=(0.28, 0.23, 0.01) — 같은 파일
- 바닥: 50x50m 절차적 평면 (Nucleus 서버 의존 없음)
- 타이밍: `sim.dt=0.01s`, `decimation=2` → 제어 주기 0.02s, `episode_length_s=8.0s` → **에피소드당 400스텝**

## 2. 로봇
[`robots/rb5_850e.py`](rb5_isaaclab/robots/rb5_850e.py) `RB5_850E_ROBOTIQ_CFG`

| 액추에이터 그룹 | 관절 | stiffness | damping | effort_limit |
|---|---|---|---|---|
| `rb5_arm` | base, shoulder, elbow, wrist1, wrist2, wrist3 (6개) | 10000 | 1000 | 150 |
| `gripper_drive` | left_knuckle (1개, 유일한 실제 구동 자유도) | 7000 | 350 | 15 |
| `gripper_mimic` | 나머지 그리퍼 관절 5개 (PhysX mimic 제약) | 0 | 0 | - |

초기 자세: `shoulder=-1.0, elbow=1.6, wrist1=-0.6, wrist2=1.57` (검증 안 된 추정값, 육안 확인 안 함), 그리퍼는 열림(0.0).

## 3. 액션 공간 (7차원)
[`config/joint_pos_env_cfg.py`](rb5_isaaclab/tasks/pick_place/config/joint_pos_env_cfg.py)

- `arm_action`: `JointPositionActionCfg`, 팔 관절 6개, `scale=0.5`, `use_default_offset=True`
- `gripper_action`: `BinaryJointPositionActionCfg`, knuckle 관절 1개만, open=0.0 / close=0.8 rad

## 4. 관측 공간 (46차원)
[`pick_place_env_cfg.py` `ObservationsCfg`](rb5_isaaclab/tasks/pick_place/pick_place_env_cfg.py) + [`mdp/observations.py`](rb5_isaaclab/tasks/pick_place/mdp/observations.py)

| 항목 | 크기 | 비고 |
|---|---|---|
| `joint_pos` | 12 | 기본 자세 대비 상대값 |
| `joint_vel` | 12 | |
| `object_position` | 3 | 로봇 root 프레임 기준 |
| `object_orientation` | 4 | 로봇 root 프레임 기준 쿼터니언 |
| `gripper_opening` | 1 | knuckle 각도 (0~0.8 rad) — **각도만 볼 뿐, 접촉 기반 아님** |
| `target_object_position` | 7 | 명령된 목표 pose (위치+쿼터니언) |
| `actions` | 7 | 직전 액션 |

`enable_corruption=True` (관측 노이즈 켜져 있음).

## 5. 보상 함수 (8개 항목)
[`pick_place_env_cfg.py` `RewardsCfg`](rb5_isaaclab/tasks/pick_place/pick_place_env_cfg.py) + [`mdp/rewards.py`](rb5_isaaclab/tasks/pick_place/mdp/rewards.py)

| 항목 | 가중치 | 설명 |
|---|---|---|
| `reaching_object` | 1.0 | tanh(EE-물체 거리 / 0.1) |
| `lifting_object` | 15.0 | 높이가 src_floor+0.04m를 넘으면 sparse하게 1.0 |
| `object_goal_tracking` | 16.0 | tanh(목표 거리 / 0.3), 들어올린 상태에서만 적용 |
| `object_goal_tracking_fine_grained` | 5.0 | 위와 동일하되 std=0.05 (정밀 유도용) |
| `place_and_release` | 25.0 | at_goal(0.03m) & settled(속도<0.05m/s) & gripper_open(<0.4rad)이 동시에 만족되면 1.0 |
| `holding_at_goal_penalty` | 0.0 → -2.0 (커리큘럼, step 5000부터) | 목표 지점인데 그리퍼가 계속 닫혀있으면 페널티 — "목표 근처서 계속 들고만 있기" 보상 해킹을 막기 위해 2026-07-28에 추가 — **아직 검증 안 됨** |
| `action_rate` | -1e-4 → -1e-1 (커리큘럼, step 10000부터) | 액션 급변 L2 페널티 |
| `joint_vel` | -1e-4 → -1e-1 (커리큘럼, step 10000부터) | 관절 속도 L2 페널티 |

8개 중 6개(`reaching_object`~`joint_vel`, 단 `place_and_release`/`holding_at_goal_penalty` 제외)는 IsaacLab 공식 Franka `Isaac-Lift-Cube-Franka-v0` 태스크의 보상(`isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py`)을 가중치까지 그대로 이식한 것입니다. `place_and_release`와 `holding_at_goal_penalty`는 이 태스크 전용으로 새로 만든 것이라 참고할 원본이 없습니다.

**참고할 만한 사실**: IsaacLab 자체의 더 완전한 배치(placement) 태스크들(`stack`, `place`, `pick_place` — GR1T2/Agibot/UR10 설정)은 전부 `rewards = None`이고, RL 보상 설계 대신 robomimic BC-RNN 모방학습으로 훈련됩니다. Farama의 `FetchPickAndPlace-v2`는 dense shaping 대신 순수 sparse reward + HER를 씁니다. 우리의 dense-shaping 방식은 robosuite의 `PickPlace` 태스크에 더 가까운데, 그쪽은 추가로 접촉 기반 grasp 감지를 쓰는 반면 우리는 아직 없습니다(위 관측 표 참고).

## 6. 종료 조건
[`pick_place_env_cfg.py` `TerminationsCfg`](rb5_isaaclab/tasks/pick_place/pick_place_env_cfg.py) + [`mdp/terminations.py`](rb5_isaaclab/tasks/pick_place/mdp/terminations.py)

- `time_out`: 400스텝
- `object_dropping`: 물체 높이 < src_floor - 0.05m
- `object_placed`: `place_and_release`와 동일 조건 — 조기 성공 종료

## 7. 이벤트 (리셋)
- `reset_object_position`: source bin 영역 내 균일 랜덤 XY + 랜덤 yaw(-π~π), Z는 고정
- `reset_all`: 매 에피소드마다 로봇을 기본 관절 자세로 리셋

## 8. PPO 하이퍼파라미터
[`config/agents/skrl_ppo_cfg.yaml`](rb5_isaaclab/tasks/pick_place/config/agents/skrl_ppo_cfg.yaml)

- 정책/가치망: MLP `[256, 128, 64]`, ELU, 공유 안 함(`separate: False`)
- `rollouts=24`, `learning_epochs=8`, `mini_batches=4`
- `discount_factor=0.99`, GAE `lambda=0.95`
- `learning_rate=1e-4` + `KLAdaptiveLR(kl_threshold=0.01)`
- `entropy_loss_scale=0.001`, `value_loss_scale=2.0`, `ratio_clip=0.2`, `value_clip=0.2`
- `trainer.timesteps=100000` → `num_envs=2048` 기준 총 경험량 약 2억 (IsaacLab 공식 Franka Lift 레시피의 약 1억 4700만보다 많음)

## 실행 이력

| 날짜 | num_envs | timesteps | 비고 |
|---|---|---|---|
| 2026-07-27 12:44 | 512 | 48000 | 첫 실제 실행. 팔의 시각/충돌 메시가 빠져있었음(`convert_robot_to_usd.py`의 package:// URI 해석 버그, 같은 날 수정됨) — 물리적으로 무효한 결과라 대체됨. |
| 2026-07-27 19:51 | 2048 | 100000 | 팔 메시 수정 + `holding_at_goal_penalty` 추가 후 실행. 40:27(2429초) 만에 에러 없이 완주. Play로 확인해보니 로봇이 태스크 수행 대신 그리퍼를 벌리고 팔을 뒤로 눕히는 동작을 보임 — 미숙련/degenerate 행동, 다음에 진단할 대상. |

## 어디를 고치면 되는지

스스로 수정해나가기 위한 파일 지도입니다. 전부 editable pip install(`pip install -e rb5_isaaclab/`를 한 번 실행해둔 상태)로 깔린 순수 Python/YAML이라, 수정 후 재설치 없이 바로 `train.py`를 다시 돌리면 반영됩니다. URDF/메시를 바꿨을 때만 `scripts/convert_robot_to_usd.sh` 재실행이 필요합니다.

**씬 / 환경 구성 ("layer")**
- [`tasks/pick_place/pick_place_env_cfg.py`](rb5_isaaclab/tasks/pick_place/pick_place_env_cfg.py) `RB5ObjectSceneCfg` — prim 추가/제거/위치 변경(물체, bin, 바닥, 조명, 향후 distractor 물체), 물체의 물리 속성(크기/질량/마찰/solver iteration)
- [`tasks/pick_place/bin_geometry.py`](rb5_isaaclab/tasks/pick_place/bin_geometry.py) — bin 위치/크기의 출처(`../rb5_binpicking/config/bin_geometry.yaml`를 읽음; bin을 옮기려면 이 파일이 아니라 그 YAML을 수정)
- [`robots/rb5_850e.py`](rb5_isaaclab/robots/rb5_850e.py) — 로봇 자체: 관절 그룹별 액추에이터 gain(stiffness/damping/effort_limit), 초기 관절 자세, 어느 링크에 마찰을 줄지
- [`config/joint_pos_env_cfg.py`](rb5_isaaclab/tasks/pick_place/config/joint_pos_env_cfg.py) — 액션 항의 배선(어느 관절, scale, open/close 명령값), `ee_frame` 센서 타겟, `_PLAY`/`_SAC` 변형의 num_envs
- [`config/ik_rel_env_cfg.py`](rb5_isaaclab/tasks/pick_place/config/ik_rel_env_cfg.py) — 관절 위치 제어 대신 IK 기반 액션 공간을 쓰고 싶을 때 대안
- `pick_place_env_cfg.py`의 `__post_init__`(파일 맨 아래) — 시뮬레이션 레벨 타이밍/물리: `sim.dt`, `decimation`, `episode_length_s`, PhysX GPU 버퍼 크기

**관측 공간**
- [`mdp/observations.py`](rb5_isaaclab/tasks/pick_place/mdp/observations.py) — 실제로 무엇을 계산할지(예: 접촉 기반 grasp 신호를 추가한다면 여기)
- `pick_place_env_cfg.py`의 `ObservationsCfg.PolicyCfg` — 위 함수들 중 실제로 어떤 걸 포함할지, 어떤 순서로(순서가 중요함 — 그대로 이어붙여져 정책 입력이 됨)

**보상 함수 설계**
- [`mdp/rewards.py`](rb5_isaaclab/tasks/pick_place/mdp/rewards.py) — 각 보상 항목의 실제 수식; 새 함수는 여기에 추가
- `pick_place_env_cfg.py`의 `RewardsCfg` — 어떤 항목이 활성화되어 있는지, `weight`, `params`(`position_threshold`, `std` 등 임계값) — 대부분의 튜닝은 `rewards.py`를 안 건드리고 여기서 끝남
- `pick_place_env_cfg.py`의 `CurriculumCfg` — 학습 중간에 가중치를 바꿔야 하는 항목(`modify_reward_weight`, `RewardsCfg`의 속성명과 일치하는 `term_name` + `num_steps`로 지정)

**종료 / 성공 기준**
- [`mdp/terminations.py`](rb5_isaaclab/tasks/pick_place/mdp/terminations.py) + `pick_place_env_cfg.py`의 `TerminationsCfg`

**학습 하이퍼파라미터**
- [`config/agents/skrl_ppo_cfg.yaml`](rb5_isaaclab/tasks/pick_place/config/agents/skrl_ppo_cfg.yaml) — PPO: 네트워크 크기, rollouts/epochs/batches, learning rate, clip 범위, 전체 `trainer.timesteps`
- [`config/agents/skrl_sac_cfg.yaml`](rb5_isaaclab/tasks/pick_place/config/agents/skrl_sac_cfg.yaml) — SAC 비교 실행용
- `train.py`의 `--num_envs` CLI 플래그 — 파일 수정 없이 실행마다 env cfg 기본 num_envs를 덮어씀

**로봇 형상/기구학 자체** (거의 건드릴 일 없음)
- [`assets/urdf/rb5_850e_robotiq_mimic.urdf`](rb5_isaaclab/assets/urdf/rb5_850e_robotiq_mimic.urdf) + [`scripts/convert_robot_to_usd.py`](rb5_isaaclab/scripts/convert_robot_to_usd.py) — 로봇/그리퍼 모델 자체를 바꿔야 할 때만; 수정 후 `convert_robot_to_usd.sh` 재실행 필요

## 남은 질문 / 다음 설계 단계

- 접촉 기반 grasp 감지가 없음(`gripper_opening`은 각도만 봄) — fingertip 링크에 `ContactSensorCfg`를 붙이는 게 후보안
- `holding_at_goal_penalty`가 아직 검증 안 됨 — 지금 관찰된 "그리퍼 벌리고 팔 눕기" degenerate 행동을 고치기는커녕 오히려 원인일 수도 있음
- 팔 초기 자세가 아직 육안으로 검증 안 됨
- 도메인 랜덤화(질량, 마찰, 관절 gain)가 아직 없음
