"""
Phase 2: Isaac Lab bin picking environment skeleton for RB5-850E.

Isaac Lab 설치 필요:
  git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
  cd ~/IsaacLab && ./isaaclab.sh --install

학습 실행:
  cd ~/IsaacLab
  ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py \
    --task RB5-BinPicking-v0

주의:
  - Isaac Sim 4.1 기준, Isaac Lab v1.x API 사용
  - RB5-850E USD는 별도로 생성 필요 (URDF → USD 변환 또는 Isaac Sim import)
  - YCB 오브젝트는 Nucleus 또는 로컬 에셋 사용
"""

from __future__ import annotations
import math
import torch

# ── Isaac Lab imports (설치 후 활성화) ──────────────────────────────────────
# from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
# from omni.isaac.lab.scene import InteractiveSceneCfg
# from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
# from omni.isaac.lab.sim import SimulationCfg
# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.managers import RewardTermCfg, TerminationTermCfg

# ════════════════════════════════════════════════════════════════════════════
# 환경 설정 (Config)
# ════════════════════════════════════════════════════════════════════════════

# @configclass
class RB5BinPickingEnvCfg:
    """RB5-850E 빈피킹 환경 설정."""

    # ── 에피소드 ─────────────────────────────────────────────────────────────
    episode_length_s: float = 12.0          # 에피소드 최대 길이 (초)
    decimation: int = 2                      # 제어 주기 = sim_dt * decimation

    # ── 시뮬레이션 ───────────────────────────────────────────────────────────
    sim_dt: float = 0.005                   # physics timestep

    # ── 관측/행동 공간 ───────────────────────────────────────────────────────
    # 관측: 팔 관절 위치(6) + 속도(6) + 그리퍼(2) + 물체 pose(7) + EE pose(7)
    num_observations: int = 28
    # 행동: 팔 6 DOF delta position + 그리퍼 1 (open/close)
    num_actions: int = 7

    # ── 보상 가중치 ──────────────────────────────────────────────────────────
    reward_reach_weight: float = 1.0        # EE가 물체에 가까워질수록
    reward_lift_weight: float = 5.0         # 물체를 bin 위로 들어올릴 때
    reward_success_weight: float = 20.0     # 목표 위치에 내려놓으면
    penalty_collision_weight: float = -0.5  # 충돌 패널티

    # ── bin/물체 파라미터 ────────────────────────────────────────────────────
    bin_position: tuple = (0.45, 0.0, 0.0)
    bin_size: tuple = (0.28, 0.22, 0.17)    # (W, D, H)
    object_init_height_above_bin: float = 0.05

    # ── 로봇 USD 경로 ─────────────────────────────────────────────────────────
    # URDF → Isaac Sim USD 변환 후 경로 지정
    robot_usd_path: str = "TODO: /path/to/rb5_850e_with_gripper.usd"


# ════════════════════════════════════════════════════════════════════════════
# 환경 클래스
# ════════════════════════════════════════════════════════════════════════════

class RB5BinPickingEnv:
    """
    RB5-850E 빈피킹 DirectRLEnv 스켈레톤.

    Isaac Lab 설치 후:
      class RB5BinPickingEnv(DirectRLEnv):
    로 변경하고 아래 메서드들을 구현하세요.
    """

    cfg: RB5BinPickingEnvCfg

    def __init__(self, cfg: RB5BinPickingEnvCfg, render_mode=None, **kwargs):
        self.cfg = cfg
        # super().__init__(cfg, render_mode, **kwargs)
        print("[INFO] RB5BinPickingEnv skeleton initialized (Isaac Lab not installed)")

    # ── 씬 설정 ──────────────────────────────────────────────────────────────
    def _setup_scene(self):
        """로봇, bin, 오브젝트 씬 구성."""
        # 1. 로봇 ArticulationCfg 적용
        # self.robot = Articulation(self.cfg.robot_cfg)
        # self.scene.articulations["robot"] = self.robot

        # 2. Bin 컨테이너 (FixedCuboid 5개)
        # _create_bin(self.cfg.bin_position, self.cfg.bin_size)

        # 3. 피킹 오브젝트 (RigidObject)
        # self.objects = [RigidObject(...) for _ in range(NUM_OBJECTS)]

        # 4. 목표 위치 마커
        # self.goal_marker = VisualSphere(radius=0.03, color=(0, 1, 0))
        pass

    # ── 물리 스텝 전처리 ─────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        정책 출력 → 조인트 명령 변환.

        actions: (num_envs, 7) — [Δjoint×6, gripper_cmd]
        """
        # delta_joints = actions[:, :6] * self.cfg.action_scale
        # gripper_cmd  = actions[:, 6]
        # self.robot.set_joint_position_target(
        #     self._current_joints + delta_joints,
        #     joint_ids=ARM_JOINT_IDS,
        # )
        # self.gripper.set_joint_position_target(
        #     gripper_cmd * KNUCKLE_MAX,
        # )
        pass

    # ── 관측 계산 ─────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        관측 벡터 반환.

        Returns:
            dict with key "policy": (num_envs, num_observations) tensor
        """
        # joint_pos = self.robot.data.joint_pos[:, ARM_JOINT_IDS]   # (N, 6)
        # joint_vel = self.robot.data.joint_vel[:, ARM_JOINT_IDS]   # (N, 6)
        # ee_pose   = self._compute_ee_pose()                        # (N, 7)
        # obj_pose  = self.objects[0].data.root_state_w[:, :7]       # (N, 7)
        # gripper   = self.robot.data.joint_pos[:, GRIP_JOINT_IDS]  # (N, 2)
        # obs = torch.cat([joint_pos, joint_vel, gripper, obj_pose, ee_pose], dim=-1)
        # return {"policy": obs}
        return {"policy": torch.zeros(1, self.cfg.num_observations)}

    # ── 보상 계산 ─────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """
        보상 = reach_reward + lift_reward + success_reward + collision_penalty

        Returns:
            (num_envs,) tensor
        """
        # ee_pos  = self._compute_ee_pose()[:, :3]
        # obj_pos = self.objects[0].data.root_state_w[:, :3]
        # dist    = torch.norm(ee_pos - obj_pos, dim=-1)
        # reach   = self.cfg.reward_reach_weight * (1 / (1 + dist))
        # lift_h  = obj_pos[:, 2] - self.cfg.bin_position[2] - self.cfg.bin_size[2]
        # lift    = self.cfg.reward_lift_weight * torch.clamp(lift_h, 0.0)
        # return reach + lift
        return torch.zeros(1)

    # ── 에피소드 종료 조건 ───────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (terminated, truncated) — each (num_envs,) bool tensor
        """
        # time_out = self.episode_length_buf >= self.max_episode_length
        # success  = self._check_success()
        # return success, time_out
        return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)

    # ── 환경 리셋 ────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        지정된 환경 ID를 초기 상태로 리셋.
        - 로봇: home 자세로 복귀
        - 오브젝트: bin 위 랜덤 위치에서 낙하
        """
        # home_joints = torch.zeros(len(env_ids), 6)
        # self.robot.set_joint_position_target(home_joints, env_ids=env_ids)
        # for obj in self.objects:
        #     random_pos = self._sample_object_pos(env_ids)
        #     obj.write_root_state_to_sim(random_pos, env_ids=env_ids)
        pass

    # ── 유틸 ─────────────────────────────────────────────────────────────────
    def _compute_ee_pose(self) -> torch.Tensor:
        """End-effector (tcp) pose 계산."""
        # return self.robot.data.body_state_w[:, TCP_BODY_ID, :7]
        return torch.zeros(1, 7)

    def _check_success(self) -> torch.Tensor:
        """목표 위치에 물체가 도달했는지 확인."""
        # obj_pos  = self.objects[0].data.root_state_w[:, :3]
        # goal_pos = torch.tensor(self.cfg.goal_position).to(obj_pos.device)
        # return torch.norm(obj_pos - goal_pos, dim=-1) < 0.05
        return torch.zeros(1, dtype=torch.bool)


# ════════════════════════════════════════════════════════════════════════════
# Isaac Lab 태스크 등록 (설치 후 활성화)
# ════════════════════════════════════════════════════════════════════════════

# gym.register(
#     id="RB5-BinPicking-v0",
#     entry_point="lab_envs.rb5_binpicking_env:RB5BinPickingEnv",
#     kwargs={"cfg": RB5BinPickingEnvCfg()},
# )


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 진행 순서 (메모)
# ════════════════════════════════════════════════════════════════════════════
#
# 1. Isaac Lab 설치
#    git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
#    cd ~/IsaacLab && ./isaaclab.sh --install
#
# 2. RB5-850E USD 생성
#    Isaac Sim → File → Import URDF → rb5_with_tools.urdf → USD 저장
#    저장 경로를 robot_usd_path 에 입력
#
# 3. 환경 클래스 완성
#    - DirectRLEnv 상속 활성화
#    - 주석 처리된 코드 언주석 후 수정
#
# 4. RL 학습 실행
#    cd ~/IsaacLab
#    ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py \
#      --task RB5-BinPicking-v0 --num_envs 256
#
# 5. 학습된 policy → ROS2 deploy
#    ./isaaclab.sh -p source/standalone/workflows/rl_games/play.py \
#      --task RB5-BinPicking-v0 --checkpoint /path/to/model.pth