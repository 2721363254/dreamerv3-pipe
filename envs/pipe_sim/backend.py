"""MockPipeInspectionTask —— 复刻 Aerial Gym NavigationTask 的 API 契约。

契约（逐条对齐 ntnu-arl/aerial_gym_simulator 的 navigation_task.py）：
  __init__(task_config, seed=None, num_envs=None, headless=None,
           device=None, use_warp=None)
  action_space = Box(-1, 1, (4,))                       # 归一化动作
  action_transformation_function: [-1,1]^4 → 速度指令   # 对应 velocity_control
  step(actions) → (task_obs, rewards, terminations, truncations, infos)
      全部为 device 上的批量 torch 张量；
      task_obs = {"observations": (N, obs_dim)}，本实现额外提供
      task_obs["depth_range_pixels"]: (N, H, W) 深度图
  reset() / reset_idx(env_ids)

将来切换到真实 Aerial Gym：只需在 A100 上实现同名类包装真实
NavigationTask（几何改由 USD/URDF 资产给出），DreamerV3 侧零改动。

物理为简化模型（一阶速度跟踪 + 扰动注入），这是有意的：
  Windows 阶段验证"接口契约 + 观测组织 + 奖励设计 + 课程逻辑"，
  真实刚体动力学与传感器留给 Isaac 阶段。

近壁扰动场（研究要素，非装饰）：
  扰动加速度 = A · exp(-clearance/λ) · f(位置)，
  f 为每回合重采样的平滑随机向量场（随机傅里叶特征），
  额外含低速增益项——静止/悬停时回流积聚更强（对应 2507.15444
  的机理观察："持续前飞可缓解自身回流"）。
  这构成域随机化：策略无法记住某一张扰动图，只能学会在线适应
  ——正是世界模型 RSSM 隐状态应当发挥作用的地方。
"""

import numpy as np
import torch

from .geometry import PipeVessel


class task_config:
    """镜像 Aerial Gym 的 task_config 风格（类属性即配置）。"""
    seed = 0
    num_envs = 1
    device = "cpu"
    headless = True
    use_warp = False

    # 观测/动作维度（向量部分布局见 process_obs_for_task）
    observation_space_dim = 15
    action_space_dim = 4
    episode_len_steps = 400
    sim_dt = 0.1  # 每步 0.1 s

    # 相机
    camera_size = (64, 64)
    camera_fov_deg = 90.0
    camera_max_range = 6.0

    # 机体与控制
    robot_radius = 0.15
    v_tau = 0.35           # 一阶速度跟踪时间常数
    max_v_xy = 1.0         # 机体系 x/y 最大速度指令 (m/s)
    max_v_z = 0.5
    max_yaw_rate = 1.0     # rad/s

    # 管道几何
    pipe_length = 12.0
    pipe_radius = 1.0
    strut_spacing = 1.5
    struts_per_ring = (1, 3)
    strut_radius = 0.06

    # 视点任务
    waypoint_spacing = 1.2
    reach_threshold = 0.35

    # 扰动场
    dist_gain = 0.9        # 峰值扰动加速度 (m/s^2)，clearance→0 时
    dist_lengthscale = 0.35  # exp(-clearance/λ) 的 λ
    dist_hover_boost = 1.0   # 低速额外增益（回流积聚代理）
    dist_n_features = 8      # 随机傅里叶特征数量（场的空间复杂度）

    # 奖励
    k_progress = 10.0
    k_reach = 10.0
    k_crash = -10.0
    k_success = 20.0
    k_smooth = 0.1
    k_proximity = 0.2


class MockPipeInspectionTask:
    def __init__(self, task_config=task_config, seed=None, num_envs=None,
                 headless=None, device=None, use_warp=None):
        # 与 Aerial Gym 相同的参数覆盖逻辑
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if device is not None:
            task_config.device = device
        self.cfg = task_config
        self.num_envs = task_config.num_envs
        self.device = torch.device(task_config.device)
        self._rng = np.random.RandomState(task_config.seed)

        # 每个并行环境独立几何（不同支撑构型 = 结构级域随机化）
        self.vessels = []
        self.waypoints = []  # list of (Wi,3)
        for i in range(self.num_envs):
            v = PipeVessel(
                length=task_config.pipe_length,
                radius=task_config.pipe_radius,
                strut_spacing=task_config.strut_spacing,
                struts_per_ring=task_config.struts_per_ring,
                strut_radius=task_config.strut_radius,
                seed=task_config.seed * 1000 + i,
            )
            wps = v.generate_waypoints(
                spacing=task_config.waypoint_spacing,
                robot_radius=task_config.robot_radius,
                seed=task_config.seed * 1000 + i,
            )
            self.vessels.append(v)
            self.waypoints.append(wps)

        N = self.num_envs
        H, W = task_config.camera_size
        self.action_space_low = -np.ones(4, dtype=np.float32)
        self.action_space_high = np.ones(4, dtype=np.float32)

        # 状态张量
        self.pos = torch.zeros((N, 3), device=self.device)
        self.vel = torch.zeros((N, 3), device=self.device)
        self.yaw = torch.zeros((N,), device=self.device)
        self.prev_action = torch.zeros((N, 4), device=self.device)
        self.wp_idx = torch.zeros((N,), dtype=torch.long, device=self.device)
        self.sim_steps = torch.zeros((N,), dtype=torch.long, device=self.device)
        self.prev_dist = torch.zeros((N,), device=self.device)

        self.rewards = torch.zeros((N,), device=self.device)
        self.terminations = torch.zeros((N,), device=self.device)
        self.truncations = torch.zeros((N,), device=self.device)
        self.infos = {}
        self.task_obs = {
            "observations": torch.zeros(
                (N, task_config.observation_space_dim), device=self.device
            ),
            "depth_range_pixels": torch.zeros((N, H, W), device=self.device),
        }

        # 扰动场参数（reset_idx 时按回合重采样）
        F = task_config.dist_n_features
        self._dist_freq = np.zeros((N, F, 3))
        self._dist_phase = np.zeros((N, F))
        self._dist_dir = np.zeros((N, F, 3))

        self.reset()

    # ---------- 动作变换（对应 velocity_control 控制器） ----------

    def action_transformation_function(self, action):
        """[-1,1]^4 → [v_x^body, v_y^body, v_z, yaw_rate]"""
        c = self.cfg
        scale = torch.tensor(
            [c.max_v_xy, c.max_v_xy, c.max_v_z, c.max_yaw_rate],
            device=self.device,
        )
        return torch.clamp(action, -1.0, 1.0) * scale

    # ---------- 扰动场 ----------

    def _resample_disturbance(self, i):
        F = self.cfg.dist_n_features
        self._dist_freq[i] = self._rng.uniform(0.5, 2.5, (F, 3))
        self._dist_phase[i] = self._rng.uniform(0, 2 * np.pi, F)
        d = self._rng.normal(size=(F, 3))
        self._dist_dir[i] = d / np.linalg.norm(d, axis=-1, keepdims=True)

    def _disturbance(self, i, p, speed, clearance):
        """位置依赖 + 近壁增强 + 低速增强的扰动加速度（numpy, (3,)）。"""
        c = self.cfg
        phase = self._dist_freq[i] @ p + self._dist_phase[i]  # (F,)
        field = (np.sin(phase)[:, None] * self._dist_dir[i]).mean(axis=0)  # (3,)
        proximity = np.exp(-max(clearance, 0.0) / c.dist_lengthscale)
        hover = 1.0 + c.dist_hover_boost * np.exp(-(speed / 0.3) ** 2)
        return c.dist_gain * proximity * hover * field

    # ---------- reset ----------

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.task_obs

    def reset_idx(self, env_ids):
        for i in env_ids.tolist():
            v = self.vessels[i]
            # 起点：入口端轴心附近，带小扰动，验证可行
            for _ in range(50):
                p = np.array([
                    0.6 + self._rng.uniform(-0.1, 0.1),
                    self._rng.uniform(-0.2, 0.2),
                    self._rng.uniform(-0.2, 0.2),
                ])
                if v.clearance(p) > self.cfg.robot_radius + 0.1:
                    break
            self.pos[i] = torch.tensor(p, dtype=torch.float32, device=self.device)
            self.vel[i] = 0.0
            self.yaw[i] = 0.0
            self.prev_action[i] = 0.0
            self.wp_idx[i] = 0
            self.sim_steps[i] = 0
            wp = self.waypoints[i][0]
            self.prev_dist[i] = float(np.linalg.norm(wp - p))
            self._resample_disturbance(i)

    # ---------- step ----------

    def step(self, actions):
        c = self.cfg
        N = self.num_envs
        cmd = self.action_transformation_function(actions)  # (N,4)

        pos_np = self.pos.cpu().numpy()
        vel_np = self.vel.cpu().numpy()
        yaw_np = self.yaw.cpu().numpy()

        self.rewards[:] = 0.0
        self.terminations[:] = 0.0
        self.truncations[:] = 0.0
        crashes = torch.zeros((N,), device=self.device)
        successes = torch.zeros((N,), device=self.device)

        for i in range(N):
            v = self.vessels[i]
            # 机体系速度指令 → 世界系（仅 yaw）
            cy, sy = np.cos(yaw_np[i]), np.sin(yaw_np[i])
            cmd_i = cmd[i].cpu().numpy()
            v_cmd_w = np.array([
                cy * cmd_i[0] - sy * cmd_i[1],
                sy * cmd_i[0] + cy * cmd_i[1],
                cmd_i[2],
            ])
            speed = float(np.linalg.norm(vel_np[i]))
            clr = float(v.clearance(pos_np[i]))
            dist_acc = self._disturbance(i, pos_np[i], speed, clr)

            # 一阶速度跟踪 + 扰动
            vel_np[i] += (
                (v_cmd_w - vel_np[i]) * (c.sim_dt / c.v_tau) + dist_acc * c.sim_dt
            )
            pos_np[i] += vel_np[i] * c.sim_dt
            yaw_np[i] += float(cmd_i[3]) * c.sim_dt

            # 碰撞检测
            new_clr = float(v.clearance(pos_np[i]))
            if new_clr < c.robot_radius:
                self.terminations[i] = 1.0
                crashes[i] = 1.0
                self.rewards[i] += c.k_crash
                continue

            # 视点序列奖励
            wps = self.waypoints[i]
            k = int(self.wp_idx[i].item())
            wp = wps[min(k, len(wps) - 1)]
            d = float(np.linalg.norm(wp - pos_np[i]))
            self.rewards[i] += c.k_progress * (float(self.prev_dist[i]) - d)
            if d < c.reach_threshold:
                self.rewards[i] += c.k_reach
                k += 1
                self.wp_idx[i] = k
                if k >= len(wps):
                    # 巡检序列完成：成功终止
                    self.rewards[i] += c.k_success
                    self.terminations[i] = 1.0
                    successes[i] = 1.0
                    continue
                wp = wps[k]
                d = float(np.linalg.norm(wp - pos_np[i]))
            self.prev_dist[i] = d

            # 平滑与近壁软惩罚
            da = actions[i] - self.prev_action[i]
            self.rewards[i] += -c.k_smooth * float((da * da).sum())
            self.rewards[i] += -c.k_proximity * float(
                np.exp(-new_clr / 0.2)
            )

        self.pos = torch.tensor(pos_np, dtype=torch.float32, device=self.device)
        self.vel = torch.tensor(vel_np, dtype=torch.float32, device=self.device)
        self.yaw = torch.tensor(yaw_np, dtype=torch.float32, device=self.device)
        self.prev_action = actions.clone()
        self.sim_steps += 1

        # 超时截断（不算 crash，与 AG 语义一致）
        self.truncations[:] = torch.where(
            (self.sim_steps > c.episode_len_steps) & (self.terminations == 0),
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        self.infos = {
            "successes": successes,
            "crashes": crashes,
            "timeouts": self.truncations.clone(),
        }

        # 复位已结束的环境（与 AG 相同：先算奖励后复位，观测反映新回合）
        done_ids = torch.nonzero(
            (self.terminations + self.truncations) > 0
        ).flatten()
        # 保存终止标志供上层读取后再复位状态
        term_flags = self.terminations.clone()
        trunc_flags = self.truncations.clone()
        if len(done_ids) > 0:
            self.reset_idx(done_ids)

        self._compute_observations()
        return self.task_obs, self.rewards, term_flags, trunc_flags, self.infos

    # ---------- 观测 ----------

    def _compute_observations(self):
        """向量观测布局（15 维，镜像 AG navigation_task 风格）：
        [0:3]  机体系下指向当前视点的单位向量
        [3]    到当前视点的距离（米）
        [4:7]  机体系线速度
        [7]    偏航角速度指令（上一步）
        [8:12] 上一步动作
        [12]   剩余视点比例（1→0，任务进度）
        [13]   当前 clearance（截断到 [0,1]，近壁感知）
        [14]   速度模长
        """
        c = self.cfg
        H, W = c.camera_size
        for i in range(self.num_envs):
            v = self.vessels[i]
            p = self.pos[i].cpu().numpy()
            yaw = float(self.yaw[i].item())
            wps = self.waypoints[i]
            k = min(int(self.wp_idx[i].item()), len(wps) - 1)
            wp = wps[k]

            vec_w = wp - p
            d = np.linalg.norm(vec_w)
            cy, sy = np.cos(-yaw), np.sin(-yaw)
            vec_b = np.array([
                cy * vec_w[0] - sy * vec_w[1],
                sy * vec_w[0] + cy * vec_w[1],
                vec_w[2],
            ])
            unit_b = vec_b / max(d, 1e-6)
            vel_w = self.vel[i].cpu().numpy()
            vel_b = np.array([
                cy * vel_w[0] - sy * vel_w[1],
                sy * vel_w[0] + cy * vel_w[1],
                vel_w[2],
            ])
            clr = float(np.clip(v.clearance(p), 0.0, 1.0))

            obs = np.zeros(c.observation_space_dim, dtype=np.float32)
            obs[0:3] = unit_b
            obs[3] = d
            obs[4:7] = vel_b
            obs[7] = float(self.prev_action[i, 3].item())
            obs[8:12] = self.prev_action[i].cpu().numpy()
            obs[12] = 1.0 - k / max(len(wps), 1)
            obs[13] = clr
            obs[14] = float(np.linalg.norm(vel_w))
            self.task_obs["observations"][i] = torch.tensor(
                obs, device=self.device
            )

            depth = v.render_depth(
                p, yaw, size=(H, W),
                fov_deg=c.camera_fov_deg, max_range=c.camera_max_range,
            )
            self.task_obs["depth_range_pixels"][i] = torch.tensor(
                depth, dtype=torch.float32, device=self.device
            )
