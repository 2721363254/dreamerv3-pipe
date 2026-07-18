"""MockPipeInspectionTask v3 —— 三层架构下的局部执行层。

三层分工：
  任务层  geometry.generate_viewpoints —— 巡检视点（真任务点，计分）
  全局层  planner.plan_inspection_path —— 标称几何上的 RRT 路径（离线，池缓存）
  局部层  本文件 —— RL 策略执行：沿路径引导 + 应对扰动/噪声/图实偏差

对 v2 的关键变更：
  - 引导：航点相对向量 → 前视 carrot（路径上前方 lookahead 弧长处的点）
  - progress 奖励：到航点距离缩短 → 沿路径弧长推进（单调索引跟踪，
    倒退不扣分也不可重复收割）
  - 到达奖励只在任务视点发放；路径顶点纯引导不计分
  - 布局池：每回合从训练池随机抽布局（修复单布局过拟合）；
    eval 模式固定 blueprint
  - 扩展域随机化：每回合采样 v_tau/速度上限 ±15%
  - 标称-扰动机制：路径在标称几何上规划，执行几何可加扰动
    （perturb_* 参数，默认 0 = 图纸即现实，作对照组）

Aerial Gym 五元组契约、课程门控、扰动场、位姿噪声、去特权观测
均自 v2 保留。
"""

import numpy as np
import torch

from .geometry import PipeVessel
from .planner import plan_inspection_path
from .pool import build_pool, instantiate


class task_config:
    seed = 0
    num_envs = 1
    device = "cpu"
    headless = True
    use_warp = False

    observation_space_dim = 15
    action_space_dim = 4
    episode_len_steps = 300
    sim_dt = 0.1

    camera_size = (64, 64)
    camera_fov_deg = 90.0
    camera_max_range = 6.0

    # 机体与课程（成功率门控，v2.2 语义不变）
    robot_radius = 0.26
    robot_radius_start = 0.15
    curriculum_enabled = True
    curriculum_gate_window = 20
    curriculum_gate_threshold = 0.8
    curriculum_radius_step = 0.01

    # 控制标称值（每回合围绕标称做域随机化）
    v_tau = 0.35
    max_v_xy = 1.0
    max_v_z = 0.5
    max_yaw_rate = 1.0
    dr_dyn_scale = 0.15        # v_tau / 速度上限的 ±随机化比例

    # 布局池（训练分布）与评估布局
    layout = "random"          # random=训练池 | blueprint=图纸评估
    difficulty = "medium"
    pool_train = 64
    pool_eval = 16
    plan_margin = 0.10

    # 标称-扰动（0 = 图纸即现实；实验轴，后续扫描）
    perturb_sigma_x = 0.0
    perturb_sigma_ang_deg = 0.0
    perturb_extra_prob = 0.0

    # 引导
    lookahead = 0.6            # carrot 前视弧长 (m)
    path_ds = 0.05             # 路径采样间距（与 planner 一致）
    search_window = 60         # 路径索引单调搜索窗口（×ds = 3 m）

    # 位姿噪声（动捕阶段量级）
    pose_noise_max = 0.02
    pose_drift_rate = 0.002

    # 扰动场
    dist_gain = 0.9
    dist_lengthscale = 0.35
    dist_hover_boost = 1.0
    dist_n_features = 8

    # 任务
    reach_threshold = 0.45

    # 奖励（v2.1 平衡结果沿用；progress 现指弧长推进）
    k_progress = 6.0
    k_reach = 10.0
    k_crash = -40.0
    k_success = 30.0
    k_smooth = 0.1
    k_proximity = 0.2


class MockPipeInspectionTask:
    def __init__(self, task_config=task_config, seed=None, num_envs=None,
                 headless=None, device=None, use_warp=None):
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

        c = self.cfg
        if c.layout == "blueprint":
            v = PipeVessel(layout="blueprint", seed=1)
            vps = v.generate_viewpoints(seed=1)
            plan = plan_inspection_path(v, vps, margin=c.plan_margin, seed=1)
            assert plan["ok"], "blueprint 布局规划失败"
            self._pool = [dict(seed=1, difficulty="blueprint", radius=v.R,
                               viewpoints=vps, path=plan["path"],
                               s=plan["s"], vp_idx=plan["vp_idx"],
                               _vessel=v)]
        else:
            pool = build_pool(n_train=c.pool_train, n_eval=c.pool_eval,
                              difficulty=c.difficulty,
                              margin=c.plan_margin, base_seed=0)
            self._pool = pool["train"]

        N = self.num_envs
        H, W = c.camera_size
        self.action_space_low = -np.ones(4, dtype=np.float32)
        self.action_space_high = np.ones(4, dtype=np.float32)

        # 每环境运行时状态
        self.vessels = [None] * N          # 执行几何（可能是扰动版）
        self.entries = [None] * N          # 当前池成员
        self.path_idx = np.zeros(N, dtype=np.int64)  # 路径单调索引
        self.wp_k = np.zeros(N, dtype=np.int64)      # 待达任务视点
        self._dyn_tau = np.full(N, c.v_tau)
        self._dyn_vmax = np.tile(
            [c.max_v_xy, c.max_v_xy, c.max_v_z, c.max_yaw_rate], (N, 1))

        self.pos = torch.zeros((N, 3), device=self.device)
        self.vel = torch.zeros((N, 3), device=self.device)
        self.yaw = torch.zeros((N,), device=self.device)
        self.prev_action = torch.zeros((N, 4), device=self.device)
        self.sim_steps = torch.zeros((N,), dtype=torch.long, device=self.device)

        self.rewards = torch.zeros((N,), device=self.device)
        self.terminations = torch.zeros((N,), device=self.device)
        self.truncations = torch.zeros((N,), device=self.device)
        self.infos = {}
        self.task_obs = {
            "observations": torch.zeros(
                (N, c.observation_space_dim), device=self.device),
            "depth_range_pixels": torch.zeros((N, H, W), device=self.device),
        }

        # 课程与噪声状态（v2.2 沿用）
        self.episode_count = np.zeros(N, dtype=np.int64)
        self._cur_radius = np.full(N, c.robot_radius_start)
        self._recent_frac = [[] for _ in range(N)]
        self._pose_sigma = np.zeros(N)
        self._pose_drift = np.zeros((N, 3))

        F = c.dist_n_features
        self._dist_freq = np.zeros((N, F, 3))
        self._dist_phase = np.zeros((N, F))
        self._dist_dir = np.zeros((N, F, 3))

        self.reset()

    # ---------- 契约接口 ----------

    def action_transformation_function(self, action):
        scale = torch.tensor(self._dyn_vmax, dtype=torch.float32,
                             device=self.device)
        return torch.clamp(action, -1.0, 1.0) * scale

    # ---------- 课程（v2.2 门控语义） ----------

    def current_robot_radius(self, i):
        c = self.cfg
        if not c.curriculum_enabled:
            return c.robot_radius
        return float(self._cur_radius[i])

    def _gate_curriculum(self, i, wp_fraction):
        c = self.cfg
        if not c.curriculum_enabled:
            return
        self._recent_frac[i].append(float(wp_fraction))
        if len(self._recent_frac[i]) > c.curriculum_gate_window:
            self._recent_frac[i].pop(0)
        if (len(self._recent_frac[i]) == c.curriculum_gate_window
                and np.mean(self._recent_frac[i]) >= c.curriculum_gate_threshold):
            old_r = self._cur_radius[i]
            self._cur_radius[i] = min(
                c.robot_radius, self._cur_radius[i] + c.curriculum_radius_step)
            print(f"[curriculum] env{i} ep{self.episode_count[i]}: "
                  f"radius {old_r:.3f} -> {self._cur_radius[i]:.3f} "
                  f"(gate mean={np.mean(self._recent_frac[i]):.2f})",
                  flush=True)
            self._recent_frac[i] = []
        if self.episode_count[i] % 50 == 0:
            rf = self._recent_frac[i]
            print(f"[curriculum] env{i} ep{self.episode_count[i]}: "
                  f"radius={self._cur_radius[i]:.3f} "
                  f"recent_frac_mean={np.mean(rf) if rf else 0:.2f} "
                  f"({len(rf)}/{c.curriculum_gate_window} eps)", flush=True)

    # ---------- 扰动场 ----------

    def _resample_disturbance(self, i):
        F = self.cfg.dist_n_features
        self._dist_freq[i] = self._rng.uniform(0.5, 2.5, (F, 3))
        self._dist_phase[i] = self._rng.uniform(0, 2 * np.pi, F)
        d = self._rng.normal(size=(F, 3))
        self._dist_dir[i] = d / np.linalg.norm(d, axis=-1, keepdims=True)

    def _disturbance(self, i, p, speed, clearance):
        c = self.cfg
        phase = self._dist_freq[i] @ p + self._dist_phase[i]
        field = (np.sin(phase)[:, None] * self._dist_dir[i]).mean(axis=0)
        proximity = np.exp(-max(clearance, 0.0) / c.dist_lengthscale)
        hover = 1.0 + c.dist_hover_boost * np.exp(-(speed / 0.3) ** 2)
        return c.dist_gain * proximity * hover * field

    # ---------- reset ----------

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.task_obs

    def reset_idx(self, env_ids):
        c = self.cfg
        for i in env_ids.tolist():
            entry = self._pool[self._rng.randint(len(self._pool))]
            self.entries[i] = entry
            if "_vessel" in entry and c.perturb_sigma_x == 0 \
                    and c.perturb_sigma_ang_deg == 0 \
                    and c.perturb_extra_prob == 0:
                self.vessels[i] = entry["_vessel"]   # blueprint 免重建
            else:
                self.vessels[i] = instantiate(
                    entry,
                    perturb_sigma_x=c.perturb_sigma_x,
                    perturb_sigma_ang_deg=c.perturb_sigma_ang_deg,
                    perturb_extra_prob=c.perturb_extra_prob,
                    perturb_seed=int(self._rng.randint(1 << 30)),
                )
            # 起点：路径起点附近（带抖动、验证可行）
            p0 = entry["path"][0]
            for _ in range(50):
                p = p0 + self._rng.uniform(-0.15, 0.15, 3) * [1, 1, 1]
                if self.vessels[i].clearance(p) > self.current_robot_radius(i) + 0.05:
                    break
            self.pos[i] = torch.tensor(p, dtype=torch.float32,
                                       device=self.device)
            self.vel[i] = 0.0
            self.yaw[i] = 0.0
            self.prev_action[i] = 0.0
            self.path_idx[i] = 0
            self.wp_k[i] = 0
            self.sim_steps[i] = 0
            self.episode_count[i] += 1
            self._pose_sigma[i] = self._rng.uniform(0, c.pose_noise_max)
            self._pose_drift[i] = 0.0
            self._resample_disturbance(i)
            # 扩展域随机化：动力学参数每回合抖动
            s = c.dr_dyn_scale
            self._dyn_tau[i] = c.v_tau * self._rng.uniform(1 - s, 1 + s)
            self._dyn_vmax[i] = np.array(
                [c.max_v_xy, c.max_v_xy, c.max_v_z, c.max_yaw_rate]
            ) * self._rng.uniform(1 - s, 1 + s, 4)

    # ---------- 路径工具 ----------

    def _advance_path_idx(self, i, p):
        """单调路径索引：只前进不后退（杜绝推进奖励收割）。"""
        entry = self.entries[i]
        k0 = int(self.path_idx[i])
        k1 = min(k0 + self.cfg.search_window, len(entry["path"]))
        d = np.linalg.norm(entry["path"][k0:k1] - p, axis=1)
        self.path_idx[i] = k0 + int(np.argmin(d))

    def _carrot(self, i):
        entry = self.entries[i]
        di = int(self.cfg.lookahead / self.cfg.path_ds)
        j = min(int(self.path_idx[i]) + di, len(entry["path"]) - 1)
        return entry["path"][j]

    # ---------- step ----------

    def step(self, actions):
        c = self.cfg
        N = self.num_envs
        cmd = self.action_transformation_function(actions)

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
            entry = self.entries[i]
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

            vel_np[i] += ((v_cmd_w - vel_np[i])
                          * (c.sim_dt / self._dyn_tau[i])
                          + dist_acc * c.sim_dt)
            pos_np[i] += vel_np[i] * c.sim_dt
            yaw_np[i] += float(cmd_i[3]) * c.sim_dt

            new_clr = float(v.clearance(pos_np[i]))
            if new_clr < self.current_robot_radius(i):
                self.terminations[i] = 1.0
                crashes[i] = 1.0
                self.rewards[i] += c.k_crash
                continue

            # 弧长推进奖励（单调）
            s_old = entry["s"][int(self.path_idx[i])]
            self._advance_path_idx(i, pos_np[i])
            s_new = entry["s"][int(self.path_idx[i])]
            self.rewards[i] += c.k_progress * float(s_new - s_old)

            # 任务视点到达（顺序计分）
            k = int(self.wp_k[i])
            vps = entry["viewpoints"]
            if k < len(vps):
                if np.linalg.norm(vps[k] - pos_np[i]) < c.reach_threshold:
                    self.rewards[i] += c.k_reach
                    k += 1
                    self.wp_k[i] = k
                    if k >= len(vps):
                        self.rewards[i] += c.k_success
                        self.terminations[i] = 1.0
                        successes[i] = 1.0
                        continue

            da = actions[i] - self.prev_action[i]
            self.rewards[i] += -c.k_smooth * float((da * da).sum())
            self.rewards[i] += -c.k_proximity * float(np.exp(-new_clr / 0.2))

        self.pos = torch.tensor(pos_np, dtype=torch.float32, device=self.device)
        self.vel = torch.tensor(vel_np, dtype=torch.float32, device=self.device)
        self.yaw = torch.tensor(yaw_np, dtype=torch.float32, device=self.device)
        self.prev_action = actions.clone()
        self.sim_steps += 1

        self.truncations[:] = torch.where(
            (self.sim_steps > c.episode_len_steps) & (self.terminations == 0),
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations))

        self.infos = {
            "successes": successes,
            "crashes": crashes,
            "timeouts": self.truncations.clone(),
            "curriculum_radius": torch.tensor(
                self._cur_radius.copy(), dtype=torch.float32,
                device=self.device),
        }

        done_ids = torch.nonzero(
            (self.terminations + self.truncations) > 0).flatten()
        term_flags = self.terminations.clone()
        trunc_flags = self.truncations.clone()
        if len(done_ids) > 0:
            for i in done_ids.tolist():
                n_wp = max(len(self.entries[i]["viewpoints"]), 1)
                frac = (1.0 if successes[i] > 0
                        else int(self.wp_k[i].item()) / n_wp)
                self._gate_curriculum(i, frac)
            self.reset_idx(done_ids)

        self._compute_observations()
        return self.task_obs, self.rewards, term_flags, trunc_flags, self.infos

    # ---------- 观测 ----------

    def _compute_observations(self):
        """向量观测布局（15 维）：
        [0:3]  机体系下指向 carrot（前视引导点）的单位向量
        [3]    剩余路径弧长（米）
        [4:7]  机体系线速度
        [7]    上一步偏航角速度指令
        [8:12] 上一步动作
        [12]   剩余任务视点比例（1→0）
        [13]   深度图最小值/max_range（近障感知，非特权）
        [14]   速度模长
        依赖定位的量（[0:3]/[3]）已注入位姿噪声（白噪声+慢漂移）。
        """
        c = self.cfg
        H, W = c.camera_size
        for i in range(self.num_envs):
            v = self.vessels[i]
            entry = self.entries[i]
            p_true = self.pos[i].cpu().numpy()
            yaw = float(self.yaw[i].item())

            depth = v.render_depth(
                p_true, yaw, size=(H, W),
                fov_deg=c.camera_fov_deg, max_range=c.camera_max_range)
            self.task_obs["depth_range_pixels"][i] = torch.tensor(
                depth, dtype=torch.float32, device=self.device)

            self._pose_drift[i] += self._rng.normal(0, c.pose_drift_rate, 3)
            p = p_true + self._pose_drift[i] + self._rng.normal(
                0, self._pose_sigma[i], 3)

            carrot = self._carrot(i)
            vec_w = carrot - p
            d = np.linalg.norm(vec_w)
            cy, sy = np.cos(-yaw), np.sin(-yaw)
            unit_b = (np.array([
                cy * vec_w[0] - sy * vec_w[1],
                sy * vec_w[0] + cy * vec_w[1],
                vec_w[2]]) / max(d, 1e-6))
            vel_w = self.vel[i].cpu().numpy()
            vel_b = np.array([
                cy * vel_w[0] - sy * vel_w[1],
                sy * vel_w[0] + cy * vel_w[1],
                vel_w[2]])
            remaining = float(entry["s"][-1] - entry["s"][int(self.path_idx[i])])
            min_depth = float(depth.min() / c.camera_max_range)

            obs = np.zeros(c.observation_space_dim, dtype=np.float32)
            obs[0:3] = unit_b
            obs[3] = remaining
            obs[4:7] = vel_b
            obs[7] = float(self.prev_action[i, 3].item())
            obs[8:12] = self.prev_action[i].cpu().numpy()
            obs[12] = 1.0 - int(self.wp_k[i]) / max(len(entry["viewpoints"]), 1)
            obs[13] = min_depth
            obs[14] = float(np.linalg.norm(vel_w))
            self.task_obs["observations"][i] = torch.tensor(
                obs, device=self.device)
