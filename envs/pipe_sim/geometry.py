"""管道容器几何：解析 SDF + 光线投射深度渲染 + 巡检视点生成。

场景定义（与用户描述对齐）：
  - 容器主体：沿 x 轴的水平圆柱（"左右内壁为圆柱"）
  - 两端：半球端盖（"前后内壁为半圆"）→ 整体为胶囊体(capsule)内腔
  - 内部：密集支撑结构（strut），建模为横跨内腔的圆柱杆，
    按"环"沿轴向分布，每环 1~3 根、方位随机
  - 巡检任务：沿轴向的视点(waypoint)序列，横向偏移以绕开支撑杆

一切基于带符号距离场（SDF）：
  clearance(p) = 到最近表面的距离（>0 在自由空间内）
  这同时服务于 (a) 碰撞检测 (b) 深度相机的 sphere tracing
  (c) 近壁气动扰动代理的距离项 (d) 视点可行性验证。

设计图先验：支撑杆位置在环境构造时已知（对应"提前拿到设计图"），
视点序列即基于该先验离线生成——这是混合规划中的"离线全局层"。
"""

import numpy as np


class PipeVessel:
    def __init__(
        self,
        length=12.0,        # 圆柱段长度（米），不含端盖
        radius=1.0,         # 内壁半径（米）
        strut_spacing=1.5,  # 支撑环的轴向间距（米）
        struts_per_ring=(1, 3),  # 每环支撑杆数量范围（含端点）
        strut_radius=0.06,  # 支撑杆半径（米）
        seed=0,
    ):
        self.L = float(length)
        self.R = float(radius)
        self.strut_radius = float(strut_radius)
        self._rng = np.random.RandomState(seed)

        # 轴线段：从 (0,0,0) 到 (L,0,0)；端盖为半径 R 的半球
        # 内腔 = {p : dist(p, 轴线段) < R}
        self._axis_a = np.array([0.0, 0.0, 0.0])
        self._axis_b = np.array([self.L, 0.0, 0.0])

        # 生成支撑杆：每根是内壁上两点之间的圆柱杆（弦）
        self.struts = []  # list of (p0[3], p1[3], r)
        n_rings = int(self.L / strut_spacing)
        for i in range(n_rings):
            x = (i + 0.5) * strut_spacing
            if x >= self.L:
                break
            n_struts = self._rng.randint(struts_per_ring[0], struts_per_ring[1] + 1)
            for _ in range(n_struts):
                # 弦的两个端点方位角：保证弦长足够（跨度 > 90°），起支撑作用
                a0 = self._rng.uniform(0, 2 * np.pi)
                a1 = a0 + self._rng.uniform(0.5 * np.pi, 1.5 * np.pi)
                # 轻微轴向倾斜，更接近真实支撑构型
                dx = self._rng.uniform(-0.2, 0.2)
                p0 = np.array([x - dx, self.R * np.cos(a0), self.R * np.sin(a0)])
                p1 = np.array([x + dx, self.R * np.cos(a1), self.R * np.sin(a1)])
                self.struts.append((p0, p1, self.strut_radius))

        # 预打包成数组便于向量化 SDF
        if self.struts:
            self._sp0 = np.stack([s[0] for s in self.struts])  # (S,3)
            self._sp1 = np.stack([s[1] for s in self.struts])  # (S,3)
            self._sr = np.array([s[2] for s in self.struts])   # (S,)
        else:
            self._sp0 = np.zeros((0, 3))
            self._sp1 = np.zeros((0, 3))
            self._sr = np.zeros((0,))

    # ---------- SDF ----------

    @staticmethod
    def _dist_point_segment(p, a, b):
        """p: (...,3); a,b: (3,) 或 (S,3) 广播。返回到线段的距离。"""
        ab = b - a
        t = np.sum((p - a) * ab, axis=-1) / np.maximum(
            np.sum(ab * ab, axis=-1), 1e-12
        )
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[..., None] * ab
        return np.linalg.norm(p - proj, axis=-1)

    def clearance(self, p):
        """到最近表面的距离，>0 表示在自由空间内。p: (...,3) → (...)"""
        p = np.asarray(p, dtype=np.float64)
        # 容器内壁：R - 到轴线段的距离
        d_wall = self.R - self._dist_point_segment(p, self._axis_a, self._axis_b)
        if len(self._sr) == 0:
            return d_wall
        # 支撑杆：到杆轴的距离 - 杆半径（在杆外为正）
        pp = p[..., None, :]  # (...,1,3)
        d_struts = (
            self._dist_point_segment(pp, self._sp0, self._sp1) - self._sr
        )  # (...,S)
        return np.minimum(d_wall, d_struts.min(axis=-1))

    # ---------- 深度相机（sphere tracing） ----------

    def render_depth(self, cam_pos, yaw, pitch=0.0, size=(64, 64),
                     fov_deg=90.0, max_range=6.0, n_steps=48):
        """针孔深度相机。返回 (H,W) float 深度（米），未命中 = max_range。

        机体假设：速度控制下滚转/俯仰小，渲染只考虑 yaw（+可选 pitch）。
        sphere tracing：沿每条光线以 SDF 值为步长推进，天然利用解析几何，
        无需网格求交，64x64 在 CPU 上也足够快。
        """
        H, W = size
        # 相机坐标系光线方向（x 前，y 左，z 上）
        f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
        u = np.arange(W) - (W - 1) / 2.0
        v = np.arange(H) - (H - 1) / 2.0
        uu, vv = np.meshgrid(u, v)
        dirs = np.stack(
            [np.full_like(uu, f), -uu, -vv], axis=-1
        )  # (H,W,3) 相机系
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # 相机系 → 世界系（yaw 绕 z，pitch 绕 y）
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rwc = R_yaw @ R_pitch
        dirs_w = dirs @ Rwc.T  # (H,W,3)

        # sphere tracing
        origin = np.asarray(cam_pos, dtype=np.float64)
        t = np.zeros((H, W))
        hit = np.zeros((H, W), dtype=bool)
        for _ in range(n_steps):
            p = origin + t[..., None] * dirs_w
            d = self.clearance(p)
            newly_hit = (~hit) & (d < 1e-3)
            hit |= newly_hit
            step = np.where(hit, 0.0, np.maximum(d, 1e-3))
            t = np.minimum(t + step, max_range)
        t[~hit & (t >= max_range - 1e-6)] = max_range
        return t

    # ---------- 视点序列生成（离线全局层，基于设计图先验） ----------

    def generate_waypoints(self, spacing=1.0, robot_radius=0.15,
                           margin=0.15, n_candidates=64, seed=None):
        """沿轴向每 spacing 米放一个视点，横向偏移绕开支撑杆。

        每个轴向位置采样 n_candidates 个横截面内候选点，
        选择 clearance 最大的（最安全的间隙）——这就是"每个障碍
        从哪一侧过"这一同伦类决策的离线求解，输入只有设计图几何。
        返回 (N,3) 视点数组，全部满足 clearance > robot_radius+margin。
        """
        rng = np.random.RandomState(seed if seed is not None else 12345)
        xs = np.arange(spacing, self.L - 0.5 * spacing, spacing)
        waypoints = []
        for x in xs:
            # 候选：横截面圆盘内（偏向中心区域）
            ang = rng.uniform(0, 2 * np.pi, n_candidates)
            rad = self.R * 0.7 * np.sqrt(rng.uniform(0, 1, n_candidates))
            cand = np.stack(
                [np.full(n_candidates, x), rad * np.cos(ang), rad * np.sin(ang)],
                axis=-1,
            )
            cand[0, 1:] = 0.0  # 始终包含轴心作为候选
            c = self.clearance(cand)
            best = int(np.argmax(c))
            if c[best] > robot_radius + margin:
                waypoints.append(cand[best])
            # 若该截面无可行点（被支撑环堵死），跳过——相邻视点间
            # 的连接可行性由局部策略负责，这正是"预估路径可能
            # 无法直达、需要世界模型局部规划"的部分
        return np.array(waypoints)

    # ---------- 网格导出（供后续 Isaac/Aerial Gym 阶段使用） ----------

    def export_obj(self, path, n_seg=48, n_cap=12):
        """导出内壁网格为 OBJ（法线朝内），支撑杆为圆柱网格。
        后续 A100 阶段可转 URDF/USD 喂给 Isaac 系仿真器。"""
        verts, faces = [], []

        def add_cylinder(p0, p1, r, n=16):
            base = len(verts)
            axis = p1 - p0
            length = np.linalg.norm(axis)
            axis = axis / max(length, 1e-9)
            # 构造正交基
            tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            for i in range(n):
                a = 2 * np.pi * i / n
                off = r * (np.cos(a) * e1 + np.sin(a) * e2)
                verts.append(p0 + off)
                verts.append(p1 + off)
            for i in range(n):
                i2 = (i + 1) % n
                a, b = base + 2 * i, base + 2 * i + 1
                c, d = base + 2 * i2, base + 2 * i2 + 1
                faces.append((a + 1, b + 1, d + 1))
                faces.append((a + 1, d + 1, c + 1))

        # 圆柱段内壁
        base = len(verts)
        for i in range(n_seg + 1):
            x = self.L * i / n_seg
            for j in range(n_cap * 2):
                a = 2 * np.pi * j / (n_cap * 2)
                verts.append(np.array([x, self.R * np.cos(a), self.R * np.sin(a)]))
        ring = n_cap * 2
        for i in range(n_seg):
            for j in range(ring):
                j2 = (j + 1) % ring
                a = base + i * ring + j
                b = base + i * ring + j2
                c = base + (i + 1) * ring + j
                d = base + (i + 1) * ring + j2
                faces.append((a + 1, c + 1, d + 1))
                faces.append((a + 1, d + 1, b + 1))
        # 半球端盖（两端）
        for (cx, sign) in [(0.0, -1.0), (self.L, 1.0)]:
            cap_base = len(verts)
            for i in range(1, n_cap + 1):
                phi = 0.5 * np.pi * i / n_cap
                for j in range(ring):
                    a = 2 * np.pi * j / ring
                    verts.append(np.array([
                        cx + sign * self.R * np.sin(phi) * 0 + sign * self.R * (1 - np.cos(phi)) * 0
                        + sign * self.R * np.sin(phi) * 1.0 * 0  # 保持轴向偏移显式
                        ,
                        0, 0,
                    ]))
                    # 上面这行保持占位，下面直接覆盖为正确坐标
                    verts[-1] = np.array([
                        cx + sign * self.R * np.sin(phi),
                        self.R * np.cos(phi) * np.cos(a),
                        self.R * np.cos(phi) * np.sin(a),
                    ])
            for i in range(n_cap - 1):
                for j in range(ring):
                    j2 = (j + 1) % ring
                    a = cap_base + i * ring + j
                    b = cap_base + i * ring + j2
                    c = cap_base + (i + 1) * ring + j
                    d = cap_base + (i + 1) * ring + j2
                    faces.append((a + 1, c + 1, d + 1))
                    faces.append((a + 1, d + 1, b + 1))
        # 支撑杆
        for p0, p1, r in self.struts:
            add_cylinder(p0, p1, r)

        with open(path, "w") as fo:
            fo.write("# pipe vessel with internal struts\n")
            for v in verts:
                fo.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for f_ in faces:
                fo.write(f"f {f_[0]} {f_[1]} {f_[2]}\n")
        return len(verts), len(faces)
