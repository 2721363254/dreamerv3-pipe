"""管道容器几何 v2：工艺内件构件库 + 程序化布局 + SDF 光线投射。

v2 变更（依据实地照片 + 工程图纸口述参数）：
  - 容器：Φ2 m × 7 m 卧式胶囊（圆柱 + 半球端盖），尺寸已由图纸确认
  - 内件从"随机弦杆"升级为构件库（板类为主，管类保留）：
      V 形挡板 baffle plate —— 两块板成 "\\/"，留角部/顶部窗口
      堰板 weir plate       —— 单板自底升起，顶部留窗（可滚转偏侧）
      防涡器 vortex breaker —— 壁面局部小凸起
      弦杆 chord strut      —— 管状内件（照片中真实存在的类别）
    外加环向加强环（壁面深度纹理 + 近壁间隙削减）
  - 两种布局：random（类型/数量/间距全随机 —— 训练分布）
              blueprint（B -1.3m- C -1.75m- D —— 图纸复刻，固定评估场景）
  - 窗口保证：每个内件截面经采样验证存在 ≥ min_window 的可行通道

SDF 约定：clearance(p) > 0 表示自由空间，值为到最近表面的距离下界
（相交处取 max 会低估距离，对 sphere tracing 是安全方向）。
"""

import numpy as np


# ---------- 基础 SDF 原语（全部向量化，p: (...,3)） ----------

def _sd_segment(p, a, b, r):
    """胶囊/圆杆：到线段距离 - 半径。正值在杆外。"""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=-1) /
                max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1) - r


def _sd_halfplane_slab(p, x0, t, u, b):
    """轴向厚度 t 的半平面板：{|x-x0|<=t/2} ∩ {u·(y,z) <= b}。
    u: (2,) 单位向量。正值在板外。"""
    s_slab = np.abs(p[..., 0] - x0) - 0.5 * t
    s_half = p[..., 1] * u[0] + p[..., 2] * u[1] - b
    return np.maximum(s_slab, s_half)


def _sd_box(p, center, half):
    """轴对齐盒。正值在盒外。"""
    q = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def _sd_ring(p, x0, t, R, rib_h):
    """环向加强环：{|x-x0|<=t/2} ∩ {d_axis >= R-rib_h}（依附壁面的环）。"""
    d_axis = np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
    s_slab = np.abs(p[..., 0] - x0) - 0.5 * t
    s_ann = (R - rib_h) - d_axis
    return np.maximum(s_slab, s_ann)


# ---------- 内件构件（每个实例携带自己的 SDF 参数） ----------

class Internal:
    """一个内件 = 若干 SDF 原语的并集（取 min）。"""
    def __init__(self, kind, x, prims):
        self.kind = kind      # 'baffle_v' / 'weir' / 'vortex' / 'strut'
        self.x = float(x)     # 轴向位置（窗口验证与视点生成用）
        self.prims = prims    # list of (fn_name, params dict)

    def sdf(self, p):
        vals = []
        for name, prm in self.prims:
            if name == "halfplane":
                vals.append(_sd_halfplane_slab(p, **prm))
            elif name == "segment":
                vals.append(_sd_segment(p, **prm))
            elif name == "box":
                vals.append(_sd_box(p, **prm))
        return np.minimum.reduce(vals) if len(vals) > 1 else vals[0]


class PipeVessel:
    """Φ2R × L 卧式胶囊容器 + 程序化内件。API 与 v1 兼容。"""

    # 难度档位：内件密度 / 窗口紧度 / 复合截面概率
    DIFFICULTY = {
        "sparse": dict(n_internals=(2, 4), spacing=(1.2, 2.0),
                       min_window=0.90, cluster_prob=0.0),
        "medium": dict(n_internals=(4, 7), spacing=(0.8, 1.5),
                       min_window=0.75, cluster_prob=0.3),
        "dense":  dict(n_internals=(7, 10), spacing=(0.5, 1.0),
                       min_window=0.65, cluster_prob=0.6),
    }

    def __init__(
        self,
        length=7.0,
        radius=1.0,
        layout="random",          # 'random' | 'blueprint' | 'empty'
        difficulty="medium",      # 'sparse' | 'medium' | 'dense'
        min_window=None,          # None = 取档位值；可显式覆盖
        n_internals=None,
        spacing=None,
        cluster_prob=None,        # 复合截面概率：两内件贴近放置(0.3~0.5m)
                                  # 且窗口方位错开，强制短距横向机动
        internal_types=("baffle_v", "weir", "vortex", "strut"),
        plate_thickness=0.03,
        strut_radius=0.09,
        ring_spacing=1.4,         # 环向加强环轴向间距（0 关闭）
        rib_height=0.10,
        seed=0,
    ):
        self.L = float(length)
        self.R = float(radius)
        d = dict(self.DIFFICULTY[difficulty])
        if min_window is not None: d["min_window"] = min_window
        if n_internals is not None: d["n_internals"] = n_internals
        if spacing is not None: d["spacing"] = spacing
        if cluster_prob is not None: d["cluster_prob"] = cluster_prob
        self.difficulty = difficulty
        self.min_window = float(d["min_window"])
        self._cluster_prob = float(d["cluster_prob"])
        n_internals = d["n_internals"]
        spacing = d["spacing"]
        self._plate_t = plate_thickness
        self._strut_r = strut_radius
        self._rib_h = rib_height
        self._rng = np.random.RandomState(seed)
        self._axis_a = np.array([0.0, 0.0, 0.0])
        self._axis_b = np.array([self.L, 0.0, 0.0])

        self.internals = []
        self.spec = []   # [(kind, x, kwargs)] —— 标称设计，供扰动复制
        self.rings = []  # x 位置列表（内件生成后放置，避开内件截面）

        if layout == "blueprint":
            self._build_blueprint()
        elif layout == "random":
            self._build_random(n_internals, spacing, internal_types)
        # 'empty'：无内件（课程起点/调试用）

        if ring_spacing and ring_spacing > 0:
            n = int(self.L / ring_spacing)
            for i in range(n):
                x0 = (i + 0.5) * ring_spacing
                if x0 >= self.L - 0.3:
                    continue
                if all(abs(x0 - it.x) > 0.35 for it in self.internals):
                    self.rings.append(x0)

        # 兼容 v1 的 struts 属性（供 OBJ 导出等旧代码路径）
        self.struts = [
            (prm["a"], prm["b"], prm["r"])
            for it in self.internals for name, prm in it.prims
            if name == "segment"
        ]

    # ---------- 构件生成器 ----------

    def _mk_baffle_v(self, x, opening=None, tilt=None, b_scales=None,
                     _record=True):
        """V 形挡板 "\\/"：两块半平面板，法线向内倾斜，
        开口方向 opening（弧度，窗口中心方位角）留出通道。"""
        opening = self._rng.uniform(0, 2 * np.pi) if opening is None else opening
        tilt = self._rng.uniform(np.deg2rad(25), np.deg2rad(55)) if tilt is None else tilt
        if b_scales is None:
            b_scales = tuple(self._rng.uniform(1.0, 1.6, 2))
        if _record:
            self.spec.append(("baffle_v", x, dict(
                opening=opening, tilt=tilt, b_scales=tuple(b_scales))))
        prims = []
        for sgn, bs in zip((+1.0, -1.0), b_scales):
            ang = opening + sgn * (0.5 * np.pi + tilt)
            u = np.array([np.cos(ang), np.sin(ang)])
            b = -0.5 * self.min_window * bs
            prims.append(("halfplane", dict(x0=x, t=self._plate_t, u=u, b=b)))
        return Internal("baffle_v", x, prims)

    def _mk_weir(self, x, roll=None, height_frac=None, _record=True):
        """堰板 "-"：单块板占据截面一侧，窗口在对侧。
        roll 控制窗口方位（默认窗口朝上 = 板从底部升起）。"""
        roll = (self._rng.uniform(0, 2 * np.pi)
                if roll is None else roll)          # 窗口中心方位角
        hf = (self._rng.uniform(0.35, 0.65)
              if height_frac is None else height_frac)  # 板占直径比例
        if _record:
            self.spec.append(("weir", x, dict(roll=roll, height_frac=hf)))
        u = np.array([np.cos(roll), np.sin(roll)])  # 指向窗口的方向
        # 板 = {u·p <= b}；窗口宽度 = R - b，保证 >= 1.1*min_window
        b_max = self.R - 1.1 * self.min_window
        b = 2.0 * self.R * hf - self.R          # 由板占直径比例 hf 推出
        b = float(np.clip(b, -0.6 * self.R, b_max))
        return Internal("weir", x, [
            ("halfplane", dict(x0=x, t=self._plate_t, u=u, b=float(b)))
        ])

    def _mk_vortex(self, x, azim=None, size=None, _record=True):
        """防涡器 "."：壁面局部小凸起（盒）。不构成主通道约束。"""
        azim = self._rng.uniform(0, 2 * np.pi) if azim is None else azim
        s = self._rng.uniform(0.10, 0.20) if size is None else size
        if _record:
            self.spec.append(("vortex", x, dict(azim=azim, size=s)))
        c = np.array([x, (self.R - 0.5 * s) * np.cos(azim),
                      (self.R - 0.5 * s) * np.sin(azim)])
        return Internal("vortex", x, [
            ("box", dict(center=c, half=np.array([s, s, s]) * 0.5))
        ])

    def _mk_strut(self, x, a0=None, a1=None, dx=None, _record=True):
        """管状弦杆（照片中的管式内件类别；v1 遗产）。"""
        a0 = self._rng.uniform(0, 2 * np.pi) if a0 is None else a0
        a1 = (a0 + self._rng.uniform(0.6 * np.pi, 1.4 * np.pi)
              if a1 is None else a1)
        dx = self._rng.uniform(-0.15, 0.15) if dx is None else dx
        if _record:
            self.spec.append(("strut", x, dict(a0=a0, a1=a1, dx=dx)))
        p0 = np.array([x - dx, self.R * np.cos(a0), self.R * np.sin(a0)])
        p1 = np.array([x + dx, self.R * np.cos(a1), self.R * np.sin(a1)])
        return Internal("strut", x, [
            ("segment", dict(a=p0, b=p1, r=self._strut_r))
        ])

    _MAKERS = {"baffle_v": "_mk_baffle_v", "weir": "_mk_weir",
               "vortex": "_mk_vortex", "strut": "_mk_strut"}

    # ---------- 布局 ----------

    def _validated_add(self, maker, x, tries=8, **kw):
        """生成内件并验证其截面存在可行窗口，失败重采样。"""
        for _ in range(tries):
            n_spec = len(self.spec)
            it = getattr(self, maker)(x, **kw)
            self.internals.append(it)
            if self._window_ok(x):
                return True
            self.internals.pop()
            del self.spec[n_spec:]
        return False

    def _window_ok(self, x, n=256):
        """截面 x 处采样验证：最大 clearance >= min_window/2。"""
        ang = self._rng.uniform(0, 2 * np.pi, n)
        rad = self.R * np.sqrt(self._rng.uniform(0, 1, n))
        pts = np.stack([np.full(n, x), rad * np.cos(ang), rad * np.sin(ang)], -1)
        return self.clearance(pts).max() >= 0.5 * self.min_window

    def _build_random(self, n_range, spacing, types):
        n = self._rng.randint(n_range[0], n_range[1] + 1)
        x = self._rng.uniform(0.8, 1.6)
        placed = 0
        # 有窗口方位概念的内件（复合截面用它们制造 S 机动）
        directional = [t for t in types if t in ("baffle_v", "weir")]
        while placed < n and x < self.L - 0.8:
            kind = types[self._rng.randint(len(types))]
            main_open = self._rng.uniform(0, 2 * np.pi)
            kw = {}
            if kind == "baffle_v": kw = dict(opening=main_open)
            elif kind == "weir": kw = dict(roll=main_open)
            if self._validated_add(self._MAKERS[kind], x, **kw):
                placed += 1
                # 复合截面：贴近追加一个方位错开的板类内件
                if (directional and placed < n
                        and self._rng.rand() < self._cluster_prob):
                    x2 = x + self._rng.uniform(0.3, 0.5)
                    if x2 < self.L - 0.8:
                        k2 = directional[self._rng.randint(len(directional))]
                        off = main_open + self._rng.uniform(
                            0.5 * np.pi, 1.5 * np.pi)
                        kw2 = (dict(opening=off) if k2 == "baffle_v"
                               else dict(roll=off))
                        if self._validated_add(self._MAKERS[k2], x2, **kw2):
                            placed += 1
                            x = x2
            x += self._rng.uniform(*spacing)

    def _build_blueprint(self):
        """图纸复刻：B(防涡器) -1.3m- C(堰板) -1.75m- D(V形挡板)。
        绝对位置取序列居中于 7 m 管体。"""
        xB = 0.5 * (self.L - (1.3 + 1.75))   # ≈1.975
        self._validated_add("_mk_vortex", xB, azim=-0.5 * np.pi)  # 底部防涡器
        self._validated_add("_mk_weir", xB + 1.3,
                            roll=0.5 * np.pi, height_frac=0.5)     # 窗口朝上
        self._validated_add("_mk_baffle_v", xB + 1.3 + 1.75,
                            opening=0.5 * np.pi)                   # 开口朝上

    # ---------- clearance / 渲染 / 视点（API 与 v1 相同） ----------

    def clearance(self, p):
        p = np.asarray(p, dtype=np.float64)
        d_wall = self.R - (np.linalg.norm(
            p - np.clip((p[..., 0:1]), 0.0, self.L) *
            np.array([1.0, 0, 0]) - self._axis_a, axis=-1
        ) if False else self._dist_axis(p))
        out = d_wall
        for x0 in self.rings:
            out = np.minimum(out, _sd_ring(p, x0, 0.06, self.R, self._rib_h))
        for it in self.internals:
            out = np.minimum(out, it.sdf(p))
        return out

    def _dist_axis(self, p):
        ab = self._axis_b - self._axis_a
        t = np.clip(p[..., 0] / self.L, 0.0, 1.0)
        proj = self._axis_a + t[..., None] * ab
        return np.linalg.norm(p - proj, axis=-1)

    def render_depth(self, cam_pos, yaw, pitch=0.0, size=(64, 64),
                     fov_deg=90.0, max_range=6.0, n_steps=48):
        H, W = size
        f = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg))
        u = np.arange(W) - (W - 1) / 2.0
        v = np.arange(H) - (H - 1) / 2.0
        uu, vv = np.meshgrid(u, v)
        dirs = np.stack([np.full_like(uu, f), -uu, -vv], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        dirs_w = dirs @ (R_yaw @ R_pitch).T
        origin = np.asarray(cam_pos, dtype=np.float64)
        t = np.zeros((H, W))
        hit = np.zeros((H, W), dtype=bool)
        for _ in range(n_steps):
            p = origin + t[..., None] * dirs_w
            d = self.clearance(p)
            newly = (~hit) & (d < 1e-3)
            hit |= newly
            t = np.minimum(t + np.where(hit, 0.0, np.maximum(d, 1e-3)),
                           max_range)
        t[~hit & (t >= max_range - 1e-6)] = max_range
        return t

    def generate_viewpoints(self, mode="axial", spacing=1.1,
                            robot_radius=0.26, margin=0.10,
                            n_candidates=96, seed=None):
        """任务层：巡检视点序列（到达即计分的真任务点）。

        axial 模式：沿轴向按固定间隔布置，每站取截面内 clearance
        最大点（安全的观察站）。注意这只是任务定义之一——视点之间
        的连接可行性完全交给全局层（RRT），此处不做窗口点/合并等
        引导性处理。wall_scan 模式（沿壁扫描摄影点）留待后续。
        """
        assert mode == "axial", "wall_scan 模式后续版本实现"
        rng = np.random.RandomState(seed if seed is not None else 12345)
        thresh = robot_radius + margin
        vps = []
        for x in np.arange(spacing, self.L - 0.5 * spacing, spacing):
            ang = rng.uniform(0, 2 * np.pi, n_candidates)
            rad = self.R * 0.8 * np.sqrt(rng.uniform(0, 1, n_candidates))
            cand = np.stack([np.full(n_candidates, x),
                             rad * np.cos(ang), rad * np.sin(ang)], -1)
            cand[0, 1:] = 0.0
            c = self.clearance(cand)
            best = int(np.argmax(c))
            if c[best] > thresh:
                vps.append(cand[best])
        return np.array(vps)

    def perturbed_copy(self, sigma_x=0.0, sigma_ang_deg=0.0,
                       extra_strut_prob=0.0, seed=0):
        """标称设计 → 施工/现实扰动版：内件轴向位移 N(0,σx)、
        角参数偏转 N(0,σang)，并以概率追加图纸外杆件。
        sigma=0 且 extra=0 时返回与标称几何完全一致的副本。
        全局层在标称上规划，执行环境用扰动版 —— 制造"图纸与
        现实的偏差"，策略必须在线适应（可扫描的实验轴）。"""
        rng = np.random.RandomState(seed)
        v = PipeVessel(length=self.L, radius=self.R, layout="empty",
                       difficulty=self.difficulty,
                       plate_thickness=self._plate_t,
                       strut_radius=self._strut_r,
                       ring_spacing=0,  # 环手动复制，保持一致
                       rib_height=self._rib_h, seed=seed)
        v.rings = list(self.rings)
        sa = np.deg2rad(sigma_ang_deg)
        for kind, x, kw in self.spec:
            x2 = x + rng.normal(0, sigma_x) if sigma_x > 0 else x
            kw2 = dict(kw)
            for key in ("opening", "roll", "azim", "a0", "a1"):
                if key in kw2 and sa > 0:
                    kw2[key] = kw2[key] + rng.normal(0, sa)
            maker = getattr(v, self._MAKERS[kind])
            v.internals.append(maker(float(np.clip(x2, 0.5, self.L-0.5)),
                                     _record=True, **kw2))
        if extra_strut_prob > 0:
            for _ in range(3):
                if rng.rand() < extra_strut_prob:
                    xr = rng.uniform(1.0, self.L - 1.0)
                    v.internals.append(v._mk_strut(xr, _record=True))
        v.struts = [
            (prm["a"], prm["b"], prm["r"])
            for it in v.internals for name, prm in it.prims
            if name == "segment"
        ]
        return v

    def export_obj(self, path, n_seg=48, n_cap=12):
        """导出壳体 + 管状内件网格（板类内件导出为薄盒近似，
        供可视化确认；Isaac 阶段用参数直接生成精确资产）。"""
        # 壳体部分沿用 v1 实现思路，此处从简：只导出壳体与杆件
        verts, faces = [], []

        def add_cyl(p0, p1, r, n=12):
            base = len(verts)
            axis = p1 - p0
            ln = np.linalg.norm(axis)
            axis = axis / max(ln, 1e-9)
            tmp = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
            e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            for i in range(n):
                a = 2 * np.pi * i / n
                off = r * (np.cos(a) * e1 + np.sin(a) * e2)
                verts.append(p0 + off); verts.append(p1 + off)
            for i in range(n):
                i2 = (i + 1) % n
                a, b_, = base + 2*i, base + 2*i + 1
                c, d = base + 2*i2, base + 2*i2 + 1
                faces.append((a+1, b_+1, d+1)); faces.append((a+1, d+1, c+1))

        ring = n_cap * 2
        base = len(verts)
        for i in range(n_seg + 1):
            x = self.L * i / n_seg
            for j in range(ring):
                a = 2 * np.pi * j / ring
                verts.append(np.array([x, self.R*np.cos(a), self.R*np.sin(a)]))
        for i in range(n_seg):
            for j in range(ring):
                j2 = (j + 1) % ring
                a = base + i*ring + j; b_ = base + i*ring + j2
                c = base + (i+1)*ring + j; d = base + (i+1)*ring + j2
                faces.append((a+1, c+1, d+1)); faces.append((a+1, d+1, b_+1))
        for p0, p1, r in self.struts:
            add_cyl(np.asarray(p0), np.asarray(p1), r)
        with open(path, "w") as fo:
            for v in verts:
                fo.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for f_ in faces:
                fo.write(f"f {f_[0]} {f_[1]} {f_[2]}\n")
        return len(verts), len(faces)
