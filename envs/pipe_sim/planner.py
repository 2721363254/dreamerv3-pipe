"""全局层：RRT-Connect 离线路径规划。

职责：在【标称】几何上，把任务层给出的视点序列连成一条连续无碰路径。
离线执行（布局池构建时规划一次并缓存），不考虑运行时效率——
对应真实部署流程"图纸建模 → 离线规划"。

设计要点：
  - RRT-Connect（双树对向生长），对几何拓扑零假设——弯管/斜桁架/
    任意 3D 障碍通吃，这是它取代 2D 截面法的根本理由
  - 碰撞检测 = vessel.clearance(p) > margin（默认 0.10 m：
    机体 0.26 之下的紧裕度，保证能通到贴壁视点；容错交给局部策略）
  - 后处理：随机捷径平滑（去 RRT 锯齿）→ 等弧长重采样（ds=0.05 m）
  - 输出携带累计弧长 s 与各视点在路径上的索引，供局部层做
    前视引导（carrot）与弧长推进奖励
"""

import numpy as np


def _seg_free(vessel, a, b, margin):
    """线段无碰检验：SDF 保守推进（sphere marching）。

    在点 p 处 clearance=c 时，沿线段前进 (c-margin) 距离内
    clearance 必不低于 margin（距离场 1-Lipschitz）——因此按
    该步长推进即可零漏检，薄板（3cm）不可能被隧穿。
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-9:
        return float(vessel.clearance(a)) > margin
    u = (b - a) / L
    t = 0.0
    while t < L:
        c = float(vessel.clearance(a + u * t))
        if c <= margin:
            return False
        t += max(c - margin, 0.004)
    return float(vessel.clearance(b)) > margin


def _rrt_connect(vessel, start, goal, margin, rng,
                 step=0.15, max_iters=4000):
    """双树 RRT-Connect。返回路径点列表（含起终点）或 None。"""
    R, L = vessel.R, vessel.L
    lo = np.array([0.0, -R, -R])
    hi = np.array([L, R, R])

    trees = [
        {"pts": [np.asarray(start, float)], "parent": [-1]},
        {"pts": [np.asarray(goal, float)], "parent": [-1]},
    ]

    def nearest(tree, q):
        pts = np.asarray(tree["pts"])
        return int(np.argmin(np.linalg.norm(pts - q, axis=1)))

    def extend(tree, q):
        """向 q 走一步；返回 ('reached'|'advanced'|'trapped', new_idx)。"""
        i = nearest(tree, q)
        p = tree["pts"][i]
        d = q - p
        dist = np.linalg.norm(d)
        p_new = q if dist <= step else p + d / dist * step
        if not _seg_free(vessel, p, p_new, margin):
            return "trapped", -1
        tree["pts"].append(p_new)
        tree["parent"].append(i)
        j = len(tree["pts"]) - 1
        return ("reached" if dist <= step else "advanced"), j

    def connect(tree, q):
        while True:
            status, j = extend(tree, q)
            if status != "advanced":
                return status, j

    for it in range(max_iters):
        q_rand = rng.uniform(lo, hi)
        if vessel.clearance(q_rand) <= margin:
            continue
        sa, ia = extend(trees[0], q_rand)
        if sa != "trapped":
            q_new = trees[0]["pts"][ia]
            sb, ib = connect(trees[1], q_new)
            if sb == "reached":
                # 回溯两树拼接
                def trace(tree, idx):
                    out = []
                    while idx != -1:
                        out.append(tree["pts"][idx])
                        idx = tree["parent"][idx]
                    return out
                path_a = trace(trees[0], ia)[::-1]
                path_b = trace(trees[1], ib)
                combined = path_a + path_b
                # 树交替生长后 trees[0] 可能是目标树 —— 按端点校正方向，
                # 否则拼接路径反向，与锚点间产生未检验跳线（隧穿源）
                if (np.linalg.norm(combined[0] - np.asarray(start, float))
                        > np.linalg.norm(combined[-1] - np.asarray(start, float))):
                    combined = combined[::-1]
                return combined
        trees = trees[::-1]  # 交替生长
    return None


def _shortcut(vessel, path, margin, rng, iters=150):
    """随机捷径平滑。"""
    path = [np.asarray(p, float) for p in path]
    for _ in range(iters):
        if len(path) < 3:
            break
        i = rng.randint(0, len(path) - 2)
        j = rng.randint(i + 2, len(path))
        if _seg_free(vessel, path[i], path[j], margin):
            path = path[: i + 1] + path[j:]
    return path


def _resample(path, ds=0.05):
    """等弧长重采样。返回 (pts (N,3), s (N,))。"""
    pts = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        L = np.linalg.norm(b - a)
        if L < 1e-9:
            continue
        n = max(int(np.ceil(L / ds)), 1)
        for t in np.linspace(0, 1, n + 1)[1:]:
            pts.append(a + t * (b - a))
    pts = np.asarray(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return pts, s


def plan_inspection_path(vessel, viewpoints, start=None, margin=0.10,
                         seed=0, ds=0.05):
    """把视点序列连成一条完整巡检路径（在标称几何上）。

    返回 dict:
      path (N,3)      等弧长采样的路径点
      s (N,)          累计弧长
      vp_idx list     每个视点对应的路径索引（到达判定与计分用）
      ok bool         全部段是否规划成功
    规划失败的段以直线占位并标记 ok=False（布局池会丢弃该布局，
    并记录——"专家级规划器都连不通的布局"是环境审计信号）。
    """
    rng = np.random.RandomState(seed)
    start = np.array([0.6, 0.0, 0.0]) if start is None else np.asarray(start)
    anchors = [start] + [np.asarray(v, float) for v in viewpoints]
    full = [anchors[0]]
    ok = True
    for a, b in zip(anchors[:-1], anchors[1:]):
        if _seg_free(vessel, a, b, margin):
            seg = [a, b]
        else:
            seg = _rrt_connect(vessel, a, b, margin, rng)
            if seg is None:
                ok = False
                seg = [a, b]  # 占位
            else:
                seg = _shortcut(vessel, seg, margin, rng)
        full.extend(seg[1:])
    pts, s = _resample(full, ds)
    # 视点 → 最近路径索引（单调推进：从上一个视点索引往后找）
    vp_idx, k0 = [], 0
    for v in viewpoints:
        d = np.linalg.norm(pts[k0:] - np.asarray(v), axis=1)
        k = k0 + int(np.argmin(d))
        vp_idx.append(k)
        k0 = k
    return dict(path=pts, s=s, vp_idx=vp_idx, ok=ok)
