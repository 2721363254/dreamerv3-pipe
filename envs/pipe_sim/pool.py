"""布局池：预生成（几何 + 任务视点 + RRT 全局路径）并缓存。

职责：
  - 训练池 / held-out 评估池切分 —— 让"泛化到新布局"成为可测量指标
  - 每个池成员 = (几何种子, 难度, 视点, 全局路径)；几何按种子重建
    （spec 决定论已验证），路径直接缓存
  - 规划失败或视点不足的布局丢弃并记录（环境审计信号）
  - 磁盘缓存（pickle）：首次构建分钟级，之后秒级加载
  - 标称→扰动：池存标称几何种子与标称路径；执行环境按
    perturb_* 参数现场生成扰动副本（sigma=0 时即标称本身）
"""

import os
import pickle

import numpy as np

from .geometry import PipeVessel
from .planner import plan_inspection_path

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_pool_cache")


def _build_entry(seed, difficulty, pipe_radius_range, margin, rng):
    R = float(rng.uniform(*pipe_radius_range))
    v = PipeVessel(layout="random", difficulty=difficulty,
                   radius=R, seed=seed)
    vps = v.generate_viewpoints(seed=seed)
    if len(vps) < 3:
        return None
    plan = plan_inspection_path(v, vps, margin=margin, seed=seed)
    if not plan["ok"]:
        return None
    return dict(seed=seed, difficulty=difficulty, radius=R,
                viewpoints=vps, path=plan["path"], s=plan["s"],
                vp_idx=plan["vp_idx"])


def build_pool(n_train=64, n_eval=16, difficulty="medium",
               pipe_radius_range=(0.95, 1.05), margin=0.10,
               base_seed=0, cache=True, verbose=True):
    """构建（或从缓存加载）布局池。返回 dict(train=[...], eval=[...])。

    train/eval 池种子空间不相交（eval 种子偏移 100000），保证
    held-out 评估的是真正没见过的布局。
    """
    key = f"pool_{difficulty}_{n_train}_{n_eval}_{base_seed}"
    path = os.path.join(CACHE_DIR, key + ".pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            pool = pickle.load(f)
        if verbose:
            print(f"[pool] 缓存加载: {path} "
                  f"(train {len(pool['train'])}, eval {len(pool['eval'])})",
                  flush=True)
        return pool

    pool = {"train": [], "eval": [], "rejected": []}
    for split, n, offset in [("train", n_train, 0),
                             ("eval", n_eval, 100000)]:
        rng = np.random.RandomState(base_seed + offset)
        seed = base_seed + offset
        while len(pool[split]) < n:
            entry = _build_entry(seed, difficulty, pipe_radius_range,
                                 margin, rng)
            if entry is None:
                pool["rejected"].append(seed)
            else:
                pool[split].append(entry)
            seed += 1
        if verbose:
            print(f"[pool] {split} 池构建完成: {n} 个布局 "
                  f"(累计丢弃 {len(pool['rejected'])})", flush=True)

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pool, f)
        if verbose:
            print(f"[pool] 已缓存: {path}", flush=True)
    return pool


def instantiate(entry, perturb_sigma_x=0.0, perturb_sigma_ang_deg=0.0,
                perturb_extra_prob=0.0, perturb_seed=0):
    """池成员 → 可执行几何。标称按种子重建（决定论），
    扰动参数非零时返回扰动副本（规划路径仍是标称的——
    "图纸与现实的偏差"由此产生）。"""
    v = PipeVessel(layout="random", difficulty=entry["difficulty"],
                   radius=entry["radius"], seed=entry["seed"])
    if perturb_sigma_x > 0 or perturb_sigma_ang_deg > 0 \
            or perturb_extra_prob > 0:
        v = v.perturbed_copy(sigma_x=perturb_sigma_x,
                             sigma_ang_deg=perturb_sigma_ang_deg,
                             extra_strut_prob=perturb_extra_prob,
                             seed=perturb_seed)
    return v
