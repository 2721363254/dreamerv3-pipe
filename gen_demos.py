"""特权专家演示生成器（gen_demos.py）。

用途（一份代码三个身份）：
  1. 演示源：生成成功巡检回合，npz 写入 <logdir>/train_eps/，
     dreamerv3-torch 启动时自动加载进回放缓冲（模型代码零改动）
  2. 论文基线：--eval 模式统计专家在指定难度下的成功率
  3. 环境审计：专家反复失败的布局会被记录

专家 = 前视 carrot 引导（沿 RRT 标称路径）+ 特权排斥项
（SDF clearance 梯度，有限差分）。特权信息只用于【控制】；
录制的观测走与训练完全相同的含噪管线（扰动场、位姿噪声、
域随机化全开）——世界模型学到的是真实训练分布下的穿越。

只保留完整成功的回合。演示统一在真实机体半径 0.26 的碰撞
判定下录制（课程各档均安全可用）。blueprint 布局被刻意排除
（保护零样本泛化评估）。

用法（先于训练执行）：
  python gen_demos.py --outdir logdir/pipe_v3_run1/train_eps --episodes 200
  python gen_demos.py --eval --episodes 50        # 仅统计基线成功率
"""

import argparse
import datetime
import io
import pathlib
import sys
import uuid

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from envs.pipe_inspection import PipeInspection  # noqa: E402
import envs.wrappers as wrappers                  # noqa: E402


def expert_action(backend, i):
    """势场专家：carrot 吸引 + 特权 clearance 梯度排斥。"""
    p = backend.pos[i].numpy()
    yaw = float(backend.yaw[i])
    v = backend.vessels[i]
    carrot = backend._carrot(i)

    d = carrot - p
    att = d / max(np.linalg.norm(d), 1e-6)
    g = np.zeros(3)
    eps = 0.03
    for k in range(3):
        e = np.zeros(3); e[k] = eps
        g[k] = (v.clearance(p + e) - v.clearance(p - e)) / (2 * eps)
    clr = float(v.clearance(p))
    rep = g * np.exp(-max(clr - 0.26, 0.0) / 0.15) * 1.2

    vw = att * 0.8 + rep
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    vb = np.array([cy * vw[0] - sy * vw[1],
                   sy * vw[0] + cy * vw[1], vw[2]])
    a = np.zeros(4, dtype=np.float32)
    a[0:3] = np.clip(vb / np.array([1.0, 1.0, 0.5]), -1, 1)
    return a


def run_episode(env, backend):
    """跑一条回合，返回 (transitions 列表, success)。
    transition 键与 tools.simulate 缓存一致：观测键 + action + reward + discount。"""
    obs = env.reset()
    backend._cur_radius[:] = 0.26  # 演示统一真实半径
    transitions = []
    tr = dict(obs)
    tr["action"] = np.zeros(4, dtype=np.float32)
    tr["reward"] = np.float32(0.0)
    tr["discount"] = np.float32(1.0)
    transitions.append(tr)
    success = False
    for t in range(320):
        a = expert_action(backend, 0)
        obs, r, done, info = env.step({"action": a})
        tr = dict(obs)
        tr["action"] = a
        tr["reward"] = np.float32(r)
        tr["discount"] = np.float32(info.get("discount", 1.0))
        transitions.append(tr)
        if done:
            success = bool(info.get("success", False))
            break
    return transitions, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=None,
                    help="npz 输出目录（通常为 <logdir>/train_eps）")
    ap.add_argument("--episodes", type=int, default=200,
                    help="目标成功回合数（--eval 模式下为尝试回合数）")
    ap.add_argument("--eval", action="store_true",
                    help="仅统计专家成功率（基线模式），不写文件")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 训练分布环境：random 布局池 + 全部随机化（与训练完全一致）
    raw = PipeInspection("mock", (64, 64), seed=args.seed, mode="train")
    backend = raw._backend          # 包装前留引用，专家控制用
    env = wrappers.NormalizeActions(raw)
    env = wrappers.TimeLimit(env, 320)
    env = wrappers.SelectAction(env, key="action")

    stats = {"success": 0, "crash": 0, "timeout": 0, "attempts": 0}
    fail_layouts = {}
    saved = 0
    outdir = pathlib.Path(args.outdir) if args.outdir else None
    if outdir and not args.eval:
        outdir.mkdir(parents=True, exist_ok=True)

    target = args.episodes
    while (stats["attempts"] < target if args.eval else saved < target):
        stats["attempts"] += 1
        if stats["attempts"] > target * 4 and not args.eval:
            print("尝试次数超上限，提前结束", flush=True)
            break
        transitions, success = run_episode(env, backend)
        layout_seed = backend.entries[0]["seed"]
        if success:
            stats["success"] += 1
            if not args.eval and outdir:
                # 拼成 episode dict 并保存（与 tools.save_episodes 同构）
                ep = {}
                for k in transitions[0]:
                    ep[k] = np.array([tr[k] for tr in transitions])
                ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                name = f"{ts}-expert{uuid.uuid4().hex}-{len(ep['reward'])}"
                with io.BytesIO() as f1:
                    np.savez_compressed(f1, **ep)
                    f1.seek(0)
                    with (outdir / f"{name}.npz").open("wb") as f2:
                        f2.write(f1.read())
                saved += 1
                if saved % 20 == 0:
                    print(f"[demos] 已保存 {saved}/{target} "
                          f"(尝试 {stats['attempts']})", flush=True)
        else:
            key = "crash"  # 简化归类：非成功即失败统计
            stats[key] = stats.get(key, 0) + 1
            fail_layouts[layout_seed] = fail_layouts.get(layout_seed, 0) + 1

    sr = stats["success"] / max(stats["attempts"], 1)
    print(f"\n专家统计: 成功率 {sr*100:.1f}% "
          f"({stats['success']}/{stats['attempts']})", flush=True)
    if fail_layouts:
        worst = sorted(fail_layouts.items(), key=lambda x: -x[1])[:5]
        print(f"失败最多的布局种子(审计): {worst}", flush=True)
    if not args.eval:
        print(f"演示已写入: {outdir} ({saved} 条)", flush=True)


if __name__ == "__main__":
    main()
