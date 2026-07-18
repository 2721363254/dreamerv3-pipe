"""离线策略评估（eval_policy.py）—— 论文主表数据生成器。

在同一组布局上对照评估【学习策略】与【势场专家】：
  - blueprint：图纸复刻场景（零样本）
  - heldout：16 个 held-out 随机布局（训练从未见过 —— 泛化指标）
  - sparse / medium / dense：三档难度的 held-out 池（难度谱）

用法：
  python eval_policy.py --ckpt logdir/pipe_v3_run1/best.pt --episodes 30
  python eval_policy.py --ckpt logdir/pipe_v3_run1/best.pt --episodes 30 --expert
      （--expert: 同一评估协议下跑势场专家，生成对照列）
  python eval_policy.py ... --splits blueprint heldout        （只跑部分 split）
  python eval_policy.py ... --json results/eval_run1.json     （结果落盘）

评估协议：真实半径 0.26（课程关闭）、扰动场/位姿噪声/动力学随机化全开
（与训练分布一致）、策略用确定性动作（training=False → mode）。
"""

import argparse
import functools
import json
import pathlib
import sys

import numpy as np
import torch
import ruamel.yaml as yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import tools                                        # noqa: E402
from dreamer import Dreamer                          # noqa: E402
from envs.pipe_inspection import PipeInspection      # noqa: E402
import envs.wrappers as wrappers                     # noqa: E402
from gen_demos import expert_action                  # noqa: E402


def load_config():
    """复刻 dreamer.py 的配置构建：defaults + pipe_mock。"""
    configs = yaml.YAML(typ="safe").load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text(
            encoding="utf-8"))
    defaults = {}

    def rupdate(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                rupdate(base[k], v)
            else:
                base[k] = v

    for name in ["defaults", "pipe_mock"]:
        rupdate(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args([])


SPLITS = {
    # split 名 → (mode, overrides)
    "blueprint": ("eval", {}),
    "heldout":   ("train", {"pool_split": "eval",
                            "curriculum_enabled": False}),
    "sparse":    ("train", {"pool_split": "eval", "difficulty": "sparse",
                            "curriculum_enabled": False}),
    "medium":    ("train", {"pool_split": "eval", "difficulty": "medium",
                            "curriculum_enabled": False}),
    "dense":     ("train", {"pool_split": "eval", "difficulty": "dense",
                            "curriculum_enabled": False}),
}


def make_env(split, seed):
    mode, overrides = SPLITS[split]
    raw = PipeInspection("mock", (64, 64), seed=seed, mode=mode,
                         overrides=dict(overrides))
    backend = raw._backend
    env = wrappers.NormalizeActions(raw)
    env = wrappers.TimeLimit(env, 320)
    env = wrappers.SelectAction(env, key="action")
    return env, backend


def build_agent(config, env, ckpt_path, device):
    config.num_actions = env.action_space.shape[0]
    logdir = pathlib.Path("/tmp/eval_policy_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, 0)
    agent = Dreamer(env.observation_space, env.action_space, config,
                    logger, dataset=None).to(device)
    agent.requires_grad_(False)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent


def rollout_policy(agent, env, device, max_steps=320):
    obs = env.reset()
    state = None
    total, steps = 0.0, 0
    for t in range(max_steps):
        obs_b = {k: np.asarray(v)[None] for k, v in obs.items()}
        done_b = np.array([False])
        with torch.no_grad():
            action, state = agent(obs_b, done_b, state, training=False)
        a = action["action"][0].detach().cpu().numpy()
        obs, r, done, info = env.step({"action": a})
        total += float(r)
        steps += 1
        if done:
            return dict(ret=total, length=steps,
                        success=bool(info.get("success", False)),
                        crash=bool(info.get("crash", False)))
    return dict(ret=total, length=steps, success=False, crash=False)


def rollout_expert(env, backend, max_steps=320):
    env.reset()
    backend._cur_radius[:] = 0.26
    total, steps = 0.0, 0
    for t in range(max_steps):
        a = expert_action(backend, 0)
        obs, r, done, info = env.step({"action": a})
        total += float(r)
        steps += 1
        if done:
            return dict(ret=total, length=steps,
                        success=bool(info.get("success", False)),
                        crash=bool(info.get("crash", False)))
    return dict(ret=total, length=steps, success=False, crash=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False, default=None,
                    help="策略 checkpoint（best.pt）。--expert 模式可省略")
    ap.add_argument("--episodes", type=int, default=30, help="每 split 回合数")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    ap.add_argument("--expert", action="store_true",
                    help="评估势场专家而非策略（同协议对照列）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", type=str, default=None, help="结果输出路径")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    agent = None
    if not args.expert:
        assert args.ckpt, "--ckpt 必填（或使用 --expert）"

    results = {}
    who = "expert" if args.expert else f"policy({args.ckpt})"
    print(f"评估对象: {who} | 每split {args.episodes} 回合\n")
    print(f"{'split':10s} {'成功率':>7s} {'撞毁率':>7s} {'超时率':>7s} "
          f"{'均回报':>8s} {'均长度':>7s}")

    for split in args.splits:
        env, backend = make_env(split, seed=args.seed)
        if agent is None and not args.expert:
            config = load_config()
            config.device = args.device
            agent = build_agent(config, env, args.ckpt, args.device)
        eps = []
        for e in range(args.episodes):
            if args.expert:
                eps.append(rollout_expert(env, backend))
            else:
                eps.append(rollout_policy(agent, env, args.device))
        sr = np.mean([e["success"] for e in eps])
        cr = np.mean([e["crash"] for e in eps])
        to = 1.0 - sr - cr
        mr = np.mean([e["ret"] for e in eps])
        ml = np.mean([e["length"] for e in eps])
        results[split] = dict(n=args.episodes, success=float(sr),
                              crash=float(cr), timeout=float(to),
                              mean_return=float(mr), mean_length=float(ml))
        print(f"{split:10s} {sr*100:6.1f}% {cr*100:6.1f}% {to*100:6.1f}% "
              f"{mr:8.1f} {ml:7.1f}")

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            dict(who=who, seed=args.seed, results=results),
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
