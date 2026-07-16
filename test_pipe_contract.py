"""管道环境三层契约测试 v2：几何构件库 → AG契约后端 → DreamerV3 适配层。"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")


def test_geometry():
    from envs.pipe_sim.geometry import PipeVessel

    # blueprint 布局：三件套按图纸间距
    v = PipeVessel(layout="blueprint", seed=3)
    kinds = [it.kind for it in v.internals]
    print(f"[几何] blueprint 内件: {kinds}")
    assert kinds == ["vortex", "weir", "baffle_v"]
    xs = [it.x for it in v.internals]
    assert abs((xs[1]-xs[0]) - 1.3) < 0.01 and abs((xs[2]-xs[1]) - 1.75) < 0.01

    # SDF 基本正确性
    assert v.clearance(np.array([0.5, 0.0, 0.0])) > 0.5   # 入口段自由
    assert v.clearance(np.array([3.5, 1.2, 0.0])) < 0      # 壁外

    # 三档难度：内件数量递增、全部种子存在可行视点
    from collections import Counter
    for diff, lo, hi in [("sparse",2,4), ("medium",4,7), ("dense",7,10)]:
        ns, ok = [], 0
        for s in range(10):
            vv = PipeVessel(layout="random", difficulty=diff, seed=s)
            ns.append(len(vv.internals))
            wps = vv.generate_waypoints(seed=s)
            if len(wps) >= 4 and min(vv.clearance(w) for w in wps) > 0.41:
                ok += 1
        print(f"[几何] {diff:7s}: 内件均值={np.mean(ns):.1f} (期望{lo}~{hi}区间), 可行 {ok}/10")
        assert ok >= 9, f"{diff} 档可行率过低"

    # 深度渲染
    depth = v.render_depth(np.array([0.8, 0, 0]), yaw=0.0)
    assert depth.shape == (64, 64) and depth.min() > 0
    print(f"[几何] 深度图范围 [{depth.min():.2f}, {depth.max():.2f}] m")
    print("✅ 几何层通过\n")


def test_backend_contract():
    from envs.pipe_sim.backend import MockPipeInspectionTask, task_config

    N = 2
    env = MockPipeInspectionTask(task_config, seed=7, num_envs=N, device="cpu")

    obs = env.reset()
    assert obs["observations"].shape == (N, 15)
    assert obs["depth_range_pixels"].shape == (N, 64, 64)
    print(f"[后端] reset → observations{tuple(obs['observations'].shape)}")

    ret = env.step(torch.tensor([[0.5, 0.0, 0.0, 0.0]] * N))
    assert len(ret) == 5
    task_obs, rewards, terms, truncs, infos = ret
    assert set(infos.keys()) >= {"successes", "crashes", "timeouts"}
    print("[后端] step → AG五元组 ✓")

    cmd = env.action_transformation_function(torch.ones(N, 4))
    assert torch.allclose(cmd[0], torch.tensor([1.0, 1.0, 0.5, 1.0]))

    # 课程：半径随回合数递进
    r0 = env.current_robot_radius(0)
    env.episode_count[0] = task_config.curriculum_episodes
    r1 = env.current_robot_radius(0)
    print(f"[后端] 课程半径: 初期={r0:.3f} → 完成={r1:.3f}")
    assert r0 < r1 and abs(r1 - 0.26) < 1e-6
    env.episode_count[0] = 1

    # 位姿噪声：sigma 每回合采样在界内
    assert (env._pose_sigma >= 0).all() and (env._pose_sigma <= task_config.pose_noise_max).all()
    # 去特权：obs[13] 为深度最小值 ∈ (0,1)
    o = env.task_obs["observations"][0].numpy()
    assert 0 < o[13] < 1
    print(f"[后端] 位姿噪声σ={env._pose_sigma.round(4).tolist()}, obs[13](min-depth)={o[13]:.3f}")

    # 扰动场：悬停漂移非零
    env.reset()
    p0 = env.pos.clone()
    for _ in range(30):
        env.step(torch.zeros(N, 4))
    drift = (env.pos - p0).norm(dim=-1)
    print(f"[后端] 悬停30步扰动漂移: {[f'{d:.3f}m' for d in drift.tolist()]}")
    assert drift.max() > 0.01
    print("✅ 后端层（AG契约+课程+噪声+去特权）通过\n")


def test_dreamer_adapter():
    import importlib
    import envs.pipe_inspection as pi
    import envs.wrappers as wrappers

    # train 模式 → random 布局
    importlib.reload(pi)
    e_train = pi.PipeInspection("mock", (64, 64), seed=11, mode="train")
    assert e_train._backend.vessels[0].difficulty in ("sparse", "medium", "dense")
    # eval 模式 → blueprint 布局 + 关闭课程
    importlib.reload(pi)
    e_eval = pi.PipeInspection("mock", (64, 64), seed=11, mode="eval")
    kinds = [it.kind for it in e_eval._backend.vessels[0].internals]
    assert kinds == ["vortex", "weir", "baffle_v"]
    print(f"[适配] train=random / eval=blueprint 分离 ✓")

    env = wrappers.SelectAction(wrappers.TimeLimit(
        wrappers.NormalizeActions(e_eval), 300), key="action")
    obs = env.reset()
    assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8
    assert obs["state"].shape == (15,) and obs["state"].dtype == np.float32

    total, steps = 0.0, 0
    for t in range(320):
        a = np.zeros(4, dtype=np.float32); a[0] = 0.6
        obs, r, done, info = env.step({"action": a})
        total += r; steps += 1
        if done:
            print(f"[适配] 盲飞终止于 {steps} 步 (crash={info.get('crash')})")
            break
    assert steps < 320
    print("✅ DreamerV3 适配层通过\n")


if __name__ == "__main__":
    test_geometry()
    test_backend_contract()
    test_dreamer_adapter()
    print("🎯 v2 三层契约全部通过")
