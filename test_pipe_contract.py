"""管道环境契约测试 v3：任务层 → 全局层(RRT) → 布局池 → 后端 → 适配层。"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")


def test_geometry_and_task_layer():
    from envs.pipe_sim.geometry import PipeVessel

    v = PipeVessel(layout="blueprint", seed=1)
    assert [k for k, _, _ in v.spec] == ["vortex", "weir", "baffle_v"]
    vps = v.generate_viewpoints(seed=1)
    assert len(vps) >= 4
    print(f"[任务层] blueprint 视点 {len(vps)} 个 ✓")

    # 标称复制决定论 + 扰动生效
    p = np.random.RandomState(0).uniform([0, -1, -1], [7, 1, 1], (1000, 3))
    v0 = v.perturbed_copy(seed=9)
    assert np.abs(v.clearance(p) - v0.clearance(p)).max() < 1e-12
    v1 = v.perturbed_copy(sigma_x=0.15, sigma_ang_deg=10, seed=9)
    assert np.abs(v.clearance(p) - v1.clearance(p)).max() > 0.01
    print("[任务层] 标称复制精确 / 扰动机制生效 ✓")


def test_planner():
    from envs.pipe_sim.geometry import PipeVessel
    from envs.pipe_sim.planner import plan_inspection_path

    ok_n, mins = 0, []
    for s in range(8):
        v = PipeVessel(layout="random", difficulty="medium", seed=200 + s)
        vps = v.generate_viewpoints(seed=200 + s)
        plan = plan_inspection_path(v, vps, margin=0.10, seed=200 + s)
        if plan["ok"]:
            ok_n += 1
            mins.append(float(v.clearance(plan["path"]).min()))
            assert all(a < b for a, b in zip(plan["vp_idx"], plan["vp_idx"][1:]))
    assert ok_n >= 7 and min(mins) > 0.07
    print(f"[全局层] RRT {ok_n}/8 成功, 路径 min-clr={min(mins):.3f} (>0.07, 无隧穿) ✓")


def test_pool():
    from envs.pipe_sim.pool import build_pool, instantiate

    pool = build_pool(n_train=16, n_eval=4, verbose=False)
    assert len(pool["train"]) == 16 and len(pool["eval"]) == 4
    st = {x["seed"] for x in pool["train"]}
    se = {x["seed"] for x in pool["eval"]}
    assert not (st & se)
    e = pool["train"][0]
    v = instantiate(e)
    assert float(v.clearance(e["path"]).min()) > 0.07
    print("[布局池] 构建/切分/重建决定论 ✓")


def test_backend():
    from envs.pipe_sim.backend import MockPipeInspectionTask, task_config

    env = MockPipeInspectionTask(task_config, seed=7, num_envs=2, device="cpu")
    obs = env.reset()
    assert obs["observations"].shape == (2, 15)
    ret = env.step(torch.tensor([[0.5, 0, 0, 0]] * 2, dtype=torch.float32))
    assert len(ret) == 5
    assert "curriculum_radius" in ret[4]
    print("[后端] AG五元组契约 + infos ✓")

    # 布局池抽样
    seeds = set()
    for _ in range(10):
        env.reset_idx(torch.tensor([0]))
        seeds.add(env.entries[0]["seed"])
    assert len(seeds) > 2
    print(f"[后端] 10次reset抽到 {len(seeds)} 个布局（池抽样修复单布局过拟合）✓")

    # 单调推进：倒退不产生负progress也不可重复收割
    env.reset_idx(torch.tensor([0]))
    i0 = int(env.path_idx[0])
    env._advance_path_idx(0, env.entries[0]["path"][min(i0 + 10, len(env.entries[0]["path"]) - 1)])
    i1 = int(env.path_idx[0])
    env._advance_path_idx(0, env.entries[0]["path"][i0])  # 试图倒退
    assert int(env.path_idx[0]) == i1 >= i0
    print("[后端] 弧长索引单调（推进奖励不可收割）✓")

    # 门控课程（v2.2 语义回归）
    r0 = env.current_robot_radius(0)
    for _ in range(task_config.curriculum_gate_window):
        env._gate_curriculum(0, 0.9)
    assert env.current_robot_radius(0) > r0
    print("[后端] 门控课程 ✓")


def test_adapter():
    import importlib
    import envs.pipe_inspection as pi
    import envs.wrappers as wrappers

    importlib.reload(pi)
    e = pi.PipeInspection("mock", (64, 64), seed=11, mode="eval")
    assert e._backend.cfg.layout == "blueprint"
    assert not e._backend.cfg.curriculum_enabled
    env = wrappers.SelectAction(wrappers.TimeLimit(
        wrappers.NormalizeActions(e), 320), key="action")
    obs = env.reset()
    assert obs["image"].shape == (64, 64, 3) and obs["state"].shape == (15,)
    for t in range(320):
        a = np.array([0.6, 0, 0, 0], dtype=np.float32)
        obs, r, done, info = env.step({"action": a})
        if done:
            break
    print(f"[适配] eval=blueprint / obs形状不变 / 盲飞终止于{t+1}步 ✓")


if __name__ == "__main__":
    test_geometry_and_task_layer()
    test_planner()
    test_pool()
    test_backend()
    test_adapter()
    print("\n🎯 v3 三层架构契约全部通过")
