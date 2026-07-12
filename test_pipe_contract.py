"""管道环境三层契约测试：几何层 → 后端层（AG 契约） → DreamerV3 适配层。"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")


def test_geometry():
    from envs.pipe_sim.geometry import PipeVessel

    v = PipeVessel(seed=3)
    print(f"[几何] 支撑杆数量: {len(v.struts)}")

    # SDF 正确性：轴心在自由空间，壁外为负
    axis_pt = np.array([6.0, 0.0, 0.0])
    wall_pt = np.array([6.0, 1.2, 0.0])  # 半径1.0之外
    c_axis, c_wall = v.clearance(axis_pt), v.clearance(wall_pt)
    print(f"[几何] clearance 轴心={c_axis:.3f} (期望≈1.0)  壁外={c_wall:.3f} (期望<0)")
    assert 0.5 < c_axis <= 1.0 + 1e-6 and c_wall < 0

    # 深度渲染：中心像素应看到远处，边缘像素看到近壁
    depth = v.render_depth(np.array([0.8, 0, 0]), yaw=0.0)
    print(f"[几何] 深度图 {depth.shape} 范围 [{depth.min():.2f}, {depth.max():.2f}] m")
    assert depth.shape == (64, 64) and depth.min() > 0

    # 视点：全部满足安全裕度
    wps = v.generate_waypoints(seed=3)
    clr = np.array([v.clearance(w) for w in wps])
    print(f"[几何] 视点 {len(wps)} 个, 最小 clearance={clr.min():.3f} (>0.30 required)")
    assert len(wps) > 5 and clr.min() > 0.30
    print("✅ 几何层通过\n")
    return v, wps


def test_backend_contract():
    from envs.pipe_sim.backend import MockPipeInspectionTask, task_config

    N = 2
    env = MockPipeInspectionTask(task_config, seed=7, num_envs=N, device="cpu")

    # 契约1: reset 返回 task_obs 字典
    obs = env.reset()
    assert "observations" in obs and obs["observations"].shape == (N, 15)
    assert obs["depth_range_pixels"].shape == (N, 64, 64)
    print(f"[后端] reset → observations{tuple(obs['observations'].shape)} "
          f"depth{tuple(obs['depth_range_pixels'].shape)}")

    # 契约2: step 返回五元组，全为 torch 张量
    actions = torch.tensor([[0.5, 0.0, 0.0, 0.0]] * N)
    ret = env.step(actions)
    assert len(ret) == 5
    task_obs, rewards, terms, truncs, infos = ret
    for name, t in [("rewards", rewards), ("terminations", terms),
                    ("truncations", truncs)]:
        assert isinstance(t, torch.Tensor) and t.shape == (N,), name
    assert set(infos.keys()) >= {"successes", "crashes", "timeouts"}
    print(f"[后端] step → 5元组 ✓  reward[0]={rewards[0].item():.3f}")

    # 契约3: 动作变换到速度指令的量纲
    cmd = env.action_transformation_function(torch.ones(N, 4))
    assert torch.allclose(cmd[0], torch.tensor([1.0, 1.0, 0.5, 1.0]))
    print(f"[后端] action_transformation: [1,1,1,1] → {cmd[0].tolist()}")

    # 契约4: 前飞若干步应产生轴向位移和视点推进
    env.reset()
    reached = 0
    for t in range(300):
        # 朝当前视点的简单比例控制（贪心基线）
        o = env.task_obs["observations"]
        a = torch.zeros(N, 4)
        a[:, 0:3] = torch.clamp(o[:, 0:3] * 2.0, -1, 1)  # 沿单位向量
        _, r, tm, tc, inf = env.step(a)
        reached = max(reached, int(env.wp_idx.max().item()))
        if tm.any() or tc.any():
            pass  # 自动复位继续
    print(f"[后端] 贪心基线 300 步内最多推进到视点 #{reached}")
    assert reached >= 2, "贪心至少应能推进几个视点"

    # 契约5: 扰动场生效——同一悬停指令下位置漂移非零
    env.reset()
    p0 = env.pos.clone()
    for _ in range(30):
        env.step(torch.zeros(N, 4))
    drift = (env.pos - p0).norm(dim=-1)
    print(f"[后端] 悬停 30 步扰动漂移: {[f'{d:.3f}m' for d in drift.tolist()]}")
    assert drift.max() > 0.01, "扰动场应产生可观测漂移"
    print("✅ 后端层（Aerial Gym 契约）通过\n")


def test_dreamer_adapter():
    import envs.pipe_inspection as pipe_inspection
    import envs.wrappers as wrappers

    env = pipe_inspection.PipeInspection("mock", (64, 64), seed=11)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, 400)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    obs = env.reset()
    assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8
    assert obs["state"].shape == (15,) and obs["state"].dtype == np.float32
    assert obs["is_first"] == True
    print(f"[适配] reset → image{obs['image'].shape} uint8, state{obs['state'].shape}, "
          f"image 亮度范围 [{obs['image'].min()}, {obs['image'].max()}]")

    total, steps = 0.0, 0
    for t in range(450):
        a = np.zeros(4, dtype=np.float32)
        a[0] = 0.6  # 前飞
        obs, r, done, info = env.step({"action": a})
        total += r
        steps += 1
        if done:
            print(f"[适配] 回合结束于 {steps} 步, 累计奖励 {total:.2f}, "
                  f"is_terminal={obs['is_terminal']}, info={ {k:v for k,v in info.items() if k!='discount'} }")
            break
    assert steps < 450, "盲目前飞应在撞支撑杆或超时前结束"
    print("✅ DreamerV3 适配层（含 wrapper 链）通过\n")


if __name__ == "__main__":
    test_geometry()
    test_backend_contract()
    test_dreamer_adapter()
    print("🎯 三层契约全部通过：Windows mock 后端与 Aerial Gym API 对齐，"
          "DreamerV3 侧可无缝接入")
