"""PipeInspection —— DreamerV3 侧环境适配器。

把 Aerial Gym 契约的后端（五元组、批量 torch 张量）适配成
dreamerv3-torch 的单环境 gym 接口（字典观测、numpy）：

  观测键:
    image    : (64,64,3) uint8 —— 深度图编码为 3 通道
               （近亮远暗；复制 3 通道以复用现有 CNN 编码器，零模型改动）
    state    : (15,) float32  —— 向量观测（视点相对量/速度/进度/clearance）
               经 configs 里 mlp_keys: 'state' 走 MLP 编码器 → 多模态早融合
    is_first / is_last / is_terminal

  动作: Box(-1,1,(4,)) 连续 —— 经 wrappers.NormalizeActions 接入，
        actor 用 normal 分布、imag_gradient 用 dynamics（连续控制路径）。

后端可替换：Windows 上用 MockPipeInspectionTask；A100 阶段换成包装
真实 Aerial Gym NavigationTask 的同契约类，本文件与 DreamerV3 零改动。
"""

import gym
import numpy as np
import torch

from .pipe_sim.backend import MockPipeInspectionTask, task_config, copy_config


class PipeInspection:
    metadata = {}

    def __init__(self, task="mock", size=(64, 64), seed=0, mode="train",
                 overrides=None):
        assert task in ("mock",), "真实 Aerial Gym 后端在 A100 阶段接入"
        cfg = copy_config(task_config)    # 实例副本，不再污染共享类
        cfg.camera_size = tuple(size)
        # 训练环境用随机布局（泛化），评估环境用图纸复刻（固定场景）
        cfg.layout = "random" if mode == "train" else "blueprint"
        if mode != "train":
            cfg.curriculum_enabled = False  # 评估始终用真实机体半径
        if overrides:
            for k, v in overrides.items():
                assert hasattr(cfg, k), f"未知配置项: {k}"
                setattr(cfg, k, v)
        self._size = tuple(size)
        self._backend = MockPipeInspectionTask(
            task_config=cfg, seed=seed, num_envs=1, device="cpu"
        )
        self._obs_dim = cfg.observation_space_dim
        self._max_range = cfg.camera_max_range
        self._done = True

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0, 255, (*self._size, 3), dtype=np.uint8
                ),
                "state": gym.spaces.Box(
                    -np.inf, np.inf, (self._obs_dim,), dtype=np.float32
                ),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def _depth_to_image(self, depth):
        """深度(米) → uint8 3 通道图。近处亮、远处暗，
        与'黑暗管道内主动传感'的物理直觉一致。"""
        d = np.clip(depth / self._max_range, 0.0, 1.0)
        img = ((1.0 - d) * 255).astype(np.uint8)
        return np.repeat(img[..., None], 3, axis=-1)

    def _extract_obs(self, task_obs, is_first):
        depth = task_obs["depth_range_pixels"][0].cpu().numpy()
        state = task_obs["observations"][0].cpu().numpy().astype(np.float32)
        return {
            "image": self._depth_to_image(depth),
            "state": state,
            "is_first": is_first,
            "is_last": False,
            "is_terminal": False,
        }

    def reset(self):
        task_obs = self._backend.reset()
        self._done = False
        return self._extract_obs(task_obs, is_first=True)

    def step(self, action):
        a = torch.tensor(
            np.asarray(action, dtype=np.float32)[None], dtype=torch.float32
        )
        task_obs, rewards, terms, truncs, infos = self._backend.step(a)
        reward = float(rewards[0].item())
        terminated = bool(terms[0].item() > 0)
        truncated = bool(truncs[0].item() > 0)
        done = terminated or truncated

        obs = self._extract_obs(task_obs, is_first=False)
        obs["is_last"] = done
        # is_terminal 语义：真终止（撞击/完成）→ discount 0；超时截断 → 1
        obs["is_terminal"] = terminated
        info = {"discount": np.float32(0.0 if terminated else 1.0)}
        if terminated or truncated:
            info["success"] = bool(infos["successes"][0].item() > 0)
            info["crash"] = bool(infos["crashes"][0].item() > 0)
        return obs, np.float32(reward), done, info

    def render(self):
        depth = self._backend.task_obs["depth_range_pixels"][0].cpu().numpy()
        return self._depth_to_image(depth)
