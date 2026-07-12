import gym
import numpy as np


class MockEnv:
    """轻量 mock 环境，接口与 crafter 完全对齐（image 观测 + 离散动作）。

    目的：在不依赖真实/重型仿真器的情况下，快速验证 DreamerV3 训练全流程
    （采数据 → 训世界模型 → 想象里训 actor-critic → 日志/checkpoint）。

    内部实现是一个「可学习」的网格世界：
      - 智能体（红块）在 grid×grid 的格子上移动，目标（绿块）随机放置；
      - 每步动作让智能体上/下/左/右移动或不动；
      - 靠近目标给正奖励，够到目标 reward=1 并终止（is_terminal=True）。
    整个格子世界渲染成 (H, W, 3) uint8 图像作为观测。

    这样世界模型的图像重建、奖励预测、cont 预测以及 actor-critic 都有
    「真实、确定、可优化」的信号——因此它是一个有效的冒烟测试，
    如果各项 loss 会下降、train_return 会上升，就说明训练管线是通的。

    接口刻意与 envs/crafter.py 保持一致，因此上层的模型结构、状态空间、
    动作空间配置都无需改动，直接复用现有的 image + onehot 离散动作路径。
    """

    metadata = {}

    def __init__(self, task="reward", size=(64, 64), grid=8, seed=0):
        assert task in ("reward", "noreward")
        self._size = tuple(size)
        self._grid = int(grid)
        self._reward = task == "reward"
        self._rng = np.random.RandomState(seed)
        self._agent = None
        self._goal = None
        # 5 个离散动作：上 / 下 / 左 / 右 / 不动
        self._num_actions = 5
        self.reward_range = [-1.0, 1.0]

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0, 255, (*self._size, 3), dtype=np.uint8
                ),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            }
        )

    @property
    def action_space(self):
        space = gym.spaces.Discrete(self._num_actions)
        # OneHotAction wrapper 依赖 Discrete；额外标记 discrete 供上层识别
        space.discrete = True
        return space

    def _render(self):
        h, w = self._size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cell_h = max(1, h // self._grid)
        cell_w = max(1, w // self._grid)

        def draw(pos, color):
            y, x = int(pos[0]), int(pos[1])
            y0, x0 = y * cell_h, x * cell_w
            img[y0 : y0 + cell_h, x0 : x0 + cell_w] = color

        # 背景轻微灰色，便于区分边界
        img[:] = (20, 20, 30)
        draw(self._goal, (40, 200, 60))  # 目标：绿
        draw(self._agent, (220, 40, 40))  # 智能体：红
        return img

    def reset(self):
        self._agent = self._rng.randint(0, self._grid, size=2)
        # 保证目标与初始位置不同，避免第一步就终止
        while True:
            self._goal = self._rng.randint(0, self._grid, size=2)
            if not np.array_equal(self._agent, self._goal):
                break
        obs = {
            "image": self._render(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs

    def step(self, action):
        moves = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
            4: (0, 0),   # 不动
        }
        dy, dx = moves[int(action)]
        prev = self._agent.copy()
        self._agent = np.clip(
            self._agent + np.array([dy, dx]), 0, self._grid - 1
        )

        prev_dist = np.abs(prev - self._goal).sum()
        cur_dist = np.abs(self._agent - self._goal).sum()
        reached = bool(np.array_equal(self._agent, self._goal))

        if self._reward:
            if reached:
                reward = 1.0
            else:
                # 势能型奖励：靠近目标 +，远离 -，每步有微弱信号
                reward = 0.1 * float(prev_dist - cur_dist)
        else:
            reward = 0.0

        done = reached
        # discount == 0 表示真正的终止（够到目标）；超时由 TimeLimit 处理
        info = {"discount": np.float32(0.0 if reached else 1.0)}

        obs = {
            "image": self._render(),
            "is_first": False,
            "is_last": done,
            "is_terminal": reached,
        }
        return obs, np.float32(reward), done, info

    def render(self):
        return self._render()
