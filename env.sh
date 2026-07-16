#!/bin/bash
# 一键环境配置。用法（必须 source，不能直接执行）：
#   source env.sh        # 默认 GPU 3
#   source env.sh 2      # 指定 GPU 2
#   source env.sh 2,3    # 多卡
# 放在仓库根目录，开新终端后第一件事就是 source 它。

# 防呆：检测直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "错误: 请用 source env.sh 而不是 ./env.sh（环境变量需注入当前shell）"
    exit 1
fi

# 1. 激活自己的 conda（显式路径，免疫 /opt/anaconda3 的 PATH 污染）
source /mnt/sda/file-416-all-user/tmy/tool/anaconda/etc/profile.d/conda.sh
conda activate dreamerv3

# 2. GPU 指定（默认 3，可传参覆盖）
export CUDA_VISIBLE_DEVICES=${1:-3}

# 3. 进入仓库目录
cd /mnt/sda/file-416-all-user/tmy/code/DreamerV3/dreamerv3-torch

# 4. 自检输出（which python 是防环境错位的哨兵）
echo "── env.sh ─────────────────────────────────"
echo " python : $(which python)"
echo " GPU 可见: $CUDA_VISIBLE_DEVICES"
echo " 当前显卡占用:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/   GPU /'
echo "───────────────────────────────────────────"

# 期望 python 路径: /mnt/sda/.../tool/anaconda/envs/dreamerv3/bin/python
# 若显示 /opt/anaconda3/... 说明环境错位，检查 ~/.bashrc 末尾的 PATH 反制行
