#!/bin/bash
# 训练启动脚本：10万步自动结束，抗终端断开，断点自动续训。
#
# 用法:
#   bash train.sh              # GPU 3, 运行名 pipe_v3_run1
#   bash train.sh 2            # GPU 2
#   bash train.sh 3 my_run2    # GPU 3, 自定义运行名（决定 logdir）
#
# 断点续训: 中断后重跑完全相同的命令即可 —— dreamer.py 检测到
#   logdir/latest.pt 会自动加载权重与优化器状态，回放缓冲从
#   logdir/train_eps/*.npz 自动重载。
# 检查点粒度: 每 eval_every(1万步)保存一次，崩溃最多回退1万步。
#
# 停止: kill $(cat logdir/<run>/train.pid)

set -u
GPU="${1:-3}"
RUN="${2:-pipe_v3_run1}"

REPO=/mnt/sda/file-416-all-user/tmy/code/DreamerV3/dreamerv3-torch
LOGDIR="$REPO/logdir/$RUN"
PIDFILE="$LOGDIR/train.pid"
LOGFILE="$LOGDIR/train.log"

# 环境（显式路径，免疫 /opt/anaconda3 污染）
source /mnt/sda/file-416-all-user/tmy/tool/anaconda/etc/profile.d/conda.sh
conda activate dreamerv3
cd "$REPO"

# 防重复启动
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "已有训练在运行 (PID $(cat "$PIDFILE"))，logdir=$LOGDIR"
    echo "查看进度: tail -f $LOGFILE"
    exit 1
fi

mkdir -p "$LOGDIR"
if [[ -f "$LOGDIR/latest.pt" ]]; then
    echo ">> 检测到 latest.pt —— 断点续训（若上次已跑满步数上限，"
    echo ">>   进程会立即正常退出；提高命令中 --steps 可继续加练）"
else
    echo ">> 全新训练"
fi

echo ">> GPU=$GPU  运行名=$RUN  日志=$LOGFILE"
nohup env CUDA_VISIBLE_DEVICES="$GPU" python dreamer.py \
    --configs pipe_mock --task pipe_mock \
    --steps 4e5 --compile True \
    --logdir "$LOGDIR" \
    >> "$LOGFILE" 2>&1 &

echo $! > "$PIDFILE"
echo ">> 已启动 (PID $!)，终端断开不影响运行。跑满10万步自动结束。"
echo ""
echo "   监控命令:"
echo "     tail -f $LOGFILE"
echo "     tensorboard --logdir $REPO/logdir --port 6006"
echo "     nvidia-smi"
echo "   结束后指标文件: $LOGDIR/metrics.jsonl"
