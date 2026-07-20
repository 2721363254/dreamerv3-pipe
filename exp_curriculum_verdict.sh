#!/bin/bash
# 课程裁决实验（单卡串行）。用法: bash exp_curriculum_verdict.sh [GPU]
#
# 裁决问题：演示前提下，半径课程换来的"终段稳定性"是否真实、是否值得
# 换来的"收敛延迟"。用【可靠的 eval 协议】而非噪声曲线来裁。
#
# 三步串行：
#  1. run2 续跑 +200k（4e5→6e5），看 304k 后的稳定平台是否维持到 600k
#  2. 修复后的 eval_policy 对 run1（无课程）与 run2（有课程）的
#     best.pt 与 latest.pt 各测一次带 95%CI 的成功率
#  3. 汇总为对照表，落 JSON
#
# 前置：eval_policy.py 已修复（隐状态均值 + 逐回合种子 + CI）

set -eu
GPU="${1:-3}"
REPO=/mnt/sda/file-416-all-user/tmy/code/DreamerV3/dreamerv3-torch
cd "$REPO"
source /mnt/sda/file-416-all-user/tmy/tool/anaconda/etc/profile.d/conda.sh
conda activate dreamerv3
export CUDA_VISIBLE_DEVICES="$GPU"

RESULTS=results/curriculum_verdict
mkdir -p "$RESULTS"
echo "=== 课程裁决实验 | GPU=$GPU | $(date) ==="

# ---------- 步骤1: run2 续跑 +200k ----------
RUN2=logdir/pipe_v3_run2_curr
if [[ -f "$RUN2/latest.pt" ]]; then
    echo ">> [1/3] run2 续跑至 6e5 步（从 latest.pt 续训）"
    python dreamer.py --configs pipe_mock --task pipe_mock \
        --steps 6e5 --compile True --logdir "$RUN2" \
        2>&1 | tee -a "$RUN2/train_extend.log" | grep -E "eval_return|curriculum|best" || true
else
    echo ">> [1/3] 跳过：未找到 $RUN2/latest.pt（run2 logdir 名称请核对）"
fi

# ---------- 步骤2: 可靠评估两个 run 的 best/latest ----------
echo ">> [2/3] 带CI评估 run1/run2 的 best.pt 与 latest.pt"
for RUN in pipe_v3_run1 pipe_v3_run2_curr; do
    for CKPT in best latest; do
        P="logdir/$RUN/$CKPT.pt"
        if [[ -f "$P" ]]; then
            echo "---- $RUN / $CKPT ----"
            python eval_policy.py --ckpt "$P" --episodes 50 \
                --splits blueprint sparse medium dense \
                --json "$RESULTS/${RUN}_${CKPT}.json" \
                2>&1 | grep -E "split|blueprint|sparse|medium|dense" || true
        else
            echo "---- $RUN / $CKPT : 缺失 $P ----"
        fi
    done
done

# ---------- 步骤3: 汇总 ----------
echo ">> [3/3] 汇总对照表"
python - << 'PYEOF'
import json, glob, pathlib
rows = []
for f in sorted(glob.glob("results/curriculum_verdict/*.json")):
    d = json.load(open(f))
    tag = pathlib.Path(f).stem
    for split, r in d["results"].items():
        ci = r.get("success_ci", [0, 0])
        rows.append((tag, split, r["success"]*100, ci[0]*100, ci[1]*100))
print(f"\n{'run/ckpt':28s} {'split':10s} {'成功率':>7s} {'95%CI':>16s}")
for tag, split, sr, lo, hi in rows:
    print(f"{tag:28s} {split:10s} {sr:6.1f}% [{lo:5.1f},{hi:5.1f}]")
print("\n裁决要点:")
print(" - run1(无课程) vs run2(有课程) 的 best.pt 成功率CI是否重叠 → 课程是否真有增益")
print(" - 各 run 的 latest vs best 差距 → 终段是否退化（run1预期退化大，run2预期小）")
PYEOF
echo "=== 完成 $(date)，结果见 $RESULTS ==="
