"""给已生成的演示 npz 就地补 logprob 键（与智能体回合键集对齐）。
用法: python fix_demo_logprob.py logdir/pipe_v3_run1/train_eps"""
import io, sys, glob, pathlib
import numpy as np

d = pathlib.Path(sys.argv[1])
fixed = skipped = 0
for f in sorted(d.glob("*expert*.npz")):
    ep = dict(np.load(f))
    if "logprob" in ep:
        skipped += 1
        continue
    ep["logprob"] = np.zeros(len(ep["reward"]), dtype=np.float32)
    with io.BytesIO() as b:
        np.savez_compressed(b, **ep)
        b.seek(0)
        f.write_bytes(b.read())
    fixed += 1
print(f"补键 {fixed} 条, 已有跳过 {skipped} 条")
