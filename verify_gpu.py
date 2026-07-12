import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device name:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("compute capability:", cap)

# 真正执行一次算子,确认 Blackwell kernel 在跑(而非静默回退)
x = torch.randn(2000, 2000, device="cuda")
y = (x @ x).sum().item()
print("matmul result:", y)

assert cap == (12, 0), "算力号不是 sm_120,torch 没认出 Blackwell"
print("✅ GPU 就绪")