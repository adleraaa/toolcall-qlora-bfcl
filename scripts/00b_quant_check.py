"""Gate check 2: prove bitsandbytes NF4 kernels run on this Blackwell GPU.

QLoRA lives or dies on bitsandbytes, and bnb ships its own CUDA kernels compiled
per-arch -- a torch build with sm_120 says nothing about whether bnb has it. On
Blackwell + Windows this is the most likely thing to break, so test it here with
a 30-second synthetic instead of discovering it 20 minutes into a training run
after downloading a model.

Tests the full path that training actually uses: quantize a Linear to NF4, run a
forward, and backprop into a LoRA-shaped fp32 side branch (the frozen NF4 weight
itself takes no gradient -- that is the whole point of QLoRA).
"""

import os

os.environ.setdefault("HF_HOME", r"D:\ToolCallFT\hf_cache")

import torch


def main() -> int:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")

    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes {bnb.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot import bitsandbytes: {type(exc).__name__}: {exc}")
        return 1

    from bitsandbytes.nn import Linear4bit

    in_f, out_f, batch = 2048, 2048, 8
    torch.manual_seed(0)

    print("\n--- NF4 forward ---")
    try:
        layer = Linear4bit(
            in_f,
            out_f,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            compress_statistics=True,
        )
        layer.weight.data = torch.randn(out_f, in_f, dtype=torch.float16)
        layer = layer.cuda()

        x = torch.randn(batch, in_f, device="cuda", dtype=torch.bfloat16)
        y = layer(x)
        torch.cuda.synchronize()
        assert torch.isfinite(y).all(), "non-finite output"
        print(f"OK   out {tuple(y.shape)} {y.dtype}  mean={y.float().mean().item():+.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: NF4 forward: {type(exc).__name__}: {exc}")
        return 1

    print("\n--- backward through a LoRA-shaped branch ---")
    try:
        r = 16
        a = torch.zeros(r, in_f, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        b = torch.zeros(out_f, r, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.nn.init.kaiming_uniform_(a, a=5**0.5)
        torch.nn.init.normal_(b, std=0.01)

        out = layer(x) + (x @ a.T) @ b.T
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()

        assert a.grad is not None and torch.isfinite(a.grad).all(), "bad grad on A"
        assert b.grad is not None and torch.isfinite(b.grad).all(), "bad grad on B"
        print(f"OK   loss={loss.item():.4f}  |gradA|={a.grad.norm().item():.4e}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: backward: {type(exc).__name__}: {exc}")
        return 1

    print("\n--- paged 8-bit optimizer ---")
    try:
        opt = bnb.optim.PagedAdamW8bit([a, b], lr=1e-4)
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        print("OK   PagedAdamW8bit stepped")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: paged optimizer: {type(exc).__name__}: {exc}")
        print("      (training can fall back to adamw_torch, at higher VRAM cost)")
        return 1

    free, total = torch.cuda.mem_get_info()
    print(f"\nfree VRAM {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
    print("PASS: QLoRA path is functional on this GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
