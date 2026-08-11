"""Gate check: prove this Blackwell (sm_120) GPU actually runs PyTorch kernels.

Detecting the GPU is not the same as being able to compute on it. RTX 50-series
cards report fine under torch.cuda.is_available() while every kernel launch fails
with "no kernel image is available for execution on the device" when the wheel
was built without sm_120. So this script does real matmuls in each dtype we plan
to train in, and fails loudly rather than letting a broken stack reach training.
"""

import os
import sys

os.environ.setdefault("HF_HOME", r"D:\ToolCallFT\hf_cache")

import torch


def header(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def main() -> int:
    header("BUILD")
    print(f"python           {sys.version.split()[0]}")
    print(f"torch            {torch.__version__}")
    print(f"built for CUDA   {torch.version.cuda}")
    print(f"compiled archs   {torch.cuda.get_arch_list()}")

    if not torch.cuda.is_available():
        print("\nFAIL: torch.cuda.is_available() is False")
        return 1

    cap = torch.cuda.get_device_capability(0)
    cap_sm = f"sm_{cap[0]}{cap[1]}"
    props = torch.cuda.get_device_properties(0)

    header("DEVICE")
    print(f"name             {props.name}")
    print(f"capability       {cap_sm}")
    print(f"total VRAM       {props.total_memory / 1024**3:.2f} GiB")
    print(f"multiprocessors  {props.multi_processor_count}")

    if cap_sm not in torch.cuda.get_arch_list():
        print(
            f"\nFAIL: this wheel has no {cap_sm} kernels. "
            f"Reinstall from a cu128/cu129 index."
        )
        return 1

    # The real test. A wheel can list an arch and still fail here.
    header("KERNEL EXECUTION")
    ok = True
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
            b = torch.randn(2048, 2048, device="cuda", dtype=dtype)
            c = (a @ b).float()
            torch.cuda.synchronize()
            assert torch.isfinite(c).all(), "non-finite result"
            print(f"{str(dtype):<22} matmul OK   mean={c.mean().item():+.4f}")
        except Exception as exc:  # noqa: BLE001 - we want the raw failure text
            ok = False
            print(f"{str(dtype):<22} FAILED      {type(exc).__name__}: {exc}")
        finally:
            torch.cuda.empty_cache()

    # bf16 is what we train in; fp16-only would change the training config.
    header("TRAINING-RELEVANT SUPPORT")
    print(f"bf16 supported   {torch.cuda.is_bf16_supported()}")
    print(f"TF32 matmul      {torch.backends.cuda.matmul.allow_tf32}")
    try:
        from torch.backends.cuda import sdp_kernel  # noqa: F401

        print("SDPA available   True")
    except Exception:
        print("SDPA available   unknown")

    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM        {free / 1024**3:.2f} GiB of {total / 1024**3:.2f} GiB")

    if not ok:
        print("\nFAIL: at least one dtype could not execute on this GPU.")
        return 1

    print("\nPASS: GPU executes real kernels. Safe to proceed to training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
