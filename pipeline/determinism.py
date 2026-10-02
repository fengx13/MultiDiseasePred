"""
determinism.py -- put every run in the same known state.

Call seed_everything() as the first line of every pipeline script. It fixes
the four things that made the previous code impossible to reproduce, and one
that only shows up on this cluster's A100s.

1. Weight initialisation and dropout were never seeded. The old notebook
   defines SEED = 42 and passes it to random_split and to the bootstrap, but
   never calls torch.manual_seed, so the network started from different
   weights on every run.

2. numpy and the stdlib random module were likewise unseeded.

3. DataLoader worker processes get their own seeds unless told otherwise.

4. cuDNN picks convolution algorithms by benchmarking, which is nondeterministic
   in itself. Irrelevant for an MLP but free to switch off.

5. TF32. On Ampere, PyTorch may compute float32 matrix multiplies on tensor
   cores in TF32, which keeps only 10 mantissa bits. Measured on this A100,
   that changes results by roughly a thousand times more than plain float32
   rounding. For a 13-expert, 256-wide network the speed it buys is
   negligible, so it is disabled here in exchange for results that reproduce
   and that match a CPU reference.

Anything the paper reports should come from a run that called this.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True,
                    verbose: bool = True) -> dict:
    """Seed every generator this project touches. Returns what it set, so the
    caller can drop it straight into config.json."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    state = {
        "seed": seed,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "deterministic": deterministic,
    }

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # cuBLAS needs this to make reductions reproducible run to run
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    state["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    state["cudnn_deterministic"] = torch.backends.cudnn.deterministic

    if verbose:
        print(f"[determinism] seed={seed} torch={torch.__version__} "
              f"tf32={state['tf32_matmul']} "
              f"cudnn_deterministic={state['cudnn_deterministic']}")

    return state


def dataloader_kwargs(seed: int = 42, num_workers: int = 2) -> dict:
    """Keyword arguments that make a DataLoader reproducible.

    Pass as: DataLoader(ds, batch_size=256, shuffle=True, **dataloader_kwargs())
    """
    def worker_init(worker_id):
        s = seed + worker_id
        np.random.seed(s)
        random.seed(s)

    g = torch.Generator()
    g.manual_seed(seed)
    return {"num_workers": num_workers, "worker_init_fn": worker_init,
            "generator": g, "persistent_workers": num_workers > 0}


def run_fingerprint(extra: dict | None = None) -> dict:
    """Everything worth recording alongside a result."""
    import platform
    import subprocess
    fp = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "hostname": platform.node(),
    }
    try:
        import sklearn
        fp["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        fp["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        fp["git_commit"] = None
    if extra:
        fp.update(extra)
    return fp
