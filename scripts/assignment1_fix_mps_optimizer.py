"""Fix CPU scalar tensor arguments in eager MPS optimizer operations.

Run from nanochat root: python -m scripts.assignment1_fix_mps_optimizer
Keeps the optimizer algorithm; uses Python scalars for MPS scalar arguments.
CUDA/CPU branches remain unchanged. Backs up original before writing.
"""
import ast
from pathlib import Path
import shutil

MARKER = "# Assignment 1: Python scalars for eager MPS optimizer operations."


def patched_source(source):
    if MARKER in source:
        return source
    replacements = [
        (
            "    p32 = p.float()",
            "    " + MARKER + "\n"
            "    # These scalar tensors live on CPU for the compiled CUDA path.\n"
            "    # Eager MPS lerp_ requires a same-device tensor or a Python number.\n"
            "    if p.device.type == 'mps':\n"
            "        step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t = (\n"
            "            scalar.item() for scalar in (step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)\n"
            "        )\n"
            "    p32 = p.float()",
        ),
        (
            "    momentum = momentum_t.to(stacked_grads.dtype)",
            "    momentum = momentum_t.item() if stacked_grads.device.type == 'mps' else momentum_t.to(stacked_grads.dtype)",
        ),
        (
            "    beta2 = beta2_t.to(g.dtype)",
            "    beta2 = beta2_t.item() if g.device.type == 'mps' else beta2_t.to(g.dtype)",
        ),
        (
            "    lr = lr_t.to(g.dtype)",
            "    lr = lr_t.item() if g.device.type == 'mps' else lr_t.to(g.dtype)",
        ),
        (
            "    wd = wd_t.to(g.dtype)",
            "    wd = wd_t.item() if g.device.type == 'mps' else wd_t.to(g.dtype)",
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Source differs from inspected version; no file changed. Expected one: {old.strip()}")
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def main():
    path = Path("nanochat/optim.py")
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print("MPS scalar fix is already installed; no changes made.")
        return
    updated = patched_source(source)
    backup = path.with_name("optim.py.before_assignment1_mps_fix")
    if backup.exists():
        raise FileExistsError(f"Backup already exists; no changes made: {backup}")
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    print(f"Backup saved: {backup}")
    print(f"MPS scalar fix installed: {path}")
    print("Python syntax check passed. MPS runtime must be verified by the 30-step test.")


if __name__ == "__main__":
    main()
