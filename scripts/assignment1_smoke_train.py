"""30-step MPS smoke test using a copy of the user's official base_train.py.

Run from nanochat root: python -m scripts.assignment1_smoke_train
Only code change in the copied trainer: synchronize MPS for correct step timing.
Compilation is disabled through PyTorch's environment setting before import.
This is a compatibility/timing test, not the final assignment training run.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time


def main():
    repo = Path.cwd()
    trainer = repo / "scripts/base_train.py"
    if not trainer.is_file():
        raise RuntimeError("Run this command from the nanochat-master project root.")
    base = Path(os.environ.get("NANOCHAT_BASE_DIR") or Path.home() / ".cache/nanochat").expanduser().resolve()
    tokenizer = base / "assignment1/tokenizers/vocab_8192"
    for name in ("tokenizer.pkl", "token_bytes.pt", "training_meta.json"):
        if not (tokenizer / name).is_file():
            raise FileNotFoundError(tokenizer / name)
    meta = json.loads((tokenizer / "training_meta.json").read_text())
    assert meta["roundtrip_and_reload_passed"] is True
    assert hashlib.sha256((tokenizer / "tokenizer.pkl").read_bytes()).hexdigest() == meta["tokenizer_sha256"]
    data = base / "base_data_climbmix"
    for name in ("shard_00000.parquet", "shard_00001.parquet", "shard_06542.parquet"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)

    source = trainer.read_text(encoding="utf-8")
    old = 'synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None'
    new = 'synchronize = torch.cuda.synchronize if device_type == "cuda" else (torch.mps.synchronize if device_type == "mps" else lambda: None)'
    if source.count(old) != 1:
        raise RuntimeError("base_train.py differs from the inspected version; send the file before proceeding.")
    parent = base / "assignment1/smoke_runs"
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="d2_8k_", dir=parent))
    (run_dir / "tokenizer").symlink_to(tokenizer, target_is_directory=True)
    (run_dir / "base_data_climbmix").symlink_to(data, target_is_directory=True)
    copied = run_dir / "base_train_smoke.py"
    copied.write_text(source.replace(old, new), encoding="utf-8")
    args = [
        "--device-type=mps", "--depth=2", "--head-dim=128", "--window-pattern=L",
        "--max-seq-len=512", "--device-batch-size=2", "--total-batch-size=2048",
        "--num-iterations=30", "--warmup-steps=5", "--eval-every=15",
        "--eval-tokens=2048", "--core-metric-every=-1", "--sample-every=-1",
        "--model-tag=smoke_d2_8k", "--run=dummy",
    ]
    env = os.environ.copy()
    env.update(NANOCHAT_BASE_DIR=str(run_dir), NANOCHAT_DTYPE="float32",
               TORCH_COMPILE_DISABLE="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Do not silently accept unsupported operations by falling back to CPU.
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    config = {"arguments": args, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "sample_sha256": meta["sample_sha256"], "tokenizer_sha256": meta["tokenizer_sha256"],
              "compile_disabled": True, "dtype": "float32", "purpose": "smoke test only",
              "modification": "MPS synchronization for timing", "run_directory": str(run_dir)}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [sys.executable, "-u", str(copied), *args]
    print(f"30-step test. Output directory: {run_dir}", flush=True)
    print("Tokenizer: 8192; model: depth 2; device: MPS; dtype: float32", flush=True)
    print("Terminal output is also saved to training.log", flush=True)
    records = []
    start = time.perf_counter()
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                match = re.search(r"step (\d+)/\d+ .*?\| dt: ([\d.]+)ms", line)
                if match:
                    records.append({"step": int(match[1]), "seconds": float(match[2]) / 1000})
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            print(f"Stopped. Diagnostic log: {run_dir / 'training.log'}")
            raise
    steady = [r["seconds"] for r in records if r["step"] >= 10]
    summary = {"returncode": returncode, "wall_seconds": time.perf_counter() - start,
               "completed_steps": len(records), "steps": records,
               "mean_seconds_per_step_after_first_10": statistics.mean(steady) if steady else None,
               "note": "Step timings include data loading and MPS sync, exclude evaluation and checkpoint overhead. Only comparable at the same model, batch and sequence settings."}
    (run_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nCompleted training steps: {len(records)}/30")
    print(f"Total wall time: {summary['wall_seconds']:.1f} seconds")
    if steady:
        avg = statistics.mean(steady)
        print(f"Mean step time after first 10 steps: {avg:.4f} seconds")
        print(f"Steady training throughput: {2048 / avg:.0f} tokens/second")
    print(f"Results and log: {run_dir}")
    if returncode != 0:
        print("Test failed. Send the error; do not start a full run.")
        raise SystemExit(returncode)
    if len(records) != 30:
        raise RuntimeError("Unexpected step count; inspect the log.")
    print("SMOKE TEST PASSED. This checkpoint is for testing, not the final handoff.")


if __name__ == "__main__":
    main()
