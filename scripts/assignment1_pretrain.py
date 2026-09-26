"""Assignment pretraining launcher, with fixed train/validation bpb evaluation.

From nanochat root:
  python -m scripts.assignment1_pretrain --mode check
  python -m scripts.assignment1_pretrain --mode train

Check: 100 steps. Train: 35,000 steps (71,680,000 presented target tokens).
Each launch creates a new run; it never resumes the smoke-test model.
Source changes are written to a reviewable copy, leaving base_train.py unchanged.
The MPS optimizer fix must already be installed in nanochat/optim.py.
Metrics use fixed, small packed subsets of train/validation, not the entire sets.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


HELPERS = r'''
# Assignment additions: structured logging and matched fixed-subset evaluation.
_a1_start_time = time.perf_counter()
_a1_eval_cache = {}

def _a1_record(kind, **values):
    row = dict(kind=kind, wall_seconds=time.perf_counter() - _a1_start_time, **values)
    path = os.path.join(get_base_dir(), "metrics.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")

def _a1_loss_sums(losses, targets, byte_lengths):
    # Loss is in nats. CE includes special targets; bpb excludes them.
    ordinary_losses, all_losses, represented_bytes = [], [], 0
    for loss, target in zip(losses, targets):
        if target < 0:
            continue
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss")
        all_losses.append(loss)
        size = byte_lengths[target]
        if size > 0:
            ordinary_losses.append(loss)
            represented_bytes += size
    return math.fsum(all_losses), len(all_losses), math.fsum(ordinary_losses), represented_bytes

def _a1_cached_batches(split):
    if split not in _a1_eval_cache:
        steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
        if steps < 1:
            raise ValueError("Evaluation budget is smaller than one batch")
        iterator = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len,
            split=split, device="cpu")
        cache = []
        for _ in range(steps):
            x_eval, y_eval = next(iterator)
            # The loader reuses its buffers: clones are essential here.
            cache.append((x_eval.clone(), y_eval.clone()))
        _a1_eval_cache[split] = cache
        torch.save(_a1_eval_cache, os.path.join(get_base_dir(), "evaluation_batches.pt"))
    return _a1_eval_cache[split]

@torch.no_grad()
def _a1_evaluate(model_eval, split, token_bytes_eval):
    byte_lengths = token_bytes_eval.detach().cpu().tolist()
    total_loss = total_ordinary_loss = 0.0
    total_tokens = total_bytes = 0
    for x_cpu, y_cpu in _a1_cached_batches(split):
        losses = model_eval(x_cpu.to(device), y_cpu.to(device), loss_reduction="none")
        sums = _a1_loss_sums(losses.detach().float().cpu().reshape(-1).tolist(),
                             y_cpu.reshape(-1).tolist(), byte_lengths)
        total_loss += sums[0]
        total_tokens += sums[1]
        total_ordinary_loss += sums[2]
        total_bytes += sums[3]
    if total_bytes <= 0 or total_tokens <= 0:
        raise RuntimeError("Empty evaluation sample")
    return dict(loss=total_loss / total_tokens,
                bpb=total_ordinary_loss / (math.log(2) * total_bytes),
                evaluated_tokens=total_tokens, evaluated_bytes=total_bytes)
'''

EVALUATION = '''    # Evaluate identical cached packed subsets at each measurement point.
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        eval_started = time.perf_counter()
        model.eval()
        with disable_fp8(orig_model):
            train_metrics = _a1_evaluate(orig_model, "train", token_bytes)
            val_metrics = _a1_evaluate(orig_model, "val", token_bytes)
            if step == 0 and os.environ.get("ASSIGNMENT1_CHECK") == "1":
                reference_batches = ((a.to(device), b.to(device)) for a, b in _a1_cached_batches("val"))
                reference_bpb = evaluate_bpb(orig_model, reference_batches, len(_a1_cached_batches("val")), token_bytes)
                if not math.isclose(reference_bpb, val_metrics["bpb"], rel_tol=2e-5, abs_tol=2e-5):
                    raise RuntimeError(f"bpb reference mismatch: {reference_bpb} vs {val_metrics['bpb']}")
                print0("BPB REFERENCE CHECK PASSED")
        val_bpb = val_metrics["bpb"]
        min_val_bpb = min(min_val_bpb, val_bpb)
        _a1_record("evaluation", step=step, tokens_seen=step * total_batch_size,
                   train=train_metrics, val=val_metrics,
                   evaluation_seconds=time.perf_counter() - eval_started)
        print0(f"Step {step:05d} | Train bpb: {train_metrics['bpb']:.6f} | Validation bpb: {val_bpb:.6f}")
        wandb_run.log({"step": step, "train/bpb": train_metrics["bpb"], "val/bpb": val_bpb,
                       "train/eval_loss": train_metrics["loss"], "val/loss": val_metrics["loss"]})
        model.train()

'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Trainer differs from inspected version; no training started. Missing/duplicate: " + old[:100])
    return source.replace(old, new, 1)


def adapt_trainer(source):
    source = replace_once(source, "print_banner()\n", "print_banner()\n" + HELPERS)
    source = replace_once(source,
        'synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None',
        'synchronize = torch.cuda.synchronize if device_type == "cuda" else (torch.mps.synchronize if device_type == "mps" else lambda: None)')
    start = source.index("    # once in a while: evaluate the val bpb (all ranks participate)")
    end = source.index("    # once in a while: estimate the CORE metric (all ranks participate)", start)
    source = source[:start] + EVALUATION + source[end:]
    source = replace_once(source, "# Learning rate schedule (linear warmup, constant, linear warmdown)",
        '_a1_record("configuration", model=model_config_kwargs, parameter_counts=param_counts, '
        'arguments=user_config, total_batch_size=total_batch_size, num_iterations=num_iterations, '
        'total_tokens=total_tokens, dtype=str(COMPUTE_DTYPE), device=str(device), '
        'evaluation_scope="Fixed packed subsets; not complete train/validation corpora")\n\n'
        '# Learning rate schedule (linear warmup, constant, linear warmdown)')
    source = replace_once(source, "    for micro_step in range(grad_accum_steps):",
        "    _a1_batch_loss = torch.zeros((), device=device)\n    for micro_step in range(grad_accum_steps):")
    source = replace_once(source, "        train_loss = loss.detach() # for logging",
        "        train_loss = loss.detach() # for logging\n        _a1_batch_loss += train_loss / grad_accum_steps")
    source = replace_once(source, "    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point",
        "    train_loss_f = _a1_batch_loss.item() # Mean CE over all gradient accumulation microbatches\n"
        "    if not math.isfinite(train_loss_f):\n        raise RuntimeError('Non-finite training loss; stopping')")
    source = replace_once(source, "    # state update\n",
        '    if (step + 1) % 10 == 0 or step + 1 == num_iterations:\n'
        '        _a1_record("training", step=step + 1, tokens_seen=(step + 1) * total_batch_size, '
        'loss=train_loss_f, smoothed_loss=debiased_smooth_loss, seconds_per_step=dt, tokens_per_second=tok_per_sec)\n\n'
        '    # state update\n')
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "train"], default="check")
    options = parser.parse_args()
    repo = Path.cwd()
    source_path = repo / "scripts/base_train.py"
    source = source_path.read_text(encoding="utf-8")
    optimizer = (repo / "nanochat/optim.py").read_text(encoding="utf-8")
    if "# Assignment 1: Python scalars for eager MPS optimizer operations." not in optimizer:
        raise RuntimeError("Install the MPS optimizer fix first.")
    modified = adapt_trainer(source)
    base = Path(os.environ.get("NANOCHAT_BASE_DIR") or Path.home() / ".cache/nanochat").expanduser().resolve()
    tokenizer = base / "assignment1/tokenizers/vocab_8192"
    meta = json.loads((tokenizer / "training_meta.json").read_text())
    assert meta["roundtrip_and_reload_passed"] is True
    assert hashlib.sha256((tokenizer / "tokenizer.pkl").read_bytes()).hexdigest() == meta["tokenizer_sha256"]
    assert (tokenizer / "token_bytes.pt").is_file()
    data = base / "base_data_climbmix"
    for name in ("shard_00000.parquet", "shard_00001.parquet", "shard_06542.parquet"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    check = options.mode == "check"
    parent = base / "assignment1" / ("logging_checks" if check else "pretraining_runs")
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="d2_8k_", dir=parent))
    shutil.copytree(tokenizer, run_dir / "tokenizer")
    # Pin exactly the two training shards and the validation shard.
    run_data = run_dir / "base_data_climbmix"
    run_data.mkdir()
    for name in ("shard_00000.parquet", "shard_00001.parquet", "shard_06542.parquet"):
        (run_data / name).symlink_to(data / name)
    copied = run_dir / "base_train_assignment.py"
    copied.write_text(modified, encoding="utf-8")
    snapshot = run_dir / "source_snapshot"
    code_hashes = {}
    for rel in ("scripts/base_train.py", "scripts/base_eval.py", "nanochat/gpt.py", "nanochat/optim.py", "nanochat/common.py", "nanochat/dataset.py", "nanochat/dataloader.py", "nanochat/tokenizer.py", "nanochat/loss_eval.py", "nanochat/flash_attention.py", "nanochat/checkpoint_manager.py", "nanochat/engine.py", "pyproject.toml", "uv.lock"):
        path = repo / rel
        if path.is_file():
            destination = snapshot / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            code_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__), run_dir / "launcher_snapshot.py")
    iterations = 100 if check else 35000
    args = ["--device-type=mps", "--depth=2", "--head-dim=128", "--window-pattern=L",
            "--max-seq-len=512", "--device-batch-size=2", "--total-batch-size=2048",
            f"--num-iterations={iterations}", "--warmup-steps=40",
            f"--eval-every={50 if check else 500}", f"--eval-tokens={4096 if check else 32768}",
            f"--save-every={50 if check else 5000}", "--core-metric-every=-1", "--sample-every=-1",
            "--model-tag=d2_8k", "--run=dummy"]
    env = os.environ.copy()
    env.update(NANOCHAT_BASE_DIR=str(run_dir), NANOCHAT_DTYPE="float32", TORCH_COMPILE_DISABLE="1",
               PYTHONUNBUFFERED="1", ASSIGNMENT1_CHECK="1" if check else "0")
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    config = dict(mode=options.mode, arguments=args, run_directory=str(run_dir), source_hashes=code_hashes,
                  sample_sha256=meta["sample_sha256"], tokenizer_sha256=meta["tokenizer_sha256"],
                  compile_disabled=True, dtype="float32", trained_from_scratch=True,
                  changes=["MPS synchronized timing", "MPS optimizer scalar fix", "fixed train/validation bpb and CE", "JSONL logging", "train CE averaged across microbatches"])
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Mode: {options.mode}. Steps: {iterations:,}. Output: {run_dir}", flush=True)
    start = time.perf_counter()
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", str(copied), *args], cwd=repo, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", start_new_session=True)
        inhibitor = None
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            inhibitor = subprocess.Popen(["caffeinate", "-i", "-w", str(process.pid)])
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            print(f"Stopped. Existing checkpoints and logs remain in {run_dir}")
            raise
        finally:
            if inhibitor is not None:
                inhibitor.terminate()
                inhibitor.wait()
    elapsed = time.perf_counter() - start
    summary = dict(returncode=result, wall_seconds=elapsed, run_directory=str(run_dir))
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nOutput directory: {run_dir}")
    if result != 0:
        raise SystemExit(result)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evaluations = [r for r in rows if r["kind"] == "evaluation"]
    assert evaluations[0]["step"] == 0 and evaluations[-1]["step"] == iterations
    assert any(r["kind"] == "training" and r["step"] == iterations for r in rows)
    final_checkpoint = run_dir / "base_checkpoints/d2_8k" / f"model_{iterations:06d}.pt"
    assert final_checkpoint.is_file(), "Final checkpoint missing"
    if check:
        assert (run_dir / "base_checkpoints/d2_8k/model_000050.pt").is_file()
        assert "BPB REFERENCE CHECK PASSED" in (run_dir / "training.log").read_text()
    print(f"Wall time: {elapsed:.1f} seconds; evaluation points: {len(evaluations)}")
    for row in (evaluations[0], evaluations[-1]):
        print(f"Step {row['step']}: train bpb={row['train']['bpb']:.6f}, val bpb={row['val']['bpb']:.6f}")
    print("Metrics: metrics.jsonl; terminal log: training.log")
    print("LOGGING CHECK PASSED" if check else "PRETRAINING COMPLETE")


if __name__ == "__main__":
    main()
