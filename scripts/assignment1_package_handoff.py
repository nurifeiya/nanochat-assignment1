"""Package the confirmed final model for the teammate, without training or deleting files.
Run in nanochat-master: python -m scripts.assignment1_package_handoff
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


README = '''# Nanochat Assignment 1: base-model handoff

This is a pretrained base model, not an instruction-tuned assistant.
Depth 2; hidden dimension 128; 1 attention head; vocabulary 8192; context 512.
3,538,986 parameters; 35,000 steps; 71,680,000 presented training tokens.
Trained on Apple M4/MPS with float32 and torch.compile disabled.
Final fixed-subset train bpb: 1.445140; validation bpb: 1.513907.
Each evaluation subset has 32,768 packed target tokens. Greedy samples repeat
and fail the tested factual/reasoning/code tasks; see evidence/ for raw outputs.

## Contents
- model_data/base_checkpoints/d2_8k/model_035000.pt: final model weights.
- model_data/base_checkpoints/d2_8k/meta_035000.json: configuration and metadata.
- model_data/tokenizer/: the matching 8192-token tokenizer and byte counts.
- code/: project Python code, tasks, run scripts, and dependency files.
- evidence/: training metrics/configuration/log, plus five raw completions if present.
- SHA256.json: integrity checksums of every other included file.

## First load (Mac or CPU)
Use the included code version; do not replace it with the newest upstream code.
From the extracted folder, enter code/ in your terminal, then:

    uv sync --extra cpu
    source .venv/bin/activate
    export NANOCHAT_BASE_DIR="$(cd ../model_data && pwd)"
    export NANOCHAT_DTYPE=float32
    export TORCH_COMPILE_DISABLE=1
    python verify_handoff.py

The check loads on CPU to be portable. It does not train or change weights.
On NVIDIA hardware, choose the appropriate CUDA environment for actual training.
Keep NANOCHAT_BASE_DIR pointing to the extracted model_data directory.

## Task 3 starting point
The exact base checkpoint can be loaded with:

    from nanochat.checkpoint_manager import load_model
    model, tokenizer, meta = load_model(
        "base", device, phase="train", model_tag="d2_8k", step=35000)

Here device is a torch.device chosen for your hardware. Set environment variables
before importing nanochat modules. Use the supplied tokenizer for every stage;
do not retrain, replace it, or select the 32768-token variant.

Implement Stage 1 with MMLU + GSM-8K only; Stage 2 with SmolTalk only,
loading Stage 1 weights. Keep stage output folders distinct to preserve both.
Evaluate the base model and both subsequent stages on ARC-Easy,
ARC-Challenge and GSM-8K under matched settings. These benchmark scores have
NOT been produced yet; the evidence here is not a substitute for benchmarks.
Default SFT settings have not been sized for the teammate's hardware.

The pretraining optimizer state and training dataset are intentionally omitted:
they are not needed to begin a new fine-tuning stage. Retain the originals on
Fiya's machine for resuming pretraining. Fine-tuning datasets must be obtained
separately through the project workflow.

The shared code repository and report (including AI-use disclosure) still need
to be submitted. This ZIP is a handoff, not a replacement for those deliverables.
'''

VERIFY = '''import os
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
from pathlib import Path
import torch
from nanochat.checkpoint_manager import load_model
os.environ.setdefault("NANOCHAT_BASE_DIR", str(Path(__file__).resolve().parent.parent / "model_data"))
model, tokenizer, meta = load_model("base", torch.device("cpu"), phase="eval", model_tag="d2_8k", step=35000)
assert meta["step"] == 35000
assert tokenizer.get_vocab_size() == 8192
assert sum(p.numel() for p in model.parameters()) == 3538986
text = "Hello world!"
assert tokenizer.decode(tokenizer.encode(text)) == text
with torch.no_grad():
    ids = torch.tensor([tokenizer.encode(text, prepend="<|bos|>")])
    assert torch.isfinite(model(ids)).all()
print("HANDOFF LOAD CHECK PASSED: step 35000, vocab 8192, finite forward pass")
'''


def main():
    repo = Path.cwd()
    if not (repo / "nanochat/gpt.py").is_file():
        raise RuntimeError("Run from the nanochat-master project root")
    run = Path.home() / ".cache/nanochat/assignment1/pretraining_runs/d2_8k_mibnmsfu"
    cfg = json.loads((run / "run_config.json").read_text())
    # Verify the project version used to train before distributing code.
    for rel, expected in cfg["source_hashes"].items():
        if hashlib.sha256((repo / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Code differs from the training version: {rel}. Stop and check before packaging.")
    files = {}
    for name in ("model_035000.pt", "meta_035000.json"):
        files[f"model_data/base_checkpoints/d2_8k/{name}"] = run / "base_checkpoints/d2_8k" / name
    for name in ("tokenizer.pkl", "token_bytes.pt", "training_meta.json"):
        files[f"model_data/tokenizer/{name}"] = run / "tokenizer" / name
    if hashlib.sha256(files["model_data/tokenizer/tokenizer.pkl"].read_bytes()).hexdigest() != cfg["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer checksum mismatch")
    meta = json.loads(files["model_data/base_checkpoints/d2_8k/meta_035000.json"].read_text())
    assert meta["step"] == 35000 and meta["model_config"]["vocab_size"] == 8192
    for name in ("metrics.jsonl", "run_config.json", "run_summary.json", "training.log"):
        files[f"evidence/{name}"] = run / name
    for name in ("base_completions_step35000.json", "base_completions_step35000.txt"):
        files[f"evidence/{name}"] = run / "analysis" / name
    # Source-only project directories; exclude caches and hidden files.
    extensions = {".py", ".json", ".sh", ".md", ".html", ".css", ".js", ".txt", ".toml", ".yaml", ".yml", ".jinja", ".jinja2"}
    for directory in ("nanochat", "scripts", "tasks", "runs"):
        for path in (repo / directory).rglob("*"):
            rel = path.relative_to(repo)
            if path.is_file() and not path.is_symlink() and path.suffix in extensions and not any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
                files["code/" + rel.as_posix()] = path
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "LICENSE.md"):
        if (repo / name).is_file():
            files[f"code/{name}"] = repo / name
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    destination = Path(tempfile.mkdtemp(prefix="nanochat_handoff_", dir=Path.home() / "Desktop"))
    archive = destination / "nanochat_A_handoff_d2_8k.zip"
    prefix = "nanochat_A_handoff/"
    checksums = {}
    print("Packaging final checkpoint, tokenizer, source and evidence ...", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for rel, path in sorted(files.items()):
            data = path.read_bytes()
            z.writestr(prefix + rel, data)
            checksums[rel] = hashlib.sha256(data).hexdigest()
        for rel, text in (("HANDOFF_README.md", README), ("code/verify_handoff.py", VERIFY)):
            data = text.encode("utf-8")
            z.writestr(prefix + rel, data)
            checksums[rel] = hashlib.sha256(data).hexdigest()
        z.writestr(prefix + "SHA256.json", json.dumps(checksums, indent=2) + "\n")
    with zipfile.ZipFile(archive) as z:
        for rel, expected in checksums.items():
            if hashlib.sha256(z.read(prefix + rel)).hexdigest() != expected:
                raise RuntimeError(f"Archive verification failed: {rel}")
    print("HANDOFF PACKAGE READY")
    print(f"ZIP: {archive}")
    print(f"Size: {archive.stat().st_size / 1_000_000:.2f} MB")
    print(f"Verified {len(checksums)} files. Original model and data unchanged.")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(archive)], check=False)


if __name__ == "__main__":
    main()
