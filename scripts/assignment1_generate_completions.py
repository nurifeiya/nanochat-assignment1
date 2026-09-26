"""Five predeclared raw completions from the final base model, no system prompt.

Place in scripts/ and run from nanochat root:
    python -m scripts.assignment1_generate_completions
Uses the confirmed run d2_8k_mibnmsfu, step 35000. No training is performed.
Greedy decoding (temperature 0), at most 96 new tokens, no KV cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

PROMPTS = [
    ("narrative", "When the rain stopped, the young woman opened the door and"),
    ("science", "Water freezes when"),
    ("geography", "The capital city of the Netherlands is"),
    ("arithmetic", "There are three apples on the table. I add two more apples. Now there are"),
    ("code", "def add_numbers(a, b):\n    "),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    base = Path(os.environ.get("NANOCHAT_BASE_DIR") or Path.home() / ".cache/nanochat").expanduser()
    run = (args.run_dir or base / "assignment1/pretraining_runs/d2_8k_mibnmsfu").expanduser().resolve()
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    if config["mode"] != "train":
        raise ValueError("Expected the completed formal training run")
    checkpoint = run / "base_checkpoints/d2_8k"
    model_file = checkpoint / "model_035000.pt"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    for rel in ("nanochat/gpt.py", "nanochat/tokenizer.py", "nanochat/common.py", "nanochat/flash_attention.py", "nanochat/checkpoint_manager.py"):
        expected = config["source_hashes"].get(rel)
        if expected and hashlib.sha256(Path(rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source differs from training: {rel}. Stop and check the version.")
    if hashlib.sha256((run / "tokenizer/tokenizer.pkl").read_bytes()).hexdigest() != config["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer checksum mismatch")
    os.environ["NANOCHAT_BASE_DIR"] = str(run)
    os.environ["NANOCHAT_DTYPE"] = "float32"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch
    from nanochat.checkpoint_manager import build_model
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this Python environment")
    device = torch.device("mps")
    torch.manual_seed(42)
    print("Loading final base model, step 35000 ...", flush=True)
    model, tokenizer, metadata = build_model(str(checkpoint), 35000, device, phase="eval")
    assert metadata["step"] == 35000
    assert sum(p.numel() for p in model.parameters()) == 3_538_986
    max_new_tokens = 96
    results = []
    for index, (category, prompt) in enumerate(PROMPTS, start=1):
        ids = tokenizer.encode(prompt, prepend="<|bos|>")
        if len(ids) < 2 or len(ids) + max_new_tokens > model.config.sequence_len:
            raise ValueError("Prompt length outside supported generation limits")
        torch.mps.synchronize()
        start = time.perf_counter()
        output = []
        reason = "max_new_tokens"
        # Native generate() with temperature=0 avoids MPS-specific RNG support.
        for token_id in model.generate(ids, max_tokens=max_new_tokens, temperature=0.0):
            if token_id == tokenizer.get_bos_token_id():
                reason = "document_boundary_bos"
                break
            output.append(token_id)
        torch.mps.synchronize()
        completion = tokenizer.decode(output)
        results.append(dict(category=category, prompt=prompt, prompt_token_ids=ids,
                            completion=completion, generated_token_ids=output,
                            generated_tokens=len(output), stop_reason=reason,
                            generation_seconds=time.perf_counter() - start))
        print(f"\nExample {index}/5 ({category})\nPROMPT: {prompt}\nCOMPLETION: {completion}\n", flush=True)
    report = dict(checkpoint_step=35000, model_tag="d2_8k", model_config=metadata["model_config"],
                  model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                  tokenizer_sha256=config["tokenizer_sha256"],
                  decoding=dict(temperature=0.0, method="greedy", max_new_tokens=max_new_tokens,
                                system_prompt=None, prompt_format="BOS + raw text", use_kv_cache=False,
                                stop_on="BOS (document boundary) or token limit"),
                  note="Five predeclared diagnostic prompts; all outputs retained without cherry-picking. Not benchmark scores.",
                  examples=results)
    folder = run / "analysis"
    folder.mkdir(exist_ok=True)
    json_path = folder / "base_completions_step35000.json"
    text_path = folder / "base_completions_step35000.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text("Raw base-model completions | step 35000 | temperature 0 | max 96 new tokens\n\n" +
        "\n\n".join(f"Example {i}: {r['category']}\nPROMPT:\n{r['prompt']}\nCOMPLETION:\n{r['completion']}\nStop: {r['stop_reason']}"
                      for i, r in enumerate(results, 1)) + "\n", encoding="utf-8")
    print("FIVE COMPLETIONS SAVED")
    print(f"JSON: {json_path}")
    print(f"Text: {text_path}")
    print("No model weights or training checkpoints were changed.")


if __name__ == "__main__":
    main()
