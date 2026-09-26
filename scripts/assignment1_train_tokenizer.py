"""Train one assignment tokenizer on the fixed sample, without overwriting others.

Run from nanochat root after placing this file in scripts/:
python -m scripts.assignment1_train_tokenizer --vocab-size 8192
python -m scripts.assignment1_train_tokenizer --vocab-size 32768
No default/active tokenizer is replaced. No language model training is started.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, choices=[8192, 32768], required=True)
    args = parser.parse_args()
    import torch
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN

    root = Path(get_base_dir()) / "assignment1"
    sample = root / "tokenizer_sample_500mb.jsonl"
    meta_path = root / "tokenizer_sample_500mb.meta.json"
    output = root / "tokenizers" / f"vocab_{args.vocab_size}"
    if output.exists():
        raise FileExistsError(f"Output already exists; not overwriting: {output}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    print("Checking sample SHA256 ...", flush=True)
    checksum = file_sha256(sample)
    if checksum != metadata["sample_sha256"]:
        raise ValueError("Sample SHA256 differs from metadata. Stop and check the files.")
    if not 499_999_997 <= metadata["actual_text_bytes"] <= 500_000_000:
        raise ValueError("Unexpected sample text size")
    counts = {"documents": 0, "bytes": 0, "characters": 0}

    def texts():
        with sample.open("r", encoding="utf-8") as stream:
            for line in stream:
                text = json.loads(line)["text"]
                if not isinstance(text, str):
                    raise TypeError("Sample document is not text")
                counts["documents"] += 1
                counts["bytes"] += len(text.encode("utf-8"))
                counts["characters"] += len(text)
                yield text

    print(f"Sample verified. Training vocabulary {args.vocab_size:,} on CPU ...", flush=True)
    start = time.perf_counter()
    tokenizer = RustBPETokenizer.train_from_iterator(texts(), args.vocab_size)
    seconds = time.perf_counter() - start
    assert counts["documents"] == metadata["documents"], "Not all sample documents consumed"
    assert counts["bytes"] == metadata["actual_text_bytes"], "Text byte count mismatch"
    assert counts["characters"] == metadata["characters"], "Character count mismatch"
    assert tokenizer.get_vocab_size() == args.vocab_size, "Unexpected vocabulary size"
    checks = [
        "Hello world! This is a tokenizer test.",
        "Numbers: 1234567890, 3.14159, -42.",
        "def square(x):\n    return x * x\n",
        "你好世界！🌍 café Nederlands ئۇيغۇرچە",
    ]
    for text in checks:
        assert tokenizer.decode(tokenizer.encode(text)) == text, "Round-trip failed"
    special_ids = {tokenizer.encode_special(s) for s in SPECIAL_TOKENS}
    token_bytes = torch.tensor([
        0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
        for i in range(args.vocab_size)
    ], dtype=torch.int32, device="cpu")

    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save(str(output))
    torch.save(token_bytes, output / "token_bytes.pt")
    reloaded = RustBPETokenizer.from_directory(str(output))
    for text in checks:
        assert reloaded.encode(text) == tokenizer.encode(text), "Reloaded tokenizer differs"
    saved_bytes = torch.load(output / "token_bytes.pt", map_location="cpu", weights_only=True)
    assert torch.equal(token_bytes, saved_bytes), "Reloaded token byte counts differ"
    versions = {}
    for name in ["rustbpe", "tiktoken", "torch"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    run = {
        "vocab_size_including_special_tokens": args.vocab_size,
        "special_token_count": len(SPECIAL_TOKENS),
        "special_tokens": SPECIAL_TOKENS,
        "split_pattern": SPLIT_PATTERN,
        "training_seconds": seconds,
        "sample_sha256": checksum,
        "sample_metadata": metadata,
        "consumed": counts,
        "document_character_cap": None,
        "roundtrip_and_reload_passed": True,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "script_sha256": file_sha256(Path(__file__)),
        "tokenizer_sha256": file_sha256(output / "tokenizer.pkl"),
    }
    (output / "training_meta.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Training successful. Vocabulary: {args.vocab_size:,}")
    print(f"Training time: {seconds:.2f} seconds")
    print(f"Consumed text bytes: {counts['bytes']:,}")
    print("Round-trip and saved-file reload checks: PASSED")
    print(f"Saved to: {output}")
    print("Files: tokenizer.pkl, token_bytes.pt, training_meta.json")
    print("Default tokenizer unchanged. No language model training started.")


if __name__ == "__main__":
    main()
