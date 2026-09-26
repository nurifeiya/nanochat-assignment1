"""Prepare a fixed ~500 MB UTF-8 text sample; does not train a tokenizer.

Place in nanochat's scripts/ and run from the repository root:
    python -m scripts.assignment1_prepare_sample
Outputs live under NANOCHAT_BASE_DIR/assignment1, outside the code repository.
MB means 1,000,000 bytes of document text, excluding JSON formatting.
The final document is truncated at a valid UTF-8 boundary (at most 3 bytes short).
No per-document character cap is applied. Validation data is never included.
"""

import hashlib
import json
import os
import platform
from pathlib import Path


def utf8_prefix(text, budget):
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text, len(raw), False
    raw = raw[:budget].decode("utf-8", errors="ignore").encode("utf-8")
    return raw.decode("utf-8"), len(raw), True


def main():
    import pyarrow.parquet as pq
    from nanochat.common import get_base_dir
    from nanochat.dataset import DATA_DIR, MAX_SHARD, index_to_filename

    target = 500_000_000
    # Explicit sources: these are the two training shards already downloaded.
    sources = [Path(DATA_DIR) / index_to_filename(i) for i in (0, 1)]
    assert all(p.name != index_to_filename(MAX_SHARD) for p in sources)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Missing training shard: {path}")
    folder = Path(get_base_dir()) / "assignment1"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "tokenizer_sample_500mb.jsonl"
    manifest = folder / "tokenizer_sample_500mb.meta.json"
    if output.exists() or manifest.exists():
        raise FileExistsError("Sample or metadata already exists; not overwriting. Ask for verification before rerunning.")

    temporary = output.with_suffix(".jsonl.tmp")
    count = nbytes = nchars = 0
    digest = hashlib.sha256()
    finished = truncated = False
    used_sources = []
    try:
        with temporary.open("xb") as stream:
            for path in sources:
                print(f"Reading {path.name} ...", flush=True)
                used_sources.append(path.name)
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str):
                            raise TypeError("Non-string document found")
                        if not text:
                            continue
                        text, size, truncated = utf8_prefix(text, target - nbytes)
                        if size:
                            record = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
                            stream.write(record)
                            digest.update(record)
                            count += 1
                            nbytes += size
                            nchars += len(text)
                        if truncated or nbytes == target:
                            finished = True
                            break
                    if finished:
                        break
                if finished:
                    break
        if not finished or not (target - 3 <= nbytes <= target):
            raise RuntimeError(f"Insufficient text: {nbytes:,} bytes; target {target:,}")
        metadata = {
            "dataset": "karpathy/climbmix-400b-shuffle",
            "source_shards": used_sources,
            "selection": "First nonempty documents in shard 0 then shard 1; final document UTF-8 truncated",
            "validation_excluded": True,
            "document_character_cap": None,
            "target_text_bytes": target,
            "actual_text_bytes": nbytes,
            "characters": nchars,
            "documents": count,
            "last_document_truncated": truncated,
            "sample_file_bytes": temporary.stat().st_size,
            "sample_sha256": digest.hexdigest(),
            "python_version": platform.python_version(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        os.replace(temporary, output)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except Exception:
        # Keep partial files for diagnosis; never treat them as a completed sample.
        print("Preparation failed. Send the error before retrying.", flush=True)
        raise
    print(f"Sample ready: {output}")
    print(f"Metadata: {manifest}")
    print(f"Documents: {count:,}")
    print(f"Text bytes: {nbytes:,} ({nbytes / 1_000_000:.6f} MB)")
    print(f"SHA256: {digest.hexdigest()}")
    print("No tokenizer training was started.")


if __name__ == "__main__":
    main()
