"""Run from nanochat root: python -m scripts.assignment1_compare_tokenizers

Small, explicitly constructed diagnostic texts, not a representative benchmark.
Character count means Python len(text), including spaces and punctuation.
No BOS or other special tokens are added to measurement inputs.
"""
import hashlib
import json
from pathlib import Path

SAMPLES = {
    "english_everyday": "On Saturday morning, Maya walked to the market to buy vegetables and bread. The streets were quiet after a night of heavy rain. She stopped at a small cafe, opened her notebook, and made a plan for the coming week. Before leaving, she called a friend to ask whether they could meet for dinner.",
    "english_science": "A language model learns to predict the next token from the tokens that came before it. During training, its parameters are updated to reduce prediction error. A lower training loss does not necessarily mean better performance on unfamiliar data. Evaluation on a separate validation set helps us examine whether the model can generalize beyond the examples it has already seen.",
    "english_technical": "Reproducibility requires more than saving the final model. Researchers should record the dataset version, preprocessing choices, tokenizer configuration, and training settings. Computational resources also affect which experiments are practical. Comparing two systems fairly requires identical evaluation inputs and a clear description of any differences in the measurement procedure.",
    "numbers": "0 1 12 123 1234 12345 1234567890 2026-09-26 3.14159265 -42 1,000,000 10.50",
    "code": "def mean(values):\n    if not values:\n        return None\n    return sum(values) / len(values)\n\nprint(mean([1, 2, 3]))\n",
    "chinese": "今天我在学习语言模型。我们比较两个分词器，看看中文、数字和代码会被怎样切分。",
    "dutch": "Vandaag vergelijk ik twee tokenizers. Welke woorden worden opgesplitst, en hoeveel tokens hebben we nodig voor dezelfde tekst?",
    "unicode": "café naïve résumé 你好 🌍 👩‍💻",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer

    root = Path(get_base_dir()) / "assignment1"
    results = {
        "measurement": "Constructed diagnostic texts; no special tokens; chars = Unicode code points including whitespace; lower tokens/char means more compact encoding on this sample only",
        "samples": SAMPLES,
        "tokenizers": {},
    }
    sample_hashes = set()
    for vocab in (8192, 32768):
        folder = root / "tokenizers" / f"vocab_{vocab}"
        meta = json.loads((folder / "training_meta.json").read_text(encoding="utf-8"))
        assert meta["vocab_size_including_special_tokens"] == vocab
        assert meta["roundtrip_and_reload_passed"] is True
        assert sha256(folder / "tokenizer.pkl") == meta["tokenizer_sha256"], "Tokenizer file changed"
        sample_hashes.add(meta["sample_sha256"])
        tokenizer = RustBPETokenizer.from_directory(str(folder))
        assert tokenizer.get_vocab_size() == vocab
        token_bytes = torch.load(folder / "token_bytes.pt", map_location="cpu", weights_only=True)
        special_ids = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
        expected = torch.tensor([0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i)) for i in range(vocab)], dtype=torch.int32)
        assert torch.equal(token_bytes, expected), "Incorrect token_bytes.pt"
        rows = {}
        english_chars = english_tokens = english_bytes = 0
        for name, text in SAMPLES.items():
            ids = tokenizer.encode(text)
            assert tokenizer.decode(ids) == text, f"Round-trip failed: {name}"
            raw = [tokenizer.decode_single_token_bytes(i) for i in ids]
            assert b"".join(raw) == text.encode("utf-8")
            rows[name] = {
                "characters": len(text), "utf8_bytes": len(text.encode("utf-8")),
                "tokens": len(ids), "tokens_per_character": len(ids) / len(text),
                "characters_per_token": len(text) / len(ids),
                "token_ids": ids,
                "token_bytes_hex": [piece.hex() for piece in raw],
                "token_display": [repr(piece.decode("utf-8", errors="backslashreplace")) for piece in raw],
                "roundtrip_passed": True,
            }
            if name.startswith("english_"):
                english_chars += len(text)
                english_tokens += len(ids)
                english_bytes += len(text.encode("utf-8"))
        results["tokenizers"][str(vocab)] = {
            "training_seconds": meta["training_seconds"],
            "sample_sha256": meta["sample_sha256"],
            "english_aggregate": {
                "characters": english_chars, "utf8_bytes": english_bytes,
                "tokens": english_tokens, "tokens_per_character": english_tokens / english_chars,
            },
            "samples": rows,
        }
    assert len(sample_hashes) == 1, "The two tokenizers used different training samples"
    print("Both tokenizers verified; training sample SHA256 matches.")
    print(f"{'Sample':<23} {'Chars':>6} {'8K tokens':>10} {'32K tokens':>11}")
    for name, text in SAMPLES.items():
        a = results["tokenizers"]["8192"]["samples"][name]["tokens"]
        b = results["tokenizers"]["32768"]["samples"][name]["tokens"]
        print(f"{name:<23} {len(text):>6} {a:>10} {b:>11}")
    for vocab in (8192, 32768):
        row = results["tokenizers"][str(vocab)]["english_aggregate"]
        print(f"English aggregate, vocab={vocab}: {row['tokens']} tokens / {row['characters']} chars = {row['tokens_per_character']:.6f} tokens/char")
    path = root / "tokenizer_comparison.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison: {path}")
    print("Token byte displays preserve partial UTF-8 bytes; full decoded text round-trips correctly.")
    print("No language model training was started.")


if __name__ == "__main__":
    main()
