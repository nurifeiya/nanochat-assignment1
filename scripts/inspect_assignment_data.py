import json
from pathlib import Path

from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    datasets = [
        (
            "MMLU",
            "cais/mmlu",
            "auxiliary_train",
            lambda: MMLU(subset="all", split="auxiliary_train"),
        ),
        (
            "GSM8K",
            "openai/gsm8k",
            "train",
            lambda: GSM8K(subset="main", split="train"),
        ),
        (
            "SmolTalk",
            "HuggingFaceTB/smol-smoltalk",
            "train",
            lambda: SmolTalk(split="train"),
        ),
    ]

    for name, source, split, create_dataset in datasets:
        print(f"\nLoading {name} ...", flush=True)
        dataset = create_dataset()

        result = {
            "dataset": source,
            "split": split,
            "number_of_examples": len(dataset),
            "samples": [
                {
                    "index_after_shuffle": i,
                    "raw_row": dataset.ds[i],
                    "training_conversation": dataset[i],
                }
                for i in range(min(2, len(dataset)))
            ],
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))

        output_path = output_dir / f"{name.lower()}_inspection.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved to: {output_path}", flush=True)
        del dataset


if __name__ == "__main__":
    main()