# nanochat Assignment 1

This repository contains our implementation and experimental results for Assignment 1 based on [nanochat](https://github.com/karpathy/nanochat).

## Repository Structure

### Task 0 — Model Hub Warm-up
Model inspection and comparison are documented in the final report.

### Task 1 — Tokenization
Main scripts:

- `scripts/assignment1_prepare_sample.py` — prepares the tokenizer training sample.
- `scripts/assignment1_train_tokenizer.py` — trains the BPE tokenizer.
- `scripts/assignment1_compare_tokenizers.py` — compares the 8K and 32K tokenizers.

### Task 2 — Pre-Training
Main scripts:

- `scripts/assignment1_pretrain.py` — pre-training pipeline.
- `scripts/assignment1_smoke_train.py` — short training sanity check.
- `scripts/assignment1_generate_completions.py` — generates qualitative examples from the pretrained model.
- `scripts/assignment1_fix_mps_optimizer.py` — Apple MPS compatibility support.
- `scripts/assignment1_package_handoff.py` — packages the pretrained checkpoint for handoff.
- `verify_handoff.py` — verifies the handoff package.

Checkpoint:

- `checkpoints/pretrained/model_035000.pt` — pretrained checkpoint at step 35,000.
- `checkpoints/pretrained/meta_035000.json` — checkpoint metadata.

### Task 3 — Mid-Training and Supervised Fine-Tuning
Main implementation:

- `scripts/assignment1_sft.py` — two-stage mid-training and SFT pipeline.

Results:

- `results/task3_benchmark_results.csv` — ARC-Easy, ARC-Challenge, and GSM8K benchmark results across training stages.

Checkpoints:

- `checkpoints/midtrain/model_067455.pt` — final mid-training checkpoint at step 67,455.
- `checkpoints/midtrain/meta_067455.json` — mid-training checkpoint metadata.
- `checkpoints/sft/model_216770.pt` — final SFT checkpoint at step 216,770.
- `checkpoints/sft/meta_216770.json` — final SFT checkpoint metadata.

### Task 4A — Inference and Deployment
Main implementation:

- `scripts/chat_cli.py` — command-line inference.
- `scripts/chat_web.py` — web inference server.
- `nanochat/ui.html` — WebUI.

Experimental outputs:

- `runs/task4/representative_examples.txt` — representative generations at different temperatures.
- `runs/task4/temperature_france.txt` — temperature comparison example.
- `runs/task4/gsm8k_diagnostic.txt` — GSM8K diagnostic generations.
- `runs/task4/inference_analysis.md` — inference experiment notes and analysis.
- `runs/task4/webui_evidence/` — WebUI screenshot and server log.
- `runs/task4/README.md` — additional notes for Task 4.

### Task 4B — Critical Reflection
The critical reflection is included in the final report.

## Reproducing Tasks 1 and 2 (Apple Silicon)

Run the commands below from the repository root. The recorded experiment used Python 3.10, an Apple M4 with 16 GB unified memory, and PyTorch MPS in float32. These assignment launchers target MPS; they are not generic CUDA/CPU training commands.

### Environment and data

```bash
uv sync --extra cpu
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

On the recorded Mac installation, the CPU dependency extra was used and MPS was available. Confirm the check prints `True` before using the MPS launchers. The MPS optimizer fix is already included in `nanochat/optim.py`.

```bash
python -m nanochat.dataset -n 2 -w 2
```

This downloads training shards 0 and 1 and validation shard 6542. Confirm all three downloads succeed. Outputs use `~/.cache/nanochat` by default; `NANOCHAT_BASE_DIR` can select another base directory. The commands below assume the default location and a fresh reproduction environment.

### Task 1: train and compare tokenizers

```bash
python -m scripts.assignment1_prepare_sample
python -m scripts.assignment1_train_tokenizer --vocab-size 8192
python -m scripts.assignment1_train_tokenizer --vocab-size 32768
python -m scripts.assignment1_compare_tokenizers
```

The sample contains 500,000,000 bytes of document text (JSON formatting excluded), selected in order from the training shards, with the last document truncated at a valid UTF-8 boundary. Validation data is excluded. Both tokenizers use the same sample. Sample preparation refuses to overwrite existing output; an existing completed experiment does not need to be rerun.

Recorded sample-file SHA-256:

```text
96f2dc98afdedff2cbd9715e2ec949fea7bb7f0289ddaae077aab3c0c9de014b
```

Tokenizers are written under `~/.cache/nanochat/assignment1/tokenizers/vocab_8192/` and `vocab_32768/`. Comparison output is `~/.cache/nanochat/assignment1/tokenizer_comparison.json`.

Committed evidence:

- [Comparison results](results/task1/tokenizer_comparison.json)
- [Sample metadata](results/task1/tokenizer_sample_500mb.meta.json)
- [8K training metadata](results/task1/training_meta_8192.json)
- [32K training metadata](results/task1/training_meta_32768.json)

On three English diagnostic passages (1,053 characters), the token counts were 221 for 8K and 192 for 32K: a 13.1% reduction. These are small diagnostic samples, not a corpus-wide benchmark.

### Task 2: check before full pretraining

First run the 100-update check:

```bash
python -m scripts.assignment1_pretrain --mode check
```

Inspect its log and confirm `LOGGING CHECK PASSED`. Only when intending to reproduce the full experiment, run:

```bash
python -m scripts.assignment1_pretrain --mode train
```

Each launch creates a new directory and starts from scratch; it does not resume a previous checkpoint. The launcher disables compilation, uses float32, and records its actual configuration. Full runs are stored under `~/.cache/nanochat/assignment1/pretraining_runs/d2_8k_*`.

| Setting | Recorded value |
| --- | --- |
| Layers / hidden width | 2 / 128 |
| Query heads / KV heads | 1 / 1 |
| Vocabulary / context length | 8,192 / 512 |
| Trainable parameters | 3,538,986 |
| Sequences per microbatch / gradient accumulation | 2 / 2 |
| Tokens per update / updates | 2,048 / 35,000 |
| Processed training tokens, including repeats | 71,680,000 |
| Evaluation interval | 500 updates, including step 0 |
| Fixed evaluation subset per split | 32,768 packed tokens |
| Final train / validation bpb | 1.4451 / 1.5139 |
| Recorded wall time | 2,068.4 seconds (about 34.5 minutes) |

Runtime is an observation from this run, not a guarantee for another environment. The final checkpoint is `base_checkpoints/d2_8k/model_035000.pt` inside the run directory.

Committed evidence:

- [Run configuration and source hashes](results/task2/run_config.json)
- [Training and evaluation metrics](results/task2/metrics.jsonl)
- [Run summary](results/task2/run_summary.json)
- [Final checkpoint metadata](results/task2/meta_035000.json)
- [Five raw completions](results/task2/base_completions_step35000.json)

To regenerate the five completions, replace `/absolute/path/to/completed_run` with the completed run directory printed by the training launcher:

```bash
python -m scripts.assignment1_generate_completions --run-dir /absolute/path/to/completed_run
```

This loads step 35,000 and its matching tokenizer, checks source/tokenizer hashes, and uses greedy decoding with at most 96 new tokens. It does not retrain the model. Output is written to `analysis/base_completions_step35000.json` in that run directory.

### Reproduction scope

Saved JSON files contain historical absolute paths from the experiment machine; use your own run directory when reproducing. The training source files in this submission match the hashes recorded in `results/task2/run_config.json`, except for the dependency lockfile `uv.lock`. Consequently, the current lockfile should not be described as an exact copy of the original training environment. The original run's source snapshot includes its original lockfile and should be retained with the checkpoint handoff. Hardware and dependency differences can affect runtime and numerical results.

## Checkpoint downloads and handoff

**Pretrained package download link: pending.** The model was handed to the teammate, but an assessor-accessible download link still needs to be added here. Mid-training and SFT checkpoint links must also be supplied by the teammate.

The pretrained model must be loaded with the matching **8K tokenizer**, not the 32K tokenizer. A usable package includes the final weights, checkpoint metadata, `tokenizer.pkl`, `token_bytes.pt`, and the code/environment information needed to load them. The JSON checkpoint metadata committed in `results/task2/` does not contain model weights.

## Authors

- Nurifeiya Anwaier
- Binghao Guo
