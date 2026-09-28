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

## Authors

- Nurifeiya Anwaier
- Binghao Guo
