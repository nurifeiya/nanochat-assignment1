# Task 4 Experimental Evidence

## Model
- Checkpoint: `assignment1_sft_checkpoints/d2_8k/model_216770.pt`
- Depth: 2
- Attention heads: 1
- KV heads: 1
- Embedding dimension: 128
- Vocabulary size: 8192
- Sequence length: 2048

## Temperature experiment
Temperatures tested:
- 0.1
- 0.7
- 1.5

Prompts:
1. What is the capital of France?
2. What is 2 plus 2?
3. Explain what a computer is in simple terms.
4. Write a short story about a cat.
5. Give me three tips for studying effectively.

## Evidence files
- `temperature_france.txt`
- `representative_examples.txt`

## Status
- Temperature sampling experiment: COMPLETE
- Five representative prompt categories: COMPLETE
- Inference server / Web UI: TODO
- Inference loop and KV-cache analysis: TODO
- Production hardening proposals: TODO
- Public-model benchmark comparison: TODO
- Reflection: TODO
