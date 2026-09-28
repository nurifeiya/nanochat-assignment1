# Task 4 — Inference Loop and KV Cache

## Autoregressive inference loop

Nanochat performs autoregressive generation in `Engine.generate()` in
`nanochat/engine.py`.

The prompt tokens are first processed together in a prefill pass. A KV cache
is created and the complete prompt is passed through the model:

- `KVCache(...)` creates the prefill cache.
- `model.forward(ids, kv_cache=kv_cache_prefill)` processes the prompt.
- The logits of the final prompt position are used to predict the first
  generated token.

After prefill, nanochat enters an autoregressive decoding loop. At each
iteration:

1. The current logits represent the model's distribution over the next token.
2. `sample_next_token()` selects the next token according to temperature and
   top-k sampling.
3. The selected token is appended to the generated sequence.
4. Only the new token is passed through the model.
5. The resulting logits are used to sample the following token.
6. This repeats until a stopping token or the maximum generation length is
   reached.

Relevant implementation:
- `nanochat/engine.py:141-156` — token sampling
- `nanochat/engine.py:176-275` — optimized generation loop

## Temperature and top-k sampling

`sample_next_token()` implements the decoding parameters.

For temperature > 0, logits are divided by the temperature before applying
softmax and multinomial sampling.

A lower temperature sharpens the probability distribution, making
high-probability tokens more dominant and generation more deterministic.
A higher temperature flattens the distribution and increases the probability
of sampling lower-probability tokens, increasing diversity but potentially
reducing coherence.

When top-k is enabled, only the k highest-logit candidate tokens are retained
before sampling.

In our experiments:

- temperature 0.1 frequently produced highly repetitive sequences;
- temperature 0.7 provided more variation but still showed severe repetition
  and semantic drift;
- temperature 1.5 substantially increased diversity but often produced
  incoherent text and malformed or invented words.

These results show that temperature controls the sampling distribution, but
increasing temperature cannot compensate for limitations in the underlying
model.

## KV cache

During autoregressive generation, recomputing attention keys and values for
all previous tokens at every decoding step would be wasteful.

Nanochat therefore stores the keys and values produced for previous tokens in
a `KVCache`.

The prompt is processed once during the prefill stage:

    logits = model.forward(ids, kv_cache=kv_cache_prefill)

Nanochat then creates the decoding cache and copies the prefilled state:

    kv_cache_decode.prefill(kv_cache_prefill)

During decoding, only the newly generated token is passed to the model:

    logits = model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]

The attention implementation retrieves the cached key/value tensors using
`kv_cache.get_layer_cache()` and uses
`flash_attn_with_kvcache` during inference.

Relevant implementation:
- `nanochat/engine.py:82-139` — KVCache implementation
- `nanochat/engine.py:197-218` — prompt prefill and cache replication
- `nanochat/engine.py:275` — cached token-by-token decoding
- `nanochat/gpt.py:108-123` — attention using the KV cache

The main benefit is that keys and values for previous tokens do not need to
be recomputed for every generated token, making autoregressive decoding much
more efficient.
