# The fixed recipe (what happens to your groups after you hand them in)

1. Each selected trajectory is expanded into one training sample per assistant turn with `expand.py`
   (official EnvScaler recipe). A 20-turn trajectory therefore contributes 20 samples that share a
   prefix; only the current turn's reply (including its `<think>` block) receives loss, and the earlier
   replies in the prompt have their `<think>` blocks removed.
2. All samples of a group are shuffled and trained with `sft.yaml`: Qwen3-4B, full-parameter, LR 5e-6,
   cosine with 10% warmup, effective batch 32, 2 epochs, seed 42, prompts truncated at 16,384 tokens.
3. The resulting model is scored on the downstream multi-turn tool-use benchmark. Identical
   hyper-parameters for every group; the group is the only variable.

`expand.py` is importable (`from recipe.expand import expand_trajectory, to_chat`) so that any
per-sample statistic you compute (loss, perplexity, length, ...) is computed on exactly the inputs the
recipe will train on.
