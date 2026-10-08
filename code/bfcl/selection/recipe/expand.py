"""expand.py — READ-ONLY reference: how the fixed recipe turns one trajectory into training samples.

An n-turn trajectory becomes n samples (the official EnvScaler "mask_history" expansion):
  sample i  prompt = system + turns 1..i-1 (assistant replies with their <think>...</think> REMOVED)
                     + user turn i
            target = assistant turn i WITH its <think>...</think>
Loss is computed on the target only. Every prompt is truncated at max_len tokens (recipe/sft.yaml
cutoff_len). This file reproduces the released training text exactly (verified on all 9,022
trajectories); use it to build inputs that match what the recipe will see. Do not modify it.
"""
import re

THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def strip_think(text):
    """Assistant text as it appears in the HISTORY of a later sample."""
    return THINK.sub("", text)


def expand_trajectory(messages):
    """messages: the pool row's list of {role, content} (system, then user/assistant alternating).
    Returns a list of samples {"system", "history": [(user, assistant_stripped), ...], "prompt", "target"}."""
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    turns = [(messages[i]["content"], messages[i + 1]["content"])
             for i in range(1 if system else 0, len(messages) - 1, 2)]
    samples = []
    for i, (user, assistant) in enumerate(turns):
        samples.append({
            "system": system,
            "history": [(u, strip_think(a)) for u, a in turns[:i]],
            "prompt": user,
            "target": assistant,
        })
    return samples


def to_chat(sample):
    """The same sample as a chat message list (for tokenizer.apply_chat_template): the prompt part is
    everything up to and including the final user message; the target is the last assistant message."""
    msgs = [{"role": "system", "content": sample["system"]}]
    for u, a in sample["history"]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": sample["prompt"]})
    return msgs, sample["target"]
