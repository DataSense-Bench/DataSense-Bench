#!/usr/bin/env python3
"""Build the EnvScaler candidate pool from trajectory metadata.

Preserve per-turn reasoning for agent inspection and compare the rendered
training samples against all 9,022 released trajectories before writing.

Outputs:
  pool.parquet: trajectories in metadata order, excluding the 60 validation rows.
  index_map.json: pool-row identifiers, validation indices and baseline indices.
"""
import json, re, hashlib, random, os, sys, argparse
from collections import Counter, defaultdict
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer

parser=argparse.ArgumentParser(description="Build the EnvScaler candidate pool")
parser.add_argument('--data-dir', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--tokenizer', required=True)
args=parser.parse_args()
D, OUT, TOK=args.data_dir, args.output, args.tokenizer
os.makedirs(OUT, exist_ok=False)
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
SEED, N_RANDOM, VAL_AT, N_VAL = 42, 200, 500, 60          # same shuffle as build_traj3/5 -> same val60 / random200
tok = AutoTokenizer.from_pretrained(TOK)

def render_asst(a, think):
    parts = []
    if a.get("content"): parts.append(a["content"])
    for tc in a.get("tool_calls") or []:
        f = tc["function"]; args = f["arguments"]
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: pass
        parts.append("<tool_call>\n" + json.dumps({"name": f["name"], "arguments": args}, ensure_ascii=False) + "\n</tool_call>")
    body = "\n".join(parts)
    return ("<think>\n" + (a.get("reasoning_content") or "") + "\n</think>\n\n" + body) if think else body

def render_system(sysm, tools):
    txt = tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": "Q"}],
                                  tools=tools, tokenize=False)
    return txt.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]

def render_turns(ms):
    turns, pend = [], []
    for x in ms:
        if x["role"] == "system": continue
        if x["role"] == "user": pend.append(x["content"])
        elif x["role"] == "tool": pend.append("<tool_response>\n" + x["content"] + "\n</tool_response>")
        elif x["role"] == "assistant": turns.append(("\n".join(pend), x)); pend = []
    return turns

def key_of(sysm, u, a_stripped):
    return hashlib.sha1((sysm + "\x00" + u + "\x00" + a_stripped).encode()).hexdigest()

def stream(path):
    dec, buf, pos = json.JSONDecoder(), "", 0
    with open(path) as f:
        while True:
            chunk = f.read(64 << 20)
            if not chunk: return
            buf, pos = buf[pos:] + chunk, 0
            while True:
                nxt = buf.find("{", pos)
                if nxt < 0: break
                try: row, end = dec.raw_decode(buf, nxt)
                except ValueError: pos = nxt; break
                pos = end; yield row

# pass over the released file: per key keep the row count and the longest-history row
count, longest = Counter(), {}
for r in stream(f"{D}/mask_history_all_traj-9K_apply_qwen3_template.json"):
    h = r.get("history") or []
    u, a = (h[0][0], h[0][1]) if h else (r["instruction"] + r["input"], THINK.sub("", r["output"]))
    k = key_of(r.get("system", ""), u, THINK.sub("", a)); count[k] += 1
    if k not in longest or len(h) > len(longest[k].get("history") or []): longest[k] = r
print(f"released rows {sum(count.values())} | groups {len(count)}", flush=True)

meta = json.load(open(f"{D}/envscaler_sft_traj_9k_metadata.json"))
rows, keys, bad = [], [], Counter()
for mi, m in enumerate(meta):
    ms, tools = json.loads(m["messages"]), json.loads(m["tools"])
    sysm = next((x["content"] for x in ms if x["role"] == "system"), "")
    sys_r = render_system(sysm, tools); turns = render_turns(ms)
    k = key_of(sys_r, turns[0][0], render_asst(turns[0][1], False)); keys.append(k)
    r = longest.get(k)
    if r is None: bad["no_group"] += 1; continue
    h = r.get("history") or []
    if count[k] != len(turns) or count[k] != int(m["steps"]): bad["count"] += 1
    if any((turns[i][0], render_asst(turns[i][1], False)) != (h[i][0], h[i][1]) for i in range(len(h))): bad["history"] += 1
    if turns[len(h)][0] != r["instruction"] + r["input"] or render_asst(turns[len(h)][1], True) != r["output"]: bad["output"] += 1
    msgs = [{"role": "system", "content": sys_r}]
    for u, a in turns: msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": render_asst(a, True)}]
    rows.append({"meta_index": mi, "env_id": m["task_info"].get("env_id", ""), "task_id": str(m["task_info"].get("task_id", "")),
                 "traj_type": m.get("traj_type", ""), "steps": int(m["steps"]), "messages": msgs})
print(f"metadata {len(meta)} | rendered {len(rows)} | unique keys {len(set(keys))} | verification failures {dict(bad)}", flush=True)
if bad or len(set(keys)) != len(meta) or len(rows) != len(meta):
    print("ABORT: renderer does not reproduce the released expansion for every trajectory", flush=True); sys.exit(1)

# the very same shuffle build_traj3/5 used, so val60 / random200 are the trajectories already trained and scored
order = sorted(keys); random.Random(SEED).shuffle(order)
k2i = {k: i for i, k in enumerate(keys)}
random200 = sorted(k2i[k] for k in order[:N_RANDOM]); val60 = sorted(k2i[k] for k in order[VAL_AT:VAL_AT + N_VAL])
val_set = set(val60)
pool = [r for r in rows if r["meta_index"] not in val_set]
pq.write_table(pa.table({c: [r[c] for r in pool] for c in ("meta_index", "env_id", "task_id", "traj_type", "steps", "messages")}),
               f"{OUT}/pool.parquet")
json.dump({"pool_rows": [{"meta_index": r["meta_index"], "key": keys[r["meta_index"]]} for r in pool],
           "val60_meta_index": val60, "random200_meta_index": random200,
           "random200_pool_rows": [i for i, r in enumerate(pool) if r["meta_index"] in set(random200)]},
          open(f"{OUT}/index_map.json", "w"))
print(f"pool.parquet {len(pool)} rows | val60 {len(val60)} | random200 in pool {sum(r['meta_index'] in set(random200) for r in pool)}", flush=True)
