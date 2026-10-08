"""Fixed OpenThoughts-Agent SFT recipe for Qwen3-4B.

Full-parameter training: global batch 96, 7 epochs, learning rate 4e-5,
cosine schedule, 10% warmup, no weight decay, and a 32,768-token cutoff.
Assistant-only loss includes reasoning in every assistant turn.
Runtime paths and resource settings are supplied through the environment.
"""
import math
import os
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


class StopOnNaN(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k in ("loss", "grad_norm"):
                v = logs.get(k)
                if v is not None:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(fv) or math.isinf(fv):
                        print(f"[StopOnNaN] {k}={v} at step {state.global_step}; stopping.", flush=True)
                        control.should_training_stop = True
        return control



def _last_complete_checkpoint(output_dir):
    """Find the newest checkpoint containing trainer_state.json."""
    import os, re
    if not os.path.isdir(output_dir): return None
    cands = []
    for d in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        p = os.path.join(output_dir, d)
        if m and os.path.isfile(os.path.join(p, "trainer_state.json")):
            cands.append((int(m.group(1)), p))
        elif m:
            print(f"[sft] ignoring incomplete checkpoint {p} (no trainer_state.json)", flush=True)
    return max(cands)[1] if cands else None

def load_ds(path):
    table = pq.read_table(path, columns=["messages"], memory_map=True)
    table = table.replace_schema_metadata(None)
    return Dataset(table)


MODEL = os.environ["MODEL_PATH"]
DATA_ROOT = os.environ.get("DATA_ROOT", ".")
OUT = os.environ["OUTPUT_DIR"]
try:
    import pyarrow.parquet as _pq
    _N_ROWS = _pq.ParquetFile(os.environ.get("TRAIN_FILE", f"{DATA_ROOT}/train.parquet")).metadata.num_rows
except Exception:
    _N_ROWS = 10**9   # unknown -> keep checkpoints

args = SFTConfig(
    output_dir=OUT,
    run_name=os.environ.get("RUN_NAME", Path(OUT).name),
    num_train_epochs=float(os.environ.get("EPOCHS", "7")),
    learning_rate=float(os.environ.get("LR", "4e-5")),
    seed=42,
    lr_scheduler_type="cosine",
    warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.1")),
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "0")),   # Fixed recipe uses zero weight decay.
    adam_beta2=float(os.environ.get("ADAM_BETA2", "0.98")),      # OT: adam_beta2 0.98
    # Clipping matches the OpenThoughts-Agent 1,000-trajectory configuration.
    max_grad_norm=float(os.environ.get("MAX_GRAD_NORM", "1e-3")),
    per_device_train_batch_size=int(os.environ.get("PER_DEVICE_BS", "1")),
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", str(max(1, 96 // int(os.environ.get("NGPUS", "8")))))),
    max_length=int(os.environ.get("MAX_LEN", "32768")),
    assistant_only_loss=True,
    use_liger_kernel=True,
    bf16=True,
    # SDPA uses PyTorch attention kernels without a separate flash-attn install.
    model_init_kwargs={
        ("dtype" if int(__import__("transformers").__version__.split(".")[0]) >= 5
         else "torch_dtype"): "bfloat16",
        "attn_implementation": "sdpa",
    },
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Downstream tasks are evaluated after training; disable in-training evaluation.
    eval_strategy="no",
    logging_steps=5,
    # Skip intermediate checkpoints when the run is shorter than the save interval.
    save_strategy=os.environ.get("SAVE_STRATEGY") or ("no" if _N_ROWS * float(os.environ.get("EPOCHS", "7")) / 96 < int(os.environ.get("SAVE_STEPS", "250")) else "steps"),
    # Retain resumable checkpoints for longer runs.
    save_steps=int(os.environ.get("SAVE_STEPS", "250")),
    save_total_limit=2,
    report_to=os.environ["REPORT_TO"].split(",") if os.environ.get("REPORT_TO") else [],
)
# Tokenizer copy with {% generation %} markers added to the assistant branch of
# the Qwen3 chat template (the stock template has none, and assistant_only_loss
# requires them). Verified: rendering is byte-identical to the original; the
# assistant mask covers exactly assistant content + <|im_end|>.
from transformers import AutoTokenizer
# Use the prepared tokenizer with assistant masks and reasoning retained in all turns.
_IS_Q3 = "Qwen3-4B/" in MODEL or MODEL.rstrip("/").endswith("Qwen3-4B")
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR") or str(Path(__file__).parent / (
    ("tokenizer_genmask_qwen3-4b-allthink" if os.environ.get("THINK_TEMPLATE", "allthink") == "allthink"
     else "tokenizer_genmask_qwen3-4b") if _IS_Q3 else "tokenizer_genmask"))
class _NoShuffleSFTTrainer(SFTTrainer):
    """OT recipe: disable_shuffling=True (Table 16/18) -> fixed example order every epoch."""
    def _get_train_sampler(self, *a, **k):
        if os.environ.get("SHUFFLE", "0") == "1":
            return super()._get_train_sampler(*a, **k)
        from torch.utils.data import SequentialSampler
        return SequentialSampler(self.train_dataset)

trainer = _NoShuffleSFTTrainer(
    model=MODEL,
    args=args,
    processing_class=AutoTokenizer.from_pretrained(TOKENIZER_DIR),
    train_dataset=load_ds(os.environ.get("TRAIN_FILE", f"{DATA_ROOT}/train.parquet")),
    eval_dataset=load_ds(os.environ.get("VAL_FILE", f"{DATA_ROOT}/val.parquet")),
    callbacks=[StopOnNaN()],
)
from transformers.trainer_utils import get_last_checkpoint
last_ckpt = _last_complete_checkpoint(OUT) if os.path.isdir(OUT) else None
if last_ckpt:
    print(f"[sft] resuming from {last_ckpt}", flush=True)
trainer.train(resume_from_checkpoint=last_ckpt)
trainer.save_model(OUT)

# Preserve base-model configuration and tokenizer compatibility with inference.
if trainer.is_world_process_zero():
    import json as _json
    import shutil as _shutil
    base = Path(MODEL)
    cfg = _json.load(open(base / "config.json"))
    _json.dump(cfg, open(Path(OUT) / "config.json", "w"), indent=2)
    for fn in ("tokenizer_config.json", "tokenizer.json", "vocab.json",
               "merges.txt", "special_tokens_map.json", "generation_config.json"):
        src = base / fn
        if src.exists():
            _shutil.copy(src, Path(OUT) / fn)
    for extra in Path(OUT).glob("chat_template.jinja"):
        extra.unlink()
    print("[sft] config/tokenizer rewritten in serving (4.x) format", flush=True)
