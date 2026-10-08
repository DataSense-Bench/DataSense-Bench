# TBLite

Run all commands from the repository root. Replace example paths with your own.
To reuse `results/assignments/tblite/`, skip selection and start at post-check after preparation.

### Preparation

Run commands from the repository root on Linux with NVIDIA GPUs. Install the
three environments from the single `requirements.txt`:

```sh
python3 -m pip install uv
python3 install.py train
python3 install.py eval
python3 install.py harbor
source .venv-train/bin/activate
```

`train` provides data preparation, selection and both SFT recipes; `eval` provides
vLLM and BFCL; `harbor` provides TBLite task execution.

For a new selection, install and authenticate your chosen agent CLI.

Download the data and model for your task:

| Input | Download | Local path used below |
|---|---|---|
| Base model | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/tree/main) | `/models/Qwen3-4B` |
| TBLite training data | [OpenThoughts-Agent-SFT-10K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-10K/tree/main) | `/data/tblite-pool/train_10k.parquet` |
| TBLite evaluation tasks | [OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite/tree/main) | `/data/tblite` |

Replace the example paths with your local paths. Download Qwen3-4B once:

```sh
hf download Qwen/Qwen3-4B --local-dir /models/Qwen3-4B
```

**TBLite.** Download the training shards and evaluation tasks, then convert the
training data:

```sh
hf download open-thoughts/OpenThoughts-Agent-SFT-10K --repo-type dataset \
  --revision d0f898f7b65290c9312f59089a7e9265fdaed00c \
  --include 'data/train-*.parquet' --local-dir /data/openthoughts-10k
hf download open-thoughts/OpenThoughts-TBLite --repo-type dataset \
  --local-dir /data/tblite
python code/tblite/selection/prepare_data.py \
  --source /data/openthoughts-10k --output /data/tblite-pool
```

The conversion preserves all 10,000 trajectories in their original order,
matching the released assignments. It writes `train_10k.parquet` and the recipe's
`validation.parquet` loader input; in-training evaluation is disabled.


### 1. Selection

Before launching selection, configure an isolated environment with read-only
dataset/model mounts and no access to evaluation tasks. The launcher supplies
the task prompt and fixed SFT recipe for data selection.

Choose the command for your task. Replace `MODEL_ID` with the selection agent's
model identifier and use a new work directory for each selection run:

```sh
# TBLite
python code/tblite/selection/run.py \
  --task tblite --backend codex --model MODEL_ID \
  --pool /data/tblite-pool/train_10k.parquet --base-model /models/Qwen3-4B \
  --workdir /experiments/tblite-run1
```

Supported backends are `claude`, `codex`, `gemini`, `deepseek` and `kimi`.

On success, the work directory contains `group_assignment.csv` and `status.json`.
Attempt directories retain CLI traces; the successful attempt includes
`outputs/method.md` describing the selection method.

The launcher allows at most two attempts within one four-hour budget. If its
final state is `failed_base_fallback`, skip stages 2–3 for that selection run and
record it as failed in stage 4.

### 2. Post-check

Use the successful assignment from stage 1. To reproduce an existing selection,
instead pass a CSV from `results/assignments/<task>/<agent>/runN.csv` with the
matching original pool.

```sh
# TBLite
python code/tblite/post-check/run.py \
  --assignment /experiments/tblite-run1/group_assignment.csv \
  --pool /data/tblite-pool/train_10k.parquet --output /data/tblite-run1-groups
```

The commands validate group sizes and unique, in-range indices, then write the
training inputs to a new output directory:

| Task | Output | Contents |
|---|---|---|
| TBLite | `group_1.parquet` … `group_5.parquet` | Exactly 1,000 trajectories per group |
| BFCL | `g1/train-data.json` … `g5/train-data.json` | Exactly 50 trajectories per group, expanded into one training sample per assistant turn |

BFCL uses `code/bfcl/selection/recipe/expand.py` to expand assistant turns into
training samples.

### 3. Post-training

For each group, train a fresh copy of Qwen3-4B using the recipe in
`code/<task>/selection/recipe/`, then evaluate that checkpoint. The examples below
show **group 1**. Repeat with group IDs and separate checkpoint/result paths for
**groups 2–5** before proceeding to ranking metrics.

#### TBLite

In `.venv-train`, prepare the training tokenizer once, then train group 1.

```sh
source .venv-train/bin/activate
python code/tblite/post-training/prepare_tokenizer.py \
  --model /models/Qwen3-4B --output /models/qwen3-training-tokenizer

MODEL_PATH=/models/Qwen3-4B \
TOKENIZER_DIR=/models/qwen3-training-tokenizer \
TRAIN_FILE=/data/tblite-run1-groups/group_1.parquet \
VAL_FILE=/data/tblite-pool/validation.parquet \
OUTPUT_DIR=/checkpoints/tblite-run1-g1 NGPUS=4 GRAD_ACCUM=24 \
bash code/tblite/post-training/train.sh
```

This example uses four GPUs and effective batch size 96. Evaluation uses the
inference and Harbor executables configured below.

**Daytona setup.** TBLite uses Daytona for task sandboxes. In the
[Daytona dashboard](https://app.daytona.io/dashboard/keys), select the organization
you want to use and create an API key with sandbox creation, access and deletion
permissions. That organization must have available credits. Export the key in
the same shell that launches evaluation:

```sh
export DAYTONA_API_KEY='YOUR_DAYTONA_API_KEY'
```

Harbor inherits the exported key. To use a private shell-compatible `.env` file
outside the repository, load it before launching:
`set -a; source /path/to/daytona.env; set +a`. Use the key for your chosen
organization. See the
[Daytona authentication documentation](https://www.daytona.io/docs/api-keys/)
for optional API endpoint and region settings.

Then run:

```sh
VLLM_PYTHON="$PWD/.venv-eval/bin/python"
HARBOR_CLI="$PWD/.venv-harbor/bin/harbor"
python code/tblite/post-training/evaluate.py \
  --model-path /checkpoints/tblite-run1-g1 --tasks /data/tblite \
  --output /results/tblite-run1-g1 \
  --vllm-python "${VLLM_PYTHON:?Set the vLLM Python executable}" \
  --harbor "${HARBOR_CLI:?Set the Harbor executable}"
```

The launcher runs three stochastic evaluations sequentially against one vLLM
server/GPU, with 100 tasks each and no explicit RNG seed flags. Stage 4 checks
that all 300 task rewards are present before aggregation.

### 4. Metric calculation

Activate the training environment, then recompute metrics from the released
checkpoint scores:

```sh
source .venv-train/bin/activate
python code/tblite/metric-calculation/calculate.py results/scores.csv --output tblite-metrics.json
```

For new evaluations, aggregate each checkpoint into your own CSV. Replace the
agent, selection run and group identifiers to match the evaluated assignment:

```sh
# TBLite: the directory containing e1/, e2/ and e3/
python code/tblite/metric-calculation/aggregate.py \
  --task tblite --agent fable51 --run 1 --group 1 \
  --input /results/tblite-run1-g1 --output /results/scores.csv
```

Run aggregation for all five groups in each selection run, using each checkpoint's
own result directory. It validates all three evaluations and rejects missing,
duplicate or unresolved results. Scores use mean reward for TBLite and accuracy
fractions for BFCL, both in [0, 1]. Each calculator selects its task from the CSV.
Then compute selection metrics:

```sh
python code/tblite/metric-calculation/calculate.py /results/scores.csv \
  --references results/baselines/baseline-aggregates-user-20260926.json \
  --output /results/metrics.json
```

The output includes group means, sample standard deviations and maxima, top-group
gain over Random, best-group accuracy and Spearman correlation. Evaluation scores
are averaged within checkpoints before averaging across selection runs. Ties for
best count as correct; a constant five-group score has undefined Spearman and is
excluded only from that metric's average. References must use the same evaluation
protocol as your checkpoint scores.

To include a failed selection, use the JSON format and mark that run with
`"failed": true`:

```json
{"base": 0.1, "random": 0.12, "runs": [
  {"id": "run1", "scores": [0.2, 0.18, 0.16, 0.14, 0.12]},
  {"id": "run2", "failed": true}
]}
```

Pass this file to `calculate.py` in place of the CSV. A failed selection receives
the base-model score, Spearman −1 and best-group accuracy 0.

