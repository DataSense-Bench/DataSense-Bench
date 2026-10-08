# BFCL

Run all commands from the repository root. Replace example paths with your own.
To reuse `results/assignments/bfcl/`, skip selection and start at post-check after preparation.

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
| BFCL training data | [EnvScaler-SFT-Traj-9K](https://huggingface.co/datasets/XXHStudyHard/EnvScaler-SFT-Traj-9K/tree/main) | `/data/envscaler` |
| BFCL evaluation data | [BFCL V3](https://github.com/RUC-NLPIR/EnvScaler/tree/96ae8b02dc0187c911b8e2101e7bb6904271597b/evaluation/bfcl_eval/data) | `/data/bfcl-v3` |

Replace the example paths with your local paths. Download Qwen3-4B once:

```sh
hf download Qwen/Qwen3-4B --local-dir /models/Qwen3-4B
```

**BFCL.** Download the two source files and build the training pool:

```sh
hf download XXHStudyHard/EnvScaler-SFT-Traj-9K --repo-type dataset \
  --include envscaler_sft_traj_9k_metadata.json mask_history_all_traj-9K_apply_qwen3_template.json \
  --local-dir /data/envscaler
python code/bfcl/selection/build_pool.py \
  --data-dir /data/envscaler --output /data/bfcl-pool \
  --tokenizer /models/Qwen3-4B
```

This writes `pool.parquet` and `index_map.json` under `/data/bfcl-pool`.


### 1. Selection

Before launching selection, configure an isolated environment with read-only
dataset/model mounts and no access to evaluation tasks. The launcher supplies
the task prompt and fixed SFT recipe for data selection.

Choose the command for your task. Replace `MODEL_ID` with the selection agent's
model identifier and use a new work directory for each selection run:

```sh
# BFCL
python code/bfcl/selection/run.py \
  --task bfcl --backend codex --model MODEL_ID \
  --pool /data/bfcl-pool/pool.parquet --base-model /models/Qwen3-4B \
  --workdir /experiments/bfcl-run1
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
# BFCL
python code/bfcl/post-check/run.py \
  --assignment /experiments/bfcl-run1/group_assignment.csv \
  --pool /data/bfcl-pool/pool.parquet --output /data/bfcl-run1-groups
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

#### BFCL

In `.venv-train`, train group 1:

```sh
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python code/bfcl/post-training/train.py \
  --data /data/bfcl-run1-groups/g1 --model /models/Qwen3-4B \
  --output /checkpoints/bfcl-run1-g1
```

Training uses two B200 GPUs with effective batch size 32. Successful
training writes the checkpoint and `training-complete.txt` to the output directory.

In the BFCL evaluation environment, prepare the evaluation data and install the
included evaluator patches once:

```sh
source .venv-eval/bin/activate
python code/bfcl/post-training/prepare_eval.py --output /data/bfcl-v3
```

This prepares the fixed V3 dataset (four categories of 200 cases) and installs
the evaluator patches in `.venv-eval`.

After training, evaluate the checkpoint from the repository root:

```sh
RELEASE_ROOT="$PWD"
(
  cd /checkpoints/bfcl-run1-g1
  MODEL=Qwen/Qwen3-4B-FC MODEL_PATH="$PWD" TAG=bfcl-run1-g1-e1 \
  BFCL_DATA_DIR=/data/bfcl-v3 BFCL_RESULTS=/results/bfcl-run1-g1 \
  bash "$RELEASE_ROOT/code/bfcl/post-training/eval.sh"
)
```

The `-e1` tag names the first of three evaluations, which share one model
server/GPU. Inference uses a separate 64K configuration with YaRN factor 2
(original context 32,768). `check_eval.py` validates each evaluation and writes:

```text
/results/bfcl-run1-g1/bfcl-run1-g1-e1/verified-summary.json
/results/bfcl-run1-g1/bfcl-run1-g1-e2/verified-summary.json
/results/bfcl-run1-g1/bfcl-run1-g1-e3/verified-summary.json
```

Each summary contains an `accuracy` fraction. Set `BFCL_REPS` to a comma-separated
subset of `1,2,3` to run fewer evaluations; already verified results are skipped.

### 4. Metric calculation

Activate the training environment, then recompute metrics from the released
checkpoint scores:

```sh
source .venv-train/bin/activate
python code/bfcl/metric-calculation/calculate.py results/scores.csv --output bfcl-metrics.json
```

For new evaluations, aggregate each checkpoint into your own CSV. Replace the
agent, selection run and group identifiers to match the evaluated assignment:

```sh
# BFCL: the directory containing this checkpoint's three -eN result folders
python code/bfcl/metric-calculation/aggregate.py \
  --task bfcl --agent fable51 --run 1 --group 1 \
  --input /results/bfcl-run1-g1 --output /results/scores.csv
```

Run aggregation for all five groups in each selection run, using each checkpoint's
own result directory. It validates all three evaluations and rejects missing,
duplicate or unresolved results. Scores use mean reward for TBLite and accuracy
fractions for BFCL, both in [0, 1]. Each calculator selects its task from the CSV.
Then compute selection metrics:

```sh
python code/bfcl/metric-calculation/calculate.py /results/scores.csv \
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

