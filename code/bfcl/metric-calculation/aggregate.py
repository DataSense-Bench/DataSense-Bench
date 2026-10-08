"""Validate three checkpoint evaluations and append their mean to a score CSV."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean

FIELDS = ["task", "agent", "run", "group", "score", "n_evaluations", "cases_per_evaluation"]
CATEGORIES = {"base", "miss_func", "miss_param", "long_context"}


def reward(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected a finite score in [0, 1], got {value!r}")
    return float(value)


def evaluate_bfcl(root):
    scores = {}
    for path in root.rglob("verified-summary.json"):
        match = re.search(r"(?:^|-)e([123])$", path.parent.name)
        if not match:
            raise ValueError(f"Cannot identify evaluation number: {path}")
        rep = int(match[1])
        if rep in scores:
            raise ValueError(f"Duplicate evaluation e{rep}")
        data = json.loads(path.read_text())
        subsets = data["subsets"]
        if set(subsets) != CATEGORIES:
            raise ValueError(f"Wrong BFCL categories: {path}")
        correct = 0
        for subset in subsets.values():
            n, c = subset["total_count"], subset["correct_count"]
            if type(n) is not int or type(c) is not int or n != 200 or not 0 <= c <= n:
                raise ValueError(f"Invalid BFCL counts: {path}")
            correct += c
        scores[rep] = correct / 800
        if not math.isclose(reward(data["accuracy"]), scores[rep], abs_tol=1e-12):
            raise ValueError(f"Accuracy disagrees with category counts: {path}")
    if set(scores) != {1, 2, 3}:
        raise ValueError("A checkpoint requires three verified BFCL evaluations")
    return sum(round(score * 800) for score in scores.values()) / 2400, 800


def evaluate_tblite(root):
    runs = {rep: {} for rep in (1, 2, 3)}
    for path in root.rglob("result.json"):
        data = json.loads(path.read_text())
        if "task_name" not in data:
            continue  # Harbor job summaries are not trial results.
        reps = [int(m[1]) for part in path.relative_to(root).parts[:-1]
                if (m := re.fullmatch(r"e([123])", part))]
        if len(reps) != 1:
            raise ValueError(f"Expected a trial below e1, e2 or e3: {path}")
        if data.get("exception_info") is not None:
            raise ValueError(f"Unresolved trial exception: {path}")
        name = data["task_name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"Missing task identity: {path}")
        values = (data.get("verifier_result") or {}).get("rewards") or {}
        value = reward(values.get("reward"))
        if name in runs[reps[0]]:
            raise ValueError(f"Duplicate task {name} in e{reps[0]}")
        runs[reps[0]][name] = value
    if any(len(v) != 100 for v in runs.values()) or not (set(runs[1]) == set(runs[2]) == set(runs[3])):
        raise ValueError("Expected the same 100 tasks with valid rewards in each of three evaluations")
    return mean(value for tasks in runs.values() for value in tasks.values()), 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("bfcl",), default="bfcl")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--group", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--input", type=Path, required=True, help="One checkpoint's three evaluation directories")
    parser.add_argument("--output", type=Path, required=True, help="Score CSV to create or extend")
    args = parser.parse_args()
    if args.run < 1 or not args.input.is_dir():
        parser.error("run must be positive and input must be an existing checkpoint result directory")
    score, count = (evaluate_tblite if args.task == "tblite" else evaluate_bfcl)(args.input)
    record = dict(zip(FIELDS, map(str, [args.task, args.agent, args.run, args.group, score, 3, count])))
    rows = []
    if args.output.exists():
        with args.output.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing CSV has a different schema")
            rows = list(reader)
        for row in rows:
            if all(row[k] == record[k] for k in FIELDS[:4]):
                if row == record:
                    print("Checkpoint already recorded with the same score")
                    return
                raise ValueError("Checkpoint already has a different score; refusing to overwrite")
    rows.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {args.task}: 3 × {count} cases, mean={score:.12g}")


if __name__ == "__main__":
    main()
