"""Download the experiment's BFCL V3 data and install the included evaluator patches."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
REVISION = "96ae8b02dc0187c911b8e2101e7bb6904271597b"
SOURCE = f"https://raw.githubusercontent.com/RUC-NLPIR/EnvScaler/{REVISION}/evaluation/bfcl_eval/data/"
CATEGORIES = ("base", "miss_func", "miss_param", "long_context")
DOCS = ("gorilla_file_system", "math_api", "message_api", "posting_api", "ticket_api",
        "trading_bot", "travel_booking", "vehicle_control")
FILES = [f"{prefix}BFCL_v4_multi_turn_{category}.json"
         for prefix in ("", "possible_answer/") for category in CATEGORIES]
FILES += [f"multi_turn_func_doc/{name}.json" for name in DOCS]


def verify(directory, manifest):
    for name in FILES:
        payload = (directory / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(f"Experiment checksum mismatch: {name}")
    for category in CATEGORIES:
        groups = []
        for prefix in ("", "possible_answer/"):
            path = directory / f"{prefix}BFCL_v4_multi_turn_{category}.json"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            ids = {row["id"] for row in rows}
            if len(rows) != 200 or len(ids) != 200:
                raise ValueError(f"Expected 200 unique cases: {path.name}")
            groups.append(ids)
        if groups[0] != groups[1]:
            raise ValueError(f"Question/answer IDs differ: {category}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true", help="Prepare data without modifying the evaluator environment")
    args = parser.parse_args()
    package = None
    if not args.data_only:
        if importlib.metadata.version("bfcl-eval") != "2026.3.23":
            parser.error("Run in the evaluation environment with bfcl-eval==2026.3.23")
        spec = importlib.util.find_spec("bfcl_eval")
        if spec is None or not spec.submodule_search_locations:
            parser.error("Cannot locate installed bfcl_eval package")
        package = Path(next(iter(spec.submodule_search_locations)))
    manifest = {entry["file"]: entry for entry in json.loads((HERE / "data_manifest.json").read_text())}
    if args.output.exists():
        verify(args.output, manifest)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
            stage = Path(temp) / "data"
            stage.mkdir()
            for name in FILES:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                url = SOURCE + name.replace("BFCL_v4_", "BFCL_v3_")
                with urllib.request.urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
            verify(stage, manifest)
            stage.rename(args.output)
    if package:
        patches = HERE / "patches" / "bfcl_eval"
        backup = package / ".datasense-originals"
        pending = []
        for source in sorted(patches.rglob("*.py")):
            relative = source.relative_to(patches)
            target, original = package / relative, backup / relative
            if target.exists() and target.read_bytes() == source.read_bytes():
                continue
            if not target.is_file():
                raise FileNotFoundError(f"Expected installed evaluator file: {target}")
            if original.exists() and target.read_bytes() != original.read_bytes():
                raise ValueError(f"Evaluator has other edits; refusing to overwrite {target}")
            pending.append((source, target, original))
        for source, target, original in pending:
            original.parent.mkdir(parents=True, exist_ok=True)
            if not original.exists():
                shutil.copy2(target, original)
            shutil.copy2(source, target)
        print(f"Evaluator patches installed; originals retained in {backup}")
    print(f"Verified BFCL V3: four categories × 200 cases, matching answers and function docs: {args.output}")


if __name__ == "__main__":
    main()
