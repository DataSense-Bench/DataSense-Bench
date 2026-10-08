"""Install one isolated environment from the profiles in requirements.txt."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def requirements(profile):
    active, selected = set(), []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        match = re.fullmatch(r"# \[([a-z,]+)\]", line.strip())
        if match:
            active = set(match.group(1).split(","))
        elif profile in active and line.strip() and not line.lstrip().startswith("#"):
            selected.append(line)
    if not selected:
        raise ValueError(f"No dependencies for profile {profile}")
    return "\n".join(selected) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("train", "eval", "harbor"))
    parser.add_argument("--env", type=Path, help="New environment directory (default: .venv-PROFILE)")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv first: python -m pip install uv")
    env = (args.env or ROOT / f".venv-{args.profile}").resolve()
    marker = env / ".datasense-profile"
    if env.exists():
        if not marker.is_file() or marker.read_text().strip() != args.profile:
            parser.error(f"Existing environment was not created for {args.profile}: {env}")
    else:
        subprocess.run([uv, "venv", "--python", "3.12", str(env)], check=True)
        marker.write_text(args.profile + "\n")
    process_env = dict(os.environ)
    for key, value in {"UV_CONCURRENT_DOWNLOADS": "4", "UV_CONCURRENT_BUILDS": "1",
                       "UV_CONCURRENT_INSTALLS": "4", "RAYON_NUM_THREADS": "4",
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}.items():
        process_env.setdefault(key, value)
    python = env / "bin" / "python"
    with tempfile.TemporaryDirectory() as temp:
        spec = Path(temp) / "requirements.txt"
        spec.write_text(requirements(args.profile))
        command = [uv, "pip", "install", "--python", str(python), "-r", str(spec)]
        if args.profile != "harbor":
            command += ["--extra-index-url", "https://download.pytorch.org/whl/cu128",
                        "--index-strategy", "unsafe-best-match"]
        subprocess.run(command, env=process_env, check=True)
    subprocess.run([uv, "pip", "check", "--python", str(python)], env=process_env, check=True)
    imports = {"train": "import torch, transformers, trl, peft, datasets, pyarrow, liger_kernel, llamafactory",
               "eval": "import torch, transformers, vllm, bfcl_eval",
               "harbor": "import harbor, daytona"}
    subprocess.run([str(python), "-c", imports[args.profile]], env=process_env, check=True)
    print(f"Ready: source {env}/bin/activate")


if __name__ == "__main__":
    main()
