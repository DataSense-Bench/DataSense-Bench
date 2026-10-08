"""Train a materialized BFCL group with the selection-time reference recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Materialized group directory')
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    recipe = Path(__file__).resolve().parents[1] / 'selection/recipe/sft.yaml'
    data, output = a.data.resolve(), a.output.resolve()
    if not (data / 'train-data.json').is_file():
        p.error('Missing train-data.json; run post-check first')
    if output.exists():
        p.error('Output already exists; choose a new directory')
    command = ['llamafactory-cli', 'train', str(recipe), 'do_train=true',
               f'model_name_or_path={a.model}', f'dataset_dir={data}',
               'dataset=selected_group', f'output_dir={output}', 'save_strategy=epoch']
    print(json.dumps(command))
    if a.dry_run:
        return
    registry = {'selected_group': {'file_name': 'train-data.json', 'formatting': 'alpaca',
                'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output', 'system': 'system', 'history': 'history'}}}
    registry_path = data / 'dataset_info.json'
    if registry_path.exists() and json.loads(registry_path.read_text()) != registry:
        p.error('Existing dataset_info.json differs; refusing to overwrite')
    registry_path.write_text(json.dumps(registry, indent=2) + '\n')
    subprocess.run(command, env={**os.environ, 'FORCE_TORCHRUN': '1'}, check=True)
    (output / 'training-complete.txt').write_text('Training completed successfully.\n')


if __name__ == '__main__':
    main()
