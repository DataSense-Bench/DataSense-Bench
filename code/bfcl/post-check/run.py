import csv,json,sys,hashlib,argparse
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'selection'))
from recipe.expand import expand_trajectory
parser=argparse.ArgumentParser()
parser.add_argument('--pool', type=Path, required=True)
parser.add_argument('--assignment', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
if args.output.exists():parser.error('Output must not exist')
rows=list(csv.DictReader(args.assignment.open()))
t=pq.read_table(args.pool).to_pydict()
run=args.output.name
assert len(rows)==250
idx=[int(r['index']) for r in rows]; assert len(set(idx))==250
assert all(0<=i<len(t['meta_index']) for i in idx)
assert all(int(r['meta_index'])==t['meta_index'][int(r['index'])] for r in rows)
assert set(int(r['group_id']) for r in rows)==set(range(1,6))
assert all(sum(int(r['group_id'])==g for r in rows)==50 for g in range(1,6))
for g in range(1,6):
 selected=[int(r['index']) for r in rows if int(r['group_id'])==g];assert len(selected)==50
 samples=[]
 for i in selected:
  for s in expand_trajectory(t['messages'][i]):
   samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
 d=args.output/f'g{g}'; d.mkdir(parents=True); path=d/'train-data.json'
 path.write_text(json.dumps(samples,ensure_ascii=False))
 (d/'selection-manifest.json').write_text(json.dumps(dict(run=run,group=g,trajectories=50,expanded_rows=len(samples),indices=selected,meta_indices=[t['meta_index'][i] for i in selected],assignment_sha256=hashlib.sha256(args.assignment.read_bytes()).hexdigest()),indent=2))
print('VALIDATED',run, '5 groups x 50')
