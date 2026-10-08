"""Validate five disjoint 1,000-trajectory groups before writing any output."""
import argparse,csv,json
from pathlib import Path

def validate(rows, pool_size):
    groups={g:[] for g in range(1,6)}; seen=set()
    for row in rows:
        i,g=int(row['index']),int(row['group_id'])
        if g not in groups: raise ValueError('group_id must be 1..5')
        if not 0<=i<pool_size: raise ValueError(f'Index out of range: {i}')
        if i in seen: raise ValueError(f'Duplicate trajectory index: {i}')
        seen.add(i);groups[g].append(i)
    if any(len(v)!=1000 for v in groups.values()):raise ValueError('Each of the five groups must contain exactly 1000 trajectories')
    return {g:sorted(v) for g,v in groups.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import pyarrow.parquet as pq
    with a.assignment.open(newline='') as f:groups=validate(list(csv.DictReader(f)),pq.ParquetFile(a.pool).metadata.num_rows)
    if a.output.exists():p.error('Output must not exist; refusing to overwrite results')
    table=pq.read_table(a.pool);a.output.mkdir(parents=True)
    for g,indices in groups.items():pq.write_table(table.take(indices),a.output/f'group_{g}.parquet')
    (a.output/'groups_index.json').write_text(json.dumps({'source':str(a.pool.resolve()),**{f'group_{g}':v for g,v in groups.items()}},indent=2)+'\n')
    print('Validated and wrote five disjoint groups of 1000 trajectories')
if __name__=='__main__':main()
