import json,sys,datetime,os
from pathlib import Path
root=Path(os.environ.get('BFCL_RESULTS','results/bfcl')).resolve()/sys.argv[1]
parts={}
for cat in ['base','miss_func','miss_param','long_context']:
 p=root/'score/Qwen_Qwen3-4B-FC/multi_turn'/f'BFCL_v4_multi_turn_{cat}_score.json'
 d=json.loads(p.open().readline());assert d['total_count']==200,(cat,d)
 assert 0<=d['correct_count']<=200
 parts[cat]=d
result=dict(tag=sys.argv[1],checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),subsets=parts,accuracy=sum(d['correct_count'] for d in parts.values())/800)
(root/'verified-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
