"""Launch a selection agent in a fresh work directory using the unchanged task prompt."""
import argparse,csv,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
"""Recognize explicit provider payment failures; generic rate limits are not billing failures."""
import json,re
PATTERNS=[r'\binsufficient[_ -](?:balance|credit[s]?|quota)\b',r'\bcredit balance is too low\b',r'\b(?:balance|credits?) (?:is |are )?(?:insufficient|exhausted|depleted)\b',r'\b(?:payment_required|billing_hard_limit_reached)\b',r'\bpayment required\b',r'余额不足',r'欠费',r'充值后重试']
def billing_reason(line):
 try:
  d=json.loads(line)
 except ValueError:
  if not re.search(r'error|exception|\b402\b|failed|错误|余额不足|欠费',line,re.I):return None
  text=line
 else:
  if not isinstance(d,dict):return None
  # Do not classify model prose or successful tool output as provider errors.
  if d.get('type')=='error' or d.get('is_error') is True or d.get('status')=='error':text=json.dumps(d,ensure_ascii=False)
  elif d.get('error'):text=json.dumps(d['error'],ensure_ascii=False)
  else:return None
 for pattern in PATTERNS:
  if re.search(pattern,text,re.I):return 'provider_payment_required'
 return None
class BillingLogScanner:
 def __init__(self,paths):self.paths=paths;self.offsets={};self.pending={}
 def poll(self):
  for path in self.paths:
   try:
    with path.open() as f:
     f.seek(self.offsets.get(str(path),0));chunk=f.read();self.offsets[str(path)]=f.tell()
   except FileNotFoundError:continue
   buf=self.pending.get(str(path),'')+chunk
   lines=buf.splitlines(keepends=True);self.pending[str(path)]=''
   for line in lines:
    if not line.endswith('\n'):self.pending[str(path)]=line
    reason=billing_reason(line)
    if reason:return reason
  return None


def command(backend,model,prompt):
    if backend in ('claude','deepseek','kimi'):
        return ['claude','--print','--verbose','--model',model,'--output-format','stream-json','--thinking-display','summarized','--dangerously-skip-permissions']
    if backend=='codex':return ['codex','--search','exec','--json','-c','model_reasoning_summary=detailed','--skip-git-repo-check','--yolo','--model',model]
    return ['gemini','-p',prompt,'-m',model,'--yolo','--output-format','stream-json']

def valid_assignment(path,task,pool):
    import pyarrow.parquet as pq
    if not path.is_file():return False
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    size=1000 if task=='tblite' else 50; n=pq.ParquetFile(pool).metadata.num_rows
    indices=[int(r['index']) for r in rows];ids=[int(r['group_id']) for r in rows]
    if len(indices)!=5*size or len(set(indices))!=len(indices) or any(i<0 or i>=n for i in indices):return False
    if set(ids)!=set(range(1,6)) or any(ids.count(g)!=size for g in range(1,6)):return False
    if task=='bfcl':
        meta=pq.read_table(pool,columns=['meta_index']).column(0).to_pylist()
        if any(int(r['meta_index'])!=meta[int(r['index'])] for r in rows):return False
    return True

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['bfcl'],default='bfcl');p.add_argument('--backend',choices=['claude','codex','gemini','deepseek','kimi'],required=True);p.add_argument('--model',required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--base-model',required=True);p.add_argument('--workdir',type=Path,required=True);p.add_argument('--budget-hours',type=float,default=4);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not 0<a.budget_hours<=4:p.error('budget-hours must be in (0,4]')
    prompt=(ROOT/'prompt.txt').read_text();cmd=command(a.backend,a.model,prompt)
    if a.dry_run:
        shown=['<unchanged task prompt>' if x==prompt else x for x in cmd];print(json.dumps({'command':shown,'task':a.task,'budget_hours':a.budget_hours,'max_attempts':2,'group_size':1000 if a.task=='tblite' else 50},indent=2));return
    if not a.pool.is_file():p.error('Pool parquet is missing')
    if a.workdir.exists():p.error('Work directory must not exist')
    if shutil.which(cmd[0]) is None:p.error(f'{cmd[0]} CLI is not installed')
    env=dict(os.environ)
    if a.backend in ('deepseek','kimi'):
        key='DEEPSEEK_API_KEY' if a.backend=='deepseek' else 'MOONSHOT_API_KEY'
        if not env.get(key):p.error(f'Set {key} in the environment')
        env['ANTHROPIC_AUTH_TOKEN']=env[key];env['ANTHROPIC_MODEL']=a.model
        env['ANTHROPIC_BASE_URL']='https://api.deepseek.com/anthropic' if a.backend=='deepseek' else env.get('KIMI_ANTHROPIC_BASE_URL','https://api.moonshot.ai/anthropic')
        for k in ('ANTHROPIC_API_KEY','CLAUDE_CODE_OAUTH_TOKEN'):env.pop(k,None)
    env['GEMINI_CLI_TRUST_WORKSPACE']='true'
    # Validate dependency before starting a paid agent.
    import pyarrow.parquet
    a.workdir.mkdir(parents=True);deadline=time.monotonic()+a.budget_hours*3600
    (a.workdir/'prompt.txt').write_text(prompt)
    history=[]
    def save(state):
        (a.workdir/'status.json').write_text(json.dumps({'state':state,'task':a.task,'model':a.model,'attempts':history},indent=2)+'\n')
    for attempt in (1,2):
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        work=a.workdir/f'attempt-{attempt}';work.mkdir();(work/'outputs').mkdir()
        shutil.copytree(ROOT/'recipe',work/'recipe')
        for f in (work/'recipe').rglob('*'):
            if f.is_file():f.chmod(0o444)
        config={'base_model':a.base_model,'dataset_path':str(a.pool.resolve()),'n_groups':5,'group_size':1000 if a.task=='tblite' else 50,'max_len':32768 if a.task=='tblite' else 16384}
        # JSON is valid YAML; task prompt remains byte-for-byte unchanged.
        (work/'config.yaml').write_text(json.dumps(config,indent=2));(work/'prompt.txt').write_text(prompt)
        log=work/'agent.jsonl';scanner=BillingLogScanner([log]);state='failed'
        with log.open('w') as output:
            proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try:
                proc.stdin.write(prompt if a.backend!='gemini' else '');proc.stdin.close()
                while proc.poll() is None:
                    if scanner.poll():state='payment_required';break
                    if time.monotonic()>=deadline:state='timeout';break
                    time.sleep(.5)
                if scanner.poll():state='payment_required'
                if state not in ('payment_required','timeout'):
                    try:
                        method=work/'outputs/method.md'
                        ok=(proc.returncode==0 and method.is_file() and bool(method.read_text().strip())
                            and valid_assignment(work/'outputs/group_assignment.csv',a.task,a.pool))
                    except (KeyError,ValueError,TypeError,OSError):ok=False
                    if ok:state='complete'
            finally:stop(proc)
        history.append({'attempt':attempt,'state':state,'returncode':proc.returncode});save(state)
        if state=='payment_required':raise SystemExit('Provider payment failure: recharge account; no retry submitted.')
        if state=='complete':
            shutil.copyfile(work/'outputs/group_assignment.csv',a.workdir/'group_assignment.csv');return
    save('failed_base_fallback');raise SystemExit('Selection failed: record base-model fallback under the benchmark scoring policy; no training submitted.')
if __name__=='__main__':main()
