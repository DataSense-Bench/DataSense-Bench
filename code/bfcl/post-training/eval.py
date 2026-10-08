"""One vLLM process and one GPU for a checkpoint's remaining BFCL repeats."""
import concurrent.futures,json,os,sys,signal,socket,subprocess,time,urllib.request
from pathlib import Path
BASE=Path(__file__).resolve().parent
RESULTS=Path(os.environ.get('BFCL_RESULTS','results/bfcl')).resolve()
CATS='multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context'
def verified(tag):
 try:
  d=json.loads((RESULTS/tag/'verified-summary.json').read_text());s=d['subsets']
  return set(s)=={'base','miss_func','miss_param','long_context'} and all(v['total_count']==200 and 0<=v['correct_count']<=200 for v in s.values())
 except (OSError,KeyError,ValueError,TypeError):return False
def evaluation_model(checkpoint, destination):
 """Create the paper's 64K inference view without changing trained weights."""
 checkpoint=Path(checkpoint).resolve();destination=Path(destination).resolve()
 config=json.loads((checkpoint/'config.json').read_text())
 config['max_position_embeddings']=65536
 config['rope_scaling']={'rope_type':'yarn','factor':2.0,'original_max_position_embeddings':32768}
 destination.mkdir(parents=True,exist_ok=True)
 for source in checkpoint.iterdir():
  if not source.is_file() or source.name=='config.json':continue
  target=destination/source.name
  if target.is_symlink():
   if target.resolve()!=source.resolve():raise ValueError(f'Conflicting inference file: {target}')
  elif target.exists():raise ValueError(f'Refusing to replace inference file: {target}')
  else:target.symlink_to(source)
 target=destination/'config.json'
 if target.exists() and json.loads(target.read_text())!=config:raise ValueError('Conflicting 64K inference config')
 target.write_text(json.dumps(config,indent=2)+'\n')
 return str(destination)

def main():
 assert Path('training-complete.txt').exists()
 prefix=os.environ['TAG'].rsplit('-e',1)[0];reps=[int(r) for r in os.environ['BFCL_REPS'].split(',')];assert reps and set(reps)<={1,2,3}
 todo=[r for r in reps if not verified(prefix+'-e'+str(r))]
 if not todo:print('All assigned repetitions already verified',flush=True);return
 model=os.environ['MODEL_PATH'];port=int(os.environ['LOCAL_SERVER_PORT'])
 with socket.socket() as s:s.bind(('127.0.0.1',port))
 root=RESULTS/(prefix+'-shared');root.mkdir(exist_ok=True,parents=True)
 model=evaluation_model(model,root/'model-64k')
 os.environ['MODEL_PATH']=model
 with (root/('server-'+os.environ['EVAL_RUN_ID']+'.log')).open('w') as log:
  server=subprocess.Popen(['vllm','serve',model,'--port',str(port),'--dtype','bfloat16','--max-model-len','65536','--tensor-parallel-size','1','--gpu-memory-utilization','0.85','--trust-remote-code'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  clients=[]
  try:
   opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));ready=False
   for _ in range(600):
    if server.poll() is not None:raise RuntimeError('Shared vLLM exited during startup')
    try:
     with opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:ready=any(d['id']==model for d in json.load(response)['data'])
    except Exception:pass
    if ready:break
    time.sleep(2)
   if not ready:raise TimeoutError('Shared vLLM not ready')
   print(json.dumps({'event':'shared_server_ready','server_pid':server.pid,'gpu_count':1,'repetitions':todo,'port':port}),flush=True)
   def repeat(rep):
    tag=prefix+'-e'+str(rep);out=RESULTS/tag;out.mkdir(exist_ok=True,parents=True)
    env={**os.environ,'TAG':tag}
    with (root/(f'e{rep}-'+os.environ['EVAL_RUN_ID']+'.log')).open('w') as f:
     commands=[['bfcl','generate','--model',env['MODEL'],'--test-category',CATS,'--temperature','0.7','--backend','vllm','--num-gpus','1','--gpu-memory-utilization','0.85','--num-threads',env.get('NTHREADS','64'),'--local-model-path',model,'--result-dir',str(out),'--skip-server-setup','-o'],['bfcl','evaluate','--model',env['MODEL'],'--test-category',CATS,'--result-dir',str(out),'--score-dir',str(out/'score')],[sys.executable,str(BASE/'check_eval.py'),tag]]
     for cmd in commands:
      p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);clients.append(p)
      if p.wait()!=0:raise RuntimeError(f'{tag}: {cmd[1]} failed; see client log')
    assert verified(tag);print(json.dumps({'event':'repeat_verified','tag':tag}),flush=True)
   with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as pool:
    futures=[pool.submit(repeat,r) for r in todo]
    for future in futures:future.result()
  finally:
   for p in clients+[server]:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for p in clients+[server]:
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
if __name__=='__main__':main()
