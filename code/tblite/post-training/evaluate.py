"""Run three Harbor evaluations sequentially against one shared vLLM server."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path

def commands(a):
    api=f'http://127.0.0.1:{a.port}/v1'
    server=[a.vllm_python,'-m','vllm.entrypoints.openai.api_server','--model',a.model_path,'--served-model-name','tb-eval','--host','127.0.0.1','--port',str(a.port),'--tensor-parallel-size',str(a.tensor_parallel),'--disable-custom-all-reduce','--reasoning-parser','qwen3','--chat-template',str(Path(__file__).with_name('chat_template_allthink.jinja'))]
    runs=[]
    for rep in (1,2,3):
        runs.append([a.harbor,'run','-p',str(a.tasks.resolve()),'-e','daytona','-o',str(a.output.resolve()),'-n','1','-k','1','-y','--timeout-multiplier','1','--environment-build-timeout-multiplier','3','--job-name',f'e{rep}','-a','terminus-2','-m','openai/tb-eval','--ek','auto_snapshot=false','--ak','api_base='+api,'--ak','temperature=0.6','--ak','parser_name=json','--ak','interleaved_thinking=true','--ak','model_info={"max_input_tokens":32768,"max_tokens":40960,"max_output_tokens":7168}'])
    return server,runs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-path',required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--vllm-python',default='python');p.add_argument('--harbor',default='harbor');p.add_argument('--port',type=int,default=33000);p.add_argument('--tensor-parallel',type=int,default=1);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.tensor_parallel<1:p.error('tensor-parallel must be positive')
    server_cmd,runs=commands(a)
    if a.dry_run:print(json.dumps({'server':server_cmd,'sequential_evaluations':runs,'gpu_count':a.tensor_parallel},indent=2));return
    if a.output.exists():p.error('Output must not exist; completed trials will not be overwritten')
    if not a.tasks.is_dir():p.error('Missing TBLite task directory')
    if len(list(a.tasks.glob('*/task.toml')))!=100:p.error('Expected the paper task set: 100 directories containing task.toml')
    if not os.environ.get('DAYTONA_API_KEY'):p.error('Set DAYTONA_API_KEY in the environment')
    with socket.socket() as s:s.bind(('127.0.0.1',a.port))
    a.output.mkdir(parents=True);env=dict(os.environ);api=f'http://127.0.0.1:{a.port}/v1';env.update(OPENAI_API_BASE=api,OPENAI_BASE_URL=api)
    env.setdefault('OPENAI_API_KEY','local-vllm');env['NO_PROXY']=env['no_proxy']='localhost,127.0.0.1,::1'
    processes=[];manifest={'status':'starting','gpu_count':a.tensor_parallel,'evaluation_runs':[],'rng_note':'Three independent stochastic repeats; The launcher does not pass an explicit per-repeat RNG seed.'}
    def save(): (a.output/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        with (a.output/'vllm.log').open('w') as log:
            server=subprocess.Popen(server_cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);processes.append(server)
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));ready=False
            for _ in range(600):
                if server.poll() is not None:raise RuntimeError('vLLM startup failed; inspect vllm.log')
                try:
                    with opener.open(api+'/models',timeout=3) as r:ready=any(x['id']=='tb-eval' for x in json.load(r)['data'])
                except (OSError,ValueError,KeyError):pass
                if ready:break
                time.sleep(2)
            if not ready:raise TimeoutError('vLLM did not become ready')
            manifest['status']='evaluating';save()
            for rep,cmd in enumerate(runs,1):
                with (a.output/f'e{rep}.log').open('w') as h:
                    proc=subprocess.Popen(cmd,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);processes.append(proc);rc=proc.wait()
                manifest['evaluation_runs'].append({'repeat':rep,'returncode':rc});save()
                if rc:raise RuntimeError(f'Harbor evaluation {rep} failed; inspect e{rep}.log')
            manifest['status']='commands_completed_pending_result_validation';save()
    except BaseException:
        manifest['status']='failed_or_interrupted';save();raise
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
if __name__=='__main__':main()
