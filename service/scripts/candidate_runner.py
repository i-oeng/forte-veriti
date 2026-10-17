"""Run a bounded comparison only on the already-authorized, idle Vast instance."""
import json, os, pathlib, signal, subprocess, time, urllib.request
from huggingface_hub import snapshot_download

ROOT=pathlib.Path('/workspace/forte')
EXP=ROOT/'experiments/candidates-20261017'
OUT=ROOT/'var/bakeoff/candidates-20261017'
OUT.mkdir(parents=True,exist_ok=True)
STATUS=OUT/'status.json'
models=[
    ('q4b',ROOT/'models/Qwen3.5-4B-AWQ-4bit',None,None),
    ('cosmos8b',ROOT/'models/Cosmos-Reason2-8B',None,None),
    ('q27b',EXP/'models/q27b','cyankiwi/Qwen3.5-27B-AWQ-4bit','5cba9b5f1bca69fab9b75c88ae7817ae22d7d898'),
    ('internvl14b',EXP/'models/internvl14b','cyankiwi/InternVL3_5-14B-AWQ-4bit','f5ccf9c80ccaa9ca579603708864bb55a4417779'),
]
state={'phase':'starting','results':{},'excluded':{'internvl38b':{
    'reason':'Checkpoint weights alone exceed available physical GPU memory; no CPU offload comparison.',
    'weights_bytes':31923014696,'gpu_total_mib':24564}},'pid':os.getpid()}
def update(**kwargs):
    state.update(kwargs); state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATUS)
    print(json.dumps(kwargs),flush=True)
def stop(proc):
    if proc is None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=15)
def main():
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>500:raise RuntimeError(f'GPU is not idle ({used} MiB); refusing to interrupt another job')
    env=os.environ.copy()
    env.update(FORTE_HOME=str(ROOT),FORTE_VLM_BASE_URL='http://127.0.0.1:8101/v1',FORTE_VLM_MODEL='vlm',
        FORTE_JOB_DEADLINE_SEC='270',FORTE_VLM_MAX_CONCURRENCY='8',FORTE_REVIEW_TYPES='CASH_OUT_OF_VIEW,other',
        FORTE_OUTPUT_LANG='ru',FORTE_DETECTOR='small',FORTE_PERCEPTION_FPS='5',FORTE_VIOLATION_THRESHOLD='0.5',
        VLLM_USE_FLASHINFER_SAMPLER='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
    libs=list((ROOT/'venv-vllm/lib').glob('python*/site-packages/nvidia/*/lib'))
    env['LD_LIBRARY_PATH']=':'.join(map(str,libs))+':'+env.get('LD_LIBRARY_PATH','')
    for name,path,repo,revision in models:
        server=evaluator=None
        try:
            update(model=name,phase='download' if repo else 'loading')
            if repo:
                snapshot_download(repo,revision=revision,local_dir=path,
                    allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.py','*.model'],max_workers=4)
            mm={'max_dynamic_patch':2} if name.startswith('internvl') else {'size':{'longest_edge':589824,'shortest_edge':3136}}
            # InternVL's installed video processor rejects max_dynamic_patch; image-only inference uses defaults.
            if name.startswith('internvl'):mm={}
            args=[str(ROOT/'venv-vllm/bin/vllm'),'serve',str(path),'--served-model-name','vlm','--host','127.0.0.1',
                  '--port','8101','--gpu-memory-utilization','0.90','--max-model-len','16384','--max-num-seqs','8',
                  '--max-num-batched-tokens','2048','--limit-mm-per-prompt','{"image":24,"video":0}',
                  '--mm-processor-kwargs',json.dumps(mm),'--enable-prefix-caching',
                  '--structured-outputs-config.backend','xgrammar','--structured-outputs-config.disable_any_whitespace','true',
                  '--seed','0']
            if name.startswith('internvl'):args+=['--trust-remote-code']
            run_dir=OUT/name;run_dir.mkdir(exist_ok=True)
            (run_dir/'settings.json').write_text(json.dumps({'command':args,'repo':repo,'revision':revision,
                'parallel':4,'deadline':270,'preview':False,'source_manifest':str(EXP/'manifest.json')},indent=2))
            with (run_dir/'server.log').open('w') as logfile:
                server=subprocess.Popen(args,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            update(phase='loading',server_pid=server.pid)
            ready=False
            for _ in range(150):
                if server.poll() is not None:break
                try:
                    urllib.request.urlopen('http://127.0.0.1:8101/v1/models',timeout=2).close();ready=True;break
                except Exception:time.sleep(4)
            if not ready:
                state['results'][name]={'status':'startup_failed','log_tail':(run_dir/'server.log').read_text(errors='replace')[-3500:]}
                update();continue
            update(phase='evaluating')
            app_libs=list((ROOT/'venv-app/lib').glob('python*/site-packages/nvidia/*/lib'))
            runenv={**env,'FORTE_VAR':str(run_dir),'LD_LIBRARY_PATH':':'.join(map(str,app_libs))+':'+os.environ.get('LD_LIBRARY_PATH','')}
            with (run_dir/'eval.log').open('w') as logfile:
                evaluator=subprocess.Popen([str(ROOT/'venv-app/bin/python'),'-u','-m','scripts.candidate_eval'],
                    cwd=EXP/'service',env=runenv,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.time();peak=0
            while evaluator.poll() is None and time.time()-started<2400:
                try:peak=max(peak,int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
                except Exception:pass
                update(phase='evaluating',completed_clips=len([p for p in run_dir.glob('*.json') if p.stem.isdigit()]),elapsed_sec=round(time.time()-started),peak_mib=peak)
                time.sleep(15)
            if evaluator.poll() is None:stop(evaluator)
            result={'status':'complete' if evaluator.returncode==0 else 'evaluation_failed','exit_code':evaluator.returncode,'peak_mib':peak}
            if (run_dir/'metrics.json').exists():result['metrics']=json.loads((run_dir/'metrics.json').read_text())
            else:result['log_tail']=(run_dir/'eval.log').read_text(errors='replace')[-3500:]
            state['results'][name]=result;update()
        except Exception as exc:
            state['results'][name]={'status':'failed','error':f'{type(exc).__name__}: {exc}'};update()
        finally:
            stop(evaluator);stop(server);time.sleep(5)
    update(phase='done')
if __name__=='__main__':
    try: main()
    except Exception as exc:update(phase='failed',error=f'{type(exc).__name__}: {exc}');raise
