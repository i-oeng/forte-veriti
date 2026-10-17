"""Pinned official-runtime smoke test; no rule pipeline or deadline-based truncation."""
import json, os, pathlib, re, subprocess, sys, time, traceback

ROOT=pathlib.Path('/workspace/forte')
EXP=ROOT/'experiments/internvideo3-20261017'
OUT=ROOT/'var/bakeoff/internvideo3-20261017'
sys.path.insert(0,str(EXP/'runtime'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

def save(name,data):
    p=OUT/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2))

def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoProcessor
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>500:raise RuntimeError(f'GPU occupied: {used} MiB; no other job interrupted')
    started=time.time()
    save('smoke-status.json',{'phase':'loading','pid':os.getpid()})
    model=AutoModelForCausalLM.from_pretrained(EXP/'model',torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map='cuda:0',trust_remote_code=True,local_files_only=True)
    processor=AutoProcessor.from_pretrained(EXP/'model',trust_remote_code=True,local_files_only=True)
    processor.image_processor.size={'longest_edge':131072,'shortest_edge':65536}
    processor.video_processor.size={'longest_edge':131072,'shortest_edge':65536}
    load_sec=time.time()-started
    src=next(p for p in (ROOT/'data/videos').glob('*.mov') if re.findall(r'\d+',p.stem)==['11'])
    # Explicitly decode eight chronological frames, respecting rotation. No labels enter the prompt.
    frames=[]
    for t in range(14,22):
        raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-ss',str(t),'-i',str(src),
            '-frames:v','1','-vf','scale=224:384','-f','rawvideo','-pix_fmt','rgb24','-'])
        frames.append(np.frombuffer(raw,dtype=np.uint8).reshape(384,224,3))
    messages=[{'role':'user','content':[{'type':'video','video':'predecoded'},
        {'type':'text','text':'Describe only visible actions in this bank CCTV sequence. Mention the door state and whether anyone leaves. Do not infer intent, legal duties, or unseen actions.'}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[text],videos=[np.stack(frames)],do_sample_frames=False,
        video_metadata=[{'fps':1.0,'total_num_frames':8,'frames_indices':list(range(8))}],return_tensors='pt')
    grids=inputs['video_grid_thw'].tolist()
    represented=sum(g[0]*processor.video_processor.temporal_patch_size for g in grids)
    if represented!=len(frames):raise RuntimeError(f'Frame loss: supplied {len(frames)}, grid represents {represented}')
    inputs=inputs.to('cuda:0')
    save('smoke-status.json',{'phase':'generating','input_tokens':inputs.input_ids.shape[1],'grid':grids})
    started=time.time()
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=160,do_sample=False,use_cache=True)
    ids=output[0,inputs.input_ids.shape[1]:]
    result={'model':'yanziang/InternVideo3-8B-Instruct','revision':'c4602918b65225650d152db2850fe34e01d21fcd',
        'runtime':'Transformers 4.57.3 / SDPA / BF16','load_sec':load_sec,'inference_sec':time.time()-started,
        'frames_requested':len(frames),'frames_represented':represented,'grid':grids,
        'input_tokens':inputs.input_ids.shape[1],'output_tokens':len(ids),'hit_output_cap':len(ids)>=160,
        'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'response':processor.decode(ids,skip_special_tokens=True),
        'caveat':'Compatibility smoke test only, not accuracy; these dimensions are only for this smoke test.'}
    save('smoke-result.json',result);save('smoke-status.json',{'phase':'done'});print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('smoke-status.json',{'phase':'failed','error':f'{type(exc).__name__}: {exc}'})
        traceback.print_exc();raise
