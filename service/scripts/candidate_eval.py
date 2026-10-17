"""Isolated candidate evaluation; errors remain errors, not clean-video successes.

Run from a frozen service copy with FORTE_VAR pointing to a unique model directory.
The label-blind probe uses annotated event windows and is diagnostic, NOT an unseen-type benchmark.
"""
from __future__ import annotations
import base64
import concurrent.futures
import json
import os
from pathlib import Path
import time
import numpy as np
# Keep the deployed default review catalogue for every candidate.
os.environ['FORTE_REVIEW_TYPES'] = 'CASH_OUT_OF_VIEW,other'
from app import config
from app.pipeline import analyze, video, vlm
from scripts.eval import load_labels, load_types, vid_num

ROOT = Path(os.environ['FORTE_HOME'])
OUT = config.VAR_DIR
OUT.mkdir(parents=True, exist_ok=True)
labels = load_labels(ROOT/'data/open_labels.xlsx')
types = load_types(ROOT/'data/Copy of sheets.xlsx')
videos = sorted((ROOT/'data/videos').glob('*.mov'), key=vid_num)

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def one(src):
    num = vid_num(src)
    started = time.time()
    try:
        result = analyze.analyze(src, OUT/'media'/str(num), started+270, lambda *_: None, preview=False)
        result['wall_sec'] = round(time.time()-started, 2)
        for item in result.get('violations', []):
            for frame in item.get('evidence_frames', []):
                frame.pop('image_base64', None)
    except Exception as exc:
        result = {'error': f'{type(exc).__name__}: {exc}', 'wall_sec': round(time.time()-started,2)}
    save(OUT/f'{num}.json', result)
    print(f'CLIP {num}: {result.get("verdict", result.get("error"))} {result["wall_sec"]}s', flush=True)
    return num,result

def probe(src):
    num=vid_num(src)
    info=video.probe(src)
    positive=[t for t in labels.get(num,[]) if t is not None]
    center=positive[0] if positive else info.duration/2
    times=np.linspace(max(0,center-4),min(info.duration-0.05,center+4),8)
    frames=[]
    # ffmpeg preserves rotation; the same exact timestamps are used for all candidates.
    import subprocess, cv2
    for t in times:
        p=subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(float(t)),'-i',str(src),
                          '-frames:v','1','-vf','scale=640:-2','-f','image2pipe','-vcodec','mjpeg','-'],
                         capture_output=True,check=True,timeout=30)
        img=cv2.imdecode(np.frombuffer(p.stdout,np.uint8),cv2.IMREAD_COLOR)
        frames.append((float(t),img))
    content=[{'type':'text','text':'Inspect this chronological CCTV sequence from a bank cash desk.'}]
    for t,img in frames:
        content += [{'type':'text','text':f't={t:.2f}s'}, {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+vlm._b64(img,640)}}]
    schema={'type':'object','properties':{'events':{'type':'array','items':{'type':'object','properties':{
        'name':{'type':'string'},'timecode_sec':{'type':'number'},'observation':{'type':'string'},
        'reason_for_concern':{'type':'string'}},'required':['name','timecode_sec','observation','reason_for_concern'],
        'additionalProperties':False}}},'required':['events'],'additionalProperties':False}
    body={'model':'vlm','temperature':0,'max_tokens':700,'chat_template_kwargs':{'enable_thinking':False},
          'messages':[{'role':'system','content':
          'Describe clearly visible cash-handling or physical-security concerns. No predefined list of violation '
          'types is supplied: give each concern a specific short name. State the visible action or condition '
          'and why it warrants review. Do not assume an action did not happen just because it is not shown. '
          'Do not invent legal duties, intent, identity, document contents or off-camera events. Ordinary '
          'customer service is not itself a concern. If no concern is supported, return an empty events list.'},
          {'role':'user','content':content}], 'response_format':{'type':'json_schema','json_schema':{
              'name':'open_review','strict':True,'schema':schema}}}
    started=time.time()
    try: result=vlm._post_json(body,time.time()+90)
    except Exception as exc: result={'error':f'{type(exc).__name__}: {exc}'}
    result.update(video=num,wall_sec=round(time.time()-started,2),window=[float(times[0]),float(times[-1])],
                  caveat='Diagnostic on known footage with oracle event windows; not unseen-type accuracy.')
    save(OUT/f'open-{num}.json',result)
    print(f'OPEN {num}: {len(result.get("events",[]))} events {result.get("error","")}',flush=True)

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(one,videos))
    counts=dict(TP=0,FP=0,TN=0,FN=0,errors=0)
    type_hits=type_total=0
    for num,result in results.items():
        truth=any(t is not None for t in labels.get(num,[]))
        if 'error' in result:
            counts['errors']+=1
            continue
        pred=result.get('verdict')=='violation'
        counts['TP' if truth and pred else 'FN' if truth else 'FP' if pred else 'TN']+=1
        if truth and pred and num in types:
            type_total+=1
            type_hits+=bool(types[num]&{v['type_id'] for v in result['extended']['violations']})
    n=len(results)
    metric={**counts,'n':n,'accuracy_all_jobs':(counts['TP']+counts['TN'])/n,
            'recall_completed':counts['TP']/max(1,counts['TP']+counts['FN']),
            'type_hits':type_hits,'type_evaluated':type_total,
            'max_seconds':max(v['wall_sec'] for v in results.values()),
            'over_300s':sum(v['wall_sec']>300 for v in results.values())}
    save(OUT/'metrics.json',metric)
    print('METRICS '+json.dumps(metric),flush=True)
    # A technical regression demonstration, not model accuracy.
    e=analyze.Event('other',2.0,.95,['visible event'],['sweep'],[(0,12)],other_name='New observed action',support=1,verified='yes')
    save(OUT/'novel_filter_check.json',{'confirmed_single_window_accepted':bool(analyze.select([e],True)),
        'expected_product_behavior':'A confirmed novel event must not require four independent windows solely because its type is new.'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(probe,videos))
    print('EVALUATION_DONE',flush=True)

if __name__=='__main__': main()
