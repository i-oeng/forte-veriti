"""Collect compact benchmark artifacts, including incomplete video coverage."""
import json,pathlib,statistics

ROOT=pathlib.Path('/workspace/forte/var/bakeoff/candidates-20261017')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    report={'status':load(ROOT/'status.json'),'models':{}}
    for name in ['q4b','cosmos8b','q27b','internvl14b']:
        folder=ROOT/name
        if not folder.exists():continue
        rows=[]
        for p in sorted(folder.glob('*.json')):
            if not p.stem.isdigit():continue
            data=load(p);ext=data.get('extended') or {};debugpath=folder/'media'/p.stem/'debug.json'
            debug=load(debugpath) if debugpath.exists() else {}
            requested=debug.get('fact_samples',0);obtained=len(debug.get('facts',[]))
            rows.append({'video':int(p.stem),'verdict':data.get('verdict'),'error':data.get('error'),
                'wall_sec':data['wall_sec'],'timing':ext.get('timing'),
                'windows_reviewed':ext.get('windows_reviewed'),'windows_skipped':ext.get('windows_skipped'),
                'fact_samples_requested':requested,'fact_samples_obtained':obtained,
                'violations':[{'type':v['type'],'type_id':v.get('type_id'),'t':v['timecode_sec'],
                    'observation':v.get('observation')} for v in ext.get('violations',[])]})
        times=sorted(x['wall_sec'] for x in rows)
        model={'completed_jobs':len(rows),'clips':rows,
               'metrics':load(folder/'metrics.json') if (folder/'metrics.json').exists() else None,
               'median_sec':statistics.median(times) if times else None,
               'p95_sec':times[min(len(times)-1,int(.95*len(times)))] if times else None,
               'incomplete_facts':sum(x['fact_samples_obtained']<x['fact_samples_requested'] for x in rows),
               'skipped_review_windows':sum(x['windows_skipped'] or 0 for x in rows),
               'open_diagnostics':[load(p) for p in sorted(folder.glob('open-*.json'))]}
        report['models'][name]=model
    temporary=ROOT/'comparison.tmp'
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(ROOT/'comparison.json')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ['clips','open_diagnostics']} for k,v in report['models'].items()},indent=2))
if __name__=='__main__':main()
