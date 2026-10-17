"""Inspect a downloaded comparison without loading the inference dependencies."""
import argparse
import json
from pathlib import Path
import re
import statistics
import openpyxl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--labels', type=Path, default=Path('data/open_labels.xlsx'))
    args = parser.parse_args()
    truth = {}
    times = {}
    for row in openpyxl.load_workbook(args.labels).active.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        num = int(re.findall(r'\d+', str(row[0]))[0])
        digits = [int(x) for x in re.findall(r'\d+', str(row[1] or ''))]
        truth[num] = truth.get(num, False) or bool(digits)
        if digits:
            times.setdefault(num, []).append(digits[0]*60+digits[1] if len(digits)==2 else digits[0])
    report = json.loads(args.report.read_text(encoding='utf-8'))
    for name, model in report['models'].items():
        clips = model['clips']
        probes = model['open_diagnostics']
        completed = [p for p in probes if not p.get('error')]
        clean = [p for p in completed if not truth[p['video']]]
        time_errors = [min(abs(v['t']-t) for v in p['violations'])
                       for p in clips if not p.get('error') and p['violations']
                       for t in times.get(p['video'], [])]
        output = {
            'model': name, 'metrics': model['metrics'],
            'median_sec': model['median_sec'], 'p95_sec': model['p95_sec'],
            'incomplete_facts': model['incomplete_facts'],
            'skipped_review_windows': model['skipped_review_windows'],
            'missed_clips': [p['video'] for p in clips if not p.get('error') and truth[p['video']] and p['verdict'] != 'violation'],
            'false_alarm_clips': [p['video'] for p in clips if not p.get('error') and not truth[p['video']] and p['verdict'] == 'violation'],
            'probe_completed': len(completed), 'probe_errors': len(probes)-len(completed),
            'clean_probe_flagged': sum(bool(p.get('events')) for p in clean),
            'clean_probe_total': len(clean),
            'timecode_median_error_sec': statistics.median(time_errors) if time_errors else None,
            'timecode_within_3s': sum(t <= 3 for t in time_errors)/len(time_errors) if time_errors else None,
            'timecode_caveat': 'Nearest predicted event for detected positive clips only; not conditioned on correct violation type.',
            'probe_caveat': 'Known-footage oracle-window diagnostic; event presence does not establish correct recognition or unseen-type accuracy.',
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
