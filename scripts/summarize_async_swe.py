"""Summarize completed runs with main-agent tokens / batch wall time."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted(a.bundle.glob('*/completed.json')):
        complete=json.loads(path.read_text())
        if complete.get('mode')!='run' or not complete.get('passed'):
            continue
        m=json.loads((path.parent/'results/metrics.json').read_text())
        rows.append(dict(method=complete['method'],budget=complete['budget'],attempt=path.parent.name,
            instances=m['instances'],resolved=sum(c['resolved'] is True for c in m['cases']),
            accuracy_percent=m['accuracy']*100,output_tokens_per_second=m['agent_output_tokens_per_second'],
            batch_wall_seconds=m['batch_wall_seconds'],agent_output_tokens=m['agent_output_tokens'],
            reflection_output_tokens=m['agentdiet_output_tokens'],eligible_for_comparison=m['eligible_for_comparison'],
            grading_errors=json.dumps(m['grading_errors']),
            exits=json.dumps({c['instance_id']:c['exit_status'] for c in m['cases']})))
    (a.bundle/'summary.json').write_text(json.dumps(rows,indent=2))
    if rows:
        with (a.bundle/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(f"{row['method']:16} {row['budget']:3} {row['resolved']}/{row['instances']} "
              f"{row['output_tokens_per_second']:.2f} tokens/s eligible={row['eligible_for_comparison']}")


if __name__=='__main__':
    main()
