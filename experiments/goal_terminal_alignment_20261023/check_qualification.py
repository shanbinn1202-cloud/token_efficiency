"""Fail closed on unreviewed GPU results or source/config changes."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
kind,model=sys.argv[1:]
p=R/'qualification'/(kind+'-'+model+'.json')
if not p.exists():raise SystemExit('GPU qualification and review are pending: '+str(p))
d=json.loads(p.read_text())
if d.get('passed') is not True:raise SystemExit('Qualification did not pass')
for n,h in d['source_sha256'].items():
 if hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h:raise SystemExit('Qualified source/config changed: '+n)
for n,h in d['evidence_sha256'].items():
 if hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h:raise SystemExit('Qualification evidence changed: '+n)
print('QUALIFIED',kind,model)
