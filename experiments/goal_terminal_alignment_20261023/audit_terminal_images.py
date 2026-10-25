"""Read-only OCI platform audit; never pulls/runs an incompatible image."""
import concurrent.futures
import json
from pathlib import Path
import time
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parent
ACCEPT = ', '.join(['application/vnd.oci.image.index.v1+json', 'application/vnd.docker.distribution.manifest.list.v2+json', 'application/vnd.docker.distribution.manifest.v2+json'])

def audit(task):
    cfg = tomllib.loads((task/'task.toml').read_text())
    image = cfg.get('environment', {}).get('docker_image')
    row = {'task':task.name,'image':image,'arm64':False}
    try:
        if not image:
            return dict(row,status='build_required')
        repo, tag = image.rsplit(':', 1)
        token=json.load(urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repo+':pull',timeout=30))['token']
        headers={'Authorization':'Bearer '+token,'Accept':ACCEPT}
        base='https://registry-1.docker.io/v2/'+repo
        with urllib.request.urlopen(urllib.request.Request(base+'/manifests/'+tag,headers=headers),timeout=30) as response:
            manifest=json.load(response);row['digest']=response.headers.get('Docker-Content-Digest')
        platforms=[m.get('platform',{}) for m in manifest.get('manifests',[])]
        if 'config' in manifest:
            with urllib.request.urlopen(urllib.request.Request(base+'/blobs/'+manifest['config']['digest'],headers=headers),timeout=30) as response:
                config=json.load(response)
            platforms=[{k:config.get(k) for k in ('os','architecture')}]
        row.update(platforms=platforms,arm64=any(p.get('os')=='linux' and p.get('architecture')=='arm64' for p in platforms),status='checked')
    except Exception as exc:
        row.update(status='error',error=type(exc).__name__+': '+str(exc))
    return row

if __name__=='__main__':
    tasks=sorted(p.parent for p in (ROOT/'terminal-bench-2').glob('*/task.toml'))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(audit,tasks):
            rows.append(row)
            target=ROOT/'terminal-image-architectures.json'
            tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps({'time':time.time(),'total':len(tasks),'checked':len(rows),'rows':rows},indent=2)+'\n');tmp.replace(target)
            print(row['task'],row['status'],row.get('platforms'),flush=True)
