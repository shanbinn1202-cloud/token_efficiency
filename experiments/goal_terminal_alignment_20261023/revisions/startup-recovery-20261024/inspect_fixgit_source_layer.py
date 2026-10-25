import urllib.request,json
repo='alexgshaw/fix-git'
token=json.load(urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repo+':pull'))['token']
def get(path,accept=None):
 h={'Authorization':'Bearer '+token}
 if accept:h['Accept']=accept
 return urllib.request.urlopen(urllib.request.Request('https://registry-1.docker.io/v2/'+repo+'/'+path,headers=h),timeout=60)
m=json.load(get('manifests/20251031','application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'))
if 'manifests' in m:
 md=next(x for x in m['manifests'] if x.get('platform',{}).get('architecture')=='amd64');m=json.load(get('manifests/'+md['digest'],'application/vnd.docker.distribution.manifest.v2+json'))
c=json.load(get('blobs/'+m['config']['digest']))
i=0
for entry in c['history']:
 if entry.get('empty_layer'):continue
 layer=m['layers'][i];print(json.dumps({'layer':i,'size':layer['size'],'digest':layer['digest'],'command':entry.get('created_by','')}));i+=1
