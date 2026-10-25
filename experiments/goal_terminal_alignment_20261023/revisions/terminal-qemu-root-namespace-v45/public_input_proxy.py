"""Local byte-identical cache of public CIFAR data; pass other HTTP through."""
import hashlib,http.server,json,sys,urllib.request,urllib.parse
from pathlib import Path
ARTIFACT=Path(sys.argv[1]);READY=Path(sys.argv[2]);AUDIT=Path(sys.argv[3])
EXPECTED='c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd'
assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()==EXPECTED
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.0'
 def log_message(self,*a):pass
 def do_HEAD(self):self.serve()
 def do_GET(self):self.serve()
 def do_POST(self):self.serve()
 def serve(self):
  url=urllib.parse.urlsplit(self.path)
  cached=(url.hostname in {'www.cs.toronto.edu','cave.cs.toronto.edu'} and
          url.path in {'/~kriz/cifar-10-binary.tar.gz','/kriz/cifar-10-binary.tar.gz'} and
          not url.query and self.command in {'GET','HEAD'})
  if cached:
   size=ARTIFACT.stat().st_size;start,end=0,size-1;status=200
   part=self.headers.get('Range')
   if part:
    import re
    m=re.fullmatch(r'bytes=(\d+)-(\d*)',part)
    if not m:self.send_error(416);return
    start=int(m[1]);end=min(size-1,int(m[2]) if m[2] else size-1);status=206
    if not 0<=start<=end<size:self.send_error(416);return
   self.send_response(status);self.send_header('Content-Length',str(end-start+1))
   self.send_header('Content-Type','application/gzip');self.send_header('Accept-Ranges','bytes')
   if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
   self.end_headers()
   if self.command!='HEAD':
    with ARTIFACT.open('rb') as f:
     f.seek(start);left=end-start+1
     while left:
      data=f.read(min(1024*1024,left));self.wfile.write(data);left-=len(data)
   with AUDIT.open('a') as f:f.write(json.dumps({'source':self.path,'sha256':EXPECTED,'range':[start,end],'cache_hit':True})+'\n')
   return
  if url.scheme not in {'http','https'} or not url.netloc:self.send_error(400);return
  skip={'host','connection','proxy-connection','proxy-authorization','transfer-encoding'}
  headers={k:v for k,v in self.headers.items() if k.lower() not in skip}
  body=self.rfile.read(int(self.headers.get('Content-Length','0'))) if self.command=='POST' else None
  request=urllib.request.Request(self.path,data=body,headers=headers,method=self.command)
  try:
   try:response=opener.open(request,timeout=180)
   except urllib.error.HTTPError as e:response=e
   with response:
    self.send_response(response.status)
    for k,v in response.headers.items():
     if k.lower() not in {'connection','transfer-encoding'}:self.send_header(k,v)
    self.end_headers()
    if self.command!='HEAD':
     while data:=response.read(1024*1024):self.wfile.write(data)
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:self.send_error(502,str(type(e).__name__))
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
READY.write_text(json.dumps({'port':server.server_port,'artifact_sha256':EXPECTED}))
server.serve_forever()
