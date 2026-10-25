from pathlib import Path
import subprocess,tempfile,time,json,sys,http.client,http.server,threading
R=Path(__file__).resolve().parent
class Origin(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  self.send_response(200);self.send_header('Content-Length','11');self.end_headers();self.wfile.write(b'passthrough')
origin=http.server.HTTPServer(('127.0.0.1',0),Origin);threading.Thread(target=origin.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory() as td:
 t=Path(td);p=subprocess.Popen([sys.executable,str(R/'public_input_proxy.py'),'/tmp/ssun4-terminal-public-cache/cifar-10-binary.tar.gz',str(t/'ready.json'),str(t/'audit.jsonl')])
 try:
  for _ in range(100):
   if (t/'ready.json').exists():break
   assert p.poll() is None;time.sleep(.1)
  port=json.loads((t/'ready.json').read_text())['port']
  def request(url,headers={}):
   c=http.client.HTTPConnection('127.0.0.1',port);c.request('GET',url,headers=headers);r=c.getresponse();data=r.read();status=r.status;h=dict(r.getheaders());c.close();return status,h,data
  status,h,data=request('http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',{'Range':'bytes=100-199'})
  assert status==206 and h['Content-Range']=='bytes 100-199/170052171'
  with open('/tmp/ssun4-terminal-public-cache/cifar-10-binary.tar.gz','rb') as f:f.seek(100);assert data==f.read(100)
  status,h,data=request('http://127.0.0.1:%d/test'%origin.server_port);assert status==200 and data==b'passthrough'
  print('PASS exact cached bytes, Range semantics, ordinary HTTP pass-through')
 finally:p.terminate();p.wait();origin.shutdown()
