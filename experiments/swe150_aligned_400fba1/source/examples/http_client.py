"""Minimal ordinary OpenAI HTTP client; external tool execution is harness-owned."""
import argparse
import json
from pathlib import Path
import urllib.request

def completion(base,messages,*,tools=None):
    body=dict(model='selective-prefix',messages=messages,max_tokens=16,temperature=0,
              seed=42,n=1,stream=False,chat_template_kwargs={'enable_thinking':False})
    if tools is not None:body['tools']=tools
    request=urllib.request.Request(base.rstrip('/')+'/chat/completions',
        data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    # No automatic retry: timeout/disconnect may follow accepted admission.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=300) as response:return json.load(response)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    parser.add_argument('--messages',required=True,type=Path)
    parser.add_argument('--tools',type=Path)
    args=parser.parse_args()
    print(json.dumps(completion(args.base_url,json.loads(args.messages.read_bytes()),
        tools=None if args.tools is None else json.loads(args.tools.read_bytes())),ensure_ascii=False))
if __name__=='__main__':main()
