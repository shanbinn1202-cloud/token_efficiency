"""Transport, original-prompt and constraint regression; no GPU needed."""
import concurrent.futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'selective-prefix/source-overlay'))
from dram_kv.schema import RetrievalGoal
from dram_kv.goals.rewriter import RewriteRequest, RewriteCommandError
from dram_kv_scripts.qwen_goal_rewriter import QwenGoalGenerator
from remote_ra_live_v2.vllm_rewriter import VLLMGoalRewriter, prompt_and_schema

class Captured(Exception):pass
class Tokenizer:
    def __init__(self, capture=False):self.capture=capture
    def apply_chat_template(self,messages,**kw):
        self.seen=(messages,kw)
        if self.capture:raise Captured()
        return json.dumps(self.seen,ensure_ascii=False)
    def __call__(self,text):return {'input_ids':list(text.encode())}

def request(trigger='entity_drift',allow=False):
    return RewriteRequest('修复问题',RetrievalGoal('Fix parser',constraints=('retain API',)),
                          'pytest tests/test_parser.py','x'*9000,{'files':['parser.py']},trigger,{},allow)

class Tests(unittest.TestCase):
    def test_prompt_matches_original_for_both_constraint_modes(self):
        for trigger,allow in [('entity_drift',False),('user_request',True)]:
            payload=request(trigger,allow).to_payload()
            original=QwenGoalGenerator.__new__(QwenGoalGenerator)
            original.tokenizer=Tokenizer(True)
            with self.assertRaises(Captured):original.generate(payload)
            tk=Tokenizer();tokens,schema=prompt_and_schema(payload,tk)
            self.assertEqual(original.tokenizer.seen,tk.seen)
            self.assertEqual('constraints' in schema['properties'],allow)
            self.assertTrue(tokens)

    def run_server(self,invalid=False,concurrent=1):
        barrier=threading.Barrier(concurrent)
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append(body)
                barrier.wait(5)
                candidate={'commands':['rm -rf /']} if invalid else {'description':'fixed parser'}
                response={'choices':[{'text':json.dumps(candidate),'finish_reason':'stop'}],
                          'usage':{'completion_tokens':8}}
                raw=json.dumps(response).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)))
                self.end_headers();self.wfile.write(raw)
        class Server(ThreadingHTTPServer):request_queue_size=64
        server=Server(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        client=VLLMGoalRewriter.__new__(VLLMGoalRewriter)
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
        client.endpoint=f'http://127.0.0.1:{server.server_port}'
        client.pin={'max_new_tokens':384,'timeout_seconds':10}
        client.runtime={'backend':'vllm-0.24.0-lm-format-enforcer','telemetry':str(Path(td.name)/'events.jsonl')}
        client.tokenizer=Tokenizer();client.closed=False
        return client,captured

    def test_sixteen_rewrites_reach_service_concurrently(self):
        client,captured=self.run_server(concurrent=16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results=list(pool.map(lambda _:client.rewrite(request()),range(16)))
        self.assertEqual(len(results),16)
        for result,body in zip(results,captured):
            self.assertEqual(result.candidate.constraints,('retain API',))
            self.assertEqual(result.candidate.description,'fixed parser')
            self.assertEqual(body['temperature'],0)
            self.assertEqual(body['max_tokens'],384)
            self.assertEqual(body['stop_token_ids'],[151645,151643])
            self.assertFalse(body['structured_outputs']['json']['additionalProperties'])

    def test_original_validator_rejects_invalid_output(self):
        client,_=self.run_server(invalid=True)
        with self.assertRaises(RewriteCommandError):client.rewrite(request())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (R/'vllm-rewriter-cpu-tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun})+'\n')
    raise SystemExit(not result.wasSuccessful())
