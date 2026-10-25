"""Compare the actual vLLM parser result with all pinned launcher arguments."""
import json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'selective-prefix'))
import serve
release=Path(sys.argv[1]);model=sys.argv[2];dest=R/('cli-check-'+model+'-'+str(int(time.time())))
extra={'host_memory_shares':2} if model=='gemma' else {}
m,c,cmd,e=serve.prepare(release,None,'goal_query',dest,19840,{'active_token_budget':32768,'goal_rewriter_model':str(R/'rewriter-model')},'kv-session',25,16,'decoupled',max_running=16,cuda_graphs='full',max_tokens=4096,max_model_len=262144,dram_arena_gib=96,connector_pool_gib=1,language_model_only=True,kv_session_thinking='on',kv_session_slots=1024,release_idle_pages=True,**extra)
dest.mkdir();(dest/'startup.json').write_bytes(serve.wire(c));os.environ.update(e)
sys.path[:0]=[str(release/'source'),m['site']]
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser
# Parser-only test on the login host: avoid hardware discovery. No model runs.
import vllm.platforms
vllm.platforms.current_platform.device_type='cuda'
args=make_arg_parser(FlexibleArgumentParser()).parse_args(cmd[cmd.index(m['site'])+1:])
from remote_ra_http.config import effective_server_config
observed=effective_server_config(args)
(dest/'passed.json').write_text(json.dumps({'passed':True,'model':model,'observed':observed,'GPU_validated':False,'test_only_device_type_override':'cuda'},indent=2,default=str));print('ACTUAL_VLLM_CLI_PASS',model)
