"""Model-specific integration only; retain current v0.24 eviction algorithms."""
from pathlib import Path
import shutil,ast,json,hashlib,difflib
B=Path(__file__).resolve().parent
E=Path('/projects/bcjw/ssun4/token_efficiency/engines/vllm_024_gemma_20261017')
Q=Path('/projects/bcjw/ssun4/token_efficiency/engines/vllm_024')
G=Path('/projects/bcjw/ssun4/gemma4_25g16_20261005/source')
R=Path('/projects/bcjw/ssun4/vllm024_triton_six_20261008')
H=B/'harness';H.mkdir()
for p in (R/'harness').iterdir():
 if p.name=='tebench':shutil.copytree(p,H/p.name,ignore=shutil.ignore_patterns('__pycache__'))
 else:(H/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
for name in ['base.py','vllm.py','kv_session.py','model_chat.py']:
 shutil.copy2(G/'tebench/backends'/name,H/'tebench/backends'/name)
# Reuse the previously verified native Gemma template/parser/EOS integration.
# Engine and selectors remain current v0.24; no FA4 settings are copied.
shutil.copy2(G/'engines/vllm_gemma/vllm/tebench/geometry.py',E/'vllm/tebench/geometry.py')
for name in ['agentkv.py','intentkv.py']:
 p=E/'vllm/tebench'/name;s=p.read_text()
 config='vllm_config' if name=='agentkv.py' else 'config'
 s=s.replace(f'    c = {config}.model_config.hf_text_config',f'    from vllm.tebench.geometry import full_attention\n\n    c = {config}.model_config.hf_text_config\n    geometry = full_attention({config})',1)
 s=s.replace('layers = [i for i, t in enumerate(c.layer_types) if t == "full_attention"]','layers = geometry.layers',1).replace('layers = [i for i, kind in enumerate(c.layer_types) if kind == "full_attention"]','layers = geometry.layers',1)
 s=s.replace('c.num_attention_heads','geometry.heads').replace('c.num_key_value_heads','geometry.kv_heads').replace('c.head_dim','geometry.dim');p.write_text(s)
# Reuse full/sliding/mamba lossless snapshot/restore, without changing selectors.
for name in ['random_attention.py','agentkv_connector.py']:
 p=E/'vllm/tebench'/name;s=p.read_text();g=(G/'engines/vllm_gemma/vllm/tebench'/name).read_text()
 for fname in ['snapshot','restore_after_zero']:
  def bounds(text):
   start=text.index('\ndef '+fname+'(');end=text.find('\ndef ',start+1);return start,len(text) if end<0 else end
  a,z=bounds(s);ga,gz=bounds(g);s=s[:a]+g[ga:gz]+s[z:]
 p.write_text(s)
p=E/'vllm/tebench/kv_session.py';s=p.read_text().replace('def gather_kv(cache, blocks, block_size, count):','def gather_kv(cache, blocks, block_size, count, start=0):').replace('positions = torch.arange(count, device=cache.device)','positions = torch.arange(start, start + count, device=cache.device)');p.write_text(s)
# IntentKV eviction uses Random's snapshot but its own restore loop.
p=E/'vllm/tebench/intentkv.py';s=p.read_text().replace('from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec','from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec')
needle='                elif isinstance(group.kv_cache_spec, MambaSpec):'
branch='''                elif isinstance(group.kv_cache_spec, SlidingWindowSpec):
                    size = group.kv_cache_spec.block_size
                    start = state.window_offsets[name]
                    write_dense(cache, req.block_ids[gid], size, values.to(cache.device), start=start)
                    if os.environ.get("VLLM_INTENTKV_VERIFY") == "1":
                        torch.testing.assert_close(
                            gather_kv(cache, req.block_ids[gid], size, len(values), start=start).cpu(),
                            values, atol=0, rtol=0,
                        )
'''
assert s.count(needle)==1;s=s.replace(needle,branch+needle);p.write_text(s)
p=E/'vllm/model_executor/models/gemma4.py';s=p.read_text();start=s.index('\n    def forward(',s.index('class Gemma4Attention'))
init='''
        self.agentkv_enabled = False
        self.intentkv_enabled = False
        self.tebench_layer_id = layer_idx
        if not self.is_sliding:
            from vllm.config import get_current_vllm_config
            from vllm.tebench import agentkv, intentkv
            self.agentkv_enabled = agentkv.ENABLED
            self.intentkv_enabled = intentkv.ENABLED
            if self.agentkv_enabled:
                agentkv.initialize(get_current_vllm_config(), self.q_norm.weight.device)
            if self.intentkv_enabled:
                intentkv.initialize(get_current_vllm_config(), self.q_norm.weight.device)
'''
s=s[:start]+init+s[start:]
needle='        attn_output = self.attn(q, k, v)';assert s.count(needle)==1
s=s.replace(needle,'''        if self.agentkv_enabled:
            from vllm.tebench.agentkv import capture_q
            capture_q(q, self.tebench_layer_id)
        if self.intentkv_enabled:
            from vllm.tebench.intentkv import capture_q as capture_intent
            capture_intent(q, self.tebench_layer_id)
'''+needle);p.write_text(s)
# CPU parser can import vLLM without loading model/GPU; architecture already supported in v0.24.
changes=[]
for p in E.rglob('*.py'):
 if '__pycache__' in str(p):continue
 q=Q/p.relative_to(E)
 if not q.exists() or p.read_bytes()!=q.read_bytes():
  ast.parse(p.read_text());changes.append(str(p.relative_to(E)))
(B/'engine-port.json').write_text(json.dumps(dict(base=str(Q),engine=str(E),changed=changes,policy='Preserve current eviction selectors, Triton kernels and async scheduler. Port only full-attention geometry, Q capture, and native local-window checkpoints from previously verified Gemma integration.'),indent=2)+'\n')
patch=''.join(''.join(difflib.unified_diff((Q/p).read_text().splitlines(True) if (Q/p).exists() else [],(E/p).read_text().splitlines(True),fromfile='base/'+p,tofile='gemma/'+p)) for p in changes)
(B/'engine-port.patch').write_text(patch)
print('PORT_READY',changes)
