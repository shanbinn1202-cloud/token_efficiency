"""Verify imported closure dependencies against the installed candidate."""
import hashlib
import importlib
import json
from pathlib import Path
from .contract import require

def verify_closure(runner):
    source=Path(__file__).resolve().parents[1]
    release=source.parent
    manifest=json.loads((release/'runtime-manifest.json').read_bytes())
    closure=manifest['source_closure'];evidence={}
    # Verify every shipped closure dependency byte, plus origins for modules
    # used in geometry/ownership/cleanup. Optional unrelated modules needn't load.
    for name in closure['files']:
        path=source/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual==manifest['source_files'][name],'source closure candidate byte mismatch: '+name)
    modules=('vllm.v1.worker.block_table','vllm.v1.worker.utils',
        'vllm.v1.worker.gpu.attn_utils','vllm.v1.worker.gpu_model_runner',
        'vllm.v1.worker.mamba_utils','vllm.v1.core.block_pool',
        'vllm.v1.core.kv_cache_coordinator','vllm.v1.core.single_type_kv_cache_manager',
        'vllm.v1.worker.kv_connector_model_runner_mixin',
        'vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker',
        'vllm.v1.kv_offload.cpu.gpu_worker')
    for name in modules:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        relative=name.replace('.','/')+'.py'
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual==manifest['source_files'][relative], 'loaded closure module hash mismatch: '+name)
        evidence[name]=dict(path=str(path),sha256=actual)
    backends={}
    index_path=release/'release-index.json'
    index=json.loads(index_path.read_bytes())['files'] if index_path.is_file() else {}
    for group in runner._kv_cache_spec_attn_group_iterator():
        name=group.backend.__module__
        if name in backends:continue
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        relative=name.replace('.','/')+'.py'
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        expected=manifest['source_files'].get(relative,index.get('source/'+relative))
        if expected is not None:
            require(actual==expected,'loaded attention backend hash mismatch: '+name)
        backends[name]=dict(path=str(path),sha256=actual,
            capture_attested=relative in closure['files'],
            installed_hash_verified=expected is not None)
    return dict(capture_manifest=closure['manifest_sha256'],modules=evidence,backends=backends)
