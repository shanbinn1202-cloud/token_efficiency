"""Run the unmodified official reference solution against the ARM64 environment."""
import asyncio
import sys
import hashlib
import json
from pathlib import Path
import shutil
import toml
from harbor.models.trial.config import TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
from harbor.models.environment_type import EnvironmentType
from harbor.trial.trial import Trial

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'terminal-runtime'))
async def main():
    name='adaptive-rejection-sampler'
    src=R/'terminal-bench-2'/name;dst=R/'terminal-arm64-tasks'/name
    if not dst.exists():shutil.copytree(src,dst)
    config=toml.load(src/'task.toml')
    config['environment']['docker_image']=str(R/'terminal-images'/f'{name}.sif')
    (dst/'task.toml').write_text(toml.dumps(config))
    proof={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and p.name!='task.toml'}
    for name,h in proof.items():
        if hashlib.sha256((dst/name).read_bytes()).hexdigest()!=h:raise RuntimeError('task content changed: '+name)
    (R/'terminal-images/task-copy-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    cfg=TrialConfig(task=TaskConfig(path=dst),trial_name='arm64-oracle-sandbox',trials_dir=R/'terminal-validation',agent=AgentConfig(name='oracle'),environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment',kwargs={'singularity_image_cache_dir':str(R/'terminal-images/cache')}))
    trial=await Trial.create(cfg)
    result=await trial.run()
    (R/'terminal-images/oracle-probe-result.json').write_text(result.model_dump_json(indent=2)+'\n')
    print(result.model_dump_json(indent=2))
if __name__=='__main__':asyncio.run(main())
