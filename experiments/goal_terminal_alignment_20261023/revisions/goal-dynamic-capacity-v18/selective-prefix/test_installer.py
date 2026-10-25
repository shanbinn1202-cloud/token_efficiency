"""Local file transactions and launch contract. No GPU, server or environment edits."""
import importlib.util
import base64
from contextlib import redirect_stdout
import csv
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tarfile
import io
import shutil
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import install
import serve

ROOT=Path(__file__).resolve().parent
def write(path,raw):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
def fixture(root):
    package=root/'package';site=root/'site';model=root/'model';release=root/'release'
    source={'vllm/example.py':b'VALUE=2\n','vllm/added.py':b'NEW=True\n','policy_runtime/__init__.py':b'ACTIVE=False\n'}
    for name,raw in source.items():write(package/'source-overlay'/name,raw)
    write(site/'vllm/example.py',b'VALUE=1\n')
    versions={'vllm':'0.24.0','transformers':'5.13.0','tokenizers':'0.22.2'}
    for name,version in versions.items():write(site/f'{name}-{version}.dist-info/METADATA',f'Name: {name}\nVersion: {version}\n'.encode())
    write_record(site)
    for name in ('transformers','tokenizers','jinja2','markupsafe'):write(site/name/'__init__.py',b'VALUE=1\n')
    write(site/'tokenizers/tokenizers.so',b'FAKE_NATIVE_NOT_LOADED')
    write(model/'config.json',b'{}');write(model/'tokenizer.json',b'{}');write(model/'tokenizer_config.json',b'{"chat_template":"fixture"}')
    pins={name:install.digest(raw) for name,raw in source.items()}
    definition=dict(format_version=3,feature_default='off',source_files=pins,payload=pins,
        expected_live={'vllm/example.py':install.digest(b'VALUE=1\n'),'vllm/added.py':None},
        source_closure={'files':{},'manifest_sha256':install.digest(b'{}')},qualified_dependency_versions=versions,
        model_pins={p.name:install.digest(p.read_bytes()) for p in model.iterdir()},
        release_id='fixture-only',model_revision='a'*40,provenance={})
    write(package/'INSTALL_RELEASE.json',install.encoded(definition))
    write(package/'SOURCE_SHA256SUMS.txt',''.join(f'{pin}  source-overlay/{name}\n' for name,pin in sorted(pins.items())).encode())
    return package,site,model,release

def write_record(site):
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    for path in sorted((site/'vllm').rglob('*')):
        if path.is_file():
            raw=path.read_bytes();pin=base64.urlsafe_b64encode(bytes.fromhex(install.digest(raw))).rstrip(b'=').decode()
            writer.writerow([path.relative_to(site).as_posix(),'sha256='+pin,len(raw)])
    # Legitimate wheel metadata and generated scripts are not copied by install.
    writer.writerow(['vllm-0.24.0.dist-info/RECORD','',''])
    writer.writerow(['../../../bin/vllm','',''])
    write(site/'vllm-0.24.0.dist-info/RECORD',stream.getvalue().encode())

class InstallerTests(unittest.TestCase):
    def test_stale_source_closure_hash_is_refused(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory))
            path=package/'INSTALL_RELEASE.json';definition=json.loads(path.read_bytes())
            definition['source_closure']['files']={'vllm/example.py':'0'*64}
            path.write_bytes(install.encoded(definition))
            with self.assertRaisesRegex(ValueError,'closure hash differs'):
                install.install(site,model,release,package_root=package,apply=True)
            self.assertFalse(release.exists())

    def test_complete_wheel_record_rejects_uncovered_mutations(self):
        for mutation in ('extra','missing','changed','no_record','missing_hash'):
            with self.subTest(mutation=mutation),TemporaryDirectory() as directory:
                package,site,model,release=fixture(Path(directory))
                write(site/'vllm/unrelated.py',b'UNTOUCHED=1\n');write_record(site)
                if mutation=='extra':write(site/'vllm/foreign.py',b'BAD=1\n')
                elif mutation=='missing':(site/'vllm/unrelated.py').unlink()
                elif mutation=='changed':write(site/'vllm/unrelated.py',b'UNTOUCHED=2\n')
                elif mutation=='no_record':(site/'vllm-0.24.0.dist-info/RECORD').unlink()
                else:write(site/'vllm-0.24.0.dist-info/RECORD',b'vllm/example.py,,\n')
                before=install.inventory(site)
                with self.assertRaisesRegex(ValueError,'RECORD'):install.install(site,model,release,package_root=package,apply=True)
                self.assertFalse(release.exists());self.assertEqual(before,install.inventory(site))

    def test_record_allows_ignored_bytecode_and_checks_sizes(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory))
            write(site/'vllm/__pycache__/example.cpython-312.pyc',b'generated')
            install.install(site,model,release,package_root=package,apply=True)
            self.assertFalse((release/'source/vllm/__pycache__').exists())
            path=site/'vllm-0.24.0.dist-info/RECORD'
            write(path,path.read_bytes().replace(b',8\r\n',b',999\r\n'))
            with self.assertRaisesRegex(ValueError,'size mismatch'):install.verify_vllm_record(site)

    def test_read_only_environment_and_rollback_preserve_source_modes(self):
        for fail in (None,2):
            with self.subTest(fail=fail),TemporaryDirectory() as directory:
                package,site,model,release=fixture(Path(directory));paths=list(site.rglob('*'))
                before=install.inventory(site);modes={p:p.stat().st_mode for p in paths}
                for path in paths:
                    if path.is_file():path.chmod(stat.S_IRUSR)
                readonly={p:p.stat().st_mode for p in paths}
                try:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'injected'):
                            install.install(site,model,release,package_root=package,apply=True,fail_after=fail)
                        self.assertFalse(release.exists())
                    else:
                        install.install(site,model,release,package_root=package,apply=True)
                        install.installed_check(release)
                    self.assertEqual(before,install.inventory(site))
                    self.assertEqual(readonly,{p:p.stat().st_mode for p in paths})
                finally:
                    for path,mode in modes.items():path.chmod(mode)

    def test_interpreter_identity_is_checked_by_installed_check_and_launcher(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory));install.install(site,model,release,package_root=package,apply=True)
            expected=install.environment_identity(site)
            for field in ('python_version','machine','prefix','executable','site'):
                altered=dict(expected);altered[field]='different'
                with patch.object(install,'environment_identity',return_value=altered):
                    with self.assertRaisesRegex(ValueError,'environment mismatch: '+field):install.installed_check(release)
                    with self.assertRaisesRegex(ValueError,'environment mismatch'):
                        serve.prepare(release,dict(instance_id='fixture',problem_statement='fix'), 'full',Path(directory)/'out',8231)

    @unittest.skipIf(os.environ.get('SELECTIVE_PREFIX_ARCHIVE_TEST')=='1','already testing extracted Git archive')
    def test_git_archive_head_preserves_pins_and_runs_full_suite(self):
        # Build a disposable repository from the candidate worktree so this
        # regression also runs before the shared repository has been committed.
        # check_git_release.py --ref HEAD is the separate release gate on the
        # actual commit. Neither operation changes this repository's index.
        import package
        import check_git_release
        with TemporaryDirectory() as directory:
            snapshot=Path(directory)/'repo';snapshot.mkdir()
            definition=install.verify_package(ROOT)
            names=[*package.PUBLIC_FILES,'.gitattributes',*('source-overlay/'+n for n in definition['source_files'])]
            for name in names:write(snapshot/name,(ROOT/name).read_bytes())
            def git(*args):
                subprocess.run(['git','-C',str(snapshot),*args],check=True,capture_output=True)
            git('init','--quiet');git('add','--all')
            git('-c','user.name=Archive Test','-c','user.email=archive-test@example.invalid','commit','--quiet','-m','Candidate snapshot')
            result=check_git_release.check(snapshot,'HEAD')
            self.assertEqual(result['status'],'GIT_ARCHIVE_TESTS_PASS')

    def test_actual_package_coverage_and_default_off(self):
        definition=install.verify_package(ROOT)
        self.assertEqual(len(definition['source_files']),len(install.inventory(ROOT/'source-overlay')))
        # Exact release counts (Stage B): a new or dropped overlay file must be deliberate.
        self.assertEqual((len(definition['source_files']),len(definition['payload'])),(323,232))
        self.assertEqual(len(definition['expected_live']),120)
        self.assertEqual(sum(v is None for v in definition['expected_live'].values()),2)
        self.assertEqual(len(definition['payload']),len(definition['source_files'])-sum(1 for n,p in definition['source_files'].items() if definition['expected_live'].get(n)==p))

    def test_check_does_not_mutate_or_enable(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory));before=install.inventory(site)
            result=install.install(site,model,release,package_root=package)
            self.assertEqual(result['status'],'CHECK_ONLY_PASS');self.assertFalse(release.exists())
            self.assertEqual(before,install.inventory(site));self.assertEqual(result['feature_default'],'off')

    def test_install_uninstall_quarantine_and_original_environment_unchanged(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory));before=install.inventory(site)
            install.install(site,model,release,package_root=package,apply=True)
            manifest=install.installed_check(release);self.assertFalse(manifest['default_enabled'])
            self.assertEqual(before,install.inventory(site))
            install.uninstall(release)
            result=install.uninstall(release,apply=True,processes_stopped=True)
            self.assertFalse(release.exists());backup=Path(result['backup']);self.assertTrue(backup.is_dir())
            backup.rename(release);install.installed_check(release)
            self.assertEqual(before,install.inventory(site))

    def test_mid_install_rollback(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory));before=install.inventory(site)
            with self.assertRaisesRegex(RuntimeError,'injected'):
                install.install(site,model,release,package_root=package,apply=True,fail_after=2)
            self.assertFalse(release.exists());self.assertEqual(before,install.inventory(site))

    def test_changed_baseline_absence_model_and_version_fail_before_write(self):
        mutations=[('vllm/example.py',b'CHANGED',False),('vllm/added.py',b'FOREIGN',False),
            ('tokenizer.json',b'CHANGED',True),('vllm-0.24.0.dist-info/METADATA',b'Name: vllm\nVersion: 0.23.0\n',False)]
        for name,raw,is_model in mutations:
            with self.subTest(name=name),TemporaryDirectory() as directory:
                package,site,model,release=fixture(Path(directory));write((model if is_model else site)/name,raw)
                with self.assertRaises(ValueError):install.install(site,model,release,package_root=package,apply=True)
                self.assertFalse(release.exists())

    def test_release_existing_and_drift_refused(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory))
            install.install(site,model,release,package_root=package,apply=True)
            with self.assertRaisesRegex(ValueError,'new release'):install.install(site,model,release,package_root=package,apply=True)
            write(release/'source/policy_runtime/__init__.py',b'USER_CHANGE')
            with self.assertRaisesRegex(ValueError,'drift'):install.uninstall(release,apply=True,processes_stopped=True)
            self.assertEqual((release/'source/policy_runtime/__init__.py').read_bytes(),b'USER_CHANGE')

    def test_uninstall_requires_stopped_and_preserves_unrelated_files(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory));install.install(site,model,release,package_root=package,apply=True)
            with self.assertRaisesRegex(ValueError,'stop every'):install.uninstall(release,apply=True)
            write(release/'user-note.txt',b'keep me')
            with self.assertRaisesRegex(ValueError,'extra files'):install.uninstall(release,apply=True,processes_stopped=True)
            self.assertTrue((release/'user-note.txt').exists())

    def test_source_package_added_unpinned_file_refused(self):
        with TemporaryDirectory() as directory:
            package,site,model,release=fixture(Path(directory));write(package/'source-overlay/extra.py',b'X=1')
            with self.assertRaisesRegex(ValueError,'coverage'):install.install(site,model,release,package_root=package)

    def test_launch_dry_run_and_four_policy_original_config_schema(self):
        spec=importlib.util.spec_from_file_location('original_config',ROOT/'source-overlay/remote_ra_http/config.py')
        config_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(config_module)
        from types import SimpleNamespace
        with TemporaryDirectory() as directory:
            root=Path(directory);package,site,model,release=fixture(root)
            install.install(site,model,release,package_root=package,apply=True)
            # The fixture release has no runtime tree; import the contract from this checkout.
            search=patch.object(sys,'path',[str(ROOT/'source-overlay')]+sys.path);search.start()
            self.addCleanup(search.stop)
            for policy in serve.POLICIES:
                out=root/policy;query=policy in ('goal_query','observation_query','random_fill')
                # Only goal_query needs a Goal-rewriter choice; observation_query and random_fill never use one.
                if policy=='goal_query':
                    with self.assertRaisesRegex(ValueError,'goal-rewriter-model'):
                        serve.prepare(release,dict(instance_id='fixture',problem_statement='fix it'),policy,out,8231)
                manifest,config,command,env=serve.prepare(release,dict(instance_id='fixture',problem_statement='fix it'),
                    policy,out,8231,dict(no_goal_rewriter=True) if policy=='goal_query' else None)
                self.assertEqual(config['schema'],'remote-ra-http-config-v2')
                self.assertEqual(config['selection_contract'] is not None,query)
                self.assertEqual(config['goal_rewriter'] is not None,query)
                self.assertEqual(command[command.index('--max-model-len')+1],'65536')
                self.assertFalse(out.exists());self.assertIn('selective-prefix',command)
                self.assertNotIn('VLLM_POLICY_RUNTIME_MANIFEST',env)
                self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'],serve.pin(config))
                with patch.object(config_module,'load',return_value=config):
                    config_module.effective_server_config(SimpleNamespace(**config['effective_server_config']))
            result=subprocess.run([sys.executable,'-B',str(ROOT/'serve.py'),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0);self.assertIn('--start',result.stdout)

    def test_observation_query_launch_pins_no_goal_rewriter(self):
        spec=importlib.util.spec_from_file_location('original_config',ROOT/'source-overlay/remote_ra_http/config.py')
        config_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(config_module)
        task=dict(instance_id='fixture',problem_statement='fix it')
        unused=dict(used=False,reason=serve.GOAL_REWRITER_UNUSED)
        with TemporaryDirectory() as directory:
            root=Path(directory);package,site,model,release=fixture(root)
            install.install(site,model,release,package_root=package,apply=True)
            search=patch.object(sys,'path',[str(ROOT/'source-overlay')]+sys.path);search.start()
            self.addCleanup(search.stop)
            rewriter=root/'qwen3-0.6b';write(rewriter/'config.json',b'{}')
            write(root/'task.json',json.dumps(task).encode())
            # Dry run: harness commands keep --goal-rewriter-model; it is recorded as unused.
            for extra,requested in (([],None),(['--goal-rewriter-model',str(rewriter)],str(rewriter)),
                                    (['--no-goal-rewriter'],None)):
                argv=['serve.py','--release-dir',str(release),'--task',str(root/'task.json'),
                      '--policy','observation_query','--output',str(root/'dry'),'--port','8231',*extra]
                stream=io.StringIO()
                with patch.object(sys,'argv',argv),redirect_stdout(stream):serve.main()
                value=json.loads(stream.getvalue())
                self.assertEqual((value['status'],value['starts_model']),('SERVICE_DRY_RUN',False))
                self.assertEqual(value['goal_rewriter'],dict(unused,requested_model=requested))
            # prepare(): no pin for observation_query even with the flag; goal_query keeps its pin.
            configs={policy:serve.prepare(release,task,policy,root/policy,8231,
                                          dict(goal_rewriter_model=str(rewriter)))[1]
                     for policy in ('observation_query','goal_query')}
            obs,goal=configs['observation_query'],configs['goal_query']
            self.assertIsNone(obs['selection_contract']['goal_rewriter'])
            self.assertEqual(obs['goal_rewriter'],dict(unused,requested_model=str(rewriter)))
            self.assertEqual(goal['selection_contract']['goal_rewriter']['model'],str(rewriter.resolve()))
            self.assertEqual(goal['goal_rewriter'],dict(used=True,reason=None,requested_model=str(rewriter)))
            # The runtime's startup load accepts both as launched and fails closed on an
            # older observation_query config that still pins the rewriter.
            def load(value):
                path=root/'startup.json';path.write_bytes(serve.wire(value))
                with patch.dict(os.environ,REMOTE_RA_HTTP_RUNTIME='1',REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value)):
                    return config_module.load()
            # load() normalizes a one-task config with tasks=[task] (multi-session stage 1).
            self.assertEqual((load(obs),load(goal)),(dict(obs,tasks=[obs['task']]),dict(goal,tasks=[goal['task']])))
            older={k:v for k,v in obs.items() if k!='goal_rewriter'}
            older['selection_contract']=goal['selection_contract']
            with self.assertRaisesRegex(RuntimeError,'observation_query does not use the Goal rewriter'):load(older)
            with self.assertRaisesRegex(RuntimeError,'record differs'):load(dict(goal,goal_rewriter=obs['goal_rewriter']))

    def test_public_docs_no_legacy_interface_or_personal_paths(self):
        for name in ('README.md','README.zh-CN.md'):
            raw=(ROOT/name).read_text(encoding='utf-8')
            for forbidden in ('/u/ycao11','I:\\','Xuanxue','offline_goal_replay','VLLM_POLICY_RUNTIME_MANIFEST'):
                self.assertNotIn(forbidden,raw)

    def test_readme_executable_examples_match_in_both_languages(self):
        def examples(name):
            raw=(ROOT/name).read_text(encoding='utf-8')
            return ['\n'.join(line for line in block.splitlines() if not line.lstrip().startswith('#'))
                    for block in re.findall(r'```(?:sh|json)\n(.*?)```',raw,re.S)]
        self.assertEqual(examples('README.md'),examples('README.zh-CN.md'))

    def test_http_client_wire_has_no_selection_metadata_or_automatic_retry(self):
        spec=importlib.util.spec_from_file_location('minimal_client',ROOT/'examples/http_client.py')
        client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
        calls=[]
        class Opener:
            def open(self,request,timeout):
                calls.append(json.loads(request.data));raise TimeoutError('uncertain admission')
        with patch.object(client.urllib.request,'build_opener',return_value=Opener()):
            with self.assertRaises(TimeoutError):client.completion('http://127.0.0.1:8000/v1',[{'role':'user','content':'task'}])
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['model'],'selective-prefix')
        self.assertFalse({'policy','session','epoch','goal','selected_pages','kv_transfer_params'}&calls[0].keys())

    def test_deterministic_core_archive_and_fresh_isolated_tests(self):
        import package
        raw=package.build(ROOT);self.assertEqual(raw,package.build(ROOT))
        with TemporaryDirectory() as directory:
            target=Path(directory)
            with tarfile.open(fileobj=io.BytesIO(raw)) as stream:
                names=stream.getnames()
                self.assertFalse(any('/experimental/' in n or '/development/' in n or 'qualification_job' in n for n in names))
                self.assertEqual(len([n for n in names if n.endswith('.md')]),2)
                for member in stream:
                    path=target/member.name
                    self.assertTrue(member.isfile() and path.resolve().is_relative_to(target.resolve()))
                    write(path,stream.extractfile(member).read())
            root=target/'selective-prefix'
            script=('import sys,unittest;sys.path.insert(0,sys.argv[1]);'
                'suite=unittest.defaultTestLoader.loadTestsFromNames(['
                '"test_installer.InstallerTests.test_actual_package_coverage_and_default_off",'
                '"test_installer.InstallerTests.test_install_uninstall_quarantine_and_original_environment_unchanged",'
                '"test_http","test_allocation","test_checkpoint","test_startup"]);'
                'result=unittest.TextTestRunner().run(suite);sys.exit(not result.wasSuccessful())')
            run=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(root)],capture_output=True,text=True,timeout=60)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)

if __name__=='__main__':unittest.main()
