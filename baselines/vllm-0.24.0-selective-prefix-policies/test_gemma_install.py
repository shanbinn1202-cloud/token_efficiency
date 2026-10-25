"""Gemma 4 stage 1 (gemma-s1 WP1): install.py --model-profile, CPU only.

A release installed with ``--model-profile gemma4-31b-it`` pins Gemma's five metadata
files and its revision (``INSTALL_RELEASE.json`` ``model_profiles``, equal to
``remote_ra_live_v2.model_profile``) and names the profile in its manifest; a Qwen
install is unchanged (no ``model_profile`` key).
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))

import install  # noqa: E402
import test_installer as INSTALL  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
GEMMA_FILES = {'chat_template.jinja': b'{{ bos_token }}fixture template\n', 'config.json': b'{"fixture": 1}',
               'generation_config.json': b'{"eos_token_id": [1, 106, 50]}', 'tokenizer.json': b'{}',
               'tokenizer_config.json': b'{"fixture": true}'}


def gemma_fixture(root):
    """test_installer's fixture plus a Gemma model directory and its model profile."""
    package, site, model, release = INSTALL.fixture(root)
    gemma = root / 'gemma'
    for name, raw in GEMMA_FILES.items():
        INSTALL.write(gemma / name, raw)
    path = package / 'INSTALL_RELEASE.json'
    definition = json.loads(path.read_bytes())
    definition['model_profiles'] = {MP.GEMMA4: dict(
        model_pins={n: install.digest(r) for n, r in GEMMA_FILES.items()}, model_revision='b' * 40)}
    path.write_bytes(install.encoded(definition))
    return package, site, model, gemma, release


class InstallerProfileTests(unittest.TestCase):
    def test_release_pins_equal_the_profile_module(self):
        definition = install.verify_package(ROOT)
        profile = definition['model_profiles'][MP.GEMMA4]
        self.assertEqual(profile['model_pins'], MP.GEMMA4_MODEL_PINS)
        self.assertEqual(profile['model_revision'], MP.GEMMA4_REVISION)
        self.assertEqual(set(definition['model_profiles']), {MP.GEMMA4})

    def test_gemma_install_names_the_profile_and_qwen_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, site, model, gemma, release = gemma_fixture(root)
            with redirect_stdout(io.StringIO()):
                result = install.install(site, gemma, release, package_root=package, apply=True,
                                         model_profile=MP.GEMMA4)
            self.assertEqual((result['status'], result['model_profile']), ('INSTALL_PASS', MP.GEMMA4))
            manifest = install.installed_check(release)
            self.assertEqual(manifest['model_profile'], MP.GEMMA4)
            self.assertEqual(manifest['model_revision'], 'b' * 40)
            self.assertEqual(manifest['model_pins'], {n: install.digest(r) for n, r in GEMMA_FILES.items()})
            self.assertEqual(MP.from_manifest(manifest), MP.GEMMA4)
            qwen = root / 'qwen-release'
            with redirect_stdout(io.StringIO()):
                result = install.install(site, model, qwen, package_root=package, apply=True)
            self.assertNotIn('model_profile', result)
            manifest = install.installed_check(qwen)
            self.assertNotIn('model_profile', manifest)
            self.assertEqual(manifest['model_revision'], 'a' * 40)
            self.assertEqual(MP.from_manifest(manifest), MP.QWEN)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, site, model, gemma, release = gemma_fixture(root)
            with self.assertRaisesRegex(ValueError, 'unknown model profile'):
                install.install(site, gemma, release, package_root=package, model_profile='llama')
            # Gemma's files do not pass for the Qwen pins, nor Qwen's for Gemma's.
            with self.assertRaisesRegex(ValueError, 'pinned model/tokenizer metadata mismatch'):
                install.install(site, gemma, release, package_root=package)
            with self.assertRaisesRegex(ValueError, 'pinned model/tokenizer metadata mismatch'):
                install.install(site, model, release, package_root=package, model_profile=MP.GEMMA4)
            path = package / 'INSTALL_RELEASE.json'
            definition = json.loads(path.read_bytes())
            definition['model_profiles'][MP.GEMMA4]['model_revision'] = 'main'
            path.write_bytes(install.encoded(definition))
            with self.assertRaisesRegex(ValueError, 'bad model profile pins'):
                install.install(site, gemma, release, package_root=package, model_profile=MP.GEMMA4)


if __name__ == '__main__':
    unittest.main()
