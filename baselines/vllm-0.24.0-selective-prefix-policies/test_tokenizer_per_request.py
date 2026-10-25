"""Per-request encoding of the XML route: the loaded HF tokenizer only, as stock vLLM.

Job 3354504 (2026-10-11) failed an agent of django__django-10097: its observation held a
Devanagari e-mail domain from Django's validator tests, which Transformers 5.13 and the
configured tokenizer.json reference split differently, and the per-request comparison of
the two refused the request with HTTP 500. The comparison is gone; the startup corpus
check stays.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
from runtime_renderer import EngineSourceRenderer  # noqa: E402

EVIDENCE = ROOT / 'source-overlay' / 'remote_ra_linux' / 'model-evidence'
DJANGO_LINE = "    (validate_email, 'test@domain.with.idn.tld.उदाहरण." \
              "परीक्षा', None),\n"


class _Loaded:
    def __init__(self, ids):
        self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return list(self.ids)


class _Reference:
    """A reference that disagrees with the loaded tokenizer: it must not be consulted."""
    def encode(self, text, add_special_tokens=False):
        raise AssertionError('the per-request path consulted the reference tokenizer')


def _renderer(loaded, reference):
    renderer = EngineSourceRenderer.__new__(EngineSourceRenderer)
    renderer.tokenizer = loaded
    renderer.producer = type('Producer', (), {'_tokenizer': reference})()
    return renderer


class PerRequestEncodingTests(unittest.TestCase):
    def test_returns_the_loaded_encoding_without_the_reference(self):
        renderer = _renderer(_Loaded([497, 13, 154855, 148651]), _Reference())
        self.assertEqual(renderer.encode_exact(DJANGO_LINE), [497, 13, 154855, 148651])

    def test_still_refuses_invalid_loaded_output(self):
        for bad in ([-1], [1.5], ['a']):
            with self.assertRaisesRegex(Exception, 'invalid loaded tokenizer output'):
                _renderer(_Loaded(bad), _Reference()).encode_exact('x')

    def test_still_refuses_text_over_the_source_limit(self):
        from runtime_renderer import MAX_SOURCE_BYTES
        with self.assertRaisesRegex(Exception, 'bounded protocol text required'):
            _renderer(_Loaded([1]), _Reference()).encode_exact('a' * (MAX_SOURCE_BYTES + 1))

    @unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'),
                         'transformers and tokenizers not installed')
    def test_pinned_tokenizers_disagree_on_the_django_line_and_the_loaded_one_wins(self):
        from tokenizers import AddedToken, Tokenizer
        from transformers import AutoTokenizer
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            shutil.copy(EVIDENCE / 'tokenizer.json', work / 'tokenizer.json')
            shutil.copy(EVIDENCE / 'tokenizer-config.json', work / 'tokenizer_config.json')
            loaded = AutoTokenizer.from_pretrained(str(work), local_files_only=True)
            config = json.loads((work / 'tokenizer_config.json').read_text(encoding='utf-8'))
            reference = Tokenizer.from_file(str(work / 'tokenizer.json'))
            for _, row in sorted((int(k), v) for k, v in (config.get('added_tokens_decoder') or {}).items()):
                token = AddedToken(row['content'], **{k: row[k] for k in
                                                      ('single_word', 'lstrip', 'rstrip', 'normalized', 'special')})
                (reference.add_special_tokens if row['special'] else reference.add_tokens)([token])
            wanted = loaded.encode(DJANGO_LINE, add_special_tokens=False)
            if reference.encode(DJANGO_LINE, add_special_tokens=False).ids == wanted:
                self.skipTest('this Transformers/tokenizers pair agrees on the line')
            renderer = _renderer(loaded, reference)
            self.assertEqual(renderer.encode_exact(DJANGO_LINE), wanted)


if __name__ == '__main__':
    unittest.main()
