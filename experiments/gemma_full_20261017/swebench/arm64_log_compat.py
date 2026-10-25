"""Parse two genuinely passing ARM-marked tests missed by the upstream parser.

This adapter changes neither pytest execution nor dataset grading labels. Raw
logs are retained. A strict XPASS is emitted as FAILED and is never overridden.
"""
import re

NODE = r'xarray/tests/test_duck_array_ops\.py::test_datetime_mean\[(?:False|True)\]'
XPASS = re.compile(r'^XPASS (?P<node>' + NODE + r') expected failure on ARM\s*$', re.MULTILINE)

def install():
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    original = MAP_REPO_TO_PARSER['pydata/xarray']
    if getattr(original, '_swe500_arm64_xpass', False):
        return

    def parse(log, spec):
        statuses = original(log, spec)
        if spec.instance_id == 'pydata__xarray-4356':
            for match in XPASS.finditer(log):
                statuses.setdefault(match.group('node'), 'PASSED')
        return statuses

    parse._swe500_arm64_xpass = True
    MAP_REPO_TO_PARSER['pydata/xarray'] = parse
