# Runtime support for the SWE-150 clients

The SWE-150 clients import the frozen `Model`, `Progress`, AgentKV settings and official compaction adapter in this directory. The snapshot retains the three local changes for AgentKV tool-observation spans, XML generation and writable Apptainer testbeds. The remaining support sources are restored from the surrounding token_efficiency repository and the pinned selective-prefix checkout. Every installed source file is checked against the hashes recorded for the evaluated implementation.

| Component | Evaluated version |
| --- | --- |
| Selective-prefix repository | `400fba1ca414904d8a48222de4315ff678e18446` |
| Shared repository sources | token_efficiency at `068efecbb47cfb5dec8f633750c0e3f5c7cac0e5`, or a checkout with matching file hashes |
| Mini-SWE-Agent | Official Python wheel, 2.4.2 |
| SWE-bench harness | `fa79f3af3e0f212d4d14b1c858c77fcaae5308ce` |
| Claude Code | Official `@anthropic-ai/claude-code-linux-arm64@2.1.325`, verified npm tarball and binary |

Run with Python 3.12 in the benchmark environment. The final flag below points to the pinned selective-prefix source tree.

```bash
python restore_support.py --repository /path/to/token_efficiency --upstream /path/to/vllm-0.24.0-selective-prefix-policies
```

The installer fetches the Mini-SWE-Agent wheel and the native Claude executable on demand. For an offline install, pass the local files with `--mini-wheel` and `--claude-archive`. Use `--destination` to build the import tree outside this checkout. Python and native dependencies, model weights and Apptainer images are provisioned by the benchmark's runtime environment.

The server resolves `adapters.agentkv.vllm_policy` and `adapters.agentkv.upstream` through the restored `archive-reference` directory on its Python module search path. The latter imports the original MiniSGL StageQ/SnapKV selection, query ring, scheduler phases and tokenizer from `archive-reference/baselines/agentkv/engine/python/minisgl`. The restored distribution includes the AgentKV Apache-2.0 license, MiniSGL engine MIT license, SWE-bench and Mini-SWE-Agent licenses, selective-prefix license and the native Claude package's license notice.
