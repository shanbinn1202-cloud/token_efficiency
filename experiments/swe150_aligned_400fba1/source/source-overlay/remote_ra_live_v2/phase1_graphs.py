"""Piecewise CUDA graphs for the probe-only passes: ``serve.py --phase1-graph-tokens T`` (option (d)).

Opt-in (0, the default, is off), Qwen profile only, with ``--cuda-graphs piecewise`` or ``full``;
pinned at startup and checked once by the frontend, the engine and the worker; nothing here runs
per request, and the per-step hooks are attribute checks.

Without it a probe-only pass (phase 1, the Goal forward) is dispatched to ``CUDAGraphMode.NONE``
(``cuda_graphs.force_eager``) and vLLM 0.24 captures graphs only up to 2 x ``max_num_seqs``
tokens (1, 2, 4, 8, 16, 24, 32 at 16 sessions), so every step above that runs without a graph.
In the benchmark runs of jobs 3539242, 3461703 (SWE-bench) and 3461661 (Terminal-Bench) the
phase-1 RPC of a continuing turn without a Goal forward took 91-104 ms at the median of every
token bin up to 512 tokens and grew by 51-84 ms per 1,000 tokens (least squares up to 4,096), so
most of a typical pass does not depend on its tokens; how much of that is the graph-less forward
is what the step profile's probe records (remote_ra_step_profile) are for. With it:

- the pinned ``--compilation-config`` gains ``cudagraph_capture_sizes``: vLLM's own sizes for
  ``max_num_seqs`` (``base_sizes``, the rule of ``VllmConfig._set_cudagraph_sizes``), followed by
  the ``LADDER`` sizes above them up to T. vLLM then captures one piecewise graph per size (the
  pieces between the splitting ops; attention, the KV-cache update and the Gated-DeltaNet core
  stay eager and read each step's metadata) and FULL decode graphs only up to ``max_num_seqs``
  rows, as before;
- a probe-only pass may replay a piecewise graph: the runner's dispatch for it is restricted to
  PIECEWISE or NONE (``cuda_graphs.probe_piecewise``), so vLLM pads a pass of n <= T tokens to
  the next captured size and runs it piecewise, and runs a longer one without a graph as before.
  It never replays a FULL graph (a one-token probe would look like a uniform decode step);
- every other step keeps vLLM's dispatch, now with the larger sizes: a mixed step of up to T
  tokens (``--phase1-in-step``'s probe row next to the decodes, or a prefill chunk next to them)
  runs piecewise instead of without a graph. Decode steps are unchanged.

The ladder: two sizes per doubling up to 256 tokens (64, 96, 128, 192, 256) and four per doubling
above, so a pass of more than 256 tokens is padded by less than 25 % of its tokens (padded tokens
are computed: at the 51-84 ms per 1,000 tokens above they are not free).

Exactness: padding changes the GEMM shapes of the compiled pieces, so the phase-1 KV, carry and
vectors may differ in the last bits from the unpadded graph-less pass (as for every graph mode,
docs/reference.md 4.8): compare policies within one setting. The plumbing is exact: the probe's
hooks run outside the captured pieces and read only the probe's own rows and positions (the
padded rows are never read); the lease, the plan, the selections from the same vectors and the
evidence fields are those of the graph-less pass.

--async-scheduling: a probe-only pass runs at a pass boundary, after the in-flight batch is
drained (async_sched), so it replays its piecewise graph like any synchronous step; a mixed step
that now runs piecewise is voided on the metadata the eager attention, KV-update and
Gated-DeltaNet ops read, as the piecewise mixed steps of up to 2 x ``max_num_seqs`` tokens are
without the flag.

The Gemma 4 profile (window layout) refuses the flag in v1: it is neither built nor tested there
(its phase-1 passes took 101 s of the 2,710-s run of job 3462014, 3.7 %).

Pins: ``startup.json`` ``phase1_graphs`` = ``{"capture_sizes": [...], "tokens": T}`` and the
effective ``compilation_config.cudagraph_capture_sizes``; the frontend compares vLLM's parsed
sizes, the engine sends the record in the worker startup record and compares the echo, the
worker requires vLLM's resolved capture sizes to be exactly the pinned ones, records
``probe_forward: "piecewise"`` and ``phase1_graph_tokens`` in its ``cuda_graphs`` record and
echoes ``phase1_graphs``; ``ready.json`` records it. Off: no argument, key, record or evidence.
"""
from __future__ import annotations

import os

from .contract import require

CONFIG_KEY = 'phase1_graphs'        # startup config, worker startup record, worker evidence, ready.json
SWITCH = '--phase1-graph-tokens'    # serve.py's option (0: off)
TOKENS = (64, 128, 256, 512, 1024, 2048)        # T: the largest captured size
LADDER = (64, 96, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048)
GRAPH_MODES = ('piecewise', 'full')  # the --cuda-graphs modes that capture piecewise graphs
PROBE_FORWARD = 'piecewise'          # the cuda_graphs record's probe_forward with the flag on
VLLM_MAX_DEFAULT = 512               # vLLM 0.24's cap of its default largest size


def base_sizes(max_num_seqs, max_num_batched_tokens):
    """vLLM 0.24's default capture sizes (VllmConfig._set_cudagraph_sizes, no speculative tokens,
    the default performance mode): 1, 2, 4, multiples of 8 below 256, multiples of 16 from 256, up
    to min(2 x max_num_seqs, 512, max_num_batched_tokens); max_num_batched_tokens itself when it
    fits."""
    require(type(max_num_seqs) is int and max_num_seqs > 0 and type(max_num_batched_tokens) is int and
            max_num_batched_tokens > 0, 'capture sizes need max_num_seqs and max_num_batched_tokens')
    largest = min(max_num_seqs * 2, VLLM_MAX_DEFAULT, max_num_batched_tokens)
    sizes = [s for s in (1, 2, 4) if s <= largest]
    if largest >= 8:
        sizes += list(range(8, min(largest + 1, 256), 8))
    if largest >= 256:
        sizes += list(range(256, largest + 1, 16))
    if max_num_batched_tokens <= largest:
        sizes.append(max_num_batched_tokens)
    return sorted(set(sizes))


def capture_sizes(max_num_seqs, tokens, max_num_batched_tokens):
    """The pinned capture sizes: vLLM's base sizes, then the LADDER sizes above them up to T."""
    require(tokens in TOKENS, '%s must be one of %s, got %r' % (SWITCH, ', '.join(map(str, TOKENS)), tokens))
    require(tokens <= max_num_batched_tokens, '%s %d exceeds the step budget' % (SWITCH, tokens))
    base = base_sizes(max_num_seqs, max_num_batched_tokens)
    return base + [s for s in LADDER if base[-1] < s <= tokens]


def record(tokens, max_num_seqs, max_num_batched_tokens):
    """The startup config's record of T for a server of ``max_num_seqs`` running sequences."""
    return dict(capture_sizes=capture_sizes(max_num_seqs, tokens, max_num_batched_tokens), tokens=tokens)


def pinned(config):
    """The pinned record ({'capture_sizes': [...], 'tokens': T}), or None (off: no key). Startup only.

    The record must be exactly the one of the config's T, running sequences and step budget, next
    to --cuda-graphs piecewise or full and the effective compilation config that pins its sizes
    (cuda_graphs.pinned checks the whole compilation config). Without the key no capture sizes may
    be pinned. The Gemma 4 profile refuses the key (v1)."""
    if config is None:
        return None
    effective = config.get('effective_server_config')
    compilation = effective.get('compilation_config') if isinstance(effective, dict) else None
    if CONFIG_KEY not in config:
        require(not (isinstance(compilation, dict) and 'cudagraph_capture_sizes' in compilation),
                'pinned CUDA-graph capture sizes need %s' % SWITCH)
        return None
    require('model_profile' not in config, '%s is not available for the Gemma 4 profile (v1)' % SWITCH)
    require(config.get('cuda_graphs') in GRAPH_MODES,
            '%s needs --cuda-graphs piecewise or full (the probe-only passes replay piecewise graphs)' % SWITCH)
    value = config[CONFIG_KEY]
    require(type(value) is dict and set(value) == {'capture_sizes', 'tokens'} and value['tokens'] in TOKENS,
            'phase1_graphs must be {capture_sizes, tokens} with tokens one of %s' % ', '.join(map(str, TOKENS)))
    require(isinstance(effective, dict) and type(effective.get('max_num_seqs')) is int and
            type(effective.get('max_num_batched_tokens')) is int, 'phase1_graphs needs the pinned server config')
    expected = record(value['tokens'], effective['max_num_seqs'], effective['max_num_batched_tokens'])
    require(value == expected, 'phase1_graphs capture sizes differ from the pinned T and running sequences')
    require(isinstance(compilation, dict) and compilation.get('cudagraph_capture_sizes') == value['capture_sizes'],
            'phase1_graphs capture sizes differ from the pinned vLLM compilation config')
    return dict(value, capture_sizes=list(value['capture_sizes']))


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None. Startup only."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def worker_record(startup):
    """Worker startup: the record the engine sent (or None), checked once against the pinned config.

    With the HTTP runtime the worker reads the startup config itself and refuses a record that
    differs; without it (CPU fixtures) only the record's shape is checked."""
    sent = startup.get(CONFIG_KEY)
    require(sent is None or (type(sent) is dict and set(sent) == {'capture_sizes', 'tokens'} and
                             sent['tokens'] in TOKENS and type(sent['capture_sizes']) is list and
                             sent['capture_sizes'] and sent['capture_sizes'][-1] == sent['tokens']),
            'phase1_graphs worker record must be {capture_sizes, tokens} or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker phase1_graphs differs from the pinned startup config')
    return None if sent is None else dict(sent, capture_sizes=list(sent['capture_sizes']))
