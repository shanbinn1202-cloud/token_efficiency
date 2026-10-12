# Random Attention on native asynchronous vLLM

Original repository: `baselines/Random-Attention`, commit
`64db9688a12f0926db45fec039370f6ceb1ab4fe`. That checkout is read-only for this
experiment. `selector_random.py` is copied byte for byte, including its license,
from `scripts/vllm_rp_bench/selector_random.py`.

## Algorithm and the agreed multi-turn protocol

The user selected `random_pp` with a persistent cache across turns. Protect the
entire initial system prompt and task (including template/tool definitions),
then retain that prefix plus the newest 64 tokens and uniformly sampled older
survivors. Sampling is independent for every layer and KV head. Selection is
sorted before copying K and V. It never ranks query attention, merges heads,
rounds selection to pages, or resurrects discarded KV.

Use the original HF `EvictLayer.post_attention_update` trigger: after one decode
token, evict if physical length exceeds B and is divisible by 64. Compress to B
total tokens, **including** the protected prefix and recent window. Do not evict
during a tool-observation prefill. Thus B is a periodic compaction target; a
prefill and the following decode interval can temporarily exceed B. The prefix
must fit below B-64; all three experiments validate this. The HF implementation
would cap an oversized protected prompt, whereas the user requested protecting
the whole initial task; this incompatible edge case is rejected explicitly.

The released vLLM selector has the same uniform selection rule on this valid
prefix domain. Its default window of 128 is changed through configuration to
the main HF algorithm's 64; its TriAttention divide-length runtime, budget
reserve, and concurrency caps are not used. The original vLLM RNG scheme is
unchanged: seed 1234 hashed with session identity, logical event position and
layer. Its random-number draw layout differs from HF; identical raw seeds do
not imply identical HF and vLLM samples. Tests provide identical random scores
to the unmodified HF method and verify exact selected K/V, on CPU and GPU.

Extending the lifetime of a single-generation algorithm across tool turns is
the explicit user-approved protocol, not a claim that the paper evaluated
this SWE-bench multi-turn setup. New observations append to the retained KV;
they do not become a newly protected prompt. RoPE uses original logical
positions. Qwen's GDN/recurrent state remains intact, as in other methods.

## Engine behavior

Official vLLM 0.28 base, native AsyncScheduler with FCFS, eight request slots,
native RECOMPUTE preemption. No trajectory admission limits or full-generation
reservations. Commit fences only serialize actual KV mutations against the
next asynchronous batch.

Between tool turns, the retained KV and recurrent state have lossless CPU
checkpoints loaded through native KVConnector hooks. Evicted KV is not saved.
Active preemption uses native recomputation, not a CPU KV swap: keep-index
records tell recomputation where and how to repeat historical evictions.
Replaying never resamples. Native full-attention pages are released only after
compaction; recurrent pages follow the native allocator.

Validation covers repeated original-HF equivalence for every budget, GPU page
copying, shared scheduler behavior, early EOS, non-EOS turn continuation,
lossless turn restores, and forced native capacity preemption with replay.

## Experiments

Bundle: `/u/ssun4/token_efficiency_results/random_attention_swe20_20260927`.
Same 20 SWE-bench Verified tasks, model snapshot, BF16, 64K logical context,
4096 output limit, 120 turns, eight workers, 5 GiB native KV pool, compilation
and CUDA graphs as the existing async baselines. B = 16384 / 12288 / 8192.
Selection/checkpoint/recompute costs are inside batch wall time. Throughput is
main-agent output tokens divided by batch wall time, excluding startup,
initial compilation/warmup and grading. Genuine context/turn-limit failures
remain in the denominator; infrastructure failures require review.
