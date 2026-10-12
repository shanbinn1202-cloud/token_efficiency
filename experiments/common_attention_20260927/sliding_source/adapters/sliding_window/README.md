# Sliding Window on the common Vanilla FA3/GQA path

This is the user-requested common-attention variant. The whole initial system
and task prefix stays protected across turns. The remainder of a compaction
budget B contains the most recent surviving KV rows, in chronological order.
All heads select the same positions. K/V values and logical RoPE positions are
preserved; no input text is truncated or re-encoded to implement eviction.

The retained rows are compacted into ordinary native FullAttentionSpec pages.
Model attention uses the same FlashAttention 3 grouped-query path as Vanilla.
There is no R-SWA mask, virtual single-head sequence, custom attention kernel,
prefix/suffix attention split, or merge. Its sequence length is the compacted
physical length, so evicted historical gaps are not traversed by attention.

## Deliberate protocol change

The former FA4 baseline enforced a visible-token bound for every query. To use
the unchanged Vanilla attention path, this variant performs cache selection
outside attention: compact after the final prefill chunk when length exceeds
B, and during one-token decode when physical length exceeds B and is divisible
by 64. B includes the fixed prefix. Tool prefills may temporarily exceed B;
decode between cuts ordinarily uses at most B+64 physical tokens. This is a
periodic compaction target, not the old per-query mask or a strict memory cap.
Initial-prefix length must be below B.

The shared persistent-cache controller is in vllm/tebench/random_attention.py;
VLLM_COMPACT_WINDOW selects this deterministic policy. The Random policy and
its original selector remain unchanged. The controller stores surviving KV
and recurrent state losslessly on CPU between turns. Discarded KV is not
archived. Native RECOMPUTE replays recorded selection boundaries and keep sets.

Async FCFS, eight request slots, native allocation/preemption, a 5 GiB KV pool,
model/generation settings, compilation and CUDA Graphs match the shared runs.
New results must be labeled as this common-GQA variant rather than replacing
or claiming equivalence to the previous strict FA4 sliding-window measurements.
