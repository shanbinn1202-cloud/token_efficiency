# IntentKV with the common Vanilla FA3/GQA attention path

This is the user-requested **GQA shared-selection variant**, not the previous
algorithm-equivalent per-query-head implementation. The original repository
at `baselines/IntentKV` remains unmodified, commit
`065f394dbb14f317d2b4dba252e8e66b91afa8c7`.

## Selection

The adapter executes the original `IntentKVCluster.update_kv` and preserves
its last-64-query BF16 scores, FP32 softmax/dtype cast, cumulative attention,
JSD, second differences, per-head query starting point, and block sums.
Immediately before the final block top-k, it takes the arithmetic mean of
those block scores over the query heads sharing each KV head. It selects a
single chronological block list for that GQA group. The recent query window
and incomplete final block remain protected exactly as upstream.

For Qwen3.6-27B this means 24 query heads grouped into four sets of six, with
one retained history per KV head. This deliberately changes the original
independent query-head histories. Averaging the original final block scores
is the only scoring-policy change; the original checkout is not patched.

## Attention and budget

Selected original K/V rows are compacted into ordinary native pages.
The backend goes through the same FA3 grouped-query attention call as Vanilla,
with 24 query heads and four KV heads. It does not use virtual one-head
sequences, per-query-head page maps, GQA union storage, custom masks or a
separate attention implementation. The KV layout uses the native default.

The initial prompt is processed in full, followed by one selection at its
boundary. New decode KV is appended. Every subsequent agent turn submits the
full rendered history and creates a fresh cache, as in the original IntentKV
generate protocol. Capacity is per KV head after prompt selection; appending
decode can exceed it. It is not a hard per-token GPU-memory cap.

Native async FCFS, eight request slots, the 5 GiB native KV pool, model and
generation configuration, compilation, and CUDA Graphs are retained. Native
RECOMPUTE saves/replays integer selection decisions, recomputing K/V and
recurrent states. Already-selected histories are not rescored during replay.

## Validation and reporting

Unit checks compare grouped selection with independently captured original
block scores and direct mean/top-k, including the real 24:4 head configuration.
Native page and replay tests require exact selected K/V equality. Real-model
checks cover long prompts, eight clients, multi-turn fresh requests and
preemption recovery. New formal results are separate from the prior
per-query-head IntentKV experiments and must be labeled as the GQA variant.
