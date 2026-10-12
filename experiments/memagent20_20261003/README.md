# MemAgent_Bench: fresh 20-question experiment

The user superseded the 40-question experiment and requested that every
method and budget restart with 20 questions. No predictions, case results
or measured inference times are reused from the cancelled experiment.

Results: `/u/ssun4/token_efficiency_results/memagent20_20261003`.

## Fixed scope

- Six existing methods, at 16K, 12K and 8K: 18 runs, 360 trajectories.
  RepairKV was excluded at the user's request; its cancelled validation
  records are preserved under `audit/repairkv-excluded-by-user`.
- Five independent QAs per skill. Retrieval: QA1 × 3 and QA2 × 2.
  Learning: one QA from each of the five classification datasets.
  Conflict resolution: SH-32K × 2, SH-64K × 1, MH-32K × 1, MH-64K × 1.
  Understanding: five book summaries.
- Selection takes the first N entries from each source in the previously
  frozen, seed-42 sampling order. Preparation reads the dataset manifest and
  histories only, never any predictions or scores.
- Full histories, lossless approximately 4,096-token chunks, unseen question
  until history ingestion finishes, and fresh independent sessions per QA.
- Native 262,144 context; 18 GiB shared KV pool; BF16; eight asynchronous
  requests; native FCFS; 4,096 output tokens; temperature 0; seed 0.
  Existing method algorithms and options remain unchanged.

The ingestion and official scoring protocol is described in the
[original harness documentation](../memagent40_20261003/README.md). All
context, reference-answer and official summary-keypoint matches were
verified again for this 20-question manifest. Three protocol tests passed.
The new dataset manifest and inference protocol are frozen in
`protocol_sources.json`; existing method files remain frozen separately.

`supervise.py` completes all methods at one budget before starting the next
budget. It uses the requested two-hour allocation command and does not
reallocate a GPU before one hour. Model startup, compilation and warmup are
excluded from throughput timing. Completed independent cases can span
allocation segments; all segment wall times count toward throughput.

GPT-4o summary scoring is deferred until the user configures the endpoint.
Copy `judge.example.env` to `judge.env` in the result directory and set
`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and the available
`MEMAGENT_JUDGE_MODEL` deployment. Keys are never logged. Run `judge.py` with
`--bundle` pointing to this result directory to apply the original judge
prompts and scoring formula.
