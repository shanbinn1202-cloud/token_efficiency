# GPU-experiment harnesses (snapshot)

English | [简体中文](README.zh-CN.md)

This directory holds a filtered snapshot of the harnesses that run our GPU experiments: the four harnesses of the
release smoke matrix. It is for the repository's members only: it is not part of the runtime, the tests or a
release (the release export drops `harnesses/**`).

## What a harness is

`examples/` holds the public benchmark clients (SWE-bench, Terminal-Bench, MemoryAgentBench) that anyone runs
against `serve.py`. A harness is our experiment machinery around one runtime commit. It:

1. packs a pinned runtime commit (a sealed Git bundle, cloned fresh on the cluster) with its release receipt (the
   `check_git_release.py` result and its log hash) into a sealed payload (`build.py` -> `dist/`);
2. defines plans: the servers (policy, `serve.py` flags, KV size), the time limit and the GPU-hours (the
   `*-config.json`, copied into `pins.json`);
3. submits one Slurm job on DeltaAI (ghx4 batch, one GH200) over one SSH connection (`run_one_connection.ps1`,
   `controller.py`);
4. in the job (`job.sh`, `job.py`) installs the release in a clean environment and starts the servers one after
   another;
5. drives the benchmark client: on the agent host for SWE-bench and Terminal-Bench (`tbench/`, deployed there by
   `tbench/deploy_tbench.ps1`, reached through an SSH tunnel), with the harness's own runner inside the job for
   MemoryAgentBench;
6. checks the evidence in the job (`smoke_checks.py`, `gate_checks.py`, `*_checks.py`);
7. fetches and verifies it locally (`postrun/fetch_*.ps1`, `postrun/verify_evidence.py`);
8. builds the tables (`postrun/*_tables.py`, `postrun/join_*.py`).

Each harness's own `README.md` holds its decisions and `STATUS.md` its jobs and results.

## The four harnesses

The snapshot of 2026-11-17 holds each harness's live state (also each harness's `note` in `SNAPSHOT.json`).
`perf-swe32` (re-synced on the evening of 2026-11-17) pins integrate-1117 `2fb7f62`, `main`'s runtime (release receipt `GIT_ARCHIVE_TESTS_PASS`, 2,287 tests, 141 skipped); the
other three still pin integrate-1112 `4ebff97` (2,059 tests, 139 skipped) and are not re-pinned to integrate-1116 yet
(`perf-gemma-swe32` also needs its session cap raised first: docs/known-limits.md 5.5). This snapshot is on
integrate-1116; `main` keeps the snapshot of 2026-11-13 until integrate-1116 is merged. The verdicts below are verifier
only (not audited); [docs/status.md](../docs/status.md) has every job.

| Harness | Pin | Payload (SHA-256 of `dist/payload.json`) | Local validation | GPU jobs from this payload |
| --- | --- | --- | --- | --- |
| `perf-swe32` | `2fb7f62` | `f62064f1...` (adds gate 2, plan `parity150`; the payload of the queued 150-instance jobs, `269a00e6...`, is frozen as `perf-swe32-269a00e6` outside the repository) | `validation\20261117T190547` | sess24ri r2 3616345 (`SWE32_FAIL_VERIFIED` from a defect of the idle-release check, fixed since); queued: full150k24 3616690, stockkeep150k24 3616705, ledger150k24 3616729 |
| `perf-gemma-swe32` | `4ebff97` | `aaa272a2...` | `validation\20261114T141219` | none (smokea 3586987, from the frozen `e5c93e1e` payload, was cancelled on 2026-11-16 before it ran) |
| `perf-tb16-strip` | `4ebff97` | `4203f013...` | `validation\20261115T015540` | none (smokeqt r2 3586720, from the frozen `98a149cd` payload, was cancelled on 2026-11-16 before it ran) |
| `perf-mab` | `4ebff97` | `bd816111...` | `validation\20261115T201624` | smqc r1 3604146 (`PMAB_PASS_VERIFIED`); smqt 3584105 and smga 3601174 (both `PMAB_PASS_VERIFIED`) ran from the frozen `89d65f99` payload |

Earlier jobs ran payloads that are now frozen copies (not in the snapshot), for example stock32ti 3583822 on the
first 4ebff97 payload of `perf-swe32` (`c7c3e8dc...`).

| Harness | What it runs | Main plans |
| --- | --- | --- |
| `perf-swe32` | SWE-bench Verified with Mini-SWE-Agent 2.4.2 on Qwen3.6-27B: the first 32 instances (the `*64*` plans: the first 64), 16 sessions | release smoke `smoke` (J1) and `smoket` (thinking on); Table 1 `full`, `obs`, `rfill`, `goalb4`, `stock` (`*i`: ghx4-interactive variants); `recentt`, `rtoolst`, `akvt`, `akvs`; thinking on `goalti`, `akvsi`, `rtoolsti`, `fullt(i)`, `fulltr(i)` (idle release); the 64-instance proxy `obs64(i)`, `goal64`, `full64`, `recent64`, `akv64`, `rfill64(i)`; the stock control with thinking on `stock32t(i)`, `stock64t`; `ctx24`, `ctx25`, `vllm25`, `stock25(x)`; phase B (2026-11-16, integrate-1116): the scripted edge-case plans `keepedge(i)`, `keepedgesync(i)`, `keepedgegoalp(i)`, the `--gdn-inplace-crossing` pairs `gdnobsp(i)`, `gdngoalp(i)`, `gdnidlep(i)`, `gdnobs24p(i)`, the 24-session test `sess24(i)`, and the 64-task plans with `--async-scheduling` `full64a`, `recent64a`, `obs64a`, `goal64a`, `ledger64a`, `stockkeep64t` (the stock control keeping the reasoning) and their K = 24 twins (`...k24`, `stockkeep64k24`); the 24-session smoke with idle release `sess24r(i)`; the first 150 instances at 24 sessions `full150k24`, `stockkeep150k24`, `ledger150k24` (`tasks/tasks-pool-150.json`; the side-by-side tables `postrun/run150_tables.py`); release gate 2 (full against stock, token for token) `parity150` |
| `perf-gemma-swe32` | the same SWE-bench run on Gemma 4 31B IT | release smoke `smoke`, `smokehf`, `smokea` (async scheduling) (J2); `full`, `obs`, `obs4k`, `rfill`, `goal`, `stock` |
| `perf-tb16-strip` | Terminal-Bench 2.0 with Harbor 0.21.0's Terminus 2, Qwen3.6-27B and Gemma 4 31B IT | release smoke `smokeq`, `smokeqs`, `smokeqt` (thinking on with the history strip) and `smokeg`, `smokegh`; `gctx25` (Gemma at 262,144 tokens); the 89-task `full`, `obs`, `rfill`, `goal`, `stock` and their Gemma `g*` plans; dry-subset plans |
| `perf-mab` | the MemoryAgentBench kv-session protocol, Qwen3.6-27B and Gemma 4 31B IT (no agent host) | release smoke `smq`, `smqt` (J5) and `smg`, `smga` (J6); GPU-PLAN `g1`, `g2`, `g4o`, `g4g`, `g4v`, `gm1`, `gm4*` |

## A snapshot, not the working copy

The live harnesses are the folders `I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\<name>\` on the
maintainer's machine. They stay the canonical working copies: their scripts, READY files and agents use those absolute
paths. This directory is a filtered copy that `sync_from_live.py` refreshes; do not edit it here (the next refresh
overwrites the change and the live harness never sees it). `SNAPSHOT.json` records, for each harness, when it was
taken, its files, bytes, listing hash, pinned commit, what was left out and a note on its state; a harness missing
there is not in the snapshot yet.

To refresh (take a snapshot only when no edit of that harness is in progress):

```
python -I -B harnesses/sync_from_live.py --dry-run [NAME ...]
python -I -B harnesses/sync_from_live.py --note "<state>" [NAME ...]
python -I -B harnesses/scan_snapshot.py
```

then review the scan and commit. The sync needs a local edit list kept outside the repository (`EDIT_RULES` in
`sync_from_live.py`); the files it changes are listed under `edited` in `SNAPSHOT.json`. Every other file is copied
byte for byte. The sync removes files the live folder no longer has and refuses a file over 2 MB that is not listed
in `ALLOW_LARGE`. Left out:

| Left out | Why |
| --- | --- |
| `validation/` | local validation logs (run outputs) |
| `dist/`, `*.bundle` | the sealed payload and the runtime bundle (`build.py` rebuilds them from the pinned commit) |
| `__pycache__/`, `*.pyc` | bytecode |
| `*.log` outside `fixtures/` | run logs (the log excerpts under `fixtures/` stay: the tests read them) |
| `*-PROGRESS.md` | agents' resumable working notes; `STATUS.md` and the READY files hold the outcome |
| `*.bak`, `*.orig`, `*.tmp`, `*~`, ... | scratch copies |
| `HELD_BACK` | single files kept out after the privacy scan: `perf-gemma-swe32/fixtures/gemma-render-job3429830.json` (model replies and token ids of a job, against the counts-only rule for fixtures), so `test_real_tokenizer.GemmaJoinRender` runs only in the live copy |

`harnesses/.gitattributes` keeps the written bytes (no end-of-line conversion: the pins hash these files) and
`harnesses/.gitignore` keeps the `fixtures/` logs that the root `.gitignore` would hide. `scan_snapshot.py` looks for
secrets and blocked terms (exit 1 on any), long text in fixtures, and account and host data. The snapshot contains
the Slurm mail address, the cluster account and user names, the agent host's name and SSH host key: kept because the
repository is private.

## Conventions

- **Freeze before re-pin.** Before a harness is re-pinned to another runtime commit or edited for new jobs, its
  current state is frozen as a sibling folder `<name>-<payload hash>` (or `-job<id>`) with a SHA-256 list and a
  `FROZEN-<hash>.md`; jobs already submitted are fetched and verified with the frozen copy, whose payload they ran.
  Frozen copies are not in the snapshot.
- **Local validation with fakes.** `local_validation.ps1` builds twice (the payloads must be identical), runs the
  harness's tests, the release-parity and re-pin checks, and SelfTest and DryRun of every plan, end to end against
  fakes of DeltaAI and the agent host (`fake_ssh.py`, `fake_stage.py`, `fake_server.py`, `fake_tbench.py`), with no
  connection. Its logs go to `validation/`.
- **READY files.** `<WHAT>_READY.txt` (`SMOKE_READY.txt`, ...) hands a validated plan to the user: the runtime commit
  and receipt, the payload and entry-script hashes, the local validation result, the frozen predecessors, and the
  exact submit, fetch and verify commands. The user submits (DeltaAI needs Duo); no READY file, no submission.

## Not runnable elsewhere as-is

The harnesses assume the maintainer's Windows machine (Windows PowerShell 5.1, the OneDrive and `D:\Remote-RA-temp`
paths, its Python), DeltaAI (the account, user, ghx4 partition and model paths) and the agent host (its host key,
tunnel key and deployed environments). They are not part of this repository's checks: `run_tests.py` and
`check_git_release.py` load only `tests/`, and the CI runs `run_tests.py`.
