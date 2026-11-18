# GPU 实验 harness（快照）

[English](README.md) | 简体中文

本目录是运行我们 GPU 实验的 harness 的过滤快照：发布 smoke 矩阵的四个 harness。它只供本仓库的成员使用，不属于
runtime、测试或任何发布（发布导出会丢弃 `harnesses/**`）。

## 什么是 harness

`examples/` 是公开的 benchmark 客户端（SWE-bench、Terminal-Bench、MemoryAgentBench），任何人都可以对 `serve.py` 运行。
harness 则是围绕一个 runtime commit 的实验机制。它：

1. 把一个固定（pin）的 runtime commit（密封的 Git bundle，在集群上全新 clone）连同它的发布回执（`check_git_release.py`
   的结果和日志哈希）打包成密封的 payload（`build.py` -> `dist/`）；
2. 定义 plan：服务（策略、`serve.py` 参数、KV 大小）、时间上限和 GPU 小时（`*-config.json`，复制进 `pins.json`）；
3. 通过一次 SSH 连接在 DeltaAI 上提交一个 Slurm 作业（ghx4 batch，一块 GH200）（`run_one_connection.ps1`、
   `controller.py`）；
4. 在作业里（`job.sh`、`job.py`）把发布安装到干净环境，依次启动各个服务；
5. 驱动 benchmark 客户端：SWE-bench 和 Terminal-Bench 在 agent 主机上运行（`tbench/`，由 `tbench/deploy_tbench.ps1`
   部署，经 SSH 隧道连接），MemoryAgentBench 用 harness 自己的 runner 在作业里运行；
6. 在作业里检查证据（`smoke_checks.py`、`gate_checks.py`、`*_checks.py`）；
7. 在本地取回并核验证据（`postrun/fetch_*.ps1`、`postrun/verify_evidence.py`）；
8. 生成表格（`postrun/*_tables.py`、`postrun/join_*.py`）。

每个 harness 自己的 `README.md` 记录它的决定，`STATUS.md` 记录它的作业和结果。

## 四个 harness

2026-11-17 的快照即各 harness 的在线状态（也见 `SNAPSHOT.json` 中各 harness 的 `note`）。`perf-swe32` 固定在
integrate-1117 `2fb7f62`，即 `main` 的 runtime（2026-11-17 晚重新同步；发布回执 `GIT_ARCHIVE_TESTS_PASS`，2,287 个测试，141 个跳过）；另外三个仍固定在 integrate-1112
`4ebff97`（2,059 个测试，139 个跳过），尚未重新固定到 integrate-1116（`perf-gemma-swe32` 还要先提高 session 上限：
docs/known-limits.zh-CN.md 5.5）。本快照在 integrate-1116 上；integrate-1116 合并之前，`main` 保留 2026-11-13 的快照。下表的
判定只经过核验程序（未审计）；每个作业见 [docs/status.zh-CN.md](../docs/status.zh-CN.md)。

| Harness | 固定的提交 | Payload（`dist/payload.json` 的 SHA-256） | 本地验证 | 从这个 payload 提交的 GPU 作业 |
| --- | --- | --- | --- | --- |
| `perf-swe32` | `2fb7f62` | `f62064f1...`（加入 gate 2，plan `parity150`；排队中的 150 个实例作业所用的 payload `269a00e6...` 冻结为仓库外的 `perf-swe32-269a00e6`） | `validation\20261117T190547` | sess24ri r2 3616345（`SWE32_FAIL_VERIFIED`，来自空闲释放检查的缺陷，已修复）；排队中：full150k24 3616690、stockkeep150k24 3616705、ledger150k24 3616729 |
| `perf-gemma-swe32` | `4ebff97` | `aaa272a2...` | `validation\20261114T141219` | 无（smokea 3586987 来自冻结的 `e5c93e1e` payload，2026-11-16 在运行前取消） |
| `perf-tb16-strip` | `4ebff97` | `4203f013...` | `validation\20261115T015540` | 无（smokeqt r2 3586720 来自冻结的 `98a149cd` payload，2026-11-16 在运行前取消） |
| `perf-mab` | `4ebff97` | `bd816111...` | `validation\20261115T201624` | smqc r1 3604146（`PMAB_PASS_VERIFIED`）；smqt 3584105 和 smga 3601174（都是 `PMAB_PASS_VERIFIED`）来自冻结的 `89d65f99` payload |

更早的作业运行的 payload 现在是冻结副本（不在快照中），例如 `perf-swe32` 第一个 4ebff97 payload（`c7c3e8dc...`）上的
stock32ti 3583822。

| Harness | 运行内容 | 主要 plan |
| --- | --- | --- |
| `perf-swe32` | SWE-bench Verified，Mini-SWE-Agent 2.4.2，Qwen3.6-27B：前 32 个实例（`*64*` plan：前 64 个），16 个会话 | 发布 smoke `smoke`（J1）和 `smoket`（thinking on）；Table 1 `full`、`obs`、`rfill`、`goalb4`、`stock`（`*i`：ghx4-interactive 变体）；`recentt`、`rtoolst`、`akvt`、`akvs`；thinking on 的 `goalti`、`akvsi`、`rtoolsti`、`fullt(i)`、`fulltr(i)`（idle release）；64 实例代理 `obs64(i)`、`goal64`、`full64`、`recent64`、`akv64`、`rfill64(i)`；thinking on 的 stock 对照 `stock32t(i)`、`stock64t`；`ctx24`、`ctx25`、`vllm25`、`stock25(x)`；phase B（2026-11-16，integrate-1116）：脚本化边界情况的 `keepedge(i)`、`keepedgesync(i)`、`keepedgegoalp(i)`，`--gdn-inplace-crossing` 的成对 plan `gdnobsp(i)`、`gdngoalp(i)`、`gdnidlep(i)`、`gdnobs24p(i)`，24 个 session 的测试 `sess24(i)`，以及开 `--async-scheduling` 的 64 任务 plan `full64a`、`recent64a`、`obs64a`、`goal64a`、`ledger64a`、`stockkeep64t`（保留 reasoning 的 stock 对照）和它们 K = 24 的版本（`...k24`、`stockkeep64k24`）；开空闲释放的 24 个 session 的 smoke `sess24r(i)`；前 150 个实例、24 个 session 的 `full150k24`、`stockkeep150k24`、`ledger150k24`（`tasks/tasks-pool-150.json`；并排对比表 `postrun/run150_tables.py`）；发布门槛 2（full 与 stock 逐 token 对比）`parity150` |
| `perf-gemma-swe32` | 同样的 SWE-bench 运行，Gemma 4 31B IT | 发布 smoke `smoke`、`smokehf`、`smokea`（异步调度）（J2）；`full`、`obs`、`obs4k`、`rfill`、`goal`、`stock` |
| `perf-tb16-strip` | Terminal-Bench 2.0，Harbor 0.21.0 的 Terminus 2，Qwen3.6-27B 和 Gemma 4 31B IT | 发布 smoke `smokeq`、`smokeqs`、`smokeqt`（thinking on，带历史 strip）以及 `smokeg`、`smokegh`；`gctx25`（Gemma，262,144 token）；89 个任务的 `full`、`obs`、`rfill`、`goal`、`stock` 及其 Gemma `g*` plan；dry 子集 plan |
| `perf-mab` | MemoryAgentBench kv-session 协议，Qwen3.6-27B 和 Gemma 4 31B IT（不用 agent 主机） | 发布 smoke `smq`、`smqt`（J5）和 `smg`、`smga`（J6）；GPU-PLAN `g1`、`g2`、`g4o`、`g4g`、`g4v`、`gm1`、`gm4*` |

## 是快照，不是工作副本

在线的 harness 是维护者机器上的文件夹 `I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\<name>\`。它们仍是
权威的工作副本：其脚本、READY 文件和 agent 都使用这些绝对路径。本目录是由 `sync_from_live.py` 刷新的过滤副本；不要在
这里修改（下次刷新会覆盖修改，在线 harness 也看不到）。`SNAPSHOT.json` 为每个 harness 记录快照时间、文件数、字节数、
清单哈希、固定的 commit、被排除的内容和状态说明；不在其中的 harness 尚未进入快照。

刷新方法（只在该 harness 没有正在进行的修改时做快照）：

```
python -I -B harnesses/sync_from_live.py --dry-run [NAME ...]
python -I -B harnesses/sync_from_live.py --note "<state>" [NAME ...]
python -I -B harnesses/scan_snapshot.py
```

然后检查扫描结果并提交。同步需要一个保存在仓库之外的本地修改清单（`sync_from_live.py` 中的 `EDIT_RULES`）；被它修改
的文件列在 `SNAPSHOT.json` 的 `edited` 下，其余文件逐字节复制。同步会删除在线文件夹已不再有的文件，并拒绝未列入
`ALLOW_LARGE` 的超过 2 MB 的文件。排除的内容：

| 排除 | 原因 |
| --- | --- |
| `validation/` | 本地验证日志（运行输出） |
| `dist/`、`*.bundle` | 密封 payload 和 runtime bundle（`build.py` 从固定的 commit 重新生成） |
| `__pycache__/`、`*.pyc` | 字节码 |
| `fixtures/` 以外的 `*.log` | 运行日志（`fixtures/` 下的日志摘录保留：测试会读取） |
| `*-PROGRESS.md` | agent 可恢复的工作笔记；结果在 `STATUS.md` 和 READY 文件里 |
| `*.bak`、`*.orig`、`*.tmp`、`*~` 等 | 临时副本 |
| `HELD_BACK` | 隐私扫描后保留在外的单个文件：`perf-gemma-swe32/fixtures/gemma-render-job3429830.json`（一个作业的模型回复和 token id，违反 fixture 只存计数的规则），因此 `test_real_tokenizer.GemmaJoinRender` 只能在在线副本中运行 |

`harnesses/.gitattributes` 保持写入的字节不变（不做行尾转换：pin 对这些文件做了哈希），`harnesses/.gitignore` 保留
根目录 `.gitignore` 会隐藏的 `fixtures/` 日志。`scan_snapshot.py` 检查密钥和禁用词（发现即退出码 1）、fixture
中的长文本以及账户和主机数据。
快照中含有 Slurm 通知邮箱、集群账户和用户名、agent 主机的主机名和 SSH 主机公钥：因仓库是私有的而保留。

## 约定

- **重新固定前先冻结。** harness 改固定到另一个 runtime commit 或为新作业修改之前，先把当前状态冻结为同级文件夹
  `<name>-<payload 哈希>`（或 `-job<id>`），附 SHA-256 清单和 `FROZEN-<hash>.md`；已提交的作业用冻结副本（即它们运行的
  payload）取回和核验。冻结副本不在快照中。
- **用 fake 做本地验证。** `local_validation.ps1` 构建两次（payload 必须相同），运行 harness 的测试、release-parity 和
  re-pin 检查，以及每个 plan 的 SelfTest 和 DryRun，端到端地针对 DeltaAI 和 agent 主机的 fake（`fake_ssh.py`、
  `fake_stage.py`、`fake_server.py`、`fake_tbench.py`），不建立任何连接。日志写入 `validation/`。
- **READY 文件。** `<WHAT>_READY.txt`（`SMOKE_READY.txt` 等）把一个验证过的 plan 交给用户：runtime commit 和回执、
  payload 和入口脚本的哈希、本地验证结果、冻结的前一版本，以及确切的提交、取回和核验命令。由用户提交（DeltaAI 需要
  Duo）；没有 READY 文件就不提交。

## 不能直接在别处运行

这些 harness 依赖维护者的 Windows 机器（Windows PowerShell 5.1、OneDrive 和 `D:\Remote-RA-temp` 路径、它的 Python）、
DeltaAI（账户、用户、ghx4 分区和模型路径）以及 agent 主机（主机公钥、隧道密钥和已部署的环境）。它们不属于本仓库的
检查：`run_tests.py` 和 `check_git_release.py` 只加载 `tests/`，CI 运行的是 `run_tests.py`。
