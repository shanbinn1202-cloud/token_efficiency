# 发布门槛

[English](release-gates.md) | 简体中文

写给修改本 runtime 或报告其数字的人：一个改动怎样进入 `main`，一个 GPU 作业在其数字可用之前必须证明什么，以及每个指标的含义。这里只写规则；已经通过的内容见 [docs/status.zh-CN.md](status.zh-CN.md)，还不能用的见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)。这是本仓库成员的流程；由维护者决定。

## 1. 分支与 `main`

- 改动在功能分支上完成，合并进集成分支 `integrate-MMDD` 并在那里测试；第 2 节的 GPU 测试通过、维护者批准推送之后，才进入 `main`。
- 本仓库在 GitHub 上有 `main`、各集成分支、数据分支 `data-swe32`（运行归档），以及合作者自己推送的分支；要合并进集成分支的功能分支留在本地。
- `main` 写明它的稳定提交：[docs/status.zh-CN.md](status.zh-CN.md#main-与稳定提交) 的第一节和 README 的开头几行写明 `main` 是哪个提交、它包含哪个 runtime 提交、稳定 tag 是哪个提交。
- **稳定 tag。** 在模型 × benchmark 矩阵（Qwen3.6-27B 和 Gemma 4 31B IT；SWE-bench Verified、Terminal-Bench 2.0、MemoryAgentBench）全部六格都通过运行的最后一个提交，供需要最高稳定性的人使用。有提交做到之后才创建；在此之前 [docs/status.zh-CN.md](status.zh-CN.md) 写明最接近的提交。

## 2. 按改动类型的 GPU 测试

2026-11-15 的 `main` 规则（它取代完整的模型 × benchmark 冒烟矩阵，作为进入 `main` 的门槛）：

| 改动 | 进入 `main` 之前 |
| --- | --- |
| 需显式开启（新开关，默认关闭；插件策略） | 它自己的 GPU 测试，以及证明默认路径不变的 CPU 证据：`tools/policy_parity.py` 与基线逐字节相同，`check_git_release.py` 通过 |
| 默认开启或在共用路径上 | release check，以及每个模型（Qwen 和 Gemma）至少一次快速的 GPU 冒烟 |
| 客户端默认值、文档 | release check（代码），或文档检查（第 5 节） |

每次推送到 `main` 都需要维护者批准，[docs/status.zh-CN.md](status.zh-CN.md) 记录该改动跑过哪些格和哪些策略。

## 3. 一个数字什么时候可以用

GPU 作业的一个数字，只有在作业的判定是离线核验过的通过、并且下面三项独立的参照检查都在它上面通过时，才能作为结果写进 [docs/status.zh-CN.md](status.zh-CN.md)，或用于论文和组内报告。harness 还没有运行某项检查时，状态表的那一行写明这一点。这些检查适用于 integrate-1116 及之后的 runtime 上的作业（2026-11-16 决定）；[docs/status.zh-CN.md](status.zh-CN.md) 中更早作业的各行没有经过这些检查（对 keep-rebuild-1116 之前 thinking 开的运行，检查 1 在 session 第一次格式错误之后不会成立，见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 1.4）。检查 2 目前推迟：在它运行之前，各行写明它没有运行。

1. **每次调用的 keep 渲染**（thinking 开，XML 协议）。`tools/keep_render_check.py` 用固定的模板（`preserve_thinking`）渲染每次调用的请求，并与服务实际的 prompt 比较：每次调用给出 `equal`、`text_equal` 和 `exact_equal`。通过：每个 session 的每次调用判定（`exact_equal` 且 `text_equal`）都成立。
2. **定期在一小段上逐 token 比较 full 与 stock。** 在一段固定的小范围上（相同的 prompt、一个解码步窗口），runtime 的 `full` 服务和使用相同 kernel 的 stock vLLM 0.24.0 给出相同的 token id。共用路径改动时运行，产出数字的每个 runtime 提交至少运行一次。通过：窗口内 token id 相同（有差异时要解释清楚，例如 CUDA graph 下的近似平局，否则数字等待）。
3. **每次关闭之后的池不变量。** 每个 session 关闭之后，关闭回执要对得上（`released.blocks` = `resident_pages` + `tail` + `gather_blocks`，[docs/design.zh-CN.md](design.zh-CN.md#8-每个服务的证据) 第 8 节），没有打开的 session 时各池回到启动时的大小。通过：运行中每次关闭都对得上，没有残留被钉住的内容。

## 4. 脚本化的边界冒烟

每种边界情况都有一个在 release check 中运行的 CPU 脚本，注明的情况还有一次 GPU 冒烟；结果写进 [docs/status.zh-CN.md](status.zh-CN.md)。

| 情况 | CPU 脚本 | 预期 | GPU 上 |
| --- | --- | --- | --- |
| 池已满：一轮等待（`CAPACITY_WAIT`），直到有关闭或释放 | `tests/test_ms16.py`、`tests/test_adm_overlap.py`、`tests/test_idle_release.py` | 没有抢占；每一轮都与从不等待的服务相同 | 作业 3583081 中的空闲释放 |
| session 关闭，它的 Page 归还 | `tests/test_ms_pool.py`、`tests/test_token_gather_e2e.py` | 关闭回执对得上（第 3 节） | 每次 harness 运行 |
| 格式错误的回滚 | `tests/test_keep_rebuild.py`、`tests/test_inplace_32.py` | 与之前完全相同的历史；被放弃的重试恢复 session | 尚未脚本化 |
| 32 个 session | `tests/test_ms32.py`、`tests/test_inplace_32.py` | launcher 的限制；没有抢占 | 未运行 |
| 每个请求一个 Gated-DeltaNet 槽位（`--gdn-inplace-crossing`） | `tests/test_gdn_inplace.py`、`tests/test_inplace_32.py` | 输出与关闭该开关时相同 | 未运行 |
| 一个回复在另一个请求解码时被 `max_tokens` 截断（`--async-scheduling`） | `tests/test_async_sched.py`（`ResolveTests`） | 不失败；它的停止与同步服务一样判定 | 作业 3604146 的 async-cut 检查 |

## 5. 提交前的检查

- **代码、固定值、清单或 payload 中的任何内容：** 完整的 release check，`python -B check_git_release.py --ref HEAD`，一台机器上同时只跑一个完整测试；负载下的计时失败，在改任何代码之前先单独重跑。
- **只改文档：** 读取文档的测试（`test_installer`、`test_chat_text_server`、`test_kv_session_server`、`test_gemma_chat_text`、`test_gemma_kv_session`、`test_ctx_256k`、`test_speed_defaults`、`test_tb_client`、`test_policy_registry`）和 `tests/test_docs_parity.py`（链接和中英一致性，`tools/docs_check.py`）；它所依托的代码的 release check 仍然有效，并要写明是哪一次。
- **集群的登录节点上：** 只做轻量检查（harness 自己的自检和 dry run，约 3 到 5 分钟）；完整测试在别处运行。

## 6. 给组里或论文用的数字

- 每个数字在给组里看或公开使用之前都要审计：由起草者之外的人（或代理）从原始证据重新推导，并通过同一作业的窗口内解码步与 stock vLLM 比较。
- 论文中的延迟和吞吐来自经过 GPU 验证的优化 runtime 与 stock vLLM 的对比；论文中不描述这些优化本身。
- 尚未审计的数字，出现在哪里都标为“未审计”。

## 7. 指标

harness 使用的定义；出处的行号是 `harnesses/` 快照 `57887c7` 中的行号。

| 指标 | 定义 | 出处 |
| --- | --- | --- |
| SWE-bench 解决数 | 评测报告写明 resolved 的实例数，占计划实例数（32） | `harnesses/perf-swe32/README.md` |
| SWE-bench 吞吐 | `tok/s (runner span)` = 客户端生成的 token / 从第一个实例开始到最后一个实例结束的时长；`throughput_all` 再加上没有留下轨迹的实例在服务端生成的 token | `harnesses/perf-swe32/README.md` 451-455 |
| Terminal-Bench 准确率 | 解决的 trial（Harbor 核验器给出奖励 1.0）/ 计划的 89 个（或子集的大小） | `harnesses/perf-tb16-strip/README.md` 1135 |
| Terminal-Bench 吞吐 | ATIF 主 agent 的完成 token / 各 agent 的时间跨度（第一个 agent 开始到最后一个 agent 结束），不含摘要子调用（2026-10-17 批准） | `harnesses/perf-tb16-strip/README.md` 1135-1139 |
| 未记录的 token | 已送达但不在轨迹中的回复（被 agent 超时截断的 trial：`Unrecorded tok (timeout-cut)`）、上下文上限回退中的简短摘要（`Short summary tok (unrecorded)`）和在长度上限处截断的回复（`Length-cut tok (unrecorded)`）在分子之外单独计数，从不计入分子；客户端与服务端的交叉核对恰好允许这些 | `harnesses/perf-tb16-strip/README.md` 1140-1173 |
| 干净 ITL | 对每个批大小 b，ITL 一直低于 bound(b) 的稳态区间；bound(b) = 不小于 b 时稳态区间平均 ITL 下四分位数 2 倍的最小 ITL 直方图边界；看一个普通解码步时读 `clean_itl_ms` / `clean_tok_s` | `harnesses/perf-tb16-strip/README.md` 134-144 |
| 损失的解码时间 | 每个采样区间的 sum(ITL) − 观测数 × 该区间批大小下的干净步时，全程求和，占 vLLM ITL 总和的比例；低 / 高值取区间两端的干净步时 | `harnesses/perf-tb16-strip/README.md` 146-151；`harnesses/perf-swe32/README.md` 445-450 |
| MemoryAgentBench 分数 | 每个 case 按其能力的指标：`exact_match`（上下文学习）、`substring_exact_match`（RULER QA、FactConsolidation）、InfBench 摘要用 LLM 评判（没有评判运行时为 `judge_pending`）；综合分数：各能力的平均，没有评判时为 `composite_score_without_judge`，即其余三项能力的平均（打分推迟，见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.1） | `harnesses/perf-mab/README.md` 850-852；[docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a11-指标) 的 A.11 |

## 8. 文档

- 中英文页面在同一个提交里修改；`tools/docs_check.py` 保证两者的标题、代码块、commit 号、数字和链接的页面一致，并且每个链接有效。
- 作业结果只写在 [docs/status.zh-CN.md](status.zh-CN.md)；其他页面指向那里。新的限制写进 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)。[docs/history.zh-CN.md](history.zh-CN.md) 只追加，不改写；冻结的记录放进 `docs/history/`。
- 页面名称以及代码和 harness 快照引用的节号保持不变。
