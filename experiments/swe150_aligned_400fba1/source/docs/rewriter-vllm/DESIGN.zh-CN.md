# vLLM 上的 Goal rewriter（`--goal-rewriter-engine vllm`，需显式开启）

[English](DESIGN.md) | 简体中文

分支 `rewriter-vllm`，基于 `cf2b28d`；2026-10-30 的设计（第 1 步）以及 2026-11-01 的 lm-format-enforcer 修复，
二者都已进入 `main`（`cd42995`；`16dcb4d`、`c07d42a`）。这个引擎是稳定配置的一部分（与
`--goal-rewriter-batch-invariant` 和 `--goal-overlap` 一起，作业 3576565）。它的 GPU 运行及其判定见
[docs/status.zh-CN.md](../status.zh-CN.md) 和[历史](../history.zh-CN.md#历史)；它的限制（子进程的 GPU 显存没有预留；
只在稳定配置的命令中验证过）见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 3.1 和 3.2。
GPU 运行为这个设计测得的结果写在下面各节中。第 2 步已实现为 `--goal-overlap`
（[docs/goal-overlap/DESIGN.zh-CN.md](../goal-overlap/DESIGN.zh-CN.md)）。

## 动机

在 Terminal-Bench 上（16 个 session，Qwen3.6-27B，`cf2b28d`），一个 `goal_query` 轮次平均花 3.1 s 等待它的
Goal：1.9 s 用在 rewriter 的生成上，1.2 s 排在其他 session 的调用后面。在 362 个轮次中调用了
rewriter 的 75 个上，等待平均为 15.0 s（中位数 9.7 s，p95 53 s）。rewriter（Qwen3-0.6B）在一个子进程中运行 Hugging Face
`generate` 加 lm-format-enforcer，在一把锁下一次服务一个最多 4 个调用的批次：
单独运行时每个生成步 72 ms，成批时 94 ms（之前的回放中，在空闲 GPU 上单独运行为 60.5 ms）。

目标是降低小模型的开销，并让它与大模型重叠运行。第 1 步（本文档）
把生成移到一个带 continuous batching 的小型 vLLM 引擎中。第 2 步（之后）让阶段 1 与
rewriter 调用重叠。

## 与现在完全相同的部分

- 何时调用 rewriter、发给它的请求（`RewriteRequest.to_payload()`）、prompt（vendored 的
  `qwen_goal_rewriter.py` 的 system 文本、规范化的 user JSON、`apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False)`）、JSON schema（`candidate_schema`）、greedy 解码、`max_new_tokens` 384、解码
  （`skip_special_tokens=True`）、`parse_exact_candidate`、父进程的校验（`validate_rewrite_candidate`）、
  每一种回退及其原因、120 s 的调用超时，以及回答行的格式。
- rewriter 对象仍是一个 `PersistentCommandGoalRewriter`（它的类名是 Goal 构造器控制哈希的一部分，
  因而也是每个 `selector_config_hash` 的一部分）；选页合同的 `goal_rewriter` pin（`REWRITER_KEYS`）不变。
  与 `--goal-rewriter-batch` 一样，在这个实例之上安装一个传输层。
- `goals.jsonl` 保留每个字段（`request`、`response`、`error`、`wait_ms`、`start_ms`、`call_ms`、`input_tokens`、
  `output_tokens`、`generation_latency_ms`）；native 文件保留 `goal_rewrite_wait_ms` / `goal_rewrite_call_ms`。
- 不开这个参数时什么都不变：没有新的启动记录，HF 子进程、它的命令和每个固定项都与现在相同。

改变的是模型调用。vLLM 的 kernel 和它的 structured-output 后端都不是 HF 的那一套，所以 Goal
可能与 HF 服务的不同（不是逐位相同，与批处理一样）。只在引擎相同的服务之间比较策略；
回放（见下文）测量候选一致的频率。

## 拓扑

一个新的子进程脚本 `source-overlay/remote_ra_live_v2/goal_rewriter_vllm_child.py` 运行一个 `AsyncLLM` 引擎，
服务固定的 Qwen3-0.6B snapshot，运行在它自己的进程（及其 EngineCore 子进程）中，使用同一张 GPU。不放在 27B
引擎内部（一个 vLLM 引擎只有一个模型），不用离线的 `LLM`（多个线程调用不安全），也不用 HTTP 服务
（不需要让 harness 的端口检查再跟踪第二个端口；协议中也不会混入日志）。

- **干净的 vLLM。** 子进程启动时 `sys.path` 只有干净环境的 site-packages（没有 overlay，
  否则会加载它打过补丁的 `vllm/` 和 gate），按文件路径加载 vendored 的 `qwen_goal_rewriter.py` 以使用它的 prompt、
  schema 和解析函数，环境中清除了 `REMOTE_RA_*`、`VLLM_POLICY_*` 和 `SELECTIVE_*`
  （`REMOTE_RA_HTTP_RUNTIME`、`REMOTE_RA_DYNAMIC_STARTUP`、`VLLM_POLICY_RUNTIME_V1` 由 `serve.py` 设置）。
- **stdout 只承载协议行。** 子进程复制 fd 1 供协议使用，然后先把 fd 1 指向 fd 2，
  再导入 vLLM，于是 vLLM 的 logger、它的 EngineCore 以及任何 C 层的打印都进入 stderr。
- **进程组。** 子进程在自己的 session 中启动；关闭它时结束整个进程组，所以不会有 EngineCore
  成为孤儿进程而一直占着 GPU 显存。
- **引擎参数**（固定，记录在 `startup.json` 中）：固定的 snapshot、`local_files_only`、bfloat16、
  `max_num_seqs` 16、`max_model_len` 16,384（在记录到的调用中，有上限的请求最多约 7.5K 个 prompt token：
  最多 7,544 个）、`max_num_batched_tokens` 8,192、`kv_cache_memory_bytes` 取自 `--goal-rewriter-vllm-kv-gib`（默认
  3 GiB，按每 token 112 KiB 约为 28K token；vLLM 日志记录为 28,080）、`gpu_memory_utilization` 0.05（KV 字节数显式给出之后，它只是 vLLM 的启动门槛：
  要求有 4.8 GiB 空闲）、关闭 prefix caching、开启 CUDA graph，capture 尺寸为 1、2、4、8、16，
  structured-output 后端取自 `--goal-rewriter-guided-backend`（见下文），`generation_config` 为 "vllm"。
- **请求。** prompt 是 HF tokenizer 对上游 prompt 文本给出的 token id（`TokensPrompt`），所以
  `input_tokens` 等于现在的计数。采样：temperature 0（greedy）、`max_tokens` 384、`stop_token_ids` 为
  tokenizer 的 `<|im_end|>` 和 EOS、`structured_outputs=StructuredOutputsParams(json=schema)`、`detokenize=False`、
  `output_kind` FINAL_ONLY。子进程用 HF tokenizer 和 `skip_special_tokens=True` 解码生成的 id，
  与上游完全相同，然后调用 `parse_exact_candidate`；解析失败时（包括在 384 token 处被截断的回复，
  `finish_reason` 为 "length"），以上游的错误行及其 token 计数作答。
- **并发。** 一个专用线程上的 asyncio loop 拥有引擎。一个读线程解析请求行，
  用 `run_coroutine_threadsafe` 逐个提交；回答在一把写锁下各写一行，
  按完成顺序写出。行协议就是上游的协议（输入 `{"request_id","request"}`，输出 `{"request_id","result"}` 或
  `{"request_id","error"[,"telemetry"]}`，readiness 记录也相同）；只是可以同时有多个调用在进行，
  回答也可能乱序返回。一行 `{"abort": request_id}` 中止一个请求。

## 父进程一侧

`source-overlay/remote_ra_live_v2/goal_rewriter_vllm.py` 安装在 `PersistentCommandGoalRewriter` 实例之上，
与 `BatchedRewriter.install` 相同：

- 一个子进程，在 27B 引擎分配完 KV cache 之后**提前**启动
  （vLLM 的启动门槛和显存顺序都要求大引擎先就位），在 READY 之前，所以最初的 Goal 调用不必承担引擎的启动（编译、
  graph capture；启动超时仍为 300 s）。`ready.json` 记录子进程的启动用时。
- 每个 `rewrite()` 都针对一个 ticket 运行上游的 `rewrite`（与 `goal_rewriter_batch._Ticket` 相同：伪造的锁、
  进程、`_readline`、`close`），所以上游的外层封装、回答检查和校验都原样运行，
  `RewriterTap` 也与现在一样记下 `locked` / `sent` / `answered`（`wait_ms`
  变成等待另一个调用重启子进程的时间，接近 0；见下文“用时”）；
- 一个读线程按 `request_id` 把回答行交给对应的 ticket；
- 调用超时会在子进程中中止该请求，并且只让这一个调用失败（现在超时会重启子进程；
  而重启一个 vLLM 引擎要花一分钟）；子进程退出会让所有进行中的调用失败，下一个调用会重启它；
- 关停时关闭子进程的进程组。

## 参数与固定项

- `--goal-rewriter-engine {hf,vllm}`（默认 `hf`）。`vllm` 需要 `goal_query` 加 `--goal-rewriter-model` 和
  `--admission-overlap K`（K >= 2；否则同一时间只会有一个调用在进行），且不能与 `--goal-rewriter-batch` 同用。
- `--goal-rewriter-guided-backend {lm-format-enforcer,xgrammar}`（默认 `lm-format-enforcer`，即 HF
  子进程使用的 enforcer）。vLLM 0.24 的 xgrammar（以及 guidance，所以 `auto` 也一样）拒绝 schema 中的 `uniqueItems`
  （没有重新检查：那次测试没有保留记录，而每个 GPU 组发给 xgrammar 的都是去掉它的 schema）；
  `xgrammar` 设置会传入去掉 `uniqueItems` 的 schema（父进程的校验仍会拒绝重复项）。
  这改变了模型看到的约束，所以它只是一个测量组，由 Goal 负责人决定。
- `--goal-rewriter-vllm-kv-gib G`（默认 3）和 `--goal-rewriter-batch-invariant`（设置 `VLLM_BATCH_INVARIANT=1`，
  只作用于子进程：它的输出不再取决于有多少调用一起运行）。在 rw3（作业 3499550；取自其离线核验，未审计）中测得：
  每次调用的速度大致相同（单独运行时墙钟时间中位数 947.54 ms，不开时 974.89 ms；16 个同时进行时 16,604.77 对 16,373.98 ms，
  慢 1.4 %），但它的引擎启动用了 73,954 ms，不开时为 35,427 ms。
- `startup.json` / `ready.json` 增加一条 `goal_rewriter_engine` 记录（引擎、后端、KV 字节数、utilization 门槛、
  batch invariance、引擎参数、启动 ms），只在用 `vllm` 时才有。Gemma 4 profile 也接受这个参数
  （同一个子进程；它的 GPU 显存见下一节）。

## GPU 显存

现在没有为 rewriter 预留任何空间；HF 子进程的批次上限让 27B 服务旁边留出约 1 GiB 空闲
（`serve.py` 的显存说明）。vLLM 子进程需要权重（约 1.2 GB）、它的 KV（默认 3 GiB）、
8,192 个批量 token 时的激活、CUDA-graph 池和一个 CUDA context：约 5-6 GiB。在 25 GiB 的 Qwen 服务旁边约有
12.4-14.1 GiB 空闲：CUDA 的 97,362 MiB 中（`serve.py` 的说明），25 GiB 的 Terminal-Bench 运行 3461661 的峰值为
86,408 MiB（带 HF 子进程，其中 1,721 MiB 是它加载时的用量），所以服务本身最多占 84,687 MiB（剩 12,675 MiB
空闲）；同一服务不带 rewriter 时峰值为 82,801 MiB，把 `serve.py` 中 32 GiB 的说明按比例换算到 25 GiB 则剩
14,428 MiB。约 7 GiB（97,362 - 90,102 MiB）是 32 GiB 服务的数字（作业 3404230）。perf-rwreplay2 的
`mem25` 和 `mem32` 阶段占住 GPU 显存，使恰好剩下这 12,675 和 7,260 MiB 空闲，并在其中启动子进程
（见它的 README）。在 rw3（作业 3499550；未审计）中测得：剩 12,676 和 7,260 MiB 空闲时，子进程（3 GiB KV）
在两种情况下都启动了，并回答了 16 次调用中的 16 次；graph capture 占 0.11 GiB，子进程峰值为 5,786 和 5,718 MiB。
Gemma（31B 权重）剩下的更少：在作业 3538523 中，带这个子进程的 Gemma 服务（gvll）峰值为 92,500 MiB，
同一服务带 Hugging Face 子进程（govl）时为 89,874 MiB；gvll 在 261.1 s 内到达 READY，它的子进程约
112 s 启动（取自该作业的离线核验，未审计）。

## 验证

- **CPU**（`test_goal_rewriter_vllm.py`）：为子进程提供一个假的 `vllm` 包（一个按脚本运行、带延迟的 `AsyncLLM`）；
  prompt id 等于 HF `prepare` 的 id；解码和解析与上游相同；16 个并发调用乱序作答并被正确路由；
  错误、在 384 处截断、超时 + 中止、子进程退出；清理过的环境和 `sys.path`
  （子进程从不导入 overlay 的 `vllm`）；stdout 只有协议行；参数校验和固定项；默认的 `hf`：
  启动配置、命令和证据与现在逐字节相同；控制哈希不变。
- **GPU 1，回放**（仿照 perf-rwreplay rw1）：记录下来的 682 次 rewriter 调用，各组为 HF（重跑）、vLLM +
  lm-format-enforcer、同样配置加 batch invariance、vLLM + xgrammar（schema 去掉 `uniqueItems`）：
  有效率和回退率、与 HF 的候选一致率（HF 自己重跑的一致率是下限：rw1 中 92 % 相同）、
  每次调用在单独运行和 16 个同时进行时的延迟、每个生成步的 ms、GPU 显存。
- **GPU 2**：一次开 `--goal-rewriter-engine vllm` 的 Terminal-Bench（或 SWE-bench）Goal dry run，按轮次分解用时，
  与 10-26 的运行对比。

## 第 1 步的实现（2026-10-30）

按上文实现：`remote_ra_live_v2/goal_rewriter_vllm_child.py`（子进程）、`goal_rewriter_vllm.py`（传输层、记录、
固定项）、`serve.py` 中用于 Qwen 和 Gemma 4 profile 的四个参数、`remote_ra_http/config.py`（加载检查）、
`query_selection.py` / `private_endpoint.py`（authority 按固定的记录启动子进程）、
`remote_ra_http/evidence.py`（`worker-attached.json` 和 `ready.json`）、[docs/reference.zh-CN.md 4.10 节](../reference.zh-CN.md#410-admission-优化需显式开启)和[docs/history.zh-CN.md](../history.zh-CN.md#历史)（两种语言）、
release 固定项（329 个源文件，238 个 payload）。测试：`test_goal_rewriter_vllm.py`（当时 27 个；在 `main` 上是
`tests/test_goal_rewriter_vllm.py`，31 个；进程组测试只在 Linux 上运行），包括上面的 CPU 清单和一次端到端运行：test_gemma_goal_rewriter 的十六个 Gemma 4
`goal_query` session 在固定了 `goal_rewriter_engine` 时，决策与使用 Hugging Face 子进程假实现时完全相同
（authority 按启动配置启动真实的子进程）。用 `hf` 时，launcher、rewriter 命令和启动证据都等于
`cf2b28d` 的（身份探测：283 项，0 处差异）。

细节以及与上文的偏差：

- **提前启动的位置。** 没有新的 hook：`compose_engine` 构建 `QuerySelectionAuthority`（经由
  `SplitPrivateEndpoint`）的时机，是在 `EngineCore.__init__` 完成 profile、分配 KV cache 并 capture 它的 graph 之后，
  B0 握手、输入线程和 READY 之前；Hugging Face 子进程现在就在这里启动。vLLM
  子进程也在同一位置启动，受 pin 中 300 s 的启动超时约束。它的启动用时经由
  `worker-attached.json`（`goal_rewriter_engine.startup_ms`，只在用 `vllm` 时）从引擎传到前端，`ready.json` 复制它。
- **子进程也按文件路径加载 `goal_rewriter_child.py`**（用它的 `prepare` 和 `error_answer`，
  只依赖标准库），所以 prompt id 恰好就是 Hugging Face `prepare` 的 id；子进程本身不导入 `torch`
  （generate 的 CUDA 调用看到的是一个主机端替身）。
- **停止 id：** checkpoint 的 generation_config 中的 eos id（Transformers 在那里停止）、tokenizer 的 EOS 和
  `<|im_end|>`：对 Qwen3-0.6B 为 {151643, 151645}，即 Hugging Face 的那一组。
- **引擎参数**增加 `tokenizer` = 模型、`distributed_executor_backend` `uni`、`tensor_parallel_size` 1 和
  `disable_log_stats`；`local_files_only` 体现为子进程的 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`（vLLM 没有这样的引擎参数）
  和 tokenizer 自己的 `local_files_only`。`--goal-rewriter-vllm-kv-gib` 取 2-16（2 GiB 是按每 token
  112 KiB 能容纳一条 16,384 token 序列的最小值）。
- **中止**会取消该请求的 task（vLLM 的 `generate` 会在引擎中中止被取消的请求）；
  被中止的请求没有回答行。已经失败的调用迟到的回答由父进程丢弃。
- **用时：** `wait_ms` 是等待另一个调用重启子进程的时间，`start_ms`
  是该调用自己的重启和写出它那一行的时间，`call_ms` 是从发送到回答；`generation_latency_ms`（子进程）
  从提交给引擎算到最终输出，所以包含该请求与其他请求共享引擎的时间。
- **失败：** 报告自己已死（`errored`）的引擎会回答它的调用并结束子进程，
  所以下一个调用会重启它；stdin 结束时取消进行中的请求并关停引擎。超过 16,384 - 384 token 的 prompt
  预计会让它的调用失败（vLLM 拒绝它），而 Hugging Face 子进程会运行它；没有实际测试过：
  记录到的最大 prompt 有 7,544 token。
- **Dry run：** dry-run 行也显示 `goal_rewriter_engine`。

当时尚未解决的问题（那时还没有任何东西在 GPU 上运行，也没有针对真实的 vLLM 0.24 wheel 运行），以及 GPU 运行给出的答案
（取自各作业的离线核验，未审计）：

- 子进程使用的 vLLM 0.24 名称（`AsyncEngineArgs` 的字段、`CompilationConfig(cudagraph_capture_sizes=...)`、
  `StructuredOutputsConfig(backend=...)`、`StructuredOutputsParams(json=...)`、`RequestOutputKind.FINAL_ONLY`、
  `TokensPrompt`、`AsyncLLM.from_engine_args`、`errored`、`shutdown`）依照 API 笔记，没有对照
  wheel 检查过；第一个 GPU 作业会检查它们。答案：回放 rw2（作业 3488202，runtime `cd42995`）通过了它的 vLLM
  API 探测。
- 25 和 32 GiB 的 Qwen 服务以及 Gemma 服务旁边的 GPU 显存；
  0.05 的 utilization 门槛在 27B 引擎分配之后能否通过；子进程的启动用时（编译和 graph capture）与 300 s 超时的比较。
  答案：子进程在真实的 25 GiB 服务旁边用 111,749 ms 启动（作业 3500565）；32 GiB 的情形只在显存占位下运行过
  （rw3 `mem32`）；Gemma 见“GPU 显存”一节（作业 3538523）。
- vLLM 的 lm-format-enforcer 后端是否约束 schema（包括 `uniqueItems`），像 Hugging Face
  集成那样；以及 `VLLM_BATCH_INVARIANT=1` 对 0.6B 引擎有什么要求（后端，eager 还是 graph）。
  第一次回放发现该后端在 vLLM 0.24 上无法使用；修复见下一节。带修复时（rw3，作业 3499550，
  runtime `c07d42a`）没有任何调用出现后端错误。
- CPU 假实现逐行独立解码，所以无法显示 vLLM 的 Goal 与 Hugging Face 的有多常不同：
  这要靠回放的一致性测量。rw3 的答案：lfe.alone 的候选在 160 次调用中有 99 次与 Hugging Face 的相同
  （0.6188，低于回放 0.70 的门槛）；开启 batch invariance 时，单独运行与 16 个同时进行给出的候选在 160 次调用中
  160 次相同。在 Gemma 上（作业 3538523），带这个子进程的服务的 Goal 在 23 个 lockstep 轮次中有 22 个与
  Hugging Face 服务的相同。

## vLLM 0.24 上的 lm-format-enforcer（rewriter-lfe-fix，2026-11-01；已在 rw3 中在 GPU 上运行，作业 3499550）

第一次 GPU 回放（perf-rwreplay2，作业 3488202，runtime `cd42995`）中，每个 lm-format-enforcer 组（`lfe`、
`lfebi`、`mem25`、`mem32`）的每次调用都在几毫秒内失败，而 xgrammar 组回答了每次调用。
子进程日志给出了原因，在 EngineCore 中每个请求一次：`preprocess_add_request` -> `grammar_init` ->
`LMFormatEnforcerBackend.__post_init__` -> `build_vllm_token_enforcer_tokenizer_data` -> vLLM 的 `LazyLoader` 导入
`lmformatenforcer.integrations.vllm`，它的第 4 行（lm-format-enforcer 0.11.3，
即干净环境中安装的版本）是 `from vllm.transformers_utils.tokenizer import MistralTokenizer`。vLLM 0.24 把它的 tokenizer 移到了
`vllm/tokenizers/`（`MistralTokenizer` 在 `vllm/tokenizers/mistral.py` 中）；导入失败，
集成把它重新抛出为 `ImportError: vllm is not installed`。后端是惰性构建的，失败后从不缓存，
所以每个请求都以同样方式失败；引擎保持运行，并在不产出 token 的情况下结束每个请求，
所以子进程把每次调用都作为解析失败（`JSONDecodeError`）作答，回放把它算作回退。换一个版本不可行：
干净环境是封存的。下一个导入也会失败：
lm-format-enforcer 的 Transformers 集成从 `transformers.tokenization_utils` 导入 `PreTrainedTokenizerBase`，
而 Transformers 5 移动了它（上游脚本为它安装一个进程内的别名）。

修复只在子进程中：

- **集成模块。** 在 vLLM 加载之前，子进程把自己的一个模块以
  `lmformatenforcer.integrations.vllm` 为名放进 `sys.modules`；之后 vLLM 的 `LazyLoader`（`importlib.import_module`）
  返回它。脚本的顶层代码在它以 `__main__`（子进程）或 `__mp_main__`（用 spawn 方式启动的
  EngineCore 会以这个名字重新运行父进程的主脚本）运行时完成这一步；fork 出的 EngineCore 继承它。
- **与 Hugging Face 子进程相同的允许 token。** 该模块的 `build_vllm_token_enforcer_tokenizer_data`（vLLM
  以 `use_bitmask=True` 和模型的词表大小调用它）返回 Hugging
  Face 子进程构建的数据（`build_transformers_prefix_allowed_tokens_fn` -> `build_token_enforcer_tokenizer_data(tokenizer)`：
  对 `len(tokenizer)` 个 id 做 `_build_regular_tokens_list`、它的 decoder 和 EOS），只是 bitmask
  宽度设为模型的词表大小（vLLM 的 bitmask 每个模型 id 一位）。
  lm-format-enforcer 自己的 vLLM 构造器会列出每个模型 id：tokenizer 之外的 padding id 解码为 ""，所以它们会位于其 token 树的根部，
  并且在每一步都被允许，而 Hugging Face 子进程从不允许它们。路径的其余部分是 vLLM 的后端，
  以及与 Hugging Face 子进程相同的 lm-format-enforcer 类（`JsonSchemaParser(schema)`、`TokenEnforcer`）：
  它的第一次调用（还没有生成的 token）是根状态，与 Hugging Face prefix function 的第一次调用（prompt）一样；
  之后的 token 只解码当前的词，所以每一步允许的集合就是 Hugging Face 子进程在生成了相同
  token 时的集合。进程内的 Transformers 别名就是上游脚本的那个。
- **启动检查。** 引擎启动之后、readiness 记录之前，一个受约束的请求（`PROBE_SCHEMA`：
  一个唯一取值为 `{"probe": "remote-ra-guided-probe", "items": ["a"]}` 的对象，带 `uniqueItems`，对 xgrammar
  与每个请求一样去掉它；greedy；128 token；停止 id；prompt 为 "Say hello."）
  必须恰好返回该值，以 stop token 结束，且只使用 tokenizer 的 id。否则子进程记录 `refused: the guided decoding
  (...) failed its startup check: ...`，并在没有 readiness 记录的情况下退出，所以服务不会启动。
  构建失败的后端通不过这项检查；完全不加约束的后端会用散文作答。只在启动时检查：
  不按调用检查任何东西。
- **测试**（`test_goal_rewriter_vllm.py`、`GuidedBackendTests`）：假的 site 采用 vLLM 0.24 的布局（没有
  `vllm/transformers_utils/tokenizer.py`，有 `vllm/tokenizers/mistral.py`）、lm-format-enforcer 0.11.3
  的两个集成模块（它们的 import 块和构造器）以及 Transformers 5 移动过的基类；
  假引擎在一个单独的解释器中为第一个结构化请求构建它的后端，该解释器由 `multiprocessing.spawn.prepare` 设置成一个
  spawn 出的 EngineCore；构建错误被记录下来，并在不产出 token 的情况下结束请求，与回放中相同。有了修复，后端得到 Hugging Face 子进程的数据（137 个 tokenizer id，lm-format-enforcer 自己的构造器会允许的 23 个
  padding id 一个都没有），放在一个 160 id 的 bitmask 中。变异：没有该模块，
  或者该模块只在子进程中（不在 spawn 出的 EngineCore 中），或者没有别名：
  子进程以集成的错误拒绝启动；没有启动检查：每次调用都像回放中那样失败；遍历每个模型 id 的数据；
  一个未受约束的回复；一个越界的 id。

## 第 2 步（不在本次改动中）：重叠

现在一次 admission 先构造它的 Goal，然后运行阶段 1，因为 Goal query 的 forward 在阶段 1 内部运行
（`private_endpoint.reserve_steps`、`phase1_coordinator.run`）。把 Goal forward 从阶段 1 中拆出来，就能让阶段 1
（新 token 的 prefill）在 rewriter 生成时运行，之后再做 Goal forward 和选页。
更大的重叠（更早启动 rewriter，或在当前轮次使用上一个 Goal）会改变 Goal 的行为，
由 Goal 负责人决定。之后作为需显式开启的 `--goal-overlap` 实现（[docs/goal-overlap/DESIGN.zh-CN.md](../goal-overlap/DESIGN.zh-CN.md)）。
