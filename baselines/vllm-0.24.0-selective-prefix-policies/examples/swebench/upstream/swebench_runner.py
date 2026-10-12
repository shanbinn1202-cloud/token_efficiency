#!/usr/bin/env python3
"""
AgentDiet unified-setting SWE-bench runner.

在 mini-swe-agent 基础上增加三个功能：
  1. 非正式提交（Informal Submission）：跟踪 git diff 输出，步数耗尽时自动用最后一次有效 diff 作为补丁
  2. 死循环检测（Loop Detection）：连续 N 步命令相同时终止实例，避免浪费计算资源
  3. AgentDiet-style trajectory reduction：在完整 step 写入 self.messages 后压缩 delayed historical step

用法（替换原来的 -m minisweagent.run.benchmarks.swebench）：
  python swebench_agentdiet_runner.py --subset verified --split test ...
"""

import os
import re
import logging
import sys
import time
from pathlib import Path

import minisweagent.run.benchmarks.swebench as _swebench_module
from minisweagent.exceptions import LimitsExceeded, TimeExceeded

logger = logging.getLogger("agentdiet_runner")

_AGENTDIET_IMPORT_ERROR = None
_AGENTDIET_ADAPTER_DIR = Path(__file__).resolve().parent
if _AGENTDIET_ADAPTER_DIR.exists():
    sys.path.insert(0, str(_AGENTDIET_ADAPTER_DIR))
try:
    from agentdiet_adapter import AgentDietAdapter
except Exception as exc:  # pragma: no cover - only exercised when optional adapter is broken/missing
    AgentDietAdapter = None
    _AGENTDIET_IMPORT_ERROR = exc

# ─── 本地 SIF 镜像优先逻辑 ────────────────────────────────────────────────────
# Optional local SIF storage path: {SIF_BASE}/{instance_id}/image.sif
_SIF_BASE_ENV = os.environ.get("SIF_BASE", "").strip()
SIF_BASE = Path(_SIF_BASE_ENV) if _SIF_BASE_ENV else None

_original_get_image_name = _swebench_module.get_swebench_docker_image_name


def _get_image_with_local_sif(instance: dict) -> str:
    """
    优先使用本地 SIF 文件，路径：{SIF_BASE}/{instance_id}/image.sif
    若不存在则回退到原始 docker:// URI（从 Docker Hub 拉取）。
    """
    iid = instance.get("instance_id", "")
    sif_path = SIF_BASE / iid / "image.sif" if SIF_BASE else None
    if sif_path and sif_path.exists():
        # singularity build --sandbox 支持直接传 SIF 路径（无需 docker:// 前缀）
        # mini-swe-agent 在 singularity 模式下会加上 docker://，所以这里返回纯路径
        # 实际在 swebench.py 第 100 行：env_config["image"] = "docker://" + image_name
        # 对于本地 SIF，需要绕过这个前缀，因此返回带 file:// 前缀以便后续处理
        logger.info("[%s] 本地 SIF 存在: %s", iid, sif_path)
        return str(sif_path)   # 返回绝对路径，singularity build --sandbox 可直接接受
    # 回退：标准 Docker Hub 镜像名
    return _original_get_image_name(instance)


# Patch：替换镜像名获取函数
_swebench_module.get_swebench_docker_image_name = _get_image_with_local_sif


# patch mini-swe-agent 2.2.x 的 get_sb_environment：
# 当 image_name 是本地绝对路径时，singularity 不能再加 docker:// 前缀
def _patch_get_sb_environment():
    """
    上游 swebench.py 逻辑（2.2.8）：
      elif env_class in ["singularity", "contree"]:
          env_config["image"] = "docker://" + image_name
    若 image_name 是 /scratch/.../image.sif，会错误变成 docker:///scratch/...，导致去 Docker Hub 拉取失败。
    """
    original_get_sb_environment = _swebench_module.get_sb_environment

    def patched_get_sb_environment(config: dict, instance: dict):
        env_config = config.setdefault("environment", {})
        env_config["environment_class"] = env_config.get("environment_class", "docker")
        image_name = _swebench_module.get_swebench_docker_image_name(instance)

        if env_config["environment_class"] in ["docker", "swerex_modal"]:
            env_config["image"] = image_name
        elif env_config["environment_class"] in ["singularity", "contree"]:
            if image_name.startswith("/") or image_name.startswith("file://"):
                local_path = image_name.replace("file://", "", 1)
                env_config["image"] = local_path
                iid = instance.get("instance_id", "")
                logger.info("[%s] 使用本地 SIF 镜像: %s", iid, local_path)
            elif image_name.startswith("docker://"):
                env_config["image"] = image_name
            else:
                env_config["image"] = "docker://" + image_name

        env = _swebench_module.get_environment(env_config)
        if startup_command := config.get("run", {}).get("env_startup_command"):
            startup_command = _swebench_module.Template(
                startup_command,
                undefined=_swebench_module.StrictUndefined,
            ).render(**instance)
            out = env.execute(startup_command)
            if out["returncode"] != 0:
                raise RuntimeError(f"Error executing startup command: {out}")
        return env

    _swebench_module.get_sb_environment = patched_get_sb_environment
    return original_get_sb_environment

_patch_get_sb_environment()
# ─────────────────────────────────────────────────────────────────────────────

# ─── 可调参数 ─────────────────────────────────────────────────────────────────
LOOP_THRESHOLD = 10        # 连续相同命令超过此次数视为死循环
MIN_DIFF_LINES = 3         # git diff 至少这么多行才算有效候选补丁
# ─────────────────────────────────────────────────────────────────────────────


class SmartAgent(_swebench_module.ProgressTrackingAgent):
    """
    功能扩展：
      - 死循环检测：连续 LOOP_THRESHOLD 次相同命令 → 终止并返回候选补丁
      - 非正式提交：跟踪每次 git diff 输出；步数耗尽时若有候选补丁则自动提交
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent_cmds: list[str] = []   # 最近执行的命令（循环检测用）
        self._candidate_patch: str = ""      # 最后一次有效的 git diff
        self._agentdiet_adapter = self._init_agentdiet_adapter()

    def _init_agentdiet_adapter(self):
        mode = os.environ.get("AGENTDIET_MODE", "").strip().lower()
        if not mode or mode == "off":
            return None
        if AgentDietAdapter is None:
            raise RuntimeError(
                "AGENTDIET_MODE is set, but AgentDietAdapter could not be imported "
                f"from {_AGENTDIET_ADAPTER_DIR}: {_AGENTDIET_IMPORT_ERROR}"
            )
        instance_id = str(getattr(self, "instance_id", ""))
        return AgentDietAdapter.from_env(instance_id=instance_id)

    def add_messages(self, *messages):
        result = super().add_messages(*messages)
        adapter = getattr(self, "_agentdiet_adapter", None)
        if adapter and adapter.enabled and self.messages:
            if self.messages[-1].get("role") != "assistant":
                adapter.maybe_compress(self.messages, event="add_messages")
        return result

    # ── 覆盖 query：步数耗尽时注入候选补丁 ─────────────────────────────────
    def query(self) -> dict:
        """替换原始 step limit 检查，步数耗尽时使用候选补丁。"""
        if 0 < self.config.step_limit <= self.n_calls:
            submission = self._candidate_patch
            status = "LimitsExceeded_InformalSubmission" if submission else "LimitsExceeded"
            if submission:
                logger.info("[%s] 步数耗尽，使用非正式补丁（%d 行）",
                            self.instance_id, submission.count("\n"))
            raise LimitsExceeded({
                "role": "exit",
                "content": status,
                "extra": {"exit_status": status, "submission": submission},
            })

        if 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({
                "role": "exit",
                "content": "LimitsExceeded",
                "extra": {"exit_status": "LimitsExceeded", "submission": ""},
            })

        if 0 < self.config.wall_time_limit_seconds <= int(
            time.time() - self._start_time
        ):
            raise TimeExceeded({
                "role": "exit",
                "content": "TimeExceeded",
                "extra": {"exit_status": "TimeExceeded", "submission": ""},
            })

        # 正常执行：计数、调模型、记录费用
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    # ── 覆盖 execute_actions：扫描输出 + 死循环检测 ─────────────────────────
    def execute_actions(self, message: dict) -> list[dict]:
        # 提取本轮 bash 命令
        for action in message.get("extra", {}).get("actions", []):
            cmd = action.get("args", {}).get("command", "") if isinstance(action, dict) else ""
            if cmd:
                self._recent_cmds.append(cmd.strip())

        # 执行（调用父类链）
        results = super().execute_actions(message)

        # 扫描输出，更新候选补丁
        for msg in results:
            content = str(msg.get("content", ""))
            if "diff --git" in content or "--- a/" in content:
                patch = self._extract_diff(content)
                if patch and patch.count("\n") >= MIN_DIFF_LINES:
                    self._candidate_patch = patch
                    logger.debug("[%s] 候选补丁已更新（%d 行）",
                                 self.instance_id, patch.count("\n"))

        # 死循环检测
        if len(self._recent_cmds) >= LOOP_THRESHOLD:
            last_n = self._recent_cmds[-LOOP_THRESHOLD:]
            if len(set(last_n)) == 1:
                submission = self._candidate_patch
                status = ("LoopDetected_InformalSubmission" if submission
                          else "LoopDetected")
                logger.warning("[%s] 死循环：连续 %d 次相同命令，终止。补丁: %s",
                               self.instance_id, LOOP_THRESHOLD,
                               "有" if submission else "无")
                raise LimitsExceeded({
                    "role": "exit",
                    "content": status,
                    "extra": {"exit_status": status, "submission": submission},
                })

        return results

    # ── 辅助：从工具输出中提取 diff 内容 ────────────────────────────────────
    @staticmethod
    def _extract_diff(content: str) -> str:
        """从 tool 输出（含 XML 包装）中提取 unified diff 补丁。"""
        output_match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
        raw = output_match.group(1).strip() if output_match else content
        diff_match = re.search(r"(diff --git .+)", raw, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        unified_match = re.search(r"(--- a/.+)", raw, re.DOTALL)
        return unified_match.group(1).strip() if unified_match else ""


# ─── 替换框架中的 ProgressTrackingAgent ─────────────────────────────────────
_swebench_module.ProgressTrackingAgent = SmartAgent


# ─── 入口 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from minisweagent.run.benchmarks.swebench import app
    app()
