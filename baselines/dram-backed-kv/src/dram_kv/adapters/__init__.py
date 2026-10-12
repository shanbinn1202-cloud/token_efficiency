"""Adapters for importing recorded experiments into the common contracts."""

from .legacy_kv_drift import LegacyReplayBundle, build_legacy_replay_bundle

__all__ = ["LegacyReplayBundle", "build_legacy_replay_bundle"]

