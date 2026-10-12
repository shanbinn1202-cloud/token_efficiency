"""DRAM uses the same raw append-only session transport as other KV policies."""
from .kv_session import KVSessionBackend


class DramVLLMBackend(KVSessionBackend):
    def __init__(self, config):
        super().__init__(config, method='dram_backed_kv')
