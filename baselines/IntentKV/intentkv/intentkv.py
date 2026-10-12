from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import *
from intentkv.intentkv_cluster import IntentKVCluster

class BaseKVCache(Cache):
    def __init__(self):
        super().__init__()
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.kv_clusters = {}
        self.kv_cluster_granularity = "layer"

        self.temp_key_cache = []
        self.temp_value_cache = []

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        raise NotImplementedError(
            "Make sure to implement `get_kv_cluster_class_config` in a subclass."
        )

    def get_kv_cluster_class(self, layer_idx: int, head_idx=None):
        cluster_name, cluster_class, cluster_config = self.get_kv_cluster_class_config(
            layer_idx, head_idx
        )
        if cluster_name not in self.kv_clusters:
            self.kv_clusters[cluster_name] = cluster_class(**cluster_config)
        return self.kv_clusters[cluster_name]

    def compresssed_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        layer_idx: int,
    ):
        if self.kv_cluster_granularity == "layer":
            kv_cluster = self.get_kv_cluster_class(layer_idx)

            key_compress, value_compress = kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
            )
            self.key_cache.append(key_compress)
            self.value_cache.append(value_compress)
        else:
            assert (
                False
            ), f"kv_cluster_granularity {self.kv_cluster_granularity} not supported"

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        update_global_past_kv = cache_kwargs.get("update_global_past_kv", True)
        query_states = cache_kwargs["query_states"]
        attention_mask = cache_kwargs["attention_mask"]
        num_key_value_groups = cache_kwargs["num_key_value_groups"]

        if key_states.size(1) != query_states.size(1):  # GQA
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        q_len = query_states.shape[-2]
        initializing_kv_cluster = False
        if (
            len(self.key_cache) == layer_idx
        ):  # initialize kv_cluster, ie, the first query/context
            initializing_kv_cluster = True
            self.compresssed_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                num_key_value_groups,
                layer_idx,
            )
        else:  # the follow up queries/contexts
            if update_global_past_kv:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )
            else:  # add KVs to temp_kv_cache
                if len(self.temp_key_cache) == layer_idx:
                    self.temp_key_cache.append(key_states)
                    self.temp_value_cache.append(value_states)
                else:
                    self.temp_key_cache[layer_idx] = torch.cat(
                        [self.temp_key_cache[layer_idx], key_states], dim=-2
                    )
                    self.temp_value_cache[layer_idx] = torch.cat(
                        [self.temp_value_cache[layer_idx], value_states], dim=-2
                    )

        torch.cuda.empty_cache()
        if not initializing_kv_cluster:  # return the compressed KV cache
            if self.temp_key_cache:  # concat global past_kv and temp_kv_cache
                key_states = torch.cat(
                    [self.key_cache[layer_idx], self.temp_key_cache[layer_idx]], dim=-2
                )
                value_states = torch.cat(
                    [self.value_cache[layer_idx], self.temp_value_cache[layer_idx]],
                    dim=-2,
                )
            else:
                key_states = self.key_cache[layer_idx]
                value_states = self.value_cache[layer_idx]
        key_states = repeat_kv(key_states, query_states.size(1) // key_states.size(1))
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def to_legacy_cache(self):
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        cache = cls()
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
        return cache

    def clear_temp_kv_cache(self):
        if self.temp_key_cache:
            self._seen_tokens -= self.temp_key_cache[-1].shape[
                -2
            ]
        self.temp_key_cache = []
        self.temp_value_cache = []

class IntentKVCache(BaseKVCache):
    def __init__(self, config):
        super().__init__()
        self.init_q_window = config.get("init_q_window", 32)
        self.max_capacity_prompt = config.get("max_capacity_prompt", 4096)
        self.block_size = config.get("block_size", 128)

    def get_kv_cluster_class_config(self, layer_idx: int, head_idx: int = None):
        cluster_config = {
            "init_q_window": self.init_q_window,
            "max_capacity_prompt": self.max_capacity_prompt,
            "block_size": self.block_size
        }
        cluster_name = ",".join(["intentkv"] + [str(i) for i in cluster_config.values()])
        return cluster_name, IntentKVCluster, cluster_config