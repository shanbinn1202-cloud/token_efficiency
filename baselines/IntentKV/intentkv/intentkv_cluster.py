import math
import torch
import torch.nn as nn

class IntentKVCluster:
    def __init__(
        self,
        init_q_window=64,
        max_capacity_prompt=256 + 64,
        block_size=128,
    ):
        self.init_q_window = init_q_window
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.init_q_window > 0
        self.block_size = block_size
        
    def reset(
        self,
        init_q_window=64,
        max_capacity_prompt=256 + 64,
        block_size=128,
    ):
        self.init_q_window = init_q_window
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.init_q_window > 0
        self.block_size = block_size
        
    def _square_root_js_divergence(self, p, q):
        p = p / p.sum(-1, keepdim=True)
        q = q / q.sum(-1, keepdim=True)
        m = (p + q) / 2
        return torch.sqrt(
            0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
        )
        
    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(
                query_states[..., -self.init_q_window :, :], key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            mask = torch.full(
                (self.init_q_window, self.init_q_window),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[
                :, :, -self.init_q_window :, -self.init_q_window :
            ] += attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights_window = attn_weights[
                :, :, -self.init_q_window :, : -self.init_q_window
            ]
            
            # cumulative attention by rows, reducing the effect of special tokens like punctuations and new line
            cumulative_attention = torch.stack(
                [attn_weights_window[0, :, i:i+3, :].sum(axis=-2) for i in range(attn_weights_window.shape[-2])],
                dim=1
            ).to(attn_weights_window.device, dtype=attn_weights_window.dtype)
            js_divergence = self._square_root_js_divergence(cumulative_attention, cumulative_attention[:, 0][:, None, :]).nan_to_num()
            # diff_one = torch.diff(js_divergence[..., 5:-10], dim=-1, prepend=js_divergence[..., 4:5])
            # we start from the 8th token, and end at the 10th last token
            # since the first tokens usually have similar attention distributions and ensure the minimum window size
            # instead of using directly the diff_one, we use the difference of the difference to imporve robustness
            diff_two = js_divergence[..., 8:-10] - js_divergence[..., 6:-12] - js_divergence[..., 5:-13] + js_divergence[..., 11:-7]
            js_divergence_indices = diff_two.topk(1, dim=-1).indices.squeeze_()+5
            
            attn_weights_sum = torch.zeros((attn_weights_window.shape[0], attn_weights_window.shape[1], attn_weights_window.shape[3]), device=attn_weights_window.device)
            for i in range(attn_weights_sum.shape[1]):
                attn_weights_sum[:, i, :] = attn_weights_window[:, i, js_divergence_indices[i]:, :].sum(dim=-2)
            
            # the remainder will be directly stored in the cache for simplicity
            remainder = attn_weights_sum.size(-1) % self.block_size
            if remainder > 0:
                attn_weights_block_sum = attn_weights_sum[:, :, :-remainder].reshape(
                    (bsz, num_heads, -1, self.block_size)
                ).sum(dim=-1)
            else:
                attn_weights_block_sum = attn_weights_sum.reshape(
                    (bsz, num_heads, -1, self.block_size)
                ).sum(dim=-1)
            indices = attn_weights_block_sum.topk(
                (self.max_capacity_prompt - self.init_q_window - remainder) // self.block_size, dim=-1
            ).indices.sort(dim=-1).values
            
            
            indices = (indices * self.block_size)[:, :, :, None].repeat(1, 1, 1, self.block_size) + torch.arange(self.block_size, device=indices.device).view(1, 1, 1, -1)
            indices = indices.flatten(-2, -1)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, : -self.init_q_window, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.init_q_window, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.init_q_window - remainder:, :]
            v_cur = value_states[:, :, -self.init_q_window - remainder:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states