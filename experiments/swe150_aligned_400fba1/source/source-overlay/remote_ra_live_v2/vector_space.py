"""Pre-final-norm hidden rows for Selector V2 vectors.

Upstream captures the output of the last decoder layer (``hidden_layer=-1``),
i.e. the residual stream before the model's final RMSNorm.  vLLM's runner hands
our hooks the normalised output ``x * rsqrt(mean(x^2) + eps) * gain``.  The
per-token factor is a positive scalar, so dividing by the per-channel ``gain``
recovers the exact direction of ``x``; every vector is L2 normalised later.
Only vector consumers see these rows: logits keep using the native output.
"""
from hashlib import sha256

from .contract import require


def resolve(model):
    import torch
    stacks = [(name, module) for name, module in model.named_modules()
              if hasattr(module, 'layers') and hasattr(module, 'norm')
              and hasattr(module, 'embed_tokens')]
    require(len(stacks) == 1, 'exactly one decoder stack with a final norm required')
    name, text = stacks[0]
    norm = text.norm
    kind = type(norm).__name__
    require(kind in ('GemmaRMSNorm', 'RMSNorm') and hasattr(norm, 'weight'),
            'unsupported final norm for pre-norm vector recovery: ' + kind)
    weight = norm.weight.detach().to(dtype=torch.float32)
    # GemmaRMSNorm (Qwen3.5/3.6) scales by (1 + weight); RMSNorm by weight.
    gain = weight + 1.0 if kind == 'GemmaRMSNorm' else weight
    require(gain.ndim == 1 and bool(torch.isfinite(gain).all())
            and float(gain.abs().min()) > 1e-3, 'final norm gain cannot be inverted')
    info = dict(module=(name + '.norm') if name else 'norm', kind=kind,
                hidden_size=int(gain.numel()), decoder_layers=len(text.layers),
                gain_sha256=sha256(gain.cpu().numpy().tobytes()).hexdigest(),
                representation='pre_final_norm_last_token')
    return info, gain


def attach(runner):
    require(not hasattr(runner, '_remote_ra_final_norm_gain'), 'vector space already attached')
    info, gain = resolve(runner.model)
    runner._remote_ra_final_norm_gain = gain
    runner._remote_ra_vector_space = info
    return dict(info)


def pre_final_norm(runner, hidden_states):
    gain = getattr(runner, '_remote_ra_final_norm_gain', None)
    require(gain is not None, 'pre-norm vector space not attached')
    require(hidden_states.ndim == 2 and hidden_states.shape[1] == gain.shape[0],
            'hidden width differs from the final norm')
    return hidden_states.float() / gain.to(device=hidden_states.device)
