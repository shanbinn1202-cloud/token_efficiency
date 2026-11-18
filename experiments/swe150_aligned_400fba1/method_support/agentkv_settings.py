"""Shared AgentKV overrides for the SWE64 paper-aligned protection settings."""
OPTIONS = {
    'eviction_interval': 128,
    'anchor_budget': 32,
    'observation_window': 32,  # Upstream recent_tokens=None inherits this value.
    'sink_tokens': 32,
    'pooling_kernel_size': 1,
}
