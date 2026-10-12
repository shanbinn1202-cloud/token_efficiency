"""Compatibility shim: AgentKV uses native vLLM FCFS scheduling."""

def phase(scheduler):
    return None
