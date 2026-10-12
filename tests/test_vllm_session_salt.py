"""A shared transport must isolate each benchmark case's prefix cache."""
from concurrent.futures import ThreadPoolExecutor
import threading
from tebench.backends.base import Backend
from tebench.backends.vllm import VLLMBackend

def test_case_cache_salt_is_thread_local_and_reset_between_cases():
    backend=VLLMBackend.__new__(VLLMBackend)
    Backend.__init__(backend,{})
    barrier=threading.Barrier(2)
    def case(_):
        first=backend._cache_salt
        barrier.wait()
        assert backend._cache_salt==first
        backend.reset()
        second=backend._cache_salt
        assert first!=second
        return first,second
    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs=list(pool.map(case,range(2)))
    assert len({salt for pair in pairs for salt in pair})==4
