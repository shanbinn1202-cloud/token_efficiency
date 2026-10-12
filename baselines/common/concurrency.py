"""Concurrency helpers shared across scripts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


async def gather_bounded(
    items: Sequence[InputT],
    worker_fn: Callable[[InputT], Awaitable[OutputT]],
    max_workers: int,
) -> list[OutputT]:
    """Run async workers with a concurrency limit while preserving input order.

    Args:
        items: Ordered input items to process.
        worker_fn: Async function that processes a single item.
        max_workers: Maximum number of concurrent workers. Must be positive.

    Returns:
        Worker results in the same order as the input items.

    Raises:
        ValueError: If ``max_workers`` is not positive.
        Exception: Re-raises the first exception produced by a worker.

    Example:
        >>> import asyncio
        >>> asyncio.run(
        ...     gather_bounded(
        ...         [1, 2, 3],
        ...         lambda value: asyncio.sleep(0, result=value * 2),
        ...         max_workers=2,
        ...     )
        ... )
        [2, 4, 6]
    """
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    if not items:
        return []

    semaphore = asyncio.Semaphore(max_workers)

    async def _run(item: InputT) -> OutputT:
        async with semaphore:
            return await worker_fn(item)

    return await asyncio.gather(*(_run(item) for item in items))


def iter_parallel_completed(
    items: Sequence[InputT],
    worker_fn: Callable[[InputT], OutputT],
    max_workers: int,
) -> Iterator[OutputT]:
    """Yield worker results as soon as each item completes.

    Args:
        items: Ordered input items to process.
        worker_fn: Function that processes a single item.
        max_workers: Maximum number of worker threads. Must be positive.

    Yields:
        Worker results in completion order rather than input order.

    Raises:
        ValueError: If ``max_workers`` is not positive.
        Exception: Re-raises the first exception produced by a worker.

    Example:
        >>> list(iter_parallel_completed([1, 2, 3], lambda value: value * 2, max_workers=2))
        [2, 4, 6]
    """
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    if not items:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()
