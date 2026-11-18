"""--thinking-history strip on the window layout (Gemma 4): the sliding window held at the cut (option G).

Form E commits a session at the cut X = e (thinking_cut; docs/thinking-cut/DESIGN.md). What crosses
turns on the window layout is the sliding window: the blocks of ``sliding_window.kept_blocks(X)``
of every sliding group, 812.5 MiB per session at P = 16 and W = 1,024. At the phase-1 end that
window exists (phase 2 starts on it), but vLLM's SlidingWindowManager frees each block that leaves
the window while phase 2 decodes past X: after about 16 decoded tokens the first one goes, after
about 1,024 all of them. So a strip request holds it (option G of the 11-09 study; the host copy,
option H, is not built):

* engine (remote_ra_native_growing.window_pins.WindowPins, adapter.EngineCoreAdapter): one more
  reference on the window at X in the request's row, at a continuing turn's install (the head is
  ``kept_blocks(e)``) or, at local turn 0, before the first allocation at a cursor at or past X;
  the commit pins the held blocks (``pin_after(..., X)``), a failed admission or a cancel releases
  them;
* worker (window_owner.WindowAllocationOwner.cut_window, window_lifecycle): the window at X from the
  native rows of the first forward that holds or ends at X, committed as the ``WindowCarry`` at X
  (``drain(..., positioned=True)``); its blocks are held for the request until its drain;
* capacity (kv_layout.WindowLayout.hold_slots H, window_capacity): pool 1 reserves H more window
  slots; the gate admits at most H requests that hold (or will hold) a window at once and counts
  each one's hold term; the launcher sizes H against pool 0 by the server's rule.

Nothing here runs per request or step: this module holds the startup attachments and pins. Without
the strip history (H = 0: every Qwen server, and every Gemma server without --thinking-history strip)
none of it is attached and nothing changes. A Gemma 4 strip server attaches it through thinking_cut
(``thinking_cut.attach_engine`` / ``attach_worker`` hand the window layout here; ``check_config``
requires H >= 1; serve.py --window-hold-slots sizes H). CPU-tested only, not on a GPU
(docs/thinking-gemma/DESIGN.md section 4.4).
"""
from __future__ import annotations

from .contract import require


def attach_engine(adapter, bridge, *, split=False):
    """compose_engine on a strip server of the window layout: the completion commits at the cut
    (``commit_at_cut``, as thinking_cut.attach_engine) and the window at X is held in the layout's
    hold slots. ``split``: also end a local turn 0 prefill forward at X (EngineBridge.scheduler_split);
    the window layout does not need it (the window's KV is exact whatever chunk computed it)."""
    window = getattr(adapter, 'window', None)
    layout = getattr(adapter, 'layout', None)
    require(window is not None and getattr(layout, 'state', None) == 'sliding',
            'the window hold needs the window layout')
    require(type(layout.hold_slots) is int and layout.hold_slots >= 1,
            '--thinking-history strip on the window layout needs hold slots in its KV layout (hold_slots >= 1)')
    require(not getattr(adapter, 'commit_at_cut', False) and not getattr(bridge, 'commit_at_cut', False) and
            not window.hold_slots and not window.holds,
            'thinking_history engine attachment collision')
    window.hold_slots = layout.hold_slots
    adapter.commit_at_cut = True
    adapter.cut_records = {}            # request -> its committed cut (the native file's truncate)
    if split:
        bridge.commit_at_cut = True


def attach_worker(owner, qwen, backend):
    """attach_window_worker on a strip server: the window owner, the window lifecycle and the window
    backend commit at the cut (the window held at X)."""
    require(type(owner).__name__ == 'WindowAllocationOwner' and getattr(qwen, 'state_kind', None) == 'sliding' and
            getattr(backend, 'window', False) is True, 'the window hold needs the window worker')
    require(not getattr(owner, 'commit_at_cut', False) and not getattr(qwen, 'commit_at_cut', False) and
            not getattr(backend, 'commit_at_cut', False), 'thinking_history worker attachment collision')
    owner.commit_at_cut = qwen.commit_at_cut = backend.commit_at_cut = True


def pinned_hold_slots(config, layout):
    """The hold slots a startup config may pin in its window layout (remote_ra_http.config.window_layout):
    the layout's H when the config pins --thinking-history strip (thinking_cut.pinned, which also
    requires H >= 1 of a Gemma 4 strip config) and refuses --async-scheduling (v1), else 0 (a layout
    that pins H without the strip history is refused). Startup only."""
    hold = getattr(layout, 'hold_slots', 0)
    if not hold:
        return 0
    from .thinking_cut import pinned
    require(pinned(config) is True, 'window hold slots need --thinking-history strip')
    from .async_sched import pinned as async_pinned
    require(not async_pinned(config), '--thinking-history strip on the window layout refuses --async-scheduling (v1)')
    return hold
