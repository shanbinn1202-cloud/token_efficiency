"""Union of official Harbor agent-execution intervals, clipped to first allocation.

Includes LLM waits, tool execution and agent work; excludes intervals spent only
on environment preparation or grading. Token counts still come from the engine.
"""
import json

def measured_metrics(raw, events):
    lower, upper = float(raw['started']), float(raw['finished'])
    starts, intervals = {}, []
    for row in events:
        key = row['trial']; stamp = float(row['time'])
        if row['event'] == 'agent-start':
            if key in starts: raise ValueError('Duplicate unclosed agent interval')
            starts[key] = stamp
        elif row['event'] == 'agent-end':
            if key not in starts: raise ValueError('Agent end without start')
            begin = starts.pop(key)
            if stamp < begin: raise ValueError('Negative agent interval')
            intervals.append((max(lower,begin), min(upper,stamp)))
        else: raise ValueError('Unknown agent event')
    intervals.extend((max(lower,begin),upper) for begin in starts.values())
    merged=[]
    for begin,end in sorted((b,e) for b,e in intervals if e>b):
        if merged and begin<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
        else: merged.append([begin,end])
    wall=sum(e-b for b,e in merged)
    if not wall: raise ValueError('No measured agent interval')
    return dict(raw, raw_capture_wall_seconds=raw['wall_seconds'], wall_seconds=wall,
        agent_tokens_per_second=raw['tokens'].get('agent',0)/wall,
        agent_intervals=merged, unfinished_agent_intervals=len(starts),
        timing_definition='union of Harbor AGENT_START/AGENT_END intervals clipped to first allocation; unfinished output counted by native engine ledger')
