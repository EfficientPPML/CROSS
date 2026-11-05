"""Measure active TPU program time from device-side XLA module intervals.

The union of intervals avoids double counting nested modules and excludes
host work and idle gaps between module launches. Never substitutes wall time.
"""
import gzip
import json
from pathlib import Path
import re


def summarize_trace(path):
    with gzip.open(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    if len(events) >= 1_000_000:
        raise RuntimeError('Viewer trace may be event-capped; use the raw XPlane file')
    processes, threads = {}, {}
    for event in events:
        if event.get('ph') != 'M':
            continue
        name = event.get('args', {}).get('name', '')
        if event.get('name') == 'process_name':
            processes[event['pid']] = name
        elif event.get('name') == 'thread_name':
            threads[event['pid'], event['tid']] = name
    intervals = {}
    for event in events:
        pid, tid = event.get('pid'), event.get('tid')
        if event.get('ph') != 'X' or not re.search(r'TPU', processes.get(pid, ''), re.I):
            continue
        if 'XLA Modules' not in threads.get((pid, tid), ''):
            continue
        start, duration = event.get('ts'), event.get('dur')
        if start is not None and duration is not None and duration > 0:
            intervals.setdefault(pid, []).append((start, start + duration))
    if not intervals:
        raise RuntimeError(f'No TPU XLA Modules intervals in {path}; inspect trace metadata instead of using wall time')
    active_ms, counts = {}, {}
    for pid, spans in intervals.items():
        spans.sort()
        start, end = spans[0]
        elapsed = 0
        for next_start, next_end in spans[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                elapsed += end - start
                start, end = next_start, next_end
        elapsed += end - start
        name = processes[pid]
        active_ms[name] = elapsed / 1000  # Chrome trace timestamps are us.
        counts[name] = len(spans)
    return dict(device_active_ms=max(active_ms.values()),
                per_device_active_ms=active_ms, module_events=counts,
                trace=str(path), metric='union of TPU XLA Modules durations; host work and inter-module idle gaps excluded')


def capture(call, directory, checkpoint_owner=None):
    import jax
    import fcntl
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f'Use a fresh trace directory: {directory}')
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    options.advanced_configuration = {'tpu_trace_mode': 'TRACE_ONLY_XLA'}
    # Compatibility with the already-running benchmark driver, which imports
    # this module only after its first full evaluation and correctness checks.
    if checkpoint_owner is None:
        checkpoint_owner = getattr(call, '__globals__', {}).get('bs')
    original_checkpoint = getattr(checkpoint_owner, '_stage_checkpoint', None)
    segment = 0
    active = False

    def start():
        nonlocal active
        jax.profiler.start_trace(str(directory / f'segment-{segment}'),
                                 profiler_options=options)
        active = True

    def stop():
        nonlocal active
        if active:
            jax.profiler.stop_trace()
            active = False

    def checkpoint(tag, ct=None):
        nonlocal segment
        original_checkpoint(tag, ct)
        if tag in {'prescale', 'C2S', 'approxmod-im', 'S2C'}:
            # Drain all live outputs before changing trace buffers. No device
            # arithmetic is added; export and the restart gap are host work.
            jax.block_until_ready(jax.live_arrays())
            stop()
            segment += 1
            start()
    # Independent chip processes may prepare concurrently, but serialize trace
    # collection because the TPU profiling service is shared by the host.
    with open('/tmp/bootstrap-comparison-profiler.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if original_checkpoint:
            checkpoint_owner._stage_checkpoint = checkpoint
        try:
            start()
            output = call()
            payload = output.polynomial if hasattr(output, 'polynomial') else output
            jax.block_until_ready(payload)
        finally:
            stop()
            if original_checkpoint:
                checkpoint_owner._stage_checkpoint = original_checkpoint
    traces = list(directory.rglob('*.xplane.pb'))
    if len(traces) != segment + 1:
        raise RuntimeError(f'Expected {segment + 1} raw traces, got {len(traces)}')
    from raw_xplane import summarize
    profiles = [summarize(path) for path in sorted(traces)]
    devices = {device for profile in profiles for device in profile['per_device_active_ms']}
    if len(devices) != 1:
        raise RuntimeError(f'Expected one device, got {devices}')
    device = next(iter(devices))
    total = sum(profile['device_active_ms'] for profile in profiles)
    result = dict(device_active_ms=total, per_device_active_ms={device: total},
                  module_events={device: sum(p['module_events'][device] for p in profiles)},
                  trace=str(directory), trace_source='raw XPlane; no viewer event cap',
                  segments=profiles,
                  metric='union of TPU XLA Modules durations, summed over consecutive drained segments; host work and idle gaps excluded')
    (directory / 'device-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
