"""Read complete XPlane timelines without the trace viewer's event cap.

Wire fields follow OpenXLA's tsl/profiler/protobuf/xplane.proto. Unknown
fields are skipped; mmap avoids materializing large host event timelines.
"""
import json
import mmap
from pathlib import Path


def varint(buf, pos):
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError('Invalid protobuf varint')


def fields(buf, start, end):
    pos = start
    while pos < end:
        tag, pos = varint(buf, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 2:
            size, pos = varint(buf, pos)
            value = (pos, pos + size)
            pos += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = (pos, pos + size)
            pos += size
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        if pos > end:
            raise ValueError('Truncated protobuf')
        yield number, wire, value


def string(buf, span):
    return buf[span[0]:span[1]].decode()


def summarize(path):
    intervals, line_info, warnings = {}, {}, []
    with open(path, 'rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        for field, wire, span in fields(buf, 0, len(buf)):
            if field in (2, 3):
                warnings.append(string(buf, span))
            if field != 1 or wire != 2:
                continue
            plane_fields = list(fields(buf, *span))
            name = next((string(buf, v) for f, w, v in plane_fields if f == 2), '')
            if 'TPU' not in name:
                continue
            metadata_names = {}
            for f, w, entry_span in plane_fields:
                if f == 4:
                    entry = {f: v for f, w, v in fields(buf, *entry_span)}
                    metadata = {f: v for f, w, v in fields(buf, *entry[2])}
                    metadata_names[entry[1]] = string(buf, metadata[2])
            spans = intervals.setdefault(name, [])
            lines = line_info.setdefault(name, [])
            for f, w, line_span in plane_fields:
                if f != 3:
                    continue
                line_fields = list(fields(buf, *line_span))
                line_name = next((string(buf, v) for f, w, v in line_fields if f == 2), '')
                timestamp = next((v for f, w, v in line_fields if f == 3), 0)
                event_fields = [v for f, w, v in line_fields if f == 4]
                lines.append(dict(name=line_name, events=len(event_fields)))
                if 'TraceMe' in line_name:
                    for event_span in event_fields:
                        event = {f: v for f, w, v in fields(buf, *event_span) if w == 0}
                        label = metadata_names.get(event.get(1), '')
                        if 'dropped' in label.lower() or 'overflow' in label.lower():
                            raise ValueError(f'Incomplete raw device trace: {label}')
                if 'XLA Modules' not in line_name:
                    continue
                for event_span in event_fields:
                    event = {f: v for f, w, v in fields(buf, *event_span) if w == 0}
                    duration = event.get(3, 0)
                    if duration > 0:
                        offset = event.get(2, 0)
                        if offset >= 2**63:
                            offset -= 2**64
                        start = timestamp * 1000 + offset
                        spans.append((start, start + duration))
    for warning in warnings:
        if any(term in warning.lower() for term in ('dropped', 'overflow', 'truncat', 'data loss')):
            raise ValueError(f'Incomplete raw device trace: {warning}')
    intervals = {name: spans for name, spans in intervals.items() if spans}
    if not intervals:
        raise ValueError(f'No raw TPU XLA Modules events: {line_info}')
    active, counts, ranges = {}, {}, {}
    for name, spans in intervals.items():
        spans.sort()
        start, end = spans[0]
        total = 0
        for left, right in spans[1:]:
            if left <= end:
                end = max(end, right)
            else:
                total += end - start
                start, end = left, right
        total += end - start
        active[name] = total / 1e9  # picoseconds -> milliseconds
        counts[name] = len(spans)
        ranges[name] = (max(right for _, right in spans) - spans[0][0]) / 1e12
    return dict(device_active_ms=max(active.values()), per_device_active_ms=active,
                module_events=counts, device_span_seconds=ranges,
                trace=str(path), trace_source='raw XPlane; no viewer event cap',
                profiler_warnings=warnings, raw_lines=line_info,
                metric='union of TPU XLA Modules durations; host work and inter-module idle gaps excluded')


if __name__ == '__main__':
    import sys
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
