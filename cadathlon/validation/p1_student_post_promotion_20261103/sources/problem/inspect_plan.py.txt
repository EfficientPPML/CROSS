"""Inspect a derived secure packing plan without generating HE keys."""
import argparse
import json
import math
from pathlib import Path

from reference import make_case
from support import cross_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-baseline', action='store_true',
                        help='inspect the same smaller secure policy as baseline_student.py')
    args = parser.parse_args(argv)
    cross_root()
    from jaxite_word import nn, packing
    if args.student_baseline:
        from baseline_student import packing_policy
        policy = packing_policy()
    else:
        policy = packing.PackingPolicy(lazy_constants=True)
    for name in ('test1.in', 'test6.in'):
        case = json.loads((Path(__file__).resolve().parents[1] / 'test' / name).read_text())
        model, shape, _, _ = make_case(case)
        packed = packing.pack(nn.vectorize(model, shape), policy)
        ring = packed.ring_config
        used = int(math.prod(shape))
        print(json.dumps({
            'profile': 'student' if args.student_baseline else 'legacy',
            'case': name, 'batch': case['batch'], 'devices': case.get('devices', 1),
            'depth': packed.depth, 'degree': ring.degree, 'num_slots': packed.num_slots,
            'slots_used_by_input': used, 'slot_occupancy': round(used / packed.num_slots, 4),
            'logical_num_q': ring.logical_num_q, 'security_bits': ring.security_bits,
            'dnum': ring.dnum, 'q_towers': list(ring.q_towers), 'p_towers': list(ring.p_towers),
            'log2_QP': math.log2(math.prod((*ring.q_towers, *ring.p_towers))),
            'operations': [op[1] for op in packed.operations],
        }))


if __name__ == '__main__':
    main()
