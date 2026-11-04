"""Inspect the actual secure packing path without materializing HE keys."""
import json
import math
from pathlib import Path
from reference import make_case
from jaxite_word import nn, packing

for name in ("test1.in", "test6.in"):
    case = json.loads((Path(__file__).resolve().parents[1] / "test" / name).read_text())
    model, shape, _, _ = make_case(case)
    packed = packing.pack(nn.vectorize(model, shape), packing.PackingPolicy(lazy_constants=True))
    ring = packed.ring_config
    used = int(math.prod(shape))
    print(json.dumps({"batch": case["batch"], "devices": case.get("devices", 1),
                      "depth": packed.depth, "degree": ring.degree,
                      "num_slots": packed.num_slots,
                      "slots_used_by_input": used,
                      "slot_occupancy": round(used / packed.num_slots, 4),
                      "logical_num_q": ring.logical_num_q,
                      "security_bits": ring.security_bits, "dnum": ring.dnum,
                      "operations": [op[1] for op in packed.operations]}))
