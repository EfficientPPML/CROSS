"""Inspect the actual secure packing path without materializing HE keys."""
import json
from pathlib import Path
from reference import make_case
from jaxite_word import nn, packing

for name in ("test1.in", "test4.in"):
    case = json.loads((Path(__file__).resolve().parents[1] / "test" / name).read_text())
    model, shape, _, _ = make_case(case)
    packed = packing.pack(nn.vectorize(model, shape), packing.PackingPolicy(lazy_constants=True))
    ring = packed.ring_config
    print(json.dumps({"model": case["model"], "depth": packed.depth,
                      "degree": ring.degree, "num_slots": packed.num_slots,
                      "logical_num_q": ring.logical_num_q,
                      "security_bits": ring.security_bits, "dnum": ring.dnum}))
