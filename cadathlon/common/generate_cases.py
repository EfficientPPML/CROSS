"""Organizer tool: explicitly regenerate the 24 public numerical fixtures."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(problem):
    spec = importlib.util.spec_from_file_location(problem + "_ref", ROOT / problem / "src" / "reference.py")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def save(problem, i, case, logits):
    folder = ROOT / problem / "test"
    (folder / f"test{i}.in").write_text(json.dumps(case, indent=2) + "\n")
    (folder / f"test{i}.out").write_text(json.dumps({"kind": "plaintext-numerical-reference", "logits": logits.tolist()}, indent=2) + "\n")


qwen = module("agent_pset/p4_qwen")
for i, (layers, length, batch) in enumerate(((1, 1, 1), (1, 2, 1), (2, 4, 1), (2, 8, 1), (3, 4, 2), (3, 16, 1)), 1):
    cfg = dict(hidden_size=16, intermediate_size=24, num_attention_heads=4,
               num_key_value_heads=2, head_dim=8, num_hidden_layers=layers,
               vocab_size=32, rms_norm_eps=1e-6, rope_theta=1000000,
               tie_word_embeddings=True, attention_bias=False, hidden_act="silu")
    case = {"purpose": "toy-decoder-semantics-not-full-Qwen", "config": cfg,
            "weight_seed": 9100 + i,
            "token_ids": [[(j * 7 + b * 3 + i) % 32 for j in range(length)] for b in range(batch)]}
    _, _, logits = qwen.make_case(case)
    save("agent_pset/p4_qwen", i, case, logits)

cnn = module("agent_pset/p3_cnn")
for i, (name, batch) in enumerate((("lenet", 1), ("lenet", 2), ("lenet", 4),
                                  ("alexnet_tiny", 1), ("alexnet_tiny", 2), ("alexnet_tiny", 4)), 1):
    case = {"purpose": "synthetic-arithmetic-not-trained-accuracy", "model": name,
            "batch": batch, "weight_seed": 7200 + i, "input_seed": 8300 + i,
            "input_kind": "zeros" if i == 2 else "alternating" if i == 5 else "uniform"}
    _, _, _, logits = cnn.make_case(case)
    save("agent_pset/p3_cnn", i, case, logits)
print("Generated six numerical input/output pairs per problem.")

bench = module("agent_pset/p1_benchmark")
for i, (batch, devices, kind) in enumerate(((1, 1, "uniform"), (2, 1, "zeros"),
                                            (4, 1, "alternating"), (4, 2, "distinct_lanes"),
                                            (8, 2, "uniform"), (8, 4, "distinct_lanes")), 1):
    case = {"purpose": "synthetic-arithmetic-not-trained-accuracy",
            "dataset": "mnist", "model": "mlp_784_50_10",
            "batch": batch, "devices": devices, "hidden": 50,
            "weight_seed": 7300 + i, "input_seed": 8400 + i, "input_kind": kind}
    _, _, _, logits = bench.make_case(case)
    save("agent_pset/p1_benchmark", i, case, logits)

from generate_bert_cases import main as generate_bert_cases
generate_bert_cases()
