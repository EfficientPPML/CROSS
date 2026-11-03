"""Offline plaintext domain audit of the pinned full BERT checkpoint.

This is neither MRPC accuracy evaluation nor encrypted inference. Three fixed
public example pairs test the full reference's semantics and whether the toy
polynomial intervals cover its observed intermediates. No weights download,
executable checkpoint deserialization, approximation fitting or clipping occurs.
Install transformers==4.51.3 and safetensors in a separate environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from support import load_module

EXPECTED_ASSETS = {
    "config.json": "ec2b6cd6ce741dda6c972aea96f9d1a449cbb4b5349e688c5370a9b2f6872d30",
    "model.safetensors": "e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034",
    "tokenizer.json": "5b3360be30cdf39081153cf9d30bfb0503c011462fc9f77f097127c83ef0f4d9",
    "tokenizer_config.json": "0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8",
    "vocab.txt": "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02",
}
EXAMPLES = (
    ("The committee approved the proposal.", "The committee approved the proposal."),
    ("A person is playing a guitar.", "Someone is performing music."),
    ("The train arrived at noon.", "The weather will be cold tomorrow."),
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hashes = {name: digest(args.snapshot / name) for name in EXPECTED_ASSETS}
    if hashes != EXPECTED_ASSETS:
        raise ValueError("Local assets do not match the pinned full BERT snapshot")

    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification

    torch.set_num_threads(4)
    benchmark = load_module(ROOT / "agent_pset/p2_bert/src/benchmark.py")
    reference = load_module(ROOT / "agent_pset/p2_bert/src/reference.py")
    planning = load_module(ROOT / "validation/plan_transformer_polynomials.py")
    sources = (Path(__file__), ROOT / "agent_pset/p2_bert/src/benchmark.py",
               ROOT / "agent_pset/p2_bert/src/reference.py",
               ROOT / "validation/plan_transformer_polynomials.py",
               ROOT / "common/encrypted_polynomial.py", ROOT / "common/encrypted_transformer.py")
    before = {str(path.relative_to(ROOT.parent)): digest(path) for path in sources}
    started = time.perf_counter()
    model, loading = BertForSequenceClassification.from_pretrained(
        args.snapshot, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float64, attn_implementation="eager", output_loading_info=True)
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Incomplete checkpoint load: {loading}")
    if set(loading["unexpected_keys"]) - {"bert.embeddings.position_ids"}:
        raise ValueError(f"Unexpected checkpoint tensors: {loading['unexpected_keys']}")
    model.eval()
    config = model.config.to_dict()
    if (config["num_hidden_layers"], config["hidden_size"], model.num_labels) != (12, 768, 2):
        raise ValueError("Full pretrained BERT architecture differs from the benchmark")
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokens = tokenizer([a for a, _ in EXAMPLES], [b for _, b in EXAMPLES],
                       max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        expected = model(**tokens).logits.numpy()
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    embeddings = reference.embed(config, weights, tokens["input_ids"].numpy(),
                                 tokens["token_type_ids"].numpy())
    plan = planning.Plan("bert", exact=True)
    actual = planning.bert_forward(config, weights, embeddings,
                                   tokens["attention_mask"].numpy().astype(bool), plan)
    error = float(np.max(np.abs(actual - expected)))
    semantic_match = bool(np.allclose(actual, expected, atol=1e-8, rtol=1e-8))
    after = {str(path.relative_to(ROOT.parent)): digest(path) for path in sources}
    report = {
        "scope": "Plaintext full-checkpoint semantics and empirical domain audit on three public examples; no MRPC accuracy or HE execution.",
        "model_id": benchmark.MODEL_ID, "revision": benchmark.MODEL_REVISION,
        "asset_sha256": hashes, "examples": EXAMPLES, "layers": 12, "sequence_length": 128,
        "weights_loaded_without_missing_or_mismatched_tensors": True,
        "loading_info": loading, "semantic_match": semantic_match,
        "max_abs_logit_error_vs_transformers": error,
        "exact_logits": actual.tolist(), "toy_contracts": plan.contracts,
        "observed_ranges": plan.ranges, "observed_toy_domain_violations": plan.violations,
        "toy_domains_cover_observed_intermediates": not plan.violations,
        "approximate_full_forward_evaluated": False,
        "limitation": "Observed ranges are examples, not certified bounds. Toy-polynomial gates cannot establish pretrained BERT readiness.",
        "elapsed_s": time.perf_counter() - started,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "torch", "transformers", "safetensors", "tokenizers")},
        "source_hashes_before": before, "source_hashes_after": after,
        "source_files_unchanged": before == after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in
                     ("semantic_match", "max_abs_logit_error_vs_transformers",
                      "toy_domains_cover_observed_intermediates", "observed_toy_domain_violations",
                      "source_files_unchanged", "elapsed_s")}, indent=2))
    return 0 if semantic_match and before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
