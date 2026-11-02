"""File adapters for the unmodified upstream benchmark's seven subprocesses.

Path dictionaries document ownership; they are not a filesystem sandbox.
The judge must deny server processes access to client inputs and secret keys.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from benchmark import BATCH_SIZES, INSTANCE_NAMES, MODEL_ID, MODEL_REVISION, STAGES


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ordered_logits(inputs, decrypted):
    """Restore input order, reject duplicate/missing IDs and invalid logits."""
    ids = [r["target_idx"] for r in inputs]
    output_ids = [r["target_idx"] for r in decrypted]
    if any(type(i) is not int for i in ids + output_ids):
        raise ValueError("target_idx must be an integer")
    if len(set(ids)) != len(ids) or len(set(output_ids)) != len(output_ids):
        raise ValueError("Duplicate sample identity")
    if set(ids) != set(output_ids):
        raise ValueError("Missing or extra decrypted samples")
    lookup = {r["target_idx"]: r["he_logits"] for r in decrypted}
    logits = np.asarray([lookup[i] for i in ids], dtype=np.float64)
    if logits.shape != (len(inputs), 2) or not np.isfinite(logits).all():
        raise ValueError("Expected two finite logits for every input")
    return logits


def stage_paths(root, size):
    io = root / "io" / INSTANCE_NAMES[size]
    intermediate = io / "intermediate"
    return {"size": size, "batch_size": BATCH_SIZES[size],
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "public_keys": io / "public_keys", "client_private": io / "client_private",
            "server_data": root / "io" / "server_data",
            "test_input": root / "datasets" / INSTANCE_NAMES[size] / "intermediate" / "test_input.txt",
            "preprocessed_input": intermediate / "client_preprocessed_input",
            "upload": io / "ciphertexts_upload", "download": io / "ciphertexts_download",
            "decrypted_results": intermediate / "decrypted_results.jsonl",
            "predictions": io / "encrypted_model_predictions.txt",
            "server_report": io / "server_reported_steps.json"}


def preprocess(paths):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    records = read_jsonl(paths["test_input"])
    if len(records) != paths["batch_size"]:
        raise ValueError("Wrong benchmark input count")
    ids = [r["target_idx"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate input identity")
    output = []
    for record in records:
        encoded = tokenizer(record["sentence1"], record["sentence2"],
                            max_length=128, padding="max_length", truncation=True)
        output.append({"target_idx": record["target_idx"], **{
            name: encoded[name] for name in ("input_ids", "token_type_ids", "attention_mask")}})
    paths["preprocessed_input"].parent.mkdir(parents=True, exist_ok=True)
    paths["preprocessed_input"].write_text("".join(json.dumps(r) + "\n" for r in output))


def postprocess(paths):
    inputs = read_jsonl(paths["preprocessed_input"])
    if len(inputs) != paths["batch_size"]:
        raise ValueError("Wrong benchmark input count")
    logits = ordered_logits(inputs, read_jsonl(paths["decrypted_results"]))
    # Match the harness's binary decision: equality selects label 1.
    predictions = (logits[:, 1] >= logits[:, 0]).astype(int)
    paths["predictions"].write_text("".join(f"{p}\n" for p in predictions))


def main(stage):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    parser = argparse.ArgumentParser(description=f"CROSS BERT: {stage}")
    parser.add_argument("size", type=int, choices=range(4))
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "harness" / "run_submission.py").is_file():
        parser.error("Run from the staged BERT benchmark directory")
    paths = stage_paths(root, args.size)
    if stage == "client_preprocess_input":
        return preprocess(paths)
    if stage == "client_postprocess":
        return postprocess(paths)
    cross = os.environ.get("CROSS_ROOT")
    if not cross or not (Path(cross) / "jaxite_word" / "API_REFERENCE.md").is_file():
        parser.error("Set CROSS_ROOT to the CROSS checkout containing your patch")
    sys.path.insert(0, str(Path(cross).resolve()))
    import bert_he
    if stage.startswith("server_"):
        permitted = {"size", "batch_size", "model_id", "model_revision", "public_keys",
                     "server_data", "upload", "download", "server_report"}
        paths = {k: v for k, v in paths.items() if k in permitted}
    return getattr(bert_he, stage)(paths)
