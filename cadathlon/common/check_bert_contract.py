"""Author checks for staging and strict BERT outputs; no HE/model inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_pset/p2_bert" / "src"))
from benchmark import BATCH_SIZES, HARNESS_SHA256, STAGES
from check_result import check
from stage_benchmark import stage
from stages import postprocess, stage_paths


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def rejected(action):
    try:
        result = action()
    except (ValueError, FileExistsError):
        return
    if isinstance(result, dict) and not result["passed"]:
        return
    raise AssertionError("Invalid case was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    verdicts = []
    with tempfile.TemporaryDirectory(prefix="cross-bert-contract-") as temporary:
        destination = Path(temporary) / "staged"
        stage(args.upstream.resolve(), destination)
        for name, digest in HARNESS_SHA256.items():
            assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
        assert (destination / "LICENSE.upstream.md").is_file()
        rejected(lambda: stage(args.upstream.resolve(), destination))
        for name in STAGES:
            subprocess.run([sys.executable, str(destination / "submissions" / (name + ".py")),
                            "--help"], cwd=destination, check=True, capture_output=True)
        for size, count in enumerate(BATCH_SIZES):
            paths = stage_paths(destination, size)
            # IDs deliberately descend; decrypted records deliberately reverse.
            inputs = [{"target_idx": 10000 - i, "sentence1": "Example A", "sentence2": "Example B"}
                      for i in range(count)]
            records = [{"target_idx": r["target_idx"], "he_logits": [0.0, 0.0] if i % 2 == 0 else [1.0, 0.0]}
                       for i, r in enumerate(inputs)]
            records.reverse()
            jsonl(paths["test_input"], inputs)
            jsonl(paths["preprocessed_input"], inputs)
            jsonl(paths["decrypted_results"], records)
            expected = [1 if i % 2 == 0 else 0 for i in range(count)]
            paths["test_input"].with_name("test_labels.txt").write_text("".join(f"{v}\n" for v in expected))
            paths["predictions"].with_name("harness_model_predictions.txt").write_text(
                "".join(f"{v}\n" for v in expected))
            postprocess(paths)
            assert check(destination, size)["passed"]
            bad_variants = {
                "missing": records[:-1], "duplicate": records + [records[0]],
                "extra": records + [{"target_idx": -1, "he_logits": [0.0, 1.0]}],
            }
            for name, value in (("nonfinite", float("nan")), ("wrong_shape", [0, 1, 2])):
                altered = copy.deepcopy(records)
                altered[0]["he_logits"] = [value, 1] if name == "nonfinite" else value
                bad_variants[name] = altered
            for bad in bad_variants.values():
                jsonl(paths["decrypted_results"], bad)
                rejected(lambda: check(destination, size))
            jsonl(paths["decrypted_results"], records)
            paths["predictions"].write_text("0\n")
            rejected(lambda: check(destination, size))
            postprocess(paths)
            paths["predictions"].write_text("2\n" * count)
            rejected(lambda: check(destination, size))
            postprocess(paths)
            verdicts.append({"size": size, "samples": count, "input_order": True,
                             "tie_selects_one": True, "invalid_outputs_rejected": list(bad_variants)
                             + ["wrong_predictions", "nonbinary"]})
        # Separate the two batch accuracy gates: 97% vs 100% must fail the
        # two-point drop rule; 74% vs 74% must fail the absolute 75% threshold.
        paths = stage_paths(destination, 1)
        inputs = [{"target_idx": i} for i in range(100)]
        jsonl(paths["test_input"], inputs)
        jsonl(paths["preprocessed_input"], inputs)
        paths["test_input"].with_name("test_labels.txt").write_text("1\n" * 100)
        for correct, clear_correct, accepted in ((97, 100, False), (74, 74, False),
                                                (98, 100, True), (75, 75, True)):
            jsonl(paths["decrypted_results"], [dict(target_idx=i, he_logits=[0, 1] if i < correct else [1, 0])
                                              for i in range(100)])
            postprocess(paths)
            paths["predictions"].with_name("harness_model_predictions.txt").write_text(
                "1\n" * clear_correct + "0\n" * (100 - clear_correct))
            assert check(destination, 1)["passed"] == accepted
    print(json.dumps({"passed": True, "staged_harness_unchanged": True, "existing_destination_rejected": True,
                      "stage_cli_count": len(STAGES), "cases": verdicts, "accuracy_boundaries": True,
                      "encrypted_inference_tested": False}, indent=2))


if __name__ == "__main__":
    main()
