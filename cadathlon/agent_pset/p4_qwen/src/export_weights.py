"""Export a LOCAL pinned Qwen safetensors snapshot to NumPy weight files.

No network access, no training, no executable checkpoint deserialization.
An output directory is used instead of one npz to allow memory-mapped arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    from safetensors import safe_open
    source = json.loads((Path(__file__).resolve().parents[1] / "test" / "model_source.json").read_text())
    config_bytes = (args.snapshot / "config.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != source["config_sha256"]:
        raise ValueError("Snapshot config does not match pinned config")
    shards = sorted(args.snapshot.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("No local safetensors weights")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"model_source": source, "tensors": {}, "shards_sha256": {}}
    for shard in shards:
        with shard.open("rb") as stream:
            manifest["shards_sha256"][shard.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in manifest["tensors"]:
                    raise ValueError(f"Duplicate tensor {name}")
                array = handle.get_tensor(name).float().numpy()
                filename = name + ".npy"
                np.save(args.output / filename, array, allow_pickle=False)
                manifest["tensors"][name] = {"file": filename, "shape": list(array.shape)}
    (args.output / "config.json").write_bytes(config_bytes)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(manifest['tensors'])} tensors; verify shard hashes against organizer manifest.")


if __name__ == "__main__":
    main()
