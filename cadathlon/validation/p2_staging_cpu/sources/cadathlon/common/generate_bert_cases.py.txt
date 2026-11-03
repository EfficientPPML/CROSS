"""Organizer tool: regenerate only the six synthetic BERT fixture pairs."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "agent_pset/p2_bert"
    spec = importlib.util.spec_from_file_location("bert_reference", folder / "src" / "reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cases = ((1, 4, 1, 0), (1, 8, 1, 3), (2, 6, 1, 0),
             (2, 8, 2, 2), (3, 12, 2, 4), (3, 16, 3, 1))
    (folder / "test").mkdir(exist_ok=True)
    for i, (layers, length, batch, pad) in enumerate(cases, 1):
        config = dict(hidden_size=16 if i <= 3 else 24, intermediate_size=32 if i <= 3 else 48,
                      num_attention_heads=4, num_hidden_layers=layers, vocab_size=32,
                      max_position_embeddings=16, type_vocab_size=2, layer_norm_eps=1e-12,
                      hidden_act="gelu", num_labels=2, pad_token_id=0,
                      hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        ids, types, masks = [], [], []
        for lane in range(batch):
            valid = length - pad - (lane if pad else 0)
            ids.append([1] + [(j * 7 + lane * 3 + i) % 29 + 3 for j in range(valid - 2)]
                       + [2] + [0] * (length - valid))
            types.append([int(valid // 2 <= j < valid) for j in range(length)])
            masks.append([int(j < valid) for j in range(length)])
        case = dict(purpose="synthetic-BERT-semantics-not-pretrained-MRPC-accuracy",
                    config=config, weight_seed=9500 + i, input_ids=ids,
                    token_type_ids=types, attention_mask=masks)
        _, _, _, logits = reference.make_case(case)
        expected = dict(kind="plaintext-numerical-reference", logits=logits.tolist())
        for suffix, data in (("in", case), ("out", expected)):
            (folder / "test" / f"test{i}.{suffix}").write_text(json.dumps(data, indent=2) + "\n")
    print("Generated six BERT numerical input/output pairs.")


if __name__ == "__main__":
    main()
