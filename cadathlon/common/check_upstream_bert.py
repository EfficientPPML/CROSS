"""Compare synthetic BERT semantics to Transformers without downloading weights.

Run in an environment with torch, numpy and transformers==4.51.3.
The benchmark's architecture is BertForSequenceClassification; this verifies
the same operators at small synthetic sizes, not pretrained MRPC accuracy.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_pset/p2_bert" / "src"))
from reference import encoder_block, forward, make_case


def main():
    torch.set_num_threads(1)
    for path in sorted((ROOT / "agent_pset/p2_bert" / "test").glob("test*.in")):
        case = json.loads(path.read_text())
        c = case["config"]
        weights, embeddings, mask, logits = make_case(case)
        config = BertConfig(**c)
        config._attn_implementation = "eager"
        model = BertForSequenceClassification(config).double().eval()
        model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
        with torch.no_grad():
            actual = model(input_ids=torch.tensor(case["input_ids"]),
                           token_type_ids=torch.tensor(case["token_type_ids"]),
                           attention_mask=torch.tensor(case["attention_mask"]),
                           output_hidden_states=True)
        comparisons = {
            "embeddings": (embeddings, actual.hidden_states[0].numpy()),
            "first_block": (encoder_block(c, weights, embeddings, mask, 0),
                            actual.hidden_states[1].numpy()),
            "encoder": (forward(c, weights, embeddings, mask, output_hidden=True),
                        actual.hidden_states[-1].numpy()),
            "logits": (logits, actual.logits.numpy()),
        }
        errors = {}
        for name, (expected, result) in comparisons.items():
            np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-10)
            errors[name] = float(np.max(np.abs(result - expected)))
        # Masked key positions must not influence CLS, even across layers.
        perturbed = embeddings.copy()
        perturbed[~mask] += 100 * np.random.default_rng(18).normal(size=perturbed[~mask].shape)
        np.testing.assert_allclose(forward(c, weights, perturbed, mask), logits, atol=1e-10, rtol=1e-10)
        print(json.dumps({"case": path.name, "transformers": transformers.__version__,
                          "max_abs_error": errors, "masked_padding_invariant": True,
                          "pretrained_accuracy_evaluated": False, "passed": True}))


if __name__ == "__main__":
    main()
