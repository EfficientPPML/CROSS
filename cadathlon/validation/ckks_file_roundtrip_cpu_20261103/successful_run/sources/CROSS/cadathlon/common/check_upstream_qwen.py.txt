"""Optional independent oracle check. Requires transformers==4.51.3 and torch."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_pset/p4_qwen" / "src"))
from reference import make_case, decoder_block, operator_forward

torch.set_num_threads(1)
for path in sorted((ROOT / "agent_pset/p4_qwen" / "test").glob("test*.in")):
    case = json.loads(path.read_text())
    weights, embeddings, expected = make_case(case)
    config = Qwen3Config(**case["config"], attn_implementation="eager",
                         sliding_window=None, use_sliding_window=False)
    model = Qwen3ForCausalLM(config).double().eval()
    state = {name: torch.from_numpy(value) for name, value in weights.items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        result = model(inputs_embeds=torch.from_numpy(embeddings), use_cache=False, output_hidden_states=True)
        actual = result.logits[:, -1, :].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if case["config"]["num_hidden_layers"] > 1:
        block = decoder_block(case["config"], weights, embeddings, 0)
        np.testing.assert_allclose(block, result.hidden_states[1].numpy(), rtol=1e-5, atol=1e-6)
    # Compose all five separate operator interfaces and compare to block 0.
    c = case["config"]
    b, s, h = embeddings.shape
    nq, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    prefix = "model.layers.0."
    def op(name, w, operands):
        return operator_forward(name, c, w, operands)
    z = op("rmsnorm", {"weight": weights[prefix + "input_layernorm.weight"]}, embeddings)
    projected = []
    for name, heads in (("q_proj", nq), ("k_proj", nkv), ("v_proj", nkv)):
        value = op("linear", {"weight": weights[prefix + "self_attn." + name + ".weight"]}, z)
        projected.append(value.reshape(b, s, heads, d).transpose(0, 2, 1, 3))
    q, k, v = projected
    q, k = op("qk_rope", {name: weights[prefix + "self_attn." + name] for name in ("q_norm.weight", "k_norm.weight")}, (q, k))
    attention = op("gqa", {}, (q, k, v))
    residual = embeddings + op("linear", {"weight": weights[prefix + "self_attn.o_proj.weight"]}, attention)
    z = op("rmsnorm", {"weight": weights[prefix + "post_attention_layernorm.weight"]}, residual)
    mlp_weights = {name + ".weight": weights[prefix + "mlp." + name + ".weight"] for name in ("gate_proj", "up_proj", "down_proj")}
    block = residual + op("gated_mlp", mlp_weights, z)
    np.testing.assert_allclose(block, decoder_block(c, weights, embeddings, 0), rtol=1e-12, atol=1e-12)
    print(json.dumps({"case": path.name, "max_abs_error": float(np.max(np.abs(actual - expected)))}))
