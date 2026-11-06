"""Exact original toy_weights definition; NumPy import occurs only after setup admission."""
import numpy as np

def toy_weights(c, seed):
    rng = np.random.default_rng(seed)
    h, m, v = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    q, kv, d = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"], c["head_dim"]
    w = {"model.embed_tokens.weight": rng.normal(0, 0.15, (v, h)),
         "model.norm.weight": rng.uniform(0.9, 1.1, h)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            w[p + name + ".weight"] = rng.uniform(0.9, 1.1, h)
        for name, shape in (("q_proj", (q, h)), ("k_proj", (kv, h)),
                            ("v_proj", (kv, h)), ("o_proj", (h, q))):
            w[p + "self_attn." + name + ".weight"] = rng.normal(0, 0.08, shape)
        for name in ("q_norm", "k_norm"):
            w[p + "self_attn." + name + ".weight"] = rng.uniform(0.9, 1.1, d)
        for name, shape in (("gate_proj", (m, h)), ("up_proj", (m, h)), ("down_proj", (h, m))):
            w[p + "mlp." + name + ".weight"] = rng.normal(0, 0.08, shape)
    return w
