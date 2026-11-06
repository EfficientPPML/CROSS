"""Instrumented exact float64 circuit; compared bitwise to unchanged reference."""
import numpy as np


def rmsnorm(x, weight, eps):
    return x * (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5 * weight


def rope(x, theta):
    # x: B, heads, S, D. Qwen rotates the two halves, not adjacent pairs.
    d = x.shape[-1]
    inv_freq = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angles = np.arange(x.shape[-2])[:, None] * inv_freq[None, :]
    angles = np.concatenate((angles, angles), axis=-1)[None, None, :, :]
    rotated = np.concatenate((-x[..., d // 2:], x[..., :d // 2]), axis=-1)
    return x * np.cos(angles) + rotated * np.sin(angles)


def forward(config, weights, embeddings, *, observe, start_layer=0, output_hidden=False):
    c, w = config, weights
    x = np.asarray(embeddings, dtype=np.float64)
    b, s, h = x.shape
    nq, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    if h != c["hidden_size"] or nq % nkv or d % 2:
        raise ValueError("Invalid Qwen dimensions")
    eps = c["rms_norm_eps"]
    def normal(value, weight, label):
        observe(label, np.mean(value * value, axis=-1, keepdims=True) + eps)
        result = rmsnorm(value, weight, eps)
        observe(label + '.exact_output', result)
        return result
    for layer in range(start_layer, start_layer + c["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        a = prefix + "self_attn."
        z = normal(x, w[prefix + "input_layernorm.weight"], f"{layer}.input.inverse_sqrt")
        q = (z @ w[a + "q_proj.weight"].T).reshape(b, s, nq, d).transpose(0, 2, 1, 3)
        k = (z @ w[a + "k_proj.weight"].T).reshape(b, s, nkv, d).transpose(0, 2, 1, 3)
        v = (z @ w[a + "v_proj.weight"].T).reshape(b, s, nkv, d).transpose(0, 2, 1, 3)
        q = rope(normal(q, w[a + "q_norm.weight"], f"{layer}.q.inverse_sqrt"), c["rope_theta"])
        k = rope(normal(k, w[a + "k_norm.weight"], f"{layer}.k.inverse_sqrt"), c["rope_theta"])
        observe(f"{layer}.q_rope.exact_output", q)
        observe(f"{layer}.k_rope.exact_output", k)
        k, v = np.repeat(k, nq // nkv, axis=1), np.repeat(v, nq // nkv, axis=1)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)
        for query in range(s):
            valid = scores[..., query, :query+1]
            observe(f"{layer}.softmax.raw_valid", valid)
            observe(f"{layer}.softmax.centered", valid - valid.mean(axis=-1, keepdims=True))
        mask = np.tril(np.ones((s, s), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)
        attended = (probs @ v).transpose(0, 2, 1, 3).reshape(b, s, nq * d)
        x = x + attended @ w[a + "o_proj.weight"].T
        z = normal(x, w[prefix + "post_attention_layernorm.weight"], f"{layer}.post.inverse_sqrt")
        m = prefix + "mlp."
        gate = z @ w[m + "gate_proj.weight"].T
        observe(f"{layer}.silu", gate)
        # Numerically stable sigmoid without overflow on negative inputs.
        sigmoid = np.exp(-np.logaddexp(0, -gate))
        mlp = (gate * sigmoid) * (z @ w[m + "up_proj.weight"].T)
        x = x + mlp @ w[m + "down_proj.weight"].T
    if output_hidden:
        return x
    last = normal(x[:, -1, :], w["model.norm.weight"], "final.inverse_sqrt")
    return last @ w["model.embed_tokens.weight"].T

