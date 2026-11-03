"""Plaintext BERT oracle for synthetic fixtures; never an encrypted backend.

Parameter names match BertForSequenceClassification. Dropout is disabled.
The client computes BERT embeddings (including embedding LayerNorm); the
server target is the encoder, pooler and two-logit classifier.
"""
import math

import numpy as np


def linear(x, weights, prefix):
    return x @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]


def layer_norm(x, weights, prefix, eps):
    centered = x - x.mean(axis=-1, keepdims=True)
    return (centered / np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + eps)
            * weights[prefix + ".weight"] + weights[prefix + ".bias"])


def gelu(x):
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def embed(config, weights, input_ids, token_type_ids):
    p = "bert.embeddings."
    positions = np.arange(input_ids.shape[1])
    x = (weights[p + "word_embeddings.weight"][input_ids]
         + weights[p + "position_embeddings.weight"][positions][None, :, :]
         + weights[p + "token_type_embeddings.weight"][token_type_ids])
    return layer_norm(x, weights, p + "LayerNorm", config["layer_norm_eps"])


def encoder_block(config, weights, x, attention_mask, layer_index):
    b, s, h = x.shape
    heads = config["num_attention_heads"]
    if h % heads or attention_mask.shape != (b, s) or not np.all(attention_mask.any(axis=1)):
        raise ValueError("Invalid BERT shape or fully masked input")
    p = f"bert.encoder.layer.{layer_index}."
    q, k, v = [linear(x, weights, p + "attention.self." + name)
               .reshape(b, s, heads, h // heads).transpose(0, 2, 1, 3)
               for name in ("query", "key", "value")]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(h // heads)
    # Bidirectional attention; mask keys, not query rows. This is not causal.
    scores = np.where(attention_mask[:, None, None, :], scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    context = (probabilities @ v).transpose(0, 2, 1, 3).reshape(b, s, h)
    x = layer_norm(x + linear(context, weights, p + "attention.output.dense"),
                   weights, p + "attention.output.LayerNorm", config["layer_norm_eps"])
    intermediate = gelu(linear(x, weights, p + "intermediate.dense"))
    return layer_norm(x + linear(intermediate, weights, p + "output.dense"),
                      weights, p + "output.LayerNorm", config["layer_norm_eps"])


def forward(config, weights, embeddings, attention_mask, *, output_hidden=False):
    x = np.asarray(embeddings, dtype=np.float64)
    mask = np.asarray(attention_mask, dtype=bool)
    for i in range(config["num_hidden_layers"]):
        x = encoder_block(config, weights, x, mask, i)
    if output_hidden:
        return x
    pooled = np.tanh(linear(x[:, 0, :], weights, "bert.pooler.dense"))
    return linear(pooled, weights, "classifier")


def toy_weights(config, seed):
    rng = np.random.default_rng(seed)
    h, m = config["hidden_size"], config["intermediate_size"]
    weights = {}

    def dense(prefix, rows, cols):
        weights[prefix + ".weight"] = rng.normal(0, 0.12, (rows, cols))
        weights[prefix + ".bias"] = rng.normal(0, 0.03, rows)

    def norm(prefix):
        weights[prefix + ".weight"] = rng.uniform(0.9, 1.1, h)
        weights[prefix + ".bias"] = rng.normal(0, 0.02, h)

    for name, count in (("word", config["vocab_size"]),
                        ("position", config["max_position_embeddings"]),
                        ("token_type", config["type_vocab_size"])):
        weights[f"bert.embeddings.{name}_embeddings.weight"] = rng.normal(0, 0.15, (count, h))
    norm("bert.embeddings.LayerNorm")
    for i in range(config["num_hidden_layers"]):
        p = f"bert.encoder.layer.{i}."
        for name in ("query", "key", "value"):
            dense(p + "attention.self." + name, h, h)
        dense(p + "attention.output.dense", h, h)
        norm(p + "attention.output.LayerNorm")
        dense(p + "intermediate.dense", m, h)
        dense(p + "output.dense", h, m)
        norm(p + "output.LayerNorm")
    dense("bert.pooler.dense", h, h)
    dense("classifier", 2, h)
    return weights


def make_case(case):
    c = case["config"]
    weights = toy_weights(c, case["weight_seed"])
    ids = np.asarray(case["input_ids"], dtype=np.int64)
    types = np.asarray(case["token_type_ids"], dtype=np.int64)
    mask = np.asarray(case["attention_mask"], dtype=bool)
    embeddings = embed(c, weights, ids, types)
    return weights, embeddings, mask, forward(c, weights, embeddings, mask)
