# Sources and decisions

Initially inspected on 2026-09-19; BERT sources added on 2026-09-21.

| Source | What the problem set uses |
|---|---|
| `/home/jianming_gatech/CADathon_Template/Example 1/README` and `doc/p2-physical.html` | Package layout, TODO files, goldens, six or more cases, runtime limits, five-section handout |
| `/home/jianming_gatech/CADathon_Template/Example_2_2019_p6/doc/p6.html` | Neural-network optimization framing, runnable starter and explicit scoring |
| `/home/jianming_gatech/CROSS_paper.pdf`, Tong et al., *Leveraging ASIC AI Chips for Homomorphic Encryption*, HPCA 2026; [online paper](https://arxiv.org/abs/2501.07047) | CKKS accelerator motivation, BAT and MAT; §§III–IV and evaluation in §V. Historical kernel/ML results are not contest baselines. |
| `/home/jianming_gatech/CROSS_Tutorial/{beginner,cross,tpu,ntt,open_challenge}.html` and `README.md`; [tutorial](https://efficientppml.github.io/CROSS_Tutorial/) | HE background, profiling, NTT/BConv, scheduling and layout optimization |
| `jaxite_word/API_REFERENCE.md`, `nn.py`, `packing.py`, `mapping.py`, `he_params.py` | Canonical rank-5 Polynomial, secure ring derivation, lazy constants, frontend limitations, bootstrap barriers |
| `demos/{canonical_demo,encrypted_demos,lenet_train,alexnet_infer,prepare_demo_assets}.py` | Model definitions, weight loading, AlexNet client pooling, baseline implementation |
| [Official Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B) and [pinned config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) | Model identity, tensor dimensions, RMSNorm/RoPE settings, tied embeddings, tokenizer mode |
| [FHE BERT benchmark, pinned commit](https://github.com/fhe-benchmarking/BERT/tree/170bfe567545d74d0fad785a052518351d37bc93) | Unmodified harness and seven submission stages; MRPC counts 1/100/200/408; 75% quality threshold; outer stage timings; output-check and repeated-seed limitations |
| [Pinned BERT checkpoint](https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/tree/f150c1d609d1e50dd5e2e5408661cfac8339277c) | Cased tokenizer and BERT-Base dimensions, LayerNorm epsilon and model identity; upstream explicitly loads BertForSequenceClassification |
| [Transformers 4.51.3 BERT implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/bert/modeling_bert.py) | Independent synthetic validation of embeddings, bidirectional attention, post-LayerNorm blocks, pooler and classifier |

The Qwen repository revision is pinned in `agent_pset/p4_qwen/test/model_source.json`.
The bundled config is unmodified; model weights/tokenizer files must be staged
from that same revision by organizers. Do not resolve `main` again on contest day.

The NumPy decoder oracle was checked against the
[Transformers 4.51.3 Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py),
including Q/K per-head normalization, half-vector RoPE rotation and KV-head
repetition. This is a semantics check on the small public cases, not a
full-checkpoint or encrypted-inference result.

The full Qwen target is beyond CROSS's current automatic CNN frontend. In
particular, replacing its SiLU with the frontend's default square is not an
equivalent Qwen implementation. The assignment allows calibrated polynomial
approximations, with explicit quality gates against the original checkpoint.
It requires encrypted attention and encrypted depth refresh as necessary.

BERT's upstream DESILO implementation informs the allowed client embedding
boundary and public attention-mask leakage, but is not copied as the CROSS
backend. `agent_pset/p2_bert/src/benchmark.py` pins all eleven harness Python files by
SHA-256. Staging retains their Apache license and leaves their contents
unchanged. The upstream `run_submission.py` sets no model revision itself;
organizers must provision a pinned offline cache for its cleartext reference.
The BERT NumPy oracle uses synthetic weights and is explicitly plaintext.
Its six cases were independently compared with Transformers; they do not
establish trained MRPC accuracy or encrypted BERT feasibility.

Security target, workload sizes, tolerances, time limits, point allocations,
speedup caps, AI-use policy and contest duration are design choices in this
problem set, not claims made by the paper or measured facts. Organizers must
validate and freeze these choices before release as described in ORGANIZER.md.
