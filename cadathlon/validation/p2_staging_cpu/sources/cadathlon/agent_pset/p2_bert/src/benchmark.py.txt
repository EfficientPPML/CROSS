"""Pinned upstream identity and file contract, verified by stage_benchmark.py."""
UPSTREAM_URL = "https://github.com/fhe-benchmarking/BERT"
UPSTREAM_COMMIT = "170bfe567545d74d0fad785a052518351d37bc93"
MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"
MODEL_REVISION = "f150c1d609d1e50dd5e2e5408661cfac8339277c"
INSTANCE_NAMES = ("single", "small", "medium", "large")
BATCH_SIZES = (1, 100, 200, 408)
STAGES = ('client_key_generation', 'server_preprocess_model', 'client_preprocess_input', 'client_encode_encrypt_input', 'server_encrypted_compute', 'client_decrypt_decode', 'client_postprocess')
HARNESS_SHA256 = {
    "harness/cleartext_impl.py": "595c8128e4f44d16d10c358d63e3503cb3d13175ffcd49379457d12333672247",
    "harness/generate_dataset.py": "4dc6d35336f8d609b9676752fd0a041a40b19c1dcde822b989e44e5e20a30d6d",
    "harness/generate_input.py": "634804f0235ca611de32e535d4e55139eda109d7620c20129586b68efc73cfe1",
    "harness/mrpc/__init__.py": "f53192db12fbcbb8760d16fc7bbdbd6a7744c94713f4594cd6674bbfb3068f65",
    "harness/mrpc/model.py": "784685ccd7f09e5267c604ede7cbe21eb5e1d7a4df8eef6bfb77a6bce74b2fb4",
    "harness/mrpc/mrpc.py": "60115e739e420b006c0baf433be0c61ee88fb26fa5c1e3edc038e2ff2d3e4321",
    "harness/mrpc/test.py": "13ed5c720cada62aaf73d88b5b7afb9de4e6fcd22ed96b0f43324ec8d0e33a24",
    "harness/params.py": "e1e12436a385e1ca6f59f3c7b4ec0f4a4d3dc38c6ab9669125a66b4df6b9cd3b",
    "harness/run_submission.py": "997fcf575999c6c0a35fa0561dd1a90d3f5565303bd1b6800c078511139f4590",
    "harness/utils.py": "e73b7ba164e18da00b57a53f8245a1df29cbfc3fc5424a7e734c0a4c7e2dabae",
    "harness/verify_result.py": "878296a9a91d7f011b8f5303c16feef5988ff390c9431e50b05c6e5f1aa3d3f7"
}
