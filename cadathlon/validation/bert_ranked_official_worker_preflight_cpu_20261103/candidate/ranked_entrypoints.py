"""Seven literal official entrypoint bodies, using the complete file backend.

The staged runner must call these in separate subprocesses for official timing.
The proof collector intentionally executes combined3/5 workers instead and may
only label those timings diagnostic. Authentication lives in the producer
workers/gates, not in a substituted plaintext implementation.
"""
import ranked_execution
NAMES=('client_key_generation','server_preprocess_model','client_preprocess_input',
       'client_encode_encrypt_input','server_encrypted_compute','client_decrypt_decode','client_postprocess')


def resolve(stage):
    if type(stage)is not str or stage not in NAMES:raise ValueError('one exact official stage name required')
    if stage=='client_preprocess_input':return ranked_execution.adapter().preprocess
    if stage=='client_postprocess':return ranked_execution.adapter().postprocess
    import file_stages
    return getattr(file_stages,stage)


def invoke(stage,paths):
    # Source-compatible original path filtering: server never receives private
    #input, secret, decoded, or prediction paths.
    handler=resolve(stage)
    if stage.startswith('server_'):
        paths={k:v for k,v in paths.items() if k in ranked_execution.SERVER_FIELDS}
    return handler(paths)
