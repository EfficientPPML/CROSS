"""Contestant entrypoint. Preserve model semantics and the validated ring."""


def build_mapping(model, input_shape, *, batch, devices):
    """Return a prepared Mapping for the supplied model, batch and devices.

    TODO: begin with baseline.py, then optimize CROSS packing or evaluation.
    Do not use the cleartext model to implement the returned execute method.
    """
    raise NotImplementedError("Implement build_mapping in cnn_opt.py")
