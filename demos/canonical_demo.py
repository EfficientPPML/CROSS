"""The canonical demo pipeline: a torch model, vectorized, packed, mapped.

Every encrypted demo follows the same visible path::

    torch.nn.Module  ->  nn.vectorize  ->  packing.pack  ->  Mapping

and nothing else. There is no second compiler entry point and no ``from_*``
convenience: a demo supplies its model and its input shape, and this module
runs that one pipeline over them.

Planning is separated from materialization on purpose. ``build_plan`` is
cheap -- it traces the model, vectorizes it and derives the secure ring, which
is everything you need to inspect what a demo will do. ``materialize_mapping``
is the expensive half and constructs exactly one Mapping for the life of the
demo. Key generation is not what makes it expensive: that is well under a
second even at degree 16384. The cost is Mapping initialization and the
physical constants -- BSGS diagonals over thousands of slots, and the
per-level operator controls -- which run to tens of gigabytes at demo width.

Weight provenance is recorded rather than glossed over. Only LoLA ships real
binaries; LeNet and AlexNet fall back to a seeded random draw, which is fine
for exercising the pipeline and useless for accuracy. Every demo says which it
got, in its metadata and in its log line.
"""

import dataclasses
import logging
import os
import sys
from typing import Any

import numpy as np

# The demos use the library's flat sibling imports (``nn``, ``packing``,
# ``mapping``). Make ``jaxite_word/`` importable from this directory so a
# demo works when launched from ``demos/`` with no PYTHONPATH, as the README
# documents.
_JAXITE_WORD_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jaxite_word')
)
if _JAXITE_WORD_DIR not in sys.path:
  sys.path.insert(0, _JAXITE_WORD_DIR)

import nn
import packing

_LOGGER = logging.getLogger(__name__)

# Demos plan against the matrix-free constant source. The packer's default is
# eager SparseDiagonals, which is exact but quadratic in ciphertext width: at
# AlexNetFull's 3711 non-zero diagonals over 32768 slots that is about a
# gigabyte of float64, spent before anything is encrypted.
DEMO_LAZY_CONSTANTS = True


# The demo layer may choose only Mapping's deployment policy. In particular it
# cannot smuggle in a parameters dict, keys, a second packed program, or a
# model-specific execution path. Keep this list aligned with the scheduling
# surface documented in jaxite_word/API_REFERENCE.md.
_MAPPING_SCHEDULING = frozenset((
    'global_batch',
    'devices',
    'dnum',
    'input_scale',
    'input_nsd',
    'headroom',
    'cache_rotation_keys',
    'compile_mode',
    'bsgs_n_jobs',
    'bsgs_streaming',
    'perf_test',
))


def _validate_mapping_scheduling(scheduling: dict) -> None:
  unknown = sorted(set(scheduling) - _MAPPING_SCHEDULING)
  if unknown:
    raise TypeError(
        f'demo Mapping options must be scheduling-only; unsupported '
        f'option(s): {unknown}. Allowed options: '
        f'{sorted(_MAPPING_SCHEDULING)}.'
    )


def _enable_jax_x64() -> None:
  """Enable the integer width required by CKKS immediately before setup."""
  import jax

  jax.config.update('jax_enable_x64', True)


@dataclasses.dataclass(frozen=True)
class WeightProvenance:
  """Where a demo's parameters came from, stated plainly."""

  source: str  # 'checkpoint' or 'seeded-random'
  detail: str
  files: tuple[str, ...] = ()

  @property
  def is_checkpoint(self) -> bool:
    return self.source == 'checkpoint'

  @property
  def is_caller_supplied(self) -> bool:
    return self.source == 'caller-supplied'

  def describe(self) -> str:
    if self.is_checkpoint or self.is_caller_supplied:
      return f'weights: {self.detail}'
    return (
        f'weights: NO CHECKPOINT SHIPPED -- {self.detail}. Outputs exercise '
        'the pipeline; they carry no accuracy meaning.'
    )


class CanonicalDemo:
  """One encrypted demo, built on the canonical pipeline.

  Subclasses supply ``build_model()``, ``input_shape`` and ``provenance``.
  Everything below is the shared path.
  """

  #: Logical shape of one unbatched sample, as the torch model expects it.
  input_shape: tuple[int, ...] = ()

  def __init__(
      self,
      *,
      lazy_constants: bool = DEMO_LAZY_CONSTANTS,
      security_bits: int = packing.DEFAULT_SECURITY_BITS,
      **scheduling,
  ):
    """``scheduling`` holds Mapping's execution knobs only.

    global_batch, devices, compile_mode, bsgs_n_jobs, bsgs_streaming,
    perf_test and the rest choose how the program runs. None of them change
    what it computes or which ring it runs on -- those follow from the model.

    ``dnum`` may only restate the value fixed by the derived ring; Mapping
    refuses a contradiction. The BSGS split is not a demo option: Mapping
    chooses it per operation from each matvec's own diagonals.
    """
    self._packing_policy = packing.PackingPolicy(
        lazy_constants=lazy_constants,
        security_bits=security_bits,
    )
    _validate_mapping_scheduling(scheduling)
    self._scheduling = dict(scheduling)
    self._model = None
    self._vectorized_program = None
    self._packed_program = None
    self._mapping = None
    self._provenance_override = None

  # -- subclass surface -----------------------------------------------------

  def build_model(self):
    """Return the torch.nn.Module that is this demo's source of truth."""
    raise NotImplementedError

  @property
  def provenance(self) -> WeightProvenance:
    """Where this demo's parameters came from right now.

    Subclasses answer for the weights they build. Once a caller overwrites
    them, ``load_weights`` takes over the answer, so metadata and logs never
    keep claiming a checkpoint or a seed that no longer describes the model.
    """
    if self._provenance_override is not None:
      return self._provenance_override
    return self.default_provenance

  @property
  def default_provenance(self) -> WeightProvenance:
    raise NotImplementedError

  #: Legacy positional weight order -> torch parameter name, for the
  #: ``precompute_plaintexts`` compatibility path. Empty means the demo has
  #: no legacy entrypoint.
  weight_order: tuple[tuple[str, str], ...] = ()

  # -- caller-supplied weights ----------------------------------------------

  def load_weights(self, arrays: dict) -> None:
    """Populate the exact torch model that nn.vectorize will trace.

    Legacy callers pass weight and bias arrays positionally. Those arrays are
    the model, so they are written into the module rather than layered on
    beside it -- a demo that kept its seeded fallback here would compile
    something other than what the caller asked for. Any existing plan is
    discarded, because it describes the previous parameters.
    """
    import torch

    model = self.model
    state = model.state_dict()
    unknown = sorted(set(arrays) - set(state))
    if unknown:
      raise ValueError(
          f'{type(model).__name__} has no parameter(s) {unknown}; it holds '
          f'{sorted(state)}.'
      )
    updates = {}
    for name, value in arrays.items():
      array = np.asarray(value, dtype=np.float64)
      expected = tuple(state[name].shape)
      if array.size != int(np.prod(expected)):
        raise ValueError(
            f'{name} has {array.size} elements, but {expected} needs '
            f'{int(np.prod(expected))}.'
        )
      if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} contains non-finite values.')
      updates[name] = torch.tensor(
          array.reshape(expected), dtype=torch.float64
      )
    with torch.no_grad():
      for name, tensor in updates.items():
        state[name].copy_(tensor)
    # The model is no longer what build_model produced, so neither its
    # provenance nor the plan describing it still holds.
    self._provenance_override = WeightProvenance(
        source='caller-supplied',
        detail=(
            f'{len(updates)} array(s) supplied by the caller and written into '
            f'{type(model).__name__}'
        ),
        files=tuple(sorted(updates)),
    )
    self.invalidate_plan()

  def load_positional_weights(self, *arrays) -> None:
    """Translate a legacy positional weight list through ``weight_order``."""
    if not self.weight_order:
      raise TypeError(
          f'{type(self).__name__} declares no legacy weight order.'
      )
    if len(arrays) != len(self.weight_order):
      raise ValueError(
          f'expected {len(self.weight_order)} arrays in the order '
          f'{[name for name, _ in self.weight_order]}, got {len(arrays)}.'
      )
    supplied = {}
    for (label, parameter), value in zip(
        self.weight_order, arrays, strict=True
    ):
      del label
      if value is None:
        # A None bias lowers to explicit zeros, as it always has. Leaving the
        # parameter alone instead would quietly keep whatever was there --
        # a shipped checkpoint value, or the seeded fallback -- and compile a
        # bias the caller did not ask for.
        supplied[parameter] = np.zeros(
            tuple(self.model.state_dict()[parameter].shape), dtype=np.float64
        )
      else:
        supplied[parameter] = value
    self.load_weights(supplied)


  def invalidate_plan(self) -> None:
    """Drop the plan and Mapping so the next build reflects new weights."""
    self._vectorized_program = None
    self._packed_program = None
    # A dropped Mapping keeps its HBM until released (see ``release``).
    self.release()

  # -- the pipeline ---------------------------------------------------------

  @property
  def model(self):
    if self._model is None:
      self._model = self.build_model()
    return self._model

  def build_plan(self):
    """Trace, vectorize and pack. Cheap: no keys, no encryption.

    Idempotent, so inspecting a demo repeatedly costs one traversal.
    """
    if self._packed_program is None:
      self._vectorized_program = nn.vectorize(self.model, self.input_shape)
      self._packed_program = packing.pack(
          self._vectorized_program, self._packing_policy
      )
      _LOGGER.info(
          '%s: %s | depth %d over %d slots -> degree %d (num_q %d) | %s',
          type(self).__name__,
          self.provenance.describe(),
          self._packed_program.depth,
          self._packed_program.num_slots,
          self._packed_program.ring_config.degree,
          int(self._packed_program.ring_config.logical_num_q),
          ' '.join(
              f'{kind}x{self._packed_program.kinds().count(kind)}'
              for kind in dict.fromkeys(self._packed_program.kinds())
          ),
      )
    return self._packed_program

  @property
  def vectorized_program(self):
    """The frontend's description of this model. Builds the plan if needed."""
    self.build_plan()
    return self._vectorized_program

  @property
  def packed_program(self):
    """The PP-ops and the ring derived for them."""
    return self.build_plan()

  @property
  def ring_config(self):
    return self.build_plan().ring_config

  def materialize_mapping(self):
    """Construct this demo's one Mapping. Expensive: expect gigabytes.

    Deliberately separate from ``build_plan`` so inspecting a demo never pays
    to materialize constants or build per-level operator controls. Scheduling
    is fixed at demo construction; a materialized Mapping cannot be retargeted
    to another batch, device tuple, or execution mode.
    """
    if self._mapping is None:
      from mapping import Mapping

      self._mapping = Mapping(self.build_plan(), **self._scheduling)
    return self._mapping

  @property
  def mapping(self):
    return self.materialize_mapping()

  def is_mapped(self) -> bool:
    """True once the Mapping exists, without building one to find out."""
    return self._mapping is not None

  def release(self) -> None:
    """Return the materialized Mapping's device memory; keep the plan.

    A materialized Mapping pins gigabytes of HBM (its compiled executable with
    the captured keys and encoded constants) and dropping the Python reference
    alone does not free them. Call this before materializing another Mapping
    in the same process, for example when measuring several batch sizes in
    one test run. The next ``mapping`` access materializes afresh from the
    cached plan.
    """
    if self._mapping is not None:
      self._mapping.release()
      self._mapping = None

  # -- inference ------------------------------------------------------------

  def cleartext(self, sample) -> np.ndarray:
    """Run the torch model itself, for reference."""
    import torch

    array = np.asarray(sample, dtype=np.float64).reshape(self.input_shape)
    with torch.no_grad():
      output = self.model(
          torch.tensor(array, dtype=torch.float64).unsqueeze(0)
      )
    return output.numpy().reshape(-1)

  def infer(self, sample, **options):
    """Encrypted inference through this demo's single Mapping."""
    return self.materialize_mapping().infer(sample, **options)

  def metadata(self) -> dict[str, Any]:
    """Everything worth recording about a planned demo."""
    packed = self.build_plan()
    ring = packed.ring_config
    return {
        'demo': type(self).__name__,
        'model': type(self.model).__name__,
        'input_shape': tuple(self.input_shape),
        'weights_source': self.provenance.source,
        'weights_detail': self.provenance.detail,
        'weights_files': tuple(self.provenance.files),
        'has_shipped_checkpoint': self.provenance.is_checkpoint,
        'layers': len(self.vectorized_program.layers),
        'depth': packed.depth,
        'num_slots': packed.num_slots,
        'degree': int(ring.degree),
        'logical_num_q': int(ring.logical_num_q),
        'dnum': int(ring.dnum),
        'security_bits': int(ring.security_bits),
        'lazy_constants': self._packing_policy.lazy_constants,
        'operations': dict(
            (kind, packed.kinds().count(kind))
            for kind in dict.fromkeys(packed.kinds())
        ),
        'program_digest': self.vectorized_program.digest,
        'packed_fingerprint': packed.fingerprint,
    }


class CanonicalDemoAdapter:
  """Shared compatibility facade for every encrypted model entrypoint.

  The model-specific files retain their historical constructors and
  ``precompute_plaintexts`` weight order, but they do not implement a second
  deployment or serving API. This adapter owns their common lifecycle:

  ``torch model -> nn.vectorize -> packing.pack -> Mapping -> Mapping.infer``.

  Subclasses may preprocess one client sample and report a planned ring. They
  must not replace the Mapping-backed ``infer``, ``encrypt`` or ``decrypt``
  boundaries.
  """

  def __init__(
      self,
      demo_type: type[CanonicalDemo],
      *,
      batch: int = 1,
      devices=None,
      dnum: int | None = None,
  ):
    if (
        isinstance(batch, bool)
        or not isinstance(batch, (int, np.integer))
        or int(batch) < 1
    ):
      raise ValueError(f'batch must be >= 1 and an int, got {batch!r}.')
    if not isinstance(demo_type, type) or not issubclass(
        demo_type, CanonicalDemo
    ):
      raise TypeError('demo_type must be a CanonicalDemo subclass.')

    self._batch = int(batch)
    self._devices = None if devices is None else tuple(devices)
    self._dnum = None if dnum is None else int(dnum)
    scheduling = {
        'global_batch': self._batch,
        'devices': self._devices,
    }
    if self._dnum is not None:
      scheduling['dnum'] = self._dnum
    self._demo = demo_type(**scheduling)
    self._prepared = False

  @property
  def demo(self) -> CanonicalDemo:
    """The canonical demo this compatibility entrypoint delegates to."""
    return self._demo

  @property
  def batch(self) -> int:
    """The Mapping's public global batch."""
    return self._batch

  @property
  def vectorized_program(self):
    return self._demo.vectorized_program

  @property
  def packed_program(self):
    return self._demo.packed_program

  @property
  def packing(self):
    """Compatibility alias for the one packed program."""
    return self._demo.packed_program

  @property
  def ring_config(self):
    return self._demo.ring_config

  def _require_prepared(self, what: str) -> None:
    if not self._prepared:
      raise RuntimeError(
          f'{what} is unavailable until precompute_plaintexts(...) has bound '
          f'this model\'s weights; reaching it now would compile fallback '
          f'parameters.'
      )

  @property
  def mapping(self):
    """The entrypoint's one Mapping, after caller weights are bound."""
    self._require_prepared('mapping')
    return self._demo.mapping

  def release(self) -> None:
    """Free this model's Mapping and its device memory.

    The bound weights stay on the demo, but the model is no longer prepared:
    ``infer`` and ``mapping`` raise until ``precompute_plaintexts`` runs
    again, so a caller cannot pay a silent multi-minute rebuild by accident.
    """
    self._demo.release()
    self._prepared = False

  @property
  def ctx(self):
    """Compatibility view of the context owned by ``mapping``."""
    return self.mapping.ctx

  def _validate_ring(self, ring_config) -> None:
    """Optional model-specific assertion over the packer's derived ring."""
    del ring_config

  def _on_plan_ready(self, packed) -> None:
    """Optional model-specific reporting before Mapping materialization."""
    del packed

  def _prepare(
      self,
      arrays,
      *,
      legacy_bsgs_ratios: dict[str, float] | None = None,
  ):
    """Bind positional weights and materialize exactly one Mapping."""
    if self._prepared:
      raise RuntimeError(
          f'This {type(self).__name__} instance is already prepared; '
          f'construct a new instance to bind different model constants.'
      )
    reject_legacy_bsgs_ratio(**(legacy_bsgs_ratios or {}))
    _enable_jax_x64()
    self._demo.load_positional_weights(*arrays)
    packed = self._demo.build_plan()
    self._validate_ring(packed.ring_config)
    self._on_plan_ready(packed)
    self._demo.materialize_mapping()
    self._prepared = True
    return packed

  def _client_preprocess(self, sample):
    """Transform one client sample before Mapping's logical input packer."""
    return sample

  def _logical_input(self, value):
    if self._batch == 1:
      return self._client_preprocess(value)
    try:
      values = list(value)
    except TypeError as error:
      raise TypeError(
          'batched demo input must be an iterable of logical samples.'
      ) from error
    if len(values) != self._batch:
      raise ValueError(
          f'batched demo input contains {len(values)} samples, expected '
          f'{self._batch}.'
      )
    return [self._client_preprocess(sample) for sample in values]

  def infer(self, value, *, trace_dir=None):
    """Run the shared Mapping inference boundary for this logical model."""
    self._require_prepared('infer(...)')
    if trace_dir:
      os.makedirs(trace_dir, exist_ok=True)
    return self._demo.infer(
        self._logical_input(value), trace_dir=trace_dir
    )

  def encrypt(self, value):
    """Pack and encrypt this Mapping's logical input or global batch."""
    self._require_prepared('encrypt')
    return self.mapping.encrypt_input(self._logical_input(value))

  def encrypt_batch(self, values):
    """Compatibility spelling for ``encrypt`` on a batched Mapping."""
    if self._batch == 1:
      raise ValueError('encrypt_batch requires a model with batch > 1.')
    return self.encrypt(values)

  def decrypt(self, ciphertext):
    """Decrypt and unpack this Mapping's logical output or global batch."""
    self._require_prepared('decrypt')
    return self.mapping.decrypt_output(ciphertext)

  def decrypt_batch(self, ciphertext):
    """Compatibility spelling for ``decrypt`` on a batched Mapping."""
    if self._batch == 1:
      raise ValueError('decrypt_batch requires a model with batch > 1.')
    return self.decrypt(ciphertext)

  def metadata(self) -> dict[str, Any]:
    return self._demo.metadata()


def reject_legacy_bsgs_ratio(**ratios) -> None:
  """Refuse a legacy global BSGS ratio, which no longer has a home.

  The baby-step/giant-step split is now chosen per operation by Mapping, from
  the diagonals each matvec actually has, rather than frozen for a whole
  program at packing time. Accepting the argument and dropping it would let a
  caller believe they had tuned something.
  """
  offenders = {
      name: float(value) for name, value in ratios.items()
      if value is not None and float(value) != 2.0
  }
  if offenders:
    raise ValueError(
        f'bsgs_ratio {offenders} cannot be honoured: the baby-step/giant-step '
        'split is chosen per matvec by Mapping from that operation\'s own '
        'diagonals, not fixed for the whole program. Drop the argument.'
    )



def seeded_parameters(model, seed: int):
  """Fill ``model`` with a deterministic draw, and say that is what happened.

  Used by the demos with no shipped checkpoint. Small enough that the packed
  matrices stay well clear of the encode floor, and reproducible so two runs
  of a demo produce the same program digest.
  """
  import torch

  generator = torch.Generator().manual_seed(int(seed))
  with torch.no_grad():
    for parameter in model.parameters():
      if parameter.ndim > 1:
        flat = torch.randn(
            parameter.numel(), generator=generator, dtype=torch.float64
        )
        fan_in = int(np.prod(parameter.shape[1:]))
        parameter.copy_(
            (flat * (0.5 / max(fan_in, 1) ** 0.5)).reshape(parameter.shape)
        )
      else:
        parameter.zero_()
  return model
