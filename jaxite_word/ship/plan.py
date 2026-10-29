"""Immutable SHIP configuration and key-free planning metadata."""

import math
from dataclasses import dataclass
from numbers import Real

from .math import is_prime


@dataclass(frozen=True)
class SHIPConfig:
  """SHIP tuning parameters for a real-slot bootstrap.

  ``weight`` is the internal signed sparse key's Hamming weight; ``theta``
  controls its rotation decomposition. ``gamma`` sets the phase range and
  ``input_scale_factor`` sets the input-transform scale. This configuration
  checks arithmetic compatibility, not cryptographic security or accuracy.
  """

  weight: int = 31
  theta: int = 9
  gamma: float = 8.0
  input_scale_factor: float = 16.0

  def __post_init__(self):
    for name in ("weight", "theta"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    for name in ("gamma", "input_scale_factor"):
      value = getattr(self, name)
      if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
      ):
        raise ValueError(f"{name} must be positive and finite")

  def plan(self, ctx):
    """Validate a context and return immutable, key-free boundary metadata."""
    n, q, p = ctx.degree, tuple(ctx.q_towers), tuple(ctx.p_towers)
    if n < 8 or n & (n - 1) or ctx.num_slots != n // 2:
      raise ValueError("SHIP requires a power-of-two ring and N/2 real slots")
    if ctx.composite_degree != 1:
      raise ValueError("SHIP requires composite_degree=1")
    if self.weight > n or self.theta > n // 2:
      raise ValueError("SHIP weight/theta exceed the ring dimensions")
    depth = (self.weight - 1).bit_length() + 1
    if len(q) < depth + 2 or len(p) < 2:
      raise ValueError("SHIP needs two output Q limbs and at least two P limbs")
    if len(set(q + p)) != len(q + p) or any(
      v < 3 or v >= 2**31 or (v - 1) % (2 * n) or not is_prime(v) for v in q + p
    ):
      raise ValueError("SHIP requires distinct NTT primes below 2**31")
    slots = n // 2
    layers = slots.bit_length() - 1
    step = min(
      slots, 1 << (layers // 2 + 1 + (1 if (1 << (layers + 1)) - 1 > 7 else 0))
    )
    rotations = tuple(sorted(set(range(1, step)) | set(range(step, slots, step))))
    input_scale = float(q[0] / self.gamma * self.input_scale_factor)
    if not math.isfinite(input_scale) or input_scale <= 0:
      raise ValueError("SHIP input scale must be positive and finite")
    return _SHIPPlan(
      self,
      len(q) - depth - 1,
      input_scale,
      float(q[-1]),
      rotations,
    )


@dataclass(frozen=True)
class _SHIPPlan:
  config: SHIPConfig
  expected_output_level: int
  input_scale: float
  output_scale: float
  input_rotation_indices: tuple[int, ...]
  input_level: int = 1
  input_nsd: int = 1
  # The input transform has its own two-limb key domain. These are not
  # ordinary rotations under Mapping's full Q chain; setup owns those keys.
  required_rotation_indices: tuple[int, ...] = ()

  def validate_input(self, level, scale, nsd):
    if level != self.input_level or nsd != 1:
      raise ValueError("SHIP input requires level 1 and noise scale degree 1")
    if scale is None or not math.isclose(float(scale), self.input_scale, rel_tol=1e-12):
      raise ValueError(f"SHIP input requires scale {self.input_scale}")
