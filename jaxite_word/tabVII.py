
import os
import sys
import math
import csv
import json
import gzip
import warnings
import statistics
import concurrent.futures
from typing import Callable, List, Any, Dict, Optional, Tuple, Union
import functools
import re
import copy
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.sharding as shd
from absl.testing import absltest
from absl.testing import parameterized

# JAX configuration
jax.config.update("jax_enable_x64", True)
ENABLE_INITIAL_COPY_PROFILE = False

# ==========================================
# CHANGE ME! Evaluation Setup
# ==========================================
BATCH_SIZE_LIST_LOW_DEGREE = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCH_SIZE_LIST_HIHG_DEGREE = [64, 128, 256, 512, 1024]
TEST_PARAMS_NTT=[('2_12', 4096, 4, BATCH_SIZE_LIST_LOW_DEGREE), ('2_13', 8192, 8, BATCH_SIZE_LIST_LOW_DEGREE), ('2_14', 16384, 16, BATCH_SIZE_LIST_LOW_DEGREE), ('2_16_L48', 65536, 48, BATCH_SIZE_LIST_HIHG_DEGREE)]


# ==========================================
# UTIL (from util.py)
# ==========================================

def _square_like_mesh_shape(device_count: int) -> Tuple[int, int]:
  """Return a near-square 2D mesh shape that covers all available devices."""
  if device_count <= 0:
    raise ValueError("At least one device is required to build a mesh.")
  sqrt_devices = math.isqrt(device_count)
  for dim0 in range(sqrt_devices, 0, -1):
    if device_count % dim0 == 0:
      return dim0, device_count // dim0
  return 1, device_count

def create_sharding():
  """Create default batch and replicated shardings for the current device mesh."""
  available_devices = jax.devices()
  if not available_devices:
    raise RuntimeError("No devices available for sharding test.")
  if len(available_devices) == 8:
    mesh_shape = (4, 2)
  elif len(available_devices) == 4:
    mesh_shape = (2, 2)
  elif len(available_devices) == 2:
    mesh_shape = (2, 1)
  else:
    mesh_shape = (1, 1)
  # mesh_shape = _square_like_mesh_shape(len(available_devices))
  mesh = jax.make_mesh(mesh_shape, ('x', 'y'))
  shd.set_mesh(mesh)

  partition_spec = jax.sharding.PartitionSpec
  return mesh, partition_spec

def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a

def extended_gcd(a, b):
  """Return a tuple of (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
  if b == 0:
    return (a, 1, 0)
  else:
    g, x, y = extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)

def modinv(x: int, q: int) -> int:
  """Returns the inverse of x mod q."""
  return int(pow(x, -1, q))

def prime_factors(n):
  """Return the set of prime factors of n."""
  factors = set()
  # Divide out factors of 2
  while n % 2 == 0:
    factors.add(2)
    n //= 2
  # Check odd factors from 3 to sqrt(n)
  p = 3
  while p**2 <= n:
    while n % p == 0:
      factors.add(p)
      n //= p
    p += 2
  if n > 1:
    factors.add(n)
  return factors

def find_generator(q):
  """Find a primitive root modulo q."""
  phi = q - 1
  factors = prime_factors(phi)

  # Test candidates from 2 to q-1.
  for g in range(2, q):
    is_generator = all(pow(g, phi // p, q) != 1 for p in factors)
    if is_generator:
      return g
  raise ValueError("No generator found, check that q is prime.")

def gcd(a, b):
    return math.gcd(a, b)

def root_of_unity(m: int, q: int) -> int:
    """Canonical primitive m-th root of unity modulo q that **works with NTT**."""
    assert (q - 1) % m == 0, "q-1 must be divisible by m"
    # Step 1: multiplicative generator of Z_q^*
    g = find_generator(q)
    # Step 2: raise to (q-1)/m to get an m-th root candidate
    r = pow(g, (q - 1) // m, q)
    # Step 3: among r^k with gcd(k,m)=1, pick the minimal value whose order is exactly m
    candidates = []
    half = m // 2
    for k in range(1, m):
        if gcd(k, m) != 1:
            continue
        psi = pow(r, k, q)
        if pow(psi, half, q) == q - 1 and pow(psi, m, q) == 1:
            candidates.append(psi)
    assert candidates, "No primitive m-th root found"
    return min(candidates)

def list_add(list1: List[Any], list2: List[Any]) -> List[Any]:
    assert len(list1) == len(list2), "The two lists must have the same length"
    return [e1 + e2 for e1, e2 in zip(list1, list2)]

NTT_PARAMETERS_BY_DEGREE = {
  16: {
    "moduli": [1073759809, 1073759041, 1073759777, 1073758337, 1073759329, 1073758849, 1073759233, 1073738273, 1073754113, 1073738753, 1073753729, 1073738977, 1073753281, 1073739041, 1073753089, 1073747137, 1073752417, 1073739169, 1073745697, 1073739361, 1073752129, 1073746337, 1073748737, 1073746529, 1073748289, 1073747393, 1073749889, 1073748449, 1073751713, 1073749153, 1073750593, 1073749409, 1073751521, 1073750017, 1073751169, 1073750497, 1073751073, 1073750113, 1073750849, 1073739617, 1073746273, 1073745473, 1073745889, 1073742881, 1073745377, 1073739649, 1073745121, 1073741953, 1073744993, 1073739937, 1073744417, 1073742913, 1073744257, 1073742113, 1073743457, 1073742209, 1073743393, 1073740609, 1073742721, 1073741441, 1073741857, 524353],
    "root_of_unity": [149761193, 17168328, 145519847, 68042513, 3491826, 21109149, 48183983, 49547540, 15369996, 12935385, 1093151, 90892563, 108899655, 56634236, 235160291, 12265314, 191995239, 21404433, 40083131, 3916344, 113671079, 34500367, 61894143, 20463380, 13205216, 60050555, 145308815, 87067229, 10533116, 133048918, 13697511, 47895671, 14807533, 10994638, 25005605, 44429319, 77617905, 22756112, 21182116, 46947055, 41148497, 163086225, 60397627, 176334344, 30766686, 77429283, 67466901, 67653750, 4536048, 135444559, 63788661, 110966687, 9716122, 12174708, 49591386, 81862273, 51874541, 12155428, 60746932, 68809976, 28870916, 19017],
  },
  4096: {
    "moduli": [268730369, 268689409, 268361729, 268582913, 268369921, 268460033, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [8801, 19068, 58939, 11033, 62736, 77090, 474, 116777451583545, 271802498405390],
  },
  8192: {
    "moduli": [269402113, 268091393, 268730369, 268271617, 269221889, 268664833, 268861441, 268369921, 268582913, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [18987, 2826, 1678, 18925, 2446, 31335, 40892, 65274, 15787, 268, 25959043411404, 100406242475323],
  },
  16384: {
    "moduli": [274726913, 272760833, 274628609, 267059201, 270499841, 267550721, 270237697, 267943937, 268861441, 268042241, 268730369, 268238849, 269844481, 268271617, 269221889, 268369921, 268664833, 557057, 1152921504606748673, 1152921504606683137, 1152921504606584833],
    "root_of_unity": [9358, 15613, 1976, 5381, 15236, 9622, 5177, 2469, 792, 63914, 9742, 12308, 3704, 7216, 7564, 10360, 2023, 19, 62213374832584, 212089012217363, 92166579128688],
  },
  65536: {
    "moduli": [384040961, 376569857, 371458049, 375521281, 371589121, 383778817, 377880577, 379453441, 323092481, 351797249, 349962241, 351404033, 260702209, 308150273, 304742401, 307888129, 302776321, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
    "root_of_unity": [1197, 4622, 9335, 5748, 719, 1497, 2281, 3163, 3548, 80, 6577, 4942, 435, 3498, 316, 4503, 1433, 5766, 440, 2739, 1792, 13, 545, 7539, 7418, 7033, 32540, 1301, 4354, 16962, 10301, 289, 4195, 3322, 1005, 1747, 13384, 7659, 2200, 1035, 2142, 6961, 2774, 910, 43, 1949, 4343, 6648, 787, 2879, 4743, 563, 3385, 5648, 5875, 9494, 2122, 852, 6279, 1335, 712, 2017, 929, 142, 5274, 3264, 8],
  },
}

moduli_28_list = {
  degree: params["moduli"]
  for degree, params in NTT_PARAMETERS_BY_DEGREE.items()
}

# ==========================================
# FINITE FIELD (from finite_field.py)
# ==========================================

class FiniteFieldContextBase():
    def __init__(self, moduli: int):
        self.moduli = moduli

    def to_computation_format(self, a: int):
        return a

    def to_original_format(self, a: int):
        return a

    def get_jax_parameters(self):
        return {}

    def modular_reduction(self, a: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def drop_last_modulus(self):
        raise NotImplementedError("Subclasses must implement this method")

class MontgomeryContext(FiniteFieldContextBase):
    def __init__(self, moduli: Union[List[int], int]):
        super().__init__(moduli)
        self.moduli = moduli
        if type(self.moduli) is int:
          self.moduli = [self.moduli]
        self.w = 32
        self.w_inv = [modinv(1 << self.w, m) for m in self.moduli]
        self.w_inv_reduction = jnp.array(self.w_inv, jnp.uint64)

        self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)

        self.moduli_inv_32 = [modinv(m, 2**32) for m in self.moduli]
        self.moduli_low16 = [m & 0xFFFF for m in self.moduli]
        self.moduli_high16 = [m >> 16 for m in self.moduli]

        self.q = jnp.array(self.moduli, dtype=jnp.uint32)
        self.q_low = jnp.array(self.moduli_low16, dtype=jnp.uint32)
        self.q_high = jnp.array(self.moduli_high16, dtype=jnp.uint32)
        self.q_inv_32 = jnp.array(self.moduli_inv_32, dtype=jnp.uint32)

    def to_computation_format(self, a: int):
        return (a << self.w) % self.moduli_reduction

    def to_original_format(self, a: jnp.ndarray):
        return (a * self.w_inv_reduction) % self.moduli_reduction

    def get_jax_parameters(self):
        return {
            "moduli": to_tuple(self.moduli),
            "moduli_inv_32": to_tuple(self.moduli_inv_32),
            "moduli_low": to_tuple(self.moduli_low16),
            "moduli_high": to_tuple(self.moduli_high16)
        }

    def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
        #Local constants
        MASK32 = 0xFFFFFFFF
        MASK16 = 0xFFFF
        SHIFT16 = 16
        SHIFT32 = 32
        # Ensure dimensions for broadcasting
        q = self.q
        q_low = self.q_low
        q_high = self.q_high
        q_inv_32 = self.q_inv_32

        # Computation
        z_low = z.astype(jnp.uint32)
        z_high = (z >> SHIFT32).astype(jnp.uint32)
        t = (z_low * q_inv_32) & MASK32
        t_low = t & MASK16
        t_high = (t >> SHIFT16) & MASK16

        prod_high = t_high * q_high  # This contributes directly to upper 32 bits
        prod_mid_high = t_high * q_low  # Upper 16 bits go to upper 32 bits
        prod_mid_low = t_low * q_high   # Upper 16 bits go to upper 32 bits
        prod_low = t_low * q_low        # Upper 16 bits contribute to middle part
        mid_low = (prod_mid_high & MASK16) + (prod_mid_low & MASK16) + (prod_low >> SHIFT16)
        mid_high = (prod_mid_high >> SHIFT16) + (prod_mid_low >> SHIFT16) + (mid_low >> SHIFT16)

        # Final upper 32 bits
        t_final = prod_high + mid_high
        b = z_high + q - t_final
        return b.astype(jnp.uint32)

    def modular_reduction_single_modulus(self, z: jnp.ndarray, limb_index: int) -> jnp.ndarray:
        # Simplified for single modulus if needed, but reusing logic
        # Actually logic is broadcastable. If z is (..., M), and props are (M), it works.
        # But if z is (...,) and we need specifc limb params:
        MASK32 = 0xFFFFFFFF
        MASK16 = 0xFFFF
        SHIFT16 = 16
        SHIFT32 = 32

        q = self.q[limb_index]
        q_low = self.q_low[limb_index]
        q_high = self.q_high[limb_index]
        q_inv_32 = self.q_inv_32[limb_index]

        z_low = z.astype(jnp.uint32)
        z_high = (z >> SHIFT32).astype(jnp.uint32)
        t = (z_low * q_inv_32) & MASK32
        t_low = t & MASK16
        t_high = (t >> SHIFT16) & MASK16

        prod_high = t_high * q_high
        prod_mid_high = t_high * q_low
        prod_mid_low = t_low * q_high
        prod_low = t_low * q_low
        mid_low = (prod_mid_high & MASK16) + (prod_mid_low & MASK16) + (prod_low >> SHIFT16)
        mid_high = (prod_mid_high >> SHIFT16) + (prod_mid_low >> SHIFT16) + (mid_low >> SHIFT16)

        t_final = prod_high + mid_high
        b = z_high + q - t_final
        return b.astype(jnp.uint32)

    def drop_last_modulus(self):
        self.moduli_reduction = self.moduli_reduction[:-1]
        self.q = self.q[:-1]
        self.q_low = self.q_low[:-1]
        self.q_high = self.q_high[:-1]
        self.q_inv_32 = self.q_inv_32[:-1]

# ==========================================
# PROFILER (from profiler.py)
# ==========================================
class DataFrameGenerator:
    """A utility class for building pandas DataFrames from column data."""

    def __init__(self):
        """Initialize an empty DataFrameGenerator."""
        self.data: Dict[str, List[Any]] = {}

    def add_data(self, column_name: str, values: List[Any]) -> None:
        """Add data to a specific column.

        Args:
            column_name: Name of the column to add data to
            values: List of values to add to the column
        """
        if not isinstance(column_name, str):
            raise ValueError("column_name must be a string")
        if not isinstance(values, list):
            raise ValueError("values must be a list")

        if column_name not in self.data:
            self.data[column_name] = []
        self.data[column_name].extend(values)

    def add_single_value(self, column_name: str, value: Any) -> None:
        """Add a single value to a specific column.

        Args:
            column_name: Name of the column to add data to
            value: Single value to add to the column
        """
        self.add_data(column_name, [value])

    def get_column_lengths(self) -> Dict[str, int]:
        """Get the length of each column.

        Returns:
            Dictionary mapping column names to their lengths
        """
        return {col: len(values) for col, values in self.data.items()}

    def is_balanced(self) -> bool:
        """Check if all columns have the same length.

        Returns:
            True if all columns have the same length, False otherwise
        """
        if not self.data:
            return True
        lengths = set(len(col) for col in self.data.values())
        return len(lengths) == 1

    def to_dataframe(self, auto_balance: bool = True) -> pd.DataFrame:
        """Convert the stored data to a pandas DataFrame.

        Args:
            auto_balance: If True, automatically trim columns to the minimum length.
                        If False, raise an error if columns have different lengths.

        Returns:
            pandas DataFrame with the stored data

        Raises:
            ValueError: If auto_balance is False and columns have different lengths
        """
        if not self.data:
            return pd.DataFrame()

        if not auto_balance and not self.is_balanced():
            lengths = self.get_column_lengths()
            raise ValueError(f"Columns have different lengths: {lengths}")

        # Find the minimum length among all columns
        min_len = min(len(col) for col in self.data.values())

        # Trim each column to the minimum length
        trimmed_data = {k: v[:min_len] for k, v in self.data.items()}

        return pd.DataFrame(trimmed_data)

    def clear(self) -> None:
        """Clear all stored data."""
        self.data.clear()

    def get_column_names(self) -> List[str]:
        """Get the names of all columns.

        Returns:
            List of column names
        """
        return list(self.data.keys())

    def has_column(self, column_name: str) -> bool:
        """Check if a column exists.

        Args:
            column_name: Name of the column to check

        Returns:
            True if the column exists, False otherwise
        """
        return column_name in self.data

    def merge(self, other_dataframe_generator: "DataFrameGenerator"):
        """Merge the stored data with another DataFrameGenerator.

        Args:
            other_dataframe_generator: Another DataFrameGenerator to merge with

        Returns:
            Merged DataFrameGenerator
        """
        if not isinstance(other_dataframe_generator, DataFrameGenerator):
            raise ValueError("other_dataframe_generator must be a DataFrameGenerator")
        # Check if this DataFrameGenerator is empty
        if not self.data:
            self.data = other_dataframe_generator.data
            return
        # Check if the other DataFrameGenerator has the same column names
        if not set(self.get_column_names()) == set(other_dataframe_generator.get_column_names()):
            print("The two DataFrameGenerators have different column names")
            return
            # raise ValueError("The two DataFrameGenerators have different column names")
        # Merge the data
        for column_name in other_dataframe_generator.get_column_names():
            self.add_data(column_name, other_dataframe_generator.data[column_name])


    def get_header(self) -> List[str]:
        """Get the header of the DataFrameGenerator.

        Returns:
            List of column names
        """
        return list(self.data.keys())

    def get_row_dict(self, index: int) -> List[Any]:
        """Get a row of the DataFrameGenerator.

        Returns:
            Dictionary of column names and values
        """
        return {column_name: self.data[column_name][index] for column_name in self.get_column_names()}


class TraceParser:
    def __init__(self, trace_dir: str):
        self.trace_dir = trace_dir

    def set_trace_dir(self, new_dir: str):
        """
        Set a new trace directory for the parser.
        """
        self.trace_dir = new_dir

    def find_trace_file(self):
        """
        Recursively search for the first .trace.json.gz file in the trace_dir.
        Returns the full path to the file, or None if not found.
        """
        for root, _, files in os.walk(self.trace_dir):
            for file in files:
                if file.endswith('.trace.json.gz'):
                    return os.path.join(root, file)
        return None

    def read_trace_json(self):
        """
        Finds, unzips, and reads the JSON content from the trace file.
        Returns the loaded JSON object, or None if not found or error.
        """
        trace_file = self.find_trace_file()
        if trace_file is None:
            print("No trace file found.")
            return None
        try:
            with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading trace file: {e}")
            return None

    def parse_trace_csv(self):
        """
        Parses the trace CSV file and returns a list of trace events.
        """
        csv_file = os.path.join(self.trace_dir, 'trace_events.csv')

        # Read the trace JSON data
        trace_data = self.read_trace_json()
        if trace_data is None:
            print("Failed to read trace data")
            return None

        # Extract trace events
        trace_events = trace_data.get('traceEvents', [])
        if not trace_events:
            print("No trace events found in the data")
            return None

        headers = ['pid', 'tid', 'ts', 'dur', 'ph', 'name', 'args']
        # Write to CSV directly
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for event in trace_events:
                # Convert args dictionary to string if it exists
                if 'args' in event:
                    event['args'] = json.dumps(event['args'])
                else:
                    event['args'] = ''

                # Write the event
                writer.writerow(event)
        print(f"Trace events written to: {csv_file}")


def calculate_statistics(data: List[Any]) -> Dict[str, Any]:
    """Calculate the statistics of the data.

    Args:
        data: List of data

    Returns:
        Dictionary containing the statistics
    """
    mean_value = statistics.mean(data)
    if len(data) == 1:
        std_value = 0
    else:
        std_value = statistics.stdev(data)
    min_value = min(data)
    max_value = max(data)
    median_value = statistics.median(data)
    return {
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "median": median_value,
    }


def list_add(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Sum two lists element-wise.

    Args:
        list1: First list to sum
        list2: Second list to sum

    Returns:
        List of the sum of the two lists
    """
    assert len(list1) == len(list2), "The two lists must have the same length"
    return [e1 + e2 for e1, e2 in zip(list1, list2)]



class KernelWrapper:
    def __init__(self,
                 kernel_name: str,
                 function_to_wrap: Callable,
                 input_structs: List[Tuple[Tuple[int, ...], jnp.dtype]],
                 mesh: Optional[jax.sharding.Mesh] = None,
                 input_shardings: Optional[Tuple[jax.sharding.Sharding, ...]] = None,
                 output_sharding: Optional[jax.sharding.Sharding] = None,
                 parameters: Optional[Dict[str, Any]] = {},
                 enable_sharding: bool = False):
        self.kernel_name = kernel_name
        self.callable_function = function_to_wrap
        self.input_structs = input_structs
        self.parameters = parameters
        self.mesh = mesh
        self.input_shardings = input_shardings
        self.output_sharding = output_sharding
        self.enable_sharding = enable_sharding

        self.jit_lower = None
        self.jit_compiled_function = None

        # Compile immediately upon initialization
        self._compile()

    def _compile(self):
        jax_input_structs = []
        if self.enable_sharding and self.input_shardings:
            for (shape, dtype), sharding in zip(self.input_structs, self.input_shardings):
                jax_input_structs.append(jax.ShapeDtypeStruct(shape, dtype, sharding=sharding))
        else:
            for shape, dtype in self.input_structs:
                jax_input_structs.append(jax.ShapeDtypeStruct(shape, dtype))

        # NOTE: Do not change the name of the function, it is used for profiling
        def compiled_kernel_function(*jax_array_inputs):
            return self.callable_function(*jax_array_inputs, parameters=self.parameters)

        if self.enable_sharding and self.mesh:
            with self.mesh:
                self.jit_lower = jax.jit(
                    jax.named_call(compiled_kernel_function, name=self.kernel_name),
                    in_shardings=self.input_shardings,
                    out_shardings=self.output_sharding,
                ).lower(*jax_input_structs)
        else:
            self.jit_lower = jax.jit(jax.named_call(compiled_kernel_function, name=self.kernel_name)).lower(*jax_input_structs)

        self.jit_compiled_function = self.jit_lower.compile()

    def get_compiled_function(self) -> Callable[..., jnp.ndarray]:
        assert self.jit_compiled_function is not None, "Kernel not compiled"
        if self.enable_sharding and self.mesh:
            def compiled_with_mesh(*jax_array_inputs):
                with self.mesh:
                    return self.jit_compiled_function(*jax_array_inputs)
            return compiled_with_mesh
        return self.jit_compiled_function

    def get_input_structs(self):
        return self.input_structs

    def get_kernel_name(self) -> str:
        return self.kernel_name

    def shard_inputs(self, input_arrays: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Place inputs on the provided sharding."""
        if self.enable_sharding and self.input_shardings:
             return [jax.device_put(arr, sharding) for arr, sharding in zip(input_arrays, self.input_shardings)]
        return input_arrays



class Profiler:
    def __init__(self, output_trace_path: str, profile_naming: str, configuration: Optional[Dict[str, Any]] = None):
        self.trace_dir = output_trace_path
        self.profiler_name = profile_naming
        self.profile_dir = os.path.join(self.trace_dir, self.profiler_name)
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)

        self.configuration = configuration or {}
        self.random_seed = self.configuration.get("random_seed", 0)
        self.iterations = self.configuration.get("iterations", 1)
        self.save_to_file = self.configuration.get("save_to_file", True)
        self.enable_sharding = self.configuration.get("enable_sharding", False)

        self.profiles: List[Dict[str, Any]] = []
        self.profile_name_list: List[str] = []

        # Storage for results
        self.storage_file = os.path.join(self.profile_dir, f"{self.profiler_name}_results.csv")

    def add_profile(self, name: str, kernel_wrapper: KernelWrapper, kernel_setting_cols: Dict[str, Any] = {}):
        if name in self.profile_name_list:
             raise ValueError(f"Profiler name {name} already exists")

        self.profile_name_list.append(name)

        profile_folder = os.path.join(self.profile_dir, name)
        if not os.path.exists(profile_folder):
            os.makedirs(profile_folder)

        self.profiles.append({
            "name": name,
            "wrapper": kernel_wrapper,
            "settings": kernel_setting_cols,
            "folder": profile_folder,
            "failed": False,
            "trace_events": None,
            "filtered_events": None,
            "stats": None
        })

    def _get_input_arrays(self, kernel_wrapper: KernelWrapper):
        def get_max_value(dtype):
            if dtype == jnp.uint8:
                return 128
            elif dtype == jnp.uint16:
                return 32768
            elif dtype == jnp.uint32:
                return 4294967295
            elif dtype == jnp.uint64:
                return 4294967295
            raise ValueError(f"Unsupported dtype: {dtype}")

        random_key = jax.random.key(self.random_seed)
        input_arrays = []
        for shape, dtype in kernel_wrapper.get_input_structs():
            if jnp.issubdtype(dtype, jnp.floating):
                input_arrays.append(jax.random.uniform(random_key, shape, dtype))
            elif jnp.issubdtype(dtype, jnp.integer):
                input_arrays.append(jax.random.randint(random_key, shape, 0, get_max_value(dtype), dtype))
            elif jnp.issubdtype(dtype, jnp.bool_):
                input_arrays.append(jax.random.bernoulli(random_key, shape, dtype))
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        for input_array in input_arrays:
            input_array.block_until_ready()

        if self.enable_sharding:
            input_arrays = kernel_wrapper.shard_inputs(input_arrays)

        return input_arrays

    def profile_all_profilers(self):
        for profile in self.profiles:
            # print(f"Profiling {profile['name']}")
            try:
                # Kernel wrapper is already compiled in its init
                compiled_function = profile['wrapper'].get_compiled_function()
                input_arrays = self._get_input_arrays(profile['wrapper'])
                compiled_function(*input_arrays).block_until_ready()

                with jax.profiler.trace(profile['folder']):
                    for _ in range(self.iterations):
                        compiled_function(*input_arrays).block_until_ready()
            except Exception as e:
                print(f"Error profiling {profile['name']}:\n {e}")
                profile['failed'] = True

    def _parse_json_trace(self, profile):
        if not os.path.exists(os.path.join(profile['folder'], "trace_events.json")):
            trace_parser = TraceParser(profile['folder'])
            profile_json = trace_parser.read_trace_json()
            if profile_json is None:
                warnings.warn(f"{profile['name']}: No trace events found in the data", UserWarning)
                profile['failed'] = True
                return None
            trace_events = profile_json.get('traceEvents', [])
            if not trace_events:
                warnings.warn(f"{profile['name']}: No trace events found in the data", UserWarning)
                profile['failed'] = True
                return None
            if self.save_to_file:
                with open(os.path.join(profile['folder'], "trace_events.json"), "w") as f:
                    json.dump(trace_events, f, indent=2)
        else:
            with open(os.path.join(profile['folder'], "trace_events.json"), "r") as f:
                trace_events = json.load(f)
        profile['trace_events'] = trace_events
        return trace_events

    def _filter_trace_events(self, profile):
        trace_events = profile['trace_events']
        if trace_events is None:
            return None

        kernel_name = profile['wrapper'].get_kernel_name()

        def merge_filtered_events_by_name(filtered_events):
            grouped = {}
            for event in filtered_events:
                event_name = event.get('name', 'unknown')
                if "args" in event.keys() and "deduplicated_name" in event['args'].keys():
                    event_name += "_" + event['args']['deduplicated_name']
                elif "custom-call" in event['name'] and "args" in event.keys() and "tf_op" in event['args'].keys():
                    event_name += "_" + event['args']['tf_op']
                if event_name not in grouped:
                    grouped[event_name] = []
                grouped[event_name].append(event)

            merged_filtered_events = {}
            for event_name, events in grouped.items():
                merged = events[0].copy()
                merged['dur'] = [e.get('dur') for e in events if 'dur' in e]
                merged['ts'] = [e.get('ts') for e in events if 'ts' in e]
                merged['repeat_count'] = len(events)
                merged_filtered_events[event_name] = merged
            return merged_filtered_events

        filtered_events_list = []
        # Check if NVIDIA is in device kind OR CPU is used as a fallback if explicit check needed
        # But generally JAX trace events differ by backend.
        # Assuming typical CPU/GPU separation.
        device_kind = jax.devices()[0].device_kind

        if "NVIDIA" in device_kind:
            for e in trace_events:
                if 'args' in e and 'tf_op' in e['args']:
                    # Loosen the check for compiled_kernel_function as it might be nested differently or named differently
                    if "compiled_kernel_function" in e['args'].get("hlo_module", "") or \
                       "compiled_kernel_function" in e['args'].get("long_name", ""):
                        merged_event = False
                        # Try to merge with existing events
                        for f in filtered_events_list:
                             # Check if correlation_id exists before accessing it
                             if 'correlation_id' in f['args'] and 'correlation_id' in e['args'] and \
                                f['args']['correlation_id'] == e['args']['correlation_id'] and f['name'] == e['name']:
                                f['dur'] = f['dur'] + e['dur']
                                merged_event = True
                        if not merged_event:
                            filtered_events_list.append(e)
            profile['filtered_events'] = merge_filtered_events_by_name(filtered_events_list)

        elif "TPU" in device_kind:
            for event in trace_events:
                if "pid" not in event.keys() or event['pid'] != 3: # ToDo: change it into automatic PID detection based on "TPU:0".
                    continue
                # if "name" in event.keys() and "args" in event.keys():
                #     filtered_events_list.append(event)
                if "args" in event.keys() and "long_name" in event['args'].keys():
                    if "tf_op" in event['args'].keys() and kernel_name in event['args']["tf_op"]:
                        filtered_events_list.append(event)
                    elif "tf_op" in event['args'].keys() and "jax_array_inputs" in event['args']["tf_op"]:
                        filtered_events_list.append(event)
                    elif ENABLE_INITIAL_COPY_PROFILE and "copy" in event['name'] and "tf_op" in event["args"].keys() and "jax_array_inputs" in event['args']["tf_op"]:
                        filtered_events_list.append(event)
                else:
                    continue
            profile['filtered_events'] = merge_filtered_events_by_name(filtered_events_list)
        else:
             # Fallback for CPU or other devices
             # CPU traces might be different. Let's try to capture events related to our kernel.
             for event in trace_events:
                 if "name" in event and "compiled_kernel_function" in event['name']:
                     filtered_events_list.append(event)
             profile['filtered_events'] = merge_filtered_events_by_name(filtered_events_list)

        # Always save filtered events if we have any
        if self.save_to_file:
             # Make sure we don't crash if profile['filtered_events'] is None
             events_to_dump = profile['filtered_events'] if profile['filtered_events'] is not None else {}
             with open(os.path.join(profile['folder'], "filtered_events.json"), "w") as f:
                  json.dump(events_to_dump, f, indent=2)

    def _calculate_profiling_statistics(self, profile):
        if profile['filtered_events'] is None:
            return

        kernel_name = profile['wrapper'].get_kernel_name()
        repeat_count = self.iterations
        kernel_duration = [0] * repeat_count

        device_kind = jax.devices()[0].device_kind

        if "NVIDIA" in device_kind:
            for event in profile['filtered_events'].values():
                if "compiled_kernel_function" in event['args'].get('hlo_module', ""):
                    kernel_duration = list_add(kernel_duration, event['dur'])
        elif "TPU" in device_kind:
            for event in profile['filtered_events'].values():
                if kernel_name in event['args']['tf_op']:
                    kernel_duration = list_add(kernel_duration, event['dur'])
                elif ENABLE_INITIAL_COPY_PROFILE and "copy" in event['name'] and "tf_op" in event["args"].keys() and "jax_array_inputs" in event['args']["tf_op"]:
                    kernel_duration = list_add(kernel_duration, event['dur'])
        else:
            # CPU logic - assuming direct name match from filtered events
             for event in profile['filtered_events'].values():
                # On CPU, events might be simpler
                if "compiled_kernel_function" in event.get('name', ""):
                    # DUR might be a single value or list depending on how it was merged
                    durations = event['dur']
                    if not isinstance(durations, list):
                        durations = [durations]

                    # If we have less durations than repeat_count, we might need to pad or it's a mismatch
                    # For now, let's just add what we have, assuming 1-to-1 or aggregated
                    if len(durations) == repeat_count:
                         kernel_duration = list_add(kernel_duration, durations)
                    elif len(durations) > repeat_count:
                         # Take first N
                         kernel_duration = list_add(kernel_duration, durations[:repeat_count])
                    else:
                         # Append 0s? Or just take what we have
                         padded = durations + [0] * (repeat_count - len(durations))
                         kernel_duration = list_add(kernel_duration, padded)

        profile['stats'] = {
            "kernel_all": kernel_duration,
        }

    def post_process_all_profilers(self):
        for profile in self.profiles:
            if profile['failed']:
                continue

            events = self._parse_json_trace(profile)
            if events is None:
                continue

            self._filter_trace_events(profile)
            self._calculate_profiling_statistics(profile)

        self.write_results()

    def get_profiling_dataframe_generator_all_profilers(self):
        df_generator = DataFrameGenerator()
        for profile in self.profiles:
            if profile['failed'] or profile['stats'] is None:
                continue

            p_df_gen = DataFrameGenerator()
            p_df_gen.add_single_value("operation_name", profile['wrapper'].get_kernel_name())

            for key, value in profile['settings'].items():
                p_df_gen.add_single_value(key, value)

            all_kernel_duration = profile['stats']['kernel_all']
            for i, duration in enumerate(all_kernel_duration):
                p_df_gen.add_single_value(f"sample_{i}", duration)

            df_generator.merge(p_df_gen)
        return df_generator

    def write_results(self):
        storage_dataframe_generator = self.get_profiling_dataframe_generator_all_profilers()
        print(storage_dataframe_generator.to_dataframe().to_csv())


# ==========================================
# NTT MM (from ntt_mm.py)
# ==========================================

def gen_twiddle_matrix(rows, cols, q, omega):
  r_idx = np.arange(rows, dtype=np.int64)[:, None]
  c_idx = np.arange(cols, dtype=np.int64)[None, :]
  exponents = r_idx * c_idx
  twiddle_matrix = np.zeros((rows, cols), dtype=int)
  def compute_row(r):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(int(omega), int(exponents[r, c]), int(q))
  with concurrent.futures.ThreadPoolExecutor() as executor:
    list(executor.map(compute_row, range(rows)))
  return twiddle_matrix

def gen_twiddle_matrix_inv(rows, cols, q, omega):
  twiddle_matrix_inv = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(int(omega), int(-r * c), int(q))
  return twiddle_matrix_inv

def matmul_bat_einsum(lhs: jax.Array, rhs: jax.Array, subscripts: str):
    lhs = jax.lax.bitcast_convert_type(lhs, new_dtype=jnp.uint8)
    shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
    i8_products = jnp.einsum(subscripts, lhs, rhs, preferred_element_type=jnp.uint32)
    return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))

class NTTCiphertextContextBase():
    def __init__(self, moduli: int, parameters: dict):
        self.ff_ctx = parameters.get("finite_field_context", None)
        self.num_bytes = 4
        self.moduli = moduli
        self.parameters = parameters
        self.r = parameters.get("r", 0)
        self.c = parameters.get("c", 0)
        self.transform_length = self.r * self.c
        self.psi_list = [root_of_unity(2 * self.transform_length, q) for q in self.moduli]
        self.omega_list = [(psi ** 2) % q for psi, q in zip(self.psi_list, self.moduli)]

        self.ntt_tf_step1, self.ntt_tf_step2, self.ntt_tf_step3 = self.ntt_coefficients_precompute()
        self.ntt_bat_tf_step1 = self.basis_aligned_transformation(self.to_computation_format(self.ntt_tf_step1))
        self.ntt_tf_step2 = self.to_computation_format(self.ntt_tf_step2).astype(jnp.uint64)
        self.ntt_bat_tf_step3 = self.basis_aligned_transformation(self.to_computation_format(self.ntt_tf_step3))

    def ntt_coefficients_precompute(self):
        tf_step1_list, tf_step2_list, tf_step3_list = [], [], []
        for i, modulus in enumerate(self.moduli):
            omega_col = pow(self.omega_list[i], self.c, modulus)
            omega_row = pow(self.omega_list[i], self.r, modulus)
            tf_step1_one_modulus = gen_twiddle_matrix(self.r, self.r, modulus, omega_col)
            tf_step2_one_modulus = gen_twiddle_matrix(self.r, self.c, modulus, self.omega_list[i])
            tf_step3_one_modulus = gen_twiddle_matrix(self.c, self.c, modulus, omega_row)
            tf_step1_list.append(tf_step1_one_modulus)
            tf_step2_list.append(tf_step2_one_modulus)
            tf_step3_list.append(tf_step3_one_modulus)
        tf_step1 = jnp.array(tf_step1_list, dtype=jnp.uint32).transpose(1,2,0)
        tf_step2 = jnp.array(tf_step2_list, dtype=jnp.uint32).transpose(1,2,0)
        tf_step3 = jnp.array(tf_step3_list, dtype=jnp.uint32).transpose(1,2,0)
        return tf_step1, tf_step2, tf_step3

    def to_computation_format(self, a: np.ndarray):
        return self.ff_ctx.to_computation_format(a.astype(jnp.uint64)).astype(jnp.uint32)

    def basis_aligned_transformation(self, matrix: np.ndarray):
        matrix_u64 = matrix.astype(np.uint64)
        matrix_u64_byteshifted = np.array([matrix_u64 << (8 * byte_idx) for byte_idx in range(self.num_bytes)], dtype=np.uint64)
        matrix_u64_byteshifted_mod_modulus = (matrix_u64_byteshifted % jnp.array(self.moduli, dtype=np.uint64)).astype(np.uint32)
        matrix_u8 = jax.lax.bitcast_convert_type(matrix_u64_byteshifted_mod_modulus, jnp.uint8).transpose(1,0,2,4,3)
        return matrix_u8

    def ntt(self, v: jax.Array):
        result_step1 = matmul_bat_einsum(v, self.ntt_bat_tf_step1, "brcmq,zqrpm->bzcmp")
        result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
        result_step2 = jnp.multiply(result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2)
        result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
        result_step3 = matmul_bat_einsum(result_step2_reduced, self.ntt_bat_tf_step3, "brcmq,cqnpm->bnrmp")
        result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
        return result_step3_reduced

class NTTCiphertextMontgomeryContext(NTTCiphertextContextBase):
    def __init__(self, moduli: int, parameters: dict):
        super().__init__(moduli, parameters)
        if type(self.moduli) is int:
            self.moduli = [self.moduli]
        if self.ff_ctx is None:
          self.ff_ctx = MontgomeryContext(moduli)

# ==========================================
# TEST (from ntt_mm_perf_test.py)
# ==========================================

DEGREE_TO_RC_MAPPING = {
    65536: (128, 512), # Make some dimension 128 is always helpful!
    32768: (128, 256),
    16384: (128, 128),
    8192: (128, 64),
    4096: (128, 32),
    2048: (128, 16),
}

def _ntt_kernel(input_array, parameters):
  """Kernel wrapper entry point used by KernelWrapper."""
  return parameters["ctx"].ntt(input_array)

class NTTMMShardedPerformanceTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.output_trace_root = "log"
    # self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
    self.profiler_config = {
        "iterations": 1,
        "save_to_file": True,
    }

  def _create_sharded_kernel_wrapper(self, kernel_name, ctx, batch, rows, cols, num_moduli, mesh, batch_sharding):
    input_shape = (batch, rows, cols, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_ntt_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"ctx": ctx},
        mesh=mesh,
        input_shardings=(batch_sharding,),
        output_sharding=batch_sharding,
        enable_sharding=True,
    )

  def _profile_context_sharded(self, profile_prefix, ctx_cls, ff_ctx_cls, degree, num_limbs, batch_size_list):
    rows, cols = DEGREE_TO_RC_MAPPING[degree]
    moduli = moduli_28_list[degree][:num_limbs]
    try:
      mesh, partition_spec = create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(batch_partition, None, None, None),
      )
    except RuntimeError as exc:
      self.skipTest(str(exc))
      return

    profiler_config = self.profiler_config.copy()
    profiler_config["enable_sharding"] = True
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"sharding_{profile_prefix}_degree_{degree}",
        configuration=profiler_config,
    )

    for batch in batch_size_list:
      ctx_parameters = {
          "r": rows,
          "c": cols,
          "finite_field_context": ff_ctx_cls(moduli=moduli),
      }
      ctx = ctx_cls(moduli=moduli, parameters=ctx_parameters)

      kernel_name = f"sharding_{profile_prefix}_batch_{batch}"
      kernel_wrapper = self._create_sharded_kernel_wrapper(
          kernel_name=kernel_name,
          ctx=ctx,
          batch=batch,
          rows=rows,
          cols=cols,
          num_moduli=len(moduli),
          mesh=mesh,
          batch_sharding=batch_sharding,
      )

      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "degree": degree,
              "num_limbs": num_limbs,
              "batch": batch,
              "rows": rows,
              "cols": cols,
              "sharding": "batch",
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_sharded_NTT_Montgomery_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context_sharded(
        profile_prefix="ntt_montgomery",
        ctx_cls=NTTCiphertextMontgomeryContext,
        ff_ctx_cls=MontgomeryContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )


if __name__ == "__main__":
  absltest.main()
