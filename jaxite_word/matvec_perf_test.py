"""Performance tests for homomorphic matrix-vector multiplication (MatVec).

Profiles the three atomic sub-operations of BSGS MatVec individually:
  1. Baby-step rotation (HERot at input level)
  2. Plaintext-ciphertext multiply + rescale (HEPtCtMul + HERescale)
  3. Giant-step rotation (HERot at output level, with one fewer limb)

The total MatVec latency is estimated from:
  T_matvec ~ (n1-1)*T_baby_rot + n1*n2*T_ptct_mul_rescale + (n2-1)*T_giant_rot

where n1*n2 = num_slots (BSGS decomposition).

Directly instantiates HERot, HEPtCtMul, and HERescale with random data,
matching the pattern of herot_perf_test.py, hemul_perf_test.py, and
rescale_perf_test.py (no CKKSContext or keygen needed).
"""

import os
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import herot
from rescale import HERescale
from ptct_mul import HEPtCtMul
from polynomial import Polynomial
from matvec import compute_bsgs_params, MatVecScannable, make_ptct_rescale_fn
from profiler import KernelWrapper, Profiler, collect_logs
import util

jax.config.update("jax_enable_x64", True)

# (name, degree, r, c, limbs, dnum, q_towers, p_towers, perf_test_flag)
TEST_PARAMS = [
    (
        'CL_256_256', 65536, 256, 256, 51, 3,
        [349962241, 306708481, 304218113, 347996161, 319291393, 347078657,
         323223553, 337248257, 323878913, 336855041, 329515009, 332660737,
         329777153, 335413249, 325844993, 330301441, 327548929, 332267521,
         328728577, 344850433, 336068609, 340000769, 261488641, 302252033,
         297664513, 299499521, 261881857, 295305217, 263323649, 277086209,
         263454721, 292159489, 279838721, 291373057, 284950529, 290455553,
         281935873, 285474817, 283508737, 288882689, 264634369, 276430849,
         270532609, 274726913, 272760833, 276037633, 265420801, 270794753,
         268042241, 269221889, 786433],
        [260702209, 260571137, 258605057, 257949697, 256770049, 256376833,
         254279681, 253493249, 253100033, 249561089, 246415361, 245760001,
         245235713, 244973569, 244842497, 241827841, 240648193, 239861761],
        True,
    ),
]


# ── Kernel wrappers ──────────────────────────────────────────────────

def _herot_kernel(input_array, parameters):
    """Profile a single HERot rotation."""
    ct = parameters["ct_input"]
    ct.polynomial = input_array
    result = parameters["herot_obj"].rotate(ct)
    return result.polynomial


def _ptct_rescale_kernel(input_array, parameters):
    """Profile a single pt-ct multiply followed by rescale."""
    ptct = parameters["ptct_obj"]
    rescale_obj = parameters["rescale_obj"]
    ct = parameters["ct_input"]
    ct.polynomial = input_array
    result = ptct.mul(ct, use_bat=False)
    rescaled_data = rescale_obj.rescale(result.polynomial)
    return rescaled_data


class PerformanceTest(parameterized.TestCase):
    """Profiles MatVec atomic sub-operations without sharding (batch=1)."""

    def setUp(self):
        super().setUp()
        self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
        self.profiler_config = {
            "iterations": 1,
            "save_to_file": True,
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        root_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Collecting logs from: {root_dir}")
        collect_logs(root_dir, output_csv_name="matvec_profiling")

    @absltest.skip("test single implementation")
    @parameterized.named_parameters(*TEST_PARAMS)
    def test_matvec(self, degree, r, c, limbs, dnum, moduli, extend_moduli,
                    perf_test):
        batch = 1
        num_elements = 2
        degree_layout = (r, c)
        num_moduli = len(moduli)
        num_moduli_out = num_moduli - 1
        moduli_out = moduli[:num_moduli_out]
        num_slots = degree // 2
        n1, n2 = compute_bsgs_params(num_slots)

        profiler_instance = Profiler(
            output_trace_path=self.output_trace_root,
            profile_naming=f"{self._testMethodName}_N{degree}",
            configuration=self.profiler_config,
        )

        # ── 1. Baby-step rotation (HERot at input level) ──
        print("Setting up baby-step rotation...")
        eval_key_a = util.random_parameters(
            (dnum, *degree_layout, num_moduli + len(extend_moduli)),
            moduli, dtype=jnp.uint32)
        eval_key_b = util.random_parameters(
            (dnum, *degree_layout, num_moduli + len(extend_moduli)),
            moduli, dtype=jnp.uint32)
        coefMap = util.random_parameters((r * c,), [r * c], dtype=jnp.uint32)

        herot_baby = herot.HERot(r, c, dnum, moduli, extend_moduli)
        herot_baby.control_gen(batch=batch, degree_layout=degree_layout,
                               perf_test=perf_test)
        herot_baby.setup_rotate(eval_key_a, eval_key_b, coefMap)

        input_shape = (batch, num_elements, *degree_layout, num_moduli)
        ct_baby = Polynomial(
            {'batch': batch, 'num_elements': num_elements, 'degree': degree,
             'precision': 32, 'num_moduli': num_moduli,
             'degree_layout': degree_layout},
            {'moduli': moduli, 'BAT_lazy': False})
        baby_wrapper = KernelWrapper(
            kernel_name=f"matvec_baby_rot_B{batch}",
            function_to_wrap=_herot_kernel,
            input_structs=[(input_shape, jnp.uint32)],
            parameters={"herot_obj": herot_baby, "ct_input": ct_baby},
        )
        profiler_instance.add_profile(
            name=f"matvec_baby_rot_B{batch}",
            kernel_wrapper=baby_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "operation": "baby_rot", "n1": n1, "n2": n2,
            },
        )

        # ── 2. Pt-ct multiply + rescale ──
        print("Setting up ptct_mul + rescale...")
        ptct_obj = HEPtCtMul(batch=batch, r=r, c=c, moduli=moduli)
        pt_ntt = util.random_parameters((r, c, num_moduli), moduli,
                                        dtype=jnp.uint32)
        ptct_obj.set_plaintext(pt_ntt)

        rescale_obj = HERescale(batch=batch, num_elements=num_elements,
                                moduli=moduli, r=r, c=c)
        rescale_obj.control_gen(composite_degree=1)

        ct_ptct = Polynomial(
            {'batch': batch, 'num_elements': num_elements, 'degree': degree,
             'precision': 32, 'num_moduli': num_moduli,
             'degree_layout': degree_layout},
            {'moduli': moduli, 'BAT_lazy': False})
        ptct_wrapper = KernelWrapper(
            kernel_name=f"matvec_ptct_rescale_B{batch}",
            function_to_wrap=_ptct_rescale_kernel,
            input_structs=[(input_shape, jnp.uint32)],
            parameters={
                "ptct_obj": ptct_obj,
                "rescale_obj": rescale_obj,
                "ct_input": ct_ptct,
            },
        )
        profiler_instance.add_profile(
            name=f"matvec_ptct_rescale_B{batch}",
            kernel_wrapper=ptct_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "operation": "ptct_mul_rescale", "n1": n1, "n2": n2,
            },
        )

        # ── 3. Giant-step rotation (HERot at output level, one fewer limb) ──
        print("Setting up giant-step rotation...")
        eval_key_a_out = util.random_parameters(
            (dnum, *degree_layout, num_moduli_out + len(extend_moduli)),
            moduli_out, dtype=jnp.uint32)
        eval_key_b_out = util.random_parameters(
            (dnum, *degree_layout, num_moduli_out + len(extend_moduli)),
            moduli_out, dtype=jnp.uint32)

        herot_giant = herot.HERot(r, c, dnum, moduli_out, extend_moduli)
        herot_giant.control_gen(batch=batch, degree_layout=degree_layout,
                                perf_test=perf_test)
        herot_giant.setup_rotate(eval_key_a_out, eval_key_b_out, coefMap)

        input_shape_out = (batch, num_elements, *degree_layout, num_moduli_out)
        ct_giant = Polynomial(
            {'batch': batch, 'num_elements': num_elements, 'degree': degree,
             'precision': 32, 'num_moduli': num_moduli_out,
             'degree_layout': degree_layout},
            {'moduli': moduli_out, 'BAT_lazy': False})
        giant_wrapper = KernelWrapper(
            kernel_name=f"matvec_giant_rot_B{batch}",
            function_to_wrap=_herot_kernel,
            input_structs=[(input_shape_out, jnp.uint32)],
            parameters={"herot_obj": herot_giant, "ct_input": ct_giant},
        )
        profiler_instance.add_profile(
            name=f"matvec_giant_rot_B{batch}",
            kernel_wrapper=giant_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "operation": "giant_rot", "n1": n1, "n2": n2,
            },
        )

        print(f"Profiling 3 atomic MatVec ops (batch={batch})")
        profiler_instance.profile_all_profilers()
        profiler_instance.post_process_all_profilers()

        print(f"\nBSGS: n1={n1}, n2={n2}, slots={n1*n2}")
        print(f"T_matvec ~ (n1-1)*T_baby + n1*n2*T_ptct_rescale + (n2-1)*T_giant")


class BatchDimensionShardingTest(parameterized.TestCase):
    """Profiles MatVec atomic sub-operations with batch-dimension sharding."""

    def setUp(self):
        super().setUp()
        self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
        self.profiler_config = {
            "iterations": 1,
            "save_to_file": True,
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        root_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Collecting logs from: {root_dir}")
        collect_logs(root_dir, output_csv_name="matvec_profiling")

    @parameterized.named_parameters(*TEST_PARAMS)
    def test_matvec_sharded(self, degree, r, c, limbs, dnum, moduli,
                            extend_moduli, perf_test):
        try:
            mesh, partition_spec = util.create_sharding()
            axis_names = mesh.axis_names
            batch_partition = (axis_names if len(axis_names) > 1
                               else axis_names[0])
        except RuntimeError as exc:
            self.skipTest(str(exc))

        num_devices = jax.device_count()
        batch = num_devices
        num_elements = 2
        degree_layout = (r, c)
        num_moduli = len(moduli)
        num_moduli_out = num_moduli - 1
        moduli_out = moduli[:num_moduli_out]
        num_slots = degree // 2
        n1, n2 = compute_bsgs_params(num_slots)

        profiler_config = self.profiler_config.copy()
        profiler_config["enable_sharding"] = True
        profiler_instance = Profiler(
            output_trace_path=self.output_trace_root,
            profile_naming=f"{self._testMethodName}_N{degree}",
            configuration=profiler_config,
        )

        # Shardings
        input_none_dims = (None,) * (1 + len(degree_layout) + 1)
        input_sharding = jax.sharding.NamedSharding(
            mesh, partition_spec(batch_partition, *input_none_dims))
        output_rot_sharding = jax.sharding.NamedSharding(
            mesh, partition_spec(batch_partition, None, None, None))
        output_rescale_sharding = jax.sharding.NamedSharding(
            mesh, partition_spec(batch_partition, *((None,) * (1 + len(degree_layout) + 1 - 1))))
        input_out_sharding = jax.sharding.NamedSharding(
            mesh, partition_spec(batch_partition, *input_none_dims))

        # ── Shared eval keys (random) ──
        eval_key_a = util.random_parameters(
            (dnum, *degree_layout, num_moduli + len(extend_moduli)),
            moduli, dtype=jnp.uint32)
        eval_key_b = util.random_parameters(
            (dnum, *degree_layout, num_moduli + len(extend_moduli)),
            moduli, dtype=jnp.uint32)
        coefMap = util.random_parameters((r * c,), [r * c], dtype=jnp.uint32)

        # ── 1. Baby-step rotation ──
        print(f"Setting up baby-step rotation (batch={batch})...")
        herot_baby = herot.HERot(r, c, dnum, moduli, extend_moduli)
        herot_baby.control_gen(batch=batch, degree_layout=degree_layout,
                               perf_test=perf_test)
        herot_baby.setup_rotate(eval_key_a, eval_key_b, coefMap)

        input_shape = (batch, num_elements, *degree_layout, num_moduli)
        ct_baby = Polynomial(
            {'batch': batch, 'num_elements': num_elements, 'degree': degree,
             'precision': 32, 'num_moduli': num_moduli,
             'degree_layout': degree_layout},
            {'moduli': moduli, 'BAT_lazy': False})
        baby_wrapper = KernelWrapper(
            kernel_name=f"matvec_baby_rot_B{batch}",
            function_to_wrap=_herot_kernel,
            input_structs=[(input_shape, jnp.uint32)],
            parameters={"herot_obj": herot_baby, "ct_input": ct_baby},
            mesh=mesh,
            input_shardings=(input_sharding,),
            output_sharding=output_rot_sharding,
            enable_sharding=True,
        )
        profiler_instance.add_profile(
            name=f"matvec_baby_rot_B{batch}",
            kernel_wrapper=baby_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "sharding": "batch",
                "operation": "baby_rot", "n1": n1, "n2": n2,
            },
        )

        # ── 2. Pt-ct multiply + rescale ──
        print(f"Setting up ptct_mul + rescale (batch={batch})...")
        ptct_obj = HEPtCtMul(batch=batch, r=r, c=c, moduli=moduli)
        pt_ntt = util.random_parameters((r, c, num_moduli), moduli,
                                        dtype=jnp.uint32)
        ptct_obj.set_plaintext(pt_ntt)

        rescale_obj = HERescale(batch=batch, num_elements=num_elements,
                                moduli=moduli, r=r, c=c)
        rescale_obj.control_gen(composite_degree=1)

        ct_ptct = Polynomial(
            {'batch': batch, 'num_elements': num_elements, 'degree': degree,
             'precision': 32, 'num_moduli': num_moduli,
             'degree_layout': degree_layout},
            {'moduli': moduli, 'BAT_lazy': False})
        ptct_wrapper = KernelWrapper(
            kernel_name=f"matvec_ptct_rescale_B{batch}",
            function_to_wrap=_ptct_rescale_kernel,
            input_structs=[(input_shape, jnp.uint32)],
            parameters={
                "ptct_obj": ptct_obj,
                "rescale_obj": rescale_obj,
                "ct_input": ct_ptct,
            },
            mesh=mesh,
            input_shardings=(input_sharding,),
            output_sharding=output_rescale_sharding,
            enable_sharding=True,
        )
        profiler_instance.add_profile(
            name=f"matvec_ptct_rescale_B{batch}",
            kernel_wrapper=ptct_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "sharding": "batch",
                "operation": "ptct_mul_rescale", "n1": n1, "n2": n2,
            },
        )

        # ── 3. Giant-step rotation (one fewer limb) ──
        print(f"Setting up giant-step rotation (batch={batch})...")
        eval_key_a_out = util.random_parameters(
            (dnum, *degree_layout, num_moduli_out + len(extend_moduli)),
            moduli_out, dtype=jnp.uint32)
        eval_key_b_out = util.random_parameters(
            (dnum, *degree_layout, num_moduli_out + len(extend_moduli)),
            moduli_out, dtype=jnp.uint32)

        herot_giant = herot.HERot(r, c, dnum, moduli_out, extend_moduli)
        herot_giant.control_gen(batch=batch, degree_layout=degree_layout,
                                perf_test=perf_test)
        herot_giant.setup_rotate(eval_key_a_out, eval_key_b_out, coefMap)

        input_shape_out = (batch, num_elements, *degree_layout, num_moduli_out)
        ct_giant = Polynomial(
            {'batch': batch, 'num_elements': num_elements, 'degree': degree,
             'precision': 32, 'num_moduli': num_moduli_out,
             'degree_layout': degree_layout},
            {'moduli': moduli_out, 'BAT_lazy': False})
        giant_wrapper = KernelWrapper(
            kernel_name=f"matvec_giant_rot_B{batch}",
            function_to_wrap=_herot_kernel,
            input_structs=[(input_shape_out, jnp.uint32)],
            parameters={"herot_obj": herot_giant, "ct_input": ct_giant},
            mesh=mesh,
            input_shardings=(input_out_sharding,),
            output_sharding=output_rot_sharding,
            enable_sharding=True,
        )
        profiler_instance.add_profile(
            name=f"matvec_giant_rot_B{batch}",
            kernel_wrapper=giant_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "sharding": "batch",
                "operation": "giant_rot", "n1": n1, "n2": n2,
            },
        )

        print(f"Profiling 3 atomic MatVec ops (batch={batch}, sharded)")
        profiler_instance.profile_all_profilers()
        profiler_instance.post_process_all_profilers()

        print(f"\nBSGS: n1={n1}, n2={n2}, slots={n1*n2}")
        print(f"T_matvec ~ (n1-1)*T_baby + n1*n2*T_ptct_rescale + (n2-1)*T_giant")


# ── End-to-end scannable MatVec profiling ─────────────────────────────

def _e2e_matvec_kernel(input_array, parameters):
    """Profile the full scannable MatVec as a single compiled kernel."""
    mv = parameters["matvec_scan"]
    return mv.mul_scan(
        input_array,
        parameters["baby_eval_a"], parameters["baby_eval_b"],
        parameters["baby_coefmaps"],
        parameters["giant_eval_a"], parameters["giant_eval_b"],
        parameters["giant_coefmaps"], parameters["diag_pts"])


class EndToEndMatVecTest(parameterized.TestCase):
    """Profiles the full MatVec as a single compiled kernel using scan."""

    def setUp(self):
        super().setUp()
        self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
        self.profiler_config = {
            "iterations": 1,
            "save_to_file": True,
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        root_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Collecting logs from: {root_dir}")
        collect_logs(root_dir, output_csv_name="matvec_profiling")

    @parameterized.named_parameters(*TEST_PARAMS)
    def test_e2e_matvec_sharded(self, degree, r, c, limbs, dnum, moduli,
                                extend_moduli, perf_test):
        batch = 1
        num_elements = 2
        degree_layout = (r, c)
        num_moduli = len(moduli)
        num_moduli_out = num_moduli - 1
        moduli_out = moduli[:num_moduli_out]
        # Use reduced slot count for memory feasibility.
        # The ring dimension (degree=65536) is unchanged; only the BSGS
        # iteration count changes. Each scan body is identical to full-scale.
        num_slots = 32  # n1=4, n2=8 -> ~2.8 GB total data
        n1, n2 = compute_bsgs_params(num_slots)
        sizeQlP = num_moduli + len(extend_moduli)
        sizeQlP_out = num_moduli_out + len(extend_moduli)
        ring_dim = r * c

        print(f"Setting up E2E scannable MatVec (batch={batch}, "
              f"n1={n1}, n2={n2}, slots={num_slots})...")

        # ── Build baby-step HERot ──
        eval_key_a = util.random_parameters(
            (dnum, *degree_layout, sizeQlP), moduli, dtype=jnp.uint32)
        eval_key_b = util.random_parameters(
            (dnum, *degree_layout, sizeQlP), moduli, dtype=jnp.uint32)
        coefMap = util.random_parameters((ring_dim,), [ring_dim],
                                         dtype=jnp.uint32)

        herot_baby = herot.HERot(r, c, dnum, moduli, extend_moduli)
        herot_baby.control_gen(batch=batch, degree_layout=degree_layout,
                               perf_test=perf_test)
        herot_baby.setup_rotate(eval_key_a, eval_key_b, coefMap)

        # ── Build giant-step HERot ──
        eval_key_a_out = util.random_parameters(
            (dnum, *degree_layout, sizeQlP_out), moduli_out, dtype=jnp.uint32)
        eval_key_b_out = util.random_parameters(
            (dnum, *degree_layout, sizeQlP_out), moduli_out, dtype=jnp.uint32)

        herot_giant = herot.HERot(r, c, dnum, moduli_out, extend_moduli)
        herot_giant.control_gen(batch=batch, degree_layout=degree_layout,
                                perf_test=perf_test)
        herot_giant.setup_rotate(eval_key_a_out, eval_key_b_out, coefMap)

        # ── Build ptct_mul + rescale ──
        ptct_obj = HEPtCtMul(batch=batch, r=r, c=c, moduli=moduli)
        pt_ntt = util.random_parameters((r, c, num_moduli), moduli,
                                        dtype=jnp.uint32)
        ptct_obj.set_plaintext(pt_ntt)

        rescale_obj = HERescale(batch=batch, num_elements=num_elements,
                                moduli=moduli, r=r, c=c)
        rescale_obj.control_gen(composite_degree=1, perf_test=perf_test)

        # ── Build MatVecScannable ──
        mv = MatVecScannable(
            herot_baby=herot_baby, herot_giant=herot_giant,
            ptct_obj=ptct_obj, rescale_obj=rescale_obj,
            n=num_slots, n1=n1, n2=n2,
            q_out=moduli_out, batch=batch)

        # ── Stack per-rotation data (generated directly at full shape) ──
        print("Generating per-rotation data...")
        key = jax.random.key(42)

        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        baby_eval_a = jax.random.randint(
            k1, (n1 - 1, dnum, *degree_layout, sizeQlP),
            0, 2**31, dtype=jnp.uint64)
        baby_eval_b = jax.random.randint(
            k2, (n1 - 1, dnum, *degree_layout, sizeQlP),
            0, 2**31, dtype=jnp.uint64)
        baby_coefmaps = jax.random.randint(
            k3, (n1 - 1, ring_dim), 0, ring_dim, dtype=jnp.int32)

        giant_eval_a = jax.random.randint(
            k4, (n2, dnum, *degree_layout, sizeQlP_out),
            0, 2**31, dtype=jnp.uint64)
        giant_eval_b = jax.random.randint(
            k5, (n2, dnum, *degree_layout, sizeQlP_out),
            0, 2**31, dtype=jnp.uint64)
        giant_coefmaps = jax.random.randint(
            k6, (n2, ring_dim), 0, ring_dim, dtype=jnp.int32)

        diag_pts = jax.random.randint(
            k7, (n2, n1, *degree_layout, num_moduli),
            0, 2**31, dtype=jnp.uint32)

        print(f"Data stacked. baby_eval_a={baby_eval_a.shape}, "
              f"diag_pts={diag_pts.shape}")

        # ── Create KernelWrapper ──
        input_shape = (batch, num_elements, *degree_layout, num_moduli)
        output_shape = (batch, num_elements, *degree_layout, num_moduli_out)

        print("Compiling scannable MatVec kernel...")
        kernel_wrapper = KernelWrapper(
            kernel_name=f"matvec_e2e_B{batch}",
            function_to_wrap=_e2e_matvec_kernel,
            input_structs=[(input_shape, jnp.uint32)],
            parameters={
                "matvec_scan": mv,
                "baby_eval_a": baby_eval_a,
                "baby_eval_b": baby_eval_b,
                "baby_coefmaps": baby_coefmaps,
                "giant_eval_a": giant_eval_a,
                "giant_eval_b": giant_eval_b,
                "giant_coefmaps": giant_coefmaps,
                "diag_pts": diag_pts,
            },
        )

        profiler_instance = Profiler(
            output_trace_path=self.output_trace_root,
            profile_naming=f"{self._testMethodName}_N{degree}",
            configuration=self.profiler_config,
        )
        profiler_instance.add_profile(
            name=f"matvec_e2e_B{batch}",
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "degree": degree, "num_limbs": limbs,
                "r": r, "c": c, "batch": batch,
                "num_elements": num_elements,
                "operation": "e2e_matvec", "n1": n1, "n2": n2,
            },
        )

        print("Profiling E2E scannable MatVec...")
        profiler_instance.profile_all_profilers()
        profiler_instance.post_process_all_profilers()

        print(f"\nBSGS: n1={n1}, n2={n2}, slots={num_slots}")
        print(f"E2E MatVec profiling complete.")


if __name__ == "__main__":
    absltest.main()
