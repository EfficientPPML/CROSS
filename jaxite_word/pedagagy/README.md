# NTT Implementation Guide

This directory (`pedagagy`) contains educational and production-oriented implementations of Number Theoretic Transform (NTT) algorithms. The code illustrates the evolution from mathematical definitions to highly optimized, hardware-accelerated kernels.

## 1. NTT Variants Include

We implement two primary mathematical variants of the NTT, essential for Homomorphic Encryption (BFV, CKKS, etc.):

1.  **Cyclic NTT / INTT**:
    *   The standard transform over a finite field.
    *   Used for polynomial multiplication in $R_q = \mathbb{Z}_q[X] / (X^N - 1)$.
2.  **Negacyclic NTT / INTT**:
    *   Used for polynomial rings of the form $R_q = \mathbb{Z}_q[X] / (X^N + 1)$.
    *   **Implementation**: Realized by "Pre-twisting" (multiplying coefficients by powers of $\psi$, where $\psi^2 = \omega$), running a Cyclic NTT, and "Post-twisting" for the inverse.

## 2. Algorithms of NTT

We provide implementations with different algorithmic complexities and structural properties:

*   **Original / Naive ($O(N^2)$)**:
    *   Direct matrix-vector multiplication.
    *   **Purpose**: Golden reference for correctness and understanding the definition.
*   **Cooley-Tukey ($O(N \log N)$)**:
    *   Standard recursive decomposition (FFT).
    *   **Implementation**: Iterative "Bit-Reverse" approach.
*   **4-Step NTT ($O(N^{3/2})$ / $O(N^{1.5})$)**:
    *   Decomposes the transform into: Column NTT $\rightarrow$ Twiddle multiplication $\rightarrow$ Row NTT.
    *   **Purpose**: Improves memory locality and parallelization structure.
*   **3-Step NTT ($O(N^{3/2})$ / $O(N^{1.5})$)**:
    *   A "Layout Invariant" or "Stockham-like" approach optimized for hardware (TPU) with high penalty for layout transformation.
    *   **Purpose**: Expresses NTT purely as a sequence of large matrix multiplications to maximize arithmetic density.

## 3. Modular Reduction Algorithms

Efficient modular reduction is critical for performance. We support:

*   **Barrett Reduction**:
    *   Uses precomputed `s_w`, `w`, `m` scalar factors to estimate quotients.
    *   Dominant in the JAX/TPU implementations.
*   **Montgomery Reduction**:
    *   Transforms values into Montgomery form to replace division with logical shifts.
    *   Used in specific high-performance variants (`ntt_three_step_bat_montgomery_batch`).

## 4. Layouts and Batching

To support various hardware configurations and use-cases, we support different data layouts:

*   **Batching**:
    *   **Batch First**: `(Batch, N, ...)`
    *   **Batch Second**: `(Moduli, Batch, N, ...)` - Optimization to keep batch dimensions adjacent to dense compute dimensions or for specific sharding strategies.
*   **Multi-Moduli (RNS)**:
    *   Native support for processing multiple Residue Number System (RNS) limbs concurrently.
*   **Sharding**:
    *   Implicit support via JAX `pmap` / `vmap` compatible structures, allowing distribution across devices by batch or modulus.

## 5. Summary Table

The following table maps the algorithmic concepts to their functional APIs in `ntt.py` and corresponding tests in `ntt_test.py`.

### Cyclic NTT

| Algorithm | Complexity | Reduction | Implementation API | Test Case (in `ntt_test.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **Original** | $O(N^2)$ | Python `%` | `ntt_original_form` | `test_C_NTT_None_Barrett_BatchFirst_none` |
| **Cooley-Tukey** | $O(N \log N)$ | Python `%` | `ntt_bit_reverse` | `test_br_C_NTT_None_Barrett_BatchFirst_none` |
| **4-Step** | $O(N^{1.5})$ | Python `%` | `ntt_four_step` | `test_4_step_C_NTT_None_Barrett_BatchFirst_none` |

### Negacyclic NTT

| Algorithm | Complexity | Reduction | Implementation API | Test Case (in `ntt_test.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **Cooley-Tukey** | $O(N \log N)$ | Python `%` | `ntt_negacyclic_bit_reverse` | `test_br_N_NTT_None_Barrett_BatchFirst_none` |
| **Cooley-Tukey** | $O(N \log N)$ | **Barrett** | `ntt_negacyclic_bit_reverse_jax` | `test_br_N_NTT_None_Barrett_BatchFirst_none_Jax` |
| **4-Step** | $O(N^{1.5})$ | Python `%` | `ntt_negacyclic_four_step` | `test_4_step_N_NTT_None_Barrett_BatchFirst_none` |
| **3-Step** | $O(N^{1.5})$ | Python `%` | `ntt_negacyclic_three_step` | `test_3_step_N_NTT_None_Barrett_BatchFirst_none` |

### Optimized / Hardware Implementations (Negacyclic)

| Category | Variant | Reduction | Implementation API | Test Case |
| :--- | :--- | :--- | :--- | :--- |
| **JAX / BAT** | 3-Step, Batched | **Barrett** | `ntt_three_step_bat_barrett_batch` | `test_3_step_N_NTT_SMB_Barrett_BatchFirst_none` |
| **JAX / BAT** | 3-Step, Multi-Mod | **Barrett** | `ntt_three_step_bat_barrett_multi_moduli` | `test_3_step_N_NTT_MMB_Barrett_BatchSecond_none` |
| **JAX / BAT** | 3-Step, Batched | **Montgomery** | `ntt_three_step_bat_montgomery_batch` | *Integrated in performance tests* |

### Configurable Knobs Reference

*   `q`: Prime modulus.
*   `psi`: Primitive $2N$-th root of unity (for negacyclic).
*   `omega`: Primitive $N$-th root of unity ($\omega = \psi^2$).
*   `r`, `c`: Row and column factors where $N = r \times c$.
*   `s_w`, `w`, `m`: Barrett reduction precomputed constants.
*   `tf_step*`: Twiddle factor matrices (precomputed control constants).
