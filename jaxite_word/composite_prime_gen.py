"""Port of OpenFHE's CompositePrimeModuliGen for generating compatible extra primes.

Generates Q-tower primes with oscillating pair products that maintain
recursive SF stability, matching OpenFHE's ckksrns-parametergeneration.cpp.
"""

import math

try:
    from sympy import isprime
except ModuleNotFoundError:
    try:
        from . import util
    except ImportError:
        import util

    isprime = util.is_prime_deterministic


def _first_prime(n_bits, m):
    """Smallest prime ≡ 1 (mod m) with n_bits bits."""
    q = (1 << n_bits)
    r = q % m
    q_new = q + 1 - r
    if r > 0:
        q_new += m
    while not isprime(q_new):
        q_new += m
    return q_new


def _next_prime(q, m):
    """Next prime ≡ 1 (mod m) after q."""
    q_new = q + m
    while not isprime(q_new):
        q_new += m
        if q_new < q:  # overflow
            raise ValueError("Overflow in NextPrime")
    return q_new


def _prev_prime(q, m):
    """Previous prime ≡ 1 (mod m) before q."""
    q_new = q - m
    while q_new > 1 and not isprime(q_new):
        q_new -= m
        if q_new > q or q_new <= 1:
            raise ValueError("Underflow in PreviousPrime")
    if q_new <= 1:
        raise ValueError("No previous prime found")
    return q_new


def composite_prime_gen(
    composite_degree: int,
    num_primes: int,
    first_mod_size: int,
    scaling_mod_size: int,
    cycl_order: int,
    register_word_size: int = 31,
):
    """Generate Q-tower primes matching OpenFHE's CompositePrimeModuliGen.

    Args:
        composite_degree: Number of primes per composite group (cd, typically 2).
        num_primes: Total number of Q-tower primes to generate.
        first_mod_size: Bit size for the first (largest) prime pair.
        scaling_mod_size: Bit size for scaling factor prime pairs.
        cycl_order: Cyclotomic order (m = 2 * ring_dim).
        register_word_size: Maximum bits per prime (default 31 for uint32).

    Returns:
        List of num_primes primes in OpenFHE's exact ordering.
    """
    cd = composite_degree
    moduli_q = [0] * num_primes
    record = set()

    # Phase 1: Generate last cd primes (scaling factor base) from scalingModSize
    rem_bits = scaling_mod_size
    for d in range(1, cd + 1):
        q_bit_size = math.ceil(rem_bits / (cd - d + 1))
        q = _first_prime(q_bit_size, cycl_order)
        q = _prev_prime(q, cycl_order)
        while (math.log2(q) > register_word_size or
               math.log2(q) > q_bit_size or
               q in record):
            q = _prev_prime(q, cycl_order)
        moduli_q[num_primes - d] = q
        record.add(q)
        rem_bits -= math.ceil(math.log2(q))

    # Phase 2: Generate subsequent groups with oscillating pair products
    sf = float(moduli_q[num_primes - 1])
    for d in range(2, cd + 1):
        sf *= float(moduli_q[num_primes - d])

    flag = True
    i = num_primes - cd
    while i >= 2 * cd:
        # Compute target scaling factor for this group
        sf = sf * sf
        for d in range(cd):
            if moduli_q[i + d] != 0:
                sf /= float(moduli_q[i + d])

        sf_sqrt = sf ** (1.0 / cd)
        sf_int = int(round(sf_sqrt))
        sf_rem = sf_int % cycl_order

        n_prev = math.ceil(cd / 2)
        n_next = cd - n_prev

        q_prev = [0] * n_prev
        q_next = [0] * n_next
        q_current = set()
        prime_product = 1

        if not flag:
            # Branch B: qPrev first (descending), then qNext (ascending)
            for step in range(n_prev):
                q = sf_int - sf_rem + 1 - cycl_order
                while True:
                    q = _prev_prime(q, cycl_order)
                    if (math.log2(q) <= register_word_size and
                        q not in record and q not in q_current):
                        break
                q_prev[step] = q
                q_current.add(q)
                prime_product *= q

            for step in range(n_next):
                q = sf_int - sf_rem + 1 + cycl_order
                while True:
                    q = _next_prime(q, cycl_order)
                    if (math.log2(q) <= register_word_size and
                        q not in record and q not in q_current):
                        break
                q_next[step] = q
                q_current.add(q)
                prime_product *= q

            # Adjust: if product > sf, decrease
            if n_next > 0:
                q_adj = q_next[n_next - 1]
                while prime_product > sf:
                    q_new = _prev_prime(q_adj, cycl_order)
                    while q_new in record or q_new in q_current:
                        q_new = _prev_prime(q_new, cycl_order)
                    prime_product = prime_product // q_adj * q_new
                    q_current.discard(q_adj)
                    q_adj = q_new
                    q_current.add(q_adj)
                q_next[n_next - 1] = q_adj

            flag = True
        else:
            # Branch A: qNext first (ascending), then qPrev (descending)
            for step in range(n_next):
                q = sf_int - sf_rem + 1 + cycl_order
                while True:
                    q = _next_prime(q, cycl_order)
                    if (math.log2(q) <= register_word_size and
                        q not in record and q not in q_current):
                        break
                q_next[step] = q
                q_current.add(q)
                prime_product *= q

            for step in range(n_prev):
                q = sf_int - sf_rem + 1 - cycl_order
                while True:
                    q = _prev_prime(q, cycl_order)
                    if (math.log2(q) <= register_word_size and
                        q not in record and q not in q_current):
                        break
                q_prev[step] = q
                q_current.add(q)
                prime_product *= q

            # Adjust: if product < sf, increase
            if n_prev > 0:
                q_adj = q_prev[n_prev - 1]
                while prime_product < sf:
                    q_new = _next_prime(q_adj, cycl_order)
                    while q_new in record or q_new in q_current:
                        q_new = _next_prime(q_new, cycl_order)
                    prime_product = prime_product // q_adj * q_new
                    q_current.discard(q_adj)
                    q_adj = q_new
                    q_current.add(q_adj)
                q_prev[n_prev - 1] = q_adj

            flag = False

        # Store this group
        for d in range(n_prev):
            moduli_q[i - 1 - d] = q_prev[d]
        for d in range(n_next):
            moduli_q[i - 1 - n_prev - d] = q_next[d]

        for d in range(1, cd + 1):
            record.add(moduli_q[i - d])

        i -= cd

    # Phase 3: Generate first cd primes (P chain) from firstModSize
    rem_bits = first_mod_size
    for d in range(1, cd + 1):
        q_bit_size = math.ceil(rem_bits / (cd - d + 1))
        q = _first_prime(q_bit_size, cycl_order)
        q = _prev_prime(q, cycl_order)
        while (math.log2(q) > q_bit_size or
               math.log2(q) > register_word_size or
               q in record):
            q = _prev_prime(q, cycl_order)
        moduli_q[d - 1] = q
        record.add(q)
        rem_bits -= q_bit_size

    return moduli_q


def _ntt_primes_in_range(lo, hi, cycl_order):
    """Return sorted list of primes p in [lo, hi] with p ≡ 1 (mod cycl_order)."""
    # Align to first candidate ≡ 1 (mod cycl_order) >= lo
    start = lo + (cycl_order - (lo % cycl_order)) % cycl_order
    if start % cycl_order != 1:
        start = start - (start % cycl_order) + 1
    if start < lo:
        start += cycl_order
    primes = []
    q = start
    while q <= hi:
        if isprime(q):
            primes.append(q)
        q += cycl_order
    return primes


def compute_recursive_sf_drift(moduli_q, composite_degree=2):
    """Compute max recursive SF drift (%) for a prime chain.

    Returns (max_drift_pct, sf_table) where sf_table[k] is the recursive SF
    at prime index k (for k % cd == 0).
    """
    cd = composite_degree
    nq = len(moduli_q)
    sf = [0.0] * nq
    sf[0] = 1.0
    for i in range(cd):
        sf[0] *= float(moduli_q[nq - cd + i])
    sf_base = sf[0]
    for k in range(1, nq):
        if k % cd == 0:
            prev = sf[k - cd]
            pair = 1.0
            for j in range(cd):
                pair *= float(moduli_q[nq - k + j])
            sf[k] = prev * prev / pair
        else:
            sf[k] = 1.0
    max_drift = 0.0
    for k in range(0, nq, cd):
        if sf[k] > 0 and math.isfinite(sf[k]):
            drift = abs(sf[k] / sf_base - 1.0) * 100
            max_drift = max(max_drift, drift)
    return max_drift, sf


def composite_prime_gen_low_drift(
    composite_degree: int,
    num_primes: int,
    first_mod_size: int,
    scaling_mod_size: int,
    cycl_order: int,
    register_word_size: int = 31,
    num_start_candidates: int = 80,
    search_radius: int = 300,
):
    """Generate Q-tower primes with near-zero recursive SF drift.

    Uses global pair-product optimization: for each pair in the chain,
    searches for two NTT-friendly primes whose product is closest to the
    target that minimizes drift at that level. Pairs may use non-uniform
    prime sizes (e.g., 29-bit + 31-bit for cycl_order=8192) as long as
    each prime satisfies the register_word_size constraint.

    This is designed for large cycl_order (≥8192) where the standard
    OpenFHE-style oscillating generator produces unacceptable drift
    (>0.1%) due to sparse NTT-friendly primes.

    For cycl_order ≤ 1024, use the standard composite_prime_gen which
    matches OpenFHE exactly.

    Args:
        composite_degree: Number of primes per group (cd, typically 2).
        num_primes: Total number of Q-tower primes.
        first_mod_size: Bit size for the first (largest) prime pair.
        scaling_mod_size: Bit size for scaling factor prime pairs.
        cycl_order: Cyclotomic order (m = 2 * ring_dim).
        register_word_size: Maximum bits per prime (default 31).
        num_start_candidates: Number of starting base pairs to try.
        search_radius: Number of pair candidates to evaluate per level.

    Returns:
        List of num_primes primes ordered as: [first_mod_pair, ..., scaling_pairs, base_pair].
    """
    if composite_degree != 2:
        raise NotImplementedError(
            "Low-drift generator currently supports composite_degree=2 only."
        )
    cd = composite_degree
    num_scaling_pairs = num_primes // cd - 1  # subtract first_mod pair
    max_prime = 2 ** register_word_size

    # Step 1: Build pool of NTT-friendly primes.
    # Allow primes from 2^(scaling_mod_size/cd - 2) to 2^register_word_size
    # to maximize pair-product coverage.
    min_prime_bits = max(scaling_mod_size // cd - 2, 20)
    prime_pool = _ntt_primes_in_range(2 ** min_prime_bits, max_prime, cycl_order)
    prime_pool.sort()

    # Step 2: Build candidate pair list.
    # For target pair product ≈ 2^scaling_mod_size, find pairs near that target.
    target_pair_product = 2.0 ** scaling_mod_size

    import bisect

    def _find_pair_candidates(target, pool, used, max_err=0.01, max_count=5000):
        """Find pairs from pool with product near target, excluding used primes."""
        candidates = []
        for p1 in pool:
            if p1 in used:
                continue
            target_p2 = target / p1
            idx = bisect.bisect_left(pool, target_p2)
            for di in range(-3, 4):
                j = idx + di
                if 0 <= j < len(pool):
                    p2 = pool[j]
                    if p2 != p1 and p2 not in used:
                        prod = p1 * p2
                        rel_err = abs(prod - target) / target
                        if rel_err < max_err:
                            candidates.append((p1, p2, prod, rel_err))
            if len(candidates) > max_count:
                break
        candidates.sort(key=lambda x: x[3])
        return candidates

    # Step 3: Select base pair (the last cd primes in the chain).
    # Try multiple base pairs and pick the one that yields the lowest max drift.
    base_candidates = _find_pair_candidates(
        target_pair_product, prime_pool, set(), max_err=0.001
    )[:num_start_candidates]

    best_chain = None
    best_max_drift = float("inf")

    for base_entry in base_candidates:
        bp1, bp2, bp_prod, _ = base_entry
        used = {bp1, bp2}
        chain = [(bp1, bp2, bp_prod)]
        sf_rec = float(bp_prod)
        sf_base = sf_rec
        max_drift_so_far = 0.0
        success = True

        for k in range(1, num_scaling_pairs):
            # Target pair product for drift[k] = 0: pair[k] = sf_rec^2 / sf_base
            target_k = sf_rec * sf_rec / sf_base

            # Find closest available pair to target_k
            best_pair = None
            best_abs_drift = float("inf")

            candidates_k = _find_pair_candidates(
                target_k, prime_pool, used, max_err=0.01
            )[:search_radius]

            for p1, p2, prod, _ in candidates_k:
                new_sf = sf_rec * sf_rec / prod
                new_drift = abs(new_sf / sf_base - 1.0)
                if new_drift < best_abs_drift:
                    best_abs_drift = new_drift
                    best_pair = (p1, p2, prod)

            if best_pair is None:
                success = False
                break

            chain.append(best_pair)
            used.add(best_pair[0])
            used.add(best_pair[1])
            sf_rec = sf_rec * sf_rec / float(best_pair[2])
            max_drift_so_far = max(max_drift_so_far, abs(sf_rec / sf_base - 1.0))

        if success and len(chain) >= num_scaling_pairs and max_drift_so_far < best_max_drift:
            best_max_drift = max_drift_so_far
            best_chain = chain

    if best_chain is None:
        raise ValueError(
            "Failed to find a low-drift prime chain. "
            "Try increasing num_start_candidates or search_radius."
        )

    # Step 4: Generate first_mod pair (positions 0, 1).
    used_primes = set()
    for p1, p2, _ in best_chain:
        used_primes.add(p1)
        used_primes.add(p2)

    target_first = 2.0 ** first_mod_size
    first_candidates = _find_pair_candidates(
        target_first, prime_pool, used_primes, max_err=0.01
    )
    if not first_candidates:
        raise ValueError("Cannot find first_mod pair from remaining prime pool.")
    first_pair = first_candidates[0]

    # Step 5: Assemble the full chain in OpenFHE ordering.
    # OpenFHE ordering: [first_mod_0, first_mod_1, scaling_pairs..., base_pair]
    # Scaling pairs are stored in REVERSE order from how we generated them
    # (base pair is last cd primes, next pair is second-to-last cd primes, etc.)
    moduli_q = [0] * num_primes

    # First mod pair at positions 0, 1
    moduli_q[0] = first_pair[0]
    moduli_q[1] = first_pair[1]

    # Base pair at positions nq-2, nq-1
    moduli_q[num_primes - 2] = best_chain[0][0]
    moduli_q[num_primes - 1] = best_chain[0][1]

    # Remaining scaling pairs: chain[1], chain[2], ... chain[num_scaling_pairs-1]
    # chain[k] corresponds to the pair at positions (nq - 2*(k+1) - 1, nq - 2*(k+1))
    # i.e., chain[1] → positions nq-4, nq-3
    #        chain[2] → positions nq-6, nq-5
    #        etc.
    for k in range(1, num_scaling_pairs):
        pos_high = num_primes - 2 * (k + 1) + 1  # e.g., nq-3 for k=1
        pos_low = num_primes - 2 * (k + 1)        # e.g., nq-4 for k=1
        moduli_q[pos_low] = best_chain[k][0]
        moduli_q[pos_high] = best_chain[k][1]

    return moduli_q


if __name__ == "__main__":
    # Generate primes matching OpenFHE's D17 config
    # cd=2, ringDim=512, firstModSize=61, scalingModSize=60, depth=17
    # numPrimes = (depth+1)*cd = 36
    primes = composite_prime_gen(
        composite_degree=2,
        num_primes=36,
        first_mod_size=61,
        scaling_mod_size=60,
        cycl_order=512,  # 2 * degree = 2 * 256
        register_word_size=31,
    )

    print(f"Generated {len(primes)} primes:")
    for i, p in enumerate(primes):
        print(f"  Q[{i}] = {p} ({math.log2(p):.3f} bits)")

    # Verify pair products
    print("\nPair products:")
    for i in range(0, len(primes), 2):
        prod = primes[i] * primes[i + 1]
        print(f"  Q[{i}]*Q[{i+1}] = {prod} ({math.log2(prod):.3f} bits)")

    # Check recursive SF stability
    cd = 2
    max_drift, sf = compute_recursive_sf_drift(primes, cd)
    nq = len(primes)
    sf_base = sf[0]
    print("\nRecursive SF stability:")
    for k in range(0, nq, cd):
        level = k // cd
        if sf[k] > 0:
            drift = abs(sf[k] / sf_base - 1.0) * 100
            print(f"  level={level:2d} k={k:2d} sf={sf[k]:.6e} "
                  f"({math.log2(sf[k]):.3f} bits) drift={drift:.6f}%")
    print(f"\nMax drift: {max_drift:.6f}%")

    # Compare with OpenFHE's actual D17 primes
    from util import CROSS_Q_TOWERS
    openfhe = CROSS_Q_TOWERS[:36]
    match = primes == openfhe
    print(f"\nMatch OpenFHE D17 exactly: {match}")
    if not match:
        for i in range(36):
            if primes[i] != openfhe[i]:
                print(f"  DIFF at Q[{i}]: generated={primes[i]} openfhe={openfhe[i]}")

    # Now generate for cycl_order=8192 (N=4096): compare original vs optimized
    print("\n" + "=" * 70)
    print("=== cycl_order=8192 (N=4096): Original vs Low-Drift Generator ===")
    print("=" * 70)

    primes_orig = composite_prime_gen(2, 48, 61, 60, 8192, 31)
    drift_orig, sf_orig = compute_recursive_sf_drift(primes_orig, 2)
    print(f"\nOriginal generator: max drift = {drift_orig:.6f}%")

    primes_opt = composite_prime_gen_low_drift(2, 48, 61, 60, 8192, 31)
    drift_opt, sf_opt = compute_recursive_sf_drift(primes_opt, 2)
    print(f"Low-drift generator: max drift = {drift_opt:.8f}%")
    print(f"Improvement: {drift_orig/drift_opt:.0f}x")

    print("\nLow-drift chain primes:")
    for i, p in enumerate(primes_opt):
        print(f"  Q[{i:2d}] = {p:>12d} ({math.log2(p):>7.3f} bits)")

    print("\nLow-drift pair products:")
    for i in range(0, len(primes_opt), 2):
        prod = primes_opt[i] * primes_opt[i + 1]
        print(f"  Q[{i:2d}]*Q[{i+1:2d}] = {prod} ({math.log2(prod):.6f} bits)")

    print("\nLow-drift recursive SF:")
    sf_base_opt = sf_opt[0]
    for k in range(0, len(primes_opt), 2):
        if sf_opt[k] > 0 and math.isfinite(sf_opt[k]):
            drift = abs(sf_opt[k] / sf_base_opt - 1.0) * 100
            print(f"  k={k:2d} sf={sf_opt[k]:.10e} drift={drift:.8f}%")
