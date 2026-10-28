# Design Document: Task 5.4b — Unified Fused Top-K Attention Autograd Kernel

Companion to Module Specification 04 (`04_triton_fused_kernel.md`). This
document is the plan of record for implementing
`FusedTopKAttentionFunction(torch.autograd.Function)` plus its wrapper in
`kernels/fused_topk_attn.py`. Written before any kernel code, per the
doc-first rule adopted after the rushed-draft failures of Task 5.4b
preparation (observations.md #2).

Status: DESIGN FINALIZED (2026-10-28, user confirmed) — split forward
(F1 + F2) is the plan of record. Kernel implementation may start.

Decision history: the user asked to check progress against the literal
"one fused forward" phrasing before finalizing. progress.md shows the
Phase-5 reference kernels were deliberately built as three separately
T4-validated stages (5.1 affinity, 5.2 top-k selector, 5.3 aggregate;
each record noting the other stages "remain subsequent tasks"), and
Task 5.4's record explicitly points at a unified end-to-end path as
future work. The split design therefore unifies the validated stages
rather than deviating from them: F1 adds the only fusion not yet built
(affinity + running top-k in registers, no dense N x M to HBM), and F2
reuses the accepted Task 5.3 aggregate kernel unchanged. The literal
single-kernel variant is possible only for small `d_v <= 32` and is a
deferred non-goal, not a change of plan.

---

## 1. Scope and Non-Goals

**Goal.** One autograd-integrated fused path for sparse top-k dictionary
cross-attention that never materializes the dense `N x M` affinity matrix
in HBM, with an analytical backward.

**In scope:**
- Forward: fused "tiled affinity -> in-register running top-k" kernel that
  writes only `(N, k)` selected scores + indices to HBM.
- Backward: analytical `dQ`, `dK`, `dV` computed exclusively on the sparse
  support (the saved top-k pairs), never reconstructing dense scores.
- Batch/head cyclorama: inputs `Q (B, H, N, d_k)`, `K (B, H, M, d_k)`,
  `V (B, H, M, d_v)`.
- float32 and float64 (for `torch.autograd.gradcheck`); float16 forward.
- `tests/test_triton_autograd.py`: forward parity vs reference
  `sparse_topk_attention`, gradcheck float64, max analytical gradient
  error < 1e-4.

**Non-goals (deferred, listed explicitly):**
- Router log-gate fusion into the affinity (the dual-domain module folds
  gates upstream today). Integration hook designed but not wired.
- Per-token adaptive `k_i` (needs a gather-style kernel; master restorer
  already collapses to a batch-mean budget — observations.md #19).
- fp16 backward (fp16 forward, fp32/fp64 backward compute).
- Causal/masking patterns (dictionary attention is dense over atoms).

---

## 2. Key Design Decision: Split Forward into 2 Kernels, Not 1

The naive reading of Spec 04 is "everything in one kernel." That fails a
z-register budget: accumulating `(B_r, d_v)` output registers while also
holding the `(B_r, next_pow2(k + B_c))` sorting array exceeds register
budget for the master model's shapes (`B_r = 64`, `k = 16..64`,
`d_v = 64`: 64 x 64 fp32 accumulator = 16 KB per program on top of the
sorter). Triton will spill or fail compile.

**Decision:**

```
Kernel F1  fused affinity + running top-k        -> (N, k) scores, (N, k) indices
Kernel F2  (reuse Task 5.3) tiled_topk_aggregate -> (N, d_v) output
```

- F1 is the new work. It eliminates the `N x M` HBM round trip — the
  actual Spec-04 objective — by fusing affinity and top-k in registers.
- F2 already exists, is validated on the T4 (GPU run 2026-10-28,
  152 passed), and re-gathers only from the tiny `(N, k)` scratch.
- The `(N, k)` scratch is NOT the dense matrix: for the master model
  (N = 4096 tokens, M = 256 atoms, k = 16) this is 16 KB vs 4 MB — a
  256x HBM reduction on the write side, and a full elimination of the
  dense-matrix read by F2 (8 MB read avoided).

Fusion *inside* F1 covers: Q@K^T tile GEMM, global-index folding for the
tie-breaker, running bitonic top-k merge, and the per-tile candidate
trim. Softmax and value gather stay in F2 (they run after ALL tiles, so
they cannot be in F1 anyway unless we hold a `(B_r, k, d_v)` gather —
the register blow-up above).

**Alternative rejected:** fusing softmax into F1 across tiles requires
online softmax over a support that is still being discovered (a
lower-ranked future tile can replace previously weighted entries),
forcing a rescale of the `(B_r, d_v)` accumulator we decided not to hold.
The softmax-over-final-top-k semantics make the end-of-loop
normalization in F2 exact, not approximate.

---

## 3. Shapes, Layout, Grid

- Inputs contiguous in last dim; kernel receives `Q, K, V` pointers as
  `(B*H, N, d_k)` / `(B*H, M, d_k)` / `(B*H, M, d_v)` (python flattens).
- Grid F1: `(ceil(N / B_r), B*H)` — `B_r = 64` programs, each owns a
  query block and streams all `M` keys.
- Key tile `B_c = 64`; feature chunk `f = 32` (matching Task 5.2 tiles);
  `input_precision="ieee"` on `tl.dot` (validated choice).
- `k` any `1 <= k <= M`, not required to be a power of two (padding used).

---

## 4. Kernel F1: Fused Affinity + Running Top-k

Per program (one query block, one (b,h) slice):

State (per row of the block):
- `buf_val  (B_r, P)` — running top-k scores, padded with `-inf`
  (P = next_pow2(k + B_c) ensures the concatenated candidate array is a
  power-of-two register array). Holds globally sorted values at
  positions `0..k-1` between tiles... NOT sorted across the full P —
  see merge below.
- `buf_idx  (B_r, P)` — matching global column indices.

Loop over key tiles `t = 0 .. M/B_c`:
1. Local GEMM: `S_tile (B_r, B_c) = Q_tile @ K_tile^T` chunked over
   `d_k` in 32-wide feature chunks, fp32 accumulate (fp64 case: fp64).
2. **Index folding:** real global index = `t*B_c + local`. Padded
   candidates beyond `M` get score `-inf` AND index `0` — safe because
   softmax weight of a `-inf` logit is exactly 0 (IEEE), so a padded
   entry can never contribute gathered mass; the tie-break comparator
   ordering `(value, index)` also keeps padded slots last.
3. **Merge:** build candidate `C = bitonic_concat(buf (P/2 packed?), tile)`
   — concretely: `C = join(buf_val, S_tile)` padded to P, then run the
   bitonic sort `P * log2(P) * (log2(P)+1)/2` compare-swaps with the
   SAME comparator as `_bitonic_topk` (descending by value; tie-break
   ascending index; NaN sorts ahead — retained for parity with 5.2).
   Truncate/keep full array as the new running buffer.
   - Cost per tile per row: `P log^2 P`. For P=128: 128 x 28 ≈ 3.6k
     scalar compare-swaps, vectorized over B_c lanes in registers.
     With M=256 (4 tiles) and N=4096: ~59M compare-ops per (b,h) —
     far below the GEMM cost (~537 MFLOP per (b,h)); acceptable.
4. After the loop, positions `0..k-1` of `buf` ARE the global top-k
   (proof: invariant — after step 3, buf holds the top-P of all keys
   seen so far, since each merge sorts the union; induction over tiles).

Store `top_scores (N, k)`, `top_indices (N, k) int32`. No dense S ever
leaves registers.

**Register inventory (fp32 lanes, per program):**
`Q_tile (64, 32)` reused per chunk + `buf (64, 128)` + `S_tile (64, 64)`
≈ 64 x 128 fp32 sort array dominates (~32 KB); with `num_warps=8` this
fits SRAM/registers; if compile shows spills, fallback constants:
`B_r = 32` (grid doubles, merging unchanged). Tuning knob recorded.

---

## 5. Kernel F2 (existing, unchanged)

`tiled_topk_aggregate(top_scores, top_indices, values)` from Task 5.3:
online softmax in fp32 over the k selected logits, gather V rows
per 32-logit block, accumulate `(N, d_v)` in fp32. Already validated
(bit-exact vs torch reference on T4).

---

## 6. Backward Design (new kernels, sparse support only)

Saved tensors: `Q, K, V, top_scores, top_indices, L (N,)` where
`L_i = m_i + log(sum_j exp(S_ij - m_i))` over the k selected (natural
log; computed in F2-adjacent helper or a tiny kernel — decision: compute
in python from `top_scores` (k is small) before backward; saves a kernel).

Downstream: `O_i = sum_j P_ij V_j`, `P_ij = exp(S_ij - L_i)` (renormalized
softmax restricted to the top-k support — identical semantics to the
reference `sparse_topk_attention`).

Standard softmax-attention backward with `G_ij = dO_i · V_j`:

```
delta_i = sum_j P_ij G_ij            (= dot(dO_i, O_i), mathematically)
dS_ij   = P_ij (G_ij - delta_i)      for (i,j) in support ONLY
dQ_i    = sum_j dS_ij K_j
dK_j   += dS_ij Q_i                  (scatter, atomics across query rows)
dV_j   += P_ij dO_i                  (scatter, atomics across query rows)
```

All three gradients use ONLY saved top-k pairs. No dense S recompute is
needed anywhere — the spec's "recompute local softmax P = exp(S - L)"
is satisfied: we recompute P from the SAVED `(N, k)` scores via exp in
SRAM, and the GEMMs are block-sparse gathers.

### Kernel B1: dQ (per query block)
Grid `(ceil(N / B_r), B*H)`; per query row i, loop over k in `k-block`
chunks (16):
- load `I (chunk)` indices, `S (chunk)` saved scores, `P = exp(S - L_i)`
- `G = (B_r, kc, d_v-chunk)` gather-dotted against `dO`: partial dots
  over `d_v` feature chunks accumulate a `(B_r, kc)` scalar; second pass
  computes `dS`; third pass gathers `K[I] (`B_r, kc, f`-chunks)` and
  weighted-sums into `dQ (B_r, f)` accumulator — registers
  `64 x 32 x [kc=16] = 32k` loads per chunk, `(B_r, f)=2048` fp32
  accumulator — fits.
No atomics; exact deterministic writes.

### Kernel B2: dK/dV (per query block, atomic scatter)
Grid `(ceil(N / B_r), B*H)`; per program:
- recompute `P` and `dS` exactly as in B1 (same helper math),
- `atomic_add` into `dV[I] += P * dO` and `dK[I] += dS * Q` via gathered
  row pointers `dV_ptr + I[:, None] * d_v + f[None, :]`.
Atomic fp32 (fp64 for gradcheck; CUDA fp64 atomics are supported, slow
but tiny shapes). Deterministic per-run ordering is not guaranteed
across query blocks — the resulting ±1-2 ulp wobble is orders below the
1e-4 gradcheck target, and we validate with fixed seeds.

Note: `delta_i` (= dot(dO_i, O_i)) requires O_i — recomputed cheaply in
python before backward launches: `O` is already computed in forward;
pass it as a saved tensor. (Cheaper than multiplying P·dO through a
kernel, and exact.)

### Zero-traffic accounting (vs dense backward)
- Saved-from-forward reads: `(N,k)` scores + indices + `L`: O(Nk).
- B1 reads: `dO (N, d_v)`, gathered `K`, `V` rows O(Nk d).
- B2 reads: same atomic-scatter sources; writes hits only touched atoms.
- Never reads or writes `N x M` anywhere. Spec requirement satisfied.

---

## 7. Autograd Wrapper

```python
class FusedTopKAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, k):          # returns (B,H,N,d_v) fp32 accum
        # flatten B*H, launch F1, F2; ctx.save Q,K,V,I,S,L? (O not needed: delta = dot(dO,O) recomputed via O saved)
    @staticmethod
    def backward(ctx, dO):                 # launch B1 (dQ) + B2 (dK,dV)

def fused_topk_attention(Q, K, V, k, selection="sparse"):  # public wrapper
    # validation mirroring sparse_topk_attention (k bounds, contiguity,
    # device=CUDA, dtype table); falls back to reference implementation
    # on CPU with a warning suppressed by default (tests target GPU)
```

- Return parity with `sparse_topk_attention`: returns output; the
  `(N,M)`-shaped dense `weights`/masked_scores side products do NOT
  exist in the fused path (that is the point), so the fused path is
  exposed as a separate function, not a drop-in of the reference
  signature.
- Numerical guard: reject inputs containing `-inf` scores?? Q·K^T
  produces finite values for finite inputs; document that NaN/inf
  inputs are unsupported in the fused path (bitonic NaN handling is
  retained only for parity).

---

## 8. Validation Plan (on T4, via the notebook workflow)

Order fixed; each step gates the next:

1. **Compile smoke:** kernel F1 compiles for (fp32, k=16, M=256, N=512),
   runs, output finite.
2. **Forward parity:** `(actual, expected)` vs reference
   `sparse_topk_attention` on random distinct-value scores:
   `max abs err <= 1e-5` (fp32) across grids:
   `(N, M, k, d_v) in {(1,13,1,24), (7,64,5,67), (65,129,17,33),
   (3,73,73,7), (4096, 256, 16, 64)}`.
3. **fp64 forward parity** (tiny shapes, fp64 dots in "ieee" mode).
4. **gradcheck float64:** `torch.autograd.gradcheck` on
   `(B=1, H=1, N=8, M=12, k=5, d_k=4, d_v=3)` and one larger
   `(N=16, M=32, k=8, d=8)`; `eps=1e-6, atol=1e-4`.
5. **Analytical error bound:** compare fused backward vs
   autograd-of-reference on fp64 inputs; `max err < 1e-4` (target is
   ~1e-7; the 1e-4 is the spec gate).
6. **Tie/-inf/padding cases:** k=M (aggregate path degenerates to dense
   softmax), padded candidates contribute exactly 0 mass, duplicate
   values resolve by ascending index consistently across backends.
7. **Micro-benchmark** (Task 5.4 prep integration): fused vs reference
   path on `(B=2, H=4, N=4096, M=256, k=16)`; record both ms and
   `torch.cuda.max_memory_allocated` deltas.

Tests land in `tests/test_triton_autograd.py` (per spec path) and use
the standard `skipif not torch.cuda.is_available()` pattern.

---

## 9. Risks and Open Items

1. **Register pressure on the P=128 sorter.** Mitigation: `B_r = 32`
   fallback; `num_warps = 8`; compile-time constexpr sweep on T4.
2. **fp64 tl.dot support.** Triton lowers fp64 dot on sm75 slowly but
   correctly (SM 75 T4); gradcheck uses tiny shapes, acceptable.
3. **Atomic fp64 determinism.** Runs may wobble at 1-2 ulp; gradcheck
   seeds inputs so comparisons are run-stable.
4. **Gather atomic scatter correctness with duplicate keys across
   blocks:** atomic_add is per-element correct; no dedup needed since
   indices are unique within a row.
5. ~~**Open question for user review:** the split-forward design (F1+F2)
   deviates from the literal "one fused forward" phrasing in
   observations.md #2~~ RESOLVED 2026-10-28: user reviewed progress.md
   history (reference kernels built as separate validated stages 5.1/
   5.2/5.3; Task 5.4 pointing at a unified end-to-end path) and
   confirmed the split forward. Single-kernel variant stays a deferred
   non-goal for small `d_v <= 32`.
6. **Router-gate fusion hook:** designed as an optional `gate_logits`
   tensor `(B,H,M)` added to tile scores inside F1's GEMM epilogue;
   trivial to add once parity is proven (kept out of 5.4b scope).
