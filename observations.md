# Engineering Observations Log (SparseDict)

Accumulated issues / observations recorded during Tasks 5.4b preparation, 6.1-6.3.
Keep this file updated as new observations arise.

## Kernel and roadmap (Task 5.4b)
1. Task 5.4b is the only remaining Phase-5 item; progress.md has no 5.4b record yet.
2. The unified fused kernel needs a FlashAttention-style design: one fused forward
   (tiled affinity + in-register bitonic top-k merge + gathered online softmax) and
   an analytical recompute backward that never materializes the dense N x M score
   matrix in HBM. Rule adopted: write/validate the design doc first, then code.
3. Triton work is not CPU-testable; 5.4b validation must run on the Colab T4
   (compile, forward correctness, then float64 gradcheck per spec).

## Haar / dual-domain (Task 6.2)
4. Tautological shape check in the first draft forward (comparing fields of the
   same tensor); replaced with a real channels check.
5. blocks[..., 0, 0] collapses the last two dims (wb, 2), not the 2x2 block-pair
   dims; subband extraction must index blocks[:, :, :, r, :, c].
6. Constant gate vectors are mathematically invariant under top-k renormalization:
   log(0.5) shifts every expert equally and softmax re-cancels it; routing tests
   need nonuniform (driven) gate tensors.
7. Zero gates must be masked to -inf after clamp-min finfo.tiny log-folding; the
   additive gate fold must happen on the 4D score tensor before flattening.
8. Mixing float64 banks with float32 module weights makes einsum fail
   (expected scalar type Float but found Double).

## Entropy gate (Task 6.3)
9. A constant row through the vocab projection is not uniform over the vocab axis
   (row-sums spread); its entropy is constant-with-small-variance. The
   constant -> 0% spec check relies on strict H_i > tau_H with sigma = 0.
10. Fallback gate=ones broke constant/0%; replaced first with current-batch
    stats, then with the DEGENERATE GUARD (RESOLVED): a zero-spread entropy
    batch (max == min, bitwise) routes everything to the bypass. Chosen after
    a 1-ulp mean-rounding flake: mean of 32 identical entropies can round up
    or down, making the strict H_i > tau_H comparison unreliable whenever
    sigma is vanishingly small. User accepted this rule by review default.
11. The k_max <= V - 1 bound was assistant-invented and dropped.
12. shannon_entropy expects vocab-axis logits (last dim = V).

## Environment quirks
13. Literal path C:\Users\Kalpeth\... intermittently fails (typo/case mismatch);
    env:USERPROFILE expansion is reliable.
14. Suite runtime ~10 s for 121 tests locally (CPU).

## Test-design notes
15. Differentiability tests assert every parameter receives a finite non-NaN grad;
    for routed banks this works because routing varies across tokens.
16. Acceptance values: coherence < 1e-6 orthogonal, > 5.0 collinear
    (recorded 3.35e-31 / 3969.0); Welch bound uses M >= d.

## Master restorer (Task 6.4)
17. MutualCoherenceLoss needs M >= d per matrix and returns a vector for
    batched input; the dual-domain dictionary (4, M_e, C) must be flattened
    to (4*M_e, C) first so the loss returns a single scalar (4*M_e = 128
    covers any d <= 128).
18. The entropy gate participates in the master model as a HARD gate
    (H_i > tau_H boolean), so EntropyGate parameters receive no gradient;
    recorded as a deliberate STE-less limitation - a soft/STE version would
    be needed for the gate itself to be trained.
19. sparse_topk_attention takes a single scalar k, so per-token adaptive
    k_i is collapsed to a batch-mean integer budget (_effective_top_k);
    true per-token k needs a dedicated gather kernel.
20. models/__init__.py had duplicate MoETopKRestorer import/__all__ entries
    (dead duplication from earlier edits); rewritten cleanly.

## GPU validation run (Colab T4, 2026-10-28)
21. Full suite on Tesla T4: 152 passed in 17.16s (torch 2.11.0+cu130,
    triton 3.6.0, Python 3.13.16, pytest 8.4.2); all 19 CPU-skipped
    Triton tests executed and passed — Phase-5 reference kernels plus
    straight-through top-k gradcheck confirmed on CUDA for the first
    time. Browser-driving friction noted: Colab's "Run all"/"Copy to
    Drive" buttons resisted synthetic clicks (shadow-DOM refs stale);
    the commit-then-clone notebook workflow bypassed both.
