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
22. Split-forward design (F1 fused affinity+topk + F2 existing
    aggregate) FINALIZED by user review on 2026-10-28: checking
    progress.md showed the Phase-5 reference kernels were always
    separate T4-validated stages (5.1/5.2/5.3), so the "one fused
    forward" phrasing was the assistant's own note (#2), not a spec
    constraint. Implementation may start against the finalized split.

## Fused autograd kernel T4 debugging (2026-10-29, 2026-11-01)
23. First T4 run of the 5.4b fused autograd tests: 12 failed /
    158 passed - a single cudaErrorIllegalAddress in the F1 launch
    poisoned the CUDA context so every later CUDA test failed with the
    same error. Root cause: the F1/B1/B2 launches passed
    `top_scores.stride(2)` where kernel signatures expect the ROW
    stride of `top_indices` (i.e. `top_indices.stride(1)`), so index
    rows collapsed onto each other and torch.empty garbage indices
    drove OOB gathers. Lesson: audit every kernel launch argument
    positionally against the signature before first CUDA run; a
    poisoned context masks all later failures as duplicates of the
    first.
24. Second T4 run after stride fixes: 167 passed / 3 failed - the three
    failing fp64 backward tests crashed the Triton COMPILER (not the
    context): `_backward_dkv`'s body referenced
    QUERY_BH_STRIDE / QUERY_ROW_STRIDE which its signature never
    declared, while the hand-written launch 'matched' the broken
    signature. Lesson: compile-coverage must include the DTYPE axis; a
    kernel that compiles in fp32 can still fail in fp64-only paths
    (gradcheck is the first fp64 consumer).
25. Third T4 run (commit 36331e4): 170 passed in 7.88s on Tesla T4
    (torch 2.11.0+cu130, triton 3.6.0) - the fused 5.4b forward parity
    (5 grids incl. k=M and N=4096), fp64 parity, double gradcheck
    shapes, analytical-vs-reference backward (max err < 1e-4 gate),
    padded-lane zero-mass and k=M edge cases all pass on-device. Local
    suite stays 139 passed / 31 skipped.
26. Colab session resume notes: re-opening the notebook after the
    Google sign-in cookie expired redirected the tab to
    accounts.google.com; navigating directly to the notebook URL after
    sign-in restores the editable copy with runtime type (T4) intact.
    Runtime connect + Run all can be driven headlessly via
    `colab.global.notebook.connectToKernel()` / `runAll()`;
    per-cell results are read from the notebook DOM (not the model
    API, which does not expose outputs in this Colab version), while
    cell SOURCE edits work through `model.textModel.setValue`.
