# Engineering Observations Log (SparseDict)

Accumulated issues / observations recorded during Tasks 5.4b, 6.1-6.3.
Keep this file updated as new observations arise.

## Kernel and roadmap (Task 5.4b)
1. The original Task 5.4b F1+F2 autograd path passed T4 validation on
   2026-11-01. The user selected Option 3 as the follow-up optimization;
   the required latency and memory targets remain unmet.
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

## Task 5.4 end-to-end benchmark + GPU diagnosis (2026-11-01)
27. Added `tests/bench_fused_e2e.py` (fused vs dense PyTorch reference with
    parity gate, CUDA-event medians, peak-memory deltas; saves
    `results/fused_e2e_benchmark.json`). First T4 run: fused is SLOWER
    end-to-end (0.18x/0.35x/0.41x at N=1024/4096/16384) but ~80x less
    memory (3.0 vs 258.5 MiB at N=4096). The 2.5x gate of Task 5.4 is NOT
    met by the current F1 kernel; recorded honestly, not faked.
28. Cross-run GPU timing on the T4 is unreliable for absolute numbers
    (throttling after heavy phases): the same `(32,64)` config measured
    5.44 ms (bench) vs 15.7-21.3 ms (sweep cells) for identical shapes.
    Every comparison used to make a decision must be same-cell,
    interleaved (reference and fused timed in one script one after the
    other). Per-call timing pattern `elapsed(reps)/reps` is fine but the
    first-listed config of a sweep inherits a thermal penalty.
29. F1 bottleneck decomposition (direct kernel timing with
    `SORT_STAGES=0`): the per-tile bitonic merge costs ~38 us per key
    tile at the default `(32,64)` config (21 gather passes x ARR=128
    lanes x 128 tiles, spill-heavy at 4 warps -> 12-17 ms of sort). Best
    tile config `(16,32)` (ARR=64) shifts the split to ~6.0 affinity +
    ~4.1 sort = ~10.0 ms. Affinity-only floors by config:
    `(32,64,w2) 2.57`, `(64,32,w4) 3.37`, `(32,32,w2) 3.31`. No config
    gets the full kernel below ~8.2 ms; best e2e speedup is 0.55x.
30. Triton affinity is 3-6x slower than cuBLAS for the same math:
    torch.matmul on `(1,4096,64) @ (1,64,4096)` (fp32, ieee, writes the
    full 67 MB score tensor) runs in 0.955 ms, while the clean tiled
    Triton Phase-A prototype (same dots, + per-tile max only) measures
    3.36 ms best (`QB32/KB128/FB32/w4/s3`). The fused kernel's masked
    K-offset lane load scheme and low occupancy (few programs) are
    suspected; re-tuning may close part of this gap.
31. torch.topk scales with width: `topk(4096x256)` 0.696 ms vs
    `topk(4096x4096)` 3.764 ms (5.4x cheaper) - supports a
    two-stage/threshold-filters-then-small-topk structure. Reference
    e2e baseline is ~5.5 ms, so the 2.5x gate (fused <= 2.2 ms) requires
    both a ~1 ms affinity AND a sub-ms selection: neither is evidenced
    yet on the T4.
32. Candidate-overflow degenerate data (in particular k=M, where every
    fresh element exceeds the threshold) cannot be served by a
    threshold-filtered selector; any such redesign must keep a fallback
    to the exact running-merge F1 kernel for flagged rows (k=M parity
    tests would hit this fallback by construction).
33. Colab output-screening friction during this session: DOM reads only
    expose the LAST stream lines of a virtualized cell; multi-line
    result sets must be printed as ONE line (join with "||") or rerun.
    Triton kernel errors inside notebook cells do NOT appear in
    `.stream` nodes - read them from the model's outputItems
    (`it.ename` / `it.evalue` / `it.traceback`). Also per Triton
    scoping: variables assigned inside a `for` loop are not visible
    after it unless pre-declared before the loop (hit twice:
    `NameError('big is not defined')`).
34. (Decision resolved, Task 5.4b, 2026-11-01.) User selected Option 3:
    cuBLAS affinity with chunked writes and Triton top-k softmax-gather.
    The controlled T4 strategy sweep (N=M=4096, k=32, d_k=d_v=64) found
    the fastest candidate was chunked PyTorch top-k with chunk 4096 at
    3.4741 ms vs 8.2937 ms reference (2.39x); the 2.5x target remains unmet.
    The lowest peak allocator use was 34.5 vs 258.5 MiB (7.49x), not ~75x,
    and is not a direct HBM traffic measurement. All measured candidates
    passed parity at max absolute error <=2.384e-7. Option 1's (16,32) F1
    tuning was not included and remains unmeasured.

35. (Dataset source flakiness, Task 7.2b, 2026-11-03.) The built-in
    `cv.snu.ac.kr/research/EDSR` host used by
    `scripts/prepare_phase7_datasets.py` for Flickr2K/Urban100 is dead
    (curl: connection refused / 21 s timeout, no response) - a known
    recurring outage for that host. CBSD68 (GitHub clausmichele master
    zip, 200 OK) and DIV2K (ETH `data.vision.ee.ethz.ch`, train-HR zip
    200 OK at 3.28 GB) are healthy. Verified alternates: Flickr2K
    HR-only zip via Hugging Face `yangtao9009/Flickr2K` resolve URL
    (10.85 GB, 200 OK; HR-only suffices since Stage B synthesizes
    degradations from clean HR images; also mirrored on Kaggle
    `duyminhle/flickr2k` 21.5 GB with HR+LR); Urban100 via HF
    `eugenesiow/Urban100` raw `Urban100_HR.tar.gz` (129 MB, 200 OK;
    official source page github.com/jbhuang0604/SelfExSR links a 1.14 GB
    copy). SOTS: official RESIDE page offers Dropbox links that require
    a working bit.ly hop; Kaggle mirror `balraj98/
    synthetic-objective-testing-set-sots-reside` (436 MB, indoor+outdoor
    GT/hazy) is simpler on Colab. Deraining/Outdoor-Rain training pairs:
    Rain13K composite mirror `dronefreak/Rain13K` on HF (13,711
    input/target pairs, standard MPRNet/Restormer protocol); Kaggle
    JRDR (2.2 GB) and `darun04/de-raining` (899 MB) also exist. Kaggle
    mirrors require the user's `kaggle.json` on Colab (assistant must
    not handle credentials); the HF resolve links work directly through
    the script's existing `--source-url` HTTPS path.

36. (Decision resolved, Task 7.2b sources, 2026-11-03.) User selected:
    Flickr2K -> HF `yangtao9009/Flickr2K` HR zip (10.85 GB); SOTS ->
    Kaggle `balraj98/synthetic-objective-testing-set-sots-reside`;
    deraining/Outdoor-Rain -> HF `dronefreak/Rain13K` (spread repo -
    the HF repo has no single zip, only train/ and test/ per-image
    trees, so it needs snapshot_download, not a resolve URL); Urban100
    -> HF `eugenesiow/Urban100` HR tar.gz (129 MB). Implemented in
    `scripts/prepare_phase7_datasets.py`: dead cv.snu.ac.kr URLs
    replaced in PUBLIC_DATASET_URLS and two `--source-url` schemes added
    (`kaggle:<owner>/<slug>` via the Kaggle CLI, `hf-dataset:<repo_id>`
    via huggingface_hub.snapshot_download). Kaggle credentials remain
    user-supplied (kaggle.json on Colab); the script never touches them.
    Focused tests 10/10; full suite 173 passed / 36 skipped locally.
    Remaining risk: Rain13K real pairs must be turned into
    `image_path,low_quality_path,task` CSV manifest rows (input/target
    layout), and Colab free disk (~200 GB on standard runtimes) still
    requires the one-corpus-at-a-time sharding flow for Flickr2K.

37. (Stage B session facts, 2026-11-03.) Colab T4 run completed 50 warmup
    epochs (5,000 steps), val PSNR 24.21->26.10, SSIM 0.714->0.793. Facts kept
    for future sessions: (a) Rain13K exact size 2.53 GB / 36,021 files (HF tree
    API). (b) patch 192 OOMs (18 GiB single backward allocation, ~patch^2);
    144 + expandable_segments works; batch is already 1 (loader hardwired).
    (c) This Colab runtime does not surface child-process stdout; the train
    cell captures child output and prints tails (kept). (d) Kaggle credentials
    cell now skips itself when SOTS already exists so Run-all never stalls on
    getpass. (e) Keep-alive interval injected into the Colab page (60 s)
    mitigates idle disconnects, not free-tier usage limits. (f) FLOPs measured
    locally via scripts/measure_flops.py (torch.profiler, torch path):
    ~8.7 GFLOPs forward per patch-144 step across all scales (output_size
    interpolates LR patches up to 144 before the backbone), est. ~26 GFLOPs
    per step including backward, ~130 TFLOPs per 50-epoch warmup; custom
    fused kernels are excluded by design. Op mix: mm/conv2d/bmm dominate.

38. (Drive web-UI empty-folder illusion, 2026-11-03.) After an interrupted drive.flush_and_unmount(), the Drive web UI showed phase7/stage_b empty across repeated hard reloads and title searches while the files existed server-side; a fresh drive.mount('/content/drive', force_remount=True) in the same session listed them immediately (best.pt 5.38 MB, latest.pt 5.39, metrics.jsonl 0.34, metrics_by_task.csv 0.26, steps.jsonl 0.28, train.csv 13.78, validation.csv 1.53, trainer_status.json ~0). Lessons: (a) trust kernel-side os.listdir over the Drive web UI when judging persistence; (b) an interrupted flush_and_unmount leaves the mount torn down (MyDrive path ENOENT) but deletes nothing on the server; (c) force_remount=True restores access in-session without data loss.
