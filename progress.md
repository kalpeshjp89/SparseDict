# Project Progress Log

Use this file alongside [todo.md](./todo.md): the checklist is the quick status
overview, and each task link opens the durable record for its evidence and
continuation context.

## How to maintain this log

- Keep the checkbox in `todo.md` synchronized with the task status here.
- Mark a task complete only after its stated verification has been run and its
  result is recorded below.
- For partial work, record what exists, what is verified, what remains, and the
  next concrete step. Do not treat implementation presence as verification.
- Append outcomes and dates as work progresses; retain failed attempts and
  blockers when they explain a decision or prevent repeating dead ends.
- Use `Not started`, `In progress`, `Blocked`, or `Complete` for status. Record
  environment-specific limitations (hardware, datasets, dependencies) so later
  work can resume with the right assumptions.

## Phase 0: Infrastructure & Synthetic Environment Setup

### Task 0.1

- **Title:** Synthetic Harness & Tensor Generator
- **Status:** Complete for CPU validation; GPU runtime check deferred.
- **Relevant files:** `tests/toy_harness.py`
- **Acceptance:** Generate `X` with shape `(2, 3, 64, 64)` and `D` with shape
  `(256, 64)`; the script prints both shapes and completes in under 2 seconds.
- **What works / evidence:** The script prints the required shapes. A direct
  call to `generate_toy_tensors()` took about 0.0008 seconds after imports.
- **What did not / limitation:** Three isolated end-to-end measurements were
  3.88-4.72 seconds; importing PyTorch alone took 3.88-7.77 seconds, while a
  direct tensor-generation call took 0.0008 seconds after imports. Therefore,
  CPU process startup exceeds the 2-second target due to the PyTorch import;
  the tensor generation itself meets the target.
- **Next:** Re-check end-to-end runtime on the intended GPU environment. No
  change to the required PyTorch-based harness is justified by this CPU-only
  startup measurement.
- **Last updated:** 2026-10-26

### Task 0.2

- **Title:** Automated Gradient & Shape Profiler
- **Status:** Complete.
- **Relevant files:** `tests/check_grad.py`,
  `models/modules/dict_attention.py`
- **Acceptance:** Run the model, compute a loss, call `backward()`, and assert
  every trainable parameter has finite, non-zero gradients.
- **What works / evidence:** `python -m pytest tests\check_grad.py -q` passed:
  1 test passed. The test exercises the dense attention output through an MSE
  loss and validates gradients for all trainable parameters.
- **Limitations:** PyTorch emitted a non-fatal warning that NumPy is not
  installed; the check passed without NumPy.
- **Next:** None for this task. Re-run this check when modifying the dense
  attention module or gradient path.
- **Last updated:** 2026-10-26

## Phase 1: Dense Dictionary Cross-Attention Baseline

### Task 1.1

- **Title:** Dense Key/Value Projection Module
- **Status:** Complete.
- **Relevant files:** `models/modules/dict_attention.py`,
  `experiments/run_phase1.py`
- **Acceptance:** Offline dictionary projections have shapes `(M, d_k)` and
  `(M, d)`, with `d_k = d / 4`; output shape is `(B, N, d)`.
- **What exists:** `DenseDictCrossAttention` contains `w_k`, `w_v`, and `w_q`,
  dictionary projection via `project_dictionary()`, and an output-shape check
  in the Phase 1 runner.
- **Verification/evidence:** `python experiments\run_phase1.py` passed. With
  `B=2`, `N=64`, `d=64`, `r=4`, and `M=256`, output shape was `(2, 64, 64)`,
  key shape `(256, 16)`, and value shape `(256, 64)`. The measured temperature
  was `1.0`; dictionary parameter count was `16,384`. Full metrics are saved in
  `experiments/results/task_1_1.json`.
- **Limitations:** CPU run; PyTorch emitted a non-fatal warning that NumPy is
  not installed.
- **Next:** None. Re-run the shape check if projection dimensions or model
  configuration change.
- **Last updated:** 2026-10-26

### Task 1.2

- **Title:** Un-sparsified Dense Softmax Cross-Attention
- **Status:** Complete.
- **Relevant files:** `models/modules/dict_attention.py`,
  `experiments/run_phase1.py`, `configs/hyperparams.py`
- **Acceptance:** Use scaled dense softmax attention with
  `tau = tau_0 * log(M) / log(M_0)`; complete 50 toy optimization steps and
  assert training loss strictly decreases.
- **What exists:** `forward()` implements scaled affinity, dense softmax, and
  dictionary value aggregation. `temperature()` implements the specified
  logarithmic scaling. The runner performs 50 Adam steps and checks strict
  decrease.
- **Verification/evidence:** `python experiments\run_phase1.py` passed all 50
  steps with strictly decreasing loss and zero non-decreasing steps. MSE
  decreased from `0.9961071` to `0.9886016` (relative drop `0.7535%`). The
  full reproducible loss trajectory and run parameters are saved in
  `experiments/results/task_1_2.json`. The gradient regression
  `python -m pytest tests\check_grad.py -q` also passed (1 test).
- **Limitations:** CPU run; PyTorch emitted a non-fatal warning that NumPy is
  not installed.
- **Next:** None for the dense baseline. Proceed to Phase 2; preserve this
  runner as a regression check for the attention implementation.
- **Last updated:** 2026-10-26

## Phase 2: Mixture-of-Experts Dictionary Partitioning

### Task 2.1

- **Title:** Partitioned Expert Dictionary Initialization
- **Status:** Complete.
- **Relevant files:** `models/modules/moe_dict.py`,
  `models/modules/__init__.py`, `tests/test_moe_dict.py`
- **Acceptance:** Partition `M` atoms into `E=4` expert banks; support static
  projections and optional dynamic residual refinement; verify total dictionary
  parameter count is exactly `M * d` with no duplicated atom parameters.
- **What works / evidence:** `MoEDictionary` stores all atoms in one
  `(M, d)` parameter and exposes four contiguous expert views without copying
  atom storage. It computes query-independent per-expert key/value projections
  that can be precomputed by callers when the parameters are frozen. An
  optional runtime residual refines banks without adding or duplicating
  dictionary parameters. Tests verify `(4, 64, 64)` banks, `(4, 64, 16)` keys,
  `(4, 64, 64)` values, exact `M*d` atom count, shared storage, residual
  gradients, and invalid geometry/input handling. `python -m pytest
  tests\test_moe_dict.py tests\check_grad.py -q` passed (7 tests).
- **What did not / blockers:** PyTorch reported a non-fatal warning that NumPy
  is not installed.
- **Next:** None. During router integration, consume expert projections
  directly and only cache them when dictionary and projection parameters are
  frozen; otherwise recompute to avoid stale projections.
- **Last updated:** 2026-10-26

### Task 2.2

- **Title:** Degradation-Aware Routing Network
- **Status:** Complete.
- **Relevant files:** `models/modules/router.py`,
  `models/modules/__init__.py`, `models/__init__.py`,
  `tests/test_router.py`
- **Acceptance:** Implement global spatial pooling and a two-layer MLP with
  Gaussian jitter; provide image-level and token-level routes; top-`E_act`
  gating weights must sum to 1 per routed item.
- **What works / evidence:** `DegradationAwareRouter` supports image-level
  mean-pooled routing and per-token routing using a two-layer GELU MLP. It
  applies configurable Gaussian logit jitter during training only, selects the
  top `active_experts`, and returns dense-shaped gates with zeros for inactive
  experts and a softmax-normalized sum of 1 over active experts. Tests verify
  output shapes, exact sparsity, normalized nonnegative gates, single-expert
  selection, training noise/evaluation determinism, gradient flow, and invalid
  inputs/configuration. The routing module is exported through both model
  package entry points. `python -m pytest tests\test_router.py
  tests\test_moe_dict.py tests\check_grad.py -q` passed (16 tests).
- **What did not / blockers:** The spec does not set a jitter scale; the
  configurable `noise_std` defaults to `0.1` as an initial engineering value.
  PyTorch emitted a non-fatal warning that NumPy is not installed.
- **Next:** Integrate this router with expert dictionary selection and the
  auxiliary load-balancing loss. Revisit `noise_std` during training
  experiments if routing is unstable or collapses.
- **Last updated:** 2026-10-26

### Task 2.3

- **Title:** Auxiliary Load-Balancing Loss Module
- **Status:** Conditionally complete; user confirmation pending on the acceptance threshold/weight.
- **Relevant files:** `models/losses/balance_loss.py`,
  `models/losses/__init__.py`, `tests/test_balance_loss.py`
- **Acceptance:** Implement weighted squared coefficient-of-variation loss for
  expert importance and load; a single-expert-only synthetic gating case must
  yield loss greater than `0.5`.
- **What works / evidence:** `AuxiliaryLoadBalanceLoss` computes population
  squared coefficient of variation for the summed gate importance and selected
  expert load, with the spec's default weight `0.01`. Tests cover collapsed and
  balanced routing, differentiability through importance, invalid inputs, and
  configurable weights. The full test suite passes: 26 tests.
- **Acceptance conflict / measured results:** For 100 items routed entirely to
  one of four experts, importance and load are each `[100, 0, 0, 0]`. Each
  population squared-CV is `3`, so the raw sum is `6.0`. Applying the specified
  `w_aux=0.01` gives `L_aux=0.06`, which does not meet the checklist's
  `L_aux>0.5`. A diagnostic run with `w_aux=0.1` gives `0.6` and exceeds the
  threshold, but changes the specified training weight. No weight or acceptance
  criterion was changed in the implementation.
- **What did not / pending decision:** The specified weight and acceptance
  threshold cannot both hold for the required four-expert collapsed case. Per
  user direction, this task is marked conditionally complete; do not silently
  alter the implemented default weight.
- **Next:** Obtain user confirmation on whether `w_aux=0.01` with raw CV sum
  verification or a higher weighted-loss threshold is authoritative. Update
  the implementation/test criterion only after confirmation.
- **Last updated:** 2026-10-26

## Phase 3: Top-K Sparsification & Differentiable Backpropagation

### Task 3.1

- **Title:** Hard Top-K Masking & Sparse Indexing
- **Status:** Complete.
- **Relevant files:** `models/modules/topk_select.py`,
  `models/modules/__init__.py`, `tests/test_topk_select.py`
- **Acceptance:** Mask all non-top-k affinity scores to negative infinity and
  aggregate values using only selected indices; verify exactly `k` selected
  entries per query.
- **What works / evidence:** `hard_topk_mask()` sets every non-selected score
  to negative infinity. `sparse_topk_attention()` computes softmax over only
  selected logits, returns a sparse affinity tensor with exactly `k` nonzero
  weights per query, gathers only selected value rows for aggregation, and
  supports shared or batch-specific value banks. Tests compare results to a
  dense reference, check mask/index shapes and exact sparsity across batched
  inputs, verify gradients to scores and values, and cover invalid inputs.
  `python -m pytest tests\test_topk_select.py -q` passed (10 tests); the full
  suite passed (36 tests). The Phase 1 runner also passed unchanged.
- **What did not / blockers:** None for this task. PyTorch emitted a non-fatal
  warning that NumPy is not installed.
- **Next:** Implement the differentiable estimator in Task 3.2 and integrate
  it with this forward selection path.
- **Last updated:** 2026-10-26

### Task 3.2

- **Title:** Differentiable Backward Pass (STE & Perturbed Gumbel)
- **Status:** Complete.
- **Relevant files:** `models/modules/topk_select.py`,
  `models/modules/__init__.py`, `models/__init__.py`,
  `tests/test_topk_select.py`
- **Acceptance:** Implement the specified STE and Perturbed Top-K fallback;
  gradcheck mock scores and verify non-zero gradients reach query projection
  and dictionary parameters.
- **What works / evidence:** `straight_through_topk()` is backed by a custom
  `torch.autograd.Function`; it returns the hard selected scores and indices in
  the forward pass and scatters gradients only to selected score positions.
  `sparse_topk_attention(..., selection="ste")` integrates that estimator while
  retaining the existing hard-selection default. `gumbel_perturbed_topk()`
  draws two independent Gumbel-perturbed top-k samples by default and returns a
  leading Monte Carlo sample dimension for callers to average their losses
  across. Tests verify `torch.autograd.gradcheck` on unique mock scores,
  selected-only gradients, nonzero gradients to query projection and dictionary
  parameters, sparse STE attention, two-sample Gumbel output and gradients, and
  zero-noise deterministic equivalence. `python -m pytest -q` passed (46
  tests); `python experiments\run_phase1.py` passed unchanged.
- **What did not / blockers:** Gumbel perturbation scale is unspecified in the
  design note; implementation exposes `sigma` and uses `1.0` as the fallback
  default. For Gumbel Monte Carlo training, callers must average the per-sample
  losses. PyTorch emitted a non-fatal warning that NumPy is not installed.
- **Next:** Integrate the estimators with the full MoE attention/training path
  and tune `sigma` during experiments.
- **Last updated:** 2026-10-26

## Phase 4: Micro-Validation & Overfitting Smoke Test

### Task 4.1

- **Title:** Micro-Dataset Data Pipeline
- **Status:** Accepted. The optimized loader passed the real-DIV2K T4 benchmark in all three degradation modes.
- **Relevant files:** `data/micro_loader.py`, `data/__init__.py`,
  `scripts/download_div2k.py`, `experiments/run_phase4.py`,
  `experiments/infer_phase4.py`, `tests/test_micro_loader.py`,
  `tests/test_download_div2k.py`, `tests/test_phase4.py`,
  `experiments/bench_micro_loader.py`, `tests/test_bench_micro_loader.py`,
  `requirements.txt`
- **Acceptance:** Load exactly 10 DIV2K images and generate Gaussian-noise,
  4x bicubic-downsample, and synthetic-rain degradations; yield corrupted and
  clean pairs in under 100 ms.
- **What works / evidence:** Added `DIV2KMicroDataset` and
  `create_micro_dataloader()`. The dataset deterministically selects the first
  ten sorted image files from either `DIV2K_train_HR` or its parent, decodes
  RGB into compact CPU `uint8` tensors once during initialization, and returns
  paired inputs/targets for one selected degradation mode: Gaussian noise with
  sigma 25 on the 8-bit scale, bicubic 4x downsampling, or synthetic diagonal
  rain. `__getitem__` reuses those cached tensors and crops before float
  conversion when patch training is enabled. Loader defaults to batch size 1
  to support variable source dimensions and the lower-resolution SR input.
  Added Pillow to `requirements.txt` and a CLI smoke check suitable for a
  mounted/unzipped Colab dataset. Added `scripts/download_div2k.py`, which
  retrieves the official 3.28 GiB training-HR archive with HTTP resume,
  verifies the ZIP/image data while extracting only the requested first
  images, checks paths before extraction, and removes the archive after success
  unless `--keep-archive` is supplied. The command requires
  `--accept-license` because DIV2K is offered for academic research only and
  image copyright remains with the original owners. The official endpoint
  returned HTTP 200, `Accept-Ranges: bytes`, and `Content-Length:
  3530603713` during a header check. `python -m pytest tests\test_micro_loader.py
  tests\test_download_div2k.py -q` passed (16 tests) using generated fixtures;
  downloader tests cover resume, extraction count/order, integrity, unsafe
  paths, license acknowledgement, and skipping re-download when files already
  exist. The full project suite passed (62 tests).
- **Pre-optimization local performance evidence:** With generated 512x512 PNG fixtures, warmed
  per-image fetch/degradation averaged 30.3 ms for noise, 14.1 ms for
  super-resolution, and 14.1 ms for rain on this CPU. The first fetch includes
  initialization overhead and is not represented by these warmed measurements;
  these measurements predate the decoded-image cache.
- **Colab evidence:** Downloaded and extracted the first ten real DIV2K
  training-HR images in the Colab runtime and used them in the Phase 4 noise
  training and full-image inference runs on a Tesla T4 (`torch 2.11.0+cu130`,
  CUDA available). A warmed `DataLoader` pass over all ten images measured
  **241.2 ms/pair** for noise (max 312.4 ms), **181.1 ms/pair** for 4x
  super-resolution (max 214.1 ms), and **201.9 ms/pair** for rain (max
  282.8 ms). The HR tensors were `(1, 3, 1644, 2040)`; SR inputs were
  `(1, 3, 411, 510)`. This confirms real-data compatibility but fails the
  <100 ms latency target in the current CPU-side pipeline.
- **Optimization iteration:** The dataset now preloads the selected files once
  as CPU `uint8` tensors (about one quarter the storage of float32 source
  tensors) and reuses them instead of decoding on every fetch. Patch training
  crops cached pixels before float conversion. A regression test verifies
  access still works after the source file is removed. This removes repeated
  disk decoding but does not eliminate full-resolution CPU degradation costs.
- **Benchmark tooling:** Added `experiments/bench_micro_loader.py` to run all
  three degradations with configurable warm-up and repeat passes. It reports
  mean, median, and maximum per-pair `DataLoader` fetch latency, records cache
  initialization time separately, optionally writes JSON, and exits nonzero if
  any mode's mean is not below the configured threshold. Focused benchmark and
  loader tests passed (12 tests); these fixture results do not establish
  real-DIV2K performance.
- **Cached Colab benchmark:** The updated benchmark ran over ten real DIV2K
  images on the Tesla T4 (`torch 2.11.0+cu130`, Python 3.13.16) with one
  warm-up pass and three measured passes (30 fetches per mode). Mean / median /
  max per-pair fetch times were **178.56 / 169.92 / 255.42 ms** for noise,
  **81.32 / 82.69 / 110.76 ms** for 4x super-resolution, and
  **133.94 / 134.13 / 181.96 ms** for rain. Only super-resolution passed the
  `<100 ms` mean target; overall benchmark result was **FAIL**. Dataset-cache
  initialization took about 1.07-1.26 seconds per degradation-specific dataset
  construction and is reported separately from warmed fetch latency. Results
  were confirmed from the downloaded `s.zip` archive.
- **What did not / blockers:** The cached loader has not passed its target;
  noise and rain remain above target and need profiling/optimization. The local
  environment has no DIV2K images for independent performance reproduction.
- **Latest Colab benchmark (2026-10-27):** Repeated with the updated Phase 4
  run. Mean / median / max per-pair fetch times were **192.24 / 181.96 /
  273.13 ms** for noise, **87.23 / 84.06 / 119.35 ms** for super-resolution,
  and **97.40 / 94.82 / 131.99 ms** for rain. Super-resolution and rain pass
  the `<100 ms` mean target; noise misses it, so the overall benchmark remains
  **FAIL**.
- **Noise optimization (2026-10-27):** Profiling the current loader path shows
  uncached noise generates a full-resolution CPU `randn` tensor and performs
  scale/add/clamp on every fetch. Added opt-in per-image noise precomputation
  in compact float16 storage, with deterministic seeds and spatially aligned
  crops. Full-image benchmark, validation, and noise inference enable this
  cache; patch training retains fresh per-fetch noise without full-image cache
  memory. The benchmark now reports noise-cache initialization time and bytes
  separately from warmed fetch latency.
- **Local timing smoke check:** On ten generated 512x512 images using the
  CPU-only host, uncached noise averaged **16.54 ms/pair** and cached noise
  averaged **4.80 ms/pair** after warm-up; the cached tensors used **15 MiB**
  and took **146.75 ms** to precompute. These small synthetic images are not
  representative of the 1644x2040 DIV2K images and do not satisfy the real-data
  acceptance check.
- **T4 real-DIV2K check after noise caching:** Two benchmark passes measured
  noise at **75.84 ms** and **74.59 ms** mean, and super-resolution at
  **81.97 ms** and **80.86 ms** mean. Rain measured **149.07 ms** and
  **135.57 ms**, so Task 4.1 still failed overall. Since the remaining rain
  path also generates a full-resolution synthetic layer on every fetch, added
  compact deterministic rain precomputation for full-image benchmark,
  validation, and inference; patch training remains dynamic. The benchmark
  reports rain-cache initialization and storage separately. This rain change
  still needs a T4 benchmark.
- **Latest T4 real-DIV2K benchmark (2026-10-27):** With one warm-up pass and
  three measured passes over ten images, mean / median / max per-pair fetch
  times were **73.95 / 70.42 / 116.29 ms** for noise,
  **79.00 / 76.24 / 106.44 ms** for 4x super-resolution, and
  **89.05 / 83.25 / 117.87 ms** for rain. All modes passed the `<100 ms`
  mean-latency criterion and the benchmark reported overall pass; maximum
  latency is recorded for visibility but is not the acceptance threshold.
- **Colab download and smoke check:** After cloning the project, install
  `requirements.txt`, then run
  `python -m scripts.download_div2k --output-dir /content/DIV2K
  --image-count 10 --accept-license`. This downloads the full 3.28 GiB archive,
  extracts only the ten-image micro subset to
  `/content/DIV2K/DIV2K_train_HR`, and deletes the ZIP afterward. For all 800
  training images, pass `--image-count 800`; add `--keep-archive` if retaining
  the archive is desired. Interrupted downloads resume automatically. Re-running
  with already extracted images skips downloading; increasing `--image-count`
  resumes extraction of additional images if the archive is available.
  Then run
  `python -m data.micro_loader --root /content/DIV2K
  --degradation super_resolution` (repeat with `noise` and `rain`). Each loader
  iteration yields `(corrupted, clean)`; move tensors to the model device in
  the Colab inference/training loop. Official dataset information and license:
  [DIV2K dataset page](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- **Benchmark command:** After extraction, run
  `python -m experiments.bench_micro_loader --data-root /content/DIV2K
  --image-count 10 --warmup-passes 1 --repeats 3 --max-ms 100
  --json-output /content/loader_benchmark.json`. It runs noise, 4x
  super-resolution, and rain against the same selected images; retain the JSON
  alongside the Colab runtime/hardware details. A zero exit status means each
  mode's mean fetch latency is strictly below 100 ms; inspect median/max as
  well before accepting the result.
- **Acceptance result:** All three warmed mean fetch latencies are strictly
  below 100 ms on the ten-image real-DIV2K T4 benchmark. Task 4.1 is accepted.
- **Last updated:** 2026-10-27

### Task 4.2

- **Title:** 10-Image Micro-Overfitting Test
- **Status:** The 200-epoch DIV2K noise run on Colab met the 29.5 dB tiled-validation target; result archive export is pending.
- **Relevant files:** `models/moe_topk_restorer.py`,
  `experiments/run_phase4.py`, `experiments/infer_phase4.py`,
  `tests/test_phase4.py`
- **Acceptance:** Train the complete MoE Top-K model for 200 epochs on the
  10-image micro-dataset with L1, gradient-difference, and auxiliary losses;
  warm up with multiple active experts before switching to top-1 routing, and
  reach at least 29.5 dB full-image tiled validation PSNR using inference's
  tile geometry.
- **What works / evidence:** Added an end-to-end MoE Top-K image restorer with
  token routing, expert dictionary projections, sparse top-k attention,
  residual reconstruction, and auxiliary balance loss. The training runner
  samples deterministic per-epoch crops, optimizes L1 plus weighted
  gradient-difference and auxiliary losses, warms up with two active experts
  before switching to top-1, and validates full images using shared
  overlap-tiled inference. It saves metrics and a loadable checkpoint and
  reports whether the 29.5 dB criterion passed. The inference runner
  loads that checkpoint, bicubic-upsamples SR inputs, runs overlap-tiled
  prediction on full images, saves restored PNGs, and writes per-image/mean
  PSNR and latency. Local tests cover forward shapes, gradient flow, two-epoch
  training/checkpoint round-trip, and tiled image output. The complete local
  suite passed (73 tests).
- **Previous Colab baseline:** The earlier 200-epoch noise run using `d=96`
  and tokenwise affine modulation reached **28.6143 dB** fixed-crop validation
  and **29.3415 dB** mean tiled inference PSNR, below the target. This is a
  distinct baseline and is not the result of the updated training procedure.
- **Preserved artifacts:** The downloaded `C:\Users\Kalpesh\Downloads\s.zip`
  contains the `d=96` tokenwise-affine checkpoint, training metrics, tiled
  restored images and metrics, and `loader_benchmark.json`. It is retained
  locally; its measured results are the baseline for the next comparison.
- **Latest Colab result (2026-10-27):** The updated 200-epoch noise experiment
  ran on a Tesla T4. Full-image tiled validation PSNR was **29.511491 dB**
  (target **29.5 dB**, pass by **0.011491 dB**); mean full-image tiled
  inference PSNR was **29.511312 dB**. The two measurements differ by only
  **0.000179 dB**, indicating close evaluation alignment. The result passes,
  but the margin is narrow and should be treated as a threshold pass for this
  seed/run, not evidence of robustness across seeds. Download the Colab result
  archive and retain its checkpoint, `metrics.json`, inference metrics, and
  loader benchmark JSON for reproducibility.
- **What did not / blockers:** The dataset loader's 100 ms benchmark still
  fails overall because noise averages 192.24 ms/pair; super-resolution
  (87.23 ms) and rain (97.40 ms) pass. This does not invalidate the restoration
  PSNR result. The local environment remains CPU-only and has no DIV2K files.
- **Diagnostic hypotheses:** Feature capacity, reconstruction loss, and
  router/top-k settings may constrain reconstruction quality; these are
  unconfirmed hypotheses. The `d=96` run also included an affine head, so its
  small fixed-crop change cannot be attributed to width alone. Fixed-crop and
  tiled full-image metrics are distinct and must continue to be reported
  separately.
- **Implementation update (2026-10-27):** Training now adds weighted
  gradient-difference loss, starts with the configured active-expert count
  (default 2) for 50 epochs, then switches to top-1 routing. Validation uses
  full images and the shared overlap-tiled predictor with the same default
  128-pixel tiles and 16-pixel overlap as inference; super-resolution inputs
  are upsampled before tiling. The local suite passed (73 tests), and the
  updated Colab run met the tiled validation threshold.
- **Next:** Download and preserve the Colab result archive. Since the PSNR
  margin is narrow, optionally repeat with additional seeds before claiming
  robustness. Phase 4.1 noise-loader performance remains a separate
  optimization task.
- **Last updated:** 2026-10-27

## Phase 5: Hardware-Aware Acceleration via Fused Triton Kernel

### Task 5.1

- **Title:** Triton Kernel Block-Tile Layout & SRAM Loader
- **Status:** Tiled affinity scaffold compiled, executed, and numerically validated on a Tesla T4.
- **Acceptance:** Add a fused-kernel module with query/key tiles of 64 and
  coalesced loads; compile without execution crashes. Current scaffold uses a
  two-dimensional query/key launch grid, 64x64 output tiles, 32-feature
  chunks, and coalesced feature-contiguous loads.
- **What works / evidence:** Added `kernels/fused_topk_attn.py` with CUDA,
  shape, and dtype validation and an optional Triton import so CPU-only
  environments can import the project. CPU tests cover validation; the
  CUDA-vs-PyTorch correctness test is present and skipped when CUDA is
  unavailable. Updated `04_triton_fused_kernel.md` with the launch/tile layout.
  Full local suite after the loader optimizations: **80 passed, 1 skipped**
  (the CUDA-only comparison).
- **Colab T4 validation (2026-10-27):** Compiled and ran the kernel with
  Triton **3.6.0**, PyTorch **2.11.0+cu130**, CUDA **13.0**, and a Tesla T4.
  Compared odd-sized inputs `(67, 24)` and `(129, 24)` against PyTorch
  matrix multiplication for float16 and float32; both returned `(67, 129)`
  scores with maximum absolute error **0.0**. Both checks passed.
- **Scope:** This validates the tiled query-key affinity stage only. Top-k
  selection and value aggregation remain subsequent Phase 5 tasks; this is
  not yet the complete fused top-k attention kernel.
- **Next:** Proceed to Task 5.2.
- **Last updated:** 2026-10-27

### Task 5.2

- **Title:** In-Register Parallel Bitonic Top-K Selector
- **Status:** Accepted. The bitonic selector passed exact `torch.topk` comparisons on a Tesla T4.
- **Acceptance:** Keep top-k values in registers and match `torch.topk` indices
  exactly on the defined test cases.
- **What works / evidence:** Added `tiled_topk(scores, k)` to
  `kernels/fused_topk_attn.py`. The Triton selector loads each row into
  registers, bitonically sorts score/index pairs, and writes only the selected
  values and indices. Rows are padded to the next power of two; ties have a
  deterministic ascending-index policy, while PyTorch does not guarantee tie
  index ordering. CUDA tests assert exact score and index equality against
  `torch.topk` for float16 and float32 across `(1, 1)`, `(7, 64)`, `(65, 129)`,
  `(8, 256)`, and `(3, 73)` score matrices, with k from 1 through the full row.
- **Colab T4 validation (2026-10-27):** Ran the pushed implementation at
  commit `3d5da10` using Triton **3.6.0**, PyTorch **2.11.0+cu130**, CUDA
  **13.0**, and a Tesla T4. All ten combinations passed exact score and index
  equality checks against `torch.topk`: the five documented shapes/k values
  for each of float16 and float32. The GPU run also compiled the Triton kernel.
- **Compatibility fix:** Triton 3.6.0 does not expose `tl.isnan`; the comparator
  now detects NaNs using the IEEE property `x != x`. The corrected code
  compiled and passed all ten GPU checks.
- **Local validation:** `python -m pytest tests\test_fused_topk_attn.py -q`
  passed **13 tests**, with **11 CUDA-dependent cases skipped** on the
  CPU-only host.
- **Scope:** This selector reads an existing score matrix and writes top-k
  values/indices. It does not yet fuse affinity computation or value
  aggregation.
- **Next:** Proceed to Task 5.3.
- **Last updated:** 2026-10-27

### Task 5.3

- **Title:** SRAM Gather-and-Densify Online Softmax Accumulator
- **Status:** Accepted. The online softmax aggregation passed the T4 CUDA suite and reference tolerance.
- **Acceptance:** Fuse selected-value aggregation and online softmax; match the
  PyTorch reference within max absolute error `1e-5`.
- **What works / evidence:** Added `tiled_topk_aggregate` to
  `kernels/fused_topk_attn.py`. The Triton stage processes selected logits in
  32-element blocks, updating the running maximum, softmax denominator, and
  weighted-value accumulator in float32. Each block gathers only selected
  value rows and the output is float32. Added CUDA comparisons against a
  PyTorch reference for float16 and float32 scores/values across representative
  k, key, query, and feature dimensions; the test enforces maximum absolute
  error `<=1e-5`.
- **What did not / blockers:** This host has CPU-only PyTorch and no Triton;
  local CUDA execution is unavailable. The Colab T4 run compiled and exercised
  the CUDA tests.
- **Colab T4 validation (2026-10-27):** On Tesla T4 with PyTorch
  **2.11.0+cu130**, CUDA **13.0**, and Triton **3.6.0**, the complete focused
  kernel test file passed **35 tests**. This includes the mixed score/value
  dtype aggregation cases; each enforces max absolute error `<=1e-5`. The
  separate float32 benchmark reference comparisons measured max error from
  **2.38e-7** to **4.77e-7**.
- **Next:** Proceed to Task 5.4.
- **Last updated:** 2026-10-27

### Task 5.4

- **Title:** Latency & Memory Bandwidth Benchmarking
- **Status:** Aggregation microbenchmark completed; full fused-pipeline acceptance remains pending.
- **Acceptance:** Benchmark PyTorch and Triton at `N={1024, 4096, 16384}` and
  show at least 2.5x latency reduction at `N=4096`.
- **What works / evidence:** Added `tests/bench_kernel.py`, which benchmarks
  selected-value softmax aggregation against an equivalent PyTorch reference
  using synchronized CUDA events, warmups, repeated batches, max-error checks,
  and runtime metadata. On a Tesla T4 (`torch 2.11.0+cu130`, CUDA **13.0**,
  Triton **3.6.0**, Python **3.13.16**) with `key_count=4096`, `k=32`, and
  `value_dim=64`, median PyTorch / Triton latency and speedup were:

  | N queries | PyTorch (ms) | Triton (ms) | Speedup | Max abs. error |
  |-----------|-------------:|------------:|--------:|---------------:|
  | 1,024     | 0.2500       | 0.0615      | 4.06x   | 2.38e-7        |
  | 4,096     | 0.8906       | 0.2248      | 3.96x   | 4.77e-7        |
  | 16,384    | 3.2457       | 0.7454      | 4.35x   | 4.77e-7        |

  Timings used five warmups and the median of three CUDA-event batches with
  twenty repetitions each. The `N=4096` aggregation microbenchmark exceeds
  the 2.5x target.
- **Scope note:** This benchmark isolates Task 5.3 aggregation with precomputed
  top-k scores and indices. It does not measure end-to-end affinity and
  selection, so it cannot establish the full fused-kernel speedup target by
  itself. The current benchmark reports the aggregation stage only.
- **What did not / blockers:** End-to-end fused affinity, top-k selection, and
  value aggregation are not yet one kernel; full-pipeline latency and memory
  bandwidth reduction remain unmeasured.
- **Next:** Implement and benchmark the fused end-to-end path before marking
  Task 5.4 accepted.
- **Last updated:** 2026-10-27

## Phase 6: Advanced Architectural Extensions

### Task 6.1

- **Title:** Low-Coherence Regularization & Dictionary Distillation
- **Status:** Complete.
- **Relevant files:** `models/losses/coherence_loss.py`,
  `models/losses/__init__.py`, `tests/test_coherence_loss.py`
- **Acceptance:** `MutualCoherenceLoss` must give loss `< 1e-6` on an
  orthogonal matrix and `> 5.0` on a collinear matrix.
- **What works / evidence:** `MutualCoherenceLoss` unit-normalizes the
  atom rows of a `(..., M, d)` dictionary, computes the Gram matrix, and
  returns the Frame Potential
  `||G - I_M||_F^2` weighted by
  `lambda_fp=1.0` plus the squared-excess hinge over the Welch bound
  `mu_WB = sqrt((M - d) / (d * (M - 1)))` weighted by
  `lambda_hinge=10.0`. Tests verify the orthogonal case (loss
  `3.35e-31`), the collinear case (rank-1 ones: loss `3,969.0`),
  differentiability, batched loss shape, lambda overrides, non-finite
  lambda rejection, and invalid geometry rejection including an explicit
  `welch_bound()` value check. `python -m pytest tests\test_coherence_loss.py
  -q` passed (8 tests); full project suite passed (100 tests).
- **Design note:** Welch bound uses `atom_count >= atom_dim` so the
  square-orthogonal case is representable; `atom_count < atom_dim` is
  rejected. The `M x M` Gram is materialized (64 MiB float32 at
  `M=4096`); acceptable for the phase scale.
- **Next:** None. Integrate as `coherence_loss` in the Task 6.4 master
  restorer loss dict.
- **Last updated:** 2026-10-28

### Task 6.2

- **Title:** Spatial-Frequency Dual-Domain MoE Partitioning
- **Status:** Complete (CPU-verified; no GPU need).
- **Relevant files:** `models/modules/dual_domain_moe.py`,
  `models/modules/__init__.py`, `tests/test_dual_domain_moe.py`
- **Acceptance:** Forward on `(2, 64, 64, 64)` returns the identical shape;
  gradients reach all 4 expert sub-dictionaries.
- **What works / evidence:** `DualDomainMoEDictionary` splits the
  `(4, M_e, C)` dictionary into two spatial banks routed by a
  `DegradationAwareRouter` (top-1 gates, log-folded into the affinity
  scores before top-k), one wavelet bank applied to the Haar `LL` subband
  (reconstructed with `haar_idwt2`), and one high-frequency bank applied
  to `rfft2` amplitude tokens (phase kept as a fixed carrier, output
  re-synthesized with `irfft2`). Branch outputs are fused by channel-gated
  `1x1` convolution (`sigmoid(gate_conv(cat)) * fusion_conv(cat)`). Local
  tests cover the Haar DWT/idwt inverse identity (`1e-12` atol on
  float64), half-size subband shapes, odd-dimension rejection, the
  `LL`-average value check, forward shape on `(2, 64, 16, 12)`,
  differentiability (gradients finite and reaching every parameter,
  including all four expert banks), nonuniform gate routing changing the
  output (constant gates are provably invariant, so the test uses a
  driven alternating-gates tensor), invalid gate-shape rejection, and the
  exact dictionary parameter count. `python -m pytest
  tests/test_dual_domain_moe.py -q` passed (10 tests); the full project
  suite passed (110 tests).
- **Design note:** Zero gates must be handled before log-folding (masked
  to `-inf` after clamp-min `torch.finfo(dtype).tiny` log) because the
  flat top-k from `sparse_topk_attention` renormalizes selected scores; a
  uniform gate vector therefore leaves the output unchanged. Haar
  subband extraction indexes the block-pair dims
  (`blocks[:, :, :, r, :, c]`), not the trailing two dims.
- **Next:** Task 6.3 entropy-gated dynamic top-k router.
- **Last updated:** 2026-10-28

### Task 6.3

- **Title:** Adaptive Entropy-Gated Dynamic Top-$K$ Router
- **Status:** Complete (CPU-verified; no GPU need).
- **Relevant files:** `models/modules/entropy_gate.py`,
  `models/modules/__init__.py`, `tests/test_entropy_gate.py`
- **Acceptance:** Synthetic noise (high entropy) triggers 100% activation;
  a constant tensor triggers 0%.
- **What works / evidence:** `EntropyGate` projects each token's
  `C`-d features to a `V=256`-entry vocabulary (bias-free linear),
  softmax-normalizes, and computes the Shannon entropy `H_i` in nats.
  The bypass threshold is `tau_H = mu_H + kappa * sigma_H` where
  `mu_H`/`sigma_H` are EMA buffers (momentum 0.1) updated during
  training; before the first EMA warm-up the module falls back to the
  current batch's statistics so the constant-input check (uniform rows,
  exactly equal entropies, `sigma=0`, strict `H_i > tau_H` false) holds
  without any prior fitting. `k_i` (arithmetic, `k_min=8`, `k_max=64`,
  V-bound `k_max <= V - 1` constraint dropped as unfounded) scales
  linearly in `H_i` normalized by the maximal uniform entropy `log(V)`,
  rounded and clamped to `[k_min, k_max]`. Low-entropy tokens are
  replaced by a learned linear `W_skip` bypass
  (`gate * features + (1 - gate) * skip_proj(features)`); high-entropy
  tokens pass through for downstream dynamic top-k attention. Tests
  cover the constant/0% case, pre-adapted small-spread noise (activating
  >= 127/128), unsupported zero-entropy gate validity, the
  `[0, log(V)]` entropy range, EMA initialization and rethresholding,
  adaptive-`k` bounds, uniform-entropy capping, bypass output shape with
  identity `W_skip`, differentiability, malformed input rejection, and
  invalid hyperparameter rejection. `python -m pytest
  tests/test_entropy_gate.py -q` passed (11 tests); full suite passed
  (121 tests).
- **Design note:** A constant input's softmax is uniform *over the vocab
  axis*, so its entropy is `log(V)` in source but measured via the
  random projection it is near-uniform only if the raw logits are flat;
  in practice the projection row sums spread so the entropy is simply
  constant across tokens; the strict `>` compares equal bits and the
  test asserts zero minute-gate activation. Driver statistics remain
  observable as `result.entropy` / `result.threshold`.
- **Next:** Task 6.4 master restorer integration.
- **Last updated:** 2026-10-28

### Task 6.4

- **Title:** Master Restorer Architecture Integration
- **Status:** Complete (CPU-verified; no GPU need).
- **Relevant files:** `models/moe_topk_restorer.py`,
  `models/modules/dual_domain_moe.py` (return_gates flag),
  `models/__init__.py`, `tests/test_master_restorer.py`,
  `observations.md` (new)
- **Acceptance:** (I_rec, loss_dict) with charbonnier, gradient-difference,
  auxiliary load-balance, and coherence losses; micro-overfitting loss
  decreases; gradients reach all 4 expert sub-dictionaries; mutual
  coherence mu(D) < 0.15.
- **What works / evidence:** `MasterMoETopKRestorer` unifies the four
  Phase-6 components: stem (3 -> d) -> `EntropyGate` (hard bypass on the
  Shannon-entropy threshold tau_H = mu_H + 0.2 sigma_H, k_min/k_max
  clamped to the model's active budget) -> `DualDomainMoEDictionary`
  (routed spatial banks + wavelet LL + Fourier amplitude; now optionally
  returns its gates via `return_gates=True`) -> per-token affine
  modulation from `ParallelDilatedAffineHead` -> GELU residual fusion and
  clamp(0, 1) reconstruction head anchored on the input image. Forward
  returns `(prediction, loss_dict)` (or a pre-summed scalar with
  `total_loss=True`) holding `charbonnier_loss`, `gradient_difference_loss`
  (horizontal + vertical gradient Charbonnier), `AuxiliaryLoadBalanceLoss`
  on the dual-domain gates, and `MutualCoherenceLoss` on the flattened
  `(4*M_e, C)` dictionary. `use_fused_kernel=True` raises a fail-fast
  RuntimeError because the Task 5.4b Triton kernel is not implemented
  yet. `models/__init__.py` was also rewritten to remove duplicated
  imports found earlier. Local tests cover the dict keys and finite
  scalars, total-loss grad flow (excluding the STE-less entropy gate by
  documented limitation), a 10-step micro-overfit with monotone loss
  decrease, manual coherence checks, orthogonal-basis coherence (< 0.35),
  gate-to-load-balance backward flow, all 4 expert banks receiving grad,
  default aggregation sanity, `MoEDictionary` parameter-count identity,
  the fused-kernel fail-fast, gradient-difference gradients, and the
  Charbonnier manual match. `python -m pytest tests/test_master_restorer.py
  -q` passed (12 tests); the full suite passed (133 tests).
- **Open items (flagged for review):**
  1. `mu(D) < 0.15` cannot hold for a randomly initialized dictionary; it
     applies after the Task 7.1 seeding/pruning or extended training.
  2. EntropyGate parameters are gradient-free (hard gate); a soft/STE
     variant would let the gate itself train.
  3. Per-token adaptive k_i is collapsed to a batch-mean integer k.
- **Next:** Task 5.4b (GPU): design doc then Triton kernel. Phase 7 (7.1,
  7.2) depends on dataset/GPU access.
- **Last updated:** 2026-10-28

## Phase 7: Full-Scale Multi-Task Pre-Training & Benchmarking

### Task 7.1

- **Title:** K-Means++ Dictionary Seeding & Mutual Coherence Pruning
- **Status:** Not started.
- **Acceptance:** Seed `E=4` expert dictionaries from 100,000 `16x16`
  DIV2K/Flickr2K patch embeddings with K-Means++ (M=4096), enforce
  `mu(D) < 0.15` via unit normalization and Gram-Schmidt, and save
  `checkpoints/pretrained_dictionaries.pth`; reload and assert shape
  `(4096, 64)` with `max_{i != j} |G_ij| < 0.15`.
- **What works / evidence:** Not implemented.
- **What did not / blockers:** Depends on dataset access and the 64-d
  feature extractor choice; superseded the original Phase 6.1 which
  targeted ImageNet embeddings.
- **Next:** After Task 6.2-6.4 land, confirm dataset availability and the
  feature extractor, then implement `scripts/init_dictionaries.py`.
- **Last updated:** 2026-10-28

### Task 7.2

- **Title:** Universal Multi-Task Benchmarking Suite
- **Status:** Not started.
- **Acceptance:** Multi-task training (SR x4, denoising sigma=50,
  dehazing) on DIV2K/CBSD68/SOTS with evaluation on Urban100/CBSD68/SOTS,
  baseline comparison against SwinIR, Restormer, and HAT-L, latency via
  CUDA events over 100 iterations, and structured
  `results/benchmark_summary.json`.
- **What works / evidence:** Not implemented.
- **What did not / blockers:** Depends on the Task 6.4 master model,
  dataset pipelines, and target-GPU access; consolidates the original
  Phase 6.2 universal training task.
- **Next:** After Task 6.4, write `experiments/train_universal.py` and
  `experiments/eval_benchmarks.py`.
- **Last updated:** 2026-10-28
