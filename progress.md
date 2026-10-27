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
- **Status:** Pipeline and decoded-image cache are implemented; the real-data benchmark ran on Colab but did not meet the <100 ms acceptance target.
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
- **Next:** Profile noise and rain generation separately from cached image
  fetch; retain super-resolution as a passing mode for this timing procedure.
- **Last updated:** 2026-10-26

### Task 4.2

- **Title:** 10-Image Micro-Overfitting Test
- **Status:** Training/inference code implemented and locally smoke-tested; the 200-epoch DIV2K noise run and full-image inference completed on Colab, but the 29.5 dB acceptance target was not met.
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
  samples deterministic per-epoch crops, optimizes L1 plus auxiliary loss,
  logs fixed-crop validation PSNR, saves metrics and a loadable checkpoint,
  and reports whether the 29.5 dB criterion passed. The inference runner
  loads that checkpoint, bicubic-upsamples SR inputs, runs overlap-tiled
  prediction on full images, saves restored PNGs, and writes per-image/mean
  PSNR and latency. Local tests cover forward shapes, gradient flow, two-epoch
  training/checkpoint round-trip, and tiled image output. The complete local
  suite passed (67 tests), and the Phase 1 runner remains green.
- **Colab run:** Executed 200 epochs on the ten-image DIV2K subset on a Tesla T4
  (`torch 2.11.0+cu130`, CUDA available), using noise degradation, `d=96`, and
  the identity-initialized tokenwise affine head. Fixed-crop validation PSNR
  was **28.6143 dB**, below the **29.5 dB** target. Tiled full-image inference
  completed for all ten images; mean full-image PSNR was **29.3415 dB**, also
  below target by about **0.1585 dB**. The previous run reported 28.6012 dB
  fixed-crop and 29.3414 dB tiled PSNR, but this iteration changed both model
  width and the modulation head, so it does not isolate the capacity effect.
  These evaluation procedures are distinct and should not be conflated.
- **Preserved artifacts:** The downloaded `C:\Users\Kalpesh\Downloads\s.zip`
  contains the `d=96` tokenwise-affine checkpoint, training metrics, tiled
  restored images and metrics, and `loader_benchmark.json`. It is retained
  locally; its measured results are the baseline for the next comparison.
- **What did not / blockers:** The experiment completed and produced a
  checkpoint, but it did not meet the PSNR acceptance criterion. The local
  environment remains CPU-only and has no DIV2K files. The baseline checkpoint
  and metrics archive is available locally; the new parallel-head comparison
  still requires Colab GPU execution.
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
  are upsampled before tiling. The local suite now passes (73 tests), but the
  updated 200-epoch DIV2K run has not yet been executed, so the 29.5 dB target
  remains unverified.
- **Next:** Re-run the parallel multi-scale dilation head at `d=96` on Colab
  with the updated loss and routing schedule. Compare full-image tiled
  validation and inference PSNR using matching tile settings; do not mark
  Task 4.2 complete unless tiled validation reaches 29.5 dB.
- **Last updated:** 2026-10-27

## Phase 5: Hardware-Aware Acceleration via Fused Triton Kernel

### Task 5.1

- **Title:** Triton Kernel Block-Tile Layout & SRAM Loader
- **Status:** Not started.
- **Acceptance:** Add a fused-kernel module with query/key tiles of 64 and
  coalesced loads; compile without execution crashes.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** Triton and compatible GPU availability have not
  been checked. Phase 5 implementation is gated on Tasks 4.1 and 4.2 meeting
  their acceptance criteria.
- **Next:** After both Phase 4 acceptance checks pass, check GPU/Triton
  availability and the supported toolchain; implement the tile/load structure
  and record compile validation.
- **Last updated:** 2026-10-26

### Task 5.2

- **Title:** In-Register Parallel Bitonic Top-K Selector
- **Status:** Not started.
- **Acceptance:** Keep top-k values in registers and match `torch.topk` indices
  exactly on the defined test cases.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** Depends on a compatible Triton development and
  test environment.
- **Next:** Implement and unit-test the selector against `torch.topk`,
  including ties and supported key-tile sizes; record exact comparison results.
- **Last updated:** 2026-10-26

### Task 5.3

- **Title:** SRAM Gather-and-Densify Online Softmax Accumulator
- **Status:** Not started.
- **Acceptance:** Fuse selected-value aggregation and online softmax; match the
  PyTorch reference within max absolute error `1e-5`.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** Depends on the fused kernel and compatible
  hardware/runtime.
- **Next:** Implement aggregation after Task 5.1/5.2; compare outputs to a
  PyTorch reference over representative shapes and record max absolute error.
- **Last updated:** 2026-10-26

### Task 5.4

- **Title:** Latency & Memory Bandwidth Benchmarking
- **Status:** Not started.
- **Acceptance:** Benchmark PyTorch and Triton at `N={1024, 4096, 16384}` and
  show at least 2.5x latency reduction at `N=4096`.
- **What works / evidence:** No benchmark or measurements recorded.
- **What did not / blockers:** Depends on a working fused kernel and compatible
  GPU; no hardware details are recorded.
- **Next:** Add `tests/bench_kernel.py` with warmup, synchronization, repeated
  measurements, and environment metadata; record results for each token length.
- **Last updated:** 2026-10-26

## Phase 6: Multi-Task Scaling & Dataset Benchmarking

### Task 6.1

- **Title:** Full Dataset Pre-Training (Self-Supervised Cluster Seeding)
- **Status:** Not started.
- **Acceptance:** Seed four expert dictionaries with K-Means++ over ImageNet
  feature embeddings and verify centroids are well-separated in latent cosine
  distance.
- **What works / evidence:** No implementation, embeddings, or measurements
  recorded.
- **What did not / blockers:** ImageNet access, feature extraction, compute
  budget, and an explicit centroid-separation threshold are not established.
- **Next:** Confirm data/compute availability and define a measurable
  separation criterion before running clustering; log embedding provenance and
  clustering parameters.
- **Last updated:** 2026-10-26

### Task 6.2

- **Title:** Multi-Task Universal Training Execution
- **Status:** Not started.
- **Acceptance:** Train across DIV2K, Flickr2K, CBSD68, and SOTS for
  super-resolution, denoising, and dehazing; monitor validation PSNR/SSIM and
  expert-load distribution.
- **What works / evidence:** No complete training run or metrics recorded.
- **What did not / blockers:** Depends on the full model, data pipelines,
  training configuration, and available compute; dataset access is not
  established.
- **Next:** After micro-validation succeeds, define dataset splits, task
  sampling, metrics, and checkpointing; run a reproducible training job and
  preserve configuration and logs.
- **Last updated:** 2026-10-26
