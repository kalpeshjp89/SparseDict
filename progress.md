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
- **Status:** In progress; shape verification passes, runtime criterion open.
- **Relevant files:** `tests/toy_harness.py`
- **Acceptance:** Generate `X` with shape `(2, 3, 64, 64)` and `D` with shape
  `(256, 64)`; the script prints both shapes and completes in under 2 seconds.
- **What works / evidence:** The script prints the required shapes. A direct
  call to `generate_toy_tensors()` took about 0.0008 seconds after imports.
- **What did not / limitation:** End-to-end script measurements were about 14.36
  seconds on the first run and 7.51 seconds on a subsequent run, so the
  under-2-second script requirement is not met in this environment. Most of
  that time is consistent with Python/PyTorch startup; no profiling has isolated
  startup cost yet. PyTorch also emitted a warning because NumPy is not
  installed.
- **Next:** Decide whether the criterion applies to tensor generation or
  process startup; if process startup is required, profile and reduce startup
  overhead without weakening the required output or adding unnecessary
  dependencies, then repeat a timed run.
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
- **Status:** In progress; implementation exists, acceptance not yet recorded.
- **Relevant files:** `models/modules/dict_attention.py`,
  `experiments/run_phase1.py`
- **Acceptance:** Offline dictionary projections have shapes `(M, d_k)` and
  `(M, d)`, with `d_k = d / 4`; output shape is `(B, N, d)`.
- **What exists:** `DenseDictCrossAttention` contains `w_k`, `w_v`, and `w_q`,
  dictionary projection via `project_dictionary()`, and an output-shape check
  in the Phase 1 runner.
- **Verification/evidence:** The Phase 1 runner has not been run in the
  recorded work.
- **Next:** Run the Task 1.1 shape check in `experiments/run_phase1.py`, inspect
  the reported projection/output shapes, and record the result before checking
  off the task.
- **Last updated:** 2026-10-26

### Task 1.2

- **Title:** Un-sparsified Dense Softmax Cross-Attention
- **Status:** In progress; implementation exists, acceptance not yet recorded.
- **Relevant files:** `models/modules/dict_attention.py`,
  `experiments/run_phase1.py`, `configs/hyperparams.py`
- **Acceptance:** Use scaled dense softmax attention with
  `tau = tau_0 * log(M) / log(M_0)`; complete 50 toy optimization steps and
  assert training loss strictly decreases.
- **What exists:** `forward()` implements scaled affinity, dense softmax, and
  dictionary value aggregation. `temperature()` implements the specified
  logarithmic scaling. The runner performs 50 Adam steps and checks strict
  decrease.
- **Verification/evidence:** The Phase 1 runner has not been run in the
  recorded work; no loss trajectory is recorded.
- **Next:** Run `python experiments\run_phase1.py`, check both task results and
  the 50-step loss trajectory, and investigate any non-decreasing step before
  marking complete.
- **Last updated:** 2026-10-26

## Phase 2: Mixture-of-Experts Dictionary Partitioning

### Task 2.1

- **Title:** Partitioned Expert Dictionary Initialization
- **Status:** Not started.
- **Acceptance:** Partition `M` atoms into `E=4` expert banks; support static
  projections and optional dynamic residual refinement; verify total dictionary
  parameter count is exactly `M * d` with no duplicated atom parameters.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** None known.
- **Next:** Implement `models/modules/moe_dict.py`; decide the representation
  for shared storage and optional residual refinement from the existing MoE
  specification, then test shapes and parameter count.
- **Last updated:** 2026-10-26

### Task 2.2

- **Title:** Degradation-Aware Routing Network
- **Status:** Not started.
- **Acceptance:** Implement global spatial pooling and a two-layer MLP with
  Gaussian jitter; provide image-level and token-level routes; top-`E_act`
  gating weights must sum to 1 per routed item.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** None known.
- **Next:** Implement `models/modules/router.py` following the routing
  specification and test sparse expert selection and normalized weights for
  both routing paths.
- **Last updated:** 2026-10-26

### Task 2.3

- **Title:** Auxiliary Load-Balancing Loss Module
- **Status:** Not started.
- **Acceptance:** Implement weighted squared coefficient-of-variation loss for
  expert importance and load; a single-expert-only synthetic gating case must
  yield loss greater than `0.5`.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** None known.
- **Next:** Implement `models/losses/balance_loss.py` and test balanced and
  collapsed routing inputs, including the required collapsed-case threshold.
- **Last updated:** 2026-10-26

## Phase 3: Top-K Sparsification & Differentiable Backpropagation

### Task 3.1

- **Title:** Hard Top-K Masking & Sparse Indexing
- **Status:** Not started.
- **Acceptance:** Mask all non-top-k affinity scores to negative infinity and
  aggregate values using only selected indices; verify exactly `k` selected
  entries per query.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** None known.
- **Next:** Implement `models/modules/topk_select.py`; test selected indices,
  masked scores, and sparse aggregation against a small dense reference.
- **Last updated:** 2026-10-26

### Task 3.2

- **Title:** Differentiable Backward Pass (STE & Perturbed Gumbel)
- **Status:** Not started.
- **Acceptance:** Implement the specified STE and Perturbed Top-K fallback;
  gradcheck mock scores and verify non-zero gradients reach query projection
  and dictionary parameters.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** None known.
- **Next:** Define and test the estimator's forward/backward contract, then
  integrate it with Task 3.1 and run the specified gradient checks.
- **Last updated:** 2026-10-26

## Phase 4: Micro-Validation & Overfitting Smoke Test

### Task 4.1

- **Title:** Micro-Dataset Data Pipeline
- **Status:** Not started.
- **Acceptance:** Load exactly 10 DIV2K images and generate Gaussian-noise,
  4x bicubic-downsample, and synthetic-rain degradations; yield corrupted and
  clean pairs in under 100 ms.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** Dataset location and availability have not been
  established.
- **Next:** Confirm how DIV2K data will be provided and its local path; build
  `data/micro_loader.py` with deterministic selection/degradation controls and
  measure loader latency.
- **Last updated:** 2026-10-26

### Task 4.2

- **Title:** 10-Image Micro-Overfitting Test
- **Status:** Not started.
- **Acceptance:** Train the complete MoE Top-K model for 200 epochs on the
  10-image micro-dataset with L1 reconstruction and auxiliary losses; reach at
  least 29.5 dB reconstruction PSNR.
- **What works / evidence:** No complete model or result recorded.
- **What did not / blockers:** Depends on the model, routing, loss, and dataset
  pipeline tasks, as well as access to the data and suitable hardware.
- **Next:** After prerequisites are verified, add a reproducible training
  script with saved configuration and PSNR/loss results; run the 200-epoch
  criterion.
- **Last updated:** 2026-10-26

## Phase 5: Hardware-Aware Acceleration via Fused Triton Kernel

### Task 5.1

- **Title:** Triton Kernel Block-Tile Layout & SRAM Loader
- **Status:** Not started.
- **Acceptance:** Add a fused-kernel module with query/key tiles of 64 and
  coalesced loads; compile without execution crashes.
- **What works / evidence:** No implementation or verification recorded.
- **What did not / blockers:** Triton and compatible GPU availability have not
  been checked.
- **Next:** Check GPU/Triton environment and supported toolchain; implement the
  tile/load structure and record compile validation.
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
