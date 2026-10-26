Engineering Implementation Roadmap: MoE-Routed Top-K Sparse Dictionary Transformer

Principles
- **Fail-Fast Methodology**: Validate mathematical correctness and gradient flow on synthetic or micro-data before executing full dataset training.
- **Strict Time Boxing**: Every sub-task is scoped to 2 to 4 hours of execution.
- **Hardware Co-Design Validation**: Verify latency speedup using hardware profiling before scaling batch sizes.
- Progress notes, results, blockers, and next steps for every task: [progress.md](./progress.md).

Phase 0: Infrastructure & Synthetic Environment Setup
- [x] Task 0.1: Synthetic Harness & Tensor Generator (2 hrs) ([progress](./progress.md#task-01))
  - Implement `tests/toy_harness.py` generating mock low-quality inputs $X \in \mathbb{R}^{B \times C \times H \times W}$ ($B=2, H=64, W=64$).
  - Generate mock dictionary matrix $D \in \mathbb{R}^{M \times d}$ ($M=256, d=64$).
  - Reference: `docs/01_architecture_overview.md#tensor-shapes--dimensions`
  - Verification: Shapes verified and generation takes 0.0008 sec after PyTorch import. End-to-end startup exceeds 2 sec on CPU due to PyTorch import; GPU runtime verification is deferred.
- [x] Task 0.2: Automated Gradient & Shape Profiler (2 hrs) ([progress](./progress.md#task-02))
  - Implement a check script (`tests/check_grad.py`) that captures model outputs, runs `loss.backward()`, and asserts non-zero, non-NaN gradients across all trainable weights.
  - Verification: `pytest tests/check_grad.py` passes without error.

Phase 1: Dense Dictionary Cross-Attention Baseline
- [x] Task 1.1: Dense Key/Value Projection Module (3 hrs) ([progress](./progress.md#task-11))
  - Build `models/modules/dict_attention.py` implementing offline key/value projections: $K_D = D W_K$, $V_D = D W_V$.
  - Implement linear reduction ratio $r=4$ for key/query projections.
  - Reference: `docs/01_architecture_overview.md#dense-dictionary-cross-attention`
  - Verification: Assert output tensor shape is $X_{\text{out}} \in \mathbb{R}^{B \times N \times d}$.
- [x] Task 1.2: Un-sparsified Dense Softmax Cross-Attention (3 hrs) ([progress](./progress.md#task-12))
  - Implement standard scaled dot-product affinity: $\alpha = \text{Softmax}\left(\frac{Q_X K_D^\top}{\tau \sqrt{d_k}}\right) V_D$.
  - Add temperature scaling factor $\tau = \tau_0 \cdot \frac{\log(M)}{\log(M_0)}$.
  - Reference: `docs/03_topk_sparsification_gradients.md#temperature-scaling`
  - Verification: Perform 50 optimization steps on toy dataset; assert training loss strictly decreases.

Phase 2: Mixture-of-Experts (MoE) Dictionary Partitioning
- [x] Task 2.1: Partitioned Expert Dictionary Initialization (3 hrs) ([progress](./progress.md#task-21))
  - Implement `models/modules/moe_dict.py` partitioning dictionary memory into $E=4$ expert banks $\{D_1, D_2, D_3, D_4\} \in \mathbb{R}^{\frac{M}{E} \times d}$.
  - Add support for both static pre-computed projections and optional dynamic residual refinement.
  - Reference: `docs/02_moe_dictionary_routing.md#expert-dictionary-partitioning`
  - Verification: Confirm parameter count matches $M \times d$ exactly without memory duplication.
- [x] Task 2.2: Degradation-Aware Routing Network $\mathcal{G}(X)$ (4 hrs) ([progress](./progress.md#task-22))
  - Build gating module in `models/modules/router.py` using global spatial pooling + 2-layer MLP with Gaussian noise jitter $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
  - Implement both Image-Level and Token-Level routing paths.
  - Reference: `docs/02_moe_dictionary_routing.md#routing-network-formulation`
  - Verification: Verify router outputs top-$E_{\text{act}}$ sparse gating weights $g_{i,e}$ summing to 1.0.
- [x] Task 2.3: Auxiliary Load-Balancing Loss Module (2 hrs; conditional, pending confirmation) ([progress](./progress.md#task-23))
  - Code coefficient of variation loss $\mathcal{L}_{\text{aux}} = w_{\text{aux}} \cdot \left(\text{CV}(\text{Imp})^2 + \text{CV}(\text{Load})^2\right)$ in `models/losses/balance_loss.py`.
  - Reference: `docs/02_moe_dictionary_routing.md#auxiliary-load-balancing-loss`
  - Verification: Test with artificial 100% single-expert gating inputs; verify $\mathcal{L}_{\text{aux}} > 0.5$.
  - Observation: With the specified $w_{\text{aux}}=0.01$, the implemented loss is 0.06 for this case; the raw CV sum is 6.0. A test-only $w_{\text{aux}}=0.1$ yields 0.6. User confirmation of the authoritative threshold/weight is pending; see [progress](./progress.md#task-23).

Phase 3: Top-$K$ Sparsification & Differentiable Backpropagation
- [x] Task 3.1: Hard Top-$K$ Masking & Sparse Indexing (3 hrs) ([progress](./progress.md#task-31))
  - Implement hard thresholding operator in `models/modules/topk_select.py` setting all non-top-$k$ unnormalized affinity scores to $-\infty$.
  - Collapse value aggregation to gather only top-$k$ rows.
  - Reference: `docs/03_topk_sparsification_gradients.md#hard-top-k-forward-pass`
  - Verification: Confirm sparse softmax affinity has exactly $k$ non-zero elements per query row.
- [x] Task 3.2: Differentiable Backward Pass (STE & Perturbed Gumbel) (4 hrs) ([progress](./progress.md#task-32))
  - Wrap the Top-$K$ operator in a custom PyTorch `torch.autograd.Function` implementing the Straight-Through Estimator (STE) and Perturbed Top-$K$ fallback.
  - Reference: `docs/03_topk_sparsification_gradients.md#differentiable-gradient-estimators`
  - Verification: Run `torch.autograd.gradcheck` on mock scores; confirm non-zero gradients flow back to $W_Q$ and dictionary parameters.

Phase 4: Micro-Validation & Overfitting Smoke Test
- [ ] Task 4.1: Micro-Dataset Data Pipeline (3 hrs; local code complete, dataset verification pending) ([progress](./progress.md#task-41))
  - Implement `data/micro_loader.py` loading exactly 10 high-resolution images (from DIV2K).
  - Add `scripts/download_div2k.py` to download/extract the first 10 DIV2K training-HR images on Colab.
  - Add synthetic degradations: Gaussian noise ($\sigma=25$), bicubic downsampling ($4\times$), and synthetic rain streaks.
  - Verification: DataLoader yields corrupted and clean pairs in $<100$ ms.
  - Status: Loader and downloader implemented and locally fixture-tested; real DIV2K/Colab latency verification is pending because the dataset and GPU runtime are not available locally.
- [ ] Task 4.2: 10-Image Micro-Overfitting Test (4 hrs; code complete, Colab run pending) ([progress](./progress.md#task-42))
  - Run 200 training epochs of the complete MoE Top-$K$ model on the 10 micro images using $L_1$ reconstruction loss + $\mathcal{L}_{\text{aux}}$.
  - Verification: Reconstruction PSNR must reach $\ge 29.5\text{ dB}$, confirming model capacity and optimization stability.

Phase 5: Hardware-Aware Acceleration via Fused Triton Kernel
- [ ] Task 5.1: Triton Kernel Block-Tile Layout & SRAM Loader (4 hrs) ([progress](./progress.md#task-51))
  - Create `kernels/fused_topk_attn.py` establishing 2D grid launches with query tiles $B_r=64$ and key tiles $B_c=64$.
  - Write SRAM load logic with memory coalescing.
  - Reference: `docs/04_triton_fused_kernel.md#tiled-memory-layout`
  - Verification: Triton kernel compiles cleanly without execution crashes.
- [ ] Task 5.2: In-Register Parallel Bitonic Top-$K$ Selector (4 hrs) ([progress](./progress.md#task-52))
  - Write bitonic sort routine inside Triton registers to maintain top-$k$ scores without HBM sorting writes.
  - Reference: `docs/04_triton_fused_kernel.md#in-register-bitonic-sorting`
  - Verification: Compare sorted indices against `torch.topk` output; verify 100% index match.
- [ ] Task 5.3: SRAM Gather-and-Densify Online Softmax Accumulator (4 hrs) ([progress](./progress.md#task-53))
  - Fused online softmax aggregation and scatter-store of final outputs directly to global memory.
  - Reference: `docs/04_triton_fused_kernel.md#gather-and-densify-accumulation`
  - Verification: Check output equivalence against PyTorch reference implementation ($\max \vert{}\Delta\vert{} < 10^{-5}$).
- [ ] Task 5.4: Latency & Memory Bandwidth Benchmarking (2 hrs) ([progress](./progress.md#task-54))
  - Run latency profiler (`tests/bench_kernel.py`) comparing pure PyTorch vs. Fused Triton Kernel across token lengths $N \in \{1024, 4096, 16384\}$.
  - Verification: Confirm wall-clock latency reduction $\ge 2.5\times$ on $N=4096$.

Phase 6: Multi-Task Scaling & Dataset Benchmarking
- [ ] Task 6.1: Full Dataset Pre-Training (Self-Supervised Cluster Seeding) (4 hrs) ([progress](./progress.md#task-61))
  - Execute $K$-Means++ clustering on ImageNet feature embeddings to seed the $E=4$ expert dictionaries.
  - Reference: `docs/02_moe_dictionary_routing.md#expert-initialization-strategies`
  - Verification: Confirm expert dictionary centroids are well-separated in latent cosine distance.
- [ ] Task 6.2: Multi-Task Universal Training Execution (4 hrs) ([progress](./progress.md#task-62))
  - Train full model on combined dataset (DIV2K, Flickr2K, CBSD68, SOTS) across Super-Resolution, Denoising, and Dehazing tasks.
  - Verification: Monitor validation PSNR/SSIM curves and expert load distribution logs.
