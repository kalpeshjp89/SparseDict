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
- [ ] Task 4.1: Micro-Dataset Data Pipeline (3 hrs; cached Colab benchmark ran but missed target) ([progress](./progress.md#task-41))
  - Implement `data/micro_loader.py` loading exactly 10 high-resolution images (from DIV2K).
  - Add `scripts/download_div2k.py` to download/extract the first 10 DIV2K training-HR images on Colab.
  - Add synthetic degradations: Gaussian noise ($\sigma=25$), bicubic downsampling ($4\times$), and synthetic rain streaks.
  - Verification: DataLoader yields corrupted and clean pairs in $<100$ ms.
  - Status: Loader and downloader were exercised on ten real DIV2K images in Colab. Uncached warmed means were 241.2 ms/pair (noise), 181.1 ms/pair (4x SR), and 201.9 ms/pair (rain). After adding the decoded CPU uint8 cache, the T4 benchmark means were 178.56 ms (noise), 81.32 ms (4x SR), and 133.94 ms (rain); only SR meets the <100 ms mean criterion. `loader_benchmark.json` was inspected in the downloaded archive; see [progress](./progress.md#task-41).
- [ ] Task 4.2: 10-Image Micro-Overfitting Test (4 hrs; Colab run completed, PSNR acceptance not met) ([progress](./progress.md#task-42))
  - Run 200 training epochs of the complete MoE Top-$K$ model on the 10 micro images using $L_1$ reconstruction loss + $\mathcal{L}_{\text{aux}}$.
  - Verification: Reconstruction PSNR must reach $\ge 29.5\text{ dB}$, confirming model capacity and optimization stability.
  - Status: The original 200-epoch run reached 28.6012 dB fixed-crop and 29.3414 dB tiled full-image PSNR. A follow-up T4 run at `d=96` with the tokenwise affine head reached 28.6143 dB fixed-crop and 29.3415 dB tiled; neither meets 29.5 dB. The results archive is downloaded and retained locally. Next, compare the parallel multi-scale dilation head at `d=96` using a separate output directory and otherwise unchanged experiment settings; see [progress](./progress.md#task-42).

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

Phase 5 (Completion): End-to-End Unified Triton Autograd Wrapper
- [ ] Task 5.4b: Unified Triton Autograd Wrapper (4 hrs) ([progress](./progress.md#task-54b))
  - Wrap tiled affinity, register bitonic top-$k$, and online gather-softmax accumulation into `FusedTopKAttentionFunction(torch.autograd.Function)` in `kernels/fused_topk_attn.py`.
  - Forward: $Q \in \mathbb{R}^{B \times H \times N \times d_k}$, $K \in \mathbb{R}^{B \times H \times M \times d_k}$, $V \in \mathbb{R}^{B \times H \times M \times d_v}$; save log-sum-exp $L$ and top-$k$ indices $\mathbf{I}_{\text{topk}}$.
  - Backward: recompute local softmax $P_{i,j} = \exp(S_{i,j} - L_i)$ in SRAM; block-sparse $dQ$, $dK$, $dV$ without materializing full $N \times M$ scores.
  - Verification: `pytest tests/test_triton_autograd.py` with `torch.autograd.gradcheck` on float64; max analytical gradient error $< 10^{-4}$.

Phase 6 (New): Advanced Architectural Extensions
- [x] Task 6.1: Low-Coherence Regularization & Dictionary Distillation (2 hrs) ([progress](./progress.md#task-61))
  - Implement `models/losses/coherence_loss.py` with `MutualCoherenceLoss`: Welch bound $\mu_{\text{WB}}$, Frame Potential $\Vert{}G - I_M\Vert{}_F^2$, and hinge loss with $\lambda_{\text{FP}}=1.0$, $\lambda_{\text{hinge}}=10.0$.
  - Verification: Orthogonal matrix gives loss $< 10^{-6}$; collinear matrix gives loss $> 5.0$.
- [x] Task 6.2: Spatial-Frequency Dual-Domain MoE Partitioning (4 hrs) ([progress](./progress.md#task-62))
  - Implement `models/modules/dual_domain_moe.py`: $E=4$ experts split into two spatial, one wavelet low-pass ($LL$), and one high-frequency (wavelet $LH/HL/HH$ + Fourier amplitude) experts; Haar DWT/rFFT forward with idwt2/irfft2; channel-gated $1\times 1$ fusion.
  - Verification: Forward on $(2, 64, 64, 64)$ returns identical shape; gradients reach all 4 expert sub-dictionaries.
- [x] Task 6.3: Adaptive Entropy-Gated Dynamic Top-$K$ Router (4 hrs) ([progress](./progress.md#task-63))
  - Implement `models/modules/entropy_gate.py`: softmax vocab projection ($V=256$), Shannon entropy vs. EMA threshold $\tau_H = \mu_H + \kappa\sigma_H$ ($\kappa=0.2$), binary bypass gate $g_i$, adaptive $k_i \in [k_{\min}, k_{\max}]$ ($8, 64$), and linear $W_{\text{skip}}$ bypass.
  - Verification: Synthetic noise (high entropy) triggers 100% activation; constant tensor triggers 0%.
- [x] Task 6.4: Master Restorer Architecture Integration (4 hrs) ([progress](./progress.md#task-64))
  - Unify parallel dilated affine modulation, dual-domain MoE dictionaries, entropy-gated dynamic top-$K$, and `FusedTopKAttentionFunction` in `models/moe_topk_restorer.py`.
  - Loss return: $(I_{\text{rec}}, \text{loss\_dict})$ with charbonnier, gradient-difference, auxiliary load-balance, and coherence losses.
  - Verification: 10-epoch micro-overfitting on 10 DIV2K images; monotone training loss decrease and mutual coherence $\mu(D) < 0.15$.

Phase 7: Full-Scale Multi-Task Pre-Training & Benchmarking
- [ ] Task 7.1: $K$-Means++ Dictionary Seeding & Mutual Coherence Pruning (4 hrs) ([progress](./progress.md#task-71))
  - Extract 100,000 $16 \times 16$ patches from DIV2K/Flickr2K; extract 64-d latent features with a pre-trained extractor.
  - $K$-Means++ clustering to $M = 4096$ centroids; unit normalization and Gram-Schmidt orthogonalization to enforce $\mu(D) < 0.15$.
  - Save `checkpoints/pretrained_dictionaries.pth`.
  - Verification: Reload checkpoint; assert shape $(4096, 64)$ and $\max_{i \neq j} \vert{}G_{i,j}\vert{} < 0.15$.
- [ ] Task 7.2: Universal Multi-Task Benchmarking Suite (4 hrs) ([progress](./progress.md#task-72))
  - Implement `experiments/train_universal.py` (SR $\times 4$, denoising $\sigma=50$, dehazing on DIV2K, CBSD68, SOTS) and `experiments/eval_benchmarks.py` (Urban100, CBSD68, SOTS test sets).
  - Compare PSNR/SSIM, FLOP reduction, and latency vs. SwinIR, Restormer, HAT-L baselines; latency via `torch.cuda.Event(enable_timing=True)` over 100 iterations.
  - Export `results/benchmark_summary.json`.
  - Verification: Script executes the test suite automatically and produces complete comparison tables matching target metrics.
