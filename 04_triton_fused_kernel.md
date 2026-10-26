# Module Specification 04: Hardware Acceleration via Fused Triton Kernel

## 1. Kernel Architecture Overview
Standard framework implementations materializing $N \times M$ affinity matrices in GPU global memory (HBM) trigger severe bandwidth bottlenecks. This custom Triton kernel executes affinity scoring, bitonic top-$k$ sorting, online softmax, and value aggregation entirely within GPU SRAM registers.

```
+-------------------------------------------------------------------------+
|                        GPU GLOBAL MEMORY (HBM)                          |
|    Query Matrix Q (N x d)              Expert Dictionary K, V (M x d)   |
+-------------------------------------------------------------------------+
                                     |
                             Streaming Read Tiles
                           (Tiled Loads into SRAM)
                                     v
+-------------------------------------------------------------------------+
|                         ON-CHIP SRAM REGISTERS                          |
|  1. Compute Query-Key Tile Affinity: S_tile = Q_tile @ K_tile.T         |
|  2. Parallel Bitonic Selection: Maintain top-k scores in registers      |
|  3. Online Softmax Normalization: Scale scores by running max/sum       |
|  4. Gather-and-Densify GEMM: Output_tile += Softmax_tile @ V_tile       |
+-------------------------------------------------------------------------+
                                     |
                           Coalesced Store Output
                          (Single HBM Write Pass)
                                     v
+-------------------------------------------------------------------------+
|                        GPU GLOBAL MEMORY (HBM)                          |
|         Final Reconstructed Feature Output Matrix  X_out (N x d)        |
+-------------------------------------------------------------------------+
```

---

## 2. Two-Level Tiled Execution Strategy
- **Logical Blocks:** $B_q = 64$ (Query block length), $B_k = 128$ (Key block length).
- **Physical Tiles:** $B_r = 32$, $B_c = 32$ matched to NVIDIA Tensor Core instruction tiles ($16 \times 16$ MMA).

---

## 3. In-Register Bitonic Sorting Algorithm
Inside the kernel's inner loop over key blocks:
1. Load key tile $K_{\text{tile}} \in \mathbb{R}^{B_c \times d_k}$ into SRAM.
2. Evaluate local GEMM: $S_{\text{block}} = Q_{\text{tile}} K_{\text{tile}}^\top$.
3. Execute bitonic sorting network across the $B_c$ score outputs, maintaining a register array of size $k$ storing $(S_{\text{val}}, \text{Index}_{\text{col}})$.
4. Discard low-scoring elements directly inside registers without HBM writes.

---

## 4. Gather-and-Densify Accumulation Pipeline
Once top-$k$ column indices $\mathcal{I}_k$ are determined for the current tile:
1. Gather corresponding value vectors $V[\mathcal{I}_k, :] \in \mathbb{R}^{k \times d_v}$ from HBM to SRAM.
2. Execute online softmax update:

$$m_{\text{new}} = \max(m_{\text{old}}, S_{\text{top}k})$$

$$d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{S_{\text{top}k} - m_{\text{new}}}$$

$$O_{\text{acc}} = O_{\text{acc}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \text{Softmax}(S_{\text{top}k}) \cdot V[\mathcal{I}_k, :]$$

3. Store normalized output $O_{\text{final}} = \frac{O_{\text{acc}}}{d_{\text{final}}}$ directly to HBM.
