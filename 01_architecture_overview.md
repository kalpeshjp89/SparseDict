# Module Specification 01: Universal Architecture & Hybrid Attention

## 1. Architectural Overview
The architecture implements a dual-pathway Vision Transformer designed for universal image restoration. For an input degraded image $I_{\text{LQ}} \in \mathbb{R}^{H \times W \times 3}$, a shallow feature convolution extracts token sequence $X_0 \in \mathbb{R}^{N \times d}$, where $N = \frac{H}{P} \times \frac{W}{P}$ and $d$ is the hidden dimension.

The backbone consists of cascaded Transformer Blocks containing two parallel pathways:
- **Local Internal Prior Pathway:** Shifted Window Multi-Head Self-Attention (SW-MSA) capturing spatial self-similarity within local $W \times W$ windows.
- **Global External Prior Pathway:** MoE-Routed Top-$K$ Token Dictionary Cross-Attention (MoE-TopK-TDCA) retrieving continuous visual primitives from learned external dictionary banks.

```
+------------------------------------+
|   Input Token Sequence X (N x d)   |
+------------------------------------+
                  |
        +---------+---------+
        |                   |
        v                   v
+---------------------------+     +---------------------------+
| Shifted Window Self-Attn  |     | MoE Router G(X) & Gating  |
|         (SW-MSA)          |     |  Top-K Dict Cross-Attn    |
+---------------------------+     +---------------------------+
        |                   |
        +---------+---------+
                  |
                  v
+-----------------------------------+
|     Adaptive Feature Fusion       |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|   Category-Aware FFN Block        |
+-----------------------------------+
```

---

## 2. Tensor Shapes & Dimensions
- Batch Size ($B$): Dynamic ($1, 2, 4, \dots$)
- Token Count ($N$): $H \times W$ (e.g., $N = 4096$ for $64 \times 64$ patch)
- Hidden Dimension ($d$): $64$ (Lightweight) or $180$ (Standard)
- Key/Query Projection Dimension ($d_k$): $d / r$, where $r = 4$ (Reduction ratio)
- Global Dictionary Atoms ($M$): $4096$ total across all experts
- Active Expert Count ($E$): $4$ experts total ($E_{\text{act}} = 1$ or $2$ active per token)
- Top-$K$ Selection Count ($k$): $32$ active atoms per query token

---

## 3. Dual-Pathway Feature Fusion
For input tokens $X^{(l)}$ at layer $l$, the outputs of both pathways are fused via learned layer scaling parameters $\gamma_1, \gamma_2 \in \mathbb{R}^d$:

$$X_{\text{fused}}^{(l)} = X^{(l)} + \gamma_1 \odot \text{SW-MSA}(X^{(l)}) + \gamma_2 \odot \text{MoE-TopK-TDCA}(X^{(l)})$$

The fused output passes into a Category-Aware Feed-Forward Network (FFN) with depthwise convolutions to produce final layer representations $X^{(l+1)}$.
