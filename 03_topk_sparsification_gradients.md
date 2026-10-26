# Module Specification 03: Top-K Sparsification & Backpropagation

## 1. Hard Top-K Selection Operator
Given query projections $Q_X = X W_Q \in \mathbb{R}^{N \times d_k}$ and routed key dictionary $K_{\text{MoE}} \in \mathbb{R}^{M_{\text{act}} \times d_k}$, compute raw affinity scores $S \in \mathbb{R}^{N \times M_{\text{act}}}$:

$$S_{i,j} = \frac{(Q_X)_{i,\cdot} (K_{\text{MoE}})_{j,\cdot}^\top}{\tau \sqrt{d_k}}$$

Let $\mathcal{I}_{i,k}$ be the set of column indices corresponding to the $k$ largest values in row $i$ of $S$. The masked affinity matrix $\tilde{S}$ is:

$$\tilde{S}_{i,j} = \begin{cases} S_{i,j} & \text{if } j \in \mathcal{I}_{i,k} \\ -\infty & \text{otherwise} \end{cases}$$

Applying Softmax yields sparse regression weights $\hat{\alpha}$:

$$\hat{\alpha}_{i,j} = \frac{\exp(\tilde{S}_{i,j})}{\sum_{m \in \mathcal{I}_{i,k}} \exp(S_{i,m})}$$

Value aggregation reduces to gathering only active rows:

$$\hat{X}_{i,\cdot} = \sum_{j \in \mathcal{I}_{i,k}} \hat{\alpha}_{i,j} (V_{\text{MoE}})_{j,\cdot}$$

## 2. Temperature Scaling ($\tau$)
To maintain high selectivity as dictionary size scales, temperature $\tau$ is set as:

$$\tau = \tau_0 \cdot \frac{\log(M_{\text{active}})}{\log(M_0)}$$

During training, $\tau_0$ anneals from $1.0$ down to $0.05$ via cosine decay.

## 3. Differentiable Gradient Estimators
### Primary Estimator: Straight-Through Estimator (STE)
During forward pass, hard top-$k$ masking is executed. During backward pass, the thresholding mask is treated as an identity operator for selected indices:

$$\text{Forward: } y = \text{Top-}K(S)$$

$$\text{Backward: } \frac{\partial \mathcal{L}}{\partial S_{i,j}} \approx \begin{cases} \frac{\partial \mathcal{L}}{\partial y_{i,j}} & \text{if } j \in \mathcal{I}_{i,k} \\ 0 & \text{otherwise} \end{cases}$$

### Fallback Estimator: Gumbel-Perturbed Top-K
For fine-tuning dictionary weights, stochastic Gumbel noise $Z \sim \text{Gumbel}(0, I)$ is added:

$$S_{\text{noisy}} = S + \sigma \cdot Z$$

Expectation gradients are evaluated using 2 Monte Carlo samples per batch pass.
