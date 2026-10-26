# Module Specification 02: Mixture-of-Experts Dictionary Routing

## 1. Expert Dictionary Partitioning
The external prior memory is divided into $E$ discrete Expert Dictionaries:

$$\mathcal{D} = \{D_1, D_2, \dots, D_E\}, \quad D_e \in \mathbb{R}^{M_e \times d}$$

where $M_e = \frac{M}{E}$ is the number of dictionary atoms assigned to expert $e$.

## 2. Routing Network Formulation
The gating router $\mathcal{G}_\theta(X)$ processes query tokens to select expert dictionaries.

### Option A: Global Image-Level Routing
For uniform degradations (e.g., global Gaussian noise, uniform downsampling), features are pooled globally:

$$h_{\text{global}} = \frac{1}{N} \sum_{i=1}^N X_i$$

$$g = \text{Softmax}\left( W_2 \cdot \text{GELU}(W_1 h_{\text{global}}) + \epsilon \right)$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise added during training to encourage expert exploration.

### Option B: Token-Level Spatial Routing
For spatially varying degradations (e.g., local rain streaks, mixed haze), routing occurs per token $i \in \{1, \dots, N\}$:

$$g_i = \text{Softmax}\left( W_2 \cdot \text{GELU}(W_1 X_i) + \epsilon \right)$$

## 3. Dynamic Key/Value Matrix Construction
Let $\mathcal{I}_{\text{act}} = \text{Top-}E_{\text{act}}(g_i)$ be the set of selected expert indices for token $i$. The composite key and value representations are constructed on-the-fly:

$$K_{\text{MoE}}(X_i) = \sum_{e \in \mathcal{I}_{\text{act}}} g_{i,e} \cdot (D_e W_K^e)$$

$$V_{\text{MoE}}(X_i) = \sum_{e \in \mathcal{I}_{\text{act}}} g_{i,e} \cdot (D_e W_V^e)$$

## 4. Auxiliary Load-Balancing Loss ($\mathcal{L}_{\text{aux}}$)
To prevent router collapse (where all tokens select the same expert), an auxiliary loss is enforced:

$$\text{Importance}(e) = \sum_{i=1}^N g_{i,e}$$

$$\text{Load}(e) = \sum_{i=1}^N \mathbb{I}(e \in \text{Top-}E_{\text{act}}(g_i))$$

$$\mathcal{L}_{\text{aux}} = w_{\text{aux}} \cdot \left( \left(\frac{\text{std}(\text{Importance})}{\text{mean}(\text{Importance})}\right)^2 + \left(\frac{\text{std}(\text{Load})}{\text{mean}(\text{Load})}\right)^2 \right)$$

where $w_{\text{aux}} = 0.01$.
