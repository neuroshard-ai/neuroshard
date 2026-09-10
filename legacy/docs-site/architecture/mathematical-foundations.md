# Mathematical Foundations

This document provides a complete mathematical treatment of all algorithms and techniques used in NeuroShard's distributed LLM training system. Every equation is explained with intuition and derivation.

## 1. Training Objective

### 1.1 Language Modeling Loss

NeuroShard trains a causal language model to predict the next token. Given a sequence of tokens $x_1, x_2, \ldots, x_T$, the objective is to maximize:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})
$$

Where:
- $\theta$ = model parameters
- $x_t$ = token at position $t$
- $x_{<t}$ = all tokens before position $t$
- $P_\theta$ = probability distribution from the model

### 1.2 Cross-Entropy Loss

The model outputs logits $z \in \mathbb{R}^V$ (where $V$ is vocabulary size), converted to probabilities via softmax:

$$
P(x_t = k | x_{<t}) = \frac{\exp(z_k)}{\sum_{j=1}^{V} \exp(z_j)} = \text{softmax}(z)_k
$$

The cross-entropy loss for a single token with true label $y$ is:

$$
\mathcal{L}_{\text{CE}} = -\log P(y) = -z_y + \log \sum_{j=1}^{V} \exp(z_j)
$$

**Gradient with respect to logits:**

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = P(k) - \mathbb{1}[k = y] = \text{softmax}(z)_k - \mathbb{1}[k = y]
$$

This elegant result shows the gradient is simply the difference between predicted probability and the one-hot target.

---

## 2. The DiLoCo Algorithm

DiLoCo (Distributed Low-Communication training) is a two-level optimization algorithm that reduces communication by orders of magnitude.

### 2.1 Algorithm Overview

**Inner Loop** (local, no communication):
$$
\theta_{t+1}^{(i)} = \theta_t^{(i)} - \eta_{\text{inner}} \cdot g_t^{(i)}
$$

Where $g_t^{(i)} = \nabla_\theta \mathcal{L}(\theta_t^{(i)}, \mathcal{B}_t^{(i)})$ is the gradient on node $i$ for batch $\mathcal{B}_t^{(i)}$.

**Pseudo-Gradient Computation** (after $H$ inner steps):
$$
\Delta\theta^{(i)} = \theta_0^{(i)} - \theta_H^{(i)} = \sum_{t=0}^{H-1} \eta_{\text{inner}} \cdot g_t^{(i)}
$$

**Aggregation** (across $N$ nodes):
$$
\bar{\Delta\theta} = \text{Aggregate}\left(\Delta\theta^{(1)}, \Delta\theta^{(2)}, \ldots, \Delta\theta^{(N)}\right)
$$

**Outer Loop** (Nesterov momentum update):
$$
\theta_{\text{new}} = \theta_0 + \eta_{\text{outer}} \cdot \text{Nesterov}(\bar{\Delta\theta})
$$

### 2.2 Why Pseudo-Gradients Approximate True Gradients

Over $H$ inner steps, the pseudo-gradient accumulates:

$$
\Delta\theta = \eta_{\text{inner}} \sum_{t=0}^{H-1} g_t
$$

By the law of large numbers, as $H \to \infty$:

$$
\frac{1}{H} \sum_{t=0}^{H-1} g_t \xrightarrow{} \mathbb{E}[g] = \nabla \mathcal{L}(\theta)
$$

Therefore:
$$
\Delta\theta \approx H \cdot \eta_{\text{inner}} \cdot \nabla \mathcal{L}(\theta)
$$

The pseudo-gradient points in the same direction as the true gradient, scaled by $H \cdot \eta_{\text{inner}}$.

### 2.3 Convergence Guarantee

Under standard assumptions (L-smooth loss, bounded variance $\sigma^2$):

$$
\mathbb{E}[\mathcal{L}(\theta_T)] - \mathcal{L}(\theta^*) \leq \mathcal{O}\left(\frac{1}{\sqrt{T \cdot H}}\right)
$$

This matches the convergence rate of synchronous SGD while requiring $H\times$ less communication.

---

## 3. The Inner Optimizer: AdamW

The inner loop uses AdamW, which combines Adam's adaptive learning rates with decoupled weight decay.

### 3.1 Algorithm

Given gradient $g_t = \nabla_\theta \mathcal{L}(\theta_t)$:

**Moment estimates:**
$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{(first moment / mean)}
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad \text{(second moment / variance)}
$$

**Bias correction:**
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Update with decoupled weight decay:**
$$
\theta_{t+1} = \theta_t - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t \right)
$$

### 3.2 Hyperparameters

| Parameter | Symbol | Default | Purpose |
|-----------|--------|---------|---------|
| Learning rate | $\eta$ | $10^{-4}$ | Step size |
| First moment decay | $\beta_1$ | 0.9 | Gradient momentum |
| Second moment decay | $\beta_2$ | 0.95 | Variance estimation |
| Epsilon | $\epsilon$ | $10^{-8}$ | Numerical stability |
| Weight decay | $\lambda$ | 0.1 | L2 regularization |

### 3.3 Intuition

- **First moment ($m_t$)**: Exponential moving average of gradients → provides momentum
- **Second moment ($v_t$)**: Exponential moving average of squared gradients → adapts learning rate per parameter
- **Bias correction**: Compensates for initialization at zero (important early in training)
- **Decoupled weight decay**: Unlike L2 regularization, applies decay directly to weights, not through gradients

---

## 4. The Outer Optimizer: Nesterov Momentum

The outer loop applies Nesterov accelerated gradient descent to pseudo-gradients.

### 4.1 Standard Momentum

Classical momentum:
$$
v_t = \mu \cdot v_{t-1} + \Delta\theta_t
$$
$$
\theta_{t+1} = \theta_t + \eta \cdot v_t
$$

### 4.2 Nesterov Momentum (Look-Ahead)

Nesterov momentum evaluates the gradient at a "look-ahead" point:

$$
v_t = \mu \cdot v_{t-1} + \Delta\theta_t
$$
$$
\theta_{t+1} = \theta_t + \eta \cdot (\mu \cdot v_t + \Delta\theta_t)
$$

**Expanded form:**
$$
\theta_{t+1} = \theta_t + \eta \cdot \mu \cdot (\mu \cdot v_{t-1} + \Delta\theta_t) + \eta \cdot \Delta\theta_t
$$

### 4.3 Why Nesterov Works Better

The key insight is that Nesterov momentum makes a correction based on where momentum will take us, not where we currently are:

```
Standard:   θ → θ + μv → evaluate gradient → update
Nesterov:   θ → θ + μv → evaluate gradient at look-ahead → correct update
```

This "look-ahead" property provides:
- Faster convergence near minima
- Better handling of curved loss surfaces
- Automatic slowdown when overshooting

### 4.4 Implementation

```python
# Nesterov momentum update
v = μ * v + Δθ                    # Update velocity
θ = θ + η * (μ * v + Δθ)          # Apply with look-ahead
```

This is equivalent to:
$$
\theta_{t+1} = \theta_t + \eta \cdot \mu^2 \cdot v_{t-1} + \eta \cdot (1 + \mu) \cdot \Delta\theta_t
$$

---

## 5. Byzantine-Tolerant Aggregation

When aggregating gradients from potentially malicious nodes, we need robust methods.

### 5.1 Problem Formulation

Given $N$ gradient contributions $\{\Delta\theta^{(1)}, \ldots, \Delta\theta^{(N)}\}$ where up to $f$ may be Byzantine (arbitrary), find an aggregate $\bar{\Delta\theta}$ such that training converges.

### 5.2 Simple Mean (Vulnerable)

$$
\bar{\Delta\theta} = \frac{1}{N} \sum_{i=1}^{N} \Delta\theta^{(i)}
$$

**Vulnerability**: A single Byzantine node can set $\Delta\theta^{(\text{bad})} = M$ for arbitrarily large $M$, corrupting the mean.

### 5.3 Coordinate-Wise Median

For each parameter $j$:
$$
\bar{\Delta\theta}_j = \text{median}\left(\Delta\theta^{(1)}_j, \ldots, \Delta\theta^{(N)}_j\right)
$$

**Robustness**: Tolerates up to $\lfloor (N-1)/2 \rfloor$ Byzantine nodes.

**Limitation**: High variance compared to mean; ignores correlation between coordinates.

### 5.4 Trimmed Mean

Remove the top and bottom $\alpha$ fraction of values, then average:

$$
\bar{\Delta\theta}_j = \frac{1}{N - 2k} \sum_{i=k+1}^{N-k} \Delta\theta^{(i)}_{j,\text{sorted}}
$$

Where $k = \lfloor \alpha \cdot N \rfloor$.

**Default**: $\alpha = 0.1$ (remove top 10% and bottom 10%)

**Robustness**: Tolerates up to $\alpha$ fraction of Byzantine nodes.

### 5.5 Krum

Select the gradient closest to the majority.

**Score function** (for each gradient $i$):
$$
S(i) = \sum_{j \in \mathcal{N}_i} \|\Delta\theta^{(i)} - \Delta\theta^{(j)}\|^2
$$

Where $\mathcal{N}_i$ is the set of $(N - f - 2)$ nearest neighbors of $i$.

**Selection:**
$$
i^* = \arg\min_i S(i)
$$
$$
\bar{\Delta\theta} = \Delta\theta^{(i^*)}
$$

**Robustness**: Provably robust when $N \geq 2f + 3$.

**Theorem (Blanchard et al., 2017)**: If at most $f$ of $N$ gradients are Byzantine, Krum selects a gradient $\Delta\theta^{(i^*)}$ such that:
$$
\|\Delta\theta^{(i^*)} - \nabla\mathcal{L}\|^2 \leq (2f+2) \cdot \sigma^2
$$

where $\sigma^2$ is the variance of honest gradients.

### 5.6 Multi-Krum

Average the top $m$ gradients by Krum score:

$$
\bar{\Delta\theta} = \frac{1}{m} \sum_{i \in \mathcal{M}} \Delta\theta^{(i)}
$$

Where $\mathcal{M}$ contains the $m = N - f$ indices with lowest Krum scores.

**Benefit**: Lower variance than Krum while maintaining robustness.

### 5.7 Geometric Median

Find the point minimizing sum of Euclidean distances:

$$
\bar{\Delta\theta} = \arg\min_x \sum_{i=1}^{N} \|x - \Delta\theta^{(i)}\|_2
$$

**Weiszfeld Algorithm** (iterative solution):
$$
x^{(t+1)} = \frac{\sum_{i=1}^{N} \frac{\Delta\theta^{(i)}}{\|x^{(t)} - \Delta\theta^{(i)}\|_2}}{\sum_{i=1}^{N} \frac{1}{\|x^{(t)} - \Delta\theta^{(i)}\|_2}}
$$

**Robustness**: Optimal breakdown point of $\lfloor (N-1)/2 \rfloor$.

### 5.8 Comparison Table

| Method | Byzantine Tolerance | Variance | Complexity |
|--------|-------------------|----------|------------|
| Mean | 0 | Lowest | $O(N)$ |
| Median | $(N-1)/2$ | High | $O(N \log N)$ |
| Trimmed Mean | $\alpha N$ | Low | $O(N \log N)$ |
| Krum | $(N-3)/2$ | Very High | $O(N^2 d)$ |
| Multi-Krum | $(N-3)/2$ | Medium | $O(N^2 d)$ |
| Geometric Median | $(N-1)/2$ | Low | $O(N \cdot \text{iter})$ |

Where $d$ is the number of parameters.

---

## 6. Gradient Validation

Before aggregation, incoming gradients are validated.

### 6.1 Cosine Similarity Check

Measures alignment between submitted gradient $g_s$ and reference gradient $g_r$:

$$
\cos(g_s, g_r) = \frac{g_s \cdot g_r}{\|g_s\|_2 \cdot \|g_r\|_2}
$$

**Rejection criterion**: $\cos(g_s, g_r) < \tau$ (default $\tau = 0.3$)

**Intuition**: Honest gradients should point in similar directions (same optimization target). Anti-correlated gradients suggest malicious intent.

### 6.2 Magnitude Ratio Check

$$
\rho = \frac{\|g_s\|_2}{\|g_r\|_2}
$$

**Rejection criterion**: $\rho > \rho_{\max}$ or $\rho < \rho_{\min}$ (default: 10× range)

**Intuition**: Gradients should have similar scale. Extreme magnitudes suggest scaling attacks.

### 6.3 Variance Ratio Check

$$
\frac{\text{Var}(g_s)}{\text{Var}(g_r)} > V_{\max}
$$

**Intuition**: Abnormally high variance suggests noise injection.

---

## 7. Gradient Compression

For bandwidth efficiency, gradients are compressed before transmission.

### 7.1 Top-K Sparsification

Keep only the $k$ largest magnitude elements:

$$
\text{TopK}(g, k) = \{(i, g_i) : i \in \text{argtopk}(|g|, k)\}
$$

**Sparsity**: $k = \lfloor 0.1 \cdot d \rfloor$ (keep 10%)

**Error bound**: The approximation error is bounded by the sum of discarded elements:
$$
\|g - \text{TopK}(g, k)\|_2^2 = \sum_{i \notin \text{TopK}} g_i^2
$$

### 7.2 Quantization

Map floating-point values to integers:

$$
q(x) = \text{round}\left(x \cdot \frac{2^{b-1} - 1}{\max|x|}\right)
$$

**Dequantization**:
$$
\hat{x} = q(x) \cdot \frac{\max|x|}{2^{b-1} - 1}
$$

**Quantization error** (per element):
$$
|x - \hat{x}| \leq \frac{\max|x|}{2^b - 2}
$$

For 8-bit quantization with $\max|x| = 1$:
$$
|x - \hat{x}| \leq \frac{1}{254} \approx 0.4\%
$$

### 7.3 Why Compression Works

**Theorem (Stich et al., 2018)**: SGD with compressed gradients converges at rate:
$$
\mathbb{E}[\mathcal{L}(\theta_T)] - \mathcal{L}(\theta^*) \leq \mathcal{O}\left(\frac{1}{\sqrt{T}} + \frac{\omega}{T}\right)
$$

Where $\omega$ is the compression ratio. The extra $\omega/T$ term vanishes asymptotically.

**Intuition**:
1. SGD gradients are already noisy (mini-batch variance)
2. Compression error is much smaller than mini-batch noise
3. Averaging across nodes cancels compression errors (Central Limit Theorem)

---

## 8. Model Architecture Mathematics

### 8.1 RMS Normalization

Root Mean Square Layer Normalization:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

Where:
$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

**Compared to LayerNorm**:
$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

RMSNorm omits the mean subtraction and bias, making it:
- ~10% faster to compute
- More stable for very deep networks
- Empirically equivalent performance

### 8.2 Rotary Position Embeddings (RoPE)

RoPE encodes position through rotation in 2D subspaces.

**Rotation matrix** for position $m$ and frequency $\theta_i$:

$$
R_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
$$

**Frequency schedule**:
$$
\theta_i = 10000^{-2i/d}
$$

**Application** to query/key vectors (treating pairs of dimensions):
$$
\text{RoPE}(x, m) = \begin{pmatrix} R_{\theta_0, m} & & \\ & R_{\theta_1, m} & \\ & & \ddots \end{pmatrix} x
$$

**Key property** (relative position awareness):
$$
\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \langle q, R_{\theta, n-m} k \rangle
$$

The attention score depends only on the relative position $(n - m)$, not absolute positions.

**Complex number formulation** (equivalent, more elegant):
$$
\text{RoPE}(x, m) = x \odot e^{im\theta}
$$

Where $x$ is viewed as complex numbers and $\odot$ is element-wise multiplication.

### 8.3 Grouped Query Attention (GQA)

Standard multi-head attention has $H$ heads for Q, K, and V. GQA uses fewer KV heads.

**Projections**:
$$
Q = xW_Q \in \mathbb{R}^{B \times L \times H \times d_h}
$$
$$
K = xW_K \in \mathbb{R}^{B \times L \times G \times d_h}
$$
$$
V = xW_V \in \mathbb{R}^{B \times L \times G \times d_h}
$$

Where $G < H$ is the number of KV groups.

**Head expansion** (repeat KV heads to match query heads):
$$
K' = \text{repeat}(K, H/G), \quad V' = \text{repeat}(V, H/G)
$$

**Attention computation**:
$$
\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{QK'^T}{\sqrt{d_h}}\right) V'
$$

**Memory savings**: KV cache reduced by factor $H/G$ (e.g., 4× for $H=8, G=2$).

### 8.4 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

Where:
- $Q \in \mathbb{R}^{L_q \times d_k}$ = queries
- $K \in \mathbb{R}^{L_k \times d_k}$ = keys
- $V \in \mathbb{R}^{L_k \times d_v}$ = values
- $M$ = causal mask ($-\infty$ for future positions)

**Why scale by $\sqrt{d_k}$?**

If $q, k$ have unit variance, then:
$$
\text{Var}(q \cdot k) = d_k
$$

Scaling by $\sqrt{d_k}$ restores unit variance:
$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1
$$

This prevents softmax saturation (extreme probabilities) which would cause vanishing gradients.

### 8.5 SwiGLU Activation

A gated linear unit with SiLU (Swish) activation:

$$
\text{SwiGLU}(x) = \text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})
$$

Where:
$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**Full FFN block**:
$$
\text{FFN}(x) = ((\text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})) W_{\text{down}}
$$

**Why gating helps**:
- Allows the network to selectively pass information
- Smoother gradients than ReLU
- Empirically better performance for LLMs

**Comparison of activations**:

| Activation | Formula | Gradient |
|------------|---------|----------|
| ReLU | $\max(0, x)$ | $\mathbb{1}[x > 0]$ |
| GELU | $x \cdot \Phi(x)$ | Smooth |
| SiLU/Swish | $x \cdot \sigma(x)$ | $\sigma(x)(1 + x(1-\sigma(x)))$ |

---

## 9. Transformer Forward Pass

### 9.1 Single Layer

For input $x \in \mathbb{R}^{B \times L \times d}$:

```
# Pre-norm attention
h = x + Attention(RMSNorm(x))

# Pre-norm FFN  
out = h + FFN(RMSNorm(h))
```

Mathematically:
$$
h = x + \text{Attention}(\text{RMSNorm}(x))
$$
$$
\text{out} = h + \text{FFN}(\text{RMSNorm}(h))
$$

### 9.2 Full Forward Pass

```
# Embedding
h_0 = Embed(tokens)

# Transformer layers
for l in range(L):
    h_{l+1} = TransformerBlock_l(h_l)

# Output
logits = LMHead(RMSNorm(h_L))
```

### 9.3 Parameter Count

For a model with:
- $d$ = hidden dimension
- $L$ = number of layers
- $H$ = attention heads
- $G$ = KV heads
- $d_h$ = head dimension = $d/H$
- $d_{ff}$ = FFN intermediate dimension
- $V$ = vocabulary size

**Per-layer parameters**:

| Component | Parameters |
|-----------|------------|
| Q projection | $d \times d$ |
| K projection | $d \times (G \cdot d_h)$ |
| V projection | $d \times (G \cdot d_h)$ |
| O projection | $d \times d$ |
| Gate projection | $d \times d_{ff}$ |
| Up projection | $d \times d_{ff}$ |
| Down projection | $d_{ff} \times d$ |
| RMSNorm (×2) | $2d$ |

**Total**:
$$
P = V \cdot d + L \cdot (2d^2 + 2d \cdot G \cdot d_h + 3d \cdot d_{ff} + 2d) + d + V \cdot d
$$

Simplified (assuming $G = H/4$, $d_{ff} = 4d$, tied embeddings):
$$
P \approx V \cdot d + L \cdot (2.5d^2 + 12d^2) \approx V \cdot d + 14.5 \cdot L \cdot d^2
$$

---

## 10. Backpropagation Through Transformers

### 10.1 Gradient Flow

The gradient of loss with respect to layer $l$ input:

$$
\frac{\partial \mathcal{L}}{\partial h_l} = \frac{\partial \mathcal{L}}{\partial h_{l+1}} \cdot \left(I + \frac{\partial \text{Block}_l}{\partial h_l}\right)
$$

The residual connection ($I$) ensures gradients flow directly, preventing vanishing gradients.

### 10.2 Gradient Clipping

Before applying gradients, clip the global norm:

$$
g' = \begin{cases}
g & \text{if } \|g\|_2 \leq c \\
\frac{c \cdot g}{\|g\|_2} & \text{otherwise}
\end{cases}
$$

Where $c$ is the maximum norm (default: 1.0).

**Purpose**: Prevents exploding gradients from destabilizing training.

---

## 11. Complete Training Algorithm

Putting it all together:

### Algorithm: NeuroShard DiLoCo Training

**Inputs:**
- Model $f_\theta$ with parameters $\theta$
- Inner optimizer (AdamW) with learning rate $\eta_{\text{inner}}$
- Outer optimizer (Nesterov) with learning rate $\eta_{\text{outer}}$, momentum $\mu$
- Inner steps $H$, nodes $N$, aggregation function $\text{Agg}$

**For each outer step $k = 1, 2, \ldots$:**

1. **Save initial weights**: $\theta_0^{(i)} \leftarrow \theta$ for all nodes $i$

2. **Inner loop** (on each node $i$ independently):
   ```
   for t = 0 to H-1:
       Sample batch B_t^{(i)}
       Compute loss: L = CrossEntropy(f_θ(B_t), labels)
       Compute gradient: g_t = ∇_θ L
       Clip gradient: g_t = clip(g_t, max_norm)
       Update: θ = AdamW(θ, g_t)
   ```

3. **Compute pseudo-gradient**:
   $$\Delta\theta^{(i)} = \theta_0^{(i)} - \theta^{(i)}$$

4. **Compress** (optional):
   $$\Delta\theta^{(i)}_{\text{compressed}} = \text{Quantize}(\text{TopK}(\Delta\theta^{(i)}))$$

5. **Exchange** via gossip protocol

6. **Validate** each received gradient:
   ```
   for each peer gradient Δθ^{(j)}:
       if cosine_sim(Δθ^{(j)}, Δθ^{(i)}) < τ: reject
       if magnitude_ratio out of bounds: reject
   ```

7. **Aggregate**:
   $$\bar{\Delta\theta} = \text{TrimmedMean}(\{\Delta\theta^{(i)}\}_{\text{valid}})$$

8. **Outer update** (Nesterov):
   ```
   v = μ * v + Δθ_bar
   θ = θ_0 + η_outer * (μ * v + Δθ_bar)
   ```

9. **Broadcast** new $\theta$ to all nodes

---

## 12. Convergence Analysis

### 12.1 Assumptions

1. **L-smoothness**: $\|\nabla \mathcal{L}(\theta) - \nabla \mathcal{L}(\phi)\| \leq L \|\theta - \phi\|$
2. **Bounded variance**: $\mathbb{E}[\|g - \nabla\mathcal{L}\|^2] \leq \sigma^2$
3. **Bounded gradients**: $\|\nabla\mathcal{L}(\theta)\| \leq G$

### 12.2 Main Result

**Theorem**: Under the above assumptions, DiLoCo with $N$ nodes, $H$ inner steps, and appropriate learning rates achieves:

$$
\frac{1}{T} \sum_{k=1}^{T} \mathbb{E}[\|\nabla\mathcal{L}(\theta_k)\|^2] \leq \mathcal{O}\left(\frac{\mathcal{L}(\theta_0) - \mathcal{L}^*}{\eta T H} + \frac{\eta L \sigma^2}{N} + \eta^2 L^2 H \sigma^2\right)
$$

**Optimal learning rate**: $\eta^* = \mathcal{O}\left(\sqrt{\frac{N}{THL\sigma^2}}\right)$

**Resulting convergence rate**:
$$
\mathcal{O}\left(\sqrt{\frac{L(\mathcal{L}_0 - \mathcal{L}^*)\sigma^2}{NTH}}\right)
$$

This shows:
- Linear speedup with $N$ nodes ✓
- Convergence improves with more inner steps $H$ ✓
- Same asymptotic rate as synchronous SGD ✓

---

## 13. Summary of Key Equations

| Concept | Equation |
|---------|----------|
| **Cross-Entropy Loss** | $\mathcal{L} = -\log \text{softmax}(z)_y$ |
| **AdamW Update** | $\theta = \theta - \eta(\hat{m}/\sqrt{\hat{v}} + \lambda\theta)$ |
| **Nesterov Momentum** | $\theta = \theta + \eta(\mu v + \Delta\theta)$ |
| **Pseudo-Gradient** | $\Delta\theta = \theta_0 - \theta_H$ |
| **Trimmed Mean** | $\bar{x} = \text{mean}(x_{k+1:n-k})$ |
| **Krum Score** | $S(i) = \sum_{j \in \mathcal{N}_i} \|g_i - g_j\|^2$ |
| **RMSNorm** | $\hat{x} = x / \text{RMS}(x) \cdot \gamma$ |
| **RoPE** | $\text{RoPE}(x,m) = x \odot e^{im\theta}$ |
| **Attention** | $\text{softmax}(QK^T/\sqrt{d_k})V$ |
| **SwiGLU** | $\text{SiLU}(xW_g) \odot (xW_u)$ |

---

## References

1. **DiLoCo**: Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models" (2023)
2. **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
3. **Nesterov**: Nesterov, "A method for solving the convex programming problem with convergence rate O(1/k²)" (1983)
4. **Krum**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (2017)
5. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
6. **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
7. **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020)
8. **Gradient Compression**: Stich et al., "Sparsified SGD with Memory" (2018)
