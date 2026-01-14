# SWGL: Stability-Weighted Gradient Learning

**A Lightweight, Memory-Free Gradient Modulation Technique for Online Learning**

[PyTorch] | [Paper (Coming Soon)] | [GitHub Issues]

</div>

---

## 🎯 Overview

**SWGL** (Stability-Weighted Gradient Learning) is an **optimizer-agnostic** gradient modulation method that reduces catastrophic interference in sequential learning scenarios—**without storing past data, without computing importance matrices, and without replay buffers**.

It functions as a **drop-in wrapper** around any PyTorch optimizer (AdamW, SGD, etc.) and operates purely **online**: stability estimates are updated in real-time during training.

---

## 🔬 Core Mechanism

For each parameter $w_i$ with gradient $g_i$:

1. **Relative Impact**  
   Measures update intensity relative to parameter magnitude:  
   $$r_i = \frac{|g_i|}{|w_i| + \epsilon}$$

2. **Stability Tracking**  
   Exponential moving average of relative impact:  
   $$s_i \leftarrow \alpha \cdot s_i + (1-\alpha) \cdot r_i$$

3. **Gradient Modulation**  
   Suppresses volatile parameters, boosts stable ones:  
   $$g_i \leftarrow g_i \cdot \frac{1}{\text{clamp}(s_i, \text{max}=B)}$$

where:
- $\alpha \in [0, 1)$ = EMA decay (default: 0.99)
- $\epsilon$ = Numerical stability (default: $10^{-8}$)
- $B$ = Maximum boost factor (default: 10.0)

**Key Property:** This modulation is **multiplicative and invariant** to the base optimizer's learning rate.

---

## ✨ Strengths

| Feature | Description | Impact |
|---|---|---|
| **Zero Memory Overhead** | No replay buffers, no data storage, no Fisher matrices | Critical for edge devices and streaming data |
| **Optimizer Agnostic** | Works seamlessly with AdamW, SGD, AdaGrad, etc. | Universal integration |
| **Purely Online** | No separate phases; updates stability in real-time | True adaptive learning |
| **Mathematically Transparent** | Based on relative gradient statistics, not heuristics | Reproducible and auditable |
| **Production-Ready** | ~30 lines of core logic; easy to audit and deploy | Low maintenance cost |
| **Complementary** | Can be combined with replay, EWC, or architecture methods | Part of a hybrid solution |

---

## 🧪 When to Use SWGL

SWGL is **most valuable** when:

- **Memory is constrained** (e.g., edge AI, IoT, smartphones)
- **Data is streaming** (non-stationary distributions, financial markets, sensor networks)
- **Latency matters** (no backward passes over stored data)
- You need a **reproducible baseline** for gradient modulation research
- You're building a **hybrid system** (e.g., SWGL for fast adaptation + tiny replay buffer for retention)

**Real-World Applications:**
- **Robotics**: Adaptation to changing environments
- **Federated Learning**: Mitigation of client drift
- **Continual Pre-training**: Efficient adaptation of language models
- **Recommendation Systems**: Evolving user preferences

---

## ⚠️ Limitations & Trade-offs

**SWGL is a stability enhancer, not a silver bullet.**

| Limitation | Explanation | Mitigation |
|---|---|---|
| **Not a Standalone Solution** | Does not fundamentally prevent forgetting | Combine with replay buffers or architectural methods for long-term retention |
| **Hyperparameter Sensitivity** | `alpha` and `lambda_reg` require task-specific tuning | Use grid search or adaptive schedules |
| **No Theoretical Guarantees** | Empirical stability improvement only | Document expected performance for your domain |
| **Per-Parameter State** | Adds ~1x parameter memory overhead (like momentum) | Acceptable for modern hardware; avoid for ultra-tiny models |
| **Learning Rate Dependence** | Extreme `lr` may destabilize EMA estimates | Use moderate learning rates; validate stability metrics |

**Bottom Line:** Use SWGL to **reduce** interference and **improve** online stability. Do not expect it to **eliminate** forgetting.

---

## 🚀 Quick Start

```python
import torch
import torch.nn as nn
from swgl import SWGL

# Step 1: Create your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Step 2: Initialize your base optimizer
base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Step 3: Wrap SWGL around the same parameters
swgl = SWGL(
    params=model.parameters(),
    alpha=0.99,          # EMA decay
    lambda_reg=0.01,     # Stability penalty
    max_boost=10.0       # Maximum gradient boost
)

# Step 4: Training loop
for batch_data, batch_labels in your_data_stream:
    base_optimizer.zero_grad()
    
    loss = model(batch_data).loss
    loss.backward()
    
    # Apply gradient modulation BEFORE optimizer step
    swgl.step()
    base_optimizer.step()
```

Critical: Call `swgl.step()` immediately after `loss.backward()` and before `base_optimizer.step()`.

---

🔧 Advanced Usage & Tips

Hyperparameter Tuning

```python
# Conservative (less interference, slower learning)
swgl = SWGL(params, alpha=0.995, lambda_reg=0.005, max_boost=5.0)

# Aggressive (faster adaptation, potential instability)
swgl = SWGL(params, alpha=0.95, lambda_reg=0.05, max_boost=20.0)
```

Monitoring Stability

```python
# Access stability statistics for debugging
for group in swgl.param_groups:
    for p in group['params']:
        stability = swgl.state[p]['stability']
        print(f"Mean stability: {stability.mean().item():.4f}")
```

Integration with Mixed Precision

```python
with torch.cuda.amp.autocast():
    loss = model(data).loss

scaler.scale(loss).backward()
swgl.step()  # Works seamlessly with scaled gradients
scaler.step(base_optimizer)
scaler.update()
```

---

📊 Expected Behavior

- Task Similarity: Higher similarity between sequential tasks → better SWGL performance
- Learning Rate: Moderate rates (1e-4 to 1e-2) yield stable EMA behavior
- `lambda_reg`: Acts as mild weight decay on stable parameters; prevents "dead weights"
- Forgetting: Will reduce catastrophic forgetting in online learning, though it will not eliminate it completely

---

🏛️ Theory & Intuition

Why does this work?

In neural networks, volatile parameters (those with high |g|/|w|) often encode task-specific features. During sequential learning, these parameters cause interference when overwritten. SWGL dampens updates to volatile parameters while boosting stable ones, which tend to encode generalizable features.

This is analogous to:
- Learning Rate Annealing: But per-parameter and adaptive
- Weight Decay: But targeted at stable parameters to prevent "freezing"
- Uncertainty-Based Modulation: But using gradient statistics instead of Bayesian estimates

---

📚 Citation

If you use SWGL in your research or product, please cite:

```bibtex
@software{swgl2024,
  title={SWGL: Stability-Weighted Gradient Learning},
  author={Walid Maxsim},
  year={2024},
  url={https://github.com/VireonLabs/SWGL},
  note={Lightweight gradient modulation for online learning}
}
```

---

🤝 Contributing

This is a minimal prototype. Issues and pull requests are welcome, especially:

- Hyperparameter studies on real datasets
- Integration examples with popular frameworks (Hugging Face, Lightning)
- Visualizations of stability dynamics
- Hybrid methods combining SWGL with other CL techniques

---

📄 License

MIT License – See `LICENSE` file for details.

---

SWGL is not perfect, but it's honest and lightweight. Use it wisely.
