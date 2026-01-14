import torch
from typing import Iterable

class SWGL(torch.optim.Optimizer):
    """
    Stability-Weighted Gradient Learning (SWGL)
    
    A lightweight online gradient modulation technique that reduces catastrophic 
    interference by weighting gradients based on per-parameter stability estimates.
    
    Key Properties:
    - No memory overhead (no replay buffers or stored data)
    - No importance matrices (Fisher, EWC, etc.)
    - Works as a drop-in wrapper for any base optimizer
    - Purely online: updates stability estimates during forward/backward passes
    
    Technical Description:
    Modulates gradients using an exponential moving average of relative gradient 
    impact: |grad| / (|param| + eps). Stable parameters (low impact) receive 
    boosted gradients while volatile parameters (high impact) are suppressed.
    
    Limitations:
    - This is NOT a complete continual learning solution
    - Does NOT prevent forgetting fundamentally
    - Best used as a stability-enhancing regularizer for online learning
    
    Args:
        params: Iterable of parameters to optimize
        alpha: EMA decay for stability estimates (default: 0.99)
        lambda_reg: L2-like penalty coefficient for high-stability parameters (default: 0.01)
        eps: Numerical stability term (default: 1e-8)
        max_boost: Maximum gradient boost factor (default: 10.0)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        alpha: float = 0.99,
        lambda_reg: float = 0.01,
        eps: float = 1e-8,
        max_boost: float = 10.0
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if lambda_reg < 0.0:
            raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")
        if max_boost < 1.0:
            raise ValueError(f"max_boost must be >= 1.0, got {max_boost}")
            
        defaults = dict(alpha=alpha, lambda_reg=lambda_reg, eps=eps, max_boost=max_boost)
        super().__init__(params, defaults)
        
        # Initialize stability estimates to neutral (1.0) for all parameters
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['stability'] = torch.ones_like(p)
    
    def step(self):
        """
        Performs a single optimization step by modulating gradients based on stability.
        Call this BEFORE your base optimizer's step().
        """
        for group in self.param_groups:
            alpha = group['alpha']
            lambda_reg = group['lambda_reg']
            eps = group['eps']
            max_boost = group['max_boost']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                param = p
                state = self.state[p]
                stability = state['stability']
                
                with torch.no_grad():
                    # Compute relative gradient impact: |grad| / (|param| + eps)
                    relative_impact = grad.abs() / (param.abs() + eps)
                    
                    # Update stability estimate with exponential moving average
                    stability.mul_(alpha).add_(relative_impact, alpha=1.0 - alpha)
                    
                    # Compute gradient scaling factor: 1 / stability (clamped)
                    lr_scaler = 1.0 / (stability + eps)
                    lr_scaler.clamp_(max=max_boost)
                    
                    # Apply modulation directly to gradient
                    grad.mul_(lr_scaler)
                    
                    # Optional: L2-like penalty on parameters with excessive stability
                    if lambda_reg > 0.0:
                        stability_excess = (stability - 1.0).clamp(min=0.0)
                        grad.add_(param * stability_excess * lambda_reg)
      
