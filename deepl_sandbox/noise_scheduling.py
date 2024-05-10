import torch
from torch import Tensor, LongTensor

from enum import Enum

class VarianceType(Enum):
    NORMAL = 0
    ADJUSTED = 1

class LinearNoiseScheduler:
    def __init__(self, start: float, end: float, max_timesteps: int, reverse_variance_type: int = VarianceType.NORMAL):
        self._betas = torch.linspace(start, end, max_timesteps)
        self._alpha_bars = (1-self._betas).cumprod(0)
        if reverse_variance_type == VarianceType.NORMAL:
            self._variance = lambda t: self._betas[t]
        elif reverse_variance_type == VarianceType.ADJUSTED:
            self._variance = lambda t: (1 - self._alpha_bars[t-1]) * self._betas[t] / (1 - self._alpha_bars[t])
        else:
            raise ValueError("The entered reverse variance type is not implemented.")
    
    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: LongTensor) -> Tensor:
        xt_list = []
        for x, eps, t in zip(x0, noise, timesteps):
            xt = self._alpha_bars[t].sqrt() * x + (1-self._alpha_bars[t]).sqrt() * eps
            xt_list.append(xt.unsqueeze(0))
        return torch.concat(xt_list, dim=0)
    
    def denoising_step(self, xt: Tensor, epsilon: Tensor, z: Tensor, t: int) -> Tensor:
        const_1 = 1. / (1-self._betas[t]).sqrt() 
        const_2 = self._betas[t] / (1-self._alpha_bars[t]).sqrt()
        std = self._variance(t).sqrt() if t > 0 else 0
        return const_1 * (xt - const_2 * epsilon) + std * z
        