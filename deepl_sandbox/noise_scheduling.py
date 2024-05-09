import torch
from torch import Tensor, LongTensor

class LinearNoiseScheduler:
    def __init__(self, start: float, end: float, max_timesteps: int):
        self._betas = torch.linspace(start, end, max_timesteps)
        self._alpha_bars = (1-self._betas).cumprod(0)
    
    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: LongTensor) -> Tensor:
        xt_list = []
        for x, eps, t in zip(x0, noise, timesteps):
            xt = self._alpha_bars[t].sqrt() * x + (1-self._alpha_bars[t]).sqrt() * eps
            xt_list.append(xt.unsqueeze(0))
        return torch.concat(xt_list, dim=0)