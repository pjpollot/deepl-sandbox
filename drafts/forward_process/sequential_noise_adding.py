"""How noise is actually used in the forward process?
"""

import torch
from torch import FloatTensor, IntTensor

from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from PIL import Image

def normalize(image: IntTensor) -> FloatTensor:
    z = 2 * image.to(torch.float32) / 255 - 1
    z.clamp_(-1, 1)
    return z

def unormalize(z : FloatTensor) -> IntTensor:
    image = 255 * (1 + z) / 2
    image.clamp_(0, 255).to(torch.uint8)
    return image

if __name__ == "__main__":
    # load image
    image = Image.open(os.path.join(
        os.path.dirname(__file__),
        "../../images/butterfly.webp"
    ))
    # generate variance schedule 
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    # data pre-processing
    image_tensor = pil_to_tensor(image)
    image_tensor = normalize(image_tensor)
    # adding gradually noise
    fig, ax = plt.subplots()
    for t in range(T):
        image_tensor = torch.sqrt(1 - betas[t]) * image_tensor + betas[t] * torch.randn_like(image_tensor)
        new_image = to_pil_image(unormalize(image_tensor))
        ax.clear()
        ax.set_title(f"timestep: {t+1}/{T}")
        ax.imshow(new_image)
        plt.pause(0.1)
