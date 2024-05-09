import os

from tqdm import tqdm

from accelerate import Accelerator

import torch
from torch.nn.functional import mse_loss
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader

from diffusers import UNet2DModel

from torchvision.transforms import Compose, ToTensor, Resize, Lambda, ToPILImage
from torchvision.datasets import MNIST

from deepl_sandbox.noise_scheduling import LinearNoiseScheduler

BATCH_SIZE = 16
N_EPOCHS = 20
TRAINING_TIMESTEPS = 1000

dataset_path = os.path.join(os.path.dirname(__file__), "../../data")
checkpoint_folder_path = os.path.join(os.path.dirname(__file__), "checkpoints")

forward_transform = Compose([
    ToTensor(),
    Resize((32, 32)),
    Lambda(lambda x: 2 * x.to(torch.float32) / 255 - 1),
    Lambda(lambda z: torch.clamp(z, -1, 1)),
])

reverse_transform = Compose([
    Lambda(lambda z: 255 * (1 + z) / 2),
    Lambda(lambda x: x.to(torch.uint8).clamp(0, 255)),
    ToPILImage(),
])

if __name__ == "__main__":
    # load data
    training_data = MNIST(
        root=dataset_path,
        transform=forward_transform,
        train=True,
        download=True,
    )
    train_loader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
    unet = UNet2DModel((32, 32), in_channels=1, out_channels=1, class_embed_type="timestep")
    noise_scheduler = LinearNoiseScheduler(1e-4, 0.02, TRAINING_TIMESTEPS)
    optimizer = torch.optim.AdamW(unet.parameters())
    # send to fastest device
    accelerator = Accelerator()
    train_loader, unet, optimizer = accelerator.prepare(train_loader, unet, optimizer)
    print(f"Training device: {unet.device}")
    # save unet config 
    os.makedirs(checkpoint_folder_path, exist_ok=True)
    unet.save_config(checkpoint_folder_path)
    # Training phase
    unet.train()
    cumul_mse = 0.0
    for epoch in range(N_EPOCHS):
        desc = f"epoch: {epoch+1}/{N_EPOCHS}"
        progbar = tqdm(
            train_loader,
            desc=f"epoch: {epoch+1}/{N_EPOCHS}",
        )
        for i, (x0, labels) in enumerate(progbar):
            optimizer.zero_grad()
            timesteps = torch.randint_like(labels, TRAINING_TIMESTEPS)
            epsilon = torch.randn_like(x0)
            xt = noise_scheduler.add_noise(x0, epsilon, timesteps)
            pred_eps = unet(xt, timesteps, labels).sample
            loss = mse_loss(pred_eps, epsilon)
            loss.backward()
            optimizer.step()
            cumul_mse += loss.item()
            progbar.set_postfix({"avg MSE": cumul_mse / (i+1)})
        # save checkpoint
        unet.save_pretrained(checkpoint_folder_path)
            