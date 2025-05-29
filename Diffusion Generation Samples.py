
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import random

# ---------- Dataset Loading ----------
data = torch.load("balanced_finetune.pt")
samples = data['samples'].float()
labels = data['labels'].long()
normal_samples = samples[labels == 0]
theft_samples = samples[labels == 1]

# ---------- 1D ResNet Backbone ----------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=7, stride=1):
        super(BasicBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, in_channels=1, num_blocks=[2, 2, 2], base_channels=64):
        super(ResNet1D, self).__init__()
        self.layer1 = self._make_layer(in_channels, base_channels, num_blocks[0])
        self.layer2 = self._make_layer(base_channels, base_channels*2, num_blocks[1])
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, num_blocks[2])
        self.output_layer = nn.Linear(base_channels*4, samples.shape[1])

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.output_layer(x)

# ---------- Diffusion Utilities ----------
class Diffusion:
    def __init__(self, timesteps=50, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1, 1).to(x.device)
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]).sqrt().view(-1, 1).to(x.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,))

# ---------- Training Loop ----------
def train_diffusion(model, loader, diffusion, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        model.train()
        for x in loader:
            x = x.to(device)
            t = diffusion.sample_timesteps(x.size(0)).to(device)
            noisy_x, noise = diffusion.add_noise(x, t)
            pred = model(noisy_x)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# ---------- Generate Samples ----------
def generate_samples(model, diffusion, n_samples=1000):
    model.eval()
    x = torch.randn(n_samples, samples.shape[1]).to(device)
    for t in reversed(range(diffusion.timesteps)):
        z = torch.randn_like(x) if t > 0 else 0
        noise_pred = model(x)
        alpha = diffusion.alpha[t].to(x.device)
        alpha_hat = diffusion.alpha_hat[t].to(x.device)
        beta = diffusion.beta[t].to(x.device)
        x = 1/alpha.sqrt() * (x - (1 - alpha)/((1 - alpha_hat).sqrt()) * noise_pred) + beta.sqrt() * z
    return x.cpu()

# ---------- Run Training ----------
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = Diffusion(timesteps=50)
train_data = theft_samples[:10000]
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

model = ResNet1D().to(device)
train_diffusion(model, train_loader, diffusion, epochs=20)

# ---------- Save Generated Samples ----------
generated = generate_samples(model, diffusion, n_samples=5000)
torch.save({'samples': generated, 'labels': torch.ones(generated.size(0), dtype=torch.long)}, 'diffusion.pt')
