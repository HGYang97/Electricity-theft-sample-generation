# cyclegan_custom_data_full.py (改进版，使用A+B数据构造训练集)
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


# ==================== 配置中心 ====================
class Config:
    data_path_A = "data/electricity/aierlan_ori_data.pt"
    data_path_B = "data/electricity/balanced_finetune.pt"
    seq_length = 1034
    batch_size = 32

    gen_filters = [64, 128, 256]
    residual_blocks = 6
    dis_filters = [64, 128, 256, 512]

    epochs = 200
    lr_g = 0.0002
    lr_d = 0.0002
    lambda_cycle = 10.0
    lambda_identity = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    checkpoint_dir = "./checkpoints/"

    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.manual_seed(self.seed)
        random.seed(self.seed)


# ==================== 数据加载模块 ====================
class PowerConsumptionDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor.unsqueeze(1).to(torch.float32)
        if self.data.dim() != 3:
            raise ValueError(f"数据维度错误: {self.data.shape}")
        if self.data.size(2) != Config.seq_length:
            raise ValueError(f"序列长度应为{Config.seq_length}，但得到{self.data.size(2)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders_from_A_and_B(a_path, b_path):
    A_data = torch.load(a_path).float()
    B_dict = torch.load(b_path)

    B_data = B_dict['samples'].float()
    B_labels = B_dict['labels'].long()

    if B_data.shape[0] != B_labels.shape[0]:
        raise ValueError("B数据和标签数量不匹配")

    B_attack = B_data[B_labels == 1][:, :Config.seq_length]  # 保证长度一致

    def normalize(data):
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        return (data - min_val) / (max_val - min_val + 1e-8)

    A_normal = normalize(A_data[:, :Config.seq_length])
    B_attack = normalize(B_attack)

    normal_dataset = PowerConsumptionDataset(A_normal)
    attack_dataset = PowerConsumptionDataset(B_attack)

    normal_loader = DataLoader(normal_dataset, batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    attack_loader = DataLoader(attack_dataset, batch_size=Config.batch_size, shuffle=True, pin_memory=True)

    return normal_loader, attack_loader


# ==================== 模型定义 ====================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, in_channels, kernel_size=7),
            nn.InstanceNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, in_channels, kernel_size=7),
            nn.InstanceNorm1d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


class TimeSeriesGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, Config.gen_filters[0], kernel_size=7),
            nn.InstanceNorm1d(Config.gen_filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(Config.gen_filters[0], Config.gen_filters[1], kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(Config.gen_filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(Config.gen_filters[1], Config.gen_filters[2], kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(Config.gen_filters[2]),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock1D(Config.gen_filters[2]) for _ in range(Config.residual_blocks)])
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(Config.gen_filters[2], Config.gen_filters[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(Config.gen_filters[1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(Config.gen_filters[1], Config.gen_filters[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(Config.gen_filters[0]),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(3),
            nn.Conv1d(Config.gen_filters[0], 1, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x[:, :, :Config.seq_length]


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, Config.dis_filters[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(Config.dis_filters[0], Config.dis_filters[1], kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(Config.dis_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(Config.dis_filters[1], Config.dis_filters[2], kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(Config.dis_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(Config.dis_filters[2], Config.dis_filters[3], kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(Config.dis_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(Config.dis_filters[3], 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ==================== 训练流程 ====================
def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def plot_training_progress(g_loss, d_loss, filename="training_progress.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(g_loss, label="Generator Loss")
    plt.plot(d_loss, label="Discriminator Loss")
    plt.legend()
    plt.title("Training Progress")
    plt.savefig(filename)
    plt.close()

def save_fake_samples(generator, original_data, output_path, device, attack_ratio=0.5):
    os.makedirs(output_path, exist_ok=True)
    total = original_data.shape[0]
    num_attack = int(total * attack_ratio)
    indices = torch.randperm(total)
    attack_indices = indices[:num_attack]
    normal_indices = indices[num_attack:]

    # Prepare input
    with torch.no_grad():
        input_tensor = original_data[attack_indices, :Config.seq_length].unsqueeze(1).to(device)
        fake_attacks = generator(input_tensor).cpu().squeeze(1)

    # 拼接回完整数据
    mixed_data = torch.zeros_like(original_data[:, :Config.seq_length])
    mixed_data[attack_indices] = fake_attacks
    mixed_data[normal_indices] = original_data[normal_indices, :Config.seq_length]

    # 标签：1 为伪攻击，0 为原始正常
    labels = torch.zeros(total, dtype=torch.long)
    labels[attack_indices] = 1

    # 保存
    torch.save({"samples": mixed_data, "labels": labels}, os.path.join(output_path, "a_modified.pt"))

def main():
    config = Config()
    print(f"使用设备: {config.device}")

    normal_loader, attack_loader = create_dataloaders_from_A_and_B(
        config.data_path_A, config.data_path_B
    )

    G_XtoY = TimeSeriesGenerator().to(config.device)
    G_YtoX = TimeSeriesGenerator().to(config.device)
    D_X = TimeSeriesDiscriminator().to(config.device)
    D_Y = TimeSeriesDiscriminator().to(config.device)

    G_XtoY.apply(init_weights)
    G_YtoX.apply(init_weights)
    D_X.apply(init_weights)
    D_Y.apply(init_weights)

    optimizer_G = torch.optim.Adam(
        list(G_XtoY.parameters()) + list(G_YtoX.parameters()),
        lr=config.lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        list(D_X.parameters()) + list(D_Y.parameters()),
        lr=config.lr_d, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    G_losses, D_losses = [], []

    print("开始训练...")
    for epoch in range(config.epochs):
        for real_X, real_Y in zip(normal_loader, attack_loader):
            real_X = real_X.to(config.device)
            real_Y = real_Y.to(config.device)

            optimizer_G.zero_grad()

            fake_Y = G_XtoY(real_X)
            fake_X = G_YtoX(real_Y)

            loss_GAN_XtoY = criterion_gan(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y)))
            loss_GAN_YtoX = criterion_gan(D_X(fake_X), torch.ones_like(D_X(fake_X)))

            loss_cycle_X = criterion_cycle(G_YtoX(fake_Y), real_X) * config.lambda_cycle
            loss_cycle_Y = criterion_cycle(G_XtoY(fake_X), real_Y) * config.lambda_cycle

            loss_id_Y = criterion_identity(G_XtoY(real_Y), real_Y) * config.lambda_identity
            loss_id_X = criterion_identity(G_YtoX(real_X), real_X) * config.lambda_identity

            total_G = loss_GAN_XtoY + loss_GAN_YtoX + loss_cycle_X + loss_cycle_Y + loss_id_X + loss_id_Y
            total_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_D_Y = 0.5 * (
                criterion_gan(D_Y(real_Y), torch.ones_like(D_Y(real_Y))) +
                criterion_gan(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))
            )
            loss_D_Y.backward()
            optimizer_D.step()

        G_losses.append(total_G.item())
        D_losses.append(loss_D_Y.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | G Loss: {total_G.item():.4f} | D Loss: {loss_D_Y.item():.4f}")

    torch.save(G_XtoY.state_dict(), os.path.join(config.checkpoint_dir, "G_XtoY_final.pth"))
    torch.save(G_YtoX.state_dict(), os.path.join(config.checkpoint_dir, "G_YtoX_final.pth"))
    plot_training_progress(G_losses, D_losses)
    print("训练结束，模型已保存")

    print("开始生成伪攻击样本并保存合成数据集...")
    A_data_raw = torch.load(config.data_path_A).float()
    save_fake_samples(G_XtoY, A_data_raw, os.path.join(config.checkpoint_dir, "fake_attacks"), config.device, attack_ratio=0.3)
    print("合成后的数据集（50%正常 + 50%伪攻击）已保存为 a_modified.pt")

if __name__ == "__main__":
    main()
