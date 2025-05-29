import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
from args import args, Test_data, Train_data_all, Train_data
from model.TimeMAE import TimeMAE # 从外部文件导入模型

# -------------------- 1. 模型加载 --------------------
def load_pretrained_model(model_path, device='cuda'):
    """加载预训练模型"""
    # 注意：需要根据实际模型定义调整参数
    args.data_shape = (1035, 1)
    model = TimeMAE(args).to(device)
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise
    model.eval()
    return model


# -------------------- 2. 数据加载与预处理 --------------------
def load_and_prepare_data(data_path, attack_ratio=0.5):
    """加载数据并选择攻击样本"""
    data = torch.load(data_path)

    # 验证数据格式
    # if not {'samples', 'labels'}.issubset(data.keys()):
    #     raise ValueError("数据格式异常，需要包含'samples'和'labels'")

    samples = data.float()  # [4232, 1034]

    # 归一化到 [0,1]
    min_val = samples.min()
    max_val = samples.max()
    normalized_samples = (samples - min_val) / (max_val - min_val + 1e-8)

    # 随机选择攻击样本
    n_total = samples.size(0)
    selected_idx = random.sample(range(n_total), int(n_total * attack_ratio))

    return normalized_samples, selected_idx


# -------------------- 3. 改进的对抗样本生成 --------------------
def generate_adversarial_samples(model, original_samples, selected_idx, device='cuda', ** params):
    """
    生成对抗样本
    :param params: 可调节参数
        - epsilon: 基础步长 (默认0.005)
        - max_iter: 最大迭代次数 (默认3)
        - decay: 步长衰减率 (默认0.5)
        - step_clip: 单步扰动限制 (默认0.03)
    """
    # 参数设置
    base_epsilon = params.get('epsilon', 0.005)
    max_iter = params.get('max_iter', 3)
    decay_factor = params.get('decay', 0.5)
    step_clip = params.get('step_clip', 0.03)

    # 准备攻击数据
    attack_samples = original_samples[selected_idx]
    dataset = TensorDataset(attack_samples,
                            torch.zeros(len(selected_idx)))  # 伪标签
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    adv_samples = []

    for batch in dataloader:
        x, _ = batch
        x = x.to(device).requires_grad_(True)

        # 保留原始数据范围
        data_min, data_max = torch.min(x), torch.max(x)

        # 迭代生成
        current_epsilon = base_epsilon
        x_adv = x.clone()

        for _ in range(max_iter):
            # 前向传播
            output = model(x_adv.unsqueeze(1))  # 添加序列维度

            # 计算损失（仅针对正确分类的样本）
            with torch.no_grad():
                pred = output.argmax(dim=1)
                mask = (pred == 0)  # 假设正常样本标签为0

            if not mask.any():
                break  # 没有可攻击样本时提前终止

            loss = torch.nn.functional.cross_entropy(output[mask],
                                                     torch.ones(mask.sum(),
                                                                dtype=torch.long).to(device))

            # 梯度计算
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]

            # 梯度归一化
            grad_norm = torch.norm(grad, p=2)
            grad = grad / (grad_norm + 1e-8)

            # 更新对抗样本
            with torch.no_grad():
                step = current_epsilon * grad.sign()
                step = torch.clamp(step, -step_clip, step_clip)

                x_adv = x_adv + step
                x_adv = torch.clamp(x_adv, data_min, data_max)  # 保持原始范围

                # 应用频谱约束
                x_adv = apply_spectral_constraint(x_adv, x)

            current_epsilon *= decay_factor

        adv_samples.append(x_adv.cpu().detach())

    return torch.cat(adv_samples, dim=0)


def apply_spectral_constraint(adv, original):
    """频谱约束（保留主要频率分量）"""
    device = adv.device
    adv_np = adv.cpu().numpy()
    orig_np = original.cpu().numpy()

    # 保留前10%低频分量
    n_keep = int(0.1 * adv.shape[1])
    for i in range(len(adv)):
        adv_fft = np.fft.fft(adv_np[i])
        orig_fft = np.fft.fft(orig_np[i])

        # 替换高频部分
        adv_fft[n_keep:-n_keep] = orig_fft[n_keep:-n_keep]

        adv_np[i] = np.real(np.fft.ifft(adv_fft))

    return torch.from_numpy(adv_np).to(device).float()


# -------------------- 4. 数据保存与验证 --------------------
def save_and_validate(original_data, adv_samples, selected_idx, output_path):
    # 替换原始数据
    final_samples = original_data.clone()
    final_samples[selected_idx] = adv_samples

    # 生成标签
    labels = torch.zeros(len(original_data), dtype=torch.long)
    labels[selected_idx] = 1

    # 构建数据字典
    data_dict = {
        'samples': final_samples.float(),
        'labels': labels
    }
    torch.save(data_dict, output_path)

    # 执行验证
    analyze_distribution(adv_samples, original_data[selected_idx])
    visualize_comparison(original_data[selected_idx], adv_samples)


def analyze_distribution(adv, original):
    diff = adv - original
    print("\n数据分布分析:")
    print(f"原始数据范围: [{original.min():.4f}, {original.max():.4f}]")
    print(f"对抗数据范围: [{adv.min():.4f}, {adv.max():.4f}]")
    print(f"平均绝对扰动: {torch.mean(torch.abs(diff)):.4f}")
    print(f"最大扰动幅度: {torch.max(torch.abs(diff)):.4f}")
    print(f"饱和值比例（<0.01或>0.99）: {((adv < 0.01) | (adv > 0.99)).float().mean().item():.2%}")


def visualize_comparison(original, adv, n=3):
    indices = random.sample(range(len(original)), n)
    plt.figure(figsize=(15, 3 * n))

    for i, idx in enumerate(indices):
        plt.subplot(n, 1, i + 1)
        plt.plot(original[idx], label='Original', alpha=0.7)
        plt.plot(adv[idx], label='Adversarial', linestyle='--')
        plt.legend()
        plt.title(f"Sample {idx} Comparison")

    plt.tight_layout()
    plt.savefig('adversarial_comparison.png')
    plt.show()


# -------------------- 主流程 --------------------
if __name__ == "__main__":
    # 配置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = "data/electricity/aierlan_ori_data.pt"
    model_path = "exp/epilepsy/test/model.pkl"
    output_path = "data/electricity/Adversarial50%.pt"

    # 1. 加载模型
    model = load_pretrained_model(model_path, device)

    # 2. 加载数据
    original_samples, selected_idx = load_and_prepare_data(data_path)

    # 3. 生成对抗样本
    adv_samples = generate_adversarial_samples(
        model=model,
        original_samples=original_samples,
        selected_idx=selected_idx,
        device=device,
        epsilon=0.01,
        max_iter=10,
        decay=0.8,
        step_clip=0.05
    )

    # 4. 保存并验证
    save_and_validate(original_samples, adv_samples, selected_idx, output_path)