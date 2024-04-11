import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from scipy.special import gamma
from torch.distributions.uniform import Uniform
import numpy as np
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def my_get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def ggd_pdf(x, alpha, beta):
    """
    计算广义高斯分布的概率密度函数值。
    参数:
    - x: 变量值。
    - alpha: 形状参数。
    - beta: 尺度参数。
    返回:
    - p: 概率密度函数值。
    """
    coef = alpha / (2 * beta * gamma(1 / alpha))
    return coef * torch.exp(-torch.pow(torch.abs(x) / beta, alpha))

def generalized_gaussian_noise(alpha, beta, shape=(1000,)):
    """
    通过拒绝采样生成广义高斯噪声。
    参数:
    - alpha: 形状参数。
    - beta: 尺度参数。
    - shape: 生成样本的形状。
    返回:
    - samples: 生成的样本。
    """
    # 使用均匀分布作为建议分布
    uniform_dist = Uniform(-3*beta, 3*beta)
    max_pdf_val = ggd_pdf(torch.tensor(0.0), alpha, beta)  # 在x=0处GGD的PDF达到最大值
    
    samples = torch.empty(shape)
    num_samples = np.prod(shape)
    samples_generated = 0

    while samples_generated < num_samples:
        # 从均匀分布中批量采样
        x = uniform_dist.sample(sample_shape=shape)
        pdf_val = ggd_pdf(x, alpha, beta)
        u = Uniform(0, max_pdf_val).sample(sample_shape=shape)
        
        # 接受采样
        accept = u <= pdf_val
        accepted_samples = x[accept]
        num_accepted = accepted_samples.numel()
        
        if samples_generated + num_accepted > num_samples:
            accepted_samples = accepted_samples[:num_samples - samples_generated]
        
        samples.view(-1)[samples_generated:samples_generated + num_accepted] = accepted_samples
        samples_generated += num_accepted
    
    return samples