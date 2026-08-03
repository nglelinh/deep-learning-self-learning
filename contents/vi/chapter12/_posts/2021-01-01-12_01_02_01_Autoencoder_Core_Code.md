---
layout: post
title: 12-01-02-01 Cài đặt Cốt lõi Autoencoder
chapter: '12'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter12
---

## 4. Đoạn Code

Hãy triển khai autoencoder từ đầu với pipeline huấn luyện đầy đủ:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    """
    Autoencoder chuẩn với các tầng fully connected.
    
    Kiến trúc: Input → Encoder → Latent (nút thắt) → Decoder → Reconstruction
    
    Nút thắt buộc nén — chiều đầu vào > chiều ẩn.
    Mạng phải học mã hóa hiệu quả cấu trúc dữ liệu.
    """
    
    def __init__(self, input_dim=784, latent_dim=32):
        """
        input_dim: kích thước đầu vào làm phẳng (28*28=784 cho MNIST)
        latent_dim: chiều nút thắt (hệ số nén = input_dim/latent_dim)
        
        Ta dùng kiến trúc encoder–decoder đối xứng với chiều giảm rồi tăng dần:
        784 → 256 → 128 → 32 → 128 → 256 → 784
        """
        super(Autoencoder, self).__init__()
        
        # Encoder: nén dần
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # Không kích hoạt — để latent không bị chặn
        )
        
        # Decoder: giải nén dần
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Sigmoid cho giá trị pixel trong [0,1]
        )
    
    def encode(self, x):
        """Ánh xạ đầu vào sang biểu diễn ẩn"""
        return self.encoder(x)
    
    def decode(self, z):
        """Tái tạo từ mã ẩn"""
        return self.decoder(z)
    
    def forward(self, x):
        """Autoencoder đầy đủ: encode rồi decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z

class DenoisingAutoencoder(nn.Module):
    """
    Autoencoder khử nhiễu: huấn luyện tái tạo dữ liệu sạch từ đầu vào hỏng.
    
    Quá trình làm hỏng buộc học đặc trưng robust nắm cấu trúc dữ liệu
    thay vì ghi nhớ ví dụ huấn luyện. Kết quả là đặc trưng tốt hơn cho
    tác vụ downstream.
    """
    
    def __init__(self, input_dim=784, latent_dim=32, noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__()
        
        self.noise_factor = noise_factor
        
        # Cùng kiến trúc với autoencoder vanilla
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Chính quy hóa thêm
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def add_noise(self, x, noise_factor=None):
        """
        Làm hỏng đầu vào bằng nhiễu.
        
        Với MNIST, ta dùng nhiễu Gaussian và clip về [0,1].
        Các loại hỏng khác: masking (zero pixel),
        muối-tiêu, hoặc nhiễu adversarial.
        """
        if noise_factor is None:
            noise_factor = self.noise_factor
        
        noisy = x + noise_factor * torch.randn_like(x)
        return torch.clamp(noisy, 0., 1.)
    
    def forward(self, x):
        """
        Huấn luyện: làm hỏng đầu vào, encode bản hỏng, decode ra bản sạch.
        
        Khác biệt then chốt so với vanilla: ta thêm nhiễu vào đầu vào trước khi encode
        nhưng tính loss so với đầu vào sạch gốc. Điều này huấn luyện mạng khử nhiễu.
        """
        # Làm hỏng đầu vào
        x_noisy = self.add_noise(x)
        
        # Encode đầu vào hỏng
        z = self.encoder(x_noisy)
        
        # Decode (nên tái tạo đầu vào sạch, không phải đầu vào nhiễu!)
        reconstruction = self.decoder(z)
        
        return reconstruction, z, x_noisy

# Tải MNIST để minh họa
print("="*70)
print("Huấn luyện Autoencoder trên MNIST")
print("="*70)

# Tải dữ liệu
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Huấn luyện autoencoder vanilla
print("\n1. Huấn luyện Autoencoder Vanilla (latent_dim=32)")
print("-" * 70)

model_ae = Autoencoder(input_dim=784, latent_dim=32)
optimizer_ae = optim.Adam(model_ae.parameters(), lr=0.001)
criterion = nn.MSELoss()

model_ae.train()
for epoch in range(10):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # Làm phẳng ảnh: (batch, 1, 28, 28) → (batch, 784)
        data = data.view(data.size(0), -1)
        
        # Forward pass
        reconstruction, latent = model_ae(data)
        loss = criterion(reconstruction, data)
        
        # Backward pass
        optimizer_ae.zero_grad()
        loss.backward()
        optimizer_ae.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

print("\n2. Huấn luyện Autoencoder Khử nhiễu (latent_dim=32, noise=0.3)")
print("-" * 70)

model_dae = DenoisingAutoencoder(input_dim=784, latent_dim=32, noise_factor=0.3)
optimizer_dae = optim.Adam(model_dae.parameters(), lr=0.001)

model_dae.train()
for epoch in range(10):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        
        # Forward pass (thêm nhiễu bên trong)
        reconstruction, latent, noisy = model_dae(data)
        
        # Loss: tái tạo dữ liệu SẠCH từ đầu vào NHIỄU
        loss = criterion(reconstruction, data)
        
        optimizer_dae.zero_grad()
        loss.backward()
        optimizer_dae.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

# Kiểm tra và trực quan hóa
print("\n" + "="*70)
print("Kiểm tra Tái tạo")
print("="*70)

model_ae.eval()
model_dae.eval()

with torch.no_grad():
    # Lấy batch kiểm tra
    test_data, _ = next(iter(test_loader))
    test_data_flat = test_data.view(test_data.size(0), -1)
    
    # Autoencoder vanilla
    recon_ae, latent_ae = model_ae(test_data_flat)
    
    # Autoencoder khử nhiễu (thêm nhiễu khi kiểm tra nữa)
    test_noisy = model_dae.add_noise(test_data_flat)
    recon_dae, latent_dae, _ = model_dae.forward(test_data_flat)
    
    # Tính lỗi tái tạo
    mse_ae = F.mse_loss(recon_ae, test_data_flat).item()
    mse_dae = F.mse_loss(recon_dae, test_data_flat).item()
    
    print(f"MSE tái tạo AE vanilla: {mse_ae:.6f}")
    print(f"MSE tái tạo AE khử nhiễu: {mse_dae:.6f}")
    
    # Trực quan hóa một số tái tạo
    n_display = 8
    print(f"\nHiển thị {n_display} ảnh kiểm tra đầu với tái tạo...")
    
    # Reshape để trực quan hóa
    originals = test_data[:n_display].cpu().numpy()
    recon_ae_imgs = recon_ae[:n_display].view(-1, 1, 28, 28).cpu().numpy()
    recon_dae_imgs = recon_dae[:n_display].view(-1, 1, 28, 28).cpu().numpy()
    
    # In shape (sẽ hiển thị trong notebook thực)
    print(f"Shape gốc: {originals.shape}")
    print(f"Shape tái tạo: {recon_ae_imgs.shape}")

```
