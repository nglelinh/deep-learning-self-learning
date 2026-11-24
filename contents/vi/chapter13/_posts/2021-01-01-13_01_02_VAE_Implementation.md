---
layout: post
title: 13-01-02 Cài đặt VAE
chapter: '13'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter13
---

Chỉ một sai sót nhỏ về tỷ lệ cộng giữa hàm mất mát tái tạo (tính tổng trên $$784$$ pixel) và phân kỳ KL (tính tổng trên $$20$$ chiều ẩn) cũng đủ khiến VAE của bạn hoặc sụp đổ thành cục mờ, hoặc đánh mất hoàn toàn khả năng sinh ảnh mới. Trong bài cài đặt này, ta xây dựng trọn vẹn VAE trên MNIST bằng PyTorch — từ nhánh song song `fc_mu` / `fc_logvar`, hàm `reparameterize`, hàm mất mát ELBO cho đến kỹ thuật KL annealing và vẽ lưới ảnh sinh trong không gian ẩn 2D.

## Bản đồ kiến thức nền: Các thành phần kỹ thuật cốt lõi khi lập trình VAE

- **Dự đoán `logvar` thay vì `std`:** Xuất ra $$\log(\sigma^2) \in (-\infty, +\infty)$$ không bị ràng buộc dấu, rồi tính `std = torch.exp(0.5 * logvar)` giúp ổn định số học và tránh đạo hàm nổ.
- **Cân bằng thang đo `reduction='sum'` trong hàm loss ELBO:** Cộng tổng BCE trên mọi pixel với tổng KL trên mọi chiều ẩn (rồi mới chia cho `batch_size`) — tránh bẫy dùng `reduction='mean'` làm suy yếu hệ số tái tạo đi $$784$$ lần.
- **Lấy mẫu mới và vẽ lưới đa tạp 2D:** Khi $$d = 2$$, ánh xạ lưới phân vị hàm phân phối chuẩn ngược (PPF) qua bộ giải mã để nhìn thấy toàn bộ bản đồ chuyển dạng mượt mà giữa 10 chữ số.
- **KL Annealing và $$\beta$$-VAE:** Tăng dần trọng số $$\beta$$ từ $$0 \to 1$$ trong những epoch đầu để tránh posterior collapse, hoặc chọn $$\beta > 1$$ để tách bạch các chiều thuộc tính.

### Khung giải thích (nên đọc chậm): Kiểm tra bẫy `reduction='mean'` vs. `reduction='sum'` trong hàm loss VAE

Xét một ảnh MNIST ($$784$$ pixel) với không gian ẩn $$d = 20$$ chiều. Giả sử tại một bước huấn luyện, lỗi BCE trung bình mỗi pixel là $$0{,}25$$ và phân kỳ KL trung bình mỗi chiều ẩn là $$0{,}50$$.

1. **Cách tính đúng chuẩn ELBO (`reduction='sum'` trên từng mẫu):**
   - Tổng BCE tái tạo: $$784 \times 0{,}25 = 196{,}0$$.
   - Tổng KL: $$20 \times 0{,}50 = 10{,}0$$.
   - Tổng loss mỗi mẫu: $$196{,}0 + 10{,}0 = 206{,}0$$ (tỷ lệ Tái tạo : KL = $$19{,}6 : 1$$ — hợp lý vì 784 pixel chứa nhiều thông tin hơn 20 chiều ẩn).
2. **Bẫy thường gặp (dùng `F.binary_cross_entropy(..., reduction='mean')` cộng với tổng KL):**
   - BCE trung bình: $$0{,}25$$.
   - Tổng KL: $$10{,}0$$.
   - Tỷ lệ bị đảo ngược thành $$0{,}25 : 10{,}0 = 1 : 40$$ — tương đương bạn vô tình chạy $$\beta$$-VAE với $$\beta = 784$$! Hậu quả: KL bị ép về $$0$$ ngay lập tức và mọi ảnh sinh ra đều mờ nhạt giống hệt ảnh trung bình của tập dữ liệu.

## 4. Đoạn mã

Phần mã dưới đây cài đặt đầy đủ Variational Autoencoder bằng PyTorch trên tập MNIST, bao gồm bước tái tham số hóa, hàm mất mát ELBO, sinh mẫu và nội suy không gian ẩn:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class VAE(nn.Module):
    """
    Variational Autoencoder cho MNIST.
    
    Kiến trúc:
    - Encoder: ánh xạ ảnh sang tham số phân phối ẩn (μ, σ)
    - Sampler: reparameterization trick cho backpropagation qua lấy mẫu
    - Decoder: ánh xạ mã ẩn sang ảnh tái tạo
    
    Loss: ELBO = tái tạo + phân kỳ KL
    """
    
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: xuất tham số phân phối Gaussian
        # Ta xuất cả μ và log(σ²) thay vì σ vì ổn định số
        # (σ phải dương, dễ đảm bảo hơn với exp(log σ²))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Tầng riêng cho mean và log-variance
        # Cho phép encoder học cả hai độc lập
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: ánh xạ mã ẩn sang tái tạo
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Đầu ra trong [0,1] cho giá trị pixel
        )
    
    def encode(self, x):
        """
        Encode đầu vào sang tham số phân phối ẩn.
        
        Returns:
            mu: mean của q(z|x)
            logvar: log variance của q(z|x)
        
        Ta trả log variance thay vì variance/std vì ổn định số.
        Variance phải dương, nên ta đảm bảo bằng cách lũy thừa logvar.
        Ổn định hơn dự đoán trực tiếp σ rồi bình phương.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ⊙ε với ε ~ N(0,I)
        
        Đây là ĐỔI MỚI then chốt khiến VAE huấn luyện được bằng backprop.
        Thay vì lấy mẫu z ~ N(μ,σ²) (không khả vi theo μ,σ),
        ta biểu diễn z như hàm tất định của μ,σ và ngẫu nhiên ngoài ε.
        
        Gradient có thể chảy qua μ và σ tới tham số encoder, cho phép
        huấn luyện end-to-end qua backpropagation chuẩn.
        """
        # Tính độ lệch chuẩn từ log variance
        # std = exp(log(σ²) / 2) = exp(logvar / 2)
        std = torch.exp(0.5 * logvar)
        
        # Lấy mẫu epsilon từ chuẩn chuẩn
        # Khi huấn luyện: ngẫu nhiên. Khi sinh: có thể dùng ε cụ thể
        eps = torch.randn_like(std)
        
        # Mẫu reparameterized: z = μ + σ * ε
        z = mu + std * eps
        
        return z
    
    def decode(self, z):
        """Ánh xạ mã ẩn sang tái tạo"""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass VAE đầy đủ.
        
        Returns:
            recon: đầu vào tái tạo
            mu: mean ẩn (cho tính KL)
            logvar: log-variance ẩn (cho tính KL)
        
        Ta trả mu và logvar riêng vì cần chúng để tính
        phân kỳ KL trong hàm loss.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples):
        """
        Sinh mẫu mới bằng cách lấy mẫu từ prior và giải mã.
        
        Đây là cách ta dùng VAE đã huấn luyện để sinh:
        1. Lấy mẫu z ~ N(0, I) (prior)
        2. Giải mã để nhận x
        
        Vì ta đã chính quy hóa q(z|x) gần N(0,I) trong huấn luyện,
        mẫu từ N(0,I) nên giải mã ra đầu ra thực tế.
        """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Loss VAE: âm ELBO = mất mát tái tạo + phân kỳ KL
    
    Args:
        recon_x: đầu vào tái tạo
        x: đầu vào gốc
        mu: mean ẩn từ encoder
        logvar: log-variance ẩn từ encoder
        beta: trọng số hạng tử KL (β-VAE dùng β≠1 cho disentanglement)
    
    Loss có hai hạng tử:
    1. Tái tạo: ta tái tạo đầu vào tốt thế nào
    2. KL: phân phối encoder khác prior bao nhiêu
    
    Ta muốn tối thiểu hóa cả hai: tái tạo tốt VÀ phân phối ẩn
    gần prior.
    """
    # Mất mát tái tạo (binary cross-entropy cho ảnh trong [0,1])
    # Coi mỗi pixel như Bernoulli độc lập
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Phân kỳ KL KL(N(μ,σ²) \| N(0,I))
    # Có dạng đóng: 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    # Ta có logvar = log(σ²), nên σ² = exp(logvar)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Tổng loss (âm ELBO)
    # Tối thiểu hóa điều này tương đương tối đa hóa ELBO
    return BCE + beta * KLD, BCE, KLD

# Huấn luyện VAE
print("="*70)
print("Huấn luyện Variational Autoencoder trên MNIST")
print("="*70)

# Tải dữ liệu
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Tạo VAE
vae = VAE(input_dim=784, latent_dim=20)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

print(f"Kiến trúc VAE:")
print(f"  Chiều đầu vào: 784 (28×28)")
print(f"  Chiều ẩn: 20 (hệ số nén: 39×)")
print(f"  Tổng tham số: {sum(p.numel() for p in vae.parameters()):,}")

# Vòng huấn luyện
print("\nHuấn luyện VAE...")
vae.train()

for epoch in range(10):
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        # Làm phẳng ảnh
        data = data.view(-1, 784)
        
        # Forward pass
        recon, mu, logvar = vae(data)
        
        # Tính loss
        loss, bce, kld = vae_loss(recon, data, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
    
    # Trung bình trên các batch
    n_batches = len(train_loader)
    avg_loss = train_loss / n_batches / 128  # Mỗi mẫu
    avg_bce = train_bce / n_batches / 128
    avg_kld = train_kld / n_batches / 128
    
    print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f} "
          f"(Recon = {avg_bce:.4f}, KL = {avg_kld:.4f})")

print("\n" + "="*70)
print("Huấn luyện VAE Hoàn tất - Phân tích Kết quả")
print("="*70)

# Kiểm tra tái tạo
vae.eval()
with torch.no_grad():
    test_data, _ = next(iter(test_loader))
    test_data_flat = test_data.view(-1, 784)
    
    # Encode (dùng mean, bỏ qua phương sai cho tính tất định)
    mu, logvar = vae.encode(test_data_flat)
    
    # Tái tạo
    recon = vae.decode(mu)
    
    # Đo lỗi tái tạo
    recon_error = F.mse_loss(recon, test_data_flat).item()
    print(f"MSE tái tạo kiểm tra: {recon_error:.6f}")
    
    # Phân tích thống kê không gian ẩn
    print(f"\nThống kê không gian ẩn (nên ~N(0,1) do phạt KL):")
    print(f"  Mean: {mu.mean(dim=0)[:5].numpy().round(3)} (5 chiều đầu)")
    print(f"  Std:  {torch.exp(0.5*logvar).mean(dim=0)[:5].numpy().round(3)}")
    print(f"  Magnitude mean tổng thể: {mu.abs().mean().item():.3f}")
    print(f"  Std tổng thể: {torch.exp(0.5*logvar).mean().item():.3f}")

# Sinh mẫu mới
print("\n" + "="*70)
print("Sinh Mẫu Mới từ VAE")
print("="*70)

with torch.no_grad():
    # Lấy mẫu từ prior N(0,I)
    num_samples = 64
    samples = vae.sample(num_samples)
    
    print(f"Đã sinh {num_samples} mẫu bằng cách lấy mẫu z ~ N(0,I)")
    print(f"Shape mẫu: {samples.shape}")  # (64, 784)
    
    # Kiểm tra thống kê mẫu
    print(f"Mean mẫu sinh: {samples.mean():.3f} (nên ~0.5)")
    print(f"Std mẫu sinh: {samples.std():.3f}")
    
    # Reshape để trực quan hóa
    samples_img = samples.view(-1, 1, 28, 28)
    print(f"Reshape để trực quan hóa: {samples_img.shape}")

# Minh họa nội suy
print("\n" + "="*70)
print("Nội suy Không gian Ẩn")
print("="*70)

with torch.no_grad():
    # Lấy hai ảnh kiểm tra
    img1 = test_data_flat[0:1]
    img2 = test_data_flat[7:8]
    
    # Encode sang mean ẩn
    mu1, _ = vae.encode(img1)
    mu2, _ = vae.encode(img2)
    
    print("Nội suy giữa hai ảnh trong không gian ẩn:")
    
    # Nội suy
    n_steps = 9
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        z_interp = (1-t) * mu1 + t * mu2
        img_interp = vae.decode(z_interp)
        
        if i % 2 == 0:  # In mỗi bước khác
            print(f"  t={t:.2f}: Ảnh nội suy {i+1}/{n_steps}")
    
    print("\nNội suy nên mượt nhờ chính quy hóa không gian ẩn!")
    print("Đây là lợi thế then chốt của VAE so với autoencoder vanilla.")

# Minh họa β-VAE (thay đổi trọng số KL)
print("\n" + "="*70)
print("β-VAE: Điều khiển Disentanglement")
print("="*70)
print("Bằng cách thay đổi β (trọng số hạng tử KL), ta điều khiển đánh đổi:")
print("  β < 1: Ưu tiên tái tạo (sắc nét hơn nhưng ít disentangled)")
print("  β = 1: VAE chuẩn (cân bằng)")
print("  β > 1: Ưu tiên tính đều đặn ẩn (disentangled hơn, mờ hơn)")
print("\nβ-VAE với β=4-10 thường học chiều ẩn diễn giải được hơn")
print("nơi mỗi chiều nắm một nhân tố ngữ nghĩa (kích thước, xoay, v.v.)")
```

## 5. Các khái niệm liên quan

So sánh giữa VAE và autoencoder chuẩn làm rõ giá trị của khung xác suất. Cả hai đều dùng kiến trúc encoder–decoder và hàm mất mát tái tạo, nhưng VAE bổ sung ba điểm cốt lõi: (1) mã hóa thành phân phối xác suất thay vì điểm tất định, (2) thành phần chính quy hóa phân kỳ KL kéo không gian ẩn về gần phân phối tiên nghiệm, và (3) cơ chế lấy mẫu tường minh để sinh dữ liệu mới. Nhờ tối ưu cận dưới log-hợp lý, VAE mang lại nền tảng xác suất rõ ràng hơn so với autoencoder chỉ tối ưu sai số tái tạo.

VAE gắn chặt với suy diễn biến phân trong thống kê Bayes. Khi có mô hình $$p(\mathbf{x}, \mathbf{z})$$ nhưng không tính trực tiếp được hậu nghiệm $$p(\mathbf{z}|\mathbf{x})$$, ta xấp xỉ nó bằng phân phối $$q(\mathbf{z}|\mathbf{x})$$ thuộc họ dễ tính toán (như phân phối chuẩn độc lập theo từng chiều) và tối đa hóa ELBO. VAE mở rộng phương pháp này lên dữ liệu nhiều chiều nhờ ba yếu tố: dùng mạng nơ-ron biểu diễn $$q$$ và $$p$$, áp dụng thủ thuật tái tham số hóa để tối ưu bằng gradient và huấn luyện ngẫu nhiên theo mini-batch.

Từ góc độ lý thuyết thông tin, ELBO có thể viết dưới dạng:

$$\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

$$= -H_{q_\phi}(p_\theta(\mathbf{x}|\mathbf{z})) - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

Số hạng đầu tiên thúc đẩy bộ giải mã tái tạo chính xác (độ bất định có điều kiện thấp), còn số hạng KL đo lượng thông tin cần mã hóa để chuyển từ phân phối tiên nghiệm $$p$$ sang hậu nghiệm $$q_\phi$$. Sự đánh đổi giữa độ trung thực tái tạo và mức độ nén này được điều khiển tường minh trong $$\beta$$-VAE thông qua hệ số $$\beta$$.

Đặt cạnh *normalizing flows*, cả hai họ đều dùng biến ẩn $$\mathbf{z}$$ nhưng đánh đổi khác nhau: normalizing flows sử dụng phép biến đổi song ánh khả nghịch có định thức Jacobian tính được nên cho log-hợp lý chính xác nhưng bị ràng buộc về kiến trúc; ngược lại, VAE cho phép thiết kế mạng encoder và decoder tự do nhưng chỉ tối ưu cận dưới xấp xỉ ELBO.

Cuối cùng, VAE liên hệ mật thiết với học biểu diễn tách nhân tố (*disentangled representation learning*), nơi mỗi chiều ẩn tương ứng với một thuộc tính độc lập (như tư thế, ánh sáng, hình dạng). Hậu nghiệm chuẩn độc lập theo từng chiều và việc tăng trọng số $$\beta > 1$$ trong $$\beta$$-VAE khuyến khích sự độc lập giữa các chiều ẩn, giúp biểu diễn dễ diễn giải và thuận tiện hơn cho các tác vụ về sau.

## 6. Các bài báo nền tảng

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Tác giả*: Diederik P. Kingma, Max Welling  
Bài báo đề xuất VAE và đưa suy diễn biến phân vào thực hành học sâu nhờ thủ thuật tái tham số hóa $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$ với $$\boldsymbol{\epsilon} \sim \mathcal{N}(0,I)$$. Nhóm tác giả trình bày đầy đủ phép suy ELBO, công thức KL dạng đóng cho phân phối chuẩn và kết quả thực nghiệm trên ảnh, mở ra một hướng mô hình sinh ổn định và có cơ sở xác suất rõ ràng.

**["Tutorial on Variational Autoencoders" (2016)](https://arxiv.org/abs/1606.05908)**  
*Tác giả*: Carl Doersch  
Bài hướng dẫn trình bày trực quan ý nghĩa của ELBO, thủ thuật tái tham số hóa cùng các lưu ý thực hành khi huấn luyện VAE cho người đọc chưa có nền tảng sâu về suy diễn biến phân. Tài liệu giải đáp nhiều câu hỏi phổ biến về hậu nghiệm chuẩn, cấu trúc không gian ẩn và mối liên hệ giữa VAE với autoencoder và GAN.

**["β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)](https://openreview.net/forum?id=Sy2fzU9gl)**  
*Tác giả*: Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner  
Bài báo giới thiệu $$\beta$$-VAE bằng cách nhân số hạng KL với trọng số $$\beta > 1$$. Phạt KL mạnh hơn khuyến khích các chiều ẩn độc lập với nhau, giúp tách riêng các nhân tố biến thiên như hình dạng, kích thước, góc xoay hay màu sắc vào từng chiều riêng biệt và cho phép điều chỉnh từng thuộc tính khi sinh mẫu.

**["Importance Weighted Autoencoders" (2016)](https://arxiv.org/abs/1509.00519)**  
*Tác giả*: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov  
Bài báo cải thiện cận dưới ELBO bằng kỹ thuật lấy mẫu trọng số quan trọng (*importance sampling*) trên $$K$$ mẫu từ $$q_\phi(\mathbf{z}|\mathbf{x})$$:

$$\mathcal{L}_{\text{IWAE}} = \mathbb{E}\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(\mathbf{x}, \mathbf{z}^{(k)})}{q_\phi(\mathbf{z}^{(k)}|\mathbf{x})}\right]$$

Cận dưới này chặt hơn ELBO chuẩn và tiến tới log-hợp lý thật khi $$K \to \infty$$, thường giúp cải thiện chất lượng mô hình sinh $$p_\theta(\mathbf{x}|\mathbf{z})$$.

**["Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)](https://arxiv.org/abs/1906.00446)**  
*Tác giả*: Ali Razavi, Aaron van den Oord, Oriol Vinyals  
Bài báo giới thiệu VQ-VAE-2 sử dụng mã ẩn rời rạc thông qua lượng tử hóa vector (*vector quantization*) kết hợp kiến trúc phân cấp cho cấu trúc toàn cục và cục bộ. Biểu diễn rời rạc cho phép huấn luyện phân phối tiên nghiệm tự hồi quy mạnh trên không gian mã ẩn, giúp nâng chất lượng ảnh độ phân giải cao ngang tầm GAN trong khi vẫn duy trì quá trình huấn luyện ổn định.

## Bẫy thường gặp và Mẹo

Hiện tượng thất bại phổ biến nhất khi huấn luyện VAE là **sụp đổ hậu nghiệm (*posterior collapse*)**, khi bộ mã hóa bỏ qua đầu vào và luôn xuất ra gần đúng phân phối tiên nghiệm $$q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z}) = \mathcal{N}(0,I)$$ cho mọi $$\mathbf{x}$$. Khi đó phân kỳ KL tiến về 0, còn bộ giải mã chỉ học tái tạo ảnh trung bình của toàn bộ tập dữ liệu vì mã ẩn không chứa thông tin về đầu vào. Hiện tượng này thường xảy ra khi bộ giải mã quá mạnh và tự tái tạo được khá tốt mà không cần đến $$\mathbf{z}$$.

Các biện pháp khắc phục gồm: (1) giảm dung lượng bộ giải mã để buộc nó phụ thuộc vào mã ẩn; (2) áp dụng *KL annealing* — bắt đầu với $$\beta = 0$$ rồi tăng dần lên $$\beta = 1$$ để bộ mã hóa kịp học biểu diễn hữu ích trước khi siết chính quy hóa; (3) dùng kỹ thuật *free bits* — chỉ phạt KL khi vượt qua một ngưỡng tối thiểu trên mỗi chiều ẩn; và (4) dùng learning rate lớn hơn cho bộ mã hóa so với bộ giải mã.

Việc chọn số chiều ẩn cần cân bằng giữa độ chi tiết tái tạo và khả năng diễn giải. Chiều ẩn lớn giúp tái tạo tốt hơn nhưng các chiều dễ tương quan với nhau; chiều ẩn nhỏ tăng áp lực nén và dễ tách nhân tố hơn nhưng có thể bỏ sót chi tiết. Thông thường, 10–20 chiều phù hợp cho MNIST và 64–256 chiều cho tập ảnh khuôn mặt như CelebA.

Trong $$\beta$$-VAE, hệ số $$\beta$$ điều chỉnh trực tiếp sự cân bằng: $$\beta = 1$$ là VAE chuẩn; $$\beta > 1$$ (khoảng 2–10) ưu tiên tách nhân tố phục vụ diễn giải và điều khiển thuộc tính; còn $$\beta < 1$$ (khoảng 0,1–0,5) ưu tiên độ sắc nét tái tạo.

Để giảm độ mờ của ảnh sinh ra, có thể thay phân phối đầu ra Gaussian/Bernoulli đơn giản bằng hỗn hợp logistic rời rạc hóa hoặc bộ giải mã tự hồi quy (dù tốc độ giải mã sẽ chậm hơn). Cuối cùng, khi lấy đặc trưng ẩn của VAE cho các tác vụ sau như phân loại hay phân cụm, dùng trực tiếp vector trung bình $$\boldsymbol{\mu}$$ thường ổn định hơn, trong khi lấy mẫu từ $$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$$ phù hợp hơn khi cần phản ánh độ bất định.

## Điểm then chốt

Variational Autoencoder kết hợp mạng nơ-ron với suy diễn biến phân để xây dựng mô hình sinh xác suất tối ưu cận dưới ELBO của log-hợp lý. Bộ mã hóa ánh xạ đầu vào sang tham số của phân phối chuẩn trong không gian ẩn, thủ thuật tái tham số hóa cho phép lan truyền ngược qua bước lấy mẫu, và bộ giải mã khôi phục dữ liệu từ mã ẩn. Thành phần phân kỳ KL giữ cho không gian ẩn liên tục, đầy đủ và bám sát phân phối tiên nghiệm để hỗ trợ lấy mẫu và nội suy mượt. So với GAN, VAE huấn luyện ổn định hơn và có khung xác suất rõ ràng, dù ảnh sinh ra thường kém sắc nét hơn nếu dùng hàm mất mát tái tạo đơn giản.

## Video tham khảo

- **Xem trước:** [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — Arxiv Insights, tiếng Anh. Minh họa luồng tính toán mà mã nguồn cài đặt: mã hóa, tái tham số hóa rồi giải mã.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Phân tích các vấn đề thực tế khi huấn luyện như sụp đổ hậu nghiệm và ảnh mờ.
- [MaDL - Kullback-Leibler Divergence](https://www.youtube.com/watch?v=vPvnpC0PNVQ) — Tübingen Machine Learning, tiếng Anh. Hỗ trợ kiểm tra lại công thức KL khi hàm mất mát có dấu hiệu bất thường.
