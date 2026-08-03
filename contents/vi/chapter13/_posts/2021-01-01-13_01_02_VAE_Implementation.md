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

## 4. Đoạn Code

Hãy triển khai VAE đầy đủ với mọi thành phần toán học tường minh:

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
    img2 = test_data_flat[7:1]
    
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

## 5. Các Khái niệm Liên quan

Mối quan hệ giữa VAE và autoencoder chuẩn làm sáng tỏ điều khung xác suất thêm vào. Cả hai dùng kiến trúc encoder–decoder và mất mát tái tạo, nhưng VAE thêm: (1) mã hóa xác suất (phân phối thay vì điểm), (2) hạng tử chính quy hóa phân kỳ KL, (3) khả năng lấy mẫu để sinh. Các khác biệt này xuất phát từ việc VAE là mô hình xác suất có nguyên tắc tối ưu cận dưới của hợp lý, trong khi autoencoder chỉ là giảm chiều với nút thắt. Góc nhìn xác suất cung cấp đảm bảo lý thuyết: VAE xấp xỉ tối đa hóa hợp lý dữ liệu, đảm bảo mẫu sinh nên thực tế nếu huấn luyện thành công. Autoencoder không có đảm bảo như vậy — chúng tối thiểu hóa lỗi tái tạo, không trực tiếp chuyển thành sinh mẫu tốt.

VAE nối sâu với suy diễn biến phân, một kỹ thuật tổng quát trong thống kê Bayesian để xấp xỉ phân phối hậu nghiệm không tractable. Ý tưởng luôn giống nhau: ta có mô hình $$p(\mathbf{x}, \mathbf{z})$$ nhưng không thể tính $$p(\mathbf{z}|\mathbf{x})$$ chính xác, nên ta xấp xỉ bằng phân phối đơn giản hơn $$q(\mathbf{z}|\mathbf{x})$$ từ họ tractable (ở đây, Gaussian phân tích thừa). Ta tối ưu xấp xỉ bằng cách tối đa hóa ELBO, cận dưới đại lượng ta thực sự quan tâm (log-hợp lý). VAE khiến suy diễn biến phân mở rộng được cho bài toán chiều cao qua: (1) dùng mạng neuron cho $$q$$ và $$p$$, cung cấp linh hoạt khổng lồ, (2) reparameterization trick cho phép tối ưu dựa trên gradient, (3) tối ưu ngẫu nhiên cho phép huấn luyện mini-batch. Hiểu VAE qua lăng kính suy diễn biến phân nối chúng với truyền thống thống kê phong phú và thúc đẩy các mở rộng như importance-weighted VAE hoặc hierarchical VAE.

Kết nối với lý thuyết thông tin cung cấp góc nhìn khác. ELBO có thể viết:

$$\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

$$= -H_{q_\phi}(p_\theta(\mathbf{x}|\mathbf{z})) - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

Hạng tử thứ nhất là âm entropy có điều kiện — ưu tiên decoder tái tạo tự tin cho mã ẩn (bất định thấp). Hạng tử KL đo chi phí thông tin của việc dùng $$q_\phi$$ thay vì prior $$p$$. Góc nhìn lý thuyết thông tin này gợi ý VAE đánh đổi giữa độ trung thực tái tạo và nén (mã ẩn thông tin thấp), góc nhìn được hình thức hóa trong khung β-VAE nơi ta điều khiển tường minh đánh đổi này bằng $$\beta$$.

VAE liên quan đến normalizing flow qua cách xử lý biến ẩn. Cả hai dùng biến ẩn $$\mathbf{z}$$ và học ánh xạ sang dữ liệu $$\mathbf{x}$$. Tuy nhiên, flow dùng ánh xạ khả nghịch, tất định với Jacobian tractable, cho phép hợp lý chính xác. VAE dùng mạng neuron linh hoạt cho encoder/decoder nhưng xấp xỉ hợp lý qua ELBO. Flow cung cấp suy diễn chính xác nhưng đòi hỏi kiến trúc bị ràng buộc. VAE cho phép kiến trúc linh hoạt nhưng cung cấp suy diễn xấp xỉ. Đánh đổi này định hình các miền ứng dụng tương ứng: flow cho mô hình hóa mật độ chính xác, VAE cho sinh linh hoạt với huấn luyện ổn định.

Cuối cùng, VAE nối với học biểu diễn và disentanglement. Biểu diễn disentangled có các chiều ẩn riêng lẻ tương ứng các nhân tố biến thiên độc lập (với khuôn mặt: một chiều cho tư thế, chiều khác cho ánh sáng, chiều khác cho danh tính). Hậu nghiệm Gaussian phân tích thừa của VAE khuyến khích độc lập giữa các chiều ẩn, và β-VAE với $$\beta > 1$$ khuyến khích disentanglement thêm bằng cách phạt phân kỳ KL mạnh hơn. Học biểu diễn disentangled có giá trị vượt ngoài sinh: tác vụ downstream hưởng lợi từ đặc trưng diễn giải được, phân tích thừa nơi ta có thể thao tác thuộc tính cụ thể độc lập. Hiểu cách huấn luyện VAE khuyến khích disentanglement nối với các câu hỏi rộng hơn về điều gì tạo nên biểu diễn tốt và cách khám phá chúng qua học không giám sát.

## 6. Các Bài báo Nền tảng

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Tác giả*: Diederik P. Kingma, Max Welling  
Bài báo nền tảng này giới thiệu VAE và khiến suy diễn biến phân thực tiễn cho deep learning qua reparameterization trick. Kingma và Welling chỉ ra bằng cách biểu diễn lấy mẫu thành $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$ với $$\boldsymbol{\epsilon} \sim \mathcal{N}(0,I)$$, ta có thể backpropagate qua tính toán ngẫu nhiên, cho phép tối ưu dựa trên gradient của ELBO. Bài báo cung cấp suy dẫn toán học rõ ràng, đề xuất chi tiết cài đặt thực tiễn (dùng encoder/decoder Gaussian, KL dạng đóng), và chứng minh kết quả trên ảnh. VAE mang lại nhiều lợi thế so với cách tiếp cận hiện có: khung xác suất có nguyên tắc (không như autoencoder), huấn luyện ổn định (không như GAN ban đầu), và cận dưới tractable của hợp lý (không như mô hình ẩn). Công trình ảnh hưởng hàng nghìn bài báo theo dõi khám phá biến thể VAE, ứng dụng, và tính chất lý thuyết. Đọc bài báo này, người ta đánh giá cả sự tinh vi toán học (nối mạng neuron với suy diễn biến phân) và hiểu biết thực tiễn (reparameterization trick) khiến phương pháp hoạt động.

**["Tutorial on Variational Autoencoders" (2016)](https://arxiv.org/abs/1606.05908)**  
*Tác giả*: Carl Doersch  
Bài tutorial này cung cấp giới thiệu dễ tiếp cận về VAE cho độc giả không có nền tảng mạnh về suy diễn biến phân. Doersch giải thích cẩn thận trực giác đằng sau ELBO (vì sao cận dưới đủ, các hạng tử nghĩa là gì), reparameterization trick (với sơ đồ trực quan cho thấy luồng gradient), và các cân nhắc huấn luyện thực tiễn. Tutorial xử lý các nhầm lẫn phổ biến (vì sao hậu nghiệm Gaussian, cấu trúc không gian ẩn nghĩa là gì, cách chọn siêu tham số) và nối VAE với các khái niệm liên quan (autoencoder, GAN, suy diễn Bayesian). Dù không giới thiệu phương pháp mới, tutorial này giúp đáng kể việc áp dụng VAE bằng cách khiến khung dễ tiếp cận với thực hành. Nó minh họa cách trình bày rõ ràng — giải thích không chỉ phương trình là gì mà vì sao chúng có nghĩa — có thể có tác động lớn đến lĩnh vực bằng cách hạ rào cản hiểu kỹ thuật tinh vi.

**["β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)](https://openreview.net/forum?id=Sy2fzU9gl)**  
*Tác giả*: Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner  
Bài báo này giới thiệu β-VAE, một sửa đổi đơn giản nơi hạng tử KL trong loss được gán trọng số bằng $$\beta > 1$$ thay vì 1. Thay đổi tưởng chừng nhỏ này có hiệu ứng sâu lên biểu diễn ẩn đã học. β cao hơn phạt phân kỳ KL mạnh hơn, khuyến khích các chiều ẩn độc lập (phân tích thừa), thường dẫn đến biểu diễn disentangled nơi mỗi chiều nắm một nhân tố biến thiên diễn giải được. Bài báo chứng minh β-VAE học tách các nhân tố như hình dạng, kích thước, xoay, và màu sắc thành các chiều ẩn khác nhau, cho phép sinh có kiểm soát bằng cách thao tác chiều cụ thể. Công trình nối VAE với nguyên lý nút thắt thông tin và chỉ ra rằng mức nén đúng (qua β) có thể cải thiện chất lượng biểu diễn cho tác vụ downstream. β-VAE đã trở thành công cụ chuẩn để học biểu diễn disentangled và minh họa cách siêu tham số (β) có thể điều khiển tính chất định tính của biểu diễn đã học, không chỉ chỉ số định lượng như lỗi tái tạo.

**["Importance Weighted Autoencoders" (2016)](https://arxiv.org/abs/1509.00519)**  
*Tác giả*: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov  
Bài báo này cải thiện ELBO của VAE qua importance sampling, tạo cận dưới chặt hơn của log-hợp lý. VAE chuẩn dùng một mẫu duy nhất từ $$q_\phi(\mathbf{z}|\mathbf{x})$$ để ước lượng ELBO. IWAE dùng nhiều mẫu và importance weighting, cho:

$$\mathcal{L}_{\text{IWAE}} = \mathbb{E}\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(\mathbf{x}, \mathbf{z}^{(k)})}{q_\phi(\mathbf{z}^{(k)}|\mathbf{x})}\right]$$

Đây là cận dưới chặt hơn (tiệm cận log-hợp lý thật khi $$K \to \infty$$) và thường sinh mẫu tốt hơn. Bài báo chỉ ra cải thiện chất lượng đến từ huấn luyện tốt hơn mô hình sinh $$p_\theta(\mathbf{x}|\mathbf{z})$$, dù mạng suy diễn $$q_\phi$$ có thể trở nên kém chính xác hơn. IWAE chứng minh rằng ngay cả với nền tảng lý thuyết vững của VAE, vẫn còn chỗ cải thiện qua xấp xỉ biến phân tốt hơn. Ý tưởng importance weighting đã ảnh hưởng công trình sau về cải thiện cận biến phân và cho thấy kỹ thuật thống kê cổ điển (importance sampling) có thể tăng cường cách tiếp cận neuron.

**["Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)](https://arxiv.org/abs/1906.00446)**  
*Tác giả*: Ali Razavi, Aaron van den Oord, Oriol Vinyals  
Bài báo này giới thiệu VQ-VAE-2, đạt chất lượng mẫu tiên tiến cho mô hình dựa trên VAE bằng cách dùng biểu diễn ẩn rời rạc và prior phân cấp. Thay vì latent Gaussian liên tục, VQ-VAE dùng vector quantization — encoder xuất chỉ số vào codebook đã học, và decoder nhận các vectơ codebook tương ứng. Tính rời rạc này cho phép dùng prior tự hồi quy mạnh trên mã ẩn, cải thiện đáng kể chất lượng mẫu. Cấu trúc phân cấp (mã ẩn riêng cho cấu trúc toàn cục và cục bộ) cho phép sinh ảnh độ phân giải cao. Dù phức tạp hơn VAE chuẩn, VQ-VAE-2 chứng minh mô hình dựa trên VAE có thể cạnh tranh với GAN về chất lượng mẫu trong khi duy trì lợi thế của VAE về huấn luyện ổn định và cấu trúc không gian ẩn. Công trình cho thấy khung VAE linh hoạt đủ để chứa latent rời rạc, cấu trúc phân cấp, và prior tinh vi, đẩy hiệu năng VAE lên mức mới.

## Bẫy thường gặp và Mẹo

Chế độ thất bại phổ biến nhất trong huấn luyện VAE là sụp đổ hậu nghiệm (posterior collapse), nơi encoder học bỏ qua đầu vào và xuất phân phối prior cho mọi đầu vào: $$q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z}) = \mathcal{N}(0,I)$$ bất kể $$\mathbf{x}$$. Hạng tử KL trở thành zero (tốt cho hạng tử đó!) nhưng hạng tử tái tạo không thể cải thiện vì mã ẩn không chứa thông tin về đầu vào. Decoder học sinh ảnh trung bình (mean của phân phối huấn luyện) bất kể mã ẩn. Triệu chứng gồm phân kỳ KL rất thấp (tiệm cận 0) và tái tạo kém. Điều này xảy ra khi decoder quá mạnh — nó có thể tái tạo khá tốt không cần dùng thông tin ẩn, nên encoder chọn đường dễ là xuất prior để tối thiểu hóa KL.

Giải pháp gồm: (1) làm yếu decoder (ít tầng/tham số hơn), buộc nó dựa vào mã ẩn; (2) KL annealing — bắt đầu huấn luyện với β=0 (không phạt KL) và tăng dần lên β=1, cho phép encoder khám phá biểu diễn hữu ích trước khi áp dụng chính quy hóa; (3) free bits — chỉ phạt KL nếu trên ngưỡng, đảm bảo các chiều ẩn duy trì nội dung thông tin tối thiểu; (4) tối ưu tốt hơn — dùng learning rate cao hơn cho encoder so với decoder, cho nó lợi thế trong cạnh tranh dung lượng. Hiểu sụp đổ hậu nghiệm như bệnh lý tối ưu hóa chứ không phải hạn chế căn bản của VAE giúp cài đặt các giải pháp này phù hợp.

Chọn chiều ẩn liên quan đến đánh đổi giữa khả năng biểu đạt và disentanglement. Chiều ẩn lớn hơn có thể nắm nhiều biến thiên hơn (tốt cho tái tạo) nhưng có xu hướng ít disentangled hơn (các chiều trở nên tương quan, khó diễn giải). Chiều ẩn nhỏ hơn buộc nén nhiều hơn và thường học biểu diễn disentangled hơn nhưng có thể không nắm mọi biến thiên dữ liệu (tái tạo kém). Với MNIST, 10–20 chiều thường đủ. Với khuôn mặt CelebA, 64–256 chiều phổ biến. Luôn kiểm chứng cả chất lượng tái tạo (định lượng) và diễn giải không gian ẩn (định tính).

Lựa chọn β trong β-VAE ảnh hưởng đáng kể kết quả. β=1 là VAE chuẩn, cân bằng tái tạo và chính quy hóa. β>1 (thường 2–10) ưu tiên disentanglement, hữu ích khi diễn giải quan trọng hơn tái tạo hoàn hảo. β<1 (thường 0.1–0.5) ưu tiên tái tạo, hữu ích khi chất lượng sinh quan trọng hơn cấu trúc không gian ẩn. Với học biểu diễn (dùng đặc trưng VAE cho tác vụ downstream), β=1–2 thường hoạt động tốt. Với sinh có kiểm soát (thao tác thuộc tính cụ thể), β=4–10 cung cấp latent disentangled hơn. Hiểu núm điều khiển này cho phép tùy chỉnh VAE theo yêu cầu ứng dụng cụ thể.

Một kỹ thuật mạnh để cải thiện chất lượng mẫu là dùng phân phối decoder tinh vi hơn Gaussian hoặc Bernoulli. Hỗn hợp logistic rời rạc hóa (mô hình hóa giá trị pixel như hỗn hợp phân phối đã bin) nắm đa mode tốt hơn Gaussian đơn. Decoder tự hồi quy (mỗi pixel dự đoán có điều kiện trên pixel trước) nắm các phụ thuộc mà giả định phân tích thừa của autoencoder bỏ lỡ. Các decoder biểu đạt hơn này thường sinh mẫu sắc nét hơn trong khi duy trì huấn luyện ổn định của VAE. Đánh đổi là chi phí tính toán — giải mã tự hồi quy chậm. Hiểu rằng lựa chọn decoder ảnh hưởng cả chất lượng mẫu lẫn tốc độ huấn luyện/sinh hướng dẫn quyết định kiến trúc phù hợp.

Khi dùng mã ẩn VAE cho tác vụ downstream (phân loại, clustering), quyết định là dùng mean $$\boldsymbol{\mu}$$ hay lấy mẫu từ $$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$$. Với tác vụ tất định (phân loại), dùng mean cung cấp đặc trưng ổn định. Với tác vụ đòi hỏi bất định (active learning, suy diễn Bayesian), lấy mẫu phản ánh bất định của encoder. Với hầu hết ứng dụng, mean hoạt động tốt và là thực hành chuẩn, nhưng hiểu rằng phân phối đầy đủ sẵn có cho phép sử dụng tinh vi hơn khi phù hợp.

## Điểm then chốt

Variational Autoencoder kết hợp mạng neuron với suy diễn biến phân để tạo khung xác suất có nguyên tắc cho mô hình hóa sinh với biến ẩn, tối ưu Evidence Lower BOund của log-hợp lý như surrogate tractable cho hợp lý thật không tractable. Encoder học ánh xạ đầu vào sang phân phối trên mã ẩn (tham số hóa như Gaussian với mean và variance đã học), trong khi decoder học tái tạo đầu vào từ mẫu ẩn, cả hai được huấn luyện chung qua mục tiêu ELBO kết hợp độ chính xác tái tạo và chính quy hóa phân phối ẩn. Reparameterization trick — biểu diễn lấy mẫu ngẫu nhiên như hàm tất định của tham số và ngẫu nhiên ngoài — cho phép backpropagation qua lấy mẫu, khiến huấn luyện end-to-end khả thi với gradient descent chuẩn. Phân kỳ KL giữa hậu nghiệm xấp xỉ và prior chính quy hóa không gian ẩn để liên tục, đầy đủ, và quanh prior, đảm bảo mẫu từ prior giải mã ra đầu ra thực tế và cho phép nội suy mượt giữa các ví dụ. β-VAE mở rộng khung bằng cách gán trọng số hạng tử KL, đánh đổi chất lượng tái tạo lấy disentanglement ẩn và cho phép biểu diễn đã học nơi các chiều riêng lẻ nắm các nhân tố biến thiên diễn giải được. VAE cung cấp huấn luyện ổn định so với GAN, cấu trúc không gian ẩn tường minh cho phép nội suy và thao tác, và khung xác suất có nguyên tắc hỗ trợ phân tích lý thuyết, dù thường tạo mẫu hơi mờ hơn mô hình huấn luyện đối kháng. Hiểu VAE đòi hỏi đánh giá sự tương tác giữa deep learning (encoder/decoder neuron), lý thuyết xác suất (mô hình biến ẩn), và tối ưu hóa (cận biến phân), khiến chúng vừa giàu lý thuyết vừa có giá trị thực tiễn cho sinh, học biểu diễn, và học bán giám sát.

Variational autoencoder minh họa cách kết hợp ý tưởng từ các lĩnh vực khác nhau — mạng neuron, suy diễn biến phân, lý thuyết thông tin — có thể tạo phương pháp mạnh hơn tổng các phần.
