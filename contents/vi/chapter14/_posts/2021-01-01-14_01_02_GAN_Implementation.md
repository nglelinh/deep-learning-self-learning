---
layout: post
title: 14-01-02 Cài đặt GAN
chapter: '14'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter14
---

Viết một mạng GAN trong PyTorch giống như điều khiển một chiếc xe đua không có phanh ABS: chỉ cần quên `.detach()` ở một dòng hay chọn `beta1=0.9` mặc định của Adam thay vì `0.5`, toàn bộ quá trình huấn luyện có thể vỡ vụn sau vài epoch. Bài thực hành này trình bày đầy đủ mã nguồn DCGAN/MLP-GAN trên MNIST kèm từng chi tiết kỹ thuật quan trọng giúp vòng lặp đối kháng vận hành trơn tru.

## Kiến thức cần nhớ: Kỹ thuật lập trình GAN ổn định trong PyTorch

- **Chuẩn hóa đầu vào về `[-1, 1]` khớp với `Tanh`:** Đảm bảo thang giá trị của ảnh thật và đầu ra Generator đồng nhất để Discriminator không gian lận bằng cách phát hiện lệch biên độ.
- **Tách đồ thị gradient đúng cách (`.detach()`):** Khi cập nhật Discriminator, bắt buộc tách `fake_imgs.detach()` để tiết kiệm bộ nhớ và tránh truyền nhầm gradient ngược vào Generator.
- **Thủ thuật huấn luyện thực chiến (GAN Hacks):** Dùng `LeakyReLU(0.2)` trong $$D$$ để tránh chết nơ-ron, giảm momentum `betas=(0.5, 0.999)` trong Adam, và áp dụng *one-sided label smoothing* (nhãn thật $$0{,}9$$ thay vì $$1{,}0$$).
- **Giám sát huấn luyện bằng vector nhiễu cố định (`fixed_noise`):** Lưu cố định một batch $$\mathbf{z}_{\text{fixed}}$$ qua mọi epoch để quan sát trực tiếp sự tiến hóa của cùng các ảnh sinh theo thời gian.

### Giải thích: Tính tay một bước cập nhật Discriminator với One-Sided Label Smoothing

Giả sử trên một mẫu thật $$x$$ và một mẫu giả $$\tilde{x} = G(z)$$, bộ phân biệt dự đoán xác suất $$D(x) = 0{,}95$$ và $$D(\tilde{x}) = 0{,}20$$.

1. **Không dùng label smoothing (nhãn thật $$y_{\text{real}} = 1{,}0$$, nhãn giả $$y_{\text{fake}} = 0{,}0$$):**
   - $$\mathcal{L}_{\text{real}} = -\ln(0{,}95) \approx 0{,}0513$$.
   - $$\mathcal{L}_{\text{fake}} = -\ln(1 - 0{,}20) = -\ln(0{,}80) \approx 0{,}2231$$.
   - Gradient theo logit mẫu thật: $$D(x) - 1{,}0 = -0{,}05$$ (vẫn tiếp tục đẩy $$D(x) \to 1{,}0$$ làm $$D$$ quá tự tin).
2. **Có One-Sided Label Smoothing (nhãn thật $$y_{\text{real}} = 0{,}9$$, nhãn giả giữ nguyên $$0{,}0$$):**
   - Gradient theo logit mẫu thật: $$D(x) - 0{,}9 = 0{,}95 - 0{,}90 = +0{,}05$$!
   - Dấu dương này **kéo nhẹ $$D(x)$$ lùi từ $$0{,}95$$ về quanh $$0{,}90$$**, ngăn bộ phân biệt rơi vào trạng thái bão hòa quá tự tin ($$D(x) \to 1$$) — nhờ đó giữ cho gradient chảy về Generator luôn khỏe mạnh.

## 4. Đoạn mã

Phần mã dưới đây cài đặt một mô hình GAN tích chập đơn giản (theo phong cách DCGAN) bằng PyTorch trên tập dữ liệu MNIST để minh họa vòng lặp huấn luyện luân phiên giữa bộ phân biệt và bộ sinh:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class Generator(nn.Module):
    """
    Generator: ánh xạ nhiễu ngẫu nhiên sang dữ liệu giả.
    
    Kiến trúc theo mẫu phổ biến cho sinh ảnh:
    - Bắt đầu với độ phân giải không gian thấp nhưng nhiều kênh
    - Upsample dần về không gian trong khi giảm kênh
    - Tầng cuối xuất ảnh với kích thước đúng
    
    Với MNIST: nhiễu (100) → (256×7×7) → (128×14×14) → (1×28×28)
    """
    
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Chiếu và reshape nhiễu
        # Tầng linear: 100 → 256*7*7, rồi reshape thành (256, 7, 7)
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        
        # Upsample qua transposed convolution
        self.deconv = nn.Sequential(
            # 256×7×7 → 128×14×14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Ổn định huấn luyện
            nn.ReLU(),
            
            # 128×14×14 → 1×28×28
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Đầu ra trong [-1, 1] (ta sẽ chuẩn hóa dữ liệu thật để khớp)
        )
    
    def forward(self, z):
        """
        z: (batch, latent_dim) nhiễu ngẫu nhiên
        Returns: (batch, 1, 28, 28) ảnh sinh
        
        Forward pass biến đổi nhiễu không cấu trúc thành ảnh có cấu trúc
        qua các phép biến đổi đã học. Sớm trong huấn luyện, đầu ra
        là nhiễu. Khi huấn luyện tiến triển, cấu trúc giống chữ số nổi lên.
        """
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)  # Reshape thành feature map
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator: phân biệt thật với giả.
    
    Kiến trúc phản chiếu generator ngược lại:
    - Đầu vào: ảnh (1×28×28)
    - Tầng conv downsample dần trong khi tăng kênh
    - Cuối: đầu ra vô hướng (xác suất ảnh là thật)
    
    Dùng LeakyReLU thay vì ReLU để ngăn đơn vị chết, và không pooling
    (dùng stride để downsample) theo thực hành tốt nhất DCGAN.
    """
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            # 1×28×28 → 64×14×14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),  # Độ dốc âm 0.2
            
            # 64×14×14 → 128×7×7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Flatten: 128×7×7 → 6272
            nn.Flatten(),
            
            # Phân loại cuối
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()  # Xác suất là thật
        )
    
    def forward(self, x):
        """
        x: (batch, 1, 28, 28) ảnh
        Returns: (batch, 1) xác suất là thật
        
        Discriminator học đặc trưng phân cấp để phát hiện:
        - Tầng sớm: cạnh, texture (phân biệt texture giả với thật)
        - Tầng giữa: hình dạng, mẫu (phát hiện chữ số sai giải phẫu)  
        - Tầng muộn: đặc trưng tổng thể (nhận diện khác biệt thống kê tinh tế)
        """
        return self.conv(x)

# Huấn luyện GAN
print("="*70)
print("Huấn luyện Mạng Sinh Đối kháng trên MNIST")
print("="*70)

# Siêu tham số
latent_dim = 100
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5  # Adam beta1 (thấp hơn mặc định 0.9 cho ổn định GAN)

# Tải dữ liệu (chuẩn hóa về [-1, 1] để khớp tanh của generator)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Chuẩn hóa về [-1, 1]
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                          shuffle=True, drop_last=True)

# Khởi tạo mạng
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Optimizer (cả hai dùng Adam với β1=0.5 cho ổn định)
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Hàm mất mát (binary cross-entropy)
criterion = nn.BCELoss()

# Nhãn cho thật và giả (dùng trong tính loss)
real_label = 1.0
fake_label = 0.0

print(f"\nTham số Generator: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Tham số Discriminator: {sum(p.numel() for p in discriminator.parameters()):,}")
print(f"\nHuấn luyện {num_epochs} epoch...")
print("Điều này minh họa động lực huấn luyện đối kháng:\n")

# Vòng huấn luyện
G_losses = []
D_losses = []

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size_actual = real_images.size(0)
        
        # ==================== Huấn luyện Discriminator ====================
        # Discriminator muốn tối đa hóa: log D(x) + log(1 - D(G(z)))
        # Tương đương, tối thiểu hóa: -log D(x) - log(1 - D(G(z)))
        
        discriminator.zero_grad()
        
        # Huấn luyện trên dữ liệu thật: tối đa hóa log D(x)
        # Loss: -log D(x) (đảo dấu vì ta tối thiểu hóa)
        labels_real = torch.full((batch_size_actual, 1), real_label)
        output_real = discriminator(real_images)
        loss_D_real = criterion(output_real, labels_real)
        
        # Huấn luyện trên dữ liệu giả: tối đa hóa log(1 - D(G(z)))
        # Loss: -log(1 - D(G(z)))
        z = torch.randn(batch_size_actual, latent_dim)
        fake_images = generator(z)
        labels_fake = torch.full((batch_size_actual, 1), fake_label)
        output_fake = discriminator(fake_images.detach())  # Detach! Không backprop qua G
        loss_D_fake = criterion(output_fake, labels_fake)
        
        # Tổng loss discriminator
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        
        # ==================== Huấn luyện Generator ====================
        # Generator muốn tối thiểu hóa: -log D(G(z))
        # Tương đương, tối đa hóa: log D(G(z)) (mục tiêu non-saturating)
        
        generator.zero_grad()
        
        # Sinh giả lại (không detach lần này — ta cần gradient qua G!)
        z = torch.randn(batch_size_actual, latent_dim)
        fake_images = generator(z)
        output_fake_for_G = discriminator(fake_images)
        
        # Generator cố khiến discriminator xuất 1 (thật) cho giả của nó
        labels_real_for_G = torch.full((batch_size_actual, 1), real_label)
        loss_G = criterion(output_fake_for_G, labels_real_for_G)
        
        loss_G.backward()
        optimizer_G.step()
        
        # Theo dõi loss
        if i == 0:  # Một lần mỗi epoch
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
    
    # In tiến độ
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}]  "
              f"D_loss: {loss_D.item():.4f}  "
              f"G_loss: {loss_G.item():.4f}  "
              f"D(x): {output_real.mean():.3f}  "
              f"D(G(z)): {output_fake.mean():.3f}")

print("\n" + "="*70)
print("Huấn luyện Hoàn tất! Phân tích Kết quả")
print("="*70)

# Sinh mẫu
generator.eval()
with torch.no_grad():
    # Sinh 64 mẫu
    z_sample = torch.randn(64, latent_dim)
    fake_samples = generator(z_sample)
    
    print(f"\nĐã sinh {fake_samples.size(0)} chữ số MNIST giả")
    print(f"Shape mẫu: {fake_samples.shape}")  # (64, 1, 28, 28)
    
    # Kiểm tra ý kiến của discriminator về mẫu sinh
    disc_scores = discriminator(fake_samples)
    print(f"Điểm discriminator cho mẫu sinh:")
    print(f"  Mean: {disc_scores.mean():.3f} (lý tưởng ~0.5)")
    print(f"  Std:  {disc_scores.std():.3f}")
    
    # Nếu mean gần 0.5, generator đánh lừa thành công discriminator
    if disc_scores.mean() > 0.4 and disc_scores.mean() < 0.6:
        print("  ✓ Generator đánh lừa thành công discriminator!")
    elif disc_scores.mean() < 0.3:
        print("  ✗ Discriminator vẫn dễ dàng phát hiện giả")
    else:
        print("  ~ Generator khá thuyết phục")

# Minh họa nội suy không gian ẩn
print("\n" + "="*70)
print("Nội suy Không gian Ẩn trong GAN")
print("="*70)

with torch.no_grad():
    # Hai mã ẩn ngẫu nhiên
    z1 = torch.randn(1, latent_dim)
    z2 = torch.randn(1, latent_dim)
    
    # Nội suy
    n_steps = 7
    print(f"Sinh {n_steps} ảnh bằng cách nội suy mã ẩn:\n")
    
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        z_interp = (1-t) * z1 + t * z2
        img = generator(z_interp)
        
        print(f"  Bước {i} (t={t:.2f}): Shape ảnh sinh {img.shape}")
    
    print("\nNội suy nên cho thấy biến hình mượt giữa các chữ số khác nhau.")
    print("Chất lượng nội suy chỉ ra cấu trúc không gian ẩn.")

print("\n" + "="*70)
print("Hiểu biết về Huấn luyện GAN")
print("="*70)
print("\nQuan sát then chốt từ huấn luyện:")
print("1. Động lực đối kháng tạo chương trình học tự nhiên")
print("2. Cân bằng giữa D và G then chốt (không bên nào nên thống trị)")
print("3. Giá trị loss không chỉ ra trực tiếp chất lượng (kiểm tra mẫu!)")
print("4. Sụp đổ mode là nguy cơ liên tục (theo dõi đa dạng)")
print("5. Mẫu sinh có thể thực tế dù cân bằng không hoàn hảo")
```

## 5. Các khái niệm liên quan

So sánh giữa GAN và Variational Autoencoder (VAE) làm rõ hai hướng đánh đổi chính của mô hình hóa sinh. VAE biểu diễn tường minh phân phối dữ liệu qua biến ẩn $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$ và tối ưu cận dưới ELBO, cho quy trình huấn luyện ổn định cùng cơ sở xác suất chặt chẽ nhưng ảnh thường mờ hơn do hàm mất mát tái tạo. GAN mô hình hóa mật độ ẩn qua bộ sinh và dùng tín hiệu đối kháng từ bộ phân biệt, nhờ đó ảnh đầu ra thường sắc nét hơn nhưng thiếu hàm mật độ tường minh và khó giữ ổn định khi huấn luyện.

Về mặt lý thuyết, GAN gắn trực tiếp với lý thuyết trò chơi qua bài toán minimax hai người tổng bằng không. Tại điểm cân bằng Nash, không bên nào có thể cải thiện đơn phương, tương ứng với phân phối của bộ sinh khớp phân phối dữ liệu thật. Tuy nhiên, việc áp dụng hạ gradient cục bộ lên một trò chơi không lồi nhiều chiều không đảm bảo hội tụ tới cân bằng Nash, giải thích vì sao quá trình huấn luyện dễ dao động và thúc đẩy các thuật toán như *unrolled optimization*.

GAN cũng có liên hệ gần với bài toán mẫu đối kháng (*adversarial examples*) và tính bền vững của bộ phân loại: thay vì thêm nhiễu nhỏ để đánh lừa bộ phân loại, bộ sinh tổng hợp toàn bộ mẫu đầu vào nhằm đánh lừa bộ phân biệt, còn bộ phân biệt học chống lại mẫu giả đó.

Ngoài nhiệm vụ sinh mẫu, GAN còn hỗ trợ học bán giám sát (*semi-supervised learning*). Khi mở rộng bộ phân biệt để vừa phân biệt thật/giả vừa phân loại nhãn trên phần dữ liệu có nhãn, mô hình có thể khai thác thêm tập dữ liệu lớn không nhãn nhằm cải thiện hiệu năng phân loại trong bối cảnh thiếu nhãn.

Từ góc nhìn học tăng cường (RL), bộ sinh không nhận nhãn đầu ra trực tiếp mà học qua tín hiệu đánh giá từ bộ phân biệt — tương tự tác nhân học từ phần thưởng (*reward*) trong các phương pháp policy gradient. Mối liên hệ này tạo tiền đề cho nhiều hướng kết hợp giữa huấn luyện đối kháng và học tăng cường.

## 6. Các bài báo nền tảng

**["Generative Adversarial Networks" (2014)](https://arxiv.org/abs/1406.2661)**  
*Tác giả*: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
Bài báo đề xuất khung GAN dưới dạng trò chơi minimax, chứng minh tại điểm cân bằng tối ưu bộ sinh khôi phục đúng phân phối dữ liệu và trình bày kết quả thực nghiệm ban đầu. Đóng góp lớn nhất là mở ra hướng mô hình hóa sinh qua cạnh tranh đối kháng thay vì tối đa hóa hàm hợp lý hay sai số tái tạo, đồng thời chỉ ra sớm các thách thức về độ ổn định và sụp đổ mode.

**["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2016)](https://arxiv.org/abs/1511.06434)**  
*Tác giả*: Alec Radford, Luke Metz, Soumith Chintala  
DCGAN đưa ra bộ quy tắc thiết kế kiến trúc giúp huấn luyện GAN tích chập ổn định hơn: thay pooling bằng tích chập có stride, dùng batch normalization ở cả hai mạng (trừ tầng đầu ra của bộ sinh và tầng đầu vào của bộ phân biệt), dùng ReLU (và tanh ở đầu ra) cho bộ sinh cùng LeakyReLU cho bộ phân biệt. Bài báo cũng cho thấy không gian ẩn của GAN hỗ trợ các phép toán vector mang ý nghĩa ngữ nghĩa trên ảnh khuôn mặt 64×64.

**["Improved Techniques for Training GANs" (2016)](https://arxiv.org/abs/1606.03498)**  
*Tác giả*: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen  
Bài báo giới thiệu hàng loạt kỹ thuật cải thiện ổn định huấn luyện như *feature matching*, *minibatch discrimination* (chống sụp đổ mode), *historical averaging*, *one-sided label smoothing* (dùng 0,9 thay vì 1,0 cho nhãn thật) và *virtual batch normalization*, đồng thời đề xuất chỉ số đánh giá tự động Inception Score.

**["Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018)](https://arxiv.org/abs/1710.10196)**  
*Tác giả*: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (NVIDIA)  
Bài báo đề xuất huấn luyện tăng dần độ phân giải (*progressive growing*): bắt đầu từ ảnh 4×4 rồi bổ sung dần các tầng cho cả bộ sinh và bộ phân biệt qua các mức 8×8, 16×16 cho tới 1024×1024. Lộ trình học từ cấu trúc thô đến chi tiết tinh giúp ổn định quá trình huấn luyện và mở khóa khả năng tổng hợp ảnh độ phân giải cao.

**["A Style-Based Generator Architecture for Generative Adversarial Networks" (2019)](https://arxiv.org/abs/1812.04948)**  
*Tác giả*: Tero Karras, Samuli Laine, Timo Aila (NVIDIA)  
StyleGAN thiết kế lại bộ sinh bằng cách ánh xạ vector nhiễu qua một mạng trung gian sang không gian phong cách $$\mathcal{W}$$, rồi điều biến từng mức độ phân giải thông qua *adaptive instance normalization*. Kiến trúc này cho phép điều khiển độc lập các đặc trưng thô (tư thế, hình dạng khuôn mặt) với chi tiết tinh (tóc, bề mặt da), đồng thời đề xuất thước đo *perceptual path length* để đánh giá độ tách nhân tố của không gian ẩn.

## Bẫy thường gặp và Mẹo

**Sụp đổ mode (*mode collapse*)** là vấn đề điển hình khi bộ sinh chỉ tập trung tạo một vài dạng mẫu dễ đánh lừa bộ phân biệt thay vì bao phủ toàn bộ phân phối (chẳng hạn chỉ sinh chữ số 1 và 7 trên MNIST). Để phát hiện, cần kiểm tra độ đa dạng trên số lượng mẫu đủ lớn chứ không chỉ xem chất lượng vài mẫu lẻ. Các hướng giảm thiểu gồm *minibatch discrimination*, *unrolled optimization* hoặc chuyển sang các hàm mất mát ổn định hơn như Wasserstein GAN.

**Bộ phân biệt áp đảo quá sớm** khiến $$D(G(\mathbf{z}))$$ nhanh chóng tiến về 0 và làm suy biến gradient cho bộ sinh. Dấu hiệu nhận biết là $$D(\mathbf{x}_{\text{real}})$$ luôn sát 1 còn $$D(G(\mathbf{z}))$$ luôn sát 0. Cách xử lý gồm giảm tần suất cập nhật hoặc learning rate của bộ phân biệt, thêm nhiễu nhẹ vào đầu vào của bộ phân biệt ở giai đoạn đầu, hoặc dùng *one-sided label smoothing* (đặt nhãn thật bằng 0,9 thay vì 1,0).

**Batch normalization với batch size nhỏ** trong bộ phân biệt có thể gây bất ổn do thống kê batch có phương sai lớn. Nên dùng batch size đủ lớn (từ 32–64 trở lên) hoặc chuyển sang layer normalization, instance normalization hay virtual batch normalization.

**Đánh giá chất lượng GAN** không thể dựa vào log-hợp lý hay giá trị loss đơn thuần (loss của $$G$$ và $$D$$ dao động liên tục theo thế cân bằng giữa hai mạng). Nên kết hợp các chỉ số tự động như Inception Score, FID cùng với kiểm tra trực quan định kỳ trên lưới mẫu sinh ra để theo dõi cả độ sắc nét lẫn độ đa dạng.

**Spectral normalization** là kỹ thuật hiệu quả và ít tốn chi phí để ổn định bộ phân biệt bằng cách chuẩn hóa ma trận trọng số theo giá trị kỳ dị lớn nhất (ràng buộc hằng số Lipschitz), ngăn gradient bùng nổ và đã trở thành cấu hình tiêu chuẩn trong nhiều kiến trúc GAN hiện đại.

## Điểm then chốt

GAN tổng hợp dữ liệu bằng cơ chế cạnh tranh minimax giữa bộ sinh và bộ phân biệt, học ngầm cách lấy mẫu từ phân phối dữ liệu mà không cần biểu diễn mật độ tường minh. Quá trình huấn luyện luân phiên giữa hai mạng đòi hỏi cân bằng chặt chẽ để tránh tình trạng bộ phân biệt áp đảo hoặc sụp đổ mode. Dù khó huấn luyện và khó đánh giá hơn VAE, GAN tạo ra mẫu sắc nét và không gian ẩn giàu cấu trúc ngữ nghĩa, đồng thời chứng minh rằng huấn luyện đối kháng là một nguyên lý hiệu quả vượt ra ngoài phạm vi mô hình hóa sinh.

## Video tham khảo

- **Xem trước:** [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, tiếng Anh. Minh họa trực quan hai bước cập nhật luân phiên được cài đặt trong mã nguồn.
- [What are GANs (Generative Adversarial Networks)?](https://www.youtube.com/watch?v=TpMIssRdhco) — IBM Technology, tiếng Anh. Nhắc lại nhanh từng mạng tối ưu hàm mất mát nào.
- [The Math Behind Generative Adversarial Networks Clearly Explained!](https://www.youtube.com/watch?v=Gib_kiXgnvA) — Normalized Nerd, tiếng Anh. Hỗ trợ đối chiếu dấu hàm mất mát giữa công thức lý thuyết và đoạn mã PyTorch.
