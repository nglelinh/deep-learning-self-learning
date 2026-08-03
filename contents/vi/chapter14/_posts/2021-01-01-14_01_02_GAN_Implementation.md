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

## 4. Đoạn Code

Hãy triển khai GAN đầy đủ từ đầu để hiểu động lực huấn luyện:

```python
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

## 5. Các Khái niệm Liên quan

Mối quan hệ giữa GAN và variational autoencoder làm sáng tỏ các cách tiếp cận khác nhau đối với mô hình hóa sinh. VAE mô hình hóa tường minh phân phối dữ liệu qua mô hình biến ẩn $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$, huấn luyện qua tối đa hóa cận dưới biến phân của hợp lý. Khung xác suất này cung cấp đảm bảo lý thuyết và cho phép suy diễn Bayesian có nguyên tắc nhưng đòi hỏi chọn dạng tham số cho phân phối và thường tạo mẫu mờ hơn do mất mát tái tạo. GAN mô hình hóa ẩn phân phối qua ánh xạ đã học của generator từ nhiễu sang dữ liệu, dùng huấn luyện đối kháng thay vì hợp lý. Điều này cho phép sinh mẫu sắc nét, thực tế hơn (vì discriminator có thể học tương tự cảm nhận thay vì tái tạo theo pixel) nhưng thiếu đảm bảo lý thuyết và năng lực ước lượng mật độ của VAE. Hiểu cả hai cách tiếp cận tiết lộ các đánh đổi khác nhau: VAE cho hiểu biết lý thuyết và mô hình hóa mật độ, GAN cho chất lượng mẫu và linh hoạt.

GAN nối với lý thuyết trò chơi qua công thức minimax. Generator và discriminator chơi trò chơi hai người tổng bằng không nơi lợi của một (discriminator nhận diện đúng giả) là thiệt của kia (giả của generator bị phát hiện). Cân bằng Nash — nơi không người chơi nào có thể cải thiện bằng cách đơn phương thay đổi chiến lược — tương ứng generator khớp phân phối dữ liệu. Tuy nhiên, đạt cân bằng Nash trong thực tiễn là thách thức vì ta dùng tối ưu dựa trên gradient, thực hiện bước cục bộ, trong trò chơi không lồi nơi cân bằng có thể không tồn tại hoặc không ổn định. Kết nối với lý thuyết trò chơi này giúp hiểu vì sao huấn luyện GAN có thể bất ổn (nhiều trò chơi không có cân bằng Nash chiến lược thuần hoặc có nhiều cân bằng) và thúc đẩy các thuật toán từ lý thuyết trò chơi như unrolled optimization.

Mối quan hệ với adversarial example và tính robust cung cấp góc nhìn thú vị. Trong nghiên cứu adversarial example, ta nhiễu loạn đầu vào nhẹ để đánh lừa bộ phân loại. Trong GAN, ta làm điều tương tự nhưng tham vọng hơn: tạo đầu vào tổng hợp hoàn toàn đánh lừa discriminator. Discriminator cố chống bị lừa tương tự huấn luyện adversarial cho bộ phân loại robust. Kết nối này gợi ý kỹ thuật từ robustness adversarial (như certified defense) có thể áp dụng để ổn định huấn luyện GAN, và ngược lại, discriminator của GAN có thể cung cấp hiểu biết về điều gì khiến bộ phân loại dễ bị adversarial example. Kết nối toán học sâu: cả hai liên quan tối ưu trên không gian đầu vào để tối đa hóa hoặc tối thiểu hóa đầu ra bộ phân loại.

Tác động của GAN lên học bán giám sát chứng tỏ cách mô hình sinh có thể cải thiện tác vụ phân biệt. Bằng cách thêm tác vụ phụ cho discriminator — không chỉ thật/giả mà còn phân loại ảnh thật vào các danh mục — ta có thể tận dụng dữ liệu không gán nhãn (dùng cho huấn luyện đối kháng) để cải thiện phân loại trên dữ liệu gán nhãn hạn chế. Discriminator học biểu diễn qua cả hai tác vụ, với tác vụ sinh cung cấp chính quy hóa và tín hiệu huấn luyện bổ sung. Khung GAN bán giám sát này đã thành công trong chế độ dữ liệu thấp, cho thấy học sinh và phân biệt có thể cùng có lợi.

Cuối cùng, GAN nối với chủ đề rộng hơn về học không giám sát trực tiếp trên tác vụ mục tiêu. Ta không bao giờ cho generator xem ví dụ đầu ra — nó học thuần từ phản hồi discriminator. Điều này tương tự reinforcement learning nơi agent học từ tín hiệu thưởng thay vì ví dụ giám sát. Thật vậy, GAN có thể được xem như áp dụng phương pháp policy gradient (từ RL) cho mô hình hóa sinh, với discriminator cung cấp thưởng (điểm cao cho giả tốt) hướng dẫn cải thiện generator. Kết nối này đã dẫn đến các cách tiếp cận lai kết hợp huấn luyện GAN với nguyên tắc reinforcement learning cho ổn định và hiệu năng tốt hơn.

## 6. Các Bài báo Nền tảng

**["Generative Adversarial Networks" (2014)](https://arxiv.org/abs/1406.2661)**  
*Tác giả*: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
Bài báo nền tảng này giới thiệu khung GAN và vẫn là một trong những bài báo ảnh hưởng nhất trong học máy hiện đại. Goodfellow nảy ra ý tưởng cơ bản — huấn luyện generator và discriminator đối kháng — được cho là trong một buổi tối, dù sự phát triển của bài báo liên quan công việc lý thuyết và thực nghiệm đáng kể. Bài báo hình thức hóa GAN như trò chơi minimax, chứng minh tại cân bằng generator học phân phối dữ liệu, và chứng minh kết quả trên nhiều tập dữ liệu. Điều khiến GAN cách mạng không chỉ là kết quả mà là sự chuyển paradigm: mô hình hóa sinh qua cạnh tranh thay vì tối đa hóa hợp lý hoặc tái tạo. Bài báo thừa nhận thách thức huấn luyện (bất ổn, sụp đổ mode) trong khi cho thấy tiềm năng của cách tiếp cận. Đọc hôm nay, người ta đánh giá cả sự rõ ràng của ý tưởng cốt lõi và sự tiên đoán về các thách thức sẽ chiếm thời gian của nhà nghiên cứu trong nhiều năm. GAN chứng tỏ rằng đôi khi cách tốt nhất để giải bài toán không phải tấn công trực tiếp (mô hình hóa mật độ tường minh) mà gián tiếp (học sinh qua phản hồi đối kháng).

**["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2016)](https://arxiv.org/abs/1511.06434)**  
*Tác giả*: Alec Radford, Luke Metz, Soumith Chintala  
Bài báo DCGAN khiến GAN thực tiễn bằng cách xác định hướng dẫn kiến trúc ổn định huấn luyện và cải thiện chất lượng mẫu. Các tác giả khám phá có hệ thống các lựa chọn thiết kế — tầng tích chập so với fully connected, vị trí batch normalization, hàm kích hoạt — tìm các tổ hợp hoạt động nhất quán. Hướng dẫn của họ: dùng strided convolution thay vì pooling, dùng batch norm trong cả hai mạng (trừ đầu ra generator và đầu vào discriminator), dùng ReLU trong generator trừ đầu ra (tanh), dùng LeakyReLU trong discriminator. Những điều này không được thúc đẩy lý thuyết mà được khám phá thực nghiệm qua thử nghiệm rộng rãi, chứng tỏ tiến bộ thực tiễn đôi khi đến từ kỹ thuật có hệ thống thay vì hiểu biết toán học. DCGAN cho thấy GAN có thể sinh ảnh chất lượng cao (khuôn mặt 64×64) và không gian ẩn đã học có cấu trúc có nghĩa — số học trong không gian ẩn (vectơ cho "người phụ nữ cười" trừ "người phụ nữ trung tính" cộng "người đàn ông trung tính") tạo "người đàn ông cười." Điều này chứng tỏ GAN học biểu diễn disentangled mã hóa thuộc tính ngữ nghĩa, khiến chúng hữu ích vượt ngoài sinh cho học biểu diễn.

**["Improved Techniques for Training GANs" (2016)](https://arxiv.org/abs/1606.03498)**  
*Tác giả*: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen  
Bài báo này xử lý bất ổn huấn luyện GAN qua nhiều kỹ thuật: feature matching (huấn luyện generator khớp thống kê đặc trưng trung gian của discriminator thay vì đánh lừa đầu ra cuối), minibatch discrimination (cho phép discriminator so sánh ví dụ trong batch để phát hiện thiếu đa dạng), historical averaging (phạt tham số lệch khỏi trung bình lịch sử), one-sided label smoothing (dùng 0.9 thay vì 1.0 cho nhãn thật để ngăn discriminator quá tự tin), và virtual batch normalization (chuẩn hóa dùng thống kê từ batch tham chiếu để giảm phương sai giữa các batch). Mỗi kỹ thuật xử lý một chế độ thất bại cụ thể: feature matching giảm bất ổn, minibatch discrimination chống sụp đổ mode, label smoothing ngăn bão hòa discriminator. Bài báo cũng giới thiệu Inception Score để định lượng chất lượng mẫu, cung cấp chỉ số tự động (dù không hoàn hảo) để đánh giá GAN. Công trình thiết lập rằng huấn luyện GAN thành công đòi hỏi nhiều mẹo bổ sung thay vì chỉ thuật toán cơ bản, cung cấp bộ công cụ đã trở thành thực hành chuẩn.

**["Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018)](https://arxiv.org/abs/1710.10196)**  
*Tác giả*: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (NVIDIA)  
Bài báo này giới thiệu huấn luyện progressive: bắt đầu với ảnh độ phân giải thấp (4×4) và dần thêm tầng cho generator và discriminator, tăng độ phân giải (8×8, 16×16, ..., lên đến 1024×1024). Cách tiếp cận này ổn định huấn luyện (dễ học phân phối độ phân giải thấp trước) và cho phép sinh ảnh độ phân giải rất cao trước đây bất khả. Bài báo cũng giới thiệu chỉ số đánh giá cải thiện và kỹ thuật huấn luyện. Khuôn mặt sinh ở độ phân giải 1024×1024 thực tế đến sốc, chứng tỏ năng lực của GAN cho sinh độ trung thực cao. Progressive growing đã ảnh hưởng công trình sau (StyleGAN xây trên nó) và chứng minh rằng chương trình huấn luyện — tăng dần độ khó tác vụ — áp dụng không chỉ cho dữ liệu (ví dụ dễ trước) mà còn cho kiến trúc (sinh đơn giản trước, phức tạp sau). Công trình cho thấy bất ổn huấn luyện GAN có thể được xử lý một phần qua thủ tục huấn luyện cẩn thận, không chỉ sửa đổi kiến trúc hoặc loss.

**["A Style-Based Generator Architecture for Generative Adversarial Networks" (2019)](https://arxiv.org/abs/1812.04948)**  
*Tác giả*: Tero Karras, Samuli Laine, Timo Aila (NVIDIA)  
StyleGAN thiết kế lại kiến trúc generator để cho phép kiểm soát tinh trên ảnh sinh. Thay vì đưa mã ẩn trực tiếp vào generator, StyleGAN ánh xạ nó qua mapping network sang không gian ẩn trung gian $$\mathcal{W}$$, rồi dùng điều này để điều khiển style ở các mức độ phân giải khác nhau qua adaptive instance normalization. Điều này cho phép kiểm soát đáng kinh ngạc: thay đổi style thô (tư thế, hình dạng khuôn mặt) độc lập với style tinh (texture tóc, lỗ chân lông da). Bài báo chứng minh chất lượng ảnh chưa từng có và giới thiệu công cụ để phân tích và cải thiện GAN (như chỉ số perceptual path length). StyleGAN sinh khuôn mặt không thể phân biệt với ảnh chụp thật, đạt cột mốc trong mô hình hóa sinh. Thành công của kiến trúc cho thấy thiết kế generator quan trọng vô cùng — không phải mọi cách ánh xạ nhiễu sang ảnh đều tốt ngang nhau. Tính chất disentanglement (khả năng điều khiển thuộc tính độc lập) khiến StyleGAN hữu ích cho chỉnh sửa ngữ nghĩa và chuyển phong cách, mở rộng GAN từ sinh thuần túy sang tổng hợp có kiểm soát.

## Bẫy thường gặp và Mẹo

Sụp đổ mode có lẽ là chế độ thất bại bực bội nhất trong huấn luyện GAN. Generator khám phá rằng nó có thể đánh lừa discriminator bằng cách chỉ sinh một vài kiểu đầu ra thay vì toàn bộ đa dạng dữ liệu. Với MNIST, điều này có thể nghĩa là chỉ sinh 1 và 7, bỏ qua các chữ số khác. Với khuôn mặt, chỉ sinh một số tư thế hoặc biểu cảm. Phát hiện đòi hỏi kiểm tra đa dạng mẫu, không chỉ chất lượng — sinh nhiều mẫu và xác minh chúng trải phân phối dữ liệu. Giải pháp gồm minibatch discrimination (cho phép discriminator thấy nhiều mẫu và phát hiện đồng nhất), unrolled optimization (cho phép generator dự đoán phản ứng discriminator), hoặc dùng hàm mất mát khác như Wasserstein GAN ít dễ sụp đổ mode hơn. Hiểu rằng sụp đổ mode xuất phát từ generator tìm tối ưu cục bộ trong trò chơi đối kháng giúp nhận ra khi nó xảy ra và thúc đẩy các giải pháp này.

Discriminator lấn át generator sớm trong huấn luyện phổ biến và phá hoại. Nếu discriminator trở nên quá tốt quá nhanh, nó gán xác suất gần 0 cho mọi đầu ra generator, cung cấp gradient biến mất cho generator không thể học. Điều này xảy ra khi discriminator quá lớn so với generator, learning rate quá cao cho discriminator, hoặc phân phối thật/giả dễ tách ban đầu (generator bắt đầu tệ). Giải pháp: huấn luyện discriminator ít thường xuyên hơn (mỗi $$k$$ cập nhật generator), dùng learning rate thấp hơn cho discriminator, thêm nhiễu vào đầu vào discriminator (làm mờ ranh giới thật/giả), hoặc dùng one-sided label smoothing (nhãn thật = 0.9 thay vì 1.0, giảm sự tự tin quá mức của discriminator). Theo dõi $$D(\mathbf{x}_{\text{real}})$$ và $$D(G(\mathbf{z}))$$ giúp: nếu thật luôn gần 1 và giả luôn gần 0, discriminator quá mạnh.

Dùng batch normalization trong discriminator có thể gây vấn đề khi batch size nhỏ vì thống kê batch trở nên không đáng tin. Với batch size 1, batch norm thất bại hoàn toàn. Giải pháp gồm dùng batch size lớn hơn (ít nhất 32–64), dùng layer normalization hoặc instance normalization thay vì batch norm, hoặc dùng virtual batch normalization (chuẩn hóa dùng thống kê từ batch tham chiếu cố định). Hiểu rằng chuẩn hóa của discriminator ảnh hưởng đặc trưng nó học giúp gỡ lỗi vấn đề huấn luyện liên quan đến batch size.

Đánh giá chất lượng GAN là thách thức vì ta không thể tính hợp lý. Inception Score đo cả chất lượng (mẫu nên được phân loại tự tin) và đa dạng (nên phủ mọi lớp) dùng bộ phân loại tiền huấn luyện, nhưng có hạn chế (không phát hiện ghi nhớ, thiên lệch về các lớp ImageNet). Fréchet Inception Distance (FID) so sánh thống kê của mẫu thật và mẫu sinh trong không gian đặc trưng, cung cấp chỉ số tốt hơn nhưng vẫn không hoàn hảo. Với công việc thực tiễn, kiểm tra bằng mắt vẫn quan trọng — sinh nhiều mẫu và kiểm tra thủ công chất lượng và đa dạng. Chỉ số định lượng bổ sung nhưng không thay thế đánh giá con người.

Một kỹ thuật mạnh cho huấn luyện ổn định là spectral normalization, ràng buộc hằng số Lipschitz của discriminator bằng cách chuẩn hóa ma trận trọng số theo spectral norm (giá trị singular lớn nhất). Điều này ngăn discriminator có gradient tùy ý lớn, ổn định động lực huấn luyện. Kỹ thuật thêm chi phí tính toán tối thiểu (tính spectral norm qua power iteration) trong khi cải thiện ổn định đáng kể. GAN hiện đại thường dùng spectral normalization trong discriminator như thực hành chuẩn, cho thấy hiểu biết lý thuyết về động lực huấn luyện (ràng buộc Lipschitz cải thiện ổn định) chuyển thành kỹ thuật thực tiễn.

## Điểm then chốt

Mạng Sinh Đối kháng học sinh dữ liệu thực tế bằng cách huấn luyện hai mạng đối kháng: generator tạo mẫu giả từ nhiễu ngẫu nhiên và discriminator phân biệt thật với giả. Mục tiêu đối kháng được công thức hóa như trò chơi minimax nơi generator tối thiểu hóa điều discriminator tối đa hóa, tạo động lực cạnh tranh đẩy cả hai mạng hướng tới năng lực cao hơn. Tại cân bằng, phân phối của generator khớp phân phối dữ liệu và discriminator không thể phân biệt thật/giả tốt hơn đoán ngẫu nhiên, dù đạt cân bằng này trong thực tiễn là thách thức. Huấn luyện luân phiên giữa cập nhật discriminator (dùng dữ liệu thật và giả của generator) và cập nhật generator (cố đánh lừa discriminator), đòi hỏi cân bằng cẩn thận để ngăn một mạng thống trị. Sụp đổ mode — generator tạo đa dạng hạn chế — vẫn là thách thức dai dẳng được xử lý qua lựa chọn kiến trúc, mục tiêu sửa đổi, và kỹ thuật huấn luyện. GAN xuất sắc ở sinh mẫu chất lượng cao, thực tế (thường vượt VAE) và học không gian ẩn với cấu trúc ngữ nghĩa cho phép nội suy và thao tác. Cách tiếp cận mô hình hóa mật độ ẩn cho phép sinh dữ liệu chiều cao phức tạp không cần công thức xác suất tường minh, dù với chi phí bất ổn huấn luyện và khó khăn đánh giá. Hiểu sâu GAN nghĩa là đánh giá cả sức mạnh sáng tạo trong sinh dữ liệu thực tế và động lực huấn luyện mong manh khiến chúng thách thức nhưng đáng để làm việc trong thực tiễn.

Khung GAN chứng tỏ rằng cạnh tranh có thể là tín hiệu học mạnh, một nguyên tắc đã ảnh hưởng deep learning xa hơn mô hình hóa sinh.
