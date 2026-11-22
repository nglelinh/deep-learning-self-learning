---
layout: post
title: 12-01-02-02 Không gian Ẩn, Bài báo và Bẫy thường gặp
chapter: '12'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter12
---

```python
# Minh họa nội suy không gian ẩn
print("\n" + "="*70)
print("Nội suy Không gian Ẩn")
print("="*70)

with torch.no_grad():
    # Lấy hai chữ số khác nhau
    idx1, idx2 = 0, 5  # Nội suy giữa ảnh kiểm tra thứ nhất và thứ sáu
    
    img1 = test_data[idx1:idx1+1].view(1, -1)
    img2 = test_data[idx2:idx2+1].view(1, -1)
    
    # Encode cả hai
    z1 = model_ae.encode(img1)
    z2 = model_ae.encode(img2)
    
    print(f"Nội suy giữa hai ảnh kiểm tra:")
    print(f"  Mean mã ẩn ảnh 1: {z1.mean().item():.3f}, std: {z1.std().item():.3f}")
    print(f"  Mean mã ẩn ảnh 2: {z2.mean().item():.3f}, std: {z2.std().item():.3f}")
    
    # Nội suy trong không gian ẩn
    n_steps = 7
    print(f"\nSinh {n_steps} ảnh trung gian bằng nội suy:")
    
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        z_interp = (1-t) * z1 + t * z2
        img_interp = model_ae.decode(z_interp)
        
        print(f"  Bước {i} (t={t:.2f}): Shape ảnh giải mã {img_interp.shape}")
    
    print("\nTrong hiển thị trực quan, bạn sẽ thấy biến hình mượt từ chữ số này sang chữ số kia.")
    print("Điều này chứng tỏ không gian ẩn đã học cấu trúc có nghĩa!")

# Minh họa giảm chiều trực quan hóa (không gian ẩn 2D)
print("\n" + "="*70)
print("Huấn luyện Autoencoder 2D để Trực quan hóa")
print("="*70)

class TinyAutoencoder(nn.Module):
    """Autoencoder với không gian ẩn 2D để trực quan hóa"""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Latent 2D để vẽ!
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model_2d = TinyAutoencoder()
optimizer_2d = optim.Adam(model_2d.parameters(), lr=0.001)

print("Huấn luyện autoencoder 2D (nén cực đoan: 784 → 2)...")
model_2d.train()

for epoch in range(5):
    for data, _ in train_loader:
        data = data.view(data.size(0), -1)
        
        recon, latent = model_2d(data)
        loss = F.mse_loss(recon, data)
        
        optimizer_2d.zero_grad()
        loss.backward()
        optimizer_2d.step()

# Trực quan hóa không gian ẩn
print("\nMã hóa tập kiểm tra vào không gian ẩn 2D...")
model_2d.eval()

latent_codes = []
labels_all = []

with torch.no_grad():
    for data, labels in test_loader:
        data = data.view(data.size(0), -1)
        _, z = model_2d(data)
        latent_codes.append(z.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

latent_codes = np.concatenate(latent_codes)
labels_all = np.concatenate(labels_all)

print(f"Shape tọa độ không gian ẩn: {latent_codes.shape}")  # (10000, 2)
print(f"\nTrong scatter plot, các chữ số khác nhau sẽ cụm trong không gian 2D.")
print("Điều này chứng tỏ autoencoder học biểu diễn có nghĩa!")
print("Chữ số 0 ở một vùng, 1 ở vùng khác, v.v.")
```

Dưới đây là phần cài đặt autoencoder tích chập (*convolutional autoencoder*) dành cho dữ liệu ảnh:

```python
class ConvAutoencoder(nn.Module):
    """
    Autoencoder tích chập cho ảnh.
    
    Dùng tầng conv trong encoder (downsample không gian qua stride)
    và transposed convolution trong decoder (upsample).
    Hiệu quả tham số hơn nhiều so với fully connected cho ảnh.
    """
    
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder: downsample bằng tầng conv
        # 28×28×1 → 14×14×32 → 7×7×64 → flatten → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14→7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        # Decoder: upsample bằng transposed conv
        # latent_dim → 7×7×64 → 14×14×32 → 28×28×1
        self.decoder_linear = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14→28
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, 64, 7, 7)  # Reshape thành feature map
        return self.decoder_conv(x)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

print("\n" + "="*70)
print("Autoencoder Tích chập cho Ảnh")
print("="*70)

conv_ae = ConvAutoencoder(latent_dim=64)
total_params = sum(p.numel() for p in conv_ae.parameters())
print(f"Tổng tham số: {total_params:,}")

# Huấn luyện nhanh
optimizer_conv = optim.Adam(conv_ae.parameters(), lr=0.001)

print("Huấn luyện autoencoder tích chập...")
conv_ae.train()

for epoch in range(3):
    for data, _ in train_loader:
        # Giữ cấu trúc 2D cho tầng conv
        recon, latent = conv_ae(data)
        loss = F.mse_loss(recon, data)
        
        optimizer_conv.zero_grad()
        loss.backward()
        optimizer_conv.step()

print("Autoencoder tích chập đã huấn luyện!")
print("Lợi ích: Ít tham số hơn, tái tạo ảnh tốt hơn")
print("Cấu trúc conv cung cấp inductive bias cho dữ liệu không gian")
```

## 5. Các khái niệm liên quan

Autoencoder có liên hệ trực tiếp với phân tích thành phần chính (PCA), một phương pháp giảm chiều tuyến tính kinh điển. Một autoencoder chỉ gồm các tầng tuyến tính tối ưu theo MSE sẽ học phép chiếu dữ liệu lên đúng không gian con sinh bởi $$k$$ thành phần chính đầu tiên như PCA. Nhờ bổ sung các hàm kích hoạt phi tuyến, autoencoder tổng quát hóa PCA để tìm đa tạp phi tuyến $$k$$ chiều phù hợp nhất với dữ liệu — rất cần thiết với dữ liệu ảnh nơi các phép biến đổi như xoay, co giãn hay biến dạng đều mang tính phi tuyến.

Autoencoder cũng gắn chặt với học biểu diễn và học chuyển giao. Trước thời kỳ ImageNet, kỹ thuật tiền huấn luyện từng tầng bằng *stacked autoencoder* là phương pháp quan trọng để khởi tạo trọng số cho mạng sâu. Dù ReLU, batch normalization và kỹ thuật khởi tạo hiện đại đã giảm nhu cầu tiền huấn luyện kiểu cũ trong học giám sát, nguyên lý cốt lõi — tận dụng dữ liệu không nhãn để học đặc trưng nền tảng trước khi tinh chỉnh — vẫn là trụ cột của học tự giám sát ngày nay.

Từ góc độ lý thuyết thông tin, autoencoder gắn với nguyên lý nút thắt thông tin (*information bottleneck*): biểu diễn ẩn $$\mathbf{z}$$ cần bảo toàn lượng thông tin hữu ích về $$\mathbf{x}$$ cho nhiệm vụ tái tạo — tối đa hóa thông tin tương hỗ $$I(\mathbf{x}; \mathbf{z})$$ — đồng thời loại bỏ thông tin nhiễu $$I(\mathbf{z}; \text{noise})$$ hoặc giới hạn độ phức tạp $$H(\mathbf{z})$$ của mã ẩn. Variational Autoencoder diễn đạt ràng buộc này tường minh hơn thông qua thành phần phân kỳ KL trên phân phối ẩn.

Nhìn rộng hơn, tiến trình từ autoencoder sang VAE và GAN cho thấy cách mỗi bước mở rộng giải quyết một giới hạn cụ thể: autoencoder chuẩn đơn giản và dễ huấn luyện nhưng không gian ẩn có khoảng trống; VAE bổ sung khung xác suất và chính quy hóa phân phối ẩn để lấy mẫu và nội suy ổn định; còn GAN thay mục tiêu tái tạo bằng huấn luyện đối kháng để tạo ảnh sắc nét hơn.

## 6. Các bài báo nền tảng

**["Reducing the Dimensionality of Data with Neural Networks" (2006)](https://www.science.org/doi/10.1126/science.1127647)**  
*Tác giả*: Geoffrey E. Hinton, Ruslan Salakhutdinov  
Bài báo công bố trên tạp chí *Science* chứng minh autoencoder sâu có thể giảm chiều tốt hơn đáng kể so với PCA hoặc mạng nông. Nhóm tác giả đề xuất chiến lược tiền huấn luyện từng tầng (sử dụng máy Boltzmann hạn chế xếp chồng) rồi tinh chỉnh toàn mạng, qua đó mở khóa khả năng huấn luyện mạng sâu trước khi có ReLU và các phương pháp khởi tạo hiện đại, đồng thời khẳng định vai trò của học biểu diễn phân cấp không giám sát.

**["Extracting and Composing Robust Features with Denoising Autoencoders" (2008)](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)**  
*Tác giả*: Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol  
Bài báo giới thiệu autoencoder khử nhiễu và giải thích cơ sở lý thuyết vì sao làm nhiễu đầu vào rồi tái tạo bản sạch giúp mạng học cấu trúc đa tạp dữ liệu thay vì ghi nhớ từng mẫu. Cơ chế này đóng vai trò chính quy hóa ngay cả khi số chiều ẩn lớn, tạo ra đặc trưng bền vững hơn cho các bài toán phân loại sau đó. Mô hình "làm khuyết rồi tái tạo" này sau đó ảnh hưởng trực tiếp tới masked language modeling trong BERT và nhiều phương pháp tự giám sát khác.

**["Contractive Auto-Encoders: Explicit Invariance During Feature Extraction" (2011)](http://www.iro.umontreal.ca/~lisa/pointeurs/ICML2011_explicit_invariance.pdf)**  
*Tác giả*: Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, Yoshua Bengio  
Bài báo đề xuất *contractive autoencoder* bằng cách bổ sung thành phần phạt chuẩn Frobenius của ma trận Jacobian của bộ mã hóa:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \|J_f(\mathbf{x})\|_F^2$$

trong đó $$J_f(\mathbf{x}) = \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}$$. Thành phần phạt này khuyến khích biểu diễn ẩn ít thay đổi khi đầu vào chịu nhiễu loạn nhỏ, giúp tăng tính bất biến cục bộ của đặc trưng dù phải đánh đổi thêm chi phí tính toán Jacobian.

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Tác giả*: Diederik P. Kingma, Max Welling  
Bài báo xây dựng khung xác suất biến phân cho autoencoder (VAE), coi bộ mã hóa là hàm dự đoán phân phối trên biến ẩn và tối ưu cận dưới log-hợp lý. Bằng cách chính quy hóa không gian ẩn gần phân phối tiên nghiệm (thường là chuẩn tắc) và dùng thủ thuật tái tham số hóa để lan truyền ngược qua phép lấy mẫu, VAE khắc phục tình trạng khoảng trống trong không gian ẩn của autoencoder chuẩn.

**["Adversarial Autoencoders" (2016)](https://arxiv.org/abs/1511.05644)**  
*Tác giả*: Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey  
Bài báo kết hợp autoencoder với huấn luyện đối kháng, sử dụng một bộ phân biệt để ép phân phối mã ẩn khớp với phân phối tiên nghiệm mong muốn thay cho thành phần phạt KL của VAE. Cách tiếp cận lai này giữ được mục tiêu tái tạo của autoencoder nhưng linh hoạt hơn trong việc chọn phân phối tiên nghiệm (không bị giới hạn ở phân phối chuẩn độc lập theo từng chiều).

## Bẫy thường gặp và Mẹo

Lỗi thường gặp nhất khi xây dựng autoencoder là đặt số chiều ẩn quá lớn, làm mất tác dụng nén của nút thắt. Khi số chiều ẩn xấp xỉ số chiều đầu vào, mạng dễ học sao chép gần như nguyên vẹn dữ liệu mà không rút trích được cấu trúc hữu ích. Cách xử lý là thu nhỏ mạnh chiều ẩn hoặc bổ sung ràng buộc (độ thưa, khử nhiễu, phạt contractive). Quy tắc khởi đầu hợp lý là nén khoảng 10–20 lần — ví dụ 32–64 chiều ẩn cho MNIST (784 chiều) — rồi điều chỉnh theo thực nghiệm.

Quên chuẩn hóa dữ liệu đầu vào cũng dễ gây bất ổn huấn luyện. Nếu giá trị điểm ảnh giữ nguyên trong [0, 255], sai số tái tạo và gradient sẽ lớn hơn nhiều so với dải [0, 1]. Cần chuẩn hóa đầu vào về [0, 1] hoặc trung bình 0, độ lệch chuẩn 1, đồng thời chọn hàm kích hoạt đầu ra của bộ giải mã cho khớp: `sigmoid` cho [0, 1], `tanh` cho [-1, 1] và tuyến tính cho dữ liệu đã chuẩn hóa z-score.

Tối ưu trực tiếp theo MSE trên ảnh dễ tạo ra kết quả tái tạo bị mờ vì MSE cân bằng sai lệch trên mọi điểm ảnh thay vì ưu tiên biên cạnh và cấu trúc như thị giác người. Khi yêu cầu cao về chất lượng thị giác, nên cân nhắc dùng hàm mất mát cảm nhận (*perceptual loss*) tính khoảng cách trên bản đồ đặc trưng của mạng tiền huấn luyện như VGG.

Ngoài nút thắt chiều, việc bổ sung chính quy hóa tường minh giúp định hình không gian ẩn theo mục tiêu thiết kế cụ thể: phạt L1 (autoencoder thưa) tăng tính diễn giải, phạt KL (VAE) giúp không gian ẩn mượt để nội suy, còn phạt Jacobian (contractive autoencoder) tăng tính bền vững trước nhiễu loạn nhỏ.

Khi tận dụng autoencoder làm bước tiền huấn luyện trong bài toán ít dữ liệu, cần quyết định rõ phần nào nên tinh chỉnh: với bài toán phân loại, thường chỉ giữ bộ mã hóa và gắn thêm tầng phân loại (tinh chỉnh tầng mới và có thể cả bộ mã hóa); với bài toán sinh, có thể cố định bộ mã hóa và tinh chỉnh bộ giải mã; còn khi thích ứng miền dữ liệu mới, tinh chỉnh cả hai thường hiệu quả hơn.

## Điểm then chốt

Autoencoder học biểu diễn dữ liệu gọn bằng cách tái tạo đầu vào qua một nút thắt thông tin. Bộ mã hóa nén dữ liệu nhiều chiều thành mã ẩn, bộ giải mã khôi phục đầu vào từ mã ẩn đó, và cả hai được huấn luyện đồng thời bằng hàm mất mát tái tạo (MSE cho dữ liệu liên tục, binary cross-entropy cho dữ liệu nhị phân). Chiều nút thắt quyết định mức độ nén và chất lượng tái tạo, trong khi các biến thể như autoencoder khử nhiễu, autoencoder thưa và contractive autoencoder bổ sung ràng buộc giúp đặc trưng bền vững và có cấu trúc hơn. Không gian ẩn học được phục vụ hiệu quả cho giảm chiều, trực quan hóa, khử nhiễu và học đặc trưng không giám sát, đồng thời là bước đệm trực tiếp để hiểu Variational Autoencoder và các phương pháp học tự giám sát hiện đại.

## Video tham khảo

- **Xem trước:** [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, tiếng Anh. Giải thích vì sao một mã ẩn chỉ tối ưu tái tạo vẫn có thể không phù hợp để lấy mẫu sinh dữ liệu mới.
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — Arxiv Insights, tiếng Anh. Cách tiếp cận biến phân khắc phục hạn chế trên, nội dung trọng tâm của chương tiếp theo.
- [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — AI Coffee Break with Letitia, tiếng Anh. Ví dụ thực tế về một không gian ẩn hiện đại được dùng trực tiếp cho quá trình sinh mẫu.
