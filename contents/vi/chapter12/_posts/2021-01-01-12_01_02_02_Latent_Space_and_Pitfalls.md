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

Triển khai autoencoder tích chập cho ảnh:

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

## 5. Các Khái niệm Liên quan

Autoencoder nối sâu với phân tích thành phần chính (principal component analysis — PCA), một kỹ thuật giảm chiều cổ điển. Autoencoder tuyến tính với mất mát MSE học chiếu dữ liệu lên không gian con trải bởi $$k$$ thành phần chính hàng đầu — chính xác điều PCA làm. Sự tương đương này tiết lộ rằng autoencoder tổng quát hóa PCA bằng cách cho phép hàm encoder và decoder phi tuyến. Nơi PCA tìm không gian con tuyến tính $$k$$ chiều tốt nhất, autoencoder tìm manifold phi tuyến $$k$$ chiều tốt nhất. Với dữ liệu có cấu trúc phi tuyến (như ảnh nơi biến thiên có nghĩa là xoay, scale, biến dạng — tất cả phi tuyến), autoencoder có thể nắm cấu trúc mà PCA bỏ lỡ. Hiểu kết nối này giúp đánh giá autoencoder như giảm chiều phi tuyến và thúc đẩy dùng chúng khi phương pháp tuyến tính thất bại.

Mối quan hệ với học biểu diễn và học chuyển giao rất sâu. Autoencoder huấn luyện trên tập dữ liệu không gán nhãn lớn học đặc trưng tổng quát thường chuyển giao tốt sang tác vụ giám sát. Ở kỷ nguyên trước ImageNet, tiền huấn luyện layer-wise tham lam dùng stacked autoencoder là then chốt để huấn luyện mạng sâu. Mỗi tầng được huấn luyện như autoencoder trên đặc trưng từ tầng trước, học dần biểu diễn phân cấp. Dù ReLU, batch normalization, và khởi tạo tốt hơn đã khiến tiền huấn luyện này ít cần thiết hơn cho học giám sát, ý tưởng cốt lõi — rằng học không giám sát trên dữ liệu không gán nhãn dồi dào có thể cung cấp khởi tạo hữu ích cho tác vụ giám sát với nhãn hạn chế — vẫn quan trọng và đã tiến hóa thành các cách tiếp cận học tự giám sát hiện đại.

Autoencoder nối với lý thuyết thông tin qua nguyên lý nút thắt thông tin (information bottleneck). Biểu diễn ẩn $$\mathbf{z}$$ nên nắm thông tin về $$\mathbf{x}$$ liên quan cho tái tạo trong khi loại bỏ chi tiết không liên quan. Lý thuyết thông tin định lượng điều này qua thông tin tương hỗ: tối đa hóa $$I(\mathbf{x}; \mathbf{z})$$ (thông tin về đầu vào bảo toàn trong ẩn) trong khi tối thiểu hóa $$I(\mathbf{z}; \text{noise})$$ hoặc ràng buộc $$I(\mathbf{z})$$ (độ phức tạp của biểu diễn ẩn). Variational autoencoder làm tường minh kết nối này bằng cách đưa vào hạng tử phân kỳ KL chính quy hóa phân phối ẩn. Hiểu autoencoder qua lý thuyết thông tin cung cấp cách có nguyên tắc để nghĩ về điều gì tạo nên biểu diễn tốt.

Sự tiến hóa từ autoencoder sang variational autoencoder (VAE) và mạng sinh đối kháng (GAN) cho thấy cách xử lý hạn chế thúc đẩy đổi mới. Autoencoder chuẩn học tái tạo nhưng không mô hình hóa tường minh phân phối dữ liệu, hạn chế khả năng sinh. VAE thêm khung xác suất, coi encoder như tính phân phối trên mã ẩn và thêm hạng tử chính quy hóa định hình phân phối này để hành xử tốt (thường Gaussian chuẩn). Điều này cho phép lấy mẫu và nội suy có nguyên tắc. GAN lấy cách tiếp cận hoàn toàn khác, dùng huấn luyện đối kháng thay vì tái tạo, thường sinh mẫu sắc nét, thực tế hơn. Mỗi cách tiếp cận có điểm mạnh: autoencoder đơn giản và ổn định để huấn luyện, VAE cung cấp khung xác suất có nguyên tắc, GAN sinh mẫu chất lượng cao nhất. Hiểu autoencoder cung cấp nền tảng để đánh giá các mô hình sinh tinh vi hơn này.

## 6. Các Bài báo Nền tảng

**["Reducing the Dimensionality of Data with Neural Networks" (2006)](https://www.science.org/doi/10.1126/science.1127647)**  
*Tác giả*: Geoffrey E. Hinton, Ruslan Salakhutdinov  
Bài báo Science mang tính bước ngoặt này chứng tỏ autoencoder sâu có thể học giảm chiều tốt hơn nhiều so với PCA hoặc autoencoder nông. Hinton và Salakhutdinov giới thiệu tiền huấn luyện layer-wise tham lam: huấn luyện mỗi tầng như autoencoder (thực ra restricted Boltzmann machine trong trường hợp của họ) trên đặc trưng từ tầng trước, xếp chồng để xây dựng biểu diễn sâu. Tiền huấn luyện này theo sau bởi fine-tuning cho phép huấn luyện mạng sâu hơn nhiều so với trước đây (đây là trước ReLU và kỹ thuật khởi tạo hiện đại). Bài báo cho thấy kết quả ấn tượng về trực quan hóa dữ liệu chiều cao và nén ảnh, chứng tỏ deep learning có thể học biểu diễn phân cấp qua học không giám sát. Công trình này có ảnh hưởng trong sự tái sinh deep learning cuối những năm 2000, chỉ ra rằng độ sâu quan trọng và tiền huấn luyện không giám sát có thể mở khóa nó. Dù học giám sát hiện đại không đòi hỏi tiền huấn luyện autoencoder (nhờ ReLU, batch norm, và khởi tạo tốt hơn), các hiểu biết về học biểu diễn phân cấp và trích xuất đặc trưng không giám sát vẫn quan trọng.

**["Extracting and Composing Robust Features with Denoising Autoencoders" (2008)](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)**  
*Tác giả*: Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol  
Bài báo này giới thiệu autoencoder khử nhiễu và cung cấp biện minh lý thuyết vì sao chúng học biểu diễn tốt hơn autoencoder vanilla. Hiểu biết then chốt là bằng cách làm hỏng đầu vào và huấn luyện tái tạo bản gốc sạch, ta buộc mạng học cấu trúc manifold dữ liệu thay vì chỉ ghi nhớ ví dụ. Việc làm hỏng đóng vai trò chính quy hóa, ngăn mạng học hàm đồng nhất ngay cả với chiều ẩn lớn. Bài báo chỉ ra cả về lý thuyết lẫn thực nghiệm rằng autoencoder khử nhiễu học biểu diễn robust với hỏng đầu vào, khiến đặc trưng hữu ích hơn cho tác vụ downstream như phân loại. Khung khử nhiễu đã ảnh hưởng nhiều phương pháp sau — masked language modeling trong BERT có thể được xem như khử nhiễu, và nhiều cách tiếp cận tự giám sát làm hỏng đầu vào và huấn luyện mạng dự đoán hoặc tái tạo bản gốc. Bài báo này thiết lập hỏng-và-tái tạo như mô hình học không giám sát mạnh.

**["Contractive Auto-Encoders: Explicit Invariance During Feature Extraction" (2011)](http://www.iro.umontreal.ca/~lisa/pointeurs/ICML2011_explicit_invariance.pdf)**  
*Tác giả*: Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, Yoshua Bengio  
Bài báo này đề xuất contractive autoencoder, thêm phạt lên chuẩn Frobenius của Jacobian của encoder. Mục tiêu trở thành:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \|J_f(\mathbf{x})\|_F^2$$

trong đó $$J_f(\mathbf{x}) = \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}$$ là Jacobian của encoder. Phạt này khuyến khích encoder không nhạy với biến thiên nhỏ của đầu vào — biểu diễn ẩn nên thay đổi chậm khi ta nhiễu loạn đầu vào nhẹ. Trực giác là đặc trưng có nghĩa nên robust với thay đổi đầu vào nhỏ (như dịch nhẹ hoặc nhiễu). Bài báo chỉ ra contractive autoencoder học biểu diễn với tính bất biến tốt hơn autoencoder vanilla hoặc khử nhiễu, dù với chi phí tính toán Jacobian và chính quy hóa nó. Công trình làm sâu hiểu biết lý thuyết về điều gì tạo nên biểu diễn tốt và cung cấp công cụ để khuyến khích các tính chất mong muốn cụ thể (bất biến, thưa, v.v.) qua chính quy hóa.

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Tác giả*: Diederik P. Kingma, Max Welling  
Dù giới thiệu VAE (bao phủ ở chương sau), bài báo này thay đổi căn bản cách ta nghĩ về autoencoder bằng cách cung cấp khung xác suất. Các tác giả chỉ ra autoencoder có thể được xem như học tối đa hóa cận dưới của hợp lý dữ liệu, nối chúng với mô hình hóa xác suất có nguyên tắc. Khung biến phân xử lý hạn chế của autoencoder chuẩn: không gian ẩn có thể có lỗ hổng nơi không có ví dụ huấn luyện nào ánh xạ, khiến lấy mẫu không đáng tin. VAE chính quy hóa không gian ẩn để theo phân phối đã biết (thường Gaussian chuẩn), đảm bảo ta có thể lấy mẫu ở bất kỳ đâu và giải mã ra đầu ra thực tế. Reparameterization trick của bài báo — lấy mẫu qua phép toán khả vi — cho phép huấn luyện qua backpropagation. VAE trở nên ảnh hưởng to lớn, sinh ra nhiều biến thể và ứng dụng trong mô hình hóa sinh, học bán giám sát, và học biểu diễn. Hiểu autoencoder vanilla là điều kiện tiên quyết để đánh giá sự tinh vi xác suất của VAE và các đảm bảo bổ sung mà nó cung cấp.

**["Adversarial Autoencoders" (2016)](https://arxiv.org/abs/1511.05644)**  
*Tác giả*: Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey  
Bài báo này kết hợp autoencoder với huấn luyện đối kháng, dùng discriminator để buộc phân phối mã ẩn khớp prior (như Gaussian chuẩn) thay vì dùng phạt phân kỳ KL (như VAE). Huấn luyện đối kháng khiến không gian ẩn khớp prior chặt hơn phạt KL của VAE trong khi duy trì mục tiêu tái tạo của autoencoder. Bài báo chứng minh cách tiếp cận lai này có thể sinh mẫu chất lượng cao trong khi linh hoạt hơn VAE trong lựa chọn prior ẩn (không giới hạn ở Gaussian phân tích thừa). Adversarial autoencoder cho thấy ý tưởng từ các khung khác nhau (autoencoder, VAE, GAN) có thể được kết hợp, dẫn đến mô hình với điểm mạnh bổ sung. Công trình minh họa sự giao phấn ý tưởng sản sinh trong deep learning — kỹ thuật phát triển cho một mục đích (huấn luyện đối kháng cho GAN) chứng tỏ hữu ích khi kết hợp với khung khác (autoencoder).

## Bẫy thường gặp và Mẹo

Chế độ thất bại phổ biến nhất trong autoencoder là dùng chiều ẩn quá lớn, làm suy yếu mục tiêu nén. Với chiều ẩn tiệm cận chiều đầu vào, mạng có thể học truyền thông tin gần như không đổi, không khám phá cấu trúc có nghĩa. Triệu chứng là tái tạo hoàn hảo nhưng mã ẩn vô dụng — chúng overcomplete và dư thừa. Giải pháp là giảm mạnh chiều ẩn hoặc thêm ràng buộc khác (thưa, khử nhiễu, phạt contractive). Heuristic hữu ích: bắt đầu với chiều ẩn nhỏ hơn chiều đầu vào 10–20×, rồi thử nghiệm. Với MNIST (784 chiều), thử 32–64 chiều ẩn. Với ảnh độ phân giải cao hơn, hệ số nén có thể lớn hơn.

Quên chuẩn hóa đầu vào gây bất ổn huấn luyện và tái tạo kém. Nếu giá trị pixel trải [0, 255], lỗi tái tạo lớn hơn hàng trăm lần so với giá trị chuẩn hóa [0,1], dẫn đến gradient khổng lồ và loss bùng nổ. Luôn chuẩn hóa đầu vào về [0,1] (chia cho 255) hoặc chuẩn hóa về mean 0, std 1. Khớp kích hoạt đầu ra decoder với lược đồ chuẩn hóa: sigmoid cho [0,1], tanh cho [-1,1], linear cho chuẩn hóa. Điều này đảm bảo decoder thực sự có thể sinh giá trị trong khoảng đúng.

Dùng mất mát MSE cho ảnh trông trực quan nhưng có vấn đề tinh tế: MSE gán trọng số ngang nhau cho mọi pixel, nhưng nhận thức con người không hoạt động như vậy. Một pixel lệch có thể gây MSE lớn dù tái tạo trông hoàn hảo với người. Ngược lại, tái tạo mờ (trung bình hóa pixel) có thể có MSE thấp trong khi trông kém về cảm nhận. Với ứng dụng nơi chất lượng cảm nhận quan trọng, cân nhắc mất mát cảm nhận — đo khoảng cách trong không gian đặc trưng của mạng tiền huấn luyện như VGG thay vì không gian pixel. Đặc trưng từ tầng sâu nắm cấu trúc mức cao (hình dạng, đối tượng) tương quan tốt hơn với nhận thức con người so với khoảng cách theo pixel.

Một mẹo mạnh cho không gian ẩn tốt hơn là thêm chính quy hóa tường minh vượt ngoài chỉ giảm chiều. Autoencoder thưa thêm phạt L1 lên kích hoạt ẩn, khuyến khích hầu hết chiều bằng zero hầu hết thời gian. Điều này buộc chuyên môn hóa — mỗi chiều ẩn nắm một khía cạnh biến thiên cụ thể. Variational autoencoder thêm phân kỳ KL tới prior, đảm bảo không gian ẩn mượt, liên tục. Contractive autoencoder phạt Jacobian của encoder, khuyến khích bất biến với nhiễu loạn đầu vào. Hiểu các tùy chọn chính quy hóa này cho phép tùy chỉnh autoencoder theo desiderata cụ thể — thưa cho diễn giải, mượt cho nội suy, robust cho tác vụ downstream.

Khi dùng autoencoder cho tiền huấn luyện (ít phổ biến hơn bây giờ nhưng vẫn hữu ích trong chế độ dữ liệu thấp), quyết định then chốt là có fine-tune encoder, decoder, hay cả hai. Với phân loại, thường đóng băng decoder (ta chỉ cần đặc trưng encoder) và thêm đầu phân loại trên biểu diễn ẩn, fine-tune chỉ đầu này và tùy chọn encoder. Với tác vụ sinh, ta có thể đóng băng encoder (nếu có mã ẩn tốt) và chỉ fine-tune decoder. Với thích ứng miền, fine-tune cả hai thường hoạt động tốt nhất. Lựa chọn phụ thuộc vào việc đặc trưng encoder, sinh decoder, hay cả hai cần thích ứng chuyên tác vụ.

## Điểm then chốt

Autoencoder học biểu diễn dữ liệu hiệu quả bằng cách huấn luyện tái tạo đầu vào qua nút thắt chiều thấp hơn, buộc nén dữ liệu chiều cao thành mã ẩn gọn nắm cấu trúc thiết yếu. Encoder ánh xạ đầu vào sang biểu diễn ẩn trong khi decoder tái tạo đầu vào từ mã ẩn, cả hai được huấn luyện chung dùng mất mát tái tạo (MSE cho dữ liệu liên tục, cross-entropy cho nhị phân). Chiều nút thắt điều khiển đánh đổi nén–độ trung thực, với chiều ẩn nhỏ hơn buộc nén mạnh hơn và có thể học đặc trưng có nghĩa hơn. Autoencoder khử nhiễu làm hỏng đầu vào trước khi encode nhưng huấn luyện tái tạo bản gốc sạch, học đặc trưng robust nắm cấu trúc dữ liệu thay vì ghi nhớ ví dụ. Không gian ẩn trong autoencoder huấn luyện tốt có cấu trúc ngữ nghĩa, với các điểm lân cận tương ứng đầu vào tương tự và nội suy mượt cho phép biến hình giữa các ví dụ. Autoencoder phục vụ nhiều mục đích: giảm chiều cho trực quan hóa hoặc tác vụ downstream, học đặc trưng cho học chuyển giao, khử nhiễu để loại bỏ hỏng, và như nền tảng cho mô hình sinh tinh vi hơn. Hiểu autoencoder cung cấp nền tảng thiết yếu cho variational autoencoder và các cách tiếp cận sinh khác trong khi chứng minh các nguyên tắc cốt lõi của học biểu diễn không giám sát thấm đẫm các phương pháp tự giám sát hiện đại.

Khung autoencoder minh họa một chủ đề lặp lại trong học máy: học qua tái tạo, nơi ta buộc mô hình khám phá cấu trúc bằng cách yêu cầu chúng tái tạo dữ liệu qua ràng buộc hoặc phép biến đổi khiến nghiệm tầm thường trở nên bất khả.
