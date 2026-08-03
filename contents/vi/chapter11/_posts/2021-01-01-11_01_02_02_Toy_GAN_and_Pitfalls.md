---
layout: post
title: 11-01-02-02 GAN Toy, Bài báo và Bẫy thường gặp
chapter: '11'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter11
---

# 2. GAN đơn giản để so sánh
print("\n2. Huấn luyện GAN (Mật độ Ẩn)")
print("-" * 70)

class ToyGenerator(nn.Module):
    """Generator đơn giản cho dữ liệu 2D"""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Xuất điểm 2D
        )
    
    def forward(self, z):
        return self.net(z)

class ToyDiscriminator(nn.Module):
    """Discriminator đơn giản cho dữ liệu 2D"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

gen = ToyGenerator(latent_dim=2)
disc = ToyDiscriminator()

gen_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

print("Huấn luyện GAN trên hỗn hợp Gaussian...")

for epoch in range(1000):
    # Huấn luyện discriminator
    for _ in range(1):  # k bước discriminator mỗi bước generator
        disc.zero_grad()
        
        # Dữ liệu thật
        batch_real = data_tensor[torch.randint(len(data_tensor), (128,))]
        labels_real = torch.ones(128, 1)
        output_real = disc(batch_real)
        loss_d_real = criterion(output_real, labels_real)
        
        # Dữ liệu giả
        z = torch.randn(128, 2)
        fake = gen(z)
        labels_fake = torch.zeros(128, 1)
        output_fake = disc(fake.detach())
        loss_d_fake = criterion(output_fake, labels_fake)
        
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        disc_optimizer.step()
    
    # Huấn luyện generator
    gen.zero_grad()
    z = torch.randn(128, 2)
    fake = gen(z)
    output = disc(fake)
    labels_real_for_g = torch.ones(128, 1)
    loss_g = criterion(output, labels_real_for_g)
    
    loss_g.backward()
    gen_optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: D_loss = {loss_d.item():.4f}, "
              f"G_loss = {loss_g.item():.4f}, "
              f"D(real) = {output_real.mean():.3f}, "
              f"D(fake) = {output_fake.mean():.3f}")

# Sinh mẫu
gen.eval()
with torch.no_grad():
    z_sample = torch.randn(1000, 2)
    samples_gan = gen(z_sample)
    
print(f"\nGAN đã sinh {len(samples_gan)} mẫu")
print(f"Kiểm tra phủ mode (mẫu nên cụm quanh 8 tâm)...")

# Kiểm tra GAN có phủ mọi mode không (phát hiện sụp đổ mode)
# Với mỗi tâm thật, đếm mẫu sinh lân cận
for i, center in enumerate(true_centers):
    distances = torch.norm(samples_gan - torch.FloatTensor(center), dim=1)
    nearby = (distances < 0.5).sum().item()
    print(f"  Mode {i} (tâm {center.round(2)}): {nearby} mẫu lân cận")

if all((torch.norm(samples_gan - torch.FloatTensor(c), dim=1) < 0.5).sum() > 50 
       for c in true_centers):
    print("✓ GAN phủ thành công mọi mode!")
else:
    print("✗ Phát hiện sụp đổ mode — một số mode có ít/không có mẫu")

print("\n" + "="*70)
print("So sánh Mô hình hóa Sinh")
print("="*70)
print("\nMô hình Tự hồi quy:")
print("  + Hợp lý chính xác tính được")
print("  + Huấn luyện ổn định")
print("  - Sinh tuần tự (chậm)")
print("  - Giả định thứ tự mạnh")

print("\nGAN:")
print("  + Sinh song song nhanh")
print("  + Thường chất lượng mẫu cao")
print("  - Không có hợp lý tường minh")
print("  - Huấn luyện có thể bất ổn")
print("  - Rủi ro sụp đổ mode")

print("\nVAE (chương sau):")
print("  + Không gian ẩn tường minh")
print("  + Huấn luyện ổn định")
print("  + Sinh nhanh")
print("  - Mẫu đôi khi mờ")
```

Minh họa đánh giá dựa trên hợp lý:

```python
print("\n" + "="*70)
print("Đánh giá Chất lượng Mô hình Sinh")
print("="*70)

# Với mô hình tự hồi quy, ta có thể tính hợp lý chính xác
ar_model.eval()
with torch.no_grad():
    # Tập kiểm tra (dữ liệu giữ lại từ cùng phân phối)
    data_test, _ = generate_mixture_data(n_samples=1000)
    data_test_tensor = torch.FloatTensor(data_test)
    
    # Tính log-hợp lý trên tập kiểm tra
    test_log_probs = ar_model.log_prob(data_test_tensor)
    avg_test_ll = test_log_probs.mean().item()
    
    print(f"Log-hợp lý Kiểm tra Mô hình Tự hồi quy: {avg_test_ll:.4f}")
    print(f"Cao hơn thì tốt hơn — mô hình gán xác suất cao cho dữ liệu kiểm tra")
    
    # Sinh và đánh giá (mẫu nên có hợp lý tương tự dữ liệu thật)
    samples_ar_eval = ar_model.sample(1000)
    samples_log_probs = ar_model.log_prob(samples_ar_eval)
    avg_sample_ll = samples_log_probs.mean().item()
    
    print(f"Log-hợp lý Mẫu Sinh: {avg_sample_ll:.4f}")
    print(f"Nên tương tự LL kiểm tra nếu mô hình tốt")
    
    diff = abs(avg_test_ll - avg_sample_ll)
    if diff < 0.5:
        print(f"✓ Chênh lệch nhỏ ({diff:.4f}) chỉ ra mô hình tốt!")
    else:
        print(f"✗ Chênh lệch lớn ({diff:.4f}) chỉ ra vấn đề")

# Với GAN, ta không tính được hợp lý, nên dùng chỉ số proxy
print("\nĐánh giá GAN (không có hợp lý tường minh):")
print("  - Kiểm tra bằng mắt (mẫu trông thực tế không?)")
print("  - Phủ mode (mẫu có trải mọi mode không?)")
print("  - Đa dạng (mẫu có biến thiên hay lặp lại?)")
print("  - Điểm discriminator (nên ~0.5 với generator tốt)")

with torch.no_grad():
    disc_scores_real = disc(data_test_tensor)
    disc_scores_fake = disc(samples_gan)
    
    print(f"\nĐiểm discriminator:")
    print(f"  Dữ liệu thật: {disc_scores_real.mean():.3f} (nên ~1.0 nếu D tốt)")
    print(f"  Mẫu sinh: {disc_scores_fake.mean():.3f} (nên ~0.5 ở cân bằng)")
    
    if disc_scores_fake.mean() > 0.4 and disc_scores_fake.mean() < 0.6:
        print("✓ Generator đánh lừa thành công discriminator!")
```

## 5. Các Khái niệm Liên quan

Mô hình sinh nối với ước lượng mật độ (density estimation), một bài toán cổ điển trong thống kê nơi ta cố ước lượng hàm mật độ xác suất từ mẫu. Các phương pháp truyền thống như ước lượng mật độ hạt nhân (kernel density estimation) hoặc khớp tham số (mô hình hỗn hợp Gaussian) hoạt động tốt ở chiều thấp nhưng mở rộng kém ra chiều cao do lời nguyền chiều (curse of dimensionality). Mô hình sinh dựa trên mạng neuron vượt qua điều này bằng cách học đặc trưng phân cấp nắm cấu trúc dữ liệu thay vì biểu diễn mật độ tường minh trong không gian đầu vào thô. Một mô hình sinh sâu thực chất thực hiện ước lượng mật độ trong không gian đặc trưng đã học nơi cấu trúc dữ liệu đơn giản hơn, rồi ánh xạ trở lại không gian đầu vào. Góc nhìn này giúp đánh giá vì sao mô hình sinh sâu thành công nơi phương pháp cổ điển thất bại.

Mối quan hệ với học không giám sát và tự giám sát (self-supervised learning) rất sâu. Mô hình sinh học biểu diễn không cần nhãn, khám phá cấu trúc thuần từ mẫu dữ liệu. Các đặc trưng học được trong mô hình hóa sinh thường chuyển giao tốt sang tác vụ giám sát downstream — autoencoder cung cấp khởi tạo tốt, discriminator của GAN học đặc trưng hữu ích, encoder của VAE tạo không gian ẩn có nghĩa. Điều này nối với chủ đề rộng hơn rằng tiền huấn luyện không giám sát quy mô lớn (như mô hình ngôn ngữ BERT hoặc GPT, là các tác vụ sinh) theo sau bởi tinh chỉnh giám sát thường vượt trội học thuần giám sát, đặc biệt trong chế độ dữ liệu hạn chế. Hiểu mô hình sinh cung cấp hiểu biết vì sao học không giám sát hoạt động và biểu diễn nào nổi lên từ mục tiêu sinh.

Mô hình sinh nối với tăng cường dữ liệu qua khả năng sinh ví dụ huấn luyện tổng hợp. Với tập dữ liệu mất cân bằng (nhiều ví dụ lớp phổ biến, ít lớp hiếm), mô hình sinh có thể tổng hợp thêm ví dụ lớp thiểu số. Với dữ liệu đắt gán nhãn (ảnh y tế cần chuyên gia chú thích), ví dụ sinh có thể tăng cường tập gán nhãn hạn chế. Tuy nhiên cần cẩn thận: nếu mô hình sinh chưa học chính xác phân phối thật, ví dụ tổng hợp có thể đưa vào thiên lệch. Thực hành tốt nhất là kiểm chứng rằng ví dụ sinh giúp chứ không hại hiệu năng tác vụ downstream.

Sự tiến hóa của chỉ số đánh giá cho mô hình sinh phản ánh các thách thức đang diễn ra trong đo chất lượng và đa dạng. GAN ban đầu dùng đánh giá con người (tốn thời gian, không tái lập) hoặc kiểm tra bộ phân loại nhị phân (bộ phân loại có phân biệt thật/giả không?). Inception Score và FID cung cấp chỉ số tự động nhưng có thiên lệch và chế độ thất bại đã biết. Nghiên cứu gần đây khám phá chỉ số cảm nhận học được (đo khoảng cách trong không gian đặc trưng đã học), đánh đổi precision-recall (định lượng chất lượng so với đa dạng riêng rẽ), và phương pháp dựa trên hợp lý (cho mô hình cung cấp hợp lý). Hiểu rằng không chỉ số nào hoàn hảo hướng dẫn thực hành dùng nhiều đánh giá bổ sung thay vì tối ưu cho bất kỳ chỉ số đơn lẻ nào.

Cuối cùng, mô hình sinh nối với câu hỏi nền tảng về điều mạng neuron học. Bằng cách huấn luyện mạng sinh dữ liệu phức tạp như ảnh hoặc văn bản, ta thực chất hỏi: mẫu hình, cấu trúc và tính đều đặn nào tồn tại trong dữ liệu này, và mạng có thể khám phá chúng tự động không? Thực tế rằng mạng neuron có thể học sinh khuôn mặt chân thực, đoạn văn mạch lạc, hoặc cấu trúc phân tử hợp lệ chứng tỏ chúng nắm các tính đều đặn thống kê sâu, không chỉ ghi nhớ. Việc học cấu trúc ẩn này có hệ quả vượt ngoài sinh — nó gợi ý mạng neuron đang khám phá biểu diễn phản ánh cấu trúc thật trong thế giới, không chỉ khớp dữ liệu huấn luyện.

## 6. Các Bài báo Nền tảng

**["A tutorial on Energy-Based Learning" (2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)**  
*Tác giả*: Yann LeCun  
Dù không dành riêng cho mô hình sinh hiện đại, tutorial này thiết lập khung dựa trên năng lượng (energy-based framework) nền tảng cho phần lớn mô hình hóa sinh. LeCun chỉ ra nhiều bài toán học có thể được công thức hóa như học hàm năng lượng gán năng lượng thấp cho đầu ra đúng/thực tế và năng lượng cao cho đầu ra sai/phi thực tế. Mô hình sinh khớp khung này: chúng học cảnh quan năng lượng nơi dữ liệu có năng lượng thấp. Tutorial bao phủ Boltzmann machine, contrastive divergence, và các kỹ thuật khác đã ảnh hưởng công trình sau về mô hình sinh sâu. Hiểu mô hình dựa trên năng lượng cung cấp nền tảng lý thuyết vì sao một số thủ tục huấn luyện (như contrastive divergence hoặc score matching) hoạt động và nối mô hình hóa sinh với vật lý thống kê và suy diễn xác suất. Dù mô hình sinh hiện đại thường dùng thủ tục huấn luyện khác (backpropagation với reparameterization cho VAE, huấn luyện đối kháng cho GAN), góc nhìn dựa trên năng lượng vẫn có giá trị để hiểu các mô hình này đang làm gì về bản chất.

**["NADE: The Neural Autoregressive Distribution Estimator" (2011)](http://proceedings.mlr.press/v15/larochelle11a.html)**  
*Tác giả*: Hugo Larochelle, Iain Murray  
Bài báo này giới thiệu NADE, một mô hình tự hồi quy hiệu quả tính tractable $$p(\mathbf{x}) = \prod_i p(x_i|\mathbf{x}_{<i})$$ dùng mạng neuron cho các điều kiện. Đổi mới then chốt là chia sẻ trọng số: thay vì huấn luyện mạng riêng cho mỗi điều kiện, NADE dùng một mạng neuron duy nhất với tham số chia sẻ, khiến nó hiệu quả và ngăn overfitting. Bài báo chứng minh mô hình tự hồi quy có thể cạnh tranh với cách tiếp cận phức tạp hơn như restricted Boltzmann machine trong khi cung cấp tính hợp lý chính xác và huấn luyện ổn định. NADE ảnh hưởng các mô hình tự hồi quy sau như PixelRNN/PixelCNN (cho ảnh) và WaveNet (cho âm thanh), thiết lập mô hình hóa tự hồi quy như cách tiếp cận khả thi cho dữ liệu phức tạp, chiều cao. Công trình cho thấy mô hình hóa mật độ tường minh — tham số hóa trực tiếp $$p(\mathbf{x})$$ — thực tế cho deep learning, không chỉ thống kê cổ điển.

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Tác giả*: Diederik P. Kingma, Max Welling  
Bài báo nền tảng này giới thiệu Variational Autoencoder, kết hợp suy diễn biến phân với mạng neuron để tạo khung mở rộng được cho mô hình hóa sinh với biến ẩn. Đóng góp then chốt là reparameterization trick: thay vì lấy mẫu $$\mathbf{z} \sim q(\mathbf{z}|\mathbf{x})$$ (không khả vi theo tham số của $$q$$), viết lại lấy mẫu thành $$\mathbf{z} = \mu + \sigma \odot \boldsymbol{\epsilon}$$ với $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$. Hàm tất định này của tham số ($$\mu, \sigma$$) và ngẫu nhiên ngoài ($$\boldsymbol{\epsilon}$$) cho phép backpropagation qua lấy mẫu, khiến suy diễn biến phân huấn luyện được qua gradient descent. Bài báo chỉ ra VAE có thể học biểu diễn ẩn có nghĩa và sinh mẫu mới trong khi cung cấp khung xác suất có nguyên tắc (không như GAN đồng thời nhưng ban đầu heuristic hơn). VAE ảnh hưởng vô số công trình sau và thiết lập rằng mô hình biến ẩn có thể mở rộng cho dữ liệu phức tạp qua thiết kế thuật toán cẩn thận. Mục tiêu ELBO và reparameterization trick đã trở thành công cụ nền tảng trong deep learning xác suất.

**["Generative Adversarial Networks" (2014)](https://arxiv.org/abs/1406.2661)**  
*Tác giả*: Ian Goodfellow và cộng sự  
Bài báo GAN cách mạng hóa mô hình hóa sinh bằng cách giới thiệu huấn luyện đối kháng như lựa chọn thay cho hợp lý cực đại. Bằng cách khung hóa sinh như một trò chơi giữa generator và discriminator, GAN cho phép học mô hình mật độ ẩn sinh mẫu chất lượng cao mà không đòi hỏi tính mật độ tường minh hay tích phân không tractable. Phân tích lý thuyết của bài báo — chỉ ra tại cân bằng Nash, generator khôi phục phân phối dữ liệu — cung cấp nền tảng trong khi kết quả thực nghiệm chứng minh tính khả thi thực tiễn. GAN sinh ra nghiên cứu khổng lồ sau đó xử lý ổn định huấn luyện, sụp đổ mode, và thiết kế kiến trúc, trở thành một trong những ý tưởng ảnh hưởng nhất trong học máy hiện đại. Khung đối kháng đã được áp dụng vượt ngoài sinh sang thích ứng miền, huấn luyện robust, và học bán giám sát, chứng tỏ một mô hình huấn luyện mới có thể tác động rộng đến lĩnh vực.

**["Normalizing Flows for Probabilistic Modeling and Inference" (2019)](https://arxiv.org/abs/1912.02762)**  
*Tác giả*: George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan  
Bài tổng quan toàn diện này thống nhất normalizing flow — mô hình sinh dựa trên phép biến đổi khả nghịch — giải thích nền tảng lý thuyết và cài đặt thực tiễn. Flow học ánh xạ song ánh $$\mathbf{x} = f(\mathbf{z})$$ nơi $$\mathbf{z}$$ có mật độ đơn giản (Gaussian) và $$f$$ khả nghịch với định thức Jacobian tractable. Điều này cho phép tính hợp lý chính xác (không như GAN) và lấy mẫu nhanh (không như mô hình tự hồi quy). Bài báo bao phủ cảnh quan kiến trúc flow (coupling flow, autoregressive flow, continuous flow), tính chất lý thuyết, và ứng dụng. Flow ít dùng hơn VAE hoặc GAN cho sinh ảnh nhưng xuất sắc trong tác vụ đòi hỏi mật độ chính xác (phát hiện bất thường, nén) hoặc cấu trúc cụ thể (sinh phân tử nơi ràng buộc hợp lệ quan trọng). Hiểu flow hoàn thiện bức tranh mô hình hóa sinh, cho thấy không gian đánh đổi giữa tractability hợp lý, hiệu quả lấy mẫu, và linh hoạt kiến trúc.

## Bẫy thường gặp và Mẹo

Sai lầm nền tảng nhất trong mô hình hóa sinh là đánh giá mô hình chỉ trên hợp lý tập huấn luyện hoặc chất lượng tái tạo. Một mô hình ghi nhớ ví dụ huấn luyện đạt hợp lý huấn luyện hoàn hảo nhưng không sinh ví dụ mới — nó thất bại như mô hình sinh dù tối ưu mục tiêu hoàn hảo. Triệu chứng gồm mẫu sinh gần như giống hệt ví dụ huấn luyện và hợp lý tập kiểm tra kém. Phát hiện đòi hỏi kiểm tra láng giềng gần nhất trong tập huấn luyện cho mỗi mẫu sinh (nếu luôn rất gần, có thể ghi nhớ) và đánh giá trên dữ liệu giữ lại. Phòng ngừa gồm chính quy hóa đúng (weight decay, dropout), dùng tập validation để chọn mô hình, và kiến trúc khuyến khích tổng quát hóa (nút thắt trong autoencoder, discriminator trong GAN buộc tính mới).

Chọn hàm mất mát tái tạo không phù hợp gây lệch cảm nhận giữa điều mô hình tối ưu và điều con người quan tâm. MSE theo pixel coi mọi pixel ngang nhau, nhưng thị giác người không đều — ta nhạy hơn với cấu trúc và cạnh hơn vùng mượt. Một tái tạo tối ưu MSE có thể mờ (trung bình hóa chi tiết) trong khi trông kém về cảm nhận. Ngược lại, tái tạo với cạnh hơi lệch (MSE cao) có thể trông tương tự về cảm nhận. Giải pháp gồm mất mát cảm nhận (perceptual loss — đo khoảng cách trong không gian đặc trưng của mạng tiền huấn luyện như VGG), mất mát đối kháng (dùng discriminator đánh giá tính thực tế), hoặc mất mát có cấu trúc (đo tương tự gradient, không chỉ tương tự pixel). Hiểu rằng hàm mất mát thể hiện giả định về điều quan trọng hướng dẫn lựa chọn phù hợp cho ứng dụng cụ thể.

Với mô hình biến ẩn, chọn chiều ẩn liên quan đến các đánh đổi tinh tế. Quá nhỏ (2–3 chiều) cho phép trực quan hóa nhưng có thể không nắm độ phức tạp dữ liệu, gây tái tạo kém. Quá lớn (tiệm cận chiều đầu vào) cho phép tái tạo hoàn hảo nhưng có thể không học cấu trúc có nghĩa — mô hình có thể dùng mỗi chiều ẩn cho một chiều đầu vào, học ánh xạ đồng nhất. Kích thước đúng phụ thuộc độ phức tạp dữ liệu và mức nén mong muốn. Heuristic hữu ích: bắt đầu với nén 10–20×, điều chỉnh theo chất lượng tái tạo và hiệu năng tác vụ downstream. Với MNIST (784 chiều), thử 32–64 chiều ẩn. Với ImageNet (224×224×3), thử 512–2048.

Khi sinh mẫu, nhiệt độ lấy mẫu (sampling temperature) thường ảnh hưởng đáng kể đánh đổi chất lượng–đa dạng. Với mô hình tự hồi quy hoặc VAE nơi ta lấy mẫu từ phân phối đã học, ta có thể scale logit bằng nhiệt độ trước softmax:

$$p(x_i | \mathbf{x}_{<i}) = \text{softmax}(\mathbf{z}_i / T)$$

Nhiệt độ thấp ($$T < 1$$) làm phân phối sắc nét hơn — tự tin hơn, ít đa dạng hơn. Nhiệt độ cao ($$T > 1$$) làm nó đồng đều hơn — đa dạng hơn nhưng có thể kém thực tế. Nhiệt độ cung cấp một núm điều khiển sau huấn luyện để đánh đổi chất lượng và đa dạng mà không huấn luyện lại. Hiểu đánh đổi này giúp sinh mẫu phù hợp cho các ứng dụng khác nhau.

Một kỹ thuật mạnh để cải thiện chất lượng mẫu là rejection sampling: sinh nhiều mẫu, chấm điểm bằng discriminator hoặc bộ phân loại, chỉ giữ mẫu điểm cao. Điều này lọc mẫu sinh theo chất lượng với chi phí hiệu quả (phải sinh nhiều mẫu hơn cần thiết). Với ứng dụng nơi chất lượng quan trọng hơn tốc độ sinh (tạo nghệ thuật, thiết kế phân tử), rejection sampling là thắng lợi dễ dàng. Hiểu rằng ta có thể hậu xử lý mẫu sinh — không chỉ dùng bất kỳ thứ gì mô hình tạo ra — mở rộng bộ công cụ cho ứng dụng thực tiễn.

## Điểm then chốt

Mô hình sinh học hiểu và tái tạo phân phối dữ liệu, cho phép tạo mẫu mới, thực tế từ mẫu hình đã học. Ba mô hình chính — mô hình tự hồi quy cung cấp phân rã mật độ tuần tự tường minh, variational autoencoder dùng biến ẩn với suy diễn biến phân, và mạng sinh đối kháng huấn luyện qua cạnh tranh đối kháng — đưa ra các đánh đổi khác nhau giữa tractability hợp lý, hiệu quả lấy mẫu, ổn định huấn luyện, và chất lượng mẫu. Hợp lý cực đại cung cấp mục tiêu huấn luyện có nguyên tắc nối với lý thuyết thông tin qua phân kỳ KL, dù đòi hỏi đánh giá mật độ tractable hoặc cận dưới. Mô hình biến ẩn đưa vào biểu diễn nén nắm các nhân tố biến thiên, cho phép lấy mẫu nhanh và thao tác diễn giải được, dù đòi hỏi thủ tục suy diễn cẩn thận. Đánh giá mô hình sinh là thách thức, đòi hỏi nhiều chỉ số (hợp lý khi có, Inception Score, FID, đánh giá con người) và kiểm chứng chuyên ngành thay vì số đơn lẻ. Ứng dụng trải tăng cường dữ liệu, siêu phân giải, chuyển phong cách, khám phá thuốc, và công cụ sáng tạo, với lựa chọn cách tiếp cận phụ thuộc vào việc ta cần ước lượng hợp lý, sinh có kiểm soát, chất lượng mẫu, hay ổn định huấn luyện. Hiểu sâu mô hình hóa sinh nghĩa là đánh giá cả nền tảng thống kê (lý thuyết xác suất, ước lượng mật độ, suy diễn biến phân) và cài đặt deep learning (kiến trúc mạng, thuật toán huấn luyện, mẹo thực tiễn) khiến việc học phân phối phức tạp trở nên tractable.

Mô hình sinh chứng tỏ mạng neuron có thể khám phá và nội hóa cấu trúc thống kê nền tảng của dữ liệu phức tạp, học biểu diễn cho phép không chỉ nhận diện mà còn sáng tạo — một năng lực tiến gần hơn điều ta có thể coi là sự hiểu thực sự.
