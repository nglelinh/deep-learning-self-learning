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

Nếu mô hình tự hồi quy giống một nghệ nhân tỉ mỉ vẽ từng nét theo công thức xác suất, thì GAN lại giống một cuộc đấu trí kịch tính giữa kẻ làm tiền giả và chuyên gia kiểm định. Trên cùng tập dữ liệu 8 cụm Gaussian nằm trên vòng tròn, chỉ cần vài chục dòng mã là bạn sẽ chứng kiến tận mắt hiện tượng đáng sợ nhất của GAN: *mode collapse* — khi bộ sinh bỏ mặc 7 cụm dữ liệu còn lại để chỉ vẽ đi vẽ lại đúng 1 cụm duy nhất nhằm đánh lừa bộ phân biệt.

## Kiến thức cần nhớ: Cơ chế thực nghiệm và các bẫy kinh điển khi huấn luyện GAN

- **Vòng lặp đối kháng xen kẽ (Alternating Optimization):** Cập nhật bộ phân biệt $$D$$ để tách mẫu thật/giả trước, rồi đóng băng $$D$$ để cập nhật bộ sinh $$G$$ theo hướng làm tăng xác suất $$D(G(\mathbf{z}))$$.
- **Chẩn đoán sụp đổ mode (Mode Collapse):** Đo độ phủ trên từng tâm cụm thật để phát hiện sớm tình trạng $$G$$ chỉ sinh ra các mẫu rất đẹp nhưng lặp đi lặp lại ở một vài vùng hẹp.
- **Đánh đổi giữa hợp lý tường minh và chất lượng mẫu:** So sánh trực tiếp trên cùng bộ dữ liệu toy giữa mô hình tự hồi quy (tính được NLL chính xác, phủ đều mode nhưng có thể tràn xác suất) và GAN (mẫu sắc nét nhưng không có log-likelihood).
- **Kỹ thuật ổn định thực hành:** Nhận biết bẫy bão hòa gradient của hàm loss gốc, lệch nhịp tốc độ học giữa $$G$$ và $$D$$, cùng giải pháp non-saturating loss.

### Giải thích: So sánh gradient của hàm loss gốc và non-saturating loss khi $$D(G(\mathbf{z}))$$ gần 0

Ở giai đoạn đầu huấn luyện, ảnh giả còn rất thô nên bộ phân biệt dễ dàng bác bỏ: giả sử $$D(G(\mathbf{z})) = 0{,}01$$ (logit $$u = \text{logit}(0{,}01) \approx -4{,}595$$).

1. **Hàm mục tiêu minimax gốc của $$G$$:** $$\mathcal{L}_G^{\text{minimax}} = \ln(1 - D(G(\mathbf{z}))) = \ln(1 - \sigma(u))$$.
   Đạo hàm theo logit $$u$$ là:
   $$\frac{\partial \mathcal{L}_G^{\text{minimax}}}{\partial u} = -\sigma(u) = -0{,}01.$$
   Độ lớn gradient chỉ bằng $$0{,}01$$ — gần như **triệt tiêu** đúng lúc $$G$$ cần học mạnh nhất!
2. **Hàm mục tiêu non-saturating (dùng trong thực tế):** $$\mathcal{L}_G^{\text{NS}} = -\ln D(G(\mathbf{z})) = -\ln \sigma(u)$$.
   Đạo hàm theo logit $$u$$ là:
   $$\frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial u} = -(1 - \sigma(u)) = -(1 - 0{,}01) = -0{,}99.$$
3. **Kết luận định lượng:** Tại cùng một trạng thái $$D(G(\mathbf{z})) = 0{,}01$$, non-saturating loss tạo ra tín hiệu gradient **mạnh gấp 99 lần** so với công thức minimax gốc, giúp $$G$$ thoát khỏi điểm chết ban đầu mà vẫn giữ nguyên điểm tối ưu toàn cục.

```python
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

Đoạn mã dưới đây minh họa cách đánh giá mô hình dựa trên log-hợp lý (với mô hình tự hồi quy) và các chỉ số thay thế (với GAN):

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

## 5. Các khái niệm liên quan

Mô hình sinh gắn trực tiếp với bài toán ước lượng mật độ (*density estimation*) trong thống kê cổ điển. Các phương pháp truyền thống như ước lượng mật độ hạt nhân (*kernel density estimation*) hay mô hình hỗn hợp Gaussian hoạt động tốt ở số chiều thấp, nhưng nhanh chóng suy giảm khi số chiều tăng cao do lời nguyền số chiều (*curse of dimensionality*). Mô hình sinh sâu vượt qua hạn chế này bằng cách học các tầng đặc trưng phân cấp nắm bắt cấu trúc dữ liệu, thực chất ước lượng mật độ trong một không gian biểu diễn có cấu trúc đơn giản hơn rồi ánh xạ ngược về không gian dữ liệu gốc.

Mô hình hóa sinh cũng liên hệ mật thiết với học không giám sát và học tự giám sát (*self-supervised learning*). Vì không đòi hỏi nhãn, mô hình buộc phải tự khám phá quy luật nội tại của dữ liệu. Các biểu diễn học được thường chuyển giao tốt sang các tác vụ giám sát về sau — chẳng hạn bộ mã hóa của autoencoder và VAE hay đặc trưng trung gian của bộ phân biệt trong GAN. Xu hướng tiền huấn luyện tự giám sát quy mô lớn (như các mô hình ngôn ngữ BERT hoặc GPT) rồi tinh chỉnh cho bài toán cụ thể cũng bắt nguồn từ cùng nguyên lý này.

Trong thực hành, mô hình sinh còn được dùng để tăng cường dữ liệu bằng mẫu tổng hợp, đặc biệt khi tập dữ liệu mất cân bằng lớp hoặc chi phí gán nhãn cao (như ảnh y khoa). Tuy nhiên, nếu mô hình sinh chưa khớp tốt với phân phối thật, dữ liệu tổng hợp có thể khuếch đại sai lệch sẵn có; vì vậy luôn cần kiểm chứng xem dữ liệu sinh thêm có thực sự cải thiện hiệu năng trên tập kiểm tra thật hay không.

Sự phát triển của các chỉ số đánh giá cũng phản ánh mức độ khó của việc định lượng đồng thời chất lượng và độ đa dạng. Sau giai đoạn dựa vào đánh giá thủ công hoặc bộ phân loại nhị phân, các chỉ số tự động như Inception Score và FID được dùng phổ biến hơn dù vẫn có những thiên lệch nhất định. Các hướng đánh giá sau này bổ sung thêm khoảng cách cảm nhận (*perceptual distance*), đường cong precision–recall để tách riêng chất lượng với độ đa dạng, cùng các phép đo dựa trên hàm hợp lý khi mô hình hỗ trợ. Do không có một chỉ số đơn lẻ nào phản ánh toàn diện, thực hành tốt nhất là kết hợp nhiều thước đo bổ trợ nhau.

## 6. Các bài báo nền tảng

**["A tutorial on Energy-Based Learning" (2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)**  
*Tác giả*: Yann LeCun  
Bài hướng dẫn này xây dựng khung học dựa trên năng lượng (*energy-based learning*), nền tảng lý thuyết của nhiều hướng mô hình hóa sinh. Ý tưởng trung tâm là học một hàm năng lượng gán giá trị thấp cho các cấu hình dữ liệu hợp lệ và giá trị cao cho các cấu hình bất hợp lý. Khung này bao quát máy Boltzmann, *contrastive divergence* và kết nối mô hình hóa sinh với vật lý thống kê. Dù các mô hình hiện đại thường dùng lan truyền ngược kết hợp tái tham số hóa hoặc huấn luyện đối kháng, góc nhìn năng lượng vẫn rất hữu ích để hiểu cơ chế hoạt động chung của các thuật toán sinh.

**["NADE: The Neural Autoregressive Distribution Estimator" (2011)](http://proceedings.mlr.press/v15/larochelle11a.html)**  
*Tác giả*: Hugo Larochelle, Iain Murray  
Bài báo giới thiệu NADE, một mô hình tự hồi quy sử dụng mạng nơ-ron chia sẻ trọng số để tính chính xác và hiệu quả tích phân rã $$p(\mathbf{x}) = \prod_i p(x_i|\mathbf{x}_{<i})$$. Nhờ chia sẻ tham số thay vì huấn luyện mạng riêng cho từng phân phối điều kiện, NADE vừa giảm chi phí tính toán vừa hạn chế quá khớp, đồng thời đạt kết quả cạnh tranh với máy Boltzmann hạn chế (RBM). NADE đặt nền móng trực tiếp cho các kiến trúc tự hồi quy về sau như PixelRNN/PixelCNN cho ảnh và WaveNet cho âm thanh.

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Tác giả*: Diederik P. Kingma, Max Welling  
Bài báo giới thiệu Variational Autoencoder (VAE), kết hợp suy diễn biến phân với mạng nơ-ron sâu cho mô hình biến ẩn. Đóng góp cốt lõi là thủ thuật tái tham số hóa (*reparameterization trick*): thay vì lấy mẫu trực tiếp $$\mathbf{z} \sim q(\mathbf{z}|\mathbf{x})$$ (không khả vi theo tham số của $$q$$), biểu diễn $$\mathbf{z} = \mu + \sigma \odot \boldsymbol{\epsilon}$$ với $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$. Cách viết này cho phép lan truyền ngược gradient qua phép lấy mẫu để tối ưu mục tiêu ELBO bằng hạ gradient, đưa VAE trở thành một trong những công cụ chuẩn của học sâu xác suất.

**["Generative Adversarial Networks" (2014)](https://arxiv.org/abs/1406.2661)**  
*Tác giả*: Ian Goodfellow và cộng sự  
Bài báo mở ra hướng huấn luyện đối kháng thay thế cho ước lượng hợp lý cực đại. Bằng cách mô hình hóa quá trình sinh như trò chơi hai người giữa bộ sinh và bộ phân biệt, GAN học trực tiếp hàm lấy mẫu mà không cần tính hàm mật độ tường minh hay tích phân biên. Phần phân tích lý thuyết chứng minh tại điểm cân bằng Nash, phân phối của bộ sinh trùng với phân phối dữ liệu thật. Ý tưởng huấn luyện đối kháng sau đó không chỉ thúc đẩy mảng sinh ảnh mà còn mở rộng sang thích ứng miền, tăng độ bền vững và học bán giám sát.

**["Normalizing Flows for Probabilistic Modeling and Inference" (2019)](https://arxiv.org/abs/1912.02762)**  
*Tác giả*: George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan  
Bài tổng quan hệ thống hóa nhóm mô hình *normalizing flows* dựa trên chuỗi phép biến đổi song ánh khả nghịch $$\mathbf{x} = f(\mathbf{z})$$ có định thức Jacobian tính được. Cách tiếp cận này đồng thời cho phép tính log-hợp lý chính xác (khác với GAN) và lấy mẫu nhanh (khác với mô hình tự hồi quy). Bài báo bao quát các kiến trúc chính (*coupling flow*, *autoregressive flow*, *continuous flow*) và chỉ ra thế mạnh của nhóm mô hình này trong các bài toán yêu cầu mật độ chính xác như phát hiện bất thường, nén dữ liệu hoặc thiết kế phân tử.

## Bẫy thường gặp và Mẹo

Một sai lầm phổ biến là chỉ đánh giá mô hình sinh qua log-hợp lý trên tập huấn luyện hoặc sai số tái tạo. Một mô hình học thuộc lòng tập huấn luyện vẫn có thể đạt giá trị tối ưu rất tốt nhưng không tạo ra được mẫu mới. Để kiểm tra hiện tượng này, cần tìm láng giềng gần nhất trong tập huấn luyện cho các mẫu sinh ra và luôn đánh giá trên tập kiểm tra độc lập. Các biện pháp phòng tránh gồm dùng chính quy hóa phù hợp (weight decay, dropout), chọn mô hình trên tập validation và thiết kế kiến trúc hạn chế sao chép trực tiếp.

Lựa chọn hàm mất mát tái tạo không phù hợp cũng dễ tạo chênh lệch giữa giá trị tối ưu và cảm nhận thị giác. MSE theo từng điểm ảnh xem mọi sai lệch là như nhau, trong khi mắt người nhạy hơn nhiều với cấu trúc và biên cạnh, dẫn đến hiện tượng ảnh tái tạo bị mờ khi tối ưu MSE. Các phương án cải thiện gồm hàm mất mát cảm nhận (*perceptual loss* tính trong không gian đặc trưng của mạng tiền huấn luyện như VGG), hàm mất mát đối kháng hoặc các hàm đo tương đồng cấu trúc.

Trong mô hình biến ẩn, kích thước không gian ẩn cần được chọn cân bằng. Số chiều quá nhỏ (2–3 chiều) thuận tiện cho việc trực quan hóa nhưng khó giữ đủ thông tin phức tạp của dữ liệu. Ngược lại, số chiều quá lớn (gần bằng số chiều đầu vào) khiến mô hình dễ học ánh xạ đồng nhất mà không rút trích được cấu trúc hữu ích. Một quy tắc khởi đầu hợp lý là tỷ lệ nén khoảng 10–20 lần rồi tinh chỉnh theo chất lượng tái tạo và tác vụ đích — ví dụ khoảng 32–64 chiều cho MNIST (784 chiều) hoặc 512–2048 chiều cho ImageNet (224×224×3).

Khi lấy mẫu từ mô hình tự hồi quy hoặc VAE, tham số nhiệt độ lấy mẫu (*sampling temperature*) giúp điều chỉnh trực tiếp sự đánh đổi giữa chất lượng và độ đa dạng bằng cách chia logit cho $$T$$ trước hàm softmax:

$$p(x_i | \mathbf{x}_{<i}) = \text{softmax}(\mathbf{z}_i / T)$$

Nhiệt độ thấp ($$T < 1$$) làm phân phối tập trung hơn, mẫu sinh ổn định nhưng kém đa dạng; nhiệt độ cao ($$T > 1$$) tăng độ đa dạng nhưng dễ xuất hiện mẫu kém hợp lý. Đây là cơ chế điều chỉnh linh hoạt ở giai đoạn suy luận mà không cần huấn luyện lại mô hình.

Ngoài ra, kỹ thuật *rejection sampling* — sinh nhiều mẫu ứng viên, chấm điểm bằng bộ phân biệt hoặc bộ phân loại rồi chỉ giữ các mẫu đạt điểm cao — là cách hậu xử lý đơn giản để nâng chất lượng đầu ra trong các ứng dụng ưu tiên chất lượng hơn tốc độ sinh (như thiết kế đồ họa hay sinh cấu trúc phân tử).

## Điểm then chốt

Mô hình sinh học cách nắm bắt và tái tạo phân phối xác suất của dữ liệu để tạo ra các mẫu mới hợp lệ. Ba hướng tiếp cận nền tảng — mô hình tự hồi quy (phân rã mật độ tuần tự tường minh), Variational Autoencoder (biến ẩn kết hợp suy diễn biến phân) và GAN (huấn luyện đối kháng mật độ ẩn) — tạo nên các điểm cân bằng khác nhau giữa khả năng tính hàm hợp lý, tốc độ lấy mẫu, độ ổn định khi huấn luyện và chất lượng mẫu. Ước lượng hợp lý cực đại cung cấp mục tiêu tối ưu có cơ sở lý thuyết chặt chẽ thông qua phân kỳ KL, còn mô hình biến ẩn mang lại không gian biểu diễn nén dễ thao tác và nội suy. Vì đánh giá mô hình sinh khó hơn bài toán phân loại, cần phối hợp nhiều thước đo (log-hợp lý khi có, Inception Score, FID, đánh giá trực quan và chỉ số chuyên ngành) thay vì dựa vào một con số duy nhất.

## Video tham khảo

- **Xem trước:** [The Math Behind Generative Adversarial Networks Clearly Explained!](https://www.youtube.com/watch?v=Gib_kiXgnvA) — Normalized Nerd, tiếng Anh. Giải thích mục tiêu minimax tương ứng với hàm mất mát của mô hình GAN đơn giản trong bài.
- [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, tiếng Anh. Trình bày trực quan trò chơi hai người giữa bộ sinh và bộ phân biệt trước khi đi vào mã nguồn.
- [What are GANs (Generative Adversarial Networks)?](https://www.youtube.com/watch?v=TpMIssRdhco) — IBM Technology, tiếng Anh. Minh họa ngắn gọn vai trò của bộ sinh và bộ phân biệt nếu bạn cần nhìn tổng quan trước phần toán học.
