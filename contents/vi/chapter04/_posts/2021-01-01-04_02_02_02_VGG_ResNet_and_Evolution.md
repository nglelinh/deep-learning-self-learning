---
layout: post
title: 04-02-02-02 VGG, ResNet và Tiến hóa Kiến trúc
chapter: '04'
order: 13
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Năm 2015, nhóm nghiên cứu của Kaiming He tại Microsoft Research chứng kiến một hiện tượng hết sức trớ trêu: khi họ xếp chồng một mạng CNN thường lên 56 tầng thay vì 20 tầng trên tập CIFAR-10, **ngay cả sai số huấn luyện (training error)** của mạng 56 tầng cũng tệ hơn mạng 20 tầng! Đó không phải overfitting — mà là **vấn đề suy thoái (degradation problem)**. Và giải pháp của họ lại đơn giản đến mức ai cũng phải thốt lên: chỉ cần cộng thêm đầu vào $$x$$ vào đầu ra của khối tích chập, tạo nên **ResNet**.

## Kiến thức cần nhớ: Bốn nguyên lý thiết kế từ VGG đến ResNet

1. **Nguyên lý VGG — Phân rã nhân lớn thành chuỗi nhân $$3 \times 3$$**:
   Hai tầng $$3 \times 3$$ liên tiếp có cùng trường tiếp nhận với một tầng $$5 \times 5$$, và ba tầng $$3 \times 3$$ tương đương một tầng $$7 \times 7$$, nhưng ít tham số hơn nhiều và có thêm nhiều lớp ReLU phi tuyến ở giữa.
2. **Nghịch lý suy thoái (Degradation) vs. Overfitting**:
   Về lý thuyết, một mạng sâu hơn ít nhất phải tốt bằng mạng nông nếu các tầng thêm vào học ánh xạ đồng nhất $$H(x) = x$$. Thực tế, bộ tối ưu cực kỳ khó ép một chuỗi tầng phi tuyến học chính xác hàm đồng nhất.
3. **Học phần dư (Residual Learning) $$H(x) = F(x) + x$$**:
   Khi viết lại đầu ra khối là $$F(x) + x$$, muốn học ánh xạ đồng nhất, mạng chỉ cần đẩy các trọng số của nhánh $$F(x)$$ về $$0$$ — điều mà khởi tạo nhỏ và weight decay làm cực kỳ dễ dàng.
4. **Đường cao tốc Gradient (Gradient Highway)**:
   Khi đạo hàm ngược qua $$y = F(x) + x$$, ta có $$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}\left(\frac{\partial F(x)}{\partial x} + I\right)$$. Số hạng đơn vị $$+I$$ đảm bảo gradient luôn chảy thẳng về các tầng đầu mà không bị triệt tiêu kể cả khi mạng sâu hơn 100 tầng.

### Giải thích: Tính tay so sánh 3 tầng $$3 \times 3$$ (VGG) với 1 tầng $$7 \times 7$$ và dòng gradient trong ResNet

**Phần A — Tại sao VGG chọn 3 tầng $$3 \times 3$$ thay vì 1 tầng $$7 \times 7$$** (với $$C_{\text{in}} = C_{\text{out}} = C$$ kênh):
- **Trường tiếp nhận**: Tầng 1 nhìn $$3\times 3$$ $$\to$$ Tầng 2 nhìn $$5\times 5$$ $$\to$$ Tầng 3 nhìn **$$7\times 7$$** (y hệt 1 tầng $$7\times 7$$).
- **Số tham số**:
  - 1 tầng $$7 \times 7$$: $$7 \times 7 \times C^2 = \mathbf{49 C^2}$$.
  - 3 tầng $$3 \times 3$$: $$3 \times (3 \times 3 \times C^2) = \mathbf{27 C^2}$$ (tiết kiệm **45%** tham số!).
- **Độ phi tuyến**: 3 lần ReLU thay vì chỉ 1 lần ReLU.

**Phần B — Kiểm chứng đường cao tốc gradient của ResNet**:
Giả sử một khối phi tuyến có đạo hàm rất nhỏ $$\frac{\partial F}{\partial x} = 0{,}1$$ và ta xếp chồng $$L = 20$$ khối:
- **Mạng thường (không skip)**: Gradient thu nhỏ theo $$0{,}1^{20} = \mathbf{10^{-20}}$$ (tắt lịm hoàn toàn!).
- **Mạng ResNet (có skip $$+x$$)**: Đạo hàm mỗi khối là $$1 + 0{,}1 = 1{,}1$$, gradient không những không tắt mà còn truyền nguyên vẹn từ tầng 20 về tầng 1 nhờ số hạng $$1$$.

### VGGNet (2014) — Đơn giản và Chiều sâu

VGG của Simonyan và Zisserman cho thấy việc **tăng chiều sâu** với một cấu trúc **đồng nhất, đơn giản** mang lại cải thiện rõ rệt về độ chính xác.

**Ý tưởng chính:** Chỉ sử dụng các tầng tích chập 3×3 xếp chồng liên tiếp.

**Kiến trúc VGG-16:**
```
Input (224×224×3)

Block 1: Conv3-64, Conv3-64, MaxPool → 112×112×64
Block 2: Conv3-128, Conv3-128, MaxPool → 56×56×128
Block 3: Conv3-256, Conv3-256, Conv3-256, MaxPool → 28×28×256
Block 4: Conv3-512, Conv3-512, Conv3-512, MaxPool → 14×14×512
Block 5: Conv3-512, Conv3-512, Conv3-512, MaxPool → 7×7×512

Flatten → FC 4096 → FC 4096 → FC 1000
```

**Vì sao chọn nhân 3×3:**
- Hai tầng 3×3 nối tiếp cho cùng trường tiếp nhận 5×5 như một tầng 5×5, nhưng:
  - Ít tham số hơn: $$2 \times (3 \times 3) = 18$$ so với $$5 \times 5 = 25$$ (trên mỗi cặp kênh)
  - Nhiều lớp phi tuyến hơn: 2 lớp ReLU so với 1
- Ba tầng 3×3 nối tiếp cho trường tiếp nhận tương đương 7×7

Tổng số tham số vào khoảng **138 triệu** (phần lớn tập trung ở các tầng liên kết đầy đủ cuối mạng).

```python
class VGG16(nn.Module):
    """Kiến trúc VGG-16"""
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Khối 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Khối 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Khối 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Khối 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Khối 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### ResNet (2015) — Bước tiến từ Kết nối Tắt

ResNet của Kaiming He và cộng sự giải quyết **bài toán suy giảm** (*degradation problem*): khi tăng chiều sâu vượt một ngưỡng nhất định, mạng sâu hơn lại có sai số *huấn luyện* cao hơn mạng nông, dù về lý thuyết có dung lượng lớn hơn.

**Nguyên nhân:** Trong các mạng quá sâu (trên khoảng 20 tầng), dù đã khởi tạo cẩn thận và dùng chuẩn hóa batch, việc tối ưu trực tiếp một chuỗi dài các biến đổi phi tuyến vẫn gặp khó khăn do gradient suy giảm hoặc khó học ánh xạ đồng nhất.

**Giải pháp:** **Kết nối dư** (*residual connections* hay *skip connections*). Thay vì học trực tiếp ánh xạ mong muốn $$H(\mathbf{x})$$, khối mạng học phần dư $$F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$$ rồi cộng đầu vào:

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

**Vì sao cơ chế này hiệu quả:**
1. **Dễ học ánh xạ đồng nhất:** Nếu một tầng chưa cần biến đổi đầu vào, mạng chỉ cần đẩy trọng số của $$F(\mathbf{x})$$ về gần 0.
2. **Đường truyền trực tiếp cho gradient:** Nhánh cộng đồng nhất giúp gradient truyền ngược ổn định qua hàng chục đến hàng trăm tầng.
3. **Hiệu ứng tổ hợp (*ensemble*):** Mạng có kết nối tắt hoạt động tương tự sự kết hợp của nhiều đường dẫn có độ sâu khác nhau.

**Khối Dư Cơ bản:**
```
         ┌────────────────────────────────────┐
         │               Identity             │
         │                                    │
    x ───┼──→ Conv 3×3 → BN → ReLU           │
         │        ↓                           │
         │   Conv 3×3 → BN                    │
         │        ↓                           │
         └───────⊕───→ ReLU → output
              (add)
```

**Khối Cổ chai** (*Bottleneck*, dùng cho mạng sâu như ResNet-50 trở lên):
```
         ┌──────────────────────────────────┐
         │             Identity             │
         │                                  │
    x ───┼──→ Conv 1×1 → BN → ReLU (giảm)  │
         │        ↓                         │
         │   Conv 3×3 → BN → ReLU           │
         │        ↓                         │
         │   Conv 1×1 → BN (mở rộng)        │
         │        ↓                         │
         └───────⊕───→ ReLU → output
```

```python
class BasicBlock(nn.Module):
    """Khối dư cơ bản cho ResNet-18/34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Để khớp kích thước
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Xử lý lệch kích thước
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Then chốt: kết nối bỏ qua!
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Khối nút thắt cho ResNet-50/101/152"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # 1×1 giảm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3×3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1×1 mở rộng
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """Cài đặt ResNet"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.in_channels = 64
        
        # Tầng conv ban đầu
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Các tầng dư
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling + bộ phân loại
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# Tạo các biến thể ResNet chuẩn
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# ============== SO SÁNH MÔ HÌNH ==============
print("=" * 60)
print("So sánh Kiến trúc CNN")
print("=" * 60)

models = {
    'LeNet-5': LeNet5(10),
    'VGG-16': VGG16(1000),
    'ResNet-18': resnet18(1000),
    'ResNet-50': resnet50(1000),
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:12s}: {params:>15,} parameters")
```

### Tóm tắt Tiến hóa Kiến trúc

| Mô hình | Năm | Độ sâu | Tham số | Lỗi Top-5 | Đổi mới chính |
|-------|------|-------|--------|-------------|----------------|
| LeNet-5 | 1998 | 5 | 60K | - | CNN ứng dụng thành công sớm nhất |
| AlexNet | 2012 | 8 | 60M | 15,3% | ReLU, Dropout, huấn luyện GPU |
| VGG-16 | 2014 | 16 | 138M | 7,3% | Chỉ dùng nhân 3×3, tăng chiều sâu |
| GoogLeNet | 2014 | 22 | 6,8M | 6,7% | Mô-đun Inception |
| ResNet-50 | 2015 | 50 | 25M | 3,6% | Kết nối tắt (*skip connections*) |
| ResNet-152 | 2015 | 152 | 60M | 3,0% | Mạng rất sâu |
| DenseNet | 2017 | 121 | 8M | 4,2% | Kết nối dày đặc giữa các tầng |
| EfficientNet | 2019 | - | 5,3M | 2,9% | Tìm kiếm kiến trúc và compound scaling |

---

## 5. Các Bài báo Nền tảng

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Giới thiệu LeNet-5 và thiết lập khuôn mẫu `conv → pool → conv → pool` cho CNN, chứng minh khả năng học đầu–cuối bằng gradient trực tiếp từ điểm ảnh thô thay cho các đặc trưng thủ công.

**["Network In Network" (2014)](https://arxiv.org/abs/1312.4400)**  
*Tác giả*: Min Lin, Qiang Chen, Shuicheng Yan  
Đề xuất tích chập 1×1 để kết hợp thông tin xuyên kênh và Global Average Pooling (GAP) thay thế các tầng liên kết đầy đủ cỡ lớn, giúp giảm mạnh số tham số và trở thành chuẩn trong các kiến trúc về sau.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Tác giả*: Karen Simonyan, Andrew Zisserman  
Chứng minh rằng mạng sâu sử dụng bộ lọc 3×3 nhỏ vượt trội mạng nông dùng bộ lọc lớn. Kiến trúc đơn giản, đồng nhất của VGGNet trở thành một mốc tham chiếu quen thuộc cho học chuyển giao (*transfer learning*).

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Đề xuất kết nối tắt giúp khắc phục hiện tượng suy giảm ở mạng sâu và huấn luyện thành công mạng hơn 100 tầng, đạt lỗi top-5 là 3,57% trên ImageNet 2015. Nguyên lý học phần dư sau đó được áp dụng rộng rãi trong Transformer, mô hình khuếch tán và hầu hết các kiến trúc sâu khác.

**["Densely Connected Convolutional Networks" (2017)](https://arxiv.org/abs/1608.06993)**  
*Tác giả*: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
DenseNet nối đầu ra của mỗi tầng tới tất cả các tầng sau trong cùng khối, khuyến khích tái sử dụng đặc trưng và tăng cường luồng gradient với số lượng tham số gọn nhẹ.

**["EfficientNet: Rethinking Model Scaling" (2019)](https://arxiv.org/abs/1905.11946)**  
*Tác giả*: Mingxing Tan, Quoc V. Le  
Sử dụng tìm kiếm kiến trúc neuron để xây dựng mạng cơ sở hiệu quả, sau đó đề xuất mở rộng đồng thời cả độ sâu, độ rộng và độ phân giải (*compound scaling*) để đạt độ chính xác cao với ít tham số và FLOPs hơn.

---

## 6. Điểm Then chốt

1. **Tầng gộp** giảm kích thước không gian trong khi giữ lại thông tin đặc trưng chính; max pooling thường ưu tiên cho phân loại, còn average pooling phù hợp hơn cho các tác vụ cần bảo toàn thông tin nền hoặc ở dạng global pooling.
2. **Global Average Pooling** thay thế các tầng liên kết đầy đủ nặng tham số, giảm khoảng 40 lần tham số ở phần đầu ra và hỗ trợ khái quát hóa tốt hơn.
3. **Lộ trình tiến hóa kiến trúc:** LeNet (đơn giản) → AlexNet (quy mô và GPU) → VGG (độ sâu đồng nhất) → ResNet (kết nối tắt) → EfficientNet (tối ưu hiệu quả).
4. **Xếp chồng tích chập 3×3** thường hiệu quả hơn dùng nhân kích thước lớn về cả số tham số lẫn số lớp phi tuyến.
5. **Kết nối tắt** là thành phần nền tảng để huấn luyện mạng sâu, xuất hiện trong ResNet, DenseNet, U-Net, Transformer và mô hình khuếch tán.
6. **Các mẫu thiết kế phổ biến:**
   - Khối `Conv → BatchNorm → ReLU`
   - Kết nối tắt sau mỗi 2–3 tầng
   - Global Average Pooling thay cho `Flatten + FC`
   - Tích chập có stride để giảm mẫu học được
7. **Mô hình tiền huấn luyện** (*transfer learning*) là điểm khởi đầu thực tế cho hầu hết bài toán thị giác, vì đặc trưng học từ ImageNet chuyển giao tốt sang nhiều miền dữ liệu khác.

**Chương tiếp theo:** Mạng Neuron Hồi quy (RNN) cho dữ liệu tuần tự.

## Video tham khảo

- **Xem trước:** [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, tiếng Anh. Diễn giải trực quan khối dư như sự kết hợp giữa các tầng tích chập kiểu VGG và đường nối tắt.
- [C4W2L04 Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) — DeepLearningAI, tiếng Anh. Giải thích rõ vì sao đường nối tắt giúp tăng độ sâu mà không làm suy giảm quá trình huấn luyện.
- [Lecture 9 | CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) — Stanford University School of Engineering, tiếng Anh. Tổng hợp mạch tiến hóa từ VGG đến ResNet theo dòng thời gian.
