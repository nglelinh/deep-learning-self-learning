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

### VGGNet (2014) — Sự Đơn giản và Độ sâu

VGG của Simonyan và Zisserman cho thấy **độ sâu quan trọng** và **đơn giản thì tốt hơn**.

**Hiểu biết then chốt**: Chỉ dùng tích chập 3×3 xếp chồng sâu.

**Kiến trúc VGG-16**:
```
Input (224×224×3)

Block 1: Conv3-64, Conv3-64, MaxPool → 112×112×64
Block 2: Conv3-128, Conv3-128, MaxPool → 56×56×128
Block 3: Conv3-256, Conv3-256, Conv3-256, MaxPool → 28×28×256
Block 4: Conv3-512, Conv3-512, Conv3-512, MaxPool → 14×14×512
Block 5: Conv3-512, Conv3-512, Conv3-512, MaxPool → 7×7×512

Flatten → FC 4096 → FC 4096 → FC 1000
```

**Vì sao 3×3 tối ưu**:
- Hai conv 3×3 = một trường tiếp nhận 5×5, nhưng:
  - Ít tham số hơn: 2×(3×3) = 18 so với 5×5 = 25
  - Nhiều phi tuyến hơn: 2 ReLU so với 1
- Ba conv 3×3 = một trường tiếp nhận 7×7

**~138 triệu tham số** (chủ yếu ở các tầng FC)

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

### ResNet (2015) — Cách mạng Kết nối Bỏ qua

ResNet của He và cộng sự giải quyết **bài toán suy giảm** (*degradation problem*): mạng sâu hơn có lỗi huấn luyện *cao hơn* mạng nông. Điều này phản trực giác — nhiều dung lượng hơn lẽ ra phải giúp, không phải làm hại.

**Vấn đề**: Trong mạng rất sâu, gradient hoặc biến mất hoặc bùng nổ, khiến tối ưu khó khăn. Ngay cả với khởi tạo cẩn thận và chuẩn hóa batch, mạng sâu hơn ~20 tầng vẫn huấn luyện kém.

**Giải pháp**: **Kết nối dư** (*residual connections* / *skip connections*)

Thay vì học $$H(x)$$ trực tiếp, học phần dư $$F(x) = H(x) - x$$, rồi:

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

**Vì sao nó hoạt động**:
1. **Ánh xạ đồng nhất dễ**: Nếu đồng nhất là tối ưu, chỉ cần đặt $$F(x) = 0$$
2. **Đường cao tốc gradient**: Gradient chảy trực tiếp qua kết nối bỏ qua
3. **Hiệu ứng ensemble**: ResNet hành xử như ensemble của các mạng nông hơn

**Khối Dư Cơ bản**:
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

**Khối Nút thắt** (*Bottleneck*, cho mạng sâu hơn như ResNet-50+):
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

| Mô hình | Năm | Độ sâu | Tham số | Lỗi Top-5 | Đổi mới then chốt |
|-------|------|-------|--------|-------------|----------------|
| LeNet-5 | 1998 | 5 | 60K | - | CNN thành công đầu tiên |
| AlexNet | 2012 | 8 | 60M | 15,3% | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 16 | 138M | 7,3% | Chỉ 3×3, độ sâu |
| GoogLeNet | 2014 | 22 | 6,8M | 6,7% | Mô-đun Inception |
| ResNet-50 | 2015 | 50 | 25M | 3,6% | Kết nối bỏ qua |
| ResNet-152 | 2015 | 152 | 60M | 3,0% | Rất sâu |
| DenseNet | 2017 | 121 | 8M | 4,2% | Kết nối dày đặc |
| EfficientNet | 2019 | - | 5,3M | 2,9% | Tìm kiếm kiến trúc neuron |

---

## 5. Bài báo Nền tảng

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Giới thiệu LeNet-5, thiết lập mẫu conv-pool-conv-pool trở thành bản thiết kế cho CNN. Chứng minh học dựa trên gradient đầu-cuối trên điểm ảnh thô, chứng tỏ đặc trưng thủ công không cần thiết cho nhận diện ảnh.

**["Network In Network" (2014)](https://arxiv.org/abs/1312.4400)**  
*Tác giả*: Min Lin, Qiang Chen, Shuicheng Yan  
Giới thiệu tích chập 1×1 cho gộp xuyên kênh và Global Average Pooling (GAP) để thay thế tầng kết nối đầy đủ. Các đổi mới này giảm mạnh tham số và trở thành chuẩn trong mọi kiến trúc sau.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Tác giả*: Karen Simonyan, Andrew Zisserman  
VGGNet chứng minh độ sâu với bộ lọc 3×3 nhỏ vượt trội mạng nông với bộ lọc lớn. Kiến trúc đơn giản, lặp lại trở thành thiết kế tham chiếu và vẫn được dùng rộng rãi cho transfer learning.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Kết nối bỏ qua của ResNet giải quyết bài toán suy giảm, cho phép huấn luyện mạng 100+ tầng. Thắng ImageNet 2015 với lỗi top-5 3,57% (vượt hiệu năng con người). Khung học dư giờ được dùng trong transformer, mô hình khuếch tán, và hầu hết kiến trúc sâu.

**["Densely Connected Convolutional Networks" (2017)](https://arxiv.org/abs/1608.06993)**  
*Tác giả*: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
DenseNet nối mỗi tầng với mọi tầng khác, cho phép tái sử dụng đặc trưng và dòng gradient mạnh. Đạt độ chính xác tiên tiến với ít tham số hơn bằng cách tối đa hóa dòng thông tin qua mạng.

**["EfficientNet: Rethinking Model Scaling" (2019)](https://arxiv.org/abs/1905.11946)**  
*Tác giả*: Mingxing Tan, Quoc V. Le  
Dùng tìm kiếm kiến trúc neuron để tìm kiến trúc baseline hiệu quả, rồi đề xuất compound scaling (độ sâu × độ rộng × độ phân giải) để mở rộng có hệ thống. Đạt độ chính xác tiên tiến với ít tham số và FLOPs hơn rất nhiều.

---

## 6. Điểm then chốt

1. **Gộp giảm kích thước** trong khi giữ đặc trưng quan trọng. Max pooling tốt nhất cho phân loại; average pooling cho các tác vụ khác.

2. **Global Average Pooling** thay thế các tầng FC nặng, giảm tham số ~40× đồng thời cải thiện tổng quát hóa.

3. **Tiến hóa kiến trúc**: LeNet (đơn giản) → AlexNet (quy mô) → VGG (độ sâu) → ResNet (kết nối bỏ qua) → EfficientNet (hiệu quả)

4. **Tích chập 3×3 là tối ưu**: Xếp chồng nhiều bộ lọc nhỏ thay vì dùng bộ lọc lớn.

5. **Kết nối bỏ qua là thiết yếu** để huấn luyện mạng sâu. Chúng xuất hiện trong ResNet, DenseNet, U-Net, Transformer và mô hình khuếch tán.

6. **Thực hành tốt nhất hiện đại**:
   - Conv → BatchNorm → ReLU
   - Kết nối bỏ qua mỗi 2–3 tầng
   - Global Average Pooling thay vì Flatten + FC
   - Tích chập stride cho giảm mẫu học được

7. **Mô hình tiền huấn luyện** (transfer learning) là điểm xuất phát chuẩn cho hầu hết tác vụ thị giác. Tiền huấn luyện ImageNet chuyển giao rất tốt sang các miền khác.

**Chương tiếp theo**: Mạng Neuron Hồi quy cho dữ liệu tuần tự!
