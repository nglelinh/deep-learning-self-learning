---
layout: post
title: 04-01-03-02 Tích chập PyTorch và Khối CNN
chapter: '04'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

### Cài đặt PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionExplained(nn.Module):
    """
    Một tầng tích chập kèm giải thích chi tiết.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Tạo tầng tích chập
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
    def forward(self, x):
        """
        Lượt xuôi kèm theo dõi shape.
        
        Args:
            x: Tensor đầu vào dạng (batch, channels, height, width)
        """
        batch_size, C_in, H_in, W_in = x.shape
        
        # Tính kích thước đầu ra
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Áp dụng tích chập
        out = self.conv(x)
        
        # Kiểm tra shape
        assert out.shape == (batch_size, self.out_channels, H_out, W_out)
        
        return out
    
    def count_parameters(self):
        """Đếm và giải thích tham số."""
        # Shape trọng số: (out_channels, in_channels, kernel_size, kernel_size)
        weight_params = self.out_channels * self.in_channels * self.kernel_size**2
        bias_params = self.out_channels if self.conv.bias is not None else 0
        
        total = weight_params + bias_params
        
        print(f"Phân rã tham số:")
        print(f"  Trọng số: {self.out_channels} bộ lọc × "
              f"({self.kernel_size}×{self.kernel_size}×{self.in_channels}) = {weight_params}")
        print(f"  Bias: {bias_params}")
        print(f"  Tổng: {total}")
        
        return total


# ============== DEMO ==============

print("=" * 60)
print("Ví dụ Tích chập PyTorch")
print("=" * 60)

# Ví dụ 1: Tích chập cơ bản
print("\n--- Ví dụ 1: Ảnh RGB thành 64 Bản đồ Đặc trưng ---")
conv_layer = ConvolutionExplained(
    in_channels=3,      # Đầu vào RGB
    out_channels=64,    # 64 bộ lọc học được
    kernel_size=3,      # Bộ lọc 3×3
    stride=1,
    padding=1           # Đệm "same"
)

# Đầu vào: batch 8 ảnh RGB, 32×32 điểm ảnh
x = torch.randn(8, 3, 32, 32)
output = conv_layer(x)

print(f"Shape đầu vào:  {x.shape}")
print(f"Shape đầu ra: {output.shape}")
conv_layer.count_parameters()


# Ví dụ 2: Giảm mẫu bằng tích chập có stride
print("\n--- Ví dụ 2: Tích chập Stride để Giảm mẫu ---")
conv_downsample = ConvolutionExplained(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=2,           # Stride 2 giảm một nửa kích thước không gian
    padding=1
)

x2 = torch.randn(8, 64, 32, 32)
output2 = conv_downsample(x2)

print(f"Shape đầu vào:  {x2.shape}")
print(f"Shape đầu ra: {output2.shape}")  # 32×32 → 16×16
conv_downsample.count_parameters()


# Ví dụ 3: Tích chập 1×1 để thao tác kênh
print("\n--- Ví dụ 3: Tích chập 1×1 (Giảm kênh) ---")
conv_1x1 = ConvolutionExplained(
    in_channels=512,
    out_channels=64,    # Giảm kênh mạnh
    kernel_size=1,
    stride=1,
    padding=0
)

x3 = torch.randn(8, 512, 16, 16)
output3 = conv_1x1(x3)

print(f"Shape đầu vào:  {x3.shape}")
print(f"Shape đầu ra: {output3.shape}")  # Cùng không gian, ít kênh hơn
conv_1x1.count_parameters()


# Ví dụ 4: Quan sát bộ lọc đã học
print("\n--- Ví dụ 4: Kiểm tra Trọng số Bộ lọc ---")
# Truy cập trọng số đã học
weights = conv_layer.conv.weight.data
print(f"Shape tensor trọng số: {weights.shape}")
print(f"  = {weights.shape[0]} bộ lọc kích thước "
      f"{weights.shape[2]}×{weights.shape[3]}×{weights.shape[1]}")

# Quan sát bộ lọc đầu tiên (sẽ hiển thị bằng matplotlib)
first_filter = weights[0]  # Shape: (3, 3, 3) = (in_channels, H, W)
print(f"Shape bộ lọc đầu: {first_filter.shape}")
print(f"Giá trị bộ lọc (kênh 0):\n{first_filter[0]}")
```

### Khối CNN Hoàn chỉnh với Thực hành Tốt nhất

```python
class ConvBlock(nn.Module):
    """
    Khối tích chập chuẩn: Conv → BatchNorm → ReLU
    
    Mẫu này dùng trong hầu hết CNN hiện đại.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super().__init__()
        
        # Tích chập (không bias khi dùng BatchNorm)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        # Chuẩn hóa batch ổn định huấn luyện
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Kích hoạt ReLU
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Khối dư với kết nối bỏ qua (từ ResNet).
    
    Hiểu biết then chốt: Học phần dư F(x) = H(x) - x thay vì H(x) trực tiếp.
    Đầu ra: y = F(x) + x
    """
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x  # Lưu đầu vào cho kết nối bỏ qua
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Kết nối bỏ qua!
        out = self.relu(out)
        
        return out


class SimpleCNN(nn.Module):
    """
    Một CNN đơn giản nhưng hiệu quả cho phân loại ảnh.
    
    Kiến trúc: 
    Input → ConvBlocks (kèm pooling) → Global Average Pool → Classifier
    """
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        # Các tầng trích xuất đặc trưng
        self.features = nn.Sequential(
            # Khối 1: 32×32×3 → 16×16×32
            ConvBlock(input_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),
            
            # Khối 2: 16×16×32 → 8×8×64
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Khối 3: 8×8×64 → 4×4×128
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling: 4×4×128 → 128
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bộ phân loại
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Trích xuất đặc trưng
        x = self.features(x)
        
        # Gộp toàn cục
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Làm phẳng
        
        # Phân loại
        x = self.classifier(x)
        
        return x


# ============== VÍ DỤ HUẤN LUYỆN ==============

def train_step(model, images, labels, optimizer, criterion):
    """Một bước huấn luyện với lượt xuôi và ngược."""
    model.train()
    
    # Lượt xuôi
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Lượt ngược
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Tính độ chính xác
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).float().mean()
    
    return loss.item(), accuracy.item()


# Tạo mô hình và các thành phần huấn luyện
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Đếm tham số
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nThống kê Mô hình:")
print(f"  Tổng tham số: {total_params:,}")
print(f"  Tham số huấn luyện được: {trainable_params:,}")

# Mô phỏng một bước huấn luyện
dummy_images = torch.randn(32, 3, 32, 32)  # Batch 32 ảnh kiểu CIFAR
dummy_labels = torch.randint(0, 10, (32,))

loss, acc = train_step(model, dummy_images, dummy_labels, optimizer, criterion)
print(f"  Bước huấn luyện - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

---
