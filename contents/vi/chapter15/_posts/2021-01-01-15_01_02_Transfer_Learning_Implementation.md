---
layout: post
title: 15-01-02 Cài đặt Học Chuyển giao
chapter: '15'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter15
---

## 4. Đoạn mã

Đoạn mã dưới đây cài đặt đầy đủ hai chiến lược học chuyển giao (trích xuất đặc trưng và tinh chỉnh với learning rate phân biệt theo tầng) trên ResNet-18 tiền huấn luyện ImageNet với một tập con 1000 ảnh của CIFAR-10:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

print("="*70)
print("Học Chuyển giao: Tinh chỉnh ResNet Tiền huấn luyện cho Tập Dữ liệu Tùy chỉnh")
print("="*70)

# Tải ResNet18 tiền huấn luyện (nhỏ hơn ResNet50 để minh họa)
print("\n1. Tải Mô hình Tiền huấn luyện")
print("-" * 70)

# weights='IMAGENET1K_V1' tải trọng số tiền huấn luyện ImageNet
model_pretrained = models.resnet18(weights='IMAGENET1K_V1')

print(f"Đã tải ResNet18 tiền huấn luyện trên ImageNet")
print(f"Tầng đầu ra gốc: {model_pretrained.fc}")
print(f"  (1000 lớp cho ImageNet)")

# Xem xét điều mô hình tiền huấn luyện đã học
print(f"\nĐặc trưng tiền huấn luyện:")
print(f"  Bộ lọc tầng 1: {model_pretrained.conv1.weight.shape}")  # (64, 3, 7, 7)
print(f"  Đây là bộ phát hiện cạnh/texture học từ ImageNet")

# 2. Thích ứng cho tác vụ mới
print("\n2. Thích ứng cho Tác vụ Tùy chỉnh (10 Lớp)")
print("-" * 70)

# Thay tầng fully-connected cuối cho tác vụ của ta
# Mọi thứ khác giữ trọng số tiền huấn luyện
num_classes = 10  # Tác vụ tùy chỉnh của ta có 10 lớp
num_features = model_pretrained.fc.in_features  # Lấy kích thước đầu vào tầng fc

print(f"Đặc trưng đầu vào FC gốc: {num_features}")
print(f"Thay tầng cuối cho {num_classes} lớp...")

model_pretrained.fc = nn.Linear(num_features, num_classes)

print(f"Tầng đầu ra mô hình mới: {model_pretrained.fc}")
print("  (10 lớp cho tác vụ tùy chỉnh của ta)")

# Tạo tập dữ liệu giả (trong thực tiễn, dùng dữ liệu thật của bạn)
# Ta sẽ mô phỏng với CIFAR-10 như tác vụ "tùy chỉnh"
print("\n3. Chuẩn bị Tập Dữ liệu Tùy chỉnh")
print("-" * 70)

transform_train = transforms.Compose([
    transforms.Resize(224),  # ResNet mong đợi 224×224 (CIFAR là 32×32)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Thống kê ImageNet
])

# Mô phỏng kịch bản dữ liệu hạn chế: chỉ dùng 1000 ảnh huấn luyện
full_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)

# Lấy tập con để mô phỏng dữ liệu hạn chế
limited_size = 1000
remaining = len(full_dataset) - limited_size
train_dataset, _ = random_split(full_dataset, [limited_size, remaining])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_train)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Huấn luyện với chỉ {limited_size} ảnh (mô phỏng dữ liệu hạn chế)")
print(f"Tập kiểm tra: {len(test_dataset)} ảnh")

# 3. Chiến lược Huấn luyện: Trích xuất Đặc trưng so với Tinh chỉnh
print("\n4. Chiến lược A: Trích xuất Đặc trưng (Đóng băng Tầng Tiền huấn luyện)")
print("-" * 70)

# Tạo bản sao mô hình cho trích xuất đặc trưng
model_features = models.resnet18(weights='IMAGENET1K_V1')
model_features.fc = nn.Linear(num_features, num_classes)

# Đóng băng mọi tầng trừ FC cuối
for param in model_features.parameters():
    param.requires_grad = False

# Mở băng tầng cuối
for param in model_features.fc.parameters():
    param.requires_grad = True

trainable_params_fe = sum(p.numel() for p in model_features.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_features.parameters())

print(f"Tổng tham số: {total_params:,}")
print(f"Tham số huấn luyện được: {trainable_params_fe:,} ({trainable_params_fe/total_params*100:.1f}%)")
print("Chỉ huấn luyện tầng phân loại cuối!")

# Optimizer cho trích xuất đặc trưng (chỉ tham số tầng fc)
optimizer_fe = optim.Adam(model_features.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Huấn luyện
model_features.train()
print("\nHuấn luyện mô hình trích xuất đặc trưng (5 epoch)...")

for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer_fe.zero_grad()
        
        # Forward (tầng conv đóng băng, chỉ fc huấn luyện)
        outputs = model_features(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer_fe.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc = 100. * correct / total
    print(f"  Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, "
          f"Train Acc = {train_acc:.2f}%")

# Đánh giá
model_features.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model_features(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

test_acc_fe = 100. * correct / total
print(f"\nĐộ chính xác Kiểm tra Trích xuất Đặc trưng: {test_acc_fe:.2f}%")

# 4. Chiến lược B: Tinh chỉnh
print("\n5. Chiến lược B: Tinh chỉnh (Cập nhật Mọi Tầng)")
print("-" * 70)

model_finetune = models.resnet18(weights='IMAGENET1K_V1')
model_finetune.fc = nn.Linear(num_features, num_classes)

# Mọi tham số huấn luyện được
trainable_params_ft = sum(p.numel() for p in model_finetune.parameters())
print(f"Tham số huấn luyện được: {trainable_params_ft:,} (100%)")

# Dùng learning rate vi phân
# LR thấp hơn cho tầng tiền huấn luyện, cao hơn cho tầng mới
optimizer_ft = optim.Adam([
    {'params': model_finetune.layer1.parameters(), 'lr': 0.0001},
    {'params': model_finetune.layer2.parameters(), 'lr': 0.0001},
    {'params': model_finetune.layer3.parameters(), 'lr': 0.0002},
    {'params': model_finetune.layer4.parameters(), 'lr': 0.0005},
    {'params': model_finetune.fc.parameters(), 'lr': 0.001}  # Cao nhất cho tầng mới
], lr=0.0001)  # Mặc định cho bất kỳ tham số nào không chỉ định

print("Dùng learning rate vi phân theo tầng:")
print("  Tầng sớm: 0.0001 (hầu như không đổi)")
print("  Tầng giữa: 0.0002")
print("  Tầng sâu: 0.0005")
print("  Tầng FC mới: 0.001 (thay đổi nhiều nhất)")

# Huấn luyện
model_finetune.train()
print("\nHuấn luyện mô hình tinh chỉnh (5 epoch)...")

for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer_ft.zero_grad()
        
        outputs = model_finetune(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer_ft.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc = 100. * correct / total
    print(f"  Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, "
          f"Train Acc = {train_acc:.2f}%")

# Đánh giá
model_finetune.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model_finetune(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

test_acc_ft = 100. * correct / total
print(f"\nĐộ chính xác Kiểm tra Tinh chỉnh: {test_acc_ft:.2f}%")

# So sánh kết quả
print("\n" + "="*70)
print("So sánh Kết quả Học Chuyển giao")
print("="*70)
print(f"Trích xuất Đặc trưng: {test_acc_fe:.2f}% độ chính xác kiểm tra")
print(f"Tinh chỉnh:           {test_acc_ft:.2f}% độ chính xác kiểm tra")
print(f"\nCả hai vượt trội đáng kể so với huấn luyện từ đầu (~60% trên 1000 ảnh)")
print("Học chuyển giao cho phép hiệu năng cạnh tranh với dữ liệu hạn chế!")
```

Đoạn mã tiếp theo minh họa cách trích xuất bản đồ đặc trưng trung gian bằng `forward_hook` để phân tích biểu diễn đã học:

```python
print("\n" + "="*70)
print("Phân tích Đặc trưng Chuyển giao")
print("="*70)

# Trích xuất đặc trưng để phân tích
def extract_features(model, dataloader, layer_name='layer4'):
    """
    Trích xuất đặc trưng từ một tầng cụ thể.
    
    Điều này cho thấy biểu diễn mô hình dùng cho phân loại.
    Đặc trưng tiền huấn luyện nên có nghĩa ngay cả cho tác vụ tùy chỉnh.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Đăng ký hook để bắt đầu ra tầng
    features_hook = []
    def hook_fn(module, input, output):
        features_hook.append(output.detach())
    
    # Lấy tầng
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            _ = model(inputs)
            features_list.append(features_hook[-1])
            labels_list.append(labels)
            features_hook.clear()
    
    handle.remove()
    
    # Nối tất cả batch
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels

# Trích xuất đặc trưng layer4 (tầng conv cuối trước FC)
features_train, labels_train = extract_features(
    model_finetune, 
    DataLoader(train_dataset, batch_size=32), 
    'layer4'
)

print(f"Đã trích xuất đặc trưng từ layer4 (tầng conv cuối)")
print(f"Shape đặc trưng: {features_train.shape}")  # (1000, 512, 7, 7)

# Global average pool để nhận vectơ 512 chiều
features_pooled = features_train.mean(dim=[2,3])  # (1000, 512)

print(f"Sau global average pooling: {features_pooled.shape}")
print(f"\nCác đặc trưng 512 chiều này mã hóa:")
print("  - Mức thấp: cạnh, texture (từ ImageNet)")
print("  - Mức trung: phần đối tượng, hình dạng (từ ImageNet)")
print("  - Mức cao: mẫu chuyên chim (thích ứng trong tinh chỉnh)")
print("\nĐặc trưng tiền huấn luyện cung cấp điểm khởi đầu mạnh!")
```

## 5. Các khái niệm liên quan

Học chuyển giao có liên hệ gần với **học đa tác vụ (*multi-task learning*)**, trong đó một mô hình duy nhất được huấn luyện đồng thời trên nhiều tác vụ liên quan thay vì tuần tự. Học đa tác vụ dùng chung thân mạng ở các tầng sớm và giữa cùng các đầu ra riêng cho từng tác vụ; việc buộc một biểu diễn phục vụ đồng thời nhiều mục tiêu tạo ra cơ chế chính quy hóa tự nhiên giúp cải thiện khả năng tổng quát hóa.

Học chuyển giao cũng liên hệ với **meta-learning** ("học cách học"). Các thuật toán như MAML tìm một điểm khởi tạo sao cho chỉ cần vài bước cập nhật gradient là thích nghi tốt với tác vụ mới. Có thể xem học chuyển giao là trường hợp đơn giản của khung này, trong đó tiền huấn luyện đóng vai trò tạo khởi tạo ban đầu và tinh chỉnh là bước thích nghi.

Trong xử lý ngôn ngữ tự nhiên (NLP), học chuyển giao đã phát triển qua nhiều giai đoạn: từ chuyển giao embedding từ tĩnh (Word2Vec, GloVe), sang biểu diễn ngữ cảnh (ELMo), đến tiền huấn luyện toàn bộ kiến trúc Transformer bằng masked language modeling rồi tinh chỉnh cho từng tác vụ (BERT), và tiếp đến là các mô hình ngôn ngữ lớn dòng GPT cho phép thích nghi vài mẫu (*few-shot*) qua ngữ cảnh prompt mà không cần cập nhật trọng số.

Từ góc nhìn **curriculum learning**, học chuyển giao tương tự lộ trình học theo từng bước: học đặc trưng tổng quát trên tập lớn trước rồi mới chuyên biệt hóa sang bài toán đích. Điều này gợi ý chiến lược chuyển giao nhiều giai đoạn — ví dụ tiền huấn luyện trên ảnh tự nhiên (ImageNet), huấn luyện trung gian trên tập ảnh y khoa tổng quát, rồi mới tinh chỉnh trên bài toán chẩn đoán cụ thể.

Cuối cùng, học chuyển giao giải quyết trực tiếp bài toán **hiệu quả mẫu (*sample efficiency*)**: chi phí học các đặc trưng nền tảng được phân bổ trên nhiều tác vụ về sau, giúp ứng dụng học sâu khả thi ở những lĩnh vực chi phí gán nhãn cao hoặc khó thu thập dữ liệu quy mô lớn.

## 6. Các bài báo nền tảng

**["How transferable are features in deep neural networks?" (2014)](https://arxiv.org/abs/1411.1792)**  
*Tác giả*: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hector Lipson  
Bài báo khảo sát có hệ thống khả năng chuyển giao đặc trưng theo từng độ sâu tầng trên ImageNet. Kết quả chỉ ra các tầng sớm mang tính tổng quát cao và chuyển giao tốt, còn các tầng sâu chuyên biệt hơn cho tác vụ nguồn. Nhóm tác giả cũng ghi nhận hiện tượng các tầng liền kề đồng thích nghi (*co-adaptation*), nên việc đóng băng ở giữa mạng có thể làm giảm hiệu năng so với tinh chỉnh toàn mạng.

**["Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks" (2014)](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)**  
*Tác giả*: Maxime Oquab, Leon Bottou, Ivan Laptev, Josef Sivic  
Bài báo chứng minh đặc trưng từ CNN tiền huấn luyện trên ImageNet chuyển giao hiệu quả sang nhiều bài toán thị giác khác như phát hiện đối tượng, phân loại cảnh và nhận diện chi tiết, vượt trội các đặc trưng thủ công và góp phần đưa học chuyển giao thành chuẩn mực trong thị giác máy tính.

**["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805)**  
*Tác giả*: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
Bài báo giới thiệu BERT và chứng minh mô hình *tiền huấn luyện rồi tinh chỉnh* trên Transformer đạt kết quả cao trên 11 bài toán NLP khác nhau (hỏi đáp, suy diễn ngôn ngữ tự nhiên, nhận diện thực thể) chỉ bằng cách gắn thêm tầng đầu ra đơn giản cho từng tác vụ.

**["A Survey on Transfer Learning" (2010)](https://ieeexplore.ieee.org/document/5288526)**  
*Tác giả*: Sinno Jialin Pan, Qiang Yang  
Bài tổng quan hệ thống hóa các dạng học chuyển giao trong học máy gồm *inductive*, *transductive* và *unsupervised transfer*, phân tích điều kiện để chuyển giao hiệu quả cũng như hiện tượng *chuyển giao tiêu cực (negative transfer)* khi miền nguồn và miền đích lệch nhau quá lớn.

**["Rethinking ImageNet Pre-training" (2019)](https://arxiv.org/abs/1811.08883)**  
*Tác giả*: Kaiming He, Ross Girshick, Piotr Dollár  
Bài báo chỉ ra rằng khi bài toán đích có đủ dữ liệu lớn (hàng chục nghìn ảnh trở lên) và thời gian huấn luyện đủ dài, huấn luyện từ đầu có thể đạt hiệu năng ngang bằng hoặc vượt tinh chỉnh từ ImageNet. Do đó, lợi thế lớn nhất của tiền huấn luyện nằm ở tốc độ hội tụ nhanh và cải thiện rõ rệt trong chế độ ít dữ liệu.

## Bẫy thường gặp và Mẹo

**Không đồng bộ tiền xử lý với giai đoạn tiền huấn luyện** là sai sót phổ biến nhất. Nếu mô hình tiền huấn luyện trên ImageNet dùng chuẩn hóa RGB (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), quá trình trích xuất đặc trưng hoặc tinh chỉnh phải áp dụng đúng kích thước và chuẩn hóa tương ứng để tránh lệch phân phối đầu vào.

**Đặt learning rate quá lớn khi tinh chỉnh** dễ phá vỡ các đặc trưng tiền huấn luyện trước khi kịp thích nghi. Quy tắc thực hành là giảm learning rate 10–100 lần so với huấn luyện từ đầu (ví dụ 0,0001–0,001), hoặc dùng learning rate phân biệt theo tầng (nhỏ ở tầng sớm, lớn hơn ở tầng sâu và tầng đầu ra mới).

**Quên chuyển sang chế độ đánh giá khi trích xuất đặc trưng** khiến batch normalization và dropout hoạt động theo chế độ huấn luyện, tạo ra đầu ra không nhất quán giữa các lần chạy. Luôn gọi `model.eval()` kết hợp `torch.no_grad()` khi trích xuất đặc trưng hoặc suy luận.

**Quên thảm họa theo độ dài chuỗi trong NLP:** BERT được tiền huấn luyện trên chuỗi tới 512 token; nếu chỉ tinh chỉnh trên văn bản rất ngắn (20–50 token), khả năng xử lý chuỗi dài có thể suy giảm. Cần bổ sung dữ liệu dài hơn nếu ứng dụng yêu cầu xử lý văn bản dài.

**Chiến lược mở băng dần (*gradual unfreezing*) và learning rate chu kỳ:** bắt đầu bằng việc chỉ huấn luyện tầng đầu ra, sau đó mở băng dần từ tầng sâu ngược về tầng sớm kết hợp điều chỉnh learning rate giúp mô hình thích nghi êm hơn mà vẫn bảo toàn đặc trưng tổng quát. Khi miền đích khác nhiều so với miền nguồn, tinh chỉnh một phần (đóng băng các tầng sớm, tinh chỉnh tầng giữa và sâu) thường cân bằng tốt giữa độ ổn định và khả năng thích nghi.

## Điểm then chốt

Học chuyển giao khai thác các mô hình tiền huấn luyện để đạt hiệu năng tốt khi dữ liệu đích còn hạn chế. Nhờ tính phân cấp của đặc trưng mạng sâu, ta có thể chọn trích xuất đặc trưng (đóng băng thân mạng, phù hợp khi dữ liệu rất ít) hoặc tinh chỉnh toàn bộ/một phần mạng với learning rate nhỏ hoặc phân biệt theo tầng (phù hợp khi có thêm dữ liệu hoặc cần thích nghi sâu hơn). Hiệu quả chuyển giao phụ thuộc vào chất lượng mô hình tiền huấn luyện và mức độ tương đồng giữa hai miền; đây đã trở thành quy trình khởi đầu tiêu chuẩn trong thị giác máy tính, xử lý ngôn ngữ tự nhiên và xử lý giọng nói.

## Video tham khảo

- **Xem trước:** [Pytorch Transfer Learning and Fine Tuning Tutorial](https://www.youtube.com/watch?v=qaDe0qQZ5AQ) — Aladdin Persson, tiếng Anh. Hướng dẫn tải mô hình tiền huấn luyện, thay tầng đầu ra và huấn luyện trong PyTorch.
- [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, tiếng Anh. Giải thích logic chọn tầng đóng băng tương ứng với phần mã nguồn trong bài.
- [PyTorch Tutorial 14 - Convolutional Neural Network (CNN)](https://www.youtube.com/watch?v=pDdP0TFzsoQ) — Patrick Loeber, tiếng Anh. Ôn lại cấu trúc các tầng trong mạng tích chập nền nếu tên mô-đun còn chưa quen thuộc.
