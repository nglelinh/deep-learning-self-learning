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

## 4. Đoạn Code

Cài đặt học chuyển giao đầy đủ:

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

Minh họa trực quan hóa đặc trưng:

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

## 5. Các Khái niệm Liên quan

Học chuyển giao nối với học đa tác vụ (multi-task learning), nơi ta huấn luyện một mô hình duy nhất trên nhiều tác vụ liên quan đồng thời thay vì tuần tự. Học đa tác vụ dùng biểu diễn chia sẻ (tầng sớm/giữa) trong khi duy trì đầu chuyên tác vụ (tầng cuối), tương tự kiến trúc học chuyển giao nhưng với huấn luyện chung. Biểu diễn chia sẻ học đặc trưng hữu ích qua các tác vụ, cung cấp chính quy hóa ẩn (một đặc trưng phải giúp nhiều tác vụ để được giữ) thường cải thiện tổng quát hóa so với huấn luyện đơn tác vụ. Hiểu kết nối này giúp đánh giá rằng học chuyển giao và học đa tác vụ xử lý bài toán tương tự — tận dụng cấu trúc chia sẻ qua các tác vụ liên quan — qua các thủ tục huấn luyện khác nhau.

Mối quan hệ với meta-learning (học cách học) tinh tế hơn nhưng quan trọng. Meta-learning nhằm học một khởi tạo hoặc thuật toán học cho phép thích ứng nhanh sang tác vụ mới với dữ liệu tối thiểu. Model-Agnostic Meta-Learning (MAML), ví dụ, học một khởi tạo cách vài bước gradient khỏi hiệu năng tốt trên bất kỳ tác vụ nào từ một phân phối. Học chuyển giao có thể được xem như dạng đơn giản của meta-learning nơi "meta-training" là tiền huấn luyện trên tác vụ nguồn và "adaptation" là tinh chỉnh trên đích. Các cách tiếp cận meta-learning tinh vi hơn mở rộng ý tưởng này để học thích ứng tốt hơn hoặc xử lý phân phối tác vụ đa dạng hơn.

Thành công của học chuyển giao trong NLP qua mô hình ngôn ngữ tiền huấn luyện minh họa sự tiến hóa chuyên miền của mô hình. Word2Vec và GloVe cung cấp embedding từ tiền huấn luyện, chuyển giao tri thức từ vựng. ELMo cung cấp biểu diễn ngữ cảnh tiền huấn luyện. BERT cách mạng hóa lĩnh vực bằng cách tiền huấn luyện toàn bộ mô hình Transformer trên kho ngữ liệu văn bản khổng lồ qua masked language modeling, rồi tinh chỉnh cho tác vụ cụ thể. GPT đẩy xa hơn với mô hình lớn đến mức tinh chỉnh không luôn cần thiết — few-shot learning qua prompting có thể thích ứng mô hình không cần cập nhật tham số nào. Tiến trình từ chuyển giao embedding sang chuyển giao mô hình hoàn chỉnh sang tránh tinh chỉnh hoàn toàn cho thấy học chuyển giao tiến hóa như thế nào khi mô hình mở rộng quy mô.

Kết nối với curriculum learning cung cấp góc nhìn khác. Học chuyển giao có thể được xem như chương trình hai giai đoạn: trước hết học đặc trưng tổng quát (tác vụ dễ hơn với dữ liệu dồi dào), rồi học đặc trưng chuyên tác vụ (tác vụ khó hơn với dữ liệu hạn chế). Cách tiếp cận theo giai đoạn này phản ánh cách con người học — giáo dục tổng quát trước chuyên môn hóa — và thường hoạt động tốt hơn nhảy trực tiếp vào bài toán khó nhất. Hiểu kết nối này gợi ý ta có thể dùng chuyển giao đa giai đoạn: tiền huấn luyện trên dữ liệu tổng quát (ImageNet), huấn luyện trung gian trên dữ liệu chuyên miền (ảnh y tế rộng), rồi tinh chỉnh trên tác vụ cụ thể (phát hiện ung thư phổi). Chuyển giao theo giai đoạn như vậy đã chứng tỏ hiệu quả trong các miền chuyên biệt.

Cuối cùng, học chuyển giao nối với câu hỏi rộng hơn về hiệu quả mẫu trong học máy. Sự thèm dữ liệu của deep learning — đòi hỏi hàng triệu ví dụ — hạn chế ứng dụng nơi dữ liệu đắt (ảnh y tế, sự kiện hiếm) hoặc bất khả thu thập quy mô lớn (dữ liệu riêng tư, kịch bản độc nhất). Học chuyển giao cải thiện đáng kể hiệu quả mẫu bằng cách khấu hao chi phí học đặc trưng tổng quát qua nhiều tác vụ downstream. Hiểu hiệu quả mẫu của học chuyển giao cung cấp hiểu biết về điều gì khiến học khó (học đặc trưng tổng quát đòi hỏi nhiều dữ liệu) so với dễ hơn (học ánh xạ chuyên tác vụ cho đặc trưng tốt cần ít dữ liệu hơn), thông tin khi nào kỳ vọng chuyển giao giúp kịch tính nhất.

## 6. Các Bài báo Nền tảng

**["How transferable are features in deep neural networks?" (2014)](https://arxiv.org/abs/1411.1792)**  
*Tác giả*: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hector Lipson  
Bài báo này điều tra có hệ thống tính chuyển giao của đặc trưng mạng neuron qua các tác vụ, cung cấp bằng chứng thực nghiệm và hiểu biết lý thuyết về khi nào chuyển giao hoạt động. Các tác giả huấn luyện mạng trên biến thể ImageNet, đóng băng số tầng khác nhau khi chuyển giao sang tác vụ mới, đo suy giảm hiệu năng so với tinh chỉnh đầy đủ. Phát hiện then chốt: tầng sớm học đặc trưng tổng quát chuyển giao gần như phổ quát; tầng giữa chuyên tác vụ hơn nhưng vẫn hữu ích rộng; tầng cuối chuyên tác vụ cao và hưởng lợi nhiều nhất từ thích ứng. Bài báo cũng chỉ ra đặc trưng đồng thích ứng (đặc trưng hoạt động tốt cùng nhau) có thể bị phá vỡ bằng cách đóng băng một số trong khi huấn luyện số khác, gợi ý tinh chỉnh mọi tầng thường hoạt động tốt hơn đóng băng nhiều. Công trình thiết lập nền tảng thực nghiệm cho thực hành tốt nhất học chuyển giao và chứng minh tính chuyển giao đặc trưng không phổ quát mà phụ thuộc độ sâu tầng và độ tương tự tác vụ.

**["Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks" (2014)](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)**  
*Tác giả*: Maxime Oquab, Leon Bottou, Ivan Laptev, Josef Sivic  
Bài báo này chứng minh đặc trưng CNN tiền huấn luyện trên ImageNet chuyển giao hiệu quả sang các tác vụ nhận diện thị giác đa dạng bao gồm phát hiện đối tượng, phân loại cảnh, và nhận diện fine-grained. Các tác giả chỉ ra rằng đơn giản dùng tầng conv tiền huấn luyện như bộ trích xuất đặc trưng và huấn luyện bộ phân loại trên các đặc trưng này đạt hiệu năng mạnh qua các tác vụ, vượt trội đặc trưng thiết kế thủ công. Công trình thiết lập học chuyển giao như thực hành chuẩn thực tiễn trong thị giác máy tính, cho thấy đặc trưng học trên một tập dữ liệu lớn (ImageNet) tổng quát hóa sang nhiều tác vụ thị giác khác. Phương pháp thực nghiệm của bài báo — đánh giá có hệ thống qua nhiều tác vụ với so sánh kiểm soát — đặt chuẩn để chứng minh hiệu quả chuyển giao và ảnh hưởng việc áp dụng rộng rãi mô hình tiền huấn luyện trong thị giác máy tính.

**["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805)**  
*Tác giả*: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
Dù chủ yếu giới thiệu BERT, bài báo này cách mạng hóa học chuyển giao trong NLP bằng cách chỉ ra rằng tiền huấn luyện Transformer trên văn bản khổng lồ qua masked language modeling, rồi tinh chỉnh trên tác vụ cụ thể, đạt kết quả tiên tiến qua mười một tác vụ NLP đa dạng bao gồm trả lời câu hỏi, suy diễn ngôn ngữ tự nhiên, và nhận diện thực thể có tên. BERT chứng minh sức mạnh của học chuyển giao ở quy mô: một mô hình tiền huấn luyện duy nhất có thể được thích ứng cho các tác vụ rất khác nhau với thay đổi kiến trúc tối thiểu (chỉ thêm đầu chuyên tác vụ đơn giản). Mô hình pre-train-then-fine-tune trở nên thống trị trong NLP, cho thấy học chuyển giao không chỉ dành cho thị giác mà là nguyên tắc tổng quát áp dụng qua các modality. Tác động của bài báo mở rộng vượt ngoài chính BERT sang thiết lập tiền huấn luyện không giám sát quy mô lớn như bước đầu chuẩn trong phát triển mô hình NLP.

**["A Survey on Transfer Learning" (2010)](https://ieeexplore.ieee.org/document/5288526)**  
*Tác giả*: Sinno Jialin Pan, Qiang Yang  
Bài tổng quan toàn diện này tổ chức và phân loại các cách tiếp cận học chuyển giao qua học máy, không chỉ deep learning. Pan và Yang định nghĩa: inductive transfer (dữ liệu đích gán nhãn), transductive transfer (không có dữ liệu đích gán nhãn), và unsupervised transfer. Họ phân tích khi nào chuyển giao hoạt động (nguồn và đích chia sẻ phân phối biên hoặc có điều kiện) so với thất bại (dịch miền lớn), cung cấp khung lý thuyết để hiểu tính chuyển giao. Dù trước các thành công kịch tính của học chuyển giao kỷ nguyên deep learning, các nền tảng lý thuyết vẫn liên quan: hiểu chuyển giao như tận dụng cấu trúc chia sẻ giữa nguồn và đích, phân tích dịch miền định lượng, và nhận ra chuyển giao tiêu cực (nơi dùng dữ liệu nguồn hại hiệu năng đích) có thể xảy ra khi các miền quá khác nhau. Tổng quan nối học chuyển giao sâu với các truyền thống học máy rộng hơn, cung cấp bối cảnh lý thuyết vì sao và khi nào chuyển giao hiệu quả.

**["Rethinking ImageNet Pre-training" (2019)](https://arxiv.org/abs/1811.08883)**  
*Tác giả*: Kaiming He, Ross Girshick, Piotr Dollár  
Bài báo này thách thức quan điểm thông thường rằng tiền huấn luyện ImageNet luôn có lợi, chỉ ra rằng với tác vụ có đủ dữ liệu (hàng chục nghìn ảnh), huấn luyện từ đầu có thể khớp hoặc vượt tinh chỉnh mô hình tiền huấn luyện, với đủ thời gian huấn luyện. Hiểu biết then chốt là lợi thế của tiền huấn luyện chủ yếu ở hội tụ nhanh hơn và hiệu năng tốt hơn với dữ liệu hạn chế, không ở đạt nghiệm căn bản tốt hơn. Với dữ liệu chuyên tác vụ dồi dào và chính quy hóa đúng, khởi tạo ngẫu nhiên có thể hoạt động tốt, dù đòi hỏi huấn luyện lâu hơn nhiều. Bài báo tinh chỉnh hiểu biết về khi nào chuyển giao giúp nhiều nhất: trong chế độ dữ liệu thấp (hàng trăm đến hàng nghìn ví dụ), tiền huấn luyện mang lại lợi thế khổng lồ; trong chế độ dữ liệu cao (hàng trăm nghìn+), lợi thế giảm. Góc nhìn tinh tế này giúp thực hành viên quyết định có hiểu biết về việc dùng mô hình tiền huấn luyện hay huấn luyện từ đầu dựa trên tính sẵn có dữ liệu và ngân sách tính toán.

## Bẫy thường gặp và Mẹo

Sai lầm phổ biến nhất là dùng mô hình tiền huấn luyện mà không khớp tiền xử lý với giao thức tiền huấn luyện. Nếu mô hình được tiền huấn luyện trên ImageNet với chuẩn hóa cụ thể (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] cho kênh RGB), tinh chỉnh hoặc trích xuất đặc trưng phải dùng chuẩn hóa giống hệt. Tiền xử lý không khớp khiến mô hình nhận đầu vào từ phân phối khác so với lúc huấn luyện, suy giảm hiệu năng đáng kể. Luôn kiểm tra giao thức tiền huấn luyện (kích thước đầu vào, thống kê chuẩn hóa, bước tiền xử lý) và tái lập chính xác cho chuyển giao.

Dùng learning rate quá cao khi tinh chỉnh phá hủy đặc trưng tiền huấn luyện trước khi chúng có thể thích ứng với tác vụ mới. Trọng số tiền huấn luyện biểu diễn đặc trưng hữu ích; cập nhật lớn có thể đẩy chúng xa khỏi vùng hữu ích này vào lãnh thổ ngẫu nhiên. Quy tắc tốt: dùng learning rate nhỏ hơn 10–100× cho tinh chỉnh so với huấn luyện từ đầu. Nếu huấn luyện bình thường dùng lr=0.1, tinh chỉnh nên dùng lr=0.001–0.01. Tốt hơn nữa: dùng learning rate vi phân với tầng sớm nhận rate nhỏ hơn (chúng tổng quát hơn, nên thay đổi ít hơn) và tầng muộn nhận rate lớn hơn (chuyên tác vụ hơn, nên thích ứng nhiều hơn).

Quên đặt mô hình ở chế độ eval khi trích xuất đặc trưng là lỗi tinh tế gây kết quả không nhất quán. Nếu mô hình chứa tầng batch normalization hoặc dropout và vẫn ở chế độ train khi trích xuất đặc trưng, các tầng này hành xử khác nhau trên các lần gọi khác nhau (batch norm dùng thống kê batch, dropout loại đơn vị ngẫu nhiên), khiến cùng đầu vào sinh đặc trưng khác nhau. Luôn gọi `model.eval()` và dùng `torch.no_grad()` khi trích xuất đặc trưng hoặc đưa ra dự đoán.

Khi tinh chỉnh mô hình NLP như BERT, vấn đề phổ biến là catastrophic forgetting trên chuỗi ngắn. BERT được tiền huấn luyện trên chuỗi 512 token. Tinh chỉnh trên tác vụ với chuỗi ngắn (tweet, tin nhắn SMS 20–50 token) có thể suy giảm khả năng xử lý chuỗi dài của mô hình. Nếu bạn cần bảo toàn năng lực này, bao gồm ví dụ chuỗi dài trong tinh chỉnh hoặc dùng hỗn hợp dữ liệu nguồn và đích (tinh chỉnh một phần).

Một mẹo mạnh cho chuyển giao tốt hơn là dùng cyclical learning rate trong tinh chỉnh. Bắt đầu với learning rate thấp, tăng dần lên đỉnh vừa phải, rồi giảm lại. Điều này cho phép thích ứng nhẹ ban đầu (không phá hủy đặc trưng tiền huấn luyện), cập nhật mạnh hơn ở đỉnh (tìm đặc trưng chuyên tác vụ), và tinh chỉnh cuối cùng (fine-tune đặc trưng đã thích ứng). Kết hợp với gradual unfreezing (bắt đầu chỉ huấn luyện đầu, rồi mở băng tầng trên, rồi tầng giữa), điều này cung cấp thích ứng mượt từ đặc trưng tiền huấn luyện sang chuyên tác vụ.

Với tác vụ rất khác miền tiền huấn luyện, tinh chỉnh một phần thường hoạt động tốt hơn tinh chỉnh đầy đủ. Đóng băng tầng sớm (tổng quát nhất, ít khả năng cần thích ứng), tinh chỉnh tầng giữa và muộn. Điều này bảo toàn đặc trưng mức thấp phổ quát trong khi thích ứng đặc trưng mức cao chuyên tác vụ. Lựa chọn nơi đóng băng liên quan đến thử nghiệm nhưng theo nguyên tắc: đóng băng điều chuyển giao tốt, thích ứng điều cần học chuyên tác vụ.

## Điểm then chốt

Học chuyển giao tận dụng mô hình tiền huấn luyện để đạt hiệu năng mạnh trên tác vụ đích với dữ liệu hạn chế bằng cách chuyển giao đặc trưng đã học từ tác vụ nguồn liên quan với dữ liệu dồi dào. Bản chất phân cấp của đặc trưng mạng sâu — đặc trưng mức thấp tổng quát ở tầng sớm, đặc trưng mức cao chuyên tác vụ ở tầng muộn — cho phép chuyển giao chọn lọc nơi ta giữ đặc trưng tổng quát hữu ích và thích ứng thành phần chuyên tác vụ. Trích xuất đặc trưng đóng băng trọng số tiền huấn luyện và chỉ huấn luyện đầu chuyên tác vụ mới, hoạt động tốt với dữ liệu rất hạn chế (hàng trăm ví dụ) và chi phí tính toán tối thiểu. Tinh chỉnh thích ứng mọi hoặc hầu hết tầng với learning rate nhỏ, thường đạt hiệu năng tốt hơn với dữ liệu vừa phải (hàng nghìn ví dụ) bằng cách chuyên hóa đặc trưng tiền huấn luyện cho miền đích. Learning rate vi phân theo tầng điều khiển thích ứng, với tầng sớm thay đổi tối thiểu (bảo toàn đặc trưng tổng quát) và tầng muộn thay đổi nhiều hơn (học đặc trưng chuyên tác vụ), ngăn catastrophic forgetting trong khi cho phép chuyên hóa hiệu quả. Hiệu quả của học chuyển giao phụ thuộc độ tương tự nguồn–đích — miền tương tự hơn cho phép chuyển giao tốt hơn — và chất lượng tiền huấn luyện — hiệu năng tác vụ nguồn tốt hơn thường cải thiện chuyển giao. Thực hành hiện đại trong thị giác máy tính bắt đầu với tiền huấn luyện ImageNet, trong NLP với tiền huấn luyện BERT/GPT, và trong giọng nói với tiền huấn luyện Wav2Vec, khiến học chuyển giao trở thành bước đầu chuẩn thay vì kỹ thuật nâng cao. Hiểu sâu học chuyển giao nghĩa là nhận ra nó như khấu hao chi phí học biểu diễn tổng quát qua nhiều tác vụ, dân chủ hóa deep learning bằng cách khiến hiệu năng cao khả thi không cần tập dữ liệu chuyên tác vụ khổng lồ, và thể hiện nguyên tắc rằng biểu diễn tốt học trên một tác vụ thường giúp trên các tác vụ liên quan — một dạng tái sử dụng tri thức nền tảng cho học hiệu quả.

Học chuyển giao minh họa cách deep learning trưởng thành từ đòi hỏi tập dữ liệu khổng lồ cho mọi tác vụ sang cho phép hiệu năng mạnh với dữ liệu chuyên tác vụ hạn chế qua tái sử dụng chiến lược tri thức đã học.
