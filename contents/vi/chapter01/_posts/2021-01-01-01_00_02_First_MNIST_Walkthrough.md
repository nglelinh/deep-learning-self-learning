---
layout: post
title: 01-00-02 Bước đầu với MNIST
chapter: '01'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter01
---

Mọi ngôn ngữ lập trình đều có bài mở đầu `"Hello, World!"`, và trong thế giới Deep Learning, **"Hello, World!" chính là nhận diện chữ số viết tay MNIST**. Bộ dữ liệu gồm 70{,}000 ảnh chữ số từ 0 đến 9 do Yann LeCun và cộng sự tập hợp từ cuối thập niên 1990 đủ nhỏ để chạy trong vài giây trên máy tính cá nhân, nhưng lại chứa đầy đủ mọi thành phần cốt lõi của một hệ thống Học sâu thực thụ: chuẩn hóa tensor, thiết kế kiến trúc nhiều tầng, tính Cross-Entropy và chạy vòng lặp tối ưu. Bài thực hành này dẫn bạn đi từng dòng code để thực sự "chạm tay" vào mạng nơ-ron đầu tiên.

## Kiến thức cần nhớ: Bốn thành phần trong một Pipeline Huấn luyện PyTorch chuẩn

1. **Dữ liệu và Tiền xử lý Tensor (`Dataset`, `DataLoader`, `Transforms`)**:
   Chuyển ảnh PIL từ giá trị nguyên $$[0, 255]$$ sang tensor thực $$[0, 1]$$, rồi chuẩn hóa về trung bình $$0$$ và độ lệch chuẩn $$1$$ (với MNIST là $$\mu \approx 0.1307, \sigma \approx 0.3081$$), đồng thời xáo trộn (*shuffle*) và gom theo mini-batch.

2. **Định nghĩa Kiến trúc (`nn.Module` & `forward`)**:
   Trải phẳng ảnh $$(B, 1, 28, 28)$$ thành ma trận $$(B, 784)$$, đưa qua các lớp `nn.Linear` xen kẽ hàm kích hoạt phi tuyến `ReLU` để xuất ra 10 điểm số thô (*logits*) tương ứng 10 chữ số.

3. **Hàm Mất mát và Bộ Tối ưu (`CrossEntropyLoss`, `Adam` / `SGD`)**:
   `nn.CrossEntropyLoss` tự động kết hợp `LogSoftmax` và `NLLLoss` (ổn định số học hơn việc tự tính `Softmax` rồi lấy `log`), còn optimizer giữ nhiệm vụ cập nhật trọng số dựa trên gradient.

4. **Vòng lặp Huấn luyện 5 bước kinh điển**:
   `optimizer.zero_grad()` $$\to$$ `outputs = model(x)` $$\to$$ `loss = criterion(outputs, y)` $$\to$$ `loss.backward()` $$\to$$ `optimizer.step()`. Quên `zero_grad()` sẽ khiến gradient cộng dồn qua các batch làm mô hình hỏng ngay lập tức!

### Giải thích: Theo dõi kích thước Tensor và số bước cập nhật của 1 Epoch MNIST

Giả sử bạn huấn luyện trên tập train MNIST có $$60{,}000$$ ảnh với `batch_size = 64` qua mạng `784 -> 128 -> 64 -> 10`:

1. **Số bước cập nhật (iterations) trong 1 epoch**:
   $$\left\lceil \frac{60{,}000}{64} \right\rceil = \lceil 937.5 \rceil = 938 \text{ bước}$$ (937 batch đủ 64 ảnh, và batch cuối cùng gồm $$60{,}000 - 937 \times 64 = 32$$ ảnh).
2. **Truy vết kích thước (*shape*) Tensor qua từng lớp ở một batch đầy đủ**:
   - Đầu vào ảnh gốc: `(64, 1, 28, 28)`
   - Sau `x.view(-1, 784)` (trải phẳng): `(64, 784)`
   - Sau `fc1` (`Linear(784, 128)`) + `ReLU`: `(64, 128)`
   - Sau `fc2` (`Linear(128, 64)`) + `ReLU`: `(64, 64)`
   - Sau `fc3` (`Linear(64, 10)` — logits): `(64, 10)`
3. **Kiểm tra độ hợp lý của Loss ngay lúc khởi tạo ngẫu nhiên**:
   Khi mạng chưa học gì, xác suất đoán cho cả 10 chữ số đều xấp xỉ $$1/10 = 0.1$$. Khi đó Cross-Entropy kỳ vọng ở bước 0 phải xấp xỉ:
   $$-\ln(0.1) = \ln(10) \approx 2.3026.$$
   Mẹo thực chiến cực hay: nếu ở epoch 0 bạn thấy loss ban đầu vọt lên $$15$$ hay tụt về $$0.01$$, hãy dừng lại kiểm tra ngay bug ở chuẩn hóa dữ liệu hoặc hàm mất mát!

## 4. Đoạn mã minh họa

Ta triển khai một ví dụ học sâu cơ bản để làm rõ các khái niệm đã nêu. Mục tiêu là xây dựng một mạng nơ-ron phân loại chữ số viết tay MNIST bằng PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Định nghĩa mạng nơ-ron sâu đơn giản
class SimpleDeepNet(nn.Module):
    """
    Mạng nơ-ron 3 tầng cho phân loại chữ số MNIST.
    
    Kiến trúc:
    - Đầu vào: 784 chiều (ảnh 28x28 làm phẳng)
    - Tầng ẩn 1: 128 nơ-ron với kích hoạt ReLU
    - Tầng ẩn 2: 64 nơ-ron với kích hoạt ReLU
    - Tầng đầu ra: 10 nơ-ron (một nơ-ron cho mỗi lớp chữ số)
    """
    def __init__(self):
        super(SimpleDeepNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Tầng ẩn thứ nhất
        self.fc2 = nn.Linear(128, 64)    # Tầng ẩn thứ hai
        self.fc3 = nn.Linear(64, 10)     # Tầng đầu ra
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Làm phẳng ảnh 28x28 thành vector 784 chiều
        x = x.view(-1, 784)
        
        # Tầng 1: học đặc trưng mức thấp
        x = self.relu(self.fc1(x))
        
        # Tầng 2: học tổ hợp đặc trưng mức trung
        x = self.relu(self.fc2(x))
        
        # Tầng đầu ra: phân loại vào 10 lớp chữ số
        x = self.fc3(x)
        return x

# Tải tập dữ liệu MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Chuẩn hóa theo mean/std của MNIST
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Khởi tạo mô hình, hàm mất mát và bộ tối ưu
model = SimpleDeepNet()
criterion = nn.CrossEntropyLoss()  # Kết hợp softmax + negative log likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Vòng lặp huấn luyện
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()  # Chuyển mô hình sang chế độ huấn luyện
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Lan truyền xuôi: tính dự đoán
        output = model(data)
        loss = criterion(output, target)
        
        # Lan truyền ngược: tính gradient
        optimizer.zero_grad()  # Xóa gradient bước trước
        loss.backward()        # Lan truyền ngược
        
        # Cập nhật trọng số
        optimizer.step()
        
        # Theo dõi độ chính xác
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} Training: Avg Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')

# Vòng lặp đánh giá
def test(model, test_loader, criterion):
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # Tắt tính gradient để tăng hiệu năng
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test: Avg Loss={test_loss:.4f}, Accuracy={accuracy:.2f}%\n')
    return accuracy

# Huấn luyện qua nhiều epoch
print("Starting training...")
for epoch in range(1, 6):  # Huấn luyện 5 epoch
    train(model, train_loader, optimizer, criterion, epoch)
    test_accuracy = test(model, test_loader, criterion)

print(f"Final test accuracy: {test_accuracy:.2f}%")
print("Training complete! The network learned to recognize digits through:")
print("1. Forward propagation (making predictions)")
print("2. Loss computation (measuring errors)")
print("3. Backpropagation (computing gradients)")
print("4. Weight updates (learning from mistakes)")
```

### Phân tích đoạn mã

Ví dụ đơn giản này minh họa đầy đủ các nguyên lý cốt lõi của học sâu:

**1. Thiết kế kiến trúc:** Ba tầng liên tiếp biến đổi đầu vào 784 chiều thành đầu ra 10 chiều thông qua các tầng biểu diễn trung gian.

**2. Học đặc trưng tự động:** Ta không cần chỉ định trước mạng phải tìm đặc trưng nào — mô hình tự học các biểu diễn hữu ích từ dữ liệu điểm ảnh.

**3. Vòng lặp huấn luyện chuẩn:** Chu trình lan truyền xuôi → tính mất mát → lan truyền ngược → cập nhật trọng số là khung vận hành chung của hầu hết quy trình học sâu.

**4. Vai trò của hàm kích hoạt phi tuyến:** Các hàm ReLU đặt giữa các tầng tuyến tính giúp mạng học được những ánh xạ phi tuyến phức tạp. Nếu bỏ qua chúng, nhiều tầng tuyến tính xếp chồng lên nhau sẽ chỉ tương đương một phép biến đổi tuyến tính duy nhất.

**5. Khả năng mở rộng:** Cùng cấu trúc mã nguồn này, khi điều chỉnh kiến trúc và dữ liệu phù hợp, có thể mở rộng sang các tập dữ liệu lớn hơn và những bài toán phức tạp hơn trong thị giác máy tính, xử lý ngôn ngữ tự nhiên và nhiều lĩnh vực khác.

Chỉ sau 5 epoch (5 lượt duyệt qua 60.000 ảnh huấn luyện), mạng đơn giản này thường đạt độ chính xác khoảng 97%, cho thấy khả năng học trực tiếp từ dữ liệu của mạng nơ-ron.

## 5. Các khái niệm liên quan

Để hiểu rõ học sâu, cần đặt nó trong mối liên hệ với các nhánh và khái niệm rộng hơn của học máy và trí tuệ nhân tạo.

### Học có giám sát, học không giám sát và học tăng cường

**Học có giám sát (*supervised learning*)** — trọng tâm chính ở giai đoạn đầu — học từ các ví dụ có nhãn, tức các cặp đầu vào–đầu ra như (ảnh, nhãn lớp) hoặc (câu nguồn, bản dịch). Mạng học ánh xạ từ đầu vào sang đầu ra tương ứng.

**Học không giám sát (*unsupervised learning*)** tìm kiếm cấu trúc ẩn trong dữ liệu chưa gán nhãn. Ví dụ, *autoencoder* học biểu diễn nén, phân cụm (*clustering*) nhóm các mẫu tương đồng, còn mô hình sinh học phân phối dữ liệu để tạo mẫu mới. Hướng tiếp cận này đặc biệt hữu ích khi việc gán nhãn tốn kém hoặc không khả thi.

**Học tăng cường (*reinforcement learning*)** học thông qua tương tác với môi trường: một tác tử (*agent*) thực hiện hành động, nhận tín hiệu phần thưởng và học chính sách nhằm tối đa hóa phần thưởng tích lũy. Cách này phù hợp với các bài toán điều khiển hành vi (như chơi trò chơi hoặc điều khiển robot), nơi không thể chỉ định sẵn hành động đúng cho mọi trạng thái mà chỉ đánh giá qua kết quả dài hạn.

Học sâu đã tạo ra bước tiến lớn ở cả ba hướng tiếp cận này, dù nguyên lý toán học của từng nhánh có nhiều điểm khác biệt. Khóa học này tập trung trước hết vào học có giám sát, sau đó mở rộng sang học không giám sát và học tăng cường ở các chương sau.

### Học máy cổ điển so với học sâu

Học máy truyền thống (như SVM, cây quyết định, hồi quy logistic) thường có các đặc điểm sau:
- Đòi hỏi thiết kế đặc trưng thủ công.
- Dựa trên các giả định mô hình tường minh (như tính tuyến tính hoặc độc lập).
- Hoạt động hiệu quả với tập dữ liệu cỡ nhỏ đến vừa (vài trăm đến vài nghìn mẫu).
- Dễ diễn giải hơn thông qua mức độ quan trọng của đặc trưng hoặc ranh giới quyết định.

Trong khi đó, học sâu:
- Tự động học đặc trưng theo hướng từ đầu đến cuối (*end-to-end*).
- Ít đặt giả định cứng nhắc hơn về cấu trúc dữ liệu.
- Thường cần tập dữ liệu lớn (từ hàng nghìn đến hàng triệu mẫu).
- Khó diễn giải trực tiếp hơn nhưng xử lý hiệu quả các mẫu phức tạp.

Không có phương pháp nào vượt trội trong mọi hoàn cảnh: học máy cổ điển thường phù hợp hơn với tập dữ liệu nhỏ, dữ liệu dạng bảng hoặc khi yêu cầu cao về khả năng giải thích; còn học sâu phát huy thế mạnh trên dữ liệu lớn, đầu vào nhiều chiều (ảnh, văn bản, âm thanh) và cấu trúc mẫu phức tạp.

### Học chuyển giao và tiền huấn luyện

Một kỹ thuật quan trọng của học sâu là **học chuyển giao (*transfer learning*)**: huấn luyện mạng trên một tác vụ nguồn (chẳng hạn phân loại ImageNet) rồi điều chỉnh lại cho tác vụ liên quan (như phân tích ảnh y khoa hoặc nhận diện động vật hoang dã). Các biểu diễn đã học — bộ phát hiện cạnh, kết cấu, hình dạng — có thể tái sử dụng giữa nhiều miền dữ liệu.

Quy trình **tiền huấn luyện (*pre-training*)** trên tập dữ liệu tổng quát quy mô lớn rồi **tinh chỉnh (*fine-tuning*)** cho tác vụ cụ thể đã trở thành chuẩn mực phổ biến. Các mô hình ngôn ngữ lớn như GPT hay BERT được tiền huấn luyện trên kho văn bản khổng lồ, sau đó tinh chỉnh cho từng ứng dụng với lượng dữ liệu chuyên biệt ít hơn nhiều, giúp giảm đáng kể chi phí thu thập và gán nhãn dữ liệu mới.

### Vai trò của kiến trúc, dữ liệu và tính toán

Hiệu quả của học sâu hiện đại bắt nguồn từ sự kết hợp của ba yếu tố chính:

- **Đổi mới kiến trúc:** Các kiến trúc như CNN, ResNet hay Transformer đưa vào thiên kiến quy nạp (*inductive bias*) phù hợp với cấu trúc bài toán, giúp mô hình học hiệu quả hơn.
- **Quy mô dữ liệu:** Tập dữ liệu lớn, đa dạng và có chất lượng cao cung cấp thông tin cần thiết để mạng học được biểu diễn bền vững và khái quát tốt.
- **Năng lực tính toán:** Phần cứng như GPU và TPU song song hóa các phép toán ma trận cốt lõi của mạng nơ-ron, rút ngắn thời gian huấn luyện từ nhiều tháng xuống còn vài giờ.

Những tiến bộ của học sâu trong thực tế là kết quả đồng thời của cả ba yếu tố: kiến trúc tốt hơn (như Transformer), dữ liệu lớn hơn (văn bản và ảnh quy mô web) và năng lực tính toán mạnh hơn (cụm GPU, TPU). Không một yếu tố đơn lẻ nào có thể giải thích trọn vẹn sự phát triển của lĩnh vực.

## 6. Các bài báo nền tảng

Điểm lại một số công trình tiêu biểu giúp hình dung bối cảnh lịch sử hình thành nên thực hành học sâu hiện nay.

**["A Logical Calculus of Ideas Immanent in Nervous Activity" (1943)](https://link.springer.com/article/10.1007/BF02478259)**  
*Tác giả:* Warren McCulloch và Walter Pitts  
Bài báo đề xuất mô hình toán học đầu tiên của nơ-ron nhân tạo, chứng minh rằng mạng gồm các đơn vị ngưỡng đơn giản có thể biểu diễn mọi hàm logic. Dù đơn giản hóa rất nhiều so với nơ-ron sinh học, công trình này đã đặt nền móng lý thuyết cho tính toán nơ-ron và ảnh hưởng lâu dài tới cả khoa học thần kinh lẫn trí tuệ nhân tạo.

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Tác giả:* David Rumelhart, Geoffrey Hinton, Ronald Williams  
Dù lan truyền ngược đã được phát hiện độc lập trước đó, bài báo này đưa kỹ thuật vào dòng chính và chứng minh hiệu quả của nó trong huấn luyện mạng nhiều tầng. Bằng cách áp dụng quy tắc dây chuyền để tính gradient qua hợp thành hàm, lan truyền ngược giúp việc huấn luyện mạng nhiều tầng trở nên khả thi trên thực tế.

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Tác giả:* Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Bài báo giới thiệu kiến trúc LeNet-5, cho thấy mạng nơ-ron tích chập có thể giải quyết hiệu quả các bài toán thực tế như đọc séc và nhận dạng chữ số. Công trình này xác lập các nguyên lý cốt lõi — kết nối cục bộ, chia sẻ trọng số và gộp (*pooling*) — vẫn giữ vai trò trung tâm trong thị giác máy tính hiện đại.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Tác giả:* Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton  
Kết quả vượt trội của AlexNet tại cuộc thi ImageNet 2012 (tỷ lệ lỗi 15,3% so với 26,2% của đội đứng thứ hai) đã mở ra giai đoạn phát triển mạnh mẽ của học sâu hiện đại. Bằng cách kết hợp kiến trúc sâu hơn, hàm kích hoạt ReLU, chính quy hóa *dropout* và huấn luyện trên GPU, bài báo chứng minh mạng nơ-ron có thể mở rộng hiệu quả sang các tập dữ liệu ảnh lớn và phức tạp.

**["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)**  
*Tác giả:* Ashish Vaswani và cộng sự (Google)  
Bài báo giới thiệu kiến trúc Transformer — nền tảng của xử lý ngôn ngữ tự nhiên hiện đại và ngày càng mở rộng sang nhiều lĩnh vực khác. Bằng cách thay thế cơ chế hồi quy bằng cơ chế chú ý (*attention*), Transformer cho phép song song hóa cao trong huấn luyện và nắm bắt tốt các phụ thuộc tầm xa, trở thành kiến trúc cốt lõi của BERT, GPT và phần lớn mô hình ngôn ngữ lớn ngày nay.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Tác giả:* Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
ResNet đưa ra kết nối tắt (*skip connection*), tạo đường truyền trực tiếp cho gradient và nhờ đó cho phép huấn luyện mạng sâu hàng trăm tầng. Bên cạnh chiến thắng tại ImageNet 2015, công trình này làm rõ rằng để tăng độ sâu hiệu quả cần các đổi mới về kiến trúc và chuẩn hóa. Nguyên lý kết nối phần dư hiện có mặt trong hầu hết các kiến trúc sâu hiện đại.

## Video tham khảo

- **Xem trước:** [PyTorch Course (2022), Part 4: Image Classification (MNIST)](https://www.youtube.com/watch?v=gBw0u_5u0qU) — Mr. P Solver, tiếng Anh. Hướng dẫn trực tiếp cách chạy vòng lặp huấn luyện phân loại MNIST bằng PyTorch tương ứng với đoạn mã trong bài.
- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Trình bày mô hình trực giác của mạng nhận dạng chữ số đứng sau vòng lặp huấn luyện trên.
- [Lecture 4 | Introduction to Neural Networks](https://www.youtube.com/watch?v=d14TUNcbn1k) — Stanford University School of Engineering, tiếng Anh. Bài giảng về mạng nơ-ron của Stanford, đặt bộ phân loại cơ bản vào khung lý thuyết của các bài học tiếp theo.
