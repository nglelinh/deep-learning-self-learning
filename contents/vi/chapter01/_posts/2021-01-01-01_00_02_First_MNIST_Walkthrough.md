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

## 4. Đoạn mã minh họa

Ta triển khai một ví dụ học sâu đơn giản để làm rõ các khái niệm. Mục tiêu là xây dựng mạng nơ-ron phân loại chữ số MNIST bằng PyTorch.

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

### Hiểu đoạn mã

Ví dụ đơn giản này thể hiện các nguyên lý cốt lõi của học sâu:

**1. Thiết kế kiến trúc**: Ba tầng biến đổi đầu vào 784 chiều thành đầu ra 10 chiều thông qua các biểu diễn được học.

**2. Học đặc trưng tự động**: Ta không chỉ cho mạng cần tìm đặc trưng nào — nó tự khám phá các biểu diễn hữu ích.

**3. Vòng lặp huấn luyện**: Mẫu chuẩn lan truyền xuôi → tính mất mát → lan truyền ngược → cập nhật trọng số, nằm dưới mọi quy trình học sâu.

**4. Phi tuyến là then chốt**: Các hàm kích hoạt ReLU giữa các tầng cho phép học hàm phức tạp, phi tuyến. Nếu không có chúng, nhiều tầng tuyến tính sẽ sụp về một phép biến đổi tuyến tính duy nhất.

**5. Khả năng mở rộng**: Cùng cấu trúc mã này, với các điều chỉnh phù hợp, hoạt động cho tập dữ liệu lớn hơn và tác vụ phức tạp hơn — thị giác máy tính, xử lý ngôn ngữ tự nhiên, v.v.

Chỉ sau 5 epoch (5 lượt qua 60.000 ảnh huấn luyện), mạng đơn giản này thường đạt khoảng 97% độ chính xác — minh họa sức mạnh của học sâu trong việc học từ dữ liệu.

## 5. Các khái niệm liên quan

Hiểu học sâu đòi hỏi nhìn mối liên hệ của nó với các khái niệm rộng hơn trong học máy và AI:

### Học có giám sát, không giám sát và tăng cường

**Học có giám sát** (*supervised learning*) (chủ đề chính ta đã thảo luận) học từ các ví dụ có nhãn: các cặp đầu vào–đầu ra như (ảnh, nhãn) hoặc (câu, bản dịch). Mạng học ánh xạ từ đầu vào sang đầu ra đúng.

**Học không giám sát** (*unsupervised learning*) khám phá cấu trúc trong dữ liệu không có nhãn. Autoencoder học biểu diễn nén. Clustering nhóm các ví dụ tương tự. Mô hình sinh học phân bố dữ liệu để tạo mẫu mới. Các kỹ thuật này quan trọng khi nhãn đắt đỏ hoặc không có sẵn.

**Học tăng cường** (*reinforcement learning*) học từ tương tác: một agent thực hiện hành động trong môi trường, nhận phần thưởng và học chính sách tối đa hóa phần thưởng tích lũy. Cách này cho phép học hành vi (chơi game, robot) khi ta không thể cung cấp hành động đúng tường minh cho mọi tình huống, mà chỉ phản hồi theo kết quả.

Học sâu đã chuyển đổi cả ba paradigm, nhưng các nguyên lý khác nhau đáng kể. Khóa học này tập trung chủ yếu vào học có giám sát ở giai đoạn đầu, các chương sau sẽ đề cập học không giám sát và học tăng cường.

### Học máy cổ điển so với học sâu

Học máy truyền thống (SVM, cây quyết định, hồi quy logistic) thường yêu cầu:
- Đặc trưng được thiết kế thủ công
- Giả định mô hình tường minh (tuyến tính, độc lập)
- Hoạt động tốt với dữ liệu vừa phải (hàng trăm đến hàng nghìn mẫu)
- Dễ diễn giải hơn (tầm quan trọng đặc trưng, ranh giới quyết định)

Học sâu:
- Học đặc trưng tự động theo kiểu đầu-cuối
- Ít giả định hơn về cấu trúc dữ liệu
- Cần tập dữ liệu lớn (hàng nghìn đến hàng triệu mẫu)
- Khó diễn giải hơn nhưng mạnh hơn với mẫu phức tạp

Không có cách tiếp cận nào vượt trội tuyệt đối — học máy cổ điển có thể tốt hơn với tập nhỏ, dữ liệu dạng bảng, hoặc khi khả năng diễn giải là then chốt. Học sâu xuất sắc với dữ liệu lớn, đầu vào chiều cao (ảnh, văn bản) và mẫu phức tạp.

### Chuyển giao học và tiền huấn luyện

Một kỹ thuật mạnh nhất của học sâu là **chuyển giao học** (*transfer learning*): huấn luyện mạng trên một tác vụ (ví dụ phân loại ImageNet) rồi thích nghi cho các tác vụ liên quan (phân tích ảnh y khoa, phát hiện động vật hoang dã). Các biểu diễn đã học — bộ phát hiện cạnh, mẫu texture, nhận dạng hình dạng — chuyển giao giữa các miền.

**Tiền huấn luyện** (*pre-training*) trên tập dữ liệu tổng quát lớn, rồi **tinh chỉnh** (*fine-tuning*) cho tác vụ cụ thể, đã trở thành thực hành chuẩn. GPT, BERT và các mô hình ngôn ngữ lớn khác được tiền huấn luyện trên kho văn bản khổng lồ, rồi chuyên biệt hóa cho ứng dụng cụ thể qua fine-tuning với ít dữ liệu đặc thù hơn nhiều. Cách này giảm mạnh yêu cầu dữ liệu cho tác vụ mới.

### Vai trò của kiến trúc, dữ liệu và tính toán

Thành công của học sâu xuất phát từ ba yếu tố phối hợp:

**Đổi mới kiến trúc** (CNN, Transformer, ResNet) cho phép học một số mẫu hiệu quả. Kiến trúc phù hợp cung cấp inductive bias thích hợp với cấu trúc bài toán.

**Quy mô dữ liệu** cung cấp nguyên liệu thô cho việc học. Dữ liệu đa dạng, chất lượng cao giúp mạng học biểu diễn bền vững và khái quát hơn.

**Quy mô tính toán** khiến việc huấn luyện mạng lớn trên dữ liệu lớn trở nên thực tế. GPU song song hóa các phép toán ma trận mà mạng nơ-ron phụ thuộc, rút ngắn thời gian huấn luyện từ tháng xuống giờ.

Tiến bộ học sâu hiện đại đến từ cả ba hướng: kiến trúc tốt hơn (Transformer), dữ liệu lớn hơn (văn bản và ảnh quy mô web), và tính toán mạnh hơn (cụm GPU, TPU). Không có yếu tố đơn lẻ nào giải thích đủ thành công của lĩnh vực.

## 6. Các bài báo nền tảng

Hiểu sự phát triển lịch sử của học sâu qua các bài báo then chốt cung cấp bối cảnh cho thực hành hiện tại và định hướng tương lai.

**["A Logical Calculus of Ideas Immanent in Nervous Activity" (1943)](https://link.springer.com/article/10.1007/BF02478259)**  
*Tác giả*: Warren McCulloch và Walter Pitts  
Bài báo nền tảng này giới thiệu mô hình toán học của nơ-ron nhân tạo, chỉ ra rằng mạng các đơn vị ngưỡng đơn giản có thể tính bất kỳ hàm logic nào. Dù đơn giản hóa rất nhiều so với nơ-ron sinh học, công trình này đặt nền tảng lý thuyết cho tính toán nơ-ron và truyền cảm hứng cho nghiên cứu sau đó trong cả khoa học thần kinh và trí tuệ nhân tạo.

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Tác giả*: David Rumelhart, Geoffrey Hinton, Ronald Williams  
Lan truyền ngược không được phát minh lần đầu tại đây (nó được khám phá độc lập nhiều lần), nhưng bài báo này đưa kỹ thuật đến sự chú ý rộng rãi và chứng minh sức mạnh của nó trong huấn luyện mạng nhiều tầng. Bằng cách chỉ ra cách tính gradient hiệu quả qua hợp thành hàm nhờ quy tắc chuỗi, lan truyền ngược khiến học sâu trở nên thực tế. Bài báo kết thúc mùa đông AI đầu tiên bằng việc chứng minh mạng nơ-ron có thể học hàm phức tạp.

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
LeNet-5, được giới thiệu trong bài báo này, chứng minh rằng mạng nơ-ron tích chập có thể đạt hiệu năng xuất sắc trên tác vụ thực tế (đọc séc, nhận dạng chữ số). Quan trọng hơn, nó thiết lập các nguyên lý thiết kế — kết nối cục bộ, chia sẻ trọng số, pooling — vẫn trung tâm trong thị giác máy tính hiện đại. Bài báo cho thấy học sâu có thể chuyển từ bài toán đồ chơi sang ứng dụng thực tiễn.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Tác giả*: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton  
Chiến thắng áp đảo của AlexNet tại cuộc thi ImageNet 2012 (lỗi 15,3% so với 26,2% của hạng nhì) khơi mào cuộc cách mạng học sâu hiện đại. Bằng cách kết hợp kiến trúc sâu hơn, kích hoạt ReLU, chính quy hóa dropout và huấn luyện GPU, công trình chứng minh mạng nơ-ron có thể mở rộng tới tập dữ liệu lớn, phức tạp. Thành công này thuyết phục cộng đồng thị giác máy tính rộng lớn áp dụng học sâu.

**["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)**  
*Tác giả*: Ashish Vaswani và cộng sự (Google)  
Kiến trúc Transformer được giới thiệu ở đây đã trở thành nền tảng của NLP hiện đại và ngày càng mở rộng sang các miền khác. Bằng cách thay recurrence bằng cơ chế attention, Transformer cho phép song song hóa hoàn toàn trong huấn luyện và nắm bắt tốt hơn các phụ thuộc tầm xa. Ảnh hưởng của bài báo vượt xa ứng dụng dịch máy ban đầu — BERT, GPT và hầu hết các mô hình ngôn ngữ lớn gần đây đều xây dựng trên kiến trúc này.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
ResNet giới thiệu skip connection, cho phép huấn luyện mạng sâu hàng trăm tầng bằng cách cung cấp đường đi gradient trực tiếp. Ngoài chiến thắng ImageNet 2015, công trình này thay đổi căn bản cách ta nghĩ về kiến trúc sâu — độ sâu là then chốt, nhưng mạng cần đổi mới kiến trúc (skip connection, chuẩn hóa cẩn thận) để huấn luyện hiệu quả. Các nguyên lý của ResNet xuất hiện trong hầu hết kiến trúc sâu hiện đại.
