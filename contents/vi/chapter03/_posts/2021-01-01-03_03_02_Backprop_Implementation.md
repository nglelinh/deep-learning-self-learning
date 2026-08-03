---
layout: post
title: 03-03-02 Triển khai Lan truyền Ngược
chapter: '03'
order: 10
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

## 4. Đoạn Mã

Ta triển khai lan truyền ngược từ đầu để hiểu mọi chi tiết, rồi chỉ ra cách PyTorch tự động hóa điều này:

```python
import numpy as np

def sigmoid(z):
    """
    Kích hoạt sigmoid: ánh xạ số thực về (0, 1)
    
    Vì sao sigmoid? Nó trơn (khả vi mọi nơi), bị chặn (đầu ra
    không bùng nổ), và có diễn giải xác suất. Đạo hàm có dạng
    đẹp: σ'(z) = σ(z)(1 - σ(z)), có thể tính từ chính kích hoạt
    mà không cần lưu z.
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip để ổn định số

def sigmoid_derivative(a):
    """Đạo hàm theo kích hoạt (không phải z!)"""
    return a * (1 - a)

def relu(z):
    """
    ReLU: max(0, z) - cho ra đầu vào nếu dương, không nếu ngược lại
    
    Vì sao ReLU? Tính toán tầm thường (chỉ ngưỡng hóa), không
    bão hòa với đầu vào dương (gradient bằng 1, không tiến về 0),
    và tạo biểu diễn thưa. Các tính chất này làm nó vượt trội
    hơn nhiều so với sigmoid/tanh cho mạng sâu.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Đạo hàm bằng 1 nơi z > 0, bằng không nơi khác
    
    Lưu ý: tại z=0, đạo hàm không xác định. Trong thực tế, ta định nghĩa bằng 0
    hoặc đôi khi 0.5, nhưng điều này hiếm khi quan trọng vì đẳng thức chính xác hiếm gặp.
    """
    return (z > 0).astype(float)

class NeuralNetworkBackprop:
    """
    Triển khai lan truyền ngược thủ công để hiểu mọi bước.
    
    Triển khai này ưu tiên rõ ràng hơn hiệu quả. Mỗi phép toán
    tường minh, gradient được tính thủ công, và ta lưu mọi thứ
    cần cho việc hiểu. Triển khai sản xuất sẽ vectorize
    mạnh hơn và dùng vi phân tự động.
    """
    
    def __init__(self, layer_sizes):
        """
        layer_sizes: danh sách như [2, 4, 3, 1] cho 2 đầu vào, lớp ẩn 4 và 3, 1 đầu ra
        
        Ta khởi tạo trọng số bằng khởi tạo He cho lớp ẩn (giả định ReLU)
        và giá trị ngẫu nhiên nhỏ cho lớp đầu ra. Lược đồ khởi tạo này đảm bảo
        kích hoạt duy trì thang đo hợp lý qua lan truyền xuôi và gradient
        duy trì thang đo hợp lý qua lan truyền ngược.
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # Số lớp trọng số
        self.params = {}
        
        for l in range(1, self.L + 1):
            n_in, n_out = layer_sizes[l-1], layer_sizes[l]
            
            if l < self.L:
                # Khởi tạo He cho lớp ReLU: phương sai = 2/n_in
                # Vì sao? ReLU triệt tiêu một nửa neuron, nên cần √2 thay vì √1
                self.params[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            else:
                # Lớp đầu ra: trọng số nhỏ hơn cho ổn định số
                self.params[f'W{l}'] = np.random.randn(n_out, n_in) * 0.01
            
            self.params[f'b{l}'] = np.zeros((n_out, 1))
        
        print(f"Initialized network with architecture: {layer_sizes}")
        print(f"Total parameters: {self.count_parameters()}")
    
    def count_parameters(self):
        """Đếm tổng số tham số có thể huấn luyện"""
        total = 0
        for l in range(1, self.L + 1):
            total += self.params[f'W{l}'].size + self.params[f'b{l}'].size
        return total
    
    def forward(self, X):
        """
        Lan truyền xuôi với cache chi tiết cho lan truyền ngược.
        
        Ta phải lưu Z (tiền kích hoạt) và A (kích hoạt) cho mỗi lớp
        vì lan truyền ngược cần chúng. Đây là đánh đổi bộ nhớ vs tính toán:
        ta có thể tính lại lan truyền xuôi trong lan truyền ngược, nhưng lưu nhanh hơn.
        """
        cache = {'A0': X}
        A = X
        
        # Lớp ẩn với ReLU
        for l in range(1, self.L):
            Z = self.params[f'W{l}'] @ A + self.params[f'b{l}']
            A = relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Lớp đầu ra với sigmoid
        Z = self.params[f'W{self.L}'] @ A + self.params[f'b{self.L}']
        A = sigmoid(Z)
        cache[f'Z{self.L}'] = Z
        cache[f'A{self.L}'] = A
        
        return A, cache
    
    def compute_loss(self, AL, Y):
        """
        Mất mát entropy chéo nhị phân với các mẹo ổn định số.
        
        Mất mát -[y log(ŷ) + (1-y) log(1-ŷ)] có vấn đề: nếu ŷ đúng bằng
        0 hoặc 1, ta tính log(0) = -∞. Ta cắt dự đoán về [ε, 1-ε] để ngăn điều này.
        """
        m = Y.shape[1]
        epsilon = 1e-8
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        loss = -(1/m) * np.sum(Y * np.log(AL_clipped) + (1-Y) * np.log(1-AL_clipped))
        return loss
    
    def backward(self, AL, Y, cache):
        """
        Lan truyền ngược: tính tất cả gradient tham số hiệu quả.
        
        Thuật toán xử lý các lớp theo thứ tự ngược, duy trì các hạng tử sai số
        và dùng các giá trị lan truyền xuôi đã cache. Gradient mỗi lớp phụ thuộc
        gradient từ các lớp trên nó, tạo luồng thông tin ngược
        mang tên thuật toán.
        """
        m = Y.shape[1]
        grads = {}
        
        # Sai số lớp đầu ra (với BCE + sigmoid điều này đơn giản đáng kể!)
        dAL = -(Y / (AL + 1e-8) - (1-Y) / (1-AL + 1e-8))  # Đạo hàm mất mát
        dZL = dAL * sigmoid_derivative(AL)  # Nhưng thực ra, dZL = AL - Y hoạt động trực tiếp
        
        # Đơn giản hóa cho BCE + Sigmoid (nên dùng trong thực tế)
        dZL = AL - Y
        
        # Gradient lớp đầu ra
        grads[f'dW{self.L}'] = (1/m) * dZL @ cache[f'A{self.L-1}'].T
        grads[f'db{self.L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        
        # Khởi tạo dA để truyền
        dA = self.params[f'W{self.L}'].T @ dZL
        
        # Lớp ẩn (ngược qua các lớp L-1 xuống 1)
        for l in reversed(range(1, self.L)):
            # Sai số lớp hiện tại
            dZ = dA * relu_derivative(cache[f'Z{l}'])
            
            # Gradient cho lớp này
            grads[f'dW{l}'] = (1/m) * dZ @ cache[f'A{l-1}'].T
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            # Truyền sai số về lớp trước (nếu không phải lớp đầu vào)
            if l > 1:
                dA = self.params[f'W{l}'].T @ dZ
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """
        Cập nhật gradient descent: θ ← θ - η ∇θ L
        
        Ta dịch tham số theo hướng giảm mất mát. Learning rate
        η điều khiển kích thước bước — quá lớn gây vượt và bất ổn,
        quá nhỏ gây hội tụ chậm.
        """
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']
    
    def train(self, X, Y, learning_rate=0.01, num_iterations=1000, print_every=100):
        """
        Vòng lặp huấn luyện đầy đủ: xuôi → mất mát → ngược → cập nhật
        
        Đây là vòng lặp huấn luyện chuẩn cho mạng neuron. Mỗi vòng lặp
        xử lý toàn bộ tập dữ liệu (batch gradient descent). Trong thực tế, ta
        dùng mini-batch vì hiệu quả.
        """
        losses = []
        
        for i in range(num_iterations):
            # Lan truyền xuôi
            AL, cache = self.forward(X)
            
            # Tính mất mát
            loss = self.compute_loss(AL, Y)
            losses.append(loss)
            
            # Lan truyền ngược
            grads = self.backward(AL, Y, cache)
            
            # Cập nhật tham số
            self.update_parameters(grads, learning_rate)
            
            # In tiến độ
            if i % print_every == 0:
                accuracy = np.mean((AL > 0.5).astype(int) == Y)
                print(f"Iteration {i:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
        
        return losses

# Minh họa trên bài toán XOR
print("\n" + "="*70)
print("Training Neural Network on XOR using Manual Backpropagation")
print("="*70)

# Tập dữ liệu XOR
X_xor = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  # Dạng: (2, 4)
Y_xor = np.array([[0, 1, 1, 0]])  # Dạng: (1, 4)

# Tạo và huấn luyện mạng
np.random.seed(42)  # Để tái lập
network = NeuralNetworkBackprop([2, 4, 4, 1])  # Kiến trúc 2→4→4→1
losses = network.train(X_xor, Y_xor, learning_rate=0.5, num_iterations=2000, print_every=500)

# Kiểm tra hiệu năng cuối
print("\n" + "="*70)
print("Final Results")
print("="*70)

AL_final, _ = network.forward(X_xor)
predictions = (AL_final > 0.5).astype(int)

for i in range(4):
    print(f"Input: {X_xor[:, i]}, True: {int(Y_xor[0, i])}, " +
          f"Predicted: {int(predictions[0, i])}, Probability: {AL_final[0, i]:.4f}")

print(f"\nFinal accuracy: {np.mean(predictions == Y_xor):.0%}")
print("\nSuccess! Backpropagation enabled the network to learn XOR.")
```

Bây giờ ta xem cách PyTorch tự động hóa tất cả điều này:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNetPyTorch(nn.Module):
    """
    Cùng mạng dùng vi phân tự động của PyTorch.
    
    Chú ý ta không triển khai backward() - PyTorch tính tất cả gradient
    tự động bằng cách xây đồ thị tính toán trong lan truyền xuôi và
    áp dụng lan truyền ngược khi ta gọi loss.backward().
    """
    
    def __init__(self):
        super(SimpleNetPyTorch, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Huấn luyện với PyTorch
model = SimpleNetPyTorch()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Chuyển sang tensor PyTorch
X_torch = torch.tensor(X_xor.T, dtype=torch.float32)  # (4, 2)
Y_torch = torch.tensor(Y_xor.T, dtype=torch.float32)  # (4, 1)

print("\n" + "="*70)
print("Training with PyTorch Automatic Differentiation")
print("="*70)

for epoch in range(2000):
    # Lan truyền xuôi
    predictions = model(X_torch)
    loss = criterion(predictions, Y_torch)
    
    # Lan truyền ngược - PyTorch thực hiện backpropagation tự động!
    optimizer.zero_grad()  # Xóa gradient cũ
    loss.backward()        # Tính gradient qua vi phân tự động
    optimizer.step()       # Cập nhật trọng số
    
    if epoch % 500 == 0:
        with torch.no_grad():
            acc = ((predictions > 0.5).float() == Y_torch).float().mean()
        print(f'Epoch {epoch:4d}: Loss = {loss.item():.4f}, Accuracy = {acc:.2%}')

# Kiểm tra cuối
model.eval()
with torch.no_grad():
    final_preds = model(X_torch)
    print("\n" + "="*70)
    print("PyTorch Final Results")
    print("="*70)
    for i in range(4):
        print(f"Input: {X_torch[i].numpy()}, Predicted: {final_preds[i].item():.4f}")
```

So sánh hé lộ sức mạnh của các framework hiện đại. Triển khai thủ công của ta mất ~100 dòng để hiện thực lan truyền ngược cho một mạng đơn giản. PyTorch xử lý các kiến trúc tùy ý tự động. Tuy nhiên, hiểu triển khai thủ công vô giá. Khi gỡ lỗi vì sao mạng không học, khi triển khai lớp tùy chỉnh, hoặc khi đọc bài báo nghiên cứu thảo luận luồng gradient, hiểu biết sâu từ triển khai thủ công là thiết yếu.

## 5. Các Khái niệm Liên quan

Lan truyền ngược không tồn tại cô lập — nó gắn bó mật thiết với nhiều khái niệm khác trong học sâu và học máy nói chung. Hiểu các kết nối này biến lan truyền ngược từ một thuật toán đơn thuần thành cửa sổ vào các nguyên lý nền tảng của hệ thống học.

Kết nối trực tiếp nhất là với gradient descent và các biến thể của nó. Lan truyền ngược giải bài toán tính gradient, nhưng chính gradient descent dùng các gradient này để cập nhật tham số. Lựa chọn thuật toán tối ưu — gradient descent gốc, SGD với momentum, Adam, v.v. — quyết định cách ta dùng gradient của lan truyền ngược. Hiểu sự tách biệt này làm rõ trách nhiệm: lan truyền ngược cho biết hướng nào giảm mất mát, trong khi bộ tối ưu quyết định đi bao xa theo hướng đó và có thể tích lũy thông tin qua các vòng lặp.

Vi phân tự động (*automatic differentiation*), công nghệ nền tảng của PyTorch và TensorFlow, là họ hàng tính toán của lan truyền ngược. Trong khi lan truyền ngược thường được mô tả như thuật toán cho mạng neuron, vi phân tự động là kỹ thuật tổng quát hơn để tính đạo hàm của các chương trình tùy ý. Các framework hiện đại xây đồ thị tính toán trong lan truyền xuôi, trong đó các nút biểu diễn phép toán (nhân ma trận, cộng, ReLU, v.v.) và các cạnh biểu diễn luồng dữ liệu. Lan truyền ngược khi đó đơn giản là vi phân tự động reverse-mode trên đồ thị này. Hiểu kết nối này giải thích vì sao framework có thể xử lý kiến trúc tùy ý — miễn là mỗi phép toán khả vi, lan truyền ngược hoạt động tự động.

Các bài toán gradient biến mất và bùng nổ là hệ quả trực tiếp của cách lan truyền ngược truyền sai số qua các lớp. Sai số mỗi lớp là sai số lớp trước nhân với trọng số và đạo hàm kích hoạt. Nếu các hệ số nhân này consistently nhỏ hơn 1 (như với kích hoạt sigmoid/tanh bão hòa), sai số co theo cấp số nhân với độ sâu — đây là gradient biến mất. Nếu hệ số nhân lớn hơn 1, sai số bùng nổ. Hiểu biết này thúc đẩy nhiều đổi mới: kích hoạt ReLU giữ gradient bằng 1 với đầu vào dương, batch normalization giữ kích hoạt trong khoảng hợp lý, kết nối dư cung cấp "đường cao tốc" gradient bỏ qua nhiều lớp, và khởi tạo cẩn thận đảm bảo không biến mất hay bùng nổ ở đầu huấn luyện.

Đồ thị tính toán và lan truyền ngược kết nối với một lĩnh vực đẹp của khoa học máy tính: vi phân tự động và phép tính biến phân. Mọi chương trình khả vi có thể được xem như định nghĩa một hàm từ đầu vào đến đầu ra, và vi phân tự động cung cấp gradient của hàm này. Tính tổng quát này nghĩa là lan truyền ngược không giới hạn ở mạng feedforward — nó hoạt động với mạng hồi quy (lan truyền ngược qua thời gian), với mạng có luồng điều khiển phức tạp, thậm chí với mạng mà kiến trúc phụ thuộc dữ liệu (mạng động). Nguyên lý luôn giống nhau: xây đồ thị tính toán, tính lan truyền xuôi, tính lan truyền ngược dùng quy tắc dây chuyền.

Cuối cùng, lan truyền ngược kết nối với câu hỏi rộng hơn về gán công (*credit assignment*) trong hệ thống học. Khi mạng mắc sai lầm, tham số nào chịu trách nhiệm? Lan truyền ngược cung cấp một câu trả lời: gán công tỷ lệ với gradient. Nhưng đây không phải câu trả lời duy nhất. Học tăng cường dùng các cơ chế gán công khác cho các bài toán quyết định tuần tự. Cơ chế attention cung cấp dạng gán công khác cho các tác vụ sequence-to-sequence. Hiểu lan truyền ngược như một giải pháp cho gán công giúp ta đánh giá cả sức mạnh lẫn hạn chế của nó, và thúc đẩy các cách tiếp cận thay thế khi các giả định của lan truyền ngược không còn đúng.

## 6. Các Bài báo Nền tảng

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Tác giả*: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams  
Bài báo Nature nền tảng này làm lan truyền ngược được biết rộng rãi và chứng minh sức mạnh trên các bài toán thực tế gồm nhận dạng giọng nói và phân loại ảnh. Bài báo trình bày thanh lịch thuật toán, chứng minh tính đúng đắn qua quy tắc dây chuyền, và cho thấy mạng nhiều lớp huấn luyện bằng lan truyền ngược có thể giải các bài toán bất khả với perceptron một lớp. Các tác giả chứng minh học biểu diễn nội tại — các lớp ẩn tự động khám phá đặc trưng hữu ích — điều mang tính khai sáng vào thời điểm đó. Bài báo thực chất khởi động cuộc cách mạng liên kết và vẫn là một trong những bài báo được trích dẫn nhiều nhất trong toàn bộ học máy. Đọc lại ngày nay, ta bị ấn tượng bởi mức độ rõ ràng mà các tác giả hiểu cả sức mạnh lẫn thách thức của thuật toán, bao gồm điều ta nay gọi là gradient biến mất.

**["Efficient BackProp" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller  
Báo cáo kỹ thuật này, dù ít nổi tiếng hơn bài báo lan truyền ngược gốc, có lẽ quan trọng hơn với người thực hành. LeCun và đồng nghiệp phân tích hệ thống những gì làm lan truyền ngược hoạt động tốt trong thực tế, bao gồm khởi tạo (vì sao trọng số ngẫu nhiên nên có phương sai được chọn cẩn thận), chuẩn hóa (vì sao chuẩn hóa đầu vào giúp), lựa chọn learning rate, và lựa chọn hàm kích hoạt. Bài báo cung cấp sự khôn ngoan thực tế tích lũy từ nhiều năm làm lan truyền ngược hoạt động trên các bài toán thực. Nhiều "mẹo" được dạy trong các khóa học sâu hiện đại — như khởi tạo He và chuẩn hóa đầu vào — có gốc rễ từ các hiểu biết trong bài báo này. Đây là tài liệu đọc thiết yếu cho bất kỳ ai muốn huấn luyện mạng hiệu quả thay vì chỉ áp dụng lan truyền ngược một cách máy móc.

**["On the difficulty of training Recurrent Neural Networks" (2013)](https://arxiv.org/abs/1211.5063)**  
*Tác giả*: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio  
Bài báo này phân tích chặt chẽ vì sao lan truyền ngược thất bại trong mạng neuron hồi quy — cụ thể, vì sao gradient biến mất hoặc bùng nổ khi truyền ngược qua thời gian. Các tác giả cho thấy khi unrolling RNN qua nhiều bước thời gian, gradient phải đi qua các phép nhân ma trận lặp lại, và nếu trị riêng lớn nhất của ma trận trọng số hồi quy nhỏ hơn 1, gradient biến mất theo cấp số nhân; nếu lớn hơn 1, chúng bùng nổ. Bài báo đề xuất gradient clipping để xử lý bùng nổ (vẫn là thực hành chuẩn ngày nay) và phân tích cách cơ chế cổng của LSTM giảm nhẹ gradient biến mất. Công trình này làm sâu hiểu biết về hạn chế của lan truyền ngược và thúc đẩy các đổi mới kiến trúc như LSTM và GRU làm lan truyền ngược hồi quy ổn định hơn.

**["Automatic differentiation in PyTorch" (2017)](https://openreview.net/forum?id=BJJsrmfCZ)**  
*Tác giả*: Adam Paszke, Sam Gross, Soumith Chintala, et al.  
Bài báo này mô tả hệ thống autograd của PyTorch, tự động hóa lan truyền ngược dùng đồ thị tính toán động. Không như các framework trước yêu cầu định nghĩa cấu trúc mạng tĩnh, PyTorch xây đồ thị trong khi thực thi xuôi, cho phép kiến trúc động (trong đó tính toán phụ thuộc dữ liệu). Bài báo giải thích cách PyTorch tính gradient dùng vi phân tự động reverse-mode — là lan truyền ngược tổng quát hóa cho mã tùy ý, không chỉ mạng neuron. Tính linh hoạt này làm PyTorch phổ biến trong nghiên cứu nơi thử nghiệm kiến trúc mới là phổ biến. Hiểu cách framework tự động hóa lan truyền ngược giúp người dùng gỡ lỗi vấn đề gradient và triển khai phép toán tùy chỉnh đúng.

**["Deep Learning" - Chương 6 (2016)](http://www.deeplearningbook.org/contents/mlp.html)**  
*Tác giả*: Ian Goodfellow, Yoshua Bengio, Aaron Courville  
Dù không phải bài báo nghiên cứu, chương sách giáo khoa này cung cấp xử lý toàn diện và chặt chẽ nhất về lan truyền ngược hiện có. Nó trình bày thuật toán từ các nguyên lý đầu tiên, thảo luận đồ thị tính toán chi tiết, phân tích độ phức tạp, và đề cập các cân nhắc thực tế như ổn định số và quản lý bộ nhớ. Chương cầu nối lý thuyết và thực hành, giải thích không chỉ những gì lan truyền ngược tính mà vì sao nó tính như vậy, cách triển khai hiệu quả, và khi nào nó có thể thất bại. Với bất kỳ ai tìm hiểu toán học đầy đủ về lan truyền ngược, chương này là tài nguyên chuẩn mực. Nó cũng miễn phí trực tuyến, tiếp cận được với mọi người học.

## Cạm bẫy và Mẹo Thường gặp

Có lẽ cạm bẫy thâm độc nhất trong lan truyền ngược là không cache các giá trị lan truyền xuôi. Trong lan truyền xuôi, ta phải lưu cả tiền kích hoạt $$\mathbf{z}^{[l]}$$ và kích hoạt $$\mathbf{a}^{[l]}$$ cho mọi lớp vì lan truyền ngược cần chúng. Quên cache các giá trị này hoặc ghi đè chúng trước khi lan truyền ngược hoàn tất nghĩa là ta phải tính lại lan truyền xuôi, gấp đôi thời gian tính toán, hoặc tệ hơn, dùng giá trị sai và nhận gradient sai. Đó là vì sao các framework hiện đại tự động xử lý caching — đồ thị tính toán nhớ mọi giá trị trung gian. Khi triển khai lan truyền ngược thủ công, duy trì tường minh một từ điển cache là thực hành tốt.

Không khớp chiều giữa gradient và tham số là lỗi phổ biến khác có thể tinh tế để gỡ. Gradient $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$ phải có đúng cùng dạng với $$\mathbf{W}^{[l]}$$ — nếu $$\mathbf{W}^{[l]}$$ là $$n_{out} \times n_{in}$$, gradient của nó cũng vậy. Khi tính $$\boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$$, nhầm thứ tự nhân hoặc quên chuyển vị có thể sinh ma trận dạng sai mà Python có thể broadcast sai, dẫn đến lỗi tinh tế. Luôn assert rằng dạng gradient khớp dạng tham số sau khi tính chúng.

Bất ổn số trong tính gradient có thể khiến huấn luyện thất bại theo cách không ngay lập tức rõ ràng. Khi tính đạo hàm sigmoid $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$, nếu $$z$$ rất lớn, $$\sigma(z) \approx 1$$ và đạo hàm trở thành $$1 \times (1-1) = 0$$ về số, dù về toán nó phải là số dương nhỏ. Điều này khiến gradient biến mất không do độ sâu mạng mà do độ chính xác số thực. Cắt các giá trị trung gian về khoảng hợp lý và dùng triển khai ổn định số (như mẹo log-sum-exp cho softmax) ngăn các vấn đề này.

Một kỹ thuật gỡ lỗi mạnh là kiểm tra gradient qua xấp xỉ số. Với mọi tham số $$\theta$$, ta có thể xấp xỉ gradient dùng sai phân hữu hạn:

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

với $$\epsilon \approx 10^{-7}$$. So sánh gradient số này với gradient lan truyền ngược hé lộ lỗi triển khai. Hiệu tương đối nên nhỏ hơn $$10^{-7}$$ với triển khai đúng. Tuy nhiên, kiểm tra gradient chậm (đòi hỏi nhiều lần lan truyền xuôi) nên chỉ dùng để gỡ lỗi, không bao giờ trong huấn luyện thực tế.

Gradient clipping xứng đáng đề cập đặc biệt như mẹo thiết yếu khi huấn luyện mạng hồi quy hoặc bất kỳ kiến trúc sâu nào dễ bị bùng nổ gradient. Ta theo dõi chuẩn gradient toàn cục $$\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_{\theta} (\frac{\partial \mathcal{L}}{\partial \theta})^2}$$ và nếu vượt ngưỡng (thường 5 hoặc 10), ta scale mọi gradient bằng $$\frac{\text{threshold}}{\|\nabla_\theta \mathcal{L}\|_2}$$. Điều này bảo toàn hướng gradient trong khi ngăn các cập nhật bùng nổ sẽ làm bất ổn huấn luyện. Đây là mẹo đơn giản làm huấn luyện nhiều kiến trúc trở nên khả thi.

Cuối cùng, hiểu rằng lan truyền ngược chỉ là triển khai hiệu quả của quy tắc dây chuyền nghĩa là ta có thể tự suy ra gradient cho các lớp tùy chỉnh. Khi triển khai một phép toán mới, suy ra gradient địa phương của nó (cách đầu ra thay đổi theo đầu vào), và lan truyền ngược tự động tích hợp nó vào gradient mạng đầy đủ. Hiểu biết này trao quyền — ta không bị giới hạn ở các lớp định sẵn mà có thể tạo bất kỳ tính toán nào bài toán yêu cầu, miễn là ta có thể vi phân chúng.

## Điểm Then chốt

Lan truyền ngược về cơ bản là áp dụng hiệu quả quy tắc dây chuyền của giải tích để tính gradient trong mạng neuron. Hiệu quả của nó — tính tất cả gradient trong thời gian tỷ lệ với một lần lan truyền xuôi — làm huấn luyện mạng sâu khả thi. Thuật toán xử lý các lớp theo thứ tự ngược, truyền sai số ngược và dùng các giá trị lan truyền xuôi đã cache để tính gradient tham số. Sự đơn giản đẹp đẽ của $$\boldsymbol{\delta}^{[L]} = \mathbf{a}^{[L]} - \mathbf{y}$$ cho lớp đầu ra với cặp mất mát/kích hoạt phù hợp không phải ngẫu nhiên mà là thiết kế cẩn thận. Các framework hiện đại tự động hóa lan truyền ngược qua vi phân tự động, xây đồ thị tính toán và áp dụng vi phân reverse-mode. Hiểu sâu lan truyền ngược nghĩa là hiểu không chỉ cơ chế mà cả vì sao — vì sao ta cache giá trị, vì sao gradient biến mất hoặc bùng nổ, vì sao một số lựa chọn thiết kế đơn giản hóa gradient — và hiểu biết này thiết yếu để gỡ lỗi thất bại huấn luyện, thiết kế kiến trúc mới, và thực sự làm chủ học sâu thay vì chỉ áp dụng nó.

Hành trình từ triển khai lan truyền ngược thủ công đến dùng nó liền mạch qua PyTorch phản ánh hành trình từ hiểu biết đến ứng dụng, và cả hai đều cần thiết cho sự thành thạo.
