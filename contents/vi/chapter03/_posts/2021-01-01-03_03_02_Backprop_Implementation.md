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

Hiểu lý thuyết Backpropagation trên giấy mới đi được nửa chặng đường; nửa chặng đường còn lại—và là trải nghiệm đáng giá nhất với mọi kỹ sư Học sâu—là **tự tay cài đặt toàn bộ mạng MLP và lan truyền ngược từ đầu chỉ bằng NumPy**, rồi thấy nó tự học giải gọn bài toán XOR mà không cần bất kỳ dòng PyTorch nào! Một lỗi nhỏ về chiều ma trận (quên chuyển vị, quên chia cho kích thước batch $$B$$, hay dùng nhầm giá trị đã kích hoạt thay vì tiền kích hoạt) có thể khiến mạng học sai lặng lẽ. Bài thực hành này giúp bạn viết mã sạch, kiểm tra kích thước chặt chẽ và kiểm chứng độ chính xác bằng Gradient Checking.

## Kiến thức cần nhớ: Bốn kỷ luật lập trình khi Tự cài Backpropagation bằng NumPy

1. **Quy tắc Khớp Chiều Ma trận (Shape Invariant Check)**:
   Gradient của mọi tham số **bắt buộc phải có kích thước giống hệt chính tham số đó**: `dW.shape == W.shape`, `db.shape == b.shape`, `dA.shape == A.shape`. Cài sẵn các dòng `assert` kiểm tra shape giúp bắt ngay 90% lỗi lập trình.

2. **Lưu Cache đúng lúc Forward để dùng lúc Backward**:
   Để tính `dW_l = A_{l-1}.T @ dZ_l / B` và `dZ_l = dA_l * act_deriv(Z_l)`, hàm `forward` phải lưu lại danh sách `(A_{l-1}, Z_l)` theo đúng thứ tự tầng để `backward` duyệt ngược từ tầng `L` về `1`.

3. **Cẩn thận phép Cộng dồn và Broadcasting cho Bias (`axis=0, keepdims=True`)**:
   Vì bias $$\mathbf{b}^{(\ell)} \in \mathbb{R}^{1 \times n_\ell}$$ được cộng vào mọi mẫu trong batch $$B$$, khi đạo hàm ngược ta phải cộng (hoặc lấy trung bình) dọc theo trục mẫu `axis=0` để giữ nguyên kích thước `(1, n_l)`.

4. **Kiểm tra Sai số Tương đối bằng Sai phân Trung tâm (Gradient Checking)**:
   So sánh `grad_analytic` và `grad_numeric` qua công thức $$\frac{\|g_a - g_n\|_2}{\|g_a\|_2 + \|g_n\|_2}$$: nếu nhỏ hơn $$10^{-7}$$ (với `float64`), cài đặt Backprop của bạn đã chính xác tuyệt đối!

### Giải thích: Tính tay một phép kiểm tra Gradient Checking với $$\varepsilon = 10^{-4}$$

Giả sử bạn nghi ngờ công thức đạo hàm giải tích của hàm $$L(w) = w^3$$ tại $$w = 2.0$$ (đạo hàm đúng là $$g_a = 3(2)^2 = 12.000000$$). Hãy kiểm tra bằng sai phân trung tâm với $$\varepsilon = 10^{-4}$$:

1. Tính $$L(2 + 10^{-4}) = (2.0001)^3 = 8.001200060001$$.
2. Tính $$L(2 - 10^{-4}) = (1.9999)^3 = 7.998800059999$$.
3. Lấy hiệu chia cho $$2\varepsilon = 2 \times 10^{-4}$$:
   $$g_n = \frac{8.001200060001 - 7.998800059999}{0.0002} = \frac{0.002400000002}{0.0002} = 12.00000001.$$
4. **Sai số tương đối**:
   $$\frac{\lvert 12.0 - 12.00000001 \rvert}{12.0 + 12.00000001} \approx 4.17 \times 10^{-10} \ll 10^{-7} \implies \textbf{ĐẠT CHUẨN!}$$
Ngược lại, nếu bạn lỡ viết nhầm đạo hàm trong code thành $$2w^2 = 8.0$$, sai số tương đối lập tức nhảy lên $$\frac{4}{20} = 0.2$$—phát hiện bug ngay trong nháy mắt!

## 4. Đoạn Mã

Dưới đây là cài đặt lan truyền ngược từ đầu bằng NumPy để nắm rõ từng bước tính toán, sau đó đối chiếu với cách PyTorch tự động hóa quá trình này:

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

Tiếp theo, hãy xem cách PyTorch tự động hóa toàn bộ quá trình tính gradient trên:

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

Sự đối chiếu giữa hai đoạn mã cho thấy rõ lợi ích của các framework hiện đại: bản cài đặt thủ công cần khoảng 100 dòng mã chỉ cho một mạng đơn giản, trong khi PyTorch xử lý tự động mọi kiến trúc. Tuy nhiên, việc tự viết lan truyền ngược vẫn rất giá trị khi cần gỡ lỗi mô hình không hội tụ, xây dựng lớp tính toán tùy chỉnh hoặc đọc hiểu các phân tích về luồng gradient trong tài liệu nghiên cứu.

## 5. Các Khái niệm Liên quan

Lan truyền ngược gắn chặt với nhiều khái niệm nền tảng khác trong học sâu và học máy. Nắm rõ các mối liên hệ này giúp hiểu vai trò thực sự của thuật toán trong toàn bộ quy trình huấn luyện.

Mối liên hệ trực tiếp nhất là với gradient descent và các bộ tối ưu hóa. Lan truyền ngược chỉ giải bài toán **tính gradient**, còn bộ tối ưu (như SGD, SGD với momentum, Adam) quyết định cách dùng gradient đó để cập nhật tham số. Tách biệt hai vai trò này giúp làm rõ cơ chế: lan truyền ngược chỉ ra hướng giảm hàm mất mát, còn bộ tối ưu quyết định độ dài bước đi và cách tích lũy thông tin từ các bước trước.

Vi phân tự động (*automatic differentiation*) — công nghệ lõi của PyTorch và TensorFlow — là dạng tổng quát hơn của lan truyền ngược. Trong khi lan truyền ngược thường được trình bày cho mạng neuron, vi phân tự động áp dụng cho mọi chương trình tính toán khả vi. Framework sẽ dựng đồ thị tính toán trong lượt xuôi (mỗi nút là một phép toán, mỗi cạnh là luồng dữ liệu), sau đó chạy vi phân tự động chế độ ngược (*reverse-mode autodiff*) trên đồ thị đó. Nhờ vậy, bất kỳ kiến trúc nào tạo từ các phép toán khả vi đều có thể huấn luyện tự động.

Hiện tượng gradient biến mất và bùng nổ là hệ quả trực tiếp của việc nhân dồn đạo hàm qua nhiều lớp. Ở mỗi lớp, tín hiệu sai số được nhân với ma trận trọng số và đạo hàm của hàm kích hoạt. Nếu các hệ số nhân này liên tục nhỏ hơn 1 (chẳng hạn khi sigmoid hoặc tanh bị bão hòa), gradient giảm theo cấp số nhân theo độ sâu; ngược lại, nếu lớn hơn 1, gradient tăng vọt mất kiểm soát. Nhận thức này là động lực cho hàng loạt cải tiến như ReLU, batch normalization, kết nối tắt (*residual connections*) và các phương pháp khởi tạo trọng số phù hợp.

Cơ chế đồ thị tính toán cũng giúp lan truyền ngược mở rộng ra ngoài mạng truyền thẳng: nó áp dụng được cho mạng hồi quy (lan truyền ngược theo thời gian — BPTT), các mạng có nhánh điều khiển phức tạp và cả những kiến trúc động thay đổi theo dữ liệu đầu vào. Nguyên lý chung không đổi: xây dựng đồ thị trong lượt xuôi và truyền ngược đạo hàm bằng quy tắc dây chuyền.

Cuối cùng, lan truyền ngược là một cách giải quyết bài toán **phân bổ đóng góp** (*credit assignment*): khi mô hình dự đoán sai, tham số nào chịu trách nhiệm bao nhiêu? Lan truyền ngược phân bổ mức trách nhiệm tỷ lệ với gradient cục bộ. Tuy đây là giải pháp rất hiệu quả cho học có giám sát khả vi, các lĩnh vực khác như học tăng cường hoặc mô hình tuần tự vẫn cần thêm các cơ chế khác (như hàm giá trị hoặc cơ chế attention) khi giả định khả vi trực tiếp không còn thỏa mãn.

## 6. Các Bài báo Nền tảng

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Tác giả*: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams  
Bài báo kinh điển trên tạp chí *Nature* đã phổ biến rộng rãi thuật toán lan truyền ngược và chứng minh tính hiệu quả trên các bài toán thực tế như nhận dạng giọng nói và phân loại ảnh. Các tác giả trình bày rõ ràng thuật toán dựa trên quy tắc dây chuyền và cho thấy các lớp ẩn có thể tự học biểu diễn đặc trưng hữu ích — vượt qua giới hạn của perceptron một lớp. Đây là một trong những công trình được trích dẫn nhiều nhất trong lịch sử học máy.

**["Efficient BackProp" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller  
Báo cáo kỹ thuật này phân tích có hệ thống các yếu tố giúp lan truyền ngược hoạt động ổn định trong thực tế: phương pháp khởi tạo trọng số với phương sai hợp lý, chuẩn hóa dữ liệu đầu vào, lựa chọn tốc độ học và hàm kích hoạt. Nhiều kỹ thuật tiêu chuẩn trong học sâu ngày nay bắt nguồn từ những đúc kết thực nghiệm trong tài liệu này.

**["On the difficulty of training Recurrent Neural Networks" (2013)](https://arxiv.org/abs/1211.5063)**  
*Tác giả*: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio  
Bài báo phân tích chặt chẽ nguyên nhân khiến lan truyền ngược theo thời gian trong RNN dễ gặp tình trạng gradient biến mất hoặc bùng nổ do phép nhân ma trận lặp lại nhiều bước: nếu trị riêng lớn nhất nhỏ hơn 1, gradient suy giảm theo hàm mũ; nếu lớn hơn 1, gradient bùng nổ. Các tác giả đề xuất kỹ thuật cắt ngưỡng gradient (*gradient clipping*) để kiểm soát bùng nổ và phân tích vai trò của cơ chế cổng trong LSTM.

**["Automatic differentiation in PyTorch" (2017)](https://openreview.net/forum?id=BJJsrmfCZ)**  
*Tác giả*: Adam Paszke, Sam Gross, Soumith Chintala, et al.  
Bài báo mô tả hệ thống `autograd` của PyTorch với đồ thị tính toán động được xây dựng ngay trong lượt lan truyền xuôi, kết hợp vi phân tự động chế độ ngược. Thiết kế linh hoạt này đặc biệt phù hợp với các kiến trúc có cấu trúc thay đổi theo dữ liệu và giúp PyTorch trở nên phổ biến trong nghiên cứu cũng như triển khai thực tế.

**["Deep Learning" - Chương 6 (2016)](http://www.deeplearningbook.org/contents/mlp.html)**  
*Tác giả*: Ian Goodfellow, Yoshua Bengio, Aaron Courville  
Chương sách giáo khoa này trình bày toàn diện và chặt chẽ nền tảng toán học của lan truyền ngược, đồ thị tính toán, độ phức tạp cũng như các vấn đề thực tế như ổn định số và quản lý bộ nhớ. Tài liệu được cung cấp miễn phí trực tuyến và là nguồn tham khảo chuẩn mực về chủ đề này.

## Cạm bẫy và Mẹo Thường gặp

Một lỗi rất hay gặp khi tự cài đặt lan truyền ngược là quên lưu đệm (*cache*) các giá trị trung gian từ lượt xuôi. Lượt ngược cần dùng lại cả tiền kích hoạt $$\mathbf{z}^{[l]}$$ lẫn kích hoạt $$\mathbf{a}^{[l]}$$ của từng lớp. Nếu bỏ sót hoặc vô tình ghi đè chúng, chương trình hoặc phải tính lại lượt xuôi (làm chậm gấp đôi), hoặc tính ra gradient sai. Khi viết thủ công, nên duy trì một cấu trúc từ điển `cache` rõ ràng cho từng lớp.

Lỗi không khớp kích thước giữa gradient và tham số cũng rất dễ gây sai sót khó phát hiện. Gradient $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$ luôn phải có đúng kích thước của $$\mathbf{W}^{[l]}$$ ($$n_{out} \times n_{in}$$). Nhầm thứ tự nhân ma trận hoặc quên chuyển vị trong biểu thức $$\boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$$ có thể bị NumPy tự động *broadcast* mà không báo lỗi ngay, dẫn tới cập nhật sai. Nên đặt câu lệnh `assert` kiểm tra kích thước gradient khớp với tham số.

Bất ổn số học khi tính đạo hàm cũng có thể làm hỏng quá trình huấn luyện. Ví dụ với đạo hàm sigmoid $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$, khi $$z$$ rất lớn, giá trị số thực của $$\sigma(z)$$ làm tròn thành $$1$$ khiến đạo hàm thành $$1 \times (1-1) = 0$$ dù về mặt toán học vẫn dương. Giới hạn miền giá trị trung gian và dùng công thức ổn định số (như log-sum-exp cho softmax) giúp hạn chế lỗi này.

Một kỹ thuật gỡ lỗi hiệu quả là **kiểm tra gradient bằng số** (*gradient checking*). Với từng tham số $$\theta$$, ta xấp xỉ đạo hàm bằng sai phân trung tâm:

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}$$

với $$\epsilon \approx 10^{-7}$$. Đối chiếu kết quả này với gradient do lan truyền ngược tính ra sẽ nhanh chóng phát hiện lỗi cài đặt (sai số tương đối nên nhỏ hơn $$10^{-7}$$). Tuy nhiên, do phương pháp này rất chậm, chỉ nên dùng khi kiểm thử mã nguồn chứ không bật trong lúc huấn luyện thật.

Cắt ngưỡng gradient (*gradient clipping*) là kỹ thuật quan trọng khi huấn luyện RNN hoặc các mạng sâu dễ bùng nổ gradient. Ta tính chuẩn toàn cục $$\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_{\theta} (\frac{\partial \mathcal{L}}{\partial \theta})^2}$$; nếu giá trị này vượt ngưỡng cho phép (thường chọn 5 hoặc 10), toàn bộ gradient được thu nhỏ theo tỷ lệ $$\frac{\text{threshold}}{\|\nabla_\theta \mathcal{L}\|_2}$$. Thao tác này giữ nguyên hướng cập nhật trong khi ngăn các bước nhảy quá lớn làm hỏng trọng số.

Cuối cùng, vì lan truyền ngược đơn thuần là quy tắc dây chuyền trên từng phép toán, bạn hoàn toàn có thể tự định nghĩa đạo hàm cục bộ cho các lớp tùy chỉnh và ghép chúng vào mạng để hệ thống autograd xử lý phần còn lại.

## Điểm Then chốt

Lan truyền ngược áp dụng quy tắc dây chuyền một cách hiệu quả để tính đồng thời gradient cho mọi tham số với chi phí tương đương một lượt lan truyền xuôi. Thuật toán duyệt ngược qua các lớp, truyền tín hiệu sai số và kết hợp với các giá trị trung gian đã lưu đệm để tính đạo hàm. Các cặp hàm mất mát và kích hoạt chuẩn mang lại biểu thức sai số đầu ra rất gọn ($$\boldsymbol{\delta}^{[L]} = \mathbf{a}^{[L]} - \mathbf{y}$$). Dù các framework hiện đại đã tự động hóa quy trình này bằng vi phân tự động chế độ ngược, việc hiểu rõ cơ chế bên trong — từ lưu đệm, kích thước ma trận đến nguyên nhân gradient biến mất hay bùng nổ — vẫn là nền tảng thiết yếu để gỡ lỗi và thiết kế mô hình hiệu quả.

## Video tham khảo

- **Xem trước:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Xây dựng từng dòng một bộ máy autograd gọn nhẹ để hiểu trọn vẹn lan truyền ngược.
- [Neural Networks Pt. 2: Backpropagation Main Ideas](https://www.youtube.com/watch?v=IN2XmBhILt4) — StatQuest with Josh Starmer, tiếng Anh. Minh họa quy tắc dây chuyền trên đại lượng vô hướng để đối chiếu song song với mã nguồn.
- [Xây dựng mạng nơ-ron nhân tạo từ con số 0 | Writing a Neural Network from scratch](https://www.youtube.com/watch?v=VBviWbusHd8) — Học Lập Trình cùng Phát, tiếng Việt. Hướng dẫn viết mạng neuron từ đầu bằng tiếng Việt, bao gồm cả bước lan truyền ngược.
