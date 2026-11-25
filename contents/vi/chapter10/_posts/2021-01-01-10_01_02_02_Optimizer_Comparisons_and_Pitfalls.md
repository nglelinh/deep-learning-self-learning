---
layout: post
title: 10-01-02-02 So sánh Bộ tối ưu, Bài báo và Cạm bẫy
chapter: '10'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter10
---

Chạy cùng một mạng neuron trên cùng một bộ dữ liệu, tại sao có bài toán Adam chỉ mất 50 epoch để chạm đáy trong khi SGD mất 500 epoch — nhưng lại có những cuộc thi thị giác máy tính mà SGD + Momentum sau khi tinh chỉnh kỹ lịch Cosine Annealing lại vượt Adam 1–2% độ chính xác kiểm định? Bài học này đặt các bộ tối ưu lên bàn cân thực nghiệm, phân tích lý do đằng sau khoảng cách tổng quát hóa và liệt kê những cạm bẫy tinh chỉnh hay gặp nhất.

## Kiến thức cần nhớ: Bốn bài học thực nghiệm khi so sánh và tinh chỉnh Bộ tối ưu

1. **Tốc độ hội tụ sớm vs. Chất lượng tổng quát hóa cuối cùng**:
   Adam/AdamW thích nghi siêu nhanh ở giai đoạn đầu và rất ít nhạy cảm với tốc độ học khởi tạo, nên là lựa chọn số 1 cho NLP/Transformer và thử nghiệm nhanh; SGD + Momentum tuy cần chỉnh `lr` kỹ hơn nhưng thường hội tụ vào các cực tiểu phẳng hơn trên CNN thị giác.
2. **Định luật phối hợp Siêu tham số (LR × Batch Size × Weight Decay)**:
   Khi tăng gấp $$k$$ lần `batch_size`, phương sai nhiễu gradient giảm đi $$k$$ lần — ta thường tăng learning rate theo quy tắc tuyến tính (Linear Scaling cho SGD) hoặc căn bậc hai ($$\sqrt{k}$$ cho Adam), kèm giai đoạn warmup.
3. **Không áp dụng Weight Decay cho Bias và LayerNorm/BatchNorm**:
   Phạt trọng số lên hệ số dịch `bias` hay hệ số co giãn $$\gamma$$ của LayerNorm/BatchNorm không giúp chống overfitting mà còn làm suy yếu khả năng biểu diễn; thực tế luôn tách `param_groups` để đặt `weight_decay=0.0` cho các tham số 1D này.
4. **Cạm bẫy kinh điển khi huấn luyện với Bộ tối ưu thích nghi**:
   Dùng chung một `lr` (ví dụ `0.1` của SGD mang sang Adam sẽ lập tức nổ loss!), quên giảm learning rate ở cuối quá trình (Cosine/Step Decay), hoặc chọn $$\beta_2 = 0{,}999$$ quá cao trên bài toán gradient thay đổi đột ngột (như GAN hay RL, nơi $$\beta_2 = 0{,}95$$ hoặc $$0{,}9$$ ổn định hơn nhiều).

### Giải thích: Tính tay quy tắc tách nhóm tham số Weight Decay và tỷ lệ điều chỉnh Learning Rate theo Batch Size

**Phần A — Tách nhóm tham số Weight Decay cho 1 tầng Linear `128 -> 64` kèm LayerNorm `64`**:
- Nhóm 1 (`weight_decay = 0.01`): ma trận trọng số `W` có $$128 \times 64 = \mathbf{8{,}192}$$ tham số (2D).
- Nhóm 2 (`weight_decay = 0.0`): `bias` (64) + `LayerNorm.weight` (64) + `LayerNorm.bias` (64) = **192** tham số (1D).

**Phần B — Điều chỉnh Learning Rate khi tăng Batch Size từ $$B_1 = 64$$ lên $$B_2 = 256$$ (gấp $$k = 4$$ lần)**:
- Với **SGD + Momentum** (Linear Scaling Rule): $$\eta_2 = k \cdot \eta_1 = 4 \times 0{,}01 = \mathbf{0{,}04}$$.
- Với **Adam / AdamW** (Square-Root Scaling Rule): $$\eta_2 = \sqrt{k} \cdot \eta_1 = \sqrt{4} \times 3\times 10^{-4} = \mathbf{6 \times 10^{-4}}$$.

Nắm vững hai quy tắc đơn giản này giúp bạn mở rộng huấn luyện từ 1 GPU lên 4–8 GPU mà giữ nguyên đường cong hội tụ!

```python
# Demo so sánh bộ tối ưu trên bài toán tối ưu 2D
print("="*70)
print("So sánh Bộ tối ưu trên Hàm Rosenbrock")
print("="*70)
print("Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²")
print("Cực tiểu tại (1, 1), nhưng thung lũng cong hẹp khiến tối ưu khó\n")

def rosenbrock(x, y):
    """Hàm kiểm thử tối ưu kinh điển với thung lũng cong hẹp"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    """Gradient của hàm Rosenbrock"""
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([[dx], [dy]])

# Khởi tạo tham số (bắt đầu xa cực tiểu)
theta_sgd = np.array([[-0.5], [0.5]])
theta_momentum = np.array([[-0.5], [0.5]])
theta_rmsprop = np.array([[-0.5], [0.5]])
theta_adam = np.array([[-0.5], [0.5]])

# Tạo bộ tối ưu
opt_sgd = type('SGD', (), {'lr': 0.001, 'params': [theta_sgd]})()
opt_momentum = SGDMomentum([theta_momentum], lr=0.001, momentum=0.9)
opt_rmsprop = RMSprop([theta_rmsprop], lr=0.01, beta=0.9)
opt_adam = Adam([theta_adam], lr=0.01, beta1=0.9, beta2=0.999)

# Theo dõi quỹ đạo
trajectories = {
    'SGD': [theta_sgd.copy()],
    'Momentum': [theta_momentum.copy()],
    'RMSprop': [theta_rmsprop.copy()],
    'Adam': [theta_adam.copy()]
}

# Tối ưu 500 bước
for step in range(500):
    # SGD thuần túy
    grad = rosenbrock_grad(theta_sgd[0,0], theta_sgd[1,0])
    theta_sgd -= opt_sgd.lr * grad
    trajectories['SGD'].append(theta_sgd.copy())
    
    # Momentum
    grad = rosenbrock_grad(theta_momentum[0,0], theta_momentum[1,0])
    opt_momentum.step([grad])
    trajectories['Momentum'].append(theta_momentum.copy())
    
    # RMSprop
    grad = rosenbrock_grad(theta_rmsprop[0,0], theta_rmsprop[1,0])
    opt_rmsprop.step([grad])
    trajectories['RMSprop'].append(theta_rmsprop.copy())
    
    # Adam
    grad = rosenbrock_grad(theta_adam[0,0], theta_adam[1,0])
    opt_adam.step([grad])
    trajectories['Adam'].append(theta_adam.copy())

# So sánh vị trí cuối
print("Vị trí cuối sau 500 bước:")
print(f"  SGD:      ({theta_sgd[0,0]:.4f}, {theta_sgd[1,0]:.4f})")
print(f"  Momentum: ({theta_momentum[0,0]:.4f}, {theta_momentum[1,0]:.4f})")
print(f"  RMSprop:  ({theta_rmsprop[0,0]:.4f}, {theta_rmsprop[1,0]:.4f})")
print(f"  Adam:     ({theta_adam[0,0]:.4f}, {theta_adam[1,0]:.4f})")
print(f"  Cực tiểu thật: (1.0000, 1.0000)")

print("\nQuan sát:")
print("- Momentum tăng tốc dọc thung lũng")
print("- RMSprop thích nghi với các độ cong khác nhau")
print("- Adam kết hợp lợi ích của cả hai")
print("- SGD thuần túy chậm nhất (mắc kẹt trong dao động)")
```

Tiếp theo là ví dụ minh họa trên quá trình huấn luyện một mạng nơ-ron thực tế:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Tạo tác vụ phân loại đơn giản
print("\n" + "="*70)
print("Huấn luyện Mạng Neuron với các Bộ tối ưu Khác nhau")
print("="*70)

# Sinh dữ liệu tổng hợp: bài toán kiểu XOR
np.random.seed(42)
n_samples = 1000

X = np.random.randn(n_samples, 2)
y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(float)  # XOR

X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y).unsqueeze(1)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Mạng đơn giản
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Huấn luyện với các bộ tối ưu khác nhau
optimizers_to_test = {
    'SGD': lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
    'SGD+Momentum': lambda model: torch.optim.SGD(model.parameters(), 
                                                   lr=0.1, momentum=0.9),
    'RMSprop': lambda model: torch.optim.RMSprop(model.parameters(), lr=0.01),
    'Adam': lambda model: torch.optim.Adam(model.parameters(), lr=0.01),
    'AdamW': lambda model: torch.optim.AdamW(model.parameters(), lr=0.01, 
                                            weight_decay=0.01)
}

results = {}

for name, optimizer_fn in optimizers_to_test.items():
    print(f"\nHuấn luyện với {name}...")
    
    # Tạo mô hình mới
    model = SimpleNet()
    optimizer = optimizer_fn(model)
    criterion = nn.BCELoss()
    
    # Huấn luyện
    losses = []
    for epoch in range(100):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            # Xuôi
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            # Ngược
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {losses[-1]:.4f}")
    
    # Độ chính xác kiểm tra
    model.eval()
    with torch.no_grad():
        pred = model(X_train)
        accuracy = ((pred > 0.5).float() == y_train).float().mean()
    
    results[name] = {
        'losses': losses,
        'final_loss': losses[-1],
        'accuracy': accuracy.item()
    }

# So sánh kết quả
print("\n" + "="*70)
print("Kết quả So sánh Bộ tối ưu")
print("="*70)
print(f"{'Bộ tối ưu':<15} | {'Loss cuối':<12} | {'Độ chính xác':<10}")
print("-" * 45)
for name, res in results.items():
    print(f"{name:<15} | {res['final_loss']:<12.4f} | {res['accuracy']:<10.2%}")

print("\nQuan sát then chốt:")
print("- Momentum tăng tốc hội tụ so với SGD thuần túy")
print("- Phương pháp thích nghi (RMSprop, Adam) hội tụ nhanh hơn")
print("- AdamW thường tổng quát hóa tốt nhất với weight decay")
print("- Lựa chọn quan trọng: chênh lệch tốc độ 2–5× là phổ biến")
```

Ví dụ về lịch trình điều chỉnh learning rate:

```python
class CosineAnnealingSchedule:
    """
    Lịch trình learning rate cosine annealing.
    
    Giảm dần learning rate theo đường cong cosine.
    Thường kết hợp warm restart để cải thiện hiệu năng.
    """
    
    def __init__(self, lr_max, lr_min, T_max):
        """
        lr_max: learning rate tối đa
        lr_min: learning rate tối thiểu
        T_max: chu kỳ của cosine (số vòng lặp)
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
    
    def get_lr(self, t):
        """Lấy learning rate tại vòng lặp t"""
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
               (1 + np.cos(np.pi * (t % self.T_max) / self.T_max))

# Ví dụ
schedule = CosineAnnealingSchedule(lr_max=0.1, lr_min=0.001, T_max=100)

print("\n" + "="*70)
print("Lịch trình Learning Rate")
print("="*70)

iterations = np.arange(300)
lrs = [schedule.get_lr(t) for t in iterations]

print("Tiến hóa learning rate (300 vòng lặp đầu):")
print(f"  Bắt đầu: {lrs[0]:.6f}")
print(f"  Sau 50 vòng: {lrs[50]:.6f}")
print(f"  Sau 100 vòng: {lrs[100]:.6f} (hết chu kỳ, khởi động lại)")
print(f"  Sau 150 vòng: {lrs[150]:.6f}")
print("\nCosine annealing giảm LR mượt, cho phép tinh chỉnh gần cực tiểu")
```

## 5. Khái niệm liên quan

Mối liên hệ giữa thuật toán tối ưu và hình học của bề mặt mất mát giải thích vì sao mỗi bộ tối ưu phát huy ưu thế trong những tình huống khác nhau. Bề mặt mất mát của mạng nơ-ron sâu có tính không lồi mạnh, chứa nhiều cực tiểu địa phương, điểm yên ngựa và vùng phẳng. Trong không gian nhiều chiều, điểm yên ngựa — nơi gradient bằng 0 nhưng không phải cực tiểu — xuất hiện phổ biến hơn nhiều so với cực tiểu địa phương. Momentum giúp vượt qua điểm yên ngựa nhờ vận tốc tích lũy đưa tham số qua vùng có gradient gần bằng 0. Trong khi đó, learning rate thích nghi phát huy hiệu quả khi độ cong giữa các chiều chênh lệch lớn, điều rất thường gặp trong mạng nơ-ron khi một số tham số (như bias) có gradient khá ổn định còn các tham số khác (như ma trận trọng số) có độ lớn gradient biến thiên mạnh.

So sánh với các phương pháp tối ưu bậc hai giúp làm rõ nền tảng lý thuyết của nhóm thuật toán thích nghi. Phương pháp Newton sử dụng đạo hàm bậc hai (ma trận Hessian) để hiệu chỉnh trực tiếp theo độ cong, cho tốc độ hội tụ nhanh hơn. Tuy nhiên, với mạng có hàng triệu tham số, việc tính và nghịch đảo ma trận Hessian đòi hỏi bộ nhớ $$O(n^2)$$ và chi phí tính toán $$O(n^3)$$, vượt quá khả năng thực tế. Các phương pháp như Adam xấp xỉ thông tin bậc hai thông qua thống kê gradient (moment bậc hai $$\mathbf{v}_t$$ liên hệ với các phần tử trên đường chéo của ma trận Hessian) mà vẫn giữ chi phí tính toán tuyến tính. Dù kém chính xác hơn phương pháp Newton đầy đủ, phép xấp xỉ này vẫn đủ để tăng tốc huấn luyện đáng kể trong thực tế.

Bộ tối ưu cũng tương tác chặt chẽ với batch normalization và các kỹ thuật chuẩn hóa khác. Batch normalization làm bề mặt mất mát mượt hơn và giảm độ nhạy với learning rate. Khi có chuẩn hóa, lợi thế của một số bộ tối ưu có thể thay đổi: chẳng hạn, Adam đi kèm batch normalization đôi khi hội tụ tới điểm cực tiểu có khả năng tổng quát hóa kém hơn SGD kết hợp momentum — hiện tượng thường gọi là "khoảng cách tổng quát hóa" (*generalization gap*). Do đó, việc chọn bộ tối ưu thường gắn liền với kiến trúc: các mô hình Transformer (sử dụng layer normalization) thường đạt kết quả tốt nhất với AdamW, trong khi các mạng ResNet (sử dụng batch normalization) thường ưu tiên SGD kết hợp momentum cho độ chính xác cuối cùng.

Sự chuyển dịch từ learning rate chỉnh tay sang các phương pháp thích nghi phản ánh một xu hướng rộng hơn trong học sâu: giảm bớt công sức tinh chỉnh siêu tham số thủ công. Trước đây, việc huấn luyện mạng nơ-ron đòi hỏi điều chỉnh rất kỹ learning rate, lịch trình giảm và hệ số momentum. Các bộ tối ưu thích nghi như Adam giúp giảm đáng kể gánh nặng này vì bộ tham số mặc định đã hoạt động ổn định trên nhiều bài toán. Dù vậy, sử dụng bộ tối ưu như một "hộp đen" mà không hiểu các giả định đằng sau vẫn có thể gây giảm hiệu năng trong những trường hợp đặc thù.

Lịch trình learning rate gắn trực tiếp với sự đánh đổi giữa khám phá và khai thác trong quá trình tối ưu. Ở giai đoạn đầu, bước học lớn giúp mô hình khám phá rộng để nhanh chóng tìm vùng tốt trong không gian tham số. Ở giai đoạn sau, bước học nhỏ hơn giúp tinh chỉnh tham số quanh điểm cực tiểu. Các lịch trình như cosine annealing hay step decay thực hiện cơ chế giảm dần này. Ngược lại, giai đoạn khởi động (*warm-up*) bắt đầu từ learning rate rất nhỏ rồi tăng dần, giúp ổn định quá trình huấn luyện khi dùng batch size rất lớn hoặc ở các bước đầu khi trọng số khởi tạo ngẫu nhiên tạo ra gradient bất ổn. Lịch trình warm-up trong bài báo Transformer gốc $$\eta_t = d_{\text{model}}^{-0.5} \min(t^{-0.5}, t \cdot \text{warmup}^{-1.5})$$ đã trở thành cấu hình tham chiếu phổ biến khi huấn luyện mô hình lớn.

## 6. Bài báo nền tảng

**["On the importance of initialization and momentum in deep learning" (2013)](http://proceedings.mlr.press/v28/sutskever13.html)**  
*Tác giả*: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton  
Bài báo này phân tích có hệ thống vai trò của momentum trong học sâu, cho thấy đây không chỉ là cải tiến phụ mà là yếu tố thiết yếu để huấn luyện mạng sâu. Các tác giả chứng minh rằng kết hợp momentum với cách khởi tạo trọng số phù hợp cho phép huấn luyện mạng sâu hơn nhiều so với SGD thuần túy, đồng thời giúp vượt qua điểm yên ngựa và giảm nhiễu từ gradient mini-batch. Bên cạnh phân tích lý thuyết nối momentum với tối ưu cổ điển, bài báo còn cho thấy Nesterov momentum đạt hiệu quả ổn định hơn momentum chuẩn trên nhiều thử nghiệm, qua đó khẳng định rằng thuật toán tối ưu cần được thiết kế phù hợp với đặc thù không lồi, nhiều chiều và có nhiễu của học sâu.

**["Adam: A Method for Stochastic Optimization" (2015)](https://arxiv.org/abs/1412.6980)**  
*Tác giả*: Diederik P. Kingma, Jimmy Ba  
Bài báo giới thiệu Adam và chứng minh hiệu quả trên nhiều bài toán như phân loại ảnh, mô hình ngôn ngữ và suy luận biến phân. Đóng góp chính là kết hợp learning rate thích nghi (tương tự RMSprop) với momentum và cơ chế hiệu chỉnh độ lệch ở các bước đầu. Với các giá trị mặc định $$\beta_1=0.9, \beta_2=0.999$$ hoạt động tốt trên nhiều bài toán, Adam giảm đáng kể yêu cầu tinh chỉnh siêu tham số và trở thành lựa chọn mặc định rộng rãi, đặc biệt trong NLP nơi đặc trưng từ vựng rất thưa. Bài báo cũng đề xuất biến thể AdaMax (dùng chuẩn $$L_\infty$$ thay cho $$L_2$$) và cung cấp phân tích cận hối tiếc (*regret bound*) trong khung tối ưu lồi trực tuyến.

**["Decoupled Weight Decay Regularization" (2019)](https://arxiv.org/abs/1711.05101)**  
*Tác giả*: Ilya Loshchilov, Frank Hutter  
Bài báo chỉ ra điểm bất hợp lý trong cách Adam xử lý chính quy hóa L2 và đề xuất AdamW. Khi cộng weight decay vào gradient rồi đưa qua phép chuẩn hóa thích nghi của Adam, mức suy giảm trọng số thực tế thay đổi theo thống kê gradient — những tham số có gradient lớn lại bị phạt nhẹ hơn. AdamW tách weight decay ra khỏi bước cập nhật gradient và áp dụng trực tiếp lên tham số. Thực nghiệm cho thấy AdamW cải thiện rõ rệt khả năng tổng quát hóa, đặc biệt trên Transformer, và dần thay thế Adam trong huấn luyện các mô hình ngôn ngữ lớn.

**["On the Variance of the Adaptive Learning Rate and Beyond" (2020)](https://arxiv.org/abs/1908.03265)**  
*Tác giả*: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han  
Bài báo phân tích nguyên nhân khiến Adam đôi khi tổng quát hóa kém hơn SGD dù hội tụ nhanh hơn ("khoảng cách tổng quát hóa"). Các tác giả chỉ ra rằng phương sai lớn của learning rate thích nghi ở giai đoạn đầu có thể đẩy mô hình vào các cực tiểu nhọn (tổng quát hóa kém), trong khi SGD với momentum thường tiến tới cực tiểu phẳng hơn. Từ đó, bài báo đề xuất RAdam (*Rectified Adam*) với cơ chế hiệu chỉnh thận trọng hơn khi thống kê gradient chưa ổn định, góp phần làm rõ sự đánh đổi giữa tốc độ hội tụ và khả năng tổng quát hóa.

**["Lookahead Optimizer: k steps forward, 1 step back" (2019)](https://arxiv.org/abs/1907.08610)**  
*Tác giả*: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba  
Bài báo đề xuất một thuật toán tối ưu bọc ngoài (*meta-optimizer*) có thể kết hợp với bất kỳ bộ tối ưu cơ sở nào như SGD hay Adam. Lookahead duy trì song song hai bộ trọng số: trọng số nhanh được cập nhật $$k$$ bước (thường 5–10 bước) bởi bộ tối ưu cơ sở, sau đó trọng số chậm được nội suy tiến về phía trọng số nhanh rồi đặt lại trọng số nhanh về vị trí trọng số chậm. Cơ chế này giúp giảm phương sai quỹ đạo tối ưu, ổn định quá trình huấn luyện và thường cải thiện khả năng tổng quát hóa trên các bài toán thị giác lẫn ngôn ngữ.

## Cạm bẫy thường gặp và Mẹo

Một lỗi phổ biến khi dùng bộ tối ưu thích nghi như Adam là thay đổi batch size nhưng giữ nguyên các siêu tham số. Với SGD thuần túy, khi tăng gấp đôi batch size, ta thường cần tăng gần gấp đôi learning rate để giữ mức cập nhật tương đương. Với Adam, mối quan hệ phức tạp hơn vì bước cập nhật đã tự chuẩn hóa theo độ lớn gradient; quy tắc kinh nghiệm thường là tăng learning rate nhẹ hơn (chẳng hạn theo tỷ lệ $$\sqrt{2}$$ thay vì $$2$$) và theo dõi kỹ kết quả trên tập validation. Khi batch size rất lớn (hàng nghìn mẫu), nên kết hợp thêm giai đoạn warm-up để tránh bất ổn ở các bước đầu.

Một điểm cần lưu ý khác là trạng thái nội bộ của bộ tối ưu khi fine-tune mô hình tiền huấn luyện. Nếu tải trọng số tiền huấn luyện và bắt đầu chạy Adam mới, các bộ đệm momentum và phương sai đều khởi tạo bằng 0, chưa phản ánh đúng thang gradient của mô hình đã gần hội tụ, dễ gây nhiễu loạn trọng số ban đầu. Cách xử lý phù hợp là dùng learning rate nhỏ hơn khi fine-tune để bộ tối ưu tích lũy trạng thái dần dần, hoặc thiết lập lại trạng thái bộ tối ưu cùng lịch trình warm-up ngắn khi chuyển sang giai đoạn huấn luyện mới.

Hệ số weight decay của AdamW cũng cần được chỉnh khác với SGD. Trong SGD, giá trị thường nằm quanh 0,0001–0,001; còn trong AdamW, do weight decay đã được tách khỏi bước chia thích nghi, giá trị phổ biến hơn nằm trong khoảng 0,01–0,1. Khi chuyển từ Adam hoặc SGD sang AdamW, không nên giữ nguyên giá trị cũ mà nên bắt đầu thử từ khoảng 0,01 rồi điều chỉnh dựa trên chênh lệch giữa tập huấn luyện và tập kiểm tra.

Gradient clipping cũng tương tác trực tiếp với các bộ tối ưu thích nghi. Nếu cắt gradient trước khi đưa vào Adam, cả ước lượng moment bậc nhất lẫn bậc hai đều bị giới hạn theo ngưỡng cắt (ví dụ, nếu chuẩn gradient bị giới hạn ở 5 thì bình phương độ lớn tối đa là 25). Tác động này vừa có thể hữu ích (ngăn learning rate hiệu dụng sụt quá nhỏ), vừa có thể hạn chế khả năng thích nghi với thang gradient thật. Thông thường, nên dùng gradient clipping cho RNN và Transformer để đảm bảo ổn định, còn với các mạng đã ổn định thì có thể bỏ qua.

Cuối cùng, lịch trình learning rate chu kỳ (*cyclical learning rate*) — thay đổi learning rate luân phiên giữa hai ngưỡng trong quá trình huấn luyện — có thể giúp mô hình thoát khỏi cực tiểu địa phương và tìm được vùng nghiệm tốt hơn. Khi kết hợp với *snapshot ensembling* (lưu lại các checkpoint ở cuối mỗi chu kỳ rồi kết hợp dự đoán của chúng), kỹ thuật này có thể cải thiện hiệu năng mà không tốn thêm đáng kể chi phí huấn luyện.

## Điểm then chốt

Các thuật toán tối ưu nâng cao cải thiện hạ gradient thuần túy thông qua hai ý tưởng chính: tích lũy momentum để tăng tốc theo hướng nhất quán và giảm dao động, cùng với việc thích nghi learning rate cho từng tham số dựa trên lịch sử gradient. SGD kết hợp momentum xây dựng vận tốc từ trung bình trượt mũ của gradient, giúp vượt qua bề mặt dạng hẻm núi và các vùng phẳng. RMSprop thích nghi bước học theo trung bình trượt mũ của bình phương gradient, tự cân bằng cập nhật theo độ lớn gradient đặc trưng của từng tham số. Adam kết hợp cả hai cơ chế cùng bước hiệu chỉnh độ lệch ở giai đoạn đầu, trở thành lựa chọn mặc định đáng tin cậy nhờ ít nhạy với siêu tham số. AdamW hoàn thiện Adam bằng cách tách weight decay khỏi bước chuẩn hóa thích nghi, đảm bảo chính quy hóa đồng đều hơn và đặc biệt hiệu quả với Transformer lớn.

Không có bộ tối ưu nào tốt nhất cho mọi bài toán; việc lựa chọn luôn đòi hỏi cân bằng giữa tốc độ hội tụ, khả năng tổng quát hóa, chi phí tính toán và mức độ nhạy cảm với siêu tham số. Trong thực tế, người làm mô hình thường dùng Adam hoặc AdamW để thử nghiệm nhanh, và có thể chuyển sang SGD kết hợp momentum ở lần huấn luyện cuối nếu ưu tiên khả năng tổng quát hóa trên các bài toán thị giác. Dù các biến thể ngày càng đa dạng, tất cả vẫn dựa trên cùng một nguyên lý: sử dụng gradient từ lan truyền ngược để tinh chỉnh tham số theo từng bước.

## Video tham khảo

- **Xem trước:** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Tổng quan so sánh trực tiếp nhóm bộ tối ưu mà bài học triển khai.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Giải thích các điểm dễ sai của Adam như hiệu chỉnh độ lệch và thang learning rate.
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI) — StatQuest with Josh Starmer, tiếng Anh. Ôn lại SGD — mốc cơ sở để đánh giá mức cải thiện của các bộ tối ưu nâng cao.
