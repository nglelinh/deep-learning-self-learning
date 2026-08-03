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

Bây giờ demo trên huấn luyện mạng neuron thực:

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

Demo lịch trình learning rate:

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
Mối quan hệ giữa thuật toán tối ưu và hình học của bề mặt loss làm sáng tỏ vì sao các bộ tối ưu khác nhau xuất sắc trong các tình huống khác nhau. Bề mặt loss của mạng neuron sâu rất không lồi, có cực tiểu địa phương, điểm yên ngựa và cao nguyên. Điểm yên ngựa — nơi gradient bằng zero nhưng ta không ở cực tiểu — đặc biệt phổ biến ở chiều cao. Momentum giúp thoát điểm yên ngựa bằng cách xây vận tốc mang qua vùng gradient zero. Learning rate thích nghi giúp khi các hướng có độ cong khác nhau rất lớn — phổ biến trong mạng neuron nơi một số tham số (như bias) nhận gradient tương tự liên tục trong khi số khác (như trọng số) có độ lớn gradient biến thiên mạnh.

Liên hệ với phương pháp tối ưu bậc hai cung cấp bối cảnh lý thuyết. Phương pháp Newton dùng đạo hàm bậc hai (ma trận Hessian) để tính đến độ cong, cho phép hội tụ nhanh hơn. Tuy nhiên, tính và nghịch đảo Hessian cho mạng với hàng triệu tham số là cấm đoán về tính toán — bộ nhớ $$O(n^2)$$ và tính toán $$O(n^3)$$. Các phương pháp learning rate thích nghi như Adam xấp xỉ thông tin bậc hai qua thống kê gradient (moment bậc hai $$\mathbf{v}_t$$ liên quan các phần tử chéo Hessian) mà không tốn chi phí cấm đoán. Thông tin độ cong xấp xỉ này, dù thô hơn Newton đầy đủ, cung cấp đủ lợi ích để tăng tốc huấn luyện rõ rệt trong khi vẫn thực tế về tính toán.

Bộ tối ưu tương tác mật thiết với chuẩn hóa batch và các kỹ thuật chuẩn hóa khác. Chuẩn hóa batch thay đổi hình học bề mặt loss, làm nó mượt hơn và giảm độ nhạy với learning rate. Tương tác này có thể tinh tế: một số bộ tối ưu hoạt động tốt không có chuẩn hóa có thể kém lợi thế hơn khi có. Adam với chuẩn hóa batch đôi khi hội tụ tới cực tiểu kém hơn SGD với momentum, hiện tượng gọi là “khoảng cách tổng quát hóa” (*generalization gap*). Hiểu các tương tác này hướng dẫn lựa chọn bộ tối ưu dựa trên kiến trúc — Transformer (dùng chuẩn hóa tầng) thường hoạt động tốt nhất với AdamW, trong khi ResNet (có chuẩn hóa batch) có thể ưa SGD với momentum cho hiệu năng cuối.

Sự tiến hóa từ learning rate chỉnh tay sang phương pháp thích nghi đại diện cho xu hướng rộng hơn trong học sâu: tự động hóa lựa chọn siêu tham số. Huấn luyện mạng neuron sớm đòi hỏi chỉnh learning rate, lịch trình và hệ số momentum rất kỹ. Các bộ tối ưu thích nghi hiện đại giảm gánh nặng này — siêu tham số mặc định của Adam hoạt động hợp lý trên nhiều tác vụ. Dân chủ hóa học sâu này làm lĩnh vực dễ tiếp cận hơn, nhưng cũng tạo rủi ro: dùng bộ tối ưu hộp đen mà không hiểu giả định của chúng có thể dẫn đến hiệu năng kém trong trường hợp biên. Những người thực hành giỏi nhất hiểu cả thuật toán lẫn khi giả định của chúng sụp đổ.

Lịch trình learning rate nối với tradeoff khám phá–khai thác trong tối ưu. Đầu huấn luyện, ta muốn khám phá rộng, bước lớn để tìm vùng tốt của không gian tham số. Sau đó, ta muốn khai thác, bước nhỏ hơn để tinh chỉnh tham số gần cực tiểu. Các lịch trình như cosine annealing hoặc step decay chính thức hóa điều này, giảm learning rate khi huấn luyện tiến triển. Lịch trình warm-up làm ngược lại ban đầu — bắt đầu với learning rate rất nhỏ và tăng dần — giúp khi dùng batch rất lớn hoặc khi tham số khởi tạo ngẫu nhiên và gradient ban đầu có thể gây nhầm. Lịch trình warm-up của bài Transformer $$\eta_t = d_{\text{model}}^{-0.5} \min(t^{-0.5}, t \cdot \text{warmup}^{-1.5})$$ đã trở thành chuẩn khi huấn luyện mô hình lớn.

## 6. Bài báo Nền tảng

**["On the importance of initialization and momentum in deep learning" (2013)](http://proceedings.mlr.press/v28/sutskever13.html)**  
*Tác giả*: Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton  
Bài báo này phân tích nghiêm ngặt lợi ích của momentum cho học sâu, cho thấy nó không chỉ là cải thiện nhỏ mà thiết yếu để huấn luyện mạng sâu hiệu quả. Các tác giả chứng minh momentum kết hợp khởi tạo đúng (họ dùng lược đồ cụ thể cho các loại tầng khác nhau) cho phép huấn luyện mạng sâu hơn nhiều so với SGD thuần túy. Họ cho thấy momentum giúp thoát điểm yên ngựa và giảm tác động của gradient nhiễu từ lấy mẫu mini-batch. Quan trọng, họ cung cấp phân tích lý thuyết về động lực của momentum, nối nó với lý thuyết tối ưu cổ điển đồng thời chứng minh ưu thế cụ thể cho bề mặt loss không lồi của mạng neuron. Bài báo thiết lập Nesterov momentum là đặc biệt hiệu quả, hơi nhưng nhất quán vượt momentum chuẩn. Công trình này ảnh hưởng hiểu biết của lĩnh vực rằng thuật toán tối ưu phải được điều chỉnh cho thách thức độc đáo của học sâu — chiều cao, không lồi, gradient nhiễu — thay vì chỉ áp dụng phương pháp tối ưu cổ điển.

**["Adam: A Method for Stochastic Optimization" (2015)](https://arxiv.org/abs/1412.6980)**  
*Tác giả*: Diederik P. Kingma, Jimmy Ba  
Bài báo này giới thiệu Adam và chứng minh hiệu quả của nó trên nhiều tác vụ gồm phân loại ảnh, mô hình ngôn ngữ và suy luận biến phân. Đóng góp then chốt là kết hợp learning rate thích nghi (như RMSprop) với momentum, đồng thời bao gồm hiệu chỉnh độ lệch để đảm bảo hành vi tốt từ cập nhật đầu tiên. Kingma và Ba cho thấy Adam đòi hỏi tối thiểu chỉnh siêu tham số — giá trị mặc định $$\beta_1=0.9, \beta_2=0.999$$ hoạt động tốt trên các bài toán — khiến nó dễ tiếp cận với người thực hành không thể chỉnh kỹ. So sánh thực nghiệm của bài cho thấy Adam nhất quán khớp hoặc vượt các bộ tối ưu khác đồng thời vững với lựa chọn learning rate. Adam trở thành bộ tối ưu mặc định cho nhiều ứng dụng, đặc biệt trong NLP nơi thích nghi với thống kê gradient giúp với từ vựng thưa. Bài cũng giới thiệu AdaMax (biến thể dùng chuẩn $$L_\infty$$ thay vì $$L_2$$) và cung cấp phân tích regret bound nối Adam với lý thuyết tối ưu lồi trực tuyến, dù các khía cạnh lý thuyết này ít được dùng hơn thuật toán thực tiễn.

**["Decoupled Weight Decay Regularization" (2019)](https://arxiv.org/abs/1711.05101)**  
*Tác giả*: Ilya Loshchilov, Frank Hutter  
Bài báo này nhận diện một lỗi tinh tế nhưng quan trọng trong cách Adam xử lý chính quy hóa L2 và đề xuất AdamW làm giải pháp. Các tác giả cho thấy việc thêm weight decay vào gradient (thực hành chuẩn) rồi áp dụng learning rate thích nghi (như Adam làm) khiến weight decay hiệu dụng biến thiên giữa các tham số dựa trên thống kê gradient của chúng. Sự ghép nối này làm suy yếu chính quy hóa — tham số với gradient lớn nhận ít weight decay hơn, ngược với điều mong muốn. AdamW tách weight decay khỏi cập nhật dựa trên gradient, áp dụng nó trực tiếp lên tham số sau cập nhật thích nghi. Bài chứng minh tổng quát hóa cải thiện trên nhiều benchmark, đặc biệt với Transformer nơi chính quy hóa đúng là then chốt. AdamW phần lớn đã thay Adam khi huấn luyện mô hình ngôn ngữ lớn và các hệ thống dựa trên Transformer. Công trình minh họa cách hiểu tương tác giữa các thành phần huấn luyện khác nhau (tối ưu + chính quy hóa) tiết lộ các vấn đề tinh tế ảnh hưởng đáng kể hiệu năng thực tiễn.

**["On the Variance of the Adaptive Learning Rate and Beyond" (2020)](https://arxiv.org/abs/1908.03265)**  
*Tác giả*: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han  
Bài báo này phân tích vì sao Adam đôi khi tổng quát hóa kém hơn SGD dù hội tụ nhanh hơn, hiện tượng gọi là “khoảng cách tổng quát hóa”. Các tác giả cho thấy learning rate thích nghi của Adam có thể dẫn đến cực tiểu nhọn (loss huấn luyện thấp nhưng tổng quát hóa kém) trong khi SGD với momentum có xu hướng tìm cực tiểu phẳng hơn (tổng quát hóa tốt hơn). Họ đề xuất RAdam (Rectified Adam), chỉnh hiệu chỉnh độ lệch bảo thủ hơn đầu huấn luyện khi thống kê gradient không tin cậy. Bài sâu hơn hiểu biết về tradeoff tối ưu–tổng quát hóa: hội tụ nhanh hơn không luôn nghĩa là hiệu năng cuối tốt hơn. Nó cho thấy phương sai trong learning rate thích nghi có thể có hại và đề xuất kỹ thuật giảm phương sai. Công trình này ảnh hưởng cách người thực hành dùng Adam — nhận ra khi cơ chế thích nghi giúp (gradient thưa, thang biến thiên) so với khi phương pháp đơn giản hơn với tính chất tổng quát hóa tốt hơn (SGD+momentum) ưu việt hơn.

**["Lookahead Optimizer: k steps forward, 1 step back" (2019)](https://arxiv.org/abs/1907.08610)**  
*Tác giả*: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba  
Bài báo này giới thiệu thuật toán meta-tối ưu bao quanh bất kỳ bộ tối ưu cơ sở nào (SGD, Adam, v.v.). Lookahead duy trì hai tập trọng số: trọng số nhanh được cập nhật bởi bộ tối ưu cơ sở và trọng số chậm đồng bộ định kỳ với trọng số nhanh. Thuật toán chạy bộ tối ưu cơ sở $$k$$ bước (thường 5–10), rồi cập nhật trọng số chậm hướng về trọng số nhanh, rồi reset trọng số nhanh về trọng số chậm. Điều này giảm phương sai quỹ đạo tối ưu và cải thiện hội tụ. Bài cho thấy Lookahead cải thiện hiệu năng của bộ tối ưu cơ sở nhất quán trên các tác vụ, cung cấp huấn luyện ổn định hơn và thường tổng quát hóa tốt hơn. Dù ít dùng hơn Adam hay SGD+momentum, Lookahead chứng minh rằng thuật toán tối ưu có thể được ghép — ta có thể xây meta-thuật toán tăng cường bộ tối ưu hiện có. Phân tích thực nghiệm của bài trên tác vụ thị giác và ngôn ngữ thiết lập rằng thiết kế bộ tối ưu vẫn là lĩnh vực nghiên cứu đang hoạt động với không gian đổi mới ngoài các cổ điển.

## Cạm bẫy Thường gặp và Mẹo

Lỗi phổ biến nhất khi dùng bộ tối ưu thích nghi như Adam là quên điều chỉnh siêu tham số khi đổi batch size. Với SGD thuần túy, nhân đôi batch size xấp xỉ đòi hỏi nhân đôi learning rate để giữ cập nhật tham số tương đương (vì gradient được trung bình trên batch). Nhưng với Adam, quan hệ phức tạp hơn vì learning rate thích nghi đã tính đến độ lớn gradient. Quy tắc thực tiễn: khi tăng batch size, tăng learning rate tỷ lệ nhưng ít mạnh hơn (có lẽ $$\sqrt{2}$$ thay vì $$2$$), và theo dõi cẩn thận hiệu năng validation. Batch size rất lớn (hàng nghìn) có thể cần warm-up learning rate để ngăn bất ổn sớm.

Một vấn đề tinh tế là tích lũy trạng thái bộ tối ưu khi fine-tune mô hình tiền huấn luyện. Nếu bạn tải mô hình tiền huấn luyện và tiếp tục huấn luyện với Adam, các ước lượng momentum và phương sai bắt đầu từ zero, không từ giá trị phù hợp với mô hình gần hội tụ. Điều này có thể gây bất ổn hoặc ngăn fine-tune cải thiện mô hình. Giải pháp: hoặc dùng learning rate thấp hơn cho fine-tune (cho phép gradient xây trạng thái bộ tối ưu an toàn) hoặc reset trạng thái bộ tối ưu khi tải checkpoint, bắt đầu mới. Hiểu rằng bộ tối ưu duy trì trạng thái nội bộ ngoài tham số giúp gỡ lỗi hành vi fine-tune bất ngờ.

Weight decay trong AdamW đòi hỏi hiệu chỉnh khác SGD. Với SGD, weight decay khoảng 0,0001–0,001 là điển hình. Với AdamW, giá trị khoảng 0,01–0,1 thường hoạt động tốt hơn vì sự tách rời thay đổi cường độ hiệu dụng. Khi chuyển từ Adam sang AdamW, đừng chỉ bật weight decay với giá trị chỉnh cho SGD — bạn rất có thể chính quy hóa quá mức. Bắt đầu với 0,01 và chỉnh dựa trên khoảng cách train–test. Điều này minh họa nguyên lý rộng hơn: siêu tham số không bất biến với kiến trúc mà phải được chỉnh trong ngữ cảnh cấu hình huấn luyện đầy đủ.

Gradient clipping tương tác với bộ tối ưu theo cách không hiển nhiên. Với Adam, cắt gradient trước khi bộ tối ưu thấy chúng ảnh hưởng cả ước lượng momentum và phương sai. Nếu gradient bị cắt về chuẩn 5, moment bậc hai tối đa trở thành 25, giới hạn scale thích nghi. Điều này có thể có lợi (ngăn learning rate hiệu dụng cực nhỏ) hoặc có hại (ngăn thích nghi với thang gradient thật). Để ổn định, cắt gradient cho RNN và Transformer. Để tối đa tính thích nghi của Adam trên mạng hành xử tốt, bỏ clipping. Hiểu tradeoff này giúp chọn cấu hình phù hợp.

Một kỹ thuật mạnh cho chỉnh siêu tham số là learning rate chu kỳ — biến thiên learning rate giữa các cận trong huấn luyện. Điều này cho phép mô hình định kỳ thoát cực tiểu địa phương mà nó có thể mắc kẹt, tiềm năng tìm nghiệm tốt hơn. Kết hợp snapshot ensembling (lưu mô hình tại các điểm khác nhau trong chu kỳ và ensemble dự đoán của chúng), điều này có thể cải thiện hiệu năng vượt huấn luyện mô hình đơn với learning rate cố định. Chi phí tính toán tối thiểu (chỉ lịch trình) trong khi lợi ích có thể đáng kể, khiến nó là mẹo chưa được tận dụng đầy đủ trong bộ công cụ của người thực hành.

## Điểm then chốt

Các thuật toán tối ưu nâng cao cải thiện hạ gradient thuần túy bằng cách kết hợp momentum để tăng tốc theo hướng nhất quán và dập dao động, và bằng cách thích nghi learning rate theo từng tham số dựa trên lịch sử gradient. SGD với momentum xây vận tốc từ trung bình gradient có trọng số mũ, giúp vượt hẻm núi và thoát cao nguyên. RMSprop thích nghi learning rate bằng trung bình mũ của bình phương gradient, tự động scale cập nhật dựa trên độ lớn gradient điển hình theo từng tham số. Adam kết hợp cả hai cơ chế đồng thời bao gồm hiệu chỉnh độ lệch cho hành vi đúng ở vòng lặp sớm, trở thành chuẩn thực tế cho nhiều ứng dụng nhờ hiệu năng vững với tối thiểu chỉnh. AdamW cải thiện Adam bằng cách tách weight decay khỏi cập nhật dựa trên gradient, đảm bảo cường độ chính quy hóa độc lập với scale thích nghi, then chốt khi huấn luyện Transformer lớn. Lựa chọn bộ tối ưu liên quan tradeoff giữa tốc độ hội tụ, hiệu năng cuối, chi phí tính toán, và độ nhạy siêu tham số, không có bộ tối ưu nào thống trị mọi tình huống. Hiểu giả định của từng bộ tối ưu — hình học bề mặt loss nào nó xử lý tốt, thống kê gradient nào nó kỳ vọng — cho phép khớp thuật toán với bài toán hiệu quả. Thực hành hiện đại thường dùng Adam hoặc AdamW cho thử nghiệm ban đầu nhờ độ vững, có thể chuyển sang SGD với momentum cho huấn luyện cuối nếu cần tổng quát hóa tốt hơn. Sự tinh vi của các thuật toán này không nên che khuất nguyên lý nền tảng: tất cả chúng đều dùng gradient tính qua lan truyền ngược để cải thiện tham số lặp, chỉ khác ở cách chúng xử lý gradient thành cập nhật tham số.

Sự tiến hóa của thuật toán tối ưu từ hạ gradient thuần túy đến phương pháp thích nghi hiện đại đại diện cho việc lĩnh vực học cách tự động hóa các khía cạnh huấn luyện trước đây đòi hỏi chỉnh tay của chuyên gia, dân chủ hóa học sâu đồng thời cũng giới thiệu các tinh tế mới mà người thực hành phải hiểu để huấn luyện mô hình hiệu quả.
