---
layout: post
title: 05-01-02-02 Mô hình ngôn ngữ cấp ký tự, Papers và Bẫy thường gặp
chapter: '05'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter05
---

```python
# Ví dụ mô hình ngôn ngữ cấp ký tự
print("\n" + "="*70)
print("Mô hình ngôn ngữ cấp ký tự với PyTorch RNN")
print("="*70)

# Tạo dataset văn bản đơn giản
text = "hello world, deep learning is amazing! transformers are powerful."
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary: {chars}")
print(f"Vocabulary size: {vocab_size}")

# Chuẩn bị chuỗi: cho "hell" dự đoán "ello"
def create_sequences(text, seq_length=10):
    """Tạo các chuỗi huấn luyện từ văn bản"""
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]
        target = text[i+1:i+seq_length+1]
        
        # Chuyển thành chỉ số
        seq_idx = [char_to_idx[ch] for ch in seq]
        target_idx = [char_to_idx[ch] for ch in target]
        
        sequences.append(seq_idx)
        targets.append(target_idx)
    
    return sequences, targets

seq_length = 15
sequences, targets = create_sequences(text, seq_length)

# Chuyển thành tensor và tạo mã hóa one-hot
def to_onehot(sequences, vocab_size):
    """Chuyển chuỗi chỉ số thành tensor one-hot"""
    one_hot = []
    for seq in sequences:
        seq_onehot = torch.zeros(len(seq), vocab_size)
        for t, idx in enumerate(seq):
            seq_onehot[t, idx] = 1
        one_hot.append(seq_onehot)
    return torch.stack(one_hot)

X = to_onehot(sequences, vocab_size)
y = torch.tensor(targets, dtype=torch.long)

print(f"\nDataset: {len(sequences)} sequences of length {seq_length}")
print(f"Input shape: {X.shape}")  # (num_sequences, seq_length, vocab_size)
print(f"Target shape: {y.shape}")  # (num_sequences, seq_length)

# Tạo model
model = RNNSequenceModel(input_size=vocab_size, hidden_size=32, 
                         output_size=vocab_size, num_layers=2)

# Loss và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Huấn luyện
print("\nHuấn luyện mô hình ngôn ngữ cấp ký tự...")
model.train()

for epoch in range(500):
    # Forward pass
    outputs, _ = model(X)  # (batch, seq_len, vocab_size)
    
    # Reshape cho cross-entropy: (batch * seq_len, vocab_size)
    outputs_flat = outputs.view(-1, vocab_size)
    targets_flat = y.view(-1)
    
    # Tính loss
    loss = criterion(outputs_flat, targets_flat)
    
    # Backward pass và tối ưu
    optimizer.zero_grad()
    loss.backward()  # BPTT diễn ra tự động ở đây!
    
    # Gradient clipping (cốt yếu cho RNN!)
    # Không có bước này, gradient có thể bùng nổ và huấn luyện phân kỳ
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

# Sinh văn bản bằng cách lấy mẫu từ phân phối đã học
print("\n" + "="*70)
print("Sinh văn bản từ RNN đã học")
print("="*70)

def generate_text(model, start_text, length=50):
    """
    Sinh văn bản tự hồi quy (autoregressive) bằng RNN đã huấn luyện.
    
    Ta bắt đầu bằng seed text, dự đoán phân phối xác suất ký tự tiếp theo,
    lấy mẫu từ đó, nối vào chuỗi, và lặp lại.
    Đây là sinh tự hồi quy — mỗi dự đoán điều kiện trên
    tất cả các dự đoán trước đó.
    """
    model.eval()
    
    # Chuyển seed text thành chỉ số
    current_seq = [char_to_idx[ch] for ch in start_text]
    generated = start_text
    
    # Trạng thái ẩn mang thông tin qua quá trình sinh
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            # Chuẩn bị đầu vào: seq_length ký tự cuối (hoặc pad nếu ngắn hơn)
            input_seq = current_seq[-seq_length:] if len(current_seq) >= seq_length else current_seq
            
            # Pad nếu cần
            while len(input_seq) < seq_length:
                input_seq = [char_to_idx[' ']] + input_seq
            
            # Chuyển thành one-hot
            x = torch.zeros(1, seq_length, vocab_size)
            for t, idx in enumerate(input_seq):
                x[0, t, idx] = 1
            
            # Dự đoán ký tự tiếp theo
            output, hidden = model(x, hidden)
            
            # Lấy xác suất cho ký tự tiếp theo (bước thời gian cuối)
            probs = torch.softmax(output[0, -1], dim=0)
            
            # Lấy mẫu từ phân phối (thú vị hơn argmax)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            
            generated += next_char
            current_seq.append(next_idx)
    
    return generated

# Sinh
seed = "deep "
generated_text = generate_text(model, seed, length=50)
print(f"Seed: '{seed}'")
print(f"Generated: '{generated_text}'")
print("\nMô hình đã học các mẫu cấp ký tự!")
print("Với nhiều dữ liệu và huấn luyện hơn, RNN có thể sinh văn bản mạch lạc.")
```

Hãy cũng minh họa vấn đề gradient biến mất một cách thực nghiệm:

```python
print("\n" + "="*70)
print("Minh họa Gradient biến mất trong RNN")
print("="*70)

def analyze_gradient_flow(sequence_lengths=[5, 10, 20, 50]):
    """
    Cho thấy gradient suy giảm thế nào theo độ dài chuỗi.
    
    Ta tạo chuỗi các độ dài khác nhau, tính gradient, và
    đo độ lớn của chúng. Điều này minh họa thực nghiệm vì sao RNN vanilla
    gặp khó với phụ thuộc tầm xa.
    """
    results = []
    
    for seq_len in sequence_lengths:
        # Tạo RNN đơn giản
        rnn_test = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
        
        # Chuỗi đầu vào ngẫu nhiên
        x = torch.randn(1, seq_len, 10, requires_grad=True)
        
        # Forward pass
        out, hidden = rnn_test(x)
        
        # Tính loss chỉ từ đầu ra bước thời gian ĐẦU TIÊN
        # Gradient phải backprop qua seq_len-1 bước để tới h_1
        loss = out[:, 0, :].sum()
        
        # Backward pass
        loss.backward()
        
        # Đo độ lớn gradient tại đầu vào
        grad_magnitude = x.grad.abs().mean().item()
        
        results.append((seq_len, grad_magnitude))
        print(f"Sequence length {seq_len:2d}: Gradient magnitude = {grad_magnitude:.6f}")
    
    # Thường thấy suy giảm cấp số nhân của độ lớn gradient
    print("\nQuan sát: Gradient suy giảm cấp số nhân theo độ dài chuỗi!")
    print("Đây là vấn đề gradient biến mất giới hạn RNN vanilla.")
    
    return results

gradient_analysis = analyze_gradient_flow()
```

## 5. Các Khái niệm Liên quan

Mối quan hệ giữa RNN và các mạng feedforward chiếu sáng các nguyên tắc cơ bản về thiết kế kiến trúc mạng. Mạng feedforward giả định các đầu vào là các mẫu độc lập, đồng nhất phân phối — thứ tự ta trình bày ảnh trong huấn luyện không quan trọng vì mỗi ảnh được xử lý cô lập. RNN, ngược lại, mô hình tường minh các phụ thuộc giữa các đầu vào tuần tự thông qua trạng thái ẩn. Sự khác biệt này không chỉ về kiến trúc; nó phản ánh các giả định khác nhau về cấu trúc dữ liệu. Khi ta chọn RNN thay vì mạng feedforward, ta mã hóa thiên kiến qui nạp rằng thứ tự thời gian hoặc tuần tự mang thông tin liên quan đến tác vụ.

Kết nối với máy trạng thái hữu hạn và hệ thống động cung cấp nhận thức lý thuyết sâu hơn. Một RNN với trạng thái ẩn rời rạc và kích hoạt ngưỡng cứng về cơ bản là một máy trạng thái hữu hạn, chuyển đổi giữa các trạng thái dựa trên đầu vào. Với trạng thái ẩn liên tục và kích hoạt trơn, RNN trở thành hệ thống động liên tục được mô tả bởi phương trình sai phân $$\mathbf{h}_{t+1} = f(\mathbf{W}_{hh}\mathbf{h}_t + \mathbf{W}_{xh}\mathbf{x}_t)$$. Tính ổn định và biểu diễn của hệ thống động này phụ thuộc vào phổ của $$\mathbf{W}_{hh}$$ — các trị riêng của nó xác định hệ thống là ổn định, hỗn loạn, hay ổn định biên. Kết nối này với lý thuyết hệ thống động giúp giải thích các hiện tượng như gradient biến mất/bùng nổ và thúc đẩy các kiến trúc như LSTM quản lý tường minh luồng thông tin qua cơ chế cổng.

Sự tiến hóa từ RNN đến LSTM đến Transformer kể một câu chuyện về việc giải quyết các hạn chế cơ bản. RNN vanilla gặp khó với phụ thuộc tầm xa do gradient biến mất. LSTM giới thiệu cơ chế cổng tạo kết nối bỏ qua qua thời gian, cho phép gradient chảy dễ dàng hơn và thông tin tồn tại lâu hơn. Nhưng LSTM vẫn xử lý chuỗi tuần tự, hạn chế song song hóa. Transformer từ bỏ hoàn toàn sự hồi quy, dùng attention để tạo kết nối trực tiếp giữa tất cả các bước thời gian, cho phép song song hóa hoàn toàn với chi phí độ phức tạp bậc hai theo độ dài chuỗi. Mỗi kiến trúc tạo các đánh đổi khác nhau giữa biểu diễn, khả năng huấn luyện, và hiệu suất tính toán.

Mối quan hệ giữa RNN và mạng tích chập tinh tế hơn nhưng cũng chiếu sáng. Tích chập thời gian — áp dụng tích chập 1D qua các chuỗi — có thể nắm bắt một số mẫu tuần tự và hoàn toàn có thể song song hóa. Tuy nhiên, trường tiếp nhận của nó chỉ tăng tuyến tính theo chiều sâu (một mạng có $$L$$ lớp kích thước kernel $$k$$ có trường tiếp nhận $$1 + L(k-1)$$), trong khi RNN về lý thuyết có trường tiếp nhận vô hạn (trạng thái ẩn có thể nhớ thông tin từ bất kỳ đâu trong quá khứ). Sự đánh đổi này giữa khả năng song song hóa (ưu tiên tích chập) và bộ nhớ lý thuyết không giới hạn (ưu tiên RNN) đã dẫn đến các kiến trúc lai kết hợp cả hai, như WaveNet để tạo âm thanh.

RNN hai chiều mở rộng kiến trúc cơ bản bằng cách xử lý chuỗi theo cả hướng tiến và lùi, duy trì hai trạng thái ẩn $$\overrightarrow{\mathbf{h}}_t$$ và $$\overleftarrow{\mathbf{h}}_t$$. Đầu ra tại mỗi bước thời gian kết hợp thông tin từ cả hai: $$\mathbf{y}_t = g(\mathbf{W}_{hy}[\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] + \mathbf{b}_y)$$. Điều này mạnh mẽ cho các tác vụ nơi ngữ cảnh tương lai có sẵn (như dịch một câu hoàn chỉnh) nhưng không thể cho dự đoán thời gian thực nơi ta phải đưa ra quyết định trước khi thấy chuỗi hoàn chỉnh. Thiết kế hai chiều minh họa cách kiến trúc nên khớp yêu cầu tác vụ — dùng ngữ cảnh tương lai khi có sẵn, xử lý nhân quả khi cần thiết.

## 6. Các Bài báo Cơ bản

**["Finding Structure in Time" (1990)](https://doi.org/10.1207/s15516709cog1402_1)**  
*Tác giả*: Jeffrey L. Elman  
Bài báo tiên phong này giới thiệu Simple Recurrent Network (SRN), nay gọi là mạng Elman, và chứng minh rằng kết nối hồi quy cho phép học các mẫu thời gian. Elman chỉ ra rằng RNN có thể học dự đoán từ tiếp theo trong các câu đơn giản, khám phá cấu trúc ngữ pháp mà không có quy tắc tường minh. Điểm hiểu biết quan trọng là trạng thái ẩn phát triển các biểu diễn nội bộ của các loại ngữ pháp (danh từ, động từ) và các phụ thuộc tuần tự mà không được bảo làm như vậy — hoàn toàn từ tác vụ dự đoán. Bài báo thiết lập RNN như một phương pháp khả thi cho mô hình hóa chuỗi và ảnh hưởng đến sự phát triển sau này của các kiến trúc hồi quy tinh vi hơn. Phân tích của Elman về động lực trạng thái ẩn — cho thấy không gian trạng thái tự tổ chức để phản ánh cấu trúc ngôn ngữ — chứng minh rằng mạng neuron có thể khám phá các biểu diễn có thể diễn giải, một chủ đề tiếp tục trong nghiên cứu học sâu hiện đại.

**["Learning to Forget: Continual Prediction with LSTM" (2000)](https://doi.org/10.1162/089976600300015015)**  
*Tác giả*: Felix A. Gers, Jürgen Schmidhuber, Fred Cummins  
Mặc dù LSTM được giới thiệu năm 1997, bài báo này đã thực hiện một sửa đổi quan trọng khiến chúng thực tế: cổng forget (*forget gate*). LSTM ban đầu có thể tích lũy thông tin trong trạng thái tế bào (*cell state*) nhưng không có cơ chế để chọn lọc quên thông tin không liên quan, dẫn đến bão hòa qua các chuỗi dài. Cổng forget, được điều khiển bởi $$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f)$$, cho phép mạng xóa bộ nhớ khi thông tin cũ trở nên không liên quan. Sự bổ sung tưởng chừng đơn giản này — để mạng học khi nào nên quên — cải thiện đáng kể hiệu năng LSTM trên chuỗi dài và trở thành chuẩn trong mọi triển khai LSTM sau này. Bài báo minh họa cách các chi tiết kiến trúc tưởng nhỏ có thể có tác động thực tiễn sâu sắc.

**["On the difficulty of training Recurrent Neural Networks" (2013)](https://arxiv.org/abs/1211.5063)**  
*Tác giả*: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio  
Bài báo này cung cấp phân tích định nghĩa về gradient biến mất và bùng nổ trong RNN, vượt qua các quan sát thực nghiệm để điều trị toán học nghiêm ngặt. Các tác giả chỉ ra rằng khi backpropagate qua $$t$$ bước thời gian, gradient liên quan đến tích của $$t$$ ma trận Jacobian, và nếu trị riêng lớn nhất của các ma trận này nhỏ hơn 1, gradient biến mất theo cấp số nhân; nếu lớn hơn 1, chúng bùng nổ theo cấp số nhân. Quan trọng hơn, họ chỉ ra đây không chỉ là vấn đề thủ thuật huấn luyện mà là tính chất cơ bản của động lực hồi quy. Bài báo đề xuất gradient clipping để xử lý bùng nổ (clip chuẩn gradient về ngưỡng tối đa, nay là thực hành chuẩn) và phân tích cách cơ chế cổng của LSTM tạo đường dẫn hiệu quả cho luồng gradient. Công trình này đào sâu hiểu biết vì sao RNN vanilla thất bại trên chuỗi dài và vì sao các đổi mới kiến trúc như LSTM là cần thiết, không phải tùy chọn.

**["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)](https://arxiv.org/abs/1412.3555)**  
*Tác giả*: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio  
Bài báo này so sánh có hệ thống kiến trúc LSTM và GRU (Gated Recurrent Unit) trên nhiều tác vụ mô hình tuần tự, cung cấp bằng chứng thực nghiệm về khi nào mỗi kiến trúc vượt trội. GRU, do Cho và cộng sự giới thiệu năm 2014, đơn giản hóa LSTM bằng cách chỉ dùng hai cổng thay vì ba và không có trạng thái tế bào riêng, giảm khoảng 25% tham số. Bài báo chỉ ra rằng GRU thường khớp hiệu năng LSTM trong khi huấn luyện nhanh hơn nhờ ít tham số hơn. Quan trọng, nó chứng minh rằng chi tiết kiến trúc quan trọng — các cơ chế hồi quy được thiết kế cẩn thận luôn vượt trội RNN vanilla trên chuỗi dài. Phương pháp thực nghiệm của bài báo — so sánh có kiểm soát trên nhiều dataset với tinh chỉnh siêu tham số cẩn thận — đặt chuẩn cho cách đánh giá đổi mới kiến trúc trong học sâu.

**["Visualizing and Understanding Recurrent Networks" (2015)](https://arxiv.org/abs/1506.02078)**  
*Tác giả*: Andrej Karpathy, Justin Johnson, Li Fei-Fei  
Bài báo này điều tra những gì RNN học bằng cách phân tích động lực trạng thái ẩn của chúng trên mô hình ngôn ngữ cấp ký tự. Bằng cách xem xét đơn vị ẩn nào kích hoạt với mẫu đầu vào nào, các tác giả phát hiện rằng RNN tự phát triển các biểu diễn nội bộ có thể diễn giải: một số neuron kích hoạt cho dấu ngoặc kép, số khác cho cân bằng ngoặc đơn, số khác cho thụt lề mã. Điều này chứng minh rằng RNN không chỉ ghi nhớ mà học cấu trúc có ý nghĩa. Bài báo cũng giới thiệu kỹ thuật trực quan hóa các mẫu giống attention trong RNN trước khi cơ chế attention tường minh trở nên phổ biến. Có lẽ ảnh hưởng nhất, nó làm cho loại phân tích diễn giải khả dụng hơn — giúp ta hiểu mạng neuron học gì — một phương pháp nay đã trở thành chuẩn cho mọi loại mô hình, không chỉ RNN.

## Các Bẫy phổ biến và Thủ thuật

Chế độ thất bại phổ biến nhất khi huấn luyện RNN là gradient bùng nổ, và nhận ra các triệu chứng là then chốt để debug. Loss huấn luyện đột ngột trở thành NaN, tham số trở thành vô cực, hoặc loss dao động dữ dội thay vì giảm mượt. Điều này xảy ra khi tích gradient qua các bước thời gian tăng theo cấp số nhân. Giải pháp chuẩn — gradient clipping — về khái niệm đơn giản nhưng phải triển khai đúng. Ta tính chuẩn gradient toàn cục trên mọi tham số $$\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_\theta (\frac{\partial \mathcal{L}}{\partial \theta})^2}$$ và nếu vượt ngưỡng (thường 5–10), ta scale tất cả gradient theo $$\frac{\text{threshold}}{\|\nabla_\theta \mathcal{L}\|_2}$$. Điều này bảo toàn hướng gradient trong khi ngăn cập nhật bùng nổ. Quan trọng là clip chuẩn toàn cục, không phải từng giá trị gradient riêng lẻ, vì ta muốn bảo toàn độ lớn tương đối của gradient cho các tham số khác nhau.

Gradient biến mất thâm hiểm hơn vì chúng không gây lỗi huấn luyện rõ ràng — mạng huấn luyện nhưng đơn giản thất bại trong việc học phụ thuộc tầm xa. Các triệu chứng bao gồm mô hình chỉ dùng ngữ cảnh gần đây (trong mô hình ngôn ngữ, chỉ xem xét vài từ cuối) hoặc không thể học các tác vụ đòi hỏi thông tin từ đầu chuỗi dài. Phát hiện đòi hỏi phân tích cẩn thận: vẽ độ lớn gradient theo số bước backpropagation hoặc kiểm thử cụ thể trên các tác vụ đòi hỏi bộ nhớ tầm xa. Giải pháp bao gồm chuyển sang LSTM/GRU (giảm bớt dù không loại bỏ hoàn toàn gradient biến mất), dùng độ dài chuỗi nhỏ hơn trong huấn luyện (BPTT cắt ngắn), hoặc thêm auxiliary loss ở các bước trung gian để cung cấp đường dẫn gradient trực tiếp hơn.

Khởi tạo trọng số hồi quy đáng được chú ý đặc biệt vì nó ảnh hưởng trực tiếp đến ổn định luồng gradient. Khởi tạo ngẫu nhiên nhỏ chuẩn $$\mathbf{W}_{hh} \sim \mathcal{N}(0, 0.01^2)$$ thường dẫn đến gradient biến mất. Cách tiếp cận tốt hơn là khởi tạo trực giao: khởi tạo $$\mathbf{W}_{hh}$$ thành ma trận trực giao ngẫu nhiên (thường sinh qua phân rã QR của ma trận ngẫu nhiên). Ma trận trực giao bảo toàn chuẩn vectơ khi nhân, giúp gradient không biến mất cũng không bùng nổ, ít nhất ban đầu. Điều này cho huấn luyện điểm xuất phát tốt hơn, dù khi trọng số cập nhật chúng lệch khỏi tính trực giao. Cách khác là khởi tạo đơn vị cộng nhiễu nhỏ: $$\mathbf{W}_{hh} = I + \mathcal{N}(0, 0.001^2)$$, khuyến khích trạng thái ẩn thay đổi chậm, có thể giúp luồng gradient.

Một vấn đề tinh tế nhưng quan trọng là chuỗi độ dài biến đổi trong huấn luyện theo batch. Khi huấn luyện trên nhiều chuỗi có độ dài khác nhau đồng thời, ta phải xử lý thực tế rằng một số chuỗi kết thúc trước các chuỗi khác. Giải pháp là padding và masking: pad các chuỗi ngắn hơn để khớp chuỗi dài nhất trong batch bằng token đệm đặc biệt, rồi mask loss để các vị trí pad không đóng góp vào gradient. Không có masking, RNN nhận tín hiệu gradient vô nghĩa từ padding, làm giảm hiệu năng. Chức năng PackedSequence của PyTorch xử lý điều này thanh lịch, tránh tính toán hoàn toàn trên các vị trí pad.

Lựa chọn chiều trạng thái ẩn liên quan đến các đánh đổi quan trọng. Chiều ẩn lớn hơn cung cấp dung lượng hơn để nhớ các mẫu phức tạp và ngữ cảnh dài hơn. Tuy nhiên, chúng tăng tham số theo bậc hai ($$\mathbf{W}_{hh}$$ có $$d_h^2$$ phần tử), làm chậm tính toán (mỗi bước thời gian cần $$O(d_h^2)$$ phép toán), và có thể dẫn đến overfitting trên dataset nhỏ. Điểm xuất phát phổ biến là khớp chiều ẩn với chiều đầu vào hoặc dùng 128–512 tùy độ phức tạp tác vụ. Với mô hình cấp ký tự, 128–256 thường đủ. Với mô hình ngôn ngữ cấp từ trên từ vựng lớn, 512–1024 là điển hình. Luôn validate trên tập held-out và theo dõi khoảng cách train–test báo hiệu overfitting.

Dùng teacher forcing trong huấn luyện nhưng sinh tự hồi quy khi suy luận tạo sự lệch train–test trong các mô hình chuỗi-sang-chuỗi. Trong huấn luyện với teacher forcing, decoder nhận token đúng trước đó làm đầu vào, đảm bảo nó thấy đầu vào tốt ngay cả khi dự đoán kém. Khi suy luận, nó phải dùng chính dự đoán của mình, có thể sai, dẫn đến sai số cộng dồn. Sự lệch này nghĩa là mô hình không bao giờ học phục hồi từ lỗi của chính mình trong huấn luyện. Giải pháp bao gồm scheduled sampling (ngẫu nhiên dùng token dự đoán thay vì token đúng trong huấn luyện với xác suất tăng dần), hoặc dùng auxiliary loss khuyến khích tính robust trước nhiễu đầu vào.

## Điểm Chính

Mạng Neuron Hồi quy giới thiệu ý tưởng cơ bản về bộ nhớ trong mạng neuron thông qua các trạng thái ẩn tồn tại qua các bước thời gian, cho phép mô hình dữ liệu tuần tự nơi thứ tự và ngữ cảnh quan trọng. Sự thanh lịch toán học của chia sẻ tham số qua thời gian — dùng cùng trọng số ở mỗi bước — cho phép RNN tổng quát hóa qua độ dài chuỗi trong khi học các mẫu thời gian. Tuy nhiên, chính sự hồi quy này tạo ra các thách thức: xử lý tuần tự ngăn cản song song hóa, làm chậm huấn luyện trên GPU; tích của Jacobian qua thời gian dẫn đến gradient biến mất hoặc bùng nổ, giới hạn khả năng học phụ thuộc tầm xa; và trạng thái ẩn kích thước cố định tạo ra nghẽn thông tin cho các chuỗi dài. Mặc dù có những hạn chế này, RNN thiết lập các nguyên tắc — rằng mạng có thể duy trì trạng thái, rằng cấu trúc tuần tự nên được mô hình tường minh, rằng ta có thể học dự đoán tương lai từ quá khứ — ảnh hưởng đến mọi kiến trúc mô hình tuần tự tiếp theo. Hiểu RNN sâu sắc nghĩa là hiểu không chỉ chúng hoạt động thế nào mà vì sao chúng được thiết kế như vậy, chúng thất bại ở đâu, và các đổi mới sau này như LSTM và Transformer giải quyết hạn chế của chúng ra sao trong khi xây trên các hiểu biết của chúng.

Hành trình từ mạng feedforward đến RNN đại diện cho một bước nhảy khái niệm quan trọng trong học sâu: từ xử lý đầu vào tĩnh độc lập đến mô hình các quá trình động với bộ nhớ và cấu trúc thời gian. Bước nhảy này mở ra vô số ứng dụng mới nhưng đưa vào những thách thức mới đã thúc đẩy hàng thập kỷ nghiên cứu và tiếp tục truyền cảm hứng cho đổi mới trong kiến trúc mô hình chuỗi ngày nay.
