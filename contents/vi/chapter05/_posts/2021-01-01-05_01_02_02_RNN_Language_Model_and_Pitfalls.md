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

Đoạn mã dưới đây minh họa hiện tượng gradient biến mất bằng thực nghiệm:

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
        
        # Tính loss từ đầu ra bước thời gian CUỐI CÙNG
        # Gradient phải backprop qua seq_len-1 bước để tới đầu vào đầu tiên x[:, 0, :]
        loss = out[:, -1, :].sum()
        
        # Backward pass
        loss.backward()
        
        # Đo độ lớn gradient tại bước thời gian đầu tiên
        grad_magnitude = x.grad[:, 0, :].abs().mean().item()
        
        results.append((seq_len, grad_magnitude))
        print(f"Sequence length {seq_len:2d}: Gradient magnitude = {grad_magnitude:.6f}")
    
    # Thường thấy suy giảm cấp số nhân của độ lớn gradient
    print("\nQuan sát: Gradient suy giảm cấp số nhân theo độ dài chuỗi!")
    print("Đây là vấn đề gradient biến mất giới hạn RNN vanilla.")
    
    return results

gradient_analysis = analyze_gradient_flow()
```

## 5. Các khái niệm liên quan

So sánh RNN với mạng truyền thẳng (feedforward) làm sáng tỏ các giả định nền tảng trong thiết kế kiến trúc mạng. Mạng feedforward coi các đầu vào là những mẫu độc lập và có phân phối đồng nhất — thứ tự đưa các hình ảnh vào khi huấn luyện không ảnh hưởng vì mỗi ảnh được xử lý tách biệt. Ngược lại, RNN mô hình hóa tường minh sự phụ thuộc giữa các phần tử liên tiếp thông qua trạng thái ẩn. Việc chọn RNN thay vì mạng feedforward thể hiện một thiên kiến quy nạp (inductive bias) rằng thứ tự thời gian hoặc thứ tự chuỗi mang thông tin quan trọng cho bài toán.

Góc nhìn từ máy trạng thái hữu hạn và hệ thống động lực mang lại hiểu biết sâu hơn về mặt lý thuyết. Một RNN với trạng thái ẩn rời rạc và hàm kích hoạt ngưỡng cứng hoạt động như một máy trạng thái hữu hạn, chuyển đổi giữa các trạng thái dựa trên đầu vào. Với trạng thái ẩn liên tục và hàm kích hoạt trơn, RNN trở thành một hệ thống động lực rời rạc theo thời gian được mô tả bởi phương trình sai phân $$\mathbf{h}_{t+1} = f(\mathbf{W}_{hh}\mathbf{h}_t + \mathbf{W}_{xh}\mathbf{x}_t)$$. Tính ổn định và năng lực biểu diễn của hệ thống này phụ thuộc vào phổ giá trị riêng của $$\mathbf{W}_{hh}$$, qua đó quyết định hệ thống ổn định, hỗn loạn hay ở biên ổn định. Liên hệ này giải thích bản chất toán học của hiện tượng gradient biến mất/bùng nổ và lý do các kiến trúc như LSTM cần kiểm soát trực tiếp luồng thông tin bằng cơ chế cổng.

Quá trình phát triển từ RNN đến LSTM rồi Transformer phản ánh nỗ lực giải quyết lần lượt từng giới hạn cốt lõi. RNN cơ bản khó học phụ thuộc tầm xa do gradient biến mất. LSTM đưa vào cơ chế cổng tạo đường truyền trực tiếp theo thời gian, giúp gradient lan truyền tốt hơn và bộ nhớ duy trì lâu hơn, nhưng vẫn phải xử lý tuần tự nên khó song song hóa. Transformer loại bỏ hoàn toàn hồi quy, dùng attention để kết nối trực tiếp mọi cặp vị trí trong chuỗi, cho phép song song hóa hoàn toàn dù phải đổi lấy độ phức tạp bậc hai theo độ dài chuỗi. Mỗi kiến trúc đại diện cho một sự đánh đổi khác nhau giữa khả năng biểu diễn, độ dễ huấn luyện và hiệu suất tính toán.

So sánh RNN với mạng tích chập (CNN) cũng làm rõ sự đánh đổi này. Tích chập 1D theo thời gian nắm bắt được nhiều mẫu tuần tự và hỗ trợ song song hóa hoàn toàn, nhưng vùng tiếp nhận (receptive field) chỉ mở rộng tuyến tính theo chiều sâu: một mạng gồm $$L$$ lớp với kích thước bộ lọc $$k$$ có vùng tiếp nhận bằng $$1 + L(k-1)$$. Trong khi đó, RNN về lý thuyết có vùng tiếp nhận không giới hạn vì trạng thái ẩn có thể lưu giữ thông tin từ bất kỳ thời điểm nào trước đó. Sự đánh đổi giữa khả năng song song hóa của tích chập và bộ nhớ lý thuyết vô hạn của RNN đã dẫn tới các kiến trúc kết hợp như WaveNet trong bài toán tổng hợp âm thanh.

RNN hai chiều (Bidirectional RNN) mở rộng mô hình gốc bằng cách xử lý chuỗi theo cả chiều xuôi và chiều ngược, duy trì hai trạng thái ẩn $$\overrightarrow{\mathbf{h}}_t$$ và $$\overleftarrow{\mathbf{h}}_t$$. Đầu ra tại mỗi bước kết hợp cả hai chiều: $$\mathbf{y}_t = g(\mathbf{W}_{hy}[\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] + \mathbf{b}_y)$$. Cách thiết kế này hiệu quả với các tác vụ có sẵn toàn bộ chuỗi đầu vào (như dịch một câu hoàn chỉnh), nhưng không áp dụng được cho dự đoán thời gian thực khi mô hình phải ra quyết định trước khi thấy phần tương lai của chuỗi.

## 6. Các bài báo nền tảng

**["Finding Structure in Time" (1990)](https://doi.org/10.1207/s15516709cog1402_1)**  
*Tác giả*: Jeffrey L. Elman  
Bài báo này giới thiệu Simple Recurrent Network (SRN), còn gọi là mạng Elman, và chứng minh kết nối hồi quy có thể học các mẫu theo thời gian. Elman cho thấy RNN có thể dự đoán từ tiếp theo trong các câu đơn giản và tự khám phá cấu trúc ngữ pháp mà không cần luật định nghĩa sẵn. Trạng thái ẩn tự hình thành biểu diễn nội bộ cho các từ loại (danh từ, động từ) và quan hệ tuần tự chỉ từ nhiệm vụ dự đoán. Công trình này đặt nền móng cho việc ứng dụng RNN trong mô hình hóa chuỗi, đồng thời cho thấy không gian trạng thái ẩn của mạng neuron có thể tự tổ chức thành những biểu diễn có thể diễn giải được.

**["Learning to Forget: Continual Prediction with LSTM" (2000)](https://doi.org/10.1162/089976600300015015)**  
*Tác giả*: Felix A. Gers, Jürgen Schmidhuber, Fred Cummins  
Dù LSTM ra đời năm 1997, bài báo này bổ sung một thành phần then chốt giúp kiến trúc này vận hành hiệu quả trong thực tế: cổng quên (*forget gate*). Phiên bản LSTM ban đầu chỉ tích lũy thông tin trong trạng thái tế bào (*cell state*) mà thiếu cơ chế loại bỏ thông tin cũ không còn phù hợp, dễ dẫn đến bão hòa trên chuỗi dài. Cổng quên $$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f)$$ cho phép mạng chủ động xóa bộ nhớ khi cần. Bổ sung nhỏ này cải thiện rõ rệt hiệu năng của LSTM trên chuỗi dài và trở thành thành phần tiêu chuẩn trong mọi cài đặt LSTM về sau.

**["On the difficulty of training Recurrent Neural Networks" (2013)](https://arxiv.org/abs/1211.5063)**  
*Tác giả*: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio  
Bài báo cung cấp phân tích toán học chặt chẽ về hiện tượng gradient biến mất và bùng nổ trong RNN, vượt xa các quan sát thực nghiệm trước đó. Các tác giả chỉ ra rằng khi lan truyền ngược qua $$t$$ bước thời gian, gradient phụ thuộc vào tích của $$t$$ ma trận Jacobian; nếu trị riêng lớn nhất nhỏ hơn 1 thì gradient suy giảm theo hàm mũ, còn nếu lớn hơn 1 thì bùng nổ theo hàm mũ. Họ khẳng định đây là đặc tính nội tại của động lực hồi quy chứ không chỉ là vấn đề kỹ thuật huấn luyện. Công trình này đề xuất kỹ thuật cắt ngưỡng gradient (*gradient clipping*) để kiểm soát hiện tượng bùng nổ và lý giải cách cơ chế cổng trong LSTM cải thiện luồng gradient.

**["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)](https://arxiv.org/abs/1412.3555)**  
*Tác giả*: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio  
Bài báo so sánh có hệ thống giữa LSTM và GRU (Gated Recurrent Unit) trên nhiều tác vụ mô hình hóa chuỗi nhằm đánh giá ưu thế của từng kiến trúc. GRU, do Cho và cộng sự đề xuất năm 2014, tinh giản LSTM xuống còn hai cổng và bỏ trạng thái tế bào riêng biệt, giảm khoảng 25% số lượng tham số. Kết quả thực nghiệm cho thấy GRU thường đạt hiệu năng tương đương LSTM nhưng huấn luyện nhanh hơn nhờ ít tham số hơn, đồng thời khẳng định các kiến trúc có cơ chế cổng luôn vượt trội RNN cơ bản trên chuỗi dài.

**["Visualizing and Understanding Recurrent Networks" (2015)](https://arxiv.org/abs/1506.02078)**  
*Tác giả*: Andrej Karpathy, Justin Johnson, Li Fei-Fei  
Bài báo khảo sát cách hoạt động bên trong của RNN thông qua phân tích trạng thái ẩn trên mô hình ngôn ngữ cấp ký tự. Khi quan sát các đơn vị ẩn kích hoạt theo mẫu đầu vào, nhóm tác giả phát hiện một số neuron đảm nhận vai trò rất cụ thể như theo dõi dấu ngoặc kép, kiểm tra cân bằng dấu ngoặc đơn hay mức thụt lề mã nguồn. Điều này chứng minh RNN thực sự học được cấu trúc có nghĩa thay vì chỉ ghi nhớ máy móc. Công trình cũng giới thiệu các kỹ thuật trực quan hóa góp phần thúc đẩy hướng nghiên cứu về khả năng diễn giải mô hình trong học sâu.

## Các bẫy phổ biến và thủ thuật thực hành

Lỗi thường gặp nhất khi huấn luyện RNN là hiện tượng gradient bùng nổ. Dấu hiệu dễ nhận biết là giá trị mất mát đột ngột thành `NaN`, tham số tiến tới vô cực hoặc hàm mất mát dao động mạnh thay vì giảm dần. Nguyên nhân đến từ việc tích các gradient qua nhiều bước thời gian tăng theo hàm mũ. Giải pháp tiêu chuẩn là cắt ngưỡng gradient (*gradient clipping*): tính chuẩn gradient toàn cục trên toàn bộ tham số $$\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_\theta (\frac{\partial \mathcal{L}}{\partial \theta})^2}$$, và nếu vượt ngưỡng (thường chọn từ 5 đến 10), nhân toàn bộ gradient với hệ số $$\frac{\text{threshold}}{\|\nabla_\theta \mathcal{L}\|_2}$$. Cách làm này giữ nguyên hướng của vectơ gradient đồng thời ngăn bước cập nhật quá lớn. Cần áp dụng cắt theo chuẩn toàn cục thay vì cắt riêng từng phần tử để bảo toàn tỷ lệ tương đối giữa gradient của các tham số.

Hiện tượng gradient biến mất khó phát hiện hơn vì quá trình huấn luyện vẫn diễn ra bình thường mà không báo lỗi, nhưng mô hình không học được các phụ thuộc tầm xa. Triệu chứng thường thấy là mô hình chỉ dựa vào ngữ cảnh rất gần (chẳng hạn chỉ xét vài từ cuối trong mô hình ngôn ngữ) hoặc thất bại ở các bài toán đòi hỏi nhớ thông tin từ đầu chuỗi dài. Để chẩn đoán, cần vẽ đồ thị độ lớn gradient theo số bước lan truyền ngược hoặc kiểm thử trên tác vụ bộ nhớ dài hạn. Các biện pháp khắc phục gồm chuyển sang LSTM/GRU, rút ngắn độ dài chuỗi lan truyền ngược (Truncated BPTT), hoặc bổ sung hàm mất mát phụ ở các bước trung gian để rút ngắn đường truyền gradient.

Khởi tạo trọng số hồi quy ảnh hưởng trực tiếp tới sự ổn định của luồng gradient. Khởi tạo ngẫu nhiên nhỏ kiểu $$\mathbf{W}_{hh} \sim \mathcal{N}(0, 0.01^2)$$ thường gây gradient biến mất ngay từ đầu. Giải pháp tốt hơn là khởi tạo trực giao: gán $$\mathbf{W}_{hh}$$ bằng một ma trận trực giao ngẫu nhiên (thường thu được qua phân rã QR). Ma trận trực giao bảo toàn chuẩn vectơ qua phép nhân, giúp gradient không bị co nhỏ hay khuếch đại quá mức trong giai đoạn đầu huấn luyện (dù tính trực giao sẽ dần mất đi khi trọng số được cập nhật). Một lựa chọn khác là khởi tạo gần ma trận đơn vị: $$\mathbf{W}_{hh} = I + \mathcal{N}(0, 0.001^2)$$, giúp trạng thái ẩn thay đổi chậm và hỗ trợ luồng gradient.

Khi huấn luyện theo batch với các chuỗi có độ dài khác nhau, một số chuỗi sẽ kết thúc sớm hơn chuỗi dài nhất trong batch. Cách xử lý chuẩn là đệm (*padding*) các chuỗi ngắn bằng token đặc biệt để đồng nhất độ dài, sau đó che (*masking*) các vị trí đệm khi tính hàm mất mát để tránh sinh gradient nhiễu. Trong PyTorch, `PackedSequence` giúp xử lý gọn vấn đề này bằng cách bỏ qua hoàn toàn phép tính trên các phần tử đệm.

Việc chọn số chiều trạng thái ẩn $$d_h$$ đòi hỏi cân nhắc kỹ lưỡng. Số chiều lớn tăng khả năng ghi nhớ các mẫu phức tạp và ngữ cảnh dài, nhưng làm số tham số của $$\mathbf{W}_{hh}$$ tăng theo bình phương ($$d_h^2$$), kéo theo chi phí tính toán $$O(d_h^2)$$ ở mỗi bước và tăng nguy cơ quá khớp trên tập dữ liệu nhỏ. Mức khởi điểm phổ biến là 128–256 cho mô hình cấp ký tự, hoặc 512–1024 cho mô hình ngôn ngữ cấp từ với bộ từ vựng lớn. Luôn cần kiểm chứng trên tập validation để phát hiện sớm hiện tượng quá khớp.

Trong các mô hình chuỗi-sang-chuỗi, việc dùng *teacher forcing* lúc huấn luyện nhưng sinh tự hồi quy lúc suy luận tạo ra độ lệch giữa hai giai đoạn. Khi huấn luyện với *teacher forcing*, decoder luôn nhận token đúng của bước trước nên ít gặp tình huống đầu vào sai; nhưng khi suy luận, nó phải dùng chính dự đoán của mình, khiến sai số dễ tích lũy dần. Để giảm độ lệch này, có thể áp dụng *scheduled sampling* (thay dần token đúng bằng token dự đoán trong quá trình huấn luyện với xác suất tăng dần) hoặc dùng hàm mất mát phụ nhằm tăng khả năng chống nhiễu đầu vào.

## Điểm chính

Mạng Neuron Hồi quy đưa khái niệm bộ nhớ vào mạng neuron thông qua trạng thái ẩn được duy trì theo thời gian, giúp mô hình hóa hiệu quả các dữ liệu tuần tự mà thứ tự và ngữ cảnh đóng vai trò quyết định. Cơ chế chia sẻ tham số qua các bước thời gian cho phép RNN tổng quát hóa sang nhiều độ dài chuỗi khác nhau. Tuy nhiên, cấu trúc hồi quy cũng kéo theo ba hạn chế lớn: xử lý tuần tự khó song song hóa trên GPU, tích các ma trận Jacobian gây gradient biến mất hoặc bùng nổ khiến khó học phụ thuộc tầm xa, và trạng thái ẩn kích thước cố định tạo nút thắt thông tin trên chuỗi dài. Dù vậy, các nguyên lý mà RNN thiết lập — duy trì trạng thái theo thời gian và mô hình hóa tường minh cấu trúc chuỗi — vẫn là nền tảng cho các kiến trúc kế tiếp như LSTM và Transformer.

Bước chuyển từ mạng feedforward sang RNN đánh dấu sự mở rộng quan trọng của học sâu: từ xử lý các đầu vào tĩnh độc lập sang mô hình hóa các quá trình động có bộ nhớ và cấu trúc thời gian, mở đường cho nhiều hướng nghiên cứu kiến trúc mô hình chuỗi hiện đại.

## Video tham khảo

- **Xem trước:** [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy (tiếng Anh). Hướng dẫn chi tiết mô hình ngôn ngữ cấp ký tự và hàm mất mát huấn luyện.
- [Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=L8HKweZIOmg) — StatQuest with Josh Starmer (tiếng Anh). Giải thích mô hình encoder-decoder chuỗi-sang-chuỗi, kiến trúc liên quan trực tiếp đến phần khái niệm và bẫy thường gặp trong bài.
- [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) — DeepLearningAI (tiếng Anh). Phân tích hiện tượng gradient biến mất và bùng nổ qua tích nhiều bước trong lan truyền ngược theo thời gian.
