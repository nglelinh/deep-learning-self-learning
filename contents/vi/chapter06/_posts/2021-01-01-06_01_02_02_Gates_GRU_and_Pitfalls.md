---
layout: post
title: 06-01-02-02 Phân tích Cổng, GRU và Bẫy thường gặp
chapter: '06'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter06
---

Làm sao biết chiếc LSTM bạn vừa huấn luyện thực sự "học được cách nhớ và quên" chứ không phải chỉ đang ghi nhớ vẹt? Cách trực quan nhất là **mở nắp ca-pô** và ghi lại giá trị mở của từng cổng $$f_t, i_t, o_t$$ qua từng bước thời gian — đồng thời đặt LSTM lên bàn cân trực tiếp với đối thủ gọn nhẹ hơn là **GRU**.

## Kiến thức cần nhớ: Bốn chủ đề phân tích sâu về Cổng, GRU và Cạm bẫy

1. **Giải phẫu hành vi cổng trên chuỗi nhiễu**:
   Trên chuỗi `[5, nhiễu, nhiễu, ..., 3]`, một LSTM tốt sẽ mở $$i_1 \approx 1$$ khi thấy số `5`, sau đó giữ $$f_t \approx 1, i_t \approx 0$$ suốt đoạn nhiễu để khóa kín bộ nhớ, và chỉ mở lại $$i_T, o_T$$ ở bước cuối.
2. **So sánh thực chiến LSTM vs. GRU**:
   GRU có ít hơn 25% tham số, huấn luyện nhanh hơn một chút và ít overfitting hơn trên tập nhỏ; LSTM nhỉnh hơn trên các bài toán đếm hoặc phụ thuộc cực dài nhờ tách biệt $$c_t$$ khỏi $$h_t$$.
3. **Bốn cạm bẫy kinh điển khi huấn luyện LSTM/GRU**:
   Quên áp dụng Dropout **chỉ giữa các tầng** (không tùy tiện đặt dropout chuẩn lên kết nối hồi tiếp làm đứt trí nhớ), quên khởi tạo $$b_f > 0$$, bỏ quên gradient clipping, và dùng chuỗi quá dài mà không dùng Attention.
4. **Định vị LSTM trong bức tranh tiến hóa**:
   Biết rõ giới hạn của LSTM (tuần tự hóa $$O(T)$$ không song song được trên GPU) để hiểu chính xác tại sao Chương 07 (Attention) và Chương 08 (Transformer) ra đời.

### Giải thích: Tính tay cơ chế nội suy lồi của GRU so với LSTM khi gặp nhiễu

Xét một đơn vị GRU 1 chiều với trạng thái cũ $$h_{t-1} = 5{,}0$$ và ứng viên mới chứa tín hiệu nhiễu $$\tilde{h}_t = -1{,}0$$. Phương trình cập nhật của GRU là:
$$h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$$

| Giá trị cổng cập nhật $$z_t$$ | Tỷ lệ giữ cũ $$(1 - z_t)$$ | Tỷ lệ nhận mới $$z_t$$ | Trạng thái mới $$h_t$$ | Ý nghĩa |
| :---: | :---: | :---: | :---: | :--- |
| **$$z_t = 0{,}05$$** (đóng cửa trước nhiễu) | $$0{,}95$$ | $$0{,}05$$ | $$0{,}95(5{,}0) + 0{,}05(-1{,}0) = \mathbf{4{,}70}$$ | Bỏ qua nhiễu, bảo toàn gần nguyên vẹn số 5 |
| **$$z_t = 0{,}50$$** (trộn đều) | $$0{,}50$$ | $$0{,}50$$ | $$0{,}50(5{,}0) + 0{,}50(-1{,}0) = \mathbf{2{,}00}$$ | Hòa trộn cũ và mới |
| **$$z_t = 0{,}95$$** (ghi đè hoàn toàn) | $$0{,}05$$ | $$0{,}95$$ | $$0{,}05(5{,}0) + 0{,}95(-1{,}0) = \mathbf{-0{,}70}$$ | Xóa ký ức cũ, nhận thông tin mới |

Khác với LSTM (cho phép cả $$f_t$$ và $$i_t$$ cùng bằng 1 để *cộng dồn* không giới hạn), GRU ràng buộc tổng trọng số cũ + mới luôn bằng đúng 1 — gọn nhẹ hơn nhưng cũng làm GRU khó học bài toán "đếm không giới hạn" bằng LSTM.

```python
# Phân tích hành vi cổng
print("\n" + "="*70)
print("Phân tích Kích hoạt Cổng LSTM")
print("="*70)

# Tạo LSTM thủ công đơn giản để theo dõi cổng
lstm_analyze = LSTM(input_size=1, hidden_size=4, output_size=1)

# Tạo chuỗi: [5, noise, noise, ..., 3]
test_sequence = [np.array([[5.0]])]
test_sequence.extend([np.array([[np.random.rand()*10]]) for _ in range(10)])
test_sequence.append(np.array([[3.0]]))

# Forward pass theo dõi tất cả cổng
output, hiddens, cells, all_gates = lstm_analyze.forward(test_sequence, 
                                                         return_sequences=True)

print("Kích hoạt cổng qua thời gian (trung bình qua các chiều ẩn):\n")
print("Time | Forget | Input | Output | Cell State (avg)")
print("-" * 60)

for t, gates in enumerate(all_gates):
    f_avg = np.mean(gates['forget'])
    i_avg = np.mean(gates['input'])
    o_avg = np.mean(gates['output'])
    c_avg = np.mean(np.abs(cells[t+1]))  # Độ lớn trạng thái tế bào
    
    marker = " <-- Đầu vào quan trọng" if t == 0 or t == len(all_gates)-1 else ""
    print(f"  {t:2d} | {f_avg:.3f}  | {i_avg:.3f} | {o_avg:.3f}  | {c_avg:.3f}{marker}")

print("\nQuan sát:")
print("- Cổng forget thường giữ cao (~0.9-1.0) để duy trì bộ nhớ")
print("- Cổng input mở cho đầu vào quan trọng (giá trị đầu và cuối)")
print("- Cổng output kiểm soát thông tin nào được phơi bày")
print("- Trạng thái tế bào tích lũy thông tin, duy trì độ lớn")
```

Tiếp theo là phần cài đặt GRU để đối chiếu trực tiếp với LSTM:

```python
class GRUCell:
    """
    GRU cell — phương án đơn giản hơn so với LSTM.
    
    GRU gộp trạng thái tế bào và trạng thái ẩn, chỉ dùng 2 cổng (so với 3 của LSTM),
    dẫn đến ~25% ít tham số hơn. Thường cho hiệu năng tương đương LSTM
    trong khi huấn luyện nhanh hơn và dễ tinh chỉnh hơn.
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        combined_size = hidden_size + input_size
        scale = 1.0 / np.sqrt(combined_size)
        
        # Hai cổng thay vì ba
        self.Wr = np.random.randn(hidden_size, combined_size) * scale  # Reset
        self.Wz = np.random.randn(hidden_size, combined_size) * scale  # Update
        self.Wh = np.random.randn(hidden_size, combined_size) * scale  # Ứng viên
        
        self.br = np.zeros((hidden_size, 1))
        self.bz = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
    
    def forward(self, x_t, h_prev):
        """
        GRU không có trạng thái tế bào riêng — đơn giản hơn!
        
        Chỉ trả về trạng thái ẩn mới (đóng vai trò cả trạng thái ẩn và tế bào)
        """
        combined = np.vstack([h_prev, x_t])
        
        # Cổng reset: dùng bao nhiêu quá khứ cho ứng viên
        r_t = self.sigmoid(self.Wr @ combined + self.br)
        
        # Cổng update: nội suy bao nhiêu giữa cũ và mới
        z_t = self.sigmoid(self.Wz @ combined + self.bz)
        
        # Trạng thái ẩn ứng viên (dùng trạng thái trước đã reset)
        combined_reset = np.vstack([r_t * h_prev, x_t])
        h_tilde = np.tanh(self.Wh @ combined_reset + self.bh)
        
        # Nội suy giữa cũ và mới
        # Khi z_t ≈ 0: giữ cũ (h_t ≈ h_prev)
        # Khi z_t ≈ 1: dùng mới (h_t ≈ h_tilde)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        gates = {'reset': r_t, 'update': z_t, 'candidate': h_tilde}
        
        return h_t, gates

# So sánh LSTM vs GRU trên cùng tác vụ
print("\n" + "="*70)
print("So sánh LSTM vs GRU")
print("="*70)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, 2, batch_first=True)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 16, 2, batch_first=True)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Huấn luyện cả hai
lstm_model = LSTMModel()
gru_model = GRUModel()

# Đếm tham số
lstm_params = sum(p.numel() for p in lstm_model.parameters())
gru_params = sum(p.numel() for p in gru_model.parameters())

print(f"LSTM parameters: {lstm_params:,}")
print(f"GRU parameters: {gru_params:,}")
print(f"GRU có {(1 - gru_params/lstm_params)*100:.1f}% ít tham số hơn")
print("\nCả hai đều có thể học phụ thuộc tầm xa hiệu quả.")
print("GRU: Đơn giản hơn, nhanh hơn. LSTM: Linh hoạt hơn, đôi khi tốt hơn trên tác vụ phức tạp.")
```

## 5. Các khái niệm liên quan

Bước chuyển từ RNN cơ bản sang LSTM phản ánh một hướng tiếp cận phổ biến trong học sâu: phân tích rõ điểm yếu toán học của kiến trúc cũ để thiết kế cải tiến trực tiếp vào cấu trúc mạng. RNN cơ bản gặp khó trên chuỗi dài do gradient biến mất trong BPTT. LSTM giải quyết bằng cách bổ sung trạng thái tế bào với phép cập nhật cộng theo phần tử kết hợp các cổng điều tiết, giúp luồng gradient truyền ổn định hơn.

Bước tinh giản từ LSTM sang GRU lại minh họa nguyên tắc tối giản hóa kiến trúc khi vẫn bảo toàn cơ chế cốt lõi. GRU đạt hiệu năng tương đương LSTM trên nhiều bài toán nhưng giảm khoảng 25% số tham số và bỏ trạng thái tế bào riêng biệt. Cổng *update* đảm nhận vai trò kết hợp giữa cổng quên và cổng vào, trong khi cổng *reset* kiểm soát mức độ tham gia của trạng thái trước đó vào ứng viên mới. Trong thực hành, ta thường bắt đầu với GRU vì đơn giản và huấn luyện nhanh hơn, rồi mới chuyển sang LSTM nếu bài toán thực sự cần thêm dung lượng biểu diễn.

Cơ chế cổng — dùng hàm kích hoạt sigmoid có đầu ra trong khoảng (0, 1) để điều tiết luồng dữ liệu — xuất hiện rộng rãi trong nhiều kiến trúc học sâu khác. Highway Network dùng cổng để kiểm soát kết nối bỏ qua; cơ chế attention dùng trọng số chuẩn hóa để chọn lọc thông tin; Neural Turing Machine dùng cổng để điều khiển thao tác đọc và ghi bộ nhớ. Điểm chung của các thiết kế này là khả năng học quy tắc điều hướng thông tin một cách trơn và khả vi.

So với cơ chế attention, LSTM giải quyết phụ thuộc tầm xa theo hướng khác. LSTM nén toàn bộ lịch sử vào trạng thái kích thước cố định được cập nhật qua các cổng, giữ độ phức tạp tuyến tính $$O(n)$$. Attention cho phép truy cập trực tiếp mọi trạng thái quá khứ mà không qua nút thắt nén cố định, nhưng đổi lại chi phí bậc hai $$O(n^2)$$. Với các bài toán xử lý ngôn ngữ có đủ tài nguyên tính toán, khả năng truy cập trực tiếp của Transformer thường chiếm ưu thế; còn với chuỗi rất dài hoặc yêu cầu xử lý trực tuyến với bộ nhớ hạn chế, LSTM và GRU vẫn có chỗ đứng.

Cơ chế quản lý bộ nhớ trong LSTM cũng gần gũi với các bài toán quản lý bộ nhớ đệm (cache) trong khoa học máy tính: quyết định dữ liệu nào cần lưu, dữ liệu nào đã cũ cần loại bỏ và khi nào cần truy xuất. Thay vì dùng quy tắc cố định như trong hệ điều hành hay CPU, LSTM học chính sách quản lý bộ nhớ trực tiếp từ dữ liệu.

Nhìn chung, chuỗi phát triển từ RNN sang LSTM rồi Transformer cho thấy quy luật tiến hóa của thiết kế kiến trúc mạng: phát hiện giới hạn qua phân tích, xây dựng cơ chế mới để tháo gỡ, rồi tiếp tục giải quyết các giới hạn mới phát sinh (như tính tuần tự và nút thắt trạng thái cố định).

## 6. Các bài báo nền tảng

**["Long Short-Term Memory" (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)**  
*Tác giả*: Sepp Hochreiter, Jürgen Schmidhuber  
Bài báo nền tảng này giới thiệu kiến trúc LSTM kèm phân tích toán học chặt chẽ lý giải vì sao RNN cơ bản thất bại khi học phụ thuộc tầm xa. Hai tác giả chứng minh trong lan truyền ngược theo thời gian, gradient sẽ biến mất hoặc bùng nổ theo hàm mũ trừ khi mạng có cơ chế truyền sai số ổn định. Họ đề xuất cơ chế *constant error carousel* (trạng thái tế bào) để duy trì luồng gradient qua nhiều bước thời gian. Dù phải mất nhiều năm sau LSTM mới được ứng dụng rộng rãi khi năng lực tính toán và dữ liệu tăng lên, bài báo này đã đặt nền móng lý thuyết cho gần hai thập kỷ phát triển của mô hình hóa chuỗi.

**["Learning to Forget: Continual Prediction with LSTM" (2000)](https://doi.org/10.1162/089976600300015015)**  
*Tác giả*: Felix A. Gers, Jürgen Schmidhuber, Fred Cummins  
Phiên bản LSTM năm 1997 không có cơ chế xóa trạng thái tế bào nên dễ bị bão hòa khi xử lý chuỗi dài. Bài báo này bổ sung cổng quên (*forget gate*), cho phép mạng loại bỏ chọn lọc thông tin lỗi thời khỏi bộ nhớ. Cải tiến gọn nhẹ này nâng cao đáng kể hiệu năng trên các bài toán dự đoán liên tục và nhanh chóng trở thành thành phần chuẩn của mọi cài đặt LSTM.

**["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014)](https://arxiv.org/abs/1406.1078)**  
*Tác giả*: Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio  
Bài báo giới thiệu Gated Recurrent Unit (GRU) như một phương án đơn giản hơn LSTM, đồng thời đề xuất khung kiến trúc encoder-decoder cho dịch máy neuron. Kết quả trên bài toán dịch máy cho thấy hai cổng (*reset* và *update*) đủ để kiểm soát luồng thông tin hiệu quả với ít tham số hơn và tốc độ huấn luyện nhanh hơn. Khung encoder-decoder từ công trình này cũng trở thành cấu trúc tiêu chuẩn cho nhiều bài toán chuỗi-sang-chuỗi về sau.

**["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)](https://arxiv.org/abs/1412.3555)**  
*Tác giả*: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio  
Bài báo thực hiện đánh giá thực nghiệm có kiểm soát đầu tiên giữa LSTM và GRU trên nhiều bài toán như mô hình hóa âm nhạc, nhận dạng giọng nói và mô hình ngôn ngữ. Kết quả cho thấy không có kiến trúc nào vượt trội tuyệt đối trong mọi trường hợp, nhưng GRU thường đạt độ chính xác tương đương LSTM với thời gian huấn luyện ngắn hơn. Công trình khẳng định việc chọn kiến trúc cần dựa vào đặc thù dữ liệu, độ dài chuỗi và ngân sách tính toán của từng bài toán cụ thể.

**["Visualizing and Understanding Recurrent Networks" (2015)](https://arxiv.org/abs/1506.02078)**  
*Tác giả*: Andrej Karpathy, Justin Johnson, Li Fei-Fei  
Bài báo phân tích biểu diễn nội bộ của LSTM trên mô hình ngôn ngữ cấp ký tự bằng cách theo dõi kích hoạt của từng đơn vị ẩn và các cổng. Nhóm tác giả phát hiện một số tế bào tự học cách theo dõi dấu ngoặc kép, mức thụt lề mã nguồn hay ranh giới câu mà không cần quy tắc lập trình sẵn. Công trình này góp phần định hình các phương pháp phân tích khả năng diễn giải cho mạng neuron sâu.

## Các bẫy phổ biến và thủ thuật thực hành

Một lỗi phổ biến khi tự cài đặt LSTM là khởi tạo độ lệch (bias) của cổng quên bằng 0 giống các tham số khác. Khi đó giá trị ban đầu của cổng quên xấp xỉ $$\sigma(0) = 0.5$$, khiến mạng quên đi một nửa thông tin trạng thái tế bào ngay sau mỗi bước thời gian và khó học các phụ thuộc tầm xa ở giai đoạn đầu. Giải pháp tiêu chuẩn là mẹo khởi tạo độ lệch cổng quên (*forget bias trick*): đặt $$\mathbf{b}_f = \mathbf{1}$$ để giá trị ban đầu của cổng quên đạt $$\sigma(0 + 1) \approx 0.73$$, thiên về giữ lại thông tin trước rồi mới học cách quên dần khi cần.

Hiện tượng gradient bùng nổ tuy ít xảy ra hơn so với RNN cơ bản nhưng vẫn có thể xuất hiện. Khi cổng quên tiến sát 1 và cổng vào liên tục mở cho các giá trị ứng viên cùng dấu, trạng thái tế bào cập nhật theo $$c_t = 1 \cdot c_{t-1} + 1 \cdot \tilde{c}_t$$ sẽ tăng tuyến tính không bị chặn theo độ dài chuỗi, dẫn đến mất ổn định số học, mất mát tăng vọt hoặc xuất hiện `NaN`. Bên cạnh cắt ngưỡng gradient (*gradient clipping*), các biện pháp hỗ trợ gồm:
- Giới hạn giá trị ứng viên bằng hàm $$\tanh$$ (đã tích hợp sẵn trong LSTM chuẩn).
- Sử dụng Layer Normalization để giữ độ lớn trạng thái ổn định.
- Khởi tạo trọng số phù hợp để tránh bão hòa cổng quá sớm.

Một điểm cần lưu ý khác là sự phối hợp giữa cổng quên và cổng vào. Do hai cổng hoạt động độc lập trong LSTM chuẩn, mô hình đôi khi có thể rơi vào trạng thái bất thường khi đồng thời quên hết thông tin cũ ($$f_t \approx 0$$) mà không ghi thông tin mới ($$i_t \approx 0$$), khiến trạng thái tế bào tiến về 0. GRU tránh tình huống này bằng cách ràng buộc trực tiếp hai trọng số là $$1 - z_t$$ và $$z_t$$. Khi gỡ lỗi huấn luyện LSTM không hội tụ, kiểm tra phân phối giá trị các cổng (xem chúng có bị kẹt sát 0 hoặc 1 hay không) thường giúp xác định nguyên nhân nhanh chóng.

Về lựa chọn giữa LSTM và GRU, kinh nghiệm thực tế là bắt đầu bằng GRU để tận dụng tốc độ huấn luyện nhanh và ít tham số hơn. Nếu hiệu năng chững lại trên bài toán có dữ liệu lớn, chuỗi dài hoặc cấu trúc thời gian phức tạp, hãy thử chuyển sang LSTM để kiểm chứng xem trạng thái tế bào tách biệt có cải thiện kết quả hay không.

Khi xếp chồng nhiều lớp LSTM, có thể áp dụng dropout lên đầu ra trạng thái ẩn truyền giữa các lớp, nhưng không nên áp dụng trực tiếp lên trạng thái tế bào hoặc kết nối hồi quy trong cùng một lớp theo cách thông thường. Tỷ lệ dropout cho LSTM thường nhỏ hơn mạng feedforward (khoảng 0.2 đến 0.3 thay vì 0.5) vì tỷ lệ quá cao sẽ làm gián đoạn luồng thông tin xuyên suốt chuỗi.

LSTM hai chiều xử lý chuỗi theo cả chiều xuôi và chiều ngược rồi ghép hai trạng thái ẩn ở mỗi bước: $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$. Thiết kế này tăng gấp đôi số tham số và chi phí tính toán nhưng mang lại biểu diễn đầy đủ hơn cho các bài toán có sẵn toàn bộ đầu vào như dịch câu hoàn chỉnh hoặc nhận dạng đoạn âm thanh đã ghi sẵn; ngược lại, không thể dùng cho dự đoán trực tuyến thời gian thực hay sinh tự hồi quy.

Cuối cùng, trực quan hóa giá trị kích hoạt của các cổng $$f_t$$, $$i_t$$, $$o_t$$ theo từng bước thời gian là công cụ phân tích rất hữu ích. Quan sát thời điểm cổng quên hạ thấp (ví dụ tại ranh giới câu), cổng vào mở (tại các từ khóa chính) hay cổng ra kích hoạt giúp vừa gỡ lỗi huấn luyện, vừa kiểm chứng xem mô hình đã học đúng cấu trúc của dữ liệu hay chưa.

## Điểm chính

LSTM khắc phục hiện tượng gradient biến mất của RNN cơ bản bằng cách bổ sung trạng thái tế bào (*cell state*) với cơ chế cập nhật cộng theo phần tử được điều khiển bởi ba cổng (*forget*, *input*, *output*). Cấu trúc này giúp gradient lan truyền ổn định qua hàng trăm bước thời gian và mang lại hiệu quả cao cho nhiều bài toán dữ liệu chuỗi. GRU tinh giản thiết kế của LSTM xuống còn hai cổng và một trạng thái duy nhất, thường đạt hiệu năng tương đương với chi phí tính toán thấp hơn. Hiểu rõ vai trò của từng cổng và lý do tồn tại của phép cập nhật cộng là chìa khóa để sử dụng và gỡ lỗi hiệu quả các mô hình hồi quy có cổng.

Thành công của LSTM cũng cho thấy một bài học quan trọng trong thiết kế mạng sâu: phân tích chặt chẽ động lực gradient giúp tạo ra những thay đổi kiến trúc giải quyết tận gốc các giới hạn huấn luyện.

## Video tham khảo

- **Xem trước:** [Simple Explanation of GRU (Gated Recurrent Units) | Deep Learning Tutorial 37 (Tensorflow & Python)](https://www.youtube.com/watch?v=tOuXgORsXJ4) — codebasics (tiếng Anh). Giải thích cách GRU gộp quyết định quên và ghi mới, khớp trực tiếp với phần so sánh trong bài.
- [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The AI Hacker (tiếng Anh). Ôn lại chi tiết từng cổng LSTM làm nền cho phần đối chiếu với GRU.
- [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest with Josh Starmer (tiếng Anh). Làm rõ cơ chế hoạt động và giới hạn dung lượng của trạng thái tế bào khi chuỗi quá dài.
