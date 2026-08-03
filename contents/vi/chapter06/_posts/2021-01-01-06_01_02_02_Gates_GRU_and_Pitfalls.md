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

Bây giờ triển khai GRU để so sánh:

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

## 5. Các Khái niệm Liên quan

Mối quan hệ giữa LSTM và RNN vanilla minh họa một mẫu lặp lại trong học sâu: nhận diện chế độ thất bại của kiến trúc đơn giản và thiết kế giải pháp có mục tiêu qua đổi mới kiến trúc. RNN vanilla thất bại trên chuỗi dài do gradient biến mất trong backpropagation through time. LSTM giải quyết bằng cách tạo đường dẫn thông tin riêng (trạng thái tế bào) với cập nhật cộng thay vì nhân, và dùng cổng để kiểm soát luồng thông tin. Đây không chỉ là sửa lỗi — đây là thay đổi kiến trúc cơ bản được thúc đẩy bởi hiểu toán học của luồng gradient.

Sự tiến hóa từ LSTM đến GRU minh họa một nguyên tắc quan trọng khác: đơn giản hơn có thể tốt hơn khi nó bảo toàn cơ chế cốt lõi. GRU đạt hiệu năng tương tự LSTM cho nhiều tác vụ trong khi có 25% ít tham số hơn và động lực đơn giản hơn (không có trạng thái tế bào riêng). Triết lý thiết kế của GRU là tối giản — dùng ít cơ chế nhất cần thiết để đạt hành vi mong muốn. Cổng update kết hợp cổng forget và input của LSTM, giảm tham số trong khi duy trì khả năng then chốt kiểm soát bộ nhớ. Cổng reset thay thế chức năng cổng output theo cách khác. Với người thực hành, điều này thường nghĩa là bắt đầu với GRU (đơn giản hơn, nhanh hơn) và chỉ chuyển sang LSTM nếu tác vụ rõ ràng hưởng lợi từ dung lượng bổ sung.

Kết nối với cơ chế cổng trong kiến trúc neuron rộng hơn tiết lộ một mẫu mạnh mẽ. Cổng — các lớp kích hoạt sigmoid xuất ra giá trị trong (0,1) dùng để điều tiết các giá trị khác — xuất hiện khắp học sâu. Highway network dùng cổng để kiểm soát skip connection. Cơ chế attention dùng cổng (trọng số attention) để chọn thông tin. Neural Turing Machine dùng cổng để kiểm soát đọc/ghi bộ nhớ. Mẫu nhất quán: khi ta cần kiểm soát có thể học về luồng thông tin, ta dùng cổng. Hiểu vì sao điều này hoạt động — khả vi mượt, diễn giải như xác suất, hiệu quả học hành vi có điều kiện — giúp đánh giá motif kiến trúc này.

LSTM và cơ chế attention có mối quan hệ thú vị. Cả hai giải quyết phụ thuộc tầm xa, nhưng theo cách khác nhau. LSTM nén tất cả thông tin quá khứ vào trạng thái kích thước cố định, cập nhật qua cổng. Attention cho phép truy cập trực tiếp mọi trạng thái quá khứ, chọn những cái liên quan qua trọng số attention. Điều này khiến attention mạnh hơn (không nén mất mát) nhưng đắt hơn ($$O(n^2)$$ thay vì $$O(n)$$). Thành công của Transformer gợi ý rằng với nhiều tác vụ NLP có đủ tài nguyên tính toán, truy cập trực tiếp của attention vượt trội hiệu quả của LSTM. Tuy nhiên với tác vụ có chuỗi rất dài hoặc ràng buộc thời gian thực, LSTM vẫn còn liên quan.

Khái niệm quản lý bộ nhớ tường minh trong LSTM kết nối với khoa học máy tính rộng hơn — ý tưởng caching thông tin quan trọng, loại bỏ dữ liệu lỗi thời, và kiểm soát truy cập. Hệ thống cơ sở dữ liệu, quản lý bộ nhớ hệ điều hành, và cache CPU đều đối mặt thách thức tương tự về quyết định nhớ gì và quên gì với dung lượng hạn chế. LSTM học các chính sách tương tự từ dữ liệu thay vì được mã hóa thủ công. Kết nối này giúp định khung những gì LSTM đang làm: chúng là hệ thống quản lý bộ nhớ có thể học, khả vi.

Cuối cùng, hiểu thành công và hạn chế của LSTM thông tin cho thiết kế kiến trúc nói chung. LSTM thành công vì chúng giải quyết một vấn đề cụ thể, được hiểu rõ (gradient biến mất) bằng giải pháp có mục tiêu (trạng thái tế bào có cổng). Hạn chế của chúng (xử lý tuần tự, nghẽn trạng thái kích thước cố định) thúc đẩy các đổi mới tiếp theo (attention, Transformer). Tiến trình từ RNN đơn giản đến LSTM phức tạp đến Transformer dựa trên attention cho thấy lĩnh vực tiến hóa ra sao: nhận diện hạn chế qua phân tích, thiết kế kiến trúc giải quyết những hạn chế đó, khám phá hạn chế mới, lặp lại. Mỗi kiến trúc dạy ta điều gì đó về thiên kiến qui nạp và cơ chế cần thiết cho các kiểu suy luận tuần tự khác nhau.

## 6. Các Bài báo Cơ bản

**["Long Short-Term Memory" (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)**  
*Tác giả*: Sepp Hochreiter, Jürgen Schmidhuber  
Bài báo nền tảng này giới thiệu kiến trúc LSTM và phân tích nghiêm ngặt vì sao RNN vanilla thất bại trong việc học phụ thuộc tầm xa. Hochreiter và Schmidhuber chỉ ra bằng toán học rằng trong backpropagation through time, gradient hoặc biến mất hoặc bùng nổ theo cấp số nhân trừ khi mạng được xây dựng cẩn thận để tránh điều này. Họ đề xuất LSTM với constant error carousel (trạng thái tế bào) như một giải pháp, chứng minh rằng LSTM về nguyên tắc có thể học phụ thuộc tầm xa tùy ý. Bài báo đáng chú ý về tính tiên tri, đề cập các vấn đề như dung lượng bộ nhớ và đề xuất giải pháp sau này trở thành chuẩn (như cổng forget, được thêm trong công trình sau). Dù LSTM mất nhiều năm để được áp dụng rộng rãi (một phần do tài nguyên tính toán và dataset hạn chế lúc đó), bài báo này thiết lập nền tảng lý thuyết và chứng minh ưu thế của LSTM trên các tác vụ được thiết kế cẩn thận đòi hỏi bộ nhớ dài hạn. Đây là một trong những bài báo được trích dẫn nhiều nhất trong toàn bộ học sâu và có thể nói đã cho phép phần lớn tiến bộ trong mô hình hóa chuỗi trong hai thập kỷ tiếp theo.

**["Learning to Forget: Continual Prediction with LSTM" (2000)](https://doi.org/10.1162/089976600300015015)**  
*Tác giả*: Felix A. Gers, Jürgen Schmidhuber, Fred Cummins  
Kiến trúc LSTM gốc thiếu cơ chế reset trạng thái tế bào — nó chỉ có thể thêm thông tin, không thể loại bỏ. Điều này dẫn đến vấn đề bão hòa trên chuỗi dài nơi trạng thái tế bào đầy thông tin lỗi thời. Bài báo này giới thiệu cổng forget, cho phép mạng chọn lọc xóa các phần bộ nhớ khi chúng không còn cần thiết. Sự bổ sung tưởng chừng đơn giản này — một cổng nữa điều tiết cập nhật trạng thái tế bào — khiến LSTM thực tiễn hơn đáng kể cho các tác vụ thực tế. Bài báo chứng minh hiệu năng cải thiện trên các tác vụ học liên tục nơi mạng phải xử lý nhiều chuỗi và reset ngữ cảnh giữa chúng. Cổng forget đã trở thành phần chuẩn của mọi triển khai LSTM, và bài báo minh họa cách các chi tiết kiến trúc tưởng nhỏ có thể có tác động thực tiễn lớn. Nó cũng chứng minh giá trị của tinh chỉnh liên tục — các kiến trúc tốt nhất thường nảy sinh qua cải tiến lặp lại giải quyết các vấn đề thực tiễn phát hiện khi áp dụng.

**["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014)](https://arxiv.org/abs/1406.1078)**  
*Tác giả*: Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio  
Bài báo này giới thiệu Gated Recurrent Unit (GRU) như phương án đơn giản hơn so với LSTM đồng thời đề xuất kiến trúc encoder-decoder cho dịch máy neuron. Thiết kế GRU được thúc đẩy bởi sự phức tạp của LSTM — liệu ta có thể đạt hiệu năng tương tự với ít tham số hơn và động lực đơn giản hơn? Bài báo chỉ ra rằng hai cổng của GRU (reset và update) có thể kiểm soát luồng thông tin gần như hiệu quả như ba cổng của LSTM, trong khi dễ triển khai hơn và huấn luyện nhanh hơn. Kết quả thực nghiệm trên dịch máy chứng minh rằng đơn giản hóa kiến trúc không nhất thiết làm giảm hiệu năng khi các cơ chế cốt lõi (cổng để kiểm soát bộ nhớ) được bảo toàn. Bài báo ảnh hưởng triết lý thiết kế kiến trúc: ưu tiên thiết kế đơn giản hơn khi chúng duy trì các tính chất then chốt, vì sự đơn giản hỗ trợ debug, tinh chỉnh, và hiểu biết. Khung encoder-decoder giới thiệu ở đây trở thành chuẩn cho các tác vụ chuỗi-sang-chuỗi, dù dùng RNN, LSTM, GRU, hay cuối cùng là Transformer.

**["Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014)](https://arxiv.org/abs/1412.3555)**  
*Tác giả*: Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio  
Bài báo này cung cấp so sánh thực nghiệm toàn diện đầu tiên về LSTM và GRU trên nhiều tác vụ mô hình tuần tự bao gồm mô hình hóa nhạc, nhận dạng giọng nói, và mô hình ngôn ngữ. Phương pháp thực nghiệm cẩn thận — kiểm soát siêu tham số, chiều sâu kiến trúc, và quy trình huấn luyện — cho phép so sánh công bằng tập trung vào sự khác biệt kiến trúc. Các phát hiện mang tính sắc thái: không kiến trúc nào thống trị nhất quán trên mọi tác vụ, nhưng GRU thường khớp hiệu năng LSTM trong khi huấn luyện nhanh hơn nhờ ít tham số. Bài báo thiết lập rằng lựa chọn kiến trúc nên phụ thuộc đặc điểm tác vụ cụ thể và ràng buộc (kích thước dataset, độ dài chuỗi, ngân sách tính toán) thay vì là khuyến nghị phổ quát. Nó cũng chứng minh cách đánh giá đúng đổi mới kiến trúc — không chỉ cho thấy một kết quả tốt mà so sánh hệ thống trên nhiều tác vụ đa dạng với độ chặt chẽ thống kê. Phương pháp này đã trở thành chuẩn trong nghiên cứu học sâu.

**["Visualizing and Understanding Recurrent Networks" (2015)](https://arxiv.org/abs/1506.02078)**  
*Tác giả*: Andrej Karpathy, Justin Johnson, Li Fei-Fei  
Bài báo này điều tra những gì LSTM học bằng cách phân tích biểu diễn nội bộ của chúng trên mô hình ngôn ngữ cấp ký tự. Bằng cách xem xét kích hoạt của từng đơn vị ẩn và cổng, Karpathy chứng minh rằng LSTM tự phát triển cấu trúc nội bộ có thể diễn giải. Một số cell theo dõi ký tự dấu ngoặc kép (kích hoạt bên trong ngoặc, tắt bên ngoài), số khác theo dõi mức thụt lề trong mã, số khác phát hiện kết thúc dòng hoặc khối comment. Cổng forget học reset tại ranh giới câu. Cấu trúc nổi lên này không được lập trình tường minh mà nảy sinh từ mục tiêu huấn luyện dự đoán ký tự tiếp theo. Phương pháp của bài báo — phân tích hệ thống từng đơn vị, kích hoạt cổng, và mẫu lỗi — thiết lập các cách tiếp cận phân tích diễn giải đã được áp dụng cho mọi loại mạng neuron. Nó cho thấy LSTM không chỉ đạt hiệu năng tốt qua tính toán mờ đục mà phát triển các biểu diễn nội bộ có ý nghĩa mà ta có thể hiểu và xác nhận. Tính diễn giải này khiến LSTM giá trị không chỉ vì hiệu năng mà vì cung cấp cái nhìn về các mẫu mô hình đã khám phá trong dữ liệu.

## Các Bẫy phổ biến và Thủ thuật

Lỗi phổ biến nhất khi triển khai LSTM là khởi tạo bias cổng forget bằng zero, như các bias khác. Điều này khiến cổng forget bắt đầu quanh 0.5 (từ sigmoid của 0), nghĩa là mạng ban đầu quên một nửa trạng thái tế bào ở mỗi bước. Với hầu hết tác vụ, việc quên hung hãn sớm trong huấn luyện ngăn mạng khám phá rằng phụ thuộc tầm xa quan trọng. Giải pháp là "forget bias trick": khởi tạo $$\mathbf{b}_f = \mathbf{1}$$ (vectơ toàn số 1). Điều này khiến cổng forget ban đầu $$\sigma(0 + 1) \approx 0.73$$, thiên về giữ lại. Khi huấn luyện tiến triển, nếu quên có lợi, mạng có thể học giảm giá trị cổng forget. Thủ thuật khởi tạo đơn giản này có thể tạo ra sự khác biệt giữa LSTM huấn luyện thành công và LSTM không bao giờ học được phụ thuộc tầm xa.

Gradient bùng nổ, dù ít vấn đề hơn trong LSTM so với RNN vanilla nhờ động lực trạng thái tế bào, vẫn có thể xảy ra. Vấn đề nay thường đến từ chính các cổng. Nếu cổng forget bão hòa ở 1 và cổng input cho phép giá trị ứng viên lớn, trạng thái tế bào có thể tăng không giới hạn: $$c_t = 1 \cdot c_{t-1} + 1 \cdot \tilde{c}_t$$ lặp nhiều lần cho tăng trưởng cấp số nhân. Điều này biểu hiện thành tham số trở thành NaN trong huấn luyện hoặc loss bùng nổ. Giải pháp chuẩn vẫn là gradient clipping, nhưng các giải pháp đặc thù LSTM bao gồm:
- Ràng buộc giá trị ứng viên qua tanh (mà LSTM đã làm)
- Dùng layer normalization để giữ trạng thái tế bào trong dải hợp lý
- Khởi tạo trọng số cẩn thận để ngăn bão hòa cổng

Một vấn đề tinh tế là sự ghép nối giữa cổng forget và input. Về nguyên tắc, các cổng này có thể học hành vi xung đột — quên thông tin cũ ($$f_t \approx 0$$) trong khi không thêm mới ($$i_t \approx 0$$), khiến trạng thái tế bào biến mất. GRU tránh điều này bằng cách ghép chúng: $$1 - z_t$$ giữ cũ, $$z_t$$ thêm mới, đảm bảo ít nhất một cái đáng kể. Một số biến thể LSTM cũng ghép cổng, dù LSTM chuẩn cho phép chúng độc lập. Trong thực tế, khởi tạo đúng và dữ liệu huấn luyện đủ thường cho phép LSTM học phối hợp cổng hợp lý, nhưng khi debug thất bại huấn luyện LSTM, kiểm tra hành vi cổng bệnh lý (tất cả cổng gần 0 hoặc 1) có thể tiết lộ vấn đề.

Lựa chọn giữa LSTM và GRU đã tạo nhiều thảo luận nhưng ít kết luận phổ quát. Như heuristic thực tiễn: bắt đầu với GRU vì đơn giản và nhanh hơn. Nếu hiệu năng bão hòa và bạn có nhiều dữ liệu, thử LSTM xem dung lượng bổ sung có giúp không. Với chuỗi rất dài hoặc mẫu thời gian phức tạp, trạng thái tế bào riêng của LSTM thường mang lại ưu thế. Với tác vụ dữ liệu hạn chế hoặc thời gian huấn luyện bị ràng buộc, hiệu quả của GRU thường khiến nó ưa thích hơn. Luôn validate trên bài toán cụ thể thay vì giả định một kiến trúc luôn tốt hơn.

Khi xếp chồng nhiều lớp LSTM, câu hỏi phổ biến là có nên áp dụng dropout giữa các lớp không. Câu trả lời: có, nhưng cẩn thận. Áp dụng dropout lên đầu ra (trạng thái ẩn) truyền giữa các lớp, không lên trạng thái tế bào hay kết nối hồi quy trong một lớp. Tỷ lệ dropout điển hình cho LSTM thấp hơn mạng feedforward — 0.2 đến 0.3 thay vì 0.5 — vì LSTM đã được regularize khá qua cơ chế cổng. Dropout quá nhiều có thể ngăn LSTM học phụ thuộc tầm xa mà chúng được thiết kế cho, vì việc drop ngẫu nhiên phá vỡ luồng thông tin qua thời gian.

LSTM hai chiều xử lý chuỗi theo cả hướng tiến và lùi, kết hợp thông tin từ cả hai ở mỗi bước thời gian: $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$. Điều này nhân đôi tham số và tính toán nhưng cung cấp biểu diễn phong phú hơn khi ngữ cảnh tương lai có sẵn. Tuy nhiên, LSTM hai chiều không thể dùng cho dự đoán tuần tự thời gian thực (nơi ta phải dự đoán trước khi thấy chuỗi hoàn chỉnh) hoặc cho sinh tự hồi quy. Chúng mạnh mẽ cho các tác vụ như dịch máy (nơi ta có câu nguồn hoàn chỉnh) hoặc nhận dạng giọng nói (nơi ta có thể xử lý toàn bộ âm thanh trước khi phiên âm), nhưng không phù hợp cho dự đoán online hoặc tác vụ sinh.

Một kỹ thuật mạnh để phân tích và debug là trực quan hóa kích hoạt cổng theo thời gian. Vẽ $$f_t$$, $$i_t$$, $$o_t$$ cho mỗi chiều khi mạng xử lý một chuỗi. Các mẫu tiết lộ những gì mạng đã học: cổng forget rơi tại ranh giới câu, cổng input mở cho từ nội dung và đóng cho từ chức năng, cổng output phơi bày thông tin khi cần quyết định. Trực quan hóa này không chỉ giúp debug vấn đề huấn luyện mà cung cấp cái nhìn về cấu trúc ngôn ngữ hoặc tuần tự mà mạng đã khám phá, khiến LSTM diễn giải được hơn nhiều kiến trúc học sâu khác.

## Điểm Chính

Mạng Long Short-Term Memory giải quyết vấn đề gradient biến mất từng giới hạn RNN vanilla bằng cách giới thiệu trạng thái tế bào (*cell state*) với kết nối có cổng cho phép thông tin chảy qua thời gian với suy giảm tối thiểu. Kiến trúc dùng ba cổng (*gate*) — forget, input, và output — mỗi cái triển khai như lớp sigmoid, để kiểm soát thông tin nào được giữ, thêm, hoặc phơi bày ở mỗi bước thời gian. Cơ chế cổng này cho phép học phụ thuộc trải dài hàng trăm bước thời gian, khiến LSTM thành công cho dịch máy, nhận dạng giọng nói, và nhiều tác vụ tuần tự khác đòi hỏi bộ nhớ dài hạn. Trạng thái tế bào cung cấp đường dẫn cập nhật cộng nơi gradient có thể chảy dễ dàng hơn qua các cập nhật nhân, phi tuyến của trạng thái ẩn RNN vanilla. Gated Recurrent Unit đơn giản hóa LSTM bằng cách dùng hai cổng thay vì ba và gộp trạng thái tế bào với trạng thái ẩn, thường đạt hiệu năng tương đương với ít tham số hơn. Lựa chọn giữa LSTM và GRU phụ thuộc độ phức tạp tác vụ, dữ liệu có sẵn, và ràng buộc tính toán, với GRU thường là điểm xuất phát tốt nhờ sự đơn giản. Hiểu LSTM sâu sắc nghĩa là đánh giá không chỉ các phương trình mà vì sao mỗi thành phần tồn tại — cổng cho phép quản lý bộ nhớ có thể học ra sao, vì sao trạng thái tế bào dùng cập nhật cộng, các lựa chọn thiết kế này cho phép luồng gradient ra sao — và nhận ra LSTM như giải pháp cho thách thức cụ thể học phụ thuộc tầm xa trong dữ liệu tuần tự qua tối ưu hóa dựa trên gradient.

Thành công của LSTM chứng minh rằng thiết kế kiến trúc cẩn thận được thông tin bởi hiểu biết về động lực gradient có thể vượt qua các hạn chế cơ bản — bài học đã ảnh hưởng thiết kế kiến trúc neuron xa hơn mạng hồi quy.
