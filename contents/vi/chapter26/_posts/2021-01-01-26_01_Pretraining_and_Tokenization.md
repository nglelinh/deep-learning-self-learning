---
layout: post
title: 26-01 Tiền huấn luyện và tokenization
chapter: '26'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter26
---

Trước khi một dòng văn bản kịp chạm tới tầng Attention đầu tiên, nó phải đi qua "cánh cửa" **Tokenizer** — nơi một câu tiếng Việt có thể bị cắt thành vài token đẹp đẽ hoặc vụn vỡ thành hàng chục mảnh byte tùy cách huấn luyện BPE. Ngay sau đó là trái tim toán học của tiền huấn luyện LLM: công thức đạo hàm siêu gọn $$\partial\mathcal{L}/\partial z = p - t$$, mẹo **Log-Sum-Exp** chống tràn số mũ (tiền đề của FlashAttention), và quy ước chuyển vị $$X W^\top$$ của PyTorch.

## Bản đồ kiến thức nền: Bốn trụ cột của Tiền huấn luyện và Tokenization

- **Mô hình hóa ngôn ngữ tự hồi quy (Next-Token Prediction):** Tối ưu hóa trung bình Negative Log-Likelihood trên toàn chuỗi $$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \ln P_\theta(x_t \mid x_{<t})$$, liên hệ trực tiếp với chỉ số Perplexity $$\text{PPL} = e^{\mathcal{L}}$$.
- **Softmax + Cross-Entropy và đạo hàm $$p - t$$:** Rút gọn đạo hàm của tổ hợp hai hàm phức tạp thành hiệu trực giác giữa phân phối dự đoán $$p$$ và nhãn one-hot $$t$$.
- **Ổn định số học Log-Sum-Exp và Online Softmax:** Trừ giá trị cực đại $$m = \max_i z_i$$ trước khi lấy hàm mũ để không bao giờ gặp `NaN`/`Inf`, đồng thời cập nhật thống kê trượt từng khối (tiling) trong FlashAttention.
- **Quy ước hình học tầng tuyến tính `nn.Linear`:** Phân biệt công thức toán học $$Y = X W$$ với cách lưu ma trận chuyển vị `(out_features, in_features)` của PyTorch để tránh sai lệch khi tính FLOPs hay ghép ma trận LoRA.

### Khung giải thích (nên đọc chậm): Tính tay Softmax ổn định số học, Cross-Entropy, Perplexity và gradient $$p - t$$

Giả sử từ điển thu nhỏ có 3 token `[mèo, chó, cá]`, nhãn đúng là `mèo` ($$\mathbf{t} = [1, 0, 0]$$), và tầng chiếu cuối xuất ra logit rất lớn $$\mathbf{z} = [1002{,}0,\; 1000{,}0,\; 1000{,}0]$$.

1. **Nếu tính ngây thơ `exp(1002)`:** Vượt quá giới hạn FP32 ($$\approx 3{,}4 \times 10^{38}$$) → lập tức trả về `Inf / Inf = NaN`!
2. **Áp dụng mẹo trừ cực đại $$m = \max(\mathbf{z}) = 1002{,}0$$:**
   - Logit dịch chuyển: $$\tilde{\mathbf{z}} = [0{,}0,\; -2{,}0,\; -2{,}0]$$.
   - Hàm mũ an toàn: $$[e^0,\; e^{-2},\; e^{-2}] \approx [1{,}0000,\; 0{,}1353,\; 0{,}1353]$$ (tổng $$Z = 1{,}2706$$).
   - Phân phối xác suất: $$\mathbf{p} \approx [0{,}7870,\; 0{,}1065,\; 0{,}1065]$$.
3. **Cross-Entropy Loss & Perplexity:**
   - $$\mathcal{L} = -\ln(0{,}7870) \approx 0{,}2395$$ → Perplexity tương ứng $$\text{PPL} = e^{0{,}2395} \approx 1{,}27$$ (mô hình chỉ phân vân tương đương $$1{,}27$$ lựa chọn).
4. **Gradient theo logit ($$\mathbf{p} - \mathbf{t}$$):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = [0{,}7870 - 1,\; 0{,}1065 - 0,\; 0{,}1065 - 0] = [-0{,}2130,\; +0{,}1065,\; +0{,}1065].$$
   Hạ gradient sẽ trừ đi giá trị này: **tăng** logit của `mèo` thêm $$0{,}213\eta$$ và **giảm** logit của `chó`, `cá` mỗi từ $$0{,}1065\eta$$!

# Tiền huấn luyện và tokenization

Mô hình ngôn ngữ lớn (LLM) hiện đại được tiền huấn luyện với mục tiêu **dự đoán token kế tiếp**. Mọi năng lực phía sau — *prompting*, tuân thủ chỉ dẫn hay sử dụng công cụ — đều xây dựng trên một mô hình nhận chuỗi token tiền tố $$x_1,\ldots,x_{t-1}$$ và xuất ra phân bố xác suất cho token kế tiếp $$x_t$$.

Bài này tập trung vào ba nền tảng cốt lõi ở mức khái niệm và toán học: hàm mục tiêu tiền huấn luyện, bộ tách token xác định tập ký hiệu đầu vào, và cửa sổ ngữ cảnh giới hạn phạm vi thông tin mô hình xử lý được.

## 1. Dự đoán token kế tiếp

Gọi $$x = (x_1,\ldots,x_T)$$ là một chuỗi token. Mô hình ngôn ngữ nhân quả phân rã xác suất đồng thời của chuỗi thành tích các xác suất có điều kiện:

$$p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).$$

Quá trình huấn luyện tối thiểu hóa log-likelihood âm trung bình (hàm mất mát cross-entropy) trên kho văn bản:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).$$

Tại mỗi vị trí, mạng xuất ra vectơ logit $$z \in \mathbb{R}^{V}$$ (với $$V$$ là kích thước từ vựng), sau đó hàm softmax chuyển đổi thành phân bố $$p_\theta(\cdot \mid x_{<t})$$. Vì “nhãn” chính là token kế tiếp có sẵn trong văn bản gốc mà không cần gán nhãn thủ công, đây là dạng **học tự giám sát** theo tinh thần Chương 16 dù hàm mất mát có dạng phân loại thông thường.

Hai hệ quả trực tiếp ảnh hưởng đến các bài sau trong chương:

- Không gian đầu ra **rất lớn** (thường $$V \sim 32\mathrm{k}$$–$$200\mathrm{k}+$$). Vì vậy, chưng cất tri thức cổ điển khớp toàn bộ softmax như trên tập 1.000 lớp (Chương 23) không còn rẻ khi chuyển sang LLM — xem thêm bài **26-05**.
- Huấn luyện sử dụng cơ chế **teacher forcing**: mô hình luôn dự đoán dựa trên tiền tố *thật* từ dữ liệu, còn khi sinh văn bản thực tế nó phải dựa trên các token do chính mình sinh ra. Sự khác biệt giữa huấn luyện và suy luận là đặc trưng cố hữu; các kỹ thuật căn chỉnh và lấy mẫu được thiết kế để xử lý hiện tượng này.

Đoạn mã minh họa hàm mất mát trong PyTorch:

```python
import torch.nn.functional as F

def next_token_loss(logits, input_ids):
    # logits: (batch, seq, vocab) cho vị trí 0..T-1
    # targets: token kế tại mỗi vị trí
    targets = input_ids[:, 1:]
    pred = logits[:, :-1, :]
    return F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,  # padding
    )
```

Các mục dưới đây tóm tắt phần toán học trực tiếp tại tầng đầu ra, tầng tuyến tính và bộ tối ưu trong bối cảnh LLM (các chứng minh đầy đủ đã có ở Chương 02–03 và Chương 10).

## 2. Softmax, cross-entropy và $$\partial\mathcal{L}/\partial z = p - t$$

Tại mỗi vị trí, mô hình sinh ra vectơ logit $$z \in \mathbb{R}^{V}$$. Hàm softmax cho xác suất từng token:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.$$

Với nhãn mục tiêu dạng *one-hot* $$t$$ (chỉ bằng 1 tại chỉ số token đúng $$t^*$$), hàm mất mát cross-entropy là:

$$\mathcal{L} = -\sum_{i=1}^{V} t_i \log p_i = -\log p_{t^*}.$$

Lấy đạo hàm theo vectơ logit $$z$$ thu được hằng đẳng thức quen thuộc:

$$\frac{\partial \mathcal{L}}{\partial z} = p - t.$$

Đây chính là kết quả giản ước của cross-entropy kết hợp softmax từ Chương 03, nhưng ở LLM phép tính này diễn ra tại mọi vị trí trong chuỗi trên từ vựng cỡ hàng chục đến hàng trăm nghìn phần tử.

**Quan hệ giữa CE, KL và entropy.** Với hai phân bố rời rạc bất kỳ $$p^*$$ và $$p$$:

$$\mathrm{CE}(p^*, p) = H(p^*) + \mathrm{KL}(p^* \| p).$$

Khi $$p^*$$ là phân bố *one-hot*, entropy $$H(p^*) = 0$$, nên cross-entropy trùng khớp hoàn toàn với $$\mathrm{KL}(\text{one-hot} \| p)$$ và log-likelihood âm. Khi mục tiêu là phân bố *mềm* từ mô hình giáo viên trong chưng cất (**26-05**), $$H(p^*)$$ là hằng số đối với mô hình học sinh nên tín hiệu tối ưu chính là phân kỳ KL.

## 3. Log-sum-exp và cơ chế “softmax trực tuyến” trong FlashAttention

Tính trực tiếp $$e^{z_i}$$ dễ gây tràn số khi giá trị logit lớn. Để ổn định số học, ta trừ đi giá trị lớn nhất trên hàng $$m = \max_j z_j$$:

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \qquad p_i = \exp\big(z_i - \mathrm{LSE}(z)\big).$$

Kỹ thuật **log-sum-exp** này có thể cập nhật *tăng dần* theo từng khối: nếu đã có giá trị lớn nhất tạm thời $$m$$ và tổng chuẩn hóa $$s = \sum e^{z-m}$$ trên phần tiền tố, khi nhận thêm một khối logit mới với cực đại $$m'$$, ta cập nhật:

$$s \leftarrow s\, e^{m - m_{\mathrm{new}}} + \sum_{\text{new}} e^{z - m_{\mathrm{new}}}, \quad m_{\mathrm{new}} = \max(m, m').$$

Kết hợp cập nhật tương tự cho tổng có trọng số của các vectơ *value*, ta thu được **softmax trực tuyến** (*online softmax*) — nền tảng số học của [FlashAttention](https://arxiv.org/abs/2205.14135). Nhờ đó, như sẽ thấy ở bài **26-07**, GPU không cần lưu toàn bộ ma trận điểm $$L \times L$$ xuống bộ nhớ HBM.

## 4. Tầng tuyến tính: Quy ước toán học $$XW$$ và PyTorch

Với một batch trạng thái ẩn $$X \in \mathbb{R}^{B \times d_{\mathrm{in}}}$$, sách giáo khoa thường viết:

$$Y = X W + b, \qquad W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}.$$

Trong khi đó, `torch.nn.Linear(n_in, n_out)` lưu thuộc tính `weight` theo kích thước **$$(n_{\mathrm{out}}, n_{\mathrm{in}})$$** và tính $$X W^\top + b$$. Hai cách biểu diễn mô tả cùng một ánh xạ tuyến tính, chỉ khác nhau ở phép chuyển vị. Khi đếm số tham số ở bài **26-07**, mỗi phép chiếu vẫn có $$d_{\mathrm{in}} \times d_{\mathrm{out}}$$ trọng số; khi cài đặt lại một bài báo, cần kiểm tra rõ quy ước chiều ma trận của tác giả.

## 5. Hàm kích hoạt trong tầng FFN

Các hàm kích hoạt đã trình bày ở Chương 02; trong LLM có ba công thức quan trọng:

$$
\begin{aligned}
\mathrm{ReLU}(z) &= \max(0, z), \\
\mathrm{SiLU}(z) = \mathrm{Swish}(z) &= z\,\sigma(z), \\
\mathrm{SwiGLU}(x) &= \big(\mathrm{SiLU}(x W_{\mathrm{gate}}) \odot (x W_{\mathrm{up}})\big) W_{\mathrm{down}}.
\end{aligned}
$$

ReLU được dùng trong Transformer năm 2017; SiLU là hàm cổng trơn; còn SwiGLU là khối FFN có cổng gồm ba ma trận trọng số kết hợp tích theo phần tử (Hadamard), được sử dụng phổ biến trong các decoder hiện đại (**26-07**). Một số mô hình kiểu GPT đời cũ dùng GELU (Chương 02).

## 6. Adam / AdamW: Moment và quy ước áp dụng weight decay

Tiền huấn luyện LLM hầu như luôn dùng **AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) thay vì Adam gốc ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)). Hai ước lượng moment bậc một và bậc hai giữ nguyên như Chương 10:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
\end{aligned}
$$

Nếu cộng trực tiếp số hạng chính quy hóa L2 vào hàm mất mát trong Adam, gradient của L2 cũng bị chia bởi $$\sqrt{\hat{v}_t}$$ nên suy giảm trọng số không còn đồng đều. AdamW **tách riêng** bước suy giảm trọng số (*decoupled weight decay*):

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda \theta_{t-1} \right).$$

**Tham số nào nhận $$\lambda > 0$$?** Theo quy ước phổ biến trong các trình huấn luyện LLM (như Hugging Face `AdamW` hay các cấu hình họ Llama), *weight decay* được áp dụng cho các **ma trận trọng số 2-D** (các phép chiếu *attention*, FFN, và thường cả bảng nhúng / đầu LM), nhưng **không** áp dụng cho các tensor 1-D như hệ số điều chỉnh (*bias*) và hệ số tỉ lệ (*gain*) của RMSNorm / LayerNorm để tránh xung đột với vai trò chuẩn hóa của chúng (xem thêm **10-01**).

## 7. Tại sao dùng tokenization dưới từ thay vì ký tự hoặc từ nguyên vẹn?

Tách theo từng ký tự khiến chuỗi đầu vào quá dài và tiêu tốn năng lực mô hình cho chính tả. Ngược lại, tách theo từ nguyên vẹn khiến từ vựng phình to và dễ gặp lỗi với từ hiếm hoặc từ mới. **Tokenization dưới từ** (*subword tokenization*) cân bằng hai cực này: giữ nguyên các từ phổ biến thành một token và tách các từ hiếm thành các mảnh ghép tái sử dụng được.

Hai thuật toán phổ biến nhất trong thực tế:

- **Byte Pair Encoding (BPE):** Bắt đầu từ tập ký tự (hoặc byte) rồi lặp lại việc gộp cặp token liền kề xuất hiện nhiều nhất thành một token mới; bảng các quy tắc gộp tạo thành bộ tokenizer. [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) áp dụng BPE cho dịch máy nơ-ron, và các tokenizer kiểu GPT-2 kế thừa trực tiếp phương pháp này. Thư viện [tiktoken](https://github.com/openai/tiktoken) của OpenAI là bản cài đặt tối ưu tốc độ của BPE.
- **Unigram / SentencePiece:** Thay vì gộp tham lam từ dưới lên, mô hình Unigram khởi tạo một tập từ vựng ứng viên lớn rồi loại bỏ dần các token ít làm giảm hàm likelihood. [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226) (SentencePiece) là công cụ phổ biến cho phép huấn luyện trực tiếp trên văn bản thô (hỗ trợ cả BPE và Unigram), được dùng rộng rãi trong nhiều mô hình đa ngôn ngữ mở.

Một số đặc điểm thực hành quan trọng:

- Tokenization là **bước tiền xử lý tất định**: mô hình chỉ nhìn thấy chuỗi chỉ số token chứ không thấy trực tiếp cấu trúc ký tự bên trong, và không thể sinh ra ID nằm ngoài bảng từ vựng.
- Cùng một từ có thể tương ứng 1 token ở bộ từ vựng này nhưng 3 token ở bộ từ vựng khác; do đó **chi phí và độ dài ngữ cảnh luôn được tính theo token, không phải theo từ**.
- Các token đặc biệt (`<bos>`, `<eos>`, thẻ định dạng hội thoại, thẻ gọi công cụ) là thành phần chính thức của từ vựng; cấu hình sai chúng có thể làm lãng phí ngữ cảnh hoặc phá vỡ khuôn mẫu hội thoại.

Bảng nhúng $$E \in \mathbb{R}^{V \times d}$$ của Chương 18 chính là tầng đầu tiên (tầng 0) của mô hình ngôn ngữ: mỗi ID token được ánh xạ thành một vectơ, sau đó thông tin vị trí (như RoPE, bài tùy chọn **08-99**) được đưa vào bên trong cơ chế *attention*.

## 8. Cửa sổ ngữ cảnh

Cơ chế *self-attention* chuẩn tính tương tác giữa mọi cặp vị trí trong cửa sổ. Với độ dài ngữ cảnh $$L$$, độ phức tạp của *attention* tại mỗi tầng là $$O(L^2)$$, khiến $$L$$ đồng thời là giới hạn về **khả năng nắm bắt ngữ cảnh** lẫn **tính toán và bộ nhớ**.

Trong thực tế, cửa sổ ngữ cảnh quy định ba điều:

- Mô hình chỉ điều kiện hóa trực tiếp trên tối đa $$L$$ token (gồm prompt, lịch sử hội thoại, đoạn truy hồi RAG từ Chương 18-99 và phần đã sinh).
- Các tài liệu quá dài phải được chia nhỏ, tóm tắt hoặc truy hồi theo đoạn; việc một ứng dụng xử lý tài liệu PDF 200 trang thường dựa vào hệ thống truy hồi đưa các đoạn liên quan vào trong giới hạn $$L$$.
- Khi giải mã, các vectơ *key* và *value* của toàn bộ tiền tố được lưu lại trong **KV-cache** để tái sử dụng (xem bài **26-04**).

Các cải tiến như cửa sổ trượt (*sliding window*), *grouped-query attention* hay mở rộng RoPE giúp giảm hằng số tính toán và bộ nhớ, nhưng không thay đổi bản chất rằng mô hình vẫn làm việc trên một vùng nhớ ngữ cảnh hữu hạn.

## 9. Dữ liệu tiền huấn luyện

Kho dữ liệu tiền huấn luyện thường kết hợp văn bản web đã lọc, sách, mã nguồn và các tài liệu chuyên ngành. Ba điểm cốt lõi cần lưu ý:

- Hàm mất mát thưởng cho **độ trôi chảy và mức độ phù hợp thống kê với ngữ liệu**, không đảm bảo tính đúng đắn sự kiện; hiện tượng ảo giác (*hallucination*) là hệ quả tự nhiên khi lấy mẫu từ một phân bố $$p_\theta$$ chưa hoàn hảo.
- Quy trình lọc dữ liệu và khử trùng lặp là một phần quan trọng quyết định chất lượng mô hình: hai mô hình giống hệt kiến trúc có thể cho hành vi rất khác nhau nếu huấn luyện trên hai kho dữ liệu khác nhau.
- Định luật mở rộng (Chương 25 và bài **26-07**) liên kết hàm mất mát với số tham số, số token và ngân sách tính toán.

## Điểm chính cần nhớ

- Tiền huấn luyện LLM là tối đa hóa xác suất dự đoán token kế tiếp dưới mặt nạ nhân quả.
- Softmax kết hợp cross-entropy cho đạo hàm đơn giản $$\partial\mathcal{L}/\partial z = p - t$$; cross-entropy bằng tổng entropy và phân kỳ KL (trùng với NLL khi nhãn là *one-hot*).
- *Log-sum-exp* ổn định phép tính softmax, còn biến thể trực tuyến theo khối là nền tảng của FlashAttention (**26-07**).
- `nn.Linear` của PyTorch lưu trọng số kích thước $$(n_{\mathrm{out}}, n_{\mathrm{in}})$$ và tính $$X W^\top$$; AdamW thường áp dụng *weight decay* cho ma trận trọng số 2-D và bỏ qua các tham số chuẩn hóa 1-D.
- Tokenizer dưới từ (BPE, SentencePiece) xác định tập ký hiệu rời rạc của mô hình.
- Độ dài ngữ cảnh $$L$$ quyết định phạm vi chú ý của *attention* và dung lượng KV-cache khi phục vụ.
- Chính kích thước từ vựng $$V$$ lớn khiến chưng cất logit cổ điển trở nên đắt đỏ với LLM (xem tiếp ở bài **26-05**).

## Video tham khảo

- **Xem trước** [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Andrej Karpathy, tiếng Anh. Hướng dẫn xây dựng chi tiết bộ tokenizer BPE, bổ sung trực tiếp cho phần tokenization trong bài.
- [LLM Tokenizers Explained: BPE Encoding, WordPiece and SentencePiece](https://www.youtube.com/watch?v=hL4ZnAWSyuU) — DataMListic, tiếng Anh. So sánh ngắn gọn giữa BPE, WordPiece và SentencePiece trước khi đi sâu vào cài đặt.
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, tiếng Anh. Minh họa hàm mất mát cross-entropy dự đoán ký tự kế tiếp — nền tảng của mục tiêu tiền huấn luyện trước khi áp dụng tokenization dưới từ.
