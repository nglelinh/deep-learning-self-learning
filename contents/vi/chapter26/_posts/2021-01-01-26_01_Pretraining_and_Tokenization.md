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

# Tiền huấn luyện và tokenization

LLM hiện đại được tiền huấn luyện như một **bộ dự đoán token kế**. Mọi thứ sau đó — prompting, tuân thủ chỉ dẫn, công cụ — nằm trên một mô hình mà, cho token $$x_1,\ldots,x_{t-1}$$, xuất phân bố trên token tiếp theo $$x_t$$.

Bài này giữ mức khái niệm. Bạn không cần cả stack huấn luyện; bạn cần mục tiêu, bộ tách token định nghĩa tập ký hiệu, và cửa sổ ngữ cảnh giới hạn những gì mô hình “nhìn thấy.”

## 1. Dự đoán token kế

Gọi $$x = (x_1,\ldots,x_T)$$ là chuỗi token. Mô hình ngôn ngữ nhân quả phân rã

$$p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).$$

Huấn luyện cực tiểu hóa log-likelihood âm trung bình (cross-entropy) trên ngữ liệu:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).$$

Tại mỗi vị trí mạng sinh logit $$z \in \mathbb{R}^{V}$$ ($$V$$ là cỡ từ vựng). Softmax biến chúng thành $$p_\theta(\cdot \mid x_{<t})$$. “Nhãn” chính là token kế quan sát được — không cần người gán nhãn. Đó là lý do việc này là **tự giám sát** theo nghĩa Chương 16, dù loss trông như phân loại thông thường.

Hai hệ quả quan trọng cho phần sau của chương:

- Không gian đầu ra **rất lớn** (thường $$V \sim 32\mathrm{k}$$–$$200\mathrm{k}+$$). Chưng cất tri thức cổ điển khớp softmax 1.000 lớp (Chương 23) không sao chép sang một cách rẻ — xem **26-05**.
- Huấn luyện dùng **teacher forcing**: mô hình luôn điều kiện trên tiền tố *đúng*, không trên mẫu do nó sinh. Lúc giải mã nó phải điều kiện trên token của chính nó. Lệch train/serve này là bình thường; căn chỉnh và mẹo giải mã sống chung với nó chứ không xóa nó.

Phác thảo loss theo kiểu PyTorch (chỉ shape):

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

Phần còn lại của bài là **toán mà tầng cuối thực sự dùng**, rồi tokenizer và cửa sổ. Nếu công thức đã sống ở Chương 02–03 hoặc 10, ta chỉ viết lại phiên bản hình LM.

## 2. Softmax, cross-entropy, và $$\partial\mathcal{L}/\partial z = p - t$$

Tại một vị trí mô hình phát logit $$z \in \mathbb{R}^{V}$$. Softmax là

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.$$

Loss LM trên mục tiêu one-hot $$t$$ (token kế quan sát được) là cross-entropy

$$\mathcal{L} = -\sum_{i=1}^{V} t_i \log p_i = -\log p_{t^*},$$

với $$t^*$$ là chỉ số token đúng. Vi phân qua softmax cho đồng nhất phỏng vấn

$$\frac{\partial \mathcal{L}}{\partial z} = p - t.$$

Đó là cùng phép giản ước Chương 03 chỉ cho CE + softmax trên bộ phân loại nhỏ. LM làm vậy ở mọi bước thời gian, với $$V$$ hàng chục hoặc hàng trăm nghìn.

**CE, KL, và entropy.** Với mọi cặp phân bố rời rạc $$p^*$$ và $$p$$,

$$\mathrm{CE}(p^*, p) = H(p^*) + \mathrm{KL}(p^* \| p).$$

One-hot $$p^*$$ có $$H(p^*) = 0$$, nên CE theo token **chính là** $$\mathrm{KL}(\text{one-hot} \| p)$$ **chính là** log-likelihood âm. Khi mục tiêu là giáo viên *mềm* (chưng cất, **26-05**), $$H(p^*)$$ là hằng đối với học sinh và tín hiệu thêm là KL. Softmax có nhiệt độ ở 26-05 là đồng nhất này với $$p^*$$ phẳng hơn.

## 3. Log-sum-exp và vì sao FlashAttention nói “softmax trực tuyến”

$$e^{z_i}$$ ngây thơ tràn số khi logit lớn. Viết lại ổn định trừ max hàng $$m = \max_j z_j$$:

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \qquad p_i = \exp\big(z_i - \mathrm{LSE}(z)\big).$$

Đó là **log-sum-exp**. Cùng đại số có thể chạy *tăng dần*: nếu bạn đã thấy một tiền tố logit với max chạy $$m$$ và tổng chạy $$s = \sum e^{z-m}$$, rồi một khối mới tới với max $$m'$$, bạn đổi thang

$$s \leftarrow s\, e^{m - m_{\mathrm{new}}} + \sum_{\text{new}} e^{z - m_{\mathrm{new}}}, \quad m_{\mathrm{new}} = \max(m, m').$$

Giữ một tổng có trọng chạy tương ứng của value và bạn có **softmax trực tuyến** — câu chuyện số trong [FlashAttention](https://arxiv.org/abs/2205.14135), không cần CUDA. Bài **26-07** dùng đây làm lý do ma trận $$L\times L$$ không phải hạ xuống HBM.

## 4. Tầng tuyến tính: toán $$XW$$ so với $$W$$ của PyTorch

Một batch trạng thái ẩn $$X \in \mathbb{R}^{B \times d_{\mathrm{in}}}$$ và sách toán viết

$$Y = X W + b, \qquad W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}.$$

`torch.nn.Linear(n_in, n_out)` lưu `weight` với shape **$$(n_{\mathrm{out}}, n_{\mathrm{in}})$$** và tính $$X W^\top + b$$. Cùng ánh xạ affine; ma trận lưu là $$W$$ sách giáo khoa *chuyển vị*. Khi đếm tham số ở **26-07**, dùng $$d_{\mathrm{in}} \times d_{\mathrm{out}}$$ và đừng đếm đôi vì layout. Khi chuyển một bài báo, kiểm tra tác giả vẽ quy ước nào.

## 5. Hàm kích hoạt mà FFN thực sự dùng

Chương 02 đã có hình. Ba hàm liên quan LM:

$$
\begin{aligned}
\mathrm{ReLU}(z) &= \max(0, z), \\
\mathrm{SiLU}(z) = \mathrm{Swish}(z) &= z\,\sigma(z), \\
\mathrm{SwiGLU}(x) &= \big(\mathrm{SiLU}(x W_{\mathrm{gate}}) \odot (x W_{\mathrm{up}})\big) W_{\mathrm{down}}.
\end{aligned}
$$

ReLU là FFN 2017. SiLU là cổng trơn. SwiGLU là *khối* (ba ma trận, cổng Hadamard) mà **26-07** đặt vào sơ đồ decoder. GELU (Chương 02) vẫn xuất hiện ở FFN kiểu GPT cũ; đó không phải lý thuyết thứ ba.

## 6. Adam / AdamW: moment, và tham số nào bị decay

Tiền huấn luyện gần như luôn dùng **AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)), không phải Adam thô ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)). Các moment là những cái ở Chương 10:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
\end{aligned}
$$

Adam áp một bước thích nghi và, nếu bạn cộng L2 vào *loss*, gradient L2 đó cũng bị $$\sqrt{\hat{v}}$$ đổi thang — nên “weight decay” không còn là kéo đều về không. AdamW **tách** decay:

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda \theta_{t-1} \right).$$

**Tham số nào nhận $$\lambda > 0$$?** Quy ước LLM thông thường (Hugging Face `AdamW`, trainer kiểu Llama): decay trên **ma trận trọng số 2-D** (chiếu attention và FFN, thường cả embedding / đầu LM). **Không** decay tensor 1-D: bias (nếu có) và gain RMSNorm / LayerNorm. Những thang đó không phải “trọng số phải co”; decay chúng đánh nhau với bộ chuẩn hóa. Chi tiết ở **10-01**; đoạn này chỉ là checklist LM.

## 7. Vì sao không huấn luyện trên ký tự thô (hay từ thô)

Ký tự làm chuỗi rất dài và lãng phí dung lượng vào chính tả. Từ nguyên vẹn làm từ vựng nổ tung và thất bại với cách viết hiếm hoặc mới. Tokenization **đơn vị con từ** (*subword*) nằm ở giữa: từ thường gặp giữ một token; từ hiếm tách thành mảnh tái sử dụng được.

Hai họ bạn sẽ thực sự gặp:

**Byte Pair Encoding (BPE).** Bắt đầu từ ký tự (hoặc byte). Lặp lại việc hợp nhất cặp kề nhau xuất hiện nhiều nhất. Bảng hợp nhất *chính là* tokenizer. [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) đưa BPE vào dịch máy nơ-ron; tokenizer kiểu GPT-2 là hậu duệ được dùng rộng. [tiktoken](https://github.com/openai/tiktoken) của OpenAI là bản cài đặt kỹ thuật của ý tưởng này, không phải thuật toán mới.

**Unigram / SentencePiece.** Thay vì hợp nhất tham lam, mô hình unigram giữ từ vựng ứng viên lớn rồi loại token làm hại mục tiêu likelihood. [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226) (SentencePiece) là gói thông dụng: huấn luyện từ văn bản thô và có thể xuất từ vựng BPE hoặc unigram. Nhiều LM mở đa ngữ dùng stack này.

Trực giác, không phải công thức huấn luyện:

- Tokenization là **front-end tất định, có mất mát**. LM không thấy các ký tự mà tokenizer đã dính, và không thể bịa một ID token ngoài bảng.
- Cùng một từ tiếng Anh có thể là 1 token ở vocab này và 3 token ở vocab khác. **Chi phí và ngữ cảnh được đo bằng token, không phải từ.**
- Token đặc biệt (`<bos>`, `<eos>`, mốc khuôn chat, thẻ công cụ) là mục từ vựng hạng nhất. Sai lệch một vị trí ở đây lãng phí cửa sổ ngữ cảnh hoặc làm hỏng định dạng hội thoại.

Bảng embedding $$E \in \mathbb{R}^{V \times d}$$ ở Chương 18 chính là tầng 0 của mô hình này: mỗi ID token thành một vector, rồi (ở decoder hiện đại) một phương pháp vị trí như RoPE được áp trong attention (bài tùy chọn **08-99**).

## 8. Cửa sổ ngữ cảnh

Self-attention trong Transformer gốc trộn mọi cặp vị trí trong cửa sổ. Nếu độ dài cửa sổ là $$L$$, một tầng tốn $$O(L^2)$$ ở attention (cộng MLP theo vị trí rẻ hơn). Vậy $$L$$ vừa là giới hạn **năng lực** vừa là giới hạn **tính toán/bộ nhớ**.

Cửa sổ thực sự chặn những gì:

- Mô hình chỉ điều kiện được trên $$L$$ token cuối của prompt + phần sinh đã nối (cộng những gì bạn truy hồi rồi dán vào — RAG, Chương 18-99).
- Tài liệu dài phải cắt, tóm tắt, hoặc truy hồi theo khối. “Mô hình đã đọc PDF 200 trang” thường nghĩa là *một hệ truy hồi nhét các khối được chọn vào $$L$$*.
- Lúc giải mã, key và value của những vị trí đó là thứ tự nhiên để **cache** (bài **26-04**).

Các mở rộng (cửa sổ trượt, grouped-query attention, kéo RoPE siêu dài) đổi hằng số; chúng không bỏ ý rằng mô hình có một băng làm việc hữu hạn.

## 9. Dữ liệu tiền huấn luyện, ngắn gọn

Ngữ liệu tiền huấn luyện trộn văn bản web, sách, mã nguồn, và bản crawl đã lọc. Bạn không cần công thức bí mật để hiểu mục tiêu. Bạn *cần* nhớ:

- Loss thưởng **độ trôi chảy và thống kê ngữ liệu**, không phải chân lý. Ảo giác không phải lỗi riêng; đó là lấy mẫu next-token từ $$p_\theta$$ không hoàn hảo.
- Bộ lọc dữ liệu và khử trùng là một phần của mô hình. Hai LM cùng kiến trúc có thể cư xử khác vì ngữ liệu khác.
- Scaling (Chương 25; phác lúc huấn luyện ở **26-07**) gắn loss với tham số, token, và compute. Bài này giả định đường cong đó tồn tại; không khớp lại nó.

## Điểm chính cần nhớ

- Tiền huấn luyện = cực đại hóa likelihood token kế dưới mask nhân quả.
- Softmax + CE rút thành $$\partial\mathcal{L}/\partial z = p - t$$. CE = entropy + KL; mục tiêu one-hot khiến đó chỉ là NLL.
- Log-sum-exp là softmax ổn định; phiên bản tăng dần là câu chuyện FlashAttention (**26-07**).
- `Linear` của PyTorch lưu $$W$$ dạng $$(n_{\mathrm{out}}, n_{\mathrm{in}})$$ và áp $$X W^\top$$. AdamW decay trọng số 2-D, không decay gain chuẩn hóa.
- Tokenizer định nghĩa bảng chữ cái rời rạc; BPE và SentencePiece là hai câu chuyện thực dụng.
- Độ dài ngữ cảnh $$L$$ là bộ nhớ làm việc của attention — và kích thước KV-cache bạn sẽ trả lúc phục vụ.
- Cùng góc nhìn softmax-trên-$$V$$ khiến tiền huấn luyện đơn giản cũng khiến **chưng cất logit cổ điển đắt** với LM (tiếp ở **26-05**).
