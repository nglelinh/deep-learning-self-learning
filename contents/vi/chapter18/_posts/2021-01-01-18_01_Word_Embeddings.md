---
layout: post
title: 18-01 Nhúng Từ và Biểu diễn Ngôn ngữ
chapter: '18'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter18
---

Phép tính nổi tiếng nhất lịch sử NLP hiện đại không phải là một định lý giải tích phức tạp, mà là một phép cộng trừ tiểu học: $$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$. Làm thế nào một mạng nơ-ron nông chỉ gồm đúng hai ma trận trọng số như **Word2Vec (Skip-gram)** hay phép phân tích ma trận đồng xuất hiện của **GloVe** lại có thể tự động chưng cất ra cả giới tính, số ít/số nhiều lẫn quan hệ thủ đô–quốc gia? Bài học này mổ xẻ trọn vẹn cơ chế toán học và cài đặt của nhúng từ.

## Kiến thức cần nhớ: Nền tảng toán học và kỹ thuật của Nhúng Từ

- **Mô hình Skip-gram và CBOW (Word2Vec):** Dự đoán từ ngữ cảnh quanh từ trung tâm (hoặc ngược lại) bằng tích vô hướng giữa hai ma trận nhúng $$\mathbf{v}_w$$ (đầu vào) và $$\mathbf{u}_c$$ (ngữ cảnh).
- **Lấy mẫu âm (Negative Sampling — SGNS):** Thay phép Softmax cực đắt trên toàn bộ $$|V|$$ từ bằng $$1$$ cặp dương và $$k$$ cặp âm rút theo phân phối $$P_n(w) \propto \text{count}(w)^{3/4}$$, giảm độ phức tạp mỗi bước từ $$O(|V|)$$ xuống $$O(k)$$.
- **GloVe (Global Vectors):** Khai thác thống kê đồng xuất hiện toàn cục bằng hồi quy bình phương có trọng số $$\mathbf{v}_i^\top \mathbf{u}_j + b_i + \tilde{b}_j \approx \ln X_{ij}$$, kết nối trực tiếp phương pháp đếm ma trận với mạng dự đoán.
- **FastText và giới hạn từ đa nghĩa:** Biểu diễn từ bằng tổng các n-gram ký tự để xử lý từ hiếm/sai chính tả, đồng thời chỉ rõ giới hạn cố hữu của nhúng tĩnh (ví dụ từ "đường" trong "đường phố" và "đường ăn" bị gộp chung một vector).

### Giải thích: Tính tay một bước hàm mất mát Skip-gram Negative Sampling (SGNS)

Xét từ trung tâm `mèo` có vector $$\mathbf{v} = (1{,}0,\; 0{,}5)$$, từ ngữ cảnh thật `ngủ` có vector $$\mathbf{u}_{+} = (0{,}8,\; 0{,}4)$$, và 2 mẫu âm ngẫu nhiên `búa` ($$\mathbf{u}_{-,1} = (-0{,}5,\; 0{,}2)$$), `thuế` ($$\mathbf{u}_{-,2} = (0{,}4,\; -0{,}8)$$). Hàm loss SGNS là:
$$\mathcal{L} = -\ln \sigma(\mathbf{u}_+^\top \mathbf{v}) - \sum_{k=1}^{2} \ln \sigma(-\mathbf{u}_{-,k}^\top \mathbf{v}).$$

1. **Tính các tích vô hướng:**
   - Cặp dương: $$\mathbf{u}_+^\top \mathbf{v} = 0{,}8(1{,}0) + 0{,}4(0{,}5) = 1{,}00$$.
   - Mẫu âm 1: $$\mathbf{u}_{-,1}^\top \mathbf{v} = -0{,}5(1{,}0) + 0{,}2(0{,}5) = -0{,}40$$ → đổi dấu thành $$+0{,}40$$.
   - Mẫu âm 2: $$\mathbf{u}_{-,2}^\top \mathbf{v} = 0{,}4(1{,}0) - 0{,}8(0{,}5) = 0{,}00$$ → đổi dấu vẫn là $$0{,}00$$.
2. **Tính hàm Sigmoid và mất mát:**
   - $$\sigma(1{,}00) \approx 0{,}7311 \implies -\ln(0{,}7311) \approx 0{,}3132$$.
   - $$\sigma(0{,}40) \approx 0{,}5987 \implies -\ln(0{,}5987) \approx 0{,}5130$$.
   - $$\sigma(0{,}00) = 0{,}5000 \implies -\ln(0{,}5000) \approx 0{,}6931$$.
   - **Tổng loss:** $$\mathcal{L} \approx 0{,}3132 + 0{,}5130 + 0{,}6931 = 1{,}5193$$.
3. **Hiệu quả tốc độ:** Thay vì tính $$50{,}000$$ tích vô hướng cho Softmax, bước cập nhật này chỉ cần đúng $$3$$ tích vô hướng (nhanh hơn hơn **16.000 lần**) mà vẫn đẩy `mèo` lại gần `ngủ` và tách xa `búa`, `thuế`!

# Nhúng Từ: Biểu diễn Ngôn ngữ trong Không gian Vectơ

![Word2Vec Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Word2vec.png/800px-Word2vec.png)
*Hình ảnh: Minh họa nhúng Word2Vec trong không gian vectơ. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

Nhúng từ (*word embedding*) là một trong những kỹ thuật nền tảng của xử lý ngôn ngữ tự nhiên, thay đổi cách biểu diễn và xử lý văn bản trong các hệ thống học máy. Ý tưởng cốt lõi khá đơn giản: biểu diễn mỗi từ bằng một vectơ dày đặc các số thực (thường có 100–300 chiều) sao cho các từ gần nghĩa có vectơ gần nhau. Biểu diễn liên tục này thay thế các biểu diễn thưa trước đây như mã hóa *one-hot* (vectơ hàng nghìn chiều, toàn số 0 trừ một vị trí) bằng các vectơ gọn hơn, phản ánh quan hệ ngữ nghĩa thông qua cấu trúc hình học trong không gian vectơ.

Để thấy rõ vai trò của nhúng từ, cần xét hạn chế của biểu diễn rời rạc. Các phương pháp truyền thống coi từ là các ký hiệu độc lập — “king”, “queen” và “car” cách đều nhau và không chia sẻ cấu trúc nào. Với từ vựng 50.000 từ, mã hóa *one-hot* tạo ra các vectơ 50.000 chiều trực giao đôi một, không mang thông tin về độ tương tự. Điều này cản trở khả năng tổng quát hóa: nếu mô hình học được mẫu câu “king lives in palace”, nó vẫn phải học lại độc lập với “queen lives in palace” dù hai từ gần nghĩa.

### Từ văn bản → token → one-hot / bag-of-words

Bước đầu tiên luôn là **tokenization** (tách văn bản thành các từ hoặc token), sau đó chuyển sang biểu diễn số.

![Tokenization](/deep-learning-self-learning/img/chapter_img/chapter18/emb_tokenization.jpg)
*Hình: Câu được tách thành các token trước khi mã hóa. (Minh họa từ video Word Embedding / Transformer)*

![Bảng one-hot](/deep-learning-self-learning/img/chapter_img/chapter18/emb_one_hot_table.jpg)
*Hình: Mỗi từ tương ứng một vector one-hot — trực giao, thưa và không mang thông tin ngữ nghĩa. (Minh họa từ video Word Embedding / Transformer)*

![Bag-of-words](/deep-learning-self-learning/img/chapter_img/chapter18/emb_bag_of_words.jpg)
*Hình: Bag-of-words gộp các vector one-hot theo câu nên làm mất thứ tự từ. (Minh họa từ video Word Embedding / Transformer)*

![Hạn chế one-hot / BoW](/deep-learning-self-learning/img/chapter_img/chapter18/emb_onehot_limitations.jpg)
*Hình: Ba hạn chế chính — mất thứ tự từ, biểu diễn thưa với số chiều lớn và thiếu quan hệ ngữ nghĩa. (Minh họa từ video Word Embedding / Transformer)*

Nhúng từ khắc phục vấn đề này bằng cách học biểu diễn liên tục, trong đó các từ tương tự nằm gần nhau. Khoảng cách giữa vectơ “king” và “queen” nhỏ (cùng thuộc nhóm hoàng gia), còn “king” và “car” nằm xa nhau. Hơn nữa, không gian nhúng còn thể hiện các quan hệ loại suy qua phép toán vectơ: hướng từ “man” đến “woman” tương tự hướng từ “king” đến “queen”, phản ánh quan hệ giới tính. Nhờ đó, ta có thể tính loại suy bằng số học vectơ: `king − man + woman ≈ queen`. Cấu trúc này không được lập trình sẵn mà tự hình thành trong quá trình huấn luyện trên văn bản.

![Không gian nhúng 2D](/deep-learning-self-learning/img/chapter_img/chapter18/emb_semantic_space_2d.jpg)
*Hình: Các từ gần nghĩa gom cụm với nhau (vua/nữ hoàng/nam/nữ…); khoảng cách phản ánh quan hệ giữa chúng. (Minh họa từ video Word Embedding / Transformer)*

![Lợi ích vector hóa](/deep-learning-self-learning/img/chapter_img/chapter18/emb_vector_arithmetic_benefits.jpg)
*Hình: Embedding hỗ trợ biểu diễn quan hệ ngữ nghĩa, suy luận từ vựng và học đặc trưng tự động. (Minh họa từ video Word Embedding / Transformer)*

Mục tiêu huấn luyện nhúng từ dựa trên *giả thuyết phân bố* (*distributional hypothesis*) trong ngôn ngữ học: những từ xuất hiện trong các ngữ cảnh tương tự thường có nghĩa tương tự. Nguyên lý này cho phép học biểu diễn không giám sát trực tiếp từ kho ngữ liệu văn bản lớn.

![Ngữ cảnh phân bố quanh “vua”](/deep-learning-self-learning/img/chapter_img/chapter18/emb_distributional_context.jpg)
*Hình: Từ “vua” thường đi cùng ngữ cảnh chiến đấu, nam giới, mạnh mẽ… — các từ đồng xuất hiện trong ngữ cảnh tương tự sẽ có embedding gần nhau. (Minh họa từ video Word Embedding / Transformer)*

Các mô hình như Word2Vec và GloVe học nhúng từ bằng cách dự đoán từ ngữ cảnh từ từ đích (hoặc ngược lại), hoặc phân rã ma trận thống kê đồng xuất hiện. Các vectơ học được lưu giữ thông tin về ngữ nghĩa từ vựng, quy luật cú pháp và một phần tri thức thế giới chỉ từ thống kê đồng xuất hiện mà không cần nhãn thủ công.

Nhúng từ đặt nền móng cho học sâu trong xử lý ngôn ngữ tự nhiên, cho phép mạng nơ-ron tận dụng kho văn bản chưa gán nhãn quy mô lớn để học biểu diễn và chuyển giao sang các tác vụ hạ nguồn. Trong giai đoạn 2013–2018, các bộ nhúng tiền huấn luyện như Word2Vec và GloVe là thành phần tiêu chuẩn trong hầu hết hệ thống NLP. Dù ngày nay nhúng theo ngữ cảnh từ BERT và GPT đã thay thế nhúng tĩnh ở nhiều tác vụ, việc nắm vững nhúng tĩnh vẫn cần thiết để hiểu sự phát triển của học biểu diễn trong NLP, cũng như phục vụ những ứng dụng ưu tiên tính đơn giản và hiệu quả tính toán.

## 2. Nền tảng toán học

Nhúng từ ánh xạ các ký hiệu rời rạc (từ) sang vectơ liên tục sao cho bảo toàn độ tương tự ngữ nghĩa. Cụ thể, với tập từ vựng $$V$$ có kích thước $$|V|$$, ta học ma trận nhúng $$\mathbf{E} \in \mathbb{R}^{d \times |V|}$$, trong đó mỗi cột $$\mathbf{e}_w \in \mathbb{R}^d$$ là vectơ nhúng của từ $$w$$. Số chiều nhúng thường là $$d = 100\text{–}300$$, nhỏ hơn rất nhiều so với kích thước từ vựng (10.000–100.000).

### Word2Vec: Mô hình Skip-gram

Mô hình *skip-gram* dự đoán các từ ngữ cảnh khi biết từ đích, dựa trên giả thuyết phân bố. Với chuỗi từ $$w_1, w_2, \ldots, w_T$$ trong kho ngữ liệu, mục tiêu tối ưu là:

$$\max_\theta \frac{1}{T}\sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t; \theta)$$

trong đó $$c$$ là kích thước cửa sổ ngữ cảnh (thường bằng 5), và $$\theta$$ bao gồm ma trận nhúng đầu vào và trọng số đầu ra.

![Cửa sổ ngữ cảnh](/deep-learning-self-learning/img/chapter_img/chapter18/emb_context_window.jpg)
*Hình: Cửa sổ ngữ cảnh kích thước 1 — từ trung tâm học từ các từ lân cận (và ngược lại trong skip-gram/CBOW). (Minh họa từ video Word Embedding / Transformer)*

![CBOW-style: dự đoán từ giữa](/deep-learning-self-learning/img/chapter_img/chapter18/emb_cbow_predict_word.jpg)
*Hình: Hai từ ngữ cảnh (one-hot) đi qua lớp ẩn nhỏ để dự đoán từ đích ở giữa. (Minh họa từ video Word Embedding / Transformer)*

![One-hot → projection ẩn](/deep-learning-self-learning/img/chapter_img/chapter18/emb_onehot_to_hidden.jpg)
*Hình: Biểu diễn chiều từ vựng (ví dụ 5) được nén xuống chiều nhúng (ví dụ 3). (Minh họa từ video Word Embedding / Transformer)*

![Ma trận W1, W2 khi huấn luyện](/deep-learning-self-learning/img/chapter_img/chapter18/emb_w1_w2_training.jpg)
*Hình: Quy trình đầy đủ: one-hot × W1 → lớp ẩn → × W2 → logit các từ; so sánh với nhãn thật để cập nhật trọng số. (Minh họa từ video Word Embedding / Transformer)*

![Embedding matrix](/deep-learning-self-learning/img/chapter_img/chapter18/emb_embedding_matrix.jpg)
*Hình: Ma trận nhúng $$|V|\times d$$ (ví dụ $$5\times 3$$) — mỗi hàng hoặc cột là vector đại diện cho một từ. (Minh họa từ video Word Embedding / Transformer)*

Xác suất có điều kiện được tính qua hàm softmax:

$$p(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}_w^T \mathbf{v}_{w_I})}$$

trong đó $$\mathbf{v}_{w_I}$$ là vectơ nhúng đầu vào của từ $$w_I$$ và $$\mathbf{v}_{w_O}$$ là vectơ nhúng đầu ra của từ $$w_O$$. Vì mẫu số phải tính tổng trên toàn bộ từ vựng nên chi phí tính toán rất cao, đòi hỏi các phương pháp xấp xỉ.

**Lấy mẫu âm** (*negative sampling*) xấp xỉ hàm softmax bằng cách chỉ lấy một vài từ âm thay vì tính tổng trên cả từ vựng:

$$\log \sigma(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i}^T \mathbf{v}_{w_I})]$$

trong đó $$\sigma$$ là hàm sigmoid, $$k$$ là số mẫu âm (thường từ 5 đến 20), và $$P_n(w)$$ là phân bố nhiễu (thường lấy phân bố unigram lũy thừa 3/4 để tăng xác suất chọn các từ hiếm). Cách này chuyển bài toán phân loại đa lớp trên toàn bộ từ vựng thành $$k+1$$ bài toán phân loại nhị phân, giúp huấn luyện khả thi với từ vựng lớn.

Một tính chất đáng chú ý của Word2Vec là các quan hệ ngữ nghĩa thường tương ứng với phép tịnh tiến tuyến tính trong không gian vectơ:

$$\mathbf{e}_{\text{queen}} \approx \mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}}$$

$$\mathbf{e}_{\text{Paris}} \approx \mathbf{e}_{\text{France}} - \mathbf{e}_{\text{Germany}} + \mathbf{e}_{\text{Berlin}}$$

Các quan hệ loại suy này không được giám sát trực tiếp mà xuất hiện tự nhiên từ giả thuyết phân bố: “king” và “queen” chia sẻ nhiều ngữ cảnh về hoàng gia (ngai vàng, cung điện), còn “king” và “man” chia sẻ ngữ cảnh giới tính, tạo nên cấu trúc hình học nhất quán.

### GloVe: Global Vectors

GloVe đi theo hướng khác bằng cách phân rã trực tiếp ma trận thống kê đồng xuất hiện toàn cục. Gọi $$X_{ij}$$ là số lần từ $$j$$ xuất hiện trong ngữ cảnh của từ $$i$$. GloVe tối thiểu hóa hàm mục tiêu:

$$J = \sum_{i,j=1}^{|V|} f(X_{ij})(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

trong đó $$\mathbf{w}_i$$ và $$\tilde{\mathbf{w}}_j$$ lần lượt là vectơ nhúng của từ và từ ngữ cảnh, $$b_i, \tilde{b}_j$$ là các hệ số điều chỉnh (*bias*), còn $$f(X_{ij})$$ là hàm trọng số:

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

Hàm trọng số này giới hạn ảnh hưởng của các cặp từ đồng xuất hiện quá thường xuyên (như từ dừng) và gán trọng số nhỏ hơn cho các cặp hiếm. Nhờ kết hợp thống kê đồng xuất hiện toàn kho ngữ liệu với ngữ cảnh cục bộ, GloVe tạo ra nhúng từ có chất lượng ngang bằng hoặc cao hơn Word2Vec trong nhiều tác vụ.

## 3. Ví dụ và trực giác

Xét bài toán học nhúng cho một tập từ vựng nhỏ: `{cat, dog, car, truck, animal, vehicle}`. Ban đầu, các vectơ được khởi tạo ngẫu nhiên. Khi duyệt qua kho văn bản:

- `"The cat is an animal"` → `"cat"` và `"animal"` đồng xuất hiện.
- `"The dog is an animal"` → `"dog"` và `"animal"` đồng xuất hiện.
- `"The car is a vehicle"` → `"car"` và `"vehicle"` đồng xuất hiện.
- `"The truck is a vehicle"` → `"truck"` và `"vehicle"` đồng xuất hiện.

Mô hình dần điều chỉnh các vectơ sao cho:
- `"cat"` và `"dog"` nằm gần nhau (cùng xuất hiện với `"animal"`).
- `"car"` và `"truck"` nằm gần nhau (cùng xuất hiện với `"vehicle"`).
- `"cat"` và `"car"` nằm xa nhau (xuất hiện trong các ngữ cảnh khác biệt).

Sau khi huấn luyện trên đủ dữ liệu, không gian nhúng 2D có thể sắp xếp tương tự sơ đồ sau:

```
     animal
        ↑
    dog • cat
        |
    ----+---- 
        |
  truck • car
        ↓
     vehicle
```

Các nhóm ngữ nghĩa (động vật và phương tiện) tách cụm rõ rệt, và trong từng nhóm các từ gần nghĩa nằm cạnh nhau. Nhờ đó ta có các quan hệ hình học như:

- `"cat" − "animal" ≈ "dog" − "animal"` (cùng biểu diễn quan hệ từ phạm trù chung đến thực thể cụ thể).
- `"car" + "vehicle" ≈ "truck"` (nằm trong cùng cụm phương tiện).

Cấu trúc hình học này giúp mô hình tổng quát hóa: thông tin học được về `"cat"` có thể chuyển giao sang `"dog"` thông qua độ tương tự giữa hai vectơ.

## 4. Mã minh họa

Ví dụ cài đặt Word2Vec Skip-gram với lấy mẫu âm:

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class Word2VecSkipGram(nn.Module):
    """
    Skip-gram Word2Vec with negative sampling.
    
    Learns word embeddings by predicting context words from target words.
    Uses negative sampling to make training efficient.
    """
    
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        
        # Input and output embeddings
        # Input: embeddings used when word is target
        # Output: embeddings used when word is context
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        self.input_embeddings.weight.data.uniform_(-0.5/embedding_dim, 
                                                   0.5/embedding_dim)
        self.output_embeddings.weight.data.zero_()
    
    def forward(self, target_words, context_words, negative_words):
        """
        target_words: (batch,) target word indices
        context_words: (batch,) context word indices (positive examples)
        negative_words: (batch, k) negative samples
        
        Returns negative log-likelihood
        """
        # Get embeddings
        target_embeds = self.input_embeddings(target_words)  # (batch, emb_dim)
        context_embeds = self.output_embeddings(context_words)  # (batch, emb_dim)
        neg_embeds = self.output_embeddings(negative_words)  # (batch, k, emb_dim)
        
        # Positive scores (target-context similarity)
        pos_scores = (target_embeds * context_embeds).sum(dim=1)  # (batch,)
        pos_loss = -torch.log(torch.sigmoid(pos_scores)).mean()
        
        # Negative scores (target-negative dissimilarity)
        neg_scores = torch.bmm(neg_embeds, target_embeds.unsqueeze(2)).squeeze(2)  # (batch, k)
        neg_loss = -torch.log(torch.sigmoid(-neg_scores)).sum(dim=1).mean()
        
        return pos_loss + neg_loss

# Prepare training data
print("="*70)
print("Training Word2Vec Embeddings")
print("="*70)

# Simple corpus for demonstration
corpus = """
the cat sat on the mat .
the dog sat on the rug .
the cat and the dog are animals .
the car is a vehicle .
the truck is a vehicle .
cats and dogs are pets .
cars and trucks are vehicles .
""".lower().split()

# Build vocabulary
vocab = list(set(corpus))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Corpus: {len(corpus)} words")
print(f"Vocabulary: {vocab_size} unique words")
print(f"Sample vocab: {vocab[:10]}")

# Generate training pairs
def generate_training_data(corpus, word_to_idx, window_size=2):
    """Generate (target, context) pairs"""
    pairs = []
    for i, word in enumerate(corpus):
        target_idx = word_to_idx[word]
        
        # Get context (words within window)
        context_start = max(0, i - window_size)
        context_end = min(len(corpus), i + window_size + 1)
        
        for j in range(context_start, context_end):
            if j != i:  # Don't pair with self
                context_idx = word_to_idx[corpus[j]]
                pairs.append((target_idx, context_idx))
    
    return pairs

pairs = generate_training_data(corpus, word_to_idx, window_size=2)
print(f"\nGenerated {len(pairs)} training pairs")
print(f"Sample pairs: {pairs[:5]}")

# Train Word2Vec
model = Word2VecSkipGram(vocab_size, embedding_dim=10)  # Small dim for demo
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTraining embeddings...")

for epoch in range(500):
    epoch_loss = 0
    
    for target_idx, context_idx in pairs:
        # Sample negatives
        neg_indices = np.random.choice(vocab_size, size=5, replace=False)
        
        # To tensors
        target = torch.LongTensor([target_idx])
        context = torch.LongTensor([context_idx])
        negatives = torch.LongTensor(neg_indices).unsqueeze(0)
        
        # Forward
        loss = model(target, context, negatives)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}: Loss = {epoch_loss/len(pairs):.4f}")

# Analyze learned embeddings
print("\n" + "="*70)
print("Analyzing Learned Embeddings")
print("="*70)

model.eval()
embeddings = model.input_embeddings.weight.data.numpy()

# Find nearest neighbors
def find_nearest(word, embeddings, word_to_idx, idx_to_word, k=3):
    """Find k nearest words to query word"""
    word_idx = word_to_idx[word]
    word_vec = embeddings[word_idx]
    
    # Compute cosine similarities
    similarities = embeddings @ word_vec / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vec) + 1e-10
    )
    
    # Get top k (excluding self)
    top_k = similarities.argsort()[-k-1:-1][::-1]
    
    return [(idx_to_word[i], similarities[i]) for i in top_k]

# Test semantic similarity
test_words = ['cat', 'dog', 'car']
for word in test_words:
    if word in word_to_idx:
        neighbors = find_nearest(word, embeddings, word_to_idx, idx_to_word)
        print(f"\nNearest to '{word}': {neighbors}")

print("\nEmbeddings learned semantic relationships from co-occurrence patterns!")
print("Similar words (cat/dog, car/truck) have similar embeddings.")
```

## 5. Mở rộng: Nhúng trong hệ thống gợi ý (Two-Tower)

Nguyên lý “vector gần nhau biểu thị mức độ tương tự” không chỉ áp dụng cho văn bản. Trong **hệ thống gợi ý** (luồng video, thương mại điện tử), ta học hai tập biểu diễn:

- **User embedding** $$\mathbf{u} \in \mathbb{R}^d$$ — biểu diễn sở thích của người dùng.
- **Item embedding** $$\mathbf{v} \in \mathbb{R}^d$$ — biểu diễn đặc trưng của nội dung hoặc sản phẩm.

Mức độ phù hợp thường được tính bằng tích vô hướng hoặc độ tương tự cosine: $$s(u,i) = \mathbf{u}^\top \mathbf{v}$$.

![Mạng sâu sinh embedding](/deep-learning-self-learning/img/chapter_img/chapter18/recsys_deep_ranking_nn.jpg)
*Hình: Mạng feedforward dùng để học biểu diễn và xếp hạng ứng viên. (Minh họa từ video giải thích recommendation system)*

![Embedding chiều 128](/deep-learning-self-learning/img/chapter_img/chapter18/recsys_embedding_128d.jpg)
*Hình: Đầu ra mạng được nén thành vector dày đặc (ví dụ $$1\times 128$$) đại diện cho sở thích người dùng hoặc đặc trưng item. (Minh họa từ video giải thích recommendation system)*

Trong **mô hình hai tháp (*two-tower*)**, một tháp mã hóa người dùng và một tháp mã hóa nội dung/sản phẩm. Mô hình được huấn luyện sao cho các cặp `(user, item)` đã tương tác nằm gần nhau trong không gian nhúng, còn các cặp âm nằm xa nhau. Ưu điểm lớn khi triển khai quy mô lớn là có thể tính sẵn toàn bộ *item embedding*; khi phục vụ trực tuyến (*serving*), hệ thống chỉ cần mã hóa người dùng rồi tìm láng giềng gần nhất xấp xỉ (ANN) trên hàng tỷ mục.

![Two-tower: khớp hai vector 128-d](/deep-learning-self-learning/img/chapter_img/chapter18/recsys_two_tower_matching.jpg)
*Hình: So khớp hai vector nhúng cùng số chiều (user và item) — cơ chế cốt lõi của truy xuất two-tower. (Minh họa từ video giải thích recommendation system)*

So với Word2Vec, thay vì ngữ cảnh từ xung quanh, tín hiệu huấn luyện ở đây là **hành vi tương tác** (nhấp chuột, xem hết video, mua hàng). Giả thuyết phân bố trở thành: người dùng hoặc sản phẩm xuất hiện trong các ngữ cảnh tương tác tương tự sẽ có vectơ nhúng gần nhau.

## 6. Khái niệm liên quan

Nhúng từ bắt nguồn từ ngữ nghĩa học phân bố (*distributional semantics*) — quan điểm cho rằng nghĩa của một từ được xác định bởi ngữ cảnh sử dụng nó. Việc học vectơ sao cho các từ có ngữ cảnh tương tự nằm gần nhau chính là cách hiện thực hóa tính toán cho lý thuyết ngôn ngữ học này.

Xét về mặt biểu diễn dữ liệu, nhúng từ cũng gần với các kỹ thuật giảm chiều như PCA hoặc autoencoder: vectơ *one-hot* có số chiều bằng kích thước từ vựng được nén xuống vectơ dày đặc chiều thấp nhưng vẫn giữ được cấu trúc ngữ nghĩa. Điều này cho thấy số chiều nội tại cần thiết để biểu diễn nghĩa của từ nhỏ hơn rất nhiều so với tổng số từ.

Quá trình phát triển từ nhúng tĩnh (Word2Vec, GloVe) sang nhúng theo ngữ cảnh (ELMo, BERT) khắc phục hạn chế của từ đa nghĩa. Nhúng tĩnh chỉ gán một vectơ duy nhất cho mỗi từ, nên “bank” (ngân hàng) và “bank” (bờ sông) có cùng biểu diễn. Ngược lại, nhúng theo ngữ cảnh sinh ra vectơ khác nhau tùy theo câu chứa nó.

![Không gian vector: trái cây vs công ty](/deep-learning-self-learning/img/chapter_img/chapter18/sa_vector_space_fruit_vs_companies.jpg)
*Hình: Cụm “táo / chuối / cam…” tách khỏi cụm “Google / Microsoft / DeepSeek…”. (Minh họa từ video Transformer / Self-Attention nhập môn)*

![Sau self-attention: “Apple” theo nghĩa công ty](/deep-learning-self-learning/img/chapter_img/chapter18/sa_contextual_apple_company_vector.jpg)
*Hình: Khi câu nhắc đến điện thoại và màu sắc, vector “Apple” dịch chuyển về phía cụm các hãng công nghệ. (Minh họa từ video Transformer / Self-Attention nhập môn)*

Sự chuyển dịch này đánh dấu bước tiến từ biểu diễn cố định ở cấp độ từ sang mô hình hóa ý nghĩa phụ thuộc ngữ cảnh của ngôn ngữ.

## 7. Các bài báo nền tảng

**["Efficient Estimation of Word Representations in Vector Space" (2013)](https://arxiv.org/abs/1301.3781)**  
*Tác giả*: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
Word2Vec giới thiệu hai kiến trúc hiệu quả (skip-gram và CBOW) cùng kỹ thuật lấy mẫu âm để học nhúng từ trên kho dữ liệu hàng tỷ từ. Bài báo cũng chỉ ra khả năng giải các phép loại suy ngữ nghĩa bằng số học vectơ, đưa nhúng từ trở thành thành phần phổ biến trong NLP.

**["GloVe: Global Vectors for Word Representation" (2014)](https://aclanthology.org/D14-1162/)**  
*Tác giả*: Jeffrey Pennington, Richard Socher, Christopher Manning  
GloVe kết hợp ưu điểm của phân rã ma trận toàn cục và cửa sổ ngữ cảnh cục bộ bằng cách phân rã ma trận đồng xuất hiện của từ. Phương pháp đạt chất lượng ngang bằng hoặc tốt hơn Word2Vec, đồng thời gắn trực tiếp các phép toán vectơ với tỷ số xác suất đồng xuất hiện.

**["Deep contextualized word representations" (2018)](https://arxiv.org/abs/1802.05365)**  
*Tác giả*: Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer  
ELMo giới thiệu biểu diễn từ theo ngữ cảnh dựa trên mạng LSTM hai chiều sâu, khắc phục hạn chế của nhúng tĩnh đối với từ đa nghĩa. Bài báo cũng chỉ ra rằng các tầng khác nhau học các loại thông tin khác nhau (tầng thấp thiên về cú pháp, tầng cao thiên về ngữ nghĩa), mở đường cho BERT và các mô hình Transformer sau này.

## Bẫy thường gặp và mẹo thực hành

Sử dụng nhúng từ tiền huấn luyện mà không xử lý sự lệch từ vựng sẽ dẫn đến vấn đề từ ngoài từ vựng (*out-of-vocabulary*, OOV). Khi dữ liệu tác vụ chứa các từ không có trong bộ nhúng tiền huấn luyện, có thể khắc phục bằng cách: dùng biểu diễn dưới từ (*subword* như BPE, WordPiece), khởi tạo từ mới dựa trên trung bình các từ liên quan (chẳng hạn xấp xỉ “coronavirus” từ “virus” và “corona”), hoặc tinh chỉnh lại bộ nhúng trên dữ liệu thuộc miền chuyên biệt.

## Điểm chính cần nhớ

Nhúng từ biểu diễn từ bằng các vectơ thực dày đặc, trong đó độ tương tự ngữ nghĩa tương ứng với sự gần gũi về mặt hình học, giúp mạng nơ-ron tổng quát hóa giữa các từ gần nghĩa. Word2Vec (skip-gram với lấy mẫu âm) và GloVe (phân rã ma trận đồng xuất hiện) đều học từ văn bản chưa gán nhãn dựa trên giả thuyết phân bố. Không gian nhúng học được thể hiện các tính chất như gom cụm ngữ nghĩa và phép toán loại suy (`king − man + woman ≈ queen`) mà không cần giám sát tường minh. Cùng cấu trúc hình học đó cũng được ứng dụng rộng rãi trong **hệ thống gợi ý** qua *user/item embedding* và mô hình *two-tower*. Trong khi các mô hình nhúng theo ngữ cảnh như BERT giải quyết tốt hiện tượng đa nghĩa, nhúng tĩnh vẫn hữu ích nhờ tốc độ và tính dễ diễn giải, đồng thời là nền tảng để hiểu học biểu diễn trong NLP và nhiều lĩnh vực khác.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến):

- [Transformer là gì? Self-Attention siêu dễ hiểu (Phần 1)](https://www.facebook.com/reel/930207546288223)
- [Word Embedding và Transformer hiểu ngôn ngữ](https://www.facebook.com/reel/1501843101324752)
- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)

## Video tham khảo

- **Xem trước** [Word Embedding and Word2Vec, Clearly Explained!!!](https://www.youtube.com/watch?v=viZrOnJclY0) — StatQuest with Josh Starmer, tiếng Anh. Giải thích rõ ràng skip-gram và giả thuyết phân bố — trọng tâm của bài.
- [The Illustrated Word2vec - A Gentle Intro to Word Embeddings in Machine Learning](https://www.youtube.com/watch?v=ISPId9Lhc1g) — Jay Alammar, tiếng Anh. Trình bày trực quan không gian vectơ từ và các ví dụ loại suy.
- [Lecture 2 | Word Vector Representations: word2vec](https://www.youtube.com/watch?v=ERibwqs9p38) — Stanford University School of Engineering, tiếng Anh. Đi sâu vào hàm mục tiêu softmax và kỹ thuật lấy mẫu âm như phần nền tảng toán học trong bài.
