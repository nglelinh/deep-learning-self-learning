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

# Nhúng Từ: Biểu diễn Ngôn ngữ trong Không gian Vectơ

![Word2Vec Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Word2vec.png/800px-Word2vec.png)
*Hình ảnh: Minh họa nhúng Word2Vec trong không gian vectơ. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm
Nhúng từ (*word embedding*) là một trong những đổi mới nền tảng nhất của xử lý ngôn ngữ tự nhiên, thay đổi cách ta biểu diễn và xử lý văn bản trong hệ thống học máy. Ý tưởng cốt lõi thanh lịch nhưng có tác động sâu: biểu diễn mỗi từ bằng một vectơ dày đặc các số thực (thường 100–300 chiều) sao cho các từ gần nghĩa có vectơ tương tự. Biểu diễn vectơ liên tục này thay thế các biểu diễn thưa cũ như mã hóa one-hot (vectơ hàng nghìn chiều, toàn số 0 trừ một vị trí) bằng các nhúng gọn, mang nghĩa, nắm bắt quan hệ ngữ nghĩa qua thuộc tính hình học trong không gian vectơ.

Để hiểu vì sao nhúng làm thay đổi NLP, cần thấy hạn chế của biểu diễn từ rời rạc. Các cách tiếp cận truyền thống coi từ là ký hiệu nguyên tử — “king”, “queen” và “car” cách đều nhau, không chia sẻ cấu trúc. One-hot biểu diễn từ vựng 50.000 từ bằng vectơ 50.000 chiều trực giao, không có khái niệm tương tự. Điều này làm việc học khó vì mô hình không thể tổng quát hóa từ “king lives in palace” sang “queen lives in palace” — nó phải học sự kiện về “queen” độc lập dù gần nghĩa với “king”.

### Từ văn bản → token → one-hot / bag-of-words

Bước đầu luôn là **tokenization** (tách từ/token), rồi gán biểu diễn số.

![Tokenization](/deep-learning-self-learning/img/chapter_img/chapter18/emb_tokenization.jpg)
*Hình: Câu được tách thành các token trước khi mã hóa. (Minh họa từ video Word Embedding / Transformer)*

![Bảng one-hot](/deep-learning-self-learning/img/chapter_img/chapter18/emb_one_hot_table.jpg)
*Hình: Mỗi từ một vector one-hot — trực giao, thưa, không mang nghĩa. (Minh họa từ video Word Embedding / Transformer)*

![Bag-of-words](/deep-learning-self-learning/img/chapter_img/chapter18/emb_bag_of_words.jpg)
*Hình: Bag-of-words gộp one-hot theo câu — mất thứ tự từ. (Minh họa từ video Word Embedding / Transformer)*

![Hạn chế one-hot / BoW](/deep-learning-self-learning/img/chapter_img/chapter18/emb_onehot_limitations.jpg)
*Hình: Ba hạn chế chính — không thứ tự, thưa + nổ chiều, không quan hệ ngữ nghĩa. (Minh họa từ video Word Embedding / Transformer)*

Nhúng từ giải quyết điều này bằng cách học biểu diễn liên tục nơi các từ tương tự tụ cụm. Khoảng cách giữa vectơ “king” và “queen” nhỏ (cùng hoàng tộc), trong khi “king” và “car” xa nhau (không liên quan ngữ nghĩa). Đáng chú ý hơn, không gian nhúng thể hiện quan hệ loại suy qua phép toán vectơ: vectơ từ “man” đến “woman” tương tự vectơ từ “king” đến “queen”, nắm bắt quan hệ giới. Ta có thể giải loại suy bằng số học vectơ đơn giản: king − man + woman ≈ queen. Cấu trúc nổi này không được lập trình tường minh mà nảy sinh từ huấn luyện trên văn bản, cho thấy nhúng nắm bắt các quy luật ngữ nghĩa sâu.

![Không gian nhúng 2D](/deep-learning-self-learning/img/chapter_img/chapter18/emb_semantic_space_2d.jpg)
*Hình: Từ gần nghĩa tụ cụm (vua/nữ hoàng/nam/nữ…); khoảng cách phản ánh quan hệ. (Minh họa từ video Word Embedding / Transformer)*

![Lợi ích vector hóa](/deep-learning-self-learning/img/chapter_img/chapter18/emb_vector_arithmetic_benefits.jpg)
*Hình: Embedding cho phép quan hệ ngữ nghĩa, suy luận từ vựng, và học đặc trưng tự động. (Minh họa từ video Word Embedding / Transformer)*

Mục tiêu huấn luyện để học nhúng được xây dựng thanh lịch qua *giả thuyết phân bố* (*distributional hypothesis*) trong ngôn ngữ học: các từ xuất hiện trong ngữ cảnh tương tự có nghĩa tương tự. Nguyên lý đơn giản này cho phép học không giám sát từ kho ngữ liệu văn bản khổng lồ.

![Ngữ cảnh phân bố quanh “vua”](/deep-learning-self-learning/img/chapter_img/chapter18/emb_distributional_context.jpg)
*Hình: Từ “vua” thường đi với ngữ cảnh chiến đấu, nam giới, mạnh mẽ… — đồng xuất hiện → embedding gần. (Minh họa từ video Word Embedding / Transformer)*

Các mô hình như Word2Vec và GloVe học nhúng bằng cách dự đoán từ ngữ cảnh từ từ đích (hoặc ngược lại), hoặc bằng cách phân rã thống kê đồng xuất hiện. Các vectơ thu được mã hóa ngữ nghĩa từ vựng, mẫu cú pháp, thậm chí một phần tri thức thế giới — tất cả khám phá thuần từ mẫu đồng xuất hiện trong văn bản mà không cần dữ liệu có nhãn.

Tác động của nhúng từ đối với NLP không thể phóng đại. Chúng đặt nền cho cuộc cách mạng học sâu trong xử lý ngôn ngữ, cho phép mạng neuron tận dụng văn bản không nhãn khổng lồ để học biểu diễn rồi chuyển giao sang tác vụ hạ nguồn. Nhúng tiền huấn luyện như Word2Vec và GloVe trở thành thành phần chuẩn trong hầu hết hệ thống NLP giai đoạn 2013–2018. Dù nhúng ngữ cảnh hiện đại từ BERT và GPT đã phần lớn thay thế nhúng từ tĩnh cho nhiều tác vụ, hiểu nhúng tĩnh vẫn then chốt để nắm lộ trình tiến hóa của học biểu diễn trong NLP và cho các ứng dụng nơi tính đơn giản và hiệu quả của chúng vẫn là lợi thế.

## 2. Nền tảng toán học
Nhúng từ ánh xạ ký hiệu rời rạc (từ) sang vectơ liên tục theo cách nắm bắt độ tương tự ngữ nghĩa. Hình thức hóa: ta có từ vựng $$V$$ kích thước $$|V|$$ và học ma trận nhúng $$\mathbf{E} \in \mathbb{R}^{d \times |V|}$$ trong đó cột $$\mathbf{e}_w \in \mathbb{R}^d$$ là nhúng của từ $$w$$. Chiều nhúng điển hình $$d = 100\text{-}300$$, nhỏ hơn nhiều so với kích thước từ vựng (10.000–100.000).

### Word2Vec: Mô hình Skip-gram

Mô hình skip-gram dự đoán từ ngữ cảnh cho trước từ đích, dựa trên giả thuyết phân bố. Với kho ngữ liệu các từ $$w_1, w_2, \ldots, w_T$$, mục tiêu là:

$$\max_\theta \frac{1}{T}\sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t; \theta)$$

trong đó $$c$$ là kích thước cửa sổ ngữ cảnh (thường 5), và $$\theta$$ gồm ma trận nhúng cùng trọng số đầu ra.

![Cửa sổ ngữ cảnh](/deep-learning-self-learning/img/chapter_img/chapter18/emb_context_window.jpg)
*Hình: Window size = 1 — từ tâm học từ láng giềng (và ngược lại trong skip-gram/CBOW). (Minh họa từ video Word Embedding / Transformer)*

![CBOW-style: dự đoán từ giữa](/deep-learning-self-learning/img/chapter_img/chapter18/emb_cbow_predict_word.jpg)
*Hình: Hai từ ngữ cảnh (one-hot) → lớp ẩn nhỏ → dự đoán từ đích. (Minh họa từ video Word Embedding / Transformer)*

![One-hot → projection ẩn](/deep-learning-self-learning/img/chapter_img/chapter18/emb_onehot_to_hidden.jpg)
*Hình: Chiều vocab (ví dụ 5) nén xuống embedding dim (ví dụ 3). (Minh họa từ video Word Embedding / Transformer)*

![Ma trận W1, W2 khi huấn luyện](/deep-learning-self-learning/img/chapter_img/chapter18/emb_w1_w2_training.jpg)
*Hình: Pipeline đầy đủ: one-hot × W1 → ẩn → × W2 → logit từ; so với nhãn thật để cập nhật. (Minh họa từ video Word Embedding / Transformer)*

![Embedding matrix](/deep-learning-self-learning/img/chapter_img/chapter18/emb_embedding_matrix.jpg)
*Hình: Ma trận nhúng $$|V|\times d$$ (ví dụ $$5\times 3$$) — mỗi hàng/cột là vector một từ. (Minh họa từ video Word Embedding / Transformer)*

Xác suất có điều kiện dùng softmax:

$$p(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}_w^T \mathbf{v}_{w_I})}$$

trong đó $$\mathbf{v}_{w_I}$$ là nhúng đầu vào của từ $$w_I$$ và $$\mathbf{v}_{w_O}$$ là nhúng đầu ra của $$w_O$$. Tính softmax này đòi hỏi tổng trên toàn từ vựng (đắt!), thúc đẩy các xấp xỉ.

**Lấy mẫu âm** (*negative sampling*) xấp xỉ softmax bằng cách lấy mẫu vài ví dụ âm thay vì tổng trên mọi từ:

$$\log \sigma(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i}^T \mathbf{v}_{w_I})]$$

trong đó $$\sigma$$ là sigmoid, $$k$$ là số mẫu âm (thường 5–20), và $$P_n(w)$$ là phân bố nhiễu (thường unigram lũy thừa 3/4 để lấy mẫu quá các từ hiếm). Điều này chuyển bài toán đa lớp thành $$k+1$$ phân loại nhị phân, khả thi ngay cả với từ vựng lớn.

Tính chất đáng chú ý của nhúng Word2Vec là quan hệ ngữ nghĩa được mã hóa như phép tịnh tiến tuyến tính trong không gian vectơ:

$$\mathbf{e}_{\text{queen}} \approx \mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}}$$

$$\mathbf{e}_{\text{Paris}} \approx \mathbf{e}_{\text{France}} - \mathbf{e}_{\text{Germany}} + \mathbf{e}_{\text{Berlin}}$$

Các loại suy này không được huấn luyện tường minh mà nảy sinh từ giả thuyết phân bố: “king” và “queen” xuất hiện trong ngữ cảnh tương tự (hoàng gia, ngai vàng, cung điện), cũng như “king” và “man” (ngữ cảnh giới), tạo hình học vectơ phản ánh các mẫu ngữ nghĩa này.

### GloVe: Global Vectors

GloVe tiếp cận khác, phân rã trực tiếp thống kê đồng xuất hiện từ. Gọi $$X_{ij}$$ là số lần từ $$j$$ xuất hiện trong ngữ cảnh của từ $$i$$. GloVe tối thiểu hóa:

$$J = \sum_{i,j=1}^{|V|} f(X_{ij})(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

trong đó $$\mathbf{w}_i$$ và $$\tilde{\mathbf{w}}_j$$ là nhúng từ và ngữ cảnh, $$b_i, \tilde{b}_j$$ là độ lệch, và $$f(X_{ij})$$ là hàm trọng số:

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

Hàm này giảm trọng số các đồng xuất hiện thường xuyên (đã được biểu diễn tốt) và giới hạn ảnh hưởng của các cặp rất thường. GloVe kết hợp lợi ích của phương pháp phân rã ma trận toàn cục (tận dụng thống kê toàn kho ngữ liệu) với phương pháp cửa sổ ngữ cảnh cục bộ (Word2Vec), thường sinh nhúng cạnh tranh hoặc vượt Word2Vec.

## 3. Ví dụ / Trực giác

Hãy tưởng tượng học nhúng cho từ vựng nhỏ: {cat, dog, car, truck, animal, vehicle}. Ban đầu, vectơ ngẫu nhiên. Khi xử lý văn bản:

"The cat is an animal" → "cat" và "animal" đồng xuất hiện  
"The dog is an animal" → "dog" và "animal" đồng xuất hiện  
"The car is a vehicle" → "car" và "vehicle" đồng xuất hiện  
"The truck is a vehicle" → "truck" và "vehicle" đồng xuất hiện

Mô hình điều chỉnh vectơ sao cho:
- "cat" và "dog" trở nên gần nhau (cùng xuất hiện với "animal")
- "car" và "truck" trở nên gần nhau (cùng xuất hiện với "vehicle")
- "cat" và "car" giữ khoảng cách (xuất hiện trong ngữ cảnh khác)

Sau khi thấy đủ văn bản, không gian nhúng 2D có thể tổ chức như:

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

Các phạm trù ngữ nghĩa (động vật vs phương tiện) tụ cụm, và trong mỗi phạm trù các mục tương tự nằm gần nhau. Ta có thể tính:

"cat" − "animal" ≈ "dog" − "animal" (cùng chỉ từ phạm trù đến thể hiện)  
"car" + "vehicle" ≈ "truck" (phạm trù + tương tự cho mục tương tự)

Cấu trúc hình học này cho phép tổng quát hóa: nếu mô hình học sự kiện về "cat", nó có thể chuyển giao sang "dog" qua độ tương tự vectơ.

## 4. Mã minh họa
Cài đặt Word2Vec đầy đủ:

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
        neg_scores = torch.bmm(neg_embeds, target_embeds.unsqueeze(2)).squeeze()  # (batch, k)
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

## 5. Mở rộng: nhúng trong hệ thống gợi ý (two-tower)

Cùng ý tưởng “vector gần nhau = tương tự” không chỉ áp dụng cho từ. Trong **hệ thống gợi ý** (feed video, e-commerce), ta học:

- **User embedding** $$\mathbf{u} \in \mathbb{R}^d$$ — “ảnh chụp” sở thích người dùng  
- **Item embedding** $$\mathbf{v} \in \mathbb{R}^d$$ — “ảnh chụp” nội dung/sản phẩm  

Điểm khớp thường là tích vô hướng hoặc cosine: $$s(u,i) = \mathbf{u}^\top \mathbf{v}$$.

![Mạng sâu sinh embedding](/deep-learning-self-learning/img/chapter_img/chapter18/recsys_deep_ranking_nn.jpg)
*Hình: Mạng feedforward dùng để học biểu diễn / xếp hạng ứng viên. (Minh họa từ video giải thích recommendation system)*

![Embedding chiều 128](/deep-learning-self-learning/img/chapter_img/chapter18/recsys_embedding_128d.jpg)
*Hình: Đầu ra mạng nén thành vector dày đặc (ví dụ $$1\times 128$$) — “ảnh chụp” sở thích hoặc item. (Minh họa từ video giải thích recommendation system)*

**Mô hình hai tháp (*two-tower*)**: một tháp encode user, một tháp encode item; huấn luyện để cặp (user, item đã tương tác) gần nhau trong không gian nhúng, cặp âm xa nhau. Lợi thế quy mô: precompute toàn bộ item embedding, lúc serving chỉ cần encode user rồi **nearest-neighbor** (ANN) trên hàng tỷ item.

![Two-tower: khớp hai vector 128-d](/deep-learning-self-learning/img/chapter_img/chapter18/recsys_two_tower_matching.jpg)
*Hình: So khớp hai embedding cùng chiều (user vs item) — trực giác của two-tower retrieval. (Minh họa từ video giải thích recommendation system)*

So với Word2Vec: thay vì ngữ cảnh từ, tín hiệu huấn luyện là **hành vi** (click, xem hết, mua). Giả thuyết phân bố trở thành: “user/item xuất hiện trong ‘ngữ cảnh’ tương tác tương tự thì embedding gần nhau”.

## 6. Khái niệm liên quan
Nhúng từ gắn với ngữ nghĩa phân bố (*distributional semantics*) — lý thuyết ngôn ngữ học cho rằng nghĩa từ được xác định bởi ngữ cảnh. Việc cài đặt tính toán — học vectơ sao cho từ trong ngữ cảnh tương tự có biểu diễn tương tự — chính là hiện thực hóa lý thuyết này. Hiểu mối liên hệ giúp đánh giá vì sao nhúng hoạt động: chúng không phải kỹ thuật đặc trưng tùy tiện mà là cài đặt các nguyên lý ngôn ngữ học cơ bản.

Nhúng liên quan đến các kỹ thuật giảm chiều như PCA hoặc autoencoder. Ta nén vectơ one-hot chiều cao (kích thước từ vựng) thành vectơ dày đặc chiều thấp (kích thước nhúng) trong khi bảo toàn thông tin ngữ nghĩa. Phép nén học được khám phá rằng quan hệ ngữ nghĩa có thể nắm bắt bằng ít chiều hơn nhiều so với danh tính ký hiệu tường minh, cho thấy chiều nội tại của ngữ nghĩa từ thấp hơn nhiều so với kích thước từ vựng.

Sự tiến hóa từ nhúng tĩnh (Word2Vec, GloVe) đến nhúng ngữ cảnh (ELMo, BERT) phản ánh sự tinh vi ngày càng tăng. Nhúng tĩnh gán một vectơ cho mỗi loại từ, nên “bank” (tài chính) và “bank” (bờ sông) có cùng biểu diễn dù nghĩa khác. Nhúng ngữ cảnh sinh vectơ khác nhau theo ngữ cảnh, giải quyết đa nghĩa.

![Không gian vector: trái cây vs công ty](/deep-learning-self-learning/img/chapter_img/chapter18/sa_vector_space_fruit_vs_companies.jpg)
*Hình: Cụm “táo / chuối / cam…” tách khỏi “Google / Microsoft / DeepSeek…”. (Minh họa từ video Transformer / Self-Attention nhập môn)*

![Sau self-attention: “Apple” theo nghĩa công ty](/deep-learning-self-learning/img/chapter_img/chapter18/sa_contextual_apple_company_vector.jpg)
*Hình: Khi câu nói về điện thoại/màu cam, vector “Apple” trôi về phía nghĩa hãng (logo). (Minh họa từ video Transformer / Self-Attention nhập môn)*

Lộ trình này cho thấy lĩnh vực tiến từ học biểu diễn cấp từ sang mô hình hóa bản chất phụ thuộc ngữ cảnh của ngôn ngữ.

## 7. Các Bài báo Nền tảng

**["Efficient Estimation of Word Representations in Vector Space" (2013)](https://arxiv.org/abs/1301.3781)**  
*Tác giả*: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
Word2Vec giới thiệu các phương pháp hiệu quả (skip-gram và CBOW) để học nhúng từ ở quy mô lớn. Thủ tục huấn luyện lấy mẫu âm cho phép xử lý hàng tỷ từ, khiến nhúng thực tiễn với từ vựng lớn. Bài báo chứng minh các tính chất ngữ nghĩa đáng chú ý — loại suy giải bằng số học vectơ — cho thấy nhúng nắm bắt mẫu ngôn ngữ tinh vi. Tính đơn giản, hiệu quả và chất lượng của Word2Vec khiến nó được áp dụng rộng rãi, thiết lập nhúng như nền tảng của NLP.

**["GloVe: Global Vectors for Word Representation" (2014)](https://aclanthology.org/D14-1162/)**  
*Tác giả*: Jeffrey Pennington, Richard Socher, Christopher Manning  
GloVe kết hợp phân rã ma trận toàn cục với ngữ cảnh cục bộ, phân rã ma trận đồng xuất hiện từ để học nhúng. Phương pháp đạt hiệu năng cạnh tranh hoặc vượt Word2Vec đồng thời cung cấp diễn giải trực quan qua thống kê đồng xuất hiện. GloVe cho thấy các mục tiêu huấn luyện khác nhau có thể sinh nhúng chất lượng cao tương tự, gợi ý bản thân biểu diễn quan trọng hơn thủ tục huấn luyện cụ thể.

**["Deep contextualized word representations" (2018)](https://arxiv.org/abs/1802.05365)**  
*Tác giả*: Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer  
ELMo giới thiệu nhúng ngữ cảnh từ LSTM hai chiều sâu, biểu diễn mỗi từ khác nhau theo ngữ cảnh câu. Điều này giải quyết hạn chế của nhúng tĩnh với đa nghĩa và nghĩa phụ thuộc ngữ cảnh. ELMo cho thấy mô hình ngôn ngữ sâu học các loại thông tin khác nhau ở các tầng khác nhau (cú pháp ở tầng thấp, ngữ nghĩa ở tầng cao), và kết hợp các tầng cải thiện tác vụ hạ nguồn. ELMo đại diện cho bước chuyển từ biểu diễn tĩnh sang ngữ cảnh, mở đường cho BERT và Transformer.

## Bẫy Thường gặp và Mẹo

Dùng nhúng tiền huấn luyện mà không căn chỉnh từ vựng đúng cách gây vấn đề ngoài từ vựng (*out-of-vocabulary*). Nếu từ vựng tác vụ chứa từ không có trong nhúng tiền huấn luyện, cần chiến lược: dùng nhúng dưới từ (BPE, WordPiece), khởi tạo từ thiếu từ các từ tương tự (nếu thiếu “coronavirus”, lấy trung bình “virus” và “corona”), hoặc tinh chỉnh nhúng trên văn bản miền chuyên biệt.

## Điểm Chính Cần Nhớ

Nhúng từ biểu diễn từ bằng vectơ liên tục dày đặc, trong đó độ tương tự ngữ nghĩa tương ứng với gần gũi hình học, cho phép mạng neuron tổng quát hóa qua các từ gần nghĩa nhờ biểu diễn vectơ dùng chung. Skip-gram Word2Vec dự đoán ngữ cảnh từ đích dùng lấy mẫu âm để hiệu quả, trong khi GloVe phân rã ma trận đồng xuất hiện — cả hai học từ văn bản không nhãn qua giả thuyết phân bố. Nhúng thu được thể hiện các tính chất đáng chú ý gồm suy luận loại suy qua số học vectơ (king − man + woman ≈ queen) và tụ cụm ngữ nghĩa (từ đồng nghĩa có vectơ tương tự), tất cả nảy sinh từ mẫu đồng xuất hiện mà không cần giám sát tường minh. Cùng hình học vector xuất hiện trong **gợi ý**: user/item embedding và mô hình two-tower khớp sở thích với nội dung ở quy mô lớn. Nhúng tiền huấn luyện như Word2Vec và GloVe chuyển giao sang tác vụ hạ nguồn, cung cấp biểu diễn ngữ nghĩa cải thiện hiệu năng trên các ứng dụng NLP từ phân tích cảm xúc đến dịch máy. Nhúng ngữ cảnh hiện đại từ BERT cung cấp biểu diễn phụ thuộc ngữ cảnh giải quyết đa nghĩa, dù nhúng tĩnh vẫn hữu ích về hiệu quả và khả năng diễn giải. Hiểu nhúng từ đặt nền cho mọi học biểu diễn trong NLP (và ngoài NLP), chứng minh cách mạng neuron có thể khám phá cấu trúc thuần từ mẫu dữ liệu.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Transformer là gì? Self-Attention siêu dễ hiểu (Phần 1)](https://www.facebook.com/reel/930207546288223)
- [Word Embedding và Transformer hiểu ngôn ngữ](https://www.facebook.com/reel/1501843101324752)
- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)
