---
layout: post
title: 16-01 Học Tự giám sát
chapter: '16'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter16
---

Trong thế giới không nhãn, mối nguy lớn nhất luôn rình rập mọi thuật toán học biểu diễn là **sụp đổ biểu diễn (Representation Collapse)**: nếu bộ mã hóa ánh xạ *mọi* bức ảnh về đúng một vector hằng số $$\mathbf{z} = (0, 0, \dots, 0)$$, thì mọi cặp ảnh giống nhau đều có khoảng cách bằng 0! Bài học này giải phẫu cơ chế toán học của **InfoNCE**, **Masked Prediction** và **BYOL/DINO** — ba con đường thông minh giúp học sâu thoát khỏi bẫy sụp đổ để tạo ra những bộ biểu diễn còn mạnh hơn cả học có giám sát.

## Kiến thức cần nhớ: Cơ chế toán học ngăn sụp đổ biểu diễn trong SSL

- **Hàm mất mát InfoNCE (NT-Xent) và nhiệt độ $$\tau$$:** Xây dựng bài toán phân loại $$(2N-1)$$ lớp bằng độ tương đồng cosin chia nhiệt độ $$\tau$$, tạo đồng thời lực hút (cặp dương) và lực đẩy (cặp âm) chống sụp đổ.
- **Liên hệ với thông tin tương hỗ (Mutual Information):** Chứng minh cực tiểu hóa InfoNCE tương đương cực đại hóa cận dưới của thông tin tương hỗ $$I(\mathbf{v}_1; \mathbf{v}_2)$$ giữa hai góc nhìn.
- **Dự đoán phần bị che (BERT / MAE):** Khai thác sự dư thừa ngữ cảnh trong chuỗi văn bản hoặc mảng ảnh; phần mất mát chỉ tính trên các vị trí bị che giúp mô hình học cấu trúc toàn cục.
- **Phá đối xứng không cần cặp âm (BYOL / SimSiam / DINO):** Dùng mạng đích cập nhật trượt (EMA), đầu dự đoán (predictor), `stop-gradient` và căn giữa (centering) để học biểu diễn giàu thông tin mà không cần batch size khổng lồ.

### Giải thích: Tính tay hàm mất mát InfoNCE (NT-Xent) cho một cặp dương và hai mẫu âm

Xét một vector truy vấn $$\mathbf{z}_i$$ (đã chuẩn hóa đơn vị $$\|\mathbf{z}\|_2 = 1$$) với cặp dương $$\mathbf{z}_j$$ và hai mẫu âm $$\mathbf{z}_{k_1}, \mathbf{z}_{k_2}$$. Giả sử độ tương đồng cosin lần lượt là:
- $$s(i, j) = 0{,}80$$ (cặp dương)
- $$s(i, k_1) = 0{,}20$$, $$s(i, k_2) = -0{,}20$$ (hai mẫu âm)
Chọn tham số nhiệt độ $$\tau = 0{,}2$$.

1. **Chia nhiệt độ $$\tau$$ để tạo logit:**
   - Cặp dương: $$0{,}80 / 0{,}2 = 4{,}0$$.
   - Hai mẫu âm: $$0{,}20 / 0{,}2 = 1{,}0$$ và $$-0{,}20 / 0{,}2 = -1{,}0$$.
2. **Tính hàm mũ và xác suất Softmax của cặp dương:**
   - $$e^{4{,}0} \approx 54{,}60$$, $$e^{1{,}0} \approx 2{,}72$$, $$e^{-1{,}0} \approx 0{,}37$$.
   - Tổng mẫu số: $$54{,}60 + 2{,}72 + 0{,}37 = 57{,}69$$.
   - Xác suất nhận đúng cặp dương: $$p = \frac{54{,}60}{57{,}69} \approx 0{,}9464$$.
3. **Tính InfoNCE loss:** $$\mathcal{L}_i = -\ln(0{,}9464) \approx 0{,}055$$.
   *Thử nghiệm tư duy:* Nếu tăng $$\tau = 1{,}0$$, các logit co lại thành $$0{,}8; 0{,}2; -0{,}2$$ và $$p$$ tụt xuống $$0{,}526$$ ($$\mathcal{L}_i = 0{,}642$$). Nhiệt độ $$\tau$$ nhỏ giúp khuếch đại gradient đặc biệt mạnh lên những **mẫu âm khó (hard negatives)** có độ tương đồng gần cặp dương!

# Học Tự giám sát: Học từ Chính Dữ liệu

## 1. Tổng quan khái niệm

Học tự giám sát (*self-supervised learning*) là một hướng tiếp cận khác trong việc khai thác dữ liệu chưa gán nhãn: tín hiệu giám sát được tạo tự động từ chính dữ liệu thay vì dựa vào nhãn thủ công tốn kém do con người thực hiện. Ý tưởng cốt lõi là bản thân dữ liệu đã chứa cấu trúc và mối quan hệ nội tại có thể dùng làm tín hiệu học: ảnh có cấu trúc không gian cho phép dự đoán một phần từ các phần còn lại; văn bản có cấu trúc tuần tự cho phép dự đoán từ bị che dựa trên ngữ cảnh; video có tính liên tục theo thời gian giúp dự đoán thứ tự các khung hình. Bằng cách xây dựng các tác vụ dự đoán khai thác những cấu trúc này, ta có thể huấn luyện mạng nơ-ron trên các tập dữ liệu chưa gán nhãn quy mô lớn và thu được biểu diễn chuyển giao hiệu quả sang các tác vụ có giám sát ở giai đoạn sau (*downstream tasks*).

Sự khác biệt giữa học tự giám sát và học không giám sát (*unsupervised learning*) tuy tinh tế nhưng quan trọng. Học không giám sát truyền thống (như phân cụm hay PCA) tìm kiếm cấu trúc trong dữ liệu mà không dùng cấu trúc đó cho bài toán dự đoán. Học tự giám sát xây dựng các tác vụ dự đoán có giám sát với nhãn được sinh tự động từ cấu trúc dữ liệu, biến dữ liệu chưa gán nhãn thành bài toán học có giám sát thông qua việc thiết kế tác vụ phù hợp. Mô hình ngôn ngữ che dấu (*masked language modeling*, MLM) trong BERT — dự đoán từ bị che dựa trên ngữ cảnh — thực chất là học có giám sát, trong đó nhãn (các từ gốc) đến từ chính văn bản chứ không phải từ người gán nhãn.

Học tự giám sát hiện đại đạt hiệu quả cao, đặc biệt trong xử lý ngôn ngữ tự nhiên (NLP), nơi tiền huấn luyện trên kho văn bản lớn với các mục tiêu tự giám sát (mô hình ngôn ngữ che dấu, dự đoán câu tiếp theo) rồi tinh chỉnh trên tác vụ có giám sát đã trở thành quy trình phổ biến. BERT, GPT và các mô hình tương tự học được biểu diễn ngôn ngữ phong phú từ văn bản chưa gán nhãn, sau đó chuyển giao sang nhiều tác vụ khác nhau. Trong thị giác máy tính, các phương pháp học đối chiếu như SimCLR và MoCo học biểu diễn ảnh bằng cách phân biệt các phiên bản tăng cường (*augmentation*) của cùng một ảnh với các ảnh khác, đạt chất lượng biểu diễn cạnh tranh với tiền huấn luyện có giám sát.

Để nắm vững học tự giám sát, cần hiểu rõ các tác vụ tiền huấn luyện (*pretext tasks*) — tức các mục tiêu giám sát tự động dùng khi tiền huấn luyện — và loại biểu diễn mà chúng tạo ra. Một tác vụ tiền huấn luyện tốt cần đáp ứng ba tiêu chí: (1) giải được chỉ từ dữ liệu mà không cần nhãn; (2) đòi hỏi mô hình hiểu cấu trúc ngữ nghĩa để giải tốt; và (3) tạo ra biểu diễn hữu ích cho các tác vụ hạ nguồn. Khó khăn chính nằm ở việc thiết kế tác vụ sao cho mô hình buộc phải học các đặc trưng tổng quát, thay vì dựa vào những “lối tắt” nông trong dữ liệu.

## 2. Nền tảng toán học

### Học đối chiếu (Contrastive Learning)

Các phương pháp đối chiếu học bằng cách phân biệt các cặp tương tự (mẫu dương) với các cặp không tương tự (mẫu âm). Cho mẫu neo (*anchor*) $$\mathbf{x}$$, mẫu dương $$\mathbf{x}^+$$ (phiên bản tăng cường của cùng một ảnh), và tập mẫu âm $$\{\mathbf{x}_i^-\}$$ (các ảnh khác):

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}^+))/\tau)}{\exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}^+))/\tau) + \sum_i \exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}_i^-))/\tau)}$$

trong đó $$f$$ là bộ mã hóa (*encoder*), $$\text{sim}$$ là độ tương tự (thường là độ tương tự cosine), và $$\tau$$ là tham số nhiệt độ (*temperature*). Hàm mất mát NT-Xent (*normalized temperature-scaled cross-entropy*) thúc đẩy biểu diễn của các phiên bản tăng cường từ cùng một ảnh nằm gần nhau, đồng thời đẩy biểu diễn của các ảnh khác ra xa nhau.

### Dự đoán phần bị che (Masked Prediction)

BERT che ngẫu nhiên 15% số token và dự đoán lại chúng từ ngữ cảnh xung quanh:

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{\mathbf{x}}\left[\sum_{i \in \mathcal{M}} \log p(x_i | \mathbf{x}_{\backslash \mathcal{M}})\right]$$

trong đó $$\mathcal{M}$$ là tập các vị trí bị che. Mô hình phải khai thác ngữ cảnh hai chiều để dự đoán các token bị che, qua đó học được cấu trúc và ngữ nghĩa của ngôn ngữ.

## 3. Ví dụ và trực giác

Xét bài toán học biểu diễn hình ảnh từ tập ảnh chưa gán nhãn. Từ một bức ảnh con mèo, ta tạo hai phiên bản tăng cường: một phiên bản cắt ngẫu nhiên kết hợp biến đổi màu sắc, phiên bản kia cắt ở vùng khác và xoay nhẹ. Đây là một cặp dương — hai góc nhìn khác nhau của cùng một đối tượng, nên cần có biểu diễn gần nhau.

Các ảnh chó, xe cộ và cây cối đóng vai trò mẫu âm. Học đối chiếu kéo biểu diễn của hai phiên bản ảnh mèo lại gần nhau trong không gian đặc trưng, đồng thời đẩy chúng ra xa biểu diễn của chó, xe và cây. Sau khi huấn luyện trên hàng triệu ảnh, các biểu diễn sẽ gom cụm theo nội dung ngữ nghĩa: các ảnh mèo nằm gần nhau, các ảnh chó nằm gần nhau; ngoài ra, mèo và chó nằm gần nhau hơn (cùng là động vật) so với xe cộ.

Những biểu diễn này có thể chuyển giao trực tiếp sang bài toán phân loại: dù chưa từng sử dụng nhãn trong giai đoạn tiền huấn luyện, bộ mã hóa đã học được cách trích xuất đặc trưng giúp phân biệt đối tượng, hỗ trợ hiệu quả cho giai đoạn phân loại có giám sát khi số lượng nhãn bị hạn chế.

## 4. Mã minh họa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """Simplified SimCLR for contrastive learning"""
    
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        
        # Encoder (e.g., ResNet)
        self.encoder = base_encoder
        
        # Projection head
        # Maps encoder output to space where contrastive loss is computed
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)  # L2 normalize

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent loss for contrastive learning
    
    z1, z2: (batch, dim) representations of augmented pairs
    """
    batch_size = z1.size(0)
    
    # Concatenate augmentations
    z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # (2*batch, 2*batch)
    
    # Positive pairs: (i, i+batch) and (i+batch, i)
    pos_sim = torch.cat([
        sim_matrix[range(batch_size), range(batch_size, 2*batch_size)],
        sim_matrix[range(batch_size, 2*batch_size), range(batch_size)]
    ])
    
    # Mask out self-similarity
    mask = ~torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix[mask].view(2*batch_size, -1)
    
    # NT-Xent loss
    loss = -pos_sim + torch.log(sim_matrix.exp().sum(dim=1))
    
    return loss.mean()

print("Self-supervised learning enables learning from unlabeled data!")
```

## 5. Khái niệm liên quan

Học tự giám sát gắn chặt với học chuyển giao (*transfer learning*) như một chiến lược tiền huấn luyện. Thay vì tiền huấn luyện có giám sát trên ImageNet, ta có thể tiền huấn luyện tự giám sát trên dữ liệu ảnh chưa gán nhãn, và trong nhiều trường hợp thu được biểu diễn chuyển giao ngang bằng hoặc tốt hơn.

## 6. Các bài báo nền tảng

**["Momentum Contrast for Unsupervised Visual Representation Learning" (2020)](https://arxiv.org/abs/1911.05722)**  
*Tác giả*: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick  
MoCo duy trì một hàng đợi lớn các mẫu âm thông qua bộ mã hóa cập nhật theo động lượng (*momentum encoder*), giúp học đối chiếu ổn định và đạt chất lượng biểu diễn cạnh tranh với tiền huấn luyện có giám sát.

**["A Simple Framework for Contrastive Learning of Visual Representations" (2020)](https://arxiv.org/abs/2002.05709)**  
*Tác giả*: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton  
SimCLR cho thấy khung học đối chiếu đơn giản kết hợp phép tăng cường dữ liệu mạnh và kích thước batch lớn có thể tạo ra biểu diễn chất lượng cao, ngang bằng hoặc vượt nhiều phương pháp tiền huấn luyện có giám sát.

**["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805)**  
Mục tiêu mô hình ngôn ngữ che dấu của BERT là một dạng học tự giám sát tiêu biểu, tạo tín hiệu giám sát trực tiếp từ văn bản gốc.

## Điểm chính cần nhớ

Học tự giám sát tạo tín hiệu giám sát tự động từ cấu trúc nội tại của dữ liệu, cho phép huấn luyện trên các tập dữ liệu chưa gán nhãn quy mô lớn thông qua các tác vụ tiền huấn luyện như dự đoán phần bị che hoặc học đối chiếu. Các phương pháp đối chiếu phân biệt các phiên bản tăng cường của cùng một mẫu với các mẫu khác, giúp biểu diễn vừa bất biến trước phép tăng cường vừa giữ được thông tin ngữ nghĩa. Các tác vụ dự đoán phần bị che như MLM của BERT yêu cầu khôi phục phần thiếu từ ngữ cảnh, buộc mô hình nắm bắt cấu trúc dữ liệu. Tiền huấn luyện tự giám sát thường đạt hiệu quả ngang bằng hoặc tốt hơn tiền huấn luyện có giám sát trong học chuyển giao, vì biểu diễn không bị ràng buộc vào một tập nhãn cố định.

## Video tham khảo

- **Xem trước** [Contrastive Learning with SimCLR | Deep Learning Animated](https://www.youtube.com/watch?v=UqJauYELn6c) — Deepia, tiếng Anh. Giải thích trực quan hàm mất mát NT-Xent trên hai phiên bản tăng cường của cùng một ảnh.
- [DINO: Emerging Properties in Self-Supervised Vision Transformers (Facebook AI Research Explained)](https://www.youtube.com/watch?v=h3ij3F3cPIk) — Yannic Kilcher, tiếng Anh. Phân tích cơ chế tự chưng cất và khả năng phân đoạn tự phát sinh của DINO.
- [But what is cross-entropy? | Compression is Intelligence Part 2](https://www.youtube.com/watch?v=GlYgs6v2YfU) — 3Blue1Brown, tiếng Anh. Giải thích hàm cross-entropy, nền tảng cho tác vụ dự đoán phần bị che kiểu BERT song song với NT-Xent.
