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

# Học Tự giám sát: Học từ Chính Dữ liệu

## 1. Tổng quan khái niệm
Học tự giám sát (*self-supervised learning*) đại diện cho một sự chuyển dịch mô hình trong cách khai thác dữ liệu không gán nhãn: tín hiệu giám sát được tạo tự động từ chính dữ liệu thay vì dựa vào chú thích tốn kém của con người. Ý tưởng then chốt là dữ liệu chứa cấu trúc và quan hệ nội tại có thể đóng vai trò tín hiệu học: ảnh có cấu trúc không gian cho phép dự đoán một phần từ các phần còn lại; văn bản có cấu trúc tuần tự cho phép dự đoán từ bị che từ ngữ cảnh; video có tính liên tục thời gian làm cho thứ tự khung hình có thể dự đoán được. Bằng cách xây dựng các tác vụ dự đoán khai thác các cấu trúc này, ta có thể huấn luyện mạng neuron trên các tập dữ liệu không nhãn quy mô lớn và học được các biểu diễn chuyển giao hiệu quả sang các tác vụ giám sát hạ nguồn.

Sự phân biệt giữa học tự giám sát và học không giám sát (*unsupervised learning*) tinh tế nhưng quan trọng. Học không giám sát truyền thống (phân cụm, PCA) khám phá cấu trúc mà không dùng cấu trúc đó cho dự đoán. Học tự giám sát xây dựng các tác vụ dự đoán có giám sát với nhãn sinh tự động từ cấu trúc dữ liệu, về bản chất chuyển dữ liệu không giám sát thành bài toán học có giám sát thông qua thiết kế tác vụ khéo léo. Mô hình ngôn ngữ che dấu (*masked language modeling*) trong BERT — dự đoán từ bị che từ ngữ cảnh — chính là học có giám sát, trong đó nhãn (các từ gốc) đến từ chính dữ liệu chứ không phải từ người gán nhãn.

Học tự giám sát hiện đại đã đạt được thành công đáng kể, đặc biệt trong NLP, nơi tiền huấn luyện trên văn bản khổng lồ với các mục tiêu tự giám sát (mô hình ngôn ngữ che dấu, dự đoán câu kế) rồi tinh chỉnh trên các tác vụ có giám sát đã trở thành chuẩn mực. BERT, GPT và các mô hình tương tự học được hiểu biết ngôn ngữ phong phú từ văn bản không nhãn, sau đó chuyển giao sang nhiều tác vụ đa dạng. Trong thị giác máy tính, các phương pháp học đối chiếu như SimCLR và MoCo học biểu diễn hình ảnh bằng cách phân biệt các phiên bản tăng cường (*augmentation*) của cùng một ảnh với các ảnh khác, đạt được biểu diễn cạnh tranh với tiền huấn luyện có giám sát.

Để hiểu sâu học tự giám sát, cần nắm rõ các *pretext task* — các mục tiêu có giám sát tự động dùng trong tiền huấn luyện — và loại biểu diễn mà chúng khuyến khích. Một pretext task tốt cần: (1) giải được chỉ từ dữ liệu, không cần nhãn; (2) đòi hỏi hiểu cấu trúc ngữ nghĩa để giải hiệu quả; (3) sinh ra biểu diễn hữu ích cho các tác vụ hạ nguồn. Nghệ thuật nằm ở việc thiết kế tác vụ sao cho việc giải chúng buộc mô hình học các đặc trưng tổng quát hữu ích, thay vì khai thác các “lối tắt” trong dữ liệu.

## 2. Nền tảng toán học
### Học đối chiếu (Contrastive Learning)

Các phương pháp đối chiếu học bằng cách phân biệt các cặp tương tự (dương) với các cặp không tương tự (âm). Cho mỏ neo (*anchor*) $$\mathbf{x}$$, mẫu dương $$\mathbf{x}^+$$ (phiên bản tăng cường của cùng ảnh), và các mẫu âm $$\{\mathbf{x}_i^-\}$$ (các ảnh khác):

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}^+))/\tau)}{\exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}^+))/\tau) + \sum_i \exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}_i^-))/\tau)}$$

trong đó $$f$$ là mạng mã hóa (*encoder*), $$\text{sim}$$ là độ tương tự (thường là cosine), $$\tau$$ là nhiệt độ (*temperature*). Hàm mất mát NT-Xent (*normalized temperature-scaled cross-entropy*) khuyến khích biểu diễn trong đó các tăng cường của cùng ảnh nằm gần nhau, trong khi các ảnh khác nằm xa nhau.

### Dự đoán bị che (Masked Prediction)

BERT che 15% token và dự đoán chúng từ ngữ cảnh:

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{\mathbf{x}}\left[\sum_{i \in \mathcal{M}} \log p(x_i | \mathbf{x}_{\backslash \mathcal{M}})\right]$$

trong đó $$\mathcal{M}$$ là tập vị trí bị che. Mô hình phải dùng ngữ cảnh hai chiều để dự đoán từ bị che, từ đó học hiểu biết ngôn ngữ sâu.

## 3. Ví dụ / Trực giác

Hãy xét bài toán học biểu diễn hình ảnh từ ảnh không nhãn. Lấy một bức ảnh con mèo. Tạo hai phiên bản tăng cường: một với cắt ngẫu nhiên và nhiễu màu, một với cắt khác và xoay. Đây là cặp dương — các góc nhìn khác nhau của cùng con mèo, cần có biểu diễn tương tự.

Lấy các ảnh chó, xe, cây làm mẫu âm. Học đối chiếu đẩy các tăng cường của mèo lại gần nhau trong không gian biểu diễn, đồng thời đẩy mèo ra xa chó/xe/cây. Sau khi huấn luyện trên hàng triệu ảnh, các biểu diễn tụ cụm theo nội dung ngữ nghĩa: mọi mèo nằm gần nhau, mọi chó nằm gần nhau; mèo và chó gần nhau hơn (cùng là động vật) so với xe.

Các biểu diễn học được này chuyển giao sang phân loại — dù chưa từng dùng nhãn trong tiền huấn luyện, bộ mã hóa đã học trích xuất đặc trưng phân biệt đối tượng, hỗ trợ trực tiếp phân loại có giám sát với nhãn hạn chế.

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
    
    # Mask out self-similarity
    mask = ~torch.eye(2*batch_size, dtype=torch.bool)
    sim_matrix = sim_matrix[mask].view(2*batch_size, -1)
    
    # Positive pairs: (i, i+batch) and (i+batch, i)
    pos_sim = torch.cat([
        sim_matrix[range(batch_size), range(batch_size, 2*batch_size)],
        sim_matrix[range(batch_size, 2*batch_size), range(batch_size)]
    ])
    
    # NT-Xent loss
    loss = -pos_sim + torch.log(sim_matrix.exp().sum(dim=1))
    
    return loss.mean()

print("Self-supervised learning enables learning from unlabeled data!")
```

## 5. Khái niệm liên quan
Học tự giám sát gắn liền với học chuyển giao (*transfer learning*) như một chiến lược tiền huấn luyện. Thay vì tiền huấn luyện có giám sát trên ImageNet, ta dùng tiền huấn luyện tự giám sát trên ảnh không nhãn, thường học được biểu diễn chuyển giao thậm chí tốt hơn.

## 6. Các Bài báo Nền tảng

**["Momentum Contrast for Unsupervised Visual Representation Learning" (2020)](https://arxiv.org/abs/1911.05722)**  
*Tác giả*: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick  
MoCo duy trì từ điển lớn các mẫu âm thông qua encoder động lượng (*momentum encoder*), cho phép học đối chiếu hiệu quả. Đạt biểu diễn cạnh tranh với tiền huấn luyện có giám sát.

**["A Simple Framework for Contrastive Learning of Visual Representations" (2020)](https://arxiv.org/abs/2002.05709)**  
*Tác giả*: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton  
SimCLR cho thấy học đối chiếu đơn giản với tăng cường mạnh và batch lớn đạt biểu diễn xuất sắc, vượt nhiều cách tiếp cận tiền huấn luyện có giám sát.

**["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805)**  
Mô hình ngôn ngữ che dấu của BERT là học tự giám sát, tạo giám sát từ chính văn bản.

## Điểm Chính Cần Nhớ

Học tự giám sát tạo tín hiệu giám sát tự động từ cấu trúc dữ liệu, cho phép học trên các tập không nhãn quy mô lớn thông qua pretext task như dự đoán bị che hoặc học đối chiếu. Các phương pháp đối chiếu học bằng cách phân biệt các góc nhìn tăng cường của cùng một mẫu với các mẫu khác, học tính bất biến với tăng cường đồng thời nắm bắt nội dung ngữ nghĩa. Các tác vụ dự đoán bị che như MLM của BERT học dự đoán phần còn thiếu từ ngữ cảnh, đòi hỏi hiểu sâu cấu trúc dữ liệu. Tiền huấn luyện tự giám sát thường ngang hoặc vượt tiền huấn luyện có giám sát trong học chuyển giao, cho thấy học bất chấp tác vụ từ dữ liệu không nhãn sinh ra biểu diễn linh hoạt và đa dụng.
