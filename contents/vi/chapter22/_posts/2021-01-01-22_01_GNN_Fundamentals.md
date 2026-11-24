---
layout: post
title: 22-01 Mạng Neuron Đồ thị
chapter: '22'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter22
---

Điều gì xảy ra nếu trong một mạng xã hội, ta chỉ đơn giản lấy *tổng* đặc trưng của mọi bạn bè để cập nhật biểu diễn của một người dùng? Một ngôi sao có 10 triệu người theo dõi sẽ có vector lớn gấp triệu lần người bình thường, khiến kích hoạt bùng nổ! Bài học nền tảng này giải phẫu cách **GCN** giải quyết bài toán đó bằng chuẩn hóa đối xứng $$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$, cách **GAT** dùng cơ chế chú ý học trọng số cạnh, và hiện tượng **Oversmoothing** khi xếp quá nhiều tầng GNN.

## Bản đồ kiến thức nền: Cơ sở toán học của GCN, GAT và các hiện tượng đặc thù trên đồ thị

- **Tích chập đồ thị chuẩn hóa (GCN — Kipf & Welling):** Bổ sung vòng khuyên tự nối $$\tilde{A} = A + I$$ để đỉnh giữ lại đặc trưng chính nó, rồi chuẩn hóa hai đầu theo căn bậc hai bậc đỉnh $$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H W$$ giúp ổn định thang độ lớn.
- **Graph Attention Networks (GAT):** Tính trọng số chú ý $$\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(\mathbf{a}^\top[W\mathbf{h}_i \| W\mathbf{h}_j]))$$ cho phép đỉnh chủ động ưu tiên các láng giềng liên quan nhất.
- **Ba cấp độ tác vụ trên đồ thị:** Phân loại đỉnh (node classification), dự đoán cạnh (link prediction), và phân loại toàn đồ thị (graph classification qua phép gộp readout toàn cục).
- **Hiện tượng quá mịn (Oversmoothing):** Khi xếp quá nhiều tầng (thường $$> 4\text{–}6$$ tầng GCN thuần), việc lấy trung bình lặp đi lặp lại làm mọi vector đỉnh hội tụ về cùng một giá trị, xóa nhòa sự khác biệt giữa các lớp.

### Khung giải thích (nên đọc chậm): Tính tay một bước truyền thông điệp GCN chuẩn hóa đối xứng cho đỉnh trung tâm

Xét đồ thị đường thẳng $$1 - 2 - 3$$ với đặc trưng vô hướng ban đầu $$h_1 = 2{,}0$$, $$h_2 = 4{,}0$$, $$h_3 = 6{,}0$$ và trọng số tuyến tính $$W = 1$$ (bỏ qua ReLU vì các số đều dương).

1. **Thêm vòng khuyên tự nối ($$\tilde{A} = A + I$$):** Bậc mới của từng đỉnh là $$\tilde{d}_1 = 2$$, $$\tilde{d}_2 = 3$$, $$\tilde{d}_3 = 2$$.
2. **Tính giá trị cập nhật cho đỉnh 2** theo công thức $$h_2' = \sum_{j \in \{1,2,3\}} \frac{1}{\sqrt{\tilde{d}_2 \tilde{d}_j}} h_j$$:
   - Đóng góp từ đỉnh 1: $$\frac{1}{\sqrt{3 \times 2}} \cdot 2{,}0 = \frac{2{,}0}{2{,}4495} \approx 0{,}8165$$.
   - Đóng góp từ chính đỉnh 2: $$\frac{1}{\sqrt{3 \times 3}} \cdot 4{,}0 = \frac{4{,}0}{3} \approx 1{,}3333$$.
   - Đóng góp từ đỉnh 3: $$\frac{1}{\sqrt{3 \times 2}} \cdot 6{,}0 = \frac{6{,}0}{2{,}4495} \approx 2{,}4495$$.
3. **Tổng hợp:** $$h_2' \approx 0{,}8165 + 1{,}3333 + 2{,}4495 = 4{,}5993$$.
   Nhận xét: Trọng số chia đều theo bậc của *cả đỉnh gửi lẫn đỉnh nhận* ($$\sqrt{\tilde{d}_i \tilde{d}_j}$$) giúp một đỉnh bậc cao không thống trị phép cập nhật của các đỉnh lân cận bậc thấp!

# Mạng Nơ-ron Đồ thị: Học sâu trên Dữ liệu Đồ thị

![Graph Neural Network](https://distill.pub/2021/gnn-intro/graph_neural_network.png)
*Hình ảnh: Kiến trúc mạng nơ-ron đồ thị với cơ chế truyền thông điệp. Nguồn: Distill.pub*

## 1. Tổng quan khái niệm

Mạng nơ-ron đồ thị (*graph neural network*, GNN) mở rộng học sâu sang dữ liệu có cấu trúc đồ thị, giúp mô hình hóa các hệ thống thực thể và quan hệ phổ biến trong thực tế: mạng xã hội kết nối người dùng, đồ thị phân tử kết nối các nguyên tử, đồ thị tri thức liên kết khái niệm, mạng trích dẫn giữa các bài báo, hay hệ thống gợi ý kết nối người dùng với sản phẩm. Khác với ảnh (lưới 2D đều đặn) hoặc chuỗi (dãy 1D có thứ tự), đồ thị có cấu trúc tùy ý — số lượng láng giềng thay đổi theo từng nút, không có thứ tự không gian cố định và sơ đồ kết nối phức tạp — đòi hỏi các kiến trúc chuyên biệt vừa tận dụng được cấu trúc đồ thị vừa khả vi để huấn luyện bằng lan truyền ngược.

Yêu cầu cơ bản nhất đối với mô hình học trên đồ thị là tính bất biến hoán vị (*permutation invariance*): biểu diễn của đồ thị không được thay đổi khi ta đổi thứ tự đánh số các nút. Dù đánh số nút là 1, 2, 3 hay 3, 2, 1, đồ thị vẫn không đổi nên kết quả biểu diễn phải nhất quán. Điều này khiến việc đưa thẳng ma trận kề vào một mạng nơ-ron truyền thống trở nên không phù hợp, vì mạng sẽ xem các hoán vị khác nhau là các đầu vào khác nhau. GNN giải quyết vấn đề này bằng cơ chế truyền thông điệp (*message passing*): qua từng tầng, mỗi nút tổng hợp thông tin từ các láng giềng bằng những phép toán bất biến hoán vị (như tổng, trung bình hoặc cực đại) kết hợp biến đổi học được. Sau $$k$$ vòng lặp, biểu diễn của mỗi nút sẽ tích hợp thông tin từ vùng lân cận trong phạm vi $$k$$ bước (*k-hop neighborhood*).

Các bài toán học trên đồ thị thường chia thành ba nhóm chính theo mức độ đầu ra: phân loại nút (*node classification*) dự đoán nhãn cho từng nút (như phân loại người dùng trên mạng xã hội); dự đoán cạnh (*link prediction*) ước lượng khả năng tồn tại của các liên kết còn thiếu hoặc phát sinh trong tương lai (như gợi ý kết bạn hay sản phẩm); và phân loại đồ thị (*graph classification*) gán nhãn cho cả đồ thị (như dự đoán hoạt tính sinh học của một phân tử thuốc). Cả ba tác vụ đều dùng chung khung truyền thông điệp nhưng khác nhau ở bước tổng hợp và đầu ra dự đoán.

GNN được ứng dụng trong nhiều lĩnh vực: trong hóa học để dự đoán độ tan hoặc độc tính của phân tử; trong mạng xã hội để phát hiện cộng đồng và tìm người dùng có ảnh hưởng; trong hệ thống gợi ý trên đồ thị hai phía (người dùng – sản phẩm); trong sinh học cấu trúc để mô hình hóa tương tác giữa các axit amin; và trong giao thông để dự báo lưu lượng dựa trên mạng lưới đường bộ.

## 2. Nền tảng toán học

Một đồ thị $$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$ gồm tập nút $$\mathcal{V} = \{v_1, \ldots, v_n\}$$ và tập cạnh $$\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$$. Mỗi nút có vectơ đặc trưng $$\mathbf{x}_i \in \mathbb{R}^d$$, và mỗi cạnh có thể kèm đặc trưng $$\mathbf{e}_{ij}$$. Ma trận kề $$\mathbf{A} \in \{0,1\}^{n \times n}$$ mã hóa cấu trúc kết nối: $$A_{ij} = 1$$ nếu $$(v_i, v_j) \in \mathcal{E}$$ và bằng 0 trong trường hợp ngược lại.

### Khung truyền thông điệp (Message Passing)

GNN cập nhật biểu diễn của các nút qua nhiều tầng truyền thông điệp. Tại tầng $$k$$, biểu diễn $$\mathbf{h}_i^{(k)}$$ của nút $$i$$ được cập nhật từ trạng thái trước đó của chính nó và của các láng giềng:

$$\mathbf{h}_i^{(k)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \text{AGGREGATE}^{(k)}\left(\{\mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i)\}\right)\right)$$

trong đó $$\mathcal{N}(i)$$ là tập các nút láng giềng của $$i$$. Hàm `AGGREGATE` gom thông tin từ láng giềng (phải bất biến hoán vị), còn hàm `UPDATE` kết hợp đặc trưng hiện tại của nút $$i$$ với thông tin đã gom được.

### Graph Convolutional Networks (GCN)

GCN xuất phát từ lý thuyết đồ thị phổ, định nghĩa phép tích chập thông qua ma trận Laplacian của đồ thị. Dạng cập nhật thực hành theo từng tầng là:

$$\mathbf{H}^{(k+1)} = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(k)}\mathbf{W}^{(k)})$$

trong đó $$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$$ là ma trận kề có bổ sung vòng tự nối (*self-loop*), $$\tilde{\mathbf{D}}$$ là ma trận bậc tương ứng, $$\mathbf{H}^{(k)}$$ là ma trận biểu diễn nút ở tầng $$k$$, và $$\mathbf{W}^{(k)}$$ là ma trận trọng số cần học. Phép toán này tính trung bình có trọng số các láng giềng (nút có bậc càng cao thì đóng góp trên mỗi cạnh càng giảm nhờ hệ số chuẩn hóa) rồi áp dụng phép biến đổi tuyến tính và hàm kích hoạt phi tuyến.

### GraphSAGE

GraphSAGE lấy mẫu một tập láng giềng có kích thước cố định và áp dụng hàm tổng hợp học được:

$$\mathbf{h}_{\mathcal{N}(i)}^{(k)} = \text{AGGREGATE}(\{\mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i)\})$$

$$\mathbf{h}_i^{(k)} = \sigma(\mathbf{W}^{(k)} \cdot [\mathbf{h}_i^{(k-1)}, \mathbf{h}_{\mathcal{N}(i)}^{(k)}])$$

Hàm `AGGREGATE` có thể là trung bình (*mean*), cực đại (*max*) hoặc LSTM trên tập láng giềng. Cách lấy mẫu này cho phép huấn luyện theo *mini-batch* và xử lý hiệu quả các đồ thị lớn có số láng giềng biến thiên mạnh.

## 3. Ví dụ và trực giác

Xét bài toán phân loại nút trên một mạng xã hội: mỗi nút là một người dùng (có đặc trưng tuổi, vị trí, sở thích), mỗi cạnh là quan hệ bạn bè, và mục tiêu là dự đoán ai sẽ quan tâm đến một sản phẩm mới.

Với một GNN 2 tầng, ban đầu $$\mathbf{h}_i^{(0)}$$ chỉ chứa đặc trưng cá nhân của người dùng $$i$$. Sau tầng thứ nhất, $$\mathbf{h}_i^{(1)}$$ tích hợp thêm thông tin từ bạn bè trực tiếp (láng giềng 1 bước) — chẳng hạn phản ánh rằng nhóm bạn của người dùng A phần lớn ở độ tuổi 25–30, sống tại đô thị và quan tâm công nghệ. Sau tầng thứ hai, $$\mathbf{h}_i^{(2)}$$ tiếp tục tổng hợp thông tin từ bạn của bạn (vùng lân cận 2 bước), nắm bắt ngữ cảnh cộng đồng rộng hơn. Nhờ kết hợp cả thuộc tính cá nhân lẫn cấu trúc lân cận, mô hình dự đoán chính xác hơn so với việc chỉ dựa vào hồ sơ riêng lẻ của từng người dùng.

## 4. Mã minh họa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        """
        x: (num_nodes, in_features)
        adj: (num_nodes, num_nodes) adjacency matrix
        """
        # Add self-loops
        adj_hat = adj + torch.eye(adj.size(0))
        
        # Normalize
        deg = adj_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = torch.diag(deg_inv_sqrt)
        adj_normalized = norm @ adj_hat @ norm
        
        # Apply convolution
        support = x @ self.weight
        output = adj_normalized @ support
        
        return output

class GCN(nn.Module):
    """2-layer Graph Convolutional Network"""
    
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.gc1 = GCNLayer(num_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Example
num_nodes = 100
num_features = 16
num_classes = 7

gcn = GCN(num_features, hidden_dim=32, num_classes=num_classes)

# Random graph
x = torch.randn(num_nodes, num_features)
adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
adj = (adj + adj.T) / 2  # Symmetric

output = gcn(x, adj)
print(f"GCN output: {output.shape}")  # (100, 7)
print("Each node gets class predictions using graph structure!")
```

## 5. Khái niệm liên quan

GNN liên hệ mật thiết với lý thuyết đồ thị phổ (*spectral graph theory*). Ma trận Laplacian và phép phân rã trị riêng cung cấp cơ sở toán học để mở rộng phép tích chập từ lưới đều trong CNN sang cấu trúc đồ thị tùy ý.

## 6. Các bài báo nền tảng

**["Semi-Supervised Classification with Graph Convolutional Networks" (2017)](https://arxiv.org/abs/1609.02907)**  
*Tác giả*: Thomas Kipf, Max Welling  
Giới thiệu GCN cho bài toán phân loại nút bán giám sát, kết hợp cấu trúc đồ thị với số lượng nhãn hạn chế.

**["Inductive Representation Learning on Large Graphs" (2017)](https://arxiv.org/abs/1706.02216)**  
*Tác giả*: William Hamilton, Rex Ying, Jure Leskovec  
GraphSAGE dùng kỹ thuật lấy mẫu lân cận để huấn luyện *mini-batch* trên đồ thị lớn, mở rộng GNN từ thiết lập chuyển dẫn (*transductive* — đồ thị cố định) sang quy nạp (*inductive* — tổng quát hóa sang nút và đồ thị chưa từng gặp).

**["Graph Attention Networks" (2018)](https://arxiv.org/abs/1710.10903)**  
*Tác giả*: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio  
GAT áp dụng cơ chế *attention* để học trọng số đóng góp khác nhau cho từng láng giềng thay vì dùng trọng số cố định theo bậc nút.

## Bẫy thường gặp và mẹo thực hành

Hiện tượng làm mịn quá mức (*oversmoothing*) dễ xảy ra khi xếp quá nhiều tầng GNN: biểu diễn của các nút dần trở nên giống nhau và mất khả năng phân biệt do tổng hợp thông tin từ vùng lân cận quá rộng. Có thể giảm thiểu bằng cách thêm kết nối tắt (*skip connection*), chuẩn hóa và giữ độ sâu vừa phải (thường 2–3 tầng là đủ cho nhiều bài toán).

## Điểm chính cần nhớ

Mạng nơ-ron đồ thị xử lý dữ liệu dạng đồ thị bằng cơ chế truyền thông điệp, cập nhật biểu diễn từng nút qua các phép tổng hợp láng giềng bất biến hoán vị. GCN sử dụng ma trận kề chuẩn hóa cho tích chập đồ thị, GraphSAGE lấy mẫu lân cận để mở rộng quy mô, còn GAT học trọng số láng giềng bằng *attention*. GNN giúp áp dụng học sâu lên dữ liệu quan hệ không đều đặn — nơi CNN và RNN truyền thống không phù hợp — phục vụ các bài toán trên mạng xã hội, phân tử, đồ thị tri thức và hệ thống gợi ý.

## Video tham khảo

- **Xem trước** [Graph Neural Networks Explained: A Clear Guide to GNN Basics & Models](https://www.youtube.com/watch?v=eGoszzMkGfU) — IBM Technology, tiếng Anh. Minh họa nút, cạnh và phép cập nhật bất biến hoán vị trong GNN.
- [Message Passing: How Graph Neural Networks Actually Work](https://www.youtube.com/watch?v=1DmNdMUbDkg) — The Synthetic Mind, tiếng Anh. Giải thích chi tiết bước tổng hợp thông tin từ láng giềng tương ứng với công thức trong bài.
- [Linear transformations and matrices | Chapter 3, Essence of linear algebra](https://www.youtube.com/watch?v=kYB8IZa5AuE) — 3Blue1Brown, tiếng Anh. Ôn lại ý nghĩa hình học của phép nhân ma trận tuyến tính áp dụng ngay sau bước tổng hợp láng giềng.
