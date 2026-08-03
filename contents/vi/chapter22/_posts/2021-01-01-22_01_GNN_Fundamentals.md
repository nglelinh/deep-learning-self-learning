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

# Mạng Neuron Đồ thị: Học sâu trên Đồ thị

![Graph Neural Network](https://distill.pub/2021/gnn-intro/graph_neural_network.png)
*Hình ảnh: Kiến trúc mạng neuron đồ thị với truyền thông điệp. Nguồn: Distill.pub*

## 1. Tổng quan khái niệm
Mạng neuron đồ thị (*graph neural network*, GNN) mở rộng học sâu sang dữ liệu có cấu trúc đồ thị, cho phép mạng neuron xử lý mạng lưới các thực thể và quan hệ tràn ngập dữ liệu thực tế: mạng xã hội kết nối con người, đồ thị phân tử kết nối nguyên tử, đồ thị tri thức liên kết khái niệm, mạng trích dẫn liên hệ bài báo, và hệ thống gợi ý kết nối người dùng với mặt hàng. Khác ảnh (lưới 2D đều) hay chuỗi (xích 1D), đồ thị có cấu trúc tùy ý — số láng giềng biến thiên, không có thứ tự không gian, mẫu kết nối phức tạp — đòi hỏi kiến trúc neuron chuyên biệt có thể tận dụng cấu trúc này trong khi vẫn khả vi và huấn luyện được qua backpropagation.

Thách thức cơ bản mà đồ thị đặt ra là bất biến hoán vị (*permutation invariance*): biểu diễn của đồ thị không được phụ thuộc cách ta đánh số nút. Nếu nút được đánh số 1,2,3 hay 3,2,1, đồ thị vẫn giống nhau nên biểu diễn cũng phải giống. Điều này loại trừ cách tiếp cận ngây thơ như đưa ma trận kề vào mạng neuron chuẩn (xử lý các thứ tự khác nhau như đầu vào khác nhau). GNN giải quyết qua truyền thông điệp (*message passing*): các nút lặp lại gộp thông tin từ láng giềng qua các phép biến đổi học được, với hàm gộp (tổng, trung bình, max) bất biến hoán vị. Sau vài vòng lặp, biểu diễn mỗi nút tích hợp thông tin từ lân cận, với lân cận lớn hơn tiếp cận được qua nhiều vòng hơn.

Hiểu GNN đòi hỏi nắm rằng các tác vụ học trên đồ thị khác nhau yêu cầu đầu ra khác nhau. Phân loại nút (*node classification*) dự đoán nhãn cho nút (phân loại người dùng trong mạng xã hội). Dự đoán cạnh (*link prediction*) dự đoán cạnh thiếu hoặc tương lai (gợi ý bạn bè hoặc sản phẩm). Phân loại đồ thị (*graph classification*) dự đoán nhãn cho toàn đồ thị (phân loại phân tử hoạt/không hoạt cho thuốc). Mỗi tác vụ dùng cùng khung truyền thông điệp nhưng khác cách gộp biểu diễn nút và dự đoán gì.

Ứng dụng của GNN trải rộng nhiều miền. Trong hóa học, đồ thị phân tử (nguyên tử là nút, liên kết là cạnh) được GNN xử lý để dự đoán tính chất như độ tan hoặc độc tính. Trong mạng xã hội, GNN phát hiện cộng đồng, dự đoán tình bạn, hoặc xác định người dùng ảnh hưởng. Trong hệ thống gợi ý, đồ thị hai phần (người dùng và mặt hàng) được xử lý để dự đoán ưa thích. Trong dự đoán cấu trúc protein, GNN mô hình tương tác amino acid. Trong dự báo giao thông, mạng đường ảnh hưởng dự báo. Tính linh hoạt này chứng tỏ cấu trúc đồ thị hiện diện khắp nơi, và GNN cung cấp khung tổng quát để học từ nó.

## 2. Nền tảng toán học
Một đồ thị $$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$ gồm các nút $$\mathcal{V} = \{v_1, \ldots, v_n\}$$ và các cạnh $$\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$$. Nút có đặc trưng $$\mathbf{x}_i \in \mathbb{R}^d$$, và cạnh có thể có đặc trưng $$\mathbf{e}_{ij}$$. Ma trận kề $$\mathbf{A} \in \{0,1\}^{n \times n}$$ mã hóa cấu trúc: $$A_{ij} = 1$$ nếu cạnh $$(v_i, v_j) \in \mathcal{E}$$, ngược lại 0.

### Khung Truyền thông điệp (Message Passing)

GNN cập nhật biểu diễn nút qua truyền thông điệp lặp. Ở tầng $$k$$, biểu diễn của nút $$i$$ $$\mathbf{h}_i^{(k)}$$ được cập nhật dựa trên láng giềng:

$$\mathbf{h}_i^{(k)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \text{AGGREGATE}^{(k)}\left(\{\mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i)\}\right)\right)$$

trong đó $$\mathcal{N}(i)$$ là láng giềng của $$i$$. AGGREGATE kết hợp đặc trưng láng giềng (phải bất biến hoán vị); UPDATE kết hợp đặc trưng riêng của nút với thông tin láng giềng đã gộp.

### Graph Convolutional Networks (GCN)

GCN dùng lý thuyết đồ thị phổ, định nghĩa tích chập qua Laplacian đồ thị. Dạng thực tiễn:

$$\mathbf{H}^{(k+1)} = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(k)}\mathbf{W}^{(k)})$$

trong đó $$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$$ (kề cộng vòng lặp tự thân), $$\tilde{\mathbf{D}}$$ là ma trận bậc, $$\mathbf{H}^{(k)}$$ là biểu diễn nút ở tầng $$k$$, $$\mathbf{W}^{(k)}$$ là trọng số học được.

Phép này thực hiện gộp có trọng số các láng giềng rồi biến đổi tuyến tính và phi tuyến, với trọng số tỉ lệ nghịch với bậc nút (nút bậc cao đóng góp ít hơn trên mỗi láng giềng).

### GraphSAGE

GraphSAGE lấy mẫu lân cận kích thước cố định và dùng gộp học được:

$$\mathbf{h}_{\mathcal{N}(i)}^{(k)} = \text{AGGREGATE}(\{\mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i)\})$$

$$\mathbf{h}_i^{(k)} = \sigma(\mathbf{W}^{(k)} \cdot [\mathbf{h}_i^{(k-1)}, \mathbf{h}_{\mathcal{N}(i)}^{(k)}])$$

AGGREGATE có thể là mean, max, hoặc LSTM trên lân cận. Điều này cho phép huấn luyện mini-batch và xử lý lân cận kích thước biến thiên.

## 3. Ví dụ / Trực giác

Xét mạng xã hội: nút là người, cạnh là tình bạn, đặc trưng nút gồm tuổi, vị trí, sở thích. Ta muốn dự đoán người dùng nào sẽ thích sản phẩm mới (phân loại nút).

GNN 2 tầng hoạt động như sau. Ban đầu, biểu diễn mỗi người dùng $$\mathbf{h}_i^{(0)}$$ là đặc trưng thô. Sau tầng 1, $$\mathbf{h}_i^{(1)}$$ tích hợp thông tin từ bạn bè trực tiếp (láng giềng 1-hop). Biểu diễn của người dùng A giờ gồm “bạn tôi tuổi 25–30, hầu hết ở đô thị, quan tâm công nghệ” — thông tin láng giềng đã gộp. Sau tầng 2, $$\mathbf{h}_i^{(2)}$$ tích hợp bạn-của-bạn (lân cận 2-hop). Biểu diễn của A nắm bắt ngữ cảnh xã hội rộng hơn.

Cho dự đoán, mạng học rằng người dùng có lân cận với mẫu nhất định (nhiều bạn quan tâm công nghệ, tụm đô thị) có khả năng thích sản phẩm công nghệ. GNN cung cấp đặc trưng nắm bắt cả thuộc tính cá nhân lẫn ngữ cảnh xã hội cho phân loại.

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
GNN gắn với lý thuyết đồ thị phổ qua nền tảng toán học. Laplacian đồ thị và phân rã trị riêng cung cấp cơ sở lý thuyết để định nghĩa tích chập trên đồ thị, tổng quát hóa tích chập của CNN từ lưới đều sang đồ thị tùy ý.

## 6. Các Bài báo Nền tảng

**["Semi-Supervised Classification with Graph Convolutional Networks" (2017)](https://arxiv.org/abs/1609.02907)**  
*Tác giả*: Thomas Kipf, Max Welling  
Giới thiệu GCN, thiết lập truyền thông điệp trên đồ thị như cách tiếp cận học sâu hiệu quả. Chứng minh phân loại nút bán giám sát dùng cấu trúc đồ thị cộng nhãn hạn chế.

**["Inductive Representation Learning on Large Graphs" (2017)](https://arxiv.org/abs/1706.02216)**  
*Tác giả*: William Hamilton, Rex Ying, Jure Leskovec  
GraphSAGE cho phép học trên đồ thị lớn qua lấy mẫu lân cận, cho phép huấn luyện mini-batch. Mở rộng GNN từ chuyển dẫn (*transductive* — đồ thị cố định) sang quy nạp (*inductive* — tổng quát hóa sang nút/đồ thị mới).

**["Graph Attention Networks" (2018)](https://arxiv.org/abs/1710.10903)**  
*Tác giả*: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio  
GAT dùng cơ chế attention để cân trọng đóng góp láng giềng, học tầm quan trọng của các láng giềng khác nhau. Linh hoạt hơn gộp cố định.

## Bẫy Thường gặp và Mẹo

Làm mịn quá mức (*oversmoothing*) xảy ra với nhiều tầng GNN — biểu diễn nút trở nên không phân biệt được khi chúng gộp từ lân cận ngày càng lớn. Dùng kết nối bỏ qua (*skip connection*), chuẩn hóa batch, hoặc chọn độ sâu cẩn thận (2–3 tầng thường đủ).

## Điểm Chính Cần Nhớ

Mạng neuron đồ thị xử lý dữ liệu cấu trúc đồ thị qua truyền thông điệp, cập nhật lặp biểu diễn nút bằng cách gộp thông tin từ láng giềng qua các hàm học được, bất biến hoán vị. GCN dùng ma trận kề chuẩn hóa cho tích chập đồ thị phổ. GraphSAGE lấy mẫu lân cận để mở rộng quy mô. Mạng attention đồ thị học tầm quan trọng láng giềng qua attention. Ứng dụng trải mạng xã hội, phân tử, đồ thị tri thức và hệ thống gợi ý. GNN cho phép học sâu trên dữ liệu quan hệ, không đều nơi CNN và RNN không áp dụng được, mở các miền cấu trúc đồ thị cho AI hiện đại.
