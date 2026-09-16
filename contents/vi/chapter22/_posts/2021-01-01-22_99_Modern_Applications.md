---
layout: post
title: 22-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '22'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter22
lesson_type: optional
---

# Tùy chọn: GNN sau truyền tin — Graph Transformer và AlphaFold 3

> Bài này là **tùy chọn**. Nó **không** thay truyền tin, bất biến hoán vị, hay cập nhật kiểu GCN. Nó phủ các mô hình 2022–2025 trộn **attention trên đồ thị** với lớp MPNN cổ điển, cộng sản phẩm dự đoán cấu trúc khiến GNN nổi ngoài recsys.

Bước MPNN

$$\mathbf{h}_i^{(t+1)} = \mathrm{UPD}\Big(\mathbf{h}_i^{(t)},\; \mathrm{AGG}_{j\in\mathcal{N}(i)} \mathrm{MSG}(\mathbf{h}_i^{(t)},\mathbf{h}_j^{(t)},e_{ij})\Big)$$

vẫn là thuật toán đầu đúng. Nhiều mô hình SOTA giờ cho **mọi nút attend nhiều nút khác** (Graphormer, GraphGPS) khi đồ thị đủ nhỏ (phân tử, protein).

## 1. Graph Transformer

[Ying et al., 2021](https://arxiv.org/abs/2106.05234) (Graphormer) và [Rampášek et al., 2022](https://arxiv.org/abs/2205.12454) (GraphGPS) thêm mã hóa Laplacian / không gian để Transformer tôn trọng khoảng cách đồ thị. Dùng trên phân tử (hàng trăm nút), không trên đồ thị xã hội 10M nút — ở đó vẫn cần lấy mẫu láng giềng (GraphSAGE) như ghi chú lý thuyết.

## 2. Ứng dụng cụ thể

### AlphaFold 3

[Abramson et al., 2024](https://www.nature.com/articles/s41586-024-07487-w) (AlphaFold 3) dự đoán phức sinh phân tử bằng mô-đun **khuếch tán** trên tọa độ nguyên tử cộng thân cặp/token. Thân cặp là hậu duệ trí tuệ của attention-trên-đồ-thị từ AlphaFold 2. Ứng dụng: khám phá thuốc và sinh học cấu trúc, không phải công thức GCN mới.

### Recsys và knowledge graph

Hệ gợi ý production vẫn chạy **GNN hai phía** hoặc embedding two-tower (họ PinSage). Paper 2023–2025 thêm đặc trưng LLM trên nút; bộ tổng hợp vẫn là GNN.

### Phần mềm

- [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) 2.x — `TransformerConv`, ví dụ GraphGPS.
- [dmlc/dgl](https://github.com/dmlc/dgl).
- [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) (mã suy luận, 2024+).

## 3. Trích dẫn (2022–2026)

- [GraphGPS (Rampášek et al., 2022)](https://arxiv.org/abs/2205.12454).
- [Graphormer (Ying et al., 2021)](https://arxiv.org/abs/2106.05234).
- [AlphaFold 3 (Abramson et al., 2024)](https://www.nature.com/articles/s41586-024-07487-w).

## 4. Bài này bổ sung gì cho ghi chú cốt lõi

Viết một lớp truyền tin trước. Bài này chỉ thêm **Graph Transformer** và **AlphaFold 3** như ứng dụng trích ý tưởng chương này trong 2024–2026.
