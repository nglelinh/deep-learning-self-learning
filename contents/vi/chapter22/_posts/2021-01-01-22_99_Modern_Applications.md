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

# Tùy chọn: GNN sau truyền thông điệp — Graph Transformer và AlphaFold 3

> Bài này là **tùy chọn**, **không** thay thế lý thuyết về truyền thông điệp, tính bất biến hoán vị hay phép cập nhật GCN. Nội dung bổ sung các mô hình giai đoạn 2022–2025 kết hợp **attention trên đồ thị** với lớp truyền thông điệp cổ điển, cùng ứng dụng dự đoán cấu trúc sinh học nổi bật ngoài hệ thống gợi ý.

Công thức truyền thông điệp (MPNN):

$$\mathbf{h}_i^{(t+1)} = \mathrm{UPD}\Big(\mathbf{h}_i^{(t)},\; \mathrm{AGG}_{j\in\mathcal{N}(i)} \mathrm{MSG}(\mathbf{h}_i^{(t)},\mathbf{h}_j^{(t)},e_{ij})\Big)$$

vẫn là nền tảng cốt lõi. Tuy nhiên, nhiều mô hình hiện đại cho phép **mọi nút chú ý trực tiếp đến nhiều nút khác** (như Graphormer, GraphGPS) khi kích thước đồ thị đủ nhỏ (ví dụ phân tử hoặc protein).

## 1. Graph Transformer

[Ying et al., 2021](https://arxiv.org/abs/2106.05234) (Graphormer) và [Rampášek et al., 2022](https://arxiv.org/abs/2205.12454) (GraphGPS) bổ sung mã hóa vị trí Laplacian hoặc mã hóa khoảng cách đồ thị để Transformer tôn trọng cấu trúc liên kết. Hướng này hiệu quả trên các đồ thị kích thước nhỏ đến trung bình (như phân tử có vài chục đến vài trăm nút), nhưng không phù hợp với đồ thị mạng xã hội hàng chục triệu nút — nơi kỹ thuật lấy mẫu láng giềng kiểu GraphSAGE trong bài lý thuyết vẫn cần thiết.

## 2. Ứng dụng cụ thể

### AlphaFold 3

[Abramson et al., 2024](https://www.nature.com/articles/s41586-024-07487-w) (AlphaFold 3) dự đoán cấu trúc phức hợp sinh học bằng mô-đun **khuếch tán** trên tọa độ nguyên tử kết hợp với nhánh biểu diễn cặp (*pairformer*) kế thừa ý tưởng *attention* trên đồ thị từ AlphaFold 2. Đây là ứng dụng tiêu biểu trong khám phá thuốc và sinh học cấu trúc, phát triển từ cùng nền tảng biểu diễn quan hệ cặp.

### Hệ thống gợi ý và đồ thị tri thức

Các hệ thống gợi ý thực tế vẫn sử dụng phổ biến **GNN trên đồ thị hai phía** hoặc mô hình nhúng *two-tower* (như họ PinSage). Các nghiên cứu giai đoạn 2023–2025 thường bổ sung đặc trưng trích xuất từ LLM cho từng nút, trong khi tầng tổng hợp cấu trúc vẫn dựa trên GNN.

### Phần mềm phổ biến

- [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) 2.x — hỗ trợ `TransformerConv` và ví dụ GraphGPS.
- [dmlc/dgl](https://github.com/dmlc/dgl).
- [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) (mã suy luận phát hành từ 2024).

## 3. Trích dẫn (2022–2026)

- [GraphGPS (Rampášek et al., 2022)](https://arxiv.org/abs/2205.12454).
- [Graphormer (Ying et al., 2021)](https://arxiv.org/abs/2106.05234).
- [AlphaFold 3 (Abramson et al., 2024)](https://www.nature.com/articles/s41586-024-07487-w).

## 4. Bài này bổ sung gì cho ghi chú cốt lõi

Trước hết nên tự cài đặt một lớp truyền thông điệp cơ bản từ bài cốt lõi. Bài này bổ sung **Graph Transformer** và **AlphaFold 3** như những ví dụ tiêu biểu ứng dụng ý tưởng của chương trong giai đoạn 2024–2026.

## Video tham khảo

- **Xem trước** [Graph Neural Networks Explained: A Clear Guide to GNN Basics & Models](https://www.youtube.com/watch?v=eGoszzMkGfU) — IBM Technology, tiếng Anh. Ôn lại cơ chế truyền thông điệp mà Graph Transformer mở rộng bằng trọng số *attention* học được.
- [Message Passing: How Graph Neural Networks Actually Work](https://www.youtube.com/watch?v=1DmNdMUbDkg) — The Synthetic Mind, tiếng Anh. Làm rõ một vòng truyền thông điệp cục bộ để đối chiếu với khối Graph Transformer.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, tiếng Anh. Giải thích cơ chế *self-attention* toàn cục khi nút có thể trao đổi thông tin vượt ra ngoài tập láng giềng trực tiếp.
