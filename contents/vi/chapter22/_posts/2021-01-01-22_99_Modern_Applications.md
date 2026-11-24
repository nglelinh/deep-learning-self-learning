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

Năm 2024, giải Nobel Hóa học gọi tên đột phá dự đoán cấu trúc protein — và ở trái tim của cuộc cách mạng khoa học đó (từ AlphaFold 2 đến **AlphaFold 3**) là sự giao thoa giữa **học sâu hình học trên đồ thị**, **Graph Transformer** và **mô hình khuếch tán 3D**. Bài cập nhật này cho thấy GNN hiện đại đã vượt xa khung truyền thông điệp cục bộ để chinh phục từ thiết kế dược phẩm đến đồ thị tri thức (GraphRAG) và hệ thống gợi ý hàng tỷ nút.

## Bản đồ kiến thức nền: Ba mặt trận hiện đại của Học sâu trên Đồ thị (2022–2026)

- **Graph Transformer (Graphormer, GPS):** Cho phép mọi cặp đỉnh chú ý trực tiếp tới nhau (giải quyết tắc nghẽn *oversquashing* tầm xa) bằng cách nhúng khoảng cách đường đi ngắn nhất và vector riêng Laplacian vào điểm chú ý.
- **AlphaFold 3 và học sâu sinh học phân tử:** Kết hợp khối *Pairformer* (cập nhật biểu diễn cặp nguyên tử/axit amin thỏa mãn bất đẳng thức tam giác) với bộ khuếch tán sinh trực tiếp tọa độ 3D của phức hợp protein–DNA–thuốc.
- **GNN quy mô web trong hệ thống gợi ý (PinSage, LightGCN):** Mô hình hóa tương tác người dùng–sản phẩm thành đồ thị hai phía để lan truyền sở thích bậc cao mà lọc cộng tác cổ điển bỏ sót.
- **Đồ thị tri thức & GraphRAG:** Kết hợp truy hồi đa bước trên đồ thị thực thể–quan hệ với LLM để trả lời các câu hỏi tổng hợp xuyên suốt hàng nghìn tài liệu.

### Khung giải thích (nên đọc chậm): Tính tay thiên lệch khoảng cách đồ thị (Spatial Encoding) trong Graph Transformer

Trong Graphormer (Graph Transformer), điểm chú ý giữa đỉnh $$i$$ và đỉnh $$j$$ được bổ sung một số hạng thiên lệch học được $$b_{\phi(i, j)}$$ phụ thuộc vào **độ dài đường đi ngắn nhất** $$\phi(i, j)$$ giữa hai đỉnh:
$$A_{ij} = \frac{(Q_i \cdot K_j)}{\sqrt{d_k}} + b_{\phi(i, j)}.$$

Giả sử trên một phân tử mạch thẳng 4 nguyên tử $$1 - 2 - 3 - 4$$, tích vô hướng $$\frac{Q_1 \cdot K_j}{\sqrt{d_k}}$$ của đỉnh 1 với các đỉnh $$2, 3, 4$$ đều bằng $$1{,}0$$, còn bảng thiên lệch khoảng cách đã học là $$b_1 = +1{,}2$$ (kề trực tiếp), $$b_2 = +0{,}3$$ (cách 2 bước), $$b_3 = -0{,}5$$ (cách 3 bước).

1. **Điểm chú ý sau khi cộng cấu trúc đồ thị:**
   - Đỉnh 2 (cách 1): $$1{,}0 + 1{,}2 = 2{,}2$$
   - Đỉnh 3 (cách 2): $$1{,}0 + 0{,}3 = 1{,}3$$
   - Đỉnh 4 (cách 3): $$1{,}0 - 0{,}5 = 0{,}5$$
2. **Xác suất Softmax:** $$e^{2{,}2} \approx 9{,}03$$, $$e^{1{,}3} \approx 3{,}67$$, $$e^{0{,}5} \approx 1{,}65$$ (tổng $$14{,}35$$) → trọng số lần lượt là $$62{,}9\%$$, $$25{,}6\%$$ và $$11{,}5\%$$.
   Nhờ cơ chế này, đỉnh 1 vừa ưu tiên láng giềng gần, vừa **vẫn trực tiếp nhận $$11{,}5\%$$ thông tin từ đỉnh 4 chỉ trong 1 tầng duy nhất** mà không bị nghẽn cổ chai qua đỉnh 2 và 3!

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
