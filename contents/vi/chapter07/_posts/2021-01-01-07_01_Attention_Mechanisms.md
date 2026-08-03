---
layout: post
title: 07-02 Cơ chế Chú ý Đi sâu
chapter: '07'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

# 07-02 Cơ chế Chú ý Đi sâu

Khối này đi **xa hơn bài cơ sở**. Người học đã biết *vì sao* attention tồn tại và các công thức cơ bản. Ở đây ta siết chặt toán học, rồi cài đặt từng thành phần chuyển động một cách cẩn thận.

## Nội dung sẽ học

1. **Toán học Attention**  
   Hàm điểm số thống nhất, dạng QKV véc-tơ hóa, self-attention như tương tác $$L\times L$$, đại số multi-head, mask padding và causal như phép toán trước softmax, độ phức tạp và lập luận độ dài đường đi.

2. **Cài đặt Attention**  
   Bahdanau và scaled attention từ đầu, module multi-head, demo masking, phác thảo seq2seq-with-attention tối giản, papers, và cạm bẫy trong production.

## Cách đọc khối này

| Thứ tự | Bài | Kết quả mong đợi |
|--------|-----|------------------|
| 1 | **07-02-01 Toán học Attention** | Viết được mọi công thức mà không cần tra cứu |
| 2 | **07-02-02 Cài đặt** (tổng quan) | Ánh xạ toán → các module code |
| 3 | **07-02-02-01 Cài đặt cốt lõi** | Code Bahdanau + self-attention chạy được |
| 4 | **07-02-02-02 Multi-Head, Papers, Cạm bẫy** | Multi-head, causal mask, tài liệu, checklist debug |

Nếu một công thức trong 07-02-01 còn trừu tượng, hãy nhảy sang demo tương ứng ở 07-02-02-01 rồi quay lại. Toán và code được thiết kế để củng cố lẫn nhau, không cạnh tranh.

## Liên hệ với phần còn lại của khóa học

- **Quay lại 07-01** để lấy trực giác và động lực lịch sử.  
- **Tiến tới Chương 08 (Transformer)** để xếp chồng self-attention với khối feed-forward, residual và positional encoding.

Khi sẵn sàng, tiếp tục với **Toán học Attention**.
