---
layout: post
title: 09-02 Chuẩn hóa Batch
chapter: '09'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

Mục này trình bày kỹ thuật chuẩn hóa theo lô (*Batch Normalization*) qua hai bài:

1. **Cơ chế** — khái niệm, công thức lượt xuôi/lượt ngược, cài đặt bằng NumPy và lý do kỹ thuật này ổn định quá trình huấn luyện.
2. **Vị trí đặt, Biến thể và Mẹo** — BatchNorm2D cho CNN, các biến thể thay thế (LayerNorm, InstanceNorm, GroupNorm), vấn đề thường gặp và kinh nghiệm thực hành.

## Video tham khảo

- **Xem trước:** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI (tiếng Anh). Giải thích lý do chuẩn hóa theo mini-batch cải thiện đáng kể quá trình huấn luyện.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy (tiếng Anh). Cài đặt thực tế BatchNorm trong vòng huấn luyện và phân tích tương tác với hàm kích hoạt.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering (tiếng Anh). Đặt BatchNorm trong hệ thống các kỹ thuật tối ưu và huấn luyện mạng sâu.
