---
layout: post
title: 26-02 Prompting, học trong ngữ cảnh và tuân thủ chỉ dẫn
chapter: '26'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter26
---

# Prompting, học trong ngữ cảnh và tuân thủ chỉ dẫn

Sau tiền huấn luyện, trọng số đã hiện thực một máy next-token. **Prompting** là cách rẻ nhất để lái máy đó: bạn đổi tiền tố $$x_{<t}$$, không đổi $$\theta$$. **Tinh chỉnh** (*fine-tuning*) đổi $$\theta$$. Hầu hết sản phẩm LLM dùng cả hai, theo thứ tự đó.

Bài này tách ba giao diện mà phỏng vấn và blog thường trộn: prompting zero-shot, học trong ngữ cảnh (few-shot), và tuân thủ chỉ dẫn sau thích nghi có giám sát.

## 1. Prompt là một phần của chuỗi

Vì $$p_\theta(x_t \mid x_{<t})$$ điều kiện trên *mọi thứ* trong cửa sổ, system prompt, tài liệu truy hồi, kết quả công cụ, và câu hỏi của người dùng chỉ là thêm token. Kiến trúc không có cổng “nhập prompt” riêng. Quy ước định dạng (khuôn chat, thẻ giống XML, schema JSON) quan trọng vì chúng là **tiền tố ổn định** mà mô hình đã thấy, hoặc sau này được huấn luyện để chờ.

Thói quen hữu ích: khi mô hình “bỏ qua chỉ dẫn,” hãy kiểm tra trước xem chỉ dẫn còn trong cửa sổ ngữ cảnh không và khuôn chat có khớp checkpoint đã thích nghi không. Đó thường là lỗi dữ liệu/định dạng hơn là lỗi kiến trúc.

## 2. Zero-shot so với học trong ngữ cảnh

**Zero-shot.** Bạn mô tả tác vụ bằng ngôn ngữ (“Dịch sang tiếng Việt:”) rồi giải mã. Mô hình khái quát từ các mẫu tiền huấn luyện trông giống lời yêu cầu đó.

**Học trong ngữ cảnh (ICL) / few-shot.** Bạn đặt vài cặp input–output vào tiền tố, rồi input mới. [Brown et al., 2020](https://arxiv.org/abs/2005.14165) (GPT-3) cho thấy ở quy mô đủ lớn việc này có thể trông như học tác vụ *mà không có bước gradient*. Cơ chế vẫn là dự đoán token kế: các ví dụ làm phần tiếp nối mong muốn likelier.

ICL **không** giống tinh chỉnh:

| | Học trong ngữ cảnh | Tinh chỉnh cổ điển |
| --- | --- | --- |
| Thứ đổi | Token trong cửa sổ | Tham số $$\theta$$ (hoặc adapter) |
| Bền vững | Mất khi prompt biến | Sống qua các request |
| Chi phí | Prefill / cache thêm | Compute huấn luyện, rồi prompt rẻ hơn |
| Dữ liệu | Vài ví dụ sửa được ngay | Một tập dữ liệu và bộ tối ưu |

Khi nên chọn ICL: tác vụ mới, nhãn ít, hoặc phải đổi hành vi theo từng request (khách khác, schema khác). Khi nên chọn tinh chỉnh: hành vi phải ổn định, prompt sẽ phình to, hoặc bạn cần chuyên gia nhỏ/nhanh hơn (thường sau chưng cất — **26-05**).

**Chuỗi suy nghĩ** (*chain-of-thought*, [Wei et al., 2022](https://arxiv.org/abs/2201.11903)) là một mẫu prompting: yêu cầu mô hình viết các bước trung gian trước đáp án. Nó không thêm module mới; nó phân bổ token giải mã cho một phép tính dài hơn, dễ kiểm hơn. Các mô hình “lý luận” sau này *huấn luyện* thói quen đó (Chương 25-99). Đừng nhầm mẹo prompt với kiến trúc mới.

## 3. Tuân thủ chỉ dẫn thường không phải tiền huấn luyện thô

LM gốc hoàn thiện văn bản. Nếu tiền tố là một bài blog, nó viết tiếp bài blog. Người dùng muốn một **trợ lý hữu ích**: làm theo yêu cầu, từ chối khi thích hợp, giữ giọng nhất quán.

Bước thích nghi chuẩn đầu tiên là **tinh chỉnh có giám sát (SFT)** trên các cặp (chỉ dẫn, câu trả lời) — minh họa do người viết hoặc lọc, hoặc do mô hình mạnh hơn sinh rồi được biên tập. Loss vẫn là cross-entropy token kế, nhưng *phân bố dữ liệu* giờ là “trả lời yêu cầu này,” không phải “viết tiếp web.”

Sau SFT, prompting vẫn quan trọng (công cụ, RAG, phong cách), nhưng mô hình dễ coi văn bản người dùng là *tác vụ* hơn là tài liệu cần nối dài. Tối ưu sở thích (bài sau) rồi xếp hạng *câu trả lời hợp lệ nào* được ưa hơn.

Học chuyển giao cổ điển (Chương 15) cập nhật backbone cho một đầu ra và tập nhãn cố định. SFT chỉ dẫn cập nhật cùng đầu sinh trên nhiều tác vụ viết bằng ngôn ngữ. Cùng họ tối ưu; khác giao diện.

## 4. Checklist prompting ngắn (tự học)

Chép các mục này và thử trên bất kỳ mô hình instruct mở nào:

1. Nêu **vai trò** và **hợp đồng đầu ra** (văn bản thường, khóa JSON, ngôn ngữ).
2. Đặt **ràng buộc không thương lượng** trước ngữ cảnh truy hồi dài (chúng có thể bị đẩy ra khỏi $$L$$ nếu bạn append cuối).
3. Với ICL, giữ ví dụ **đồng khuôn** với câu hỏi bạn sẽ hỏi.
4. Nếu cần nguồn, truy hồi trước (Chương 18-99); đừng bảo mô hình bịa trích dẫn.
5. Nếu cần hành vi bền, hãy lên kế hoạch SFT / adapter — đừng để prompt 8k token phình mãi.

## Điểm chính cần nhớ

- Prompting sửa tiền tố điều kiện; tinh chỉnh sửa trọng số.
- ICL là few-shot *trong cửa sổ*, không phải bộ tối ưu vòng trong ẩn bạn được miễn phí.
- Tuân thủ chỉ dẫn chủ yếu là đổi **dữ liệu** (SFT) trên cùng loss next-token.
- Sở thích căn chỉnh là một đổi tiếp — bài **26-03** — không thay cho một prompt rõ.
