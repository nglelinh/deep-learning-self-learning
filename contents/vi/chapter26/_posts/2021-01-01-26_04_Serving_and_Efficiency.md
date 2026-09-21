---
layout: post
title: 26-04 Phục vụ và hiệu năng cho LLM
chapter: '26'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter26
---

# Phục vụ và hiệu năng cho LLM

Một decoder đã tiền huấn luyện, đã dạy chỉ dẫn, đã căn chỉnh vẫn đắt **theo từng token sinh ra**. Huấn luyện là hóa đơn một lần (hoặc hiếm); phục vụ là hóa đơn tăng theo người dùng. Bài này liệt kê các nút bạn nên gọi tên trước khi với tới chưng cất.

Toán nén — lưới lượng tử, điểm cắt tỉa, KD cổ điển — ở lại **Chương 23**. Các công thức riêng cho LLM (GPTQ, AWQ, GGUF, speculative decoding) ở **23-99**. Ở đây ta chỉ trả lời: *thứ gì đắt lúc giải mã, và khi nào học sinh nhỏ hơn là bước đúng tiếp theo?*

## 1. Prefill so với decode

Sinh tự hồi quy có hai pha:

**Prefill.** Toàn bộ prompt (hệ thống + lịch sử + người dùng + văn bản truy hồi) được tiêu thụ song song. Bạn trả attention trên độ dài prompt một lần và ghi **KV-cache**: với mỗi tầng và mỗi token prompt, lưu vector key và value.

**Decode.** Bạn phát từng token mới một. Không cache, bạn sẽ tính lại attention từ đầu trên tiền tố đang dài (xấp xỉ $$O(t^2)$$ tại bước $$t$$). Có cache, bạn append một $$k,v$$ mới và attention trên tiền tố đã lưu — khoảng $$O(t)$$ công attention mỗi token mới, cộng MLP.

Đó chính là KV-cache đã được nêu ở **08-99**. Hệ phục vụ (vLLM, SGLang, llama.cpp) chủ yếu là: *quản nhiều cache, gom nhân ma trận, và chồng I/O*.

Bộ nhớ, không chỉ FLOP, thường chiếm ưu thế. Cache cho một hội thoại dài là

$$\mathrm{bytes} \approx 2 \cdot L \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{kv\_heads}} \cdot d_{\mathrm{head}} \cdot b$$

(hệ số 2 là key và value; $$b$$ là số byte mỗi phần tử; grouped-query attention thu nhỏ $$n_{\mathrm{kv\_heads}}$$). Đó là lý do ngữ cảnh dài là quyết định **sản phẩm**, không chỉ quyết định chất lượng.

## 2. Lượng tử hóa là thắng lợi rẻ đầu tiên

Lưu trọng số 16-bit thay vì 32-bit, rồi 8-bit hoặc 4-bit, cắt bộ nhớ và có thể cắt thời gian decode bị giới hạn băng thông. Chương 23 cho hình dung làm tròn; **23-99** gọi tên GPTQ / AWQ / GGUF như các công thức người ta thực sự chạy trên decoder 7B–70B.

Lượng tử hóa **không** huấn luyện mô hình mới. Nó xấp xỉ cùng $$\theta$$ bằng ít bit hơn. Dùng khi:

- bạn sở hữu hoặc được phép phân phối checkpoint đó,
- sụt chất lượng chấp nhận được sau một vòng đánh giá nhanh,
- bạn cần *cùng* hành vi với RAM thấp hơn (laptop, một GPU, biên).

Nếu 4-bit vẫn trượt mục tiêu độ trễ hoặc chi phí, bạn cần ít tầng/độ rộng hơn hoặc ít tham số kích hoạt hơn (định tuyến MoE — cũng ở 23-99) — hoặc một **học sinh**.

## 3. Các đòn bẩy phục vụ khác (để không quá thiên về chưng cất)

- **Gom batch.** Continuous batching lấp chỗ trống GPU khi các chuỗi kết thúc khác lúc.
- **Speculative decoding.** Một mô hình nháp rẻ đề xuất vài token; mô hình lớn xác nhận một tiền tố trong một lượt forward song song ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)). Cùng phân bố nếu cài đúng; thêm độ phức tạp.
- **Cache prompt / chia sẻ tiền tố.** Nhiều request chung system prompt; tái sử dụng prefill đó.
- **Truy hồi vs ngữ cảnh dài.** Nhét cả cuốn sách vào $$L$$ thường chậm hơn và kém hơn truy hồi (18-99).
- **Adapter (LoRA / QLoRA).** Chuyên biệt hóa mà không phục vụ thêm một bản sao đầy đủ cho mọi chuyên gia (Chương 15 tùy chọn).

Không mục nào *chuyển tri thức vào một kiến trúc nhỏ hơn*. Chúng làm giáo viên rẻ hơn khi chạy.

## 4. Khi nào nên chưng cất

Với tới chưng cất (**26-05**) khi ít nhất một điều sau đúng:

1. **Kiến trúc phải thu nhỏ.** Lượng tử hóa không xóa được tầng. Giáo viên 70B phải thành học sinh 7–8B (hoặc mobile) cần một $$\theta$$ mới.
2. **Bạn muốn tầng sản phẩm rẻ hơn** với *hành vi* tương tự, không phải bản sao bit-exact — trong nhà, trên giáo viên bạn được phép huấn luyện từ đó.
3. **Bạn sẽ tinh chỉnh nhiều chuyên gia.** Chưng cất một học sinh tổng quát một lần, rồi LoRA từng chuyên gia.
4. **Nội tại hộp trắng sẵn có** và bạn muốn khớp hidden state hoặc phân bố token, không chỉ văn bản.

**Đừng** chưng cất trước nếu chưa thử trọng số 4-bit, giới hạn ngữ cảnh hợp lý, và gom batch. Chưng cất là dự án huấn luyện; lượng tử hóa thường chỉ là một cờ chuyển đổi.

Cũng đừng chưng cất giáo viên mà bạn không được cấp phép hoặc không có hợp đồng để huấn luyện từ đó. Bài sau tách chưng cất trong nhà / giấy phép mở thông thường khỏi việc gặt một API bạn không kiểm soát.

## Điểm chính cần nhớ

- Chi phí decode = MLP + attention trên **KV-cache** đang lớn dần.
- Lượng tử hóa và gom batch là công cụ hiệu năng mặc định (Chương 23 / 23-99).
- Chưng cất là công cụ cho một mạng *mới, nhỏ hơn* *bắt chước* giáo viên.
- Quyết định “cache / lượng tử / truy hồi” trước “huấn luyện học sinh.”
