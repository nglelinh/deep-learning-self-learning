---
layout: post
title: 26-06 Luyện phỏng vấn
chapter: '26'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter26
lesson_type: optional
---

# Tùy chọn: câu hỏi phỏng vấn LLM

> Bài này **tùy chọn**. Nó **không** thay 26-01–26-05. Các câu do khóa viết. Chúng **không** lấy từ Kashani & Ivry, *Deep Learning Interviews*, hay từ blog nào.

Dùng chúng khi đóng sách, sau khi bạn phác được loss token kế, ICL so với SFT, pipeline SFT → sở thích, KV-cache, và ba họ chưng cất. Bản đồ luyện toàn khóa vẫn ở **01-98**.

## P1. Đang huấn luyện thực sự cái gì?

Bạn có hai phút. Định nghĩa LLM theo nghĩa chương này trong một câu, rồi nêu loss tiền huấn luyện và điều mà loss đó *không* bảo đảm.

**Gợi ý.** Chỉ-decoder, token kế, không phải chân lý.

**Thảo luận.** Một LM Transformer nhân quả huấn luyện bằng cross-entropy theo token trên ngữ liệu. Loss thưởng likelihood ngữ liệu, không phải tính đúng sự kiện, không phải tuân thủ chỉ dẫn, và không phải dấu chân bộ nhớ nhỏ. Những thứ đó đến từ các thích nghi sau (26-02–26-04).

## P2. Prompt so với tinh chỉnh so với truy hồi

Một đồng đội muốn “mô hình luôn trả lời theo schema JSON hóa đơn của công ty.” Cho một cách prompting, một cách cập nhật trọng số, và một lý do truy hồi vẫn có thể cần.

**Gợi ý.** Khuôn / ICL; SFT hoặc LoRA; sự kiện vs định dạng.

**Thảo luận.** Schema cố định thuộc về prompt hoặc SFT/LoRA để định dạng ổn. Truy hồi (18-99) dành cho *điều khoản hay SKU nào* tồn tại tuần này — căn chỉnh và prompting không bịa được catalog sản phẩm.

## P3. Vì sao không luôn chưng cất logit?

Nêu hai lý do một nhóm chỉ có API chat lưu trữ sẽ chọn **chưng cất phản hồi** thay vì KD đủ phân bố token, dù họ thích loss 2015 trên giấy.

**Gợi ý.** Quyền truy cập và $$V$$.

**Thảo luận.** API thường không lộ phân bố $$V$$ chiều mỗi bước. Dù có, lưu hoặc khớp vector đó trên chuỗi dài vẫn đắt; văn bản lấy mẫu là thứ bạn có thể thu *về mặt máy* (và chỉ khi điều khoản cho phép huấn luyện trên đó).

## P4. Nhiệt độ như thiết bị dạy

Một giáo viên đạt 99,9% trên đúng lớp ImageNet. Vì sao tăng $$T$$ trước KL, và điều gì sai nếu bạn để $$T$$ cực lớn rồi bỏ hẳn số hạng nhãn cứng?

**Gợi ý.** Tri thức tối vs súp đều.

**Thảo luận.** $$T$$ cao nâng khối ngoài đường chéo đáng học (tri thức tối). Nếu $$T\to\infty$$, mục tiêu tiến tới đều và học sinh học “mọi lớp như nhau,” không phải giáo viên hữu ích. Số hạng CE cứng (hoặc $$T$$ vừa) neo mode. Cùng trực giác áp cho phân bố token nhọn, nên KD logit cho LM thường dùng top-$$k$$ thay vì chỉ $$T$$ rất lớn.

## P5. Phục vụ trước học sinh

Nêu hai thay đổi bạn sẽ thử **trước khi** huấn luyện học sinh 8B từ giáo viên 70B bạn sở hữu, và một tình huống bạn sẽ bỏ qua chúng và vẫn chưng cất.

**Gợi ý.** 26-04 rồi 26-05.

**Thảo luận.** Lượng tử (4-bit / GGUF), cắt ngữ cảnh không dùng, gom batch, hoặc speculate. Vẫn chưng cất khi *kiến trúc* phải thu (bộ nhớ biên, SKU rẻ hơn, nhiều chuyên gia LoRA trên một base nhỏ) hoặc khi bạn muốn một $$\theta$$ mới bắt chước hành vi chứ không phải cùng checkpoint với ít bit hơn.

## P6. Quyền, không phải tiêu đề

Trong bốn câu, đối chiếu chưng cất hợp lệ với việc gặt không được phép **mà không** nêu vụ kiện hay tin 2026.

**Gợi ý.** Ai sở hữu giáo viên; API để làm gì.

**Thảo luận.** Nếu bạn sở hữu giáo viên hoặc giấy phép mời học sinh trong họ / dữ liệu tổng hợp, chưng cất là công cụ nén và tầng sản phẩm. Nếu bạn chỉ có API khách, văn bản là đầu ra dịch vụ; nhiều điều khoản cấm dùng nó làm ngữ liệu huấn luyện cho mô hình cạnh tranh. Căng thẳng là cấu trúc: API hữu ích rò năng lực qua cùng token mà họ bán. Đánh giá cáo buộc công khai nằm ngoài khóa này.

## Cách dùng

Nói đáp án thành tiếng trong 90 giây, rồi đối chiếu phần thảo luận. Nếu cần toán RL, mở **21-99**, không phải trang này. Nếu cần công thức KD cổ điển, mở **23-01** và **26-05**.
