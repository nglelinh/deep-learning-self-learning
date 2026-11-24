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

Một trong những phát hiện làm ngỡ ngàng chính các tác giả GPT-3 vào năm 2020 là **Học trong Ngữ cảnh (In-Context Learning — ICL)**: không hề chạy một bước lan truyền ngược hay cập nhật bất kỳ trọng số nào, chỉ cần bạn đặt 3 ví dụ minh họa ngay trong phần mở đầu câu lệnh, mô hình bỗng "hiểu luật chơi" và giải đúng mẫu thứ 4! Bài học này giải mã bản chất thực sự của Prompting, sự khác biệt giữa mô hình nền (base model) với mô hình đã qua SFT, và cách viết prompt như một kỹ sư hệ thống.

## Bản đồ kiến thức nền: Bốn trụ cột của Prompting và Học trong Ngữ cảnh (ICL)

- **Prompt là điều kiện hóa xác suất trên chuỗi đầu vào:** Về mặt toán học, mọi kỹ thuật viết prompt đều là nghệ thuật chọn tiền tố $$x_{1:k}$$ sao cho phân phối có điều kiện $$P_\theta(x_{k+1:T} \mid x_{1:k})$$ tập trung vào vùng lời giải mong muốn.
- **Zero-shot, Few-shot và In-Context Learning:** Cung cấp cặp (đầu vào → đầu ra) trong ngữ cảnh giúp các đầu chú ý (như Induction Heads) tự căn chỉnh định dạng, phân phối nhãn và phong cách mà không đổi trọng số.
- **Vai trò quyết định của SFT trong tuân thủ chỉ dẫn:** Hiểu vì sao mô hình nền thuần tiền huấn luyện thường *viết tiếp* câu hỏi của bạn thay vì trả lời, còn mô hình hậu huấn luyện SFT biết đóng vai trợ lý tuân thủ chỉ dẫn.
- **Kỹ thuật thiết kế Prompt có hệ thống:** Chỉ rõ vai trò, ràng buộc định dạng đầu ra, phân tách rõ dữ liệu với chỉ dẫn (chống prompt injection) và cung cấp ví dụ biên (edge cases).

### Khung giải thích (nên đọc chậm): Phân tích sự thay đổi phân phối xác suất token khi chuyển từ Zero-shot sang Few-shot ICL

Xét bài toán phân loại cảm xúc đánh giá sản phẩm chỉ được trả về đúng 1 từ trong tập `{Tích cực, Tiêu cực}`:

1. **Prompt Zero-shot mơ hồ trên mô hình nền:** `Đánh giá: Giao hàng chậm quá! Cảm xúc:`
   - Phân phối xác suất token kế tiếp bị phân tán rộng: `rất` ($$0{,}22$$), `khó` ($$0{,}18$$), `bực` ($$0{,}15$$), `Tiêu` ($$0{,}08$$), các từ khác ($$0{,}37$$) — entropy cao và dễ trả lời sai định dạng JSON/nhãn.
2. **Prompt Few-shot (2 ví dụ định khung):**
   `Đánh giá: Máy chạy êm -> Nhãn: Tích cực` + `Đánh giá: Pin tụt nhanh -> Nhãn: Tiêu cực` + `Đánh giá: Giao hàng chậm quá! -> Nhãn:`
   - Các đầu chú ý so khớp mẫu `-> Nhãn:` lặp lại 2 lần trước đó và dồn khối xác suất trực tiếp vào đúng 2 token nhãn đã thấy: `Tiêu` ($$0{,}93$$), `Tích` ($$0{,}05$$), khác ($$0{,}02$$).
   - Entropy Shannon giảm mạnh từ $$> 2{,}5\text{ nat}$$ xuống còn $$-0{,}93\ln(0{,}93) - 0{,}05\ln(0{,}05) - 0{,}02\ln(0{,}02) \approx 0{,}295\text{ nat}$$ — độ tin cậy và tính tuân thủ định dạng tăng vọt mà không cần cập nhật một trọng số nào!

# Prompting, học trong ngữ cảnh và tuân thủ chỉ dẫn

Sau giai đoạn tiền huấn luyện, bộ trọng số $$\theta$$ hiện thực một mô hình dự đoán token kế tiếp. **Prompting** là cách ít tốn kém nhất để điều hướng mô hình: ta chỉ thay đổi chuỗi tiền tố $$x_{<t}$$ mà không thay đổi $$\theta$$. Ngược lại, **tinh chỉnh** (*fine-tuning*) cập nhật trực tiếp $$\theta$$. Trong thực tế, phần lớn hệ thống LLM kết hợp cả hai cách tiếp cận này.

Bài này phân biệt rõ ba khái niệm thường bị trộn lẫn: *prompting zero-shot*, học trong ngữ cảnh (*in-context learning*, ICL / *few-shot*) và khả năng tuân thủ chỉ dẫn sau tinh chỉnh có giám sát.

## 1. Prompt là một phần của chuỗi đầu vào

Do $$p_\theta(x_t \mid x_{<t})$$ điều kiện hóa trên *toàn bộ* chuỗi trong cửa sổ ngữ cảnh, các thành phần như *system prompt*, tài liệu truy hồi, kết quả trả về từ công cụ và câu hỏi người dùng thực chất đều là các token nối tiếp nhau. Kiến trúc Transformer không có cổng nhập “chỉ dẫn” tách biệt. Các quy ước định dạng (khuôn mẫu hội thoại, thẻ dạng XML, cấu trúc JSON) có tác dụng vì chúng tạo thành những **tiền tố có cấu trúc ổn định** mà mô hình đã gặp khi huấn luyện.

Khi mô hình có vẻ “bỏ qua chỉ dẫn”, trước hết nên kiểm tra xem chỉ dẫn có bị đẩy ra khỏi cửa sổ ngữ cảnh hay khuôn mẫu hội thoại (*chat template*) có khớp với phiên bản mô hình hay không — đây thường là lỗi định dạng hoặc cắt chuỗi hơn là lỗi kiến trúc.

## 2. Zero-shot và học trong ngữ cảnh (In-Context Learning)

**Zero-shot:** Người dùng mô tả trực tiếp yêu cầu bằng ngôn ngữ tự nhiên (ví dụ “Dịch sang tiếng Việt:”) rồi để mô hình sinh tiếp. Mô hình tổng quát hóa từ các mẫu tương tự đã xuất hiện trong dữ liệu tiền huấn luyện.

**Học trong ngữ cảnh (ICL / few-shot):** Người dùng đặt một vài cặp ví dụ đầu vào–đầu ra vào ngay trong tiền tố trước câu hỏi mới. [Brown et al., 2020](https://arxiv.org/abs/2005.14165) (GPT-3) cho thấy ở quy mô đủ lớn, mô hình có thể thích nghi với tác vụ mới *mà không cần cập nhật gradient*. Về bản chất, cơ chế vẫn là dự đoán token kế tiếp: các ví dụ mẫu làm tăng xác suất của kiểu câu trả lời mong muốn.

So sánh học trong ngữ cảnh với tinh chỉnh cổ điển:

| Tiêu chí | Học trong ngữ cảnh (ICL) | Tinh chỉnh cổ điển (Fine-tuning) |
| --- | --- | --- |
| Thành phần thay đổi | Chuỗi token trong cửa sổ ngữ cảnh | Tham số $$\theta$$ (hoặc *adapter*) |
| Tính bền vững | Không lưu lại sau khi đổi prompt | Giữ cố định qua mọi truy vấn |
| Chi phí chính | Tốn thêm token *prefill* và bộ nhớ cache mỗi lần gọi | Tốn tính toán huấn luyện ban đầu, nhưng prompt ngắn và rẻ hơn khi phục vụ |
| Yêu cầu dữ liệu | Vài ví dụ minh họa, chỉnh sửa tức thì | Tập dữ liệu huấn luyện và quy trình tối ưu |

Nên ưu tiên ICL khi tác vụ mới, dữ liệu có nhãn còn ít, hoặc cần thay đổi hành vi linh hoạt theo từng truy vấn (ví dụ từng khách hàng hoặc từng định dạng khác nhau). Ngược lại, nên chọn tinh chỉnh khi cần hành vi ổn định lâu dài, khi prompt ngày càng phình to hoặc khi cần chuyển sang một mô hình chuyên biệt nhỏ và nhanh hơn (thường kết hợp chưng cất ở bài **26-05**).

**Chuỗi suy nghĩ** (*chain-of-thought*, [Wei et al., 2022](https://arxiv.org/abs/2201.11903)) là một kỹ thuật *prompting* yêu cầu mô hình trình bày các bước suy luận trung gian trước khi đưa ra đáp án cuối. Kỹ thuật này không bổ sung mô-đun kiến trúc mới mà dành thêm token giải mã để chia nhỏ phép tính phức tạp thành các bước dễ kiểm soát hơn. Về sau, các mô hình suy luận chuyên biệt *huấn luyện* trực tiếp thói quen này (xem Chương 25-99).

## 3. Khả năng tuân thủ chỉ dẫn đến từ SFT

Mô hình ngôn ngữ thô sau tiền huấn luyện có xu hướng viết tiếp theo ngữ cảnh: nếu đầu vào là một đoạn bài viết, nó sẽ viết tiếp bài viết đó thay vì trả lời câu hỏi. Tuy nhiên, người dùng cần một **trợ lý** biết thực hiện yêu cầu, từ chối khi phù hợp và giữ phong cách nhất quán.

Bước thích nghi đầu tiên thường là **tinh chỉnh có giám sát (SFT)** trên tập dữ liệu các cặp `(chỉ dẫn, câu trả lời)` do con người biên soạn, chọn lọc hoặc do một mô hình mạnh hơn sinh ra và được kiểm duyệt. Hàm mất mát vẫn là cross-entropy dự đoán token kế tiếp, nhưng *phân bố dữ liệu* đã chuyển từ “viết tiếp văn bản web” sang “trả lời đúng yêu cầu”.

Sau SFT, *prompting* vẫn rất hữu ích (cho gọi công cụ, RAG và định dạng), nhưng mô hình đã nhận diện văn bản người dùng như một *tác vụ cần thực hiện*. Giai đoạn tối ưu theo sở thích ở bài sau tiếp tục xếp hạng những câu trả lời hợp lệ nào được ưa chuộng hơn.

So với học chuyển giao cổ điển (Chương 15) vốn thay đầu phân loại cho một tập nhãn cố định, SFT theo chỉ dẫn vẫn dùng chung đầu sinh ngôn ngữ cho nhiều tác vụ khác nhau mô tả bằng văn bản — cùng cơ chế tối ưu nhưng khác giao diện tác vụ.

## 4. Danh mục kiểm tra khi viết prompt

Khi thử nghiệm trên các mô hình *instruct*:

1. Xác định rõ **vai trò** và **định dạng đầu ra** (văn bản thuần, các khóa JSON, ngôn ngữ).
2. Đặt các **ràng buộc bắt buộc** trước phần ngữ cảnh truy hồi dài để tránh bị cắt khỏi cửa sổ $$L$$.
3. Khi dùng *few-shot* (ICL), giữ các ví dụ minh họa **đồng nhất định dạng** với truy vấn thực tế.
4. Nếu cần dữ liệu trích dẫn xác thực, hãy truy hồi trước bằng RAG (Chương 18-99) thay vì yêu cầu mô hình tự nhớ.
5. Nếu cần duy trì hành vi cố định trên quy mô lớn, nên lên kế hoạch SFT hoặc dùng *adapter* thay vì để prompt phình dài tới hàng nghìn token.

## Điểm chính cần nhớ

- *Prompting* điều chỉnh tiền tố điều kiện; tinh chỉnh cập nhật trực tiếp trọng số mô hình.
- Học trong ngữ cảnh (ICL) là cơ chế *few-shot* ngay trong cửa sổ ngữ cảnh, không cập nhật tham số.
- Khả năng tuân thủ chỉ dẫn chủ yếu đến từ việc thay đổi phân bố **dữ liệu** qua SFT trên cùng hàm mất mát dự đoán token kế tiếp.
- Căn chỉnh theo sở thích (**26-03**) là bước bổ sung tiếp theo, không thay thế cho việc viết prompt rõ ràng.
- Các tham số lấy mẫu khi giải mã (nhiệt độ, *top-k*, *top-p*) thuộc lớp phục vụ (**26-04**), không phải dạng *prompting* mới.

## Video tham khảo

- **Xem trước** [What Is In-Context Learning in Deep Learning?](https://www.youtube.com/watch?v=As9a15poQHs) — Deep Learning with Yacine, tiếng Anh. Giải thích cơ chế học trong ngữ cảnh thông qua các ví dụ đặt trực tiếp trong prompt.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, tiếng Anh. Minh họa cách prompt chỉ là các token tiền tố đi vào cùng một mô hình dự đoán token kế tiếp.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, tiếng Anh. Cho thấy trực tiếp trong mã nguồn rằng prompt chính là chuỗi tiền tố để bộ lấy mẫu sinh tiếp.
