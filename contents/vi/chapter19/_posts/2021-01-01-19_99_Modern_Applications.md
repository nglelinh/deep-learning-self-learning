---
layout: post
title: 19-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '19'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter19
lesson_type: optional
---

Chỉ vài năm trước, muốn xây dựng một hệ thống nhận dạng tiếng nói cho tiếng Việt hoặc một chuyên ngành hẹp, bạn phải gom hàng nghìn giờ ghi âm căn chỉnh thủ công theo từng âm vị cùng mô hình ngôn ngữ rời rạc. Sự xuất hiện của **Whisper**, các **Neural Audio Codec** (EnCodec, Mimi) và hệ thống hội thoại giọng nói song công (full-duplex speech-to-speech) đã thống nhất hoàn toàn thế giới âm thanh với kiến trúc Transformer.

## Bản đồ kiến thức nền: Bốn trụ cột của công nghệ Tiếng nói hiện đại (2022–2026)

- **Whisper và huấn luyện giám sát yếu đa nhiệm:** Dùng một kiến trúc Encoder–Decoder duy nhất trên 680.000+ giờ âm thanh đa ngữ để đồng thời nhận dạng ngôn ngữ, ghi chép có dấu câu, gắn mốc thời gian và dịch sang tiếng Anh.
- **Thước đo chất lượng ASR — Word Error Rate (WER) và CER:** Đo lường chính xác sai số thay thế (S), xóa (D) và chèn (I) dựa trên khoảng cách biên tập Levenshtein.
- **Residual Vector Quantization (RVQ) Audio Codec:** Nén sóng âm thành nhiều tầng mã rời rạc tốc độ bit cực thấp ($$1{,}5\text{–}6\text{ kbps}$$), biến sinh giọng nói thành bài toán dự đoán token kế tiếp giống LLM.
- **Giao tiếp giọng nói song công (Full-Duplex Speech Models):** Giảm độ trễ phản hồi xuống dưới 300 ms bằng cách xử lý luồng âm thanh trực tiếp (audio-in, audio-out) thay vì nối chuỗi chậm chạp `ASR → LLM → TTS`.

### Khung giải thích (nên đọc chậm): Tính tay Word Error Rate (WER) và tốc độ bit của một Neural Audio Codec

**Phần A — Tính tay WER:** Giả sử câu chuẩn (*reference*) có $$N = 8$$ từ: `hôm nay trời nắng đẹp và rất mát`, còn mô hình ASR xuất ra: `hôm nay trời rất nắng và rất mát mẻ`.
- So khớp tối ưu: chèn thêm `rất` trước `nắng` ($$I = 1$$), xóa `đẹp` ($$D = 1$$), chèn thêm `mẻ` ở cuối ($$I = 1$$), không có thay thế trực tiếp (hoặc tính tương đương: $$S=1, D=0, I=1$$? Chú ý: `nắng đẹp` → `rất nắng` là chèn `rất` + xóa `đẹp` = 2 phép, cộng chèn `mẻ` = 1 phép, tổng khoảng cách biên tập $$= 3$$).
- $$\text{WER} = \frac{S + D + I}{N} = \frac{3}{8} = 37{,}5\%.$$

**Phần B — Tốc độ bit của RVQ Audio Codec:** Giả sử codec chạy ở tần số khung $$f_{\text{frame}} = 50\text{ Hz}$$ (mỗi giây tạo 50 khung), dùng $$Q = 8$$ tầng sổ mã (*codebooks*), mỗi sổ mã có $$1{,}024 = 2^{10}$$ mã ($$10\text{ bit/mã}$$).
- Tốc độ bit: $$50 \times 8 \times 10 = 4{,}000\text{ bit/giây} = 4{,}0\text{ kbps}$$ — nhỏ hơn **64 lần** so với âm thanh PCM 16 kHz / 16-bit gốc ($$256\text{ kbps}$$) nhưng vẫn giữ đủ ngữ điệu và âm sắc người nói!

# Tùy chọn: Tiếng nói và âm thanh sau Whisper — ASR đa ngữ, Codec và TTS

> Bài này là **tùy chọn**. Phần mở đầu chương đã giới thiệu Whisper và WaveNet; bài này cập nhật các **công cụ và mô hình thực tế giai đoạn 2022–2026** như ASR quy mô Whisper, codec nơ-ron và dịch tiếng nói đa ngữ. Nội dung không thay thế phần lý thuyết nền tảng về spectrogram và MFCC.

## 1. Whisper trở thành chuẩn mực cho ASR

[Radford et al., 2022](https://arxiv.org/abs/2212.04356) huấn luyện mô hình Transformer *encoder–decoder* trên dữ liệu tiếng nói đa ngữ giám sát yếu. Bộ giải mã (*decoder*) phân rã xác suất đầu ra theo dạng tự hồi quy:

$$p(y\mid x) = \prod_t p(y_t \mid y_{<t}, x),$$

trong đó $$x$$ là phổ đồ log-mel (*log-mel spectrogram*). Các phiên bản trọng số **large-v2 / large-v3** cùng thư viện [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (dựa trên CTranslate2) là lựa chọn phổ biến trong quy trình Python thực tế. Về kiến trúc, mô hình dựa trên Transformer ở Chương 08, nhưng cho phép nhận dạng tiếng nói ổn định mà không cần xây dựng hệ thống HMM riêng cho từng ngôn ngữ.

## 2. Ứng dụng cụ thể

### Dịch tiếng nói và hệ thống giao tiếp song công

[SeamlessM4T (2023)](https://arxiv.org/abs/2308.11596) hợp nhất ba tác vụ ASR, dịch máy và TTS trong cùng một khung mô hình. Các API thời gian thực / đa phương thức (*realtime / omni*) giai đoạn 2024–2025 đưa mô hình này vào sản phẩm theo dạng luồng: nhận luồng âm thanh đầu vào và trả về luồng token hoặc âm thanh đầu ra trực tiếp.

### Codec nơ-ron làm nền cho mô hình sinh âm thanh

[EnCodec](https://arxiv.org/abs/2210.13438) và [DAC](https://arxiv.org/abs/2306.06546) (đã nhắc ở bài tùy chọn Chương 12) chuyển dạng sóng âm thanh thành chuỗi token rời rạc để các mô hình như MusicGen hay AudioLM xử lý. Trong tổng hợp tiếng nói (TTS), hướng tiếp cận cũng chuyển từ sinh trực tiếp từng mẫu sóng kiểu WaveNet sang mô hình **codec-LM kết hợp vocoder** (như CosyVoice, XTTS, OpenVoice).

### Phân tách người nói và ghi chú cuộc họp

Kết hợp Whisper với [pyannote](https://github.com/pyannote/pyannote-audio) là quy trình phổ biến cho hệ thống ghi chú cuộc họp giai đoạn 2023–2026: trước tiên nhận dạng tiếng nói (ASR), tiếp theo gán nhãn người nói (*speaker diarization*), và cuối cùng tóm tắt nội dung bằng LLM (xem bài tùy chọn Chương 18).

## 3. Phần mềm phổ biến

- [openai/whisper](https://github.com/openai/whisper) và [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication).
- [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) và [coqui-ai/TTS](https://github.com/coqui-ai/TTS).

## 4. Trích dẫn (2022–2026)

- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356).
- [SeamlessM4T (2023)](https://arxiv.org/abs/2308.11596).
- [EnCodec (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438) và [DAC (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Phần mở đầu chương phác họa bức tranh nền tảng (spectrogram, DeepSpeech, WaveNet), còn bài này cập nhật các **mô hình tiền huấn luyện và codec** thường được cài đặt và sử dụng trong thực tế đến năm 2026.

## Video tham khảo

- **Xem trước** [OpenAI's Whisper Model Explained](https://www.youtube.com/watch?v=uFOkMme19Zs) — What's AI by Louis-François Bouchard, tiếng Anh. Trình bày cơ chế của Whisper, hệ thống ASR mặc định được đề cập trong bài.
- [Mel Frequency Cepstral Coefficients (MFCC) Explained](https://www.youtube.com/watch?v=SJo7vPgRlBQ) — DataMListic, tiếng Anh. Giải thích các đặc trưng thang mel làm đầu vào cho các mô hình tiếng nói.
- [Mel-Frequency Cepstral Coefficients Explained Easily](https://www.youtube.com/watch?v=4_SH2nfbQZ8) — Valerio Velardo - The Sound of AI, tiếng Anh. Hướng dẫn cách tính các đặc trưng mel để hiểu rõ bản chất đầu vào của Whisper.
