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
