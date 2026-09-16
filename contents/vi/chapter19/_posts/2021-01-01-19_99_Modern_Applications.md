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

# Tùy chọn: tiếng nói và âm thanh sau Whisper — ASR đa ngữ, codec, TTS

> Bài này là **tùy chọn**. Phần mở đầu chương đã nêu Whisper và WaveNet. Ghi chú này đi tới **phần mềm 2022–2026** thực sự được dùng: ASR quy mô Whisper, codec neuron, và dịch tiếng nói đa ngữ — không viết lại lý thuyết spectrogram / MFCC (thân chương vẫn mỏng; đừng coi đây là thay lý thuyết).

## 1. Whisper như mặc định ASR

[Radford et al., 2022](https://arxiv.org/abs/2212.04356) huấn luyện Transformer encoder–decoder trên tiếng nói đa ngữ giám sát yếu. Checkpoint **large-v2 / large-v3** sau đó và [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) là thứ đội dữ liệu gọi từ Python. Kiến trúc là Chương 08; ứng dụng là ASR bền mà không cần HMM theo ngôn ngữ.

## 2. Ứng dụng cụ thể

### Dịch tiếng nói và stack song công

[SeamlessM4T (2023)](https://arxiv.org/abs/2308.11596) thống nhất ASR, dịch, và TTS. API realtime / omni (2024–2025) là dạng sản phẩm: luồng audio vào, luồng token hoặc audio ra.

### Codec neuron dưới audio sinh

[EnCodec](https://arxiv.org/abs/2210.13438) và [DAC](https://arxiv.org/abs/2306.06546) (cũng ở bài tùy chọn Chương 12) biến dạng sóng thành token rời rạc cho mô hình kiểu MusicGen / AudioLM. TTS chuyển từ mẫu WaveNet sang **codec-LM + vocoder** (CosyVoice, XTTS, OpenVoice).

### Diarization và ghi chú họp

Whisper + [pyannote](https://github.com/pyannote/pyannote-audio) là pipeline ghi chú họp 2023–2026: ASR, rồi nhãn người nói, rồi tóm tắt LLM (bài tùy chọn Chương 18).

## 3. Phần mềm phổ biến

- [openai/whisper](https://github.com/openai/whisper) và [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication).
- [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) và [coqui-ai/TTS](https://github.com/coqui-ai/TTS).

## 4. Trích dẫn (2022–2026)

- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356).
- [SeamlessM4T (2023)](https://arxiv.org/abs/2308.11596).
- [EnCodec (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438) và [DAC (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Dùng phần mở đầu chương làm bản đồ (spectrogram, DeepSpeech, WaveNet). Bài này chỉ cập nhật **checkpoint và codec** bạn sẽ cài năm 2026.
