---
layout: post
title: 19-99 Modern Applications and Updates (2022–2026)
chapter: '19'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter19
lesson_type: optional
---

Until 2022, building a production speech stack meant wiring together fragile acoustic models, pronunciation lexicons, and language-model decoders tuned separately for each language. OpenAI's Whisper blew up that pipeline by training a single encoder–decoder Transformer on 680,000 hours of weakly supervised multilingual audio—and by 2024–2026, neural audio codecs and full-duplex speech-to-speech models turned spoken conversation into native token-stream modeling.

## Background Knowledge Map: Modern Speech & Audio Stacks (2022–2026)

- **Pillar 1 — Weakly Supervised Multilingual ASR (Whisper)**: Pads/chunks audio into 30-second 80/128-bin log-Mel windows, downsamples $2\times$ via strided convs into 1500 encoder frames, and uses special task tokens (`<|transcribe|>`, `<|translate|>`, timestamps) for multitask decoding.
- **Pillar 2 — Neural Audio Codecs & Discrete Speech Tokens**: Quantizes continuous speech into semantic + acoustic RVQ token streams (EnCodec, Mimi, DAC) so LLMs can read and speak audio end-to-end without an intermediate text bottleneck.
- **Pillar 3 — Full-Duplex Speech & Low-Latency Streaming**: Handles simultaneous listening and speaking (interruptions, backchannels) by predicting overlapping user/assistant token streams at fixed frame rates (e.g., 12.5 Hz).
- **Pillar 4 — Speaker Diarization & Meeting Pipelines**: Combines Voice Activity Detection (VAD), speaker embeddings (`pyannote`), and Whisper ASR to answer *"who spoke what and when?"*

### Step-by-Step Worked Example (Read Slowly): Computing Whisper's Encoder Frame Budget and Real-Time Factor (RTF)

Whisper processes $30\text{ s}$ of $16\text{ kHz}$ audio using a $10\text{ ms}$ hop length ($3{,}000$ Mel frames) followed by two 1D convolutions where the second has `stride=2`.

1. **Encoder token length per 30-second chunk**:
   $$L_{\text{enc}} = \frac{3{,}000}{2} = 1{,}500\text{ frames} \implies \frac{30\text{ s}}{1{,}500} = 20\text{ ms per encoder token (50 Hz)}.$$
2. **Real-Time Factor (RTF) for streaming vs. batch**:
   Suppose `faster-whisper` (CTranslate2 INT8) transcribes a $600\text{ s}$ (10-minute) podcast in $12.0\text{ s}$ of GPU compute.
   $$\text{RTF} = \frac{T_{\text{compute}}}{T_{\text{audio}}} = \frac{12.0}{600.0} = 0.02 \quad (50\times\text{ faster than real time!}).$$
3. **Takeaway**: Every Whisper encoder token spans exactly $20\text{ ms}$ of speech, and tracking $\text{RTF} \ll 1$ tells you immediately whether a model can run comfortably in live production.

# Optional: speech and audio after Whisper — multilingual ASR, codecs, TTS

> This lesson is **optional**. The chapter intro already names Whisper and WaveNet. This note goes to the **2022–2026 software** actually used: Whisper-scale ASR, neural codecs, and multilingual speech translation — without rewriting spectrogram / MFCC theory (the chapter body is still thin; do not treat this as a theory replacement).

## 1. Whisper as the ASR default

[Radford et al., 2022](https://arxiv.org/abs/2212.04356) train an encoder–decoder Transformer on weakly supervised multilingual speech. The decoder factorizes

$$p(y\mid x) = \prod_t p(y_t \mid y_{<t}, x),$$

where $$x$$ is a log-mel spectrogram. Later **large-v2 / large-v3** checkpoints and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) are what data teams call from Python. The architecture is Chapter 08; the application is robust ASR without a language-specific HMM.

## 2. Concrete applications

### Speech translation and full-duplex stacks

[Communication et al., 2023](https://arxiv.org/abs/2308.11596) (SeamlessM4T) unify ASR, translation, and TTS. OpenAI’s realtime / omni-class APIs (2024–2025) are the product form: stream audio in, stream tokens or audio out.

### Neural codecs under generative audio

[EnCodec](https://arxiv.org/abs/2210.13438) and [DAC](https://arxiv.org/abs/2306.06546) (also in Chapter 12 optional) turn waveforms into discrete tokens for MusicGen / AudioLM-style models. TTS moved from WaveNet samples toward **codec-LM + vocoder** (CosyVoice, XTTS, OpenVoice).

### Diarization and meeting DS

Whisper + [pyannote](https://github.com/pyannote/pyannote-audio) is the 2023–2026 meeting-notes pipeline: ASR, then speaker labels, then an LLM summary (Chapter 18 optional).

## 3. Widely used software

- [openai/whisper](https://github.com/openai/whisper) and [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication).
- [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) (MusicGen) and [coqui-ai/TTS](https://github.com/coqui-ai/TTS).

## 4. Citations (2022–2026)

- [Robust Speech Recognition via Large-Scale Weak Supervision — Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356).
- [SeamlessM4T (Communication et al., 2023)](https://arxiv.org/abs/2308.11596).
- [EnCodec (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438) and [DAC (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546).

## 5. How this complements the core notes

Use this chapter’s intro as the map (spectrograms, DeepSpeech, WaveNet). This lesson only updates the **checkpoints and codecs** you would install in 2026.

## Video tham khảo

- **xem trước** [OpenAI's Whisper Model Explained](https://www.youtube.com/watch?v=uFOkMme19Zs) — What's AI by Louis-François Bouchard, English. Whisper as the default ASR system this note describes.
- [Mel Frequency Cepstral Coefficients (MFCC) Explained](https://www.youtube.com/watch?v=SJo7vPgRlBQ) — DataMListic, English. The mel features that kind of model still starts from.
- [Mel-Frequency Cepstral Coefficients Explained Easily](https://www.youtube.com/watch?v=4_SH2nfbQZ8) — Valerio Velardo - The Sound of AI, English. How those features are built, so the Whisper input is not a black box.
