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

# Optional: speech and audio after Whisper — multilingual ASR, codecs, TTS

> This lesson is **optional**. The chapter intro already names Whisper and WaveNet. This note goes to the **2022–2026 software** actually used: Whisper-scale ASR, neural codecs, and multilingual speech translation — without rewriting spectrogram / MFCC theory (the chapter body is still thin; do not treat this as a theory replacement).

## 1. Whisper as the ASR default

[Radford et al., 2022](https://arxiv.org/abs/2212.04356) train an encoder–decoder Transformer on weakly supervised multilingual speech. Later **large-v2 / large-v3** checkpoints and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) are what data teams call from Python. The architecture is Chapter 08; the application is robust ASR without a language-specific HMM.

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
