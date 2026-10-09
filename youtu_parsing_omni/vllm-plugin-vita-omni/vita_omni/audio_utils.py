"""Audio preprocessing utilities for VITA OmniEncoder plugin.

Converts audio input (file path, numpy array, or torch Tensor) to a mel
spectrogram in the format the OmniEncoder audio path expects: [T, num_mel_bins].

ALIGNMENT: this must reproduce the native transformers implementation
``feature_extraction_youtu_vita.py`` (MelFilterBankTokenizer.encode), which uses
``WhisperFeatureExtractor.from_pretrained(<whisper path>)`` with
``padding="do_not_pad", truncation=False``. Verified bit-exact (max_abs_diff=0)
against ``YoutuVITAFeatureExtractor.process_audio`` on real audio.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torchaudio

# Native config (processor_config.json feature_extractor):
#   audio_tokenizer_type = "melfilterbank"
#   audio_tokenizer_path = <whisper-large-v3 feature extractor directory>
#   feature_size = 128, sampling_rate = 16000, n_fft = 400, hop_length = 160
# The whisper path is resolved at runtime (``YOUTU_VITA_AUDIO_TOKENIZER_PATH``,
# then processor_config.json ``feature_extractor.audio_tokenizer_path``) and
# passed in by the caller. The bundled copy of the whisper-large-v3
# ``preprocessor_config.json`` below is the fallback when none is usable; only
# this file is needed by ``WhisperFeatureExtractor``.
_DEFAULT_WHISPER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "whisper-large-v3"
)
_TARGET_SR = 16000

# Lazily-loaded singleton WhisperFeatureExtractor (matches native encode path).
# Keyed by whisper_path so different configs get their own extractor.
_FEATURE_EXTRACTOR: dict = {}
_RESAMPLE_BUFFER: dict[int, torchaudio.transforms.Resample] = {}


def _get_feature_extractor(whisper_path: str | None = None):
    path = whisper_path or _DEFAULT_WHISPER_PATH
    fe = _FEATURE_EXTRACTOR.get(path)
    if fe is None:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(path)
        _FEATURE_EXTRACTOR[path] = fe
    return fe


def _to_waveform(audio_input, target_sr: int) -> torch.Tensor:
    """Return a 1-D float32 waveform at ``target_sr`` (mono).

    Mirrors MelFilterBankTokenizer.encode's front-end: accept (waveform, sr) /
    path / array / tensor, average channels to mono, resample if needed.
    """
    if isinstance(audio_input, str):
        # soundfile first (handles wav/flac), torchaudio fallback
        try:
            import soundfile as sf
            data, sr = sf.read(audio_input, dtype="float32")
            audio = torch.from_numpy(np.asarray(data, dtype=np.float32))
        except Exception:
            audio, sr = torchaudio.load(audio_input)
    elif isinstance(audio_input, tuple) and len(audio_input) == 2:
        audio, sr = audio_input
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        elif not torch.is_tensor(audio):
            audio = torch.tensor(audio, dtype=torch.float32)
    elif isinstance(audio_input, np.ndarray):
        audio = torch.from_numpy(np.asarray(audio_input, dtype=np.float32))
        sr = target_sr
    elif torch.is_tensor(audio_input):
        audio = audio_input.float()
        sr = target_sr
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Channels -> mono. torchaudio.load gives [C, T]; average over channels.
    if audio.dim() == 2:
        audio = audio.mean(0)

    if sr != target_sr:
        if sr not in _RESAMPLE_BUFFER:
            _RESAMPLE_BUFFER[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr)
        audio = _RESAMPLE_BUFFER[sr](audio[None, :])[0, :]

    return audio.contiguous().float()


def process_audio(audio_input, whisper_path: str | None = None) -> torch.Tensor:
    """Convert audio to a mel spectrogram [T, num_mel_bins=128].

    Reproduces native MelFilterBankTokenizer.encode:
        features = WhisperFeatureExtractor(audio, sampling_rate=16000,
            return_attention_mask=True, padding="do_not_pad", truncation=False)
        return features["input_features"].squeeze(0).permute(1, 0)  # [T, 128]

    Args:
        whisper_path: path to the whisper feature extractor (read from
            processor_config.json ``feature_extractor.audio_tokenizer_path``
            by the caller). When None, falls back to ``_DEFAULT_WHISPER_PATH``.

    Returns:
        torch.Tensor [T, 128] — the exact mel the OmniEncoder audio path expects.
    """
    audio = _to_waveform(audio_input, _TARGET_SR)

    fe = _get_feature_extractor(whisper_path)
    features = fe(
        audio.numpy(),
        sampling_rate=_TARGET_SR,
        return_attention_mask=True,
        return_tensors="pt",
        padding="do_not_pad",
        truncation=False,
    )
    # input_features: [1, num_mel_bins, T] -> [T, num_mel_bins]
    mel = features["input_features"].squeeze(0).permute(1, 0).contiguous()
    return mel
