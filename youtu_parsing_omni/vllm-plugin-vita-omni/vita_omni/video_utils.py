"""Video preprocessing utilities for VITA OmniEncoder plugin.

Ports the native HF ``YoutuVITAVideoProcessor`` video pipeline:
1. Frame extraction (decord, matching native ``get_video_frames``).
2. Audio extraction (ffmpeg subprocess -> 16kHz mono wav -> whisper mel).
3. Per-frame mel slicing (matching native ``process_video`` audio split).
4. Audio/video chunk interleaving (``_chunk_audio_video_frames``).

The output mirrors the native processor's ``process_video`` + ``add_video_input_discrete_or_contiguous``
so that the plugin's ``_process_video_fusion`` can build the exact same token structure.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import os
from typing import Optional

import numpy as np
import torch

_LOG = logging.getLogger(__name__)

# Mel frame rate: 16000 / hop_length(160) = 100 frames/s
_MEL_FPS = 100.0


def extract_video_frames_and_audio(
    video_path: str,
    max_num_frames: int = 64,
    max_fps: float = 1.0,
    use_audio_in_video: bool = True,
    use_vision_in_video: bool = True,
    whisper_path: Optional[str] = None,
) -> dict:
    """Extract frames and audio from a video file, porting native ``process_video``.

    Native pipeline (video_processing_youtu_vita.py):
    1. ``get_video_frames``: decord VideoReader, step = max(fps/max_fps, total/max_frames),
       indices = [int(i*step) for i in range(0, max_frames)], timestamps = [idx/fps * idx].
    2. ``get_image_and_audio``: torchaudio.load full audio -> resample 16kHz.
    3. ``process_video``: image_processor.process_images(frames) -> image_frames, grid.
       audio_processor.process_audio -> mel [T, 128]; split mel per frame by timestamps.
    4. Return (image_frames, audio_frames, audio_token_length_func, grid_thw, second_per_grids, timestamps, duration).

    This function does steps 1-2 (frame extraction + audio extraction + per-frame mel split).
    Step 3 (image preprocessing) is done by the caller via ``process_image``.

    Returns:
        dict with keys:
            - frames: list of PIL.Image (extracted video frames)
            - fps: float (native sample_fps = 1 / (step_size / fps))
            - timestamps: list of float (seconds per frame)
            - duration: float (video duration in seconds)
            - audio_mels_per_frame: list of [T_i, 128] mel tensors (per frame), or None
            - audio_token_length_func: callable (the whisper output length function)
    """
    import decord
    decord.bridge.set_bridge("native")

    # ---- 1. Frame extraction (port native get_video_frames) ----
    vid = decord.VideoReader(video_path, num_threads=1)
    fps = vid.get_avg_fps()
    fps = round(fps)

    step_size = len(vid) / max_num_frames
    step_size = max(fps / max_fps, step_size)

    indices = [int(i * step_size) for i in range(0, max_num_frames)]
    indices = [i for i in indices if i < len(vid)]

    sample_fps = 1.0 / (step_size / fps) if fps > 0 else 1.0
    timestamps = [1.0 / fps * i for i in indices]
    duration_seconds = len(vid) / vid.get_avg_fps()

    frames_np = [vid[i].asnumpy() for i in indices]

    from PIL import Image
    frames_pil = [Image.fromarray(arr) for arr in frames_np]

    # ---- 2. Audio extraction (ffmpeg -> wav -> mel) ----
    audio_mel = None  # full mel [T, 128]
    audio_total_time = None  # waveform duration (matches ref total_time)
    if use_audio_in_video:
        _am = _extract_audio_mel(video_path, whisper_path=whisper_path)
        if _am is not None:
            audio_mel, audio_total_time = _am

    # ---- 3. Per-frame mel split (port native process_video L249-268) ----
    # ref uses total_time = len(audio_waveform)/sampling_rate (waveform duration),
    # NOT T_mel/_MEL_FPS. Using the waveform duration matches ref's split points
    # exactly (avoids per-chunk audio_pad off-by-one when T_mel is rounded).
    audio_mels_per_frame = None
    if audio_mel is not None and audio_mel.shape[0] > 0 and len(frames_pil) > 0:
        total_time = audio_total_time
        full_ts = timestamps + [total_time]
        audio_mels_per_frame = []
        for fidx in range(len(frames_pil)):
            st = full_ts[fidx]
            ed = full_ts[fidx + 1]
            st_mel = int(st / total_time * audio_mel.shape[0])
            ed_mel = int(ed / total_time * audio_mel.shape[0])
            if ed_mel <= st_mel:
                ed_mel = st_mel + 1  # ensure at least 1 frame
            audio_mels_per_frame.append(audio_mel[st_mel:ed_mel])

    return {
        "frames": frames_pil,
        "fps": sample_fps,
        "timestamps": timestamps,
        "duration": duration_seconds,
        "audio_mels_per_frame": audio_mels_per_frame,
    }


def _video_has_audio_track(video_path: str) -> bool:
    """Probe whether the container declares an audio stream (no decoding).

    Uses ffprobe on container metadata to distinguish a legitimately trackless
    video (valid vision-only fallback) from a decoder/environment failure where
    an audio stream exists but torchaudio can't read it (must hard-fail).

    Raises RuntimeError if ffprobe is missing or the probe itself fails — in
    those cases we cannot make the trackless distinction safely, so the caller
    must not get a silent False that would mask a real decode problem.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe not found on PATH; cannot determine whether videos have "
            "an audio track. Install ffmpeg/ffprobe so audio-decode failures "
            "can be distinguished from trackless videos."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ffprobe timed out probing audio stream for {video_path}: {e!r}"
        ) from e

    if result.returncode != 0:
        # ffprobe returns rc=0 with empty stdout for a valid-but-trackless
        # file; a non-zero rc means the file/container itself is unreadable.
        raise RuntimeError(
            f"ffprobe failed (rc={result.returncode}) probing {video_path}: "
            f"{result.stderr.strip()!r}; cannot determine audio-track presence."
        )
    return bool(result.stdout.strip())


def _extract_audio_mel(
    video_path: str,
    whisper_path: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Extract audio from video file -> 16kHz mono waveform -> whisper mel [T, 128].

    Mirrors the native ``YoutuVITAVideoProcessor.get_image_and_audio`` audio
    front-end EXACTLY (video_processing_youtu_vita.py:198-211):
        audio, sr = torchaudio.load(video_path)   # decodes the audio track
        if audio.dim() == 2: audio = audio.mean(0) # channels -> mono
        if sr != 16000: audio = Resample(sr, 16000)(audio)
    then feeds the waveform (not a re-encoded wav) to the same whisper mel path.

    NOTE: an ffmpeg subprocess (-ar 16000 -ac 1 -> wav) is not used here: its
    decode/resample differs numerically from torchaudio and makes the video-audio
    mel (and downstream embeddings) diverge from the reference.
    """
    import torchaudio

    from .audio_utils import process_audio

    try:
        audio, sr = torchaudio.load(video_path)
    except Exception as e:
        # Distinguish the ONE legitimate fallback (video has no audio track)
        # from every other failure (missing decoder/torchcodec, corrupt file,
        # unsupported codec). Only a genuinely trackless video may pass through
        # as vision-only; an existing audio stream that won't decode is an
        # environment/data problem that must surface as a hard error rather than
        # silently make the model "deaf".
        if _video_has_audio_track(video_path):
            raise RuntimeError(
                f"torchaudio.load failed on {video_path} which DOES contain an "
                f"audio stream (per ffprobe): {e!r}. This is a decoder/"
                f"environment problem (verify `torchcodec` is importable and "
                f"ffmpeg present), not a trackless video. Refusing to silently "
                f"fall back to vision-only; fix the environment or set "
                f"use_audio_in_video=False explicitly."
            ) from e
        _LOG.warning(
            "torchaudio.load failed on %s but it has no audio stream "
            "(per ffprobe); legitimate vision-only fallback.", video_path,
        )
        return None

    if audio.dim() == 2:
        audio = audio.mean(0)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio[None, :])[0, :]

    # process_audio accepts a (waveform, sr) tuple; front-end will keep it mono
    # and skip resample since it's already at target_sr.
    mel = process_audio((audio.contiguous().float(), target_sr), whisper_path=whisper_path)
    # Waveform-based duration (matches ref's total_time = len(audio)/sampling_rate,
    # used to split mel per frame). Using T_mel/_MEL_FPS instead diverges from ref
    # when mel extraction rounds T_mel != len(waveform)/hop exactly, causing
    # per-chunk audio_pad off-by-one.
    total_time = audio.shape[0] / target_sr
    return mel, total_time


def chunk_audio_video_frames(
    audio_mels_per_frame: list[torch.Tensor],
    frame_timestamps: list[float],
    duration_seconds: float,
    video_audio_chunk_min_second: float = 1.5,
    video_audio_chunk_max_second: float = 29.5,
) -> tuple[list[list[int]], list[list[float]], list[list[torch.Tensor]], list[list[float]]]:
    """Port native ``_chunk_audio_video_frames`` exactly.

    Splits image/audio frames into time-aligned chunks based on audio duration limits.

    Args mirror native YoutuVITAVideoProcessor._chunk_audio_video_frames.

    Returns:
        (image_chunks, image_second_chunks, audio_chunks, audio_second_chunks)
        - image_chunks: list of chunks, each chunk is list of frame indices
        - image_second_chunks: timestamps per frame in each chunk
        - audio_chunks: list of chunks, each chunk is list of mel tensors (merged into 1)
        - audio_second_chunks: timestamps per audio in each chunk
    """
    if not audio_mels_per_frame:
        return [list(range(len(frame_timestamps)))], [frame_timestamps], [[]], [[]]

    # second_per_audio = total_time / total_audio_frames
    total_audio_frames = sum(m.shape[0] for m in audio_mels_per_frame)
    second_per_audio = 1.0 * duration_seconds / total_audio_frames if total_audio_frames > 0 else 1.0

    image_chunks = []
    image_second_chunks = []
    audio_chunks = []
    audio_second_chunks = []

    image_chunk = []
    image_second_chunk = []
    audio_chunk = []
    audio_second_chunk = []

    for frame_idx, (audio_mel, second_frame) in enumerate(
        zip(audio_mels_per_frame, frame_timestamps)
    ):
        audio_second = audio_mel.shape[0] * second_per_audio
        audio_chunk_second = sum(x.shape[0] * second_per_audio for x in audio_chunk)

        if audio_second + audio_chunk_second < video_audio_chunk_min_second:
            image_chunk.append(frame_idx)
            image_second_chunk.append(second_frame)
            audio_chunk.append(audio_mel)
            audio_second_chunk.append(second_frame)

        elif audio_second + audio_chunk_second <= video_audio_chunk_max_second:
            image_chunk.append(frame_idx)
            image_second_chunk.append(second_frame)
            audio_chunk.append(audio_mel)
            audio_second_chunk.append(second_frame)

            image_chunks.append(image_chunk)
            image_second_chunks.append(image_second_chunk)

            merged_audio = torch.cat(audio_chunk, dim=0)
            audio_chunks.append([merged_audio])
            audio_second_chunks.append([audio_second_chunk[0]])

            image_chunk = []
            image_second_chunk = []
            audio_chunk = []
            audio_second_chunk = []

        else:
            # Single frame audio exceeds max: split it
            assert audio_chunk_second == 0, f"expected empty chunk, got {audio_chunk_second}"

            image_chunk.append(frame_idx)
            image_second_chunk.append(second_frame)

            audio_chunk_size = int(video_audio_chunk_max_second / second_per_audio)
            if audio_chunk_size <= 0:
                audio_chunk_size = 1
            split_chunks = list(torch.split(audio_mel, audio_chunk_size))
            cur_second = second_frame
            split_seconds = []
            for x in split_chunks:
                split_seconds.append(cur_second)
                cur_second += x.shape[0] * second_per_audio

            image_chunks.append(image_chunk)
            image_second_chunks.append(image_second_chunk)
            audio_chunks.append(split_chunks)
            audio_second_chunks.append(split_seconds)

            image_chunk = []
            image_second_chunk = []
            audio_chunk = []
            audio_second_chunk = []

    if image_chunk:
        image_chunks.append(image_chunk)
        image_second_chunks.append(image_second_chunk)
        if audio_chunk:
            merged_audio = torch.cat(audio_chunk, dim=0)
            audio_chunks.append([merged_audio])
            audio_second_chunks.append([audio_second_chunk[0]])
        else:
            audio_chunks.append([])
            audio_second_chunks.append([])

    return image_chunks, image_second_chunks, audio_chunks, audio_second_chunks
