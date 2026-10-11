"""Multimodal processor for VITA OmniEncoder on vLLM 0.19.

Handles image / audio / video preprocessing and placeholder expansion.
Video always interleaves vision frames + audio chunks inside one
<|video_start|>…<|video_end|> region (the native processor's only video path).
What goes inside is decided solely by the content flags use_vision_in_video /
use_audio_in_video (+ whether the video actually has an audio track).
"""
from __future__ import annotations

import logging
import math
import os
import time as _time_mod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import BatchFeature

from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

_LOG = logging.getLogger(__name__)

# Placeholder "tag" tokens emitted by the chat template (one per mm item).
# These are expanded by _get_prompt_updates into the native token structure.
_IMAGE_TAG_TOKEN = "<|image|>"
_AUDIO_TAG_TOKEN = "<|audio|>"
_VIDEO_TAG_TOKEN = "<|video|>"

_IMAGE_PAD_TOKEN = "<|image_pad|>"
_AUDIO_PAD_TOKEN = "<|audio_pad|>"
_VIDEO_PAD_TOKEN = "<|video_pad|>"

# Video region special tokens (same IDs as image/audio but within video region)
_VIDEO_START_TOKEN = "<|video_start|>"
_VIDEO_END_TOKEN = "<|video_end|>"
_VISION_START_TOKEN = "<|vision_start|>"
_VISION_END_TOKEN = "<|vision_end|>"
_AUDIO_START_TOKEN = "<|audio_start|>"
_AUDIO_END_TOKEN = "<|audio_end|>"


def _audio_feat_output_lengths(T: int) -> int:
    """Whisper CNN output length for T mel frames.

    Verbatim port of native ``_get_feat_extract_output_lengths``
    (modular_youtu_vita.py:362 / omni_encoder._get_feat_extract_output_lengths).
    Each 100-frame block yields 13 tokens; tail computed via 3× stride-2 convs.
    """
    leave = T % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (T // 100) * 13


def _compute_audio_output_tokens(T: int, temporal_merge_size: int = 1) -> int:
    """Number of AUD_CONTEXT placeholder tokens for a mel chunk of T frames.

    Matches native add_audio_input_discrete_or_contiguous
    (feature_extraction_youtu_vita.py:384-386):
        (audio_token_length_func(T) + tms - 1) // tms
    NOTE: standalone audio uses temporal_merge_size=1 (processor_config
    feature_extractor); the video path uses its own value (=2).
    """
    n = _audio_feat_output_lengths(T)
    return (n + temporal_merge_size - 1) // temporal_merge_size


def _chunk_audio_mel(
    mel: torch.Tensor,
    duration_seconds: float,
    audio_chunk_min_second: float = 2.0,
    audio_chunk_max_second: float = 30.0,
) -> tuple[list[torch.Tensor], list[float]]:
    """Split one audio's mel [T, n_mels] into chunks + per-chunk start seconds.

    Verbatim port of native add_audio_input_discrete_or_contiguous chunking
    (feature_extraction_youtu_vita.py:322-347). Each chunk becomes an
    independent element fed to the encoder, so per-chunk token counts
    (f(T_chunk)) line up with the placeholder expansion.
    """
    T = mel.shape[0]
    if T == 0:
        return [], []
    second_per_audio = 1.0 * duration_seconds / T

    audio_chunks: list[torch.Tensor] = []
    audio_second_chunks: list[float] = []
    audio_chunk: list[torch.Tensor] = []
    for i in range(T):
        frame = mel[i]
        audio_second = 1 * second_per_audio
        audio_chunk_second = 1.0 * len(audio_chunk) * second_per_audio
        if audio_second + audio_chunk_second < audio_chunk_min_second:
            audio_chunk.append(frame)
        elif audio_second + audio_chunk_second <= audio_chunk_max_second:
            audio_chunk.append(frame)
        else:
            audio_second_chunks.append(
                sum(len(x) * second_per_audio for x in audio_chunks))
            audio_chunks.append(torch.stack(audio_chunk, dim=0))
            audio_chunk = [frame]
    if len(audio_chunk) > 0:
        audio_second_chunks.append(
            sum(len(x) * second_per_audio for x in audio_chunks))
        audio_chunks.append(torch.stack(audio_chunk, dim=0))
    return audio_chunks, audio_second_chunks


# ===========================================================================
# ProcessingInfo
# ===========================================================================


class VITAProcessingInfo(BaseProcessingInfo):
    """Processing info for VITA OmniEncoder models."""

    def get_data_parser(self):
        from vllm.multimodal.parse import MultiModalDataParser
        return MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None, "video": None}

    def _get_processor_config(self) -> dict:
        """The checkpoint's ``processor_config.json`` as a dict. ``model_id``
        may be a local directory or a Hugging Face repo id (resolved from the
        HF cache / Hub at the served revision, as vLLM does for its own
        configs). Cached per instance; ``{}`` if it cannot be loaded."""
        cached = getattr(self, "_processor_config", None)
        if cached is not None:
            return cached
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
        cfg = None
        try:
            revision = getattr(self.ctx.model_config, "revision", None)
            cfg = get_hf_file_to_dict("processor_config.json", self.model_id, revision)
        except Exception as _e:  # noqa: BLE001
            _LOG.warning("processor_config.json of %s could not be loaded: %s",
                         self.model_id, _e)
        if not isinstance(cfg, dict):
            _LOG.warning("processor_config.json not found for %s; using the "
                         "native processor defaults", self.model_id)
            cfg = {}
        self._processor_config = cfg
        return cfg

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        """Conservative upper bounds for memory profiling."""
        # Single-image upper bound MUST match the dynamic pixel budget derived
        # from processor_config.json (image_max_num_tokens), otherwise a large
        # image can exceed the pre-allocated encoder cache and vLLM raises
        # "image item ... exceeds the pre-allocated encoder cache size".
        # merged tokens = max_pixels / patch_size^2 / spatial_merge_size^2
        #               = image_max_num_tokens (by construction).
        img_budget = self.get_image_token_budget()
        max_img_tokens = int(img_budget["image_max_num_tokens"])
        max_audio_tokens = 600
        # Video: up to 64 frames, each up to 880 merged tokens
        max_video_tokens = 64 * 880

        return {
            "image": max_img_tokens,
            "audio": max_audio_tokens,
            "video": max_video_tokens,
        }

    def _get_token_id(self, token: str) -> int:
        tokenizer = self.get_tokenizer()
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == getattr(tokenizer, 'unk_token_id', None):
            return -1
        return int(tid)

    def get_video_token_budget(self) -> dict:
        """Video per-frame pixel budget params, read from the checkpoint's
        ``processor_config.json`` ``video_processor`` block, with optional
        runtime overrides from the ``YOUTU_VITA_VIDEO_*`` environment contract.
        Mirrors native
        ``YoutuVITAVideoProcessor.__init__`` fields. Cached per instance.

        Returns a dict with keys: patch_size, spatial_merge_size,
        video_min_num_tokens, video_max_num_tokens, video_image_min_num_tokens,
        video_image_max_num_tokens.
        """
        cached = getattr(self, "_video_token_budget", None)
        if cached is not None:
            return cached

        # Native code defaults (video_processing_youtu_vita.py __init__),
        # used only as fallback if processor_config.json is missing a field.
        budget = {
            "patch_size": 14,
            "spatial_merge_size": 2,
            "video_min_num_tokens": 64,
            "video_max_num_tokens": 8192,
            "video_image_min_num_tokens": 4,
            "video_image_max_num_tokens": 256,
        }
        try:
            vp = self._get_processor_config().get("video_processor", {}) or {}
            for k in list(budget.keys()):
                if k in vp and vp[k] is not None:
                    budget[k] = vp[k]
        except Exception as _e:
            _LOG.warning("get_video_token_budget: fallback to defaults: %s", _e)

        env_overrides = {
            "video_min_num_tokens": os.environ.get(
                "YOUTU_VITA_VIDEO_MIN_TOKENS"),
            "video_max_num_tokens": os.environ.get(
                "YOUTU_VITA_VIDEO_TOTAL_MAX_TOKENS")
                or os.environ.get("YOUTU_VITA_VIDEO_MAX_TOKENS"),
            "video_image_min_num_tokens": os.environ.get(
                "YOUTU_VITA_VIDEO_IMAGE_MIN_TOKENS"),
            "video_image_max_num_tokens": os.environ.get(
                "YOUTU_VITA_VIDEO_IMAGE_MAX_TOKENS"),
        }
        for key, raw_value in env_overrides.items():
            if not raw_value:
                continue
            try:
                value = int(raw_value)
                if value <= 0:
                    raise ValueError("must be positive")
                if "max_num_tokens" in key:
                    budget[key] = min(int(budget[key]), value)
                else:
                    budget[key] = value
            except (TypeError, ValueError) as _e:
                _LOG.warning("Ignoring invalid %s=%r: %s", key, raw_value, _e)

        budget["video_min_num_tokens"] = min(
            int(budget["video_min_num_tokens"]),
            int(budget["video_max_num_tokens"]),
        )
        budget["video_image_min_num_tokens"] = min(
            int(budget["video_image_min_num_tokens"]),
            int(budget["video_image_max_num_tokens"]),
        )

        self._video_token_budget = budget

        _LOG.info("get_video_token_budget: %s", budget)
        return budget

    def get_image_token_budget(self) -> dict:
        """Single-image pixel budget params, read from the checkpoint's
        ``processor_config.json`` ``image_processor`` block, capped by the
        optional ``YOUTU_VITA_IMAGE_MAX_TOKENS`` runtime limit. Mirrors native
        ``YoutuVITAImageProcessor.__init__`` fields. Cached per instance.

        Following the native __init__, the actual min/max_pixels are DERIVED:
            min_pixels = (patch_size * spatial_merge_size)^2 * image_min_num_tokens
            max_pixels = (patch_size * spatial_merge_size)^2 * image_max_num_tokens

        Returns a dict with keys: patch_size, spatial_merge_size,
        image_min_num_tokens, image_max_num_tokens, min_pixels, max_pixels.
        """
        cached = getattr(self, "_image_token_budget", None)
        if cached is not None:
            return cached

        # Native code defaults (YoutuVITAImageProcessor.__init__), used only as
        # fallback if processor_config.json is missing a field.
        budget = {
            "patch_size": 14,
            "spatial_merge_size": 2,
            "image_min_num_tokens": 4,
            "image_max_num_tokens": 256,
        }
        try:
            ip = self._get_processor_config().get("image_processor", {}) or {}
            for k in list(budget.keys()):
                if k in ip and ip[k] is not None:
                    budget[k] = ip[k]
        except Exception as _e:
            _LOG.warning("get_image_token_budget: fallback to defaults: %s", _e)

        env_max_tokens = os.environ.get("YOUTU_VITA_IMAGE_MAX_TOKENS")
        if env_max_tokens:
            try:
                max_tokens = int(env_max_tokens)
                if max_tokens <= 0:
                    raise ValueError("must be positive")
                budget["image_max_num_tokens"] = min(
                    int(budget["image_max_num_tokens"]), max_tokens)
                budget["image_min_num_tokens"] = min(
                    int(budget["image_min_num_tokens"]),
                    budget["image_max_num_tokens"],
                )
            except (TypeError, ValueError) as _e:
                _LOG.warning(
                    "Ignoring invalid YOUTU_VITA_IMAGE_MAX_TOKENS=%r: %s",
                    env_max_tokens,
                    _e,
                )

        factor_sq = (budget["patch_size"] * budget["spatial_merge_size"]) ** 2
        budget["min_pixels"] = factor_sq * budget["image_min_num_tokens"]
        budget["max_pixels"] = factor_sq * budget["image_max_num_tokens"]
        
        self._image_token_budget = budget

        _LOG.info("get_image_token_budget: %s", budget)
        return budget

    def _get_whisper_path(self) -> str:
        """Path to the whisper feature extractor.

        Resolution order:
            1. ``YOUTU_VITA_AUDIO_TOKENIZER_PATH`` if it points to a directory;
            2. the checkpoint's ``processor_config.json``
               ``feature_extractor.audio_tokenizer_path`` (a list, take the first
               entry) unless it is an absolute path that does not exist on this
               machine (checkpoints may record the training-cluster path);
            3. ``audio_utils._DEFAULT_WHISPER_PATH`` (bundled whisper-large-v3
               ``preprocessor_config.json``).
        Cached per instance.
        """
        cached = getattr(self, "_whisper_path", None)
        if cached is not None:
            return cached
        configured = os.environ.get("YOUTU_VITA_AUDIO_TOKENIZER_PATH", "").strip()
        if configured and os.path.isdir(configured):
            self._whisper_path = configured
            return configured
        path = None
        try:
            fe = self._get_processor_config().get("feature_extractor", {}) or {}
            atp = fe.get("audio_tokenizer_path")
            if isinstance(atp, list):
                atp = atp[0] if atp else None
            if isinstance(atp, str) and atp:
                if os.path.isabs(atp) and not os.path.isdir(atp):
                    _LOG.info(
                        "get_whisper_path: %s does not exist; using the bundled "
                        "feature extractor config", atp)
                else:
                    path = atp
        except Exception as _e:
            _LOG.warning("get_whisper_path: fallback to default: %s", _e)
        if not path:
            from .audio_utils import _DEFAULT_WHISPER_PATH
            path = _DEFAULT_WHISPER_PATH
        self._whisper_path = path
        return path


# ===========================================================================
# DummyInputsBuilder
# ===========================================================================


class VITADummyInputsBuilder(BaseDummyInputsBuilder):
    """Builds dummy inputs for memory profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        parts = []
        for _ in range(mm_counts.get("image", 0)):
            parts.append(_IMAGE_TAG_TOKEN)
        for _ in range(mm_counts.get("video", 0)):
            parts.append(_VIDEO_TAG_TOKEN)
        for _ in range(mm_counts.get("audio", 0)):
            parts.append(_AUDIO_TAG_TOKEN)
        return "".join(parts) + "Hello"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        data: dict[str, Any] = {}

        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            dummy_img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
            data["image"] = [dummy_img] * num_images

        num_videos = mm_counts.get("video", 0)
        if num_videos > 0:
            # Dummy video: 1 frame of 224x224 as numpy [1, H, W, C] + metadata
            dummy_frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            dummy_metadata = {"fps": 1.0, "duration": 1.0, "total_num_frames": 1,
                              "frames_indices": [0], "do_sample_frames": True,
                              "video_backend": "dummy"}
            data["video"] = [(dummy_frames, dummy_metadata)] * num_videos

        num_audios = mm_counts.get("audio", 0)
        if num_audios > 0:
            data["audio"] = [np.zeros(16000, dtype=np.float32)] * num_audios

        return data


# ===========================================================================
# MultiModalProcessor
# ===========================================================================


class VITAMultiModalProcessor(BaseMultiModalProcessor):
    """VITA OmniEncoder multimodal processor.

    Supports image, audio, and video modalities.
    Video always interleaves vision frames + audio chunks (omni fusion); the two
    content flags use_vision_in_video / use_audio_in_video decide what is filled in.
    """

    info: VITAProcessingInfo

    @staticmethod
    def _extract_audio_from_video_bytes(
        video_bytes: bytes, whisper_path: str | None = None,
    ) -> torch.Tensor | None:
        """Extract audio from video bytes using ffmpeg, return mel [T, 128]."""
        import subprocess, tempfile, os
        from .audio_utils import process_audio
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_in = os.path.join(tmp_dir, "in.mp4")
                tmp_wav = os.path.join(tmp_dir, "out.wav")
                with open(tmp_in, "wb") as f:
                    f.write(video_bytes)
                cmd = ["ffmpeg", "-y", "-i", tmp_in, "-vn",
                       "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav]
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode != 0:
                    return None
                return process_audio(tmp_wav, whisper_path=whisper_path)
        except Exception as e:
            _LOG.warning(f"Failed to extract audio from video: {e}")
            return None

    def _process_video_fusion(
        self,
        video_data: list,
        nframes: int,
        use_audio_in_video: bool,
        use_vision_in_video: bool,
        tokenizer,
        spatial_merge: int,
        patch_size: int,
        video_audio_chunk_min_second: float,
        video_audio_chunk_max_second: float,
        video_file_paths: list[str] | None = None,
    ) -> dict:
        """Process video (omni fusion: interleaved vision frames + audio chunks).

        Ports native ``YoutuVITAVideoProcessor.add_video_input_discrete_or_contiguous``
        token structure exactly:
        - Vision frames use ``<|image_pad|>`` (IMG_CONTEXT, 128257)
        - Audio chunks use ``<|audio_pad|>`` (AUD_CONTEXT, 128265)
        - NOT ``<|video_pad|>`` (old plugin was wrong)

        Produces:
            - video_input_ids: list[int] (the <vid_start>...<vid_end> fragment)
            - video_images: Tensor [N_patches_all, patch_dim]
            - video_image_grid_thw: Tensor [N_frames, 3]
            - video_audios: list[Tensor] (each [T_i, 128] mel chunk)
            - video_split: Tensor [N_videos, 2]
        """
        from .image_utils import process_image
        from .video_utils import extract_video_frames_and_audio, chunk_audio_video_frames

        vid_start_id = tokenizer.convert_tokens_to_ids(_VIDEO_START_TOKEN)
        vid_end_id = tokenizer.convert_tokens_to_ids(_VIDEO_END_TOKEN)
        img_start_id = tokenizer.convert_tokens_to_ids(_VISION_START_TOKEN)
        img_end_id = tokenizer.convert_tokens_to_ids(_VISION_END_TOKEN)
        img_pad_id = tokenizer.convert_tokens_to_ids(_IMAGE_PAD_TOKEN)
        aud_start_id = tokenizer.convert_tokens_to_ids(_AUDIO_START_TOKEN)
        aud_end_id = tokenizer.convert_tokens_to_ids(_AUDIO_END_TOKEN)
        aud_pad_id = tokenizer.convert_tokens_to_ids(_AUDIO_PAD_TOKEN)
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        all_video_images = []
        all_video_grid_thw = []
        all_video_audios = []  # list of mel tensors per chunk
        video_split_list = []
        all_input_ids = []
        all_interleave_orders = []  # per-video list of (modality, idx) events
        # Per-video sizes (for vLLM to split the concatenated video fields into
        # N items, one per <|video|> placeholder). Mirrors ref's per-video
        # regions + video_split design (multi-video support).
        video_input_ids_lens = []     # token length of each video's fusion region
        video_patch_counts = []       # sum(t*h*w) over each video's frames
        video_audio_frame_counts = [] # sum(mel frames) over each video's audio chunks

        for vid_idx, video_item in enumerate(video_data):
            # ---- Extract frames + audio via native port ----
            # If we have a video file path, use the native decord pipeline.
            # Otherwise fall back to the vLLM-parsed (frames, metadata) tuple.
            video_path = None
            if video_file_paths and vid_idx < len(video_file_paths):
                video_path = video_file_paths[vid_idx]

            frames_pil = None
            sample_fps = None
            timestamps = None
            duration_seconds = 0.0
            audio_mels_per_frame = None

            if video_path is not None and os.path.isfile(video_path):
                # Native path: decord frame extraction + ffmpeg audio
                extraction = extract_video_frames_and_audio(
                    video_path,
                    max_num_frames=nframes,
                    max_fps=1.0,
                    use_audio_in_video=use_audio_in_video,
                    use_vision_in_video=use_vision_in_video,
                    whisper_path=self.info._get_whisper_path(),
                )
                frames_pil = extraction["frames"]
                sample_fps = extraction["fps"]
                timestamps = extraction["timestamps"]
                duration_seconds = extraction["duration"]
                audio_mels_per_frame = extraction["audio_mels_per_frame"]
            else:
                # Fallback: vLLM-parsed frames (numpy/tuple + metadata)
                if isinstance(video_item, tuple) and len(video_item) == 2:
                    frames, metadata = video_item
                else:
                    frames = video_item
                    metadata = None

                if isinstance(frames, np.ndarray) and frames.ndim == 4:
                    n_frames = frames.shape[0]
                    if n_frames > nframes:
                        indices = np.linspace(0, n_frames - 1, nframes, dtype=int)
                        frames = frames[indices]
                    frames_pil = [PILImage.fromarray(frames[i]) for i in range(frames.shape[0])]
                elif isinstance(frames, (list, tuple)):
                    frame_list = list(frames)
                    if len(frame_list) > nframes:
                        indices = np.linspace(0, len(frame_list) - 1, nframes, dtype=int)
                        frame_list = [frame_list[i] for i in indices]
                    frames_pil = frame_list
                else:
                    frames_pil = []

                if metadata is not None:
                    duration_seconds = metadata.get("duration", 0.0)
                    fps_meta = metadata.get("fps", 1.0)
                    frames_indices = metadata.get("frames_indices", [])
                    if frames_indices:
                        timestamps = [idx / fps_meta for idx in frames_indices]
                        if len(timestamps) > len(frames_pil) and len(frames_pil) > 0:
                            ts_indices = np.linspace(
                                0, len(timestamps) - 1, len(frames_pil), dtype=int)
                            timestamps = [timestamps[i] for i in ts_indices]
                        else:
                            timestamps = timestamps[:len(frames_pil)]
                    else:
                        timestamps = [i * duration_seconds / max(len(frames_pil), 1)
                                      for i in range(len(frames_pil))]

                    # Try to extract audio from video bytes (if available)
                    if use_audio_in_video:
                        video_bytes = metadata.get("original_video_bytes")
                        if video_bytes is not None:
                            full_mel = self._extract_audio_from_video_bytes(
                                video_bytes, whisper_path=self.info._get_whisper_path())
                            if full_mel is not None and full_mel.shape[0] > 0 and frames_pil:
                                total_mel_time = full_mel.shape[0] / 100.0
                                full_ts = timestamps + [duration_seconds]
                                audio_mels_per_frame = []
                                for fidx in range(len(frames_pil)):
                                    st = full_ts[fidx]
                                    ed = full_ts[fidx + 1]
                                    st_mel = int(st / total_mel_time * full_mel.shape[0])
                                    ed_mel = int(ed / total_mel_time * full_mel.shape[0])
                                    if ed_mel <= st_mel:
                                        ed_mel = st_mel + 1
                                    audio_mels_per_frame.append(full_mel[st_mel:ed_mel])

            n_images = len(frames_pil) if frames_pil else 0

            # Fallback timestamps if still empty
            if not timestamps and n_images > 0:
                if duration_seconds > 0:
                    timestamps = [i * duration_seconds / n_images for i in range(n_images)]
                else:
                    timestamps = [float(i) for i in range(n_images)]

            # ---- Process frames through image pipeline ----
            # CRITICAL: cap per-frame resolution like native process_video
            # (video_processing_youtu_vita.py:229-236). Native spreads a TOTAL
            # token budget (video_max_num_tokens) across all frames (``// n``)
            # and also clamps each frame to video_image_max_num_tokens. Without
            # this, each frame is processed at the single-image max_pixels and a
            # long / hi-res video blows past the model context (e.g. 155k tokens).
            #   factor² = (patch_size * spatial_merge)²  (pixels per merged token)
            #   min_pixels = max(factor² * video_min_num_tokens // n, factor² * video_image_min_num_tokens)
            #   max_pixels = min(factor² * video_max_num_tokens // n, factor² * video_image_max_num_tokens)
            # Budget params come from the checkpoint's processor_config.json
            # (golden source), NOT hardcoded — see get_video_token_budget().
            frame_pixel_values = []
            frame_grid_thws = []
            if use_vision_in_video and frames_pil:
                budget = self.info.get_video_token_budget()
                b_factor = budget["patch_size"] * budget["spatial_merge_size"]
                factor_sq = b_factor ** 2
                n = max(n_images, 1)
                frame_min_pixels = max(
                    factor_sq * budget["video_min_num_tokens"] // n,
                    factor_sq * budget["video_image_min_num_tokens"])
                frame_max_pixels = min(
                    factor_sq * budget["video_max_num_tokens"] // n,
                    factor_sq * budget["video_image_max_num_tokens"])
                for pil_img in frames_pil:
                    result = process_image(
                        pil_img,
                        patch_size=patch_size,
                        spatial_merge_size=spatial_merge,
                        min_pixels=frame_min_pixels,
                        max_pixels=frame_max_pixels,
                    )
                    frame_pixel_values.append(result["pixel_values"])
                    frame_grid_thws.append(result["image_grid_thw"])

            n_images = len(frame_pixel_values)

            # ---- Chunk audio+video frames (port native _chunk_audio_video_frames) ----
            if audio_mels_per_frame is None:
                audio_mels_per_frame = []

            image_chunks, image_second_chunks, audio_chunks, audio_second_chunks = \
                chunk_audio_video_frames(
                    audio_mels_per_frame,
                    timestamps,
                    duration_seconds,
                    video_audio_chunk_min_second,
                    video_audio_chunk_max_second,
                )

            # ---- Build token structure (port native add_video_input_discrete_or_contiguous) ----
            video_ids = [vid_start_id]
            num_images_for_split = 0
            num_audios_for_split = 0
            # Record the interleaved order of vision/audio events in prompt order.
            # Each entry: (0, frame_idx_in_all_frames) or (1, audio_chunk_idx_in_all_chunks).
            # The model uses this to scatter embeddings in the correct prompt order.
            vid_interleave_order = []
            frame_global_idx = 0
            audio_global_idx = 0
            vid_audio_frames = 0  # sum of mel frames for THIS video's audio chunks

            for img_chunk_indices, img_seconds, aud_chunk_mels, aud_seconds in zip(
                image_chunks, image_second_chunks, audio_chunks, audio_second_chunks
            ):
                # Vision frames in this chunk
                if use_vision_in_video:
                    for frame_idx_in_chunk, frame_second in zip(img_chunk_indices, img_seconds):
                        # Timestamp (HH:MM:SS)
                        timestamp_str = _time_mod.strftime(
                            "%H:%M:%S", _time_mod.gmtime(round(frame_second)))
                        ts_ids = tokenizer.encode(timestamp_str, add_special_tokens=False)
                        video_ids.extend(ts_ids)

                        # <|vision_start|> "H*W" \n [ <|image_pad|>*cols \n ]*rows <|vision_end|>
                        video_ids.append(img_start_id)
                        grid_thw = frame_grid_thws[frame_idx_in_chunk]  # [1, 3]
                        t, h, w = int(grid_thw[0, 0]), int(grid_thw[0, 1]), int(grid_thw[0, 2])
                        res_h = h * patch_size
                        res_w = w * patch_size
                        res_str = f"{res_h}*{res_w}"
                        res_ids = tokenizer.encode(res_str, add_special_tokens=False)
                        video_ids.extend(res_ids)

                        # Image pad tokens with newlines (use image_pad_id, NOT video_pad!)
                        rows = t * h // spatial_merge
                        cols = w // spatial_merge
                        for _ in range(rows):
                            video_ids.extend([img_pad_id] * cols)
                            video_ids.extend(nl_ids)

                        video_ids.append(img_end_id)
                        vid_interleave_order.append((0, frame_global_idx))
                        frame_global_idx += 1
                        num_images_for_split += 1

                # Audio chunks in this chunk
                if use_audio_in_video and aud_chunk_mels:
                    for aud_mel, aud_second in zip(aud_chunk_mels, aud_seconds):
                        if aud_mel.shape[0] == 0:
                            continue
                        # Timestamp (HH:MM:SS)
                        timestamp_str = _time_mod.strftime(
                            "%H:%M:%S", _time_mod.gmtime(round(aud_second)))
                        ts_ids = tokenizer.encode(timestamp_str, add_special_tokens=False)
                        video_ids.extend(ts_ids)

                        # <|audio_start|> <|audio_pad|>*N <|audio_end|>
                        # Use audio_pad_id (NOT video_pad!), matching native structure.
                        # Audio placeholder count MUST use temporal_merge_size=1:
                        # the omni audio_merger weight is [2560, 1024] (=hidden*1),
                        # i.e. the checkpoint's audio encoder does NOT temporally
                        # merge (omni_config.temporal_merge_size=1). Using 2 here
                        # produced f(T)//2 placeholders while the model emitted
                        # f(T) embeddings -> misaligned audio -> transcription
                        # degenerated into repetition.
                        video_ids.append(aud_start_id)
                        n_audio_tokens = _compute_audio_output_tokens(
                            aud_mel.shape[0], temporal_merge_size=1)
                        video_ids.extend([aud_pad_id] * n_audio_tokens)
                        video_ids.append(aud_end_id)

                        vid_interleave_order.append((1, audio_global_idx))
                        audio_global_idx += 1
                        all_video_audios.append(aud_mel)
                        vid_audio_frames += int(aud_mel.shape[0])
                        num_audios_for_split += 1

            video_ids.append(vid_end_id)
            all_input_ids.extend(video_ids)
            all_interleave_orders.append(vid_interleave_order)

            # Collect frame data
            if frame_pixel_values:
                all_video_images.append(torch.cat(frame_pixel_values, dim=0))
                all_video_grid_thw.append(torch.cat(frame_grid_thws, dim=0))

            video_split_list.append((num_images_for_split, num_audios_for_split))

            # Per-video sizes (for vLLM per-item splitting of the concatenated
            # video fields). video_input_ids lens = this region's token count;
            # patch count = sum(t*h*w) over this video's frames; audio frames =
            # sum of mel rows over this video's chunks.
            video_input_ids_lens.append(len(video_ids))
            video_patch_counts.append(
                sum(int(g[0, 0]) * int(g[0, 1]) * int(g[0, 2]) for g in frame_grid_thws)
                if frame_grid_thws else 0)
            video_audio_frame_counts.append(vid_audio_frames)

        # Assemble outputs
        result = {"video_input_ids": all_input_ids}

        if all_video_images:
            result["video_images"] = torch.cat(all_video_images, dim=0)
            result["video_image_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
        if all_video_audios:
            result["video_audios"] = all_video_audios
            result["video_audio_lens"] = torch.tensor(
                [m.shape[0] for m in all_video_audios], dtype=torch.long)
        if video_split_list:
            result["video_split"] = torch.tensor(video_split_list, dtype=torch.long)
            # Per-video sizes for vLLM per-item field splitting (multi-video).
            result["video_input_ids_lens"] = torch.tensor(
                video_input_ids_lens, dtype=torch.long)
            result["video_patch_counts"] = torch.tensor(
                video_patch_counts, dtype=torch.long)
            result["video_audio_frame_counts"] = torch.tensor(
                video_audio_frame_counts, dtype=torch.long)
        if all_interleave_orders:
            # Encode interleave orders as a flat tensor: [n_total_events, 2]
            # each row = (modality, idx) where modality 0=vision, 1=audio
            all_events = []
            for vid_order in all_interleave_orders:
                for modality, idx in vid_order:
                    all_events.append([modality, idx])
            result["video_interleave_orders"] = torch.tensor(
                all_events, dtype=torch.long)

        return result

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process multimodal data."""
        from .image_utils import process_image

        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()
        omni_config = hf_config.omni_config
        spatial_merge = omni_config.spatial_merge_size
        patch_size = getattr(omni_config, "patch_size", 16)

        # Get processing parameters from mm_kwargs
        nframes = int(mm_kwargs.get("nframes", 64))
        use_audio_in_video = bool(mm_kwargs.get("use_audio_in_video", True))
        use_vision_in_video = bool(mm_kwargs.get("use_vision_in_video", True))
        # Video always uses vision frames; audio-only video is not supported.
        use_vision_in_video = True
        video_audio_chunk_min_second = float(mm_kwargs.get("video_audio_chunk_min_second", 1.5))
        video_audio_chunk_max_second = float(mm_kwargs.get("video_audio_chunk_max_second", 29.5))

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, **tok_kwargs)
        outputs = {"input_ids": torch.tensor([input_ids])}


        # ---- Handle images ----
        images_data = mm_data.get("image") or mm_data.get("images")
        if images_data is not None:
            if not isinstance(images_data, (list, tuple)):
                images_data = [images_data]

            # Dynamic per-image pixel budget from the checkpoint's
            # processor_config.json image_processor block (golden source),
            # mirroring native YoutuVITAImageProcessor.__init__.
            img_budget = self.info.get_image_token_budget()

            all_pixel_values = []
            all_grid_thw = []
            for img in images_data:
                result = process_image(
                    img,
                    min_pixels=img_budget["min_pixels"],
                    max_pixels=img_budget["max_pixels"],
                )
                all_pixel_values.append(result["pixel_values"])
                all_grid_thw.append(result["image_grid_thw"])

            outputs["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            outputs["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)

        # ---- Handle video ----
        video_data = mm_data.get("video") or mm_data.get("videos")
        if video_data is not None:
            if not isinstance(video_data, (list, tuple)):
                video_data = [video_data]
            # Handle case where it's a single video item (frames, metadata)
            if (len(video_data) == 2 and isinstance(video_data[0], np.ndarray)
                    and isinstance(video_data[1], (dict, type(None)))):
                video_data = [video_data]

            # This model ALWAYS interleaves vision frames + audio chunks inside
            # one <|video_start|>…<|video_end|> region (the native processor's
            # only video path). What goes inside is decided solely by the two
            # content flags use_vision_in_video / use_audio_in_video (+ whether
            # the video actually has an audio track). There is no independent
            # "frames-as-images" path.
            video_file_paths = mm_kwargs.get("_video_file_paths")
            fusion_result = self._process_video_fusion(
                video_data, nframes, use_audio_in_video, use_vision_in_video,
                tokenizer, spatial_merge, patch_size,
                video_audio_chunk_min_second, video_audio_chunk_max_second,
                video_file_paths=video_file_paths,
            )

            # Store the expanded video region tokens for prompt replacement
            outputs["_video_fusion_input_ids"] = torch.tensor(
                fusion_result["video_input_ids"], dtype=torch.long)

            if "video_images" in fusion_result:
                outputs["video_images"] = fusion_result["video_images"]
                outputs["video_image_grid_thw"] = fusion_result["video_image_grid_thw"]
            if "video_audios" in fusion_result:
                audios_list = fusion_result["video_audios"]
                outputs["video_audios"] = torch.cat(audios_list, dim=0)
                outputs["video_audio_lens"] = fusion_result["video_audio_lens"]
            if "video_split" in fusion_result:
                outputs["video_split"] = fusion_result["video_split"]
                # Per-video sizes for vLLM per-item field splitting (multi-video)
                outputs["video_input_ids_lens"] = fusion_result["video_input_ids_lens"]
                outputs["video_patch_counts"] = fusion_result["video_patch_counts"]
                outputs["video_audio_frame_counts"] = fusion_result["video_audio_frame_counts"]
            if "video_interleave_orders" in fusion_result:
                outputs["video_interleave_orders"] = fusion_result["video_interleave_orders"]

        # ---- Handle audio ----
        audio_data = mm_data.get("audio") or mm_data.get("audios")
        if audio_data is not None:
            from .audio_utils import process_audio
            if not isinstance(audio_data, (list, tuple)):
                audio_data = [audio_data]

            audio_chunk_min = float(mm_kwargs.get("audio_chunk_min_second", 2.0))
            audio_chunk_max = float(mm_kwargs.get("audio_chunk_max_second", 30.0))

            # Each API audio item is split into 1+ chunks (native chunking).
            # Every chunk is an independent mel tensor for the encoder; we track
            # per-chunk frame counts (audio_lens) and per-item chunk counts so
            # _get_prompt_updates can rebuild the per-chunk START/pad/END structure.
            all_chunk_mels: list[torch.Tensor] = []
            audio_lens: list[int] = []           # frames T per chunk
            audio_chunk_seconds: list[float] = []  # start-second per chunk
            audio_num_chunks: list[int] = []      # chunks per API audio item
            for aud in audio_data:
                mel = process_audio(aud, whisper_path=self.info._get_whisper_path())
                duration_seconds = mel.shape[0] / 100.0  # mel frame rate = 100/s
                chunks, seconds = _chunk_audio_mel(
                    mel, duration_seconds, audio_chunk_min, audio_chunk_max)
                if not chunks:
                    chunks, seconds = [mel], [0.0]
                audio_num_chunks.append(len(chunks))
                for cmel, csec in zip(chunks, seconds):
                    all_chunk_mels.append(cmel)
                    audio_lens.append(cmel.shape[0])
                    audio_chunk_seconds.append(csec)

            outputs["audios"] = torch.cat(all_chunk_mels, dim=0)
            outputs["audio_lens"] = torch.tensor(audio_lens, dtype=torch.long)
            outputs["audio_chunk_seconds"] = torch.tensor(
                audio_chunk_seconds, dtype=torch.float32)
            outputs["audio_num_chunks"] = torch.tensor(
                audio_num_chunks, dtype=torch.long)

        return BatchFeature(outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        cfg: dict[str, MultiModalFieldConfig] = {}

        # Image fields
        if "pixel_values" in hf_inputs and "image_grid_thw" in hf_inputs:
            image_grid_thw = hf_inputs["image_grid_thw"]
            num_patches_per_image = (
                image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
            )
            cfg["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches_per_image)
            cfg["image_grid_thw"] = MultiModalFieldConfig.batched("image")

        # Video fields (omni fusion): N items, one per <|video|> placeholder
        # (multi-video). Per-video sizes come from video_split [N,2] + the
        # per-video counts produced by _process_video_fusion. vLLM splits each
        # concatenated field into N items; get_video_fusion_replacement(item_idx)
        # then indexes per video. Single video -> N=1 (backward compatible).
        video_split = hf_inputs.get("video_split")
        n_videos = (video_split.shape[0]
                    if video_split is not None and video_split.numel() > 0 else 0)

        if n_videos > 0:
            frame_counts = video_split[:, 0]           # [N] frames per video
            chunk_counts = video_split[:, 1]           # [N] audio chunks per video
            event_counts = video_split.sum(dim=1)      # [N] interleave events per video
            patch_counts = hf_inputs["video_patch_counts"]              # [N]
            audio_frame_counts = hf_inputs["video_audio_frame_counts"]  # [N]
            ids_lens = hf_inputs["video_input_ids_lens"]                # [N]

            if "video_images" in hf_inputs:
                cfg["video_images"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", patch_counts)
                cfg["video_image_grid_thw"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", frame_counts)
            if "video_audios" in hf_inputs:
                cfg["video_audios"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", audio_frame_counts)
                cfg["video_audio_lens"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", chunk_counts)
            # video_split: one [2] row per video -> batched (stacks to [N,2])
            cfg["video_split"] = MultiModalFieldConfig.batched("video")
            # Fusion input_ids for prompt replacement (per-video region tokens)
            if "_video_fusion_input_ids" in hf_inputs:
                cfg["_video_fusion_input_ids"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", ids_lens)
            # Interleave order (encoded as [N_events_total, 2] flat tensor)
            if "video_interleave_orders" in hf_inputs:
                cfg["video_interleave_orders"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", event_counts)
        elif "video_images" in hf_inputs and "video_image_grid_thw" in hf_inputs:
            # Defensive fallback (no video_split): treat as a single video item.
            video_image_grid_thw = hf_inputs["video_image_grid_thw"]
            total_patches = int((video_image_grid_thw[:, 0] * video_image_grid_thw[:, 1]
                                 * video_image_grid_thw[:, 2]).sum())
            n_frames = video_image_grid_thw.shape[0]
            cfg["video_images"] = MultiModalFieldConfig.flat_from_sizes(
                "video", torch.tensor([total_patches], dtype=torch.long))
            cfg["video_image_grid_thw"] = MultiModalFieldConfig.flat_from_sizes(
                "video", torch.tensor([n_frames], dtype=torch.long))

        # Audio fields. One API audio item -> 1+ chunks. We key all audio
        # fields per-item so vLLM's item indexing (one item per API audio)
        # stays consistent. Within an item, _get_prompt_updates rebuilds the
        # per-chunk START/pad/END structure using audio_num_chunks/lens/seconds.
        if "audios" in hf_inputs and "audio_lens" in hf_inputs:
            audio_lens = hf_inputs["audio_lens"]          # [n_chunks_total] frames/chunk
            audio_num_chunks = hf_inputs["audio_num_chunks"]  # [n_items] chunks/item
            # audios: split by per-item total frames (sum of that item's chunks)
            item_frame_sizes = []
            chunk_off = 0
            for nc in audio_num_chunks.tolist():
                item_frame_sizes.append(int(audio_lens[chunk_off:chunk_off + nc].sum()))
                chunk_off += nc
            cfg["audios"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", torch.tensor(item_frame_sizes, dtype=torch.long))
            # per-chunk fields split by per-item chunk count
            cfg["audio_lens"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_num_chunks)
            cfg["audio_chunk_seconds"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_num_chunks)
            cfg["audio_num_chunks"] = MultiModalFieldConfig.batched("audio")

        return cfg

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Define how placeholders get expanded."""
        updates: list[PromptUpdate] = []
        hf_config = self.info.get_hf_config()
        omni_config = hf_config.omni_config
        spatial_merge = omni_config.spatial_merge_size

        tokenizer = self.info.get_tokenizer()
        image_tag_id = self.info._get_token_id(_IMAGE_TAG_TOKEN)
        image_pad_id = self.info._get_token_id(_IMAGE_PAD_TOKEN)
        audio_pad_id = self.info._get_token_id(_AUDIO_PAD_TOKEN)
        video_pad_id = self.info._get_token_id(_VIDEO_PAD_TOKEN)
        patch_size = getattr(omni_config, "patch_size", 16)
        vision_start_id = self.info._get_token_id(_VISION_START_TOKEN)
        vision_end_id = self.info._get_token_id(_VISION_END_TOKEN)
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        # Image: expand <|image|> to the native token structure:
        #   <|vision_start|> "H*W"(digit tokens) \n
        #   [ <|image_pad|>*cols \n ] * rows
        #   <|vision_end|>
        # where rows = t*h // merge, cols = w // merge.
        # (verified 1:1 against image_processing_youtu_vita.py
        #  add_image_input_discrete_or_contiguous, contiguous path)
        # Only <|image_pad|> positions carry vision embeddings, so we wrap the
        # structure in select_token_id(image_pad_id).
        image_items = out_mm_kwargs.get("image") or []
        if image_items:

            def get_image_replacement(item_idx: int):
                item = image_items[item_idx]
                grid_thw = item.get("image_grid_thw")
                if grid_thw is not None:
                    grid = grid_thw.data
                    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
                else:
                    t, h, w = 1, 28, 28

                rows = t * h // spatial_merge
                cols = w // spatial_merge

                res_str = f"{h * patch_size}*{w * patch_size}"
                res_ids = tokenizer.encode(res_str, add_special_tokens=False)

                token_ids: list[int] = [vision_start_id]
                token_ids += res_ids
                token_ids += nl_ids
                for _ in range(rows):
                    token_ids += [image_pad_id] * cols
                    token_ids += nl_ids
                token_ids.append(vision_end_id)

                return PromptUpdateDetails.select_token_id(token_ids, image_pad_id)

            updates.append(PromptReplacement(
                modality="image",
                target=[image_tag_id],
                replacement=get_image_replacement,
            ))

        # Video: expand the <|video|> tag to the full interleaved structure built
        # by _process_video_fusion (vision frames -> <|image_pad|>, audio chunks
        # -> <|audio_pad|>, in prompt order). This is the model's only video path.
        video_items = out_mm_kwargs.get("video") or []
        if video_items:
            # Target the <|video|> tag (128260); embedding positions are both
            # image_pad_id (128257, vision frames) and audio_pad_id (128265, audio).
            video_tag_id = self.info._get_token_id(_VIDEO_TAG_TOKEN)

            def get_video_fusion_replacement(item_idx: int):
                item = video_items[item_idx]
                fusion_ids = item.get("_video_fusion_input_ids")
                if fusion_ids is not None:
                    data = fusion_ids.data if hasattr(fusion_ids, 'data') else fusion_ids
                    if isinstance(data, torch.Tensor):
                        token_list = data.tolist()
                    else:
                        token_list = list(data)
                else:
                    token_list = [image_pad_id]
                # Select both image_pad and audio_pad positions for embedding
                return PromptUpdateDetails.select_token_ids(
                    token_list, [image_pad_id, audio_pad_id])

            updates.append(PromptReplacement(
                modality="video",
                target=[video_tag_id],
                replacement=get_video_fusion_replacement,
            ))

        # Audio: expand <|audio|> to the native per-chunk structure:
        #   for each chunk of this audio item:
        #     [timestamp tokens] (only if item has >1 chunk)
        #     <|audio_start|> <|audio_pad|>*f(T_chunk) <|audio_end|>
        # where f = _audio_feat_output_lengths, temporal_merge_size=1 (standalone).
        # (verified against feature_extraction_youtu_vita.py:352-418)
        audio_items = out_mm_kwargs.get("audio") or []
        if audio_items:
            audio_tag_id = self.info._get_token_id(_AUDIO_TAG_TOKEN)
            audio_start_id = self.info._get_token_id(_AUDIO_START_TOKEN)
            audio_end_id = self.info._get_token_id(_AUDIO_END_TOKEN)

            def _as_list(field):
                if field is None:
                    return None
                data = field.data if hasattr(field, "data") else field
                if isinstance(data, torch.Tensor):
                    return data.tolist()
                return list(data)

            def get_audio_replacement(item_idx: int):
                item = audio_items[item_idx]
                lens = _as_list(item.get("audio_lens")) or []       # frames per chunk
                seconds = _as_list(item.get("audio_chunk_seconds")) or []
                n_chunks = len(lens)

                token_ids: list[int] = []
                for ci in range(n_chunks):
                    # Timestamp only when the item was split into multiple chunks
                    if n_chunks > 1:
                        sec = int(round(seconds[ci])) if ci < len(seconds) else 0
                        ts = _time_mod.strftime("%H:%M:%S", _time_mod.gmtime(sec))
                        token_ids += tokenizer.encode(ts, add_special_tokens=False)
                    n_pad = _compute_audio_output_tokens(int(lens[ci]),
                                                         temporal_merge_size=1)
                    token_ids.append(audio_start_id)
                    token_ids += [audio_pad_id] * n_pad
                    token_ids.append(audio_end_id)

                return PromptUpdateDetails.select_token_id(token_ids, audio_pad_id)

            updates.append(PromptReplacement(
                modality="audio",
                target=[audio_tag_id],
                replacement=get_audio_replacement,
            ))

        return updates
