"""Youtu-Parsing-Omni vLLM plugin.

Registers YoutuVITAForCausalLM with the vLLM model registry and binds the
multimodal processor for image / audio / video.
"""
from __future__ import annotations

import logging
import os
from urllib.parse import unquote, urlparse

_LOG = logging.getLogger(__name__)

_MODEL_TYPE = "youtu_vita"


# ---------------------------------------------------------------------------
# Video file paths: derived from ``file://`` video_url items, so that the
# processor can decode the audio track of a video together with its frames.
# ---------------------------------------------------------------------------

def _is_youtu_vita(model_config) -> bool:
    """True if ``model_config`` describes a Youtu-Parsing-Omni checkpoint."""
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == _MODEL_TYPE


def _file_url_to_local_path(url) -> str | None:
    """Convert ``file:///path`` or plain path to a local filesystem path."""
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme == "":
        return url
    if parsed.scheme == "file":
        return unquote(parsed.path)
    return None


def _patch_serving_chat_inject_audio() -> None:
    """Monkey-patch ``OpenAIServingChat.create_chat_completion`` so that the
    local paths of ``file://`` videos reach the processor (youtu_vita only).
    """
    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
    except Exception:
        _LOG.warning(
            "OpenAIServingChat not found; use_audio_in_video patch SKIPPED."
        )
        return

    orig = getattr(OpenAIServingChat, "create_chat_completion", None)
    if orig is None or getattr(orig, "_vita_uaiv_patched", False):
        return

    async def _patched(self, request, raw_request=None, *args, **kwargs):
        if not _is_youtu_vita(getattr(self, "model_config", None)):
            return await orig(self, request, raw_request, *args, **kwargs)
        try:
            mmpk = getattr(request, "mm_processor_kwargs", None)

            # Video always interleaves vision frames + audio chunks. So whenever
            # there are videos, extract their local file paths and pass them to
            # the processor via ``_video_file_paths`` so it can decode the audio
            # track as well. Without this the processor falls back to vLLM's
            # decoded frames, which carry no audio.
            #
            # Only paths derived from ``file://`` video_url items are used; vLLM
            # checks those against ``--allowed-local-media-path``. A
            # ``_video_file_paths`` sent by the client is always discarded.
            messages = getattr(request, "messages", None)
            video_paths = []
            if isinstance(messages, list):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get("content")
                    if not isinstance(content, list):
                        continue
                    for ele in content:
                        if not isinstance(ele, dict):
                            continue
                        if ele.get("type") != "video_url":
                            continue
                        vu = ele.get("video_url")
                        url = vu.get("url") if isinstance(vu, dict) else vu
                        path = _file_url_to_local_path(url)
                        if path:
                            video_paths.append(path)

            base = mmpk if isinstance(mmpk, dict) else {}
            if video_paths or "_video_file_paths" in base:
                # use_audio_in_video is dropped when paths are passed (the
                # processor handles audio internally via the file-path pipeline).
                drop = {"_video_file_paths"}
                if video_paths:
                    drop.add("use_audio_in_video")
                new_mmpk = {k: v for k, v in base.items() if k not in drop}
                if video_paths:
                    new_mmpk["_video_file_paths"] = video_paths
                try:
                    setattr(request, "mm_processor_kwargs", new_mmpk)
                except Exception:
                    if isinstance(mmpk, dict):
                        for key in drop:
                            mmpk.pop(key, None)
                        if video_paths:
                            mmpk["_video_file_paths"] = video_paths

            if video_paths:
                _LOG.info(
                    "VITA-patch: passed %d video path(s) to processor (fusion)",
                    len(video_paths),
                )
        except Exception as exc:
            _LOG.warning("VITA-patch: video path injection skipped: %s", exc)
        return await orig(self, request, raw_request, *args, **kwargs)

    _patched._vita_uaiv_patched = True  # type: ignore[attr-defined]
    OpenAIServingChat.create_chat_completion = _patched  # type: ignore[assignment]
    _LOG.info("VITA use_audio_in_video patch installed")


def _patch_llm_chat_inject_video_paths() -> None:
    """Offline ``LLM.chat`` path: extract video file paths from the OpenAI-style
    ``video_url`` items in the messages and inject them as ``_video_file_paths``
    (a list of plain path STRINGS) into ``mm_processor_kwargs`` — mirroring the
    serve-path patch (_patch_serving_chat_inject_audio).

    Why: offline llm.chat users otherwise can't pass the local video path to the
    processor (vLLM only forwards decoded frames), so the processor's native
    decord+ffmpeg audio pipeline is skipped (audio lost). The serve path solves
    this by patching OpenAIServingChat; this does the same for LLM.chat so the
    caller passes NOTHING extra.

    Hash-safety: we inject only path STRINGS extracted from ``video_url``. vLLM
    deep-hashes ``mm_processor_kwargs`` for the multimodal cache; strings are
    atomic to the hasher, so this never recurses."""
    try:
        from vllm.entrypoints.llm import LLM
    except Exception as exc:
        _LOG.warning("vllm.entrypoints.llm.LLM not found; offline video-path patch SKIPPED: %s", exc)
        return
    orig = getattr(LLM, "chat", None)
    if orig is None or getattr(orig, "_vita_vpath_patched", False):
        return

    def _extract_video_paths(messages) -> list[str]:
        # messages: a single conversation (list of msg dicts) or a list of them.
        paths: list[str] = []
        if not isinstance(messages, list) or not messages:
            return paths
        convs = messages if isinstance(messages[0], list) else [messages]
        for conv in convs:
            if not isinstance(conv, list):
                continue
            for msg in conv:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for ele in content:
                    if not isinstance(ele, dict) or ele.get("type") != "video_url":
                        continue
                    vu = ele.get("video_url")
                    url = vu.get("url") if isinstance(vu, dict) else vu
                    p = _file_url_to_local_path(url)
                    if p:
                        paths.append(p)
        return paths

    def _patched(self, messages, *args, **kwargs):
        try:
            mpk = kwargs.get("mm_processor_kwargs")
            if (isinstance(mpk, dict) and "_video_file_paths" not in mpk
                    and _is_youtu_vita(getattr(self, "model_config", None))):
                paths = _extract_video_paths(messages)
                if paths:
                    new_mpk = dict(mpk)
                    new_mpk["_video_file_paths"] = paths  # plain strings
                    kwargs["mm_processor_kwargs"] = new_mpk
                    _LOG.info("VITA-patch: LLM.chat injected %d video path(s)", len(paths))
        except Exception as exc:
            _LOG.warning("VITA-patch: LLM.chat video-path inject skipped: %s", exc)
        return orig(self, messages, *args, **kwargs)

    _patched._vita_vpath_patched = True  # type: ignore[attr-defined]
    LLM.chat = _patched  # type: ignore[assignment]
    _LOG.info("VITA-patch: LLM.chat video-path injection installed (offline path)")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register() -> None:
    """vLLM general-plugin entry point."""
    # 1. Register the config class with transformers
    from transformers import AutoConfig
    from .vita_config import YoutuVITAConfig
    try:
        AutoConfig.register(_MODEL_TYPE, YoutuVITAConfig)
    except Exception:
        pass

    # 2. Register the model and its multimodal processor with vLLM
    from vllm import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY

    from .vita_for_causal_lm import YoutuVITAForCausalLM
    from .vita_processor import (
        VITAMultiModalProcessor,
        VITAProcessingInfo,
        VITADummyInputsBuilder,
    )

    MULTIMODAL_REGISTRY.register_processor(
        VITAMultiModalProcessor,
        info=VITAProcessingInfo,
        dummy_inputs=VITADummyInputsBuilder,
    )(YoutuVITAForCausalLM)
    ModelRegistry.register_model("YoutuVITAForCausalLM", YoutuVITAForCausalLM)

    # 3. Video-path injection for the online server and the offline LLM.chat
    _patch_serving_chat_inject_audio()
    _patch_llm_chat_inject_video_paths()

    _LOG.info("Youtu-Parsing-Omni plugin registered (youtu_vita)")


__all__ = ["register"]
