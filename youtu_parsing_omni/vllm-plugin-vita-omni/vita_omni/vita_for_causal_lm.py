"""Youtu-Parsing-Omni (omni encoder + dense MLA decoder) for vLLM."""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
)
from vllm.model_executor.models.interfaces import (
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.sequence import IntermediateTensors

from .omni_encoder import VITAOmniEncoder

_LOG = logging.getLogger(__name__)

# Special tokens
_IMAGE_PAD_TOKEN = "<|image_pad|>"
_AUDIO_PAD_TOKEN = "<|audio_pad|>"
_VIDEO_PAD_TOKEN = "<|video_pad|>"


class YoutuVITAForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    """VITA OmniEncoder + dense MLA (Youtu / DeepSeek-style) decoder for vLLM.

    The LM backbone is vLLM's official ``DeepseekV2ForCausalLM`` (dense MLA
    config). ``YoutuVITAConfig`` shapes the text_config so vLLM drives it
    through DeepseekV2 (see ``_apply_mla_text_defaults``); DeepseekV2's own
    weight loader handles the q_a_proj / kv_a_proj_with_mqa fusion.
    """

    # Packed layouts of the decoder, needed by LoRA.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.omni_model.": "omni_encoder.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # OmniEncoder (tensor-parallel body). Pass quant_config + prefix so the
        # encoder's ColumnParallel/RowParallel layers register correct names.
        text_hidden_size = config.text_config.hidden_size
        quant_config = vllm_config.quant_config
        self.omni_encoder = VITAOmniEncoder(
            config.omni_config, text_hidden_size,
            quant_config=quant_config, prefix="omni_encoder")

        # LM decoder: dense MLA, served by vLLM's DeepseekV2ForCausalLM.
        self.language_model = DeepseekV2ForCausalLM(
            vllm_config=vllm_config, prefix="language_model")

        # Token IDs (looked up from tokenizer)
        self._inject_token_ids(vllm_config)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Restrict LoRA replacement to the language-model backbone."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["omni_encoder.vision_merger", "omni_encoder.audio_merger"],
            tower_model="omni_encoder",
        )

    @staticmethod
    def _inject_token_ids(vllm_config: VllmConfig) -> None:
        """Lookup special token ids and store on hf_config."""
        from vllm.transformers_utils.tokenizer import get_tokenizer

        model_cfg = vllm_config.model_config
        tokenizer = get_tokenizer(
            model_cfg.tokenizer,
            tokenizer_mode=model_cfg.tokenizer_mode,
            trust_remote_code=model_cfg.trust_remote_code,
            revision=model_cfg.tokenizer_revision,
        )

        def _lookup(token: str) -> int:
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid is None or tid == getattr(tokenizer, 'unk_token_id', None):
                _LOG.warning(f"Token {token!r} not found in tokenizer")
                return -1
            return int(tid)

        hf = model_cfg.hf_config
        hf.image_pad_token_id = _lookup(_IMAGE_PAD_TOKEN)
        hf.audio_pad_token_id = _lookup(_AUDIO_PAD_TOKEN)
        hf.video_pad_token_id = _lookup(_VIDEO_PAD_TOKEN)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("video"):
            return "<|video|>"
        if modality.startswith("audio"):
            return "<|audio|>"
        return None

    # ------------------------------------------------------------------
    # Core vLLM interface
    # ------------------------------------------------------------------

    def embed_multimodal(self, **kwargs) -> (
        list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None
    ):
        """Encode all modalities through OmniEncoder."""
        embeddings: list[torch.Tensor] = []

        # Image
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            image_grid_thw = kwargs["image_grid_thw"]
            # Handle NestedTensors
            if isinstance(pixel_values, (list, tuple)):
                pixel_values = torch.cat([p if p.dim() == 2 else p.flatten(0, -2)
                                         for p in pixel_values], dim=0)
            if isinstance(image_grid_thw, (list, tuple)):
                image_grid_thw = torch.cat(image_grid_thw, dim=0)

            # Ensure correct dtype (processor outputs float32, model is bf16)
            target_dtype = self.omni_encoder.vision_embeddings.patch_embedding.weight.dtype
            pixel_values = pixel_values.to(dtype=target_dtype)

            img_embeds = self.omni_encoder.forward_vision(pixel_values, image_grid_thw)

            # Split into per-image embeddings (vLLM expects one tensor per item)
            spatial_merge = self.omni_encoder.spatial_merge_size
            offset = 0
            for i in range(image_grid_thw.shape[0]):
                t, h, w = int(image_grid_thw[i, 0]), int(image_grid_thw[i, 1]), int(image_grid_thw[i, 2])
                n_tokens = t * (h // spatial_merge) * (w // spatial_merge)
                embeddings.append(img_embeds[offset:offset + n_tokens])
                offset += n_tokens

        # Audio
        audios = kwargs.get("audios")
        if audios is not None:
            # audios may be flat [sum_T, n_mels] with audio_lens, or list of tensors.
            # audio_lens is PER-CHUNK (one API audio item can be split into
            # multiple chunks); audio_num_chunks gives chunks-per-item so we can
            # merge each item's chunk embeddings back into ONE embedding — vLLM
            # requires len(mm_embeddings) == number of audio items.
            audio_lens = kwargs.get("audio_lens")
            if isinstance(audios, torch.Tensor) and audio_lens is not None:
                # Flat packed: split back to per-chunk list
                if isinstance(audio_lens, torch.Tensor):
                    lens_list = audio_lens.tolist()
                else:
                    lens_list = list(audio_lens)
                audios = list(audios.split(lens_list, dim=0))
            elif isinstance(audios, torch.Tensor):
                audios = [audios]

            # Per-chunk encode
            aud_embeds, aud_lens = self.omni_encoder.forward_audio(audios)
            per_chunk = []
            offset = 0
            for l in aud_lens.tolist():
                per_chunk.append(aud_embeds[offset:offset + l])
                offset += l

            # Merge chunks -> per-item embeddings using audio_num_chunks
            num_chunks = kwargs.get("audio_num_chunks")
            if num_chunks is not None:
                if isinstance(num_chunks, torch.Tensor):
                    nc_list = num_chunks.flatten().tolist()
                else:
                    nc_list = [int(x) for x in (
                        num_chunks if isinstance(num_chunks, (list, tuple)) else [num_chunks])]
            else:
                nc_list = [1] * len(per_chunk)

            per_item = []
            ci = 0
            for nc in nc_list:
                nc = int(nc)
                if nc <= 0:
                    continue
                per_item.append(torch.cat(per_chunk[ci:ci + nc], dim=0))
                ci += nc
            # Safety: if accounting didn't consume everything, fall back to per-chunk
            if ci != len(per_chunk):
                per_item = per_chunk

            # Audio embeddings are added without tag — standard _merge_multimodal_embeddings
            # scatters them to audio_pad positions by prompt order.
            embeddings.extend(per_item)

        # Video: interleaved omni fusion (vision frames + audio chunks in one
        # <|video_start|>…<|video_end|> region). This is the only video path.
        video_split = kwargs.get("video_split")
        video_images = kwargs.get("video_images")
        video_interleave_orders = kwargs.get("video_interleave_orders")
        if video_images is not None:
            video_image_grid_thw = kwargs.get("video_image_grid_thw")
            video_audios_flat = kwargs.get("video_audios")
            video_audio_lens = kwargs.get("video_audio_lens")

            if isinstance(video_images, (list, tuple)):
                video_images = torch.cat([v if v.dim() == 2 else v.flatten(0, -2)
                                         for v in video_images], dim=0)
            if isinstance(video_image_grid_thw, (list, tuple)):
                video_image_grid_thw = torch.cat(video_image_grid_thw, dim=0)

            # Convert flat video_audios + lens to list of tensors
            video_audios = None
            if video_audios_flat is not None and video_audio_lens is not None:
                if isinstance(video_audios_flat, torch.Tensor) and isinstance(video_audio_lens, torch.Tensor):
                    lens_list = video_audio_lens.tolist()
                    video_audios = list(video_audios_flat.split(lens_list, dim=0))
                elif isinstance(video_audios_flat, (list, tuple)):
                    video_audios = list(video_audios_flat)

            # Ensure correct dtype
            target_dtype = self.omni_encoder.vision_embeddings.patch_embedding.weight.dtype
            video_images = video_images.to(dtype=target_dtype)

            vis_embeds, aud_result = self.omni_encoder.forward_video(
                video_images, video_image_grid_thw, video_audios, video_split,
                video_interleave_orders)

            # Split vision per frame
            spatial_merge = self.omni_encoder.spatial_merge_size
            vis_per_frame = []
            vis_offset = 0
            for i in range(video_image_grid_thw.shape[0]):
                t, h, w = int(video_image_grid_thw[i, 0]), int(video_image_grid_thw[i, 1]), int(video_image_grid_thw[i, 2])
                n_tokens = t * (h // spatial_merge) * (w // spatial_merge)
                vis_per_frame.append(vis_embeds[vis_offset:vis_offset + n_tokens])
                vis_offset += n_tokens

            # Split audio per chunk
            aud_per_chunk = []
            if aud_result is not None:
                aud_embeds, aud_lens = aud_result
                aud_offset = 0
                for l in aud_lens.tolist():
                    aud_per_chunk.append(aud_embeds[aud_offset:aud_offset + l])
                    aud_offset += l

            # Build per-video interleaved embeddings. video_split [N,2] gives
            # (num_frames, num_audio_chunks) per video; interleave_orders idx is
            # PER-VIDEO-LOCAL (reset each video in the processor), so we slice
            # vis_per_frame / aud_per_chunk per video and index each video's slice
            # with its local idx. Emits N embeddings (one per <|video|> region);
            # single video -> N=1 (backward compatible).
            if isinstance(video_split, (list, tuple)):
                video_split = torch.stack(
                    [s if isinstance(s, torch.Tensor) else torch.tensor(s) for s in video_split])
            if video_split is not None and torch.is_tensor(video_split) \
                    and video_split.numel() > 0:
                frame_counts = video_split[:, 0].tolist()
                chunk_counts = video_split[:, 1].tolist()
                event_counts = video_split.sum(dim=1).tolist()

                # orders: flatten to a [N_events_total, 2] tensor
                orders = video_interleave_orders
                if isinstance(orders, (list, tuple)):
                    orders = orders[0] if len(orders) == 1 else orders
                if orders is not None and not torch.is_tensor(orders):
                    orders = torch.tensor(orders)

                vis_off = aud_off = evt_off = 0
                for vi in range(len(frame_counts)):
                    nf, na, ne = frame_counts[vi], chunk_counts[vi], event_counts[vi]
                    vis_vi = vis_per_frame[vis_off:vis_off + nf]
                    aud_vi = aud_per_chunk[aud_off:aud_off + na]
                    parts = []
                    if orders is not None and ne > 0:
                        for row in orders[evt_off:evt_off + ne]:
                            m, idx = int(row[0].item()), int(row[1].item())
                            if m == 0 and idx < len(vis_vi):
                                parts.append(vis_vi[idx])
                            elif m == 1 and idx < len(aud_vi):
                                parts.append(aud_vi[idx])
                    else:
                        parts = vis_vi + aud_vi
                    if parts:
                        embeddings.append(torch.cat(parts, dim=0))
                    vis_off += nf
                    aud_off += na
                    evt_off += ne
            else:
                # Defensive fallback (video_split is always produced by the
                # processor for the video path): no per-video info -> concat.
                parts = vis_per_frame + aud_per_chunk
                if parts:
                    embeddings.append(torch.cat(parts, dim=0))

        if not embeddings:
            return None
        return tuple(embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter multimodal embeddings into text embeddings."""
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or is_multimodal is None:
            return inputs_embeds

        # Standard scatter: all multimodal embeddings (image, audio, video)
        # are flattened and scattered to is_multimodal positions in order.
        # Video uses image_pad (128257) for vision frames and audio_pad (128265)
        # for audio chunks in the prompt — both are selected by is_multimodal,
        # and the embedding order matches because _process_video_fusion builds
        # the interleaved embedding in the same order as the prompt tokens.
        merged = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return merged

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass."""
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            mm_embeddings = self.embed_multimodal(**kwargs)
            is_multimodal = kwargs.get("is_multimodal")
            inputs_embeds = self.embed_input_ids(
                input_ids, mm_embeddings, is_multimodal=is_multimodal)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors,
            inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
