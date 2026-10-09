"""Lightweight VITA config classes for vLLM plugin.

These bypass the transformers version incompatibility by constructing
config objects directly from the raw config.json dict.
vLLM's hf_config will be an opaque object; we attach omni_config and
text_config as attribute namespaces.

The ``youtu_vita`` model uses an MLA decoder (DeepSeek-style) and an omni encoder
with video_group_attention / video_fusion_layer_freq /
video_omni_chunked_mthw_rope and temporal_merge_size=1.

For ``youtu_vita`` the text_config is shaped so that vLLM can drive it through
its built-in ``DeepseekV2ForCausalLM`` (dense variant, ``n_routed_experts=None``).
"""
from __future__ import annotations

import logging

from transformers import PretrainedConfig

_LOG = logging.getLogger(__name__)


class _Namespace:
    """Simple attribute namespace from a dict.

    Implements the minimal interface that vLLM expects from a text_config:
    - get_text_config() -> self
    - standardize_rope_params() -> no-op
    - to_dict() / to_json_string()
    """
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"_Namespace({self.__dict__})"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def to_json_string(self):
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def get_text_config(self, *args, **kwargs):
        """vLLM calls config.get_text_config() to find the LM config."""
        return self

    def standardize_rope_params(self):
        """vLLM calls this to normalize RoPE config. No-op for us."""
        pass

    def validate_rope(self):
        """vLLM calls this after standardize_rope_params. No-op."""
        pass

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _set_default(obj, key, value):
    if not hasattr(obj, key) or getattr(obj, key) is None:
        setattr(obj, key, value)


# ===========================================================================
# youtu_vita (MLA decoder + omni encoder)
# ===========================================================================
# YARN passthrough fields (besides rope_type / rope_theta). vLLM's YaRN rope
# (get_rope) + DeepseekV2 MLA read these from rope_parameters.
_YARN_PASSTHROUGH_KEYS = (
    "factor",
    "original_max_position_embeddings",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "attn_factor",
    "truncate",
)


def _normalize_rope_parameters(tc) -> dict:
    """Build a vLLM-compatible ``rope_parameters`` dict for the MLA decoder.

    HF ``YoutuVITATextConfig`` carries either a top-level ``rope_theta`` or a
    nested ``rope_parameters`` dict. vLLM's DeepseekV2 reads
    ``config.rope_parameters`` (a dict with at least ``rope_type`` and
    ``rope_theta``).

    Behavior:
    - If the checkpoint's ``rope_parameters`` declares a NON-``default``
      ``rope_type`` (e.g. ``yarn``), we PASS THROUGH the YaRN config
      (factor / original_max_position_embeddings / beta_* / mscale*) so YaRN
      length extrapolation is enabled. vLLM's DeepseekV2 MLA + ``get_rope``
      then build the YaRN rotary embedding and apply the mscale correction.
    - Otherwise we emit ``rope_type="default"`` (plain RoPE, no YaRN).
    """
    rp = getattr(tc, 'rope_parameters', None)

    # Resolve rope_theta (nested rope_parameters > flat rope_theta > 10000).
    theta = None
    if isinstance(rp, dict):
        theta = rp.get('rope_theta')
    if theta is None:
        theta = getattr(tc, 'rope_theta', None)
    if theta is None:
        theta = 10000.0

    rope_type = None
    if isinstance(rp, dict):
        # HF may store the type under "rope_type" or "type".
        rope_type = rp.get('rope_type') or rp.get('type')

    # YaRN enabled: pass through the scaling config verbatim.
    if isinstance(rp, dict) and rope_type and rope_type != 'default':
        out: dict = {"rope_type": str(rope_type), "rope_theta": float(theta)}
        for k in _YARN_PASSTHROUGH_KEYS:
            if k in rp and rp[k] is not None:
                out[k] = rp[k]
        # Log the effective YaRN parameters fed to vLLM. This normalize step is
        # on the always-executed config path, so it reliably reflects what the
        # MLA decoder (vLLM's DeepseekV2ForCausalLM) actually uses.
        factor = out.get("factor")
        original = out.get("original_max_position_embeddings")
        target = None
        try:
            target = int(original) * float(factor)
        except (TypeError, ValueError):
            pass
        _LOG.info(
            "[YaRN] enabled: rope_type=%s rope_theta=%s factor=%s "
            "original_max_position_embeddings=%s beta_fast=%s beta_slow=%s "
            "mscale=%s mscale_all_dim=%s extrapolation_target~=%s",
            out.get("rope_type"), out.get("rope_theta"), factor, original,
            out.get("beta_fast"), out.get("beta_slow"),
            out.get("mscale"), out.get("mscale_all_dim"), target)
        return out

    # Plain RoPE (no YaRN).
    return {"rope_type": "default", "rope_theta": float(theta)}


def _apply_mla_text_defaults(tc) -> None:
    """Ensure the text_config has every field vLLM DeepseekV2 (dense MLA) needs."""
    # Core dims
    _set_default(tc, 'hidden_size', 2560)
    _set_default(tc, 'num_hidden_layers', 40)
    _set_default(tc, 'num_attention_heads', 32)
    _set_default(tc, 'num_key_value_heads', 8)
    _set_default(tc, 'intermediate_size', 9728)
    _set_default(tc, 'vocab_size', 133632)
    _set_default(tc, 'rms_norm_eps', 1e-6)
    _set_default(tc, 'hidden_act', 'silu')
    _set_default(tc, 'max_position_embeddings', 4096000)
    _set_default(tc, 'attention_bias', False)
    _set_default(tc, 'tie_word_embeddings', True)

    # ---- MLA-specific (DeepSeek-style) ----
    _set_default(tc, 'q_lora_rank', 1536)
    _set_default(tc, 'kv_lora_rank', 512)
    _set_default(tc, 'qk_nope_head_dim', 128)
    _set_default(tc, 'qk_rope_head_dim', 64)
    _set_default(tc, 'v_head_dim', 128)
    # qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    _set_default(tc, 'qk_head_dim',
                 int(tc.qk_nope_head_dim) + int(tc.qk_rope_head_dim))

    # ---- Force DENSE: no MoE in youtu_vita text LM ----
    # Set n_routed_experts=0 (not None) so that:
    # 1) __init__: ``n_routed_experts is not None`` → True, but
    #    ``layer_idx >= first_k_dense_replace(=num_hidden_layers)`` → always
    #    False → every layer uses DeepseekV2MLP (dense SwiGLU).
    # 2) load_weights: ``n_routed_experts + 0`` → 0 (no NoneType arithmetic
    #    error in SharedFusedMoE.make_expert_params_mapping).
    tc.n_routed_experts = 0
    _set_default(tc, 'first_k_dense_replace', int(tc.num_hidden_layers))
    _set_default(tc, 'moe_layer_freq', 1)
    _set_default(tc, 'n_shared_experts', 0)
    _set_default(tc, 'num_experts_per_tok', 0)
    _set_default(tc, 'moe_intermediate_size', 0)
    _set_default(tc, 'norm_topk_prob', False)
    _set_default(tc, 'routed_scaling_factor', 1.0)

    # ---- vLLM MLA flags / misc ----
    _set_default(tc, 'use_mla', True)
    _set_default(tc, 'num_nextn_predict_layers', 0)
    # vLLM uses model_type to pick the model class only at the top level;
    # the text sub-config is consumed directly by DeepseekV2ForCausalLM, so we
    # label it accordingly to keep DeepseekV2 internal asserts happy.
    tc.model_type = 'deepseek_v2'

    # ---- RoPE ----
    tc.rope_parameters = _normalize_rope_parameters(tc)
    # keep a flat rope_theta too (some vLLM paths read it)
    _set_default(tc, 'rope_theta', tc.rope_parameters['rope_theta'])


def _apply_omni_defaults(oc, temporal_merge_size: int = 1) -> None:
    """Ensure omni_config has all fields the encoder needs (incl. new features)."""
    _set_default(oc, 'hidden_size', 1024)
    _set_default(oc, 'num_attention_heads', 16)
    _set_default(oc, 'num_key_value_heads', 8)
    _set_default(oc, 'num_hidden_layers', 28)
    _set_default(oc, 'intermediate_size', 3072)
    _set_default(oc, 'head_dim', 128)
    _set_default(oc, 'rms_norm_eps', 1e-6)
    _set_default(oc, 'rope_theta', 10000.0)
    _set_default(oc, 'patch_size', 16)
    _set_default(oc, 'num_channels', 3)
    _set_default(oc, 'spatial_merge_size', 2)
    _set_default(oc, 'temporal_merge_size', temporal_merge_size)
    _set_default(oc, 'num_mel_bins', 128)
    _set_default(oc, 'downsample_hidden_size', 512)
    _set_default(oc, 'conv_chunksize', 500)
    _set_default(oc, 'n_window', 50)
    _set_default(oc, 'n_window_infer', 800)
    _set_default(oc, 'temporal_patch_size', 1)
    _set_default(oc, 'out_hidden_size', getattr(oc, 'hidden_size', 1024))
    _set_default(oc, 'merger_hidden_size', getattr(oc, 'out_hidden_size', 2560))

    # ---- new omni features (youtu_vita) ----
    _set_default(oc, 'video_group_attention', False)
    _set_default(oc, 'video_fusion_layer_freq', None)
    _set_default(oc, 'video_omni_chunked_mthw_rope', False)
    _set_default(oc, 'video_omni_interleaved_mthw_rope', False)
    _set_default(oc, 'video_omni_interleaved_thw_section', None)
    _set_default(oc, 'rope_m_dim', 0)
    _set_default(oc, 'rope_theta_m', 100.0)


class YoutuVITAConfig(PretrainedConfig):
    """Minimal config for VITA OmniEncoder + MLA (Youtu) LM.

    text_config is shaped to be consumed by vLLM's DeepseekV2ForCausalLM in its
    dense (no-MoE) configuration; omni_config carries the new encoder features.
    """
    model_type = "youtu_vita"

    def __init__(self, **kwargs):
        text_config_dict = kwargs.pop("text_config", {}) or {}
        omni_config_dict = kwargs.pop("omni_config", {}) or {}
        # youtu_vita uses omni only; vision/audio sub-configs are null.
        kwargs.pop("vision_config", None)
        kwargs.pop("audio_config", None)

        self._text_config_dict = text_config_dict if isinstance(text_config_dict, dict) else {}
        self._omni_config_dict = omni_config_dict if isinstance(omni_config_dict, dict) else {}

        # top-level flag controlling joint video fusion forward
        self.video_omni_fusion = bool(kwargs.pop("video_omni_fusion", False))

        super().__init__(**kwargs)

        self.text_config = _Namespace(text_config_dict) if isinstance(
            text_config_dict, dict) else text_config_dict
        self.omni_config = _Namespace(omni_config_dict) if isinstance(
            omni_config_dict, dict) else omni_config_dict

        if not hasattr(self, 'vocab_size') or self.vocab_size is None:
            self.vocab_size = getattr(self.text_config, 'vocab_size', 133632)
        if not hasattr(self, 'hidden_size') or self.hidden_size is None:
            self.hidden_size = getattr(self.text_config, 'hidden_size', 2560)

        _apply_mla_text_defaults(self.text_config)
        _apply_omni_defaults(self.omni_config, temporal_merge_size=1)

        # ---- Flatten LM/MLA fields to TOP LEVEL ----
        # vLLM's DeepseekV2ForCausalLM.__init__ reads MLA/LM fields directly
        # from ``vllm_config.model_config.hf_config`` (the TOP-LEVEL config),
        # NOT from ``hf_config.text_config``. So mirror every field the MLA
        # decoder needs onto ``self`` (omni still reads ``self.omni_config``).
        self._mirror_text_fields_to_top_level()

    _LM_TOP_LEVEL_FIELDS = (
        'hidden_size', 'num_hidden_layers', 'num_attention_heads',
        'num_key_value_heads', 'intermediate_size', 'vocab_size',
        'rms_norm_eps', 'hidden_act', 'max_position_embeddings',
        'attention_bias', 'tie_word_embeddings',
        # MLA
        'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim',
        'v_head_dim', 'qk_head_dim',
        # dense/MoE control
        'n_routed_experts', 'first_k_dense_replace', 'moe_layer_freq',
        'n_shared_experts', 'num_experts_per_tok', 'moe_intermediate_size',
        'norm_topk_prob', 'routed_scaling_factor',
        # MLA flags / rope
        'use_mla', 'num_nextn_predict_layers', 'rope_parameters', 'rope_theta',
    )

    def _mirror_text_fields_to_top_level(self) -> None:
        tc = self.text_config
        for f in self._LM_TOP_LEVEL_FIELDS:
            # Always mirror fields that exist on text_config, including None
            # values. DeepseekV2DecoderLayer does ``config.n_routed_experts is
            # not None`` which requires the attribute to EXIST (not just be
            # truthy); skipping None values causes AttributeError.
            if hasattr(tc, f):
                setattr(self, f, getattr(tc, f))
        # IMPORTANT: keep top-level model_type as "youtu_vita" so vLLM picks our
        # registered model class; the TEXT sub-config is labeled "deepseek_v2"
        # (set in _apply_mla_text_defaults) only for DeepseekV2 internal asserts.
        # DeepseekV2ForCausalLM checks ``config.model_type == "deepseek"`` for
        # MHA; with qk_*_head_dim>0 it goes MLA regardless, so top-level
        # model_type "youtu_vita" is fine.

    def get_text_config(self, *args, **kwargs):
        return self.text_config

    def to_dict(self):
        output = super().to_dict()
        output["text_config"] = self._text_config_dict
        output["omni_config"] = self._omni_config_dict
        output["video_omni_fusion"] = self.video_omni_fusion
        output.pop("_text_config_dict", None)
        output.pop("_omni_config_dict", None)
        return output
