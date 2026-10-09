# vllm-plugin-vita-omni

Out-of-tree [vLLM](https://github.com/vllm-project/vllm) plugin that serves
**Youtu-Parsing-Omni** (`model_type = youtu_vita`, architecture
`YoutuVITAForCausalLM`) with image, audio and audio-visual video inputs.

The plugin is registered through the `vllm.general_plugins` entry point, so it
is loaded automatically once installed:

```bash
pip install --no-deps ./vllm-plugin-vita-omni
python -c "import vita_omni; print('vita_omni OK')"
```

Tested stack (see [`requirements/vllm.txt`](../requirements/vllm.txt)):
`vllm==0.19.0`, `transformers==5.2.0`, `torchcodec==0.10.0`, `flash-attn==2.8.3`,
`decord==0.6.0`, `soundfile==0.13.1`, `torchaudio==2.10.0` and an `ffmpeg` binary
(used to extract the audio track of videos). `peft` must not be installed in the
serving environment. `bash scripts/setup_env.sh` from the `youtu_parsing_omni` directory
installs this stack together with the plugin.

## What it registers

| Item | Value |
| --- | --- |
| Config class | `youtu_vita` (`YoutuVITAConfig`) |
| Model class | `YoutuVITAForCausalLM` (MLA decoder) |
| Multimodal processor | `VITAMultiModalProcessor` (image / audio / video with interleaved audio track) |
| Online-serving patch | `OpenAIServingChat.create_chat_completion`: passes the local path of every `file://` `video_url` to the processor so that the audio track is decoded (`youtu_vita` only) |
| Offline patch | `LLM.chat`: the same video-path injection for offline inference (`youtu_vita` only) |

## Runtime knobs

The processor reads its token budgets from the checkpoint `processor_config.json`
(`image_processor` / `video_processor`) and then applies the following optional
environment overrides. The `*_MAX_TOKENS` variables can only lower the checkpoint
value; the `*_MIN_TOKENS` variables replace it.

| Variable | `scripts/vllm.sh` value | Meaning |
| --- | ---: | --- |
| `YOUTU_VITA_IMAGE_MAX_TOKENS` | 16384 | cap on merged visual tokens per image |
| `YOUTU_VITA_VIDEO_TOTAL_MAX_TOKENS` (falls back to `YOUTU_VITA_VIDEO_MAX_TOKENS`) | 16384 | cap on visual tokens per video |
| `YOUTU_VITA_VIDEO_IMAGE_MAX_TOKENS` | 512 | cap on visual tokens per video frame |
| `YOUTU_VITA_VIDEO_IMAGE_MIN_TOKENS` | 4 | minimum visual tokens per frame |
| `YOUTU_VITA_VIDEO_MIN_TOKENS` | 4 | minimum visual tokens per video |
| `YOUTU_VITA_AUDIO_TOKENIZER_PATH` | not set | directory with the whisper-large-v3 `preprocessor_config.json` |

The whisper feature-extractor config is resolved in this order:
`YOUTU_VITA_AUDIO_TOKENIZER_PATH` (if it is a directory), then
`feature_extractor.audio_tokenizer_path` of the checkpoint `processor_config.json`
(skipped if it is an absolute path that does not exist on this machine), then the
copy bundled in `vita_omni/assets/whisper-large-v3`.

`scripts/vllm.sh` sets these variables together with the `vllm serve` flags used
for the reported results.

The processor reads the following keys from the merged server
(`--mm-processor-kwargs`) and request (`mm_processor_kwargs`) arguments:

| Key | Default | Meaning |
| --- | ---: | --- |
| `nframes` | 64 | maximum number of sampled video frames |
| `use_audio_in_video` | `true` | decode the audio track of a video (see [Video requests](#video-requests)) |
| `use_vision_in_video` | `true` | always forced to `true` (audio-only video is not supported) |
| `video_audio_chunk_min_second` / `video_audio_chunk_max_second` | 1.5 / 29.5 | length bounds of the audio chunks interleaved with video frames |
| `audio_chunk_min_second` / `audio_chunk_max_second` | 2.0 / 30.0 | length bounds of the chunks of audio inputs |

Video frames are sampled at no more than 1 fps and at most `nframes` frames. The
media token budgets are not read from these arguments but from the checkpoint
and the environment variables above. `scripts/vllm.sh` passes
`{"video_audio_chunk_max_second": 29.5, "video_audio_chunk_min_second": 1.5}`.

## Video requests

Send a single video per request. When serving through the OpenAI API the plugin
extracts the local path from the `video_url` item and decodes frames with
`decord` and the audio track with `torchaudio`/`ffmpeg`. The video has to be sent
as a `file://` URL below `--allowed-local-media-path`; other URLs fall back to the
frames decoded by vLLM, which carry no audio. The plugin only applies to
`youtu_vita` models and ignores any `_video_file_paths` sent by the client. When a
`file://` video is passed this way, `use_audio_in_video` is dropped from the
request and the audio track is always decoded if the video has one.
The recommended request additionally sends:

```json
"mm_processor_kwargs": {"nframes": 64, "use_audio_in_video": true,
                         "use_vision_in_video": true}
```

Offline `LLM.chat` gets the same treatment when it is called with an
`mm_processor_kwargs` dict and `file://` `video_url` items (below the engine's
`allowed_local_media_path`). Batching several videos in one `LLM.chat` call is not
supported: every sample would be decoded from the first video.
