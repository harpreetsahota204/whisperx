# Whisper (Transformers) — FiftyOne Remote Zoo Model

<div align="center">
<p align="center">

<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">

**The open-source tool for building high-quality datasets and computer vision
models**

---

<!-- prettier-ignore -->
<a href="https://voxel51.com/fiftyone?utm_source=harpreet-gh">Website</a> •
<a href="https://docs.voxel51.com?utm_source=harpreet-gh">Docs</a> •
<a href="https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb?utm_source=harpreet-gh">Try it Now</a> •
<a href="https://docs.voxel51.com/getting_started_guides/index.html?utm_source=harpreet-gh">Getting Started Guides</a> •
<a href="https://docs.voxel51.com/tutorials/index.html?utm_source=harpreet-gh">Tutorials</a> •
<a href="https://voxel51.com/blog/?utm_source=harpreet-gh">Blog</a> •
<a href="https://discord.gg/fiftyone-community?utm_source=harpreet-gh">Community</a>

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</p>
</div>


Wraps OpenAI [Whisper](https://huggingface.co/openai/whisper-large-v3)
speech-to-text, run via the 🤗 Transformers `automatic-speech-recognition`
pipeline, as a remotely-sourced FiftyOne zoo model. Attach timestamped
transcripts to a video dataset with a single `dataset.apply_model(...)` call.

In a single pass the model writes **two** sample fields: a
`fo.TemporalDetections` of segments (label `"speech"`, each with a `text`
attribute) and a flat `transcript` string field.

> **Why Transformers and not WhisperX?** WhisperX's CTranslate2 backend only
> ships GPU wheels for x86, so it falls back to CPU on ARM GPUs (e.g. NVIDIA
> GB10 / DGX Spark). The Transformers pipeline is pure PyTorch, so it uses the
> GPU consistently across CUDA (x86 *and* ARM), MPS, and CPU.

## Requirements

- A FiftyOne dataset of **videos** (with audio tracks).
- `torch`, `transformers>=4.56`, `accelerate`, plus a working **ffmpeg** on
  PATH (the pipeline demuxes audio with it). FiftyOne checks these at load
  time; pass `install_requirements=True` to `load_zoo_model()` to install
  them automatically.
- **GPU recommended** for `large-v3`. `torch_dtype` auto-selects `float16` on
  cuda/mps and `float32` on cpu.

## Usage

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import load_from_hub

# Instructional videos with audio; ships with its own ASR `transcript` /
# `transcript_segments` fields, so we write Whisper's output to `whisper_*`
dataset = load_from_hub("Voxel51/action100m_tiny_subset", max_samples=10)

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/whisperx", overwrite=True
)

model = foz.load_zoo_model("whisper-large-v3")

# REQUIRED: from_timestamps() needs metadata.frame_rate to map seconds -> frames
dataset.compute_metadata()

# predict() returns {"segments": TemporalDetections, "transcript": str}.
# A dict label_field names both fields exactly, in one pass:
dataset.apply_model(
    model,
    label_field={"segments": "whisper_segments", "transcript": "whisper_transcript"},
)
```

Then inspect the results:

```python
sample = dataset.first()

# Flat transcript: a top-level StringField, queryable on its own
print(sample.whisper_transcript)

# Timestamped segments: TemporalDetections with text per segment
for det in sample.whisper_segments.detections:
    print(det.support, det.text)
```

`label_field` controls how the two outputs are named:

| `label_field` | Resulting fields |
| --- | --- |
| `{"segments": "whisper_segments", "transcript": "whisper_transcript"}` | `whisper_segments`, `whisper_transcript` |
| `"audio"` (a string prefix) | `audio_segments`, `audio_transcript` |
| `None` | `segments`, `transcript` |

### Available models

| `base_name`             | HF model                       | Notes                                   |
| ----------------------- | ------------------------------ | --------------------------------------- |
| `whisper-large-v3`      | `openai/whisper-large-v3`      | Best accuracy.                          |
| `whisper-large-v3-turbo`| `openai/whisper-large-v3-turbo`| Higher throughput, small accuracy cost. |
| `whisper-large-v2`      | `openai/whisper-large-v2`      | Previous-generation large model.        |

### Load-time options

```python
model = foz.load_zoo_model(
    "whisper-large-v3",
    device=None,          # None auto-selects cuda > mps > cpu
    torch_dtype=None,     # None -> float16 on cuda/mps, float32 on cpu
    batch_size=16,
    chunk_length_s=None,  # None = accurate sequential long-form; set 30 for faster chunked
    language="en",        # force a language; None auto-detects
    # model_id="openai/whisper-large-v3",  # override the HF id directly
)
```

Device memory is freed automatically when `apply_model` finishes.

### From the App (Apply Model operator)

After registering the source, you can run Whisper from the App with no code:

1. Launch the App (`session = fo.launch_app(dataset)`) on a **video** dataset and run `dataset.compute_metadata()` first.
2. Open the operator palette (backtick `` ` ``) and choose **Apply Model**.
3. Pick the model from the list — `whisper-large-v3` / `-large-v3-turbo` / `-large-v2`.
4. Fill the form: **Batch size**, **Language** (empty = auto-detect).
5. Set the **label field** (a string prefix, e.g. `audio`) and execute.

The operator collects a single string label field, so it writes both outputs
prefixed: e.g. label field `audio` produces `audio_segments`
(`TemporalDetections`) and `audio_transcript` (string).

### Runtime options (no reload)

`batch_size`, `chunk_length_s`, and `language` are per-call parameters exposed
as setters, so you can change them without reconstructing the model (and
reloading weights):

```python
model.batch_size = 8
model.chunk_length_s = 30
model.language = "es"
dataset.apply_model(
    model,
    label_field={"segments": "whisper_segments", "transcript": "whisper_transcript"},
)
```

## Output schema

Using `label_field={"segments": "whisper_segments", "transcript": "whisper_transcript"}`:

- `whisper_segments`: `fo.TemporalDetections` — one detection per Whisper
  segment, frame-aligned, each with `label="speech"` and `text`. Scrubbable in
  the App.
- `whisper_transcript`: top-level `StringField` with the flat transcript —
  queryable on its own (text filters, `match`, embeddings, VLM context).

## Notes

- Timestamps are **segment-level** (from Whisper's `return_timestamps=True`).
  This is coarser than WhisperX's wav2vec2 forced alignment; word-level
  (`return_timestamps="word"`) and speaker diarization are possible future
  additions.
- Video models are applied via a per-sample loop, not a PyTorch DataLoader, so
  there is no `num_workers` consideration here.
- Videos with **no audio stream** make the pipeline's ffmpeg read fail for
  that sample. `apply_model` skips failed samples by default
  (`skip_failures=True`), logging a warning and leaving both fields unset;
  pass `skip_failures=False` to raise instead.
