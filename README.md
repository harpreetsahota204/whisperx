# WhisperX — FiftyOne Remote Zoo Model

Wraps [WhisperX](https://github.com/m-bain/whisperX) speech-to-text as a
remotely-sourced FiftyOne zoo model. Attach timestamped, optionally
speaker-labeled transcripts to a video dataset with a single
`dataset.apply_model(...)` call.

Each ASR segment becomes a `fo.TemporalDetection` (label `"speech"`, with
`text` and `speaker` attributes), and the flat transcript is stored as a
dynamic `transcript` attribute on the returned `fo.TemporalDetections`.

## Requirements

- A FiftyOne dataset of **videos**.
- `whisperx`, `torch`, `torchaudio` (installed automatically when you register
  and load the model, or `pip install whisperx`).
- **GPU recommended.** `large-v3` in `float16` fits comfortably on a 48GB card;
  use `int8_float16` for headroom.
- **Diarization** (`diarize=True`) additionally needs an `HF_TOKEN` environment
  variable and acceptance of the pyannote model license on Hugging Face.

## Usage

```python
import fiftyone as fo
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/whisperx", overwrite=True
)

model = foz.load_zoo_model("whisperx-large-v3")  # or pass diarize=True

dataset = fo.load_dataset("my-videos")

# REQUIRED: from_timestamps() needs metadata.frame_rate to map seconds -> frames
dataset.compute_metadata()

dataset.apply_model(model, label_field="transcript_segments")

# Materialize the flat transcript as a top-level StringField. The model stashes
# it as a dynamic attribute on the TemporalDetections; surface it, then copy it
# up to a real, queryable field.
dataset.add_dynamic_sample_fields()
dataset.set_values(
    "transcript", dataset.values("transcript_segments.transcript")
)
```

### Available models

| `base_name`               | Notes                                              |
| ------------------------- | -------------------------------------------------- |
| `whisperx-large-v3`       | Best accuracy.                                      |
| `whisperx-large-v3-turbo` | Higher throughput, small accuracy cost; English-strong. |
| `whisperx-large-v2`       | Previous-generation large model.                    |

### Load-time options

```python
model = foz.load_zoo_model(
    "whisperx-large-v3",
    compute_type="int8_float16",  # ctranslate2 compute type
    device="cuda",                # or "cpu"
    diarize=True,                 # speaker labels (needs HF_TOKEN + license)
    batch_size=16,
    language="en",                # force a language; None auto-detects
)
```

### Runtime options (no reload)

`batch_size` and `language` are per-call parameters exposed as setters, so you
can change them without reconstructing the model (and reloading weights):

```python
model.batch_size = 8
model.language = "es"
dataset.apply_model(model, label_field="transcript_segments")
```

## Output schema

- `transcript_segments`: `fo.TemporalDetections` — one detection per ASR
  segment, frame-aligned, each with `label="speech"`, `text`, and `speaker`.
  Scrubbable in the App and filterable by speaker.
- `transcript_segments.transcript`: flat transcript string (dynamic attribute).

## Notes

- Video models are applied via a per-sample loop, not a PyTorch DataLoader, so
  there is no multi-worker / `num_workers` consideration here.
- Word-level timings exist in the WhisperX result but are not stored by default
  (heavy and rarely needed).
- Mixed-language datasets load one alignment model per detected language into
  memory.
