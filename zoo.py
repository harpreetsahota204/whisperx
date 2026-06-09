"""
Whisper speech-to-text (Hugging Face Transformers) as a FiftyOne zoo model.

Uses the Transformers ``automatic-speech-recognition`` pipeline rather than
WhisperX/CTranslate2 so the same code runs on whatever device PyTorch supports
(CUDA on x86 *and* ARM/Grace-Blackwell, MPS, CPU) — CTranslate2 ships GPU wheels
for x86 only, so it was CPU-bound on aarch64 GPUs like the GB10.

Contract:
  - media_type = "video"  -> FiftyOne routes through ``_apply_video_model``
                             (a per-sample loop, NOT a DataLoader).
  - SamplesMixin          -> predict() gets the sample so we can read
                             ``sample.filepath``; the ASR pipeline demuxes audio
                             from the media file with ffmpeg.
  - predict() returns two sample-level outputs in one pass:
        "segments"   -> fo.TemporalDetections (segment-level timestamps, text)
        "transcript" -> flat transcript string

    Map the dict keys to fields via ``label_field`` (dict = exact names, string
    = prefix, None = keys verbatim)::

        dataset.apply_model(
            model,
            label_field={"segments": "transcript_segments", "transcript": "transcript"},
        )

Precondition: run ``dataset.compute_metadata()`` before ``apply_model`` so that
``TemporalDetection.from_timestamps()`` can convert seconds -> frame support.
"""

import torch

import fiftyone as fo
from fiftyone.core.models import Model, SamplesMixin


def get_device():
    """Returns the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhisperModel(SamplesMixin, Model):
    """Whisper ASR (Transformers pipeline) wrapped as a FiftyOne video model.

    Construction-only parameters (changing them reloads weights):
        model_id: Hugging Face model id, e.g. ``"openai/whisper-large-v3"``
        device: ``"cuda"`` / ``"mps"`` / ``"cpu"``; ``None`` (default)
            auto-selects via :func:`get_device`
        torch_dtype: torch dtype for the model; ``None`` (default) picks
            ``float16`` on cuda/mps and ``float32`` on cpu

    Runtime parameters (per-call setters, no reload needed):
        batch_size: number of chunks decoded per batch
        chunk_length_s: ``None`` (default) uses Whisper's native sequential
            long-form decoding (more accurate); set a value (e.g. 30) to use
            the faster but less accurate chunked algorithm
        language: force a language code (e.g. ``"en"``); ``None`` auto-detects
    """

    def __init__(
        self,
        model_id="openai/whisper-large-v3",
        device=None,
        torch_dtype=None,
        batch_size=16,
        chunk_length_s=None,
        language=None,
    ):
        SamplesMixin.__init__(self)

        # fom.Model convention: the framework probes model.config
        self.config = None

        from transformers import pipeline

        self.device = device or get_device()
        if torch_dtype is None:
            torch_dtype = (
                torch.float16
                if self.device in ("cuda", "mps")
                else torch.float32
            )

        # runtime parameters (see setters below)
        self._batch_size = batch_size
        self._language = language
        self._chunk_length_s = chunk_length_s

        # loaded once at load_zoo_model time, reused across predict() calls
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            dtype=torch_dtype,
            device=self.device,
        )

    def __exit__(self, *args):
        # Free cached device memory once apply_model finishes with the model.
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    @property
    def media_type(self):
        return "video"

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def chunk_length_s(self):
        return self._chunk_length_s

    @chunk_length_s.setter
    def chunk_length_s(self, value):
        self._chunk_length_s = value

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value

    def predict(self, arg, sample=None):
        # ``arg`` is the FFmpegVideoReader opened by apply_model; unused here.
        # The ASR pipeline reads the media file directly and demuxes its audio.
        generate_kwargs = {"task": "transcribe"}
        if self._language:
            generate_kwargs["language"] = self._language

        pipe_kwargs = {
            "batch_size": self._batch_size,
            "return_timestamps": True,  # segment-level timestamps
            "generate_kwargs": generate_kwargs,
        }
        # None -> Whisper's native sequential long-form (accurate); a value
        # switches to the faster chunked algorithm.
        if self._chunk_length_s is not None:
            pipe_kwargs["chunk_length_s"] = self._chunk_length_s

        result = self._pipe(sample.filepath, **pipe_kwargs)

        detections = []
        for chunk in result.get("chunks", []):
            start, end = chunk.get("timestamp") or (None, None)
            text = (chunk.get("text") or "").strip()
            if not text or start is None:
                continue
            # the final chunk's end is occasionally None; clamp to start
            if end is None:
                end = start
            detections.append(
                fo.TemporalDetection.from_timestamps(
                    [start, end],
                    sample=sample,  # uses sample.metadata.frame_rate
                    label="speech",
                    text=text,
                )
            )

        return {
            "segments": fo.TemporalDetections(detections=detections),
            "transcript": (result.get("text") or "").strip(),
        }
