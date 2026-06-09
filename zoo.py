"""
WhisperX transcription as a FiftyOne remote zoo model.

Maps spoken-word ASR onto the zoo Model contract:
  - media_type = "video"            -> FiftyOne treats samples as videos and
                                       routes through ``_apply_video_model``
                                       (a per-sample loop, NOT a DataLoader),
                                       so there is no worker-pickle surface.
  - SamplesMixin                    -> predict() receives the sample so we can
                                       read ``sample.filepath``; WhisperX demuxes
                                       audio with ffmpeg itself and wants the
                                       file, not decoded frames.
  - predict() returns a dict of two sample-level outputs, written in a single
    apply_model pass:
        "segments"   -> fo.TemporalDetections (timestamped, optionally
                        speaker-labeled segments; scrub/filter in the App)
        "transcript" -> flat transcript string (text search, embeddings, VLM
                        context)

FiftyOne's ``add_labels`` maps these dict keys to fields via ``label_field``:
a dict names them exactly, a string prefixes them, and ``None`` uses the keys
verbatim. Recommended::

    dataset.apply_model(
        model,
        label_field={"segments": "transcript_segments", "transcript": "transcript"},
    )

Precondition: run ``dataset.compute_metadata()`` before ``apply_model`` so that
``TemporalDetection.from_timestamps()`` can convert seconds -> frame support.
"""

import os

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


class WhisperXModel(SamplesMixin, Model):
    """WhisperX speech-to-text wrapped as a FiftyOne video zoo model.

    Construction-only parameters (changing them reloads weights):
        model_size: WhisperX/faster-whisper model size, e.g. ``"large-v3"``
        compute_type: ctranslate2 compute type, e.g. ``"float16"``,
            ``"int8_float16"``, ``"int8"``
        device: ``"cuda"``, ``"mps"``, or ``"cpu"``. ``None`` (default)
            auto-selects via :func:`get_device`. Note that WhisperX's ASR
            backend (CTranslate2) has no MPS support, so ``"mps"`` runs the
            transcription on CPU while alignment/diarization use MPS.
        diarize: whether to build the pyannote diarization pipeline (needs
            ``HF_TOKEN`` and acceptance of the pyannote model license)

    Runtime parameters (per-call, exposed as setters so callers never have to
    reconstruct the model):
        batch_size: transcription batch size
        language: force a language code (e.g. ``"en"``); ``None`` auto-detects
    """

    def __init__(
        self,
        model_size="large-v3",
        compute_type="float16",
        device=None,
        batch_size=16,
        diarize=False,
        language=None,
    ):
        SamplesMixin.__init__(self)

        import whisperx  # lazy: provided via manifest requirements

        self._whisperx = whisperx
        self.device = device or get_device()

        # runtime parameters (see setters below)
        self._batch_size = batch_size
        self._language = language

        # CTranslate2 (faster-whisper) has no MPS backend, so the ASR runs on
        # CPU when MPS is selected; alignment/diarization still use self.device.
        asr_device = "cpu" if self.device == "mps" else self.device

        # loaded once at load_zoo_model time, reused across predict() calls
        self._asr = whisperx.load_model(
            model_size, asr_device, compute_type=compute_type
        )
        self._align_cache = {}  # language_code -> (align_model, align_meta)
        self._diarizer = self._build_diarizer() if diarize else None

    def _build_diarizer(self):
        # The diarization entry point moved between whisperx releases; try the
        # current location first, then the legacy top-level alias.
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            DiarizationPipeline = self._whisperx.DiarizationPipeline

        return DiarizationPipeline(
            use_auth_token=os.environ.get("HF_TOKEN"), device=self.device
        )

    def __exit__(self, *args):
        # Free cached device memory once apply_model finishes with the model.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

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
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value

    def _align_model(self, language_code):
        if language_code not in self._align_cache:
            self._align_cache[language_code] = self._whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self._align_cache[language_code]

    def predict(self, arg, sample=None):
        # ``arg`` is the FFmpegVideoReader opened by apply_model; unused here.
        # WhisperX reads the file directly and demuxes its own audio.
        audio = self._whisperx.load_audio(sample.filepath)

        result = self._asr.transcribe(
            audio, batch_size=self._batch_size, language=self._language
        )
        lang = result["language"]

        align_model, align_meta = self._align_model(lang)
        result = self._whisperx.align(
            result["segments"],
            align_model,
            align_meta,
            audio,
            self.device,
            return_char_alignments=False,
        )

        if self._diarizer is not None:
            diarize_df = self._diarizer(audio)
            result = self._whisperx.assign_word_speakers(diarize_df, result)

        detections = []
        texts = []
        for s in result["segments"]:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            detections.append(
                fo.TemporalDetection.from_timestamps(
                    [s["start"], s["end"]],
                    sample=sample,  # uses sample.metadata.frame_rate
                    label="speech",
                    text=text,
                    speaker=s.get("speaker", "UNKNOWN"),
                )
            )

        return {
            "segments": fo.TemporalDetections(detections=detections),
            "transcript": " ".join(texts),
        }
