"""
FiftyOne remote zoo source entry points for WhisperX.

Exposes ``download_model`` and ``load_model`` as required by FiftyOne's zoo
machinery. See ``zoo.py`` for the model implementation.
"""

from .zoo import WhisperXModel


def _model_size_from_name(model_name):
    """Maps a manifest ``base_name`` to a WhisperX model size.

    ``"whisperx-large-v3"`` -> ``"large-v3"``. Falls back to ``"large-v3"`` if
    the name does not carry the ``whisperx-`` prefix.
    """
    if model_name and model_name.startswith("whisperx-"):
        return model_name[len("whisperx-"):]
    return "large-v3"


def download_model(model_name, model_path):
    """No-op: WhisperX pulls its own weights from Hugging Face on first load.

    Idempotent by construction — there is nothing to fetch or verify here, so
    it is always safe to call.
    """


def load_model(model_name=None, model_path=None, **kwargs):
    """Loads a :class:`WhisperXModel`.

    The model size is derived from ``model_name`` (the manifest ``base_name``)
    unless an explicit ``model_size`` is passed via ``kwargs``. All other
    ``kwargs`` (``compute_type``, ``device``, ``batch_size``, ``diarize``,
    ``language``) flow straight through from ``load_zoo_model``.
    """
    kwargs.setdefault("model_size", _model_size_from_name(model_name))
    return WhisperXModel(**kwargs)


def resolve_input(model_name, ctx):
    """Defines the App "Apply Model" operator form for WhisperX.

    Model size is chosen by which ``base_name`` the user selects (``-large-v3``
    / ``-turbo`` / ``-large-v2``), so it is intentionally not a form field. We
    only collect the per-run options here.
    """
    from fiftyone.operators import types

    inputs = types.Object()

    inputs.view(
        "metadata_notice",
        types.Notice(
            label=(
                "Run dataset.compute_metadata() before applying this model so "
                "segment timestamps can be mapped to frame supports."
            )
        ),
    )
    inputs.bool(
        "diarize",
        default=False,
        label="Diarize (speaker labels)",
        description=(
            "Assign a speaker to each segment. Requires the HF_TOKEN "
            "environment variable and acceptance of the pyannote model license."
        ),
        view=types.CheckboxView(),
    )
    inputs.int(
        "batch_size",
        default=16,
        label="Batch size",
        description="Transcription batch size.",
    )
    inputs.str(
        "language",
        default=None,
        required=False,
        label="Language",
        description=(
            "Force a language code (e.g. 'en'). Leave empty to auto-detect "
            "the language of each sample."
        ),
    )

    return types.Property(inputs)


def parse_parameters(model_name, ctx, params):
    """Formats the App operator inputs before they reach :func:`load_model`.

    An empty ``language`` field means "auto-detect", which WhisperX expresses
    as ``None`` rather than an empty string.
    """
    language = params.get("language")
    if isinstance(language, str) and not language.strip():
        params["language"] = None
