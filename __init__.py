"""
FiftyOne remote zoo source entry points for Whisper (Transformers).

Exposes ``download_model`` and ``load_model`` as required by FiftyOne's zoo
machinery. See ``zoo.py`` for the model implementation.
"""

from .zoo import WhisperModel


def _model_id_from_name(model_name):
    """Maps a manifest ``base_name`` to a Hugging Face model id.

    ``"whisper-large-v3"`` -> ``"openai/whisper-large-v3"``.
    """
    if not model_name or not model_name.startswith("whisper"):
        raise ValueError(
            "Unrecognized model name '%s'; pass an explicit `model_id` "
            "(e.g. 'openai/whisper-large-v3') instead" % model_name
        )

    return "openai/" + model_name


def download_model(model_name, model_path):
    """No-op: Transformers pulls weights from the Hugging Face cache on first
    load.

    Idempotent by construction — there is nothing to fetch or verify here, so
    it is always safe to call.
    """


def load_model(model_name=None, model_path=None, **kwargs):
    """Loads a :class:`WhisperModel`.

    The HF model id is derived from ``model_name`` (the manifest ``base_name``)
    unless an explicit ``model_id`` is passed via ``kwargs``. All other
    ``kwargs`` (``device``, ``torch_dtype``, ``batch_size``, ``chunk_length_s``,
    ``language``) flow straight through from ``load_zoo_model``.
    """
    if "model_id" not in kwargs:
        kwargs["model_id"] = _model_id_from_name(model_name)

    return WhisperModel(**kwargs)


def resolve_input(model_name, ctx):
    """Defines the App "Apply Model" operator form for Whisper.

    The model is chosen from the list (``whisper-large-v3`` / ``-turbo`` /
    ``-large-v2``), so only the per-run options are collected here.
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
    inputs.int(
        "batch_size",
        default=16,
        label="Batch size",
        description="Number of audio chunks decoded per batch.",
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

    An empty ``language`` field means "auto-detect", which Whisper expresses as
    ``None`` rather than an empty string.
    """
    language = params.get("language")
    if isinstance(language, str) and not language.strip():
        params["language"] = None
