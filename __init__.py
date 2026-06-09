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
