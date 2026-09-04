# SPDX-License-Identifier: Apache-2.0
"""Pluggable detectors (Build Spec Section 7.3).

Nothing here imports a heavy ML dependency at module load. Import a concrete
detector class directly (``from ragwarden.detectors.nli import NLIDetector``);
its ``__init__`` lazy-imports its extra and raises a clear error if missing.
"""

from ragwarden.detectors.base import Detector, MissingExtraError

__all__ = ["Detector", "MissingExtraError", "get_detector"]

# name -> "module:class" (resolved lazily)
_DETECTORS: dict[str, str] = {
    "nli": "ragwarden.detectors.nli:NLIDetector",
    "hhem": "ragwarden.detectors.hhem:HHEMDetector",
    "lettucedetect": "ragwarden.detectors.lettucedetect:LettuceDetectAdapter",
    "minicheck": "ragwarden.detectors.minicheck:MiniCheckDetector",
    "keyword_stub": "ragwarden.detectors.stub:KeywordStubDetector",
}


def get_detector(name: str, **kwargs: object) -> Detector:
    """Instantiate a registered detector by name (lazy import of its extra)."""
    import importlib

    try:
        target = _DETECTORS[name]
    except KeyError:
        raise ValueError(f"unknown detector {name!r}; known: {sorted(_DETECTORS)}") from None
    module_name, class_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)  # type: ignore[no-any-return]
