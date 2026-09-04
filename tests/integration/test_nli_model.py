# SPDX-License-Identifier: Apache-2.0
"""Phase 2 real-model check (opt-in): run `pytest -m model`.

Downloads the default DeBERTa-v3-base NLI checkpoint (~370 MB) on first run.
"""

from __future__ import annotations

import pytest

from ragwarden.contracts import ClaimStatus
from ragwarden.models import Chunk

pytestmark = pytest.mark.model

EVIDENCE = [
    Chunk("The Eiffel Tower was completed in 1889 for the 1889 World's Fair in Paris.", 0.9, "d1"),
]


@pytest.fixture(scope="module")
def detector():
    nli = pytest.importorskip("ragwarden.detectors.nli")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return nli.NLIDetector()


def test_supported_claim(detector) -> None:
    v = detector.check("The Eiffel Tower was finished in 1889.", EVIDENCE)
    assert v.status is ClaimStatus.SUPPORTED
    assert v.confidence > 0.6
    assert v.supporting_evidence == ["d1"]


def test_contradicted_claim(detector) -> None:
    v = detector.check("The Eiffel Tower was completed in 1950.", EVIDENCE)
    assert v.status is ClaimStatus.CONTRADICTED


def test_unsupported_claim(detector) -> None:
    v = detector.check("The Eiffel Tower is painted bright green every winter.", EVIDENCE)
    assert v.status in {ClaimStatus.UNSUPPORTED, ClaimStatus.UNRESOLVED}
