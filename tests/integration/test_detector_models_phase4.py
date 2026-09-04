# SPDX-License-Identifier: Apache-2.0
"""Phase 4 real-model checks (opt-in): `pytest -m model`.

Each test skips unless its extra is installed:
    pip install 'ragwarden[hhem]'
    pip install 'ragwarden[lettucedetect]'
"""

from __future__ import annotations

import pytest

from ragwarden.contracts import ClaimStatus
from ragwarden.models import Chunk

pytestmark = pytest.mark.model

EVIDENCE = [Chunk("The Eiffel Tower was completed in 1889 in Paris, France.", 0.9, "d1")]


def test_hhem_detector() -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("HHEM-2.1-Open trust_remote_code is not transformers-5.x compatible yet")
    from ragwarden.detectors.hhem import HHEMDetector

    det = HHEMDetector()
    good = det.check("The Eiffel Tower was completed in 1889.", EVIDENCE)
    bad = det.check("The Eiffel Tower is made of solid gold.", EVIDENCE)
    assert good.status is ClaimStatus.SUPPORTED
    assert bad.status in {ClaimStatus.UNSUPPORTED, ClaimStatus.UNRESOLVED}
    assert 0.0 <= good.confidence <= 1.0


def test_lettucedetect_adapter() -> None:
    pytest.importorskip("lettucedetect")
    from ragwarden.detectors.lettucedetect import LettuceDetectAdapter

    det = LettuceDetectAdapter()
    good = det.check("The Eiffel Tower was completed in 1889.", EVIDENCE)
    bad = det.check("The Eiffel Tower was completed in 1750 by Napoleon.", EVIDENCE)
    assert good.status in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
    assert bad.status in {ClaimStatus.UNSUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED}
