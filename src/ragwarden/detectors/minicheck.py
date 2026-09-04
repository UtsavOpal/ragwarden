# SPDX-License-Identifier: Apache-2.0
"""MiniCheck Tier-1 detector (Build Spec Section 7.3, item 4).

MiniCheck (Tang et al., EMNLP 2024) is a sentence-level fact-checker: given a
document and a sentence, it predicts whether the sentence is supported.

**Packaging caveat (verified 2026-09):** there is no usable PyPI distribution.
The PyPI name ``minicheck`` is an unrelated model checker; ``bespokelabs`` is now
a hosted-API client. Install the real library from source::

    pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    pip install "ragwarden[minicheck]"   # transformers + torch

Some MiniCheck checkpoints (the 7B Bespoke variant) are Llama-derived and *not*
OSI-licensed — never a default dependency. The Flan-T5 / RoBERTa / DeBERTa
checkpoints are permissively licensed; prefer those.
"""

from __future__ import annotations

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.detectors.base import BaseDetector, require

__all__ = ["MiniCheckDetector"]

DEFAULT_MODEL_NAME = "flan-t5-large"  # permissive; avoid the Llama-derived 7B by default


class MiniCheckDetector(BaseDetector):
    tier = 1
    name = "minicheck"

    def __init__(
        self, model_name: str = DEFAULT_MODEL_NAME, *, cache_dir: str | None = None
    ) -> None:
        require("torch", extra="minicheck")
        mc = require("minicheck.minicheck", extra="minicheck", package="minicheck (from git)")
        self.model_name = model_name
        self._scorer = mc.MiniCheck(model_name=model_name, cache_dir=cache_dir)

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        if not evidence:
            return ClaimVerdict(
                claim, ClaimStatus.UNSUPPORTED, 0.0, 1, explanation="no evidence provided"
            )
        docs = [c.text for c in evidence]
        claims = [claim] * len(docs)
        _preds, raw_probs, _, _ = self._scorer.score(docs=docs, claims=claims)
        probs = [float(p) for p in raw_probs]
        best = max(range(len(probs)), key=lambda i: probs[i])
        p = probs[best]
        if p >= 0.5:
            return ClaimVerdict(
                claim,
                ClaimStatus.SUPPORTED,
                round(p, 4),
                1,
                supporting_evidence=[evidence[best].source_id] if evidence[best].source_id else [],
                explanation=f"MiniCheck support probability {p:.2f}",
            )
        return ClaimVerdict(
            claim,
            ClaimStatus.UNSUPPORTED,
            round(1.0 - p, 4),
            1,
            explanation=f"MiniCheck support probability {p:.2f} (below 0.5)",
        )
