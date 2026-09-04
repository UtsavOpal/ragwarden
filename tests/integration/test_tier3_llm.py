# SPDX-License-Identifier: Apache-2.0
"""Phase 6 real-LLM check (opt-in): `pytest -m model`.

Skips unless a local Ollama server is reachable with a small instruct model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from ragwarden import Policy, gate
from ragwarden.contracts import ClaimStatus, ClaimVerdict, GateAction
from ragwarden.detectors.stub import ScriptedStubDetector
from ragwarden.models import Answer, Chunk, Context

pytestmark = pytest.mark.model

_OLLAMA = "http://localhost:11434"
_MODEL = "llama3.2:3b"


def _ollama_judge(prompt: str) -> str:
    body = json.dumps(
        {"model": _MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        f"{_OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["response"]


@pytest.fixture(scope="module")
def judge():
    try:
        urllib.request.urlopen(f"{_OLLAMA}/api/tags", timeout=3)
    except (urllib.error.URLError, OSError):
        pytest.skip("no local Ollama server")
    return _ollama_judge


CTX = Context(
    query="When was the Eiffel Tower completed?",
    chunks=[
        Chunk("The Eiffel Tower was completed in 1889 for the Exposition Universelle.", 0.9, "d1")
    ],
    retrieval_method="hybrid",
)
# Force every claim past Tier 1/2 into the judge.
_FORCE_TIER3 = Policy.default()
_FORCE_TIER3.tier2.enabled = False
_FORCE_TIER3.tier1.confidence_threshold = 0.99
# A local Ollama call can take several seconds; the library default budget
# (max_latency_ms=800, call_timeout_s=15) is tuned for fast hosted judges, not
# this test's slow local model — widen it so the real call gets to finish.
_FORCE_TIER3.tier3.max_latency_ms = 60_000
_FORCE_TIER3.tier3.call_timeout_s = 60.0
_STUB = ScriptedStubDetector(lambda c, e: ClaimVerdict(c, ClaimStatus.UNSUPPORTED, 0.3, 1))


def test_judge_supports_a_grounded_claim(judge) -> None:
    r = gate(
        CTX,
        Answer("The Eiffel Tower was completed in 1889."),
        policy=_FORCE_TIER3,
        detectors=[_STUB],
        judge_fn=judge,
    )
    assert r.claim_verdicts[0].resolved_at_tier == 3
    assert r.claim_verdicts[0].status is ClaimStatus.SUPPORTED
    assert r.action is GateAction.ALLOW


def test_judge_flags_a_fabricated_claim(judge) -> None:
    r = gate(
        CTX,
        Answer("The Eiffel Tower was completed in 1925 by Gustave Courbet."),
        policy=_FORCE_TIER3,
        detectors=[_STUB],
        judge_fn=judge,
    )
    assert r.claim_verdicts[0].status in {ClaimStatus.CONTRADICTED, ClaimStatus.UNSUPPORTED}
    assert r.action is not GateAction.ALLOW
