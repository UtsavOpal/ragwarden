# SPDX-License-Identifier: Apache-2.0
"""Phase 0 acceptance: contracts import and satisfy structural typing."""

from __future__ import annotations

from ragwarden.contracts import (
    ClaimStatus,
    ClaimVerdict,
    GateAction,
    GateResult,
    Generation,
    RetrievalContext,
    RetrievedChunk,
    StageTrace,
)
from ragwarden.models import Answer, Chunk, Context


def test_top_level_imports() -> None:
    import ragwarden

    assert ragwarden.__version__
    assert callable(ragwarden.gate)
    assert hasattr(ragwarden, "Policy")


def test_concrete_models_satisfy_protocols() -> None:
    chunk = Chunk(text="x", score=1.0, source_id="s", metadata={})
    ctx = Context(query="q", chunks=[chunk], retrieval_method="hybrid")
    ans = Answer(text="a")
    assert isinstance(chunk, RetrievedChunk)
    assert isinstance(ctx, RetrievalContext)
    assert isinstance(ans, Generation)


def test_core_import_is_light() -> None:
    # Must run in a fresh interpreter: other tests in this session may have
    # imported torch/transformers via the [nli] extra.
    import subprocess
    import sys

    code = (
        "import sys, ragwarden; from ragwarden import gate, Policy; "
        "leaked = {'torch','transformers','numpy','pandas','scipy'} & set(sys.modules); "
        "raise SystemExit(f'leaked {sorted(leaked)}' if leaked else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_enums_are_str() -> None:
    assert GateAction.ALLOW == "allow"
    assert ClaimStatus.SUPPORTED == "supported"


def test_dataclasses_construct() -> None:
    v = ClaimVerdict(
        claim_text="c", status=ClaimStatus.SUPPORTED, confidence=0.9, resolved_at_tier=1
    )
    r = GateResult(
        action=GateAction.ALLOW,
        reliability_score=1.0,
        claim_verdicts=[v],
        output_text="c",
        stage_trace=[StageTrace("tier0", 0.1, 0)],
        explanation="ok",
    )
    assert r.claim_verdicts[0].resolved_at_tier == 1
