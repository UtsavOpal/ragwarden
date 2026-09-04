# SPDX-License-Identifier: Apache-2.0
"""NLI-based Tier-1 detector (Build Spec Section 7.3, item 2).

Wraps a DeBERTa-v3-class NLI model. Each claim is checked against each evidence
chunk (premise = chunk, hypothesis = claim); the per-claim verdict aggregates:

  * SUPPORTED     if any chunk entails the claim above threshold
  * CONTRADICTED  if the strongest signal is contradiction above threshold
  * UNSUPPORTED   if the model is confident but neutral (no chunk entails)
  * (low-confidence verdicts are left for the orchestrator to escalate)

All (claim, chunk) pairs for a request are tokenised and run through the model in
mini-batches — a single ``gate()`` call is one or two forward passes, not one per
pair — to keep Tier-1 latency within the Spec's sub-500ms-on-CPU budget (G5).

Default checkpoint: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (MIT,
CPU-fast). Set ``model_id`` to the ``-large`` variant for accuracy.
Requires ``pip install 'ragwarden[nli]'``.
"""

from __future__ import annotations

from ragwarden.contracts import ClaimStatus, ClaimVerdict, RetrievedChunk
from ragwarden.detectors.base import BaseDetector, require

__all__ = ["DEFAULT_MODEL_ID", "NLIDetector"]

DEFAULT_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLIDetector(BaseDetector):
    tier = 1
    name = "nli"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        max_evidence_chunks: int = 8,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
    ) -> None:
        torch = require("torch", extra="nli")
        transformers = require("transformers", extra="nli")

        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_evidence_chunks = max_evidence_chunks
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model.to(device)

        # Resolve label positions from the model config rather than assuming order.
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._idx = {v: k for k, v in id2label.items()}
        for required in ("entailment", "neutral", "contradiction"):
            if required not in self._idx:
                raise ValueError(
                    f"model {model_id!r} is not a 3-way NLI model "
                    f"(labels: {sorted(id2label.values())})"
                )

    def _probs_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        """Return (p_entail, p_neutral, p_contradiction) for each (premise, hypothesis)."""
        torch = self._torch
        ent, neu, con = self._idx["entailment"], self._idx["neutral"], self._idx["contradiction"]
        out: list[tuple[float, float, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            enc = self._tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).tolist()
            out.extend((row[ent], row[neu], row[con]) for row in probs)
        return out

    def _top_chunks(self, evidence: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ranked = sorted(evidence, key=lambda c: getattr(c, "score", 0.0), reverse=True)
        return ranked[: self.max_evidence_chunks]

    def check(self, claim: str, evidence: list[RetrievedChunk]) -> ClaimVerdict:
        return self.check_batch([claim], evidence)[0]

    def check_batch(self, claims: list[str], evidence: list[RetrievedChunk]) -> list[ClaimVerdict]:
        if not claims:
            return []
        chunks = self._top_chunks(evidence)
        if not chunks:
            return [
                ClaimVerdict(c, ClaimStatus.UNSUPPORTED, 0.0, 1, explanation="no evidence provided")
                for c in claims
            ]

        pairs = [(chunk.text, claim) for claim in claims for chunk in chunks]
        probs = self._probs_batch(pairs)

        verdicts: list[ClaimVerdict] = []
        width = len(chunks)
        for ci, claim in enumerate(claims):
            best_entail = (0.0, "")
            best_contra = (0.0, "")
            for j, chunk in enumerate(chunks):
                p_ent, _p_neu, p_con = probs[ci * width + j]
                if p_ent > best_entail[0]:
                    best_entail = (p_ent, chunk.source_id)
                if p_con > best_contra[0]:
                    best_contra = (p_con, chunk.source_id)
            verdicts.append(self._verdict(claim, best_entail, best_contra))
        return verdicts

    def _verdict(
        self, claim: str, best_entail: tuple[float, str], best_contra: tuple[float, str]
    ) -> ClaimVerdict:
        p_ent, ent_src = best_entail
        p_con, con_src = best_contra
        if p_con >= self.contradiction_threshold and p_con > p_ent:
            return ClaimVerdict(
                claim,
                ClaimStatus.CONTRADICTED,
                round(p_con, 4),
                1,
                supporting_evidence=[con_src] if con_src else [],
                explanation=f"NLI contradiction p={p_con:.2f}",
            )
        if p_ent >= self.entailment_threshold:
            return ClaimVerdict(
                claim,
                ClaimStatus.SUPPORTED,
                round(p_ent, 4),
                1,
                supporting_evidence=[ent_src] if ent_src else [],
                explanation=f"NLI entailment p={p_ent:.2f}",
            )
        conf = max(p_ent, p_con)
        return ClaimVerdict(
            claim,
            ClaimStatus.UNSUPPORTED,
            round(conf, 4),
            1,
            explanation=f"NLI neutral (entail p={p_ent:.2f}, contra p={p_con:.2f})",
        )
