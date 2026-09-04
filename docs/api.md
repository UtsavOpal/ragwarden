# API reference

The public surface is deliberately small. Everything under `ragwarden.cascade`,
`ragwarden.detectors`, `ragwarden.adapters`, `ragwarden.benchmark` and `ragwarden.observability` is
implementation detail and may change without a major version bump until v1.0.

## `ragwarden`

::: ragwarden.gate.gate

::: ragwarden.gate.GateRequest

## Contracts

The frozen core interfaces (Build Spec Section 6). Host objects satisfy them structurally — nothing
needs to subclass a RagWarden type.

::: ragwarden.contracts

## Concrete models

::: ragwarden.models

## Policy

::: ragwarden.policy.Policy

::: ragwarden.policy.decide

::: ragwarden.policy.register_trigger

## Scoring

::: ragwarden.scoring.SeverityWeightedScoring

## Decomposition

::: ragwarden.decomposition.ClaimDecomposer

::: ragwarden.decomposition.SentenceSplitDecomposer
