# Security Policy

## Supported versions

RagWarden is pre-1.0. Security fixes are applied to the latest release only until v1.0 ships.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub's
[private security advisory](https://github.com/utsavopal/ragwarden/security/advisories/new) form,
or by email to the maintainers. Do not open a public issue for security problems.

We aim to acknowledge reports within 5 business days and to ship a fix or mitigation within 90 days,
coordinating disclosure with the reporter.

## Scope notes

- RagWarden loads third-party models from Hugging Face when optional extras are installed. Model
  weights are outside this project's control; review the model card and license before deploying.
- The Tier 2 / Tier 3 tiers call host-supplied callables (your LLM). RagWarden does not transmit
  data anywhere itself.
- Supply-chain: releases are published via PyPI Trusted Publishing (OIDC) with attestations, and an
  SBOM is attached to each release (from Phase 11).
