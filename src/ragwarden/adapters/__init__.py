# SPDX-License-Identifier: Apache-2.0
"""Adapters: translate a host application's native objects into the
:mod:`ragwarden.contracts` shapes (Build Spec Section 13).

Adapters live behind extras and must not be required for core usage. Import the
one you need directly, e.g.
``from ragwarden.adapters.opensearch import from_opensearch_hybrid_response``.
"""
