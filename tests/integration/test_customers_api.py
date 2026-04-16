"""Auxiliary endpoints: `/customers` and `/configs/models`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_get_customers_returns_seeded_list(client, seed_fixtures) -> None:
    response = await client.get("/customers")

    assert response.status_code == 200
    body = response.json()
    ids = {item["id"] for item in body["items"]}
    expected = {c["id"] for c in seed_fixtures["customers"]}
    assert expected.issubset(ids)

    sample = body["items"][0]
    assert set(sample.keys()) >= {"id", "name", "segment"}


async def test_get_customers_does_not_leak_transactions(client, seed_fixtures) -> None:
    """The dropdown endpoint should expose IDs + names only, not transaction history."""
    response = await client.get("/customers")
    body = response.json()

    for item in body["items"]:
        assert "transactions" not in item
        assert "accounts" not in item


async def test_get_configs_models_returns_variants(client) -> None:
    response = await client.get("/configs/models")

    assert response.status_code == 200
    body = response.json()
    assert body["items"], "no model variants registered"

    for variant in body["items"]:
        assert variant["id"]
        assert variant["provider"]
        assert variant["model"]
        assert "cost_per_1k_in_usd" in variant
        assert "cost_per_1k_out_usd" in variant
