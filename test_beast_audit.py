"""Tests pinning the core behavior of beast_audit under ub-mode."""

import numpy as np
import pytest
from beast_audit import (
    BACI_THRESHOLD,
    Site,
    baci_verdict,
    change_sentence,
    grid_size,
    money,
    month_gap,
    render_table,
    signed,
    status_label,
    zone_masks,
)


def test_baci_verdict():
    # Net delta >= threshold -> ISOLATED_IMPACT
    assert baci_verdict(0.05, 0.03) == "ISOLATED_IMPACT"
    assert baci_verdict(0.025, 0.025) == "ISOLATED_IMPACT"

    # Core delta >= threshold, but net delta < threshold -> REGIONAL_CLIMATE_SHIFT
    assert baci_verdict(0.05, 0.01) == "REGIONAL_CLIMATE_SHIFT"
    assert baci_verdict(0.03, -0.01) == "REGIONAL_CLIMATE_SHIFT"

    # Local degradation
    assert baci_verdict(-0.05, -0.03) == "LOCAL_DEGRADATION"

    # Marginal / noise
    assert baci_verdict(0.01, 0.01) == "MARGINAL_OR_NOISE"
    assert baci_verdict(0.0, 0.0) == "MARGINAL_OR_NOISE"


def test_status_label():
    assert status_label("NDVI", "ISOLATED_IMPACT") == "Verified Growth"
    assert status_label("NDWI", "ISOLATED_IMPACT") == "Verified Growth"
    assert status_label("NDBI", "ISOLATED_IMPACT") == "New Construction"

    assert status_label("NDVI", "REGIONAL_CLIMATE_SHIFT") == "Weather Only"
    assert status_label("NDWI", "REGIONAL_CLIMATE_SHIFT") == "Weather Only"

    assert status_label("NDVI", "LOCAL_DEGRADATION") == "Tree Loss"
    assert status_label("NDWI", "LOCAL_DEGRADATION") == "Drier Soil"
    assert status_label("NDBI", "LOCAL_DEGRADATION") == "More Greenery / Less Bare Ground"

    assert status_label("NDVI", "MARGINAL_OR_NOISE") == "No Change"
    assert status_label("NDVI", "NO_DATA") == "No Data"


def test_month_gap():
    assert month_gap(3, 3) == 0
    assert month_gap(1, 12) == 1
    assert month_gap(12, 1) == 1
    assert month_gap(2, 8) == 6
    assert month_gap(1, 7) == 6
    assert month_gap(11, 2) == 3


def test_grid_size():
    assert grid_size("NDVI") == 300
    assert grid_size("NDWI") == 300
    assert grid_size("NDBI") == 150


def test_zone_masks():
    core, control = zone_masks(300)
    assert core.shape == (300, 300)
    assert control.shape == (300, 300)
    # Core and control should never overlap
    assert not np.any(core & control)
    # Core should be centered
    assert core[150, 150]
    # Center is not in control
    assert not control[150, 150]


def test_money_and_signed():
    assert money(1000000) == "1,000,000"
    assert money(None) == "-"
    assert money("bad") == "-"

    assert signed(0.048) == "+0.048"
    assert signed(-0.071) == "-0.071"
    assert signed(None) == "-"
    assert signed(float("nan")) == "-"


def test_render_table():
    headers = ["Name", "Score"]
    rows = [["Alice", "95"], ["Bob", "100"]]
    table = render_table(headers, rows)
    assert "Alice" in table
    assert "Bob" in table
    assert "100" in table


def test_change_sentence():
    site = Site("Test_Site", "Test Place", "Country", "Reforestation", 0.0, 0.0, "NDVI")
    record = {
        "BACI_Verdict": "ISOLATED_IMPACT",
        "Net_Delta": 0.05,
    }
    s = change_sentence(site, record)
    assert "+5.0% more greenery" in s
    assert "Verified Growth" in s
