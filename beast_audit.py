#!/usr/bin/env python3
"""Public-interest audit of philanthropic claims using IRS Form 990 data and Sentinel-2 L2A.

Methodology:
- Module 1 (Financials): Ingests IRS Form 990 filings via ProPublica Nonprofit Explorer API.
- Module 2 (BACI Satellite Audit): Computes Before-After-Control-Impact (BACI) spectral deltas
  (NDVI, NDWI, NDBI) on Copernicus Sentinel-2 Level-2A imagery via Element 84 STAC API.
- Module 3 (Artifacts): Generates publication-ready comparative charts, maps, and visual cards.
"""

from __future__ import annotations

import logging
import os
import textwrap
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

matplotlib.use("Agg")
logger = logging.getLogger("beast_audit")

# --------------------------------------------------------------------------- #
# Domain Constants & Configuration
# --------------------------------------------------------------------------- #

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
PROPUBLICA = "https://projects.propublica.org/nonprofits/api/v2"
USER_AGENT = "beast-audit/1.0 (public-interest research)"

HALF_SIZE_M = 1500
WINDOW_M = HALF_SIZE_M * 2
GRID_10M = WINDOW_M // 10
GRID_20M = WINDOW_M // 20
CORE_RADIUS_M = 400
CTRL_INNER_M = 600

# Water masking leaves less usable land in coastal windows; 25% is sufficient for a stable mean.
MIN_VALID_FRACTION = 0.25
BACI_THRESHOLD = 0.025
DELTA_LIMIT = 0.35

HERE = Path(__file__).resolve().parent
VISUALS_DIR = HERE / "visuals"

BASELINE_START = "2019-01-01T00:00:00Z"
BASELINE_END = "2020-12-31T23:59:59Z"
CURRENT_START = "2025-01-01T00:00:00Z"
CURRENT_END = "2026-10-29T23:59:59Z"

# SCL (Scene Classification Layer) class 6 is water. Masking removes rivers and lakes from land rings.
BAD_SCL = {0, 1, 3, 6, 8, 9, 10, 11}

BACI_COLORS = {
    "ISOLATED_IMPACT": "#1a7f37",
    "REGIONAL_CLIMATE_SHIFT": "#0969da",
    "MARGINAL_OR_NOISE": "#6e7781",
    "LOCAL_DEGRADATION": "#cf222e",
    "NO_DATA": "#8c959f",
}

DELTA_CMAPS = {
    "NDVI": "RdYlGn",
    "NDWI": "BrBG",
    "NDBI": "RdBu_r",
}

CHANGE_KEYS = {
    "NDVI": "Green = More Trees/Foliage | Red = Tree Loss",
    "NDWI": "Green/Blue = Wetter Soil | Brown = Drier Soil",
    "NDBI": "Red/Orange = New Buildings/Cleared Land | Blue = More Greenery",
}

STATUS_COLORS = {
    "Verified Growth": "#1a7f37",
    "New Construction": "#bc4c00",
    "Weather Only": "#0969da",
    "No Change": "#6e7781",
    "Tree Loss": "#cf222e",
    "Drier Soil": "#9a6700",
    "More Greenery / Less Bare Ground": "#0f766e",
    "No Data": "#6e7781",
}

WHITE = "#ffffff"
INK = "#111111"
MUTED = "#555555"
RULE = "#d0d7de"

EXTENT = (-HALF_SIZE_M, HALF_SIZE_M, -HALF_SIZE_M, HALF_SIZE_M)

ORGANIZATIONS = [
    ("Beast Philanthropy", "852067214"),
    ("National Arbor Day Foundation", "237169265"),
    ("Ocean Conservancy", "237245152"),
    ("The Ocean Cleanup (US)", "815132355"),
]


# --------------------------------------------------------------------------- #
# Domain Models (Type-system discipline)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Site:
    """Target site for satellite and geographic verification."""

    site_id: str
    place: str
    country: str
    category: str
    lat: float
    lon: float
    metric: str
    months: tuple[int, ...] | None = None
    min_core_land: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.place}, {self.country}"


SITES = [
    Site("Tyndall_AFB_FL_US", "Tyndall AFB, Florida", "USA", "Reforestation",
         30.0460, -85.4750, "NDVI", min_core_land=0.95),
    Site("Elephant_Hill_BC_CA", "Elephant Hill, British Columbia", "Canada", "Reforestation",
         50.9500, -121.3500, "NDVI"),
    Site("Minqin_Gansu_CN", "Minqin, Gansu", "China", "Reforestation", 38.6650, 102.9850, "NDVI"),
    Site("Kijabe_Forest_KE", "Kijabe Forest", "Kenya", "Reforestation", -0.9250, 36.6050, "NDVI"),
    Site("Narok_Wells_Cluster_KE", "Narok wells", "Kenya", "Water & Irrigation",
         -1.1450, 35.7450, "NDWI", (1, 2, 7, 8, 9), min_core_land=0.95),
    Site("Mityana_Wells_Cluster_UG", "Mityana wells", "Uganda", "Water & Irrigation",
         0.4450, 31.9850, "NDWI", (1, 2, 6, 7, 8), min_core_land=0.95),
    Site("Binga_Water_Project_ZW", "Binga water project", "Zimbabwe", "Water & Irrigation",
         -17.6550, 27.3850, "NDWI", (5, 6, 7, 8, 9, 10), min_core_land=0.80),
    Site("Chirundu_Bridge_And_Village_ZW", "Chirundu bridge and village", "Zimbabwe",
         "Infrastructure", -16.0550, 28.8650, "NDBI", min_core_land=0.95),
    Site("Ashanti_Cocoa_Village_Zone_GH", "Ashanti cocoa village", "Ghana",
         "Infrastructure", 6.8150, -1.4650, "NDBI"),
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


@dataclass
class SceneLayers:
    """Spectral layer and zonal statistics for a single Sentinel-2 acquisition."""

    index: np.ndarray
    valid: np.ndarray
    core: float
    control: float
    item: dict[str, Any]
    core_valid_fraction: float
    core_water_fraction: float


@dataclass
class SiteResult:
    """Combined audit outcome comparing baseline and current scene layers."""

    record: dict[str, Any]
    before: SceneLayers | None = None
    after: SceneLayers | None = None
    delta: np.ndarray | None = None
    rgb_before: np.ndarray | None = None
    rgb_after: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# Pure Helper Functions (Formatting & Math)
# --------------------------------------------------------------------------- #

def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format data into a clean, text-based aligned ASCII table."""
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "+".join("-" * (w + 2) for w in widths)
    out = [line, "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers))), line]
    for row in rows:
        out.append("|".join(f" {row[i]:<{widths[i]}} " for i in range(len(headers))))
    out.append(line)
    return "\n".join(out)


def money(value: Any) -> str:
    """Format numeric values as comma-separated integers or '-' if invalid."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def signed(value: Any) -> str:
    """Format float values with explicit sign (+/-) to three decimal places."""
    return "-" if pd.isna(value) else f"{float(value):+.3f}"


def grid_size(metric: str) -> int:
    """Return pixel grid resolution for a given metric (10m vs 20m)."""
    return GRID_20M if metric == "NDBI" else GRID_10M


def month_gap(a: int, b: int) -> int:
    """Compute shortest circular calendar month difference (0 to 6)."""
    diff = abs(a - b)
    return min(diff, 12 - diff)


# --------------------------------------------------------------------------- #
# Module 1: Financial Forensics (IRS Form 990)
# --------------------------------------------------------------------------- #

def fetch_org(ein: str) -> dict[str, Any] | None:
    """Fetch organization Form 990 data from ProPublica Nonprofit Explorer API."""
    try:
        resp = SESSION.get(f"{PROPUBLICA}/organizations/{ein}.json", timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException as err:
        logger.warning("Failed to fetch EIN %s: %s", ein, err)
    return None


def _parse_org_filings(
    name: str, ein: str, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract up to 4 recent Form 990 filing records and organizational notes."""
    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    org = payload.get("organization", {})
    legal = org.get("name")
    if legal and legal.lower() not in name.lower():
        notes.append(f"{name}: IRS legal name is '{legal}' (EIN {ein})")

    filings = sorted(
        payload.get("filings_with_data") or [],
        key=lambda f: f.get("tax_prd_yr") or 0,
        reverse=True,
    )[:4]

    for filing in filings:
        revenue = filing.get("totrevenue")
        expenses = filing.get("totfuncexpns")
        ratio = (expenses / revenue * 100) if revenue else None
        rows.append({
            "Organization": name,
            "EIN": ein,
            "Tax_Year": filing.get("tax_prd_yr"),
            "Total_Revenue_USD": revenue,
            "Total_Expenses_USD": expenses,
            "Executive_Comp_USD": filing.get("compnsatncurrofcr"),
            "Net_Assets_USD": filing.get("totnetassetend"),
            "Expense_to_Revenue_Ratio_%": round(ratio, 1) if ratio is not None else None,
            "Legal_Name": legal,
        })

    for unextracted in payload.get("filings_without_data") or []:
        year = unextracted.get("tax_prd_yr") or 0
        if year >= 2023:
            notes.append(f"{name}: FY{year} filing exists but financial data is not extracted")

    return rows, notes


def financial_rows() -> tuple[list[dict[str, Any]], list[str]]:
    """Iterate through monitored organizations and aggregate financial records."""
    all_rows: list[dict[str, Any]] = []
    all_notes: list[str] = []

    for name, ein in ORGANIZATIONS:
        payload = fetch_org(ein)
        time.sleep(0.4)
        if not payload or not payload.get("organization"):
            all_notes.append(f"{name}: no record for EIN {ein}")
            continue

        rows, notes = _parse_org_filings(name, ein, payload)
        all_rows.extend(rows)
        all_notes.extend(notes)

    return all_rows, all_notes


# --------------------------------------------------------------------------- #
# Module 2: BACI Sentinel-2 Remote Sensing Engine
# --------------------------------------------------------------------------- #

def site_bbox(lat: float, lon: float) -> list[float]:
    """Calculate 3 km by 3 km UTM bounding box projected to WGS84 coordinates."""
    zone = int((lon + 180) / 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    x, y = fwd.transform(lon, lat)
    west, south = inv.transform(x - HALF_SIZE_M, y - HALF_SIZE_M)
    east, north = inv.transform(x + HALF_SIZE_M, y + HALF_SIZE_M)
    return [west, south, east, north]


def search_scenes(site: Site, start: str, end: str, max_cloud: int = 25) -> list[dict[str, Any]]:
    """Query Element 84 STAC API for Sentinel-2 Level-2A surface reflectance items."""
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": site_bbox(site.lat, site.lon),
        "datetime": f"{start}/{end}",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    resp = SESSION.post(STAC_URL, json=body, timeout=60)
    resp.raise_for_status()
    items = resp.json().get("features", [])
    if site.months:
        items = [i for i in items if int(i["properties"]["datetime"][5:7]) in site.months]
    return items


def read_band(
    href: str, bounds: tuple[float, float, float, float],
    shape: tuple[int, int], resampling: Resampling,
) -> np.ndarray:
    """Read and resample a single raster window using rasterio."""
    with rasterio.open(href) as ds:
        window = from_bounds(*bounds, transform=ds.transform)
        return ds.read(
            1, window=window, out_shape=shape,
            resampling=resampling, boundless=True, fill_value=0,
        )


def scene_bounds(item: dict[str, Any], lat: float, lon: float) -> tuple[Any, tuple[float, ...]]:
    """Get UTM scene projection and target bounding window."""
    href = item["assets"]["red"]["href"]
    with rasterio.open(href) as ds:
        crs = ds.crs
    fwd = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = fwd.transform(lon, lat)
    return crs, (x - HALF_SIZE_M, y - HALF_SIZE_M, x + HALF_SIZE_M, y + HALF_SIZE_M)


@lru_cache(maxsize=4)
def zone_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute 800m core square and 1.2km-3km surrounding control ring masks."""
    pixel_m = WINDOW_M / size
    axis = (np.arange(size, dtype=np.float32) + 0.5) * pixel_m - HALF_SIZE_M
    distance = np.maximum(np.abs(axis)[:, None], np.abs(axis)[None, :])
    core = distance <= CORE_RADIUS_M
    control = (distance >= CTRL_INNER_M) & (distance <= HALF_SIZE_M)
    return core, control


def zone_mean(index: np.ndarray, mask: np.ndarray) -> float | None:
    """Compute finite pixel mean across masked region."""
    values = index[mask]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def load_layers(item: dict[str, Any], site: Site) -> SceneLayers | None:
    """Load spectral bands, apply quality masks, and compute zonal index means."""
    assets = item["assets"]
    bands_map = {
        "NDVI": ("nir", "red"),
        "NDWI": ("green", "nir"),
        "NDBI": ("swir16", "nir"),
    }
    band_names = bands_map[site.metric]
    if any(key not in assets for key in band_names) or "scl" not in assets:
        return None

    _, bounds = scene_bounds(item, site.lat, site.lon)
    size = grid_size(site.metric)
    a = read_band(assets[band_names[0]]["href"], bounds, (size, size), Resampling.average)
    b = read_band(assets[band_names[1]]["href"], bounds, (size, size), Resampling.average)
    scl = read_band(assets["scl"]["href"], bounds, (size, size), Resampling.nearest)

    core_mask, control_mask = zone_masks(size)
    water = scl == 6
    valid = (a > 0) & (b > 0) & ~np.isin(scl, list(BAD_SCL))

    if valid[core_mask].mean() < MIN_VALID_FRACTION:
        return None
    if valid[control_mask].mean() < MIN_VALID_FRACTION:
        return None

    a_f = a.astype("float32")
    b_f = b.astype("float32")
    denominator = a_f + b_f
    index = np.full(a.shape, np.nan, dtype=np.float32)
    ok = valid & (denominator != 0)
    index[ok] = (a_f[ok] - b_f[ok]) / denominator[ok]

    core = zone_mean(index, core_mask)
    control = zone_mean(index, control_mask)
    if core is None or control is None:
        return None

    return SceneLayers(
        index=index,
        valid=valid,
        core=core,
        control=control,
        item=item,
        core_valid_fraction=float(valid[core_mask].mean()),
        core_water_fraction=float(water[core_mask].mean()),
    )


def stretch_rgb(raw: np.ndarray) -> np.ndarray:
    """Normalize and 2-98% percentile stretch true-color channels to 8-bit RGB."""
    out = np.zeros_like(raw, dtype=np.float32)
    for channel_index in range(3):
        channel = raw[..., channel_index]
        sample = channel[channel > 0]
        if sample.size == 0:
            continue
        low, high = np.percentile(sample, (2, 98))
        if high <= low:
            low, high = float(sample.min()), float(sample.max())
        if high <= low:
            continue
        out[..., channel_index] = np.clip((channel - low) / (high - low), 0, 1)
    return (out * 255).astype(np.uint8)


def read_rgb(item: dict[str, Any], site: Site) -> np.ndarray:
    """Extract RGB true-color imagery from scene assets."""
    assets = item["assets"]
    _, bounds = scene_bounds(item, site.lat, site.lon)
    size = GRID_10M

    if "visual" in assets:
        try:
            with rasterio.open(assets["visual"]["href"]) as ds:
                window = from_bounds(*bounds, transform=ds.transform)
                rgb = ds.read(
                    [1, 2, 3], window=window, out_shape=(3, size, size),
                    resampling=Resampling.average, boundless=True, fill_value=0,
                )
            return np.clip(np.moveaxis(rgb, 0, -1), 0, 255).astype(np.uint8)
        except rasterio.errors.RasterioError:
            pass

    needed = ("red", "green", "blue")
    if any(key not in assets for key in needed):
        raise ValueError(f"RGB assets missing for scene {item['id']}")
    bands = [
        read_band(assets[key]["href"], bounds, (size, size), Resampling.average)
        for key in needed
    ]
    return stretch_rgb(np.stack(bands, axis=-1))


def collect_scenes(site: Site, start: str, end: str) -> list[dict[str, Any]]:
    """Search for cloud-filtered scenes, relaxing cloud limits if initially empty."""
    for cloud in (20, 35, 50):
        try:
            items = search_scenes(site, start, end, cloud)
        except requests.RequestException:
            items = []
        if items:
            return items
    return []


def pair_scenes(
    before: list[dict[str, Any]], after: list[dict[str, Any]], attempts: int = 6
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair baseline and current scenes prioritizing matching months and minimal clouds."""
    pairs = sorted(
        (
            (
                month_gap(
                    int(b["properties"]["datetime"][5:7]),
                    int(a["properties"]["datetime"][5:7]),
                ),
                float(b["properties"].get("eo:cloud_cover", 0))
                + float(a["properties"].get("eo:cloud_cover", 0)),
                b,
                a,
            )
            for b in before
            for a in after
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    return [(b, a) for _, _, b, a in pairs[:attempts]]


def baci_verdict(core_delta: float, net_delta: float) -> str:
    """Determine BACI verdict from project core delta and net control-corrected delta."""
    if net_delta >= BACI_THRESHOLD:
        return "ISOLATED_IMPACT"
    if core_delta >= BACI_THRESHOLD and abs(net_delta) < BACI_THRESHOLD:
        return "REGIONAL_CLIMATE_SHIFT"
    if abs(core_delta) < BACI_THRESHOLD and abs(net_delta) < BACI_THRESHOLD:
        return "MARGINAL_OR_NOISE"
    if net_delta <= -BACI_THRESHOLD:
        return "LOCAL_DEGRADATION"
    return "MARGINAL_OR_NOISE"


def empty_record(site: Site) -> dict[str, Any]:
    """Return initialized dictionary record for a site with null metrics."""
    return {
        "Site_ID": site.site_id,
        "Place": site.place,
        "Country": site.country,
        "Category": site.category,
        "Primary_Metric": site.metric,
        "Latitude": site.lat,
        "Longitude": site.lon,
        "Baseline_Date": "-",
        "Current_Date": "-",
        "Core_Before": None,
        "Core_After": None,
        "Core_Delta": None,
        "Ctrl_Before": None,
        "Ctrl_After": None,
        "Ctrl_Delta": None,
        "Net_Delta": None,
        "BACI_Verdict": "NO_DATA",
        "Baseline_Scene_ID": "-",
        "Current_Scene_ID": "-",
        "Baseline_Cloud_%": None,
        "Current_Cloud_%": None,
        "Month_Gap": None,
    }


def _evaluate_candidate_pair(
    site: Site, bitem: dict[str, Any], aitem: dict[str, Any]
) -> SiteResult | None:
    """Evaluate candidate baseline and current scene pair for BACI computation."""
    try:
        before = load_layers(bitem, site)
        after = load_layers(aitem, site)
    except (rasterio.errors.RasterioError, requests.RequestException, KeyError, ValueError) as err:
        logger.debug("Failed layer load for %s / %s: %s", bitem.get("id"), aitem.get("id"), err)
        return None

    if before is None or after is None:
        return None

    core_delta = round(after.core - before.core, 3)
    ctrl_delta = round(after.control - before.control, 3)
    net_delta = round(core_delta - ctrl_delta, 3)
    delta = after.index - before.index
    delta[~np.isfinite(delta)] = np.nan

    record = {
        "Site_ID": site.site_id,
        "Place": site.place,
        "Country": site.country,
        "Category": site.category,
        "Primary_Metric": site.metric,
        "Latitude": site.lat,
        "Longitude": site.lon,
        "Baseline_Date": bitem["properties"]["datetime"][:10],
        "Current_Date": aitem["properties"]["datetime"][:10],
        "Core_Before": round(before.core, 3),
        "Core_After": round(after.core, 3),
        "Core_Delta": core_delta,
        "Ctrl_Before": round(before.control, 3),
        "Ctrl_After": round(after.control, 3),
        "Ctrl_Delta": ctrl_delta,
        "Net_Delta": net_delta,
        "BACI_Verdict": baci_verdict(core_delta, net_delta),
        "Baseline_Scene_ID": bitem["id"],
        "Current_Scene_ID": aitem["id"],
        "Baseline_Cloud_%": round(float(bitem["properties"].get("eo:cloud_cover", 0)), 2),
        "Current_Cloud_%": round(float(aitem["properties"].get("eo:cloud_cover", 0)), 2),
        "Month_Gap": month_gap(
            int(bitem["properties"]["datetime"][5:7]),
            int(aitem["properties"]["datetime"][5:7]),
        ),
    }

    try:
        rgb_before = read_rgb(bitem, site)
        rgb_after = read_rgb(aitem, site)
    except (rasterio.errors.RasterioError, requests.RequestException, KeyError, ValueError) as err:
        logger.debug("RGB imagery unavailable for %s: %s", site.site_id, err)
        rgb_before = None
        rgb_after = None

    return SiteResult(record, before, after, delta, rgb_before, rgb_after)


def audit_site(site: Site) -> SiteResult:
    """Execute complete satellite audit workflow for a single target site."""
    record = empty_record(site)
    before_items = collect_scenes(site, BASELINE_START, BASELINE_END)
    after_items = collect_scenes(site, CURRENT_START, CURRENT_END)
    if not before_items or not after_items:
        return SiteResult(record)

    for bitem, aitem in pair_scenes(before_items, after_items):
        result = _evaluate_candidate_pair(site, bitem, aitem)
        if result is not None:
            return result

    return SiteResult(record)


# --------------------------------------------------------------------------- #
# Module 3: Reporting & Visual Artifact Generators
# --------------------------------------------------------------------------- #

def status_label(metric: str, verdict: str) -> str:
    """Map technical BACI verdict and metric into human-readable plain language."""
    if verdict == "ISOLATED_IMPACT":
        return "Verified Growth" if metric in {"NDVI", "NDWI"} else "New Construction"
    if verdict == "REGIONAL_CLIMATE_SHIFT":
        return "Weather Only"
    if verdict == "LOCAL_DEGRADATION":
        return {
            "NDVI": "Tree Loss",
            "NDWI": "Drier Soil",
            "NDBI": "More Greenery / Less Bare Ground",
        }[metric]
    if verdict == "NO_DATA":
        return "No Data"
    return "No Change"


def change_sentence(site: Site, record: dict[str, Any]) -> str:
    """Synthesize clean, neutral English summary explaining the spectral delta."""
    verdict = str(record["BACI_Verdict"])
    status = status_label(site.metric, verdict)

    if verdict == "NO_DATA" or pd.isna(record["Net_Delta"]):
        return f"{site.label}: no usable satellite pair was available ({status})."

    pct = float(record["Net_Delta"]) * 100

    phenomena = {
        "NDVI": ("greenery inside the planting zone", "less greenery inside the planting zone"),
        "NDWI": ("surface water and wet soil inside the project zone", "less surface water and wet soil inside the project zone"),
        "NDBI": ("buildings and bare ground inside the project zone", "more greenery and less bare ground inside the project zone"),
    }
    gain_desc, loss_desc = phenomena[site.metric]

    if verdict == "ISOLATED_IMPACT":
        return f"{site.label}: {pct:+.1f}% more {gain_desc} than on the surrounding land ({status})."
    if verdict == "REGIONAL_CLIMATE_SHIFT":
        qualifier = "the whole region got greener from seasonal rain" if site.metric == "NDVI" else (
            "the whole region got wetter from seasonal rain" if site.metric == "NDWI" else "the whole area changed at a similar rate"
        )
        return f"{site.label}: {pct:+.1f}% local difference; {qualifier} ({status})."
    if verdict == "LOCAL_DEGRADATION":
        return f"{site.label}: {abs(pct):.1f}% {loss_desc} than on the surrounding land ({status})."

    return f"{site.label}: {pct:+.1f}% local difference, too small to separate from normal year-to-year variation ({status})."


def add_zone_boxes(ax: plt.Axes, on_image: bool = True) -> None:
    """Draw solid project core box and dashed comparison control ring."""
    core_size = CORE_RADIUS_M * 2
    control_size = CTRL_INNER_M * 2
    styles = [("#000000", 2.1, 1.8, 0.45), ("#ffffff", 1.1, 0.9, 0.95)] if on_image else [(INK, 1.0, 0.8, 0.55)]

    for color, core_width, control_width, alpha in styles:
        core = Rectangle(
            (-CORE_RADIUS_M, -CORE_RADIUS_M), core_size, core_size,
            fill=False, edgecolor=color, linewidth=core_width, alpha=alpha,
        )
        control = Rectangle(
            (-CTRL_INNER_M, -CTRL_INNER_M), control_size, control_size,
            fill=False, edgecolor=color, linewidth=control_width,
            linestyle=(0, (4, 3)), alpha=alpha,
        )
        ax.add_patch(core)
        ax.add_patch(control)


def show_rgb(ax: plt.Axes, rgb: np.ndarray | None, title: str, boxes: bool = True) -> None:
    """Render RGB image onto matplotlib axes with consistent styling."""
    if rgb is None:
        ax.text(
            0.5, 0.5, "Image unavailable", transform=ax.transAxes,
            ha="center", va="center", color=MUTED, fontsize=11,
        )
    else:
        ax.imshow(rgb, extent=EXTENT, origin="upper", interpolation="bilinear")
        if boxes:
            add_zone_boxes(ax, on_image=True)

    ax.set_title(title, fontsize=13, color=INK, pad=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(WHITE)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def verify_core_land(site: Site, result: SiteResult) -> None:
    """Enforce strict land validity thresholds on project core."""
    if site.min_core_land <= 0 or result.before is None or result.after is None:
        return
    valid = [result.before.core_valid_fraction, result.after.core_valid_fraction]
    water = [result.before.core_water_fraction, result.after.core_water_fraction]
    if min(valid) <= site.min_core_land:
        raise ValueError(
            f"{site.site_id}: core valid land {min(valid) * 100:.1f}% "
            f"does not clear the {site.min_core_land * 100:.0f}% requirement"
        )
    if max(water) > 0:
        raise ValueError(
            f"{site.site_id}: core contains water ({max(water) * 100:.2f}% of pixels)"
        )
    print(
        f"LAND CHECK  {site.site_id}: core valid land "
        f"{valid[0] * 100:.1f}% before / {valid[1] * 100:.1f}% after; "
        f"water {water[0] * 100:.1f}% / {water[1] * 100:.1f}%"
    )


def save_site_visual(position: int, site: Site, result: SiteResult) -> Path:
    """Generate high-resolution 3-panel Before/After/Delta comparison card."""
    if result.before is None or result.after is None or result.delta is None:
        raise ValueError(f"No visual data for {site.site_id}")

    verify_core_land(site, result)

    record = result.record
    before_date = str(record["Baseline_Date"])
    after_date = str(record["Current_Date"])

    fig = plt.figure(figsize=(18, 7.6), facecolor=WHITE)
    grid = fig.add_gridspec(
        1, 3, left=0.025, right=0.975, top=0.93, bottom=0.30, wspace=0.055,
    )
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]

    show_rgb(axes[0], result.rgb_before, f"Before ({before_date})")
    show_rgb(axes[1], result.rgb_after, f"After ({after_date})")

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-DELTA_LIMIT, vmax=DELTA_LIMIT)
    axes[2].imshow(
        result.delta, extent=EXTENT, origin="upper", interpolation="nearest",
        cmap=DELTA_CMAPS[site.metric], norm=norm,
    )
    axes[2].set_title("What Changed", fontsize=13, color=INK, pad=10, loc="left")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_facecolor(WHITE)
    axes[2].set_aspect("equal")
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    add_zone_boxes(axes[2], on_image=False)

    key = textwrap.fill(CHANGE_KEYS[site.metric], width=58)
    axes[2].text(
        0.5, -0.055, key, transform=axes[2].transAxes,
        ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.35,
    )

    fig.text(
        0.5, 0.165,
        "Solid box = project area. Dashed box = surrounding comparison area. "
        "White areas are water, cloud, or missing data.",
        ha="center", va="center", fontsize=9, color=MUTED,
    )
    sentence = textwrap.fill(change_sentence(site, record), width=124)
    fig.text(
        0.5, 0.075, sentence, ha="center", va="center",
        fontsize=11.5, color=INK, linespacing=1.4,
    )

    VISUALS_DIR.mkdir(exist_ok=True)
    path = VISUALS_DIR / f"{position:02d}_{site.site_id}.png"
    fig.savefig(path, dpi=200, facecolor=WHITE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def build_contact_sheet(results: list[tuple[Site, SiteResult]]) -> Path:
    """Generate dense contact sheet displaying all verified sites side-by-side."""
    rows = len(results)
    fig = plt.figure(figsize=(11.5, 0.5 + rows * 2.35), facecolor=WHITE)
    grid = fig.add_gridspec(
        rows + 1, 4, width_ratios=[1, 1, 1, 1],
        height_ratios=[0.18] + [1] * rows,
        left=0.015, right=0.985, top=0.985, bottom=0.015,
        hspace=0.03, wspace=0.02,
    )
    headers = [
        "Before (2019\u20132020)",
        "After (2025\u20132026)",
        "What Changed\n(Green = Gain, Red = Loss)",
    ]
    for column, header in enumerate(headers, start=1):
        ax = fig.add_subplot(grid[0, column])
        ax.axis("off")
        ax.text(
            0.5, 0.45, header, ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK, linespacing=1.25,
        )

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-DELTA_LIMIT, vmax=DELTA_LIMIT)
    for row, (site, result) in enumerate(results):
        verdict = str(result.record["BACI_Verdict"])
        status = status_label(site.metric, verdict)
        color = STATUS_COLORS.get(status, MUTED)

        label_ax = fig.add_subplot(grid[row + 1, 0])
        label_ax.axis("off")
        label_ax.text(
            0.96, 0.58, textwrap.fill(site.label, width=20),
            ha="right", va="center", fontsize=9, fontweight="bold",
            color=INK, linespacing=1.25,
        )
        label_ax.text(
            0.96, 0.30, status, ha="right", va="center",
            fontsize=8, color=color,
        )

        show_rgb(fig.add_subplot(grid[row + 1, 1]), result.rgb_before, "")
        show_rgb(fig.add_subplot(grid[row + 1, 2]), result.rgb_after, "")

        heat_ax = fig.add_subplot(grid[row + 1, 3])
        heat_ax.set_facecolor(WHITE)
        heat_ax.set_xticks([])
        heat_ax.set_yticks([])
        heat_ax.set_aspect("equal")
        for spine in heat_ax.spines.values():
            spine.set_visible(False)

        if result.delta is not None:
            contact_delta = -result.delta if site.metric == "NDBI" else result.delta
            heat_ax.imshow(
                contact_delta, extent=EXTENT, origin="upper", interpolation="nearest",
                cmap="RdYlGn", norm=norm,
            )
            add_zone_boxes(heat_ax, on_image=False)
        else:
            heat_ax.text(
                0.5, 0.5, "No usable image", transform=heat_ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10,
            )

    path = HERE / "beast_visual_contact_sheet.png"
    fig.savefig(path, dpi=160, facecolor=WHITE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def _plot_financials_bar(ax: plt.Axes, fin: pd.DataFrame) -> None:
    """Subplot rendering latest financial revenue vs expense comparisons."""
    if fin.empty:
        return

    latest = fin.sort_values("Tax_Year").groupby("Organization").tail(1).copy()
    latest["Short_Name"] = (
        latest["Organization"]
        .str.replace("National ", "", regex=False)
        .str.replace("The Ocean Cleanup (US)", "Ocean Cleanup (US)", regex=False)
    )
    latest = latest.sort_values("Total_Revenue_USD")
    positions = np.arange(len(latest))
    revenue = latest["Total_Revenue_USD"] / 1e6
    expenses = latest["Total_Expenses_USD"] / 1e6

    ax.barh(positions + 0.19, revenue, height=0.36, color="#2da44e", label="Money in")
    ax.barh(positions - 0.19, expenses, height=0.36, color="#bf8700", label="Money out")
    ax.set_yticks(positions)
    ax.set_yticklabels(latest["Short_Name"], fontsize=9, color=INK)
    ax.set_ylim(-0.65, len(latest) - 0.35)
    ax.set_xlim(0, max(revenue.max(), expenses.max()) * 1.28)

    for position, value in zip(positions + 0.19, revenue, strict=False):
        ax.text(value + 1.2, position, f"${value:.1f}M", va="center", fontsize=7.5, color=MUTED)
    for position, value in zip(positions - 0.19, expenses, strict=False):
        ax.text(value + 1.2, position, f"${value:.1f}M", va="center", fontsize=7.5, color=MUTED)

    ax.set_xlabel("USD, millions", fontsize=9, color=MUTED)
    ax.set_title("Latest IRS filing: money in and money out", fontsize=13, color=INK, loc="left", pad=10)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)


def _plot_satellite_baci_bar(ax: plt.Axes, sat: pd.DataFrame) -> None:
    """Subplot rendering satellite net local difference against 2.5-point threshold."""
    values = sat.dropna(subset=["Net_Delta"]).copy()
    if values.empty:
        return

    values["Status"] = [
        status_label(metric, verdict)
        for metric, verdict in zip(values["Primary_Metric"], values["BACI_Verdict"], strict=False)
    ]
    values = values.sort_values("Net_Delta")
    positions = np.arange(len(values))
    change = values["Net_Delta"] * 100
    colors = [STATUS_COLORS.get(status, MUTED) for status in values["Status"]]

    ax.barh(positions, change, height=0.62, color=colors)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.axvline(2.5, color=RULE, linewidth=0.8, linestyle="--")
    ax.axvline(-2.5, color=RULE, linewidth=0.8, linestyle="--")

    for position, value in zip(positions, change, strict=False):
        offset = 0.35 if value >= 0 else -0.35
        align = "left" if value >= 0 else "right"
        ax.text(value + offset, position, f"{value:+.1f}", va="center", ha=align, fontsize=7.5, color=INK)

    ax.set_yticks(positions)
    ax.set_yticklabels(values["Place"] + ", " + values["Country"], fontsize=8.5)
    ax.set_ylim(-0.65, len(values) - 0.35)
    limit = max(float(np.abs(change).max()) * 1.22, 3.5)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(
        "Percentage-point difference (project area minus surrounding area)",
        fontsize=9, color=MUTED,
    )
    ax.set_title(
        "Did the project area change more than its neighborhood?",
        fontsize=13, color=INK, loc="left", pad=10,
    )

    handles = [
        Patch(facecolor=STATUS_COLORS.get(status, MUTED), label=status)
        for status in dict.fromkeys(values["Status"])
    ]
    handles.append(
        Line2D([0], [0], color=RULE, linewidth=0.8, linestyle="--", label="2.5-point mark")
    )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=3, fontsize=7.5, frameon=False, handlelength=1.2, columnspacing=1.2,
    )


def build_chart(fin: pd.DataFrame, sat: pd.DataFrame) -> None:
    """Generate side-by-side financial and satellite audit summary dashboard."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8.5), facecolor=WHITE,
        gridspec_kw={"width_ratios": [1, 1.1]},
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.84, bottom=0.24, wspace=0.48)

    _plot_financials_bar(ax1, fin)
    _plot_satellite_baci_bar(ax2, sat)

    for ax in (ax1, ax2):
        ax.set_facecolor(WHITE)
        ax.tick_params(length=0, colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(
        HERE / "beast_audit_dashboard.png", dpi=200, facecolor=WHITE,
        bbox_inches="tight", pad_inches=0.2,
    )
    plt.close(fig)


def _render_map_popup(row: dict[str, Any]) -> str:
    """Build clean HTML popup card for interactive Folium global map."""
    def fmt(value: Any, digits: int = 3) -> str:
        return "-" if pd.isna(value) else f"{float(value):.{digits}f}"

    baseline = f"{row['Baseline_Date']} &middot; {row['Baseline_Scene_ID']}"
    current = f"{row['Current_Date']} &middot; {row['Current_Scene_ID']}"
    cloud = f"{fmt(row['Baseline_Cloud_%'])} / {fmt(row['Current_Cloud_%'])}"
    core = f"{fmt(row['Core_Before'])} &rarr; {fmt(row['Core_After'])}"
    control = f"{fmt(row['Ctrl_Before'])} &rarr; {fmt(row['Ctrl_After'])}"

    return f"""
    <div style="font-family:system-ui;width:330px">
      <h4 style="margin:0 0 6px">{row["Site_ID"]}</h4>
      <table style="font-size:12px;border-collapse:collapse">
        <tr><td style="color:#666">Category</td><td><b>{row["Category"]}</b></td></tr>
        <tr><td style="color:#666">Metric</td><td><b>{row["Primary_Metric"]}</b></td></tr>
        <tr><td style="color:#666">Baseline</td><td>{baseline}</td></tr>
        <tr><td style="color:#666">Current</td><td>{current}</td></tr>
        <tr><td style="color:#666">Month gap</td><td>{fmt(row["Month_Gap"], 0)}</td></tr>
        <tr><td style="color:#666">Cloud %</td><td>{cloud}</td></tr>
        <tr><td style="color:#666">Core</td><td>{core}</td></tr>
        <tr><td style="color:#666">Control</td><td>{control}</td></tr>
        <tr><td style="color:#666">Core delta</td><td>{signed(row["Core_Delta"])}</td></tr>
        <tr><td style="color:#666">Ctrl delta</td><td>{signed(row["Ctrl_Delta"])}</td></tr>
        <tr><td style="color:#666">Net BACI</td><td><b>{signed(row["Net_Delta"])}</b></td></tr>
        <tr><td style="color:#666">Verdict</td><td><b>{row["BACI_Verdict"]}</b></td></tr>
      </table>
    </div>"""


def build_map(sat: pd.DataFrame) -> None:
    """Generate interactive global Leaflet/Folium map with BACI verdict markers."""
    fmap = folium.Map(location=[10, 15], zoom_start=2, tiles=None, control_scale=True)
    fmap.get_root().html.add_child(
        folium.Element("<title>BEAST AUDIT - BACI Satellite Verification Map</title>")
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri World Imagery", max_zoom=19,
    ).add_to(fmap)

    for row in sat.to_dict("records"):
        popup_html = _render_map_popup(row)
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]], radius=8,
            color=BACI_COLORS.get(row["BACI_Verdict"], MUTED), fill=True, fill_opacity=0.85,
            weight=1, popup=folium.Popup(popup_html, max_width=370), tooltip=row["Site_ID"],
        ).add_to(fmap)

    legend_rows = "".join(
        f'<span style="color:{color}">&#9679;</span> {name}<br>'
        for name, color in BACI_COLORS.items()
    )
    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:#0d1117;
                color:#e6edf3;padding:10px 12px;border:1px solid #30363d;border-radius:6px;
                font:12px system-ui">
      <b>BACI verdict</b><br>{legend_rows}
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True).add_to(fmap)
    fmap.save(str(HERE / "beast_global_map.html"))


# --------------------------------------------------------------------------- #
# CLI Pipeline Orchestration
# --------------------------------------------------------------------------- #

def main() -> int:
    """Execute complete end-to-end philanthropic audit pipeline."""
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

    print("MODULE 1  IRS Form 990 via ProPublica Nonprofit Explorer")
    fin_rows, notes = financial_rows()
    fin = pd.DataFrame(fin_rows)
    fin.to_csv(HERE / "beast_financials.csv", index=False)

    table_rows = [
        [
            r["Organization"][:30],
            str(r["Tax_Year"]),
            money(r["Total_Revenue_USD"]),
            money(r["Total_Expenses_USD"]),
            money(r["Executive_Comp_USD"]),
            money(r["Net_Assets_USD"]),
            f"{r['Expense_to_Revenue_Ratio_%']:.1f}" if r["Expense_to_Revenue_Ratio_%"] is not None else "-",
        ]
        for _, r in fin.iterrows()
    ]
    print()
    print(render_table([
        "Organization", "Tax_Year", "Total_Revenue_USD", "Total_Expenses_USD",
        "Executive_Comp_USD", "Net_Assets_USD", "Expense_to_Revenue_Ratio_%",
    ], table_rows))

    for note in notes:
        print(f"NOTE  {note}")

    print("\nMODULE 2  BACI Sentinel-2 L2A audit (3000 m window)")
    results: list[tuple[Site, SiteResult]] = []
    for i, site in enumerate(SITES, 1):
        print(f"[{i}/{len(SITES)}] {site.site_id} ({site.metric})")
        results.append((site, audit_site(site)))

    sat = pd.DataFrame([result.record for _, result in results])
    sat.to_csv(HERE / "beast_satellite_audit.csv", index=False)

    print()
    baci_rows = [
        [
            r.Site_ID, r.Category, r.Primary_Metric,
            signed(r.Core_Delta), signed(r.Ctrl_Delta),
            signed(r.Net_Delta), r.BACI_Verdict,
        ]
        for r in sat.itertuples()
    ]
    print(render_table([
        "Site_ID", "Category", "Metric", "Core_Delta",
        "Ctrl_Delta", "Net_Delta", "BACI_Verdict",
    ], baci_rows))

    print("\nMODULE 3  artifacts")
    build_chart(fin, sat)
    build_map(sat)

    VISUALS_DIR.mkdir(exist_ok=True)
    for stale in VISUALS_DIR.glob("*.png"):
        stale.unlink()

    visual_paths: list[Path] = []
    for i, (site, result) in enumerate(results, 1):
        if result.delta is None:
            print(f"NOTE  {site.site_id}: visual skipped, insufficient valid pixels")
            continue
        visual_paths.append(save_site_visual(i, site, result))

    contact_sheet = build_contact_sheet(results)

    artifacts = [
        HERE / "beast_financials.csv",
        HERE / "beast_satellite_audit.csv",
        HERE / "beast_audit_dashboard.png",
        HERE / "beast_global_map.html",
        contact_sheet,
        *visual_paths,
    ]
    for path in artifacts:
        print(f"  {path.relative_to(HERE)!s:<52} {path.stat().st_size / 1024:9.1f} KB")

    counts = sat["BACI_Verdict"].value_counts().to_dict()
    print(f"\nSUMMARY  isolated={counts.get('ISOLATED_IMPACT', 0)}"
          f"  regional={counts.get('REGIONAL_CLIMATE_SHIFT', 0)}"
          f"  marginal={counts.get('MARGINAL_OR_NOISE', 0)}"
          f"  degradation={counts.get('LOCAL_DEGRADATION', 0)}"
          f"  no_data={counts.get('NO_DATA', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
