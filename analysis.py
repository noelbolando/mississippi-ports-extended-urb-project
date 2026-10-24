from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import (
    DATA_DIR,
    FACILITIES_CSV,
    FACILITY_COLOR,
    FACILITY_EDGE_COLOR,
    MAP_CRS,
    load_facilities,
    load_parishes,
    load_port_boundary,
)

TRACTS_SHP = DATA_DIR / "tracts" / "tracts.shp"
PLACES_CSV = DATA_DIR / "places_tracts_3parishes.csv"
# EPA AirToxScreen 2020 block-level cancer risk, filtered to the three parishes.
AIRTOXSCREEN_CSV = DATA_DIR / "airtoxscreen_2020_blocks_3parishes.csv"
COUNTIES_SHP = DATA_DIR / "us_counties" / "cb_2019_us_county_5m.shp"
STATES_SHP = DATA_DIR / "us_states" / "cb_2019_us_state_5m.shp"
# EPA AirToxScreen 2020 county-average cancer risk for the whole US.
COUNTY_RISK_XLSX = DATA_DIR / "airtoxscreen_2020_county_cancer_risk.xlsx"
OUTPUT_DIR = Path(__file__).parent / "output"

LOUISIANA_FIPS = "22"
# Alaska, Hawaii and the territories are left off the US map.
NON_CONTIGUOUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CANCER_RISK_LABEL = "Cancer risk from air toxics (per million)"
# One color scale for every cancer risk map so the scales can be compared directly.
CANCER_RISK_RANGE = (20, 100)

EMISSION_COLUMNS = {
    "ghg_total_tco2e_2023": "GHG (t CO2e, 2023)",
    "tri_total_releases_lb_2024": "TRI releases (lb, 2024)",
    "tri_dioxin_releases_g_2024": "TRI dioxin (g, 2024)",
}
PLACES_MEASURES = ["CASTHMA", "COPD", "CANCER", "CSMOKING"]


def weighted_mean(df: pd.DataFrame, value: str, weight: str) -> float:
    w = df[weight]
    if w.sum() > 0:
        return float(np.average(df[value], weights=w))
    return float(df[value].mean())


def parish_emissions() -> pd.DataFrame:
    df = pd.read_csv(FACILITIES_CSV)
    out = df.groupby("parish").agg(
        facilities=("facility_name", "count"),
        **{label: (col, "sum") for col, label in EMISSION_COLUMNS.items()},
    )
    return out


def tract_cancer_risk() -> pd.DataFrame:
    """Population-weighted cancer risk (per million) per census tract."""
    blocks = pd.read_csv(AIRTOXSCREEN_CSV, dtype={"FIPS": str, "Block": str})
    blocks["GEOID"] = blocks["Block"].str[:11]
    point_col = "PT-StationaryPoint Cancer Risk (per million)"
    rows = []
    for geoid, g in blocks.groupby("GEOID"):
        rows.append({
            "GEOID": geoid,
            "population": g["Population"].sum(),
            "cancer_risk_per_million": weighted_mean(g, "Total Cancer Risk (per million)", "Population"),
            "point_source_risk_per_million": weighted_mean(g, point_col, "Population"),
        })
    return pd.DataFrame(rows).set_index("GEOID")


def tract_places() -> pd.DataFrame:
    p = pd.read_csv(PLACES_CSV, dtype={"locationid": str})
    p = p[p["measureid"].isin(PLACES_MEASURES)]
    wide = p.pivot_table(index="locationid", columns="measureid", values="data_value")
    wide.index.name = "GEOID"
    wide["adult_population"] = p.groupby("locationid")["totalpop18plus"].first()
    return wide


def build_tracts() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(TRACTS_SHP)[["GEOID", "COUNTYFP", "geometry"]]
    tracts = tracts.merge(tract_cancer_risk(), on="GEOID", how="left")
    tracts = tracts.merge(tract_places(), on="GEOID", how="left")
    parish_names = {"089": "St. Charles", "093": "St. James", "095": "St. John the Baptist"}
    tracts["parish"] = tracts["COUNTYFP"].map(parish_names)
    return tracts


def parish_summary(tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for parish, g in tracts.groupby("parish"):
        row = {
            "parish": parish,
            "tracts": len(g),
            "population": g["population"].sum(),
            "cancer_risk_per_million": weighted_mean(g, "cancer_risk_per_million", "population"),
            "point_source_risk_per_million": weighted_mean(g, "point_source_risk_per_million", "population"),
        }
        for m in PLACES_MEASURES:
            valid = g.dropna(subset=[m])
            row[f"{m}_pct"] = weighted_mean(valid, m, "adult_population")
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("parish")
    return summary.join(parish_emissions())


def county_risk() -> gpd.GeoDataFrame:
    """US counties joined to EPA's county-average cancer risk (per million)."""
    counties = gpd.read_file(COUNTIES_SHP)[["GEOID", "STATEFP", "NAME", "geometry"]]
    risk = pd.read_excel(COUNTY_RISK_XLSX)
    risk["GEOID"] = risk["FIPS"].astype(str).str.zfill(5)
    risk = risk.rename(columns={"Total Cancer Risk (per million)": "cancer_risk_per_million"})
    return counties.merge(risk[["GEOID", "cancer_risk_per_million"]], on="GEOID", how="left")


def draw_map(shaded, column, label, filename, cmap, outline, boundaries=None,
             edge_width=0.4, boundary_width=1.2, zoom="auto", value_range=None,
             show_facilities=True, alpha=0.7, rotate_ticks=False):
    """Shade polygons by `column` over satellite imagery, outline the subject area in
    yellow, and optionally mark the facilities in black."""
    shaded = shaded.to_crs(MAP_CRS)

    legend_kwds = {"shrink": 0.7, "aspect": 18, "pad": 0.01}
    range_kwds = {}
    if value_range:
        range_kwds = {"vmin": value_range[0], "vmax": value_range[1]}
        legend_kwds["extend"] = "max"

    fig, ax = plt.subplots(figsize=(18, 13))
    shaded.plot(ax=ax, column=column, cmap=cmap, alpha=alpha, edgecolor="white",
                linewidth=edge_width, legend=True, legend_kwds=legend_kwds,
                missing_kwds={"color": "lightgrey", "alpha": 0.5}, zorder=1, **range_kwds)
    if boundaries is not None:
        boundaries.to_crs(MAP_CRS).plot(ax=ax, facecolor="none", edgecolor="white",
                                        linewidth=boundary_width, zorder=2)
    outline.to_crs(MAP_CRS).plot(ax=ax, facecolor="none", edgecolor="#ffd54f",
                                 linewidth=2.5, zorder=3)
    if show_facilities:
        load_facilities().to_crs(MAP_CRS).plot(ax=ax, color=FACILITY_COLOR, markersize=150,
                                               edgecolor=FACILITY_EDGE_COLOR, linewidth=2,
                                               zorder=4)

    colorbar_ax = fig.axes[-1]
    colorbar_ax.tick_params(labelsize=20)
    if rotate_ticks:
        # Same reading direction as the title.
        for tick_label in colorbar_ax.get_yticklabels():
            tick_label.set_rotation(90)
            tick_label.set_va("center")
    colorbar_ax.set_ylabel(label, fontsize=22, labelpad=15)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=MAP_CRS, zoom=zoom, zorder=0)
    ax.set_axis_off()
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    tracts = build_tracts()
    summary = parish_summary(tracts)

    tracts.drop(columns="geometry").to_csv(OUTPUT_DIR / "tract_summary.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "parish_summary.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.1f}".format):
        print(summary.T.to_string())

    port = load_port_boundary()
    parishes = load_parishes()
    draw_map(tracts, "cancer_risk_per_million", CANCER_RISK_LABEL, "tract_cancer_risk.png",
             "YlOrRd", outline=port, boundaries=parishes, value_range=CANCER_RISK_RANGE,
             alpha=0.5)
    draw_map(tracts, "CASTHMA", "Current adult asthma (% of adults)", "tract_asthma.png",
             "PuBu", outline=port, boundaries=parishes)

    counties = county_risk()
    louisiana = counties[counties["STATEFP"] == LOUISIANA_FIPS]
    draw_map(louisiana, "cancer_risk_per_million", CANCER_RISK_LABEL,
             "louisiana_cancer_risk.png", "YlOrRd", outline=louisiana.dissolve(),
             edge_width=0.6, zoom=9, value_range=CANCER_RISK_RANGE, show_facilities=False)

    contiguous = counties[~counties["STATEFP"].isin(NON_CONTIGUOUS_FIPS)]
    states = gpd.read_file(STATES_SHP)
    states = states[~states["STATEFP"].isin(NON_CONTIGUOUS_FIPS)]
    draw_map(contiguous, "cancer_risk_per_million", CANCER_RISK_LABEL, "us_cancer_risk.png",
             "YlOrRd", outline=states.dissolve(), boundaries=states, edge_width=0.15,
             boundary_width=0.8, zoom=6, value_range=CANCER_RISK_RANGE, show_facilities=False,
             rotate_ticks=True)

if __name__ == "__main__":
    main()
