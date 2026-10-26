from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import searoute as sr
from matplotlib import patheffects
from shapely.geometry import LineString, Point

from main import DATA_DIR, FACILITY_COLOR, FACILITY_EDGE_COLOR, MAP_CRS, load_port_boundary

EIA_IMPORTS_XLSX = DATA_DIR / "eia814_company_level_imports_2025.xlsx"
COUNTRIES_SHP = DATA_DIR / "countries" / "ne_110m_admin_0_countries.shp"
OUTPUT_DIR = Path(__file__).parent / "output"

CRUDE_PROD_CODE = 25
# EIA-814 processing sites (PCOMP_SITEID) inside the three port parishes.
PORT_REGION_SITES = {
    "517": "Valero St. Charles (Norco)",
    "530": "Marathon Garyville",
    "4414": "Marathon Garyville - asphalt",
    "4784": "Shell/Equilon St. James terminal",
}
# EIA port-of-entry codes (PORT_CODE) for ports inside the three parishes: Gramercy is in
# St. James Parish and St. Rose is in St. Charles Parish (St. Rose has no crude in 2025).
PORT_REGION_ENTRY_PORTS = {"2010": "Gramercy, LA", "2013": "St. Rose, LA"}
FLOW_COLOR = "#00e5ff"
# Label offsets from each country's marker point (Web Mercator meters), so labels don't collide.
LABEL_OFFSETS = {
    "Argentina": (-500_000, 900_000),
    "Canada": (2_775_000, -1_790_000),
    "Colombia": (-1_300_000, -350_000),
    "Venezuela": (500_000, 900_000),
    "Guyana": (1_300_000, -150_000),
    "Brazil": (900_000, 0),
    "Iraq": (-900_000, 600_000),
    "Kuwait": (700_000, -700_000),
}
# Absolute label positions (lon, lat), for countries whose labels would otherwise collide.
LABEL_POSITIONS = {
    "Venezuela": (-61.8, 14.9),
    "Trinidad and Tobago": (-52.5, 10.5),
    "Guyana": (-48.2, 4.4),
    "Guatemala": (-95.5, 11.5),
}
# A representative crude export port (lon, lat) for each origin country. EIA only reports the
# country, so these are illustrative, not the actual loading terminals.
EXPORT_PORTS = {
    "MEXICO": (-93.18, 18.43),            # Dos Bocas
    "VENEZUELA": (-64.86, 10.17),         # Jose
    "COLOMBIA": (-75.69, 9.41),           # Covenas
    "BRAZIL": (-43.17, -22.95),           # Rio de Janeiro
    "GUYANA": (-57.0, 7.0),               # offshore Stabroek block
    "IRAQ": (48.9, 29.7),                 # Basrah
    "KUWAIT": (48.15, 29.07),             # Mina al-Ahmadi
    "CANADA": (-54.2, 47.2),              # Placentia Bay (Come By Chance)
    "TRINIDAD AND TOBAGO": (-61.47, 10.32),  # Pointe-a-Pierre
    "GUATEMALA": (-88.61, 15.70),         # Santo Tomas de Castilla
    "ARGENTINA": (-62.0, -38.9),          # Puerto Rosales (Bahia Blanca)
}
# Intermediate waypoints for routes the network would otherwise send the wrong way round.
ROUTE_VIA = {"ARGENTINA": [(-43.17, -22.95)]}  # up the Atlantic coast, not around Cape Horn
# Passages to avoid per country. Loaded VLCCs can't transit Suez and Red Sea traffic has been
# diverted, so Gulf crude is routed around the Cape of Good Hope.
ROUTE_RESTRICTIONS = {"IRAQ": ["suez", "northwest"], "KUWAIT": ["suez", "northwest"]}
MISSISSIPPI_MOUTH = (-89.2, 28.9)
# Map extent in Web Mercator meters; the bottom edge extends if an export port is further south.
MAP_XLIM = (-14_000_000, 6_500_000)
MAP_YLIM = (-3_800_000, 8_700_000)
MAX_LINE_WIDTH = 16


def load_crude() -> pd.DataFrame:
    imports = pd.read_excel(EIA_IMPORTS_XLSX, dtype={"PORT_CODE": str, "PCOMP_SITEID": str})
    return imports[imports["PROD_CODE"] == CRUDE_PROD_CODE].copy()


def processed_in_region(crude: pd.DataFrame) -> pd.DataFrame:
    """Crude processed at sites in the port parishes, whichever port it entered through."""
    out = crude[crude["PCOMP_SITEID"].isin(PORT_REGION_SITES)].copy()
    out["site"] = out["PCOMP_SITEID"].map(PORT_REGION_SITES)
    return out


def entered_at_region_ports(crude: pd.DataFrame) -> pd.DataFrame:
    """Crude that entered the US at a port inside the port parishes, wherever it went next."""
    out = crude[crude["PORT_CODE"].isin(PORT_REGION_ENTRY_PORTS)].copy()
    out["site"] = out["PCOMP_SNAM"].fillna("UNKNOWN")
    return out


def by_country(crude: pd.DataFrame) -> pd.DataFrame:
    totals = crude.groupby("CNTRY_NAME")["QUANTITY"].sum().sort_values(ascending=False)
    out = pd.DataFrame({"thousand_barrels": totals})
    out["share_pct"] = out["thousand_barrels"] / out["thousand_barrels"].sum() * 100
    return out


def smooth(coords, iterations=2) -> np.ndarray:
    """Chaikin corner cutting, keeping the end points fixed."""
    pts = np.asarray(coords, dtype=float)
    for _ in range(iterations):
        near = 0.75 * pts[:-1] + 0.25 * pts[1:]
        far = 0.25 * pts[:-1] + 0.75 * pts[1:]
        cut = np.empty((2 * len(near), 2))
        cut[0::2], cut[1::2] = near, far
        pts = np.vstack([pts[:1], cut, pts[-1:]])
    return pts


def sea_route(country_key, destination, include_final_leg=True) -> LineString:
    """Shipping route (lon/lat) from the country's export port to the Mississippi mouth,
    then up the river to the port region."""
    origin = EXPORT_PORTS[country_key]
    stops = [origin, *ROUTE_VIA.get(country_key, []), MISSISSIPPI_MOUTH]
    coords = [origin]
    for start, end in zip(stops[:-1], stops[1:]):
        restrictions = ROUTE_RESTRICTIONS.get(country_key, ["northwest"])
        leg = sr.searoute(list(start), list(end), units="km", restrictions=restrictions)
        coords.extend(leg["geometry"]["coordinates"])
    if include_final_leg:
        coords.append(destination)
    return LineString(smooth(coords))


def build(crude: pd.DataFrame, countries: pd.DataFrame, map_name: str) -> None:

    world = gpd.read_file(COUNTRIES_SHP)
    world["key"] = world["ADMIN"].str.upper()
    origins = world[world["key"].isin(countries.index)].copy()
    origins["thousand_barrels"] = origins["key"].map(countries["thousand_barrels"])
    origins["share"] = origins["thousand_barrels"] / origins["thousand_barrels"].sum()

    port = load_port_boundary().to_crs("EPSG:4326")
    destination = port.union_all().representative_point() if hasattr(port, "union_all") \
        else port.unary_union.representative_point()
    origins_m = origins.to_crs(MAP_CRS)

    dest_lonlat = (destination.x, destination.y)
    arcs = gpd.GeoDataFrame(
        origins[["ADMIN", "key", "share"]],
        geometry=[sea_route(key, dest_lonlat) for key in origins["key"]],
        crs="EPSG:4326",
    ).to_crs(MAP_CRS)

    ports = gpd.GeoSeries([Point(EXPORT_PORTS[k]) for k in origins["key"]], crs="EPSG:4326")
    ports_m = ports.to_crs(MAP_CRS)
    route_minx, route_miny, route_maxx, route_maxy = arcs.total_bounds
    xmax = max(MAP_XLIM[1], route_maxx + 250_000)
    ymin = min(MAP_YLIM[0], ports_m.y.min() - 400_000, route_miny - 300_000)
    width = 18
    fig, ax = plt.subplots(figsize=(width, width * (MAP_YLIM[1] - ymin) / (xmax - MAP_XLIM[0])))
    origins_m.plot(ax=ax, color=FLOW_COLOR, alpha=0.35, edgecolor="white", linewidth=0.8, zorder=1)
    for _, arc in arcs.sort_values("share", ascending=False).iterrows():
        gpd.GeoSeries([arc.geometry], crs=MAP_CRS).plot(
            ax=ax, color=FLOW_COLOR, linewidth=1 + MAX_LINE_WIDTH * arc["share"] / arcs["share"].max(),
            alpha=0.9, zorder=2)
    ports_m.plot(ax=ax, color="white", markersize=60, edgecolor="black", linewidth=1, zorder=3)
    gpd.GeoSeries([destination], crs="EPSG:4326").to_crs(MAP_CRS).plot(
        ax=ax, color=FACILITY_COLOR, markersize=400, edgecolor=FACILITY_EDGE_COLOR,
        linewidth=2.5, zorder=4)

    outline = [patheffects.withStroke(linewidth=3, foreground="black")]
    for _, row in origins_m.iterrows():
        point = row.geometry.representative_point()
        dx, dy = LABEL_OFFSETS.get(row["ADMIN"], (0, 0))
        x, y = point.x + dx, point.y + dy
        if row["ADMIN"] in LABEL_POSITIONS:
            x, y = Point(LABEL_POSITIONS[row["ADMIN"]]).x, Point(LABEL_POSITIONS[row["ADMIN"]]).y
            x, y = gpd.GeoSeries([Point(x, y)], crs="EPSG:4326").to_crs(MAP_CRS).iloc[0].coords[0]
        ax.text(x, y, f"{row['ADMIN']}\n{row['share'] * 100:.1f}%",
                color="white",
                fontsize=15, fontweight="bold", ha="center", va="center",
                path_effects=outline, zorder=5)

    ax.set_xlim(MAP_XLIM[0], xmax)
    ax.set_ylim(ymin, MAP_YLIM[1])
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=MAP_CRS, zoom=4, zorder=0)
    ax.set_axis_off()
    fig.tight_layout()

    path = OUTPUT_DIR / map_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    crude = load_crude()
    views = {
        "processed": (processed_in_region(crude), "", "crude_imports_to_port_region.png"),
        "entered": (entered_at_region_ports(crude), "_port_of_entry",
                    "crude_imports_via_port_region_ports.png"),
    }
    shares = {}
    for name, (selected, suffix, map_name) in views.items():
        countries = by_country(selected)
        countries.to_csv(OUTPUT_DIR / f"crude_imports_by_country_2025{suffix}.csv")
        selected.groupby(["site", "CNTRY_NAME", "PORT_CITY"])["QUANTITY"].sum().rename(
            "thousand_barrels").to_csv(OUTPUT_DIR / f"crude_imports_by_site_2025{suffix}.csv")
        print(f"--- {name}: {selected['QUANTITY'].sum():,.0f} thousand barrels")
        print(countries.round(1).to_string())
        shares[name] = countries["thousand_barrels"]
        build(selected, countries, map_name)

    comparison = pd.DataFrame({
        "thousand_barrels_processed_in_region": shares["processed"],
        "thousand_barrels_entered_at_region_ports": shares["entered"],
    }).fillna(0).sort_values("thousand_barrels_processed_in_region", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "crude_imports_comparison_2025.csv")


if __name__ == "__main__":
    main()
