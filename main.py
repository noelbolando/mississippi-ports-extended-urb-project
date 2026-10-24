from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PORT_SHP = DATA_DIR / "Port_Statistical_Area_7283592976755831424" / "Ports.shp"
PARISHES_SHP = DATA_DIR / "parishes" / "parishes.shp"
FACILITIES_CSV = DATA_DIR / "port_south_louisiana_refining_petrochem_facilities.csv"
OUTPUT_PNG = Path(__file__).parent / "output" / "port_south_louisiana_facilities.png"

# Facility coordinates are WGS84 lat/long; the shapefile is NAD83 (EPSG:4269).
FACILITY_CRS = "EPSG:4326"
# Web Mercator, which web map tiles are served in.
MAP_CRS = "EPSG:3857"

# Facility marker colors.
FACILITY_COLOR = "#ff3fd0"
FACILITY_EDGE_COLOR = "#b0006f"

# Facilities left off the map.
EXCLUDED_FACILITIES = ("Air Products Garyville SMR",)


def load_port_boundary() -> gpd.GeoDataFrame:
    return gpd.read_file(PORT_SHP)


def load_parishes() -> gpd.GeoDataFrame:
    return gpd.read_file(PARISHES_SHP)


def load_facilities() -> gpd.GeoDataFrame:
    df = pd.read_csv(FACILITIES_CSV)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[~df["facility_name"].isin(EXCLUDED_FACILITIES)]
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=FACILITY_CRS,
    )


def main() -> None:
    port = load_port_boundary().to_crs(MAP_CRS)
    parishes = load_parishes().to_crs(MAP_CRS)
    facilities = load_facilities().to_crs(MAP_CRS)

    # Tag each facility as inside/outside the port boundary.
    port_area = port.union_all() if hasattr(port, "union_all") else port.unary_union
    facilities["in_port"] = facilities.within(port_area)
    inside = facilities[facilities["in_port"]]
    outside = facilities[~facilities["in_port"]]

    print(f"{len(inside)} of {len(facilities)} facilities fall within the port boundary")
    if not outside.empty:
        print("Outside the boundary:")
        print(outside[["facility_name", "parish"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(18, 13))
    parishes.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=0.8, zorder=1)
    port.plot(ax=ax, facecolor="none", edgecolor="#ffd54f", linewidth=2, zorder=2)
    inside.plot(ax=ax, color=FACILITY_COLOR, markersize=150, edgecolor=FACILITY_EDGE_COLOR,
                linewidth=2,
                label=f"Inside port ({len(inside)})", zorder=3)
    if not outside.empty:
        outside.plot(ax=ax, color="#7f8c8d", markersize=150, marker="x",
                     label=f"Outside port ({len(outside)})", zorder=3)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=MAP_CRS)
    ax.set_axis_off()
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved map to {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
