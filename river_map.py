from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt

from analysis import STATES_SHP
from main import DATA_DIR, MAP_CRS, load_port_boundary

RIVERS_SHP = DATA_DIR / "rivers" / "ne_10m_rivers_lake_centerlines.shp"
OUTPUT_PNG = Path(__file__).parent / "output" / "mississippi_river.png"

RIVER_COLOR = "#00e5ff"
# Padding around the river, in Web Mercator meters.
PADDING = (150_000, 100_000)


def load_mississippi() -> gpd.GeoDataFrame:
    rivers = gpd.read_file(RIVERS_SHP)
    return rivers[rivers["name"] == "Mississippi"].to_crs(MAP_CRS)


def main() -> None:
    river = load_mississippi()
    states = gpd.read_file(STATES_SHP).to_crs(MAP_CRS)
    port = load_port_boundary().to_crs(MAP_CRS)

    fig, ax = plt.subplots(figsize=(9, 14))
    states.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=0.8, zorder=1)
    river.plot(ax=ax, color=RIVER_COLOR, linewidth=3, zorder=2)
    port.plot(ax=ax, facecolor="none", edgecolor="#ffd54f", linewidth=2.5, zorder=3)

    minx, miny, maxx, maxy = river.total_bounds
    ax.set_xlim(minx - PADDING[0], maxx + PADDING[0])
    ax.set_ylim(miny - PADDING[1], maxy + PADDING[1])

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=MAP_CRS, zoom=7, zorder=0)
    ax.set_axis_off()
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved map to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
