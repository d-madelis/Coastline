"""
Coastline Analysis for Synthetic Geophysical Data.

This module processes synthetic EM point data relative to a coastline derived
from a Digital Terrain Model (DTM) raster. The workflow is:

    1. Load the DTM raster and the EM point dataset.
    2. Approximate the coastline via an alpha shape on sub-sampled land cells.
    3. Filter the EM points to keep only those on land.
    4. Build an ordered LineString representation of the coastline.
    5. Compute the distance from each land point to the coastline.
    6. Identify the nearest and farthest points from the coast.
    7. Render three diagnostic maps:
         (a) DTM + coastline,
         (b) DTM + coastline + filtered EM data,
         (c) DTM + coastline + nearest / farthest points joined to the coast.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely import geometry
from scipy.spatial import KDTree
import alphashape


# TODO: update paths
DEFAULT_RASTER_PATH = r"C:\Users\dmadelis\Documents\git\Coastline-main\data\dtm_region4.tif"
DEFAULT_EM_DATA_PATH = r"C:\Users\dmadelis\Documents\git\Coastline-main\data\em_data.h5"
DEFAULT_OUTPUT_DIR = r"C:\Users\dmadelis\Documents\git\Coastline-main\results"
DEFAULT_OUTPUT_DATA_PATH = r"C:\Users\dmadelis\Documents\git\Coastline-main\results\coastal_data.h5"

# Shared plot extent
PLOT_XLIM = (21100, 27050)
PLOT_YLIM = (379000, 382000)


class CoastlineAnalyzer:
    """Encapsulate coastline-analysis operations for a single survey region.

    The analyzer loads a DTM raster plus a point dataset, approximates the
    coastline via an alpha shape, filters the points to keep only land
    locations, and computes each point's distance to the coastline.

    Attributes:
        raster_path: Filesystem path to the DTM raster (GeoTIFF).
        data_path: Filesystem path to the HDF5 file with EM point data.
        synthetic_data: DataFrame with the raw point data (columns X, Y, ...).
        dtm_array: 2D NumPy array of elevation values from the raster.
        transform: Affine transform relating raster indices to world coords.
        hull_polygon: Shapely Polygon approximating the coastline.
        coastline_line: Shapely LineString ordered along the coastline.
        coastal_points_df: DataFrame with points that fall on land.
        distances: NumPy array of point-to-coastline distances (meters).
    """

    def __init__(self, raster_path: str, data_path: str):
        """Initialize the analyzer with file paths.

        Args:
            raster_path: Path to the DTM raster file (GeoTIFF).
            data_path: Path to the HDF5 file containing EM point data.
        """
        self.raster_path = raster_path
        self.data_path = data_path

        self.synthetic_data = None
        self.dtm_array = None
        self.transform = None
        self.hull_polygon = None
        self.coastline_line = None
        self.coastal_points_df = None
        self.distances = None

    def load_synthetic_data(self):
        """Load the EM point dataset from HDF5 into a DataFrame.

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
            KeyError: If the expected 'data' key is missing in the file.
        """
        self.synthetic_data = pd.read_hdf(self.data_path, key="data")
        print("EM data loaded successfully.")

    def load_raster(self, max_elevation: float = 1e3):
        """Load the DTM raster, masking values above a given elevation.

        Args:
            max_elevation: Values strictly above this threshold (meters) are
                replaced with NaN to remove nodata artifacts.

        Raises:
            rasterio.RasterioIOError: If the raster file cannot be read.
        """
        with rasterio.open(self.raster_path) as src:
            band1 = src.read(1).astype(float)
            band1[band1 > max_elevation] = np.nan
            self.dtm_array = band1
            self.transform = src.transform
        print("DTM raster loaded successfully.")

    def compute_alpha_shape_hull(self,
                                 elevation_threshold: float = 100000,
                                 x_min: float = 21000, x_max: float = 32000,
                                 y_min: float = 377000, y_max: float = 383000,
                                 downsample_step: int = 400,
                                 alpha: float = 0.03):
        """Approximate the coastline with an alpha shape over the DTM.

        All DTM cells with elevation below the threshold and inside the given
        bounding box are flattened, sub-sampled, and passed to the
        ``alphashape`` library. If the result is a MultiPolygon, the polygon
        with the largest area is retained.

        Args:
            elevation_threshold: Maximum elevation (m) to include as land.
            x_min: Minimum Easting of the bounding box.
            x_max: Maximum Easting of the bounding box.
            y_min: Minimum Northing of the bounding box.
            y_max: Maximum Northing of the bounding box.
            downsample_step: Keep every Nth candidate point to speed up the
                alpha-shape computation.
            alpha: Alpha parameter controlling hull concavity. Smaller
                values produce looser (more convex) hulls.

        Raises:
            RuntimeError: If the raster has not been loaded yet.
            ValueError: If no DTM cells match the filter criteria.
        """
        if self.dtm_array is None or self.transform is None:
            raise RuntimeError("Raster must be loaded before computing alpha shape.")

        n_rows, n_cols = self.dtm_array.shape
        ulx, uly = self.transform * (0, 0)
        xres = self.transform.a
        yres = self.transform.e

        x_coords = ulx + np.arange(n_cols) * xres + (xres / 2)
        y_coords = uly + np.arange(n_rows) * yres + (yres / 2)
        xg, yg = np.meshgrid(x_coords, y_coords)

        flat_elev = self.dtm_array.ravel()
        flat_x = xg.ravel()
        flat_y = yg.ravel()

        mask = (
            (flat_elev < elevation_threshold)
            & (flat_x > x_min) & (flat_x < x_max)
            & (flat_y > y_min) & (flat_y < y_max)
        )
        indices = np.where(mask)[0]

        if indices.size == 0:
            raise ValueError("No grid points found within the specified range and threshold.")

        subsample_indices = indices[::downsample_step]
        points = np.column_stack((flat_x[subsample_indices], flat_y[subsample_indices]))

        raw_hull = alphashape.alphashape(points, alpha)
        if isinstance(raw_hull, geometry.MultiPolygon):
            raw_hull = max(raw_hull.geoms, key=lambda p: p.area)

        self.hull_polygon = raw_hull
        print("Coastline (alpha shape) computed successfully.")

    def filter_coastal_points(self):
        """Keep only EM points that fall inside the coastline polygon.

        Points whose (X, Y) coordinates are contained by ``hull_polygon`` are
        stored in ``coastal_points_df``; the rest are discarded.

        Raises:
            RuntimeError: If the alpha-shape hull has not been computed yet.
        """
        if self.hull_polygon is None:
            raise RuntimeError("Alpha shape hull must be computed before filtering points.")

        inside_mask = self.synthetic_data.apply(
            lambda row: self.hull_polygon.contains(geometry.Point(row["X"], row["Y"])),
            axis=1
        )
        self.coastal_points_df = self.synthetic_data[inside_mask].copy()
        print(f"Filtering completed. {len(self.coastal_points_df)} land points retained.")

    def compute_ordered_coastline_line(self,
                                       start_point: np.ndarray = np.array([21138, 380221]),
                                       end_point: np.ndarray = np.array([27214, 381242])):
        """Build a LineString of the coastline between two reference points.

        Snaps ``start_point`` and ``end_point`` to their nearest vertices on
        the hull exterior (via a KD-tree), then slices the exterior coords in
        order to produce a single LineString running between them.

        Args:
            start_point: (X, Y) world coordinates of the desired start of the
                coastline segment.
            end_point: (X, Y) world coordinates of the desired end of the
                coastline segment.

        Raises:
            RuntimeError: If the alpha-shape hull has not been computed yet.
        """
        if self.hull_polygon is None:
            raise RuntimeError("Alpha shape hull must be computed before creating coastline line.")

        hull_coords = np.asarray(self.hull_polygon.exterior.coords)
        tree = KDTree(hull_coords)
        start_idx = tree.query(start_point)[1]
        end_idx = tree.query(end_point)[1]

        if start_idx < end_idx:
            ordered_coords = hull_coords[start_idx: end_idx + 1]
        else:
            ordered_coords = np.vstack((hull_coords[start_idx:], hull_coords[: end_idx + 1]))

        self.coastline_line = geometry.LineString(ordered_coords)

    def calculate_distances_to_coast(self):
        """Compute the distance from each land point to the coastline.

        Populates the ``DISTANCE_COASTLINE`` column of ``coastal_points_df``
        and mirrors the values in ``self.distances``.

        Raises:
            RuntimeError: If either the coastline LineString or the filtered
                land points are missing.
        """
        if self.coastline_line is None or self.coastal_points_df is None:
            raise RuntimeError("Both coastline line and coastal points must be available.")

        distances = self.coastal_points_df.apply(
            lambda row: self.coastline_line.distance(geometry.Point(row["X"], row["Y"])),
            axis=1
        ).to_numpy()

        self.coastal_points_df["DISTANCE_COASTLINE"] = distances
        self.distances = distances

    def identify_extreme_distance_points(self):
        """Return the land points with the smallest and largest distance to coast.

        Returns:
            A tuple ``(min_point, max_point)`` where each element is a pandas
            Series with the fields X, Y, and DISTANCE_COASTLINE.

        Raises:
            RuntimeError: If distances have not been calculated yet.
        """
        if "DISTANCE_COASTLINE" not in self.coastal_points_df.columns:
            raise RuntimeError("Distances must be calculated before identifying extremes.")

        min_idx = self.coastal_points_df["DISTANCE_COASTLINE"].idxmin()
        max_idx = self.coastal_points_df["DISTANCE_COASTLINE"].idxmax()

        min_point = self.coastal_points_df.loc[min_idx, ["X", "Y", "DISTANCE_COASTLINE"]]
        max_point = self.coastal_points_df.loc[max_idx, ["X", "Y", "DISTANCE_COASTLINE"]]
        return min_point, max_point

    # ------------------------------------------------------------------ #
    # Plotting helpers                                                   #
    # ------------------------------------------------------------------ #

    def _base_figure(self, title: str):
        """Create a figure with the DTM raster + standard formatting.

        Args:
            title: Title shown at the top of the figure.

        Returns:
            A tuple ``(fig, ax, img)`` with the figure, axes, and the AxesImage
            returned by ``rasterio.plot.show`` (needed for the colorbar).
        """
        src = rasterio.open(self.raster_path)
        fig, ax = plt.subplots(figsize=(14, 8))
        img = show(src, ax=ax, cmap="gray", alpha=0.85)

        ax.grid(True, linestyle="--", color="lightgrey", alpha=0.7, zorder=0)
        ax.set_xlim(*PLOT_XLIM)
        ax.set_ylim(*PLOT_YLIM)
        ax.set_xlabel("Easting (X)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Northing (Y)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

        return fig, ax, img

    def _attach_elevation_colorbar(self, ax, img):
        """Attach a horizontal elevation colorbar below the given axes.

        Args:
            ax: The main axes the colorbar should be attached to.
            img: The AxesImage returned by ``rasterio.plot.show``.
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.6)
        cbar = plt.colorbar(img.get_images()[0], cax=cax, orientation="horizontal")
        cbar.set_label("Elevation (m)", fontsize=12, fontweight="bold")

    def plot_coastline_only(self, output_path: str):
        """Render and save a map showing the DTM and the coastline only.

        Args:
            output_path: File path where the PNG figure is saved.

        Raises:
            RuntimeError: If the coastline LineString has not been built yet.
        """
        if self.coastline_line is None:
            raise RuntimeError("Coastline must be computed before plotting.")

        print("Generating coastline-only map...")
        fig, ax, img = self._base_figure("Coastline (Alpha-Shape Approximation)")

        x, y = self.coastline_line.xy
        ax.plot(x, y, color="green", linewidth=2.5, label="Coastline", zorder=2)
        ax.legend(fontsize=11, loc="upper right", framealpha=1.0)

        self._attach_elevation_colorbar(ax, img)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Coastline map saved to: {output_path}")
        plt.show()

    def plot_coastline_with_data(self, output_path: str):
        """Render and save a map with the DTM, coastline, and EM data.

        Args:
            output_path: File path where the PNG figure is saved.

        Raises:
            RuntimeError: If land points have not been filtered yet.
        """
        if self.coastal_points_df is None:
            raise RuntimeError(
                "filter_coastal_points() must be run before plotting this map."
            )

        print("Generating coastline + EM data map...")
        fig, ax, img = self._base_figure("Survey Region with EM Measurements")

        # Coastline
        x, y = self.coastline_line.xy
        ax.plot(x, y, color="green", linewidth=2.5, label="Coastline", zorder=2)

        # Filtered EM points
        ax.scatter(self.coastal_points_df["X"], self.coastal_points_df["Y"],
                   color="blue", s=15, label="EM Data", zorder=3)

        ax.legend(fontsize=11, loc="upper right", framealpha=1.0)

        self._attach_elevation_colorbar(ax, img)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Coastline + EM data map saved to: {output_path}")
        plt.show()

    def plot_extreme_distance_points(self, output_path: str):
        """Render and save a map highlighting the nearest and farthest points.

        The figure shows the DTM, the coastline, and two marker points — the
        EM point closest to the coastline and the one farthest from it. Each
        marker is connected to its closest point on the coastline by a dashed
        line annotated with the distance in meters.

        Args:
            output_path: File path where the PNG figure is saved.

        Raises:
            RuntimeError: If distances to coast have not been calculated.
        """
        if "DISTANCE_COASTLINE" not in (self.coastal_points_df.columns
                                        if self.coastal_points_df is not None
                                        else []):
            raise RuntimeError(
                "calculate_distances_to_coast() must be run before this plot."
            )

        print("Generating extreme-distance points map...")
        min_point, max_point = self.identify_extreme_distance_points()

        fig, ax, img = self._base_figure(
            "Nearest & Farthest EM Points from the Coastline"
        )

        # Coastline
        x, y = self.coastline_line.xy
        ax.plot(x, y, color="green", linewidth=2.5, label="Coastline", zorder=2)

        # Helper to draw a dashed connector from a point to its nearest spot on the coast.
        def _connector(point, color, marker_label, connector_label):
            """Plot the marker and a dashed line to the closest coastline vertex."""
            p = geometry.Point(point["X"], point["Y"])
            # Interpolate the closest point on the LineString to p
            closest = self.coastline_line.interpolate(self.coastline_line.project(p))

            ax.scatter(point["X"], point["Y"],
                       color=color, edgecolors="black", linewidths=1.2,
                       marker="o", s=120, zorder=5, label=marker_label)
            ax.plot([point["X"], closest.x], [point["Y"], closest.y],
                    color=color, linestyle="--", linewidth=1.8,
                    zorder=4, label=connector_label)

        _connector(min_point, "orange",
                   f"Nearest point ({min_point['DISTANCE_COASTLINE']:.1f} m)",
                   "Distance to coast (nearest)")
        _connector(max_point, "red",
                   f"Farthest point ({max_point['DISTANCE_COASTLINE']:.1f} m)",
                   "Distance to coast (farthest)")

        ax.legend(fontsize=10, loc="upper right", framealpha=1.0)

        self._attach_elevation_colorbar(ax, img)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Extreme-distance map saved to: {output_path}")
        plt.show()

    def save_coastal_data(self, filepath: str):
        """Save the filtered land points (with distances) to an HDF5 file.

        Args:
            filepath: Destination path for the HDF5 file. The data is written
                under the 'data' key and any existing file is overwritten.
        """
        self.coastal_points_df.to_hdf(filepath, key="data", mode="w")


def main():
    """Run the full coastline-analysis workflow with default paths."""
    print("Starting coastline analysis...")

    analyzer = CoastlineAnalyzer(
        raster_path=DEFAULT_RASTER_PATH,
        data_path=DEFAULT_EM_DATA_PATH,
    )

    analyzer.load_synthetic_data()
    analyzer.load_raster(max_elevation=1e3)

    analyzer.compute_alpha_shape_hull(
        elevation_threshold=100000,
        x_min=21000, x_max=32000,
        y_min=377000, y_max=383000,
        downsample_step=400,
        alpha=0.03,
    )

    analyzer.filter_coastal_points()

    analyzer.compute_ordered_coastline_line(
        start_point=np.array([21138, 380221]),
        end_point=np.array([27214, 381242]),
    )

    analyzer.calculate_distances_to_coast()

    # Three diagnostic figures
    analyzer.plot_coastline_only(
        output_path=rf"{DEFAULT_OUTPUT_DIR}\01_coastline_only.png"
    )
    analyzer.plot_coastline_with_data(
        output_path=rf"{DEFAULT_OUTPUT_DIR}\02_coastline_with_data.png"
    )
    analyzer.plot_extreme_distance_points(
        output_path=rf"{DEFAULT_OUTPUT_DIR}\03_extreme_distance_points.png"
    )

    analyzer.save_coastal_data(DEFAULT_OUTPUT_DATA_PATH)

    print("Execution completed.")


if __name__ == "__main__":
    main()