"""
Coastline Analysis for Synthetic Geophysical Data

This script processes synthetic geophysical data to:
1. Remove points below the coastline (sea points) using an alpha shape.
2. Calculate distances from each point to the coastline.
3. Identify and plot the nearest and farthest points from the coast.
4. Display plots interactively as figures in a standard Python environment.

Replace file paths and parameters as needed for your environment.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely import geometry
import alphashape
from scipy.spatial import KDTree
import geopandas as gpd

# Ενεργοποίηση interactive mode ώστε τα plt.show() να εμφανίζουν παράθυρα γραφημάτων
plt.ion()


class CoastlineAnalyzer:
    """
    Class to encapsulate coastline analysis operations for Region4:
    - Loading raster and point data
    - Computing alpha shape hull (approximate coastline)
    - Filtering terrestrial points
    - Calculating distances to coastline
    - Plotting results interactively (no file output)
    """

    def __init__(self, raster_path: str, data_path: str):
        """
        Initialize paths.

        :param raster_path: Path to the Region4 DTM GeoTIFF.
        :param data_path: Path to the Region4 selected data HDF5.
        """
        self.raster_path = raster_path
        self.data_path = data_path

        # Attributes to be populated later
        self.synthetic_data = None        # pandas DataFrame of point data
        self.dtm_array = None             # 2D NumPy array of elevation
        self.transform = None             # Affine transform of raster
        self.hull_polygon = None          # Shapely Polygon for coastline
        self.coastline_line = None        # Shapely LineString ordered along coastline
        self.coastal_points_df = None     # DataFrame of terrestrial (inland) points
        self.distances = None             # NumPy array of distances to coastline

    def load_synthetic_data(self):
        """
        Load the Region4 point data from an HDF5 file into a pandas DataFrame.
        """
        try:
            self.synthetic_data = pd.read_hdf(self.data_path, key="data")
        except Exception as e:
            raise IOError(f"Failed to load synthetic data: {e}")

    def load_raster(self, max_elevation: float = 1e3):
        """
        Load the Region4 DTM raster, mask out invalid elevations, and store the array and transform.

        :param max_elevation: Maximum valid elevation value; values above become NaN.
        """
        with rasterio.open(self.raster_path) as src:
            band1 = src.read(1).astype(float)
            band1[band1 > max_elevation] = np.nan
            self.dtm_array = band1
            self.transform = src.transform

    def compute_alpha_shape_hull(self,
                                 elevation_threshold: float = 100000,
                                 x_min: float = 21000, x_max: float = 32000,
                                 y_min: float = 377000, y_max: float = 383000,
                                 downsample_step: int = 400,
                                 alpha: float = 0.03):
        """
        Compute the alpha shape (approximate coastline) from the raster grid for Region4.
        """
        if self.dtm_array is None or self.transform is None:
            raise RuntimeError("Raster must be loaded before computing alpha shape.")

        # Get raster dimensions
        n_rows, n_cols = self.dtm_array.shape
        ulx, uly = self.transform * (0, 0)
        xres = self.transform.a
        yres = self.transform.e

        # Create coordinate arrays for each pixel center
        x_coords = ulx + np.arange(n_cols) * xres + (xres / 2)
        y_coords = uly + np.arange(n_rows) * yres + (yres / 2)
        xg, yg = np.meshgrid(x_coords, y_coords)

        # Flatten arrays and apply filters
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

        # Downsample candidate points
        subsample_indices = indices[::downsample_step]
        points = np.column_stack((flat_x[subsample_indices], flat_y[subsample_indices]))

        # Compute the alpha shape hull
        raw_hull = alphashape.alphashape(points, alpha)
        if isinstance(raw_hull, geometry.MultiPolygon):
            raw_hull = max(raw_hull.geoms, key=lambda p: p.area)

        self.hull_polygon = raw_hull

    def filter_coastal_points(self):
        """
        Filter synthetic_data to keep only points inside the alpha shape polygon (terrestrial points).
        """
        if self.hull_polygon is None:
            raise RuntimeError("Alpha shape hull must be computed before filtering points.")

        inside_mask = self.synthetic_data.apply(
            lambda row: self.hull_polygon.contains(geometry.Point(row["X"], row["Y"])),
            axis=1
        )
        self.coastal_points_df = self.synthetic_data[inside_mask].copy()

    def compute_ordered_coastline_line(self,
                                       start_point: np.ndarray = np.array([21138, 380221]),
                                       end_point: np.ndarray = np.array([27214, 381242])):
        """
        Compute an ordered LineString along the hull perimeter from start to end for Region4.
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
        """
        Calculate perpendicular distances from each terrestrial point to the coastline line.
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
        """
        Identify the points with minimum and maximum distances to the coast.

        :return: Tuple of two Series: (min_point, max_point)
        """
        if "DISTANCE_COASTLINE" not in self.coastal_points_df.columns:
            raise RuntimeError("Distances must be calculated before identifying extremes.")

        min_idx = self.coastal_points_df["DISTANCE_COASTLINE"].idxmin()
        max_idx = self.coastal_points_df["DISTANCE_COASTLINE"].idxmax()

        min_point = self.coastal_points_df.loc[min_idx, ["X", "Y", "DISTANCE_COASTLINE"]]
        max_point = self.coastal_points_df.loc[max_idx, ["X", "Y", "DISTANCE_COASTLINE"]]
        return min_point, max_point

    def plot_topo_and_coastline(self):
        """
        Plot the DTM raster with the coastline overlay interactively for Region4,
        with a smaller, horizontal colorbar below the figure.
        """
        with rasterio.open(self.raster_path) as src:
            fig, ax = plt.subplots(figsize=(14, 12))
            img = show(src, ax=ax, cmap="terrain")
            x, y = self.coastline_line.xy
            ax.plot(x, y, color="black", linewidth=2, label="Coastline")

            cbar = plt.colorbar(
                img.get_images()[0], ax=ax,
                orientation="horizontal",
                shrink=0.5,
                pad=0.1
            )
            cbar.set_label("Elevation (m)", fontsize=12)

            ax.set_xlim(21000, 27290)
            ax.set_ylim(378800, 382000)
            ax.legend(fontsize=14)
            ax.set_title("Topographic Map with Coastline", fontsize=18, fontweight="bold")
            plt.tight_layout()
            plt.show()

    def plot_extreme_points(self, min_point: pd.Series, max_point: pd.Series):
        """
        Plot the DTM, coastline, and highlight the nearest and farthest points interactively.
        """
        with rasterio.open(self.raster_path) as src:
            fig, ax = plt.subplots(figsize=(14, 12))
            img = show(src, ax=ax, cmap="terrain")
            x, y = self.coastline_line.xy
            ax.plot(x, y, color="k", linewidth=2, label="Coastline")

            # Plot nearest point
            ax.scatter(min_point["X"], min_point["Y"], color="green", s=100, zorder=3, label="Nearest Point")
            np_proj = self.coastline_line.interpolate(
                self.coastline_line.project(geometry.Point(min_point["X"], min_point["Y"]))
            )
            ax.plot([min_point["X"], np_proj.x], [min_point["Y"], np_proj.y], color="green", linestyle="--", linewidth=2)
            ax.text(min_point["X"] + 50, min_point["Y"] + 50,
                    f'{min_point["DISTANCE_COASTLINE"]:.2f} m', fontsize=12, ha="right", color="green")

            # Plot farthest point with purple dashed line
            ax.scatter(max_point["X"], max_point["Y"], color="purple", s=100, zorder=3, label="Farthest Point")
            fp_proj = self.coastline_line.interpolate(
                self.coastline_line.project(geometry.Point(max_point["X"], max_point["Y"]))
            )
            ax.plot(
                [max_point["X"], fp_proj.x],
                [max_point["Y"], fp_proj.y],
                color="purple", linestyle="--", linewidth=2
            )
            ax.text(max_point["X"] + 50, max_point["Y"] + 50,
                    f'{max_point["DISTANCE_COASTLINE"]:.2f} m', fontsize=12, ha="left", color="purple")

            cbar = plt.colorbar(
                img.get_images()[0], ax=ax,
                orientation="horizontal",
                shrink=0.5,
                pad=0.1
            )
            cbar.set_label("Elevation (m)", fontsize=12)

            ax.set_xlim(21200, 27500)
            ax.set_ylim(379000, 382000)
            ax.legend(fontsize=12)
            ax.set_title("Nearest and Farthest Points from Coast", fontsize=20, fontweight="bold")
            ax.set_xlabel("X Coordinates", fontsize=12)
            ax.set_ylabel("Y Coordinates", fontsize=12)
            plt.tight_layout()
            plt.show()

    def plot_all_coastal_points(self):
        """
        Plot the DTM, coastline, and all terrestrial points interactively.
        """
        with rasterio.open(self.raster_path) as src:
            fig, ax = plt.subplots(figsize=(14, 12))
            img = show(src, ax=ax, cmap="terrain")
            x, y = self.coastline_line.xy
            ax.plot(x, y, color="k", linewidth=3, label="Coastline")

            ax.scatter(self.coastal_points_df["X"], self.coastal_points_df["Y"],
                       c="red", s=10, label="Terrestrial Points")

            cbar = plt.colorbar(
                img.get_images()[0], ax=ax,
                orientation="horizontal",
                shrink=0.5,
                pad=0.1
            )
            cbar.set_label("Elevation (m)", fontsize=12)

            ax.set_xlim(21200, 27290)
            ax.set_ylim(379000, 382000)
            ax.legend(fontsize=12)
            ax.set_title("Coastline and Terrestrial Geophysical Points", fontsize=18, fontweight="bold")
            ax.set_xlabel("X Coordinates", fontsize=12)
            ax.set_ylabel("Y Coordinates", fontsize=12)
            plt.tight_layout()
            plt.show()

    def save_coastal_data(self, filepath: str):
        """
        Save the filtered coastal DataFrame with distance column to HDF5.
        """
        self.coastal_points_df.to_hdf(filepath, key="data", mode="w")

    def save_coastline_geojson(self, filename: str = "D:/Διπλωματική ΠΜΣ/Region4/coastline.geojson"):
        """
        Save the coastline LineString as a GeoJSON file for future use.
        """
        coastline_gdf = gpd.GeoDataFrame(geometry=[self.coastline_line], crs="EPSG:28992")
        coastline_gdf.to_file(filename, driver="GeoJSON")


def main():
    # Paths for Region4
    raster_file = r"D:/Διπλωματική ΠΜΣ/git/dtm_region.tif"
    synthetic_data_file = r"D:/Διπλωματική ΠΜΣ/git/em_data.h5"

    analyzer = CoastlineAnalyzer(
        raster_path=raster_file,
        data_path=synthetic_data_file
    )

    analyzer.load_synthetic_data()
    analyzer.load_raster(max_elevation=1e3)

    analyzer.compute_alpha_shape_hull(
        elevation_threshold=100000,
        x_min=21000, x_max=32000,
        y_min=377000, y_max=383000,
        downsample_step=400,
        alpha=0.03
    )

    analyzer.filter_coastal_points()

    analyzer.compute_ordered_coastline_line(
        start_point=np.array([21138, 380221]),
        end_point=np.array([27214, 381242])
    )

    analyzer.plot_topo_and_coastline()

    analyzer.calculate_distances_to_coast()
    min_pt, max_pt = analyzer.identify_extreme_distance_points()
    print(f"Nearest point: X={min_pt['X']:.2f}, Y={min_pt['Y']:.2f}, Distance={min_pt['DISTANCE_COASTLINE']:.2f} m")
    print(f"Farthest point: X={max_pt['X']:.2f}, Y={max_pt['Y']:.2f}, Distance={max_pt['DISTANCE_COASTLINE']:.2f} m")

    analyzer.plot_extreme_points(min_pt, max_pt)
    analyzer.plot_all_coastal_points()

    analyzer.save_coastal_data(r"D:/Διπλωματική ΠΜΣ/Region4/coastal_data.h5")
    # analyzer.save_coastline_geojson()  # Will save to Region4/coastline.geojson by default


if __name__ == "__main__":
    main()
