# Coastline Analysis for Synthetic Geophysical Data

## Description
This project provides a Python script (`coastline.py`) in the `src/` directory to analyze a coastline using a raster DTM (Digital Terrain Model) and synthetic EM geophysical point data.

Key features:
1. Loading a DTM raster and point data from HDF5.
2. Building an alpha-shape approximation of the coastline.
3. Filtering terrestrial points located inland of the coastline.
4. Computing the distance from each point to the coastline.
5. Identifying the nearest and farthest points.
6. Producing three diagnostic figures with `matplotlib`:
   - DTM + coastline only
   - DTM + coastline + filtered EM data
   - DTM + coastline + nearest/farthest points joined to the coast by dashed connectors
7. Saving filtered data (with per-point distance to coast) for further use.

---

## Repository Structure

```
.
├── data/                              ← Input data (not tracked, download separately)
│   ├── dtm_region.tif                 ← DTM raster
│   └── em_data.h5                     ← Synthetic EM data (dataset key = "data")
├── results/                           ← Generated output
│   ├── 01_coastline_only.png
│   ├── 02_coastline_with_data.png
│   ├── 03_extreme_distance_points.png
│   └── coastal_data.h5                ← Filtered points with DISTANCE_COASTLINE (not tracked)
├── src/
│   └── coastline.py                   ← Main analysis script
├── .gitattributes
├── .gitignore
├── LICENSE
├── REQUIREMENTS.txt
└── README.md
```

Example output figures are included under `results/` so you can preview the expected result before running the analysis yourself.

---

## Requirements

- Python 3.7 or newer
- GDAL (installed automatically via `rasterio`)
- Python packages:

  numpy
  pandas
  rasterio
  matplotlib
  shapely
  alphashape
  scipy

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/d-madelis/Coastline.git
   cd Coastline
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Linux/macOS
   .venv\Scripts\activate          # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r REQUIREMENTS.txt
   ```

---

## Data Download

The input data files are hosted on Google Drive because they exceed GitHub's file-size limit. Download both files and place them inside the `data/` folder:

- [Download DTM GeoTIFF](https://drive.google.com/file/d/1pVoRnMPMzOfYJFvHJnhJdQU6-Gryr4I_/view) → save as `data/dtm_region.tif`

---

## Configuration

Open `src/coastline.py` and update the path constants at the top of the file (marked `# TODO: update paths`):

```python
DEFAULT_RASTER_PATH      = r"data/dtm_region.tif"
DEFAULT_EM_DATA_PATH     = r"data/em_data.h5"
DEFAULT_OUTPUT_DIR       = r"results"
DEFAULT_OUTPUT_DATA_PATH = r"results/coastal_data.h5"
```

- **`DEFAULT_RASTER_PATH`**: Path to your GeoTIFF DTM.
- **`DEFAULT_EM_DATA_PATH`**: Path to your HDF5 file containing EM point data (dataset key = "data").
- **`DEFAULT_OUTPUT_DIR`**: Folder where the three diagnostic PNGs will be written.
- **`DEFAULT_OUTPUT_DATA_PATH`**: Destination HDF5 for the filtered point cloud with distances.

---

## Usage

Run the analysis script:

```bash
python src/coastline.py
```

This will execute the full workflow and generate:

- `results/01_coastline_only.png`
- `results/02_coastline_with_data.png`
- `results/03_extreme_distance_points.png`
- `results/coastal_data.h5`

---

## Outputs

**Figures (`results/`):**
- `01_coastline_only.png` — Topographic map with the extracted coastline.
- `02_coastline_with_data.png` — Coastline plus filtered terrestrial EM points.
- `03_extreme_distance_points.png` — Coastline with the nearest and farthest EM points highlighted; each is connected to its closest point on the coast by a dashed line annotated with the distance in meters.

**Data (`results/coastal_data.h5`):**
HDF5 file (key = `"data"`) containing the filtered EM points with an added `DISTANCE_COASTLINE` column giving the straight-line distance (m) from each point to the coastline.

---

## Contributing

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/xyz`).
3. Commit your changes (`git commit -m "Add xyz"`).
4. Push to your branch (`git push origin feature/xyz`).
5. Open a Pull Request.κκ

---

## License

This project is released under the MIT License (see LICENSE).
