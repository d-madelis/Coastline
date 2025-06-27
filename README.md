# Coastline Analysis for Synthetic Geophysical Data

## Description
This project provides a Python script (`coastline.py`) in the `src/` directory to analyze a coastline using a raster DTM (Digital Terrain Model) and synthetic geophysical point data.  
Key features include:
1. Loading a DTM raster and point data from HDF5.  
2. Building an alpha‐shape approximation of the coastline.  
3. Filtering terrestrial points located inland of the coastline.  
4. Computing distances from each point to the coastline.  
5. Identifying the nearest and farthest points.  
6. Visualizing results with `matplotlib` (DTM + coastline + points).  
7. Saving filtered data and the coastline geometry for further use.

---

## Repository Structure

```
.
├── data/                  ← HDF5 point data
│   └── em_data.h5         ← Synthetic geophysical data (dataset key = "data")
├── results/               ← Generated output
│   ├── coast.png          ← Coastline-only figure
│   └── coastal_data.png   ← Coastline + points figure
├── src/                   ← Source code
│   └── coastline.py       ← Main analysis script
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.txt
```

---

## Requirements

- Python 3.7 or newer  
- GDAL (or via `rasterio`)  
- Python packages:

  numpy  
  pandas  
  rasterio  
  matplotlib  
  shapely  
  alphashape  
  scipy  
  geopandas  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/coastline-analysis.git
   cd coastline-analysis
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Linux/macOS
   .venv\Scripts\activate        # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Download

Before running the analysis, download the GeoTIFF DTM file from Google Drive:

[Download DTM GeoTIFF](https://drive.google.com/file/d/1pVoRnMPMzOfYJFvHJnhJdQU6-Gryr4I_/view?usp=drive_link)

Save the downloaded file to `data/dtm_region.tif`.

---

## Configuration

Open `src/coastline.py` and set the file paths inside the `main()` function:

```python
raster_file = r"data/dtm_region.tif"
synthetic_data_file = r"data/em_data.h5"
```

- **`raster_file`**: Path to your GeoTIFF DTM.  
- **`synthetic_data_file`**: Path to your HDF5 file containing point data (dataset key = "data").

---

## Usage

Run the analysis script:

```bash
python src/coastline.py
```

This will generate:

- `results/coast.png`  
- `results/coastal_data.png`

and save filtered point data with distances to `results/coastal_data.h5`.

---

## Outputs

- **Figures** (`results/`):  
  - `coast.png`: Topographic map with extracted coastline.  
  - `coastal_data.png`: Coastline and terrestrial geophysical points.  

- **Data** (`results/coastal_data.h5`):  
  HDF5 file (key = `"data"`) with a `DISTANCE_COASTLINE` column for each point.

---

## Contributing

1. Fork this repository.  
2. Create a feature branch (`git checkout -b feature/xyz`).  
3. Commit your changes (`git commit -m "Add xyz"`).  
4. Push to your branch (`git push origin feature/xyz`).  
5. Open a Pull Request.

---

## License

This project is released under the MIT License (see LICENSE).


