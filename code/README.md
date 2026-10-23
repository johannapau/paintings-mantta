# Code for the Hyperspectral Dataset of Paintings

Python code and notebooks for working with the [Hyperspectral Dataset of Paintings](DATA_DOI): preprocessing raw FX10/FX17/IQ data into reflectance cubes, and widgets for inspecting the cubes.

- **Data:** [DATA_DOI] (place it in `data/`, see below)
- **Documentation of the data:** `dataset_description.pdf` in the data record

## Structure

```
.
├── environment.yml
├── LICENSE
├── README.md
└── code/
    ├── cube_tools.py               # widgets: CubeClipper, BandSurfer, SpectralExplorer
    ├── stitching_functions.py      # strip alignment and stitching (FX10, FX17)
    ├── resize.py                   # resizing to the physical painting dimensions
    ├── stitching.ipynb             # example: stitching FX10/FX17 strips
    ├── resizing.ipynb              # example: resizing the stitched images
    ├── cube_tools_example.ipynb    # example: cube inspection widgets
    └── pigments.ipynb              # paint reference spectra: inspection and extraction
```

## Getting started

### 1. Environment

```bash
conda env create -f environment.yml
conda activate stitch
```

Tested with Python [version]. The widgets run in Jupyter [and need `ipympl` / a `%matplotlib widget` backend, if that is the case].

### 2. Data

Download the dataset from [DATA_DOI] and place it in a `data/` folder next to `code/`, keeping its folder layout (`data/<camera>/raw`, `reflectance`, `gt`). The notebooks expect `[path or variable to edit]`. The cubes are large, so the code reads them with memory mapping where possible [and needs roughly X GB of RAM for full-size FX10 images].

## Preprocessing

Reflectance is computed as (raw - dark) / (white - dark), then the FX10 and FX17 strips are aligned, stitched and resized.

| Step | Camera | Where |
|---|---|---|
| Strip alignment and stitching | FX10, FX17 | `stitching.ipynb` (functions in `stitching_functions.py`) |
| Resizing to painting dimensions | FX10, FX17 | `resizing.ipynb` (function in `resize.py`) |
| Reflectance conversion | IQ | the IQ section of `pigments.ipynb` |

The notebooks are run in the order above. They take [raw data format] as input and save [output format] to `data/<camera>/reflectance/`.

## Widgets

Defined in `cube_tools.py`, with examples in [`code/cube_tools_example.ipynb`](code/cube_tools_example.ipynb). They take a cube of shape (Y, X, B), or a path to a `.npy` file.

| Widget | Purpose |
|---|---|
| `CubeClipper` | crop regions of interest from a cube and save them as `.npy` |
| `BandSurfer` | scroll through the bands of a cube |
| `SpectralExplorer` | plot spectra by clicking or lassoing a region |

## Paint reference spectra

`pigments.ipynb` shows how the average reflectance spectra of the four acrylics were extracted. The results are in `data/<camera>/gt/`.

## Citation

If you use this code, please cite the code record ([CODE_DOI]) and the dataset ([DATA_DOI]).

## License

MIT, see `LICENSE`.

## Contact

[Name, email]