# Rat Brain Atlas Implant Visualizer

Python tools for visualizing electrode implant coordinates in the rat brain atlas.
This is a Python port of MATLAB code originally used to generate coronal, sagittal, and horizontal atlas slices with implant markers.

Original MATLAB API written by Matt Gaidica at: [Rat Brain Atlas API](https://github.com/mattgaidica/RatBrainAtlasAPI) along with [Rat Brain Atlas](https://labs.gaidi.ca/rat-brain-atlas/).
The code queries an online API for the [Gaidica Rat Brain Atlas](http://labs.gaidi.ca/rat-brain-atlas/api.php), and demarcates implant coordinates.

---

## Features
- Query rat brain atlas slices (coronal, sagittal, horizontal) by **AP/ML/DV coordinates** (in mm).
- Draw markers for implant positions
  - Optionally show top and bottom of implant span if vertical span is specified
- Plot results via notebook

---

## Install
Clone Repository
```bash
git clone https://github.com/kimqi/rat-atlas-implant-viz.git
cd rat-atlas-implant-viz
```

Create virtual environment and install requirements
```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .\.venv\Scripts\Activate     # Windows PowerShell

pip install -r requirements.txt
```

---

## Usage
Launch Jupyter and run the notebook
```
jupyter lab
# jupyter notebook
```

Or use the API directly in Python
```python
from rat_brain_atlas_api import plot_implant_coords

# Example: AP = -3.6 mm, ML = 2.5 mm, DV = 2.8 mm, angle = 30 degrees and plot 200 micron span (top to bottom)
Scomb_bot, Scomb_top = plot_implant_coords(
    AP=-3.6,
    ML=2.5,
    DV=2.8,
    angle=30,
    span=750,      # microns
    skull_t=500,   # microns
    vert_span=200  # microns (optional top span)
)
```

## Notes
- Requires internet connection: images are fetched on demand from labs.gaidi.ca.
- If the API is down, images will be unavailable
- Pillow is required for markers
