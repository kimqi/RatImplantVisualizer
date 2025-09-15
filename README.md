# Rat Brain Atlas Implant Visualizer

Python tools for visualizing electrode implant coordinates in the rat brain atlas.
This is a Python port of MATLAB code originally used to generate coronal, sagittal, and horizontal atlas slices with implant markers.

Original MATLAB API written by Matt Gaidica at: [Rat Brain Atlas API](https://github.com/mattgaidica/RatBrainAtlasAPI) along with [Rat Brain Atlas](https://labs.gaidi.ca/rat-brain-atlas/)
The code queries an online API for the [Gaidi Rat Brain Atlas](http://labs.gaidi.ca/rat-brain-atlas/api.php), and demarcates implant coordinates.

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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage
Launch Jupyter
```
jupyter lab
```

Or use the API directly in Python
```python
from rat_brain_atlas_api import plot_implant_coords

# Example: AP = -3.6 mm, ML = 2.5 mm, DV = 2.8 mm, angle = 15 degrees
Scomb_bot, Scomb_top = plot_implant_coords(
    AP=-3.6,
    ML=2.5,
    DV=2.8,
    angle=15,
    span=750,       # microns
    skull_t=500,    # microns
    vert_span=2000  # microns (optional top span)
)
```

## Notes
- Requires internet connection: images are fetched on demand from labs.gaidi.ca.
- If the API is down, you will see “Image unavailable” placeholders.
- Pillow is optional, **but without it markers will not be drawn on the images.**
