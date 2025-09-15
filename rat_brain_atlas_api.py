from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
import io
import json
import math
import requests
import numpy as np
import matplotlib.pyplot as plt

# If no Pillow just show the atlas at the coordinates.
try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# -------------------------
# Low-level API Structures
# -------------------------

# Represent a single slice (coronal, sagittal, horizontal slices) from rat brain atlas API with metadata
# Optionally load images
@dataclass
class PlaneInfo:
    top: int
    left: int
    image_url: str
    image: Optional["Image.Image"] = None
    image_marked: Optional["Image.Image"] = None

# Bundle the three planes together that the API returns
@dataclass
class AtlasResponse:
    coronal: PlaneInfo
    sagittal: PlaneInfo
    horizontal: PlaneInfo

    @staticmethod
    def from_json(d: Dict[str, Any]):
        """
        Construct Atlas coordinates from JSON

        Parameters
        d : dict
            JSON response for c/s/h slices

        Returns
        AtlasResponse
            Object wrapping PlaneInfo for each plane
        """
        def _plane(key: str):
            p = d[key]
            return PlaneInfo(
                top=int(p["top"]),
                left=int(p["left"]),
                image_url=str(p["image_url"]),
                image=None,
                image_marked=None,
            )
        return AtlasResponse(
            coronal=_plane("coronal"),
            sagittal=_plane("sagittal"),
            horizontal=_plane("horizontal"),
        )

    def as_dict(self):
        """
        Convert AtlasResponse into dict
        """
        return asdict(self)


# -------------------------
# API helpers
# -------------------------

API_BASE = "http://labs.gaidi.ca/rat-brain-atlas/api.php"

def atlas_url(ml: float, ap: float, dv: float):
    """Build query URL for atlas API"""
    return f"{API_BASE}?ml={ml}&ap={ap}&dv={dv}"

def _webread(url: str):
    """Fetch raw bytes from URL with timeout"""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def _read_image(url: str):
    """Download image and return a Pillow image"""
    try:
        data = _webread(url)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def rat_brain_atlas(ml: float, ap: float, dv: float):
    """Query atlas API for c/s/h planes"""

    # Fetch url
    url = atlas_url(ml, ap, dv)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Unable to complete web request to {url!r}.") from e

    # JSON Parse
    try:
        d = r.json()  # Fast path
    except ValueError:
        # If no JSON headers (might be deprecated)
        try:
            d = json.loads(r.content.decode("utf-8", errors="replace"))
        except Exception as e:
            sample = r.content[:200]
            raise RuntimeError(
                "Failed to parse JSON from atlas API. "
                f"First 200 bytes of response: {sample!r}"
            ) from e

    # API level error
    if isinstance(d, dict) and d.get("error"):
        raise RuntimeError(f"Atlas API error: {d.get('error')}")

    # Get dataclasses from JSON
    slice_views = AtlasResponse.from_json(d)

    # Fetch plane images
    for plane in (slice_views.coronal, slice_views.sagittal, slice_views.horizontal):
        plane.image = _read_image(plane.image_url)

    # If Pillow is present, overlay implant locations
    if _HAS_PIL:
        for plane in (slice_views.coronal, slice_views.sagittal, slice_views.horizontal):
            if plane.image is not None:
                im = plane.image.copy()
                draw = ImageDraw.Draw(im)
                rpx = 10
                x, y = plane.left, plane.top
                draw.ellipse((x - rpx, y - rpx, x + rpx, y + rpx), fill=(255, 0, 0))
                plane.image_marked = im

    return slice_views


# -------------------------
# Plot Implants
# -------------------------

@dataclass
class CombinedAtlas:
    """Container mimicking the MATLAB combined struct list"""
    entries: List[AtlasResponse]

def _consolidate(Sleft: AtlasResponse, Scenter: AtlasResponse, Sright: AtlasResponse):
    """Wrap CombinedAtlases"""
    return CombinedAtlas(entries=[Sleft, Scenter, Sright])

def _insert_markers_on_planes(
    atlas_struct: CombinedAtlas,
    radius_px: int,
    multi_mark_horizontal: Optional[List[Tuple[int, int, int]]] = None,
):
    """
    Adds red markers:
      - Coronal: mark electrode
      - Horizontal: mark all provided coords
    """
    if not _HAS_PIL:
        return atlas_struct  # No Pillow

    for entry in atlas_struct.entries:
        # Coronal: one marker per entry
        if entry.coronal.image is not None:
            im = entry.coronal.image.copy()
            d = ImageDraw.Draw(im)
            x, y = entry.coronal.left, entry.coronal.top
            d.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=(255, 0, 0))
            entry.coronal.image_marked = im

        # Horizontal: L/C/R or just original entry
        if entry.horizontal.image is not None:
            im = entry.horizontal.image.copy()
            d = ImageDraw.Draw(im)
            if multi_mark_horizontal:
                for (x, y, r) in multi_mark_horizontal:
                    d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
            else:
                x, y = entry.horizontal.left, entry.horizontal.top
                d.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=(255, 0, 0))
            entry.horizontal.image_marked = im
    return atlas_struct


def _sind(deg: float):
    return math.sin(math.radians(deg))

def _cosd(deg: float):
    return math.cos(math.radians(deg))


def plot_implant_coords(
    AP: float,
    ML: float,
    DV: float,
    angle: float,
    *,
    span: float = 750.0,       # microns
    skull_t: float = 500.0,    # microns
    vert_span: float = float("nan"),  # microns
    plot_radius: int = 5       # pixels for drawn circle
):
    """
    Python port of plot_implant_coords. Returns (s_comb_bot, s_comb_top).
    Fetch and visualize atlas slices for three electrode targets (left/center/right), given base AP/ML/DV, implant angle, and geometric spans.

    Units:
      - AP/ML/DV: mm
      - span, skull_t, vert_span: microns
    """
    # Convert microns to mm
    span_mm = span / 1000.0
    skull_mm = skull_t / 1000.0
    vert_mm = (vert_span / 1000.0) if not math.isnan(vert_span) else float("nan")
    #angle = -float(angle)  # flip rotation sense: +θ now behaves like previous -θ

    # Bottom (tip) coordinates in mm (center and r/l offsets with angle)
    center_bot = np.array([AP, ML, DV + skull_mm])
    left_bot   = np.array([AP - _sind(-angle) * span_mm / 2.0,
                           ML - _cosd(-angle) * span_mm / 2.0,
                           DV + skull_mm])
    right_bot  = np.array([AP + _sind(-angle) * span_mm / 2.0,
                           ML + _cosd(-angle) * span_mm / 2.0,
                           DV + skull_mm])

    # Optional top coordinates if vert_span is provided
    if not math.isnan(vert_span):
        center_top = center_bot - np.array([0.0, 0.0, vert_mm])
        left_top   = left_bot   - np.array([0.0, 0.0, vert_mm])
        right_top  = right_bot  - np.array([0.0, 0.0, vert_mm])
    else:
        center_top = left_top = right_top = None

    # Helper to query one coordinate triple (note ML/AP/DV order for API)
    def S_at(coords) -> AtlasResponse:
        ap, ml, dv = float(coords[0]), float(coords[1]), float(coords[2])
        return rat_brain_atlas(ml=ml, ap=ap, dv=dv)

    s_left = S_at(left_bot)
    s_center = S_at(center_bot)
    s_right  = S_at(right_bot)

    # Consolidate and add markers
    s_comb_bot = _consolidate(s_left, s_center, s_right)

    # For horizontal images, insert all three markers on each image
    horiz_triplet = [
        (s_left.horizontal.left,   s_left.horizontal.top,   plot_radius),
        (s_center.horizontal.left, s_center.horizontal.top, plot_radius),
        (s_right.horizontal.left,  s_right.horizontal.top,  plot_radius),
    ]
    s_comb_bot = _insert_markers_on_planes(s_comb_bot, radius_px=plot_radius, multi_mark_horizontal=horiz_triplet)

    # Optional top if vert_span is provided
    s_comb_top = None
    if center_top is not None:
        s_left_t   = S_at(left_top)
        s_center_t = S_at(center_top)
        s_right_t  = S_at(right_top)
        s_comb_top = _consolidate(s_left_t, s_center_t, s_right_t)

        horiz_triplet_t = [
            (s_left_t.horizontal.left,   s_left_t.horizontal.top,   plot_radius),
            (s_center_t.horizontal.left, s_center_t.horizontal.top, plot_radius),
            (s_right_t.horizontal.left,  s_right_t.horizontal.top,  plot_radius),
        ]
        s_comb_top = _insert_markers_on_planes(s_comb_top, radius_px=plot_radius, multi_mark_horizontal=horiz_triplet_t)

    # --- Plotting helpers
    def _imshow(ax, pil_img, title=None):
        if pil_img is None:
            ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            ax.axis("off")
            if title:
                ax.set_title(title)
            return
        ax.imshow(pil_img)
        ax.axis("off")
        if title:
            ax.set_title(title)

    # Bottom figure: 2 rows × 3 columns (coronal on top, horizontal below)
    fig = plt.figure(figsize=(18, 6))
    axes = [plt.subplot(2, 3, i) for i in range(1, 7)]

    for j, entry in enumerate(s_comb_bot.entries):
        _imshow(axes[j], entry.coronal.image_marked or entry.coronal.image)
    for j, entry in enumerate(s_comb_bot.entries):
        _imshow(axes[3 + j], entry.horizontal.image_marked or entry.horizontal.image)

    axes[1].set_title(f"Bottom Electrode Locations {angle}°", fontsize=12)

    # Top figure if required
    if s_comb_top is not None:
        fig2 = plt.figure(figsize=(18, 6))
        axes2 = [plt.subplot(2, 3, i) for i in range(1, 7)]
        for j, entry in enumerate(s_comb_top.entries):
            _imshow(axes2[j], entry.coronal.image_marked or entry.coronal.image)
        for j, entry in enumerate(s_comb_top.entries):
            _imshow(axes2[3 + j], entry.horizontal.image_marked or entry.horizontal.image)
        axes2[1].set_title(f"Top Electrode Locations {angle}°", fontsize=12)

    plt.tight_layout()
    return s_comb_bot, s_comb_top