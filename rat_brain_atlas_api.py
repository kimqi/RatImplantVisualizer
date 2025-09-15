from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
import io
import math
import requests
import numpy as np

import json

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

import matplotlib.pyplot as plt


# -------------------------
# Low-level API structures
# -------------------------

@dataclass
class PlaneInfo:
    top: int
    left: int
    image_url: str
    image: Optional["Image.Image"] = None
    image_marked: Optional["Image.Image"] = None


@dataclass
class AtlasResponse:
    coronal: PlaneInfo
    sagittal: PlaneInfo
    horizontal: PlaneInfo

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "AtlasResponse":
        def _plane(key: str) -> PlaneInfo:
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

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# API helpers
# -------------------------

API_BASE = "http://labs.gaidi.ca/rat-brain-atlas/api.php"

def atlas_url(ml: float, ap: float, dv: float) -> str:
    return f"{API_BASE}?ml={ml}&ap={ap}&dv={dv}"

def _webread(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def _read_image(url: str) -> Optional["Image.Image"]:
    try:
        data = _webread(url)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def rat_brain_atlas(ml: float, ap: float, dv: float) -> AtlasResponse:
    url = atlas_url(ml, ap, dv)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Unable to complete web request to {url!r}.") from e

    # --- Robust JSON parse ---
    try:
        d = r.json()  # Fast path
    except ValueError:
        # Fallback if server didn't send proper JSON headers
        try:
            d = json.loads(r.content.decode("utf-8", errors="replace"))
        except Exception as e:
            sample = r.content[:200]
            raise RuntimeError(
                "Failed to parse JSON from atlas API. "
                f"First 200 bytes of response: {sample!r}"
            ) from e

    if isinstance(d, dict) and d.get("error"):
        raise RuntimeError(f"Atlas API error: {d.get('error')}")

    S = AtlasResponse.from_json(d)

    # Fetch images
    for plane in (S.coronal, S.sagittal, S.horizontal):
        plane.image = _read_image(plane.image_url)

    if _HAS_PIL:
        for plane in (S.coronal, S.sagittal, S.horizontal):
            if plane.image is not None:
                im = plane.image.copy()
                draw = ImageDraw.Draw(im)
                rpx = 10
                x, y = plane.left, plane.top
                draw.ellipse((x - rpx, y - rpx, x + rpx, y + rpx), fill=(255, 0, 0))
                plane.image_marked = im

    return S


# -------------------------
# Visualization port
# -------------------------

@dataclass
class CombinedAtlas:
    """Container mimicking the MATLAB combined struct list (left, center, right)."""
    entries: List[AtlasResponse]

def _consolidate(Sleft: AtlasResponse, Scenter: AtlasResponse, Sright: AtlasResponse) -> CombinedAtlas:
    return CombinedAtlas(entries=[Sleft, Scenter, Sright])

def _insert_markers_on_planes(
    S: CombinedAtlas,
    radius_px: int,
    multi_mark_horizontal: Optional[List[Tuple[int, int, int]]] = None,
) -> CombinedAtlas:
    """
    Adds red filled circles:
      - Coronal: mark each entry's own (left, top)
      - Horizontal: mark all provided coords on every entry (like MATLAB's combined insert)
    """
    if not _HAS_PIL:
        return S  # no-op if Pillow not available

    for entry in S.entries:
        # Coronal: one marker per entry (its own)
        if entry.coronal.image is not None:
            im = entry.coronal.image.copy()
            d = ImageDraw.Draw(im)
            x, y = entry.coronal.left, entry.coronal.top
            d.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=(255, 0, 0))
            entry.coronal.image_marked = im

        # Horizontal: either one marker (own) or multiple combined markers
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
    return S


def _sind(deg: float) -> float:
    return math.sin(math.radians(deg))

def _cosd(deg: float) -> float:
    return math.cos(math.radians(deg))


def plot_implant_coords(
    AP: float,
    ML: float,
    DV: float,
    angle: float,
    *,
    span: float = 750.0,       # microns
    skull_t: float = 500.0,    # microns (your MATLAB default noted)
    vert_span: float = float("nan"),  # microns
    plot_radius: int = 5       # pixels for drawn circle
) -> Tuple[CombinedAtlas, Optional[CombinedAtlas]]:
    """
    Python port of plot_implant_coords. Returns (Scomb_bot, Scomb_top).

    Units:
      - AP/ML/DV: millimeters (like MATLAB)
      - span, skull_t, vert_span: microns
    """
    span_mm = span / 1000.0
    skull_mm = skull_t / 1000.0
    vert_mm = (vert_span / 1000.0) if not math.isnan(vert_span) else float("nan")

    # Bottom (tip) coordinates in mm
    center_bot = np.array([AP, ML, DV + skull_mm])
    left_bot   = np.array([AP - _sind(angle) * span_mm / 2.0,
                           ML - _cosd(angle) * span_mm / 2.0,
                           DV + skull_mm])
    right_bot  = np.array([AP + _sind(angle) * span_mm / 2.0,
                           ML + _cosd(angle) * span_mm / 2.0,
                           DV + skull_mm])

    # Optional top coordinates (if vert_span provided)
    if not math.isnan(vert_span):
        center_top = center_bot - np.array([0.0, 0.0, vert_mm])
        left_top   = left_bot   - np.array([0.0, 0.0, vert_mm])
        right_top  = right_bot  - np.array([0.0, 0.0, vert_mm])
    else:
        center_top = left_top = right_top = None

    # --- Fetch atlas for each target (ML, AP, DV order for API) ---
    def S_at(coords) -> AtlasResponse:
        ap, ml, dv = float(coords[0]), float(coords[1]), float(coords[2])
        return rat_brain_atlas(ml=ml, ap=ap, dv=dv)

    Sleft   = S_at(left_bot)
    Scenter = S_at(center_bot)
    Sright  = S_at(right_bot)

    # Consolidate and add markers
    Scomb_bot = _consolidate(Sleft, Scenter, Sright)

    # For horizontal images, MATLAB inserts all three markers on each image
    horiz_triplet = [
        (Sleft.horizontal.left,   Sleft.horizontal.top,   plot_radius),
        (Scenter.horizontal.left, Scenter.horizontal.top, plot_radius),
        (Sright.horizontal.left,  Sright.horizontal.top,  plot_radius),
    ]
    Scomb_bot = _insert_markers_on_planes(Scomb_bot, radius_px=plot_radius, multi_mark_horizontal=horiz_triplet)

    # Optional top
    Scomb_top = None
    if center_top is not None:
        Sleft_t   = S_at(left_top)
        Scenter_t = S_at(center_top)
        Sright_t  = S_at(right_top)
        Scomb_top = _consolidate(Sleft_t, Scenter_t, Sright_t)

        horiz_triplet_t = [
            (Sleft_t.horizontal.left,   Sleft_t.horizontal.top,   plot_radius),
            (Scenter_t.horizontal.left, Scenter_t.horizontal.top, plot_radius),
            (Sright_t.horizontal.left,  Sright_t.horizontal.top,  plot_radius),
        ]
        Scomb_top = _insert_markers_on_planes(Scomb_top, radius_px=plot_radius, multi_mark_horizontal=horiz_triplet_t)

    # --- Plot like MATLAB (2 rows x 3 cols): top row = coronal, bottom = horizontal ---
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

    # Bottom (tip) figure
    fig = plt.figure(figsize=(18, 6))
    axes = [plt.subplot(2, 3, i) for i in range(1, 7)]

    for j, entry in enumerate(Scomb_bot.entries):
        _imshow(axes[j], entry.coronal.image_marked or entry.coronal.image)
    for j, entry in enumerate(Scomb_bot.entries):
        _imshow(axes[3 + j], entry.horizontal.image_marked or entry.horizontal.image)

    axes[1].set_title(f"Bottom Electrode Locations {angle}°", fontsize=12)

    # Top (if requested)
    if Scomb_top is not None:
        fig2 = plt.figure(figsize=(18, 6))
        axes2 = [plt.subplot(2, 3, i) for i in range(1, 7)]
        for j, entry in enumerate(Scomb_top.entries):
            _imshow(axes2[j], entry.coronal.image_marked or entry.coronal.image)
        for j, entry in enumerate(Scomb_top.entries):
            _imshow(axes2[3 + j], entry.horizontal.image_marked or entry.horizontal.image)
        axes2[1].set_title(f"Top Electrode Locations {angle}°", fontsize=12)

    plt.tight_layout()
    return Scomb_bot, Scomb_top