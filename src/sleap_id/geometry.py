"""v1's crop method, ported unchanged in behaviour.

Source: src/sleap_idtracker/build_id_images.py and rotate_boxes.py (v1, the
version used for the oline run). Every number this module computes -- the
rotation, the crop size, the hull, the mask -- is computed the way v1 computes
it, so the crops are pixel-identical to v1's when the input frames are.

What changed is only the plumbing:
  * v1 read `config` module globals; here the settings are passed in.
  * v1 discovered "the hull collapses after int32 rounding" only while writing
    (and trimmed the file afterwards). That check depends on keypoints alone,
    so it runs here, in step 0, and those instances simply get no crop_row.

The method, per instance:
  1. Rotate about CENTROID_NODE so TOP_NODE (or TOP_NODE_FALLBACK when TOP_NODE
     is missing) sits directly above it.
  2. Cut a W x H window with the centroid at the horizontal centre and `cy`
     pixels from the top. W, H, cy come from population statistics over every
     instance in the file, and are scaled down to fit MAX_CROP_AREA.
  3. Mask everything outside the body outline: the convex hull of
     SURROUNDING_KEYPOINTS when all are present (cv2.convexHull), otherwise a
     Graham scan over every surviving surrounding + bounded keypoint.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# crop_status codes, stored per instance in selection.h5.
OK = 0
NO_CENTROID = 1           # CENTROID_NODE missing
NO_TOP_NODE = 2           # TOP_NODE missing and TOP_NODE_FALLBACK missing/unset
INSUFFICIENT_KEYPOINTS = 3  # fewer than min_hull_keypoints outline points
DEGENERATE_HULL = 4       # enough points, but collinear/coincident
DEGENERATE_AFTER_ROUNDING = 5  # hull collapses once scaled and rounded to pixels
STATUS_NAMES = {
    OK: "ok",
    NO_CENTROID: "no_centroid",
    NO_TOP_NODE: "no_top_node",
    INSUFFICIENT_KEYPOINTS: "insufficient_keypoints",
    DEGENERATE_HULL: "degenerate_hull",
    DEGENERATE_AFTER_ROUNDING: "degenerate_after_rounding",
}


class CropGeometryError(ValueError):
    """This instance cannot produce a crop."""


class InsufficientKeypointsError(CropGeometryError):
    """Fewer than min_hull_keypoints finite keypoints to build a hull from."""


class DegenerateHullError(CropGeometryError):
    """Enough points, but no polygon: they are collinear or coincident."""


def heading_angle_deg(top_xy: np.ndarray, centroid_xy: np.ndarray) -> float:
    """Angle to hand cv2 so that `top` ends up directly above `centroid`.

    cv2.getRotationMatrix2D(c, a, 1) maps an offset (dx, dy) from c to
    x' = cos(a)dx + sin(a)dy, y' = -sin(a)dx + cos(a)dy. Requiring x' = 0 and
    y' < 0 (above, since +y is down) gives a = atan2(dx, -dy).
    """
    dx, dy = top_xy - centroid_xy
    return float(np.degrees(np.arctan2(dx, -dy)))


def rotation_matrix(pts, i_top, i_cen, i_top_fb):
    """-> (M, used_fallback, status). M is None when the instance can't be aligned."""
    cen = pts[i_cen]
    if not np.isfinite(cen).all():
        return None, False, NO_CENTROID
    top = pts[i_top]
    used_fallback = False
    if not np.isfinite(top).all():
        if i_top_fb is None:
            return None, False, NO_TOP_NODE
        top = pts[i_top_fb]
        if not np.isfinite(top).all():
            return None, False, NO_TOP_NODE
        used_fallback = True
    M = cv2.getRotationMatrix2D((float(cen[0]), float(cen[1])),
                                heading_angle_deg(top, cen), 1.0)
    return M, used_fallback, OK


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (M @ np.c_[pts, np.ones(len(pts))].T).T


def graham_scan(points: np.ndarray) -> np.ndarray:
    """Convex hull by Graham scan. Returns the hull vertices, (k, 2).

    Lowest point (lowest y, then x) as pivot; sort the rest by angle, ties by
    distance; sweep with a stack, popping right turns and collinear points.
    """
    P = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(P) < 3:
        raise DegenerateHullError(f"convex hull needs 3 distinct points, got {len(P)}")
    pivot = P[np.lexsort((P[:, 0], P[:, 1]))[0]]
    rest = P[~np.all(P == pivot, axis=1)]
    if len(rest) < 2:
        raise DegenerateHullError(f"convex hull needs 3 distinct points, got {len(P)}")
    d = rest - pivot
    order = np.lexsort((np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])))
    ordered = rest[order]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    stack = [pivot, ordered[0]]
    for pt in ordered[1:]:
        while len(stack) > 1 and cross(stack[-2], stack[-1], pt) <= 0:
            stack.pop()
        stack.append(pt)
    hull = np.asarray(stack)
    if len(hull) < 3:
        raise DegenerateHullError(f"all {len(P)} points are collinear; no polygon exists")
    return hull


def hull_points(pts, i_surround, i_bounded, min_keypoints):
    """The keypoints forming one instance's outline, and whether we fell back.

    Primary: SURROUNDING_KEYPOINTS alone, when every one is present.
    Fallback: every available surrounding + bounded keypoint.
    """
    surround = pts[i_surround]
    if np.isfinite(surround).all():
        src, fallback = surround, False
    else:
        both = pts[list(i_surround) + list(i_bounded)]
        src, fallback = both[np.isfinite(both).all(axis=1)], True
    if len(src) < min_keypoints:
        raise InsufficientKeypointsError(
            f"{len(src)} finite outline keypoints; at least {min_keypoints} needed")
    return src, fallback


def node_indices(cfg: dict, node_names: list[str]) -> dict:
    ix = node_names.index
    return dict(
        i_top=ix(cfg["top_node"]),
        i_cen=ix(cfg["centroid_node"]),
        i_bot=ix(cfg["bottom_node"]),
        i_top_fb=None if cfg["top_node_fallback"] is None else ix(cfg["top_node_fallback"]),
        i_surround=[ix(n) for n in cfg["surrounding_keypoints"]],
        i_bounded=[ix(n) for n in cfg["bounded_keypoints"]],
        min_keypoints=cfg["min_hull_keypoints"],
    )


def collect_geometry(xy: np.ndarray, cfg: dict, node_names: list[str]):
    """Pass 1 of v1, over every instance: crop size, per-instance affine, status.

    Returns (geo, M, status, top_fallback, hull_fallback):
      geo           crop geometry shared by every crop (W, H, cy, scale, ...)
      M             (N, 2, 3) rotation per instance (NaN where status != OK)
      status        (N,) int8 crop_status code
      top_fallback  (N,) bool, aligned on top_node_fallback
      hull_fallback (N,) bool, outline rebuilt by Graham scan
    """
    ni = node_indices(cfg, node_names)
    n = len(xy)
    M_all = np.full((n, 2, 3), np.nan)
    status = np.zeros(n, np.int8)
    top_fb = np.zeros(n, bool)
    hull_fb = np.zeros(n, bool)
    x_spread, y_extent, d_bottom = [], [], []

    for k in range(n):
        pts = xy[k]
        M, used_fb, st = rotation_matrix(pts, ni["i_top"], ni["i_cen"], ni["i_top_fb"])
        if M is None:
            status[k] = st
            continue
        # Hull feasibility on the raw keypoints (affine-invariant), as v1 pass 1.
        try:
            src, hull_fb[k] = hull_points(pts, ni["i_surround"], ni["i_bounded"],
                                          ni["min_keypoints"])
            graham_scan(src)
        except InsufficientKeypointsError:
            status[k] = INSUFFICIENT_KEYPOINTS
            continue
        except DegenerateHullError:
            status[k] = DEGENERATE_HULL
            continue
        M_all[k] = M
        top_fb[k] = used_fb
        ok = np.isfinite(pts).all(axis=1)
        rel = apply(M, pts[ok]) - pts[ni["i_cen"]]
        x_spread.append(rel[:, 0].max() - rel[:, 0].min())
        y_extent.append(rel[:, 1].max() - rel[:, 1].min())
        if np.isfinite(pts[ni["i_bot"]]).all():
            b = (M @ np.r_[pts[ni["i_bot"]], 1.0]) - pts[ni["i_cen"]]
            d_bottom.append(b[1])  # +ve = below the centroid

    if not x_spread:
        raise CropGeometryError("no instance in this file can produce a crop")
    if not d_bottom:
        raise CropGeometryError(
            f"bottom_node {cfg['bottom_node']!r} is missing on every croppable instance")

    stat = np.median if cfg["size_stat"] == "median" else np.mean
    W_raw = int(round(float(stat(np.asarray(x_spread)))))
    H_raw = int(round(float(stat(np.asarray(y_extent)))))
    up_from_bottom = float(np.median(np.asarray(d_bottom)))
    cy_raw = int(round(H_raw - up_from_bottom))

    # Clamp the AREA, scaling both sides by one factor so the shape is kept.
    # Round first; if rounding up overshoots the bound, floor instead.
    max_area = cfg["max_crop_area"]
    scale, W, H, cy = 1.0, W_raw, H_raw, cy_raw
    if max_area is not None and W_raw * H_raw > max_area:
        scale = math.sqrt(max_area / float(W_raw * H_raw))
        W, H = int(round(W_raw * scale)), int(round(H_raw * scale))
        if W * H > max_area:
            W, H = int(W_raw * scale), int(H_raw * scale)
        W, H = max(1, W), max(1, H)
        cy = int(round(cy_raw * scale))

    geo = dict(W=W, H=H, cy=cy, W_raw=W_raw, H_raw=H_raw, cy_raw=cy_raw,
               scale=scale, up_from_bottom=up_from_bottom,
               body_length=H_raw, **ni)

    # v1 found this case only while writing; it depends on keypoints alone.
    for k in np.flatnonzero(status == OK):
        try:
            crop_outline(xy[k], M_all[k], geo)
        except CropGeometryError:
            status[k] = DEGENERATE_AFTER_ROUNDING
            M_all[k] = np.nan
    return geo, M_all, status, top_fb, hull_fb


def crop_affine(pts: np.ndarray, M: np.ndarray, geo: dict) -> np.ndarray:
    """Rotation + translation into the W x H window (+ area scale), as one 2x3."""
    cen = pts[geo["i_cen"]]
    A = M.copy()
    A[0, 2] += geo["W_raw"] / 2.0 - cen[0]  # centroid -> horizontal centre
    A[1, 2] += geo["cy_raw"] - cen[1]       # centroid -> cy_raw from the top edge
    if geo["scale"] != 1.0:
        A = A * geo["scale"]
    return A


def crop_outline(pts: np.ndarray, M: np.ndarray, geo: dict):
    """-> (A, hull) where hull is the int32 polygon in crop coordinates."""
    A = crop_affine(pts, M, geo)
    src, fallback = hull_points(pts, geo["i_surround"], geo["i_bounded"],
                                geo["min_keypoints"])
    xy = apply(A, src)
    hull = (graham_scan(xy) if fallback
            else cv2.convexHull(xy.astype(np.int32))).astype(np.int32)
    if len(hull) < 3:
        raise DegenerateHullError(f"hull has {len(hull)} vertices after rounding")
    return A, hull


def canonical_crop(image: np.ndarray, pts: np.ndarray, M: np.ndarray,
                   geo: dict) -> np.ndarray:
    """One W x H hull-masked crop via a single warpAffine (v1's canonical_crop)."""
    A, hull = crop_outline(pts, M, geo)
    crop = cv2.warpAffine(image, A, (geo["W"], geo["H"]), flags=cv2.INTER_LINEAR)
    mask = np.zeros((geo["H"], geo["W"]), np.uint8)
    cv2.fillConvexPoly(mask, hull.reshape(-1, 1, 2), 255)
    return cv2.bitwise_and(crop, crop, mask=mask)
