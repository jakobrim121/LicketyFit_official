# This code was produced by ChatGPT 5 based on the following prompts:

# PROMPT #1
###########
# A large water Cherenkov detector consists of thousands of photo-sensors located on the walls and endcaps of a
# large cylindrical can filled with ultra pure water. The sensors are pointed inward to detect Cherenkov photons
# produced by relativistic charged particles travelling in the water. The pattern, timing, and amplitude of the
# light signals are used to infer the paths of the charged particles. What approaches can show, in a 2D representation,
# the pattern of the light signals from a single charged particle observed by this detector that is insightful
# and visually appealing?

# Response:
###########
# Here are 2D views that actually help you *see* one particle’s light pattern, not just decorate it.
# Each keeps the detector geometry honest and encodes timing/charge in ways that expose the track and outliers.
#
# # 1) Cylinder unwrap (“φ–z map”)
#
# * **What**: Flatten the barrel to a rectangle: x = sensor azimuth φ (−π→π), y = height z. Put the two endcaps
# as separate disks (see #2) or append them to top/bottom edges.
# * **Why**: A single straight track gives a **sinusoidal** Cherenkov ridge on the barrel (cone ∩ cylinder).
# The sinusoid’s phase/slope pins down the 3D direction; its amplitude relates to track tilt.
# * **Encoding**: dot size ∝ √(PE), hue = **time residual** (t_hit − t_pred) w.r.t. your best-fit vertex/track,
# lightness = total charge. Overlay the analytic cone–surface intersection from the fitted track so residuals pop.
#
# # 2) Endcap polar rings (“R–φ disks”)
#
# * **What**: Two polar plots for the upstream/downstream endcaps (r = radius, φ = azimuth).
# * **Why**: Cone ∩ plane → **circle/ellipse** on each cap. You immediately see whether the vertex is
# upstream/downstream and how centered the ring is.
# * **Encoding**: same as above; optionally overlay the predicted circle from the fit and draw **isochrones**
# (expected arrival-time contours) to judge vertex timing.
#
# # 3) Unified surface projection (“unrolled + caps” panel)
#
# * **What**: One canvas with the unrolled barrel in the middle and the two endcap disks attached at the appropriate z.
# Think of it as the detector’s “skin” laid flat.
# * **Why**: Lets you trace a single Cherenkov cone continuously across wall→cap transitions without changing
# coordinate systems. Great for QA and outreach.
# * **Encoding**: continuous colorbar for time residual; thin gray geodesics every Δφ, Δz for orientation.
#
# # 4) Angle-of-arrival map (“photon sky”)
#
# * **What**: For a seed vertex, convert each sensor direction to **track-centric spherical angles** (θ′, φ′)
# where +z′ is the reconstructed track direction. Plot on an equal-area map (e.g., Mollweide or Lambert azimuthal).
# * **Why**: The Cherenkov cone collapses to a **ring near θ′ ≈ θC** (≈42° in water). Mis-reconstructed vertices smear
# the ring radially; multiple tracks split it.
# * **Encoding**: radius from θC shows angular residual; color = time residual. Tight ring → good vertex/β≈1 consistency.
#
# # 5) Hough/Radon spaces (diagnostic “parameter maps”)
#
# * **What**:
#
#   * **Barrel**: Hough in (k, φ0) for sinusoids y = A sin(φ + φ0) + z0 (or use a Radon transform tailored to the
#   cylinder-cone curve).
#   * **Endcaps**: Standard circle Hough (center_x, center_y, R).
# * **Why**: Puts ring finding into a **peak-finding** problem. Single clean peak → one track; multiple
# peaks → overlaps/decays.
# * **How to show**: A small inset heatmap of accumulator space next to the hit view; mark the chosen maximum.
#
# # 6) Space–time residual carpets
#
# * **What**:
#
#   * **φ–t_res** for the barrel at fixed z-bands (or all sensors unrolled), and **r–t_res** for each cap.
# * **Why**: For a correct track/vertex, time residuals line up along **straight bands** (isochrones).
# Curvature/bows indicate vertex bias or scattering; fat tails flag dark noise/afterpulses.
# * **Encoding**: density as grayscale, median and ±1σ bands overlaid; outliers marked.
#
# # 7) Along-ring 1D profiles
#
# * **What**: Project each hit to the **closest point on the predicted ring/curve**, then plot:
#
#   * charge vs ring-arc-length,
#   * time residual vs ring-arc-length.
# * **Why**: Collapses 2D clutter into a **quality control strip**. Hot spots show Michel electrons or localized
# reflections; phase shifts expose timing calibration offsets.
#
# # 8) Likelihood/back-projection heatmaps
#
# * **What**: For a fixed track direction, back-project each hit’s timing to a **vertex likelihood** on a 2D
# slice (e.g., r–z through the cylinder axis) or make a 2D **ring-likelihood** on the detector surface.
# * **Why**: Produces a single smooth map where the maximum is the vertex or ring. Intuitive for non-experts;
# useful for debugging fits.
#
# # 9) HEALPix-style detector map
#
# * **What**: Treat each sensor normal as a point on the unit sphere; bin hits into an equal-area tessellation and
# show a **charge/time heatmap**.
# * **Why**: Geometry-agnostic summary that compares different detectors directly; the Cherenkov ring is a band at
# fixed polar angle relative to direction.
#
# # 10) Aesthetic but honest touches (matter for legibility)
#
# * Use **perceptually uniform** colormaps for timing (e.g., cyclic map for phase/wrap; linear for residuals).
# Keep charge in lightness or size, not saturated color.
# * Show **predicted-curve overlays** and a **single residual colorbar**; annotate θC and vertex (with uncertainty
# ellipse) where relevant.
# * Down-weight saturation for hits with large time uncertainty; outline sensors with thin strokes for dense regions.
# * For publications: anti-aliased small markers, √PE sizing to tame dynamic range, and subtle KDE smoothing that
# doesn’t move peaks.
#
# # Picking the right pair (practical combos)
#
# * **Reconstruction dev**: #1 (φ–z) + #5 (Hough insets) + residual colors.
# * **Vertex tuning**: #6 (carpets) + #8 (likelihood slice).
# * **Talk/outreach**: #3 (unified surface) + #2 (cap disks) with clean overlays.
# * **Ambiguous events** (multi-ring): #4 (“photon sky”) + #7 (along-ring profiles).
#
# If you want, I can generate a minimal, publication-ready Matplotlib template that takes sensor (x,y,z), normals, and
# per-hit (t, charge), and emits: (a) φ–z unroll with predicted intersection and residual coloring, (b) twin endcap
# polar plots, and (c) an along-ring profile.

# PROMPT #2
###########
# We describe the cylindrical can as having its axis along the y direction, and the azimuthal angle, phi,
# is 0 along the z direction. The locations of the sensors are specified as points with Cartesian
# coordinates (x,y,z) which are near, but not exactly on the cylinder of radius, r, and the top and bottom
# endcaps are located at yt and yb, with yt > yb. For purposes of making the display, points with yb < y < yt
# can be moved onto the cylindrical surface by transforming to cylindrical coordinates and then setting the
# radius coordinate to r. Points with y < yb, can be moved to the bottom endcap surface by setting y = yb.
# Points with y > yt can be moved to the top endcap surface by setting y = yt.
#
# Please prepare separate Matplotlib templates for the following options you presented. In each case, the template
# should accept an list of Cartesian points to represent sensor locations, and one or more lists containing the
# numerical values to show via hue, brightness, or size of dot for each sensor. There should be the option to
# provide a list of Cartesian points to use to draw a curve, such as the edge of a Cherenkov ring to overlay on
# the same plot. There should be the option to allow text to be drawn at specified Cartesian points.
#
# #3: As specified, with the option to specify the azimuthal angle that cuts the cylindrical wall.
#
# #4: The seed vertex and direction is specified by Cartesian coordinates for the start of the track (x0,y0,z0)
# and the unit vector direction (cx,cy,cz).
#
# #5: As specified.

# water_cerenkov_2d_templates.py
# Matplotlib templates for options #3 (unified surface), #4 (photon sky),
# and #5 (Hough diagnostics) for a cylindrical water Cherenkov detector.
#
# Conventions:
# - Cylinder axis is y. Azimuth φ is defined with φ=0 along +z, φ = atan2(x, z).
# - Sensors are projected to detector surfaces before plotting:
#     yb < y < yt  → clamp to cylindrical wall (set radius = r, preserve φ)
#     y ≤ yb       → move to bottom cap (y = yb)
#     y ≥ yt       → move to top cap    (y = yt)
# - “Brightness” is mapped to marker alpha; size maps to marker area.
# - Overlay curve points and text annotations are transformed consistently.
#
# Dependencies: numpy, matplotlib (no seaborn).
# Each function draws a single figure and returns it.

from typing import Optional, Sequence, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

#cmap = 'viridis'
cmap = 'plasma'
#cmap = 'inferno'

# -----------------------------
# Geometry + styling helpers
# -----------------------------

def _to_numpy_xyz(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return arr

def cartesian_to_cylindrical_phi(points: np.ndarray) -> np.ndarray:
    """φ=atan2(x, z) with φ=0 along +z, range (-π, π]."""
    x, _, z = points.T
    return np.arctan2(x, z)

def project_to_detector_surfaces(
    points: Sequence[Sequence[float]], r: float, yb: float, yt: float
) -> np.ndarray:
    """
    Move points to cylinder or caps:
      - y in (yb, yt): set radius to r in x–z plane (preserve φ)
      - y <= yb: y = yb  (bottom cap)
      - y >= yt: y = yt  (top cap)
    """
    P = _to_numpy_xyz(points).copy()
    x, y, z = P.T
    phi = np.arctan2(x, z)

    # Barrel projection
    m_barrel = (y > yb) & (y < yt)
    x[m_barrel] = r * np.sin(phi[m_barrel])
    z[m_barrel] = r * np.cos(phi[m_barrel])

    # Cap clamps
    m_bottom = y <= yb
    m_top = y >= yt
    y[m_bottom] = yb
    y[m_top] = yt

    return np.column_stack([x, y, z])

def unwrap_phi(phi: np.ndarray, phi_cut: float = 0.0) -> np.ndarray:
    """Shift φ by -phi_cut and wrap to [0, 2π)."""
    two_pi = 2.0 * np.pi
    return (phi - phi_cut) % two_pi

def sensor_style(
    values_color: Optional[Sequence[float]] = None,
    values_size: Optional[Sequence[float]] = None,
    values_alpha: Optional[Sequence[float]] = None,
    size_min: float = 0.5,
    size_max: float = 10.0,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
):
    """
    Normalize style arrays for color, size, and alpha.
    - color: returned as numpy array or None; colormap handled by Matplotlib.
    - size: linear map to [size_min, size_max] (marker area).
    - alpha: linear map to [alpha_min, alpha_max].
    """
    c = None; s = None; a = None
    if values_color is not None:
        c = np.asarray(values_color, dtype=float)

    if values_size is not None:
        vs = np.asarray(values_size, dtype=float)
        vmin, vmax = np.nanmin(vs), np.nanmax(vs)
        if vmax > vmin:
            s = size_min + (vs - vmin) * (size_max - size_min) / (vmax - vmin)
        else:
            s = np.full_like(vs, (size_min + size_max) * 0.5)

    if values_alpha is not None:
        va = np.asarray(values_alpha, dtype=float)
        vmin, vmax = np.nanmin(va), np.nanmax(va)
        if vmax > vmin:
            a = alpha_min + (va - vmin) * (alpha_max - alpha_min) / (vmax - vmin)
        else:
            a = np.full_like(va, (alpha_min + alpha_max) * 0.5)
    return c, s, a

def _apply_alpha(scatter: PathCollection, alpha: Optional[np.ndarray]):
    """Apply per-point alpha by modifying facecolors."""
    if alpha is None:
        return
    facecolors = scatter.get_facecolors()
    if facecolors is None or len(facecolors) == 0:
        plt.gcf().canvas.draw()  # ensure colors exist
        facecolors = scatter.get_facecolors()
    if len(facecolors) == 0:
        return
    if len(facecolors) == 1:
        facecolors = np.tile(facecolors, (scatter.get_offsets().shape[0], 1))
    facecolors[:, 3] = alpha
    scatter.set_facecolors(facecolors)

# -----------------------------
# #3 Unified surface projection
# -----------------------------

def plot_unified_surface(
    sensor_xyz: Sequence[Sequence[float]],
    r: float,
    yb: float,
    yt: float,
    phi_cut: float = 0.0,
    values_color: Optional[Sequence[float]] = None,
    label_color: Optional[str] = "color values",
    vmin_color: Optional[float] = None,
    vmax_color: Optional[float] = None,
    values_size: Optional[Sequence[float]] = None,
    values_alpha: Optional[Sequence[float]] = None,
    overlay_curve_xyzs: Optional[List[Sequence[Sequence[float]]]] = None,
    text_annotations: Optional[List[Tuple[float, float, float, str]]] = None,
    figsize: Tuple[float, float] = (20, 8),
    width_ratios: Tuple[float, float] = (2.82, 1.0),
):
    """
    Draw unrolled barrel (u=r·unwrap(φ−φ_cut), y) plus top/bottom endcap insets.
    φ is computed with φ=atan2(x,z). The azimuthal cut is selectable via phi_cut.
    """
    P = project_to_detector_surfaces(sensor_xyz, r, yb, yt)
    phi = cartesian_to_cylindrical_phi(P)
    u = unwrap_phi(phi, phi_cut=phi_cut)
    u_len = r * u
    y = P[:, 1]

    c, s, a = sensor_style(values_color, values_size, values_alpha)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=width_ratios, height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.4)
    ax_barrel = fig.add_subplot(gs[:, 0], aspect='equal', adjustable='box')
    ax_cap_top = fig.add_subplot(gs[0, 1], aspect='equal', adjustable='box')
    ax_cap_bot = fig.add_subplot(gs[1, 1], aspect='equal', adjustable='box')

    # Barrel
    sc = ax_barrel.scatter(u_len, y, c=c, s=s, cmap=cmap, vmin=vmin_color, vmax=vmax_color)
    _apply_alpha(sc, a)
    if c is not None:
        fig.colorbar(sc, ax=ax_barrel, label=label_color, shrink=0.5)

    ax_barrel.set_xlim(0, 2 * np.pi * r)
    ax_barrel.set_ylim(yb, yt)
    ax_barrel.set_xlabel("Unrolled arc length u = r·(φ − φ_cut) (mm)")
    ax_barrel.set_ylabel("y (mm)")
    #ax_barrel.set_title("Unrolled barrel (φ–y) with endcap insets")
    # invert x-axis for conventional view (inside the tank, looking outward, z into page)
    ax_barrel.invert_xaxis()

    # Barrel overlay curve
    if overlay_curve_xyzs is not None:
        for overlay_curve_xyz in overlay_curve_xyzs:
            C = project_to_detector_surfaces(overlay_curve_xyz, r, yb, yt)
            phi_c = cartesian_to_cylindrical_phi(C)
            u_c = unwrap_phi(phi_c, phi_cut=phi_cut)
            ax_barrel.plot(r * u_c, C[:, 1], linewidth=0.5)

    # Barrel text
    if text_annotations:
        for (xt, yt_, zt, label) in text_annotations:
            if yb < yt_ < yt:
                pt = project_to_detector_surfaces([(xt, yt_, zt)], r, yb, yt)[0]
                phi_t = cartesian_to_cylindrical_phi(pt[None, :])[0]
                u_t = unwrap_phi(np.array([phi_t]), phi_cut=phi_cut)[0]
                ax_barrel.text(r * u_t, pt[1], label, fontsize=7)

    # Endcaps
    cap_top_mask = P[:, 1] >= yt - 1e-9
    cap_bot_mask = P[:, 1] <= yb + 1e-9

    def _plot_cap(ax, pts_mask, title):
        pts = P[pts_mask]
        if pts.shape[0] == 0:
            ax.set_axis_off()
            return
        # Subset styles to match this axis
        local_c = (c[pts_mask] if c is not None else None)
        local_s = (s[pts_mask] if s is not None else None)
        local_a = (a[pts_mask] if a is not None else None)

        sc2 = ax.scatter(pts[:, 0], pts[:, 2], c=local_c, s=local_s, cmap=cmap, vmin=vmin_color, vmax=vmax_color)
        _apply_alpha(sc2, local_a)
        if local_c is not None:
            plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)

        # Cap outline
        theta = np.linspace(0, 2*np.pi, 360)
        ax.plot(r*np.sin(theta), r*np.cos(theta), linewidth=1.0)
        ax.set_xlim(-r*1.05, r*1.05)
        ax.set_ylim(-r*1.05, r*1.05)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(title)

    _plot_cap(ax_cap_top, cap_top_mask, f"Top endcap y={yt:.3f}")
    _plot_cap(ax_cap_bot, cap_bot_mask, f"Bottom endcap y={yb:.3f}")
    # invert axes for conventional views
    ax_cap_top.invert_xaxis()
    ax_cap_top.invert_yaxis()
    ax_cap_bot.invert_xaxis()

    # Endcap overlay curves
    if overlay_curve_xyzs is not None:
        for overlay_curve_xyz in overlay_curve_xyzs:
            C = project_to_detector_surfaces(overlay_curve_xyz, r, yb, yt)
            mtop = C[:, 1] >= yt - 1e-9
            mbot = C[:, 1] <= yb + 1e-9
            if np.any(mtop):
                # sort these in x order to avoid the wrap line
                mtop_sorted_indices = np.argsort(C[mtop, 0])
                mtop = np.where(mtop)[0][mtop_sorted_indices]
                ax_cap_top.plot(C[mtop, 0], C[mtop, 2], linewidth=0.5)
            if np.any(mbot):
                ax_cap_bot.plot(C[mbot, 0], C[mbot, 2], linewidth=0.5)

    # Endcap text
    if text_annotations:
        for (xt, yt_, zt, label) in text_annotations:
            pt = project_to_detector_surfaces([(xt, yt_, zt)], r, yb, yt)[0]
            if abs(pt[1] - yt) < 1e-9:
                ax_cap_top.text(pt[0], pt[2], label, fontsize=7)
            elif abs(pt[1] - yb) < 1e-9:
                ax_cap_bot.text(pt[0], pt[2], label, fontsize=7)

    #fig.suptitle("Unified surface projection (#3)")
    #fig.tight_layout()
    return fig

# -----------------------------
# #4 Angle-of-arrival (“photon sky”)
# -----------------------------

def _orthonormal_basis_from_direction(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given direction c, return orthonormal basis (ex, ey, ez) with ez aligned to c.
    """
    ez = c / np.linalg.norm(c)
    tmp = np.array([0.0, 1.0, 0.0]) if abs(ez[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ex = np.cross(tmp, ez); ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return ex, ey, ez

def lambert_azimuthal_equal_area(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LAEA projection centered at θ=0. θ is polar angle (0..π), φ is azimuth (-π, π].
    Radius of hemisphere maps to √2.
    """
    rho = np.sqrt(2) * np.sin(theta / 2.0)
    X = rho * np.cos(phi)
    Y = rho * np.sin(phi)
    return X, Y

def plot_photon_sky(
    sensor_xyz: Sequence[Sequence[float]],
    vertex_xyz: Tuple[float, float, float],
    direction_c: Tuple[float, float, float],
    values_color: Optional[Sequence[float]] = None,
    label_color: Optional[str] = "color values",
    values_size: Optional[Sequence[float]] = None,
    values_alpha: Optional[Sequence[float]] = None,
    overlay_curve_xyzs: Optional[List[Sequence[Sequence[float]]]] = None,
    text_annotations: Optional[List[Tuple[float, float, float, str]]] = None,
    figsize: Tuple[float, float] = (7, 7),
):
    """
    Track-centric equal-area map. For each sensor, take the unit vector from vertex to sensor,
    express it in the (ex,ey,ez) basis aligned with the input direction, compute (θ',φ'),
    and project with LAEA.
    """
    P = _to_numpy_xyz(sensor_xyz)
    v0 = np.asarray(vertex_xyz, dtype=float)
    cdir = np.asarray(direction_c, dtype=float); cdir /= np.linalg.norm(cdir)
    ex, ey, ez = _orthonormal_basis_from_direction(cdir)

    V = P - v0
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    vx, vy, vz = Vn @ ex, Vn @ ey, Vn @ ez

    theta_p = np.arccos(np.clip(vz, -1.0, 1.0))
    phi_p = np.arctan2(vy, vx)

    X, Y = lambert_azimuthal_equal_area(theta_p, phi_p)
    c, s, a = sensor_style(values_color, values_size, values_alpha)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(X, Y, c=c, s=s, cmap=cmap)
    _apply_alpha(sc, a)
    if c is not None:
        fig.colorbar(sc, ax=ax, label=label_color)

    # Optional guideline at θC≈42° in water
    #theta_c = np.deg2rad(42.0)
    #t = np.linspace(-np.pi, np.pi, 512)
    #Rc = np.sqrt(2) * np.sin(theta_c / 2.0)
    #ax.plot(Rc * np.cos(t), Rc * np.sin(t), linewidth=1.0)

    # Overlay curve points
    if overlay_curve_xyzs is not None:
        for overlay_curve_xyz in overlay_curve_xyzs:
            C = _to_numpy_xyz(overlay_curve_xyz)
            U = C - v0
            Un = U / np.linalg.norm(U, axis=1, keepdims=True)
            ux, uy, uz = Un @ ex, Un @ ey, Un @ ez
            theta_cu = np.arccos(np.clip(uz, -1.0, 1.0))
            phi_cu = np.arctan2(uy, ux)
            Xc, Yc = lambert_azimuthal_equal_area(theta_cu, phi_cu)
            ax.plot(Xc, Yc, linewidth=0.5)

    # Text annotations
    if text_annotations:
        for (xt, yt, zt, label) in text_annotations:
            pt = np.array([xt, yt, zt], dtype=float)
            u = pt - v0
            un = u / np.linalg.norm(u)
            ux, uy, uz = un @ ex, un @ ey, un @ ez
            theta_t = np.arccos(np.clip(uz, -1.0, 1.0))
            phi_t = np.arctan2(uy, ux)
            Xt, Yt = lambert_azimuthal_equal_area(np.array([theta_t]), np.array([phi_t]))
            ax.text(Xt[0], Yt[0], label, fontsize=7)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("LAEA X (track-centric)")
    ax.set_ylabel("LAEA Y (track-centric)")
    ax.set_title("Angle-of-arrival map (#4) — Lambert azimuthal equal-area")
    Rmax = np.sqrt(2) * 1.01
    ax.set_xlim(-Rmax, Rmax)
    ax.set_ylim(-Rmax, Rmax)
    fig.tight_layout()
    return fig

# -----------------------------
# #5 Hough/Radon diagnostic spaces
# -----------------------------

def hough_circle_endcap(
    cap_points_xz: np.ndarray,
    r_search: Tuple[float, float],
    n_center: int = 64,
    n_radius: int = 64,
    x_bounds: Optional[Tuple[float, float]] = None,
    z_bounds: Optional[Tuple[float, float]] = None,
):
    """
    Simple circle Hough accumulator for endcap hits in the cap plane (x,z).
    Returns (H, xgrid, zgrid, rgrid) where H has shape (n_center, n_center, n_radius).
    """
    if cap_points_xz.size == 0:
        raise ValueError("No points provided for endcap Hough.")

    X, Z = cap_points_xz[:, 0], cap_points_xz[:, 1]
    if x_bounds is None:
        x_bounds = (np.min(X), np.max(X))
    if z_bounds is None:
        z_bounds = (np.min(Z), np.max(Z))

    xgrid = np.linspace(x_bounds[0], x_bounds[1], n_center)
    zgrid = np.linspace(z_bounds[0], z_bounds[1], n_center)
    rgrid = np.linspace(r_search[0], r_search[1], n_radius)

    H = np.zeros((n_center, n_center, n_radius), dtype=np.uint32)

    # O(N * n_center^2) voting (template clarity over speed)
    for xi, zi in zip(X, Z):
        dx = xi - xgrid[None, :]          # (1, n_center)
        dz = zi - zgrid[:, None]          # (n_center, 1)
        dist = np.sqrt(dx**2 + dz**2)     # (n_center, n_center)
        # nearest radius bin
        ridx = np.clip(
            np.rint((dist - rgrid[0]) / (rgrid[1] - rgrid[0])).astype(int),
            0, len(rgrid) - 1
        )
        for iz in range(n_center):
            for ix in range(n_center):
                H[iz, ix, ridx[iz, ix]] += 1

    return H, xgrid, zgrid, rgrid

def plot_hough_circle_endcap(
    sensor_xyz: Sequence[Sequence[float]],
    r: float,
    y_plane: float,
    values_color: Optional[Sequence[float]] = None,
    values_size: Optional[Sequence[float]] = None,
    values_alpha: Optional[Sequence[float]] = None,
    r_search: Tuple[float, float] = (0.2, 1.2),
    n_center: int = 64,
    n_radius: int = 64,
    figsize: Tuple[float, float] = (10, 4),
):
    """
    Plot endcap hits and a circle-Hough accumulator (max over R).
    y_plane should be either yb or yt. Only points on that cap after projection are used.
    """
    # Project all sensors to the chosen cap plane
    P = project_to_detector_surfaces(sensor_xyz, r, y_plane, y_plane)
    mask = np.isclose(P[:, 1], y_plane, atol=1e-9)
    XZ = P[mask][:, [0, 2]]
    if XZ.shape[0] == 0:
        raise ValueError("No points on the specified endcap after projection.")

    # Style arrays, subset to this cap
    c, s, a = sensor_style(values_color, values_size, values_alpha)
    local_c = (c[mask] if c is not None else None)
    local_s = (s[mask] if s is not None else None)
    local_a = (a[mask] if a is not None else None)

    # Default radius search relative to detector r
    if r_search == (0.2, 1.2):
        r_search = (0.05 * r, 1.05 * r)

    # Hough
    H, xgrid, zgrid, rgrid = hough_circle_endcap(
        XZ, r_search, n_center=n_center, n_radius=n_radius
    )

    # Display
    fig, (ax_points, ax_accum) = plt.subplots(1, 2, figsize=figsize)

    sc = ax_points.scatter(XZ[:, 0], XZ[:, 1], c=local_c, s=local_s, cmap=cmap)
    _apply_alpha(sc, local_a)
    if local_c is not None:
        fig.colorbar(sc, ax=ax_points, label="color values")

    theta = np.linspace(0, 2 * np.pi, 360)
    ax_points.plot(r * np.sin(theta), r * np.cos(theta), linewidth=1.0)
    ax_points.set_aspect('equal', adjustable='box')
    ax_points.set_xlabel("x (cap plane)")
    ax_points.set_ylabel("z (cap plane)")
    ax_points.set_title(f"Endcap hits at y={y_plane:.3f}")

    Hmax = H.max(axis=2)
    im = ax_accum.imshow(
        Hmax,
        origin='lower',
        extent=[xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]],
        aspect='equal'
    )
    fig.colorbar(im, ax=ax_accum, label="votes")
    ax_accum.set_xlabel("center x")
    ax_accum.set_ylabel("center z")
    ax_accum.set_title("Circle Hough accumulator (max over R)")

    fig.suptitle("Endcap circle Hough (#5)")
    fig.tight_layout()
    return fig, (H, xgrid, zgrid, rgrid)

def plot_hough_barrel_sinusoid(
    sensor_xyz: Sequence[Sequence[float]],
    r: float,
    yb: float,
    yt: float,
    phi_cut: float = 0.0,
    A_grid: Optional[np.ndarray] = None,
    n_phi0: int = 90,
    n_y0: int = 90,
    figsize: Tuple[float, float] = (7, 6),
):
    """
    Coarse sinusoid Hough on unrolled barrel hits.
    Model: y ≈ A * sin(φ + φ0) + y0, with φ from unwrap(φ−φ_cut).
    Votes into (A, φ0, y0) and shows two max-projection heatmaps.
    """
    P = project_to_detector_surfaces(sensor_xyz, r, yb, yt)
    y = P[:, 1]
    phi = cartesian_to_cylindrical_phi(P)

    # Keep only barrel hits
    mask = (y > yb) & (y < yt)
    if not np.any(mask):
        raise ValueError("No barrel hits after projection.")
    yb_hits = y[mask]
    phi_hits = unwrap_phi(phi[mask], phi_cut=phi_cut)  # radians in [0, 2π)

    # Parameter grids
    if A_grid is None:
        A_grid = np.linspace(0.0, 0.6 * (yt - yb), 50)
    phi0_grid = np.linspace(-np.pi, np.pi, n_phi0, endpoint=False)
    y0_grid = np.linspace(yb, yt, n_y0)
    dy0 = y0_grid[1] - y0_grid[0]

    H = np.zeros((len(A_grid), len(phi0_grid), len(y0_grid)), dtype=np.uint32)

    # Voting: for each A, compute y0_i(φ0) = y_i − A sin(φ_i + φ0) and bin
    for ia, A in enumerate(A_grid):
        S = np.sin(phi_hits[None, :] + phi0_grid[:, None])  # (n_phi0, n_hits)
        Y0 = yb_hits[None, :] - A * S                      # (n_phi0, n_hits)
        idx = np.clip(np.floor((Y0 - y0_grid[0]) / dy0).astype(int), 0, len(y0_grid) - 1)
        for k in range(len(phi0_grid)):
            bincount = np.bincount(idx[k], minlength=len(y0_grid))
            H[ia, k, :] += bincount.astype(np.uint32)

    # Display: two max-projections
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    im1 = ax1.imshow(
        H.max(axis=0),  # (φ0, y0)
        origin='lower',
        aspect='auto',
        extent=[y0_grid[0], y0_grid[-1], phi0_grid[0], phi0_grid[-1]]
    )
    fig.colorbar(im1, ax=ax1, label="votes")
    ax1.set_xlabel("y0")
    ax1.set_ylabel("φ0")
    ax1.set_title("Accumulator max over A")

    im2 = ax2.imshow(
        H.max(axis=1),  # (A, y0)
        origin='lower',
        aspect='auto',
        extent=[y0_grid[0], y0_grid[-1], A_grid[0], A_grid[-1]]
    )
    fig.colorbar(im2, ax=ax2, label="votes")
    ax2.set_xlabel("y0")
    ax2.set_ylabel("A")
    ax2.set_title("Accumulator max over φ0")

    fig.suptitle("Barrel sinusoid Hough (#5)")
    fig.tight_layout()
    return fig, (H, A_grid, phi0_grid, y0_grid)
