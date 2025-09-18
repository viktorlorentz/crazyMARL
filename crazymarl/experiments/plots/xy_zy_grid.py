#!/usr/bin/env python3
import seaborn as sns
import os, sys, glob
import numpy as np
import colorsys
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
sns.set_theme(style="white", palette="muted", font_scale=0.8)

import mpl_fontkit as fk
# 1. Install and register Inter
fk.install("Inter")  
fk.set_font("Inter")  

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
try:
    from scipy.optimize import least_squares
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Ensure we can import the local package when run from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crazymarl.experiments.experiment_loader import Experiment

sns.set_theme(style="white", palette="muted", font_scale=0.8)
plt.rcParams['figure.dpi'] = 300

#font size 8
plt.rcParams['font.size'] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# Width similar to your notebook config
TEXTWIDTH = 2 * 3.4127  # inches


def find_latest_asdf(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    matches.sort()
    return matches[-1]


def load_figure_eight_experiments():
    base = os.path.join(REPO_ROOT, "experiments_data")
    p1 = find_latest_asdf(os.path.join(base, "1_quads_figure_eight_*", "1_quads_figure_eight.crazy.asdf"))
    p2 = find_latest_asdf(os.path.join(base, "2_quads_figure_eight_*", "2_quads_figure_eight.crazy.asdf"))
    p3 = find_latest_asdf(os.path.join(base, "3_quads_figure_eight_*", "3_quads_figure_eight.crazy.asdf"))
    return Experiment(p1), Experiment(p2), Experiment(p3)


def _pad_limits(arr: np.ndarray):
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.5
    return (vmin - pad, vmax + pad)


def _adjust_lightness(rgb, factor: float):
    """Lighten (factor>1) or darken (factor<1) an RGB tuple."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0, min(1, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


def cable_path(p0, p1, cable_length, resolution=0.01):
    """Compute a catenary-like cable path between p0 and p1 of given length.
    Falls back to straight line if length is insufficient.
    """
    p0, p1 = np.array(p0, float), np.array(p1, float)
    straight_dist = np.linalg.norm(p1 - p0)
    if cable_length <= straight_dist or straight_dist < 1e-6:
        t = np.linspace(0, 1, int(np.ceil(max(straight_dist, 1e-6) / resolution)) + 1)
        return np.outer(1 - t, p0) + np.outer(t, p1)

    delta = p1 - p0
    dh = np.linalg.norm(delta[:2])
    e_h = (delta[:2] / dh).tolist() + [0.0] if dh >= 1e-6 else [1.0, 0.0, 0.0]
    S = dh
    z0, z1 = p0[2], p1[2]

    if _HAS_SCIPY:
        def eqs(vars):
            a, u = vars
            return [
                a * (np.cosh((S - u) / a) - np.cosh(u / a)) - (z1 - z0),
                a * (np.sinh((S - u) / a) + np.sinh(u / a)) - cable_length
            ]

        a0 = cable_length**2 / (8 * max(cable_length - S, 1e-6))
        guess = [a0, S / 2]
        bounds = ([1e-6, 0], [np.inf, max(S, 1e-6)])
        sol = least_squares(eqs, guess, bounds=bounds)
        a, u = sol.x

        C = z0 - a * np.cosh(u / a)
        xs = np.linspace(0, S, int(np.ceil(max(S, 1e-6) / resolution)) + 1)
        zs = a * np.cosh((xs - u) / a) + C
    else:
        # fallback: straight-line discretization to approximate a cable
        t = np.linspace(0, 1, int(np.ceil(max(S, 1e-6) / resolution)) + 1)
        points = np.outer(1 - t, p0) + np.outer(t, p1)
        return points

    eh = np.array(e_h)
    points = p0 + np.outer(xs, eh) + np.outer(zs - z0, [0, 0, 1])
    return points


def make_xy_xz_grid(exps, labels, runs=slice(0, 5), width=TEXTWIDTH):
    """Grid of XY (top) and XZ (bottom) plots for multiple experiments."""
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width, width * 0.5),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex='all',  # share X across every axis
        sharey='row',
        constrained_layout=False
    )

    x_limits_per_col = []
    y_vals_all, z_vals_all = [], []
    cmap = plt.get_cmap('viridis')
    v_vals = []

    for exp in exps:
        all_runs = np.arange(exp.num_runs)
        run_idxs = all_runs[runs] if runs is not None else np.array([0])
        completed = np.intersect1d(exp.full_runs, run_idxs)
        if completed.size:
            x_vals = exp.payload_pos[:, completed, 0].ravel()
            y_vals = exp.payload_pos[:, completed, 1].ravel()
            z_vals = exp.payload_pos[:, completed, 2].ravel()
            x_limits_per_col.append(_pad_limits(x_vals))
            y_vals_all.append(y_vals)
            z_vals_all.append(z_vals)
            v = exp.payload_linvel[:, completed, 0, :]
            v_vals.append(np.linalg.norm(v, axis=-1).ravel())
        else:
            x_limits_per_col.append((-1.0, 1.0))
            y_vals_all.append(np.array([0.0]))
            z_vals_all.append(np.array([0.0]))
            v_vals.append(np.array([0.0]))

    # Global limits
    y_all = np.concatenate(y_vals_all) if y_vals_all else np.array([0.0])
    z_all = np.concatenate(z_vals_all) if z_vals_all else np.array([0.0])
    y_limits = _pad_limits(y_all)
    z_limits = _pad_limits(z_all)
    # Heavily increase Z padding
    z_min, z_max = z_limits
    if z_max > z_min:
        span = (z_max - z_min)
        extra = span  # add 100% extra span on both sides ("a lot")
        z_limits = (z_min - extra, z_max + extra)

    # Use a single global X limit so equal aspect keeps Y identical across columns
    x_global = (min(l for l, _ in x_limits_per_col), max(r for _, r in x_limits_per_col))

    v_all = np.concatenate(v_vals) if v_vals else np.array([0.0])
    vmin = float(np.min(v_all))
    # Cap the color scale so that 1.5 already reaches the maximum color
    vmax = 1.5
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for c, (exp, col_label) in enumerate(zip(exps, labels)):
        ax_xy = axes[0, c]
        ax_xz = axes[1, c]
        all_runs = np.arange(exp.num_runs)
        run_idxs = all_runs[runs] if runs is not None else np.array([0])
        completed = np.intersect1d(exp.full_runs, run_idxs)
        if completed.size == 0:
            ax_xy.set_xlim(x_global)
            ax_xz.set_xlim(x_global)
            subtitle = f"Q={exp.num_quads}"
            ax_xy.text(0.98, 0.98, subtitle, transform=ax_xy.transAxes, ha='right', va='top', fontsize=8)
            continue

        traj = exp.trajectory
        if traj is not None and getattr(traj, 'shape', None) and traj.shape[1] == 3:
            ax_xy.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.4, lw=1)
            ax_xz.plot(traj[:, 0], traj[:, 2], color='gray', alpha=0.4, lw=1)

        base_colors = sns.color_palette("muted", n_colors=len(completed))
        rots_flat = getattr(exp, 'agent_rot_flat', None)
        cable_length = exp.env_config.get('cable_length', 0.3)
        quad_w, quad_h = 0.10, 0.02

        for run_i, idx in enumerate(completed):
            base = base_colors[run_i]
            pos = exp.payload_pos[:, idx, :]
            v = exp.payload_linvel[:, idx, 0, :]
            vnorm = np.linalg.norm(v, axis=-1)
            ax_xy.scatter(pos[:, 0], pos[:, 1], c=vnorm, cmap=cmap, norm=norm, s=1, alpha=0.6)
            ax_xz.scatter(pos[:, 0], pos[:, 2], c=vnorm, cmap=cmap, norm=norm, s=1, alpha=0.6)
            start = pos[0]
            ax_xy.scatter(start[0], start[1], s=10, marker='o', color='black', zorder=5)
            ax_xz.scatter(start[0], start[2], s=10, marker='o', color='black', zorder=5)
            for q in range(exp.num_quads):
                quad = exp.quad_pos[:, idx, q, :]
                if q == 0:
                    color = _adjust_lightness(base, 1.2); ls = '--'
                elif q == 1:
                    color = _adjust_lightness(base, 0.8); ls = ':'
                else:
                    color = base; ls = '-.'
                ax_xy.plot(quad[:, 0], quad[:, 1], linestyle=ls, lw=0.5, alpha=0.7, color=color)
                ax_xz.plot(quad[:, 0], quad[:, 2], linestyle=ls, lw=0.5, alpha=0.7, color=color)
                ax_xy.scatter(quad[0, 0], quad[0, 1], marker='x', s=40, color=color, zorder=6)
                cable_pts = cable_path(start, quad[0], cable_length)
                cable_color = _adjust_lightness(color, 0.3)
                ax_xy.plot(cable_pts[:, 0], cable_pts[:, 1], '-', lw=1, color=cable_color, alpha=0.8, zorder=5)
                ax_xz.plot(cable_pts[:, 0], cable_pts[:, 2], '-', lw=1, color=cable_color, alpha=0.8, zorder=5)
                if rots_flat is not None:
                    try:
                        R = rots_flat[0, idx, q, :].reshape(3, 3)
                        wvec = R[:, 0] * quad_w
                        angle_deg_xz = np.degrees(np.arctan2(wvec[2], wvec[0]))
                        x0, z0 = quad[0, 0], quad[0, 2]
                        rect = Rectangle((x0 - quad_w/2, z0 - quad_h/2), quad_w, quad_h,
                                         edgecolor=color, facecolor=color, lw=1, zorder=6)
                        t_mat = Affine2D().rotate_deg_around(x0, z0, angle_deg_xz) + ax_xz.transData
                        rect.set_transform(t_mat)
                        ax_xz.add_patch(rect)
                    except Exception:
                        pass
        ax_xy.set_xlim(x_global)
        ax_xz.set_xlim(x_global)
        subtitle = f"Q={exp.num_quads}"
        ax_xy.text(0.5, 0.98, subtitle, transform=ax_xy.transAxes, ha='center', va='top', fontsize=8)

    # Apply identical Y & Z limits across columns
    for c in range(ncols):
        axes[0, c].set_ylim(y_limits)
        axes[1, c].set_ylim(z_limits)

    # Remove equal aspect to ensure identical physical x span top/bottom
    # (Maintaining global x limits already guarantees same data range.)
    # for r in range(nrows):
    #     for c in range(ncols):
    #         axes[r, c].set_aspect('equal', adjustable='box')

    # Re-apply uniform global X limits to all axes to guarantee identical width
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].set_xlim(x_global)

    for c in range(ncols):
        axes[0, c].set_xlabel("")
    axes[0, 0].set_ylabel('Y [m]')
    for c in range(1, ncols):
        axes[0, c].set_ylabel("")
    for c in range(ncols):
        axes[1, c].set_xlabel('X [m]')
    axes[1, 0].set_ylabel('Z [m]')
    for c in range(1, ncols):
        axes[1, c].set_ylabel("")

    # Make room at bottom
    fig.subplots_adjust(bottom=0.24, left=0.07, right=0.97, top=0.98, wspace=0.05, hspace=0.08)

    # Legend (3 columns) and colorbar side-by-side at bottom
    handles = [
        Line2D([], [], color='gray', lw=1, label='Ref Traj'),
        Line2D([], [], marker='o', lw=0.8, linestyle='None', color='black', label='Payload Start'),
        Line2D([], [], marker='x', linestyle='None', color='gray', label='Quad Start'),
        Line2D([], [], color='gray', lw=1.0, label='Cable', alpha=0.8),
        Line2D([], [], color='gray', lw=0.8, ls='--', label='Quad Path'),
    ]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.07, -0.035), ncol=3, frameon=True)

    cbar_ax = fig.add_axes([0.65, 0.07, 0.3, 0.025])  # x,y,w,h in figure fraction

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    if hasattr(cbar, "solids") and cbar.solids is not None:
        cbar.solids.set_alpha(0.6)
    cbar.set_label('Payload Velocity [m/s]',labelpad=0)

    return fig


def main():
    os.makedirs(os.path.join(REPO_ROOT, 'out'), exist_ok=True)
    one, two, three = load_figure_eight_experiments()
    exps = [one, two, three]
    labels = ['1 quad', '2 quads', '3 quads']

    fig = make_xy_xz_grid(exps, labels, runs=slice(5, 10), width=TEXTWIDTH)
    out_pdf = os.path.join(REPO_ROOT, 'out', 'xy_xz_grid.pdf')
    out_png = os.path.join(REPO_ROOT, 'out', 'xy_xz_grid.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()