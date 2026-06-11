"""
make_all_figures.py
===================

Self-contained reference listing of ALL the code that produces Figures 1-7
of the paper "Optimal Charger Location for Electric Service Operation
Vessels in Offshore Wind Farms".

This file collects, verbatim, the figure-generating code that lives in the
project's three plotting scripts, organised by figure number:

    Figure 1  layout map ............ from plot_layout_map.py
    Figure 2  cost vs K ............. from plot.py  (fig_dogger_costs)
    Figure 3  break-even ........... from plot_timecost.py
    Figure 4  optimal route ........ from plot.py  (fig_dogger_main)
    Figure 5  robust map ........... from plot.py  (fig_multiday_map)
    Figure 6  distribution+ranking . from plot.py  (fig_multiday_compare)
    Figure 7  per-scenario ......... from plot.py  (fig_multiday_scenarios)

Dependencies: numpy, matplotlib, scipy. The figure functions read CSVs
produced by the optimisation pipeline (solve.py); the expected files are
listed in load_data() / load_layout() below.

Usage (with the data/ directory next to this file):
    python make_all_figures.py --data ./data --out ./figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull


# =============================================================================
# SHARED HELPERS  (label formatting, style, data loading, lease hull)
# =============================================================================

def tlabel(i):
    """Display label for a turbine. Internal indices are 0-based; turbines
    are presented to the reader as 1-based (T1..T95)."""
    return f"T{int(i) + 1}"


def sites_label(s):
    """Format a costs_vs_k 'sites' string (0-based, '+'-separated, or
    'none') as a 1-based, comma-separated, T-prefixed display label."""
    s = str(s).strip()
    if s in ("none", "", "infeasible"):
        return "none"
    return ", ".join(tlabel(int(tok)) for tok in s.split("+"))


# Colour palette (shared by every figure)
SEA      = "#e8eef2"
LEASE    = "#dde6ec"
TURBINE  = "#4a5a6a"
CHARGER  = "#c75b2f"
SUB      = "#7a1f1f"
ROUTE    = "#1f3a4d"
TASK     = "#f0b840"
CAND     = "#888888"
ROBUST   = "#2d6a4f"
SINGLE   = "#c75b2f"
DIESEL   = "#7a3a3a"
OPEX     = "#3a7d7a"
TIMECOST = "#4a5a6a"
GRID     = "#d8d4c8"
BG       = "#fbfaf6"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#2b2b2b",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": BG,
        "savefig.facecolor": "white",
    })


def lease_polygon(turbines: np.ndarray, expand_km: float = 0.8):
    """Convex hull of turbines, slightly expanded outward (lease boundary)."""
    hull = ConvexHull(turbines)
    pts = turbines[hull.vertices]
    cen = pts.mean(axis=0)
    v = pts - cen
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    return pts + expand_km * vn


def load_data(data_dir: str | Path) -> dict:
    """Load every CSV the figures need (written by the optimisation pipeline)."""
    d = Path(data_dir)

    def load(name, **kwargs):
        return np.loadtxt(d / name, delimiter=",", skiprows=1, **kwargs)

    out = {
        "turbines": load("turbines.csv"),       # (95, 2) x_km, y_km
        "sub":      load("substation.csv"),      # (2,)
        "cand":     load("candidates.csv", dtype=int),    # 0-based indices
        "tasks":    load("tasks_single.csv", dtype=int),  # single-day demand
        "route":    load("route_k1.csv", dtype=int).flatten(),
        "charger":  int(load("charger_k1.csv", dtype=int)),
    }
    # costs_vs_k.csv has a string column ('sites'), so parse manually
    costs = []
    with open(d / "costs_vs_k.csv") as f:
        header = next(f).strip().split(",")
        has_breakdown = "elec_eur" in header and "time_eur" in header
        for line in f:
            parts = line.strip().split(",")
            row = {"K": int(parts[0]), "capex": float(parts[1]),
                   "opex": float(parts[2]), "total": float(parts[3]),
                   "tour_km": float(parts[4]), "sites": parts[5]}
            if has_breakdown and len(parts) >= 8:
                row["elec"] = float(parts[6]); row["time"] = float(parts[7])
            else:
                row["elec"] = float(parts[2]); row["time"] = 0.0
            costs.append(row)
    out["costs_vs_k"] = costs

    if (d / "multiday_k1.csv").exists():
        out["multiday"] = load("multiday_k1.csv")
    if (d / "scenario_costs_pair.csv").exists():
        with open(d / "scenario_costs_pair.csv") as f:
            header = f.readline().strip().split(",")
        out["scenarios"] = np.loadtxt(d / "scenario_costs_pair.csv",
                                      delimiter=",", skiprows=1)
        out["scenarios_labels"] = [int(h.split("_")[0][1:]) for h in header]
    return out


# =============================================================================
# FIGURE 1 — Turbine layout and candidate charger sites
#   (source: plot_layout_map.py)
# =============================================================================

def figure1_layout_map(data, out_path):
    turbines = data["turbines"]; sub = data["sub"]
    cand = set(int(i) for i in data["cand"])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_facecolor(SEA)
    ax.add_patch(Polygon(lease_polygon(turbines), facecolor=LEASE,
                         edgecolor="#7a8b9a", linewidth=1.2, linestyle="--",
                         zorder=1))

    # all turbines, with 1-based index labels offset above-right of each marker
    for i, (x, y) in enumerate(turbines):
        is_cand = i in cand
        if is_cand:
            ax.scatter(x, y, s=150, facecolors="none", edgecolors=CHARGER,
                       linewidths=2.0, zorder=4)
            ax.scatter(x, y, s=30, c=CHARGER, zorder=4)
        else:
            ax.scatter(x, y, s=34, c=TURBINE, alpha=0.65, zorder=3,
                       edgecolors="none")
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(4.5, 4.5), fontsize=7.0,
                    color=(CHARGER if is_cand else "#34404c"),
                    fontweight="bold" if is_cand else "normal", zorder=6)

    ax.scatter(sub[0], sub[1], s=340, marker="^", c=SUB,
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.annotate("Substation", (sub[0], sub[1]), textcoords="offset points",
                xytext=(0, 11), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=SUB, zorder=8)

    legend = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=SUB,
               markersize=13, label="Offshore substation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=CHARGER, markeredgewidth=2.0, markersize=12,
               label=f"Candidate charger site ({len(cand)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TURBINE,
               markersize=8, label=f"Turbine ({len(turbines)} total)"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.set_title("Dogger Bank A — turbine layout and candidate charger sites\n"
                 "95 turbines, 21 candidates; numbers are turbine indices",
                 fontweight="bold")
    ax.set_aspect("equal"); ax.set_xlim(-2, 34); ax.set_ylim(-2, 19)
    ax.grid(color="#c8d2da", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}")


# =============================================================================
# FIGURE 2 — Cost decomposition vs number of chargers K
#   (source: plot.py  fig_dogger_costs)
# =============================================================================

def figure2_cost_vs_K(data, out_path):
    costs = data["costs_vs_k"]; diesel = 4.2
    fig, ax = plt.subplots(figsize=(10, 6))
    Ks      = [r["K"] for r in costs]
    capex_m = [r["capex"] / 1e6 for r in costs]
    elec_m  = [r["elec"]  / 1e6 for r in costs]
    time_m  = [r["time"]  / 1e6 for r in costs]
    total_m = [r["total"] / 1e6 for r in costs]
    sites   = [r["sites"] for r in costs]
    has_time = any(t > 1e-9 for t in time_m)

    ax.bar(Ks, capex_m, 0.6, color=CHARGER, label="Charger capex (annualised)",
           edgecolor="white", linewidth=1.2)
    ax.bar(Ks, elec_m, 0.6, bottom=capex_m, color=OPEX,
           label="Vessel electricity (opex)", edgecolor="white", linewidth=1.2)
    if has_time:
        bottom_te = [c + e for c, e in zip(capex_m, elec_m)]
        ax.bar(Ks, time_m, 0.6, bottom=bottom_te, color=TIMECOST,
               label="Vessel time (opex)", edgecolor="white", linewidth=1.2)

    ax.axhline(diesel, color=DIESEL, linestyle="--", linewidth=1.5,
               label=f"Diesel SOV reference (\u20ac{diesel:.1f}M/yr)")

    kbest = min(range(len(total_m)),
                key=lambda i: total_m[i] if total_m[i] > 0 else float("inf"))
    ymax = max(max(total_m), diesel)
    for x, t, lab in zip(Ks, total_m, sites):
        if t > 0:
            ax.text(x, t + ymax * 0.015, f"\u20ac{t:.2f}M", ha="center",
                    fontsize=9, fontweight="bold" if x == Ks[kbest] else "normal")
            ax.text(x, -ymax * 0.05, sites_label(lab), ha="center",
                    fontsize=7.5, style="italic", color="#555")

    if total_m[kbest] > 0:
        ax.annotate("INTEGRATED OPTIMUM", xy=(Ks[kbest], total_m[kbest]),
                    xytext=(Ks[kbest] + 0.7, total_m[kbest] + ymax * 0.16),
                    ha="center", fontsize=10, fontweight="bold", color=ROBUST,
                    arrowprops=dict(arrowstyle="->", color=ROBUST, lw=1.5))
    if total_m[0] > 0 and has_time:
        ax.annotate('K=0: substation-only\n(detour returns inflate\nvessel-time cost)',
                    xy=(0, total_m[0]), xytext=(0.0, total_m[0] + ymax * 0.18),
                    ha="center", fontsize=8.5, color="#555",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    ax.set_xlabel("Number of in-field chargers K")
    ax.set_ylabel("Annual cost (\u20ac million)")
    subtitle = ("integrated objective: capex + electricity + vessel time"
                if has_time else "capex + electricity")
    ax.set_title("Dogger Bank A \u2014 Cost Decomposition vs. K\n"
                 f"95 turbines, 12 tasks/day \u2014 {subtitle}", fontweight="bold")
    ax.set_xticks(Ks); ax.legend(loc="upper left", framealpha=0.95)
    ax.set_ylim(-ymax * 0.08, ymax * 1.30); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}")


# =============================================================================
# FIGURE 3 — Break-even sensitivity to the value of vessel time
#   (source: plot_timecost.py)
#
# Sweeps c_time analytically from the two tour lengths, using the ABSOLUTE
# vessel-time objective consistent with the cost table:
#   total(c_time) = capex + electricity + N_op * (c_time / v) * tour_km
# NOTE: annual_op_days (300) and vessel_speed v (20 km/h) are model
# parameters; here they are passed in via `p` (a small namespace).
# =============================================================================

def figure3_breakeven(data, out_path, annual_op_days=300, vessel_speed=20.0,
                      chosen_ctime=1500.0):
    setup_style()
    rows = data["costs_vs_k"]
    r0 = next(r for r in rows if r["K"] == 0)
    r1 = next(r for r in rows if r["K"] == 1)
    days, v = annual_op_days, vessel_speed

    def total_of(r, ctime):
        return r["capex"] + r["elec"] + r["tour_km"] / v * ctime * days

    ctime = np.linspace(0, 5500, 500)
    c0 = total_of(r0, ctime); c1 = total_of(r1, ctime)
    hours_saved_yr = (r0["tour_km"] - r1["tour_km"]) / v * days
    be = (r1["capex"] - (r0["elec"] - r1["elec"])) / hours_saved_yr

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ctime, c0 / 1e6, color=SINGLE, lw=2.4, label="K=0  (substation only)")
    ax.plot(ctime, c1 / 1e6, color=ROBUST, lw=2.4,
            label="K=1  (one in-field charger, T67)")

    be_cost = total_of(r0, be) / 1e6
    ax.axvline(be, color="#444", ls=":", lw=1.4)
    ax.plot([be], [be_cost], "o", color="#222", ms=7, zorder=6)
    ax.annotate(f"break-even\n\u2248 {be:,.0f} \u20ac/vessel-hour",
                xy=(be, be_cost), xytext=(be + 250, be_cost - 0.32),
                fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#222", lw=1.3))

    band_lo, band_hi = 40000 / 12, 60000 / 12  # 40-60k/day over ~12 working hours
    ax.axvspan(band_lo, band_hi, color=ROBUST, alpha=0.10, zorder=0)
    ax.text((band_lo + band_hi) / 2, ax.get_ylim()[1] * 0.30,
            "realistic SOV\ncharter band", rotation=90, ha="center",
            va="center", fontsize=8.5, color=ROBUST)

    ax.axvline(chosen_ctime, color="#999", ls="-", lw=1.0, alpha=0.8)
    ax.text(chosen_ctime, ax.get_ylim()[1] * 0.96,
            f"  chosen\n  {chosen_ctime:,.0f} \u20ac/h",
            fontsize=8.5, color="#666", va="top")

    ax.text(be * 0.45, total_of(r0, be * 0.45) / 1e6 + 0.15, "K=0 cheaper",
            color=SINGLE, fontsize=10, fontweight="bold", ha="center")
    ax.text(min(be * 2.4, 2600), total_of(r1, min(be * 2.4, 2600)) / 1e6 + 0.18,
            "K=1 cheaper", color=ROBUST, fontsize=10, fontweight="bold", ha="center")

    ax.set_xlabel("Value of vessel time, $c^{\\mathrm{time}}$ (\u20ac per vessel-hour)")
    ax.set_ylabel("Integrated annual cost (M\u20ac/yr)")
    ax.set_title("Break-even: when does an in-field charger pay for itself?",
                 fontweight="bold", pad=14)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); ax.set_xlim(0, 5500)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor=GRID, fontsize=9.5)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}  (break-even {be:,.0f} EUR/vessel-hour)")


# =============================================================================
# FIGURE 4 — Optimal K=1 solution and SOV route
#   (source: plot.py  fig_dogger_main)
# =============================================================================

def figure4_optimal_route(data, out_path):
    turbines = data["turbines"]; sub = data["sub"]
    cand = data["cand"]; tasks = data["tasks"]
    route = data["route"]; charger = data["charger"]
    k1 = next(r for r in data["costs_vs_k"] if r["K"] == 1)
    all_nodes = np.vstack([sub[None, :], turbines])  # node 0 = substation

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(SEA)
    ax.add_patch(Polygon(lease_polygon(turbines), facecolor=LEASE,
                         edgecolor="#7a8b9a", linewidth=1.2, linestyle="--",
                         zorder=0))
    ax.scatter(turbines[:, 0], turbines[:, 1], s=55, c=TURBINE, marker="o",
               edgecolors="white", linewidths=1.0, zorder=3)
    for ci in cand:
        ax.add_patch(Circle(turbines[ci], 0.45, fill=False, edgecolor=CAND,
                            linewidth=1.0, linestyle=":", zorder=2))
    for ti in tasks:
        ax.add_patch(Circle(turbines[ti], 0.65, fill=False, edgecolor=TASK,
                            linewidth=2.2, zorder=2.5))

    pt = turbines[charger]
    ax.scatter(*pt, s=320, c=CHARGER, marker="s", edgecolors="white",
               linewidths=1.6, zorder=4)
    ax.annotate(f"CHARGER\n({tlabel(charger)})", pt, xytext=(10, 10),
                textcoords="offset points", fontsize=10, fontweight="bold",
                color=CHARGER,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.scatter(*sub, s=260, c=SUB, marker="v", edgecolors="white",
               linewidths=1.6, zorder=4)
    ax.annotate("OFFSHORE\nSUBSTATION", sub, xytext=(10, -25),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color=SUB,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    for i in range(len(route) - 1):
        a = all_nodes[route[i]]; b = all_nodes[route[i + 1]]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                     color=ROUTE, linewidth=1.6, alpha=0.85,
                                     zorder=3.5))

    # Electrification headline: capital + electricity only (a diesel SOV incurs
    # essentially the same transit time, so vessel time is excluded here).
    diesel = 4.2e6
    capex_elec = k1["capex"] + k1["elec"]
    savings = 100 * (diesel - capex_elec) / diesel

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TURBINE,
               markersize=9, label="Turbine (95 total)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=TASK, markeredgewidth=2.2, markersize=11,
               label=f"Maintenance task ({len(tasks)} today)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=CAND, linestyle=":", markeredgewidth=1.2,
               markersize=11, label="Candidate charger site"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CHARGER,
               markersize=12, label="Optimal charger (K=1)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=SUB,
               markersize=11, label="Offshore substation (SOV base)"),
        Line2D([0], [0], color=ROUTE, linewidth=1.8, label="SOV route"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", framealpha=0.95, fontsize=9)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.set_title(f"Dogger Bank A — Optimal Charger Location (K=1)\n"
                 f"Charger at turbine {tlabel(charger)} · Tour {k1['tour_km']:.0f} km · "
                 f"capex+electricity \u2212{savings:.0f}% vs diesel · "
                 f"integrated \u20ac{k1['total']/1e6:.2f}M/yr", fontweight="bold")
    ax.set_xlim(-2, 35); ax.set_ylim(-2, 19); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}")


# =============================================================================
# FIGURE 5 — Robust charger location: expected cost across scenarios
#   (source: plot.py  fig_multiday_map)
# =============================================================================

def figure5_robust_map(data, out_path):
    if "multiday" not in data:
        print("  (skipped: no multiday data)"); return
    turbines = data["turbines"]; sub = data["sub"]
    md = data["multiday"]
    single_t = int(data["charger"])
    ti_col   = md[:, 0].astype(int)
    annual_m = md[:, 5] / 1e6
    xs, ys   = md[:, 6], md[:, 7]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(SEA)
    ax.add_patch(Polygon(lease_polygon(turbines), facecolor=LEASE,
                         edgecolor="#7a8b9a", linewidth=1.2, linestyle="--",
                         zorder=0))
    ax.scatter(turbines[:, 0], turbines[:, 1], s=30, c=TURBINE, alpha=0.35,
               edgecolors="none", zorder=2)
    ax.scatter(*sub, s=240, c=SUB, marker="v", edgecolors="white",
               linewidths=1.6, zorder=4)
    ax.annotate("SUBSTATION", sub, xytext=(8, -22), textcoords="offset points",
                fontsize=9, fontweight="bold", color=SUB,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=annual_m.min(), vmax=annual_m.max())
    best_i = int(np.argmin(annual_m)); best_t = int(ti_col[best_i])

    for i, ti in enumerate(ti_col):
        c = cmap(norm(annual_m[i]))
        ax.scatter(xs[i], ys[i], s=260, c=[c], marker="o", edgecolors="black",
                   linewidths=1.0, zorder=3)
        if i == best_i:
            ax.add_patch(Circle((xs[i], ys[i]), 1.4, fill=False,
                                edgecolor=ROBUST, linewidth=3.0, zorder=5))
            ax.annotate(f"ROBUST OPTIMUM\n{tlabel(ti)} · €{annual_m[i]:.2f}M/yr",
                        (xs[i], ys[i]), xytext=(15, 15),
                        textcoords="offset points", fontsize=10,
                        fontweight="bold", color=ROBUST,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        if ti == single_t and ti != best_t:
            single_rank = int(np.argsort(annual_m).tolist().index(i)) + 1
            ax.add_patch(Circle((xs[i], ys[i]), 1.0, fill=False,
                                edgecolor=SINGLE, linewidth=2.5,
                                linestyle="--", zorder=4.5))
            ax.annotate(f"Single-day optimum\n{tlabel(single_t)} (rank {single_rank})",
                        (xs[i], ys[i]), xytext=(-90, -40),
                        textcoords="offset points", fontsize=9,
                        fontweight="bold", color=SINGLE,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                        arrowprops=dict(arrowstyle="->", color=SINGLE, lw=1.2))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Expected annual cost (€M)", fontsize=10)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.set_title("Dogger Bank A — Robust Charger Location (K=1)\n"
                 "Expected cost across 50 task realisations · workload 8–15 tasks/day",
                 fontweight="bold")
    ax.set_xlim(-2, 35); ax.set_ylim(-2, 19); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}")


# =============================================================================
# FIGURE 6 — Daily cost distributions + top-10 ranking
#   (source: plot.py  fig_multiday_compare)
# =============================================================================

def figure6_distribution_ranking(data, out_path):
    if "scenarios" not in data or "multiday" not in data:
        print("  (skipped: no multiday data)"); return
    sc = data["scenarios"]; md = data["multiday"]
    robust_t, single_t = data["scenarios_labels"]
    t_robust = sc[~np.isnan(sc[:, 0]), 0]
    t_single = sc[~np.isnan(sc[:, 1]), 1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: histogram of daily costs
    ax = axes[0]
    bins = np.linspace(min(t_robust.min(), t_single.min()),
                       max(t_robust.max(), t_single.max()), 20)
    ax.hist(t_robust, bins=bins, alpha=0.6, color=ROBUST,
            label=f"{tlabel(robust_t)} (robust)\nmean €{t_robust.mean():.0f}, "
                  f"P95 €{np.percentile(t_robust, 95):.0f}",
            edgecolor="white", linewidth=1.0)
    ax.hist(t_single, bins=bins, alpha=0.55, color=SINGLE,
            label=f"{tlabel(single_t)} (single-day)\nmean €{t_single.mean():.0f}, "
                  f"P95 €{np.percentile(t_single, 95):.0f}",
            edgecolor="white", linewidth=1.0)
    ax.axvline(t_robust.mean(), color=ROBUST, linestyle="--", linewidth=1.5)
    ax.axvline(t_single.mean(), color=SINGLE, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Daily vessel operating cost (€)")
    ax.set_ylabel("Number of days (out of 50 scenarios)")
    ax.set_title("Daily cost distributions", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9); ax.grid(alpha=0.3)

    # Right: top-10 ranking with +/-1 SD error bars
    ax = axes[1]
    order = np.argsort(md[:, 5]); top10 = md[order][:10]
    ys = np.arange(10)
    totals_m = top10[:, 5] / 1e6
    std_m    = top10[:, 2] * 300 / 1e6
    labels   = [tlabel(t) for t in top10[:, 0]]
    best_t   = int(top10[0, 0])
    colors = []
    for t in top10[:, 0].astype(int):
        if t == best_t:     colors.append(ROBUST)
        elif t == single_t: colors.append(SINGLE)
        else:               colors.append("#6a8aa3")
    ax.barh(ys, totals_m, xerr=std_m, color=colors, edgecolor="white",
            linewidth=1.0, error_kw=dict(ecolor="#444", capsize=3, lw=1))
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
    for i, t in enumerate(totals_m):
        ax.text(t + 0.02, i, f"€{t:.3f}M", va="center", fontsize=8.5)
    ax.set_xlabel("Expected annual cost (€M, ± 1 SD)")
    ax.set_title("Top 10 candidate sites by expected cost", fontweight="bold")
    ax.set_xlim(totals_m.min() - 0.02, totals_m.max() + 0.05)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[
        Line2D([0], [0], color=ROBUST, lw=8, label=f"Robust optimum ({tlabel(best_t)})"),
        Line2D([0], [0], color=SINGLE, lw=8, label=f"Single-day optimum ({tlabel(single_t)})"),
        Line2D([0], [0], color="#6a8aa3", lw=8, label="Other candidates"),
    ], loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}")


# =============================================================================
# FIGURE 7 — Per-scenario comparison of robust vs single-day site
#   (source: plot.py  fig_multiday_scenarios)
# =============================================================================

def figure7_per_scenario(data, out_path):
    if "scenarios" not in data:
        print("  (skipped: no scenario data)"); return
    sc = data["scenarios"]
    robust_t, single_t = data["scenarios_labels"]
    t_robust, t_single = sc[:, 0], sc[:, 1]
    order = np.argsort(t_robust)
    rs, ss = t_robust[order], t_single[order]
    x = np.arange(1, len(rs) + 1)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    worse = ss > rs
    ax.fill_between(x, rs, ss, where=worse, color=SINGLE, alpha=0.15,
                    label=f"{tlabel(single_t)} worse than {tlabel(robust_t)}")
    ax.fill_between(x, rs, ss, where=~worse, color=ROBUST, alpha=0.15,
                    label=f"{tlabel(robust_t)} worse than {tlabel(single_t)}")
    ax.plot(x, rs, "o-", color=ROBUST, markersize=5, linewidth=1.5,
            label=f"{tlabel(robust_t)} (robust)")
    ax.plot(x, ss, "s-", color=SINGLE, markersize=5, linewidth=1.5,
            label=f"{tlabel(single_t)} (single-day opt)")

    n_wins = int(worse.sum())
    ax.set_xlabel(f"Scenario (sorted by {tlabel(robust_t)} cost, ascending)")
    ax.set_ylabel("Daily vessel operating cost (€)")
    ax.set_title(f"Per-scenario comparison: {tlabel(robust_t)} beats {tlabel(single_t)} "
                 f"in {n_wins}/{len(rs)} scenarios\n"
                 f"{tlabel(robust_t)} is more robust — lower mean AND lower tail risk",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"  -> {out_path}")


# =============================================================================
# DRIVER
# =============================================================================

FIGURES = [
    ("Figure1_layout_map.png",            figure1_layout_map),
    ("Figure2_cost_vs_K.png",             figure2_cost_vs_K),
    ("Figure3_breakeven.png",             figure3_breakeven),
    ("Figure4_optimal_route.png",         figure4_optimal_route),
    ("Figure5_robust_map.png",            figure5_robust_map),
    ("Figure6_distribution_ranking.png",  figure6_distribution_ranking),
    ("Figure7_per_scenario.png",          figure7_per_scenario),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./figures")
    args = ap.parse_args()

    setup_style()
    data = load_data(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {out}/")
    for fname, fn in FIGURES:
        fn(data, out / fname)


if __name__ == "__main__":
    main()
