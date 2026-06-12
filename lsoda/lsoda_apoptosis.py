"""
=============================================================
Track A — ODE Model + Book Reproduction
Schiesser Chapter 3: Apoptosis
=============================================================

Reproduces all results from Listings 3.1, 3.2, 3.3, and 3.4 of:
Schiesser, W.E. (2014). Differential Equation Analysis in
Biomedical Science and Engineering: ODE Applications with R.

Cases reproduced:
  ncase=1 : Base case    — y_hif(0)=1, rest zero
  ncase=2 : IC variation — all zeros IC
  ncase=3 : ODE variation — a_12 decays exponentially with t

Authors : [Your names here]
Course  : SBEG108 — Numerical Methods in Biomedical Engineering
Date    : Spring 2026
=============================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import os
output_dir = r"C:\Users\ebrah\Downloads"
os.makedirs(output_dir, exist_ok=True)
# ─────────────────────────────────────────────────────────────
# SECTION 1 — PARAMETERS  (Table 3.3, p.148)
# ─────────────────────────────────────────────────────────────
a_hif = 1.52
a_o2  = 1.80
a_p53 = 0.05
a_3   = 0.90
a_4   = 0.20
a_5   = 0.001
a_7   = 0.70
a_8   = 0.06
a_9   = 0.10
a_10  = 0.70
a_11  = 0.20
a_12  = 0.10
a_13  = 0.10
a_14  = 0.05

# ─────────────────────────────────────────────────────────────
# SECTION 2 — TIME SPAN  (Listing 3.1, p.155)
# ─────────────────────────────────────────────────────────────
t_start = 0
t_end   = 100
nout    = 101                                   # 0,1,2,...,100
t_eval  = np.linspace(t_start, t_end, nout)    # every 1 unit

# ─────────────────────────────────────────────────────────────
# SECTION 3 — THREE CASES
# ─────────────────────────────────────────────────────────────
# ncase=1  base case        : y_hif(0)=1, others=0, a_12 constant
# ncase=2  IC variation     : all zeros IC,          a_12 constant
# ncase=3  ODE variation    : y_hif(0)=1, others=0, a_12*exp(-0.1t)
cases = {
    1: {"y0": [1, 0, 0, 0, 0, 0], "a12_decay": False,
        "label": "ncase=1: Base case  (y_hif₀=1)"},
    2: {"y0": [0, 0, 0, 0, 0, 0], "a12_decay": False,
        "label": "ncase=2: Zero IC    (y_hif₀=0)"},
    3: {"y0": [1, 0, 0, 0, 0, 0], "a12_decay": True,
        "label": "ncase=3: a₁₂ decay  (a₁₂·e^{-0.1t})"},
}

# ─────────────────────────────────────────────────────────────
# SECTION 4 — ODE FUNCTION  (Listing 3.2 / 3.4, pp.161–172)
# ─────────────────────────────────────────────────────────────
def apoptosis_ode(t, y, a12_decay=False):
    """
    Six coupled ODEs for the apoptosis model eqs.(3.1)-(3.6).

    State vector y:
        y[0] = y_hif   hypoxia inducible factor
        y[1] = y_o2    oxygen concentration
        y[2] = y_p300  p300 co-activator
        y[3] = y_p53   p53 tumor suppressor
        y[4] = y_casp  caspase
        y[5] = y_kp    potassium ions

    a12_decay : if True, a_12 is replaced by a_12*exp(-0.1*t)
                (Listing 3.4, ncase=2 ODE variation)
    """
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = y

    # effective a_12 — constant or exponentially decaying
    a12_eff = a_12 * np.exp(-0.1 * t) if a12_decay else a_12

    # eq. (3.1) — HIF
    dyhif  = ( a_hif
             - a_3  * y_o2  * y_hif
             - a_4  * y_hif * y_p300
             - a_7  * y_p53 * y_hif )

    # eq. (3.2) — O2
    dyo2   = ( a_o2
             - a_3  * y_o2  * y_hif
             + a_4  * y_hif * y_p300
             - a_11 * y_o2  )

    # eq. (3.3) — p300
    dyp300 = ( a_8
             - a_4  * y_hif * y_p300
             - a_5  * y_p300 * y_p53 )

    # eq. (3.4) — p53
    dyp53  = ( a_p53
             - a_5  * y_p300 * y_p53
             - a_9  * y_p53  )

    # eq. (3.5) — caspase  (a_12 may decay)
    dycasp = ( a12_eff
             + a_9  * y_p53
             - a_13 * y_casp )

    # eq. (3.6) — K+
    dykp   = ( - a_10 * y_casp * y_kp
               + a_11 * y_o2
               - a_14 * y_kp  )

    return [dyhif, dyo2, dyp300, dyp53, dycasp, dykp]

# ─────────────────────────────────────────────────────────────
# SECTION 5 — LSODA SOLVER  (equivalent to R's lsoda/ode)
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  LSODA SOLUTION  —  Schiesser Chapter 3 Reproduction")
print("=" * 65)

solutions = {}   # store sol object per case

# book only prints t = 0, 25, 50, 100  (it=1,26,51,101)
print_times = [0.0, 25.0, 50.0, 100.0]
var_names   = ["y_hif", "y_o2 ", "y_p300", "y_p53 ", "y_casp", "y_kp  "]

for ncase, cfg in cases.items():
    call_count = [0]

    def ode_counted(t, y, cfg=cfg):
        call_count[0] += 1
        return apoptosis_ode(t, y, cfg["a12_decay"])

    start = time.time()
    sol = solve_ivp(
        fun    = ode_counted,
        t_span = (t_start, t_end),
        y0     = cfg["y0"],
        method = "LSODA",
        t_eval = t_eval,
        rtol   = 1e-8,
        atol   = 1e-10,
    )
    elapsed = time.time() - start
    solutions[ncase] = sol

    # ── print table matching book Table 3.4 / 3.5 format ──
    print(f"\n  ncase = {ncase}   {cfg['label']}")
    print(f"  {'─'*58}")
    for t_val in print_times:
        idx = np.argmin(np.abs(sol.t - t_val))
        print(f"\n    t = {sol.t[idx]:5.1f}")
        for vi, vname in enumerate(var_names):
            print(f"      {vname} = {sol.y[vi][idx]:8.4f}", end="   ")
            if vi % 2 == 1:
                print()
    print(f"\n  ncall = {call_count[0]}   |   solve time = {elapsed*1000:.2f} ms")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — VALIDATION TABLE
# ─────────────────────────────────────────────────────────────
# Expected equilibrium at t=100 from book Table 3.4
expected = {
    "y_hif" : 0.4991,
    "y_o2"  : 2.8646,
    "y_p300": 0.5978,
    "y_p53" : 0.4970,
    "y_casp": 1.4968,
    "y_kp"  : 0.5219,
}

print("\n" + "=" * 65)
print("  VALIDATION — Equilibrium at t=100  (ncase=1 vs book)")
print("=" * 65)
print(f"\n  {'Variable':<10} {'Computed':>12} {'Book':>12} {'|Error|':>12}")
print("  " + "─" * 48)

sol1   = solutions[1]
idx100 = np.argmin(np.abs(sol1.t - 100.0))
vkeys  = ["y_hif","y_o2","y_p300","y_p53","y_casp","y_kp"]

for vi, key in enumerate(vkeys):
    computed = sol1.y[vi][idx100]
    book_val = expected[key]
    error    = abs(computed - book_val)
    print(f"  {key:<10} {computed:>12.4f} {book_val:>12.4f} {error:>12.6f}")

# ─────────────────────────────────────────────────────────────
# SECTION 7 — EQUILIBRIUM VERIFICATION (derivatives → 0)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EQUILIBRIUM CHECK — derivatives at t=100 should be ≈ 0")
print("=" * 65)
y_eq   = sol1.y[:, idx100]
derivs = apoptosis_ode(100.0, y_eq, a12_decay=False)
dnames = ["dy_hif/dt", "dy_o2/dt ", "dy_p300/dt",
          "dy_p53/dt", "dy_casp/dt","dy_kp/dt  "]
print()
for dn, dv in zip(dnames, derivs):
    print(f"  {dn} = {dv:+.6f}")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — FIGURE 3.2  (6 subplots, one per variable)
# Reproduces the 3×2 figure in Schiesser p.167
# ─────────────────────────────────────────────────────────────
titles = [
    "Hypoxia Inducible Factor  y_hif(t)",
    "Oxygen Level  y_o2(t)",
    "Co-activator  y_p300(t)",
    "Tumor Suppressor Gene  y_p53(t)",
    "Caspases  y_casp(t)",
    "Potassium Ions  y_kp(t)",
]
ylabels = ["y_hif(t)", "y_o2(t)", "y_p300(t)",
           "y_p53(t)", "y_casp(t)", "y_kp(t)"]
colors  = ["steelblue", "darkorange", "forestgreen"]

fig, axes = plt.subplots(3, 2, figsize=(13, 13))
fig.suptitle(
    "Apoptosis ODE Model — Schiesser Chapter 3\n"
    "Python Reproduction using LSODA  (ncase = 1, 2, 3)",
    fontsize=13, fontweight="bold"
)
axes = axes.flatten()

for vi in range(6):
    ax = axes[vi]
    for ncase, cfg in cases.items():
        sol = solutions[ncase]
        ax.plot(sol.t, sol.y[vi],
                color=colors[ncase - 1],
                linewidth=2,
                linestyle=["-", "--", ":"][ncase - 1],
                label=cfg["label"])
    ax.set_title(titles[vi], fontsize=10)
    ax.set_xlabel("t  (dimensionless)", fontsize=9)
    ax.set_ylabel(ylabels[vi], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig_apoptosis_6vars.png",
            dpi=150, bbox_inches="tight")
print("\n  Saved: fig_apoptosis_6vars.png")

# ─────────────────────────────────────────────────────────────
# SECTION 9 — FIGURE: IC comparison (ncase 1 vs 2 overlay)
# Shows same equilibrium regardless of IC (Section 3.6)
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(3, 2, figsize=(13, 13))
fig2.suptitle(
    "IC Variation Study — ncase=1 (y_hif₀=1)  vs  ncase=2 (y_hif₀=0)\n"
    "Both Reach the Same Equilibrium  (Schiesser Section 3.6)",
    fontsize=12, fontweight="bold"
)
axes2 = axes2.flatten()

for vi in range(6):
    ax = axes2[vi]
    for ncase in [1, 2]:
        cfg = cases[ncase]
        sol = solutions[ncase]
        ax.plot(sol.t, sol.y[vi],
                color=colors[ncase - 1],
                linewidth=2,
                linestyle=["-", "--"][ncase - 1],
                label=cfg["label"])
    # mark equilibrium value
    eq_val = list(expected.values())[vi]
    ax.axhline(eq_val, color="gray", linestyle=":", linewidth=1.2,
               label=f"Equilibrium = {eq_val:.4f}")
    ax.set_title(titles[vi], fontsize=10)
    ax.set_xlabel("t  (dimensionless)", fontsize=9)
    ax.set_ylabel(ylabels[vi], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig_apoptosis_IC_comparison.png",
            dpi=150, bbox_inches="tight")
print("  Saved: fig_apoptosis_IC_comparison.png")

# ─────────────────────────────────────────────────────────────
# SECTION 10 — FIGURE: a_12 ODE variation (ncase 1 vs 3)
# Focus on y_casp and y_kp which are most affected
# ─────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
fig3.suptitle(
    "ODE Variation — Constant a₁₂  vs  Decaying a₁₂·e^{-0.1t}\n"
    "Effect on Caspase y_casp and Potassium y_kp  (Schiesser Section 3.7)",
    fontsize=12, fontweight="bold"
)

plot_vars = [4, 5, 2, 3]   # casp, kp, p300, p53 — most affected
plot_titles = [
    "Caspase y_casp(t) — most affected",
    "Potassium y_kp(t) — most affected",
    "Co-activator y_p300(t)",
    "Tumor suppressor y_p53(t)",
]
axes3 = axes3.flatten()

for i, vi in enumerate(plot_vars):
    ax = axes3[i]
    for ncase in [1, 3]:
        cfg = cases[ncase]
        sol = solutions[ncase]
        ax.plot(sol.t, sol.y[vi],
                linewidth=2,
                linestyle=["-", ":"][ncase != 1],
                color=["steelblue", "crimson"][ncase != 1],
                label=cfg["label"])
    ax.set_title(plot_titles[i], fontsize=10)
    ax.set_xlabel("t  (dimensionless)", fontsize=9)
    ax.set_ylabel(ylabels[vi], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig_apoptosis_a12_variation.png",
            dpi=150, bbox_inches="tight")
print("  Saved: fig_apoptosis_a12_variation.png")

# ─────────────────────────────────────────────────────────────
# SECTION 11 — FULL NUMERICAL OUTPUT TABLE (all 3 cases)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FULL NUMERICAL OUTPUT  —  All 3 Cases")
print("=" * 65)

all_print_times = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]

for ncase, cfg in cases.items():
    sol = solutions[ncase]
    print(f"\n  ncase = {ncase}   {cfg['label']}")
    print(f"  {'t':>6}  {'y_hif':>8}  {'y_o2':>8}  {'y_p300':>8}"
          f"  {'y_p53':>8}  {'y_casp':>8}  {'y_kp':>8}")
    print("  " + "─" * 62)
    for t_val in all_print_times:
        idx = np.argmin(np.abs(sol.t - t_val))
        row = [sol.y[vi][idx] for vi in range(6)]
        print(f"  {sol.t[idx]:6.1f}  "
              + "  ".join(f"{v:8.4f}" for v in row))

plt.show()
print("\n✓  All done. 3 figures saved to outputs.")


# ─────────────────────────────────────────────────────────────
# SECTION 12 — CSV EXPORT
# ─────────────────────────────────────────────────────────────
import csv, os

output_dir = "/mnt/user-data/outputs"

for ncase, cfg in cases.items():
    sol      = solutions[ncase]
    filename = os.path.join(output_dir, f"apoptosis_ncase{ncase}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "y_hif", "y_o2", "y_p300",
                         "y_p53", "y_casp", "y_kp"])
        for i in range(len(sol.t)):
            writer.writerow([
                f"{sol.t[i]:.4f}",
                f"{sol.y[0][i]:.6f}",
                f"{sol.y[1][i]:.6f}",
                f"{sol.y[2][i]:.6f}",
                f"{sol.y[3][i]:.6f}",
                f"{sol.y[4][i]:.6f}",
                f"{sol.y[5][i]:.6f}",
            ])
    print(f"  Saved CSV: apoptosis_ncase{ncase}.csv")

# equilibrium summary CSV
with open(os.path.join(output_dir, "apoptosis_equilibrium.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variable", "equilibrium_t100",
                     "book_value", "absolute_error"])
    for vi, key in enumerate(vkeys):
        comp = solutions[1].y[vi][np.argmin(np.abs(solutions[1].t - 100))]
        book = list(expected.values())[vi]
        writer.writerow([key, f"{comp:.6f}", f"{book:.4f}",
                         f"{abs(comp-book):.8f}"])
print("  Saved CSV: apoptosis_equilibrium.csv")
