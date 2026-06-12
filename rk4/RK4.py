import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

# Create an organized directory for Track B artifacts
output_dir = r"C:\Users\youss\OneDrive\Desktop\New folder (2)"
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — SYSTEM PARAMETERS (Directly Mirroring Track A)
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

t_start = 0
t_end   = 100

cases = {
    1: {"y0": [1, 0, 0, 0, 0, 0], "a12_decay": False, "label": "Case 1: Base Case"},
    2: {"y0": [0, 0, 0, 0, 0, 0], "a12_decay": False, "label": "Case 2: All-Zero ICs"},
    3: {"y0": [1, 0, 0, 0, 0, 0], "a12_decay": True,  "label": "Case 3: Exp a_12"},
}

# ─────────────────────────────────────────────────────────────
# SECTION 2 — SYSTEM STATE DEFINITION
# ─────────────────────────────────────────────────────────────
def apoptosis_ode(t, y, a12_decay=False):
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = y
    a12_eff = a_12 * np.exp(-0.1 * t) if a12_decay else a_12

    dyhif  = a_hif - a_3 * y_o2 * y_hif - a_4 * y_hif * y_p300 - a_7 * y_p53 * y_hif
    dyo2   = a_o2 - a_3 * y_o2 * y_hif + a_4 * y_hif * y_p300 - a_11 * y_o2
    dyp300 = a_8 - a_4 * y_hif * y_p300 - a_5 * y_p300 * y_p53
    dyp53  = a_p53 - a_5 * y_p300 * y_p53 - a_9 * y_p53
    dycasp = a12_eff + a_9 * y_p53 - a_13 * y_casp
    dykp   = -a_10 * y_casp * y_kp + a_11 * y_o2 - a_14 * y_kp

    return np.array([dyhif, dyo2, dyp300, dyp53, dycasp, dykp])

# ─────────────────────────────────────────────────────────────
# SECTION 3 — FIXED-STEP RUNGE-KUTTA 4 (RK4) CORE ENGINE
# ─────────────────────────────────────────────────────────────
def rk4_solver(ode_func, t_span, y0, h, a12_decay):
    t_array = np.arange(t_span[0], t_span[1] + h, h)
    if t_array[-1] > t_span[1]:
        t_array = t_array[:-1]
        
    y_array = np.zeros((len(t_array), len(y0)))
    y_array[0] = y0

    for i in range(1, len(t_array)):
        t_curr = t_array[i-1]
        y_curr = y_array[i-1]

        k1 = ode_func(t_curr,       y_curr,                a12_decay)
        k2 = ode_func(t_curr + h/2, y_curr + (h/2) * k1,   a12_decay)
        k3 = ode_func(t_curr + h/2, y_curr + (h/2) * k2,   a12_decay)
        k4 = ode_func(t_curr + h,   y_curr + h * k3,       a12_decay)

        y_array[i] = y_curr + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return t_array, y_array

# ─────────────────────────────────────────────────────────────
# SECTION 4 — TRACK A REFERENCE REPRODUCTION FOR PROPER RMSE
# ─────────────────────────────────────────────────────────────
from scipy.integrate import solve_ivp
print("Generating Ground Truth Reference Solution vectors using LSODA...")
ref_solutions = {}
for ncase, cfg in cases.items():
    sol = solve_ivp(
        fun=lambda t, y: apoptosis_ode(t, y, cfg["a12_decay"]),
        t_span=(t_start, t_end),
        y0=cfg["y0"],
        method="LSODA",
        rtol=1e-10,
        atol=1e-12
    )
    ref_solutions[ncase] = sol

# ─────────────────────────────────────────────────────────────
# SECTION 5 — EXECUTION AND COMPUTATIONAL PROFILING LOOP
# ─────────────────────────────────────────────────────────────
step_sizes = [0.1, 0.01]
rk4_results = {}

print("\n" + "=" * 75)
print(f"{'Case':<8}{'h':<8}{'RMSE y_hif':<15}{'RMSE y_casp':<15}{'Runtime':<15}")
print("=" * 75)

for ncase, cfg in cases.items():
    rk4_results[ncase] = {}
    ref_sol = ref_solutions[ncase]
    
    for h in step_sizes:
        start_time = time.time()
        t_rk4, y_rk4 = rk4_solver(apoptosis_ode, (t_start, t_end), cfg["y0"], h, cfg["a12_decay"])
        elapsed = time.time() - start_time
        
        rmse_list = []
        for vi in range(6):
            ref_interp = np.interp(t_rk4, ref_sol.t, ref_sol.y[vi])
            rmse_val = np.sqrt(np.mean((y_rk4[:, vi] - ref_interp) ** 2))
            rmse_list.append(rmse_val)
            
        rk4_results[ncase][h] = {"t": t_rk4, "y": y_rk4, "rmse": rmse_list, "runtime": elapsed}
        
        print(f"Case {ncase:<4}{h:<8}{rmse_list[0]:<15.7f}{rmse_list[4]:<15.7f}{elapsed*1000:<10.2f} ms")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — FULL NUMERICAL GRID SNAPSHOT PRINTING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("            TRACK B RK4 COMPLETE NUMERICAL VALUE STATE VECTOR PRINTS")
print("=" * 85)

target_print_times = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]

for ncase, cfg in cases.items():
    print(f"\n{'='*85}")
    print(f" NETWORK RUN: {cfg['label']}")
    print(f"{'='*85}")
    
    for h in step_sizes:
        print(f"\n [Evaluation Parameters: Grid Spacing Step Size h = {h}]")
        print(f"  {'t':>6}  {'y_hif':>10}  {'y_o2':>10}  {'y_p300':>10}  {'y_p53':>10}  {'y_casp':>10}  {'y_kp':>10}")
        print(f"  {'─'*79}")
        
        t_arr = rk4_results[ncase][h]["t"]
        y_arr = rk4_results[ncase][h]["y"]
        
        for target_t in target_print_times:
            idx = np.argmin(np.abs(t_arr - target_t))
            actual_t = t_arr[idx]
            states = y_arr[idx]
            
            print(f"  {actual_t:6.1f}  {states[0]:10.4f}  {states[1]:10.4f}  {states[2]:10.4f}  {states[3]:10.4f}  {states[4]:10.4f}  {states[5]:10.4f}")

# ─────────────────────────────────────────────────────────────
# ADDED SECTION — FILE DATA EXPORTER (.CSV) FOR TEAM TRACKS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 75)
print("Exporting high-fidelity numerical arrays to track_b_outputs/ ...")
for ncase, h_dicts in rk4_results.items():
    for h, data in h_dicts.items():
        csv_filename = os.path.join(output_dir, f"rk4_simulation_case{ncase}_h{h}.csv")
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "y_hif", "y_o2", "y_p300", "y_p53", "y_casp", "y_kp"])
            for t_val, state_vec in zip(data["t"], data["y"]):
                writer.writerow([f"{t_val:.4f}"] + [f"{val:.8f}" for val in state_vec])

# ─────────────────────────────────────────────────────────────
# REBUILT SECTION 7 — 3x6 MULTI-CASE COMPREHENSIVE MATRIX PLOT
#   (now generates ONE figure PER step size: h=0.1 AND h=0.01)
# ─────────────────────────────────────────────────────────────
var_names = ["y_hif", "y_o2", "y_p300", "y_p53", "y_casp", "y_kp"]
col_titles = ["y_hif", "y_o2", "y_p300", "y_p53", "y_casp", "y_kp"]
line_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#e377c2", "#17becf"]

# Specific validation marker nodes to show reference checkpoints matching the book
book_markers = [0, 25, 50, 100]

# Loop over BOTH step sizes — generates 2 separate figures
for h_val in step_sizes:  # [0.1, 0.01]

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle(f"RK4 Solution — All 3 Cases — Apoptosis ODE  (h = {h_val})",
                  fontsize=14, fontweight="bold", y=0.96)

    for row_idx, ncase in enumerate([1, 2, 3]):
        cfg = cases[ncase]
        # Grab results for THIS step size
        res = rk4_results[ncase][h_val]
        ref_sol = ref_solutions[ncase]

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]

            # Plot continuous RK4 simulation trace line
            ax.plot(res["t"], res["y"][:, col_idx], color=line_colors[col_idx], linewidth=1.5)

            # Plot exact validation checkpoint nodes as black diamonds
            for t_mark in book_markers:
                ref_idx = np.argmin(np.abs(ref_sol.t - t_mark))
                ax.plot(ref_sol.t[ref_idx], ref_sol.y[col_idx, ref_idx], 'kd', markersize=5)

            # Clean formatting
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.set_xlim(0, 100)
            ax.set_ylim(bottom=-0.05 * np.max(res["y"][:, col_idx]))
            ax.tick_params(axis='both', which='major', labelsize=8)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=12)

            ax.set_xlabel("Time", fontsize=8, labelpad=2)
            ax.set_ylabel(var_names[col_idx], fontsize=8, labelpad=2)

            if col_idx == 0:
                ax.text(-0.38, 0.5, f"{cfg['label']}\n(y_hif0={cfg['y0'][0]})",
                        transform=ax.transAxes, fontsize=9, fontweight="semibold",
                        va='center', ha='right', rotation=90)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.08, wspace=0.35, hspace=0.35)

    # Filename includes the step size — so h=0.1 and h=0.01 don't overwrite each other
    h_str = str(h_val).replace('.', 'p')  # 0.1 -> 0p1, 0.01 -> 0p01
    plot_filename = os.path.join(output_dir, f"rk4_comprehensive_3x6_grid_h{h_str}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {plot_filename}")

    plt.show()
