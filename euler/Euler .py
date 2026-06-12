"""
Apoptosis ODE Model - Euler Method Solver
Course: SBEG108 Numerical Methods in Biomedical Engineering
Chapter 3: Apoptosis (Schiesser, 2014)

3 Cases implemented:
  Case 1: Base IC (y_hif=1, rest=0), constant a12
  Case 2: Zero IC  (all=0),          constant a12
  Case 3: Zero IC ,          a12*exp(-0.1*t)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────
#  Parameters (Table 3.3)
# ─────────────────────────────────────────
p = {
    'a_hif': 1.52, 'a_o2': 1.8,  'a_p53': 0.05,
    'a3':    0.9,  'a4':   0.2,   'a5':    0.001,
    'a7':    0.7,  'a8':   0.06,  'a9':    0.1,
    'a10':   0.7,  'a11':  0.2,   'a12':   0.1,
    'a13':   0.1,  'a14':  0.05,
}

# ─────────────────────────────────────────
#  Book reference values (Tables 3.4, 3.5, 3.6)
#  [y_hif, y_o2, y_p300, y_p53, y_casp, y_kp]
# ─────────────────────────────────────────
book_ref = {
    1: {
        0.0:   [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        25.0:  [0.5420, 2.6645, 0.4706, 0.4576, 1.2735, 0.5706],
        50.0:  [0.5038, 2.8407, 0.5771, 0.4940, 1.4708, 0.5267],
        100.0: [0.4991, 2.8646, 0.5978, 0.4970, 1.4968, 0.5219],
    },
    2: {
        0.0:   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        25.0:  [0.5392, 2.6779, 0.4800, 0.4575, 1.2735, 0.5740],
        50.0:  [0.5036, 2.8419, 0.5783, 0.4940, 1.4708, 0.5269],
        100.0: [0.4991, 2.8646, 0.5978, 0.4970, 1.4968, 0.5219],
    },
    3: {
        0.0:   [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        25.0:  [0.5392, 2.6779, 0.4800, 0.4575, 0.5608, 1.1832],
        50.0:  [0.5036, 2.8419, 0.5783, 0.4940, 0.5112, 1.3837],
        100.0: [0.4991, 2.8646, 0.5978, 0.4970, 0.4973, 1.4390],
    },
}

labels     = ['y_hif', 'y_o2', 'y_p300', 'y_p53', 'y_casp', 'y_kp']
colors     = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#00BCD4']
titles     = ['HIF', 'Oxygen', 'p300', 'p53', 'Caspase', 'Potassium']
case_names = {
    1: 'Case 1: Base IC (y_hif=1), constant a12',
    2: 'Case 2: Zero IC, constant a12',
    3: 'Case 3: Zero IC, a12*exp(-0.1t)',
}

# ─────────────────────────────────────────
#  ODE system
# ─────────────────────────────────────────
def apoptosis(y, t, p, ncase):
    hif, o2, p300, p53, casp, kp = y
    dhif  = p['a_hif'] - p['a3']*o2*hif  - p['a4']*hif*p300 - p['a7']*p53*hif
    do2   = p['a_o2']  - p['a3']*o2*hif  + p['a4']*hif*p300 - p['a11']*o2
    dp300 = -p['a4']*hif*p300 - p['a5']*p300*p53 + p['a8']
    dp53  = p['a_p53'] - p['a5']*p300*p53 - p['a9']*p53
    if ncase == 3:
        dcasp = p['a9']*p53 + p['a12']*np.exp(-0.1*t) - p['a13']*casp
    else:
        dcasp = p['a9']*p53 + p['a12'] - p['a13']*casp
    dkp   = -p['a10']*casp*kp + p['a11']*o2 - p['a14']*kp
    return np.array([dhif, do2, dp300, dp53, dcasp, dkp])

# ─────────────────────────────────────────
#  Euler solver (accepts any h)
# ─────────────────────────────────────────
def euler_solve(ncase, h=0.01):
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if ncase in [1, 3] \
         else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    t0, tf = 0.0, 100.0
    t_arr  = np.arange(t0, tf + h, h)
    N      = len(t_arr)
    y_arr  = np.zeros((N, 6))
    y_arr[0] = y0

    start = time.perf_counter()
    for i in range(N - 1):
        y_arr[i+1] = y_arr[i] + h * apoptosis(y_arr[i], t_arr[i], p, ncase)
    elapsed = time.perf_counter() - start

    return t_arr, y_arr, elapsed

# ─────────────────────────────────────────
#  Error metrics vs book reference points
# ─────────────────────────────────────────
def compute_errors(ncase, t_arr, y_arr, h):
    """
    Computes MAE and RMSE at the 4 book reference time points
    (t = 0, 25, 50, 100) for all 6 variables.
    Returns per-variable arrays and overall scalars.
    """
    ref        = book_ref[ncase]
    ref_times  = [0.0, 25.0, 50.0, 100.0]
    euler_vals = []
    ref_vals   = []

    for tc in ref_times:
        idx = int(round(tc / h))
        idx = min(idx, len(t_arr) - 1)   # guard against rounding past end
        euler_vals.append(y_arr[idx])
        ref_vals.append(ref[tc])

    euler_vals = np.array(euler_vals)   # shape (4, 6)
    ref_vals   = np.array(ref_vals)     # shape (4, 6)
    abs_err    = np.abs(euler_vals - ref_vals)

    # Per-variable metrics  (axis=0 → over the 4 time points)
    MAE_per_var  = np.mean(abs_err, axis=0)          # shape (6,)
    RMSE_per_var = np.sqrt(np.mean((euler_vals - ref_vals)**2, axis=0))

    # Overall scalar metrics (over all 4 times × 6 variables)
    MAE_overall  = np.mean(abs_err)
    RMSE_overall = np.sqrt(np.mean((euler_vals - ref_vals)**2))

    return MAE_per_var, RMSE_per_var, MAE_overall, RMSE_overall, abs_err, euler_vals, ref_vals

# ─────────────────────────────────────────
#  Run: h = 0.01  AND  h = 0.1
# ─────────────────────────────────────────
h_values = [0.01, 0.1]
all_results = {}

for h in h_values:
    all_results[h] = {}
    for nc in [1, 2, 3]:
        t_arr, y_arr, elapsed = euler_solve(nc, h)
        all_results[h][nc] = {'t': t_arr, 'y': y_arr, 'time': elapsed}
        print(f"  h={h}  Case {nc} — {elapsed*1000:.1f} ms")

# ─────────────────────────────────────────
#  Plot: 3x6 grid for each h value
#  Rows = Cases 1,2,3   Cols = 6 variables
#  Matches the RK4 style layout
# ─────────────────────────────────────────
case_row_labels = [
    'Case 1: Base Case',
    'Case 2: All-Zero ICs',
    'Case 3: Exp a_12',
]

for h in h_values:
    fig, axes = plt.subplots(3, 6, figsize=(22, 12))
    fig.suptitle(f'Euler Solution — All 3 Cases — Apoptosis ODE  (h = {h})',
                 fontsize=15, fontweight='bold', y=1.01)

    for row, nc in enumerate([1, 2, 3]):
        t_arr = all_results[h][nc]['t']
        y_arr = all_results[h][nc]['y']
        ref   = book_ref[nc]
        ref_t = list(ref.keys())

        for col in range(6):
            ax = axes[row, col]
            ref_vals = [ref[t][col] for t in ref_t]

            ax.plot(t_arr, y_arr[:, col], color=colors[col], lw=1.5)
            ax.scatter(ref_t, ref_vals, color='black', s=30,
                       zorder=5, marker='D')

            # Title only on top row
            if row == 0:
                ax.set_title(labels[col], fontsize=11, fontweight='bold')

            # Case label only on leftmost column
            if col == 0:
                ax.set_ylabel(f'{case_row_labels[row]}\n{labels[col]}',
                              fontsize=8)
            else:
                ax.set_ylabel(labels[col], fontsize=8)

            ax.set_xlabel('Time', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────
#  Print comparison tables + MAE + RMSE
# ─────────────────────────────────────────
check_times = [0.0, 25.0, 50.0, 100.0]

for h in h_values:
    for nc in [1, 2, 3]:
        t_arr = all_results[h][nc]['t']
        y_arr = all_results[h][nc]['y']
        ref   = book_ref[nc]

        MAE_var, RMSE_var, MAE_all, RMSE_all, abs_err, euler_v, ref_v = \
            compute_errors(nc, t_arr, y_arr, h)

        print(f"\n{'='*72}")
        print(f"  {case_names[nc]}   |   h = {h}")
        print(f"  Euler runtime : {all_results[h][nc]['time']*1000:.2f} ms")
        print(f"  Overall MAE   : {MAE_all:.6f}")
        print(f"  Overall RMSE  : {RMSE_all:.6f}")
        print(f"{'='*72}")

        for k, tc in enumerate(check_times):
            idx = int(round(tc / h))
            idx = min(idx, len(t_arr) - 1)
            print(f"\n  t = {tc:.1f}")
            print(f"  {'Var':<8} {'Euler':>10} {'Book RKF45':>12} {'Abs Error':>12}")
            print(f"  {'-'*46}")
            for j in range(6):
                err = abs(y_arr[idx][j] - ref[tc][j])
                print(f"  {labels[j]:<8} {y_arr[idx][j]:>10.4f} "
                      f"{ref[tc][j]:>12.4f} {err:>12.4e}")

        print(f"\n  Per-variable summary:")
        print(f"  {'Var':<8} {'MAE':>12} {'RMSE':>12}")
        print(f"  {'-'*34}")
        for j in range(6):
            print(f"  {labels[j]:<8} {MAE_var[j]:>12.6f} {RMSE_var[j]:>12.6f}")

# ─────────────────────────────────────────
#  h comparison summary table
# ─────────────────────────────────────────
print(f"\n{'='*65}")
print("  H COMPARISON — Overall MAE and RMSE across all cases")
print(f"{'='*65}")
print(f"  {'h':>6}  {'Case':>6}  {'Overall MAE':>14}  {'Overall RMSE':>14}  {'Runtime(ms)':>12}")
print(f"  {'-'*58}")
for h in h_values:
    for nc in [1, 2, 3]:
        t_arr = all_results[h][nc]['t']
        y_arr = all_results[h][nc]['y']
        _, _, MAE_all, RMSE_all, _, _, _ = compute_errors(nc, t_arr, y_arr, h)
        rt = all_results[h][nc]['time'] * 1000
        print(f"  {h:>6.3f}  {nc:>6}  {MAE_all:>14.6f}  {RMSE_all:>14.6f}  {rt:>12.3f}")

print("\n✓ All done!")
