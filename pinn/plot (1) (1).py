import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# 1. SETUP & CONSTANTS
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A_HIF,  B_HIF  = 1.52, 3.05
A_O2,   B_O2   = 1.80, 2.10
A_P300, B_P300 = 1.35, 2.70
A_P53,  B_P53  = 1.60, 3.20
A_CASP, B_CASP = 1.20, 0.80
A_KP,   B_KP   = 0.90, 1.72
T_END          = 100.0

VAR_META = [
    (0, "y_HIF", "HIF-1α"), (1, "y_O2", "Oxygen / ROS"), 
    (2, "y_p300", "p300 Co-activator"), (3, "y_p53", "p53 Tumour Suppressor"), 
    (4, "y_Casp", "Caspase"), (5, "y_Kp", "Potassium (Kp)")
]

# =============================================================================
# 2. STANDARDIZED ARCHITECTURES (All 128 Neurons)
# =============================================================================
# Case 1: 128 neurons + Softplus (to prevent negative concentrations)
class ApoptosisPINN_Case1(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 6))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        
    def forward(self, t):
        return self.net(t)

# Cases 2 & 3: 128 neurons, no Softplus
class ApoptosisPINN_Case23(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 6))
        self.net = nn.Sequential(*layers)
        
    def forward(self, t):
        return self.net(t)

def load_model(path, case_num):
    ckpt = torch.load(path, map_location=DEVICE)
    
    hidden_dim = ckpt.get("hidden_dim", 128) if isinstance(ckpt, dict) and "hidden_dim" in ckpt else 128
    num_layers = ckpt.get("num_layers", 6) if isinstance(ckpt, dict) and "num_layers" in ckpt else 6
    
    if case_num == 1:
        model = ApoptosisPINN_Case1(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    else:
        model = ApoptosisPINN_Case23(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
        
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
        
    model.eval()
    return model

# =============================================================================
# 3. UNIFIED RK4 SIMULATOR
# =============================================================================
def get_alpha12(case_num, t):
    return 0.10 if case_num in [1, 2] else 0.10 * np.exp(-0.05 * t)

def rk4_step(t, y, case_num, dt):
    def rhs(t_val, y_val):
        hif, o2, p300, p53, casp, kp = y_val
        a12 = get_alpha12(case_num, t_val)
        
        d_hif  = A_HIF  - B_HIF*hif  + a12*kp   - hif*o2
        d_o2   = A_O2   - B_O2*o2   + a12*hif
        d_p300 = A_P300 - B_P300*p300 + a12*hif
        d_p53  = A_P53  - B_P53*p53  + a12*p300 - p53*casp
        d_casp = A_CASP - B_CASP*casp + a12*p53
        d_kp   = A_KP   - B_KP*kp   + a12*p53
        return np.array([d_hif, d_o2, d_p300, d_p53, d_casp, d_kp])

    k1 = rhs(t, y)
    k2 = rhs(t + dt/2, y + dt/2 * k1)
    k3 = rhs(t + dt/2, y + dt/2 * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_ground_truth(case_num, n_steps=500):
    y0 = np.array([0.0, 0, 0, 0, 0, 0]) if case_num == 2 else np.array([1.0, 0, 0, 0, 0, 0])
    dt = T_END / n_steps
    t_arr = np.linspace(0.0, T_END, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, 6))
    y_arr[0] = y0
    
    y = y0.copy()
    for i in range(n_steps):
        y = rk4_step(t_arr[i], y, case_num, dt)
        y_arr[i+1] = y
    return t_arr, y_arr

# =============================================================================
# 4. EXECUTION & ERROR CALCULATION
# =============================================================================
script_dir = Path(__file__).resolve().parent

models = {
    1: load_model(script_dir / "pinn_case1.pth", 1),
    2: load_model(script_dir / "pinn_case2.pth", 2),
    3: load_model(script_dir / "pinn_case3.pth", 3)
}

# 501 points to perfectly match the length of the 500-step RK4 array
t_dense = torch.linspace(0, T_END, 501, device=DEVICE).unsqueeze(1)
t_np = t_dense.cpu().numpy().flatten()

predictions = {}
rk4_data = {}  # Added to store RK4 curves for individual plotting
errors = {1: [], 2: [], 3: []}

with torch.no_grad():
    for case_num in [1, 2, 3]:
        pred = models[case_num](t_dense).cpu().numpy()
        predictions[case_num] = pred
        
        t_rk4, y_rk4 = simulate_ground_truth(case_num)
        rk4_data[case_num] = (t_rk4, y_rk4)
        
        for var_idx in range(6):
            denom = np.linalg.norm(y_rk4[:, var_idx]) + 1e-12
            l2_err = np.linalg.norm(pred[:, var_idx] - y_rk4[:, var_idx]) / denom
            errors[case_num].append(l2_err * 100.0) 

# =============================================================================
# 5. MASTER PLOT GENERATION (All Cases Together)
# =============================================================================
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3})

colors = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}
labels = {1: "Case 1: Base", 2: "Case 2: Zero IC", 3: "Case 3: Exp Decay"}

fig_master = plt.figure(figsize=(18, 12), constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig_master)
axes_master = [fig_master.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

fig_master.suptitle("PINN v3 — All 3 Cases — Apoptosis ODE", fontsize=18, fontweight="bold", y=1.02)

for idx, symbol, fullname in VAR_META:
    ax = axes_master[idx]
    for case_num in [1, 2, 3]:
        ax.plot(t_np, predictions[case_num][:, idx], 
                color=colors[case_num], linewidth=2.5, label=labels[case_num])
    
    err1, err2, err3 = errors[1][idx], errors[2][idx], errors[3][idx]
    title_text = (f"{fullname} ({symbol})\n"
                  f"Error: C1 [{err1:.2f}%] | C2 [{err2:.2f}%] | C3 [{err3:.2f}%]")
    
    ax.set_title(title_text, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Time t")
    ax.set_ylabel(symbol)
    ax.set_xlim(0, T_END)
    if idx == 0:
        ax.legend(loc="upper right", framealpha=0.9)

save_path_master = script_dir / "pinn_all_cases_evaluated.png"
fig_master.savefig(save_path_master, dpi=300, bbox_inches="tight")
print(f"Master plot successfully generated and saved to: {save_path_master}")
plt.close(fig_master)

# =============================================================================
# 6. INDIVIDUAL PLOT GENERATION (PINN vs RK4 per Case)
# =============================================================================
case_titles = {
    1: "Case 1: Base Parameters (IC: y_hif=1)",
    2: "Case 2: IC Independence (All-Zero ICs)",
    3: "Case 3: Time-Varying Coupling (Decaying α₁₂)"
}

for case_num in [1, 2, 3]:
    fig_indiv, axes_indiv = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig_indiv.suptitle(f"PINN vs RKF45 Ground Truth — {case_titles[case_num]}", fontsize=16, fontweight="bold")
    
    axes_indiv = axes_indiv.flatten()
    t_rk4, y_rk4 = rk4_data[case_num]
    
    # Thin out the RK4 points so the scatter plot doesn't look like a solid thick line
    rk4_every = max(1, len(t_rk4) // 50)
    
    for idx, symbol, fullname in VAR_META:
        ax = axes_indiv[idx]
        
        # 1. Plot continuous PINN line
        ax.plot(t_np, predictions[case_num][:, idx], 
                color=colors[case_num], linewidth=2.5, label="PINN (Continuous)")
        
        # 2. Plot discrete RK4 scatter points
        ax.scatter(t_rk4[::rk4_every], y_rk4[::rk4_every, idx], 
                   edgecolors='black', facecolors='none', s=40, linewidths=1.2, 
                   zorder=4, label="RK4 (Discrete Truth)")
        
        err = errors[case_num][idx]
        ax.set_title(f"{fullname} ({symbol})\nL2 Relative Error: {err:.3f}%", fontsize=11)
        ax.set_xlabel("Time t")
        ax.set_ylabel(symbol)
        ax.set_xlim(0, T_END)
        
        if idx == 0:
            ax.legend(loc="best", framealpha=0.9)
            
    save_path_indiv = script_dir / f"pinn_case{case_num}_individual.png"
    fig_indiv.savefig(save_path_indiv, dpi=300, bbox_inches="tight")
    print(f"Individual plot generated and saved to: {save_path_indiv}")
    plt.close(fig_indiv)

print("\n[DONE] All visuals have been generated successfully!")