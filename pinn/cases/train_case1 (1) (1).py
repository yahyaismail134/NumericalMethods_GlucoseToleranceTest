import torch
import torch.nn as nn
import torch.optim as optim
import time

# =============================================================================
# 1. UNIFIED ARCHITECTURE (128 Neurons, 6 Layers, Softplus)
# =============================================================================
class ApoptosisPINN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super(ApoptosisPINN, self).__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 6))
        layers.append(nn.Softplus()) # Prevents negative concentrations
        
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

# =============================================================================
# 2. THE NEW PHYSICS ENGINE (Matches Cases 2 & 3)
# =============================================================================
A_HIF,  B_HIF  = 1.52, 3.05
A_O2,   B_O2   = 1.80, 2.10
A_P300, B_P300 = 1.35, 2.70
A_P53,  B_P53  = 1.60, 3.20
A_CASP, B_CASP = 1.20, 0.80
A_KP,   B_KP   = 0.90, 1.72
ALPHA_12       = 0.10  # Constant for Case 1

def physics_loss(model, t):
    t.requires_grad_(True)
    y_pred = model(t)
    
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = [y_pred[:, i:i+1] for i in range(6)]
    
    # Calculate gradients
    dydt = torch.zeros_like(y_pred)
    for i in range(6):
        dydt[:, i:i+1] = torch.autograd.grad(y_pred[:, i].sum(), t, create_graph=True)[0]
    
    dy_hif_dt, dy_o2_dt, dy_p300_dt, dy_p53_dt, dy_casp_dt, dy_kp_dt = [dydt[:, i:i+1] for i in range(6)]
    
    # NEW Simplified ODEs
    f_hif  = dy_hif_dt - (A_HIF - B_HIF*y_hif + ALPHA_12*y_kp - y_hif*y_o2)
    f_o2   = dy_o2_dt - (A_O2 - B_O2*y_o2 + ALPHA_12*y_hif)
    f_p300 = dy_p300_dt - (A_P300 - B_P300*y_p300 + ALPHA_12*y_hif)
    f_p53  = dy_p53_dt - (A_P53 - B_P53*y_p53 + ALPHA_12*y_p300 - y_p53*y_casp)
    f_casp = dy_casp_dt - (A_CASP - B_CASP*y_casp + ALPHA_12*y_p53)
    f_kp   = dy_kp_dt - (A_KP - B_KP*y_kp + ALPHA_12*y_p53)
    
    return torch.mean(f_hif**2 + f_o2**2 + f_p300**2 + f_p53**2 + f_casp**2 + f_kp**2)

# =============================================================================
# 3. TRAINING PIPELINE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Case 1 on device: {device}")

hidden_dim, num_layers = 128, 6
model = ApoptosisPINN(hidden_dim, num_layers).to(device)

t_collocation = (torch.rand(5000, 1) * 100.0).to(device).requires_grad_(True)

# Case 1 Initial Conditions: y_hif = 1.0, rest = 0.0
t_ic = torch.tensor([[0.0]], device=device).requires_grad_(True)
y_ic_target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)

def total_loss_fn():
    loss_phys = physics_loss(model, t_collocation)
    y_ic_pred = model(t_ic)
    loss_ic = torch.mean((y_ic_pred - y_ic_target)**2)
    return loss_phys + (1000.0 * loss_ic)

# --- Phase 1: Adam ---
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
print("\n--- Starting Phase 1: Adam ---")
for epoch in range(8000): # Matched to 8000 for consistency
    optimizer_adam.zero_grad()
    loss = total_loss_fn()
    loss.backward()
    optimizer_adam.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch:04d} | Total Loss: {loss.item():.6f}')

# --- Phase 2: L-BFGS ---
optimizer_lbfgs = optim.LBFGS(
    model.parameters(), 
    lr=0.1, 
    max_iter=10000, 
    tolerance_grad=1e-7, 
    tolerance_change=1e-9, 
    history_size=50,
    line_search_fn="strong_wolfe" # Prevents NaN explosions
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = total_loss_fn()
    loss.backward()
    return loss

print("\n--- Starting Phase 2: L-BFGS ---")
optimizer_lbfgs.step(closure)

print(f'\nFinal Total Loss: {total_loss_fn().item():.6f}')

# Save the unified weights
torch.save({
    "model_state_dict": model.state_dict(),
    "hidden_dim": hidden_dim,
    "num_layers": num_layers
}, "pinn_case1.pth")
print("Model saved to pinn_case1.pth")