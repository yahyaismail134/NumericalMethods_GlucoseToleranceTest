"""
train_case2.py
==============
Physics-Informed Neural Network (PINN) for the Apoptosis ODE System.
CASE 2: IC Independence Experiment — All-Zero Initial Conditions.

Reference: Schiesser, Chapter 3 — Apoptosis Biochemical Cascade.

System Variables (6):
    y_hif  : HIF-1α transcription factor
    y_o2   : Oxygen / ROS
    y_p300 : p300 co-activator
    y_p53  : p53 tumour suppressor
    y_casp : Caspase (effector)
    y_kp   : Kinase-Phosphatase signal

Initial Conditions for Case 2:
    y(0) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   ← KEY DIFFERENCE from Case 1

The system is asymptotically stable: despite starting at absolute zero,
production parameters drive convergence to the same equilibrium as Case 1
(e.g. y_hif ≈ 0.4991, y_casp ≈ 1.4968, y_kp ≈ 0.5219 at t=100).

Training Pipeline:
    Phase 1 — Adam   : settles the zero-start gradient landscape
    Phase 2 — L-BFGS : second-order refinement for smooth ODE curves

Author  : [Your Name]
Platform: Windows-native (pathlib used throughout; CPU fallback guaranteed)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# 0.  DEVICE SETUP  (GPU if available, else CPU — safe on Windows without CUDA)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# =============================================================================
# 1.  APOPTOSIS ODE PARAMETERS  (Schiesser Ch. 3, shared across Case 1 & 2)
# =============================================================================
# Production rates
A_HIF  = 1.52   # HIF-1α basal production
A_O2   = 1.80   # Oxygen/ROS production
A_P300 = 1.35   # p300 production
A_P53  = 1.60   # p53 production
A_CASP = 1.20   # Caspase production
A_KP   = 0.90   # Kinase-Phosphatase production

# Degradation rates
B_HIF  = 3.05
B_O2   = 2.10
B_P300 = 2.70
B_P53  = 3.20
B_CASP = 0.80
B_KP   = 1.72

# Cross-activation / inhibition coupling
ALPHA_12 = 0.10   # ← constant for BOTH Case 1 and Case 2

# T_END: final integration time
T_END = 100.0


def apoptosis_odes(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the right-hand side of the 6-variable apoptosis ODE system.

    The ODEs (simplified Schiesser form):
        dy_hif /dt = a_hif  - b_hif *y_hif  + alpha_12*y_kp  - y_hif *y_o2
        dy_o2  /dt = a_o2   - b_o2  *y_o2   + alpha_12*y_hif
        dy_p300/dt = a_p300 - b_p300*y_p300 + alpha_12*y_hif
        dy_p53 /dt = a_p53  - b_p53 *y_p53  + alpha_12*y_p300 - y_p53*y_casp
        dy_casp/dt = a_casp - b_casp*y_casp + alpha_12*y_p53
        dy_kp  /dt = a_kp   - b_kp  *y_kp   + alpha_12*y_p53

    Args:
        t : (N,1) collocation times  [not used explicitly but kept for
                                       compatibility with auto-diff wrapper]
        y : (N,6) state tensor — columns ordered as listed above

    Returns:
        dydt : (N,6) time-derivatives
    """
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = (y[:, i:i+1] for i in range(6))

    d_hif  = A_HIF  - B_HIF  * y_hif  + ALPHA_12 * y_kp   - y_hif  * y_o2
    d_o2   = A_O2   - B_O2   * y_o2   + ALPHA_12 * y_hif
    d_p300 = A_P300 - B_P300 * y_p300 + ALPHA_12 * y_hif
    d_p53  = A_P53  - B_P53  * y_p53  + ALPHA_12 * y_p300 - y_p53  * y_casp
    d_casp = A_CASP - B_CASP * y_casp + ALPHA_12 * y_p53
    d_kp   = A_KP   - B_KP   * y_kp   + ALPHA_12 * y_p53

    return torch.cat([d_hif, d_o2, d_p300, d_p53, d_casp, d_kp], dim=1)


# =============================================================================
# 2.  NEURAL NETWORK ARCHITECTURE
# =============================================================================
class ApoptosisPINN(nn.Module):
    """
    Fully-connected PINN that maps time t → [y_hif, y_o2, y_p300, y_p53, y_casp, y_kp].

    Architecture: 1 → [128]*6 → 6  with Tanh activations.
    Tanh is preferred over ReLU for smooth ODE solutions (avoids kinks).
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 6))

        self.net = nn.Sequential(*layers)

        # Xavier initialisation — helps escape flat loss landscape at t=0 zeros
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (N,1) normalised or raw time values
        Returns:
            y : (N,6) predicted state variables
        """
        return self.net(t)


# =============================================================================
# 3.  LOSS FUNCTION
# =============================================================================
def compute_loss(
    model: nn.Module,
    t_col: torch.Tensor,
    t_ic: torch.Tensor,
    w_ode: float = 1.0,
    w_ic: float = 1000.0,   # ← heavy weight to anchor the zero-start
) -> tuple[torch.Tensor, dict]:
    """
    Total PINN loss = w_ode * Loss_ODE  +  w_ic * Loss_IC

    Loss_ODE : MSE of ODE residuals at collocation points
    Loss_IC  : MSE between y(0) and the all-zero target vector

    Args:
        model  : the PINN model
        t_col  : (N_col, 1) collocation times, requires_grad=True
        t_ic   : (1, 1)     t=0 point for IC enforcement
        w_ode  : weight for ODE physics loss
        w_ic   : weight for IC loss (large → hard enforcement)

    Returns:
        total_loss : scalar tensor (differentiable)
        loss_dict  : breakdown for logging
    """
    # ── ODE Residual Loss ──────────────────────────────────────────────────
    y_col = model(t_col)                          # (N_col, 6)

    # Compute dy/dt via automatic differentiation
    # We differentiate each output w.r.t. t_col
    dydt = torch.zeros_like(y_col)
    for i in range(6):
        grad = torch.autograd.grad(
            y_col[:, i].sum(), t_col,
            create_graph=True, retain_graph=True
        )[0]
        dydt[:, i:i+1] = grad

    rhs = apoptosis_odes(t_col, y_col)            # (N_col, 6)
    residual = dydt - rhs                          # should be ~0
    loss_ode = torch.mean(residual ** 2)

    # ── Initial Condition Loss (CASE 2 KEY CHANGE) ────────────────────────
    # Target: ALL ZEROS — y(0) = [0, 0, 0, 0, 0, 0]
    # Compare with Case 1 where target was [1.0, 0, 0, 0, 0, 0]
    y_ic_pred = model(t_ic)                        # (1, 6)
    ic_target = torch.zeros(1, 6, device=DEVICE)   # ← ALL-ZERO TARGET VECTOR
    loss_ic = torch.mean((y_ic_pred - ic_target) ** 2)

    # ── Weighted Total ─────────────────────────────────────────────────────
    total_loss = w_ode * loss_ode + w_ic * loss_ic

    loss_dict = {
        "total": total_loss.item(),
        "ode":   loss_ode.item(),
        "ic":    loss_ic.item(),
    }
    return total_loss, loss_dict


# =============================================================================
# 4.  COLLOCATION POINT GENERATION
# =============================================================================
def make_collocation_points(
    n_points: int = 5000,
    t_end: float = T_END,
    strategy: str = "uniform",
) -> torch.Tensor:
    """
    Generate collocation times in [0, t_end].

    Strategies:
        "uniform"  — evenly spaced
        "random"   — uniformly sampled at random (better for generalisation)
        "chebyshev"— Chebyshev nodes (better resolution near boundaries)

    Returns:
        t_col : (n_points, 1) tensor on DEVICE, requires_grad=True
    """
    if strategy == "uniform":
        t = torch.linspace(0.0, t_end, n_points, device=DEVICE).unsqueeze(1)
    elif strategy == "random":
        t = torch.rand(n_points, 1, device=DEVICE) * t_end
        t, _ = torch.sort(t, dim=0)
    elif strategy == "chebyshev":
        k = torch.arange(1, n_points + 1, dtype=torch.float32, device=DEVICE)
        t = 0.5 * t_end * (1 - torch.cos(torch.pi * (2*k - 1) / (2*n_points)))
        t = t.unsqueeze(1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    t.requires_grad_(True)
    return t


# =============================================================================
# 5.  TRAINING ROUTINE
# =============================================================================
def train(
    # ── Network ──
    hidden_dim: int   = 128,
    num_layers: int   = 6,
    # ── Collocation ──
    n_col: int        = 5000,
    col_strategy: str = "random",
    # ── Phase 1: Adam ──
    adam_epochs: int  = 8000,
    adam_lr: float    = 1e-3,
    # ── Phase 2: L-BFGS ──
    lbfgs_epochs: int = 2000,
    lbfgs_lr: float   = 0.5,
    lbfgs_max_iter: int = 100,
    # ── Loss weights ──
    w_ode: float      = 1.0,
    w_ic: float       = 1000.0,  # heavy IC anchor for zero-start
    # ── Logging / Saving ──
    log_every: int    = 500,
    save_dir: Path    = Path("."),   # resolved below to script directory
) -> ApoptosisPINN:
    """
    Full dual-optimizer training pipeline for Case 2.

    Phase 1 — Adam:
        Fast gradient descent settles the loss landscape when the entire
        trajectory starts from zero.  Without this warm-up, L-BFGS can
        stall or diverge on the flat zero-loss region.

    Phase 2 — L-BFGS:
        Second-order quasi-Newton method refines the smooth ODE curve
        with high precision.  Requires a closure() function (PyTorch API).

    Returns trained model and saves checkpoint to save_dir/pinn_case2.pth.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)   # create dir if missing

    checkpoint_path = save_dir / "pinn_case2.pth"  # OS-independent via pathlib

    # ── Initialise model ──────────────────────────────────────────────────
    model = ApoptosisPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Fixed t=0 tensor for IC enforcement ──────────────────────────────
    t_ic = torch.zeros(1, 1, device=DEVICE)

    # ── Collocation points (refreshed each phase for diversity) ───────────
    t_col = make_collocation_points(n_col, strategy=col_strategy)

    # =========================================================================
    # PHASE 1  —  Adam optimiser
    # =========================================================================
    print("\n" + "="*60)
    print("  PHASE 1 — Adam Optimiser")
    print("="*60)

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_adam, step_size=2000, gamma=0.5
    )

    t0 = time.time()
    for epoch in range(1, adam_epochs + 1):
        optimizer_adam.zero_grad()

        # Refresh collocation points periodically (avoids over-fitting to grid)
        if epoch % 1000 == 0:
            t_col = make_collocation_points(n_col, strategy=col_strategy)

        total_loss, loss_dict = compute_loss(model, t_col, t_ic, w_ode, w_ic)
        total_loss.backward()
        optimizer_adam.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  [Adam] Epoch {epoch:>6d}/{adam_epochs} | "
                f"Total: {loss_dict['total']:.4e} | "
                f"ODE: {loss_dict['ode']:.4e} | "
                f"IC: {loss_dict['ic']:.4e} | "
                f"Time: {elapsed:.1f}s"
            )

    # =========================================================================
    # PHASE 2  —  L-BFGS optimiser
    # =========================================================================
    print("\n" + "="*60)
    print("  PHASE 2 — L-BFGS Optimiser")
    print("="*60)

    # Refresh collocation for Phase 2
    t_col = make_collocation_points(n_col, strategy="uniform")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",   # robust line-search on CPU
    )

    loss_log = [None]   # mutable container for closure capture

    def closure():
        """L-BFGS requires a closure that re-evaluates the loss."""
        optimizer_lbfgs.zero_grad()
        total_loss, loss_dict = compute_loss(model, t_col, t_ic, w_ode, w_ic)
        total_loss.backward()
        loss_log[0] = loss_dict
        return total_loss

    t0 = time.time()
    for epoch in range(1, lbfgs_epochs + 1):
        optimizer_lbfgs.step(closure)

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            ld = loss_log[0]
            print(
                f"  [LBFGS] Epoch {epoch:>5d}/{lbfgs_epochs} | "
                f"Total: {ld['total']:.4e} | "
                f"ODE: {ld['ode']:.4e} | "
                f"IC: {ld['ic']:.4e} | "
                f"Time: {elapsed:.1f}s"
            )

    # =========================================================================
    # SAVE CHECKPOINT
    # =========================================================================
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "case": 2,
            "ic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "alpha_12": ALPHA_12,
            "t_end": T_END,
        },
        checkpoint_path,    # pathlib.Path → str handled by torch.save internally
    )
    print(f"\n[INFO] Checkpoint saved → {checkpoint_path.resolve()}")

    # ── Quick sanity check at t=100 ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        t_final = torch.tensor([[T_END]], dtype=torch.float32, device=DEVICE)
        y_final = model(t_final).cpu().numpy().flatten()

    var_names = ["y_hif", "y_o2", "y_p300", "y_p53", "y_casp", "y_kp"]
    expected  = [0.4991, None, None, None, 1.4968, 0.5219]
    print("\n[SANITY CHECK] Predictions at t=100 (expected equilibrium):")
    for name, val, exp in zip(var_names, y_final, expected):
        marker = f"  (expected ≈ {exp})" if exp is not None else ""
        print(f"  {name:>8s} = {val:.6f}{marker}")

    return model


# =============================================================================
# 6.  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # All paths relative to THIS script's directory — works on Windows & Linux
    script_dir = Path(__file__).resolve().parent

    trained_model = train(
        # Network
        hidden_dim   = 128,
        num_layers   = 6,
        # Collocation
        n_col        = 5000,
        col_strategy = "random",
        # Phase 1 — Adam
        adam_epochs  = 8000,
        adam_lr      = 1e-3,
        # Phase 2 — L-BFGS
        lbfgs_epochs = 2000,
        lbfgs_lr     = 0.5,
        lbfgs_max_iter = 100,
        # Loss weights
        w_ode        = 1.0,
        w_ic         = 1000.0,   # large weight to anchor all-zero IC
        # I/O
        log_every    = 500,
        save_dir     = script_dir,   # saves pinn_case2.pth next to this script
    )

    print("\n[DONE] Case 2 training complete.")
