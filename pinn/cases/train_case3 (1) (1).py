"""
train_case3.py
==============
Physics-Informed Neural Network (PINN) — Apoptosis Cascade
CASE 3: Time-Varying Parameter Experiment

Key difference from Cases 1 & 2:
    alpha_12 is NO LONGER constant. It decays exponentially:
        alpha_12(t) = 0.1 * exp(-0.05 * t)

    This means the coupling strength between variables weakens over time,
    producing dynamic transient behaviour rather than a static equilibrium.

Initial Conditions (same as Case 1):
    y(0) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Training Pipeline:
    Phase 1 — Adam   (8000 epochs) : navigates the dynamic, time-varying curve
    Phase 2 — L-BFGS (2000 iters)  : second-order physical convergence

Platform : Windows-native (pathlib throughout; CPU fallback guaranteed)
Reference: Schiesser, Chapter 3 — Apoptosis Biochemical Cascade
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 0.  DEVICE SETUP
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# =============================================================================
# 1.  SYSTEM PARAMETERS
#     Production & degradation rates are constant across all three cases.
#     Only alpha_12 changes — from a fixed scalar to a time-varying tensor.
# =============================================================================
A_HIF,  B_HIF  = 1.52, 3.05
A_O2,   B_O2   = 1.80, 2.10
A_P300, B_P300 = 1.35, 2.70
A_P53,  B_P53  = 1.60, 3.20
A_CASP, B_CASP = 1.20, 0.80
A_KP,   B_KP   = 0.90, 1.72

# Time-varying coupling — evaluated per-point inside the ODE function.
# alpha_12(t) = ALPHA_BASE * exp(-ALPHA_DECAY * t)
ALPHA_BASE  = 0.10
ALPHA_DECAY = 0.05

T_END = 100.0


def alpha12_torch(t: torch.Tensor) -> torch.Tensor:
    """
    Time-varying coupling coefficient (PyTorch version for autograd).

    Args:
        t : (N, 1) collocation time tensor

    Returns:
        a12 : (N, 1) coupling values at each time point
    """
    return ALPHA_BASE * torch.exp(-ALPHA_DECAY * t)


# =============================================================================
# 2.  APOPTOSIS ODE RIGHT-HAND SIDE  (time-varying alpha_12)
# =============================================================================
def apoptosis_odes(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    RHS of the 6-variable apoptosis system with time-varying alpha_12(t).

    The coupling term is now a FUNCTION of t, evaluated at each collocation
    point. This requires t to be passed through explicitly — unlike Cases 1/2
    where ALPHA_12 was a module-level constant.

    ODEs:
        dy_hif /dt = a_hif  - b_hif *y_hif  + α(t)*y_kp   - y_hif*y_o2
        dy_o2  /dt = a_o2   - b_o2  *y_o2   + α(t)*y_hif
        dy_p300/dt = a_p300 - b_p300*y_p300 + α(t)*y_hif
        dy_p53 /dt = a_p53  - b_p53 *y_p53  + α(t)*y_p300 - y_p53*y_casp
        dy_casp/dt = a_casp - b_casp*y_casp + α(t)*y_p53
        dy_kp  /dt = a_kp   - b_kp  *y_kp   + α(t)*y_p53

    Args:
        t : (N, 1) time tensor — REQUIRED for time-varying alpha_12
        y : (N, 6) state tensor

    Returns:
        dydt : (N, 6) time-derivatives
    """
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = (y[:, i:i+1] for i in range(6))

    # ── Evaluate coupling at each collocation time ────────────────────────
    a12 = alpha12_torch(t)    # (N, 1) — decays as t increases

    d_hif  = A_HIF  - B_HIF  * y_hif  + a12 * y_kp   - y_hif  * y_o2
    d_o2   = A_O2   - B_O2   * y_o2   + a12 * y_hif
    d_p300 = A_P300 - B_P300 * y_p300 + a12 * y_hif
    d_p53  = A_P53  - B_P53  * y_p53  + a12 * y_p300 - y_p53  * y_casp
    d_casp = A_CASP - B_CASP * y_casp + a12 * y_p53
    d_kp   = A_KP   - B_KP   * y_kp   + a12 * y_p53

    return torch.cat([d_hif, d_o2, d_p300, d_p53, d_casp, d_kp], dim=1)


# =============================================================================
# 3.  NEURAL NETWORK ARCHITECTURE
# =============================================================================
class ApoptosisPINN(nn.Module):
    """
    Fully-connected PINN: t ∈ ℝ  →  [y_hif, y_o2, y_p300, y_p53, y_casp, y_kp] ∈ ℝ⁶

    Architecture: 1 → [128]×6 → 6, Tanh activations.

    The time-varying alpha_12 introduces higher-frequency content into the
    solution trajectory. A deeper/wider network with Tanh (rather than ReLU)
    is better suited to represent smooth but non-monotone dynamics.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 6))

        self.net = nn.Sequential(*layers)

        # Xavier init — important for Case 3 since the IC is non-zero (y_hif=1)
        # and the trajectory has more curvature than Cases 1 & 2
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# =============================================================================
# 4.  LOSS FUNCTION
# =============================================================================
def compute_loss(
    model:  nn.Module,
    t_col:  torch.Tensor,      # (N_col, 1)  collocation times
    t_ic:   torch.Tensor,      # (1, 1)      t = 0
    w_ode:  float = 1.0,
    w_ic:   float = 1000.0,    # heavy weight — anchors the non-zero IC firmly
) -> tuple[torch.Tensor, dict]:
    """
    PINN total loss = w_ode * Loss_ODE  +  w_ic * Loss_IC

    Loss_ODE:
        Penalises deviation from the physics at every collocation point.
        Because alpha_12 is time-varying, t_col must be forwarded into the
        ODE RHS explicitly — not just the state vector y.

    Loss_IC:
        Enforces y(0) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]   (Case 3 = Case 1)

    Args:
        model   : ApoptosisPINN
        t_col   : collocation times (requires_grad=True for autograd)
        t_ic    : t=0 point
        w_ode   : ODE residual weight
        w_ic    : IC enforcement weight

    Returns:
        total_loss : differentiable scalar
        loss_dict  : breakdown dict for logging
    """
    # ── ODE Residual ──────────────────────────────────────────────────────
    y_col = model(t_col)        # (N_col, 6)

    # Compute dy/dt via automatic differentiation w.r.t. t_col
    dydt = torch.zeros_like(y_col)
    for i in range(6):
        grad = torch.autograd.grad(
            y_col[:, i].sum(), t_col,
            create_graph=True, retain_graph=True
        )[0]
        dydt[:, i:i+1] = grad

    # Pass BOTH t_col and y_col — RHS needs t for alpha_12(t)
    rhs      = apoptosis_odes(t_col, y_col)   # (N_col, 6)
    residual = dydt - rhs
    loss_ode = torch.mean(residual ** 2)

    # ── Initial Condition Loss (CASE 3 = CASE 1 IC) ───────────────────────
    # y(0) = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y_ic_pred = model(t_ic)                              # (1, 6)
    ic_target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32, device=DEVICE
    )                                                    # ← Case 3 IC vector
    loss_ic = torch.mean((y_ic_pred - ic_target) ** 2)

    # ── Weighted Total ─────────────────────────────────────────────────────
    total_loss = w_ode * loss_ode + w_ic * loss_ic

    return total_loss, {
        "total": total_loss.item(),
        "ode":   loss_ode.item(),
        "ic":    loss_ic.item(),
    }


# =============================================================================
# 5.  COLLOCATION POINT GENERATION
# =============================================================================
def make_collocation_points(
    n_points:   int   = 5000,
    t_end:      float = T_END,
    strategy:   str   = "random",
) -> torch.Tensor:
    """
    Sample collocation times in [0, t_end].

    "random" is preferred for Case 3 because the time-varying parameter
    creates non-uniform curvature: early times [0, 20] have high coupling
    (alpha_12 ≈ 0.1 → 0.037) while late times [80, 100] are nearly decoupled
    (alpha_12 ≈ 0.018). Random sampling naturally spreads coverage.

    Returns:
        t : (n_points, 1) tensor on DEVICE, requires_grad=True
    """
    if strategy == "random":
        t = torch.rand(n_points, 1, device=DEVICE) * t_end
        t, _ = torch.sort(t, dim=0)
    elif strategy == "uniform":
        t = torch.linspace(0.0, t_end, n_points, device=DEVICE).unsqueeze(1)
    elif strategy == "chebyshev":
        k = torch.arange(1, n_points + 1, dtype=torch.float32, device=DEVICE)
        t = 0.5 * t_end * (1 - torch.cos(torch.pi * (2*k - 1) / (2*n_points)))
        t = t.unsqueeze(1)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    t.requires_grad_(True)
    return t


# =============================================================================
# 6.  TRAINING ROUTINE
# =============================================================================
def train(
    # Network
    hidden_dim:      int   = 128,
    num_layers:      int   = 6,
    # Collocation
    n_col:           int   = 5000,
    col_strategy:    str   = "random",
    # Phase 1 — Adam
    adam_epochs:     int   = 8000,
    adam_lr:         float = 1e-3,
    # Phase 2 — L-BFGS
    lbfgs_epochs:    int   = 2000,
    lbfgs_lr:        float = 0.5,
    lbfgs_max_iter:  int   = 100,
    # Loss weights
    w_ode:           float = 1.0,
    w_ic:            float = 1000.0,
    # I/O
    log_every:       int   = 500,
    save_dir:        Path  = Path("."),
) -> "ApoptosisPINN":
    """
    Dual-optimizer training pipeline for Case 3.

    Phase 1 — Adam:
        The time-varying alpha_12 creates a loss landscape that transitions
        from highly coupled (early t) to nearly independent (late t). Adam's
        adaptive learning rate handles this non-stationary gradient magnitude.

    Phase 2 — L-BFGS:
        Refines the solution curve with second-order precision. The strong
        Wolfe line search is critical on CPU where step sizes need careful control.

    Saves: <save_dir>/pinn_case3.pth
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "pinn_case3.pth"   # pathlib — Windows-safe

    # ── Model ─────────────────────────────────────────────────────────────
    model = ApoptosisPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")

    # ── Fixed IC point ────────────────────────────────────────────────────
    t_ic = torch.zeros(1, 1, device=DEVICE)

    # ── Initial collocation grid ──────────────────────────────────────────
    t_col = make_collocation_points(n_col, strategy=col_strategy)

    # =========================================================================
    # PHASE 1  —  Adam
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1 — Adam Optimiser")
    print("=" * 60)

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    # Decay LR every 2000 steps — prevents overshooting the dynamic curve
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_adam, step_size=2000, gamma=0.5
    )

    t0 = time.time()
    for epoch in range(1, adam_epochs + 1):
        optimizer_adam.zero_grad()

        # Refresh random collocation points periodically for diversity
        if epoch % 1000 == 0:
            t_col = make_collocation_points(n_col, strategy=col_strategy)

        loss, ld = compute_loss(model, t_col, t_ic, w_ode, w_ic)
        loss.backward()
        optimizer_adam.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  [Adam] Epoch {epoch:>6d}/{adam_epochs} | "
                f"Total: {ld['total']:.4e} | "
                f"ODE: {ld['ode']:.4e} | "
                f"IC: {ld['ic']:.4e} | "
                f"Elapsed: {time.time()-t0:.1f}s"
            )

    # =========================================================================
    # PHASE 2  —  L-BFGS
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2 — L-BFGS Optimiser")
    print("=" * 60)

    # Switch to uniform grid for Phase 2 — L-BFGS benefits from deterministic,
    # evenly-spaced residual evaluation points
    t_col = make_collocation_points(n_col, strategy="uniform")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    loss_cache = [None]

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, ld = compute_loss(model, t_col, t_ic, w_ode, w_ic)
        loss.backward()
        loss_cache[0] = ld
        return loss

    t0 = time.time()
    for epoch in range(1, lbfgs_epochs + 1):
        optimizer_lbfgs.step(closure)

        if epoch % log_every == 0 or epoch == 1:
            ld = loss_cache[0]
            print(
                f"  [LBFGS] Epoch {epoch:>5d}/{lbfgs_epochs} | "
                f"Total: {ld['total']:.4e} | "
                f"ODE: {ld['ode']:.4e} | "
                f"IC: {ld['ic']:.4e} | "
                f"Elapsed: {time.time()-t0:.1f}s"
            )

    # =========================================================================
    # SAVE CHECKPOINT
    # =========================================================================
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim":  hidden_dim,
            "num_layers":  num_layers,
            "case":        3,
            "ic":          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "alpha_base":  ALPHA_BASE,
            "alpha_decay": ALPHA_DECAY,
            "t_end":       T_END,
        },
        checkpoint_path,
    )
    print(f"\n[INFO] Checkpoint saved → {checkpoint_path.resolve()}")

    # ── Sanity check: final values at t=100 ──────────────────────────────
    model.eval()
    with torch.no_grad():
        t_final = torch.tensor([[T_END]], dtype=torch.float32, device=DEVICE)
        y_final = model(t_final).cpu().numpy().flatten()

    var_names = ["y_hif", "y_o2", "y_p300", "y_p53", "y_casp", "y_kp"]
    print("\n[SANITY] Predicted values at t=100 (Case 3 — NOT same equilibrium as Cases 1/2):")
    for name, val in zip(var_names, y_final):
        print(f"  {name:>8s} = {val:.6f}")

    return model


# =============================================================================
# 7.  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    train(
        hidden_dim      = 128,
        num_layers      = 6,
        n_col           = 5000,
        col_strategy    = "random",
        adam_epochs     = 8000,
        adam_lr         = 1e-3,
        lbfgs_epochs    = 2000,
        lbfgs_lr        = 0.5,
        lbfgs_max_iter  = 100,
        w_ode           = 1.0,
        w_ic            = 1000.0,
        log_every       = 500,
        save_dir        = script_dir,
    )

    print("\n[DONE] Case 3 training complete.")
