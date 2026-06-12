import torch

# Biological Parameters for Case 1 (Base Case)
PARAMS = {
    'a_hif': 1.52, 'a_o2': 1.80, 'a_p53': 0.05,
    'a3': 0.90, 'a4': 0.20, 'a5': 0.001,
    'a7': 0.70, 'a8': 0.06, 'a9': 0.10,
    'a10': 0.70, 'a11': 0.20, 'a12': 0.10,
    'a13': 0.10, 'a14': 0.05
}

def physics_loss(model, t):
    # Ensure t tracks gradients for the backward pass
    t.requires_grad_(True)
    
    y_pred = model(t)
    
    # Unpack the 6 outputs
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = [y_pred[:, i] for i in range(6)]
    
    # Compute the derivatives of the outputs with respect to t
    dy_hif_dt = torch.autograd.grad(y_hif, t, grad_outputs=torch.ones_like(y_hif), create_graph=True)[0][:, 0]
    dy_o2_dt = torch.autograd.grad(y_o2, t, grad_outputs=torch.ones_like(y_o2), create_graph=True)[0][:, 0]
    dy_p300_dt = torch.autograd.grad(y_p300, t, grad_outputs=torch.ones_like(y_p300), create_graph=True)[0][:, 0]
    dy_p53_dt = torch.autograd.grad(y_p53, t, grad_outputs=torch.ones_like(y_p53), create_graph=True)[0][:, 0]
    dy_casp_dt = torch.autograd.grad(y_casp, t, grad_outputs=torch.ones_like(y_casp), create_graph=True)[0][:, 0]
    dy_kp_dt = torch.autograd.grad(y_kp, t, grad_outputs=torch.ones_like(y_kp), create_graph=True)[0][:, 0]
    
    # Define the ODE residuals
    f_hif = dy_hif_dt - (PARAMS['a_hif'] - PARAMS['a3']*y_o2*y_hif - PARAMS['a4']*y_hif*y_p300 - PARAMS['a7']*y_p53*y_hif)
    f_o2 = dy_o2_dt - (PARAMS['a_o2'] - PARAMS['a3']*y_o2*y_hif + PARAMS['a4']*y_hif*y_p300 - PARAMS['a11']*y_o2)
    f_p300 = dy_p300_dt - (PARAMS['a8'] - PARAMS['a4']*y_hif*y_p300 - PARAMS['a5']*y_p300*y_p53)
    f_p53 = dy_p53_dt - (PARAMS['a_p53'] - PARAMS['a5']*y_p300*y_p53 - PARAMS['a9']*y_p53)
    f_casp = dy_casp_dt - (PARAMS['a12'] + PARAMS['a9']*y_p53 - PARAMS['a13']*y_casp)
    f_kp = dy_kp_dt - (-PARAMS['a10']*y_casp*y_kp + PARAMS['a11']*y_o2 - PARAMS['a14']*y_kp)
    
    # Calculate Mean Squared Error
    loss_f = torch.mean(f_hif**2 + f_o2**2 + f_p300**2 + f_p53**2 + f_casp**2 + f_kp**2)
    return loss_f