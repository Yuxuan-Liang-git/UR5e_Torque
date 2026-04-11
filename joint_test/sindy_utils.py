import numpy as np

def pool_data_motor(yin, poly_order=2, include_sign=True):
    """
    Constructs a library of candidate functions for SINDy based on motor dynamics.
    yin: [q, dq, u] - states and control inputs
    """
    n_samples, n_vars = yin.shape
    
    # 1. Constant term (1)
    out = [np.ones((n_samples, 1))]
    
    # 2. Polynomial order 1
    out.append(yin)
    
    # 3. Polynomial order 2
    if poly_order >= 2:
        for i in range(n_vars):
            for j in range(i, n_vars):
                out.append((yin[:, i] * yin[:, j]).reshape(-1, 1))
                
    # 4. Polynomial order 3
    if poly_order >= 3:
        for i in range(n_vars):
            for j in range(i, n_vars):
                for k in range(j, n_vars):
                    out.append((yin[:, i] * yin[:, j] * yin[:, k]).reshape(-1, 1))
                    
    # 5. Add Coulomb Friction term: sign(dq)
    # Based on MATLAB poolDataMotor: dq is item 1 (index starting from 0) if yin = [q, dq, u]
    if include_sign:
        dq = yin[:, 1].reshape(-1, 1)
        out.append(np.sign(dq))
        
    # 6. Add Stribeck-related terms (Exponential decay)
    # tau_f = (Fs - Fc) * exp(-(dq/vs)^2) * sign(dq)
    # We can add a few candidates with different vs values or use a specific one
    # Here we use vs = 0.05 as a default characteristic velocity or a list
    vs_candidates = [0.01, 0.05, 0.1]
    for vs in vs_candidates:
        dq = yin[:, 1].reshape(-1, 1)
        stribeck_term = np.exp(-(dq / vs)**2) * np.sign(dq)
        out.append(stribeck_term)
        
    return np.hstack(out)

def sparsify_dynamics(Theta, dXdt, lambdas, n_states, alpha_ridge=1e-4):
    """
    Sequential Thresholded Least Squares (STLS) with normalization and Ridge regularization.
    Theta: Library matrix
    dXdt: Derivative matrix (target)
    lambdas: List of thresholds for each state
    n_states: Number of states to identify
    """
    n_lib = Theta.shape[1]
    
    # 1. Normalization
    Theta_norm = np.linalg.norm(Theta, axis=0)
    Theta_norm[Theta_norm == 0] = 1.0
    Theta_reg = Theta / Theta_norm
    
    dXdt_norm = np.linalg.norm(dXdt, axis=0)
    dXdt_norm[dXdt_norm == 0] = 1.0
    dXdt_reg = dXdt / dXdt_norm
    
    # 2. Initial guess using Ridge Regression
    # (Theta'Theta + alpha*I) * Xi = Theta'dXdt
    # Note: Using normalized matrices
    A = Theta_reg.T @ Theta_reg + alpha_ridge * np.eye(n_lib)
    B = Theta_reg.T @ dXdt_reg
    Xi = np.zeros((n_lib, n_states))
    
    for i in range(n_states):
        Xi[:, i] = np.linalg.solve(A, B[:, i])
    
    # 3. Iterative Thresholding
    for _ in range(10):
        for i in range(n_states):
            # Thresholding
            small_inds = np.abs(Xi[:, i]) < lambdas[i]
            Xi[small_inds, i] = 0
            
            # Re-regress on big indices
            big_inds = ~small_inds
            if np.any(big_inds):
                # Solving only for selected terms
                sub_Theta = Theta_reg[:, big_inds]
                sub_A = sub_Theta.T @ sub_Theta + alpha_ridge * np.eye(np.sum(big_inds))
                sub_B = sub_Theta.T @ dXdt_reg[:, i]
                Xi[big_inds, i] = np.linalg.solve(sub_A, sub_B)
                
    # 4. Denormalization
    # Xi_coeff = (D_dx / D_Theta) * Xi_norm
    for i in range(n_states):
        Xi[:, i] = Xi[:, i] * dXdt_norm[i]
        
    for j in range(n_lib):
        Xi[j, :] = Xi[j, :] / Theta_norm[j]
        
    return Xi

def get_library_labels(vars_names, poly_order, include_sign=True):
    """
    Generates string labels for the library columns.
    """
    labels = ["1"]
    
    # Order 1
    labels.extend(vars_names)
    
    # Order 2
    if poly_order >= 2:
        for i in range(len(vars_names)):
            for j in range(i, len(vars_names)):
                labels.append(f"{vars_names[i]}*{vars_names[j]}")
                
    # Order 3
    if poly_order >= 3:
        for i in range(len(vars_names)):
            for j in range(i, len(vars_names)):
                for k in range(j, len(vars_names)):
                    labels.append(f"{vars_names[i]}*{vars_names[j]}*{vars_names[k]}")
                    
    if include_sign:
        labels.append(f"sign({vars_names[1]})") # assumes dq is index 1
        
    # Stribeck labels
    vs_candidates = [0.01, 0.05, 0.1]
    for vs in vs_candidates:
        labels.append(f"exp(-(dq/{vs})^2)*sign(dq)")
        
    return labels
