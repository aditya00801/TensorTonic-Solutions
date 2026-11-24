import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    
    grad = np.asarray(grad)

    # Update first moment
    new_m =  beta1 * m + (1 - beta1) * grad

    # update second moment 
    new_v  = beta2 * v + (1 - beta2)*(grad ** 2)

    # Calculate Nesterov-adjusted first moment
    m_hat  = beta1 * new_m + (1-beta1) * grad

     # Update weights using adaptive learning rate

    new_w = w - lr* m_hat/ np.sqrt(new_v) + eps

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }
    