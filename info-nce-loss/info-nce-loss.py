import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)

    #  Calculate similarity matrix

    S = (Z1 @ (Z2.T))/ temperature

    #Subtract maximum of each row
    raw_max = np.max(S, axis = 1, keepdims = True)
    Sub = S - raw_max # numerical stabilization.

    exp_S = np.exp(Sub)

    # Numerator: Positive pair
    numerator = np.diag(exp_S)
    
    # Denominator: Sum of each row
    denominator = np.sum(exp_S, axis=1)

    # Calculate probabilities
    probabilities =  numerator / denominator

    losses = -np.log(probabilities)

    return float(np.mean(losses))

    
    

    



    