import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    eps = 1e-7

    p = np.clip(np.array(p), eps, 1 - eps)
    y = np.array(y)

    loss = (
        -((1 - p) ** gamma) * y * np.log(p)
        -(p ** gamma) * (1 - y) * np.log(1 - p)
    )

    return float(np.mean(loss))