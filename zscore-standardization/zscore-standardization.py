import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:

    X = np.asarray(X, dtype = float)

    mean = np.mean(X,axis = axis, keepdims = True)
    std =  np.std(X,axis = axis, keepdims = True)

    safe_std = np.where(std == 0, eps, std) 

    result = (X - mean)/ safe_std

    return result

    