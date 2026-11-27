import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    train = np.asarray(X_train, dtype=float)
    test = np.asarray(X_test, dtype=float)

    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    indices = np.full((len(test), k), -1, dtype=int)
    n = min(k, len(train))

    if n:
        distances = np.linalg.norm(test[:, None] - train[None, :], axis=2)
        indices[:, :n] = np.argsort(distances, axis=1, kind="stable")[:, :n]

    return indices