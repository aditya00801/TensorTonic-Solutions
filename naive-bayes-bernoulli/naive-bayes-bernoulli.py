import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.unique(y_train)
    results = []

    for c in classes:
        X_c = X_train[y_train == c]

        n = len(X_c)
        prior = np.log(n / len(y_train))

        theta = (np.sum(X_c, axis=0) + 1) / (n + 2)

        score = X_test * np.log(theta) + (1 - X_test) * np.log(1 - theta)

        results.append(prior + np.sum(score, axis=1))

    return np.column_stack(results)