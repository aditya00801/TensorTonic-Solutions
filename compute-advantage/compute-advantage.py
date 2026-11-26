import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    states = np.asarray(states)
    rewards = np.asarray(rewards)
    V = np.asarray(V)
    returns = np.zeros(len(rewards))
    G = 0

    for t in range(len(rewards)-1, -1 , -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    compute_advantage = returns - V[states]

    return np.round(compute_advantage, 4)
        



    