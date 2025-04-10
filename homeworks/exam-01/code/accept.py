import numpy as np

def f(x):
    return (2 / np.sqrt(np.pi)) * np.sqrt(x) * np.exp(-x)

def g(x):
    return 0.5 * (1 + x) * np.exp(-x)

def accept_reject(n, seed=787):
    rng = np.random.default_rng(seed)
    samples = []
    M = 2 / np.sqrt(np.pi)

    while len(samples) < n:

        coin = rng.integers(0, 2)
        if coin == 0:
            Y = rng.gamma(shape=1, scale=1)
        else:
            Y = rng.gamma(shape=2, scale=1)

        U = rng.uniform()

        if U <= f(Y) / (M * g(Y)):
            samples.append(Y)

    return np.array(samples)


if __name__ == "__main__":
    n_samples = 10000
    samples = accept_reject(n_samples)
