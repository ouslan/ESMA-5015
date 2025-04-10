import numpy as np
import scipy.stats as stats


def h(X):
    return (X**5) / (1 + (X - 3) ** 2) * (X >= 0)


if __name__ == "__main__":
    samples = stats.t.rvs(df=12, size=1000)
    monte_carlo_estimate = np.mean(h(samples))
