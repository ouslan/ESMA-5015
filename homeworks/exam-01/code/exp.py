import numpy as np
import scipy.stats as stats


def h(X: np.ndarray):
    return (X**5) / (1 + (X - 3) ** 2) * (X >= 0)


def g_exponential(x: np.ndarray):
    return np.exp(-x) * (x >= 0)


def importance_sampling_exponential(samples: np.ndarray):
    f_samples = stats.t.pdf(samples, df=12)

    g_samples = g_exponential(samples)

    weights = f_samples / g_samples
    h_values = h(samples)

    estimate = np.mean(weights * h_values)

    return estimate


if __name__ == "__main__":
    samples_exponential = np.random.exponential(1, size=1000)
    exponential_estimate = importance_sampling_exponential(samples_exponential)
    print(exponential_estimate)
