import numpy as np
import scipy.stats as stats


def h(X):
    return (X**5) / (1 + (X - 3) ** 2) * (X >= 0)


def g_cauchy(x):
    return 1 / (np.pi * (1 + x**2))


def importance_sampling_cauchy(samples, v):
    f_samples = stats.t.pdf(samples, df=v)
    g_samples = g_cauchy(samples)
    weights = f_samples / g_samples
    h_values = h(samples)

    estimate = np.mean(weights * h_values)

    return estimate


if __name__ == "__main__":
    samples_cauchy = np.random.standard_cauchy(size=1000)
    cauchy_estimate = importance_sampling_cauchy(samples_cauchy, 12)
    print(cauchy_estimate)
