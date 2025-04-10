import numpy as np
import scipy.stats as stats


def h(X):
    return (X**5) / (1 + (X - 3) ** 2) * (X >= 0)


def g_normal(x, v):
    return stats.norm.pdf(x, loc=0, scale=np.sqrt(v / (v - 2)))


def importance_sampling_normal(samples, v):
    f_samples = stats.t.pdf(samples, df=v)
    g_samples = g_normal(samples, v)

    weights = f_samples / g_samples
    h_values = h(samples)
    return np.mean(weights * h_values)


if __name__ == "__main__":
    samples_normal = np.random.normal(0, np.sqrt(12 / (12 - 2)), size=100)

    normal_estimate = importance_sampling_normal(samples_normal, 12)
    print(normal_estimate)
