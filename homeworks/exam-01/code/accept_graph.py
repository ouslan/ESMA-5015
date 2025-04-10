import matplotlib.pyplot as plt
import numpy as np
from accept import accept_reject
from scipy.stats import gamma


def main(n: int) -> None:
    samples = accept_reject(n)

    # Traficar el histograma
    x = np.linspace(0, 10, 1000)
    plt.hist(
        samples,
        bins=50,
        density=True,
        alpha=0.6,
        color="b",
        label="Histograma (muestras)",
    )
    plt.plot(x, gamma.pdf(x, 3 / 2, scale=1), "r-", label="Distribución Gamma(3/2, 1)")

    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.title("Histograma de la Distribución Generada vs. Distribución Objetivo")
    plt.savefig("gamma_hist.png")


if __name__ == "__main__":
    n_samples = 10000
    main(n_samples)
