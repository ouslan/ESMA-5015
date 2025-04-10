import numpy as np


# Función de densidad de la distribución objetivo (Gamma(3/2, 1))
def f(x):
    return (2 / np.sqrt(np.pi)) * np.sqrt(x) * np.exp(-x)


# Función de densidad de la distribución candidata (Gamma(1, 2))
def g(x):
    return x * np.exp(-x)


# Algoritmo de aceptación y rechazo
def accept_reject(n, seed=787):
    # Semilla para la generación de números aleatorios
    rng = np.random.default_rng(seed=seed)
    samples = []
    M = 2 / np.sqrt(np.pi)

    while len(samples) < n:
        # Paso 1: Generar una muestra de la distribución candidata (Gamma(1, 2))
        Y = rng.gamma(1, 2)

        # Paso 2: Generar una variable aleatoria uniforme U
        U = rng.uniform(0, 1)

        # Paso 3: Aceptar o rechazar
        if U <= f(Y) / (M * g(Y)):
            samples.append(Y)

    return np.array(samples)


if __name__ == "__main__":
    n_samples = 10000
    samples = accept_reject(n_samples)
