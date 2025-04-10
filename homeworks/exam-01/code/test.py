# Estimar E[X^2] y construir la gráfica de la convergencia de los "running means"
running_means = np.cumsum(samples) / np.arange(1, n_samples + 1) + (np.cumsum(samples) / np.arange(1, n_samples + 1))**2

# Graficar la convergencia
plt.plot(running_means, label="Running Mean")
plt.axhline(y=3.75, color="r", linestyle="--", label="Media Estimada")
plt.xlabel("Número de Muestras")
plt.ylabel("Running Mean")
plt.legend()
plt.title("Convergencia del Running Mean a la Media Estimada")
plt.show()
