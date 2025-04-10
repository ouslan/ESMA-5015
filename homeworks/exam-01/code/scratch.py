import numpy as np
import pandas as pd
import seaborn as sns

rng = np.random.default_rng(seed=787)
alpha = 5
beta = 7
a = alpha - 1
b = beta + 1

df = pd.DataFrame()
df["candidate"] = rng.gamma(a, b, 2000)
df["target"] = rng.gamma(alpha, beta, 2000)


# Plot both distributions using seaborn's displot with kind='kde'
g = sns.displot(
    df.melt(value_vars=["candidate", "target"]),
    x="value",
    hue="variable",
    kind="kde",
)

# Save the figure
g.savefig("fig1.png")
