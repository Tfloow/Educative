import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
labels = ["Sun", "Moon", "Jupiter", "Venus"]
values = []
values.append(np.random.normal(100, 10, 200))
values.append(np.random.normal(90, 20, 200))
values.append(np.random.normal(120, 25, 200))
values.append(np.random.normal(130, 30, 200))

fig, axe = plt.subplots(dpi=800)
axe.boxplot(values, labels=labels)

fig.savefig("Matplotlib_for_Python/boxplot.png")