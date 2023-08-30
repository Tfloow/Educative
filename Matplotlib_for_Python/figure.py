import numpy as np
import matplotlib.pyplot as plt

points = np.linspace(-5, 5, 256)
y1 = np.tanh(points) + 0.5
y2 = np.sin(points) - 0.2

fig, axe = plt.subplots(dpi=600)
axe.plot(points, y1)
axe.plot(points, y2)
axe.set_xticks(np.linspace(-5,5,9))
fig.savefig("Matplotlib for Python Visually Represent Data with Plots/tick.png")
plt.close(fig)