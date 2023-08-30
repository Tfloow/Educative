import numpy as np
import matplotlib.pyplot as plt

points = np.linspace(-5, 5, 256)
y1 = np.tanh(points) + 0.5
y2 = np.sin(points) - 0.2

fig, axe = plt.subplots(figsize=(7, 3.5), dpi=300)
axe.plot(points, y1)
axe.plot(points, y2)
axe.legend(["tanh", "sin"])
axe.annotate("1.464=tanh(2)+0.5", xy=(2, 1.464), xycoords="data",
             xytext=(0.4, -40), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
axe.minorticks_on()
axe.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
axe.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
fig.savefig("Matplotlib_for_Python/grid.png")