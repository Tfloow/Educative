import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = [1, 2, 3, 4, 5]
y = [1, 2, 4, 8, 16]
y1 = y+np.random.randint(1,5,5)
y2 = y+ np.random.randint(1,5,5)
y3 = y+np.random.randint(1,5,5)
y4 = y+np.random.randint(1,5,5)
y5 = y+np.random.randint(1,5,5)
y6 = y+np.random.randint(1,5,5)

labels = ["Jan", "Feb", "Mar", "Apr", "May"]

fig, axe = plt.subplots(dpi=800)
axe.stackplot(x, y, y1, y2, y3, y4, y5, y6,
              labels=["A", "B", "C", "D", "E", "F", "G"])
axe.set_xticks(x)
axe.set_xticklabels(labels)
axe.set_title("car sales from Jan to May")
axe.legend(loc='upper left')

plt.savefig("Matplotlib_for_Python/stack.png")