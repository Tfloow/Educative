import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"yearID": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007],
                   "HR": [49, 73, 46, 45, 45,  5, 26, 28]})

df.plot(kind="line", x="yearID", y="HR")
plt.savefig("Machine_learning/plotExample.png")