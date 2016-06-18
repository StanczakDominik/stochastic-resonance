import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

rc('font', **{'family':'DejaVu Sans'})
filename = "data3.hdf5"

df = pd.read_csv("wykres.csv", header=None)
x_plot, y_plot = df[0], df[1]

id_max = np.argmax(y_plot)

fig, ax = plt.subplots()
# ax.set_xlim(-0.5, x_plot.max())
ax.plot(np.log(x_plot), y_plot, "bo--")
ax.set_title("$[\log |X(\omega)|^2 - \log <|X|>^2](D)$")
ax.set_xlabel("$\log D$ - wariancja losowych drgań układu")
ax.set_ylabel("$\log |X(\omega)|^2 - \log <|X|>^2$")
xmax, ymax = x_plot[id_max], y_plot[id_max]
ax.annotate("Maksimum: {:.2f} dla $D = ${:.3f}".format(ymax, xmax),
    (np.log(xmax), ymax), (-3, 2), arrowprops=dict(facecolor='black', shrink=0.05))
ax.grid()
fig.savefig("ostateczny_wykres.png")
plt.show()
