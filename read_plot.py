import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.fftpack as fft
import os
import pandas as pd
filename = "data3.hdf5"

df = pd.read_csv("wykres.csv", header=None)
x_plot, y_plot = df[0], df[1]

fig, ax = plt.subplots()
ax.plot(x_plot, y_plot)
plt.show()
