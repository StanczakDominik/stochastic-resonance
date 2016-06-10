import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.fftpack as fft
from scipy.signal import savgol_filter


filename = "data2.hdf5"

def fourier_analysis(filename = "data2.hdf5"):
    with h5py.File("data.hdf5") as f:
        for dataset_name, dataset in f.items():
            print(dataset_name)
            attrs = dataset.attrs
            random_variance = attrs['random_variance']
            NT = int(attrs['NT'])
            dt = attrs['dt']
            T = attrs['T']
            frequency = attrs['force_frequency']
            x = dataset[...] * 2 -1
            if (x == 1).all() or ( x == -1).all():
                continue
            X = fft.rfft(x)
            Omega = fft.rfftfreq(NT, dt)
            # dOmega = Omega[1] - Omega[0]
            SPD = np.log(np.abs(X)**2)

            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(np.linspace(0,T,NT), x)
            ax1.set_ylim(-1.3, 1.3)
            ax1.set_xlabel("t")
            ax1.set_ylabel("x")
            ax1.set_title("Noise variance: {}".format(random_variance))

            min_line = SPD.min()
            if np.isinf(min_line):
                min_line = 0
            print(frequency, SPD.min(), SPD.max())
            ax2.plot(Omega, SPD, label="Signal power density")
            ax2.vlines(frequency, min_line, SPD.max())
            ax2.set_xlim(-0.5*frequency, frequency*2.5)

            # noise = np.log(savgol_filter(np.abs(X)**2, 201, 3))
            # ax2.plot(Omega, noise, "r--", label="avg noise")
            # ax2.legend(loc = 'best')

            ax2.set_ylabel("ln(|x|^2)")
            ax2.set_xlabel("f (Hz)")
            ax2.grid()
            plt.show()
if __name__=="__main__":
    fourier_analysis()
