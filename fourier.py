import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.fftpack as fft
import os

filename = "data3.hdf5"

def fourier_analysis(filename = filename):
    data = {}
    with h5py.File(filename) as f:
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

            fig, (ax1, ax2, ax3) = plt.subplots(3)
            ax1.plot(np.linspace(0,T,NT), x)
            ax1.set_ylim(-1.3, 1.3)
            ax1.set_xlabel("t")
            ax1.set_ylabel("x")
            ax1.set_title("Noise variance: {}".format(random_variance))

            min_line = SPD.min()
            if np.isinf(min_line):
                min_line = 0
            print(frequency, SPD.min(), SPD.max())
            ind_plot2 = Omega < frequency*2.5
            ax2.plot(Omega[ind_plot2], SPD[ind_plot2], label="Signal power density")
            ax2.vlines(frequency, min_line, SPD.max())
            # ax2.set_xlim(-0.5*frequency, frequency*2.5)

            N_average = 11
            def running_mean(x, N):
                cumsum = np.cumsum(np.insert(x, 0, 0))
                return (cumsum[N:] - cumsum[:-N]) / N
            omega_average = Omega[int(N_average/2):-int(N_average/2)]
            indices = omega_average < frequency*2.5
            # averaged_signal = np.convolve(x, np.ones((N_average,))/N_average, mode='valid')
            averaged_signal = running_mean(np.abs(X)**2, N_average)
            averaged_fourier = np.log(averaged_signal)
            # import ipdb; ipdb.set_trace()
            ax2.plot(omega_average[indices], averaged_fourier[indices], "r--", label="avg noise")
            y_plot3 = SPD[int(N_average/2):-int(N_average/2)][indices] - averaged_fourier[indices]
            ax3.plot(omega_average[indices], y_plot3,
                        "r-", label="avg noise")
            ax3.vlines(frequency, y_plot3.min(), y_plot3.max())
            ax2.legend(loc = 'best')

            id_freq = np.argmin(np.abs(omega_average[indices] - frequency))
            data[random_variance] = y_plot3[id_freq]

            ax2.set_ylabel("ln(|x|^2)")
            ax2.set_xlabel("f (Hz)")
            ax2.grid()
            picname = "{}.png".format(random_variance)
            if not os.path.isfile(picname):
                plt.savefig(picname)
            plt.clf()
    fig, ax = plt.subplots()
    ax.plot(np.asarray(data.keys()), np.asarray(data.values()))
    plt.show()
if __name__=="__main__":
    fourier_analysis()
