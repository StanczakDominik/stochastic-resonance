import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numba import autojit
import h5py
import time

a = 1
b = 1
c = 0.25
D = 0.05
dt = 0.005
steps = 1024
periods = 64
T = steps * dt
w = 2 * np.pi / T
NT = periods * steps
x0 = 2

def stochastic(a=a, b=b, c=c, D=D, w=w, NT=NT, dt = dt, x0=x0):
    fig, (ax1, ax2) = plt.subplots(2)
    parameters = {"a": a, "b":b, "c":c, "D":D, "steps":steps, "periods":periods,
        "w":w, "T":T, "NT":NT, "dt":dt, "x0":x0}

    def f(x, t):
        return a*x - b*x**3 + c*np.sin(w*t)

    def V(x, t):
        return -0.5*a*x**2 + 0.25 * b * x**4 - x*c*np.sin(w*t)

    t = np.arange(NT)*dt

    # initial conditions
    x_history = np.zeros(NT)
    x_history[0] = np.array(x0)

    @autojit
    def euler(x,t,dt):
        dx = f(x, t)*dt + np.sqrt(2*D*dt)*np.random.normal()
        return x + dx

    @autojit
    def run_euler(x_history, t, NT, dt):
        for i in range(1,NT):
            x_history[i] = euler(x_history[i-1], t[i], dt)
        return x_history

    x_history = run_euler(x_history, t, NT, dt)
    x_binary = x_history > 0
    x_binary_plot = x_binary*2-1
    ax2.plot(t, x_history)
    ax2.plot(t, x_binary_plot)
    time_dot, = ax2.plot([],[], "ro")
    ax2.set_xlim(t.min(), t.max())
    ax2.set_xlabel("t")
    ax2.set_ylabel("x_history")
    ax2.plot(t, np.sin(w*t))

    # Omega = fft.rfftfreq(NT, dt)
    # dOmega = Omega[1]-Omega[0]
    # X = fft.rfft(x_history)
    # power = np.abs(X)**2
    # idx = np.argmax(power)
    # Omega_max = Omega[idx]
    # limits = (Omega < 4*Omega_max) * (Omega > 0)

    max_distance = np.max((np.abs(x_history.min()),np.abs(x_history.max())))
    x_v = np.linspace(-max_distance,max_distance,30)
    potentials = V(x_v[:,np.newaxis], t[np.newaxis,:])
    v_v = V(x_v,0)
    potential_plot, = ax1.plot(x_v, v_v)
    potential_dot, = ax1.plot([],[], "ro")
    ax1.set_xlabel("x_history")
    ax1.set_ylabel("V(x_history, t)")

    ax1.set_ylim(potentials.min(), potentials.max())
    ax1.set_xlim(-max_distance, max_distance)
    particle_potentials = V(x_history, t)

    def animation_init():
        potential_plot.set_data(x_v,np.zeros_like(x_v))
        potential_dot.set_data([],[])
        time_dot.set_data([],[])
        return potential_plot, potential_dot, time_dot

    def animate(i):
        potential_plot.set_ydata(potentials[:,i])
        potential_dot.set_data(x_history[i], particle_potentials[i])
        time_dot.set_data(t[i], x_history[i])
        return potential_plot, potential_dot, time_dot

    animation = anim.FuncAnimation(fig, animate, frames=range(NT),
        init_func=animation_init, interval=1, blit=True,
        repeat=True, repeat_delay = 100,
        )
    plt.show()
    parameters["xfin"] = x_history[-1]
    parameters["tfin"] = t[-1]
    return x_binary, parameters

def fourier_analysis():
    D_points = []
    P_points = []
    with h5py.File("data.hdf5") as f:
        for dataset_name, dataset in f.items():
            attrs = dataset.attrs
            D = attrs['D']
            NT = int(attrs['NT'])
            dt = attrs['dt']
            X = fft.rfft(dataset[...])
            power = np.abs(X)**2
            Omega = fft.rfftfreq(NT, dt)
            max_power = power.max()
            ind = (Omega > 0) * (power < max_power*3)
            plt.plot(Omega[ind],np.log(power[ind]))
            plt.show()
            # D_points.append(D)
            # P_points.append(max)

def run_stochastic(D=D):
    dataset, attrs = stochastic(D=D)
    with h5py.File("data.hdf5") as f:
        file_dataset = f.create_dataset(time.strftime("%c"), data = dataset.astype(bool))
        for key, value in attrs.items():
            file_dataset.attrs[key] = value

        print(file_dataset)
        for attr, val in file_dataset.attrs.items():
            print(attr, val)

if __name__=="__main__":
    fourier_analysis()
    # run_stochastic(D=0.3)
