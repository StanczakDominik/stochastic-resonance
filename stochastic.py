import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numba import autojit
import h5py

"""
f_s = 3.125
"""
filename = "data3.hdf5"

x1_force = 1
x3_force = 1
periodic_force_amplitude = 0.25
random_variance = 0.05
dt = 0.005
steps = 1024
periods = 512 * 8 * 4 
T = steps * dt
# force_frequency = 2 * np.pi / T
force_frequency = 1/T
NT = periods * steps
x0 = 2

def stochastic(x1_force=x1_force, x3_force=x3_force, periodic_force_amplitude=periodic_force_amplitude, random_variance=random_variance, force_frequency=force_frequency, NT=NT, dt = dt, x0=x0, plotting=True):
    parameters = {"x1_force": x1_force, "x3_force":x3_force, "periodic_force_amplitude":periodic_force_amplitude, "random_variance":random_variance, "steps":steps, "periods":periods,
        "force_frequency":force_frequency, "T":T, "NT":NT, "dt":dt, "x0":x0}

    def f(x, t):
        return x1_force*x - x3_force*x**3 + periodic_force_amplitude*np.sin(2*np.pi*force_frequency*t)

    def V(x, t):
        return -0.5*x1_force*x**2 + 0.25 * x3_force * x**4 - x*periodic_force_amplitude*np.sin(force_frequency*t)

    t = np.arange(NT)*dt

    # initial conditions
    x_history = np.zeros(NT)
    x_history[0] = np.array(x0)

    @autojit
    def euler(x,t,dt):
        dx = f(x, t)*dt + np.sqrt(2*random_variance*dt)*np.random.normal()
        return x + dx

    @autojit
    def run_euler(x_history, t, NT, dt):
        for i in range(1,NT):
            x_history[i] = euler(x_history[i-1], t[i], dt)
        return x_history

    x_history = run_euler(x_history, t, NT, dt)
    x_binary = x_history > 0

    if plotting:
        fig, (ax1, ax2) = plt.subplots(2)
        x_binary_plot = x_binary*2-1
        ax2.plot(t, x_history)
        ax2.plot(t, x_binary_plot)
        time_dot, = ax2.plot([],[], "ro")
        ax2.set_xlim(t.min(), t.max())
        ax2.set_xlabel("t")
        ax2.set_ylabel("x_history")
        ax2.plot(t, np.sin(force_frequency*t))

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



def run_stochastic(random_variance=random_variance, plotting=False):
    name = str(random_variance)
    with h5py.File(filename) as f:
        if name in f:
            print("{} already done".format(name))
            return

        dataset, attrs = stochastic(random_variance=random_variance, plotting=plotting)

        file_dataset = f.create_dataset(name, data = dataset.astype(bool))
        print(file_dataset)
        for key, value in attrs.items():
            file_dataset.attrs[key] = value
            print(key, value)


if __name__=="__main__":
    for d in np.logspace(-2,1,200):
        print("Running for d={}".format(d))
        run_stochastic(random_variance=d)
