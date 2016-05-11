import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numba import autojit
a = 1
b = 2
c = 1
std = 1
D = 0.5
w = 1
NT = 20000
dt = 0.005
x0 = 0
v0 = 1


def stochastic(a=a, b=b, c=c, std=std, D=D, w=w, NT=NT, dt = dt, x0=x0, v0=v0):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    def f(x, t):
        return a*x - b*x**3 + c*np.random.normal(0, std) + D*np.sin(w*t)

    def V(x, t):
        return -0.5*a*x**2 + 0.25 * b * x**4 - x*D*np.sin(w*t)

    t = np.arange(NT)*dt

    # initial conditions
    r_history = np.zeros([NT, 2])
    r_history[0] = np.array([x0,v0])

    @autojit
    def euler(r,t,dt):
        dx = r[1]*dt
        dv = f(r[0], t)*dt
        return r + np.array([dx, dv])

    @autojit
    def run_euler(r_history, t, NT, dt):
        for i in range(1,NT):
            r_history[i] = euler(r_history[i-1], t[i], dt)
        return r_history

    r_history = run_euler(r_history, t, NT, dt)
    x, v = r_history.T

    ax2.plot(t,x)
    time_dot, = ax2.plot([],[], "ro")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")

    phase_plot, = ax3.plot(x,v)
    phase_dot, = ax3.plot([], [], "ro")
    ax3.set_xlabel("x")
    ax3.set_ylabel("v")

    Omega = fft.rfftfreq(NT, dt)
    dOmega = Omega[1]-Omega[0]
    X = fft.rfft(x)
    power = np.abs(X)**2
    idx = np.argmax(power)
    Omega_max = Omega[idx]
    limits = (Omega < 4*Omega_max) * (Omega > 0)

    ax4.bar(Omega[limits], np.log(power[limits]), width=dOmega)
    ax4.set_xlim(0, 2*Omega_max)
    ax4.set_xlabel("Omega")
    ax4.set_ylabel("ln(|FFT(x)|^2)")

    x_v = np.linspace(x.min(),x.max(),30)
    potentials = V(x_v[:,np.newaxis], t[np.newaxis,:])
    v_v = V(x_v,0)
    potential_plot, = ax1.plot(x_v, v_v)
    potential_dot, = ax1.plot([],[], "ro")
    ax1.set_xlabel("x")
    ax1.set_ylabel("V(x, t)")
    ax1.set_ylim(potentials.min(), potentials.max())
    particle_potentials = V(x, t)

    def animation_init():
        potential_plot.set_data(x_v,np.zeros_like(x_v))
        potential_dot.set_data([],[])
        time_dot.set_data([],[])
        phase_dot.set_data([],[])
        return potential_plot, potential_dot, time_dot, phase_dot
    def animate(i):
        potential_plot.set_ydata(potentials[:,i])
        potential_dot.set_data(x[i], particle_potentials[i])
        time_dot.set_data(t[i], x[i])
        phase_dot.set_data(x[i], v[i])
        return potential_plot, potential_dot, time_dot, phase_dot
    animation = anim.FuncAnimation(fig, animate, frames=range(NT),
        init_func=animation_init, interval=1, blit=True,
        repeat=True, repeat_delay = 100,
        )
    plt.show()

if __name__=="__main__":
    stochastic()
