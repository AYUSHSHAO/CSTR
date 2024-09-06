import math
import numpy as np
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
pow = math.pow
exp = np.exp
tanh = np.tanh

#comjdj
# a new change


global dt
dt = 0.05


def CSTR(action, ti, x0):
    Fa0 = action

    def model(y, ti):
        x1 = y[0]  # Propylene oxide
        x2 = y[1]  # water with H2SO4
        x3 = y[2]  # Propylene glycol
        x4 = y[3]  # Methanol
        x5 = y[4]  # Temperature
        parameters = [1000, 75, 16000, 60, 100, 1000]
        Fb0, T0, UA, Ta1, Fm0, mc = parameters
        V = (1 / 7.484) * 500
        k = 16.96e12 * math.exp(-32400 / 1.987 / (y[4] + 460))
        # ra = -k*Ca
        # rb = -k*Ca
        # rc = k*Ca
        Nm = y[3] * V
        Na = y[0] * V
        Nb = y[1] * V
        Nc = y[2] * V

        ThetaCp = 35 + Fb0 / Fa0[0] * 18 + Fm0 / Fa0[0] * 19.5
        v0 = Fa0[0] / 0.923 + Fb0 / 3.45 + Fm0 / 1.54
        Ta2 = y[4] - (y[4] - Ta1) * exp(-UA / (18 * mc))
        Ca0 = Fa0[0] / v0
        Cb0 = Fb0 / v0
        Cm0 = Fm0 / v0
        Q = mc * 18 * (Ta1 - Ta2)
        tau = V / v0
        NCp = Na * 35 + Nb * 18 + Nc * 46 + Nm * 19.5

        dx1_dt = 1 / tau * (Ca0 - x1) - k * x1
        dx2_dt = 1 / tau * (Cb0 - x2) - k * x1
        dx3_dt = 1 / tau * (0 - x3) + k * x1
        dx4_dt = 1 / tau * (Cm0 - x4)
        dx5_dt = (Q - Fa0[0] * ThetaCp * (x5 - T0) + (-36000) * (-k * x1) * V) / NCp

        return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt])

    t = np.linspace(ti, ti + dt, 100)
    y = odeint(model, x0, t)
    All = y[-1]
    p = All[2]
    A = y[99, 0]
    B = y[99, 1]
    C = y[99, 2]
    D = y[99, 3]
    E = y[99, 4]
    rewards = -np.abs((All[2] - 0.143))

    return All, rewards



x0 = [0, 3.45, 0, 0, 75]
high = np.array([10, 10, 10, 10, 200])

observation_space = spaces.Box(
    low=np.array([0, 3.45, 0, 0, 75]),
    high=high,
    dtype=np.float32
)
high = np.array([80], dtype=np.float32)

action_space = spaces.Box(
    low=np.array([0]),
    high=high,
    dtype=np.float32
)

def norm_action(action):
    low = -1
    high = 1

    action = ((action - low) / (high - low))
    action = 78 + action

    return action


def plot_G(propylene_glycol, tot_time, flowrate):
    time = np.linspace(0, tot_time, int(tot_time / dt))

    T1 = 0.143  # target
    ta = np.ones(int(tot_time / dt)) * T1
    fig, ax1 = plt.subplots()
    # time = np.linspace(0,T,int(T/dt))
    font1 = {'family': 'serif', 'size': 15}
    font2 = {'family': 'serif', 'size': 15}
    color = 'tab:red'
    ax1.set_xlabel('time (min)', fontdict=font1)
    ax1.set_ylabel('Propylene Glycol', fontdict=font2, color=color)
    ax1.plot(time, propylene_glycol, color=color)
    ax1.plot(time, ta, color='tab:orange', linewidth=4, label='reference concentration')
    leg = ax1.legend(loc='lower right')

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('flowrate', fontdict=font2, color=color)  # we already handled the x-label with ax1
    ax2.step(time, flowrate, where='post', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color='g', linestyle='-', linewidth=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('deeprl_test1_batch1.jpg')
    plt.show()

# In[10]:




