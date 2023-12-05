import numpy as np
import numpy.linalg as LA
from flightcontrol.Utils import *
from flightcontrol.VTOL import *
from scipy.integrate import solve_ivp
from functools import partial
import math

import matplotlib.pyplot as plt


if __name__ == "__main__":
    dynamics = QuadVTOL()
    controller = PID()

    pos0 = np.array([0., 0., 0.])
    vel0 = np.array([0., 0., 0.])
    quat0 = np.array([1., 0., 0., 0.])
    pqr0 = np.array([0., 0., 0.])
    tet_r0 = np.array([0., 0., 0.])
    x = np.concatenate([
        pos0, vel0, quat0, pqr0, tet_r0
    ])
    ref = np.array([2.0, 0.0, 0.0, 0.0])
    
    t_step = 0.01
    t_span = np.arange(0, 10, t_step)

    # Iterate through time
    state = np.nan * np.ones((16, 1000))
    control = np.nan * np.ones((6, 1000))
    for i,t in enumerate(t_span):
        # Control
        u = controller.run(ref, x)
        
        # Simulate dynamics
        sol = solve_ivp(
            partial(dynamics, u=u),
            t_span=[t, t+t_step],
            t_eval=[t+t_step],
            # method='Radau',
            y0=x
        )
        print(t, sol.status)
        x = sol.y[:,-1]

        # Log state
        state[:,i] = sol.y[:,-1]
        control[:,i] = u
    
    """Plot"""
    t = t_span
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    pqr = state[10:13]
    tet_r = state[13:15]

    dE = control[0:2]
    # rpm = control[2:5]
    # tet_c = control[5:7]
    TLMN = control[2:6]
    
    fig, ax = plt.subplots(3, 1, sharey=True)
    [ ax[i].plot(t, vel[i]) for i in range(3) ]
    ax[0].set_title("Velocity")
    ax[0].set_ylabel("vx (m/s)")
    ax[1].set_ylabel("vy (m/s)")
    ax[2].set_ylabel("vz (m/s)")
    ax[2].set_xlabel("t (s)")

    fig, ax = plt.subplots(3, 1, sharey=True)
    [ ax[i].plot(t, pqr[i]) for i in range(3) ]
    ax[0].set_title("Body Rate")
    ax[0].set_ylabel("p (m/s)")
    ax[1].set_ylabel("q (m/s)")
    ax[2].set_ylabel("r (m/s)")
    ax[2].set_xlabel("t (s)")

    plt.show()

