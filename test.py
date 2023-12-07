import numpy as np
import numpy.linalg as LA
from flightcontrol.Utils import *
from flightcontrol.VTOL import *
from scipy.integrate import solve_ivp
from functools import partial
import math
from time import sleep

import matplotlib.pyplot as plt


if __name__ == "__main__":
    t_step = 1/100
    t_span = np.arange(0, 10, t_step)

    dynamics = QuadVTOL()
    controller = PID(t_step)

    pos0 = np.array([0., 0., 0.])
    vel0 = np.array([0., 0., 0.])
    quat0 = np.array([1., 0., 0., 0.])
    pqr0 = np.array([0., 0., 0.])
    tet_r0 = np.array([0., 0., 0.])
    x = np.concatenate([
        pos0, vel0, quat0, pqr0, tet_r0
    ])
    ref = np.array([2.0, 0.0, 0.0, 0.0])
    
    # Iterate through time
    log_x = np.nan * np.ones((16, len(t_span)))
    log_u = np.nan * np.ones((6, len(t_span)))
    desired = np.nan * np.ones((6, len(t_span)))
    for i,t in enumerate(t_span):
        # Control
        u, (eul_des, pqr_des) = controller.run(ref, x)
        # Simulate dynamics
        sol = solve_ivp(
            dynamics,
            args=(u,),
            t_span=[t, t+t_step],
            t_eval=[t+t_step],
            method="RK45",
            y0=x
        )
        print(sol.t.item(), sol.status)
        x = sol.y[:,-1]

        # Log state
        log_x[:,i] = sol.y[:,-1]
        desired[0:3,i] = eul_des
        desired[3:6,i] = pqr_des
        log_u[:,i] = u
    
    """Plot"""
    t = t_span
    pos = log_x[0:3]
    vel = log_x[3:6]
    quat = log_x[6:10]
    pqr = log_x[10:13]
    tet_r = log_x[13:15]
    eul = quat2eul(quat)

    dE = log_u[0:2]
    # rpm = log_u[2:5]
    # tet_c = log_u[5:7]
    TLMN = log_u[2:6]

    vel_des = np.repeat(ref[0:3, np.newaxis], len(t), axis=1)
    eul_des = desired[0:3]
    pqr_des = desired[3:6]


    fig, ax = plt.subplots(3, 1, sharey=True)
    [ ax[i].plot(t, vel[i]) for i in range(3) ]
    [ ax[i].plot(t, vel_des[i]) for i in range(3) ]
    ax[0].set_title("Velocity")
    ax[0].set_ylabel("vx (m/s)")
    ax[1].set_ylabel("vy (m/s)")
    ax[2].set_ylabel("vz (m/s)")
    ax[2].set_xlabel("t (s)")

    # Attitude
    fig, ax = plt.subplots(3, 1, sharey=True)
    [ ax[i].plot(t, eul[i]) for i in range(3) ]
    [ ax[i].plot(t, eul_des[i]) for i in range(3) ]
    ax[0].set_title("Euler Angles")
    ax[0].set_ylabel("phi (rad)")
    ax[1].set_ylabel("theta (rad)")
    ax[2].set_ylabel("psi (rad)")
    ax[2].set_xlabel("t (s)")

    # Body Rate
    fig, ax = plt.subplots(3, 1, sharey=True)
    [ ax[i].plot(t, pqr[i]) for i in range(3) ]
    [ ax[i].plot(t, pqr_des[i]) for i in range(3) ]
    ax[0].set_title("Body Rate")
    ax[0].set_ylabel("p (rad/s)")
    ax[1].set_ylabel("q (rad/s)")
    ax[2].set_ylabel("r (rad/s)")
    ax[2].set_xlabel("t (s)")

    # Quaternion
    if 0:
        fig, ax = plt.subplots(4, 1, sharey=True)
        [ ax[i].plot(t, quat[i]) for i in range(4) ]
        # [ ax[i].plot(t, eul_des[i]) for i in range(3) ]
        ax[0].set_title("Quaternion")
        ax[0].set_ylabel("w")
        ax[1].set_ylabel("x")
        ax[2].set_ylabel("y")
        ax[3].set_ylabel("z")
        ax[3].set_xlabel("t (s)")

    plt.show()

