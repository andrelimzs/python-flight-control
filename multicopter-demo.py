# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:control-systems] *
#     language: python
#     name: conda-env-control-systems-py
# ---

# +
import numpy as np
import numpy.linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control

import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from Quadcopter import *
from Controller import *
from Utils import *

# Create a new visualizer
vis = meshcat.Visualizer()

# + tags=[]
# Specify physical parameters
params = { 'mass': 1, 'J': np.array([0.01, 0.01, 0.02])}

# Compute state feedback
z3 = np.zeros((3,3))
I3 = np.eye(3)
A = np.block([[z3, I3],[z3, z3]])
B = np.block([[z3],[I3]])
p = np.repeat([-3,-4], 3)
kTranslation = control.place(A, B, p)

# Compute state feedback
z3 = np.zeros((3,3))
I3 = np.eye(3)
A = np.block([[z3, I3],[z3, z3]])
B = np.block([[z3],[I3]])
p = np.repeat([-10,-12], 3)
kRotation = control.place(A, B, p)

# Instantiate Controller
controller = Controller(kTranslation, kRotation)


# +
# %%time
# Generate trajectory
def ref_function(t):
    t = np.atleast_1d(t)
    nan  = np.nan * t
    zero = 0 * t
    
    ref = {}
    ref['pos']  = stack_squeeze( [nan, nan, nan] )
    ref['pos'][0] = 5*np.sin(1*t)
    ref['pos'][1] = 5*np.sin(0.5*t)
    ref['pos'][2] = -1 * (t > 0).astype(int)
    
    ref['vel']  = stack_squeeze( [nan, nan, nan] )
    
    ref['att']  = stack_squeeze( [nan, nan, zero] )
    # ref['att'][0] = 0.5*np.sin(2*t)
    # ref['att'][1] = 0.5*np.sin(1*t)
    # ref['att'][2] = 0*np.sin(5*t)
    
    ref['rate'] = stack_squeeze( [nan, nan, nan] )
    
    return ref

# Run Simulation
Ts     = 1/100
t_span = [0,20]
t_eval = np.arange(t_span[0], t_span[1], Ts)

# Instantiate Quadcopter
quadcopter = Quadcopter('Quaternion', params=params, reference=ref_function, control=controller)

# Run simulation (solve ODE)
x0 = quadcopter.generate_x0()
sol = solve_ivp(quadcopter, t_span, x0, t_eval=t_eval)

# Unpack state & reference
t = sol.t
state = quadcopter.unpack_state(sol.y)
ref = ref_function(sol.t)

# Recreate control input
u = np.zeros((4, len(t)))
for i in range(len(t)):
    x = quadcopter.unpack_state(sol.y[:,i])
    r = ref_function(t[i])
    u[:,i] = controller(r, x)
T = u[0]
LMN = u[1:4]

# +
# %matplotlib inline
plt.rcParams["figure.figsize"] = (20,5)

fig,ax = plt.subplots(3,2, sharex=True)
for i in range(3):
    ax[i,0].plot(t, state['pos'][i])
    ax[i,0].plot(t, ref['pos'][i])
    
    ax[i,1].plot(t, np.rad2deg(state['eul'][i]))
    ax[i,1].plot(t, np.rad2deg(ref['att'][i]))

[ ax[i,j].grid() for i in range(3) for j in range(2) ]
ax[0,0].set_title('Pos')
ax[0,1].set_title('Att')
        
fig,ax = plt.subplots(3,2, sharex=True)
for i in range(3):
    ax[i,0].plot(t, state['vel'][i])
    ax[i,0].plot(t, ref['vel'][i])
    
    ax[i,1].plot(t, np.rad2deg(state['rate'][i]))
    ax[i,1].plot(t, np.rad2deg(ref['rate'][i]))
    # ax[i,3].plot(time, LMN[i])
    
[ ax[i,j].grid() for i in range(3) for j in range(2) ]
ax[0,0].set_title('Vel')
ax[0,1].set_title('Rate')


# -

# **Visualise**

# + tags=[]
def create_quadcopter(vis):
    arm_length = 0.25
    rotor_radius = 0.125

    # Arms
    vis['drone']['left_arm'].set_object(g.Box([2*arm_length, 0.05, 0.05]))
    vis['drone']['left_arm'].set_transform(tf.rotation_matrix(np.deg2rad(45), [0,0,1]))
    vis['drone']['right_arm'].set_object(g.Box([2*arm_length, 0.05, 0.05]))
    vis['drone']['right_arm'].set_transform(tf.rotation_matrix(np.deg2rad(-45), [0,0,1]))

    # Rotors
    for i in range(1,5):
        theta = np.deg2rad(45 + 90*i)
        offset = np.array([arm_length * np.sin(theta), arm_length * np.cos(theta), 0.05])

        # Compute transformation
        T = tf.rotation_matrix(np.deg2rad(90), [1,0,0])
        T[0:3,3] = offset

        vis['drone'][f'rotor{i}'].set_object(g.Cylinder(0.01, rotor_radius))
        vis['drone'][f'rotor{i}'].set_transform(T)
        
    return vis['drone']

vis.delete()
drone = create_quadcopter(vis)

# Set camera
T = tf.rotation_matrix(np.deg2rad(45), [0,0,1])
T[0:3,3] = [3,3,2]
vis['/Cameras'].set_transform(T)

anim = meshcat.animation.Animation()

for i in range(0, len(t), 3):   
    with anim.at_frame(vis, i) as frame:
        # Convert NED to ENU
        pos_enu = np.array([1,-1,-1]) * state['pos'][:,i]
        att_enu = np.array([1,-1,-1]) * state['eul'][:,i]

        # Form homogeneous transformation
        T = tf.euler_matrix(*att_enu)
        T[0:3,3] = pos_enu

        # Apply
        frame.set_transform(T)

vis.set_animation(anim)
vis.render_static()