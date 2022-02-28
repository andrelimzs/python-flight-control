import numpy as np

import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

def create_quadcopter(vis, arm_length=0.25, rotor_radius=0.125):
    """Create a quadcopter in Meshcat from two boxes (arms) and four cylinders (rotors)"""
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

def move_camera(vis, axis=[0,0,1], angle=45, offset=[3,3,2]):
    """Move camera by rotating {angle} degrees about {axis}, then translating by {offset}"""
    T = tf.rotation_matrix(np.deg2rad(angle), axis)
    T[0:3,3] = offset
    vis['/Cameras'].set_transform(T)

def homogeneous_transform(att_enu, pos_enu):
    """Form homogeneous transformation matrix"""
    T = tf.euler_matrix(*att_enu)
    T[0:3,3] = pos_enu
    return T

def homogeneous_transform_NED(att_ned, pos_ned):
    """Convert NED to ENU, then form homogeneous transformation"""
    att_enu = np.array([1,-1,-1]) * att_ned
    pos_enu = np.array([1,-1,-1]) * pos_ned
    return homogeneous_transform(att_enu, pos_enu)