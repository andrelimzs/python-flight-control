import numpy as np
from scipy.spatial.transform import Rotation as R

def rotx(a):
    """Rotation matrix about x"""
    return R.from_euler('X', a).as_matrix()

def roty(a):
    """Rotation matrix about y"""
    return R.from_euler('Y', a).as_matrix()

def rotz(a):
    """Rotation matrix about z"""
    return R.from_euler('Z', a).as_matrix()

def dot_2d(v1,v2) -> np.ndarray:
    """For (D,N) vectors, return (N,) dot products
    Compute the dot product along each column (N times)"""
    if v1.ndim == 1 and v2.ndim == 1:
        return np.atleast_1d(np.dot(v1,v2))
    elif v1.ndim == 2 and v2.ndim == 2:
        N = v1.shape[1]
        output = np.zeros(N)
        for i in range(N):
            output[i] = v1[:,i].T @ v2[:,i]
        return output
    else:
        raise ValueError(f"Cannot compute 2d dot product, v1 {v1.shape} and v2 {v2.shape} dimensions do not match")

def apply(M, vec) -> np.ndarray:
    """ Apply a sequence of linear transformations (N,D,D) of length N and size D x D
    to either a vector (D,) or a sequence of vectors (D,N) """
    # Check for the single Rot matrix case (3,3) and return instantly
    if M.ndim < 3:
        return M @ vec
    
    # Apply N rotations to a single vector
    if vec.ndim == 1:
        return M @ vec
        
    # Apply N rotations to N vectors
    elif M.shape[0] == vec.shape[1]:
        N = M.shape[0]
        D = M.shape[1]
        output = np.zeros((D,N))
        for i,R in enumerate(M):
            output[:,i] = R @ vec[:,i]
        return output
    
    # N rotations cannot be applied to M vectors
    else:
        raise ValueError("Cannot apply different number of transformations and vectors")

def hatmap(vec):
    """ Map a (R^3) vector to the SO(3) Lie algebra
    to either a vector (3,) -> (3,3) or a sequence of vectors (3,N) -> (N,3,3) """
    a1 = vec[0].reshape(-1,1,1)
    a2 = vec[1].reshape(-1,1,1)
    a3 = vec[2].reshape(-1,1,1)
    zero = np.zeros_like(a1)
    M = np.block( [[zero, -a3,    a2],
                   [ a3,  zero,  -a1],
                   [-a2,   a1,   zero]] )
    return np.squeeze(M)

# The hatmap is also the skew symmetric matrix
skew = hatmap

def veemap(M):
    """ Map the SO(3) Lie algebra to a (R^3) vector
    Input can be either (3,3) -> (3,) or (N,3,3) -> (3,N) """
    if M.ndim == 2:
        v = np.zeros(3)
        v[0] = M[2,1]
        v[1] = M[0,2]
        v[2] = M[1,0]
    elif M.ndim == 3:
        N = M.shape[0]
        v = np.concatenate([ M[:,2,1].reshape(1,N),
                             M[:,0,2].reshape(1,N),
                             M[:,1,0].reshape(1,N) ])        
    else:
        raise ValueError(f"{M.shape} cannot be interpreted as a SO(3) map")
    return v

def atan2(Y, X) -> np.ndarray:
    """Numpy's arctan2, but output is at least 1D array"""
    return np.atleast_1d(np.arctan2(Y,X))

def asin(X) -> np.ndarray:
    """Numpy's arcsin, but output is at least 1D array"""
    return np.atleast_1d(np.arcsin(X))

def stack_squeeze(arr) -> np.ndarray:
    """Stack along axis 0, then squeeze to remove any trailing dimensions of size 1"""
    return np.squeeze(np.stack( arr ))

def eul2rotm(eul):
    """Euler Angle to Rotation Matrix"""
    return R.from_euler('ZYX', eul[::-1].T).as_matrix()

def eul2quat(eul):
    """Euler Angle to Quaternion"""
    quat = R.from_euler('ZYX', eul[::-1].T).as_quat()
    # Rearrange from (x,y,z,w) to (w,x,y,z)
    return np.concatenate([quat[3:4], quat[0:3]])

def quat2eul(quat) -> np.ndarray:
    """Euler Angles from Quaternion
    (4,N) --> (3,N) """
    # Convert from (w,x,y,z) to (x,y,z,w)
    quat_scipy = np.concatenate([quat[1:4], quat[0:1]])
    eul = R.from_quat(quat_scipy.T).as_euler('ZYX').T
    return np.squeeze(eul[::-1])

def quat2rotm(quat) -> np.ndarray:
    """Rotation matrix from quaternion
    (4,N) --> (N,3,3) """
    # Convert from (w,x,y,z) to (x,y,z,w)
    quat_scipy = np.concatenate([quat[1:4], quat[0:1]])
    return R.from_quat(quat_scipy.T).as_matrix()

def rotm2eul(rotm) -> np.ndarray:
    eul = R.from_matrix(rotm).as_euler('ZYX')
    return np.squeeze(eul[::-1])

def rotm2quat(rotm) -> np.ndarray:
    quat = R.from_matrix(rotm).as_quat()
    # Rearrange from (x,y,z,w) to (w,x,y,z)
    return np.concatenate([quat[3:4], quat[0:3]])