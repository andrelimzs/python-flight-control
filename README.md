# python-flight-control

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andrelimzs/python-flight-control/structure-as-package?labpath=examples%2Fmulticopter-demo.ipynb)

A demonstration of flight control algorithms, and associated physics models and simulation.



## Installation

1. Clone this repository

```
git clone https://github.com/andrelimzs/python-flight-control.git
```

2. Install required packages with pip

````
pip -r requirements.txt
````

3. Run `examples/multicopter-demo.ipynb` 



## Overview

This is a simulation of a quadcopter, written as a set of ordinary differential equations (ODEs) that are fed into an ODE solver.



## Modular ODE Model

The model is written as a set of modular ODEs, allowing existing modules to be changed or new modules to be added. Each module contains the states and derivative calculation it needs to be run in a solver.

python-flight-control contains the following default modules:

- `Translation`
  Responsible for the integration of acceleration into velocity and position.
- `EulerRotation` or `QuaternionRotation`
  Responsible for the integration of angular acceleration into body rates and either Euler angles or Quaternions.
- `DefaultRotor`
  Models the motor, ESC, propeller subsystem. This is a direct passthrough with no dynamics.
- `DefaultAero`
  Models aerodynamic effects on the quadcopter. This is a placeholder returning zero force/moment.

### Adding New Models

1. Create a new class
2. Add class variable `num_states`
   Which contains the number of states it needs
3. Add class variable `x0`
   Which contains a default initial state vector
   (This can be all zeros. An exception would be a quaternion, which should be initialised to [1,0,0,0].)
4. Add derivative calculation to `__call__(self, params, ...)` method

Each model class must return the same number of derivatives as the number of states it contains.



## Quaternion ODE

This codes implements

> Rucker, Caleb. "Integrating rotations using nonunit quaternions." IEEE Robotics and Automation Letters 3.4 (2018): 2979-2986.

to simulate the rotational dynamics of the quadcopter. Quaternions allow for rotations of more than 90 degrees without gimbal lock or singularities.

Non-unit quaternions allows the use of general ODE solvers, instead of special geometric solvers which preserve the SO(3) structure or expensive/slow exponential updates.

### Derivation 

The relationship between angular velocity $\omega$ and the rotation $R$ is

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_1.svg)

Which simplifies to

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_2.svg)

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_3.svg)

To get $\dot{q}$ from $\omega$

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_4.svg)

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_5.svg)

### Final Equation

The final differential equation is:

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_6.svg)

![eqn_1](https://raw.githubusercontent.com/andrelimzs/python-flight-control/53afcbfcaa35e483e6ee8a74a9e407eb26851f46/docs/equations/eqn_7.svg)

which is used in the ODE solver.

