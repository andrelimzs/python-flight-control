# python-flight-control

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andrelimzs/python-flight-control/structure-as-package?labpath=examples%2Fmulticopter-demo.ipynb)

A demonstration of flight control algorithms, and associated physics models and simulations.

## Installation

1. Clone this repository

```
git clone https://github.com/andrelimzs/python-flight-control.git
```

2. Install required packages with pip

````
 pip -r requirements.txt
````

3. Or run `examples/multicopter-demo.ipynb` [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andrelimzs/python-flight-control/structure-as-package?labpath=examples%2Fmulticopter-demo.ipynb)

## Overview

This is a simulation of a quadcopter, written as a set of ordinary differential equations (ODE) that are fed into an ODE solver. 

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



