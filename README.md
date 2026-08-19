# MuJoCo-Based DMDc Quadrotor System Identification

## Overview

This branch implements the **MuJoCo-based system identification** part of our quadrotor project using **Dynamic Mode Decomposition with Control (DMDc)**.

The objective is to learn a data-driven approximation of the quadrotor's dynamics from simulated flight data, identify the system matrices **A** and **B**, and evaluate the learned model on previously unseen validation data.

## Pipeline

Manual quadrotor movement  
→ MuJoCo simulation  
→ State + motor data collection  
→ DMDc system identification  
→ Identify **A, B**  
→ Independent validation  
→ DMDc prediction vs MuJoCo actual  
→ RMSE and trajectory plots

## Quadrotor State and Inputs

The model uses an 8-dimensional state:

$$
x =
\begin{bmatrix}
z_{error} & v_z & roll & pitch & yaw & p & q & r
\end{bmatrix}^{T}
$$

where:

- `z_error` — altitude error
- `vz` — vertical velocity
- `roll`, `pitch`, `yaw` — attitude
- `p`, `q`, `r` — angular velocities

The four motor commands form the control vector:

$$
u =
\begin{bmatrix}
u_1 & u_2 & u_3 & u_4
\end{bmatrix}^{T}
$$

The nominal hover motor value is approximately **2.4525**, based on the 1 kg quadrotor model and gravity.

## DMDc Model

The quadrotor dynamics are approximated using:


$$
x_{k+1} = Ax_k + Bu_k
$$

The collected data is converted into the snapshot matrices \(X\), \(X'\), and \(U\). DMDc then identifies the matrices:

- **A** — state-transition dynamics
- **B** — influence of motor inputs on the state

The identified model is then used for recursive multi-step prediction.

## Data Collection

The quadrotor is manually controlled in MuJoCo. Keyboard commands are passed through the control logic and converted into four motor commands.

The recorded data contains:

- Time
- 8 state variables
- 4 actual motor inputs

The motor inputs used for DMDc are the actual commands applied to the MuJoCo simulation.

## Training and Validation

The current experiment uses separate datasets:

- `manual_training.csv` — used to identify the DMDc model
- `manual_validation.csv` — independent data used to evaluate the model

The validation process recursively predicts the state:

$$
\hat{x}_{k+1} = A\hat{x}_k + Bu_k
$$

and compares the prediction against the actual MuJoCo trajectory.

## Current Results

The current 8-state DMDc model achieved the following validation RMSE:

| State | RMSE |
|---|---:|
| `z_error` | 0.1235 |
| `vz` | 0.0102 |
| `roll` | 0.0077 |
| `pitch` | 0.0085 |
| `yaw` | 0.0371 |
| `p` | 0.0052 |
| `q` | 0.0083 |
| `r` | 0.0110 |

Prediction-versus-actual plots are generated for the individual state variables.

An initial rank-4 model was also evaluated. Independent validation showed that important dynamics were lost, so the current implementation retains all 8 state dimensions.

## Repository Contents

- `dmdc_manual_pipeline.py` — main DMDc identification and validation pipeline
- `manual_control.py` — manual control and data collection
- `quadrotor.xml` — MuJoCo quadrotor model
- `inspect_model.py` — MuJoCo model inspection
- `run_simulation.py` — simulation utility
- `dmdc_pipeline.py` — earlier DMDc/baseline implementation
- `data/` — training and independent validation datasets
- `results/` — identified model outputs, metrics, and prediction plots

## Scope

This branch focuses on **MuJoCo-based DMDc system identification and validation**.

LQR control is **not implemented in this branch**.

The current objective is to demonstrate that a mathematical model of the quadrotor dynamics can be identified from observed state and motor-control data and evaluated on independent simulated behaviour.

## Future Work

Before the final evaluation, the model can be strengthened through:

- Multiple training and validation trajectories
- More diverse manual excitation
- Testing different operating conditions
- Further model-order analysis
- Comparison with the corresponding ArduPilot-based identification
- Potential use of the identified model for data-driven control
