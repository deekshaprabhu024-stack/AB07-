# AB07 — Drone System Identification Using DMDc

A Python-based system-identification project that simulates a single quadrotor in **PyBullet** (via `gym-pybullet-drones`), collects flight data under **PID control**, and identifies a reduced linear discrete-time dynamical model of the drone using **Dynamic Mode Decomposition with Control (DMDc)** and **Singular Value Decomposition (SVD)**.

The identified model takes the form:

```
x_(k+1) = A x_k + B u_k
```

where `A ∈ R^(6×6)` and `B ∈ R^(6×4)` are learned directly from simulated flight data, rather than derived analytically from the full nonlinear quadrotor dynamics.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Drone Simulation](#2-drone-simulation)
3. [Data Collection](#3-data-collection)
4. [Why System Identification Is Used](#4-why-system-identification-is-used)
5. [State-Space Model](#5-state-space-model)
6. [Dynamic Mode Decomposition with Control](#6-dynamic-mode-decomposition-with-control)
7. [Snapshot Matrices](#7-snapshot-matrices)
8. [Singular Value Decomposition](#8-singular-value-decomposition)
9. [DMDc Implementation](#9-dmdc-implementation)
10. [One-Step Prediction](#10-one-step-prediction)
11. [Multi-Step Rollout](#11-multi-step-rollout)
12. [RMSE Evaluation](#12-rmse-evaluation)
13. [Results](#13-results)
14. [Repository Structure](#14-repository-structure)
15. [Installation](#15-installation)
16. [How to Run](#16-how-to-run)
17. [Command-Line Options](#17-command-line-options)
18. [Implementation Notes](#18-implementation-notes)
19. [Limitations](#19-limitations)
20. [Quick Reference](#20-quick-reference)

---

## 1. Project Overview

This project studies the dynamics of a simulated quadrotor by combining classical control (PID) with data-driven modeling (DMDc). The overall pipeline is:

```
Reference Position
        ↓
PID Controller
        ↓
Four Motor RPM Commands
        ↓
PyBullet Quadrotor Simulation
        ↓
State + Motor Data Logging
        ↓
Snapshot Matrices
        ↓
SVD-Based DMDc
        ↓
Identify A and B
        ↓
One-Step Prediction
        ↓
Multi-Step Rollout
        ↓
RMSE Evaluation
        ↓
Plots + Saved Model
```

The project uses the identified model for:

- One-step prediction
- Multi-step rollout
- RMSE evaluation
- Position prediction visualization
- Velocity prediction visualization
- Singular-value visualization
- Saving the identified DMDc model to disk

---

## 2. Drone Simulation

The simulation and data-collection stage is implemented in:

```
project/run_pid_dmdc.py
```

It relies on:

- **Python**
- **NumPy**
- **gym-pybullet-drones**
- `CtrlAviary`
- `DSLPIDControl`
- `DroneModel.CF2X`
- **PyBullet**

### 2.1 Simulation Setup

A **single** quadrotor (`DroneModel.CF2X`) is simulated with the following parameters:

| Parameter | Value |
|---|---|
| Physics frequency | 240 Hz |
| Control frequency | 48 Hz |
| Default simulation duration | 14 seconds |
| Random seed | 42 |
| GUI | Optional |
| Obstacles | None |
| Recording | Disabled |

The drone starts at approximately `[0.0, 0.0, 0.10]`.

### 2.2 Reference Trajectory

A **PID controller** (`DSLPIDControl.computeControlFromState(...)`) drives the drone to follow a stepped position reference that changes over time:

| Time window | Reference position `[x, y, z]` |
|---|---|
| `t < 2 s` | `[0.0, 0.0, 0.30]` |
| `2 ≤ t < 5 s` | `[0.0, 0.0, 0.70]` |
| `5 ≤ t < 8 s` | `[0.30, 0.0, 0.70]` |
| `8 ≤ t < 11 s` | `[0.30, 0.0, 0.40]` |
| `t ≥ 11 s` | `[0.0, 0.0, 0.40]` |

A **changing** reference trajectory is deliberately used because system identification requires the drone to visit a variety of states under a variety of control inputs. A trajectory that never changes would not excite the dynamics enough to produce informative data for identification.

The PID controller outputs four **motor RPM commands** at every control step. These RPM commands, together with the resulting drone state, are what gets logged for later use.

> **Important distinction:** PID control is used only to *fly the simulated drone and generate data*. It is **not** the DMDc identification model. DMDc is a separate, subsequent step that learns a model from the data the PID-controlled flight produces.

---

## 3. Data Collection

The simulation writes its recorded data to:

```
data/pid_dmdc_log.csv
```

### 3.1 Logged Columns

The CSV file contains the following columns:

| Column | Description |
|---|---|
| `time` | Simulation timestamp |
| `target_x`, `target_y`, `target_z` | Reference position given to the PID controller |
| `x`, `y`, `z` | Drone position |
| `qx`, `qy`, `qz`, `qw` | Drone orientation quaternion |
| `vx`, `vy`, `vz` | Linear velocity |
| `wx`, `wy`, `wz` | Angular velocity |
| `rpm1`, `rpm2`, `rpm3`, `rpm4` | Commanded motor RPMs |

### 3.2 What DMDc Actually Uses

Although the log contains position, quaternion attitude, linear velocity, and angular velocity, the DMDc identification script uses only a **reduced subset** of these columns:

**State vector (6 dimensions):**
```
x, y, z, vx, vy, vz
```

**Input vector (4 dimensions):**
```
rpm1, rpm2, rpm3, rpm4
```

The quaternion (`qx, qy, qz, qw`) and angular velocity (`wx, wy, wz`) values **are logged by the simulation but are not part of the six-dimensional DMDc state** used in `dmdc_identify.py`. They remain available in the CSV for reference or future extension, but the identified model does not depend on them.

---

## 4. Why System Identification Is Used

System identification is the process of observing how a system responds to inputs and using those observations to build an approximate mathematical model of its behavior — as opposed to deriving that model purely from first-principles physics.

In this project, instead of manually deriving the complete nonlinear equations of motion for a quadrotor, the observed input/state data (motor RPMs and resulting position/velocity) is used to identify a **linear, discrete-time approximation**:

```
x_(k+1) = A x_k + B u_k
```

Such a model can be useful for:

- **Prediction** — estimating future states given current state and inputs
- **Analysis** — studying how the system behaves near the operating conditions covered by the data
- **Control design** — providing a simplified model that downstream controllers could use
- **Understanding system dynamics** — revealing which state/input relationships dominate the response
- **Data-driven modeling** — building a model directly from observed behavior rather than from analytical derivation

This model is a **linear approximation** identified from a specific dataset. It is **not** claimed to be an exact physical representation of the real quadrotor dynamics.

---

## 5. State-Space Model

### 5.1 State Vector

```
x_k = [x, y, z, vx, vy, vz]^T
```

where:

- `x, y, z` — position of the drone
- `vx, vy, vz` — linear velocity of the drone

### 5.2 Control (Input) Vector

```
u_k = [rpm1, rpm2, rpm3, rpm4]^T
```

where each entry is the commanded RPM of one of the drone's four motors.

### 5.3 Identified Model

```
x_(k+1) = A x_k + B u_k
```

- `A ∈ R^(6×6)` — describes how the **current state** contributes to the **next state**.
- `B ∈ R^(6×4)` — describes how the **motor inputs** contribute to the **next state**.

Both `A` and `B` are **identified from data** using DMDc — they are not manually derived from the full nonlinear quadrotor equations of motion.

---

## 6. Dynamic Mode Decomposition with Control

**Dynamic Mode Decomposition (DMD)** is a data-driven technique for extracting a linear operator that best describes how a system's state evolves over time, based purely on measured snapshots of that state.

Standard DMD only considers state evolution and does not account for external control inputs. Since this project's drone is actively controlled by motor RPM commands, an extension is required: **DMDc (DMD with Control)**.

DMDc explicitly includes an input/control matrix `B` alongside the state-transition matrix `A`, so that the effect of control inputs on the state evolution can be separated from the system's natural (uncontrolled) dynamics.

### 6.1 Formulation

The project solves for `A` and `B` such that:

```
X_next ≈ A X + B U
```

where:

- `X` — matrix of current-state snapshots
- `X_next` — matrix of next-state snapshots (one step ahead of `X`)
- `U` — matrix of control-input snapshots

To solve for both matrices simultaneously, they are combined into a single unknown matrix:

```
G = [A  B]
```

and the state and input snapshots are stacked into a combined data matrix:

```
Ω = [X]
    [U]
```

so that the identification problem becomes:

```
X_next ≈ G Ω
```

This reframes the DMDc problem as a single linear least-squares problem: find the matrix `G` that best maps the combined state/input matrix `Ω` to the next-state matrix `X_next`. Once `G` is found, it is split back into its `A` (first 6 columns) and `B` (last 4 columns) components.

---

## 7. Snapshot Matrices

The time-series data recorded during simulation is reorganized into **snapshot matrices** for DMDc.

Given a sequence of measured states `x_1, x_2, ..., x_N` and inputs `u_1, u_2, ..., u_(N-1)`:

```
X      = [x_1  x_2  x_3  ...  x_(N-1)]
X_next = [x_2  x_3  x_4  ...  x_N]
U      = [u_1  u_2  u_3  ...  u_(N-1)]
```

`X` and `X_next` are shifted by exactly one timestep: column `i` of `X_next` is the state that immediately follows column `i` of `X`.

Each corresponding column triple `(x_i, u_i, x_(i+1))` represents one training example:

```
current state + current motor input → next state
```

Stacking many such examples across the full recorded trajectory gives DMDc enough data to estimate `A` and `B` via least squares.

---

## 8. Singular Value Decomposition

To solve the least-squares problem robustly, the implementation applies **Singular Value Decomposition (SVD)** to the combined state/input matrix:

```
Ω = [X]
    [U]
```

The SVD decomposes `Ω` as:

```
Ω = U_Ω Σ V_Ω^T
```

- **`U_Ω`** — orthonormal matrix whose columns represent dominant directions ("modes") in the combined state/input space.
- **`Σ`** — diagonal matrix of **singular values**, ordered from largest to smallest, indicating how much each corresponding mode contributes to the overall data.
- **`V_Ω`** — orthonormal matrix relating the singular vectors back to individual data snapshots (time samples).

### 8.1 Why SVD Is Used

SVD provides a numerically stable way to compute a matrix pseudoinverse, which is required to solve the least-squares regression `X_next ≈ G Ω` for `G`. It also reveals the **rank structure** of the collected data: singular values that drop off sharply suggest that the effective dimensionality of the data is lower than the number of raw variables, while a slowly decaying spectrum suggests the data spans most of the available directions. Examining the singular values therefore gives insight into how much independent information the collected dataset actually contains.

### 8.2 Truncation Rank

The implementation truncates the SVD to:

```
rank = 10
```

This value follows directly from the dimensionality of the combined state/input space:

```
6 state variables + 4 input variables = 10 dimensions
```

### 8.3 Computing G

Using the truncated SVD factors (`U_r`, `Σ_r`, `V_r`), the combined matrix is computed as:

```
G = X_next V_r Σ_r^(-1) U_r^T
```

This is the (truncated) least-squares solution for `G` that best satisfies `X_next ≈ G Ω`. Once `G` is obtained, it is split into:

```
G = [A  B]
```

giving the identified state-transition matrix `A` and input matrix `B`.

---

## 9. DMDc Implementation

The identification logic lives in:

```
project/dmdc_identify.py
```

### 9.1 Key Functions

| Function | Role |
|---|---|
| `fit_dmdc()` | Builds the snapshot matrices, performs the truncated SVD, and computes `A` and `B` from the recorded data. |
| `rollout()` | Uses the identified `A` and `B` matrices repeatedly to simulate ("roll out") a predicted state trajectory over multiple timesteps. |
| `main()` | Orchestrates the full pipeline: loading data, calling `fit_dmdc()`, running one-step and rollout predictions, computing RMSE, generating plots, and saving the model. |

### 9.2 How They Work Together

`main()` reads the logged CSV, extracts the six-dimensional state and four-dimensional input columns, and passes them to `fit_dmdc()` to obtain `A` and `B`. It then uses those matrices directly for one-step prediction, and passes them into `rollout()` for multi-step prediction. Finally, `main()` computes RMSE metrics from both predictions, generates the result plots, and saves the identified model to disk.

---

## 10. One-Step Prediction

One-step prediction evaluates the model using the **actual** recorded state at every timestep:

```
x_(k+1) = A x_k + B u_k
```

In the implementation, this is computed in a vectorized form across all timesteps at once:

```python
one_step = A @ X + B @ U
```

At every step, the true, measured state `x_k` (not a previous prediction) is fed into the model along with the true recorded input `u_k`. This isolates the **local accuracy** of the identified model — i.e., how well `A` and `B` predict a single step forward, without any compounding of past prediction errors.

---

## 11. Multi-Step Rollout

Multi-step rollout evaluates how well the model performs when used to predict an entire trajectory, not just a single step.

- The **first** state in the rollout comes from the actual simulation data.
- From that point on, the identified model repeatedly predicts:

```
x_(k+1) = A x_k + B u_k
```

- Critically, each **predicted** state is fed back in as the input state for the next prediction, rather than using the true recorded state.

### 11.1 One-Step vs. Multi-Step

```
ONE-STEP:
Actual state → DMDc model → next predicted state
(repeated fresh from actual data at every step)

MULTI-STEP (ROLLOUT):
Initial actual state
      ↓
DMDc prediction
      ↓
Predicted state
      ↓
DMDc prediction
      ↓
Predicted state
      ↓
      ...
```

Multi-step rollout is inherently more difficult than one-step prediction because prediction errors from earlier steps propagate and accumulate into later steps, rather than being reset by fresh ground-truth data at every step.

The rollout uses the **recorded motor-input sequence** (`rpm1`–`rpm4` from the CSV) at each step, so only the state trajectory is predicted — the inputs themselves are taken directly from the logged data.

---

## 12. RMSE Evaluation

The project quantifies prediction accuracy using **Root Mean Squared Error (RMSE)**, computed separately for:

- One-step prediction → `one_step_rmse`
- Multi-step rollout → `rollout_rmse`

RMSE is computed per state variable:

```
x, y, z, vx, vy, vz
```

### 12.1 RMSE Definition

```
RMSE = sqrt( (1/N) * Σ (y_i − ŷ_i)^2 )
```

where:

- `y_i` — actual (simulated) value
- `ŷ_i` — predicted value from the DMDc model
- `N` — number of samples

RMSE gives a single scalar measure, per state variable, of how far the DMDc predictions deviate from the true simulated trajectory. Comparing `one_step_rmse` and `rollout_rmse` shows how prediction error changes when the model is used for a single step versus an extended, self-fed rollout.

> This README does not report specific numerical RMSE values — those are produced by running the identification script on collected data, and will vary depending on the dataset used.

---

## 13. Results

Running the full pipeline produces the following artifacts in the `results/` directory:

```
results/dmdc_model.npz
results/dmdc_position_prediction.png
results/dmdc_singular_values.png
results/dmdc_velocity_prediction.png
```

### 13.1 `dmdc_model.npz`

A NumPy archive containing the identified model and supporting metadata:

| Key | Contents |
|---|---|
| `A` | Identified 6×6 state-transition matrix |
| `B` | Identified 6×4 input matrix |
| `singular_values` | Singular values from the SVD used during identification |
| `state_columns` | Names of the columns used as the state vector (`x, y, z, vx, vy, vz`) |
| `input_columns` | Names of the columns used as the input vector (`rpm1, rpm2, rpm3, rpm4`) |

### 13.2 `dmdc_position_prediction.png`

Compares the simulated (ground-truth) trajectory against the DMDc rollout prediction for the position components: `x`, `y`, `z`.

### 13.3 `dmdc_velocity_prediction.png`

Compares the simulated (ground-truth) trajectory against the DMDc rollout prediction for the velocity components: `vx`, `vy`, `vz`.

### 13.4 `dmdc_singular_values.png`

Plots the singular values of the combined state/input matrix `Ω` on a logarithmic scale. This visualization is used to inspect the rank structure and information content of the collected dataset — how quickly the singular values decay indicates how many effective dimensions the data spans.

---

## 14. Repository Structure

```
AB07-/
├── README.md
├── data/
│   ├── .gitkeep
│   └── pid_dmdc_log.csv
├── project/
│   ├── .gitkeep
│   ├── dmdc_identify.py
│   └── run_pid_dmdc.py
└── results/
    ├── .gitkeep
    ├── dmdc_model.npz
    ├── dmdc_position_prediction.png
    ├── dmdc_singular_values.png
    └── dmdc_velocity_prediction.png
```

| Directory | Purpose |
|---|---|
| `data/` | Stores the recorded simulation dataset (`pid_dmdc_log.csv`). |
| `project/` | Contains the simulation/data-generation script (`run_pid_dmdc.py`) and the DMDc identification script (`dmdc_identify.py`). |
| `results/` | Contains the identified model (`.npz`) and the generated visualization plots. |

---

## 15. Installation

This project requires **Python 3** along with the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `gym-pybullet-drones`

There is currently no `requirements.txt` in this repository, so dependencies should be installed manually.

### 15.1 Setting Up an Environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 15.2 Installing Dependencies

```bash
# Core scientific packages used by the DMDc identification script
pip install numpy pandas matplotlib

# Simulation environment used by the data-collection script
pip install gym-pybullet-drones
```

> `gym-pybullet-drones` is only required to run the simulation (`run_pid_dmdc.py`). The identification script (`dmdc_identify.py`) only requires NumPy, Pandas, and Matplotlib.

---

## 16. How to Run

The full workflow consists of two steps: run the simulation to generate data, then run the DMDc identification on that data.

### Step 1 — Run the Simulation

Default run:

```bash
python project/run_pid_dmdc.py
```

With the PyBullet GUI enabled:

```bash
python project/run_pid_dmdc.py --gui
```

With a custom simulation duration:

```bash
python project/run_pid_dmdc.py --duration 20
```

With a custom output CSV path:

```bash
python project/run_pid_dmdc.py --output data/my_log.csv
```

By default, output is written to:

```
data/pid_dmdc_log.csv
```

### Step 2 — Run DMDc Identification

```bash
python project/dmdc_identify.py
```

This reads:

```
data/pid_dmdc_log.csv
```

and produces:

```
results/dmdc_model.npz
results/dmdc_position_prediction.png
results/dmdc_velocity_prediction.png
results/dmdc_singular_values.png
```

> **Note:** If the simulation output path is changed using `--output`, `dmdc_identify.py` will still look for `data/pid_dmdc_log.csv` by default. In that case, its input path must be updated directly in the script.

---

## 17. Command-Line Options

`run_pid_dmdc.py` supports the following command-line arguments:

| Argument | Description | Default |
|---|---|---|
| `--gui` | Enables the PyBullet GUI during simulation | Disabled |
| `--duration` | Sets the simulation duration in seconds | `14.0` |
| `--output` | Sets the output path for the recorded CSV log | `data/pid_dmdc_log.csv` |

---

## 18. Implementation Notes

The complete software flow across both scripts is:

```
run_pid_dmdc.py
        ↓
PyBullet simulation
        ↓
PID control
        ↓
CSV logging
        ↓
dmdc_identify.py
        ↓
Snapshot matrices
        ↓
SVD
        ↓
A and B
        ↓
Prediction
        ↓
RMSE
        ↓
Plots + NPZ model
```

`run_pid_dmdc.py` and `dmdc_identify.py` are decoupled by the CSV log file: the simulation script's only responsibility is to produce accurate, timestamped state/input data, while the identification script's only responsibility is to consume that data and produce a model plus evaluation artifacts. This separation allows the simulation to be re-run independently (e.g., with a different duration or GUI setting) without modifying the identification logic, and vice versa.

---

## 19. Limitations

- The identified model is derived from **simulated** data (PyBullet), not from a physical drone.
- The model uses only **six state variables** (`x, y, z, vx, vy, vz`); attitude (quaternion) and angular velocity are logged but not modeled.
- The model is a **linear approximation** of what is, in reality, nonlinear quadrotor dynamics.
- The quality of the identified model depends directly on the data collected — the trajectory, duration, and control excitation used during simulation.
- **Multi-step rollout** predictions can accumulate error over time, since each predicted state feeds into the next prediction.
- The identified `A` and `B` matrices reflect the specific reference trajectory and PID controller used during data collection, and may not generalize to substantially different flight conditions.
- This project does not include any real-world hardware validation or deployment on a physical drone.

---

## 20. Quick Reference

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib
pip install gym-pybullet-drones

# 2. Run the simulation (default settings)
python project/run_pid_dmdc.py

# 2a. Run the simulation with the PyBullet GUI
python project/run_pid_dmdc.py --gui

# 3. Run DMDc identification on the collected data
python project/dmdc_identify.py

# 4. View generated results
#    results/dmdc_model.npz
#    results/dmdc_position_prediction.png
#    results/dmdc_velocity_prediction.png
#    results/dmdc_singular_values.png
```

**Expected output files after a full run:**

| File | Produced by | Description |
|---|---|---|
| `data/pid_dmdc_log.csv` | `run_pid_dmdc.py` | Recorded simulation state/input log |
| `results/dmdc_model.npz` | `dmdc_identify.py` | Identified `A`, `B`, singular values, and column metadata |
| `results/dmdc_position_prediction.png` | `dmdc_identify.py` | Simulated vs. predicted position (`x, y, z`) |
| `results/dmdc_velocity_prediction.png` | `dmdc_identify.py` | Simulated vs. predicted velocity (`vx, vy, vz`) |
| `results/dmdc_singular_values.png` | `dmdc_identify.py` | Singular value spectrum of the combined state/input matrix |
