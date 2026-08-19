import os
import numpy as np

from drone_env import DroneEnv
from pid_controller import PIDController


def collect_data(
    num_steps=5000,
    dt=1.0 / 100.0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    env = DroneEnv(dt=dt, gui=False)

    hover_rpm = np.sqrt(
        (env.mass * 9.81 / 4.0) / env.k_f
    )

    controller = PIDController(
        hover_rpm=hover_rpm,
        dt=dt,
    )

    X_list = []
    X_next_list = []
    U_list = []

    state, _ = env.reset()

    targets = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.0, 1.2]),
        np.array([0.0, 0.5, 1.2]),
        np.array([-0.5, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.8]),
        np.array([0.0, 0.0, 1.0]),
    ]

    target_duration = 700

    for step in range(num_steps):

        target_index = min(
            step // target_duration,
            len(targets) - 1,
        )

        target = targets[target_index]

        position = state[:3]
        velocity = state[3:]

        attitude, angular_velocity = env.get_attitude()

        rpms = controller.compute(
            position=position,
            velocity=velocity,
            target=target,
            attitude=attitude,
            angular_velocity=angular_velocity,
            k_f=env.k_f,
            mass=env.mass,
            arm_length=env.arm_length,
            k_m=env.k_m,
        )

        # Small deterministic excitation.
        # This helps DMDc identify the effect of the inputs.
        excitation = 8.0 * np.sin(
            2.0 * np.pi * step / 80.0
        )

        rpms = rpms.copy()

        rpms[0] += excitation
        rpms[2] -= excitation

        rpms = np.clip(
            rpms,
            0.0,
            env.max_rpm,
        )

        next_state, _, terminated, _, _ = env.step(rpms)

        if np.all(np.isfinite(state)) and np.all(
            np.isfinite(next_state)
        ):
            X_list.append(state)
            U_list.append(rpms)
            X_next_list.append(next_state)

        state = next_state

        if terminated:
            # If the drone becomes unstable, restart.
            state, _ = env.reset()
            controller.reset()

    env.close()

    X = np.asarray(X_list, dtype=float).T
    X_next = np.asarray(X_next_list, dtype=float).T
    U = np.asarray(U_list, dtype=float).T

    os.makedirs("data", exist_ok=True)

    np.savez(
        "data/collected_data.npz",
        X=X,
        X_next=X_next,
        U=U,
    )

    print(f"Collected {X.shape[1]} valid samples.")
    print("X shape:", X.shape)
    print("X_next shape:", X_next.shape)
    print("U shape:", U.shape)

    return X, X_next, U