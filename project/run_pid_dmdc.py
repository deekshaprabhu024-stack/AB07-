import os
import time
import argparse
import numpy as np
import pandas as pd

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync


def reference_trajectory(t):
    """
    Generates a takeoff -> circular flight -> hover reference trajectory.

    Returns:
        target_pos : desired [x, y, z] position in metres
        target_vel : desired [vx, vy, vz] velocity in m/s
        target_rpy : desired [roll, pitch, yaw] attitude in radians
    """
    takeoff_end = 3.0
    circle_end = 17.0

    radius = 0.35
    altitude = 0.70
    omega = 0.35

    if t < takeoff_end:
        # Hold above the origin while the drone takes off and stabilizes.
        target_pos = np.array([0.0, 0.0, altitude])
        target_vel = np.zeros(3)
        target_rpy = np.zeros(3)

    elif t < circle_end:
        # Time measured from the beginning of the circular segment.
        circle_t = t - takeoff_end

        target_pos = np.array([
            radius * np.cos(omega * circle_t),
            radius * np.sin(omega * circle_t),
            altitude
        ])

        target_vel = np.array([
            -radius * omega * np.sin(omega * circle_t),
             radius * omega * np.cos(omega * circle_t),
             0.0
        ])

        # Points the drone approximately along the tangent of the circle.
        target_yaw = omega * circle_t + np.pi / 2.0
        target_rpy = np.array([0.0, 0.0, target_yaw])

    else:
        # End by returning to a stable hover at the circle centre.
        target_pos = np.array([0.0, 0.0, altitude])
        target_vel = np.zeros(3)
        target_rpy = np.zeros(3)

    return target_pos, target_vel, target_rpy


def run(gui=True, duration=22.0, output_csv="data/pid_dmdc_log.csv"):
    os.makedirs("data", exist_ok=True)

    drone_model = DroneModel.CF2X
    physics = Physics.PYB
    sim_freq_hz = 240
    control_freq_hz = 48

    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.10]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=physics,
        pyb_freq=sim_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=False
    )

    controller = DSLPIDControl(drone_model=drone_model)
    obs, info = env.reset(seed=42)

    action = np.zeros((1, 4))
    rows = []
    start = time.time()
    num_steps = int(duration * control_freq_hz)

    print("Running PID-controlled circular-flight experiment...")
    print("Logging position, velocity, attitude, angular rate, and four motor RPM commands.")

    for i in range(num_steps):
        t = i * env.CTRL_TIMESTEP

        state = obs[0]

        target_pos, target_vel, target_rpy = reference_trajectory(t)

        rpm, _, _ = controller.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel
        )

        action[0, :] = rpm

        obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs[0]

        rows.append({
            "time": t,

            "target_x": target_pos[0],
            "target_y": target_pos[1],
            "target_z": target_pos[2],

            "target_vx": target_vel[0],
            "target_vy": target_vel[1],
            "target_vz": target_vel[2],

            "target_roll": target_rpy[0],
            "target_pitch": target_rpy[1],
            "target_yaw": target_rpy[2],

            "x": next_state[0],
            "y": next_state[1],
            "z": next_state[2],

            "qx": next_state[3],
            "qy": next_state[4],
            "qz": next_state[5],
            "qw": next_state[6],

            "vx": next_state[10],
            "vy": next_state[11],
            "vz": next_state[12],

            "wx": next_state[13],
            "wy": next_state[14],
            "wz": next_state[15],

            "rpm1": rpm[0],
            "rpm2": rpm[1],
            "rpm3": rpm[2],
            "rpm4": rpm[3]
        })

        if gui:
            env.render()
            sync(i, start, env.CTRL_TIMESTEP)

        if terminated or truncated:
            obs, info = env.reset(seed=42)

    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} samples to: {output_csv}")
    print(df.head())
    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open PyBullet's 3D GUI")
    parser.add_argument("--duration", type=float, default=22.0, help="Simulation duration in seconds")
    parser.add_argument("--output", type=str, default="data/pid_dmdc_log.csv", help="Output CSV path")
    args = parser.parse_args()

    run(gui=args.gui, duration=args.duration, output_csv=args.output)