import time
import numpy as np

from drone_env import DroneEnv
from pid_controller import PIDController


def reached_target(position, velocity, target):
    """
    Check whether the drone has reached a waypoint
    and is reasonably stable there.
    """

    position_error = np.linalg.norm(target - position)
    speed = np.linalg.norm(velocity)

    position_tolerance = 0.06
    velocity_tolerance = 0.12

    return (
        position_error < position_tolerance
        and speed < velocity_tolerance
    )


def main():

    env = DroneEnv(
        dt=1.0 / 100.0,
        gui=True
    )

    hover_rpm = np.sqrt(
        (env.mass * 9.81 / 4.0) / env.k_f
    )

    controller = PIDController(
        hover_rpm=hover_rpm,
        dt=env.dt
    )

    state, _ = env.reset()

    # --------------------------------------------------
    # DESIRED FLIGHT PATH
    # --------------------------------------------------

    targets = [
        np.array([0.0, 0.0, 1.0]),   # 1. Takeoff + hover
        np.array([0.5, 0.0, 1.0]),   # 2. Move along X
        np.array([0.5, 0.5, 1.0]),   # 3. Move along Y
        np.array([0.0, 0.0, 1.0]),   # 4. Return to start
        np.array([0.0, 0.0, 0.10]),  # 5. Descend
    ]

    try:

        for target_number, target in enumerate(targets, start=1):

            print()
            print("--------------------------------")
            print(f"Target {target_number}: {target}")
            print("--------------------------------")

            reached_count = 0

            # Maximum number of simulation steps
            # allowed for this waypoint.
            max_steps = 1200

            for step in range(max_steps):

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

                state, _, terminated, _, _ = env.step(rpms)

                time.sleep(env.dt)

                if terminated:
                    print("Drone became unstable.")
                    return

                # Require the drone to remain close to the
                # target for several consecutive frames.
                if reached_target(
                    state[:3],
                    state[3:],
                    target
                ):
                    reached_count += 1
                else:
                    reached_count = 0

                # Stable at target
                if reached_count >= 40:

                    print(
                        f"Reached target {target_number} "
                        f"after {step + 1} steps."
                    )

                    print(
                        "Position:",
                        np.round(state[:3], 3)
                    )

                    break

            else:

                print(
                    f"Warning: target {target_number} "
                    f"was not fully reached."
                )

        # --------------------------------------------------
        # FINAL LANDING
        # --------------------------------------------------

        print()
        print("--------------------------------")
        print("LANDING")
        print("--------------------------------")

        # Once the drone reaches the low altitude,
        # switch the motors off.
        for _ in range(150):

            rpms = np.zeros(4)

            state, _, terminated, _, _ = env.step(rpms)

            time.sleep(env.dt)

            if terminated:
                break

        print("Landing complete.")

    finally:

        input("Press Enter to close PyBullet...")
        env.close()


if __name__ == "__main__":
    main()