import numpy as np


class PIDController:

    def __init__(self, hover_rpm, dt):

        self.hover_rpm = hover_rpm
        self.dt = dt

        # Position gains
        self.kp = np.array([
            70.0,    # x
            70.0,    # y
            180.0    # z
        ])

        # Velocity damping
        self.kd = np.array([
            35.0,    # vx
            35.0,    # vy
            60.0     # vz
        ])

    def reset(self):
        pass

    def compute(
        self,
        position,
        velocity,
        target,
        **kwargs
    ):

        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        target = np.asarray(target, dtype=float)

        error = target - position

        # PID-like position/velocity control
        control = (
            self.kp * error
            - self.kd * velocity
        )

        # Keep the horizontal corrections small.
        horizontal_x = np.clip(
            control[0],
            -80.0,
            80.0
        )

        horizontal_y = np.clip(
            control[1],
            -80.0,
            80.0
        )

        # Vertical control
        vertical = np.clip(
            control[2],
            -200.0,
            200.0
        )

        base = self.hover_rpm + vertical

        # Motor mixing
        rpm1 = base + horizontal_x + horizontal_y
        rpm2 = base - horizontal_x + horizontal_y
        rpm3 = base - horizontal_x - horizontal_y
        rpm4 = base + horizontal_x - horizontal_y

        rpms = np.array([
            rpm1,
            rpm2,
            rpm3,
            rpm4
        ])

        return np.clip(
            rpms,
            0.0,
            1000.0
        )