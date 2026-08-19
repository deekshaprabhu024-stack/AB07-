import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class DroneEnv(gym.Env):

    def __init__(
        self,
        dt=0.01,
        mass=0.5,
        arm_length=0.2,
        k_f=3.0e-6,
        k_m=1.0e-7,
        gui=False,
    ):
        super().__init__()

        self.dt = dt
        self.mass = mass
        self.arm_length = arm_length
        self.k_f = k_f
        self.k_m = k_m

        self.max_rpm = 1000.0

        self.action_space = spaces.Box(
            low=0.0,
            high=self.max_rpm,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        self.client = p.connect(
            p.GUI if gui else p.DIRECT
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.81,
            physicsClientId=self.client
        )

        p.setTimeStep(
            self.dt,
            physicsClientId=self.client
        )

        p.setRealTimeSimulation(
            0,
            physicsClientId=self.client
        )

        # Store visual parts of the quadcopter
        self.visual_parts = []

        self._build_world()

    # ==========================================================
    # WORLD + DRONE
    # ==========================================================

    def _build_world(self):

        p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client
        )

        # ------------------------------------------------------
        # PHYSICAL BODY
        # ------------------------------------------------------
        #
        # This is still the actual physics body.
        # We make its visual transparent because we will
        # display a separate quadcopter model.
        #

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.12, 0.12, 0.03],
            physicsClientId=self.client
        )

        invisible_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.12, 0.12, 0.03],
            rgbaColor=[0.1, 0.4, 0.9, 0.0],
            physicsClientId=self.client
        )

        self.drone_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=invisible_visual,
            basePosition=[0, 0, 0.5],
            physicsClientId=self.client
        )

        p.changeDynamics(
            self.drone_id,
            -1,
            linearDamping=0.15,
            angularDamping=0.5,
            physicsClientId=self.client
        )

        # Create the visual quadcopter
        self._create_drone_visual()

    # ==========================================================
    # CREATE QUADCOPTER VISUAL
    # ==========================================================

    def _create_drone_visual(self):

        # Dark drone color
        black = [0.03, 0.03, 0.03, 1.0]
        dark_gray = [0.12, 0.12, 0.12, 1.0]

        # Remove any previous visual parts
        self.visual_parts = []

        # ------------------------------------------------------
        # CENTRAL BODY
        # ------------------------------------------------------

        body_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.055, 0.04, 0.012],
            rgbaColor=black,
            physicsClientId=self.client
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=body_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.client
        )

        self.visual_parts.append(
            (body_id, [0, 0, 0.055], [0, 0, 0, 1])
        )

        # ------------------------------------------------------
        # FOUR ARMS
        # ------------------------------------------------------

        arm_length = 0.14
        arm_width = 0.012
        arm_height = 0.008

        arm_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                arm_length,
                arm_width,
                arm_height
            ],
            rgbaColor=dark_gray,
            physicsClientId=self.client
        )

        # X configuration
        arm_angles = [
            45,
            135,
            225,
            315
        ]

        for angle in arm_angles:

            angle_rad = np.deg2rad(angle)

            local_x = 0.0
            local_y = 0.0

            local_quat = p.getQuaternionFromEuler(
                [0, 0, angle_rad]
            )

            arm_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=arm_shape,
                basePosition=[0, 0, 0],
                baseOrientation=local_quat,
                physicsClientId=self.client
            )

            self.visual_parts.append(
                (
                    arm_id,
                    [local_x, local_y, 0.055],
                    local_quat
                )
            )

        # ------------------------------------------------------
        # FOUR MOTORS
        # ------------------------------------------------------

        motor_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.022,
            length=0.018,
            rgbaColor=black,
            physicsClientId=self.client
        )

        motor_positions = []

        for angle in arm_angles:

            angle_rad = np.deg2rad(angle)

            x = arm_length * np.cos(angle_rad)
            y = arm_length * np.sin(angle_rad)

            motor_positions.append(
                [x, y, 0.06]
            )

            motor_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=motor_shape,
                basePosition=[0, 0, 0],
                physicsClientId=self.client
            )

            self.visual_parts.append(
                (
                    motor_id,
                    [x, y, 0.06],
                    [0, 0, 0, 1]
                )
            )

        # ------------------------------------------------------
        # FOUR PROPELLERS
        # ------------------------------------------------------

        prop_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.055,
            length=0.004,
            rgbaColor=[0.02, 0.02, 0.02, 0.65],
            physicsClientId=self.client
        )

        for position in motor_positions:

            prop_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=prop_shape,
                basePosition=[0, 0, 0],
                physicsClientId=self.client
            )

            self.visual_parts.append(
                (
                    prop_id,
                    [position[0], position[1], 0.075],
                    [0, 0, 0, 1]
                )
            )

        self._update_drone_visual()

    # ==========================================================
    # UPDATE VISUAL MODEL
    # ==========================================================

    def _update_drone_visual(self):

        drone_position, drone_orientation = (
            p.getBasePositionAndOrientation(
                self.drone_id,
                physicsClientId=self.client
            )
        )

        for visual_id, local_position, local_orientation in self.visual_parts:

            world_position, world_orientation = (
                p.multiplyTransforms(
                    drone_position,
                    drone_orientation,
                    local_position,
                    local_orientation
                )
            )

            p.resetBasePositionAndOrientation(
                visual_id,
                world_position,
                world_orientation,
                physicsClientId=self.client
            )

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self, seed=None, start_pos=None):

        super().reset(seed=seed)

        if start_pos is None:
            start_pos = [0, 0, 0.5]

        p.resetBasePositionAndOrientation(
            self.drone_id,
            start_pos,
            [0, 0, 0, 1],
            physicsClientId=self.client
        )

        p.resetBaseVelocity(
            self.drone_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.client
        )

        self._update_drone_visual()

        return self._get_state(), {}

    # ==========================================================
    # STATE
    # ==========================================================

    def _get_state(self):

        position, _ = p.getBasePositionAndOrientation(
            self.drone_id,
            physicsClientId=self.client
        )

        velocity, _ = p.getBaseVelocity(
            self.drone_id,
            physicsClientId=self.client
        )

        return np.array(
            [
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ],
            dtype=np.float32
        )

    # ==========================================================
    # ATTITUDE
    # ==========================================================

    def get_attitude(self):

        _, quat = p.getBasePositionAndOrientation(
            self.drone_id,
            physicsClientId=self.client
        )

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)

        _, angular_velocity = p.getBaseVelocity(
            self.drone_id,
            physicsClientId=self.client
        )

        return (
            np.array([roll, pitch, yaw]),
            np.array(angular_velocity)
        )

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action):

        rpm = np.clip(
            np.asarray(action, dtype=float),
            0,
            self.max_rpm
        )

        # Motor thrust
        motor_thrust = self.k_f * rpm**2

        total_thrust = np.sum(motor_thrust)

        # --------------------------------------------------
        # Horizontal control
        # --------------------------------------------------

        front_back = (
            motor_thrust[0]
            + motor_thrust[1]
            - motor_thrust[2]
            - motor_thrust[3]
        )

        left_right = (
            motor_thrust[0]
            + motor_thrust[3]
            - motor_thrust[1]
            - motor_thrust[2]
        )

        horizontal_scale = 0.15

        fx = horizontal_scale * left_right
        fy = horizontal_scale * front_back

        # --------------------------------------------------
        # Apply force at drone's current position
        # --------------------------------------------------

        current_position, _ = (
            p.getBasePositionAndOrientation(
                self.drone_id,
                physicsClientId=self.client
            )
        )

        p.applyExternalForce(
            self.drone_id,
            -1,
            [fx, fy, total_thrust],
            current_position,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client
        )

        p.stepSimulation(
            physicsClientId=self.client
        )

        # Update the visible quadcopter
        self._update_drone_visual()

        state = self._get_state()

        terminated = bool(
            state[2] < 0.05
            or abs(state[0]) > 5
            or abs(state[1]) > 5
        )

        return (
            state,
            0.0,
            terminated,
            False,
            {}
        )

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(self):

        if self.client is not None:

            p.disconnect(
                physicsClientId=self.client
            )

            self.client = None