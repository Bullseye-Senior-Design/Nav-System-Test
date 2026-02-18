"""Extended Kalman Filter for 9-axis IMU (accel, gyro, mag) and UWB ranges.

Trimmed variant: state vector (10) = [pos(3), vel(3), quat(4)].
- pos: x, y, z
- vel: vx, vy, vz
- quat: qx, qy, qz, qw (unit quaternion)

Error-state (9) = [pos_err(3), vel_err(3), att_err(3)].

This implementation removes accel/gyro bias states from the filter. It is
intended for use when biases are handled outside the filter (e.g., by the IMU
preprocessor) or not modeled.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
import threading
from dataclasses import dataclass
from typing import Tuple
from Robot.MathUtil import MathUtil
import time

GRAVITY = np.array([0.0, 0.0, -9.80665])

@dataclass
class State:
    pos: Tuple[float, float, float]      # shape (3,)
    vel: Tuple[float, float, float]      # shape (3,)
    quat: Tuple[float, float, float, float]  # shape (4,)

class KalmanStateEstimator:
    _instance = None

    # When a new instance is created, sets it to the same global instance
    def __new__(cls):
        # If the instance is None, create a new instance
        # Otherwise, return already created instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._start()
        return cls._instance
    
    def _start(self, dt: float = 0.01):
        self.dt = dt
        # Reentrant lock to allow safe concurrent access from multiple threads
        self._lock = threading.RLock()

        # Flag to track if filter has been initialized with first UWB measurement
        self.is_initialized = False

        # Full state: pos(3), vel(3), quat(4) => 10 elements
        self.x = np.zeros(10)
        # init quat as identity
        self.x[6:10] = np.array([0.0, 0.0, 0.0, 1.0])

        # Error-state covariance for [pos(3), vel(3), att_err(3)] => 9x9
        P_pos = np.eye(3) * 1e-0
        P_vel = np.eye(3) * 1e-0
        P_att = np.eye(3) * 1e-0
        self.P = np.zeros((9, 9))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel
        self.P[6:9, 6:9] = P_att

        # Process noise (continuous) in error-state (9x9)
        q_pos = 1e-2
        q_vel = 1e-1
        q_att = 1e-1
        self.Qc = block_diag(np.eye(3) * q_pos, np.eye(3) * q_vel, np.eye(3) * q_att)

        # Measurement noise templates
        self.R_uwb_range = 0.1 ** 2  # 10 cm sigma default (variance)
        # IMU attitude measurement noise (small-angle residuals)
        # default sigma ~0.02 rad (~1.15 deg)
        self.imu_attitude_sigma = 0.02
        self.R_imu_attitude = np.eye(3) * (self.imu_attitude_sigma ** 2)
        # Encoder velocity measurement noise (m/s)^2
        self.R_encoder_velocity = (0.05 ** 2)  # 0.05 m/s sigma
        
        # Bicycle model parameters
        self.L = 0.5  # Wheelbase: distance from rear to front axle [m]
        
        # Control inputs (updated externally before predict)
        self.u_velocity = 0.0  # Rear wheel velocity [m/s]
        self.u_steering = 0.0  # Front wheel steering angle [rad]
        
        # Batch UWB measurement storage
        self._uwb_batch = {}  # {tag_id: (timestamp, position, offset, use_offset)}
        self._uwb_batch_timeout = 0.025  # Maximum wait time in seconds
        
        threading.Thread(target=self._run_loop, daemon=True).start()

    # --- Helpers to access parts of the full state
    @property
    def pos(self) -> np.ndarray:
        with self._lock:
            return self.x[0:3].copy()

    @property
    def vel(self) -> np.ndarray:
        with self._lock:
            return self.x[3:6].copy()

    @property
    def quat(self) -> np.ndarray:
        with self._lock:
            return self.x[6:10].copy()
    
    @property
    def euler(self) -> np.ndarray:
        # quat property already locks so just call it
        q = self.quat
        return MathUtil.quat_to_euler(q)
    
    def get_state(self) -> State:
        with self._lock:
            return State(
                pos=tuple(self.x[0:3]),
                vel=tuple(self.x[3:6]),
                quat=tuple(self.x[6:10]),
            )
    
    # --- Control input setters (called by sensors before predict) ---
    def set_rear_wheel_velocity(self, v: float):
        """Set rear wheel velocity control input [m/s]."""
        with self._lock:
            self.u_velocity = float(v) if np.isfinite(v) else 0.0
    
    def set_steering_angle(self, angle: float):
        """Set front wheel steering angle control input [rad]."""
        with self._lock:
            self.u_steering = float(angle) if np.isfinite(angle) else 0.0
    
    def _run_loop(self):
        """Background thread to run predict at fixed dt intervals."""
        import time
        next_time = time.time()
        while True:
            next_time += self.dt
            # Only run predict if filter has been initialized
            if self.is_initialized:
                self.constant_velocity_predict()
            sleep_duration = next_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
    
    def constant_velocity_predict(self):
        if not self.is_initialized:
            return
        
        with self._lock:
            dt = self.dt
            
            # Position integration
            self.x[0:3] = self.pos + self.vel * dt
            
            # Velocity remains constant
            self.x[3:6] = self.vel
            
            # Attitude remains constant
            self.x[6:10] = self.quat
            
            # Error-state Jacobian F (9x9)
            F = np.zeros((9, 9))
            F[0:3, 3:6] = np.eye(3)  # position ← velocity
            
            # Discretize and propagate covariance
            Phi = np.eye(9) + F * dt
            Qd = Phi @ (self.Qc * dt) @ Phi.T
            self.P = Phi @ self.P @ Phi.T + Qd

    def predict(self):
        """Bicycle kinematic model prediction.
        
        Model: Two-wheel bicycle (front steerable, rear drive)
        Control inputs:
          - u_velocity: rear wheel velocity [m/s]
          - u_steering: front wheel steering angle [rad]
        
        Kinematics:
          Yaw rate: ω = (v / L) * tan(δ)
          Body velocity: v_body = [v, 0, 0]
          World velocity: v_world = R(q) @ v_body
        
        Integration:
          pos = pos + v_world * dt
          q = q ⊗ exp(0.5 * [0, 0, ω] * dt)
        """
        # Don't predict if not yet initialized
        if not self.is_initialized:
            return

        with self._lock:
            dt = self.dt
            
            # Get control inputs
            v = self.u_velocity
            delta = self.u_steering
            
            # Bicycle model: yaw rate from steering geometry
            # ω = (v / L) * tan(δ)
            if abs(delta) < 1e-6 or abs(v) < 1e-6:
                omega_z = 0.0
            else:
                omega_z = (v / self.L) * np.tan(delta)
            
            # Body-frame velocity (rear wheel point)
            v_body = np.array([v, 0.0, 0.0])
            
            # Current rotation (body to world)
            q = MathUtil.quat_normalize(self.quat)
            R = MathUtil.quat_to_rotmat(q)
            
            # Transform to world frame
            v_world = R @ v_body
            
            # Position integration
            self.x[0:3] = self.pos + v_world * dt
            
            # Velocity state
            self.x[3:6] = v_world
            
            # Orientation integration
            omega_body = np.array([0.0, 0.0, omega_z])
            q_dot = 0.5 * MathUtil.quat_mul(q, np.append(omega_body, 0.0))
            q_new = q + q_dot * dt
            self.x[6:10] = MathUtil.quat_normalize(q_new)
            
            # Error-state Jacobian F (9x9)
            F = np.zeros((9, 9))
            F[0:3, 3:6] = np.eye(3)  # position ← velocity
            
            # velocity ← attitude (∂(R @ v_body)/∂θ ≈ R @ [v_body]ₓ)
            def skew(vec):
                return np.array([[0, -vec[2], vec[1]],
                                 [vec[2], 0, -vec[0]],
                                 [-vec[1], vec[0], 0]])
            F[3:6, 6:9] = R @ skew(v_body)
            
            # Discretize and propagate covariance
            Phi = np.eye(9) + F * dt
            Qd = Phi @ (self.Qc * dt) @ Phi.T
            self.P = Phi @ self.P @ Phi.T + Qd

    def update_uwb_range(self, tag_pos_meas: np.ndarray, tag_offset: np.ndarray | None = None, use_offset: bool = True):
        """EKF update using the fused UWB tag world position (POS), not per-anchor ranges.

        Measurement model:
            z (3x1) = h(x) + v,  h(x) = p + R(q) @ o_b
        where
            p is robot position in world (state),
            R(q) is body->world rotation from quaternion,
            o_b is tag offset in body frame (default 0),
            v ~ N(0, R_meas).

        Args:
            tag_pos_meas: (3,) measured tag position in world frame [m]
            tag_offset:   (3,) tag offset in body frame from robot center [m]; None => [0,0,0]
                +x : tag is forward of the robot center
                +y : tag is to the robot's left side
        """        
        with self._lock:
            z = np.asarray(tag_pos_meas, dtype=float).reshape(3)
            if not np.all(np.isfinite(z)):
                return

            # Initialize filter with first UWB measurement
            if not self.is_initialized:
                self.x[0:3] = z  # Set initial position to first UWB measurement
                self.is_initialized = True
                return

            # Decide whether to apply the offset
            o_b = np.zeros(3)
            if use_offset and tag_offset is not None:
                o_b = np.asarray(tag_offset, dtype=float).reshape(3)

            # rotation matrix from body to world
            q = MathUtil.quat_normalize(self.quat)
            R = MathUtil.quat_to_rotmat(q)

            # prediction h(x)
            h = self.pos + R @ o_b
            y = z - h

            # Jacobian H (3x9): [ I3  03  R[o]_x ] wrt error-state [pos, vel, att_err]
            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)

            if np.any(o_b):
                def skew(v):
                    return np.array([[0, -v[2], v[1]],
                                     [v[2], 0, -v[0]],
                                     [-v[1], v[0], 0]])
                H[:, 6:9] = R @ skew(o_b)

            R_meas = np.eye(3) * self.R_uwb_range

            S = H @ self.P @ H.T + R_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return

            K = self.P @ H.T @ Sinv

            dx = (K @ y).flatten()

            # inject and update P under lock
            self._inject_error_state(dx)

            # Joseph form for numerical stability (9x9)
            I = np.eye(9)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T

    def update_imu_attitude(self, q_meas: np.ndarray):
        """EKF attitude update using an external IMU rotation estimate.

        Provide a quaternion q_meas = [qx,qy,qz,qw] The measurement residual is
        the small-angle vector from the quaternion error:

            q_err = q_meas ⊗ conj(q_est)  (ensure qw >= 0)
            y ≈ 2 * q_err.xyz  (3x1)

        The measurement Jacobian for the error-state is simply H = [0 0 I 0 0]
        for the attitude block, making this a direct attitude observation.

        Args:
            q_meas: quaternion [qx,qy,qz,qw] (preferred).
        """
        # Don't update attitude if not yet initialized
        if not self.is_initialized:
            return

        with self._lock:
            q_est = MathUtil.quat_normalize(self.quat)

            # quaternion conjugate (inverse for unit quaternion)
            q_conj = np.array([-q_est[0], -q_est[1], -q_est[2], q_est[3]], dtype=float)
            q_err = MathUtil.quat_mul(q_meas, q_conj)
            # Make scalar part positive to keep smallest rotation
            if q_err[3] < 0:
                q_err = -q_err

            # small-angle residual (3,)
            y = 2.0 * q_err[0:3]
            if not np.all(np.isfinite(y)):
                return

            # H selects the attitude error block in the 9-element error-state
            # error-state: [pos(0:3), vel(3:6), att_err(6:9)]
            H = np.zeros((3, 9))
            H[:, 6:9] = np.eye(3)

            # Use preconfigured IMU attitude measurement covariance
            R_meas = self.R_imu_attitude

            S = H @ self.P @ H.T + R_meas
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return
            K = self.P @ H.T @ Sinv

            dx = (K @ y).flatten()
            self._inject_error_state(dx)

            # Joseph form for numerical stability (9x9)
            I = np.eye(9)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T

    def update_encoder_velocity(self, v_enc: float):
        """EKF update using encoder velocity measurement (forward speed in body frame).

        Measurement model:
            z = v_enc (scalar, forward velocity in body frame)
            h(x) = e_x^T @ R^T @ v_world
        where R is body->world rotation, e_x = [1,0,0] is body forward axis.

        Args:
            v_enc: measured forward velocity in m/s (positive = forward motion)
        """
        # Don't update if not yet initialized
        if not self.is_initialized:
            return

        with self._lock:
            z = float(v_enc)
            if not np.isfinite(z):
                return

            # Get rotation matrix (body to world)
            q = MathUtil.quat_normalize(self.quat)
            R = MathUtil.quat_to_rotmat(q)
            R_T = R.T  # world to body

            # Predicted body-frame forward velocity
            # h = [1,0,0] @ R^T @ v_world = first row of R^T times v_world
            v_world = self.vel
            h = R_T[0, :] @ v_world  # scalar

            y = z - h  # innovation (scalar)

            #print(f"predicted body frame forward velocity {z:.3f} m/s, measured {h:.3f} m/s, residual {y:.3f} m/s")

            # Jacobian H (1x9): wrt error-state [pos, vel, att_err]
            H = np.zeros((1, 9))
            # ∂h/∂vel = R^T[0,:]
            H[0, 3:6] = R_T[0, :]
            
            # ∂h/∂θ: derivative of (R^T @ v) w.r.t. attitude
            # For small angle δθ: δ(R^T @ v) ≈ -R^T @ [v]_x @ δθ
            # So ∂h/∂θ = -e_x^T @ R^T @ [v]_x = -R^T[0,:] @ [v]_x
            def skew(v):
                return np.array([[0, -v[2], v[1]],
                                 [v[2], 0, -v[0]],
                                 [-v[1], v[0], 0]])
            H[0, 6:9] = -R_T[0, :] @ skew(v_world)

            R_meas = self.R_encoder_velocity  # scalar variance

            S = H @ self.P @ H.T + R_meas  # scalar
            K = (self.P @ H.T) / S  # 9x1

            dx = (K * y).flatten()

            # inject and update P under lock
            self._inject_error_state(dx)

            # Joseph form for numerical stability (9x9)
            I = np.eye(9)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + (K * R_meas) @ K.T

    def batch_uwb(self, tag_id: int, tag_pos_meas: np.ndarray, tag_offset: np.ndarray | None = None, use_offset: bool = True):
        """Batch UWB update that waits for measurements from two tags before processing.
        
        This function stores measurements from each tag and waits up to 0.025s for
        measurements from both tags. When both are available (or timeout occurs),
        it processes all available measurements together in a single stacked Kalman update
        to avoid ping-pong errors from sequential updates.
        
        Args:
            tag_id: Integer identifier for the UWB tag (e.g., 0 or 1)
            tag_pos_meas: (3,) measured tag position in world frame [m]
            tag_offset: (3,) tag offset in body frame from robot center [m]; None => [0,0,0]
            use_offset: Whether to apply the tag offset
        """
        with self._lock:
            current_time = time.time()
            
            # Store this measurement
            self._uwb_batch[tag_id] = (
                current_time,
                np.asarray(tag_pos_meas, dtype=float).reshape(3),
                tag_offset,
                use_offset
            )
            
            # Check if we should process the batch
            should_process = False
            
            if len(self._uwb_batch) >= 2:
                # We have measurements from at least 2 tags - process immediately
                should_process = True
            elif len(self._uwb_batch) == 1:
                # Only one measurement - check if we should wait or timeout
                # Don't wait here, just return and let the next call trigger processing
                # Check age of oldest measurement
                oldest_time = min(t for t, _, _, _ in self._uwb_batch.values())
                if current_time - oldest_time >= self._uwb_batch_timeout:
                    should_process = True
            
            if should_process:
                # Process all stored measurements as a single stacked update
                self._process_stacked_uwb_update()
                # Clear the batch
                self._uwb_batch.clear()
    
    def _process_stacked_uwb_update(self):
        """Process stored UWB measurements as a single stacked update.
        
        Stacks all measurements into a single measurement vector and performs
        one Kalman update to avoid ping-pong errors from sequential updates.
        """
        # This assumes _lock is already held by caller
        if not self._uwb_batch:
            return
        
        # Check for any valid measurement to initialize
        for _, (_, pos, _, _) in self._uwb_batch.items():
            if not self.is_initialized:
                if np.all(np.isfinite(pos)):
                    self.x[0:3] = pos  # Set initial position
                    self.is_initialized = True
                    return
        
        # Don't update if still not initialized
        if not self.is_initialized:
            return
        
        # Get current rotation matrix (shared for all tags)
        q = MathUtil.quat_normalize(self.quat)
        R = MathUtil.quat_to_rotmat(q)
        
        # Build stacked measurement vector and Jacobian
        z_list = []
        h_list = []
        H_list = []
        
        def skew(v):
            return np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
        
        for tag_id in sorted(self._uwb_batch.keys()):
            _, pos, offset, use_off = self._uwb_batch[tag_id]
            
            # Skip non-finite measurements
            if not np.all(np.isfinite(pos)):
                continue
            
            # Decide whether to apply the offset
            o_b = np.zeros(3)
            if use_off and offset is not None:
                o_b = np.asarray(offset, dtype=float).reshape(3)
            
            # Measurement
            z_list.append(pos)
            
            # Prediction h(x) = p + R @ o_b
            h = self.pos + R @ o_b
            h_list.append(h)
            
            # Jacobian H (3x9): [ I3  03  R[o]_x ]
            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)
            if np.any(o_b):
                H[:, 6:9] = R @ skew(o_b)
            H_list.append(H)
        
        # Check if we have valid measurements
        if not z_list:
            return
        
        # Stack into single vectors/matrices
        z_stacked = np.concatenate(z_list)  # (3*N, )
        h_stacked = np.concatenate(h_list)  # (3*N, )
        H_stacked = np.vstack(H_list)       # (3*N, 9)
        
        # Innovation
        y = z_stacked - h_stacked
        
        # Measurement covariance (block diagonal)
        n_meas = len(z_list)
        R_single = np.eye(3) * self.R_uwb_range
        R_stacked = block_diag(*[R_single for _ in range(n_meas)])
        
        # Kalman update
        S = H_stacked @ self.P @ H_stacked.T + R_stacked
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        K = self.P @ H_stacked.T @ Sinv
        dx = (K @ y).flatten()
        
        # Inject error state
        self._inject_error_state(dx)
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(9)
        self.P = (I - K @ H_stacked) @ self.P @ (I - K @ H_stacked).T + K @ R_stacked @ K.T

    def _inject_error_state(self, dx: np.ndarray):
        """Inject 9-vector error state into the full state x and renormalize quaternion.

        dx layout: [dp(3), dv(3), dtheta(3)]
        """
        if dx.shape[0] != 9:
            raise ValueError("dx must be length 9")
        with self._lock:
            # sanitize non-finite
            if not np.all(np.isfinite(dx)):
                dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)

            dp = dx[0:3]
            dv = dx[3:6]
            dtheta = dx[6:9]

            # pos
            self.x[0:3] = self.pos + dp
            # vel
            self.x[3:6] = self.vel + dv
            # attitude: apply small-angle
            dq = MathUtil.small_angle_quat(dtheta)
            q = self.quat
            q_new = MathUtil.quat_mul(dq, q)
            q_new = MathUtil.quat_normalize(q_new)
            self.x[6:10] = q_new
