import numpy as np
import casadi as ca
from scipy.interpolate import interp1d
import time
import threading
import logging
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from structure.Subsystem import Subsystem
from Robot.Constants import Constants

logger = logging.getLogger(f"{__name__}.PathFollowing")
logger.setLevel(logging.INFO)


class PathFollowing(Subsystem):
    """Model Predictive Control Navigator for path following.
    
    This class wraps the MPC solver and provides a threaded interface for
    continuous path following using feedback from the KalmanStateEstimator.
    """
    
    _instance = None

    def __new__(cls):
        """Singleton pattern: return the same instance every time."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.start()
        return cls._instance
    
    def start(self):
        """Initialize MPC Navigator with default parameters."""
        # ────────────────────────────────────────────────
        # Parameters & Constants
        # ────────────────────────────────────────────────
        self.Ts = 0.1
        self.p = 12
        self.L = 0.25
        self.v_nom = Constants.rear_motor_top_speed / 2.0
        self.ds = self.v_nom * self.Ts
        self.ds_ref = 0.1  # Fixed arc-length spacing for reference trajectory (10 cm per MPC step)
        
        # Weights (Q for state, R for input, Rd for rate of change, V for speed tracking)
        self.Q_diag = np.array([10.0, 10.0, 1.0])
        self.R_diag = np.array([0.1, 0.1])
        self.Rd_diag = np.array([1.0, 5.0])
        self.V_weight = 5.0  # Weight for speed tracking cost
        
        # Constraints
        self.v_bounds = [-Constants.rear_motor_top_speed, Constants.rear_motor_top_speed]
        self.delta_bounds = [-np.deg2rad(30), np.deg2rad(30)]
        
        # State bounds
        self.lbx = np.array([-np.inf, -np.inf, -np.inf] * (self.p + 1) + 
                           [self.v_bounds[0], self.delta_bounds[0]] * self.p)
        self.ubx = np.array([np.inf, np.inf, np.inf] * (self.p + 1) + 
                           [self.v_bounds[1], self.delta_bounds[1]] * self.p)
        
        # Setup MPC solver
        self.solver, self.n_states, self.n_controls = self._setup_mpc()
        
        # Constraint bounds (g=0 for initial state and dynamics)
        self.lbg = np.zeros((self.p + 1) * self.n_states)
        self.ubg = np.zeros((self.p + 1) * self.n_states)
        
        # Path matrix
        self.path_matrix = None
        
        # Thread control
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        
        # Current commands (output)
        self._v_cmd = 0.0
        self._delta_cmd = 0.0
        
        # MPC state
        self._last_u = np.array([0.0, 0.0])
        self._x_prev = None  # For warm starting
        
        # Path completion tracking
        self.goal_tolerance = 0.1  # meters - distance threshold to consider goal reached
        
        # Get reference to state estimator
        self.state_estimator = KalmanStateEstimator()
    
    def _setup_mpc(self):
        """Setup the MPC solver using CasADi."""
        # Symbolic states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.size1()
        
        # Symbolic inputs
        v = ca.SX.sym('v')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(v, delta)
        n_controls = controls.size1()
        
        # Right hand side (Bicycle Model)
        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v/self.L * ca.tan(delta))
        f = ca.Function('f', [states, controls], [rhs])
        
        # Optimization variables
        U = ca.SX.sym('U', n_controls, self.p)
        X = ca.SX.sym('X', n_states, self.p + 1)
        # Parameters: initial state (3) + prev control (2) + pose refs (3*(p+1)) + speed refs (p+1)
        P = ca.SX.sym('P', n_states + n_controls + (self.p+1)*3 + (self.p+1))
        
        cost_fn = 0
        g = []
        
        # Unpack parameters
        x_init = P[0:3]
        u_prev = P[3:5]
        pose_ref_end = 5 + (self.p+1)*3
        ref_traj = ca.reshape(P[5:pose_ref_end], 3, self.p+1)
        v_ref = P[pose_ref_end:pose_ref_end + self.p + 1]
        
        # Initial state constraint
        g.append(X[:, 0] - x_init)
        
        for k in range(self.p):
            st = X[:, k]
            con = U[:, k]
            
            # Tracking cost
            cost_fn += ca.mtimes([(st - ref_traj[:, k]).T, np.diag(self.Q_diag), 
                                  (st - ref_traj[:, k])])
            # Input effort cost
            cost_fn += ca.mtimes([con.T, np.diag(self.R_diag), con])
            # Speed tracking cost (track desired speed reference)
            cost_fn += self.V_weight * (con[0] - v_ref[k])**2
            # Smoothness cost
            u_compare = u_prev if k == 0 else U[:, k-1]
            cost_fn += ca.mtimes([(con - u_compare).T, np.diag(self.Rd_diag), 
                                  (con - u_compare)])
            
            # Dynamics constraint
            st_next = X[:, k+1]
            f_value = f(st, con)
            st_next_euler = st + (self.Ts * f_value)
            g.append(st_next - st_next_euler)
        
        # Terminal cost
        cost_fn += ca.mtimes([(X[:, self.p] - ref_traj[:, self.p]).T, 
                             np.diag(self.Q_diag), (X[:, self.p] - ref_traj[:, self.p])])
        
        # Reshape for solver
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        nlp_prob = {
            'f': cost_fn,
            'x': opt_vars,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.max_iter': 100
        }
        
        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts), n_states, n_controls
    
    def _generate_reference(self, cur_state):
        """Generate reference trajectory from path matrix using fixed arc-length spacing.
        
        Uses constant arc-length intervals independent of speed to decouple path planning
        from speed control. Speed is handled separately via the speed reference trajectory.
        """
        if self.path_matrix is None:
            return np.zeros((self.p + 1, 3))
        
        x_wp = self.path_matrix[:, 0]
        y_wp = self.path_matrix[:, 1]
        theta_wp = self.path_matrix[:, 2]
        
        dx, dy = np.diff(x_wp), np.diff(y_wp)
        s_wp = np.cumsum(np.sqrt(dx**2 + dy**2))
        s_wp = np.insert(s_wp, 0, 0.0)
        
        interp_x = interp1d(s_wp, x_wp, kind='cubic', fill_value='extrapolate')
        interp_y = interp1d(s_wp, y_wp, kind='cubic', fill_value='extrapolate')
        interp_theta = interp1d(s_wp, theta_wp, kind='cubic', fill_value='extrapolate')
        
        distances = np.sqrt((x_wp - cur_state[0])**2 + (y_wp - cur_state[1])**2)
        closest_idx = np.argmin(distances)
        
        # Interpolate arc-length at robot's position for better accuracy
        if closest_idx == len(x_wp) - 1:
            s_cur = s_wp[-1]
        else:
            # Linear interpolation between two closest waypoints
            sum_distances = distances[closest_idx] + distances[closest_idx + 1]
            alpha = distances[closest_idx] / sum_distances if sum_distances > 0 else 0
            s_cur = s_wp[closest_idx] * (1 - alpha) + s_wp[closest_idx + 1] * alpha
        
        # Fixed arc-length spacing (independent of speed changes)
        ref = np.zeros((self.p + 1, 3))
        for i in range(self.p + 1):
            s_f = min(s_cur + i * self.ds_ref, s_wp[-1])
            ref[i, :] = [interp_x(s_f), interp_y(s_f), interp_theta(s_f)]
        return ref
    
    def set_path(self, path_matrix):
        """Set the path to follow.
        
        Args:
            path_matrix: Nx3 array of [x, y, theta] waypoints
        """
        with self._lock:
            self.path_matrix = np.asarray(path_matrix, dtype=float)
    
    def set_nominal_speed(self, speed_percent):
        """Set the desired nominal speed for path following.
        
        Args:
            speed_percent: Speed as a percentage of maximum (0-100).
                          Negative values indicate reverse direction.
        """
        with self._lock:
            clamped_percent = np.clip(speed_percent, -100, 100)
            self.v_nom = (clamped_percent / 100.0) * Constants.rear_motor_top_speed
            self.ds = self.v_nom * self.Ts
            logger.debug(f"Set nominal speed: {speed_percent}% -> {self.v_nom:.3f} m/s")
    
    def get_nominal_speed(self):
        """Get the current nominal speed setting.
        
        Returns:
            tuple: (speed_m_s, speed_percent) - speed in m/s and percentage
        """
        with self._lock:
            speed_percent = (self.v_nom / Constants.rear_motor_top_speed) * 100.0
            return self.v_nom, speed_percent
    
    def set_speed_tracking_weight(self, weight):
        """Set the weight for speed tracking in the MPC cost function.
        
        Higher weight = MPC prioritizes tracking desired speed over other objectives.
        Lower weight = MPC has more freedom to deviate from speed for better path tracking.
        
        Args:
            weight: Positive float value. Typical range: 0.5 - 20.0 (default: 5.0)
        """
        if weight <= 0:
            logger.warning(f"Speed tracking weight must be positive. Got {weight}, ignoring.")
            return
        
        with self._lock:
            self.V_weight = weight
            logger.debug(f"Set speed tracking weight: {weight}")
    
    def get_path(self):
        """Get the current reference path.
        
        Returns:
            Nx3 numpy array of [x, y, theta] waypoints, or None if no path set
        """
        with self._lock:
            return self.path_matrix.copy() if self.path_matrix is not None else None
    
    def start_path_following(self):
        """Start the MPC path following in a separate thread."""
        with self._lock:
            if self._running:
                logger.debug("Path following already running")
                return
            
            if self.path_matrix is None:
                logger.error("No path set. Call set_path() first.")
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()
            logger.info("MPC path following started")
    
    def stop_path_following(self):
        """Stop the MPC path following."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            logger.info("MPC path following stopped")
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
    
    def get_current_commands(self):
        """Get the current velocity and steering commands.
        
        Returns:
            tuple: (velocity [m/s], steering_angle [rad])
        """
        with self._lock:
            return self._v_cmd, self._delta_cmd
    
    def is_running(self):
        """Check if path following is active."""
        with self._lock:
            return self._running
    
    def is_at_goal(self, tolerance=None):
        """Check if the robot has reached the end of the path.
        
        Args:
            tolerance: Distance threshold in meters to consider goal reached.
                      If None, uses self.goal_tolerance (default 0.1m)
        
        Returns:
            bool: True if robot is within tolerance of the final waypoint
        """
        if tolerance is None:
            tolerance = self.goal_tolerance
            
        with self._lock:
            if self.path_matrix is None:
                logger.warning("is_at_goal called but no path set")
                return False
            
            current_state = self.state_estimator.get_state()
            distance_to_goal = self._get_distance_to_goal(current_state.pos)
            return distance_to_goal is not None and distance_to_goal <= tolerance
    
    def get_distance_to_goal(self):
        """Get the current distance to the end of the path.
        
        Returns:
            float: Distance in meters to the final waypoint, or None if no path set
        """
        with self._lock:
            return self._get_distance_to_goal(self.state_estimator.get_state().pos)
    
    def _get_distance_to_goal(self, current_state):
        """Helper function to compute distance to goal from a given state."""
        with self._lock:
            if self.path_matrix is None:
                return None
            
            current_x = current_state[0]
            current_y = current_state[1]
            
            goal_x = self.path_matrix[-1, 0]
            goal_y = self.path_matrix[-1, 1]
            
            distance_to_goal = np.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)
            return distance_to_goal
    
    def _control_loop(self):
        """Main control loop running in separate thread."""
        logger.info("MPC control loop started")
        next_time = time.time()
        
        while True:
            with self._lock:
                if not self._running:
                    break
            
            next_time += self.Ts
            start_time = time.time()
            
            try:
                # Get current state from Kalman filter
                state = self.state_estimator.get_state()
                cur_state = np.array([state.pos[0], state.pos[1], 
                                     self.state_estimator.euler[2]])  # x, y, yaw
                
                # Generate reference trajectory
                refs = self._generate_reference(cur_state)
                
                # Generate speed reference trajectory (constant nominal speed)
                with self._lock:
                    v_nom_current = self.v_nom
                v_ref = np.full(self.p + 1, v_nom_current)
                
                # Prepare parameters
                params = np.concatenate([cur_state, self._last_u, refs.flatten(), v_ref])
                
                # Solve MPC
                solver_args = {
                    'lbx': self.lbx, 
                    'ubx': self.ubx, 
                    'lbg': self.lbg, 
                    'ubg': self.ubg, 
                    'p': params
                }
                if self._x_prev is not None:
                    solver_args['x0'] = ca.DM(self._x_prev)
                
                res = self.solver(**solver_args)
                
                # Extract control commands
                u_offset = self.n_states * (self.p + 1)
                u_opt = res['x'][u_offset : u_offset + self.n_controls]
                v_cmd = float(u_opt[0])
                delta_cmd = float(u_opt[1])
                
                # Update stored values
                with self._lock:
                    self._v_cmd = v_cmd
                    self._delta_cmd = delta_cmd
                    self._last_u = np.array([v_cmd, delta_cmd])
                    self._x_prev = res['x']
                
                elapsed = time.time() - start_time
                logger.debug(
                    "MPC: V=%.2f m/s | δ=%.1f° | Pos=(%.2f, %.2f) | Time=%.3fs",
                    v_cmd,
                    np.degrees(delta_cmd),
                    cur_state[0],
                    cur_state[1],
                    elapsed,
                )
                
            except Exception:
                logger.exception("MPC error")
                with self._lock:
                    self._v_cmd = 0.0
                    self._delta_cmd = 0.0
            
            # Timing control
            sleep_duration = next_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        
        logger.info("MPC control loop stopped")