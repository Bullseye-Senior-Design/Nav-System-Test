"""Test script for MPC Path Following with dummy feedback data."""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Robot.Constants import Constants
from tests.debug_mpc.PathFollowing import PathFollowing



class PathType(Enum):
    """Enumeration for reference path types."""
    STRAIGHT = "straight"
    SINWAVE = "sinwave"


class DisturbanceModel(Enum):
    """Enumeration for disturbance models."""
    NONE = "none"
    NOISE = "noise"
    DRIFT = "drift"
    WHEEL_SLIP = "wheel_slip"
    HEADING_BIAS = "heading_bias"
    MEASUREMENT_NOISE = "measurement_noise"


class DummyStateEstimator:
    """Mock state estimator for testing."""
    
    def __init__(self):
        self.state_pos = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.euler = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        
    def set_state(self, x, y, theta):
        """Set the state position."""
        self.state_pos = np.array([x, y, theta])
        self.euler[2] = theta  # yaw
    
    def get_state(self):
        """Return a mock state object."""
        class State:
            def __init__(self, pos):
                self.pos = pos
        return State(self.state_pos[:2])


def create_reference_path():
    """Create a reference path with straight line and 90-degree turn.
    
    Path: 1m straight, then 90 degree right turn, then 1m straight
    """
    # First segment: straight ahead (x-axis)
    segment1_x = np.linspace(0, 1, 15)
    segment1_y = np.zeros(15)
    segment1_theta = np.zeros(15)
    
    # Turn segment: 90-degree right turn (arc)
    turn_angles = np.linspace(0, np.pi/2, 15)
    radius = 0.5
    turn_center_x = 1.0
    turn_center_y = -radius
    segment2_x = turn_center_x + radius * np.sin(turn_angles)
    segment2_y = turn_center_y + radius * np.cos(turn_angles)
    segment2_theta = -turn_angles  # heading changes by the turn angle (clockwise)
    
    # Second segment: straight (negative y-axis)
    segment3_x = np.full(15, 1.0 + radius)
    segment3_y = np.linspace(-radius, -radius - 1, 15)
    segment3_theta = np.full(15, -np.pi/2)  # pointing down (negative y direction)
    
    # Combine segments
    path_x = np.concatenate([segment1_x, segment2_x[1:], segment3_x[1:]])
    path_y = np.concatenate([segment1_y, segment2_y[1:], segment3_y[1:]])
    path_theta = np.concatenate([segment1_theta, segment2_theta[1:], segment3_theta[1:]])
    
    path_matrix = np.column_stack([path_x, path_y, path_theta])
    return path_matrix


def create_sinwave_reference_path(length=5.0, amplitude=0.5, frequency=1.0, num_points=50):
    """Create a sinusoidal reference path.
    
    Path follows a sine wave pattern along the x-axis.
    
    Args:
        length: Total length of the path in meters (default: 5.0)
        amplitude: Amplitude of the sine wave in meters (default: 0.5)
        frequency: Frequency of the sine wave in cycles per length (default: 1.0)
        num_points: Number of waypoints in the path (default: 50)
        
    Returns:
        Nx3 numpy array of [x, y, theta] waypoints
    """
    # Generate x coordinates along the forward direction
    path_x = np.linspace(0, length, num_points)
    
    # Generate y coordinates following a sine wave
    path_y = amplitude * np.sin(2 * np.pi * frequency * path_x / length)
    
    # Calculate heading angle (tangent to the curve)
    # dy/dx = amplitude * cos(2*pi*f*x/length) * (2*pi*f/length)
    path_theta = np.arctan2(
        np.gradient(path_y, path_x),
        np.ones_like(path_x)
    )
    
    path_matrix = np.column_stack([path_x, path_y, path_theta])
    return path_matrix


def simulate_bicycle_model(state, v_cmd, delta_cmd, dt=0.1, L=0.25):
    """Simulate one step of bicycle model dynamics.
    
    Args:
        state: [x, y, theta] current state
        v_cmd: velocity command (m/s)
        delta_cmd: steering angle command (rad)
        dt: time step
        L: wheelbase
        
    Returns:
        Updated state [x, y, theta]
    """
    x, y, theta = state
    
    # Bicycle model
    x_dot = v_cmd * np.cos(theta)
    y_dot = v_cmd * np.sin(theta)
    theta_dot = v_cmd / L * np.tan(delta_cmd)
    
    # Euler integration
    x_new = x + x_dot * dt
    y_new = y + y_dot * dt
    theta_new = theta + theta_dot * dt
    
    return np.array([x_new, y_new, theta_new])


def apply_disturbance(state, v_cmd, disturbance_model=DisturbanceModel.NONE, time_step=0, dt=0.1, L=0.25):
    """Apply disturbances to feedback data to simulate model mismatch or sensor errors.
    
    Args:
        state: [x, y, theta] current state
        v_cmd: velocity command (m/s)
        disturbance_model: DisturbanceModel enum specifying the type of disturbance
        time_step: Current simulation step (for time-dependent disturbances)
        dt: Time step
        L: Wheelbase
        
    Returns:
        Modified state [x, y, theta]
    """
    state = np.copy(state)
    
    if disturbance_model == DisturbanceModel.NONE:
        return state
    
    elif disturbance_model == DisturbanceModel.NOISE:
        # Add white Gaussian noise to position (10mm standard deviation)
        state[0] += np.random.normal(0, 0.01)  # x noise
        state[1] += np.random.normal(0, 0.01)  # y noise
        
    elif disturbance_model == DisturbanceModel.DRIFT:
        # Accumulating drift bias (grows quadratically with time)
        drift_per_step = 0.001 * (time_step * dt)
        state[0] += drift_per_step
        state[1] += drift_per_step * 0.5
        
    elif disturbance_model == DisturbanceModel.WHEEL_SLIP:
        # Simulate asymmetric wheel slip (left wheel slips more)
        slip_factor = 1.1  # Left wheel 10% slower
        x, y, theta = state
        
        # Effective velocity is reduced on one side
        v_eff = v_cmd / slip_factor
        x_dot = v_eff * np.cos(theta)
        y_dot = v_eff * np.sin(theta)
        theta_dot = v_cmd / L * np.tan(np.deg2rad(5))  # Also adds turning bias
        
        state[0] = x + x_dot * dt
        state[1] = y + y_dot * dt
        state[2] = theta + theta_dot * dt
        
    elif disturbance_model == DisturbanceModel.HEADING_BIAS:
        # Systematic heading bias (compass error)
        heading_error = np.deg2rad(2)  # 2 degree persistent heading error
        state[2] += heading_error * (0.1 + np.sin(time_step * dt * 2 * np.pi / 10) * 0.05)
        
    elif disturbance_model == DisturbanceModel.MEASUREMENT_NOISE:
        # Gaussian noise on all measurements
        state[0] += np.random.normal(0, 0.015)
        state[1] += np.random.normal(0, 0.015)
        state[2] += np.random.normal(0, 0.02)
        
    return state


def test_path_following(path_type=PathType.STRAIGHT, disturbance_model=DisturbanceModel.NONE, timeout=None, nominal_speed_percent=75):
    """Test the MPC path following with dummy feedback.
    
    Args:
        path_type: PathType enum specifying the reference path type
        disturbance_model: DisturbanceModel enum specifying the disturbance model
        timeout: Maximum simulation time in seconds (None for unlimited)
    """
    
    print(f"Creating {path_type.value} reference path...")
    if path_type == PathType.SINWAVE:
        path_matrix = create_sinwave_reference_path(length=2.5, amplitude=0.25, frequency=1.5, num_points=60)
        max_time = timeout if timeout is not None else 40.0
        title_suffix = " (Sinusoidal Path)"
    else:
        path_matrix = create_reference_path()
        max_time = timeout if timeout is not None else 30.0
        title_suffix = " (90° Turn)"
    
    if disturbance_model != DisturbanceModel.NONE:
        title_suffix += f" [Disturbance: {disturbance_model.value}]"
    
    print("Initializing MPC path following...")
    pf = PathFollowing()
    pf.set_path(path_matrix)
    pf.set_nominal_speed(nominal_speed_percent)  # 30% of max speed
    
    # Replace the state estimator with dummy
    dummy_estimator = DummyStateEstimator()
    pf.state_estimator = dummy_estimator
    
    print("Starting path following...")
    print("(Press Ctrl+C to stop the simulation)")
    pf.start_path_following()
    
    # Simulation loop
    current_state = np.array([0.0, 0.0, 0.0])  # Start with some initial heading error
    dummy_estimator.set_state(*current_state)
    
    history_state = [current_state.copy()]
    history_commands = []
    history_time = [0.0]
    
    dt = 0.1
    
    start_time = time.time()
    time_step = 0
    stop_reason = "timeout"
    
    print(f"Running simulation with disturbance model: '{disturbance_model.value}'...")
    try:
        while time.time() - start_time < max_time:
            # Get MPC commands
            v_cmd, delta_cmd = pf.get_current_commands()
            history_commands.append([v_cmd, delta_cmd])
            
            # Simulate one step
            current_state = simulate_bicycle_model(current_state, v_cmd, delta_cmd, dt)
            
            # Apply disturbance to feedback
            current_state = apply_disturbance(current_state, v_cmd, disturbance_model, time_step, dt)
            
            # Update state estimator
            dummy_estimator.set_state(*current_state)
            
            history_state.append(current_state.copy())
            history_time.append(time.time() - start_time)
            time_step += 1
            
            # Check if at goal
            if pf.is_at_goal():
                print(f"Reached goal at t={time.time() - start_time:.2f}s")
                stop_reason = "goal_reached"
                break
            
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        stop_reason = "user_interrupt"
    
    pf.stop_path_following()
    
    # Convert to numpy arrays
    history_state = np.array(history_state)
    history_commands = np.array(history_commands)
    history_time = np.array(history_time)
    
    print(f"Simulation complete. Collected {len(history_state)} states. Reason for stopping: {stop_reason}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Path and robot trajectory
    ax = axes[0, 0]
    ax.plot(path_matrix[:, 0], path_matrix[:, 1], 'g-', linewidth=2, label='Reference path')
    ax.plot(history_state[:, 0], history_state[:, 1], 'b-', linewidth=1.5, label='Robot trajectory')
    ax.scatter(path_matrix[0, 0], path_matrix[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(path_matrix[-1, 0], path_matrix[-1, 1], color='red', s=100, marker='s', label='Goal', zorder=5)
    ax.scatter(history_state[0, 0], history_state[0, 1], color='blue', s=80, marker='o', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Path Following Trajectory' + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # Plot 2: Heading angle
    ax = axes[0, 1]
    ax.plot(history_time, np.degrees(path_matrix[np.linspace(0, len(path_matrix)-1, len(history_state), dtype=int), 2]), 
            'g--', linewidth=1.5, label='Reference heading')
    ax.plot(history_time, np.degrees(history_state[:, 2]), 'b-', linewidth=1.5, label='Robot heading')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (°)')
    ax.set_title('Heading Angle')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Velocity command
    ax = axes[1, 0]
    ax.plot(history_time[:-1], history_commands[:, 0], 'r-', linewidth=1.5, label='Velocity')
    ax.axhline(y=pf.v_nom, color='r', linestyle='--', alpha=0.5, label=f'Nominal speed ({pf.v_nom:.2f} m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Command')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Steering command
    ax = axes[1, 1]
    ax.plot(history_time[:-1], np.degrees(history_commands[:, 1]), 'purple', linewidth=1.5, label='Steering angle')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering Angle (°)')
    ax.set_title('Steering Command')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / 'mpc_test_output.png'
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Stop reason: {stop_reason}")
    print(f"Initial position: ({history_state[0, 0]:.2f}, {history_state[0, 1]:.2f})")
    print(f"Final position: ({history_state[-1, 0]:.2f}, {history_state[-1, 1]:.2f})")
    print(f"Final heading: {np.degrees(history_state[-1, 2]):.1f}°")
    print(f"Total distance traveled: {np.sum(np.linalg.norm(np.diff(history_state[:, :2], axis=0), axis=1)):.2f} m")
    print(f"Total time: {history_time[-1]:.2f} s")
    print(f"Max velocity: {np.max(history_commands[:, 0]):.3f} m/s")
    print(f"Max steering angle: {np.degrees(np.max(np.abs(history_commands[:, 1]))):.1f}°")
    print("="*60)


if __name__ == "__main__":
    # Configure test parameters here
    path_type = PathType.SINWAVE
    disturbance_model = DisturbanceModel.NOISE
    timeout = 80  # None for unlimited, or set a value in seconds like 20.0
    
    test_path_following(path_type, disturbance_model, timeout)
