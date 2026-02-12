import numpy as np

# Create a simple straight path: 0.5 meters forward from current position
distance = 1  # meters
num_points = 100
self.path_matrix = np.zeros((num_points, 3))
# Path goes forward in the direction of current yaw
self.path_matrix[:, 0] = start_x + np.linspace(0, distance, num_points) * np.cos(start_yaw)
self.path_matrix[:, 1] = start_y + np.linspace(0, distance, num_points) * np.sin(start_yaw)
self.path_matrix[:, 2] = start_yaw  # Keep same heading

self._last_update_time = 0.0