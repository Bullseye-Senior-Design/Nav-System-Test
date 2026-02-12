import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define ground truth waypoints (x, y positions)
table_size = [((2.667, 0.914), (4.191, 0.914)), ((4.191, 0.914), (4.191, 4.595)), ((4.191, 4.595), (2.667, 4.595)), ((2.667, 4.595), (2.667, 0.914))]

def load_csv_files(data_dir):
    """Load state estimator and UWB position CSV files."""
    state_estimator_path = Path(data_dir) / 'state_estimator.csv'
    if not state_estimator_path.exists():
        state_estimator_path = Path(data_dir) / 'kalman.csv'
    uwb_positions_path = Path(data_dir) / 'uwb_positions.csv'
    
    print("Loading data files...")
    state_estimator_df = pd.read_csv(state_estimator_path)
    uwb_positions_df = pd.read_csv(uwb_positions_path)
    
    return state_estimator_df, uwb_positions_df

def adjust_ground_truth_by_robot_size(table_waypoints, robot_size):
    """Inset each segment by half the robot size both perpendicular to and along the segment."""
    adjusted_waypoints = []
    half_size = robot_size / 2

    for segment in table_waypoints:
        (x1, y1), (x2, y2) = segment

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            nx = dx / length  # direction along the segment
            ny = dy / length
        else:
            nx = ny = 0

        px = -ny  # unit normal (perpendicular)
        py = nx

        # Pull the entire segment inward and shorten it to avoid the original corner points
        new_x1 = x1 + px * half_size + nx * half_size
        new_y1 = y1 + py * half_size + ny * half_size
        new_x2 = x2 + px * half_size - nx * half_size
        new_y2 = y2 + py * half_size - ny * half_size

        adjusted_waypoints.append(((new_x1, new_y1), (new_x2, new_y2)))

    return adjusted_waypoints

def distance_point_to_point(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Calculate perpendicular distance from point (px, py) to line segment."""
    # Vector from start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # If line segment has zero length
    if dx == 0 and dy == 0:
        return distance_point_to_point(px, py, x1, y1)
    
    # Parameter t of closest point on line (clamped to [0,1] for segment)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return distance_point_to_point(px, py, closest_x, closest_y)

def analyze_state_estimator(state_estimator_df, ground_truth):
    """Analyze state estimator data against ground truth."""
    print("\n" + "="*70)
    print("STATE ESTIMATOR ANALYSIS")
    print("="*70)
    
    # Extract position columns
    positions = state_estimator_df[['px', 'py']].values
    
    # Calculate minimum distance from each position to any ground truth line segment
    distances = []
    for pos in positions:
        min_distance = float('inf')
        for segment in ground_truth:
            x1, y1 = segment[0]
            x2, y2 = segment[1]
            
            distance = distance_point_to_line_segment(pos[0], pos[1], x1, y1, x2, y2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    distances = np.array(distances)
    
    print(f"Total samples: {len(distances)}")
    print(f"Mean distance from ground truth: {np.mean(distances):.4f} meters")
    print(f"Median distance from ground truth: {np.median(distances):.4f} meters")
    print(f"Std deviation: {np.std(distances):.4f} meters")
    print(f"Min distance: {np.min(distances):.4f} meters")
    print(f"Max distance: {np.max(distances):.4f} meters")
    print(f"RMSE: {np.sqrt(np.mean(distances**2)):.4f} meters")
    
    return distances, positions

def analyze_uwb_positions(uwb_positions_df, ground_truth):
    """Analyze UWB position data against ground truth."""
    print("\n" + "="*70)
    print("UWB POSITIONS ANALYSIS")
    print("="*70)

    # Extract UWB tag 1 and tag 2 positions
    uwb_data = uwb_positions_df[['x1', 'y1', 'x2', 'y2']].copy()

    # Collect tag-specific samples for plotting
    tag1_positions = uwb_data[['x1', 'y1']].dropna(subset=['x1', 'y1']).to_numpy()
    tag2_positions = uwb_data[['x2', 'y2']].dropna(subset=['x2', 'y2']).to_numpy()

    # Average the two UWB positions when both are available (used for analysis)
    positions = []
    for _, row in uwb_data.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

        valid_positions = []
        if pd.notna(x1) and pd.notna(y1):
            valid_positions.append((x1, y1))
        if pd.notna(x2) and pd.notna(y2):
            valid_positions.append((x2, y2))

        if len(valid_positions) == 2:
            avg_x = (valid_positions[0][0] + valid_positions[1][0]) / 2
            avg_y = (valid_positions[0][1] + valid_positions[1][1]) / 2
            positions.append([avg_x, avg_y])
        elif len(valid_positions) == 1:
            positions.append(list(valid_positions[0]))

    if len(positions) == 0:
        print("No valid UWB position data available!")
        return None, None, tag1_positions, tag2_positions

    positions = np.array(positions)
    print(f"Total valid UWB samples (averaged for analysis): {len(positions)}")

    # Calculate minimum distance from each position to any ground truth line segment
    distances = []
    for pos in positions:
        min_distance = float('inf')
        for segment in ground_truth:
            x1, y1 = segment[0]
            x2, y2 = segment[1]

            distance = distance_point_to_line_segment(pos[0], pos[1], x1, y1, x2, y2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)

    if len(distances) > 0:
        distances = np.array(distances)
        print(f"Mean distance from ground truth: {np.mean(distances):.4f} meters")
        print(f"Median distance from ground truth: {np.median(distances):.4f} meters")
        print(f"Std deviation: {np.std(distances):.4f} meters")
        print(f"Min distance: {np.min(distances):.4f} meters")
        print(f"Max distance: {np.max(distances):.4f} meters")
        print(f"RMSE: {np.sqrt(np.mean(distances**2)):.4f} meters")
        return distances, positions, tag1_positions, tag2_positions

    print("Could not calculate distances!")
    return None, None, tag1_positions, tag2_positions

def compare_systems(state_distances, uwb_distances):
    """Compare Sensor Fusion and UWB system performance."""
    print("\n" + "="*70)
    print("SYSTEM COMPARISON")
    print("="*70)
    
    if uwb_distances is None or len(uwb_distances) == 0:
        print("UWB comparison not available (insufficient data)")
        return
    
    state_rmse = np.sqrt(np.mean(state_distances**2))
    uwb_rmse = np.sqrt(np.mean(uwb_distances**2))
    
    print(f"\nSensor Fusion - Mean Error: {np.mean(state_distances):.4f} m")
    print(f"Sensor Fusion - RMSE: {state_rmse:.4f} m")
    print(f"UWB Positions  - Mean Error: {np.mean(uwb_distances):.4f} m")
    print(f"UWB Positions  - RMSE: {uwb_rmse:.4f} m")
    print(f"Mean Error Difference: {abs(np.mean(state_distances) - np.mean(uwb_distances)):.4f} m")
    print(f"RMSE Difference: {abs(state_rmse - uwb_rmse):.4f} m")
    
    if np.mean(state_distances) < np.mean(uwb_distances):
        print("→ Sensor Fusion performs BETTER (by mean error)")
    else:
        print("→ UWB Positions performs BETTER (by mean error)")
    
    if state_rmse < uwb_rmse:
        print("→ Sensor Fusion performs BETTER (by RMSE)")
    else:
        print("→ UWB Positions performs BETTER (by RMSE)")

def plot_comparison(state_distances, state_positions, uwb_distances, uwb_positions, ground_truth, uwb_tag1_positions=None, uwb_tag2_positions=None):
    """Create visualization of the comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot ground truth lines
    for i, segment in enumerate(ground_truth):
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3, label='Ground Truth' if i == 0 else '')
        ax.plot([x1, x2], [y1, y2], 'ko', markersize=8)

    # Plot State Estimator positions
    if state_positions is not None:
        ax.scatter(state_positions[:, 0], state_positions[:, 1], c='blue', alpha=0.6, s=30, label='Sensor Fusion')

    # Plot UWB tag-specific positions
    if uwb_tag1_positions is not None and len(uwb_tag1_positions) > 0:
        ax.scatter(uwb_tag1_positions[:, 0], uwb_tag1_positions[:, 1], c='orange', alpha=0.6, s=30, label='UWB Tag 1')
    if uwb_tag2_positions is not None and len(uwb_tag2_positions) > 0:
        ax.scatter(uwb_tag2_positions[:, 0], uwb_tag2_positions[:, 1], c='purple', alpha=0.6, s=30, label='UWB Tag 2')

    # Plot averaged UWB positions used for analysis
    if uwb_positions is not None and len(uwb_positions) > 0:
        ax.scatter(uwb_positions[:, 0], uwb_positions[:, 1], c='green', alpha=0.6, s=30, marker='x', label='UWB Avg of 1 & 2')

    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_title('Position Estimates vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig('position_comparison.png', dpi=100)
    print("\nPlot saved as 'position_comparison.png'")
    plt.show()

def main():
    """Main analysis pipeline."""
    data_dir = Path(__file__).parent
    
    # Load data
    state_estimator_df, uwb_positions_df = load_csv_files(data_dir)
    
    robot_size = 0.2477 
    ground_truth = adjust_ground_truth_by_robot_size(table_size, robot_size)
    
    # Analyze both systems
    state_distances, state_positions = analyze_state_estimator(state_estimator_df, ground_truth)
    uwb_distances, uwb_positions, uwb_tag1_positions, uwb_tag2_positions = analyze_uwb_positions(uwb_positions_df, ground_truth)
    
    # Compare systems
    compare_systems(state_distances, uwb_distances)
    
    # Create visualization
    try:
        plot_comparison(state_distances, state_positions, uwb_distances, uwb_positions, ground_truth, uwb_tag1_positions, uwb_tag2_positions)
    except Exception as e:
        print(f"\nNote: Could not create plot: {e}")

if __name__ == '__main__':
    main()
