import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_reference_csv(folder: str | None = None, ax = None) -> str:
	"""Plot the first CSV in folder that starts with 'reference'.

	Also plots state_estimator.csv when present, with synchronized
	path_following commands overlaid as velocity vectors.

	Returns the path of the plotted file.
	"""
	base_dir = folder or os.path.dirname(__file__)
	pattern = os.path.join(base_dir, "reference*.csv")
	matches = sorted(glob.glob(pattern))
	if not matches:
		raise FileNotFoundError(f"No reference CSV found in {base_dir}")

	ref_path = matches[0]
	df = pd.read_csv(ref_path)
	if not {"x", "y"}.issubset(df.columns):
		raise ValueError("Reference CSV must include 'x' and 'y' columns")

	fig = None
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 8))
	ax.plot(df["x"], df["y"], label="reference path", linewidth=2)

	# Load and plot state estimator data
	state_path = os.path.join(base_dir, "state_estimator.csv")
	if os.path.exists(state_path):
		state_df = pd.read_csv(state_path)
		if {"px", "py", "yaw", "timestamp"}.issubset(state_df.columns):
			ax.plot(state_df["px"], state_df["py"], label="state estimator", linewidth=2)

			# Load and sync path following commands by timestamp
			pf_path = os.path.join(base_dir, "path_following.csv")
			if os.path.exists(pf_path):
				pf_df = pd.read_csv(pf_path)
				if {"timestamp", "motor_speed_mps", "steering_angle_rad"}.issubset(pf_df.columns):
					# Merge on nearest timestamp
					state_df_sorted = state_df.sort_values("timestamp")
					pf_df_sorted = pf_df.sort_values("timestamp")

					# Use pandas merge_asof to join on nearest timestamp
					merged = pd.merge_asof(
						state_df_sorted,
						pf_df_sorted[["timestamp", "motor_speed_mps", "steering_angle_rad"]],
						on="timestamp",
					)

					# Downsample for clearer visualization
					step = max(1, len(merged) // 50)
					merged_sparse = merged.iloc[::step]

					# Plot velocity vectors (scaled for visibility)
					scale = 0.5
					direction_scale = 0.1
					for _, row in merged_sparse.iterrows():
						px, py = row["px"], row["py"]
						yaw = np.radians(row["yaw"])
						speed = row["motor_speed_mps"]
						steering = row["steering_angle_rad"]

						# Body frame velocity with steering applied
						# In bicycle model: heading changes due to steering
						heading = yaw + steering

						# Body velocity vector rotated by yaw
						vx = speed * np.cos(heading) * scale
						vy = speed * np.sin(heading) * scale

						ax.arrow(px, py, vx, vy, head_width=0.2, head_length=0.2,
								fc="red", ec="red", alpha=0.6, linewidth=1)

						# Robot direction vector (based on yaw angle)
						dir_x = np.cos(yaw) * direction_scale
						dir_y = np.sin(yaw) * direction_scale
						ax.arrow(px, py, dir_x, dir_y, head_width=0.1, head_length=0.1,
								fc="blue", ec="blue", alpha=0.7, linewidth=1)

	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.axis("equal")
	ax.grid(True, alpha=0.3)
	
	# Create custom legend entries for vectors
	from matplotlib.patches import Patch
	legend_elements = [
		Patch(facecolor='red', alpha=0.6, label='velocity vector'),
		Patch(facecolor='blue', alpha=0.7, label='robot direction'),
	]
	ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_elements)
	ax.set_title(os.path.basename(ref_path))
	
	if fig is not None:
		fig.tight_layout()
		plt.show()

	return ref_path


def plot_position_comparison(folder: str | None = None, ax = None) -> str:
	"""Plot comparison of reference path, state estimator, and UWB positions.

	Returns the path of the reference file.
	"""
	base_dir = folder or os.path.dirname(__file__)
	pattern = os.path.join(base_dir, "reference*.csv")
	matches = sorted(glob.glob(pattern))
	if not matches:
		raise FileNotFoundError(f"No reference CSV found in {base_dir}")

	ref_path = matches[0]
	df_ref = pd.read_csv(ref_path)
	if not {"x", "y"}.issubset(df_ref.columns):
		raise ValueError("Reference CSV must include 'x' and 'y' columns")

	fig = None
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 8))
	
	# Plot reference path
	ax.plot(df_ref["x"], df_ref["y"], label="reference path", linewidth=2, color="green")

	# Load and plot state estimator data
	state_path = os.path.join(base_dir, "state_estimator.csv")
	if os.path.exists(state_path):
		state_df = pd.read_csv(state_path)
		if {"px", "py"}.issubset(state_df.columns):
			ax.plot(state_df["px"], state_df["py"], label="state estimator", linewidth=2, color="orange")

	# Load and plot UWB positions
	uwb_path = os.path.join(base_dir, "uwb_positions.csv")
	if os.path.exists(uwb_path):
		uwb_df = pd.read_csv(uwb_path)
		if {"x", "y", "tag_id"}.issubset(uwb_df.columns):
			# Plot each tag with a different color
			colors = ["purple", "brown"]
			for idx, tag_id in enumerate(sorted(uwb_df["tag_id"].unique())):
				tag_data = uwb_df[uwb_df["tag_id"] == tag_id].sort_values("timestamp")
				color = colors[idx % len(colors)]
				ax.plot(tag_data["x"], tag_data["y"], label=f"UWB Tag {tag_id}", linewidth=2, color=color, linestyle="--")
		elif {"x", "y"}.issubset(uwb_df.columns):
			ax.plot(uwb_df["x"], uwb_df["y"], label="UWB positions", linewidth=2, color="purple", linestyle="--")


	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.axis("equal")
	ax.grid(True, alpha=0.3)
	ax.legend()
	ax.set_title("Position Comparison: Reference vs State Estimator vs UWB")
	
	if fig is not None:
		fig.tight_layout()
		plt.show()

	return ref_path


def plot_side_by_side(folder: str | None = None) -> None:
	"""Display both plots side by side in one window."""
	base_dir = folder or os.path.dirname(__file__)
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
	
	plot_reference_csv(base_dir, ax1)
	plot_position_comparison(base_dir, ax2)
	
	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	plot_side_by_side()

