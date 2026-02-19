import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_reference_csv(folder: str | None = None) -> str:
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
						direction="nearest"
					)

					# Downsample for clearer visualization
					step = max(1, len(merged) // 15)
					merged_sparse = merged.iloc[::step]

					# Plot velocity vectors (scaled for visibility)
					scale = 0.2
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

						ax.arrow(px, py, vx, vy, head_width=0.05, head_length=0.03,
								fc="red", ec="red", alpha=0.6, linewidth=1)

	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.axis("equal")
	ax.grid(True, alpha=0.3)
	ax.legend()
	ax.set_title(os.path.basename(ref_path))
	fig.tight_layout()
	plt.show()

	return ref_path


if __name__ == "__main__":
	plot_reference_csv()

