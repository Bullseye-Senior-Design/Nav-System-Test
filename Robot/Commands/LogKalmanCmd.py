from structure.commands.Command import Command
import time
import csv
import os
from pathlib import Path
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
import numpy as np


class LogKalmanCmd(Command):
    """Periodic logger that prints the Kalman filter state and covariance to console."""

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.start_time = time.time()
        # create timestamped folder under ./sim_log/<timestamp>/
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.start_time))
        base_log_dir = Path.cwd() / 'sim_log'
        self.log_dir = base_log_dir / ts_str
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        self.csv_file = str(self.log_dir / 'kalman.csv')

        # ensure header exists
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                # fields: timestamp, px,py,pz, vx,vy,vz, yaw,pitch,roll, diag0..diag8, trace
                fieldnames = ['timestamp', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'yaw_deg', 'pitch_deg', 'roll_deg']
                # covariance diagonal entries
                fieldnames += [f'diag{i}' for i in range(9)]
                fieldnames += ['trace']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def execute(self):
        try:
            kf = KalmanStateEstimator()
            state = kf.get_state()
            euler = kf.euler  # radians
            yaw = float(euler[2]) * 180.0 / np.pi
            pitch = float(euler[1]) * 180.0 / np.pi
            roll = float(euler[0]) * 180.0 / np.pi

            ts = time.time()
            # write a CSV row containing state and covariance diagonal+trace
            row = {
                'timestamp': ts,
                'px': float(state.pos[0]),
                'py': float(state.pos[1]),
                'pz': float(state.pos[2]),
                'vx': float(state.vel[0]),
                'vy': float(state.vel[1]),
                'vz': float(state.vel[2]),
                'yaw_deg': float(yaw),
                'pitch_deg': float(pitch),
                'roll_deg': float(roll),
            }
            
            try:
                with open(self.csv_file, 'a', newline='') as f:
                    fieldnames = ['timestamp', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'yaw_deg', 'pitch_deg', 'roll_deg']
                    fieldnames += [f'diag{i}' for i in range(9)]
                    fieldnames += ['trace']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)
            except Exception as e:
                print(f"LogKalmanCmd: failed to write CSV: {e}")

        except Exception as e:
            print(f"LogKalmanCmd execute error: {e}")

    def end(self, interrupted):
        print("LogKalmanCmd ended")

    def is_finished(self) -> bool:
        # Run indefinitely until cancelled
        return False
