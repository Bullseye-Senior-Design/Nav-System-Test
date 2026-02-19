#!/usr/bin/env python3
"""CSV-driven control inputs simulator (velocity + steering angle).

This simulator reads control input data from a CSV file (timestamp, velocity_mps,
steering_angle_rad, steering_angle_deg) and updates the Kalman filter control
inputs at the appropriate time steps.

Usage: ControlInputs()
"""
import csv
import time
import threading
import logging
from typing import Optional
import numpy as np
from Debug import Debug
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator


logger = logging.getLogger(__name__)


class ControlInputs:
    """Singleton simulator for control input data from CSV files."""

    _instance = None

    def __new__(cls, inputs_csv: Optional[str] = None, interval: float = 0.01):
        if cls._instance is None:
            cls._instance = super(ControlInputs, cls).__new__(cls)
            cls._instance._init(inputs_csv, interval)
        return cls._instance

    def _init(self, inputs_csv: Optional[str], interval: float):
        # CSV path (relative to repository by default)
        import os
        root = os.path.dirname(__file__)
        sim_files = os.path.join(root, 'sim_files')

        # prefer a simulation control inputs file in sim_files
        self.inputs_csv = inputs_csv or os.path.join(sim_files, 'control_inputs.csv')
        if not os.path.exists(self.inputs_csv):
            logger.warning(f'Control inputs CSV not found: {self.inputs_csv}')

        self.interval = interval

        # internal state
        self.is_reading = False
        self.read_thread: Optional[threading.Thread] = None
        self.data_lock = threading.RLock()

        # current control inputs
        self.control_data = {
            'timestamp': 0.0,
            'velocity_mps': 0.0,
            'steering_angle_rad': 0.0,
        }

        self._timeline = []  # list of dicts with timestamp, velocity_mps, steering_angle_rad
        self._data_index = 0

        self.state_estimator = KalmanStateEstimator()

        try:
            self._load_control_data()
        except Exception as e:
            logger.debug(f'No control inputs CSV loaded or parse error: {e}')

        threading.Thread(target=self.start_continuous_reading, daemon=True).start()

    def _load_control_data(self):
        """Load control inputs from CSV file."""
        rows = []
        with open(self.inputs_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get('timestamp') or 0.0)
                    velocity = float(row.get('velocity_mps', 0.0))
                    steering_rad = float(row.get('steering_angle_rad', 0.0))
                except Exception:
                    continue

                rows.append({
                    'timestamp': ts,
                    'velocity_mps': velocity,
                    'steering_angle_rad': steering_rad,
                })

        # sort by timestamp
        rows.sort(key=lambda r: r['timestamp'])
        self._timeline = rows
        logger.info(f'ControlInputs loaded {len(rows)} data points')

    def start_continuous_reading(self, interval: float = 0.01):
        """Start continuous reading and playback of control inputs."""
        if self.is_reading:
            logger.warning('ControlInputs already reading')
            return

        self.is_reading = True
        if interval is not None:
            self.interval = interval

        def read_loop():
            n = len(self._timeline)
            if n == 0:
                logger.warning('No control inputs loaded for ControlInputs')
                self.is_reading = False
                return

            idx = 0
            while self.is_reading and idx < n:
                entry = self._timeline[idx]
                ts = entry['timestamp']
                velocity = entry['velocity_mps']
                steering_rad = entry['steering_angle_rad']

                # update control_data
                with self.data_lock:
                    self.control_data = {
                        'timestamp': ts,
                        'velocity_mps': velocity,
                        'steering_angle_rad': steering_rad,
                    }

                # Feed control inputs to EKF
                if np.isfinite(velocity):
                    self.state_estimator.set_rear_wheel_velocity(velocity)
                if np.isfinite(steering_rad):
                    self.state_estimator.set_steering_angle(steering_rad)

                # advance index and sleep according to desired interval
                idx += 1
                if idx < n:
                    next_ts = self._timeline[idx]['timestamp']
                    dt = max(0.0, min(1.0, next_ts - ts))
                    time.sleep(max(self.interval, dt) / Debug.time_scale)
                else:
                    break

            self.is_reading = False
            logger.info('ControlInputs playback finished (EOF reached)')

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
        logger.info('ControlInputs started continuous reading')

    def stop_reading(self):
        """Stop the reading thread."""
        self.is_reading = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
        logger.info('ControlInputs stopped reading')

    def get_velocity(self) -> float:
        """Get the latest velocity in m/s."""
        with self.data_lock:
            return self.control_data['velocity_mps']

    def get_steering_angle(self) -> float:
        """Get the latest steering angle in radians."""
        with self.data_lock:
            return self.control_data['steering_angle_rad']

    def get_timestamp(self) -> float:
        """Get the timestamp of the latest control data."""
        with self.data_lock:
            return self.control_data['timestamp']


__all__ = ['ControlInputs']