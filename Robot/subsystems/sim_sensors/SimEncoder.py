#!/usr/bin/env python3
"""CSV-driven encoder simulator that feeds velocity data to the Kalman filter.

This simulator reads encoder data from a CSV file (timestamp, count, velocity)
and plays it back in time order, feeding velocity measurements to the EKF.
When the CSV reaches EOF, the simulator stops reading.

Usage: SimEncoder()
"""
import csv
import time
import threading
import logging
import numpy as np
from typing import Optional
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Debug import Debug


logger = logging.getLogger(__name__)


class SimEncoder:
    """Singleton simulator for encoder velocity data from CSV files."""

    _instance = None

    def __new__(cls, encoder_csv: Optional[str] = None, interval: float = 0.01):
        if cls._instance is None:
            cls._instance = super(SimEncoder, cls).__new__(cls)
            cls._instance._init(encoder_csv, interval)
        return cls._instance

    def _init(self, encoder_csv: Optional[str], interval: float):
        # CSV path (relative to repository by default)
        import os
        root = os.path.dirname(__file__)
        sim_files = os.path.join(root, 'sim_files')

        # prefer a simulation encoder file in sim_files
        self.encoder_csv = encoder_csv or os.path.join(sim_files, 'encoder_data.csv')
        if not os.path.exists(self.encoder_csv):
            logger.warning(f'Encoder CSV not found: {self.encoder_csv}')

        self.interval = interval

        # internal state
        self.is_connected = False
        self.is_reading = False
        self.read_thread: Optional[threading.Thread] = None
        self.data_lock = threading.RLock()

        # current encoder data
        self.encoder_data = {'count': 0, 'velocity': 0.0, 'timestamp': 0.0}

        self._encoder_timeline = []  # list of dicts with timestamp, count, velocity
        self._data_index = 0

        self.state_estimator = KalmanStateEstimator()

        try:
            self._load_encoder_data()
        except Exception as e:
            logger.debug(f'No encoder CSV loaded or parse error: {e}')
            
        threading.Thread(target=self.start_continuous_reading, daemon=True).start()

    def _load_encoder_data(self):
        """Load encoder data from CSV file."""
        rows = []
        with open(self.encoder_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get('timestamp') or 0.0)
                    count = int(float(row.get('count', 0)))
                    velocity = float(row.get('velocity', 0.0))
                except Exception:
                    continue

                rows.append({
                    'timestamp': ts,
                    'count': count,
                    'velocity': velocity
                })

        # sort by timestamp
        rows.sort(key=lambda r: r['timestamp'])
        self._encoder_timeline = rows
        logger.info(f'SimEncoder loaded {len(rows)} data points')

    def connect(self) -> bool:
        """Pretend to open a connection for the simulator."""
        self.is_connected = True
        logger.info('SimEncoder connected (CSV playback)')
        return True

    def disconnect(self):
        """Disconnect and stop any playback thread."""
        try:
            self.stop_reading()
        except Exception:
            pass

        self.is_connected = False
        with self.data_lock:
            self.encoder_data = {'count': 0, 'velocity': 0.0, 'timestamp': 0.0}

        logger.info('SimEncoder disconnected')

    def start_continuous_reading(self, interval: float = 0.01):
        """Start continuous reading and playback of encoder data."""
        if self.is_reading:
            logger.warning('SimEncoder already reading')
            return

        self.is_reading = True
        if interval is not None:
            self.interval = interval

        def read_loop():
            # play through encoder timeline; when finished, stop
            n = len(self._encoder_timeline)
            if n == 0:
                logger.warning('No encoder data loaded for SimEncoder')
                self.is_reading = False
                return

            idx = 0
            while self.is_reading and idx < n:
                entry = self._encoder_timeline[idx]
                ts = entry['timestamp']
                count = entry['count']
                velocity = entry['velocity']

                # update encoder_data
                with self.data_lock:
                    self.encoder_data = {
                        'timestamp': ts,
                        'count': count,
                        'velocity': velocity
                    }

                # Feed velocity to EKF
                if self.state_estimator.is_initialized and velocity is not None and np.isfinite(velocity):
                    # print(f"SimEncoder feeding velocity: {velocity:.3f} m/s at ts={ts}")
                    self.state_estimator.update_encoder_velocity(velocity)

                # advance index and sleep according to desired interval
                idx += 1
                # compute wait from next timestamp if available, otherwise use self.interval
                if idx < n:
                    next_ts = self._encoder_timeline[idx]['timestamp']
                    dt = max(0.0, min(1.0, next_ts - ts))
                    time.sleep(max(self.interval, dt) / Debug.time_scale)
                else:
                    # EOF reached
                    break

            self.is_reading = False
            logger.info('SimEncoder playback finished (EOF reached)')

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
        logger.info('SimEncoder started continuous reading')

    def stop_reading(self):
        """Stop the reading thread."""
        self.is_reading = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
        logger.info('SimEncoder stopped reading')

    def get_velocity(self) -> float:
        """Get the latest encoder velocity in m/s."""
        with self.data_lock:
            return self.encoder_data['velocity']

    def get_count(self) -> int:
        """Get the latest encoder count."""
        with self.data_lock:
            return self.encoder_data['count']

    def get_timestamp(self) -> float:
        """Get the timestamp of the latest encoder data."""
        with self.data_lock:
            return self.encoder_data['timestamp']


__all__ = ['SimEncoder']
