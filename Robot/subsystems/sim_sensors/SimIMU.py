#!/usr/bin/env python3
"""Simulated IMU that reads from a CSV file.

This class implements the same public API as `Robot.subsystems.sensors.IMU`
but sources data from a CSV (timestamp,yaw,pitch,roll,ax,ay,az,gx,gy,gz,mx,my,mz).
When the CSV is exhausted the sensor values are set to None.
"""
import threading
import time
import csv
import numpy as np
import logging
from typing import Optional, Tuple
from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.MathUtil import MathUtil
from Debug import Debug


logger = logging.getLogger(__name__)


class SimIMU():
    """Singleton simulated IMU that reads sequential rows from a CSV file.

    Public API mirrors the real `IMU` where practical: get_accel(), get_gyro(),
    get_mag(), get_quat(), get_euler(), set_yaw_offset(), etc. After the CSV
    reaches EOF the sensor attributes are set to None and the update thread
    stops.
    """
    _instance = None

    def __new__(cls):
        # Keep a single global instance like the real IMU class
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._start()
        return cls._instance

    def _start(self):
        # data storage
        self._csv_path = "Robot/subsystems/sim_sensors/sim_files/imu_orientation.csv"
        self._rows = []
        self._idx = 0

        # sensor state (None when not available)
        self.acceleration: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
        self.gyro: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
        self.magnetic: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
        # sensor quaternion in sensor ordering (w,x,y,z) to match real sensor
        # libraries; None when no data
        self.quat: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0)

        # timing and config
        self.interval = 0.01
        self.mag_interval = self.interval * 5
        self._last_mag_time = 0.0

        # helpers
        self._lock = threading.RLock()
        self.state_estimator = KalmanStateEstimator()

        # load CSV into memory (best for small sim files)
        try:
            with open(self._csv_path, newline='') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    # keep raw dict rows (we'll parse floats during playback)
                    self._rows.append(r)
        except Exception as e:
            logger.error(f"SimIMU: failed to open CSV '{self._csv_path}': {e}")
            self._rows = []

        # control
        self._is_running = False
        self._thread = None

        # start playback automatically
        self.begin()

    def begin(self):
        if self._is_running:
            return

        self._is_running = True

        def _loop():
            while self._is_running:
                try:
                    self._step()
                except Exception as e:
                    logger.debug(f"SimIMU update error: {e}")
                    # on error, stop playback and set values to None
                    self._set_none_and_stop()
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def _set_none_and_stop(self):
        with self._lock:
            self.acceleration = None
            self.gyro = None
            self.magnetic = None
            self.quat = None
        self._is_running = False

    def _step(self):
        # If no rows, set none and stop
        if not self._rows or self._idx >= len(self._rows):
            self._set_none_and_stop()
            return

        row = self._rows[self._idx]

        # parse expected columns safely
        try:
            ts = float(row.get('timestamp', 0.0)) if row.get('timestamp') else None
            # CSV columns: timestamp,yaw,pitch,roll,ax,ay,az,gx,gy,gz,mx,my,mz
            yaw = float(row.get('yaw')) if row.get('yaw') else 0.0
            pitch = float(row.get('pitch')) if row.get('pitch') else 0.0
            roll = float(row.get('roll')) if row.get('roll') else 0.0

            ax = float(row.get('ax')) if row.get('ax') else None
            ay = float(row.get('ay')) if row.get('ay') else None
            az = float(row.get('az')) if row.get('az') else None

            gx = float(row.get('gx')) if row.get('gx') else None
            gy = float(row.get('gy')) if row.get('gy') else None
            gz = float(row.get('gz')) if row.get('gz') else None

            mx = float(row.get('mx')) if row.get('mx') else None
            my = float(row.get('my')) if row.get('my') else None
            mz = float(row.get('mz')) if row.get('mz') else None
        except Exception as e:
            logger.debug(f"SimIMU: malformed CSV row at idx={self._idx}: {e}")
            # skip malformed row
            self._idx += 1
            return

        # build sensor quaternion: MathUtil.euler_to_quat expects radians and roll,pitch,yaw order
        try:
            euler_rad = np.radians(np.array([roll, pitch, yaw], dtype=float))
            q_est = MathUtil.euler_to_quat(euler_rad)  # returns [qx,qy,qz,qw]
            # convert to sensor ordering (w,x,y,z)
            qw = float(q_est[3])
            qx = float(q_est[0])
            qy = float(q_est[1])
            qz = float(q_est[2])
            sensor_quat = (qw, qx, qy, qz)
        except Exception:
            sensor_quat = None

        # update internal state under lock
        with self._lock:
            self.acceleration = None if any(v is None for v in (ax, ay, az)) else (ax, ay, az) # type: ignore
            self.gyro = None if any(v is None for v in (gx, gy, gz)) else (gx, gy, gz) # type: ignore
            self.magnetic = None if any(v is None for v in (mx, my, mz)) else (mx, my, mz) # type: ignore
            self.quat = sensor_quat
            # ensure we pass an ndarray or None to the estimator to satisfy its type signature
            q_meas_sen = np.asarray(sensor_quat, dtype=float)
            q_meas = MathUtil.quat_sensor_to_estimator(q_meas_sen)  # validate conversion
            self.state_estimator.update_imu_attitude(q_meas)
            if self.magnetic is not None:
                self._last_mag_time = time.time()

        # advance index and sleep using timestamp delta when available
        next_idx = self._idx + 1
        sleep_for = self.interval
        if next_idx < len(self._rows):
            try:
                t0 = float(self._rows[self._idx].get('timestamp', 0.0)) if self._rows[self._idx].get('timestamp') else None
                t1 = float(self._rows[next_idx].get('timestamp', 0.0)) if self._rows[next_idx].get('timestamp') else None
                if t0 is not None and t1 is not None:
                    dt = float(t1 - t0)
                    # clamp dt to reasonable range
                    if dt <= 0 or not np.isfinite(dt):
                        dt = self.interval
                    sleep_for = min(max(dt, 0.0), 1.0)
            except Exception:
                sleep_for = self.interval

        self._idx = next_idx
        # print(f"sleep_for={sleep_for:.3f}s")
        time.sleep(sleep_for / Debug.time_scale)

    # Public API methods to match real IMU
    def get_gyro(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self.gyro

    def get_accel(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self.acceleration

    def get_mag(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self.magnetic

    def get_quat(self) -> Optional[Tuple[float, float, float, float]]:
        with self._lock:
            return self.quat

    def get_aligned_quat(self) -> Optional[Tuple[float, float, float, float]]:
        with self._lock:
            raw = self.quat
            if raw is None:
                return None
            q_sensor = np.asarray(raw, dtype=float)
            q_est = MathUtil.quat_sensor_to_estimator(q_sensor)
            return tuple(q_est)

    def get_euler(self) -> Optional[Tuple[float, float, float]]:
        """Return (yaw, roll, pitch) in degrees or None if no quat available."""
        with self._lock:
            raw = self.quat
            yaw_off = float(getattr(self, '_yaw_offset_rad', 0.0))

        if raw is None:
            return None

        try:
            q_sensor = np.asarray(raw, dtype=float)
            q_est = MathUtil.quat_sensor_to_estimator(q_sensor)
            if getattr(self, '_is_offset_set', False) and abs(yaw_off) > 1e-12:
                q_yaw = MathUtil.euler_to_quat(np.array([0.0, 0.0, yaw_off]))
                q_est = MathUtil.quat_mul(q_yaw, q_est)
                q_est = MathUtil.quat_normalize(q_est)
            rpy = MathUtil.quat_to_euler(q_est)  # roll,pitch,yaw radians
            roll = float(rpy[0]); pitch = float(rpy[1]); yaw = float(rpy[2])
            deg = np.degrees
            return (deg(yaw), deg(roll), deg(pitch))
        except Exception:
            return None

    def mag_interval_elapsed(self) -> bool:
        return (time.time() - self._last_mag_time) >= self.mag_interval

    def end(self):
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
