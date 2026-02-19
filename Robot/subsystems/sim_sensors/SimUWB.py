#!/usr/bin/env python3
"""CSV-driven UWB simulator that mirrors the UWBTag API.

This simulator reads two CSVs (anchors and positions) and plays them back
in time order. When the position file reaches EOF the simulator sets
anchors and position to None to indicate no more data.

Usage: SimUWB(anchors_csv=None, positions_csv=None)
"""
import csv
import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, cast
import numpy
import numpy as np
from Debug import Debug


from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator
from Robot.Constants import Constants

logger = logging.getLogger(__name__)


@dataclass
class Position:
    x: float
    y: float
    z: float
    quality: int
    timestamp: float


class SimUWB:
    """Singleton-like simulator for UWB tag data from CSV files."""

    _instance = None

    def __new__(cls, anchors_csv: Optional[str] = None, positions_csv: Optional[str] = None, interval: float = 0.01):
        if cls._instance is None:
            cls._instance = super(SimUWB, cls).__new__(cls)
            cls._instance._init(anchors_csv, positions_csv, interval)
        return cls._instance

    def _init(self, anchors_csv: Optional[str], positions_csv: Optional[str], interval: float):
        # CSV paths (relative to repository by default)
        import os
        root = os.path.dirname(__file__)
        sim_files = os.path.join(root, 'sim_files')

        # prefer a simulation positions file in sim_files; if not present fall back to example_output
        self.positions_csv = positions_csv or os.path.join(sim_files, 'uwb_positions.csv')
        if not os.path.exists(self.positions_csv):
            # try example_output location (useful for repository examples)
            example_path = os.path.join(root, '..', '..', 'example_output')
            # do a simple search for a file named uwb_positions.csv in example_output
            for dirpath, _, files in os.walk(example_path):
                if 'uwb_positions.csv' in files:
                    self.positions_csv = os.path.join(dirpath, 'uwb_positions.csv')
                    break

        self.interval = interval
        self.batch_data = False

        # internal state
        self.is_connected = False
        self.is_reading = False
        self.read_thread: Optional[threading.Thread] = None
        self.position_lock = threading.RLock()

        # tag_info mirrors real UWBTag usage
        self.tag_info = {'anchors': None, 'position': None, 'individual_positions': None}

        self._anchors_timeline = []  # list of (timestamp, anchors_list)
        self._positions_timeline = []  # list of dicts rows
        self._pos_index = 0

        self.state_estimator = KalmanStateEstimator()

        try:
            self._load_positions()
        except Exception:
            logger.debug('No positions CSV loaded or parse error')
            
        threading.Thread(target=self.start_continuous_reading, daemon=True).start()

    def _load_positions(self):
        # positions CSV columns: timestamp,tag_id,x,y,z,quality
        rows = []
        with open(self.positions_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get('timestamp') or 0.0)
                    tag_id = int(row.get('tag_id', 0))
                    x = row.get('x', '')
                    y = row.get('y', '')
                    z = row.get('z', '')
                    quality = row.get('quality', '')
                    
                    # Skip rows with missing position data
                    if x == '' or y == '' or z == '':
                        continue
                    
                    rows.append({
                        'timestamp': ts,
                        'tag_id': tag_id,
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'quality': int(float(quality)) if quality != '' else 0
                    })
                except Exception:
                    continue

        # sort by timestamp
        rows.sort(key=lambda r: r['timestamp'])
        self._positions_timeline = rows

    def connect(self) -> bool:
        """Pretend to open a connection for the simulator."""
        self.is_connected = True
        logger.info('SimUWB connected (CSV playback)')
        return True

    def disconnect(self):
        """Disconnect and stop any playback thread."""
        try:
            self.stop_reading()
        except Exception:
            pass

        self.is_connected = False
        with self.position_lock:
            self.tag_info['anchors'] = None
            self.tag_info['position'] = None
            self.tag_info['individual_positions'] = None

        logger.info('SimUWB disconnected')

    def start_continuous_reading(self, interval: float = 0.01, debug: bool = False):
        if self.is_reading:
            logger.warning('SimUWB already reading')
            return

        self.is_reading = True
        if interval is not None:
            self.interval = interval

        def read_loop():
            # play through positions timeline; when finished, set values to None
            n = len(self._positions_timeline)
            if n == 0:
                logger.warning('No positions loaded for SimUWB')
                # nothing to play; mark outputs None and stop
                with self.position_lock:
                    self.tag_info['anchors'] = None
                    self.tag_info['position'] = None
                self.is_reading = False
                return

            idx = 0
            while self.is_reading and idx < n:
                entry = self._positions_timeline[idx]
                ts = entry['timestamp']
                tag_id = entry['tag_id']
                x = entry['x']
                y = entry['y']
                z = entry['z']
                quality = entry['quality']

                # Create position object
                pos = Position(x=x, y=y, z=z, quality=quality, timestamp=ts)

                # update tag_info
                with self.position_lock:
                    self.tag_info['position'] = pos # type: ignore
                    # Store individual positions as a list of Position objects
                    self.tag_info['individual_positions'] = [pos] # type: ignore

                # Look up tag offset from Constants based on tag_id
                tag_offset = None
                for tag_data in Constants.uwb_tag_data:
                    if tag_data.id == tag_id:
                        tag_offset = np.array(tag_data.offset, dtype=float)
                        break
                
                # Feed the EKF with tag position and offset
                tag_pos_meas = np.array([pos.x, pos.y, pos.z], dtype=float)
                # print(f"Feeding EKF with tag_id={tag_id}, position={tag_pos_meas}, offset={tag_offset}")
                if self.batch_data:
                    self.state_estimator.batch_uwb(tag_id, tag_pos_meas, tag_offset)
                else:
                    self.state_estimator.update_uwb_range(tag_pos_meas, tag_offset)

                # advance index and sleep according to desired interval
                idx += 1
                # compute wait from next timestamp if available, otherwise use self.interval
                if idx < n:
                    next_ts = self._positions_timeline[idx]['timestamp']
                    dt = max(0.0, min(1.0, next_ts - ts))
                    time.sleep(max(self.interval, dt) / Debug.time_scale)
                else:
                    # EOF reached
                    break

            # At EOF set None values and stop reading
            with self.position_lock:
                self.tag_info['anchors'] = None
                self.tag_info['position'] = None
                self.tag_info['individual_positions'] = None

            self.is_reading = False
            logger.info('SimUWB playback finished (EOF reached)')

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
        logger.info('SimUWB started continuous reading')

    def stop_reading(self):
        self.is_reading = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
        logger.info('SimUWB stopped reading')

    def get_latest_position(self) -> Optional[Position]:
        with self.position_lock:
            return self.tag_info['position']

    def get_latest_anchor_info(self) -> Optional[List[Dict[str, Any]]]:
        with self.position_lock:
            anchors = cast(Optional[List[Dict[str, Any]]], self.tag_info.get('anchors'))
            if anchors is None:
                return None
            return [a.copy() for a in anchors]

    def get_individual_positions(self) -> Optional[List[Position]]:
        """Get the list of individual tag positions (before averaging)."""
        with self.position_lock:
            return self.tag_info.get('individual_positions')
