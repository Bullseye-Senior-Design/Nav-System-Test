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
from Debug import Debug


from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator

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
        # positions CSV columns like: timestamp,x1,y1,z1,quality1,x2,y2,z2,quality2,...
        rows = []
        with open(self.positions_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get('timestamp') or 0.0)
                except Exception:
                    continue

                rows.append({'timestamp': ts, 'row': row})

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
                row = entry['row']

                # Collect all available position blocks in the row and average them
                # fields come in groups like (x1,y1,z1,quality1),(x2,y2,z2,quality2),...
                pos = None
                pos_samples = []
                # allow a generous number of groups in case CSV has many
                for g in range(1, 50):
                    xs = row.get(f'x{g}', '')
                    ys = row.get(f'y{g}', '')
                    zs = row.get(f'z{g}', '')
                    qs = row.get(f'quality{g}', '')
                    if xs not in (None, '') and ys not in (None, '') and zs not in (None, ''):
                        try:
                            x = float(xs)
                            y = float(ys)
                            z = float(zs)
                            q = int(float(qs)) if qs not in (None, '') and qs != '' else 0
                            pos_samples.append((x, y, z, q))
                        except Exception:
                            # skip malformed group
                            continue

                if pos_samples:
                    nx = sum(p[0] for p in pos_samples) / len(pos_samples)
                    ny = sum(p[1] for p in pos_samples) / len(pos_samples)
                    nz = sum(p[2] for p in pos_samples) / len(pos_samples)
                    nq = int(round(sum(p[3] for p in pos_samples) / len(pos_samples)))
                    pos = Position(x=nx, y=ny, z=nz, quality=nq, timestamp=ts)
                    # print(f"SimUWB position at ts={ts}: x={nx}, y={ny}, z={nz}, quality={nq}")

                # update tag_info
                with self.position_lock:
                    self.tag_info['position'] = pos # type: ignore
                    # Store individual positions as a list of Position objects
                    self.tag_info['individual_positions'] = [ # type: ignore
                        Position(x=p[0], y=p[1], z=p[2], quality=p[3], timestamp=ts)
                        for p in pos_samples
                    ] if pos_samples else None 

                # if we have a valid (possibly averaged) position, feed the EKF like a real device
                if pos is not None:
                    tag_pos_meas = numpy.array([pos.x, pos.y, pos.z], dtype=float)
                    # Send position to EKF with no offset
                    # print("Feeding EKF with position:", tag_pos_meas)
                    self.state_estimator.update_uwb_range(tag_pos_meas, None, False)
                else:
                    print("No valid position data available for EKF update.")

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
