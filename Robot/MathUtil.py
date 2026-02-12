import numpy as np

class MathUtil:
    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q)

    @staticmethod
    def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
        # q = [qx, qy, qz, qw]
        qx, qy, qz, qw = q
        R = np.empty((3, 3))
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[0, 1] = 2 * (qx * qy - qz * qw)
        R[0, 2] = 2 * (qx * qz + qy * qw)
        R[1, 0] = 2 * (qx * qy + qz * qw)
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[1, 2] = 2 * (qy * qz - qx * qw)
        R[2, 0] = 2 * (qx * qz - qy * qw)
        R[2, 1] = 2 * (qy * qz + qx * qw)
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
        return R
    
    @staticmethod
    def quat_sensor_to_estimator(q_sensor: np.ndarray) -> np.ndarray:
        """Convert sensor quaternion (w, x, y, z) to estimator order [qx,qy,qz,qw].

        Many IMU libraries (e.g. Adafruit BNO055) return quaternions as (w,x,y,z).
        This helper makes the conversion explicit and safe.
        """
        if q_sensor is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        try:
            w, x, y, z = (0.0 if v is None else float(v) for v in q_sensor)
        except Exception:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return np.array([x, y, z, w], dtype=float)

    @staticmethod
    def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        # Hamilton product, q = q1 * q2
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return np.array([x, y, z, w])
    
    @staticmethod
    def small_angle_quat(dtheta: np.ndarray) -> np.ndarray:
        # dtheta: 3-vector small rotation
        theta = np.linalg.norm(dtheta)
        if theta < 1e-8:
            q = np.concatenate((0.5 * dtheta, np.array([1.0])))
        else:
            axis = dtheta / theta
            s = np.sin(theta / 2.0)
            q = np.concatenate((axis * s, np.array([np.cos(theta / 2.0)])))
        return MathUtil.quat_normalize(q)

    @staticmethod
    def quat_to_euler(q: np.ndarray) -> np.ndarray:
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        qx, qy, qz, qw = q
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def euler_to_quat(euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to quaternion [qx, qy, qz, qw].

        Assumes intrinsic rotations about x (roll), y (pitch), z (yaw) with the
        same convention used by quat_to_euler.
        """
        roll, pitch, yaw = float(euler[0]), float(euler[1]), float(euler[2])
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return MathUtil.quat_normalize(np.array([qx, qy, qz, qw], dtype=float))
