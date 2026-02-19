from Robot.Constants import Constants
from structure.commands.InstantCommand import InstantCommand
from structure.commands.SequentialCommandGroup import SequentialCommandGroup
from Robot.subsystems.sim_sensors.SimIMU import SimIMU
from Robot.subsystems.sim_sensors.SimUWB import SimUWB
from Robot.subsystems.sim_sensors.SimEncoder import SimEncoder
from Robot.Commands.PlotStateCmd import PlotStateCmd
from Robot.Commands.LogKalmanCmd import LogKalmanCmd
from Robot.subsystems.sim_sensors.ControlInputs import ControlInputs

class RobotContainer:
    def __init__(self):
        self.sim_imu = SimIMU()
        self.sim_uwb = SimUWB()
        self.sim_encoder = SimEncoder()
        self.control_inputs = ControlInputs()
                    
    def begin_data_log(self):
        LogKalmanCmd().schedule()
        PlotStateCmd().schedule()