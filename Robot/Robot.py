from structure.CommandRunner import CommandRunner
from Robot.RobotContainer import RobotContainer

class Robot:    
    def __init__(self):
        self.command_runner = CommandRunner()
        self.robot_container = RobotContainer()
    
    def robot_init(self):
        self.command_runner.turn_on()
        self.robot_container.begin_data_log()
    
    def robot_periodic(self):
        self.command_runner.run_commands()
    
    def teleop_init(self):
        pass
            
    def teleop_periodic(self):
        pass

    def test_init(self):
        pass
    
    def test_periodic(self):
        pass
    
    def disabled_init(self):
        pass