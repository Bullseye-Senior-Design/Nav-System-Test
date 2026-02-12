from Debug import Debug
from Robot.Robot import Robot
from structure.RobotState import RobotState

import time

def main():
    
    robot = Robot()
    robot_state = RobotState()


    robot.robot_init()
    robot_state.enable_teleop()
    
    try:
        while True:        
            # Run periodic functions
            robot.robot_periodic()
            
            # Teleop (human operated) mode
            if robot_state.should_init_teleop():
                print("Teleop mode enabled")
                robot.teleop_init()
            
            if robot_state.is_teleop_enabled():
                robot.teleop_periodic()
            
            # Test mode
            if robot_state.should_init_test():
                print("Test mode enabled")
                robot.test_init()
            
            if robot_state.is_test_enabled():
                robot.test_periodic()
            
            # Disabled check
            if robot_state.should_init_disable():
                robot.disabled_init()
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down robot...")
        
if __name__ == "__main__":
    main()