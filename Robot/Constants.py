class UWBTagInfo:
    def __init__(self, port, id, offset):
        self.port = port
        self.id = id
        self.offset = offset

class Constants:
    # Encoder Constants
    back_right_encoder_pin = 4
    wheel_circumference = 0.25  # meters
    counts_per_revolution = 6  # encoder pulses per wheel rotation
    
    # Motor speed constants
    rear_motor_top_speed = 0.13
    
    # Clutches Constants
    left_clutch_pin = 17
    right_clutch_pin = 27
    
    # For the tag offsets:
    # +x : tag is forward of the robot center
    # +y : tag is to the robot's left side
    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyACM0", id=0, offset=(-24.77 / 2 / 100, 22.225 / 2 / 100, 0.0)), # back left tag
                    UWBTagInfo(port="/dev/ttyACM1", id=1, offset=(24.77 / 2 / 100, -22.225 / 2 / 100, 0.0)) ] # front right tag

