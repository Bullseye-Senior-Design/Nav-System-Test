class UWBTagInfo:
    def __init__(self, port, offset):
        self.port = port
        self.offset = offset

class Constants:
    # UWB Tag Data
    # +x : tag is forward of the robot center
    # +y : tag is to the robot's left side
    uwb_tag_data = [ UWBTagInfo(port="/dev/ttyACM0", offset=(-24.77 / 2 / 100, 22.225 / 2 / 100, 0.0)), # back left tag
                    UWBTagInfo(port="/dev/ttyACM1", offset=(24.77 / 2 / 100, -22.225 / 2 / 100, 0.0)) ] # front right tag
