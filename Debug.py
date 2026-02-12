class Debug:
    # If true, will display a list of active commands in the command runner
    displayActiveCommands = False
    
    # Time scale multiplier for simulation speed (1.0 = real-time, 2.0 = 2x speed, etc.)
    # Affects KalmanStateEstimator, SimIMU, and SimUWB but not main loop
    time_scale = 10.0