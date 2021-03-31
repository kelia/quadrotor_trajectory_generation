import casadi as cs


class RandomTrajectory:
    def __init__(self, duration):
        # define position trajectory symbolically
        self.freq_x = 0.9
        self.freq_y = 0.7
        self.freq_z = 0.7
