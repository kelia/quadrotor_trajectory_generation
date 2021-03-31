import casadi as cs


class GeometricTrajectory:
    def __init__(self, duration):
        # define position trajectory symbolically
        self.t = cs.MX.sym("t")

        #################################################################################
        # time mapping, ensures that trajectories start and end in hover
        #################################################################################
        tau = self.t / duration
        self.t_adj = 1.524 * duration * (-(
                8 * cs.cos(tau * cs.pi) * cs.constpow(cs.sin(tau * cs.pi), 5) + 10 * cs.cos(tau * cs.pi) * cs.constpow(
            cs.sin(tau * cs.pi), 3) + 39 * cs.sin(tau * cs.pi) * cs.cos(tau * cs.pi) + 12 * cs.sin(
            2 * tau * cs.pi) * cs.cos(2 * tau * cs.pi) - 63 * tau * cs.pi) / (96 * cs.pi))

        #################################################################################
        # describe your trajectory here as a function of t_adj
        #################################################################################
        # sphere trajectory rotating around x-axis
        # radius_x = 5.0
        # radius_y = 3.5
        # radius_z = 2.5
        # fast config
        # freq_slow = 0.009
        # freq_fast = 0.33
        # slow config
        # freq_slow = 0.02
        # freq_fast = 0.12
        # pos_x = 3.0 + radius_x * (cs.sin(2.0 * cs.pi * freq_fast * t_adj) * cs.cos(2.0 * cs.pi * freq_slow * t_adj))
        # pos_y = 1.0 + radius_y * (cs.cos(2.0 * cs.pi * freq_fast * t_adj))
        # pos_z = 3.5 + radius_z * (cs.sin(2.0 * cs.pi * freq_fast * t_adj) * cs.sin(2.0 * cs.pi * freq_slow * t_adj))

        # circle
        radius = 1.5
        freq = 0.45
        self.pos_x = 1.5 + radius * cs.sin(2.0 * cs.pi * freq * self.t_adj)
        self.pos_y = 0.0 + radius * cs.cos(2.0 * cs.pi * freq * self.t_adj)
        self.pos_z = 1.0
