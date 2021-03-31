import os

import numpy as np
import yaml


class Settings:
    def __init__(self, settings_yaml):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            # --- general ---
            general = settings['general']
            self.generate_plots = general['generate_plots']
            self.num_traj = general['n_trajectories']
            self.debug = general['debug']

            # --- quadrotor ---
            quadrotor = settings['quadrotor']
            self.quad_mass = quadrotor['mass']
            self.quad_inertia = quadrotor['inertia']
            self.quad_max_mot_thrust = quadrotor['max_thrust_per_motor']
            self.rotor_drag_coeff = quadrotor['rotor_drag_coeff']
            self.quad_arm_length = quadrotor['arm_length']

            # --- trajectory ---
            trajectory = settings['trajectory']
            self.traj_type = trajectory['type']
            self.traj_duration = trajectory['duration']
            self.traj_dt = trajectory['dt']
            self.seed = trajectory['seed']
            self.bound_min = np.array(trajectory['bound_min'])
            self.bound_max = np.array(trajectory['bound_max'])
            self.dt_gen = trajectory['dt_gen']

            random_traj = trajectory['random']
            self.rand_traj_freq_x = random_traj['freq_x']
            self.rand_traj_freq_y = random_traj['freq_y']
            self.rand_traj_freq_z = random_traj['freq_z']
