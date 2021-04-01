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

            assert self.quad_mass > 0.0
            assert all(type(value) is float for value in self.quad_inertia)
            assert all(value > 0.0 for value in self.quad_inertia)

            # --- trajectory ---
            trajectory = settings['trajectory']
            self.traj_type = trajectory['type']
            self.traj_duration = trajectory['duration']
            self.traj_dt = trajectory['dt']
            self.seed = trajectory['seed']
            self.bound_min = np.array(trajectory['bound_min'])
            self.bound_max = np.array(trajectory['bound_max'])
            self.dt_gen = trajectory['dt_gen']
            assert np.all(self.bound_min <= self.bound_max)

            random_traj = trajectory['random']
            self.rand_num_kernels = random_traj['num_kernels']
            self.rand_max_length_scale = random_traj['max_length_scale']
            self.rand_min_length_scale = random_traj['min_length_scale']
            self.rand_max_period = random_traj['max_period']
            self.rand_min_period = random_traj['min_period']

            assert self.rand_min_length_scale <= self.rand_max_length_scale
            assert self.rand_min_period <= self.rand_max_period
