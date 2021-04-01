import argparse
import time

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared

from config.settings import Settings
from utils.geometric_traj import GeometricTrajectory
from utils.quadrotor import Quad
from utils.trajectory import Trajectory
from utils.visualization import draw_poly


class TrajectoryGenerator:
    def __init__(self, config):
        self.config = config
        self.quad = Quad(self.config)

    def generate(self):
        start_time = time.time()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_fn = "generated_trajectories/" + self.config.traj_type + "_" + timestr + ".csv"

        if self.config.traj_type == 'geometric':
            trajectory = self.compute_geometric_trajectory()
        elif self.config.traj_type == 'random':
            trajectory = self.compute_random_trajectory()
        else:
            print("Unknown trajectory type.")
            exit()

        if trajectory.check_integrity():
            if self.config.generate_plots:
                draw_poly(trajectory)
            self.export(trajectory, output_fn)
        else:
            print("Trajectory check failed.")
            exit()
        print("Trajectory generation took [%.3f] seconds." % (time.time() - start_time))

    def export(self, trajectory, output_fn):
        '''
        Save trajectory to csv
        '''
        take_every_nth = int(self.config.traj_dt / self.config.dt_gen)

        # save trajectory to csv
        df_traj = pd.DataFrame()
        df_traj['t'] = trajectory.t[::take_every_nth]
        df_traj['p_x'] = trajectory.pos[::take_every_nth, 0]
        df_traj['p_y'] = trajectory.pos[::take_every_nth, 1]
        df_traj['p_z'] = trajectory.pos[::take_every_nth, 2]

        df_traj['q_w'] = trajectory.att[::take_every_nth, 0]
        df_traj['q_x'] = trajectory.att[::take_every_nth, 1]
        df_traj['q_y'] = trajectory.att[::take_every_nth, 2]
        df_traj['q_z'] = trajectory.att[::take_every_nth, 3]

        df_traj['v_x'] = trajectory.vel[::take_every_nth, 0]
        df_traj['v_y'] = trajectory.vel[::take_every_nth, 1]
        df_traj['v_z'] = trajectory.vel[::take_every_nth, 2]

        df_traj['w_x'] = trajectory.omega[::take_every_nth, 0]
        df_traj['w_y'] = trajectory.omega[::take_every_nth, 1]
        df_traj['w_z'] = trajectory.omega[::take_every_nth, 2]

        df_traj['a_lin_x'] = trajectory.acc[::take_every_nth, 0]
        df_traj['a_lin_y'] = trajectory.acc[::take_every_nth, 1]
        df_traj['a_lin_z'] = trajectory.acc[::take_every_nth, 2]

        df_traj['a_rot_x'] = trajectory.acc_rot[::take_every_nth, 0]
        df_traj['a_rot_y'] = trajectory.acc_rot[::take_every_nth, 1]
        df_traj['a_rot_z'] = trajectory.acc_rot[::take_every_nth, 2]

        df_traj['u_1'] = trajectory.inputs[::take_every_nth, 0]
        df_traj['u_2'] = trajectory.inputs[::take_every_nth, 1]
        df_traj['u_3'] = trajectory.inputs[::take_every_nth, 2]
        df_traj['u_4'] = trajectory.inputs[::take_every_nth, 3]

        df_traj['jerk_x'] = trajectory.jerk[::take_every_nth, 0]
        df_traj['jerk_y'] = trajectory.jerk[::take_every_nth, 1]
        df_traj['jerk_z'] = trajectory.jerk[::take_every_nth, 2]

        df_traj['snap_x'] = trajectory.snap[::take_every_nth, 0]
        df_traj['snap_y'] = trajectory.snap[::take_every_nth, 1]
        df_traj['snap_z'] = trajectory.snap[::take_every_nth, 2]

        print("Saving trajectory to [%s]." % output_fn)
        df_traj.to_csv(output_fn, index=False)

    def compute_geometric_trajectory(self):
        print("Computing geometric trajectory!")

        dt = self.config.dt_gen

        # define position trajectory symbolically
        geom_traj = GeometricTrajectory(self.config.traj_duration)

        # TODO: define yaw trajectory
        pos = cs.vertcat(geom_traj.pos_x, geom_traj.pos_y, geom_traj.pos_z)
        vel = cs.jacobian(pos, geom_traj.t)
        acc = cs.jacobian(vel, geom_traj.t)
        jerk = cs.jacobian(acc, geom_traj.t)
        snap = cs.jacobian(jerk, geom_traj.t)

        t_vec, dt = np.linspace(0.0, self.config.traj_duration, int(self.config.traj_duration / dt), endpoint=False,
                                retstep=True)

        f_t_adj = cs.Function('t_adj', [geom_traj.t], [geom_traj.t_adj])
        f_pos = cs.Function('f_pos', [geom_traj.t], [pos])
        f_vel = cs.Function('f_vel', [geom_traj.t], [vel])
        f_acc = cs.Function('f_acc', [geom_traj.t], [acc])
        f_jerk = cs.Function('f_jerk', [geom_traj.t], [jerk])
        f_snap = cs.Function('f_snap', [geom_traj.t], [snap])

        pos_list = []
        vel_list = []
        alin_list = []
        jerk_list = []
        snap_list = []
        t_adj_list = []
        for t_curr in t_vec:
            t_adj_list.append(f_t_adj(t_curr).full().squeeze())
            pos_list.append(f_pos(t_curr).full().squeeze())
            vel_list.append(f_vel(t_curr).full().squeeze())
            alin_list.append(f_acc(t_curr).full().squeeze())
            jerk_list.append(f_jerk(t_curr).full().squeeze())
            snap_list.append(f_snap(t_curr).full().squeeze())

        t_adj_np = np.array(t_adj_list)
        pos_np = np.array(pos_list)
        vel_np = np.array(vel_list)
        alin_np = np.array(alin_list)
        jerk_np = np.array(jerk_list)
        snap_np = np.array(snap_list)

        if self.config.debug:
            plt.plot(t_adj_np)
            plt.show()

        trajectory, motor_inputs, t_vec = self.compute_full_traj(t_vec, pos_np, vel_np, alin_np, jerk_np, snap_np)

        return trajectory, motor_inputs, t_vec

    def compute_random_trajectory(self):
        print("Computing random trajectory!")

        if self.config.seed == -1:
            self.config.seed = np.random.randint(0, 9999999)

        # kernel to generate functions that repeat exactly
        print("seed is: %d" % self.config.seed)
        np.random.seed(self.config.seed)

        kernel_x = ExpSineSquared(length_scale=np.random.uniform(self.config.rand_min_length_scale,
                                                                 self.config.rand_max_length_scale),
                                  periodicity=np.random.uniform(self.config.rand_min_period,
                                                                self.config.rand_max_period))
        kernel_y = ExpSineSquared(length_scale=np.random.uniform(self.config.rand_min_length_scale,
                                                                 self.config.rand_max_length_scale),
                                  periodicity=np.random.uniform(self.config.rand_min_period,
                                                                self.config.rand_max_period))
        kernel_z = ExpSineSquared(length_scale=np.random.uniform(self.config.rand_min_length_scale,
                                                                 self.config.rand_max_length_scale),
                                  periodicity=np.random.uniform(self.config.rand_min_period,
                                                                self.config.rand_max_period))

        for i_kernel in range(self.config.rand_num_kernels):
            kernel_x += ExpSineSquared(length_scale=np.random.uniform(self.config.rand_min_length_scale,
                                                                      self.config.rand_max_length_scale),
                                       periodicity=np.random.uniform(self.config.rand_min_period,
                                                                     self.config.rand_max_period))
            kernel_y += ExpSineSquared(length_scale=np.random.uniform(self.config.rand_min_length_scale,
                                                                      self.config.rand_max_length_scale),
                                       periodicity=np.random.uniform(self.config.rand_min_period,
                                                                     self.config.rand_max_period))
            kernel_z += ExpSineSquared(length_scale=np.random.uniform(self.config.rand_min_length_scale,
                                                                      self.config.rand_max_length_scale),
                                       periodicity=np.random.uniform(self.config.rand_min_period,
                                                                     self.config.rand_max_period))

        gp_x = GaussianProcessRegressor(kernel=kernel_x)
        gp_y = GaussianProcessRegressor(kernel=kernel_y)
        gp_z = GaussianProcessRegressor(kernel=kernel_z)

        t_coarse = np.linspace(0.0, self.config.traj_duration, int(self.config.traj_duration / 0.1), endpoint=False)
        t_vec, dt = np.linspace(0.0, self.config.traj_duration, int(self.config.traj_duration / self.config.dt_gen),
                                endpoint=False, retstep=True)

        t = cs.MX.sym("t")
        # t_speed is a function starting at zero and ending at zero that modulates time
        # casadi cannot do symbolic integration --> write down the integrand by hand of 2.0*sin^2(t)
        # t_adj = 2.0 * (t / 2.0 - cs.sin(2.0 / duration * cs.pi * t) / (4.0 * cs.pi / duration))
        tau = t / self.config.traj_duration
        t_adj = 1.524 * self.config.traj_duration * (-(
                8 * cs.cos(tau * cs.pi) * cs.constpow(cs.sin(tau * cs.pi), 5) + 10 * cs.cos(tau * cs.pi) * cs.constpow(
            cs.sin(tau * cs.pi), 3) + 39 * cs.sin(tau * cs.pi) * cs.cos(tau * cs.pi) + 12 * cs.sin(
            2 * tau * cs.pi) * cs.cos(2 * tau * cs.pi) - 63 * tau * cs.pi) / (96 * cs.pi))

        f_t_adj = cs.Function('t_adj', [t], [t_adj])
        scaled_time = f_t_adj(t_vec)

        print("sampling x...")
        x_sampled = gp_x.sample_y(t_coarse[:, np.newaxis], 1, random_state=self.config.seed)
        print("sampling y...")
        y_sampled = gp_y.sample_y(t_coarse[:, np.newaxis], 1, random_state=self.config.seed + 1)
        print("sampling z...")
        z_sampled = gp_z.sample_y(t_coarse[:, np.newaxis], 1, random_state=self.config.seed + 2)

        pos_sampled = np.concatenate([x_sampled, y_sampled, z_sampled], axis=1)
        # scale to arena bounds
        max_traj = np.max(pos_sampled, axis=0)
        min_traj = np.min(pos_sampled, axis=0)
        pos_centered = pos_sampled - (max_traj + min_traj) / 2.0
        pos_scaled = pos_centered * (self.config.bound_max - self.config.bound_min) / (max_traj - min_traj)
        pos_coarse = pos_scaled + (self.config.bound_max + self.config.bound_min) / 2.0

        if self.config.debug:
            plt.plot(pos_coarse[:, 0], label="x")
            plt.plot(pos_coarse[:, 1], label="y")
            plt.plot(pos_coarse[:, 2], label="z")
            plt.legend()
            plt.show()

        spl_x = UnivariateSpline(t_coarse, pos_coarse[:, 0], k=5)
        spl_y = UnivariateSpline(t_coarse, pos_coarse[:, 1], k=5)
        spl_z = UnivariateSpline(t_coarse, pos_coarse[:, 2], k=5)

        trajectory = Trajectory()
        trajectory.t = t_vec
        trajectory.pos = np.concatenate([spl_x(scaled_time),
                                         spl_y(scaled_time),
                                         spl_z(scaled_time)], axis=1)

        # compute derivatives via numeric differentiation
        trajectory.vel = np.gradient(trajectory.pos, axis=0) / dt
        trajectory.acc = np.gradient(trajectory.vel, axis=0) / dt
        trajectory.jerk = np.gradient(trajectory.acc, axis=0) / dt
        trajectory.snap = np.gradient(trajectory.jerk, axis=0) / dt

        trajectory.compute_full_traj(self.quad)

        # remove boundary effects due to numeric differentiation
        trajectory.fix_start_end()

        return trajectory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a quadrotor trajectory.')
    parser.add_argument('--settings_file', help='Path to configuration file.', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file

    settings = Settings(settings_filepath)

    traj_generator = TrajectoryGenerator(settings)

    for i in range(settings.num_traj):
        traj_generator.generate()
