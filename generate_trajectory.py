import argparse
import time

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared

from config.settings import Settings
from utils.check_trajectory import check_trajectory
from utils.geometric_traj import GeometricTrajectory
from utils.q_funcs import q_dot_q, quaternion_inverse
from utils.quadrotor import Quad
from utils.smooth import smooth
from utils.visualization import debug_plot, draw_poly


class TrajectoryGenerator:
    def __init__(self, config):
        self.config = config
        self.quad = Quad(self.config)

    def generate(self):
        start_time = time.time()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_fn = "generated_trajectories/" + self.config.traj_type + "_" + timestr + ".csv"

        if self.config.traj_type == 'geometric':
            trajectory, motor_inputs, t_vec = self.compute_geometric_trajectory()
        elif self.config.traj_type == 'random':
            trajectory, motor_inputs, t_vec = self.compute_random_trajectory()
        else:
            print("Unknown trajectory type.")
            exit()

        if check_trajectory(trajectory, motor_inputs, t_vec, self.config.debug):
            if self.config.debug:
                debug_plot(trajectory, motor_inputs, t_vec)
            if self.config.generate_plots:
                draw_poly(trajectory[:, :13], motor_inputs, t_vec)
            self.export(t_vec, trajectory, motor_inputs, output_fn)
        else:
            print("Trajectory check failed.")
            exit()
        print("Trajectory generation took [%.3f] seconds." % (time.time() - start_time))

    def export(self, t_vec, trajectory, motor_inputs, output_fn):
        take_every_nth = int(self.config.traj_dt / self.config.dt_gen)

        # save trajectory to csv
        df_traj = pd.DataFrame()
        df_traj['t'] = t_vec[::take_every_nth]
        df_traj['p_x'] = trajectory[::take_every_nth, 0]
        df_traj['p_y'] = trajectory[::take_every_nth, 1]
        df_traj['p_z'] = trajectory[::take_every_nth, 2]

        df_traj['q_w'] = trajectory[::take_every_nth, 3]
        df_traj['q_x'] = trajectory[::take_every_nth, 4]
        df_traj['q_y'] = trajectory[::take_every_nth, 5]
        df_traj['q_z'] = trajectory[::take_every_nth, 6]

        df_traj['v_x'] = trajectory[::take_every_nth, 7]
        df_traj['v_y'] = trajectory[::take_every_nth, 8]
        df_traj['v_z'] = trajectory[::take_every_nth, 9]

        df_traj['w_x'] = trajectory[::take_every_nth, 10]
        df_traj['w_y'] = trajectory[::take_every_nth, 11]
        df_traj['w_z'] = trajectory[::take_every_nth, 12]

        df_traj['a_lin_x'] = trajectory[::take_every_nth, 13]
        df_traj['a_lin_y'] = trajectory[::take_every_nth, 14]
        df_traj['a_lin_z'] = trajectory[::take_every_nth, 15]

        df_traj['a_rot_x'] = trajectory[::take_every_nth, 16]
        df_traj['a_rot_y'] = trajectory[::take_every_nth, 17]
        df_traj['a_rot_z'] = trajectory[::take_every_nth, 18]

        df_traj['u_1'] = motor_inputs[::take_every_nth, 0]
        df_traj['u_2'] = motor_inputs[::take_every_nth, 1]
        df_traj['u_3'] = motor_inputs[::take_every_nth, 2]
        df_traj['u_4'] = motor_inputs[::take_every_nth, 3]

        df_traj['jerk_x'] = trajectory[::take_every_nth, 19]
        df_traj['jerk_y'] = trajectory[::take_every_nth, 20]
        df_traj['jerk_z'] = trajectory[::take_every_nth, 21]

        df_traj['snap_x'] = trajectory[::take_every_nth, 22]
        df_traj['snap_y'] = trajectory[::take_every_nth, 23]
        df_traj['snap_z'] = trajectory[::take_every_nth, 24]

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

        # kernel to map functions that repeat exactly
        print("seed is: %d" % self.config.seed)
        kernel_y = ExpSineSquared(length_scale=self.config.rand_traj_freq_x, periodicity=17) \
                   + ExpSineSquared(length_scale=3.0, periodicity=23) \
                   + ExpSineSquared(length_scale=4.0, periodicity=51)
        kernel_x = ExpSineSquared(length_scale=self.config.rand_traj_freq_y, periodicity=37)  # \
        # + ExpSineSquared(length_scale=3.0, periodicity=61) \
        # + ExpSineSquared(length_scale=4.0, periodicity=13)
        kernel_z = ExpSineSquared(length_scale=self.config.rand_traj_freq_z, periodicity=19) \
                   + ExpSineSquared(length_scale=3.0, periodicity=29) \
                   + ExpSineSquared(length_scale=4.0, periodicity=53)

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
        x_sample_hr = gp_x.sample_y(t_coarse[:, np.newaxis], 1, random_state=self.config.seed)
        print("sampling y...")
        y_sample_hr = gp_y.sample_y(t_coarse[:, np.newaxis], 1, random_state=self.config.seed + 1)
        print("sampling z...")
        z_sample_hr = gp_z.sample_y(t_coarse[:, np.newaxis], 1, random_state=self.config.seed + 2)

        pos_np = np.concatenate([x_sample_hr, y_sample_hr, z_sample_hr], axis=1)
        # scale to arena bounds
        max_traj = np.max(pos_np, axis=0)
        min_traj = np.min(pos_np, axis=0)
        pos_centered = pos_np - (max_traj + min_traj) / 2.0
        pos_scaled = pos_centered * (self.config.bound_max - self.config.bound_min) / (max_traj - min_traj)
        pos_arena = pos_scaled + (self.config.bound_max + self.config.bound_min) / 2.0

        if self.config.debug:
            plt.plot(pos_arena[:, 0], label="x")
            plt.plot(pos_arena[:, 1], label="y")
            plt.plot(pos_arena[:, 2], label="z")
            plt.legend()
            plt.show()

        # rescale time to get smooth start and end states
        pos_blub_x = interpolate.interp1d(t_coarse, pos_arena[:, 0], kind="cubic", fill_value="extrapolate")
        pos_blub_y = interpolate.interp1d(t_coarse, pos_arena[:, 1], kind="cubic", fill_value="extrapolate")
        pos_blub_z = interpolate.interp1d(t_coarse, pos_arena[:, 2], kind="cubic", fill_value="extrapolate")
        pos_arena = np.concatenate([pos_blub_x(scaled_time),
                                    pos_blub_y(scaled_time),
                                    pos_blub_z(scaled_time)], axis=1)

        pos_arena = np.concatenate([smooth(np.squeeze(pos_arena[:, 0]), window_len=11)[:, np.newaxis],
                                    smooth(np.squeeze(pos_arena[:, 1]), window_len=11)[:, np.newaxis],
                                    smooth(np.squeeze(pos_arena[:, 2]), window_len=11)[:, np.newaxis]], axis=1)

        # compute numeric derivative & smooth things
        vel_arena = np.gradient(pos_arena, axis=0) / dt
        vel_arena = np.concatenate([smooth(np.squeeze(vel_arena[:, 0]), window_len=11)[:, np.newaxis],
                                    smooth(np.squeeze(vel_arena[:, 1]), window_len=11)[:, np.newaxis],
                                    smooth(np.squeeze(vel_arena[:, 2]), window_len=11)[:, np.newaxis]], axis=1)
        acc_arena = np.gradient(vel_arena, axis=0) / dt
        acc_arena = np.concatenate([smooth(np.squeeze(acc_arena[:, 0]), window_len=11)[:, np.newaxis],
                                    smooth(np.squeeze(acc_arena[:, 1]), window_len=11)[:, np.newaxis],
                                    smooth(np.squeeze(acc_arena[:, 2]), window_len=11)[:, np.newaxis]], axis=1)
        t_np = t_vec

        jerk_np = np.zeros_like(acc_arena)
        snap_np = np.zeros_like(acc_arena)

        trajectory, motor_inputs, t_vec = self.compute_full_traj(t_np, pos_arena, vel_arena, acc_arena, jerk_np,
                                                                 snap_np)

        return trajectory, motor_inputs, t_vec

    def compute_full_traj(self, t_np, pos_np, vel_np, alin_np, jerk_np, snap_np):
        len_traj = t_np.shape[0]
        dt = np.mean(np.diff(t_np))

        # Compute thrust direction
        gravity = 9.81
        thrust_np = alin_np + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
        # Compute body axes
        z_b = thrust_np / np.sqrt(np.sum(thrust_np ** 2, 1))[:, np.newaxis]
        # computing the quaternion based on two vectors:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        e_z = np.array([[0.0, 0.0, 1.0]])
        q_w = 1.0 + np.sum(e_z * z_b, axis=1)
        q_xyz = np.cross(e_z, z_b)
        att_np = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
        att_np = att_np / np.sqrt(np.sum(att_np ** 2, 1))[:, np.newaxis]

        rate_np = np.zeros_like(pos_np)
        f_t = np.zeros((len_traj, 1))

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(att_np, axis=0) / dt
        w_int = np.zeros((len_traj, 3))
        for i in range(len_traj):
            w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(att_np[i, :]), q_dot[i])[1:]
            f_t[i, 0] = self.quad.mass * z_b[i].dot(thrust_np[i, :].T)
        rate_np[:, 0] = w_int[:, 0]
        rate_np[:, 1] = w_int[:, 1]
        rate_np[:, 2] = w_int[:, 2]

        minimize_yaw_rate = True
        n_iter_yaw_refinement = 20
        if minimize_yaw_rate:
            for iter_yaw_refinement in range(n_iter_yaw_refinement):
                print("Maximum yawrate before refinement %d / %d: %.6f" % (
                    iter_yaw_refinement, n_iter_yaw_refinement, np.max(np.abs(rate_np[:, 2]))))
                q_new = att_np
                yaw_corr_acc = 0.0
                for i in range(1, len_traj):
                    yaw_corr = -rate_np[i, 2] * dt
                    yaw_corr_acc += yaw_corr
                    q_corr = np.array([np.cos(yaw_corr_acc / 2.0), 0.0, 0.0, np.sin(yaw_corr_acc / 2.0)])
                    q_new[i, :] = q_dot_q(att_np[i, :], q_corr)
                    w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(att_np[i, :]), q_dot[i])[1:]

                q_new_dot = np.gradient(q_new, axis=0) / dt
                for i in range(1, len_traj):
                    w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q_new[i, :]), q_new_dot[i])[1:]

                att_np = q_new
                rate_np[:, 0] = w_int[:, 0]
                rate_np[:, 1] = w_int[:, 1]
                rate_np[:, 2] = w_int[:, 2]
                print("Maximum yawrate after refinement: %.3f" % np.max(np.abs(rate_np[:, 2])))
                if np.max(np.abs(rate_np[:, 2])) < 0.005:
                    break

        arot_np = np.gradient(rate_np, axis=0)
        trajectory = np.concatenate([pos_np, att_np, vel_np, rate_np, alin_np, arot_np, jerk_np, snap_np], axis=1)
        motor_inputs = np.zeros((pos_np.shape[0], 4))

        # Compute inputs
        rate_dot = np.gradient(rate_np, axis=0) / dt
        rate_x_Jrate = np.array([(self.quad.J[2] - self.quad.J[1]) * rate_np[:, 2] * rate_np[:, 1],
                                 (self.quad.J[0] - self.quad.J[2]) * rate_np[:, 0] * rate_np[:, 2],
                                 (self.quad.J[1] - self.quad.J[0]) * rate_np[:, 1] * rate_np[:, 0]]).T

        tau = rate_dot * self.quad.J[np.newaxis, :] + rate_x_Jrate
        b = np.concatenate((tau, f_t), axis=-1)
        a_mat = np.concatenate((self.quad.y_f[np.newaxis, :],
                                -self.quad.x_f[np.newaxis, :],
                                self.quad.z_l_tau[np.newaxis, :],
                                np.ones_like(self.quad.z_l_tau)[np.newaxis, :]), 0)

        for i in range(len_traj):
            motor_inputs[i, :] = np.linalg.solve(a_mat, b[i, :])

        return trajectory, motor_inputs, t_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a quadrotor trajectory.')
    parser.add_argument('--settings_file', help='Path to configuration file.', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file

    settings = Settings(settings_filepath)

    traj_generator = TrajectoryGenerator(settings)

    for i in range(settings.num_traj):
        traj_generator.generate()
