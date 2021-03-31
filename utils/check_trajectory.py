import matplotlib.pyplot as plt
import numpy as np

from .q_funcs import q_dot_q, quaternion_inverse


def check_trajectory(trajectory, inputs, tvec, plot=False):
    """
    @param trajectory:
    @param inputs:
    @param tvec:
    @param plot:
    @return:
    """

    print("Checking trajectory integrity...")

    dt = np.expand_dims(np.gradient(tvec, axis=0), axis=1)
    numeric_derivative = np.gradient(trajectory, axis=0) / dt

    errors = np.zeros((dt.shape[0], 3))

    num_bodyrates = []

    for i in range(dt.shape[0]):
        # 1) check if velocity is consistent with position
        numeric_velocity = numeric_derivative[i, 0:3]
        analytic_velocity = trajectory[i, 7:10]
        errors[i, 0] = np.linalg.norm(numeric_velocity - analytic_velocity)
        if not np.allclose(analytic_velocity, numeric_velocity, atol=0.05, rtol=0.05):
            print("inconsistent linear velocity at i = %d" % i)
            print(numeric_velocity)
            print(analytic_velocity)
            return False

        # 2) check if attitude is consistent with acceleration
        gravity = 9.81
        numeric_thrust = numeric_derivative[i, 7:10] + np.array([0.0, 0.0, gravity])
        numeric_thrust = numeric_thrust / np.linalg.norm(numeric_thrust)
        analytic_attitude = trajectory[i, 3:7]
        if np.abs(np.linalg.norm(analytic_attitude) - 1.0) > 1e-6:
            print("quaternion does not have unit norm at i = %d" % i)
            print(analytic_attitude)
            print(np.linalg.norm(analytic_attitude))
            return False

        e_z = np.array([0.0, 0.0, 1.0])
        q_w = 1.0 + np.dot(e_z, numeric_thrust)
        q_xyz = np.cross(e_z, numeric_thrust)
        numeric_attitude = 0.5 * np.array([q_w] + q_xyz.tolist())
        numeric_attitude = numeric_attitude / np.linalg.norm(numeric_attitude)
        # the two attitudes can only differ in yaw --> check x,y component
        q_diff = q_dot_q(quaternion_inverse(analytic_attitude), numeric_attitude)
        errors[i, 1] = np.linalg.norm(q_diff[1:3])
        if not np.allclose(q_diff[1:3], np.zeros(2, ), atol=0.05, rtol=0.05):
            print("Attitude and acceleration do not match at i = %d" % i)
            print(analytic_attitude)
            print(numeric_attitude)
            print(q_diff)
            return False

        # 3) check if bodyrates agree with attitude difference
        numeric_bodyrates = 2.0 * q_dot_q(quaternion_inverse(trajectory[i, 3:7]), numeric_derivative[i, 3:7])[1:]
        num_bodyrates.append(numeric_bodyrates)
        analytic_bodyrates = trajectory[i, 10:13]
        errors[i, 2] = np.linalg.norm(numeric_bodyrates - analytic_bodyrates)
        if not np.allclose(numeric_bodyrates, analytic_bodyrates, atol=0.05, rtol=0.05):
            print("inconsistent angular velocity at i = %d" % i)
            print(numeric_bodyrates)
            print(analytic_bodyrates)
            return False

    print("Trajectory check successful")
    print("Maximum linear velocity error: %.5f" % np.max(errors[:, 0]))
    print("Maximum attitude error: %.5f" % np.max(errors[:, 1]))
    print("Maximum angular velocity error: %.5f" % np.max(errors[:, 2]))

    if plot:
        num_bodyrates = np.stack(num_bodyrates)
        plt.figure()
        for i in range(3):
            plt.subplot(3, 2, i * 2 + 1)
            plt.plot(numeric_derivative[:, i], label='numeric')
            plt.plot(trajectory[:, 7 + i], label='analytic')
            plt.ylabel('m/s')
            if i == 0:
                plt.title("Velocity check")
            plt.legend()

        for i in range(3):
            plt.subplot(3, 2, i * 2 + 2)
            plt.plot(num_bodyrates[:, i], label='numeric')
            plt.plot(trajectory[:, 10 + i], label='analytic')
            plt.ylabel('rad/s')
            if i == 0:
                plt.title("Body rate check")
            plt.legend()
        plt.suptitle('Integrity check of reference trajectory')
        plt.show()

    return True
