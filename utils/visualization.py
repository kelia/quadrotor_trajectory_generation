import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def draw_poly(trajectory, target_points=None, target_t=None):
    """
    Plots the generated trajectory of length n with the used keypoints.
    :param traj: Full generated reference trajectory. Numpy array of shape nx13
    :param u_traj: Generated reference inputs. Numpy array of shape nx4
    :param t: Timestamps of the references. Numpy array of length n
    :param target_points: m position keypoints used for trajectory generation. Numpy array of shape 3 x m.
    :param target_t: Timestamps of the reference position keypoints. If not passed, then they are extracted from the
    t vector, assuming constant time separation.
    """

    ders = 2
    dims = 3

    y_labels = [r'pos $[m]$', r'vel $[m/s]$', r'acc $[m/s^2]$', r'jer $[m/s^3]$']
    dim_legends = ['x', 'y', 'z']

    if target_t is None and target_points is not None:
        target_t = np.linspace(0, t[-1], target_points.shape[1])

    p_traj = trajectory.pos
    a_traj = trajectory.att
    v_traj = trajectory.vel
    r_traj = trajectory.omega

    plt_traj = [p_traj, v_traj]

    fig = plt.figure(figsize=(10, 7))
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1)

        for dim in range(dims):

            plt.plot(trajectory.t, plt_traj[d_ord][:, dim], label=dim_legends[dim])

            if d_ord == 0 and target_points is not None:
                plt.plot(target_t, target_points[dim, :], 'bo')

        plt.gca().set_xticklabels([])
        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])

    dim_legends = [['w', 'x', 'y', 'z'], ['x', 'y', 'z']]
    y_labels = [r'att $[quat]$', r'rate $[rad/s]$']
    plt_traj = [a_traj, r_traj]
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1 + ders * 2)
        for dim in range(plt_traj[d_ord].shape[1]):
            plt.plot(trajectory.t, plt_traj[d_ord][:, dim], label=dim_legends[d_ord][dim])

        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])
        if d_ord == ders - 1:
            plt.xlabel(r'time $[s]$')
        else:
            plt.gca().set_xticklabels([])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    cmap = cm.get_cmap('jet')
    rgb = cmap(trajectory.t / trajectory.t[-1])
    ax.scatter(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2], facecolors=rgb)

    if target_points is not None:
        plt.plot(target_points[0, :], target_points[1, :], target_points[2, :], 'bo')
    plt.title('Target position trajectory')
    ax.set_xlabel(r'$p_x [m]$')
    ax.set_ylabel(r'$p_y [m]$')
    ax.set_zlabel(r'$p_z [m]$')

    plt.subplot(ders + 1, 2, (ders + 1) * 2)
    for i in range(trajectory.inputs.shape[1]):
        plt.plot(trajectory.t, trajectory.inputs[:, i], label=r'$u_{}$'.format(i))
    plt.grid()
    plt.legend()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel(r'time $[s]$')
    plt.ylabel(r'single thrusts $[N]$')
    plt.title('Control inputs')

    plt.suptitle('Generated trajectory')

    plt.show()

