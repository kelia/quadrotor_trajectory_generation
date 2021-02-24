# Quadrotor Trajectory Generation

This repo contains some (hacky) code to generate quadrotor trajectories.

## Installation

Install `casadi`.

## Usage

```
python generate_trajectory.py --traj_type [random | geometric]
```

The script either generates `geometric` trajectories or `random` trajectories.

To edit the random trajectories: edit `freq_x`, `freq_y`, `freq_z` in the `main` function to control the aggressiveness
of the generated trajectory.

To edit the geometric trajectory: edit the symbolic functions describing the evolution of the position over time, i.e.
edit `pos_x`, `pos_y`, `pos_z` in the function `compute_geometric_trajectory`.