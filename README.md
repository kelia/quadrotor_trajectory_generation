# Quadrotor Trajectory Generation

This repo contains some (hacky) code to generate quadrotor trajectories.

## Installation

Install `casadi`.

## Usage

```
python generate_trajectory.py --settings_file=config/settings.yaml
```

The script either generates `geometric` trajectories or `random` trajectories.

To edit the generated trajectories (duration, sampling frequency, ...): edit `config/settings.yaml`. 
The geometric trajectory type allows to define an arbitrary trajectory that can be expressed as a symbolic function. To define such function, edit `utils/geometric_traj.py` and define the position as a function of time in casadi syntax.
To edit the random trajectories, edit `freq_x`, `freq_y`, `freq_z` in the settings file to control the aggressiveness
of the generated trajectory.
