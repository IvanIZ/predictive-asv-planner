# Autonomous Navigation in Ice-Covered Waters with Learned Predictions on Ship-Ice Interactions
Code for our work [paper](https://arxiv.org/abs/2302.11601) "Autonomous Navigation in Ice-Covered Waters with Learned Predictions on Ship-Ice Interactions".


![my image](./assets/demo_fig.png)



## Installation

This project requires [ROS Humble](https://docs.ros.org/en/humble/Installation.html) and has been testbed on Ubuntu 22.04. with Python 3.10.

1. Clone the project
```bash
git clone https://github.com/IvanIZ/predictive-asv-planner.git
```

2. install dependencies. Note, may need to install the
[dubins](https://github.com/AndrewWalker/pydubins) package manually
(see [here](https://github.com/AndrewWalker/pydubins/issues/16#issuecomment-1138899416) for instructions).
```bash
pip install -r requirements.txt
```

## Usage
The simulation runs with two ROS nodes - a navigation node and planner node.

Run the following command to start the navigation node, which loads the environments and runs the physics simulations
```bash
python asv_navigation.py
```

In a new terminal, run one of the following commands to start a specific planner. To run our planner:
```bash
python planners/predictive.py
```

To run the lattice planner
```bash
python planners/lattice.py
```

To run the skeleton planner
```bash
python planners/skeleton.py
```

To run the straight-line planner
```bash
python planners/straight.py
```

#### Configuration File
All parameters are set in the config.yaml file in the `config` directory. 

#### Experiment Logs
After each trial, the navigation visualization and collision statistics can be found in the ```logs``` directory. 