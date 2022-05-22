# DRL_Environment_Exploration

This is the repository to for Autonomous waypoints planning and trajectory generation for multi-rotor UAVs

## Table of contents
- [Introduction](#Introduction)

- [Environment](#Environment)

- [Example](#Example)

- [Publications](#Publications)

- [Acknowledgement](#Acknowledgement)

## Introduction

In this work, we have developed an agent which could plan an obstcal-free trajectory from a strat point to the end, based on the observation of the environment.


- Agent

The core of the agent is a Deep Reinforcement Model shown in figure below.

![](https://github.com/Dr-Qiu-s-research-team/DRL_Environment_Exploration/blob/main/image/network.png)

The input is the observation of the agent. And the output is the next action of the UAV.

If we consider a 3x3 voxel world. The uav(red) could go to the adjecent 26 grids except the center which is the location of the uav shown in the figure below.

![](https://github.com/Dr-Qiu-s-research-team/DRL_Environment_Exploration/blob/main/image/action_space.png)

- Encoding

We encode the environment, observation and the reward as below.

|envionment|value|
|-|-|
|kOutOfBound_Encode|-1|
|kObstacle_Encode|-1|
|kEgoPosition_Encode|0.5|
|kGoalPosition_Encode|1|

|reward|value|
|-|-|
|OBSTACLE_REWARD|-2|
|GOAL_REWARD|10|
|DIST_REWARD|0.1|
|DRONE_POSITION|1|
  
## Environment

Please refer to the [UAV_data_repository](https://github.com/Dr-Qiu-s-research-team/UAV_data_repository) for more information. 

## Example

To train the model.

```
pyhton3 main.py
```

There are lots of parameters could be setup in the main.py which could be easily modefied for different kinds of purposes.

|patameter|defination|type|
|-|-|-|
|lr|learning rate|float|
|mode|the kind of model|'linear','conv'|
|batch size|batch size for trainnning|int|
|optimizer|the kind of optimizer|'adam','sgd'|
|env-size|the size of the environment|int|
|sensing-range|the sensing range for the agent|int|
|grid-resolution|the resolution for each grid|float|
|num-obst|number of uccupied grids|int|
|num-objs|number of shaped obstacles|int|
|state-dim|maximum obstical number|int|
|action-dim|the size of the action space|6,26|
|eval|load pretrained model and evalue|bool|
|buffer-size|the size of the replay buffer|int|
|gamma|the decay of learning rate|float|
|enable-epsilon|use epsilon greedy policy|bool|
|epsilon|the maximum probability of the random process|float|
|epsilon-min|the minimum probability of the random process|float|
|max-steps|the maximum steps of one epoch|int|
|save-epochs|the number of epochs to save the model|int|
|save-weights-dir|the location to save or load the model|str|
|load-pretrained|load the pretrained model|bool|
|thrust-reward|use thrust reward|bool|
|obst-generation-mode|the type of different environments|'voxel_random', 'plane_random', 'voxel_constrain', 'test', 'random', 'gazebo_random', 'demo'|


## Publications

The original paper proposes a Deep Reinforcement Learning Model to generate a set of obstacle-free waypoints in a known environment.
- [Autonomous waypoints planning and trajectory generation for multi-rotor UAVs](https://dl.acm.org/doi/abs/10.1145/3313151.3313163)

The latest paper implements ADMM method to prune the DRL model and get a much smaller but better performence model.
- [Neural Network Pruning and Fast Training for DRL-based UAV Trajectory Planning](https://ieeexplore.ieee.org/abstract/document/9712561)

## Acknowledgement

This project is partially supported by SRC(Semiconductor Research Corporation)

[![](https://www.src.org/web/img/SRC_logo_blue.png)](https://www.src.org/)

