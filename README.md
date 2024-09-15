# Multi-Skill  Learning for Robotic Manipulation Tasks

## Overview
This is the implementation of the Multi-Phase Training for the Sliding Puzzle from our  pape "Multi-Skill Learning for Robotic Manipulation Tasks".
Please note that the code is not cleaned up. It is only intended for reproducing our results.

It contains the implementation for our Multi-phase training on the Sliding Puzzle, as well as a reimplementation of SEADS, adapted to the Sliding Puzzle task. For the original implementation of SEADS please refer to ["Learning Temporally Extended Skills in Continuous
Domains as Symbolic Actions for Planning"](https://arxiv.org/abs/2207.05018) (Achterhold, Krimmel, and Stueckler 2022).

## Setup
As physics simulation environment we use [Robotic Python - Robotic Control Interface & Manipulation Planning Library](https://github.com/MarcToussaint/robotic).
We use [PyTorch with Cuda](https://pytorch.org/), [gymnasium](https://gymnasium.farama.org/index.html), and the SAC implementation from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

## Usage
To run the multi-phase training for different number of skills execute the file `run-multi-phase-training.sh`
inside the folder "SAC/".
To run our implementation of SEADS execute the file `run-seads.sh` inside the folder "SAC/".
Also have a look at those two files to see how to execute individual training runs with different sets of parameters.
