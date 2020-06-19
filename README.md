# Rethinking_OR_with_RL

*Disclaimer: This repository is designed for the "OR vs RL" project at the EPFL, Lausanne, under the supervision of Prof. A. Alahi, VITA laboratory, Lausanne, Switzerland. This project is based on the previous work from N. Bolón Brun, available in the following repository*: [https://github.com/natbolon/Rethinking_OR].

## Aim of the project

This project aims at exploring the advantages of deep Reinforcement Learning
in the case of traditional Operations Research problems
such as the Dial-a-Ride Problem.

More information can be found in the report located in the `Papers/` folder.

## Repository organization

The folder 'Bibliography/' lists the bibliography used for this work.
Its only file is a detailed list of it,
the .pdf files not being uploaded to github for space and/or property reasons.

The code can be found in the `CodeMC/Python/` folder.

## Environment

Language Python was used for this project.
The list of the project overall requirements can be found under requirements.txt,
with installation methods.

Minimal requirement for this repository are:
PyTorch,
Matplotlib,
NumPy

## Setup

This code relies on external files to limit memory usage.
Notable saved data are input and comparison tables, and models.

To this end, the user must have a folder dedicated to these saves.
This folder is to contain the following subfolders:
- `situations_tables/`
- `assignements_tables/`
- `Models/`

The following command create such a folder (may vary with OS):
- *mkdir situations_tables/ assignements_tables/ Models/*
- *mkdir Data/ Data/situations_tables/ Data/assignements_tables/ Data/Models/*

*Note: Default considered folder is local folder.*

## Data Generation
Before actually training a network, we have to generate its data.
We use a simple simulation for this.

We generate a set of situations,
in which a request is issued on a square grid with agent located around.
This data is stored and loaded from the `situations_tables/` folder.

To generate a set of situations, the user may run the following command:
- *python3 data_generator.py N*

Where N is the number of tables to be generated.
By default, this will generate N*500 examples of size 30*30 with 10 agents.

This can be further parametrized.
Most notable parameters are:
- -i S to set the image to size S*S;
- -f S to set the fleet to size S;
- -r K to create K examples per table;
- -p PATH to set the saving path to PATH.

Run *python3 data_generator.py -h* for further information.

After creating the situations,
we have to generate the assignements expected from them.
This data is stored and loaded from the `assignements_tables/` folder.

To do so,  the user may run the following command:
- *python3 policies.py N*

Where N is the number of tables to be generated.
This command may be parametrized as the previous one,
and should be if some default parameters were changed during generation.

Another notable parameter is:
- -pol P to set the policy;
only Nearest Neighbour (NN) and Top Corner (TC) are implemented as of today,
but the command supports further addition.

## Network training.
The network may now be trained, once the data has been generated.

To run a network training and subsequent training,
the user might run the following command:
- *python3 training_accuracy.py N S*

Where N is the number of tables to be used
and S the size of the image
(which is 30 by default).

This will run a training and testing set of epochs (200 by default),
with evaluation of the network.
Its versions will be stored in the `Models/` folder.

Further parameters can be used.
Notable parameters include:
  - -net NET to define the network category.
    MonoLoco (ML) and (FC) are fully implemented.
    Convolutional network (CO) are also implemented but with less parameters.
  - -la A to define the amount of fully-connected layers.
  - -ls S to define the size of fully-connected layers.
  - -e E to define a run of E epochs.
  - -adlr E1 LR1 E2 LR2 to define adaptative,
  epoch-fixed learning rate evolutions at epoch Ei for learning rate LRi.
  - -p PATH to set the saving and loading path to PATH.

For more precisions, the user may run *python3 training_accuracy.py -h*



WIP

June 2020,

Maxime Cauté
