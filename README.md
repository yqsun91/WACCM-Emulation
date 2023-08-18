# WACCM-Emulation
This is a DNN written with PyTorch to Emulate the gravity wave drag (GWD, both zonal and meridional ) in the WACCM Simulation.


# DemoData
Sample output data from WACCM emulation.
It is 3D global output from the WACCM model, on the original model grid.


# data loader
load 3D WACCM data and reshaping them to the NN input.

# Using a FNN to train and predict the GWD
train.py train the files and generate the weights for NN.

NN-pred.py load the weights and do prediction.

# Coupling ? future work
replace original GWD scheme in WACCM with this emulator.
a. the emulator can be trained offline
b. training the emulator online


