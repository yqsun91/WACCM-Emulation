import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import scipy.stats as st
import xarray as xr

import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


## load mean and std for normalization
fm = np.load('Demodata/conv_mean_avg.npz')
fs = np.load('Demodata/conv_std_avg.npz')

Um        = fm['U']
Vm        = fm['V']
Tm        = fm['T']
DSEm      = fm['DSE']
NMm       = fm['NM']
NETDTm    = fm['NETDT']
Z3m    = fm['Z3']
RHOIm    = fm['RHOI']
PSm    = fm['PS']
latm    = fm['lat']
lonm    = fm['lon']
UTGWSPECm    = fm['UTGWSPEC']
VTGWSPECm    = fm['VTGWSPEC']

Us        = fs['U']
Vs        = fs['V']
Ts        = fs['T']
DSEs      = fs['DSE']
NMs       = fs['NM']
NETDTs    = fs['NETDT']
Z3s    = fs['Z3']
RHOIs    = fs['RHOI']
PSs    = fs['PS']
lats    = fs['lat']
lons    = fs['lon']
UTGWSPECs    = fs['UTGWSPEC']
VTGWSPECs    = fs['VTGWSPEC']

dim_NN =int(564)
dim_NNout =int(140)


