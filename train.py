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

model = FullyConnected()

train_losses = []
val_losses = [0]

learning_rate = 1e-4
epochs = 4000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=1e-5


s_list = list(range(1, 500))

for iter in s_list:
    if (iter > 1):
        model.load_state_dict(torch.load('conv_torch.pth'))
    print ('data loader iteration',iter)
    filename  = '/glade/scratch/pahlavan/waccm_emu/Q_norm_conv/train_data_' + str(iter) + '.npz'

    F = np.load(filename)
    PS = np.asarray(F['PS'])
    Z3 = np.asarray(F['Z3'])
    U = np.asarray(F['U'])
    V = np.asarray(F['V'])
    T = np.asarray(F['T'])
    lat = F['lat']
    lon = F['lon']

    DSE = np.asarray(F['DSE'])
    RHOI = np.asarray(F['RHOI'])
    NETDT = np.asarray(F['NETDT'])
    NM = np.asarray(F['NM'])

    UTGWSPEC = np.asarray(F['UTGWSPEC'])
    VTGWSPEC = np.asarray(F['VTGWSPEC'])

    x_train,y_train = data_loader(U,V,T, DSE, NM, NETDT, Z3, RHOI, PS,lat,lon,UTGWSPEC, VTGWSPEC)

    data = myDataset(X=x_train, Y=y_train)

    batch_size = 1024

    split_data = torch.utils.data.random_split(data, [0.75, 0.25], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(split_data[0], batch_size=batch_size, shuffle=True, num_workers=18)
    val_dataloader = DataLoader(split_data[1], batch_size=len(split_data[1]), shuffle=True, num_workers=18)


     # training
    early_stopper = EarlyStopper(patience=3, min_delta=0) # Note the hyper parameters.
    for t in range(epochs):
        if t % 2 ==0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(val_losses[-1])
            print('counter=' + str(early_stopper.counter))
        train_loss = train_loop(train_dataloader, model, nn.MSELoss(), optimizer)
        train_losses.append(train_loss)
        val_loss = val_loop(val_dataloader, model, nn.MSELoss())
        val_losses.append(val_loss)
        if early_stopper.early_stop(val_loss):
            print("BREAK!")
            break
                                                                                                             
