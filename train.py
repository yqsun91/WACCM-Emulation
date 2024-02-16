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
import Model
from loaddata import newnorm, data_loader



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

            #save model
            torch.save(model.state_dict(), 'conv_torch.pth')

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




## load mean and std for normalization
fm = np.load('Demodata/mean_demo.npz')
fs = np.load('Demodata/std_demo.npz')

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

ilev = 93

dim_NN =int(8*ilev+4)
dim_NNout =int(2*ilev)

model = Model.FullyConnected()

train_losses = []
val_losses = [0]

learning_rate = 1e-5
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=1e-5


s_list = list(range(1, 6))

for iter in s_list:
    if (iter > 1):
        model.load_state_dict(torch.load('conv_torch.pth'))
    print ('data loader iteration',iter)
    filename  = './Demodata/newCAM_demo_' + str(iter).zfill(1) + '.nc'
    print('working on: ', filename)

    F = nc.Dataset(filename)
    PS = np.asarray(F['PS'][0,:])
    PS = newnorm(PS, PSm, PSs)
    
    Z3 = np.asarray(F['Z3'][0,:,:])
    Z3 = newnorm(Z3, Z3m, Z3s)
    
    U = np.asarray(F['U'][0,:,:])
    U = newnorm(U, Um, Us)
    
    V = np.asarray(F['V'][0,:,:])
    V = newnorm(V, Vm, Vs)
    
    T = np.asarray(F['T'][0,:,:])
    T = newnorm(T, Tm, Ts)
    
    lat = F['lat']
    lat = newnorm(lat, np.mean(lat), np.std(lat))
    
    lon = F['lon']
    lon = newnorm(lon, np.mean(lon), np.std(lon))
    
    DSE = np.asarray(F['DSE'][0,:,:])
    DSE = newnorm(DSE, DSEm, DSEs)
    
    RHOI = np.asarray(F['RHOI'][0,:,:])
    RHOI = newnorm(RHOI, RHOIm, RHOIs)
    
    NETDT = np.asarray(F['NETDT'][0,:,:])
    NETDT = newnorm(NETDT, NETDTm, NETDTs)
    
    NM = np.asarray(F['NMBV'][0,:,:])
    NM = newnorm(NM, NMm, NMs)
    
    UTGWSPEC = np.asarray(F['UTGWSPEC'][0,:,:])
    UTGWSPEC = newnorm(UTGWSPEC, UTGWSPECm, UTGWSPECs)
    
    VTGWSPEC = np.asarray(F['VTGWSPEC'][0,:,:])
    VTGWSPEC = newnorm(VTGWSPEC, VTGWSPECm, VTGWSPECs)
    
    x_train,y_train = data_loader(U,V,T, DSE, NM, NETDT, Z3, RHOI, PS,lat,lon,UTGWSPEC, VTGWSPEC)

    data = Model.myDataset(X=x_train, Y=y_train)

    batch_size = 128

    split_data = torch.utils.data.random_split(data, [0.75, 0.25], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(split_data[0], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(split_data[1], batch_size=len(split_data[1]), shuffle=True)

     # training
    early_stopper = EarlyStopper(patience=5, min_delta=0) # Note the hyper parameters.
    for t in range(epochs):
        if t % 2 ==0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(val_losses[-1])
            print('counter=' + str(early_stopper.counter))
        train_loss = Model.train_loop(train_dataloader, model, nn.MSELoss(), optimizer)
		
        train_losses.append(train_loss)
        val_loss = Model.val_loop(val_dataloader, model, nn.MSELoss())
        val_losses.append(val_loss)
        if early_stopper.early_stop(val_loss):
            print("BREAK!")
            break
                                                                                                             
