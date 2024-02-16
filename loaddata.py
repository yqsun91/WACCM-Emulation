import numpy as np

ilev = 93
dim_NN =int(8*ilev+4)
dim_NNout =int(2*ilev)

def newnorm(var, varm, varstd):
  dim=varm.size
  if dim > 1 :
    vara = var - varm[:, :]
    varstdmax = varstd
    varstdmax[varstd==0.0] = 1.0
    tmp = vara / varstdmax[:, :]
  else:
    tmp = ( var - varm ) / varstd
  return tmp


def data_loader (U,V,T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC):

  Nlat = U.shape[1]
  Nlon = U.shape[2]
  Ncol = Nlat*Nlon
   
  x_train = np.zeros([dim_NN,Ncol])
  y_train = np.zeros([dim_NNout,Ncol])


  x_train [0:ilev, : ] = U.reshape(ilev, Ncol)
  x_train [ilev:2*ilev, :] = V.reshape(ilev, Ncol)
  x_train [2*ilev:3*ilev,:] = T.reshape(ilev, Ncol)
  x_train [3*ilev:4*ilev, :] = DSE.reshape(ilev, Ncol)
  x_train [4*ilev:5*ilev, :] = NM.reshape(ilev, Ncol)
  x_train [5*ilev:6*ilev, :] = NETDT.reshape(ilev, Ncol)
  x_train [6*ilev:7*ilev, :] = Z3.reshape(ilev, Ncol)
  x_train [7*ilev:8*ilev+1, :] = RHOI.reshape(ilev+1, Ncol)
  x_train [8*ilev+1:8*ilev+2, :] = PS.reshape(1, Ncol)
  x_train [8*ilev+2:8*ilev+3, :] = np.transpose(np.tile(lat, (Nlon, 1))).reshape(1, Ncol)
  x_train [8*ilev+3:ilev*ilev+4, :] = np.tile(lon, (1, Nlat))

  y_train [0:ilev, :] = UTGWSPEC.reshape(ilev, Ncol)
  y_train [ilev:2*ilev, :] = VTGWSPEC.reshape(ilev, Ncol)

  return x_train,y_train

