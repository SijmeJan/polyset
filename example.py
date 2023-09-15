#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from polyset.dsi_mode import DSIMode
from polyset.tanhsinh import TanhSinh

dust_to_gas_ratio = 0.001
stokes_range = [0.01, 0.01]
search_domain = [[-2,2],[0,1]]
sound_speed_over_eta = 1/np.sqrt(0.001)
z_over_H = 1.0

dm = DSIMode(dust_to_gas_ratio,
             stokes_range,
             search_domain[0],
             search_domain[1],
             sound_speed_over_eta=sound_speed_over_eta,
             single_size_flag=True,
             tanhsinh_integrator=TanhSinh(),
             z0_over_H=z_over_H)

kx = np.logspace(-1,4,10)
kz = np.logspace(-1,4,10)

omega = np.zeros((len(kx),len(kz)), dtype=complex)

for i in range(0, len(kx)):
    for j in range(0, len(kz)):
        wave_number_x = kx[i]
        wave_number_z = kz[j]

        res = dm.calculate(wave_number_x, wave_number_z)

        print(100.0*(i*len(kz) + j)/(len(kx)*len(kz)), wave_number_x, wave_number_z, res)
        if len(res) > 0:
            omega[i,j]=res[res.imag.argmax()]


plt.xscale('log')
plt.yscale('log')

plt.contourf(kx, kz, np.log10(np.imag(omega)).transpose())
#plt.imshow(np.log10(np.imag(omega)))
plt.colorbar()

plt.show()
