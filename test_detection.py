import cv2

import math as m
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from core import Core

from global_var import COLORS

matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Palatino Linotype')
matplotlib.rc('text', usetex='false')
matplotlib.rcParams.update({'font.size': 20})

folder = r'C:\SPRUP_data_Jenda\2019_03_13_Jenda_microscopy\20_12_11_BC5/'.replace('\\', '/')

file = 'raw_02_'


core = Core(folder, file+str(1))
# core.downsample(5)
core.k = 10
core.type = 'diff'
core.downsample(5)



fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.set_title(file)
axes.set_xlabel('frame')
axes.set_ylabel('intensity [a. u.]')


axes.imshow(
    core.   count_nps(),
    cmap='gray',
    zorder=0,
    # vmin=core.range[0],
    # vmax=core.range[1]
)

plt.show()

# fig.savefig('images/intensity_fluctiations_SLED.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
