from core import Core
from view import View
import time
import copy
time_start = time.time()
main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '20_04_21_L3_tomas/'
folder = main_folder + '20_04_20_Q4/'

folder = r'C:\SPRUP_data_Jenda\2020_09_25_Jenda_prism_grating\20_10_16_L3/'.replace('\\', '/')

file = 'raw_02_1'
core = Core(folder, file)
core.k = 10


file = 'raw_02_2'
core2 = Core(folder, file)
core2.k = 10

# file = 'raw_02_3'
# core3 = Core(folder, file)
# core3.k = 10
#
# file = 'raw_02_4'
# core4 = Core(folder, file)
# core4.k = 10


view = View()
view.add_core(core)
view.add_core(core2)
# view.add_core(core3)
# view.add_core(core4)

for i, core in enumerate(view.core_list):
    print('channel {}.'.format(i))
    core.make_intensity_raw()
    core.make_intensity_int()
    core.make_std_int()

view.orientation = False
view.show_img()
view.show_plots()


print(core.shape)

print('\n--elapsed time--\n{:.2f} s'.format(time.time()-time_start))
