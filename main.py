import copy
import time

from core import Core
from view import View

t = time.time()

main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '20_06_03_K5_tomas/'
file = 'raw_200603 cycle 1 strep 80nm PBS_1'

core = Core(folder, file)
core.k = 10


file = file[:-1] + '2'
core2 = Core(folder, file)
core2.k = 10

file = file[:-1] + '3'
core3 = Core(folder, file)
core3.k = 10

file = file[:-1] + '4'
core4 = Core(folder, file)
core4.k = 10

view = View()
view.add_core(core)
view.add_core(core2)
view.add_core(core3)
view.add_core(core4)
view.show()


print('elapsed time: {:.1f}'.format(time.time() - t))