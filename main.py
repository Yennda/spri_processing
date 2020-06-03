from core import Core
from view import View
import copy

main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '20_04_21_L3_tomas/'
file = 'raw_02_1'
core = Core(folder, file)
core.k = 10


file = 'raw_02_2'
core2 = Core(folder, file)
core2.k = 10

view = View()
view.add_core(core)
view.add_core(core2)
view.show()


print(core.shape)