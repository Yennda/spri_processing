from core import Core
from view import View
import copy

main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '20_05_26_K5/'
file = 'raw_01_2'

core = Core(folder, file)
core.k = 10


view = View()
view.add_core(core)
view.add_core(copy.deepcopy(core))
view.orientation = False
view.show()


print(core.shape)