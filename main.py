from core import Core
from view import View

main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '20_05_26_K5/'
file = 'raw_01_2'

core = Core(folder, file)
core.k = 10
core.type = 'int'

view = View(core)
view.show()


print(core.shape)