from core import Core

main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '20_04_21_L3_tomas/'
file = 'raw_01_4'

core = Core(folder, file)

print(core.file)

print(core.load_stats())
core.load_data()
print(core._video.shape)
