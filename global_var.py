FOLDER_EXPORTS = 'exports'
FOLDER_EXPORTS_NP = 'exports_np'
FOLDER_IDEAS = 'ideas'
FOLDER_BIOEXPORTS = '/exports_bio'
FOLDER_SAVED = 'saved_data'

NAME_RAW = 'raw'
NAME_LOCAL_SPR = 'spr'
NAME_GLOBAL_SPR = 'spr_global'

# INIT_RANGE = [-0.01. 0.01]
# INIT_RANGE = [-0.003, 0.003]
INIT_RANGE = [-0.0075, 0.0075]
INIT_CORR = [-50, 200]
INIT_FOUR = [-50, 50]

PX = 2.93e-3 #mm

# LMIN, LMAX = [0, -1]
# SMIN, SMAX = [0, -1]
#vertical
LMIN, LMAX = [400, 800]
SMIN, SMAX = [0, -1]

# horizontal
# YMIN, YMAX = [400, 800]
# XMIN, XMAX = [0, -1]

yellow = '#ffb200'
red = '#DD5544'
blue = '#0284C0'
black = '#000000'
green = '#008000'
gray = '#555555'
purple = '#DD84C0'

COLORS = [yellow, blue, red, green, purple, black]

SIDES = ['left', 'right', 'bottom', 'top']

HELP = '''
---------------------
FILE NAMES CONVENTION
---------------------
    - the raw file starts with 'raw' and ends with the underscore and a nuber of the channel (1 - 4). Different formats do not work.
        eg. raw_02_1, raw_200603 cycle 2 1kDa 60nm PBS_4, etc.
        
    - the spr data files have to be named exactly the same as the corresponding raw files, starting with 'spr' instead of 'raw'.
        eg. spr_02_1, spr_200603 cycle 2 1kDa 60nm PBS_4, etc.

------------------
KEYBOARD SHORTCUTS
---------------------
'8'/'5' increases/decreases contrast
Mouse scrolling moves the time 
'1' and '3' jumps 1 frames in time
'4' and '6' jumps 10 frames in time
'7' and '9' jumps 100 frames in time
'8' and '5' increase or decrease contrast
'ctrl + 1 or 2 or 3 or 4 or 5' switches to raw, integral, differential image, correlation or differential without any filters
'alt + 1 or 2 or 3' switches to fourier transformed raw, integral or differential image
'alt + 4' shows position of all the detected NPs
'i' sets the current frame as a reference for the integral image
'f' turns on/off all the filters

The shortcuts work locally for the active channel (mouse within its area) or globally for all the channels (mouse outside any channel).

Clicking into graphs also works as a navigation.

Official MATPLOTLIB shortcuts at https://matplotlib.org/users/navigation_toolbar.html

-------
LICENCE
-------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

Jan Bukáček, bukacek@ufe.cz
https://github.com/Yennda/spri_processing

Institute of Photonics and Elctronics, Czech Academy of Sciences
www.ufe.cz

2021




'''
