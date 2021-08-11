import numpy as np
import scipy.signal
import os
import random
from scipy.ndimage import gaussian_filter
import scipy.optimize as opt
from global_var import FOLDER_NP_IMAGES
import matplotlib.pyplot as plt

from scipy import ndimage


def frame_times(file_content):
    time0 = int(file_content[1].split()[0])
    time_info = []
    time_last = time0

    for line in file_content[1:]:
        time_actual = int(line.split()[0])
        time_info.append([
            (time_actual - time0) / 1e7,
            (time_actual - time_last) / 1e7
        ])
        time_last = time_actual

    return np.array(time_info)


def SecToMin(sec):
    return '{:.0f}:{:.1f}'.format(sec // 60, sec % 60)


def BoolFromCheckBox(value):
    if value.checkState() == 0:
        return False
    else:
        return True


def read_file_info(path):
    with open(path + '.tsv') as f:
        next(f)
        lines = f.readlines()
    t0, width, height, __, ets, avg, ___ = lines[0].split('\t')
    t2, *_ = lines[1].split('\t')

    return int(width), int(height), (int(t2) - int(t0)) / 1e7, int(avg), int(len(lines)), float(ets)


def fourier_filter(img, level, longpass=True):
    f = np.fft.fft2(img)
    magnitude_spectrum = 20 * np.log(np.abs(f))

    # mask = np.abs(magnitude_spectrum) > level
    mask = np.full(magnitude_spectrum.shape, longpass, dtype=bool)
    mask[
    int(img.shape[0] / 2 * (1 - level / 100)): int(img.shape[0] / 2 * (1 + level / 100)),
    int(img.shape[1] / 2 * (1 - level / 100)): int(img.shape[1] / 2 * (1 + level / 100))
    ] = not longpass

    f[mask] = 0
    return np.real(np.fft.ifft2(f))


def fourier_filter_gauss(img, level):
    f = np.fft.fft2(img)
    magnitude_spectrum = 20 * np.log(np.abs(f))

    magnitude_spectrum = gaussian_filter(magnitude_spectrum, level)

    return np.real(np.fft.ifft2(np.exp(magnitude_spectrum / 20)))


def fourier_filter_threshold(img, level):
    f = np.fft.fft2(img)
    f[f > np.exp(level / 20)] = 0

    f[:25, :25] = 0
    f[-25:, -25:] = 0
    return np.real(np.fft.ifft2(f))


def spectral_wiener_filter(img, size, noise=None):
    f = np.fft.fft2(img)
    print(np.std(f))
    print(np.var(f))

    return np.real(np.fft.ifft2(scipy.signal.wiener(f, size, noise)))


def true_coordinate(x):
    return int((x + 0.5) // 1)


def random_color():
    return (random.random(), random.random(), random.random())


def before_save_file(path):
    if os.path.isfile(path):
        print('=' * 80)

        print(
            'Old data in {} will be overwriten.'
            'Type "y" as yes or "n"'
            'as no bellow in the command line.'.format(path))

        print('=' * 80)
        result = input()

        if result == 'y':
            os.remove(path)

            return True

        return False

    return True


def np_analysis(npp, folder, file, image = False):
    def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                           + c * ((y - yo) ** 2)))
        return g.ravel()

    initial_guess = (0.03, 25, 25, 2, 2, 0, 10)

    x = np.linspace(0, npp.shape[0] - 1, npp.shape[0])
    y = np.linspace(0, npp.shape[1] - 1, npp.shape[1])
    x, y = np.meshgrid(x, y)

    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), np.abs(npp).reshape(npp.shape[0] * npp.shape[1]),
                                   p0=initial_guess)
    except RuntimeError:
        return False

    data_fitted = twoD_Gaussian((x, y), *popt)
    lim = np.average(np.abs(npp)) * 4
    perr = np.sqrt(np.diag(pcov))

    threshold = data_fitted.reshape(npp.shape) > popt[6] * 3

    area = np.sum(threshold * 1)
    intensity = np.sum(np.abs(npp)[threshold])
    intensity_px = np.average(np.abs(npp)[threshold])
    intensity_bg_px = np.average(np.abs(npp)[threshold == False])

    snr = intensity_px / intensity_bg_px

    if any(perr > 1) or any(np.isnan(perr)) or np.abs(popt[3]) > 15 or np.abs(popt[4]) > 15 or popt[0] < 0 or popt[
        0] < lim:
        return False
    if image:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(npp, cmap='gray', origin='lower')
        ax.imshow(threshold, cmap='plasma', origin='lower', alpha=0.5)

        if not os.path.isdir(folder + FOLDER_NP_IMAGES):
            os.mkdir(folder + FOLDER_NP_IMAGES)

        file_name = folder + FOLDER_NP_IMAGES + '/' + file + '_fit'

        i = 1
        while os.path.isfile(file_name + '_{:02d}.png'.format(i)):
            i += 1
        file_name += '_{:02d}'.format(i)

        fig.savefig(file_name + '.png', dpi=300, bbox_inches='tight')

    return [area, intensity, intensity_px, intensity_bg_px, snr]
