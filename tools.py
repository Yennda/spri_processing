import numpy as np


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

    return time_info


def SecToMin(sec):
    return '{:.0f}:{:.1f}'.format(sec // 60, sec % 60)


def read_file_info(path):
    with open(path + '.tsv') as f:
        next(f)
        lines = f.readlines()
    t0, width, height, __, ets, avg, ___ = lines[0].split('\t')
    t2, *_ = lines[1].split('\t')

    return int(width), int(height), (int(t2) - int(t0)) / 1e7, int(avg), int(len(lines)), float(ets)


def fourier_filter(img, level):
    f = np.fft.fft2(img)
    magnitude_spectrum = 20 * np.log(np.abs(f))

    print(np.average(magnitude_spectrum))
    print(np.min(magnitude_spectrum))
    print(np.max(magnitude_spectrum))

    mask = np.abs(magnitude_spectrum) > level

    mask = np.full(magnitude_spectrum.shape, False, dtype=bool)

    print(img.shape)
    print(int(img.shape[1] / 2 * (1 - level / 100)))
    print(int(img.shape[1] / 2 * (1 + level / 100)))

    mask[
    int(img.shape[0] / 2 * (1 - level / 100)): int(img.shape[0] / 2 * (1 + level / 100)),
    int(img.shape[1] / 2 * (1 - level / 100)): int(img.shape[1] / 2 * (1 + level / 100))
    ] = True

    f[mask] = 0
    return np.real(np.fft.ifft2(f))
