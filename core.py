import numpy as np

from global_var import *
import tools as tl


class Core(object):
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file

        self._raw = None
        self._time_info = None
        self._ref_frame = 0
        self._range = [-0.01, 0.01]

        self.__video_stats = None

        self.k = 1
        self.type = 'diff'
        self.fouriere_level = 30
        self.postprocessing = [self.fourier]

        self.spr_time = None
        self.spr_signal = None
        self.reference = None


        self._load_data()
        self.ref_frame = 0

    def _load_data(self):
        self.__video_stats = self._load_stats()
        self._raw = self._load_video()
        self.spr_time, self.spr_signal = self._load_spr()

    def _load_stats(self):
        suffix = '.tsv'
        with open(self.folder + self.file + suffix, mode='r') as fid:
            file_content = fid.readlines()

        self._time_info = tl.frame_times(file_content)
        stats = file_content[1].split()
        video_length = len(file_content) - 1
        video_width = int(stats[1])
        video_height = int(stats[2])

        return video_width, video_height, video_length

    def _load_video(self):
        with open(self.folder + self.file + '.bin', mode='rb') as fid:
            video = np.fromfile(fid, dtype=np.float64)
            fid.close()

        video = np.reshape(video, (self.__video_stats[0],
                                   self.__video_stats[1],
                                   self.__video_stats[2]), order='F')

        return np.swapaxes(video, 0, 1)

    def _load_spr(self):
        with open(self.folder + NAME_LOCAL_SPR + self.file[-5:] + '.tsv', 'r') as spr:
            contents = spr.readlines()

        time = []
        signal = []

        for line in contents[:-1]:
            line_split = line.split('\t')
            time.append(float(line_split[0]))
            signal.append(float(line_split[1]))

        return time, signal

    def __len__(self):
        return self._raw.shape[2]

    @property
    def shape(self):
        return self._raw.shape

    @property
    def shape_img(self):
        return self._raw.shape[:2]

    @property
    def ref_frame(self):
        return self._ref_frame

    @ref_frame.setter
    def ref_frame(self, f):
        self._ref_frame = f
        self.reference = np.sum(
            self._raw[:, :, self.ref_frame: self.ref_frame + self.k],
            axis=2
        ) / self.k

    def frame(self, f):
        if f < 2 * self.k:
            image = np.zeros(self.shape_img)
            return {
                'time': self._time_info[f],
                'range': self._range,
                'image': image
            }

        if self.type == 'diff':
            current = np.sum(
                self._raw[:, :, f - self.k + 1: f + 1],
                axis=2
            ) / self.k
            previous = np.sum(
                self._raw[:, :, f - 2 * self.k + 1: f - self.k + 1],
                axis=2
            ) / self.k
            image = current - previous

        elif self.type == 'int':
            current = np.average(
                self._raw[:, :, f // self.k * self.k - self.k: f // self.k * self.k],
                axis=2
            )
            image = current - self.reference

        if len(self.postprocessing) != 0:
            for p in self.postprocessing:
                image = p(image)

        return {
            'time': self._time_info[f],
            'range': self._range,
            'image': image
        }

    def fourier(self, image):
        f = np.fft.fft2(image)
        # magnitude_spectrum = 20 * np.log(np.abs(f))
        # mask = np.real(magnitude_spectrum) > self.fourier_level
        mask = np.abs(f) > np.exp(self.fouriere_level / 20)
        f[mask] = 0
        return np.real(np.fft.ifft2(f))
