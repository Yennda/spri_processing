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
        self._range = {
            'diff': INIT_RANGE,
            'raw': [0, 1],
            'int': INIT_RANGE
        }
        self.__video_stats = None

        self.k = 1
        self.type = 'diff'
        self.fourier_level = 30
        self.postprocessing = []

        self.graphs = dict()
        self.spr_time = None
        self.zero_time = None
        self.reference = None

        self._load_data()
        self.ref_frame = 10

    def _load_data(self):
        self.__video_stats = self._load_stats()
        self._raw = self._load_video()
        self.spr_time, self.graphs['spr_signal'] = self._load_spr()

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
        with open(self.folder + self.file.replace(NAME_RAW, NAME_LOCAL_SPR) + '.tsv', 'r') as spr:
            contents = spr.readlines()
        time = []
        signal = []

        for line in contents[:-1]:
            line_split = line.split('\t')
            time.append(float(line_split[0]))
            signal.append(float(line_split[1]))

        return time, np.array(signal)

    def synchronize(self):
        try:
            with open(self.folder + NAME_GLOBAL_SPR + self.file[-2:] + '.tsv', 'r') as spr:
                contents = spr.readlines()
        except:
            self.zero_time = 0
            return

        time = []
        signal = []

        for line in contents[:-1]:
            line_split = line.split('\t')
            time.append(float(line_split[0]))
            signal.append(float(line_split[1]))
        beginning = list(self.graphs['spr_signal'][:2])
        for i in range(len(signal)):
            if signal[i:i + 2] == beginning:
                self.zero_time = time[i]
                break
            elif i == len(signal) - 2:
                self.zero_time = 0
                # raise Exception('Could not match global and local SPR signals.')

    def __len__(self):
        return self._raw.shape[2]

    @property
    def shape(self):
        return self._raw.shape

    @property
    def area(self):
        return self._raw.shape[0] * self._raw.shape[1]

    @property
    def shape_img(self):
        return self._raw.shape[:2]

    @property
    def ref_frame(self):
        return self._ref_frame

    @ref_frame.setter
    def ref_frame(self, f):
        if f > self.k:
            self._ref_frame = f // self.k * self.k
        else:
            self._ref_frame = self.k

        self.reference = np.sum(
            self._raw[:, :, self.ref_frame - self.k: self.ref_frame],
            axis=2
        ) / self.k

    @property
    def range(self):
        return self._range[self.type]

    @range.setter
    def range(self, r):
        self._range[self.type] = r

    @property
    def intensity(self):
        type_buffer = self.type
        self.type = 'raw'
        intensity = np.sum(self.frame(0))
        self.type = type_buffer
        return intensity

    def frame(self, f):
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

        elif self.type == 'raw':
            image = self._raw[:, :, f]

        if len(self.postprocessing) != 0:
            for p in self.postprocessing:
                image = p(image)

        return image

    def fourier(self, image):
        f = np.fft.fft2(image)
        mask = np.abs(f) > np.exp(self.fourier_level / 20)
        f[mask] = 0
        return np.real(np.fft.ifft2(f))

    def apply_function(self, fn, progress_callback):
        out = []
        for f in range(len(self)):
            out.append(fn(self.frame(f)))
            # print('\r\t{}/ {}'.format(f + 1, len(self)), end='')
            progress_callback.emit((f + 1) / len(self) * 100)
        return np.array(out)

    def make_intensity_raw(self, progress_callback):
        # print('Processing raw intensity')
        type_buffer = self.type
        self.type = 'raw'
        self.graphs['intensity_raw'] = self.apply_function(np.sum, progress_callback) / self.area
        self.type = type_buffer
        return 'done'

    def make_intensity_int(self, progress_callback):
        # print('Processing int. intensity')
        type_buffer = self.type
        self.type = 'int'
        self.graphs['intensity_int'] = self.apply_function(np.sum, progress_callback) / self.area / self.intensity
        self.type = type_buffer
        return 'done'

    def make_std_int(self, progress_callback):
        # print('Processing int. std')
        type_buffer = self.type
        self.type = 'int'
        self.graphs['std_int'] = self.apply_function(np.std, progress_callback) / self.intensity
        self.type = type_buffer
        return 'done'
