import numpy as np

import tools as tl


class Core(object):
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file

        self._raw = None
        self._time_info = None

        self.__video_stats = None

        self.k = 1

        self._load_data()

    def _load_data(self):
        self.__video_stats = self._load_stats()
        self._raw = self._load_video()

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
        self.spr_signals = []

        for c in self._channels:
            f = open(self.folder + NAME_GLOBAL_SPR + '_{}.tsv'.format(c + 1), 'r')
            contents = f.readlines()

            time = []
            signal = []

            for line in contents[:-1]:
                line_split = line.split('\t')

                if c == 0:
                    time.append(float(line_split[0]))
                signal.append(float(line_split[1]))

            if c == 0:
                self.spr_time = time
            self.spr_signals.append(signal)

    def __len__(self):
        return self._raw.shape[2]

    @property
    def shape(self):
        return self._raw.shape
