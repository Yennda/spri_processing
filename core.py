import numpy as np

import tools as tl

class Core(object):
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file

        self._video = None
        self._time_info = None

    def load_data(self):
        self.__video_stats = self.load_stats()
        self._video = self.load_video()


    def load_stats(self):
        suffix = '.tsv'
        with open(self.folder + self.file + suffix, mode='r') as fid:
            file_content = fid.readlines()

        self.time_info = tl.frame_times(file_content)
        stats = file_content[1].split()
        video_length = len(file_content) - 1
        video_width = int(stats[1])
        video_hight = int(stats[2])
        video_fps = float(stats[4]) * int(stats[5])

        return [video_fps, [video_width, video_hight, video_length]]

    def load_video(self):
        with open(self.folder + self.file + '.bin', mode='rb') as fid:
            video = np.fromfile(fid, dtype=np.float64)
            fid.close()

        video = np.reshape(video, (self.__video_stats[1][0],
                                   self.__video_stats[1][1],
                                   self.__video_stats[1][2]), order='F')

        return np.swapaxes(video, 0, 1)