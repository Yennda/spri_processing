import os
import time

import cv2
import numpy as np
import scipy as sc
import scipy.signal
from scipy import ndimage

from global_var import *
import tools as tl
from nanoparticle import NanoParticle


class Core(object):
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file

        self._data_raw = None
        self._data_mask = None
        self._data_corr = None

        self._time_info = None
        self._ref_frame = 0
        self._range = {
            'diff': INIT_RANGE,
            'raw': [0, 1],
            'mask': [0, 1],
            'int': INIT_RANGE,
            'corr': INIT_CORR,
            'four_r': INIT_FOUR,
            'four_d': INIT_FOUR,
            'four_i': INIT_FOUR

        }
        self.__video_stats = None

        self.k = 1
        self.type = 'diff'
        self._f = int()
        self.fourier_level = 30
        self.postprocessing = True
        self.postprocessing_filters = dict()

        # np_counting
        self.stack_frames = []
        self.dist = 4
        self.np_container = []
        self.nps_in_frame = []
        self.show_nps = False

        self.graphs = dict()
        self.spr_time = None
        self.zero_time = None
        self.reference = None

        self.idea3d = None

        self._load_data()
        self.load_idea()
        self.ref_frame = 10

    def _load_data(self):
        self.__video_stats = self._load_stats()
        self._data_raw = self._load_video()

        if self._data_raw.shape[0] < self._data_raw.shape[1]:
            self._data_raw = self._data_raw[SMIN: SMAX, LMIN: LMAX, :]
        else:
            self._data_raw = self._data_raw[LMIN: LMAX, SMIN: SMAX, :]

        self.spr_time, self.graphs['spr_signal'] = self._load_spr()
        self._synchronize()

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
        try:
            with open(self.folder + self.file.replace(NAME_RAW, NAME_LOCAL_SPR) + '.tsv', 'r') as spr:
                contents = spr.readlines()
            time = []
            signal = []

            for line in contents[:-1]:
                line_split = line.split('\t')
                time.append(float(line_split[0]))
                signal.append(float(line_split[1]))

            return time, np.array(signal)
        except FileNotFoundError:
            print('SPR file not found. Diseable ploting of SPR. ')
            return None, None

    def downsample(self, k):
        if k == 0:
            return
        self._data_raw = scipy.signal.decimate(self._data_raw, k, axis=2)
        self._time_info = scipy.signal.decimate(self._time_info, k, axis=0)

        if self.spr_time is not None:
            self.spr_time = scipy.signal.decimate(self.spr_time, k)
        for key in self.graphs:
            if self.graphs[key] is not None:
                self.graphs[key] = scipy.signal.decimate(self.graphs[key], k)

    def _synchronize(self):
        try:
            with open(self.folder + NAME_GLOBAL_SPR + self.file[-2:] + '.tsv', 'r') as spr:
                contents = spr.readlines()
        except FileNotFoundError:
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
        return self._data_raw.shape[2]

    @property
    def shape(self):
        return self._data_raw.shape

    @property
    def area(self):
        return self._data_raw.shape[0] * self._data_raw.shape[1]

    @property
    def shape_img(self):
        return self._data_raw.shape[:2]

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
            self._data_raw[:, :, self.ref_frame - self.k: self.ref_frame],
            axis=2
        ) / self.k

    @property
    def range(self):
        return self._range[self.type]

    @range.setter
    def range(self, r):
        self._range[self.type] = r

    def intensity(self, f):
        type_buffer = self.type
        self.type = 'raw'
        intensity = np.sum(self.frame(f))
        self.type = type_buffer
        return intensity

    def save_data(self, name=None):
        if not os.path.isdir(self.folder + FOLDER_SAVED):
            os.mkdir(self.folder + FOLDER_SAVED)

        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_SAVED + '/' + name

        data = np.zeros(self.shape)
        for f in range(len(self)):
            data[:, :, f] = self.frame(f)

        if tl.before_save_file(file_name) or name == self.file:
            np.save(file_name + '.npy', data)
            # np.save(file_name + '_frame' + '.npy', np.array(self._idea_frame))
            # np.save(file_name + '_spanx' + '.npy', self._idea_span_x)
            # np.save(file_name + '_spany' + '.npy', self._idea_span_y)

            print('Pattern saved')

        else:
            print('Could not save the pattern.')

    def save_idea(self, name=None):
        if not os.path.isdir(self.folder + FOLDER_IDEAS):
            os.mkdir(self.folder + FOLDER_IDEAS)

        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_IDEAS + '/' + name

        if tl.before_save_file(file_name) or name == self.file:
            np.save(file_name + '.npy', self.idea3d)
            # np.save(file_name + '_frame' + '.npy', np.array(self._idea_frame))
            # np.save(file_name + '_spanx' + '.npy', self._idea_span_x)
            # np.save(file_name + '_spany' + '.npy', self._idea_span_y)

            print('Pattern saved')

        else:
            print('Could not save the pattern.')

    def load_idea(self, name=None):
        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_IDEAS + '/' + name
        try:
            self.idea3d = np.load(file_name + '.npy')
            return True
        except FileNotFoundError:
            return False

    def frame_diff(self, f):
        current = np.sum(
            self._data_raw[:, :, f - self.k + 1: f + 1],
            axis=2
        ) / self.k
        previous = np.sum(
            self._data_raw[:, :, f - 2 * self.k + 1: f - self.k + 1],
            axis=2
        ) / self.k
        return current - previous

    def frame(self, f):
        self._f = f
        if self.type == 'diff':
            image = self.frame_diff(f)

        elif self.type == 'int':
            current = np.average(
                self._data_raw[:, :, f // self.k * self.k - self.k: f // self.k * self.k],
                axis=2
            )
            image = current - self.reference

        elif self.type == 'raw':
            image = self._data_raw[:, :, f]

        elif self.type == 'mask':
            image = self._data_mask[:, :, f]

        elif self.type == 'four_r':
            image_pre = self._data_raw[:, :, f]
            image = np.real(20 * np.log(np.abs(np.fft.fft2(image_pre))))

        elif self.type == 'four_i':
            current = np.average(
                self._data_raw[:, :, f // self.k * self.k - self.k: f // self.k * self.k],
                axis=2
            )
            image_pre = current - self.reference
            image = np.real(20 * np.log(np.abs(np.fft.fft2(image_pre))))

        elif self.type == 'four_d':
            image_pre = self.frame_diff(f)
            image = np.real(20 * np.log(np.abs(np.fft.fft2(image_pre))))

        elif self.type == 'corr':
            if self.idea3d is None:
                image = np.zeros(self.shape_img[0])
                print('No selected NP patter for file {}'.format(self.file))
                return image

            if self._data_corr is not None:
                image = self._data_corr[:, :, f]
                # print('Shape of corr: {}'.format(self._raw_corr.shape))
                #
                # print('Shape of image: {}'.format(image.shape))
            else:
                # self.make_correlation()
                # image = self._data_corr[:, :, f]

                sequence_diff = np.zeros((self.shape_img[0], self.shape_img[1], 4 * self.k))
                for i in range(4 * self.k):
                    diff_image = self.frame_diff(f + 2 * self.k - i)
                    if self.postprocessing and len(self.postprocessing_filters) != 0:
                        for p in self.postprocessing_filters.values():
                            diff_image = p(diff_image)

                    sequence_diff[:, :, i] = diff_image

                out = scipy.signal.correlate(
                    sequence_diff,
                    self.idea3d,
                    mode='same'
                ) * 1e5

                image = out[:, :, 2 * self.k]

        # if self.postprocessing and len(self.postprocessing_filters) != 0 and self.type != 'corr':
        if self.postprocessing and len(self.postprocessing_filters) != 0:
            for p in self.postprocessing_filters.values():
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
        self.graphs['intensity_int'] = self.apply_function(np.sum, progress_callback) / self.area / self.intensity(0)
        self.type = type_buffer
        return 'done'

    def make_std_int(self, progress_callback):
        # print('Processing int. std')
        type_buffer = self.type
        self.type = 'int'
        self.graphs['std_int'] = self.apply_function(np.std, progress_callback) / self.intensity(0)
        self.type = type_buffer
        return 'done'

    def make_correlation(self):
        if self.idea3d is None:
            image = np.zeros(self.shape_img[0])
            raise Exception('No selected NP patter for file {}'.format(self.file))

        img_type = self.type
        self.type = 'diff'

        raw_diff = np.zeros(self.shape)
        print('Processing data for correlation')
        for f in range(len(self)):
            print('\r\t{}/ {}'.format(f + 1, len(self)), end='')
            raw_diff[:, :, f] = self.frame(f)

        self._data_corr = scipy.signal.correlate(
            raw_diff,
            self.idea3d,
            mode='same'
        ) * 1e5

        self.type = img_type

    def histogram(self):
        frame = self.frame(self._f)
        n = 20
        values = np.linspace(np.min(frame), np.max(frame), n)
        counts = ndimage.measurements.histogram(
            frame,
            np.min(frame),
            np.max(frame),
            n
        )
        return values, counts

    def frame_np(self, f):
        positions = []
        colors = []

        for idnp in self.nps_in_frame[f]:
            nnp = self.np_container[idnp]
            positions.append(nnp.position(f))
            colors.append(nnp.color)

        return positions, colors

    def detect_spots(self, f, threshold):
        mask_positions = np.zeros(self.shape_img)

        frame = self.frame(f)
        mask = (frame < -1e-4 * threshold / 100 * 6) * 1
        mask = mask.astype(np.uint8)
        _, threshold_mask = cv2.threshold(
            mask,
            100,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        patterns = cv2.findContours(
            threshold_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE
        )[-2][: -1]

        for p in patterns:
            (x, y), radius = cv2.minEnclosingCircle(p)
            if radius <= 1.1:
                continue
            center = (int(y), int(x))
            mask_positions[center] = 1
            # threshold_mask = cv2.circle(threshold_mask, (int(x), int(y)), int(_), (0, 255, 0), 2)

        self._data_mask[:, :, f] = threshold_mask
        return mask_positions

    def track_nps(self, frame_first):
        del self.stack_frames[0]
        tracked_nps = []

        start_position = np.nonzero(frame_first)

        for (x, y) in np.transpose(start_position):
            np_time_evolution = [np.array([x, y])]

            for i in range(self.k * 2 - 1):

                surroundings = self.stack_frames[i][
                               x - self.dist:x + self.dist,
                               y - self.dist:y + self.dist
                               ]
                new_positions = np.transpose(surroundings.nonzero())

                if len(new_positions) == 0:
                    if len(np_time_evolution) >= 2:
                        tracked_nps.append(np_time_evolution)

                    break
                else:
                    new_position = np.array([x, y]) - np.array([self.dist] * 2) + np.average(new_positions, 0).astype(
                        np.int)
                    np_time_evolution.append(new_position)
                    self.stack_frames[i][
                    x - self.dist:x + self.dist,
                    y - self.dist:y + self.dist] = 0
                    (x, y) = new_position

        return tracked_nps

    def process_binary(self, data):
        data[data > 0.1] = 1
        data = data.astype(int)
        labeled_data, num = ndimage.label(data, np.ones((3, 3, 3)))
        objects = ndimage.find_objects(labeled_data)
        objects_out = []

        for o in objects:
            dim = labeled_data[o].shape
            volume_rough = dim[0] * dim[1] * dim[2]
            volume = len(labeled_data[o].nonzero()[0])

            if volume < 6 or volume_rough / volume > 4 or volume / dim[2] <= 2:
                data[o] = 0

            elif dim[0] * dim[1] < 3:
                data[o] = 0

            elif dim[0] < 2 or dim[1] < 2:
                print('ej')
                data[o] = 0

            elif 4 > dim[2] > self.k * 2 - 2:
                data[o] = 0
            else:
                print(dim)
                objects_out.append(o)

        return data, objects_out

    def count_nps(self, start, stop, threshold):
        time0 = time.time()
        print('Detecting NPs')
        self.nps_in_frame = [[] for i in range(len(self))]
        self.np_container =[]

        data = np.zeros(self.shape)
        self._data_mask = np.zeros(self.shape)
        for f in range(len(self)):
            data[:, :, f] = self.frame(f)

        data, np_slices = self.process_binary(data)

        idnp = 0
        for np_slice in np_slices:
            x = (np_slice[0].start + np_slice[0].stop) / 2
            y = (np_slice[1].start + np_slice[1].stop) / 2
            dt = int(np_slice[2].stop - np_slice[2].start)

            nnp = NanoParticle(idnp, np_slice[2].start, [np.array([x, y])] * dt)
            self.np_container.append(nnp)
            for i in range(dt):
                self.nps_in_frame[np_slice[2].start + i].append(idnp)
            idnp += 1

        self._data_mask = data
        self.show_nps = True

        return

        img_type = self.type
        # self.type = 'corr'
        # self.make_correlation()

        self._data_mask = np.zeros(self.shape)
        self.nps_in_frame = [[] for i in range(len(self))]
        self.stack_frames = [np.zeros(self.shape_img)] * 2 * self.k

        idnp = 0
        for f in range(start + self.k * 2, stop):
            print('\r\t{}/ {}'.format(f + 1, stop), end='')

            # tracked_nps = self.track_nps(self.stack_frames[0])
            #
            # for np_positions in tracked_nps:
            #     nnp = NanoParticle(idnp, f - self.k * 2, np_positions)
            #
            #     self.np_container.append(nnp)
            #
            #     for i in range(len(np_positions)):
            #         self.nps_in_frame[f - self.k * 2 + i].append(idnp)
            #
            #     # for i, (y, x) in enumerate(np_positions):
            #     #     self._data_mask[y, x, f - 2 * self.k + i] = 1
            #
            #     idnp += 1

            mask_positions = self.detect_spots(f, threshold)
            self.stack_frames.append(mask_positions)

        self.type = img_type
        self.show_nps = True

        print('\n--elapsed time--\n{:.2f} s'.format(time.time() - time0))
        return mask_positions
