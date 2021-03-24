import os
import time

import numpy as np
import scipy.signal
from scipy import ndimage
from numba import jit

from global_var import *
import tools as tl
import tools_numba as ntl
from nanoparticle import NanoParticle


class Core(object):
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file

        self._data_raw = None
        self._data_mask = None
        self._data_corr = None
        self._data_diff_std = None
        self._data_corr_std = None

        self._mask_fourier = None
        self._mask_ommit = None

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
        self.threshold = False
        self.threshold_value = 0
        self.threshold_adaptive = False

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
        self.autocorrelation_max = None

        time0 = time.time()
        self._load_data()
        self.load_idea()
        self._mask_ommit = np.zeros(self.shape_img)
        self.ref_frame = 10
        print('\n--elapsed time--\n{:.2f} s'.format(time.time() - time0))

    def _load_data(self):
        self.__video_stats = self._load_stats()
        self._data_raw = self._load_video()
        self.spr_time, self.graphs['spr_signal'] = self._load_spr()
        self._synchronize()


    def crop(self):
        if self._data_raw.shape[0] < self._data_raw.shape[1]:
            self._data_raw = self._data_raw[SMIN: SMAX, LMIN: LMAX, :]
        else:
            self._data_raw = self._data_raw[LMIN: LMAX, SMIN: SMAX, :]

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
        if FOLDER_SAVED in self.folder:
            print(self.folder)
            video = np.load(self.folder + self.file + '.npy')
            self.folder = self.folder.replace(FOLDER_SAVED, '')
            print(self.folder)

            return video

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

            for line in contents[:]:
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
        for name_global_spr in [NAME_GLOBAL_SPR, 'spr_integral']:
            try:
                with open(self.folder + name_global_spr + self.file[-2:] + '.tsv', 'r') as spr:
                    contents = spr.readlines()

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
                    return

            except FileNotFoundError:
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

    @property
    def active_area(self):
        if self._mask_ommit is not None:
            return -np.sum(self._mask_ommit * 1 - 1)
        else:
            return self.area

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

    def export_data(self, start, stop, name=None):
        if not os.path.isdir(self.folder + FOLDER_SAVED):
            os.mkdir(self.folder + FOLDER_SAVED)

        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_SAVED + '/' + name

        data = np.zeros(self.shape)
        for f in range(start, stop):
            data[:, :, f] = self.frame(f)

        if tl.before_save_file(file_name) or name == self.file:
            np.save(file_name + '.npy', data)
            # np.save(file_name + '_frame' + '.npy', np.array(self._idea_frame))
            # np.save(file_name + '_spanx' + '.npy', self._idea_span_x)
            # np.save(file_name + '_spany' + '.npy', self._idea_span_y)

            print('Data exported')

        else:
            print('Could not export the data')

    def export_csv(self, name=None):
        if not os.path.isdir(self.folder + FOLDER_SAVED):
            os.mkdir(self.folder + FOLDER_SAVED)

        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_SAVED + '/' + name

        corr_std = [0 for i in range(self.k * 3)]
        avg = np.average(self._data_corr_std[self.k * 3:])
        for cs in self._data_corr_std[self.k * 3:]:
            if cs / avg > 1:
                corr_std.append((cs / avg) ** self.threshold_adaptive)
            else:
                corr_std.append(1)

        with open(file_name + '.csv', mode='w') as f:
            nps_add = [sum(self.graphs['nps_pos'][:i]) for i in range(len(self))]
            for i in range(len(self)):
                f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    i,
                    self.spr_time[i],
                    self.spr_time[i] + self.zero_time,
                    self.graphs['nps_pos'][i],
                    nps_add[i],
                    self.graphs['nps_pos'][i] / self.active_area / PX ** 2,
                    nps_add[i] / self.active_area / PX ** 2,
                    corr_std[i],
                    self.active_area,
                    self.active_area * PX ** 2
                ))

        with open(file_name + '_log.txt', mode='w') as f:
            f.write(
                'frame, time [s], global time [min], NPs adsorbed in frame,' +
                'sum of adsorbed NPs, NPs adsorbed in frame [/mm2],sum of adsorbed NPs [/mm2],' +
                'threshold, area [px], area [mm2]'
            )

        print('Data exported')

    def export_np_csv(self, name=None):
        if not os.path.isdir(self.folder + FOLDER_SAVED):
            os.mkdir(self.folder + FOLDER_SAVED)

        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_SAVED + '/nps_' + name

        with open(file_name + '.csv', mode='w') as f:
            for npp in self.np_container:
                if npp.color == green:
                    f.write(
                        '{}, {}, {}, {}\n'.format(
                            npp.positions[0][0],
                            npp.positions[0][1],
                            npp.first_frame,
                            len(npp.positions)
                        ))

        with open(file_name + '_log.txt', mode='w') as f:
            f.write('x, y, first_frame, duration [frames]')

            print('Data exported')

    def import_np_csv(self, name=None):
        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_SAVED + '/nps_' + name
        try:
            with open(file_name + '.csv', 'r') as csv:
                contents = csv.readlines()

            self.np_container = []
            self.nps_in_frame = [[] for i in range(len(self))]
            self.graphs['nps_pos'] = np.array([0 for i in range(len(self))])

            for i, line in enumerate(contents):
                line_split = line.split(', ')

                first_frame = int(line_split[2])
                duration = int(line_split[3])
                positions = [np.array([float(line_split[0]), float(line_split[1])])] * duration

                nnp = NanoParticle(
                    i,
                    first_frame,
                    positions,
                    positive=True
                )
                nnp.color = green

                self.graphs['nps_pos'][first_frame] += 1

                for j in range(duration):
                    self.nps_in_frame[first_frame + j].append(i)

                self.np_container.append(nnp)

            self.show_nps = True
            print('NPs succesfully imported.')

        except FileNotFoundError:
            print('"{}"  not found. Diseable ploting of SPR. '.format(file_name))
            return None, None

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

            self.autocorrelation_max = np.max(scipy.signal.correlate(
                self.idea3d,
                self.idea3d,
                mode='same'
            ) * 1e5)

        else:
            print('Could not save the pattern.')

    def load_idea(self, name=None):
        if name == None:
            name = self.file

        file_name = self.folder + FOLDER_IDEAS + '/' + name
        try:
            self.idea3d = np.load(file_name + '.npy')

            self.autocorrelation_max = np.max(scipy.signal.correlate(
                self.idea3d,
                self.idea3d,
                mode='same'
            ) * 1e5)

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
        no_postpro = ['raw', 'four_d', 'four_i', 'mask', 'corr']
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
            if self._data_mask is None:
                image = np.zeros(self.shape_img)
            else:
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

            else:
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

        if self.type not in no_postpro:
            if self._mask_fourier is not None and self.postprocessing:
                f = np.fft.fft2(image)
                f[self._mask_fourier] = 0
                image = np.real(np.fft.ifft2(f))

            if self.postprocessing and len(self.postprocessing_filters) != 0:

                for p in self.postprocessing_filters.values():
                    image = p(image)

        if self.threshold and self.type == 'corr':
            if self.threshold_value > 0:
                image = ndimage.maximum_filter(image, size=2)

                # scikit.
                # image = (image > np.std(image) * self.threshold_value) * self._range[self.type][1]

                level = self._data_corr_std[f] / np.average(self._data_corr_std[self.k * 3:])

                if level > 1:

                    image = (
                                    image > level ** self.threshold_adaptive * self.autocorrelation_max * self.threshold_value) * \
                            self._range[self.type][1]
                else:
                    image = (image > self.autocorrelation_max * self.threshold_value) * self._range[self.type][1]

                image = (image > self.autocorrelation_max * self.threshold_value / 50) * self._range[self.type][1]

            else:
                image = -ndimage.maximum_filter(-image, size=2)
                image = (image < self._data_corr_std[f] / np.average(
                    self._data_corr_std[self.k * 3:]) * self.autocorrelation_max * self.threshold_value) * \
                        self._range[self.type][1]

                # image = (image < self.autocorrelation_max * self.threshold_value/50) * self._range[self.type][1]

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
        time0 = time.time()
        if self.idea3d is None:
            raise Exception('No selected NP patter for the file {}'.format(self.file))

        img_type = self.type
        self.type = 'diff'

        raw_diff = ntl.raw_diff(self._data_raw, self.k, self.idea3d)

        # raw_diff = np.zeros(self.shape)
        # self._data_diff_std = []
        # start = time.time()
        # for f in range(len(self)):
        #     raw_diff[:, :, f], std = ntl.frame_diff(self._data_raw, self.k, f)
        #     self._data_diff_std.append(std)

        # print('\n--numba time--\n{:.2f} s'.format(time.time() - start))

        # raw_diff = np.zeros(self.shape)
        # self._data_diff_std = []

        print('Processing data for correlation')

        # for f in range(len(self)):
        #     print('\r\t{}/ {}'.format(f + 1, len(self)), end='')
        #     raw_diff[:, :, f] = self.frame(f)
        #     self._data_diff_std.append(np.std(self.frame(f)))
        # print('\n--elapsed time--\n{:.2f} s'.format(time.time() - time0))

        self._data_corr = scipy.signal.correlate(
            raw_diff,
            self.idea3d,
            mode='same'
        ) * 1e5

        self._data_corr_std = np.std(self._data_corr, axis=(0, 1))
        self.type = img_type
        self._range['corr'] = [- np.max(self._data_corr[:, :, self.k * 3:]),
                               np.max(self._data_corr[:, :, self.k * 3:])]

        print('\n--correlation elapsed time--\n{:.2f} s'.format(time.time() - time0))

    def histogram(self):
        frame = self.frame(self._f)
        n = 100
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

    def run_count_nps(self, start, stop, dpx):
        self.np_container = []
        self.nps_in_frame = [[] for i in range(len(self))]

        self.count_nps(start, stop, dpx)

        # self.threshold_value *= -1
        # self.count_nps(start, stop, dpx)
        # self.threshold_value *= -1

    def count_nps(self, start, stop, dpx, ):
        if self.threshold_value > 0:
            color = green
            plot = 'nps_pos'
        else:
            color = red
            plot = 'nps_neg'

        duration = False
        success = True
        minor = False
        bright_defect = False

        # duration = purple
        # success = green
        # minor = black

        def check_np(slice):
            x0_np = np.array([
                slice[0].start,
                slice[1].start,
                slice[2].start
            ])
            x1_np = np.array([
                slice[0].stop,
                slice[1].stop,
                slice[2].stop
            ])

            # print('len {}'.format(5 > x1_np[2] - x0_np[2]))
            # print('len {}'.format(x1_np[2] - x0_np[2]))

            if 5 > x1_np[2] - x0_np[2]:
                return duration

            amx_np = np.argmax(data_corr[slice])
            amx_np = np.unravel_index([amx_np], data_corr[slice].shape)
            mx_np = np.max(data_corr[slice][:, :, amx_np[2]])

            # if mx_np > 2 * self.autocorrelation_max:
            #     return bright_defect

            slice_extended = np.s_[
                             x0_np[0] - dpx: x1_np[0] + dpx,
                             x0_np[1] - dpx: x1_np[1] + dpx,
                             x0_np[2]: x1_np[1]
                             ]

            data_np_labeled = data_labeled[slice_extended]

            for npi in np.unique(data_np_labeled):
                if npi - 1 in blacklist_npid or npi == 0:
                    continue

                amx_i = np.argmax(data_corr[np_slices[npi - 1]])
                amx_i = np.unravel_index([amx_i], data_corr[np_slices[npi - 1]].shape)
                mx_i = np.max(data_corr[np_slices[npi - 1]])

                x0_i = np.array([
                    np_slices[npi - 1][0].start,
                    np_slices[npi - 1][1].start,
                    np_slices[npi - 1][2].start,
                ])

                if np.abs(x0_i[2] + amx_i[2] - x0_np[2] - amx_np[2]) < dpx and mx_i < mx_np:
                    blacklist_npid.append(i - 1)
                    continue
                elif np.linalg.norm(x0_i + np.array(amx_i) - x0_np - np.array(amx_np)) <= 3 * dpx and mx_i > mx_np:
                    return minor

            return success

        time0 = time.time()
        print('\nDetecting NPs')

        self.graphs[plot] = np.array([0 for i in range(len(self))])
        blacklist_npid = []

        data_threshold = np.zeros(self.shape)
        data_corr = np.zeros(self.shape)
        self._data_mask = np.zeros(self.shape)

        for f in range(start + self.k * 2, stop):
            data_threshold[:, :, f] = self.frame(f)

        data_threshold[data_threshold > 0.1] = 1
        data_threshold = data_threshold.astype(np.uint8)

        if self.threshold_value > 0:
            data_corr[:, :, start + self.k * 2:stop] = self._data_corr[:, :, start + self.k * 2:stop]
        else:
            data_corr[:, :, start + self.k * 2:stop] = self._data_corr[:, :, start + self.k * 2:stop] * -1

        data_labeled, _ = ndimage.label(data_threshold, np.ones((3, 3, 3)))
        np_slices = ndimage.find_objects(data_labeled)

        for np_slice in np_slices:
            idnp = len(self.np_container)
            if check_np(np_slice):
                x = (np_slice[0].start + np_slice[0].stop) / 2
                y = (np_slice[1].start + np_slice[1].stop) / 2

                dt = int(np_slice[2].stop - np_slice[2].start)

                nnp = NanoParticle(idnp, np_slice[2].start, [np.array([x, y])] * dt, positive=self.threshold_value > 0)
                nnp.color = color
                # nnp.color = check_np(np_slice)

                if idnp in blacklist_npid:
                    nnp.color = yellow

                self.np_container.append(nnp)

                if self._mask_ommit[int(x), int(y)]:
                    nnp.color = red
                else:
                    self.graphs[plot][np_slice[2].start] += 1
                    self._data_mask[int(x), int(y), int(np_slice[2].start):] = 1

                for i in range(dt):
                    self.nps_in_frame[np_slice[2].start + i].append(idnp)
            # else:
            #     self.np_container.append(None)

        # self._data_mask = data_threshold
        self.show_nps = True

        print('\n--elapsed time--\n{:.2f} s'.format(time.time() - time0))
        return
