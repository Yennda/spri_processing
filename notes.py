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

def make_correlation(self):
    time0 = time.time()
    if self.idea3d is None:
        raise Exception('No selected NP patter for the file {}'.format(self.file))

    img_type = self.type
    self.type = 'diff'

    raw_diff = np.zeros(self.shape)
    self._data_diff_std = []

    print('Processing data for correlation')

    for f in range(len(self)):
        print('\r\t{}/ {}'.format(f + 1, len(self)), end='')
        raw_diff[:, :, f] = self.frame(f)
        self._data_diff_std.append(np.std(self.frame(f)))

    self._data_corr = scipy.signal.correlate(
        raw_diff,
        self.idea3d,
        mode='same'
    ) * 1e5

    self._data_corr_std = np.std(self._data_corr, axis=(0, 1))
    self.type = img_type
    self._range['corr'] = [- np.max(self._data_corr[:, :, self.k * 3:]),
                           np.max(self._data_corr[:, :, self.k * 3:])]

    print('\n--elapsed time--\n{:.2f} s'.format(time.time() - time0))