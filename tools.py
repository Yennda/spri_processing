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
