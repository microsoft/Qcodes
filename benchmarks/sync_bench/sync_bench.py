import sys
import os
import pickle
import glob
from statistics import mean

import matplotlib.pyplot as plt

from qcodes.instrument_drivers.Keysight.Infiniium import Infiniium


DATA_FOLDER = 'sync_bench_results'


def get_channel_results(scope: Infiniium):
    scope.trigger_enabled(True)
    scope.trigger_edge_source('CHANnel1')
    scope.timebase_range(50 * 10e-9)

    data = scope.get_current_traces()

    return data


def get_closest_points(p, list_p):
    return sorted(list_p, key=lambda x: abs(p - x))[0]


def filter_outliers(diffs):
    m = mean(diffs)
    l = [i for i in diffs if i < 5 * m]
    return l


def plot_results(channel_results):
    t = channel_results['time']

    v_ch1 = channel_results['ch1']
    v_ch2 = channel_results['ch2']
    v_ch3 = channel_results['ch3']
    v_ch4 = channel_results['ch4']

    plt.figure()
    plt.plot(t, v_ch1, label='channel 1')
    plt.plot(t, v_ch2, label='channel 2')
    plt.plot(t, v_ch3, label='channel 3')
    plt.plot(t, v_ch4, label='channel 4')
    plt.legend(loc='upper left')

    thresh = mean([v_ch1.min(), v_ch1.max()])

    ch1_t_thresh = []
    ch2_t_thresh = []
    ch3_t_thresh = []
    ch4_t_thresh = []

    for i in range(v_ch1.shape[0] - 1):
        if v_ch1[i] < thresh and v_ch1[i+1] > thresh:
            ch1_t_thresh.append(mean([t[i], t[i+1]]))
        if v_ch2[i] < thresh and v_ch2[i + 1] > thresh:
            ch2_t_thresh.append(mean([t[i], t[i+1]]))
        if v_ch3[i] < thresh and v_ch3[i + 1] > thresh:
            ch3_t_thresh.append(mean([t[i], t[i+1]]))
        if v_ch4[i] < thresh and v_ch4[i + 1] > thresh:
            ch4_t_thresh.append(mean([t[i], t[i+1]]))

    plt.plot(ch1_t_thresh, [thresh for i in range(len(ch1_t_thresh))], '.')
    plt.plot(ch2_t_thresh, [thresh for i in range(len(ch2_t_thresh))], '.')
    plt.plot(ch3_t_thresh, [thresh for i in range(len(ch3_t_thresh))], '.')
    plt.plot(ch4_t_thresh, [thresh for i in range(len(ch4_t_thresh))], '.')

    diffs1_2 = []
    diffs1_3 = []
    diffs1_4 = []
    for t in ch1_t_thresh:
        t2 = get_closest_points(t, ch2_t_thresh)
        t3 = get_closest_points(t, ch3_t_thresh)
        t4 = get_closest_points(t, ch4_t_thresh)
        d1_2 = t - t2
        d1_3 = t - t3
        d1_4 = t - t4
        diffs1_2.append(d1_2)
        diffs1_3.append(d1_3)
        diffs1_4.append(d1_4)

    #diffs1_2 = filter_outliers(diffs1_2)
    #diffs1_3 = filter_outliers(diffs1_3)
    #diffs1_4 = filter_outliers(diffs1_4)

    plt.figure()
    plt.hist(diffs1_2, label='diff between ch1 and ch2')
    plt.hist(diffs1_3, label='diff between ch1 and ch3')
    plt.hist(diffs1_4, label='diff between ch1 and ch4')
    plt.legend()
    #print(diffs1_2)
    #print(diffs1_3)
    #print(diffs1_4)
    print(mean(diffs1_2 + diffs1_3 + diffs1_4) * 1e9, ' ns')
    return diffs1_2, diffs1_3, diffs1_4


def save_res(result, awg_channels):
    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    res = {}
    res['ch1'] = awg_channels[0]
    res['ch2'] = awg_channels[1]
    res['ch3'] = awg_channels[2]
    res['ch4'] = awg_channels[3]
    res['data'] = result

    with open(DATA_FOLDER + '/' + "_".join(awg_channels) + '.pickle',
              'wb') as fil:
        pickle.dump(res, fil)


def get_all_results(df=DATA_FOLDER):
    pathname = df + '/*.pickle'
    fnames = glob.glob(pathname)
    objs = []
    for fname in fnames:
        with open(fname, 'rb') as fil:
            objs.append(pickle.load(fil))

    return objs


def get_args():
    args = sys.argv[1:]
    try:
        assert len(args) == 4
    except:
        raise Exception("will only take 4 args")
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)

    scope_address = 'TCPIP0::A-PCSERNO-47466.local::inst0::INSTR'
    scope_channels = ['ch1', 'ch2', 'ch3', 'ch4']
    scope_name = 'msos104A'
    scope = Infiniium(scope_name, scope_address)
    res = get_channel_results(scope)
    # save results
    save_res(res, args)
    plot_results(res)
    plt.show()
    # get results from disk
    #res = get_all_results()
    # plot_results
    #for r in res:
    #    plot_results(r['data'])
    #plt.show()
