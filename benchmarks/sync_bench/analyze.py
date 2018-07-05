import sys
import re
from statistics import mean


import numpy as np
import matplotlib.pyplot as plt


from sync_bench import get_all_results


def analyze_single_run(r):
    channels = [r['ch1'], r['ch2'], r['ch3'], r['ch4']]
    data = r['data']
    sig1 = data['ch1']
    r = {}

    for i in range(1, 4):
        ch = channels[i]
        sig2 = data['ch%s' % (i + 1)]
        lag = get_lag(sig1, sig2)
        lt = data['time'][1] * lag
        print('channel: ', ch)
        print('lag: ', lag)
        print('lagtime: ', lt)
        r[ch] = lt

    return r


def get_lag(sig1, sig2):
    corr = np.correlate(sig1, sig2, mode='full')
    lag = corr.argmax() - (corr.shape[0] - 1) / 2
    return abs(lag)


def merge_dicts(dicts):
    r = {}
    for d in dicts:
        for key in d:
            r[key] = d[key]
    return r


def make_bar_plot(res_dict):
    plt.figure()
    vals = []
    xticks = []
    for key in res_dict:
        val = res_dict[key]
        vals.append(val)
        xticks.append(key)

    xticks = np.array(xticks)
    vals = np.array(vals)

    sindx = np.argsort(xticks)
    xticks = xticks[sindx]
    vals = vals[sindx]

    x = np.arange(len(vals))
    plt.bar(x, vals)
    plt.xticks(x, xticks)
    plt.ylabel('lagtime from ch1 s')
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()

#    plt.show()

def get_mean(res_dict):
    vals = [res_dict[key] for key in res_dict]
    return mean(vals)

def filter_markers(res_dicts):
    res = []
    for res_dict in res_dicts:
        for key in res_dict:
            if re.match('ch[0-9]', key):
                val = res_dict[key]
                if re.match('mk[0-9]_[0-9]', val):
                    res.append(res_dict)
                    break
    return res

if __name__ == '__main__':
    df = sys.argv[1]
    res = get_all_results(df)
#    res = filter_markers(res)
    r = merge_dicts([analyze_single_run(r) for r in res])
#    print(get_mean(r))
    make_bar_plot(r)
    plt.savefig(df + '.png', dpi=120)
    #plt.show()
