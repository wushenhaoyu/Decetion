import numpy as np


def smooth(x, window_len=25, window='blackman'):
    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')
    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
            )
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def smoothen_luminance(predictions, percs):
    percs = np.array(percs, dtype='float32').transpose()
    smooth_percs = []
    for i, perc in enumerate(percs):
        smooth_percs.append(smooth(perc))
    ret = []
    smooth_percs = np.array(smooth_percs).transpose()
    percs = percs.transpose()
    for i, pred in enumerate(predictions):
        smooth_pred = np.interp(pred, percs[i], smooth_percs[i]).astype(
            'float32')
        ret.append(smooth_pred.clip(0, 1))
    return ret
