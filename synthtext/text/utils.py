import numpy as np


def sample_weighted(p_dict, debug):
    ps = list(p_dict.keys())
    key = np.random.choice(ps)
    if debug:
        key = ps[0]
    return p_dict[key]


def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:, None, None]


def crop_safe(arr, rect, bbs=[], pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2 * pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [
        min(arr.shape[0], rect[0] + rect[2]),
        min(arr.shape[1], rect[1] + rect[3])
    ]
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= v0[0]
            bbs[i, 1] -= v0[1]
        return arr, bbs
    else:
        return arr


