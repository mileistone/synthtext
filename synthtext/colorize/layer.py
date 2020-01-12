import numpy as np


class Layer(object):
    def __init__(self, alpha, color):
        # alpha for the whole image:
        assert alpha.ndim == 2
        self.alpha = alpha
        [n, m] = alpha.shape[:2]

        color = np.atleast_1d(np.array(color)).astype('uint8')
        # color for the image:
        if color.ndim == 1:  # constant color for whole layer
            ncol = color.size
            if ncol == 1:  # grayscale layer
                self.color = color * np.ones((n, m, 3), 'uint8')
            if ncol == 3:
                self.color = np.ones((n, m, 3), 'uint8') * color[None, None, :]
        elif color.ndim == 2:  # grayscale image
            self.color = np.repeat(color[:, :, None], repeats=3,
                                   axis=2).copy().astype('uint8')
        elif color.ndim == 3:  #rgb image
            self.color = color.copy().astype('uint8')
        else:
            raise Exception('Data type not understood: color shape %s' % color.shape)
