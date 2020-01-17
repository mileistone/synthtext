import numpy as np
import pickle

import cv2

from synthtext.config import load_cfg


class FontColor(object):
    """
    """
    def __init__(self, col_file):
        with open(col_file, 'rb') as f:
            self.rgb_colors = pickle.load(f)
        self.ncol = self.rgb_colors.shape[0]

        # convert color-means from RGB to LAB for better nearest neighbour
        # computations:
        self.lab_colors = np.r_[self.rgb_colors[:, 0:3],
                                self.rgb_colors[:, 6:9]].astype('uint8')
        self.lab_colors = np.squeeze(
            cv2.cvtColor(self.lab_colors[None, :, :], cv2.COLOR_RGB2Lab))

    def sample_normal(self, col_mean, col_std):
        """
        sample from a normal distribution centered around COL_MEAN 
        with standard deviation = COL_STD.
        """
        rand_num = np.random.randn()
        col_sample = col_mean + col_std * rand_num
        return np.clip(col_sample, 0, 255).astype('uint8')

    def sample_color(self, bg_mat):
        """
        bg_mat : this is a nxmx3 RGB image.
        
        returns a tuple : (RGB_foreground, RGB_background)
        each of these is a 3-vector.
        """
        bg_orig = bg_mat.copy()
        bg_mat = cv2.cvtColor(bg_mat, cv2.COLOR_RGB2Lab)
        bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]), 3))
        bg_mean = np.mean(bg_mat, axis=0)

        norms = np.linalg.norm(self.lab_colors - bg_mean[None, :], axis=1)
        # choose a random color amongst the top 3 closest matches:
        #nn = np.random.choice(np.argsort(norms)[:3])
        nn = np.argmin(norms)

        ## nearest neighbour color:
        data_col = self.rgb_colors[np.mod(nn, self.ncol), :]

        col1 = self.sample_normal(data_col[:3], data_col[3:6])
        col2 = self.sample_normal(data_col[6:9], data_col[9:12])

        # TODO: debug
        if nn < self.ncol:
            return (col2, col1)
        else:
            # need to swap to make the second color close to the input backgroun color
            return (col1, col2)

    def complement(self, rgb_color):
        """
        return a color which is complementary to the RGB_COLOR.
        """
        col_hsv = np.squeeze(
            cv2.cvtColor(rgb_color[None, None, :], cv2.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128  #uint8 mods to 255
        col_comp = np.squeeze(
            cv2.cvtColor(col_hsv[None, None, :], cv2.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        """
        Returns a color which is "opposite" to both col1 and col2.
        """
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv2.cvtColor(col1[None, None, :], cv2.COLOR_RGB2HSV))
        col2 = np.squeeze(cv2.cvtColor(col2[None, None, :], cv2.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1:
            h1, h2 = h2, h1  #swap
        dh = h2 - h1
        if dh < 127:
            dh = 255 - dh
        col1[0] = h1 + dh / 2
        return np.squeeze(cv2.cvtColor(col1[None, None, :], cv2.COLOR_HSV2RGB))

    # no use
    '''
    def mean_color(self, arr):
        col = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        col = np.reshape(col, (np.prod(col.shape[:2]), 3))
        col = np.mean(col, axis=0).astype('uint8')
        return np.squeeze(cv2.cvtColor(col[None, None, :], cv2.COLOR_HSV2RGB))

    def invert(self, rgb):
        rgb = 127 + rgb
        return rgb

    def change_value(self, col_rgb, v_std=50):
        col = np.squeeze(cv2.cvtColor(col_rgb[None, None, :], cv2.COLOR_RGB2HSV))
        x = col[2]
        vs = np.linspace(0, 1)
        ps = np.abs(vs - x / 255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(
            np.random.choice(vs, p=ps) + 0.1 * np.random.randn(), 0, 1)
        col[2] = 255 * v_rand
        return np.squeeze(cv2.cvtColor(col[None, None, :], cv2.COLOR_HSV2RGB))
    '''
