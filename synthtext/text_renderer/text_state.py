import os
import os.path as osp
import pickle
import numpy as np

from pygame import freetype

from synthtext.config import load_cfg


class TextState(object):
    """
    Defines the random state of the font rendering  
    """
    def __init__(self):
        load_cfg(self)

        # get character-frequencies in the English language:
        with open(self.char_freq_fp, 'rb') as fd:
            self.char_freq = pickle.load(fd)

        # get the model to convert from pixel to font pt size:
        with open(self.font_model_fp, 'rb') as fd:
            self.font_model = pickle.load(fd)

        self.fonts = []
        with open(self.font_list_fp, 'r') as fd:
            # get the names of fonts to use:
            for line in fd:
                font_fp = osp.join(self.data_dir, line.strip())
                self.fonts.append(font_fp)

    def init_font(self, fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        char_spacing = fs['char_spacing']
        font.antialiased = True
        font.origin = True
        return font

    def sample_font_state(self):
        """
        Samples from the font state distribution
        """
        font_state = dict()

        idx = int(np.random.randint(0, len(self.fonts)))
        font_state['font'] = self.fonts[idx]

        #idx = int(np.random.randint(0, len(self.capsmode)))
        #font_state['capsmode'] = self.capsmode[idx]

        std = np.random.randn()
        font_state['size'] = self.size[1] * std + self.size[0]

        std = np.random.randn()
        font_state['underline_adjustment'] = max(
            2.0,
            min(
                -2.0, self.underline_adjustment[1] * std +
                self.underline_adjustment[0]))

        std = np.random.rand()
        font_state['strength'] = (self.strength[1] - self.strength[0]) * std + \
                self.strength[0]

        std = np.random.beta(self.kerning[0], self.kerning[1])
        font_state['char_spacing'] = int(self.kerning[3] * std +
                                         self.kerning[2])

        flag = np.random.rand() < self.underline
        font_state['underline'] = flag

        flag = np.random.rand() < self.strong
        font_state['strong'] = flag

        flag = np.random.rand() < self.oblique
        font_state['oblique'] = flag

        #flag = np.random.rand() < self.border
        #font_state['border'] = flag

        #flag = np.random.rand() < self.random_caps
        #font_state['random_caps'] = flag

        #flag = np.random.rand() < self.curved
        #font_state['curved'] = flag

        font = self.init_font(font_state)
        return font

    def get_font_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12  # doesn't matter as we take the RATIO
        chars = ''.join(self.char_freq.keys())
        w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars, size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes, w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:, [3, 4]]
            r = np.abs(sizes[:, 1] / sizes[:, 0])  # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w * r)
            return r_avg
        except:
            return 1.0

    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0] * font_size_px + m[1]  #linear model
