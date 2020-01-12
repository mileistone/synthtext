import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.ndimage as scim
import scipy.ndimage.interpolation as sii
import os
import os.path as osp
import pickle as cp
from PIL import Image

import cv2

from synthtext.synth.poisson_reconstruct import blit_images
from synthtext.config import load_cfg

from .font import FontColor
from .layer import Layer


class Colorize(object):
    def __init__(self):
        # # get a list of background-images:
        # imlist = [osp.join(im_path,f) for f in os.listdir(im_path)]
        # self.bg_list = [p for p in imlist if osp.isfile(p)]
        load_cfg(self)
        self.font_color = FontColor(
            col_file=osp.join(self.data_dir, 'models/colors_new.cp'))


    def drop_shadow(self, alpha, theta, shift, size, op=0.80):
        """
        alpha : alpha layer whose shadow need to be cast
        theta : [0,2pi] -- the shadow direction
        shift : shift in pixels of the shadow
        size  : size of the GaussianBlur filter
        op    : opacity of the shadow (multiplying factor)

        @return : alpha of the shadow layer
                  (it is assumed that the color is black/white)
        """
        if size % 2 == 0:
            size -= 1
            size = max(1, size)
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        [dx, dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = op * sii.shift(
            shadow, shift=[dx, dy], mode='constant', cval=0)
        return shadow.astype('uint8')

    def border(self, alpha, size, kernel_type='RECT'):
        """
        alpha : alpha layer of the text
        size  : size of the kernel
        kernel_type : one of [rect,ellipse,cross]

        @return : alpha layer of the border (color to be added externally).
        """
        kdict = {
            'RECT': cv2.MORPH_RECT,
            'ELLIPSE': cv2.MORPH_ELLIPSE,
            'CROSS': cv2.MORPH_CROSS
        }
        kernel = cv2.getStructuringElement(kdict[kernel_type], (size, size))
        border = cv2.dilate(alpha, kernel, iterations=1)  # - alpha
        return border

    def blend(self, cf, cb, mode='normal'):
        return cf

    def merge_two(self, fore, back, blend_type=None):
        """
        merge two FOREground and BACKground layers.
        ref: https://en.wikipedia.org/wiki/Alpha_compositing
        ref: Chapter 7 (pg. 440 and pg. 444):
             http://partners.adobe.com/public/developer/en/pdf/PDFReference.pdf
        """
        a_f = fore.alpha / 255.0
        a_b = back.alpha / 255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f * a_b
        if blend_type != None:
            c_blend = self.blend(c_f, c_b, blend_type)
            c_r = (((1 - a_f) * a_b)[:, :, None] * c_b +
                   ((1 - a_b) * a_f)[:, :, None] * c_f +
                   (a_f * a_b)[:, :, None] * c_blend)
        else:
            c_r = (((1 - a_f) * a_b)[:, :, None] * c_b + a_f[:, :, None] * c_f)

        return Layer((255 * a_r).astype('uint8'), c_r.astype('uint8'))

    def merge_down(self, layers, blends=None):
        """
        layers  : [l1,l2,...ln] : a list of LAYER objects.
                 l1 is on the top, ln is the bottom-most layer.
        blend   : the type of blend to use. Should be n-1.
                 use None for plain alpha blending.
        Note    : (1) it assumes that all the layers are of the SAME SIZE.
        @return : a single LAYER type object representing the merged-down image
        """
        nlayers = len(layers)
        if nlayers > 1:
            [n, m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2, -nlayers - 1, -1):
                blend = None
                if blends is not None:
                    blend = blends[i + 1]
                    out_layer = self.merge_two(fore=layers[i],
                                               back=out_layer,
                                               blend_type=blend)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.BICUBIC))

    def occlude(self):
        """
        somehow add occlusion to text.
        """
        pass

    def color_border(self, col_text, col_bg):
        """
        Decide on a color for the border:
            - could be the same as text-color but lower/higher 'VALUE' component.
            - could be the same as bg-color but lower/higher 'VALUE'.
            - could be 'mid-way' color b/w text & bg colors.
        """
        choice = np.random.choice(3)

        col_text = cv2.cvtColor(col_text, cv2.COLOR_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]), 3))
        col_text = np.mean(col_text, axis=0).astype('uint8')

        vs = np.linspace(0, 1)

        def get_sample(x):
            ps = np.abs(vs - x / 255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(
                np.random.choice(vs, p=ps) + 0.1 * np.random.randn(), 0, 1)
            return 255 * v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice == 0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0])  # saturation
            col_text = np.squeeze(
                cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))
        elif choice == 1:
            # get the complementary color to text:
            col_text = np.squeeze(
                cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.complement(col_text)
        else:
            # choose a mid-way color:
            col_bg = cv2.cvtColor(col_bg, cv2.COLOR_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]), 3))
            col_bg = np.mean(col_bg, axis=0).astype('uint8')
            col_bg = np.squeeze(
                cv2.cvtColor(col_bg[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = np.squeeze(
                cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.triangle_color(col_text, col_bg)

        # now change the VALUE channel:
        col_text = np.squeeze(
            cv2.cvtColor(col_text[None, None, :], cv2.COLOR_RGB2HSV))
        col_text[2] = get_sample(col_text[2])  # value
        return np.squeeze(
            cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))

    def color_text(self, text_arr, h, bg_arr):
        """
        Decide on a color for the text:
            - could be some other random image.
            - could be a color based on the background.
                this color is sampled from a dictionary built
                from text-word images' colors. The VALUE channel
                is randomized.

            H : minimum height of a character
        """
        bg_col, fg_col, i = 0, 0, 0
        fg_col, bg_col = self.font_color.sample_from_data(bg_arr)
        return Layer(alpha=text_arr, color=fg_col), fg_col, bg_col

    def process(self, text_arr, bg_arr, min_h):
        """
        text_arr : one alpha mask : nxm, uint8
        bg_arr   : background image: nxmx3, uint8
        min_h    : height of the smallest character (px)

        return text_arr blit onto bg_arr.
        """
        # decide on a color for the text:
        l_text, fg_col, bg_col = self.color_text(text_arr, min_h, bg_arr)
        bg_col = np.mean(np.mean(bg_arr, axis=0), axis=0)
        l_bg = Layer(alpha=255 * np.ones_like(text_arr, 'uint8'), color=bg_col)

        l_text.alpha = l_text.alpha * np.clip(0.88 + 0.1 * np.random.randn(),
                                              0.72, 1.0)
        layers = [l_text]
        blends = []

        # add border:
        if np.random.rand() < self.p_border:
            if min_h <= 15: bsz = 1
            elif 15 < min_h < 30: bsz = 3
            else: bsz = 5
            border_a = self.border(l_text.alpha, size=bsz)
            l_border = Layer(border_a,
                             self.color_border(l_text.color, l_bg.color))
            layers.append(l_border)
            blends.append('normal')

        # add shadow:
        if np.random.rand() < self.p_drop_shadow:
            # shadow gaussian size:
            if min_h <= 15: bsz = 1
            elif 15 < min_h < 30: bsz = 3
            else: bsz = 5

            # shadow angle:
            theta = np.pi / 4 * np.random.choice([1, 3, 5, 7
                                                  ]) + 0.5 * np.random.randn()

            # shadow shift:
            if min_h <= 15: shift = 2
            elif 15 < min_h < 30: shift = 7 + np.random.randn()
            else: shift = 15 + 3 * np.random.randn()

            # opacity:
            op = 0.50 + 0.1 * np.random.randn()

            shadow = self.drop_shadow(l_text.alpha, theta, shift, 3 * bsz, op)
            l_shadow = Layer(shadow, 0)
            layers.append(l_shadow)
            blends.append('normal')

        l_bg = Layer(alpha=255 * np.ones_like(text_arr, 'uint8'), color=bg_col)
        layers.append(l_bg)
        blends.append('normal')
        l_normal = self.merge_down(layers, blends)
        # now do poisson image editing:
        l_bg = Layer(alpha=255 * np.ones_like(text_arr, 'uint8'), color=bg_arr)
        l_out = blit_images(l_normal.color, l_bg.color.copy())

        # plt.subplot(1,3,1)
        # plt.imshow(l_normal.color)
        # plt.subplot(1,3,2)
        # plt.imshow(l_bg.color)
        # plt.subplot(1,3,3)
        # plt.imshow(l_out)
        # plt.show()

        if l_out is None:
            # poisson recontruction produced
            # imperceptible text. In this case,
            # just do a normal blend:
            layers[-1] = l_bg
            return self.merge_down(layers, blends).color

        return l_out

    def color(self, bg_arr, text_arr, hs, place_order=None, pad=20):
        """
        Return colorized text image.

        text_arr : list of (n x m) numpy text alpha mask (unit8).
        hs : list of minimum heights (scalar) of characters in each text-array. 
        text_loc : [row,column] : location of text in the canvas.
        canvas_sz : size of canvas image.
        
        return : nxmx3 rgb colorized text-image.
        """
        bg_arr = bg_arr.copy()
        if bg_arr.ndim == 2 or bg_arr.shape[2] == 1:  # grayscale image:
            bg_arr = np.repeat(bg_arr[:, :, None], 3, 2)

        # get the canvas size:
        canvas_sz = np.array(bg_arr.shape[:2])

        # initialize the placement order:
        if place_order is None:
            place_order = np.array(range(len(text_arr)))

        rendered = []
        for i in place_order[::-1]:
            # get the "location" of the text in the image:
            ## this is the minimum x and y coordinates of text:
            loc = np.where(text_arr[i])
            lx, ly = np.min(loc[0]), np.min(loc[1])
            mx, my = np.max(loc[0]), np.max(loc[1])
            l = np.array([lx, ly])
            m = np.array([mx, my]) - l + 1
            text_patch = text_arr[i][l[0]:l[0] + m[0], l[1]:l[1] + m[1]]

            # figure out padding:
            ext = canvas_sz - (l + m)
            num_pad = pad * np.ones(4, dtype='int32')
            num_pad[:2] = np.minimum(num_pad[:2], l)
            num_pad[2:] = np.minimum(num_pad[2:], ext)
            text_patch = np.pad(text_patch,
                                pad_width=((num_pad[0], num_pad[2]),
                                           (num_pad[1], num_pad[3])),
                                mode='constant')
            l -= num_pad[:2]

            w, h = text_patch.shape
            bg = bg_arr[l[0]:l[0] + w, l[1]:l[1] + h, :]

            rdr0 = self.process(text_patch, bg, hs[i])
            rendered.append(rdr0)

            bg_arr[l[0]:l[0] + w, l[1]:l[1] + h, :] = rdr0  #rendered[-1]

            return bg_arr

        return bg_arr
