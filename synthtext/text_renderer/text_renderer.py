import numpy as np
import scipy.io as sio
import os
import os.path as osp
import random
import scipy.signal as ssig
import math
from PIL import Image

import pygame
import pygame.locals
import cv2

from synthtext.config import load_cfg

from .text_state import TextState
from .curvature import Curvature
from .corpora import Corpora
from .utils import move_bb, crop_safe
from .viz import visualize_bb


class TextRenderer(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """
    def __init__(self):
        # distribution over the type of text:
        load_cfg(self)

        self.curvature = Curvature()

        # text-source : gets english text:
        self.corpora = Corpora()

        # get font-state object:
        self.text_state = TextState()

        pygame.init()

    def render_multiline(self, font, text):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1

        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0 * line_bounds.width),
                 round(1.25 * line_spacing * len(lines)))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect('O')
        x, y = 0, 0
        for l in lines:
            x = 0  # carriage-return
            y += line_spacing  # line-feed

            for ch in l:  # render each character
                if ch.isspace():  # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x, y), ch)
                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        words = ' '.join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf),
                                  rect_union,
                                  bbs,
                                  pad=5)
        surf_arr = surf_arr.swapaxes(0, 1)
        #self.visualize_bb(surf_arr,bbs)
        return surf_arr, words, bbs, False

    def render_curved(self, font, word_text):
        """
        use curved baseline for rendering word
        """
        wl = len(word_text)
        isword = len(word_text.split()) == 1

        # do curved iff, the length of the word <= 10
        rand_num = np.random.rand()
        if not isword or wl > 10 or rand_num > self.p_curved:
            #print('no curve')
            return self.render_multiline(font, word_text)
        #print('curve')

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0 * lbound.width), round(3 * lspace))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl // 2
        BS = self.curvature.sample_curvature()
        curve = [BS['curve'](i - mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl - 1)
        rots = [
            -int(
                math.degrees(
                    math.atan(BS['diff'](i - mid_idx) / (font.size / 2))))
            for i in range(wl)
        ]

        bbs = []
        # place middle char
        rect = font.get_rect(word_text[mid_idx])
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery += curve[mid_idx]
        ch_bounds = font.render_to(surf,
                                   rect,
                                   word_text[mid_idx],
                                   rotation=rots[mid_idx])
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        # render chars to the left and right:
        last_rect = rect
        ch_idx = []
        for i in range(wl):
            #skip the middle character
            if i == mid_idx:
                bbs.append(mid_ch_bb)
                ch_idx.append(i)
                continue

            if i < mid_idx:  #left-chars
                i = mid_idx - 1 - i
            elif i == mid_idx + 1:  #right-chars begin
                last_rect = rect

            ch_idx.append(i)
            ch = word_text[i]

            newrect = font.get_rect(ch)
            newrect.y = last_rect.y
            if i > mid_idx:
                newrect.topleft = (last_rect.topright[0] + 2,
                                   newrect.topleft[1])
            else:
                newrect.topright = (last_rect.topleft[0] - 2,
                                    newrect.topleft[1])
            newrect.centery = max(
                newrect.height,
                min(fsize[1] - newrect.height, newrect.centery + curve[i]))
            try:
                bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, ch)
            bbrect.x = newrect.x + bbrect.x
            bbrect.y = newrect.y - bbrect.y
            bbs.append(np.array(bbrect))
            last_rect = newrect

        # correct the bounding-box order:
        bbs_sequence_order = [None for i in ch_idx]
        for idx, i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf),
                                  rect_union,
                                  bbs,
                                  pad=5)
        surf_arr = surf_arr.swapaxes(0, 1)
        return surf_arr, word_text, bbs, True

    def get_nline_nchar(self, mask_size, font_height, font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H, W = mask_size
        nline = int(np.ceil(H / (2 * font_height)))
        nchar = int(np.floor(W / font_width))
        return nline, nchar

    def place_text(self, text_arrs, back_arr, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(np.float), 0, 255)
            ba[ba > 127] = 1e8
            intersect = ssig.fftconvolve(ba, ta[::-1, ::-1], mode='valid')
            safemask = intersect < 1e8

            if not np.any(safemask):  # no collision-free position:
                #warn("COLLISION!!!")
                return back_arr, locs[:i], bbs[:i], order[:i]

            minloc = np.transpose(np.nonzero(safemask))
            rand_num = np.random.choice(minloc.shape[0])
            loc = minloc[rand_num, :]
            locs[i] = loc

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i], loc[::-1])

            # blit the text onto the canvas
            w, h = text_arrs[i].shape
            out_arr[loc[0]:loc[0] + w, loc[1]:loc[1] + h] += text_arrs[i]

        return out_arr, locs, bbs, order

    def robust_HW(self, mask):
        m = mask.copy()
        m = (~mask).astype('float') / 255
        rH = np.median(np.sum(m, axis=0))
        rW = np.median(np.sum(m, axis=1))
        return rH, rW

    def sample_font_height_px(self, h_min, h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0, 2.0)
        rnd = rnd

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range * rnd)
        return f_h

    def bb_xywh2coords(self, bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n, _ = bbs.shape
        coords = np.zeros((2, 4, n))
        for i in range(n):
            coords[:, :, i] = bbs[i, :2][:, None]
            coords[0, 1, i] += bbs[i, 2]
            coords[:, 2, i] += bbs[i, 2:4]
            coords[1, 3, i] += bbs[i, 3]
        return coords

    # main method
    def render_text(self, mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        font = self.text_state.sample_font_state()
        #H,W = mask.shape
        H, W = self.robust_HW(mask)
        f_asp = self.text_state.get_font_aspect_ratio(font)

        # find the maximum height in pixels:
        max_font_h = min(0.9 * H, (1 / f_asp) * W / (self.min_nchar + 1))
        max_font_h = min(max_font_h, self.max_font_h)
        if max_font_h < self.min_font_h:  # not possible to place any text here
            return  #None

        # let's just place one text-instance for now
        ## TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            # sample a random font-height:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            #print "font-height : %.2f (min: %.2f, max: %.2f)"%(f_h_px, self.min_font_h,max_font_h)
            # convert from pixel-height to font-point-size:
            f_h = self.text_state.get_font_size(font, f_h_px)

            # update for the loop
            max_font_h = f_h_px
            i += 1

            font.size = f_h  # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline, nchar = self.get_nline_nchar(mask.shape[:2], f_h, f_h * f_asp)
            #print "  > nline = %d, nchar = %d"%(nline, nchar)

            assert nline >= 1 and nchar >= self.min_nchar

            text = self.corpora.sample_text(nline, nchar)
            #print(text)
            if len(text) == 0 or np.any([len(line) == 0 for line in text]):
                continue
            #print colorize(Color.GREEN, text)

            # render the text:
            txt_arr, txt, bb, curve_flag = self.render_curved(font, text)
            bb = self.bb_xywh2coords(bb)

            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[mask.shape[:2]]):
                #warn("text-array is bigger than mask")
                continue

            # position the text within the mask:
            text_mask, loc, bb, _ = self.place_text([txt_arr], mask, [bb])
            if len(loc) > 0:  #successful in placing the text collision-free:
                return text_mask, loc[0], bb[0], text, curve_flag
        return  #None
