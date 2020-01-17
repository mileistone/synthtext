import numpy as np

import cv2

from synthtext.config import load_cfg
import synthtext.synth as synth


class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.
    """
    def __init__(self):
        load_cfg(self)

    def get_hw(self, pt, return_rot=False):
        pt = pt.copy()
        R = synth.unrotate2d(pt)
        mu = np.median(pt, axis=0)
        pt = (pt - mu[None, :]).dot(R.T) + mu[None, :]
        h, w = np.max(pt, axis=0) - np.min(pt, axis=0)
        if return_rot:
            return h, w, R
        return h, w

    def filter(self, seg, area, label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        good = label[area > self.min_area]
        area = area[area > self.min_area]
        filt, R = [], []
        for idx, i in enumerate(good):
            mask = seg == i
            xs, ys = np.where(mask)

            coords = np.c_[xs, ys].astype('float32')
            rect = cv2.minAreaRect(coords)
            box = np.array(cv2.boxPoints(rect))
            h, w, rot = self.get_hw(box, return_rot=True)

            f = (h > self.min_height and w > self.min_width
                 and self.min_aspect < w / h < self.max_aspect
                 and area[idx] / w * h > self.p_area)
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label': good, 'rot': R, 'area': area[aidx]}
        return filter_info

    def sample_grid_neighbours(self, mask, nsample, step=3):
        '''
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        '''
        if 2 * step >= min(mask.shape[:2]):
            return  #None

        y_m, x_m = np.where(mask)
        mask_idx = np.zeros_like(mask, 'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i], x_m[i]] = i

        xp, xn = np.zeros_like(mask), np.zeros_like(mask)
        yp, yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:, :-2 * step] = mask[:, 2 * step:]
        xn[:, 2 * step:] = mask[:, :-2 * step]
        yp[:-2 * step, :] = mask[2 * step:, :]
        yn[2 * step:, :] = mask[:-2 * step, :]
        valid = mask & xp & xn & yp & yn

        ys, xs = np.where(valid)
        N = len(ys)
        if N == 0:  #no valid pixels in mask:
            return  #None
        nsample = min(nsample, N)
        idx = np.random.choice(N, nsample, replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs, ys = xs[idx], ys[idx]
        s = step
        X = np.transpose(np.c_[xs, xs + s, xs + s, xs - s, xs - s][:, :, None],
                         (1, 2, 0))
        Y = np.transpose(np.c_[ys, ys + s, ys - s, ys + s, ys - s][:, :, None],
                         (1, 2, 0))
        sample_idx = np.concatenate([Y, X], axis=1)
        mask_nn_idx = np.zeros((5, sample_idx.shape[-1]), 'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:, i] = mask_idx[sample_idx[:, :, i][:, 0],
                                         sample_idx[:, :, i][:, 1]]
        return mask_nn_idx

    def filter_depth(self, xyz, seg, regions):
        plane_info = {
            'label': [],
            'coeff': [],
            'support': [],
            'rot': [],
            'area': []
        }
        for idx, l in enumerate(regions['label']):
            mask = seg == l
            pt_sample = self.sample_grid_neighbours(mask,
                                                    self.ransac_fit_trials,
                                                    step=3)
            if pt_sample is None:
                continue  #not enough points for RANSAC
            # get-depths
            pt = xyz[mask]
            plane_model = synth.isplanar(pt, pt_sample, self.dist_thresh,
                                         self.num_inlier,
                                         self.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2]) > self.min_z_projection:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

        return plane_info

    def get_regions(self, xyz, seg, area, label):
        regions = self.filter(seg, area, label)
        # fit plane to text-regions:
        regions = self.filter_depth(xyz, seg, regions)
        return regions

    def filter_rectified(self, mask):
        """
        mask : 1 where 'ON', 0 where 'OFF'
        """
        wx = np.median(np.sum(mask, axis=0))
        wy = np.median(np.sum(mask, axis=1))
        return wx > self.min_rectified_w and wy > self.min_rectified_w


TEXT_REGIONS = TextRegions()
