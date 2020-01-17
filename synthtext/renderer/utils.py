import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

import cv2

import synthtext.synth as synth
from .text_regions import TEXT_REGIONS


def rescale_frontoparallel(p_fp, box_fp, p_im):
    """
    The fronto-parallel image region is rescaled to bring it in 
    the same approx. size as the target region size.

    p_fp : nx2 coordinates of countour points in the fronto-parallel plane
    box  : 4x2 coordinates of bounding box of p_fp
    p_im : nx2 coordinates of countour in the image

    NOTE : p_fp and p are corresponding, i.e. : p_fp[i] ~ p[i]

    Returns the scale 's' to scale the fronto-parallel points by.
    """
    l1 = np.linalg.norm(box_fp[1, :] - box_fp[0, :])
    l2 = np.linalg.norm(box_fp[1, :] - box_fp[2, :])

    n0 = np.argmin(np.linalg.norm(p_fp - box_fp[0, :][None, :], axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp - box_fp[1, :][None, :], axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp - box_fp[2, :][None, :], axis=1))

    lt1 = np.linalg.norm(p_im[n1, :] - p_im[n0, :])
    lt2 = np.linalg.norm(p_im[n1, :] - p_im[n2, :])

    s = max(lt1 / l1, lt2 / l2)
    if not np.isfinite(s):
        s = 1.0
    return s


def get_text_placement_mask(xyz, mask, plane, pad=2, viz=False):
    """
    Returns a binary mask in which text can be placed.
    Also returns a homography from original image
    to this rectified mask.

    XYZ  : (HxWx3) image xyz coordinates
    MASK : (HxW) : non-zero pixels mark the object mask
    REGION : DICT output of TEXT_REGIONS.get_regions
    PAD : number of pixels to pad the placement-mask by
    """
    contour, hier = cv2.findContours(mask.copy().astype('uint8'),
                                     mode=cv2.RETR_CCOMP,
                                     method=cv2.CHAIN_APPROX_SIMPLE)
    contour = [np.squeeze(c).astype('float') for c in contour]
    #plane = np.array([plane[1],plane[0],plane[2],plane[3]])
    H, W = mask.shape[:2]

    # bring the contour 3d points to fronto-parallel config:
    pts, pts_fp = [], []
    center = np.array([W, H]) / 2
    n_front = np.array([0.0, 0.0, -1.0])
    for i in range(len(contour)):
        cnt_ij = contour[i]
        xyz = synth.DepthCamera.plane2xyz(center, cnt_ij, plane)
        R = synth.rot3d(plane[:3], n_front)
        xyz = xyz.dot(R.T)
        pts_fp.append(xyz[:, :2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    box = np.array(cv2.boxPoints(rect))
    R2d = synth.unrotate2d(box.copy())
    box = np.vstack([box, box[0, :]])  #close the box for visualization

    mu = np.median(pts_fp[0], axis=0)
    pts_tmp = (pts_fp[0] - mu[None, :]).dot(R2d.T) + mu[None, :]
    boxR = (box - mu[None, :]).dot(R2d.T) + mu[None, :]

    # rescale the unrotated 2d points to approximately
    # the same scale as the target region:
    s = rescale_frontoparallel(pts_tmp, boxR, pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s * ((pts_fp[i] - mu[None, :]).dot(R2d.T) + mu[None, :])

    # paint the unrotated contour points:
    minxy = -np.min(boxR, axis=0) + pad // 2
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:, 0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:, 1]).T))

    place_mask = 255 * np.ones(
        (int(np.ceil(COL)) + pad, int(np.ceil(ROW)) + pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i] + minxy[None, :]).astype('int32')
                  for i in range(len(pts_fp))]
    cv2.drawContours(place_mask,
                     pts_fp_i32,
                     -1,
                     0,
                     thickness=cv2.FILLED,
                     lineType=8,
                     hierarchy=hier)

    if not TEXT_REGIONS.filter_rectified((~place_mask).astype('float') / 255):
        return

    # calculate the homography
    H, _ = cv2.findHomography(pts[0].astype('float32').copy(),
                              pts_fp_i32[0].astype('float32').copy(),
                              method=0)

    Hinv, _ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                 pts[0].astype('float32').copy(),
                                 method=0)
    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(mask)
        plt.subplot(1, 2, 2)
        plt.imshow(~place_mask)
        plt.hold(True)
        for i in range(len(pts_fp_i32)):
            plt.scatter(pts_fp_i32[i][:, 0],
                        pts_fp_i32[i][:, 1],
                        edgecolors='none',
                        facecolor='g',
                        alpha=0.5)
        plt.show()

    return place_mask, H, Hinv


def get_bounding_rect(min_area_rect):
    """Get bounding rect of a rotated rect.
     
        Args:
            min_area_rect, np.ndarray, 2x4xN, N is the total of rect
        Return:
            four_points, np.ndarray, 2x4xN, [top_left, top_right, bottom_right, bottom_left]
            extreme_points, [xmin, ymin, xmax, ymax]
    """
    xmin = min_area_rect[0].min(axis=0)
    xmax = min_area_rect[0].max(axis=0)
    ymin = min_area_rect[1].min(axis=0)
    ymax = min_area_rect[1].max(axis=0)

    xx = np.concatenate([xmin[None, :], xmax[None, :], xmax[None, :], xmin[None, :]], axis=0) 
    yy = np.concatenate([ymin[None, :], ymin[None, :], ymax[None, :], ymax[None, :]], axis=0) 
    four_points = np.concatenate([xx[None, :, :], yy[None, :, :]], axis=0)
    extreme_points = [xmin, ymin, xmax, ymax]
    return four_points, extreme_points 


def get_crops(img, rects):
    """Get crops from image based on rects.

    Args:
        img, np.ndarray
        rects, [xmins, ymins, xmaxs, ymaxs], each entry has a length of N (total of rect), (xmin, ymin) is the top left point of a rect, (xmax, ymax) is the bottom right point of a rect
            xmins, np.ndarray
    Return:
        crops, list of crop, which is a np.ndarray
    """
    xmins, ymins, xmaxs, ymaxs = rects
    assert len(xmins) == len(ymins) == len(xmaxs) == len(ymaxs)
    crops_total = len(xmins)
    crops = []
    valid_flags = []
    for idx in range(crops_total):
        xmin = xmins[idx]
        ymin = ymins[idx]
        xmax = xmaxs[idx]
        ymax = ymaxs[idx]
        crop = get_crop(img, xmin, ymin, xmax, ymax)
        crops.append(crop)
        is_valid =  crop is not None
        valid_flags.append(is_valid)
    return crops, valid_flags


def get_crop(img, xmin, ymin, xmax, ymax):
    """Get crop from img based on a rect.

        Args:
            xmins, np.ndarray
            (xmin, ymin) is the top left point of a rect, (xmax, ymax) is the bottom right point of a rect
    """
    rows, cols = img.shape[:2]
    # height, width, channel
    x_start = int(xmin)
    x_end = int(xmax + 1)
    y_start = int(ymin)
    y_end = int(ymax + 1)
    if x_start >= 0 and y_start >= 0 and \
            x_end <= cols and y_end <= rows:
        crop = img[y_start : y_end, x_start : x_end]
        return crop
    else:
        return None


def filter_valid(raw_list, valid_flags):
    valid_list = []
    for idx, is_valid in enumerate(valid_flags):
        if is_valid:
            valid_list.append(raw_list[idx])
    return valid_list
