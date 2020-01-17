import numpy as np
import math
import matplotlib.pyplot as plt

import synthtext.synth as synth


def viz_masks(fignum, rgb, seg, depth, label):
    """
    img,depth,seg are images of the same size.
    visualizes depth masks for top NOBJ objects.
    """
    def mean_seg(rgb, seg, label):
        mim = np.zeros_like(rgb)
        for i in np.unique(seg.flat):
            mask = seg == i
            col = np.mean(rgb[mask, :], axis=0)
            mim[mask, :] = col[None, None, :]
        mim[seg == 0, :] = 0
        return mim

    mim = mean_seg(rgb, seg, label)

    img = rgb.copy()
    for i, idx in enumerate(label):
        mask = seg == idx
        rgb_rand = (255 * np.random.rand(3)).astype('uint8')
        img[mask] = rgb_rand[None, None, :]

    #import scipy
    # scipy.misc.imsave('seg.png', mim)
    # scipy.misc.imsave('depth.png', depth)
    # scipy.misc.imsave('txt.png', rgb)
    # scipy.misc.imsave('reg.png', img)

    plt.close(fignum)
    plt.figure(fignum)
    ims = [rgb, mim, depth, img]
    for i in range(len(ims)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(ims[i])
    plt.show(block=False)


def viz_regions(img, xyz, seg, planes, labels):
    """
    img,depth,seg are images of the same size.
    visualizes depth masks for top NOBJ objects.
    """
    # plot the RGB-D point-cloud:
    synth.plot_xyzrgb(xyz.reshape(-1, 3), img.reshape(-1, 3))

    # plot the RANSAC-planes at the text-regions:
    for i, l in enumerate(labels):
        mask = seg == l
        xyz_region = xyz[mask, :]
        synth.visualize_plane(xyz_region, np.array(planes[i]))

    mym.view(180, 180)
    mym.orientation_axes()
    mym.show(True)


def viz_textbb(fignum, text_im, bb_list, alpha=1.0):
    """
    text_im : image containing text
    bb_list : list of 2x4xn_i boundinb-box matrices
    """
    plt.close(fignum)
    plt.figure(fignum)
    plt.imshow(text_im)
    H, W = text_im.shape[:2]
    for i in range(len(bb_list)):
        bbs = bb_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:, :, j]
            bb = np.c_[bb, bb[:, 0]]
            plt.plot(bb[0, :], bb[1, :], 'r', linewidth=2, alpha=alpha)
    plt.gca().set_xlim([0, W - 1])
    plt.gca().set_ylim([H - 1, 0])
    plt.show(block=False)


def viz_images(fignum, imgs, texts):
    plt.close(fignum)
    plt.figure(fignum)
    imgs_total = len(imgs)
    cols = math.ceil(math.sqrt(imgs_total))
    rows = cols
    
    for idx in range(imgs_total):
        row_idx = idx // cols
        col_idx = idx % cols
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(imgs[idx])
        plt.xlabel(texts[idx])
    plt.show(block=False)
