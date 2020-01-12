"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
import os
import os.path as osp
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import h5py


def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    #plt.hold(True)
    H, W = text_im.shape[:2]

    # plot the character-BB:
    if False:
        for i in range(len(charBB_list)):
            bbs = charBB_list[i]
            ni = bbs.shape[-1]
            for j in range(ni):
                bb = bbs[:, :, j]
                bb = np.c_[bb, bb[:, 0]]
                plt.plot(bb[0, :], bb[1, :], 'r', alpha=alpha / 2)

    # plot the word-BB:
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:, :, i]
        bb = np.c_[bb, bb[:, 0]]
        plt.plot(bb[0, :], bb[1, :], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r', 'g', 'b', 'k']
        for j in range(4):
            plt.scatter(bb[0, j], bb[1, j], color=vcol[j])

    plt.gca().set_xlim([0, W - 1])
    plt.gca().set_ylim([H - 1, 0])
    plt.show(block=False)
    #plt.savefig('xx.png')


def viz_all(db):
    dsets = sorted(db['data'].keys())
    print('Total number of images : ', len(dsets))
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        viz_textbb(rgb, [charBB], wordBB)
        print('Image name        : ', k)
        print('  ** no. of chars : ', charBB.shape[-1])
        print('  ** no. of words : ', wordBB.shape[-1])
        print('  ** text         : ', txt)

        if 'q' in input('next? ("q" to exit): '):
            break


if __name__ == '__main__':
    db_fname = 'results/SynthText.h5'
    db = h5py.File(db_fname, 'r')
    visualize(db)
    db.close()
