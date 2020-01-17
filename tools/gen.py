import numpy as np
import sys
import h5py
import os, sys, traceback
import os.path as osp
import tarfile
import argparse
from PIL import Image

import wget

sys.path.insert(0, './')

from synthtext.renderer import Renderer
from synthtext.common import set_random_seed
from synthtext.renderer.utils import get_bounding_rect, get_crops, filter_valid


def get_data(db_fp):
    """
    Download the image,depth and segmentation data:
    Returns, the h5 database.
    """
    data_url = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
    if not osp.exists(db_fp):
        print('Downloading data (56 M) from: ' + data_url)
        sys.stdout.flush()
        out_fname = 'data.tar.gz'
        wget.download(data_url, out=out_fname)
        tar = tarfile.open(out_fname)
        tar.extractall()
        tar.close()
        os.remove(out_fname)
        print('Data saved at:' + db_fp)
        sys.stdout.flush()
    # open the h5 file and return:
    return h5py.File(db_fp, 'r')


def render(engine, img, depth, seg, area, label, ninstance, viz=False):
    # re-size uniformly:
    sz = depth.shape[:2][::-1]
    img = np.array(img.resize(sz, Image.ANTIALIAS))
    seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

    res = engine.render(img, depth, seg, area, label, ninstance, viz)
    return res


def get_all_crops(idict):
    img = idict['img']
    min_area_rect = idict['wordBB']
    words = [] 
    for line in idict['txt']:
        segs = line.split() 
        words.extend(segs)
    four_points, extreme_points = get_bounding_rect(min_area_rect)
    crops, valid_flags = get_crops(img, extreme_points)
    valid_crops = filter_valid(crops, valid_flags)
    valid_words = filter_valid(words, valid_flags)
    return valid_crops, valid_words


def save_crops(imgname, res, save_folder):
    ninstance = len(res)
    for ii in range(ninstance):
        crops, words = get_all_crops(res[ii])
        ncrops = len(crops)
        for jj in range(ncrops):
            fp = '%s/%s_ins%d_crop%d_%s.png' % (save_folder, \
                    imgname.replace('.jpg', ''), ii, jj, words[jj])
            im = Image.fromarray(crops[jj])
            im.save(fp)



def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        db['data'].create_dataset(dname, data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        db['data'][dname].attrs['txt'] = res[i]['txt']


def main():
    ## Define some configuration variables:
    viz = False
    nimg = -1  # no. of images to use for generation (-1 to use all available):
    ninstance = 3  # no. of times to use the same image
    secs_per_img = 5  #max time per image in seconds

    # path to the data-file, containing image, depth and segmentation:
    resource_dir = 'data'
    save_folder = 'output'
    in_fp = osp.join(resource_dir, 'dset.h5')
    # url of the data (google-drive public file):
    out_fp = 'results/SynthText.h5'

    ## Processing
    # open databases:
    print('Getting data..')
    in_db = get_data(in_fp)
    print('\t-> Done')

    # open the output h5 file:
    #out_db = h5py.File(out_fp, 'w')
    #out_db.create_group('/data')
    #print('Storing the output in: ' + out_fp)

    # get the names of the image files in the dataset:
    imnames = sorted(in_db['image'].keys())
    nimg = len(imnames) if nimg < 0 else min(len(imnames), nimg)

    engine = Renderer()
    for i in range(nimg):
        imname = imnames[i]
        # get the image:
        img = Image.fromarray(in_db['image'][imname][:])
        # get the pre-computed depth:
        #  there are 2 estimates of depth (represented as 2 "channels")
        #  here we are using the second one (in some cases it might be
        #  useful to use the other one):
        depth = in_db['depth'][imname][:].T
        depth = depth[:, :, 1]
        # get segmentation:
        seg = in_db['seg'][imname][:].astype('float32')
        area = in_db['seg'][imname].attrs['area']
        label = in_db['seg'][imname].attrs['label']

        print('%d of %d' % (i, nimg - 1))
        res = render(engine, img, depth, seg, area, label, ninstance, viz)

        if len(res) > 0:
            save_crops(imname, res, save_folder)
            # non-empty : successful in placing text:
            #add_res_to_db(imname, res, out_db)
    in_db.close()
    #out_db.close()


if __name__ == '__main__':
    set_random_seed()
    main()
