import numpy as np
import os.path as osp

data_dir = 'data'
min_nchar = 2

## renderer
# renderer
Renderer = dict(
    min_char_height=8,  #px
    min_asp_ratio=0.4,  #
    max_text_regions=7,
    max_time=5,
    # re-use each region five times
    num_repeat=5,
)

# text_regions
TextRegions = dict(
    min_width=30,  #px
    min_height=30,  #px
    min_aspect=0.3,  # w > 0.3*h
    max_aspect=7,
    min_area=100,  # number of pix
    p_area=0.60,  # area_obj/area_minrect >= 0.6
    # RANSAC planar fitting params:
    dist_thresh=0.10,  # m
    num_inlier=90,
    ransac_fit_trials=100,
    min_z_projection=0.25,
    min_rectified_w=20,
)

## text_renderer
# text_renderer
TextRenderer = dict(
    min_nchar=min_nchar,
    ## TEXT PLACEMENT PARAMETERS:
    #f_shrink=0.90,
    max_shrink_trials=5,
    # px : 0.6*12 ~ 7px <= actual minimum height
    # 16
    min_font_h=16,  #25,
    # 120
    max_font_h=120,  #60,
    p_flat=0.10,
    # curved baseline:
    p_curved=1.0,  # 1.0
)

# text_state
TextState = dict(
    data_dir=data_dir,
    char_freq_fp=osp.join(data_dir, 'models/char_freq.pkl'),
    font_model_fp=osp.join(data_dir, 'models/font_px2pt.pkl'),
    font_list_fp=osp.join(data_dir, 'fonts/fontlist.txt'),
    # normal dist mean, std
    size=[50, 10],
    underline=0.05,
    strong=0.5,
    oblique=0.2,
    wide=0.5,
    # uniform dist in this interval
    strength=[0.05, 0.1],
    # normal dist mean, std
    underline_adjustment=[1.0, 2.0],
    # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    kerning=[2, 5, 0, 20],
    #border=0.25,
    # don't recapitalize : retain the capitalization of the lexicon
    #random_caps=-1,
    #curved=0,  #0.2,
    # lower case, upper case, proper noun
    #capsmode=[str.lower, str.upper, str.capitalize],
)

# text_state
Curvature = dict(
    a=[0.5, 0.05],  #a=[0.5, 0.05],
    p_sgn=0.5,
)

# corpora
Corpora = dict(
    corpora_fp=osp.join(data_dir, 'newsgroup/alpha_words.txt'),
    # whether to get a single word, paragraph or a line:
    p_text={
        0.0: 'WORD',
        #0.0: 'LINE',
        #1.0: 'PARA'
    },
    # the minimum number of characters that should fit in a mask
    # to define the maximum font height.
    min_nchar=min_nchar,
    # distribution over line/words for LINE/PARA:
    p_line_nline=np.array([0.85, 0.10, 0.05]),
    # normal: (mu, std)
    p_line_nword=[4, 3, 12],
    # [1.7,3.0] # beta: (a, b), max_nline
    p_para_nline=[1.0, 1.0],
    # beta: (a,b), max_nword
    p_para_nword=[1.7, 3.0, 10],
    # bability to center-align a paragraph:
    center_para=0.5,
)

## colorizer
# probabilities of different text-effects:
Colorizer = dict(
    font_fp=osp.join(data_dir, 'models/colors_new.pkl'),
    # add bevel effect to text
    #p_bevel=0, #0.05,
    # just keep the outline of the text
    #p_outline=0, #0.05,
    p_drop_shadow=0.15,
    p_border=0.15,
    # add background-based bump-mapping
    #p_displacement=0, #0.30,
    # use an image for coloring text
    #p_texture=0,
)
