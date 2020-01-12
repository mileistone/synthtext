import numpy as np

data_dir = 'data'


RenderFont = dict(
    # whether to get a single word, paragraph or a line:
    data_dir=data_dir,
    p_text={
        0.0: 'WORD',
        0.0: 'LINE',
        1.0: 'PARA'
    },
    ## TEXT PLACEMENT PARAMETERS:
    f_shrink=0.90,
    max_shrink_trials=5,
    # the minimum number of characters that should fit in a mask
    # to define the maximum font height.
    min_nchar=2,
    # px : 0.6*12 ~ 7px <= actual minimum height
    # 16
    min_font_h=25,
    # 120
    max_font_h=60,
    p_flat=0.10,
    # curved baseline:
    p_curved=0,  # 1.0
)


BaselineState = dict(
    a=[0.50, 0.50],
    p_sgn=0.5,
)


FontState = dict(
    data_dir=data_dir,
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
    border=0.25,
    # don't recapitalize : retain the capitalization of the lexicon
    random_caps=-1,
    curved=0,  #0.2,
    random_kerning=0.2,
    random_kerning_amount=0.1,
    # lower case, upper case, proper noun
    capsmode=[str.lower, str.upper, str.capitalize],
)


TextSource = dict(
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


# probabilities of different text-effects:
Colorize = dict(
    data_dir=data_dir,
    # add bevel effect to text
    p_bevel=0, #0.05,
    # just keep the outline of the text
    p_outline=0, #0.05,
    p_drop_shadow=0, #0.15,
    p_border=0, #0.15,
    # add background-based bump-mapping
    p_displacement=0, #0.30,
    # use an image for coloring text
    p_texture=0,
)


TextRegions = dict(
    minWidth=30, #px
    minHeight=30,   #px
    minAspect=0.3,   # w > 0.3*h
    maxAspect=7, 
    minArea=100,   # number of pix
    pArea=0.60,   # area_obj/area_minrect >= 0.6

    # RANSAC planar fitting params:
    dist_thresh=0.10,   # m
    num_inlier=90, 
    ransac_fit_trials=100, 
    min_z_projection=0.25, 
    minW=20,
)
