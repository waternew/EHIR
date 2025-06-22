#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import toml
import numpy as np
from functools import partial
from easydict import EasyDict as edict
import cv2

import json
import pickle
from copy import deepcopy

from ehir.base import file_name, imread, imsave, Image, transform_img
from ehir.target import process_align_camera
from ehir.utils import run_parallel
from ehir.draw import draw_defects

from ehir.align_edge import get_edge_img
from ehir.draw import verify_align_camera

import click

def process_one(name, root, verify, C):
    name = name.split('_')[0]
    # name = name[:-]
    print('-------------', name)

    root_par = os.getcwd()  

    
    tpl_f = osp.join(root_par, root, f'{name}_temp.jpg')
    img_f = osp.join(root_par, root, f'{name}_test.jpg')

    img = imread(img_f)
    tpl = imread(tpl_f)

    imdic = get_edge_img(
        {'img': tpl},
        with_edge=bool('edge'),
        with_distmap=bool('distmap'),
        distmap_kws={'max_dist': 200}
    )

    tpl_distmap = imdic['distmap']
    cv2.imwrite(osp.join(root_par, root, f'{name}_distmap.jpg'), tpl_distmap)
    
    cfg=C.target.align_camera

    tpl_h, tpl_w = tpl_distmap.shape[:2]

    print('w, h', tpl_w, tpl_h)
    init_bbox = [0, 0, tpl_w, tpl_h]
    ret = process_align_camera(
        img, 
        tpl_distmap, 
        init_bbox,
        tform_tp='projective',       # cfg.tform_tp,
        # tform_tp='affine',       # cfg.tform_tp,
        # tform_tp='similarity',       # cfg.tform_tp,
        msg_prefix=f'{name}: ',
        **cfg.edge_align,
    )

    H = np.asarray(ret['H_21'])
    warped_img = transform_img(img, H, (tpl_w, tpl_h))
    warp_img_f = osp.join(root_par, root, f'{name}_test_warp.jpg')
    cv2.imwrite(warp_img_f, warped_img)

    valid_mask = np.ones((tpl_h, tpl_w, 3), dtype='u1')*255
    # valid_mask = np.array((tpl_h, tpl_w, 3),3)
	# valid_mask = valid_mask * 255
    warped_valid_mask = transform_img(valid_mask, H, (tpl_w, tpl_h))
    warped_valid_mask_f = osp.join(root_par, root, f'{name}_test_valid_mask_warp.jpg')
    print('check ', warped_valid_mask_f)
    cv2.imwrite(warped_valid_mask_f, warped_valid_mask)
    
    # sub = cv2.subtract(tpl,warped_img)
    # sub_inter = cv2.bitwise_and(sub,warped_valid_mask)
    # check = np.concatenate((tpl, img, warped_img, sub,sub_inter), axis=1)  
    # check_f = osp.join(root_par, root, f'{name}_test_warped_check.jpg')
    # print('check ', check_f)
    # cv2.imwrite(check_f, check)


    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if tpl.ndim == 2:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2RGB)
    init_canvas, final_canvas = verify_align_camera(H, img, tpl)
    # init_canvas_2x = cv2.resize(init_canvas,(tpl_w//2, tpl_h//2))
    # final_canvas_2x = cv2.resize(final_canvas,(tpl_w//2, tpl_h//2))
    init_canvas_2x = cv2.resize(init_canvas,(tpl_w, tpl_h))
    final_canvas_2x = cv2.resize(final_canvas,(tpl_w, tpl_h))
    cv2.imwrite(osp.join(root_par, root, name + 'verify_init.jpg'), init_canvas_2x)
    cv2.imwrite(osp.join(root_par, root, name + 'verify_warp.jpg'), final_canvas_2x)



@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--debug', is_flag=True)
@click.option('--jobs', default=1)
@click.option('--verify', default='')
@click.argument('root')                                                                                   # test 0,255-0,1 
def main(cfg, debug, jobs, verify, root):
    C = edict(toml.load(cfg))

    tsks = sorted(set(file_name(i) for i in os.listdir(root) if 'test.jpg' in i))    # pre
    
    tsks = [i for i in tsks]

    worker = partial(
        process_one,
        root=root,
        verify=verify,
        C=C,
    )

    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
