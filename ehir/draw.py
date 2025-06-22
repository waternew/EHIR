import numpy as np
import cv2
import os
import os.path as osp
from .base import transform_img, imsave


def draw_match_image(img1, img2):
    return (img1.astype('i4') * [1, 0.5, 0] + img2.astype('i4') * [0, 0.5, 1]).astype('u1')

def draw_segmap(segmap, classes):
    imh, imw = segmap.shape
    canvas = np.zeros((imh, imw, 3), dtype='u1')
    for cls_name, v in classes.items():
        canvas[segmap & v.label > 0] = v.color
    return canvas

def verify_align_camera(H_10, src_img, tpl_img):
    H_10 = np.asarray(H_10)
    tpl_h, tpl_w = tpl_img.shape[:2]
    eye = np.eye(3)
    src_warped = transform_img(src_img, np.eye(3), (tpl_w, tpl_h))
    canvas1 = draw_match_image(src_warped, tpl_img)

    warped_img = transform_img(src_img, H_10, (tpl_w, tpl_h))
    canvas2 = draw_match_image(warped_img, tpl_img)
    return canvas1, canvas2

def cal_crop_bbox(bbox_ori, img_wh):
    x0, y0, x1, y1 = bbox_ori
    w, h = img_wh

    x_center = x0 + (x1 - x0) // 2
    y_center = y0 + (y1 - y0) // 2

    bbox_ori_w = x1 - x0
    bbox_ori_h = y1 - y0
    bbox_level = max(bbox_ori_w, bbox_ori_h)

    if bbox_level <= 64:
        crop_y0 = max((y_center-128/2), 0)
        crop_y1 = min((y_center+128/2), h)
        crop_x0 = max((x_center-128/2), 0)
        crop_x1 = min((x_center+128/2), w)
        if crop_x0 == 0:
            crop_x1 = crop_x0 + 128
        if crop_y0 == 0:
            crop_y1 = crop_y0 + 128
        if crop_x1 == w:
            crop_x0 = crop_x1 - 128
        if crop_y1 == h:
            crop_y0 = crop_y1 - 128
    if bbox_level > 64 and bbox_level <= 128:
        crop_y0 = max((y_center-256/2), 0)
        crop_y1 = min((y_center+256/2), h)
        crop_x0 = max((x_center-256/2), 0)
        crop_x1 = min((x_center+256/2), w)
        if crop_x0 == 0:
            crop_x1 = crop_x0 + 256
        if crop_y0 == 0:
            crop_y1 = crop_y0 + 256
        if crop_x1 == w:
            crop_x0 = crop_x1 - 256
        if crop_y1 == h:
            crop_y0 = crop_y1 - 256
    if bbox_level > 128 and bbox_level <= 256:
        crop_y0 = max((y_center-512/2), 0)
        crop_y1 = min((y_center+512/2), h)
        crop_x0 = max((x_center-512/2), 0)
        crop_x1 = min((x_center+512/2), w)
        if crop_x0 == 0:
            crop_x1 = crop_x0 + 512
        if crop_y0 == 0:
            crop_y1 = crop_y0 + 512
        if crop_x1 == w:
            crop_x0 = crop_x1 - 512
        if crop_y1 == h:
            crop_y0 = crop_y1 - 512
    if bbox_level > 256 and bbox_level <= 512:
        crop_y0 = max((y_center-1024/2), 0)
        crop_y1 = min((y_center+1024/2), h)
        crop_x0 = max((x_center-1024/2), 0)
        crop_x1 = min((x_center+1024/2), w)
        if crop_x0 == 0:
            crop_x1 = crop_x0 + 1024
        if crop_y0 == 0:
            crop_y1 = crop_y0 + 1024
        if crop_x1 == w:
            crop_x0 = crop_x1 - 1024
        if crop_y1 == h:
            crop_y0 = crop_y1 - 1024
    if bbox_level > 512 and bbox_level <= 1024:
        crop_y0 = max((y_center-2048/2), 0)
        crop_y1 = min((y_center+2048/2), h)
        crop_x0 = max((x_center-2048/2), 0)
        crop_x1 = min((x_center+2048/2), w)
        if crop_x0 == 0:
            crop_x1 = crop_x0 + 2048
        if crop_y0 == 0:
            crop_y1 = crop_y0 + 2048
        if crop_x1 == w:
            crop_x0 = crop_x1 - 2048
        if crop_y1 == h:
            crop_y0 = crop_y1 - 2048
    if bbox_level > 1024 and bbox_level <= 2048:
        crop_y0 = max((y_center-4096/2), 0)
        crop_y1 = min((y_center+4096/2), h)
        crop_x0 = max((x_center-4096/2), 0)
        crop_x1 = min((x_center+4096/2), w)
        if crop_x0 == 0:
            crop_x1 = crop_x0 + 4096
        if crop_y0 == 0:
            crop_y1 = crop_y0 + 4096
        if crop_x1 == w:
            crop_x0 = crop_x1 - 4096
        if crop_y1 == h:
            crop_y0 = crop_y1 - 4096
    if bbox_level > 2048:
        crop_y0 = max((y_center-4096/2), 0)
        crop_y1 = min((y_center+4096/2), h)
        crop_x0 = max((x_center-4096/2), 0)
        crop_x1 = min((x_center+4096/2), w)
    # else:
    #     print(bbox_ori, bbox_ori_w, bbox_ori_h)

    return int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)

def draw_defects(detect_type, name, img, canvas, d, filter_fn=None, box_fn=None, cfg=None):
    if filter_fn is None:
        filter_fn = lambda x: False

    if box_fn is None:
        box_fn = lambda x: None

    groups = d['groups']
    # # max_label = max(int(i) for i in groups.keys()) + 1
    # max_label = max((int(i) for i in groups.keys()),0) + 1

    objs = d['objects']

    # print('------------------', groups)
    # print('==================', objs)
    
    # print('objs',objs)
    for k, obj in objs.items():
        if filter_fn(obj):
            continue

        color = cfg[obj['type']].color

        if 'mask' in obj:
            # assert o['mask'].dtype == bool, f'{tp} mask type wrong!'
            assert obj['mask'].dtype == bool, f'xxx mask type wrong!'
            x0, y0, x1, y1 = obj['bbox']
            ww, hh = x1 - x0, y1 - y0
            m = obj['mask']
            canvas[y0:y1, x0:x1][m] = color + [255]

        if 'contour' in obj:
            p = obj['contour']
            canvas[p[:, 1], p[:, 0]] = color + [255]
    
    # ori 
    r = 64
    # DeepPCB
    r = 20  #10

    # just draw
    # for gid, v in groups.items():
    #     if int(gid) < 0:
    #         for oid in v['children']:
    #             o = objs[oid]
    #             draw_kws = box_fn(objs[oid])
    #             if not draw_kws:
    #                 continue

    #             x0, y0, x1, y1 = o['bbox']
    #             c = draw_kws['color'] + [255]
    #             cv2.putText(canvas, f'S{o["id"]}', (x0-r, y0-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
    #             cv2.rectangle(canvas, (x0-r, y0-r), (x1+r, y1+r), c, draw_kws['thickness'])

    #     else:
    #         draw_kws = box_fn(v)
    #         if not draw_kws:
    #             continue

    #         x0, y0, x1, y1 = v['bbox']
    #         c = draw_kws['color'] + [255]
    #         cv2.putText(canvas, f'G{gid}', (x0-r, y0-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
    #         cv2.rectangle(canvas, (x0-r, y0-r), (x1+r, y1+r), c, 2)


    # SAVE PATCHES
    h, w = canvas.shape[:2]

    # deviations
    if detect_type == 'deviations':
        spur_dir = 'deviations_patches/spur'
        mousebite_dir = 'deviations_patches/mousebite'
        copper_dir = 'deviations_patches/copper'
        pinhole_dir = 'deviations_patches/pinhole'
        open_dir = 'deviations_patches/open'
        short_dir = 'deviations_patches/short'
        deviations_group_dir = 'deviations_patches/group'
        os.makedirs(spur_dir, exist_ok=True)
        os.makedirs(mousebite_dir, exist_ok=True)
        os.makedirs(copper_dir, exist_ok=True)
        os.makedirs(pinhole_dir, exist_ok=True)
        os.makedirs(open_dir, exist_ok=True)
        os.makedirs(short_dir, exist_ok=True)
        os.makedirs(deviations_group_dir, exist_ok=True)
    # foreigns
    if detect_type == 'foreigns':
        black_dir = 'foreigns_patches/black'
        gray_dir = 'foreigns_patches/gray'
        white_dir = 'foreigns_patches/white'
        foreigns_group_dir = 'foreigns_patches/group'
        os.makedirs(black_dir, exist_ok=True)
        os.makedirs(gray_dir, exist_ok=True)
        os.makedirs(white_dir, exist_ok=True)
        os.makedirs(foreigns_group_dir, exist_ok=True)


    # print('222222222222',groups)
    # raise


    for gid, v in groups.items():
        if int(gid) < 0:
            for oid in v['children']:
                o = objs[oid]
                # print('o',o)
                # raise
                draw_kws = box_fn(objs[oid])
                # print('draw_kws',draw_kws)
                # raise
                if not draw_kws:
                    continue

                x0, y0, x1, y1 = o['bbox']
                c = draw_kws['color'] + [255]
                # cv2.putText(canvas, f'S{o["id"]}-{o["type"]}', (x0-r, y0-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
                # cv2.putText(canvas, f'S{o["type"]}', (x0-r, y0-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (x0-r, y0-r), (x1+r, y1+r), c, draw_kws['thickness'])

                # save patches
                # patch = img[y0-r:y1+r, x0-r:x1+r]
                crop_x0, crop_y0, crop_x1, crop_y1 = cal_crop_bbox([x0, y0, x1, y1], [w, h])
                patch = img[crop_y0:crop_y1, crop_x0:crop_x1]
                # patch = img[max((y0-r), 0):min((y1+r), h), max((x0-r), 0):min((x1+r), w)]
                if detect_type == 'deviations':
                    if o['type'] == 'copper':
                        save_path = osp.join(copper_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                    if o['type'] == 'pinhole':
                        save_path = osp.join(pinhole_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                    if o['type'] == 'spur':
                        save_path = osp.join(spur_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                    if o['type'] == 'mousebite':
                        save_path = osp.join(mousebite_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                    if o['type'] == 'open':
                        raise
                        save_path = osp.join(open_dir, name + f'_S{o["id"]}.jpg')
                    if o['type'] == 'short':
                        raise
                        save_path = osp.join(short_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                if detect_type == 'foreigns':
                    if o['level'] == 'black':
                        save_path = osp.join(black_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                    if o['level'] == 'gray':
                        save_path = osp.join(gray_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)
                    if o['level'] == 'white':
                        save_path = osp.join(white_dir, name + f'_S{o["id"]}.jpg')
                        imsave(save_path, patch)

        else:
            draw_kws = box_fn(v)
            # if not draw_kws:
            #     continue

            # print('gid,v', gid,v)
            # raise
            x0, y0, x1, y1 = v['bbox']
            # c = draw_kws['color'] + [255]     # ori rgb for g
            # c = [255, 188, 0] + [255]
            c = [255, 0, 0] + [255]
            # cv2.putText(canvas, f'G{gid}', (x0-r, y0-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
            # cv2.putText(canvas, f'G{v["type"]}', (x0-r, y0-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
            # cv2.rectangle(canvas, (x0-r, y0-r), (x1+r, y1+r), c, 2)
            cv2.rectangle(canvas, (x0-r, y0-r), (x1+r, y1+r), c, 10)

            # save patches
            # patch = img[y0-r:y1+r, x0-r:x1+r]
            crop_x0, crop_y0, crop_x1, crop_y1 = cal_crop_bbox([x0, y0, x1, y1], [w, h])
            patch = img[crop_y0:crop_y1, crop_x0:crop_x1]
            # patch = img[max((y0-r), 0):min((y1+r), h), max((x0-r), 0):min((x1+r), w)]
            if detect_type == 'deviations':
                # print('v', v)
                # raise
                save_path = osp.join(deviations_group_dir, name + f'_G{gid}.jpg')                  ### ori
                imsave(save_path, patch)
                
                # print('v', v)
                # raise
                if v['type'] == 'open':
                    save_path = osp.join(open_dir, name + f'_G{gid}.jpg')
                    imsave(save_path, patch)
                if v['type'] == 'short':
                    save_path = osp.join(short_dir, name + f'_G{gid}.jpg')
                    imsave(save_path, patch)
                else:
                    save_path = osp.join(deviations_group_dir, name + f'_G{gid}.jpg')
                    imsave(save_path, patch)

            if detect_type == 'foreigns':
                save_path = osp.join(foreigns_group_dir, name + f'_G{gid}.jpg')
                imsave(save_path, patch)