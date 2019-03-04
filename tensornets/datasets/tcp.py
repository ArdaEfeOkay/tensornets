"""Collection of TcP utils

The codes were adopted from voc.py and coco.py
"""

import os
import glob
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def get_files(data_dir, total_num=None):
    print(data_dir)
    image_names = glob.glob(data_dir + '/*/*.jpg')

    if total_num is not None:
        image_names = image_names[:total_num]

    return image_names

def load_train(data_dir,
               batch_size=64, shuffle=True,
               target_size=416, anchors=5, classes=20,
               total_num=None, dtype=np.float32):

    assert cv2 is not None, '`load_train` requires `cv2`.'

    file_names = get_files(data_dir, total_num)

    dirs = np.zeros(len(file_names), dtype=np.int)

    total_num = len(file_names)
    #for f in file_names:
    #    annotations[f] = reduce(lambda x, y: x + y, annotations[f])

    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    feature_size = [x // 32 for x in target_size]
    
    b = 0
    while True:
        if b == 0:
            if shuffle is True:
                idx = np.random.permutation(total_num)
            else:
                idx = np.arange(total_num)
        if b + batch_size > total_num:
            b = 0
            yield None, None
        else:
            batch_num = batch_size

        imgs = np.zeros((batch_num,) + target_size + (3,), dtype=dtype)
        probs = np.zeros((batch_num, cells, anchors, classes), dtype=dtype)
        confs = np.zeros((batch_num, cells, anchors), dtype=dtype)
        coord = np.zeros((batch_num, cells, anchors, 4), dtype=dtype)
        proid = np.zeros((batch_num, cells, anchors, classes), dtype=dtype)
        prear = np.zeros((batch_num, cells, 4), dtype=dtype)
        areas = np.zeros((batch_num, cells, anchors), dtype=dtype)
        upleft = np.zeros((batch_num, cells, anchors, 2), dtype=dtype)
        botright = np.zeros((batch_num, cells, anchors, 2), dtype=dtype)

        for i in range(batch_num):
            d = data_dir[dirs[idx[b + i]]]
            f = files[idx[b + i]]
            x = cv2.imread("%s/JPEGImages/%s.jpg" % (d, f))
            h, w = x.shape[:2]
            cellx = 1. * w / feature_size[1]
            celly = 1. * h / feature_size[0]

            processed_objs = []
            for obj in annotations[f]:
                bbox = obj['bbox']
                centerx = .5 * (bbox[0] + bbox[2])  # xmin, xmax
                centery = .5 * (bbox[1] + bbox[3])  # ymin, ymax
                cx = centerx / cellx
                cy = centery / celly
                if cx >= feature_size[1] or cy >= feature_size[0]:
                    continue
                processed_objs += [[
                    classidx(obj['name']),
                    cx - np.floor(cx),  # centerx
                    cy - np.floor(cy),  # centery
                    np.sqrt(float(bbox[2] - bbox[0]) / w),
                    np.sqrt(float(bbox[3] - bbox[1]) / h),
                    int(np.floor(cy) * feature_size[1] + np.floor(cx))
                ]]

            # Calculate placeholders' values
            for obj in processed_objs:
                probs[i, obj[5], :, :] = [[0.] * classes] * anchors
                probs[i, obj[5], :, obj[0]] = 1.
                proid[i, obj[5], :, :] = [[1.] * classes] * anchors
                coord[i, obj[5], :, :] = [obj[1:5]] * anchors
                prear[i, obj[5], 0] = obj[1] - obj[3]**2 * .5 * feature_size[1]
                prear[i, obj[5], 1] = obj[2] - obj[4]**2 * .5 * feature_size[0]
                prear[i, obj[5], 2] = obj[1] + obj[3]**2 * .5 * feature_size[1]
                prear[i, obj[5], 3] = obj[2] + obj[4]**2 * .5 * feature_size[0]
                confs[i, obj[5], :] = [1.] * anchors

            # Finalise the placeholders' values
            ul = np.expand_dims(prear[i, :, 0:2], 1)
            br = np.expand_dims(prear[i, :, 2:4], 1)
            wh = br - ul
            area = wh[:, :, 0] * wh[:, :, 1]
            upleft[i, :, :, :] = np.concatenate([ul] * anchors, 1)
            botright[i, :, :, :] = np.concatenate([br] * anchors, 1)
            areas[i, :, :] = np.concatenate([area] * anchors, 1)

            imgs[i] = cv2.resize(x, target_size,
                                 interpolation=cv2.INTER_LINEAR)
        yield imgs, [probs, confs, coord, proid, areas, upleft, botright]
        b += batch_size
