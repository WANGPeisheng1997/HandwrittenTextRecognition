import numpy as np
import selectivesearch


def NMS(dets, thresh):
    # dets: nparray[[x1,y1,x2,y2,confidence_score]...]
    # thresh: theshold scaler in [0,1]
    # return the remaining indexes

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        # ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        ious = overlaps / areas[index[1:]] # only calculate the ratio of smaller pictures

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # because index start from 1

    return keep


def selective_search(image, min_size=30):
    # image: skimage
    # return: [x, y, w, h]

    img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=min_size)

    # 创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
    candidates = set()
    for r in regions:
        # 排除重复的候选区
        if r['rect'] in candidates:
            continue
        # 排除太大或者太小候选区域
        if r['size'] < 200 or r['size']>10000:
            continue
        # 排除扭曲的候选区域边框  即只保留近似正方形的
        # x, y, w, h = r['rect']
        # if w / h > 1.2 or h / w > 1.2:
        #     continue
        candidates.add(r['rect'])

    return candidates
