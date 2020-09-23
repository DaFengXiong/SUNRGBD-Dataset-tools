import numpy as np
import cv2
import sys
import os

dataset_root = '/remote-home/source/LiFeng/MyStructureSegDataset/SUNRGBD'

if len(sys.argv) == 2:
    print(sys.argv)
else:
    print('please addpend dataset file index file!')
    exit(0)


def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 37, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [245, 245, 245],
        2: [255, 222, 173 ],
        3: [102, 205, 170],
        4: [205, 133, 63],
        5: [160, 32, 240],
        6: [255, 64, 64],
        7: [139, 69, 19],
        8: [47, 79, 79],
        9: [132, 112, 255 ],
        10: [139, 139, 0 ],
        11: [205, 92, 92 ],
        12: [139, 58, 58 ],
        13: [255, 140, 0 ],
        14: [139, 90, 43],
        15: [208, 32, 144 ],
        16: [255, 240, 245 ],
        17: [255, 248, 220],
        18: [100, 149, 237],
        19: [102, 205, 170],
        20: [205, 133, 63],
        21: [160, 32, 240],
        22: [255, 64, 64],
        23: [0, 128, 128],
        24: [128, 128, 128],
        25: [64, 0, 0],
        26: [192, 0, 0],
        27: [64, 128, 0],
        28: [192, 128, 0],
        29: [64, 0, 128],
        30: [192, 0, 128],
        31: [64, 128, 128],
        32: [192, 128, 128],
        33: [0, 64, 0],
        34: [128, 64, 0],
        35: [0, 192, 0],
        36: [128, 192, 0],
        37: [2, 64, 128],
    }
    print(label2color_dict)
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno



image_label_txt = sys.argv[1]
with open(image_label_txt) as fp:
    #cv2.namedWindow('image and it\'s label')
    for line in fp:
        filenames = line.split()
        image_path = os.path.join(dataset_root, filenames[0])
        label_path = os.path.join(dataset_root, filenames[2])

        print('image_path: ', image_path)
        print('label_path: ', label_path)

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_colored = create_visual_anno(label)
        align_image = np.hstack((image, label_colored))

        #cv2.imshow('image and it\'s label', align_image)
        cv2.imwrite('image and label.png', align_image)
        cv2.waitKey(50000)


