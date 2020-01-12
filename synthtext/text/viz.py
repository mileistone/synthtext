import matplotlib.pyplot as plt

import cv2


def visualize_bb(self, text_arr, bbs):
    ta = text_arr.copy()
    for r in bbs:
        cv2.rectangle(ta, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]),
                     color=128,
                     thickness=1)
    plt.imshow(ta, cmap='gray')
    plt.show()
