from typing import List

import cv2
import numpy as np

"""
Здесь может быть любой код, который необходим Вам для расчёта маски переднего плана на фото
"""


def get_foreground_mask(image_path: str) -> List[tuple]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img, 30, 30)
    canny = canny / 255

    for step in range(2, 15):
        for i in range(0, canny.shape[0] - step, step):
            for j in range(0, canny.shape[1] - step, step):
                if np.sum(canny[i:i + step, j: j + step]) > step * step // 2:
                    canny[i:i + step, j: j + step] = 1

    pred_points = np.argwhere(canny)

    return pred_points
