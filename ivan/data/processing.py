import numpy as np
from PIL import Image


def filter_label(label: np.ndarray):
    label_filtred = label
    label_filtred = label_filtred // 130 * 255
    label_filtred = label_filtred.reshape(-1, 3)

    for i in range(len(label_filtred)):
        if all(label_filtred[i] == [255, 0, 255]):
            label_filtred[i] = [255, 0, 0]

        if all(label_filtred[i] == [255, 255, 0]):
            label_filtred[i] = [255, 0, 0]

        if all(label_filtred[i] == [0, 255, 255]):
            label_filtred[i] = [0, 255, 0]

        if all(label_filtred[i] == [255, 255, 255]):
            label_filtred[i] = [0, 0, 0]

    label_filtred = label_filtred.reshape(label.shape)

    return label_filtred


# def torch2pil(tensor) -> Image:
