import matplotlib.pyplot as plt
import math
import io
import numpy as np
import cv2


def draw_classifications(image, predictions, targets, class_names, num_pred=1, num_target=1):
    ...


def fig2np(fig: plt.Figure):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def image_grid(images: list, n_columns: int = 1, figsize: tuple = (10, 10)):
    figure = plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(math.ceil(len(images) / n_columns), n_columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
    return fig2np(figure)
