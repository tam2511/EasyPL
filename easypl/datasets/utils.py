import cv2
import numpy as np


def read_image(image_path: str):
    '''
    Read image from file
    :param image_path: path of file
    :return: opencv image or None
    '''
    with open(image_path, 'rb') as f:
        nparr = np.fromstring(f.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image