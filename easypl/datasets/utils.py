import cv2
import numpy as np


def read_image(image_path: str, read_flag=cv2.IMREAD_COLOR, to_rgb=True):
    '''
    Read image from file
    :param image_path: path of file
    :param read_flag: cv2 flag for IMREAD
    :param to_rgb: if True will convert to rgb image
    :return: opencv image or None
    '''
    with open(image_path, 'rb') as f:
        nparr = np.fromstring(f.read(), np.uint8)
        image = cv2.imdecode(nparr, read_flag)
        if to_rgb:
            # TODO: create correct image converting to rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
