import cv2
from skimage import exposure


def load_cascade(cascade_name):
    """
    :param cascade_name: cascade name
    :return: haar cascade
    """
    if cascade_name=="face":
        cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_alt2.xml')
    elif cascade_name=="left-eye":
        cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_lefteye_2splits.xml')
    elif cascade_name =="mouth":
        cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_mouth.xml')
    else:
        raise Exception
    return cascade

def get_capture(size):
    """
    :param size: frame size
    :return: camera object to capture frames
    """
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    except:
        raise Error("Camera is not found.")
    return cap

def preprocess_frame(img):
    """
    :param img: frame to be preprocessed
    :return: preprocessed frame
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray_gamma = exposure.adjust_gamma(gray, gamma=0.7, gain=1)
    return gray_gamma
