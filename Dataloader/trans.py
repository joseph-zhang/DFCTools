from .params import *

def convert_labels(Lorig, toLasStandard=True):
    """
    Convert the labels from the original CLS file to consecutive integer values starting at 0
    :param Lorig: numpy array containing original labels
    :param toLasStandard: determines if labels are converted from the las standard labels to training labels
        or from training labels to the las standard
    :return: Numpy array containing the converted labels
    """
    L = Lorig.copy()
    if toLasStandard:
        labelMapping = LABEL_MAPPING_TRAIN2LAS
    else:
        labelMapping = LABEL_MAPPING_LAS2TRAIN

    for key, val in labelMapping.items():
        L[Lorig==key] = val
    return L
