import os
import glob

import cv2
from util.misc import load
import json
import numpy as np

from util.mx_tools import calibration_matrix

MUPO_TS_PATH = None

OPENPOSE25_NAMES = np.array(['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist',
                             'hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                             'right_eye', 'left_eye', 'right_ear', 'left_ear',
                             'left_bigtoe', 'left_smalltoe', 'left_heel', 'right_bigtoe', 'right_smalltoe', 'right_heel'])

# 17 joint long, the same for 2D and 3D
MUPOTS_NAMES = np.array(["head_top", 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',  # 0-4
                         'left_shoulder', 'left_elbow', 'left_wrist',  # 5-7
                         'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',  # 8-13
                         'hip', 'spine', 'head/nose'])  # 14-16

# The joints that occur in Muco, OpenPose and MuPo-TS
COMMON2D14_NAMES = np.array(['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'thorax',
                             'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'])
OPENPOSE25_TO_COMMON2D14 = [8, 9, 10, 11, 12, 13, 14,  # hip, rleg, lleg
                            1, 5, 6, 7, 2, 3, 4]  # neck/thorax, larm, rarm

# Only use for graphing
OPENPOSE25_TO_MUPOTS = [0, 1, 2, 3, 4, 5, 6, 7,  # head, rarm, larm
                        9, 10, 11, 12, 13, 14,  # rleg, lleg,
                        8, 8, 0]

OPENPOSE_STABLEJOINTS = np.arange(17)

MUPOTS17_TO_COMMON2D14 = [14, 8, 9, 10, 11, 12, 13, 1, 5, 6, 7, 2, 3, 4]


def set_path(path):
    global MUPO_TS_PATH
    MUPO_TS_PATH = path


def _decode_sequence(sequence):
    assert isinstance(sequence, int) or isinstance(sequence, basestring), "sequence must be an int or string"

    if isinstance(sequence, int):
        assert 1 <= sequence <= 20, "sequence id must be between 1 and 20"
        sequence = "TS" + str(sequence)

    return sequence


def get_frame_files(sequence):
    """
    Returns the list of jpg files for a given video sequence.

    Parameters:
        sequence: either an in beween 1 and 20 or a string in the form TSx.
    """
    sequence = _decode_sequence(sequence)

    folder = os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence)
    assert os.path.isdir(folder), "Could not find " + folder

    return sorted(glob.glob(folder + '/*.jpg'))


def _concat_raw_gt(gt, field, dtype):
    """ Concatenates gt annotations coming from the annot.mat files. """
    data = np.empty(gt.shape + gt[0, 0][field][0, 0].T.shape, dtype=dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = gt[i, j][field][0, 0].T

    return data


def load_gt_annotations(sequence):
    """
    Returns a dict. Has the following keys:

    - annot2: (nFrames, nPoses, 17, 2), float32
    - annot3: (nFrames, nPoses, 17, 3), float32
    - univ_annot3: (nFrames, nPoses, 17, 3), float32
    - isValidFrame: (nFrames, nPoses), bool
    - occlusions: (nFrames, nPoses, 17), bool
    """
    data = load_raw_gt_annotations(sequence)
    occlusions = load_raw_gt_occlusions(sequence)

    occ_out = np.empty(occlusions.shape + (17,), dtype='bool')
    for i in range(occlusions.shape[0]):
        for j in range(occlusions.shape[1]):
            occ_out[i, j] = occlusions[i, j][0]

    result = {'annot2': _concat_raw_gt(data, 'annot2', 'float32'),
              'annot3': _concat_raw_gt(data, 'annot3', 'float32'),
              'univ_annot3': _concat_raw_gt(data, 'univ_annot3', 'float32'),
              'isValidFrame': _concat_raw_gt(data, 'isValidFrame', 'bool').squeeze(),
              'occlusions': occ_out}

    return result


def load_raw_gt_annotations(sequence):
    sequence = _decode_sequence(sequence)
    return load(os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence, 'annot.mat'))['annotations']


def load_raw_gt_occlusions(sequence):
    sequence = _decode_sequence(sequence)
    return load(os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence, 'occlusion.mat'))['occlusion_labels']


def load_openpose_predictions(sequence):
    sequence = _decode_sequence(sequence)
    return load(os.path.join(MUPO_TS_PATH, "openpose", sequence + '.pkl'))


def all_sequences():
    """ Returns every available sequence's name. """
    folder = os.path.join(MUPO_TS_PATH, "MultiPersonTestSet")
    assert os.path.isdir(folder), "Could not find " + folder

    return sorted(os.listdir(folder))



def _sequence2num(sequence):
    """ Returns the input sequence as a number. """
    if isinstance(sequence, basestring):
        sequence = int(sequence[2])

    return sequence


def get_fps(sequence):
    sequence = _sequence2num(sequence)
    return 30 if sequence <= 5 else 60


def get_image(sequence, frameind):
    """

    :param sequence: sequence id
    :param frameind: zero based index of the image
    :return:
    """
    sequence = _decode_sequence(sequence)
    img = cv2.imread(os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence, 'img_%06d.jpg' % frameind))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_size(sequence):
    """
    Returns:
        width, height
    """
    sequence = _sequence2num(sequence)
    return (2048, 2048) if sequence <= 5 else (1920, 1080)


def get_calibration_matrices():
    calibs = {}
    for seq in range(1, 21):
        annot = load_gt_annotations(seq)

        valid = np.logical_and(annot['isValidFrame'][:, :, np.newaxis], annot['occlusions'])

        pose2d = annot['annot2'][valid]
        pose3d = annot['annot3'][valid]

        calibs[seq], reproj, resx, resy, _, _ = calibration_matrix(pose2d, pose3d)

    return calibs
