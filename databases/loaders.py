import numpy as np
from torch.utils.data import Dataset

from databases import mupots_3d
from util.misc import assert_shape
from util.mx_tools import normalize_arr
from util.training import preprocess_2d, preprocess_3d
import cv2


def scale_and_round_pose(poses, img_width, img_height, target_width, target_height, verbose=True):
    """
    Converts the input OpenPose coordinates to integer pixel numbers on a resized image,
    keeping only the Common14 joints.

    Parameters:
        poses: (nFrames, nPoses, 25, 3[x, y, score])
        img_width, img_height: width and height of the original image
        target_width, target_height: width and height of the resized image
    """
    assert_shape(poses, (None, None, 25, 3))

    poses = poses[:, :, mupots_3d.OPENPOSE25_TO_COMMON2D14, :2]
    poses[:, :, :, 0] *= target_width / float(img_width)
    poses[:, :, :, 1] *= target_height / float(img_height)
    poses = np.around(poses)

    if verbose:
        if np.any(poses < -1):
            print "too small value"
        if np.any(poses[:, :, :, 0] > target_width):
            print "too large width"
        if np.any(poses[:, :, :, 1] > target_height):
            print "too large height"

    poses[:, :, :, 0] = np.clip(poses[:, :, :, 0], 0, target_width - 1)
    poses[:, :, :, 1] = np.clip(poses[:, :, :, 1], 0, target_height - 1)
    poses = poses.astype('int64')

    return poses


def img_preprocess(img, depth_width, depth_height):
    """ Preprocesses an image for the MegaDepth subnetwork. """
    img = cv2.resize(img, (depth_width, depth_height)).astype('float32')
    img = img / 255.0
    img = img.transpose((2, 0, 1))

    return img

class MuPoTsDataset(Dataset):
    def __init__(self, seq, calibs, mean3d, std3d):
        """
        seq: sequence number
        calibs: calibration matrices for MuPo-TS, as returned by `mupots_3d.get_calibration_matrices()`
        """
        assert 1 <= seq <= 20, "Invalid seq: "+str(seq)

        gt = mupots_3d.load_gt_annotations(seq)
        op = mupots_3d.load_openpose_predictions(seq)

        pose2d = op['pose']
        pose3d = gt['annot3']

        good_poses = gt['isValidFrame'].squeeze()
        good_poses = np.logical_and(good_poses, op['valid_pose'])
        good_poses = np.logical_and(good_poses, pose2d[:, :, 8, 2] > 0.5)
        self.valid = good_poses

        self.orig_frame = np.tile(np.arange(len(good_poses)).reshape((-1, 1)), (1, good_poses.shape[1]))
        self.orig_pose = np.tile(np.arange(good_poses.shape[1]).reshape((1, -1)), (good_poses.shape[0], 1))

        assert pose2d.shape[:2] == good_poses.shape
        assert pose3d.shape[:2] == good_poses.shape
        assert self.orig_frame.shape == good_poses.shape
        assert self.orig_pose.shape == good_poses.shape
        assert pose2d.shape[2:] == (25, 3)
        assert pose3d.shape[2:] == (17, 3)
        assert good_poses.ndim == 2

        # Keep only those frames where there is at least one pose
        # Having empty frames causes issues with the model forward pass creating zero dimensions
        good_frame = np.any(good_poses, axis=1)
        pose2d = pose2d[good_frame]
        pose3d = pose3d[good_frame]
        self.orig_frame = self.orig_frame[good_frame]
        self.orig_pose = self.orig_pose[good_frame]
        good_poses = good_poses[good_frame]

        self.orig_frame = self.orig_frame[good_poses]
        self.orig_pose = self.orig_pose[good_poses]

        width, height = mupots_3d.image_size(seq)
        depth_width = 512
        depth_height = 512 if seq <= 5 else 288

        self.normed_poses = preprocess_2d(pose2d,  calibs[seq][0, 0], calibs[seq][0, 2],
                                          calibs[seq][1, 1], calibs[seq][1, 2])

        pose2d = scale_and_round_pose(pose2d, width, height, depth_width, depth_height)

        pose3d = preprocess_3d(pose3d, True)
        pose3d = normalize_arr(pose3d, mean3d, std3d)

        assert len(pose2d) == len(pose3d)

        self.seq = seq
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.pose2d = pose2d
        self.pose3d = pose3d[good_poses]
        self.good_poses = good_poses.astype('uint8')

    def __len__(self):
        return len(self.pose2d)

    def __getitem__(self, index):
        img = mupots_3d.get_image(self.seq, index)
        img = img_preprocess(img, self.depth_width, self.depth_height)
        return (img, self.pose2d[index], self.normed_poses[index], self.good_poses[index])
