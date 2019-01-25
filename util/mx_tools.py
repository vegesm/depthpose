import numpy as np

from util.misc import assert_shape


def project_points(calib, points3d):
    """
    Projects 3D points using a calibration matrix.

    Parameters:
        points3d: ndarray of shape (nPoints, 3)
    """
    assert points3d.ndim == 2 and points3d.shape[1] == 3

    p = np.empty((len(points3d), 2))
    p[:, 0] = points3d[:, 0] / points3d[:, 2] * calib[0, 0] + calib[0, 2]
    p[:, 1] = points3d[:, 1] / points3d[:, 2] * calib[1, 1] + calib[1, 2]

    return p


def calibration_matrix(points2d, points3d):
    """
    Calculates camera calibration matrix (no distortion) from 3D points and their projection.
    Only works if all points are away from the camera, eg all z coordinates>0.

    Returns:
        calib, reprojection error, x residuals, y residuals, x singular values, y singular values
    """
    assert points2d.ndim == 2 and points2d.shape[1] == 2
    assert points3d.ndim == 2 and points3d.shape[1] == 3

    A = np.column_stack([points3d[:, 0] / points3d[:, 2], np.ones(len(points3d))])
    px, resx, _, sx = np.linalg.lstsq(A, points2d[:, 0], rcond=None)

    A = np.column_stack([points3d[:, 1] / points3d[:, 2], np.ones(len(points3d))])
    py, resy, _, sy = np.linalg.lstsq(A, points2d[:, 1], rcond=None)

    calib = np.eye(3)
    calib[0, 0] = px[0]
    calib[1, 1] = py[0]
    calib[0, 2] = px[1]
    calib[1, 2] = py[1]

    # Calculate mean reprojection error
    # p = np.empty((len(points3d), 2))
    # p[:, 0] = points3d[:, 0] / points3d[:, 2] * calib[0, 0] + calib[0, 2]
    # p[:, 1] = points3d[:, 1] / points3d[:, 2] * calib[1, 1] + calib[1, 2]
    p = project_points(calib, points3d)
    reproj = np.mean(np.abs(points2d - p))

    return calib, reproj, resx, resy, sx, sy


def procrustes_depth(coords2d, coords3d, focal_length, verbose=False, approximate=False):
    """
    Absolute depth prediction based on Mehta et al. (https://arxiv.org/pdf/1611.09813.pdf) .

    Parameters:
        pose3d: ndarray(nJoints, 3[x,y,z), the relative 3D coordinates
        pose2d: ndarray(nJoints, 3[x,y]), the 2D coordinates, relative to the centerpoint of the camera
        focal_length: scalar, focus distance
        approximate: if True, uses the formula in https://arxiv.org/pdf/1611.09813.pdf, otherwise uses the solution without
                     any approximation. The latter gives better results.
    Returns:
        ndarray(3,), the optimal translation vector
    """
    assert len(coords2d) == len(coords3d)
    assert coords2d.ndim == 2
    assert coords3d.ndim == 2
    assert coords3d.shape[1] == 3

    coords3d = coords3d[:, :2]
    mean2d = np.mean(coords2d, axis=0, keepdims=True)
    mean3d = np.mean(coords3d, axis=0, keepdims=True)

    assert_shape(mean2d, (1, 2))
    assert_shape(mean3d, (1, 2))

    # orig method using an approximation (does not provide any visible speedup)
    if approximate:
        numer = np.sqrt(np.sum(np.square(coords3d - mean3d)))
        denom = np.sqrt(np.sum(np.square(coords2d - mean2d)))
    else:
        # no cos approximation
        numer = np.sum(np.square(coords3d - mean3d))
        denom = np.trace(np.dot((coords2d - mean2d), (coords3d - mean3d).T))

    if verbose:
        print "proc: %f / %f" % (numer, denom)
    return numer / denom * np.array([mean2d[0, 0], mean2d[0, 1], focal_length]) - np.array([mean3d[0, 0], mean3d[0, 1], 0])


def procrustes_translations(pose3d, pose2d, focus, cx, cy):
    """
    Calculates the translations for a set of poses using optimal reprojection loss.

    Parameters:
        pose3d: ndarray(nPoses, nJoints, 3), the relative 3D coordinates
        pose2d: ndarray(nPoses, nJoints, 3[x,y,score]), the 2D coordinates and OpenPose score.
        focus: scalar, focus distance
        cx,cy: scalars, camera principal points
    Returns:
        ndarray(nPoses, 3), the optimal translation vectors
    """
    assert pose3d.shape == pose2d.shape
    assert pose2d.ndim == 3, pose2d.ndim

    pose2d = pose2d.copy()
    pose3d = pose3d.copy()

    t = np.zeros((len(pose3d), 3))

    for i in range(len(pose3d)):
        # only use visible joints
        good = pose2d[i][:, 2] > 0.2
        # assert np.all(np.sum(good) >= 4)

        pose2d[i][:, 0] -= cx
        pose2d[i][:, 1] -= cy

        t[i] = procrustes_depth(pose2d[i][:, :2][good, :], pose3d[i][:][good, :], focus, approximate=False)

    return t


def normalize_arr(data, mean, std):
    """ Normalizes `data` by removing the mean and std parameters. """
    assert mean.shape == std.shape
    assert data.shape[-1:] == mean.shape

    return (data - mean) / std


def insert_zero_joint(data, ind=14):
    """ Adds back a hip with zeros in a hip-relative MuPo-TS pose. """
    assert data.shape[-2] == 16 and data.ndim >= 2

    shape = list(data.shape)
    shape[-2] = 17
    result = np.zeros(shape, dtype=data.dtype)
    result[..., :ind, :] = data[..., :ind, :]
    result[..., ind + 1:, :] = data[..., ind:, :]

    return result


def remove_root(data, root_ind):
    """
    Removes  a joint from the dataset by moving to the origin and removing it from the array.

    :param data: (nPoses, nJoints, 3) array
    :param root_ind: index of the joint to be removed
    :return: (nPoses, nJoints-1, 3) array
    """
    assert data.ndim >= 3 and data.shape[-1] in (2, 3)

    roots = data[..., [root_ind], :]
    # roots = roots.reshape((len(roots), 1, 3))
    data = data - roots
    data = np.delete(data, root_ind, axis=-2)

    return data


def remove_openpose_root(data, root_ind):
    """
    Removes  a joint from an openpose dataset by moving to the origin and removing it from the array.

    :param data: (nPoses, nJoints, 3) array
    :param root_ind: index of the joint to be removed
    :return: (nPoses, nJoints-1, 3) array
    """
    assert data.ndim >= 3 and data.shape[-1] == 3

    roots = data[..., [root_ind], :2]  # ndarray(...,1,2)
    data[..., :2] = data[..., :2] - roots
    data = np.delete(data, root_ind, axis=-2)

    return data


# Normalize - first centrize with image data then normalize for
def img_normalize(data, width, height, scale_only=False):
    """
    Normalizes a set of points by changing the coordinate system centered to the image
    and scaling back by the image size. If ``scale_only`` is False than the coordinate system
    recentering is not done.

    :param data: (nFrames, nPoses, [x,y,score])
    :return: ndarray(nFrames, nPoses, [x,y,score])
    """
    assert data.shape[2] == 3 and data.ndim == 3

    if scale_only:
        data[:, :, 0] = data[:, :, 0] / width * 2
        data[:, :, 1] = data[:, :, 1] / height * 2
    else:
        data[:, :, 0] = (data[:, :, 0] - width / 2) / width * 2
        data[:, :, 1] = (data[:, :, 1] - height / 2) / height * 2

    return data


def relative_pose_to_absolute(data3d, std3d, mean3d):
    """ 3D result postprocess: the first 16*3 values are relative poses, the last one is the hip. """
    assert data3d.ndim == 2 and data3d.shape[1] == 48

    data3d = data3d.copy() * std3d + mean3d
    data3d = data3d.reshape((len(data3d), 16, 3))

    data3d = insert_zero_joint(data3d, ind=14)

    return data3d


def combine_pose_and_trans(data3d, std3d, mean3d):
    """ 3D result postprocess: the first 16*3 values are relative poses, the last one is the hip. """
    assert data3d.ndim == 2 and data3d.shape[1] == 51

    data3d = data3d * std3d + mean3d
    hip = data3d[:, -3:]
    rel_pose = data3d[:, :-3].reshape((len(data3d), 16, 3))

    hip[:, 2] = np.exp(hip[:, 2])

    root_ind = 14
    rel_pose += hip[:, np.newaxis, :]

    result = np.zeros((len(data3d), 17, 3), dtype='float32')
    result[:, :root_ind, :] = rel_pose[:, :root_ind, :]
    result[:, root_ind, :] = hip
    result[:, root_ind + 1:, :] = rel_pose[:, root_ind:, :]

    return result
