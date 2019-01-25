import cv2
import pickle
import scipy.io
import os
import numpy as np


def assert_shape(data, shape):
    # assert data.ndim == len(shape), "Unexpected shape: " + str(data.shape)

    star_pos = len(shape)
    for i, j in enumerate(shape):
        if j == "*":
            if star_pos < len(shape):
                raise Exception("Only one asterisk (*) character allowed")

            star_pos = i

    for i in range(0, star_pos):
        if shape[i] is not None:
            assert data.shape[i] == shape[i], "Unexpected shape: " + str(data.shape)

    for i in range(star_pos + 1, len(shape)):
        ind = i - len(shape)
        if shape[ind] is not None:
            assert data.shape[ind] == shape[ind], "Unexpected shape: " + str(data.shape)


def ensuredir(path):
    """
    Creates a folder if it doesn't exists.

    :param path: path to the folder to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load(path):
    """
    Loads the content of a file. It is mainly a convenience function to
    avoid adding the ``open()`` contexts. File type detection is based on extensions.
    Can handle the following types:

    - .pkl: pickles
    - .txt: text files, result is a list of strings ending whitespace removed

    :param path: path to the file
    """
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            return [x.rstrip('\n\r') for x in list(f)]
    elif path.endswith('.mat'):
        return scipy.io.loadmat(path)
    else:
        raise NotImplementedError("Unknown extension: " + os.path.splitext(path)[1])


def save(path, var, varname=None):
    """
    Saves the variable ``var`` to the given path. The file format depends on the file extension.
    List of supported file types:

    - .pkl: pickle
    - .npy: numpy
    - .mat: matlab, needs ``varname`` keyword argument defined
    - .png, .jpg, .gif: pictures
    - .txt: text file, one element per line. ``var`` must be a string or list of strings.
    """
    if path.endswith(".pkl"):
        with open(path, 'wb') as f:
            pickle.dump(var, f, 2)
    elif path.endswith(".mat"):
        assert varname is not None, "when using matlab format the variable name must be defined"
        scipy.io.savemat(path, {varname: var})
    elif path.endswith(".npy"):
        np.save(path, var)
    elif path.endswith((".png", ".jpg")):
        cv2.imwrite(path, var)
    elif path.endswith(".txt"):
        with open(path, 'w') as f:
            if isinstance(var, basestring):
                f.write(var)
            else:
                for i in var:
                    f.write(i)
                    f.write('\n')
    else:
        raise NotImplementedError("Unknown extension: " + os.path.splitext(path)[1])
