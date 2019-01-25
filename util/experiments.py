import os

from util.misc import load


class Params(object):
    """ A simple dictionary that has its keys as attributes available. """

    def __init__(self):
        pass

    def __str__(self):
        s = ""
        for name in sorted(self.__dict__.keys()):
            s += "%-18s %s\n" % (name + ":", self.__dict__[name])
        return s

    def __repr__(self):
        return self.__str__()


def model_state_path_for(experiment_id):
    return os.path.join(experiment_id, 'model_params.h5')


def preprocess_params(experiment_id):
    return load(os.path.join(experiment_id, 'normalisation.pkl'))
