import argparse

import numpy as np
import torch

from databases import mupots_3d
from model.torch_martinez import martinez_net
from util.experiments import preprocess_params, Params, model_state_path_for
from util.mx_tools import normalize_arr, procrustes_translations, relative_pose_to_absolute, remove_root
from util.training import preprocess_2d, torch_predict, eval_results


def get_config():
    p = Params()
    p.batch_size = 256
    p.learning_rate = 0.001
    p.optimiser = "adam"

    p.normclip_enabled = False
    p.dense_size = 1024
    p.n_layers_in_block = 2

    p.activation = 'relu'
    p.dropout = 0.5
    p.residual_enabled = True
    p.batchnorm_enabled = True
    p.n_blocks_in_model = 2

    return p


def get_mupo_poses(seq):
    """
    Loads valid MuPo-TS poses for given sequence. A pose is valid if the hip is visible (has a score>0.5)
    and was detected by OpenPose
    """
    gt = mupots_3d.load_gt_annotations(seq)
    op = mupots_3d.load_openpose_predictions(seq)

    valid = gt['isValidFrame'].squeeze()
    op_valid = op['valid_pose']

    assert valid.dtype == 'bool'
    assert op_valid.dtype == 'bool'
    valid = np.logical_and(valid, op_valid)

    pose2d = op['pose']
    pose3d = gt['annot3']

    assert pose2d.shape[:2] == valid.shape
    assert pose3d.shape[:2] == valid.shape
    assert pose2d.shape[2:] == (25, 3)
    assert pose3d.shape[2:] == (17, 3)
    assert valid.ndim == 2

    pose2d = pose2d[valid]
    pose3d = pose3d[valid]

    good_poses = pose2d[:, 8, 2] > 0.5
    pose2d = pose2d[good_poses]
    pose3d = pose3d[good_poses]

    assert len(pose2d) == len(pose3d)

    return pose2d, pose3d


def eval_baseline(model_folder, mupots_path, relative):
    """
    Evaluates a baseline model using the translation optimisation final step.

    Parameters:
         model_folder: the folder that contains the saved model parameters and normalisation parameters
         mupots_path: path to the MuPo-TS dataset fodler
         relative: if true, relative errors are calculated
    """
    mupots_3d.set_path(mupots_path)
    p = get_config()

    device = 'cuda:0'
    m = martinez_net(p, 42, 48)[1]
    m.to(device)
    m.eval()

    m.load_state_dict(torch.load(model_state_path_for(model_folder), map_location=device))
    norm_params = preprocess_params(model_folder)

    mean2d = norm_params['mean2d']
    mean3d = norm_params['mean3d']
    std2d = norm_params['std2d']
    std3d = norm_params['std3d']

    mupots_calibs = mupots_3d.get_calibration_matrices()

    # Calculate errors on the given model
    preds = {}
    gts = {}
    for seq in range(1, 21):
        pose2d, gt3d = get_mupo_poses(seq)

        processed2d = preprocess_2d(pose2d, mupots_calibs[seq][0, 0], mupots_calibs[seq][0, 2],
                                    mupots_calibs[seq][1, 1], mupots_calibs[seq][1, 2])
        processed2d = normalize_arr(processed2d, mean2d, std2d)

        pred = torch_predict(m, processed2d, 256)
        pred = relative_pose_to_absolute(pred, std3d, mean3d)

        t = procrustes_translations(pred[:, mupots_3d.MUPOTS17_TO_COMMON2D14], pose2d[:, mupots_3d.OPENPOSE25_TO_COMMON2D14],
                                    mupots_calibs[seq][0, 0], mupots_calibs[seq][0, 2], mupots_calibs[seq][1, 2])
        pred = pred + t.reshape((-1, 1, 3))

        if relative:
            pred = remove_root(pred, 14)
            gt3d = remove_root(gt3d.copy(), 14)

        preds[seq] = pred
        gts[seq] = gt3d

    eval_results(preds, gts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative", help="calculate relative error", action="store_true")
    parser.add_argument("--model-path", help="folder where the model is saved or the experiment's id", type=str)
    parser.add_argument("--mupots", help="path to the parent MuPoTS directory", type=str)
    args = parser.parse_args()

    eval_baseline(args.model_path, args.mupots, args.relative)


if __name__ == "__main__":
    main()
