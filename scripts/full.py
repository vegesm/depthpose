import argparse
import os

import torch
from torch.utils.data import DataLoader

from databases import mupots_3d
from databases.loaders import MuPoTsDataset
from model.end2end import End2EndModel
from util.experiments import preprocess_params, Params
from util.mx_tools import combine_pose_and_trans, remove_root
from util.training import torch_predict, eval_results


def get_config():
    p = Params()
    p.batch_size = 30
    p.batch_splits = 1  # each batch is split into this many sub-batches for accumulated training
    p.learning_rate = 1e-4
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


def eval_model(model_path, mupots_path, relative):
    mupots_3d.set_path(mupots_path)
    norm_params = preprocess_params(os.path.dirname(model_path))
    mean3d = norm_params['mean3d']
    std3d = norm_params['std3d']

    p = get_config()
    p.norm_params_path = os.path.dirname(model_path)

    m = End2EndModel(p)
    m.load_state_dict(torch.load(model_path))
    m.cuda()
    m.eval()

    # Load calibration matrix for MuPoTS
    mupots_calibs = mupots_3d.get_calibration_matrices()

    preds = {}
    gts = {}
    for seq in range(1, 21):
        dataset = MuPoTsDataset(seq, mupots_calibs, mean3d, std3d)
        loader = DataLoader(dataset, batch_size=30, pin_memory=True, num_workers=3)

        pred = torch_predict(m, loader)
        pred = combine_pose_and_trans(pred, std3d, mean3d)

        gt3d = dataset.pose3d
        gt3d = combine_pose_and_trans(gt3d, std3d, mean3d)

        if relative:
            pred = remove_root(pred, 14)
            gt3d = remove_root(gt3d.copy(), 14)

        preds[seq] = pred
        gts[seq] = gt3d

    eval_results(preds, gts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative", help="calculate relative error", action="store_true")
    parser.add_argument("--model-path", help="folder where the model is saved", type=str)
    parser.add_argument("--mupots", help="path to the parent MuPoTS directory", type=str)

    args = parser.parse_args()

    eval_model(args.model_path, args.mupots, args.relative)


if __name__ == "__main__":
    main()
