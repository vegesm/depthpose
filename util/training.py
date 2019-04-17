from itertools import izip_longest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from databases import mupots_3d
import torch

from util.misc import assert_shape
from util.mx_tools import remove_root, remove_openpose_root


def torch_predict(model, input, batch_size=None, device='cuda'):
    """

    :param model: PyTorch Model(nn.Module)
    :param input: a numpy array or a PyTorch dataloader
    :param batch_size: if ``input`` is a numpy array, this is the batch size used for evaluation
    :return:
    """
    model.eval()

    if isinstance(input, np.ndarray):
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input).to(device)), batch_size)
        needs_move = False
    else:
        data_loader = input
        needs_move = True

    result = []
    with torch.no_grad():
        for batch in data_loader:
            if needs_move:
                batch = map(lambda x: x.to(device), batch)
            result.append(model(*batch).cpu().numpy())
        result = np.concatenate(result)

    return result


def preprocess_2d(data, fx, cx, fy, cy):
    assert_shape(data, ("*", 25, 3))
    assert not isinstance(fx, np.ndarray) or len(fx) == len(data)
    assert not isinstance(fy, np.ndarray) or len(fy) == len(data)

    if isinstance(fx, np.ndarray):
        N = len(data)
        shape = [1] * (data.ndim - 1)
        shape[0] = N
        fx = fx.reshape(shape)
        fy = fy.reshape(shape)
        cx = cx.reshape(shape)
        cy = cy.reshape(shape)

    root_ind = 0  # 0-common14, 8-openpose
    data = data[..., mupots_3d.OPENPOSE25_TO_COMMON2D14, :]

    data[..., :, 0] -= cx
    data[..., :, 1] -= cy
    data[..., :, 0] /= fx
    data[..., :, 1] /= fy

    hip2d = data[..., root_ind, :].copy()
    data = remove_openpose_root(data, root_ind)  # (nPoses, 13, 3)

    bad_frames = data[..., 2] < 0.1

    # replace joints having low scores with 1700/focus
    # this is to prevent leaking cx/cy
    if isinstance(fx, np.ndarray):
        fx = np.tile(fx, (1,) + data.shape[1:-1])
        fy = np.tile(fy, (1,) + data.shape[1:-1])
        data[bad_frames, 0] = -1700 / fx[bad_frames]
        data[bad_frames, 1] = -1700 / fy[bad_frames]
    else:
        data[bad_frames, 0] = -1700 / fx
        data[bad_frames, 1] = -1700 / fy

    # stack hip next to pose
    data = data.reshape(data.shape[:-2] + (-1,))  # (nPoses, 13*3)
    data = np.concatenate([data, hip2d], axis=-1)  # (nPoses, 14*3)
    return data.astype('float32')


def preprocess_3d(data, add_hip):
    """

    :param data:
    :param add_hip: True if the absolute coordinates of the hip should be included in the output
    :return:
    """
    assert_shape(data, ("*", 17, 3))

    hip3d = data[..., 14, :].copy()
    hip3d[..., 2] = np.log(hip3d[..., 2])
    data = remove_root(data, 14)  # (nFrames[*nPoses], 16, 3)
    data = data.reshape(data.shape[:-2] + (-1,))  # (nFrames[*nPoses], 16*3)
    if add_hip:
        data = np.concatenate([data, hip3d], axis=-1)  # (nFrames[*nPoses], 17*3)

    return data.astype('float32')


def eval_results(pred3d, gt3d, verbose=True, pck_threshold=150, pctiles=[99]):
    """
    Evaluates the results by printing various statistics. Also returns those results.
    Poses can be represented either in hipless 16 joints or 17 joints with hip format.
    Order is MuPo-TS order in all cases.

    Parameters:
        pred3d: dictionary of predictions in mm, seqname -> (nSample, [16|17], 3)
        gt3d: dictionary of ground truth in mm, seqname -> (nSample, [16|17], 3)
        verbose: if True, a table of the results is printed
        pctiles: list of percentiles of the erros to calculate
    Returns:
        sequence_means, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles
    """

    sequence_means = {}
    sequence_pcks = {}
    sequence_common14_pcks = {}  # pck for the common 14 joints (used by mehta)
    sequence_pctiles = {}
    all_errs = []

    has_hip = pred3d.values()[0].shape[1] == 17  # whether it contains the hip or not

    # assert has_hip, pred3d.values()[0].shape
    seq_hip_errs = {}

    for k in sorted(pred3d.keys()):
        pred = pred3d[k]
        gt = gt3d[k]

        assert pred.shape == gt.shape
        assert (not has_hip and pred.shape[1:] == (16, 3)) or (has_hip and pred.shape[1:] == (17, 3)), \
            "Unexpected shape:" + str(pred.shape)

        errs = np.linalg.norm(pred - gt, axis=2, ord=2)  # (nSample, 16)

        sequence_pctiles[k] = np.percentile(errs, pctiles)
        sequence_pcks[k] = np.mean((errs < pck_threshold).astype(np.float64))
        common14_joints = mupots_3d.MUPOTS17_TO_COMMON2D14 if has_hip else \
            np.array(mupots_3d.MUPOTS17_TO_COMMON2D14[1:]) - 1
        sequence_common14_pcks[k] = np.mean((errs[:, common14_joints] < pck_threshold).astype(np.float64))
        sequence_means[k] = np.mean(errs)

        # Adjusting results for missing hip
        if not has_hip:
            sequence_pcks[k] = sequence_pcks[k] * (16. / 17.) + 1. / 17.
            sequence_common14_pcks[k] = sequence_common14_pcks[k] * (16. / 17.) + 1. / 17.
            sequence_means[k] = sequence_means[k] * float(16. / 17.)

        seq_hip_errs[k] = np.mean(np.linalg.norm((pred - gt)[:, 14, :], axis=1, ord=2))

        all_errs.append(errs)

    all_errs = np.concatenate(all_errs)  # errors per joint, (nPoses, 16)
    joint_means = np.mean(all_errs, axis=0)
    joint_pctiles = np.percentile(all_errs, pctiles, axis=0)

    joint_names = mupots_3d.MUPOTS_NAMES.copy()
    if not has_hip:
        joint_names = np.delete(joint_names, 14)  # remove hip

    num_joints = 17 if has_hip else 16
    assert len(all_errs.shape) == 2 and all_errs.shape[1] == num_joints, all_errs.shape
    assert joint_means.shape == (num_joints,), joint_means.shape
    assert joint_pctiles.shape == (len(pctiles), num_joints), joint_pctiles.shape

    if verbose:
        # Index of the percentile that will be printed. If 99 is calculated it is selected,
        # otherwise the last one
        pctile_ind = len(pctiles) - 1
        if 99 in pctiles:
            pctile_ind = pctiles.index(99)

        print(" ----- Per action and joint errors in millimeter on the validation set ----- ")
        print " %s    %6s     %5s     %6s   \t %22s  %6s     %6s" % ('Sequence', 'Avg', 'PCK', str(pctiles[pctile_ind]) + '%', '',
                                                                     'Avg', str(pctiles[pctile_ind]) + '%')
        for seq, joint_id in izip_longest(sorted(pred3d.keys()), range(num_joints)):
            if seq is not None:
                seq_str = " %-8s:   %6.2f mm   %4.1f%%   %6.2f mm\t " \
                          % (str(seq), sequence_means[seq], sequence_pcks[seq] * 100, sequence_pctiles[seq][pctile_ind])
            else:
                seq_str = " " * 56

            if joint_id is not None:
                print('%s%15s (#%2d):  %6.2f mm   %6.2f mm ' % (seq_str, joint_names[joint_id], joint_id,
                                                                joint_means[joint_id], joint_pctiles[pctile_ind, joint_id]))
            else:
                print(seq_str)

        mean_sequence_err = np.mean(np.asarray(sequence_means.values(), dtype=np.float32))
        mean_sequence_pck = np.mean(np.asarray(sequence_pcks.values(), dtype=np.float32))
        mean_sequence_common14_pck = np.mean(np.asarray(sequence_common14_pcks.values(), dtype=np.float32))
        pctile99 = np.percentile(all_errs, 99)
        print("\n Mean sequence error is %6.2f mm, mean PCK is %4.1f%% (%4.1f%%), total 99%% prctile is %6.2f." %
              (mean_sequence_err, mean_sequence_pck * 100, mean_sequence_common14_pck * 100, pctile99))
        print(" ---------------------------------------------------------------- ")

    return sequence_means, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles, seq_hip_errs
