import torch
import numpy as np

from model.HG_model import HGModel
from model.torch_martinez import martinez_net
from util.experiments import model_state_path_for, preprocess_params


class End2EndModel(torch.nn.Module):
    def __init__(self, params):
        """
        params.trunk_finetune controls which layers are trained. top_conv means the very last convolutional layer,
        hg_top means the last convolutional layer and the last layer in the hourglass model together with the
        last block in the residual path.
        """
        super(End2EndModel, self).__init__()

        self.trunk = HGModel(params)
        self.trunk_net = self.trunk.netG
        self.pose_net = martinez_net(params, 56, 51)[1]
        # self.pose_net.cuda()

        norm_params = preprocess_params(params.norm_params_path)
        self.mean2d = torch.from_numpy(norm_params['mean2d']).float().cuda()
        self.std2d = torch.from_numpy(norm_params['std2d']).float().cuda()

        self.frames = np.arange(30).reshape((-1, 1, 1))
        self.frames = np.tile(self.frames, (1, 4, 14))

    def eval(self):
        self.trunk.switch_to_eval()
        self.pose_net.eval()

    def train(self):
        # disable network as necessary
        if self.trunk_finetune != 'all':
            self._freeze_params()

        self.pose_net.train()
        self._make_batchnorm_untrainable()

    def forward(self, imgs, poses, normed_poses, good_pose):
        """
        Parameters:
            imgs: input images
            poses: 2D poses scaled to input size of the MegaDepth network
            normed_poses: 2D poses split into relative/hip and normed with the inverse calibration matrix
            good_pose: (nBatch, nFrames, 4) whether the given pose was detected by OpenPose or not
        """
        depths = self.trunk_net(imgs).squeeze(dim=1)  # need to remove channel dimension

        depth_at_kps = depths[self.frames[:len(imgs)][:, :poses.shape[1]], poses[:, :, :, 1], poses[:, :, :, 0]]
        posenet_input = torch.cat([normed_poses[good_pose], depth_at_kps[good_pose]], dim=1)
        posenet_input = (posenet_input - self.mean2d) / self.std2d
        return self.pose_net(posenet_input)
