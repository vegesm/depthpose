import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, dense_size, n_layers, activation, dropout, residual_enabled,
                 batchnorm_enabled, normclip_enabled, name=None):
        super(DenseBlock, self).__init__()
        assert residual_enabled, "residual_enabled==False is not implemented"
        self.residual = DenseBlock._residual_branch(dense_size, n_layers, activation, dropout, residual_enabled,
                                                    batchnorm_enabled, normclip_enabled, name)

    @staticmethod
    def _residual_branch(dense_size, n_layers, activation, dropout, residual_enabled,
                         batchnorm_enabled, normclip_enabled, name=None):

        assert not normclip_enabled

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(dense_size, dense_size))
            if batchnorm_enabled:
                layers.append(nn.BatchNorm1d(1024, momentum=0.01))  # Note track_running_stats should be False in eval?

            assert activation == 'relu', "Only ReLU is implemented"
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.residual(x)
        out += x

        return out


def martinez_net(params, input_size, output_size):
    layers = []
    layers.append(nn.Linear(input_size, params.dense_size))
    for _ in range(params.n_blocks_in_model):
        layers.append(DenseBlock(params.dense_size, params.n_layers_in_block, params.activation,
                                 params.dropout, params.residual_enabled, params.batchnorm_enabled,
                                 params.normclip_enabled))
    layers.append(nn.Linear(params.dense_size, output_size))

    model = nn.Sequential(*layers)

    # initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return "martinez", model
