import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        if hasattr(opt, "megadepth_pretrained_weights"):
            self.pretrained_model_path = opt.megadepth_pretrained_weights
        else:
            self.pretrained_model_path = None

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        raise RuntimeError("save_network not implemented")
        # save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        # save_path = os.path.join(self.save_dir, save_filename)
        # torch.save(network.cpu().state_dict(), save_path)
        # if len(gpu_ids) and torch.cuda.is_available():
        #     network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self):
        """ Returns the saved pretrained model weights. """
        print(self.pretrained_model_path)
        model_weights = torch.load(self.pretrained_model_path)
        return model_weights
        # network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        pass
