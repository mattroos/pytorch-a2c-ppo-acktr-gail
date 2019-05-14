import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

import pdb


SCALES = [1, 3, 5]
IM_PIX = 256
FOV_PIX = 32
N_CNN_CHANNELS_OUT = [16, 16, 16, 16]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class GlimpseBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(GlimpseBase, self).__init__(recurrent, num_inputs, hidden_size)
        # This is loosely modeled on the glimpse network by Mnih et al., Recurrent Models of Visual Attention, 2014.

        self.n_rnn_layers = 1   # Only use 1 until determine how to use higher.
        self.sz_loc_out = 32
        self.sz_glimpse_out = 256

        self.scales = SCALES
        self.n_scales = len(SCALES)
        self.im_pix = IM_PIX
        self.fov_pix = FOV_PIX
        self.fov_pix_half = FOV_PIX//2
        self.n_cnn_channels_out = N_CNN_CHANNELS_OUT
        self.n_pix_cnn_out = (FOV_PIX // (2**len(self.n_cnn_channels_out))) ** 2
        self.sz_rnn_hidden = hidden_size
        # self.rnn_hidden = torch.zeros(self.n_rnn_layers, 1, hidden_size).cuda()

        p_dropout = 0.0

        ## Build ModuleList of ModuleLists, each containing a layer
        #  block for a CNN at a particular scale. Outer ModuleList is
        #  for scales, inner ModuleLists are for layer blocks.
        n_chans = [3] + self.n_cnn_channels_out  # Assuming three color channels for all scales.
        k = 3   # kernel size
        self.cnns = nn.ModuleList()
        for i_scale in range(self.n_scales):
            self.cnns.append(nn.ModuleList())
            for i_layerblock in range(len(self.n_cnn_channels_out)):
                self.cnns[i_scale].append(nn.Sequential(
                    nn.Conv2d(n_chans[i_layerblock], n_chans[i_layerblock+1], k, padding=1),
                    nn.BatchNorm2d(n_chans[i_layerblock+1]),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(p=p_dropout)  # Channel dropout. Do this or not?
                    ))
            # self.cnns[i_scale].append(nn.Dropout(p=p_dropout))

        ## Fixation location layer
        #  Not clear why this is needed, but Mnih et al. used it.  Seems like location info
        #  could inatead be input to the RNN layer(s) directory, rather than first transforming
        #  and then combining with the CNN outputs.
        #  Input should be (x,y) location, with range [-1,1] where (0,0) is the image center.
        self.fc_loc = nn.Sequential(
            nn.Linear(2, self.sz_loc_out),
            nn.ReLU(),
            )

        # Combining CNN and locations layer outputs with "glimpse" layer
        self.fc_glimpse = nn.Sequential(
            nn.Linear(self.n_pix_cnn_out*self.n_cnn_channels_out[-1]*self.n_scales + self.sz_loc_out, self.sz_glimpse_out),
            nn.ReLU(),
            )

        # RNN layer (can't use RNN+dropout in Sequential, because RNN gives two outputs in a tuple)
        # self.rnn = nn.Sequential( \
        #     nn.RNN(self.sz_glimpse_out, sz_rnn, num_layers=n_rnn_layers, nonlinearity='relu'), \
        #     nn.Dropout(p=p_dropout), \
        #     )
        self.rnn = nn.RNN(self.sz_glimpse_out, hidden_size, num_layers=self.n_rnn_layers, nonlinearity='relu', bias=False)
        #self.rnn = nn.GRU(self.sz_glimpse_out, hidden_size, num_layers=self.n_rnn_layers)
        self.rnn_dropout = nn.Dropout(p=p_dropout)


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # Not clear what masks if for. Originally in forward() of MLPBase class.

        n_batches = inputs.shape[0]

        images = inputs[:, 0:self.n_scales*3*self.fov_pix**2].view([n_batches, self.n_scales, 3, self.fov_pix, self.fov_pix])
        fix_loc = inputs[:, -2:]

        # Put scaled images through CNNs. Separate CNN for each scale/image.
        cnns_out = []
        for i_scale in range(self.n_scales):
            cnns_out.append([])
            # cnns_out[i_scale] = images[i_scale].view([1] + list(images[i_scale].shape))
            cnns_out[i_scale] = images[:,i_scale,:,:,:]
            for i_nn in range(len(self.cnns[i_scale])):
                cnns_out[i_scale] = self.cnns[i_scale][i_nn](cnns_out[i_scale])
            cnns_out[i_scale] = cnns_out[i_scale].view(n_batches, -1)

        # Fixation locations
        loc_out = self.fc_loc(fix_loc)

        # Concatentate all features (CNN features and location features)
        # and process with glimpse layer.
        glimpse_out = torch.cat(cnns_out + [loc_out], 1)
        glimpse_out = self.fc_glimpse(glimpse_out)

        # TODO: PROBLEM - I don't think gym is every resetting the hidden state, even when env is 'done'.
        # rnn_out, self.rnn_hidden = self.rnn(glimpse_out.view([1,1,-1]), self.rnn_hidden)
        rnn_out, rnn_hxs = self.rnn(glimpse_out.view([1,n_batches,-1]), rnn_hxs.view([1,n_batches,-1]))
        rnn_hxs = rnn_hxs.view([n_batches,-1])
        # rnn_out = self.rnn_dropout(rnn_out) # Probably shouldn't have this dropout. B/C just before classification layer.

        hidden_critic = self.critic(rnn_out.view([n_batches,-1]))
        hidden_actor = self.actor(rnn_out.view([n_batches,-1]))

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
