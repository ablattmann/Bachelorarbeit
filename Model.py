import torch
import torch.nn as nn
from architecture_ops import E, Decoder
from ops import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec, total_loss, get_heat_map


class Model(nn.Module):
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg
        self.mode = arg.mode
        self.n_parts = arg.n_parts
        self.n_features = arg.n_features
        self.device = arg.device
        self.depth_s = arg.depth_s
        self.depth_a = arg.depth_a
        self.residual_dim = arg.residual_dim
        self.covariance = arg.covariance
        self.L_mu = arg.L_mu
        self.L_cov = arg.L_cov
        self.scal = arg.scal
        self.E_sigma = E(self.depth_s, self.n_parts, residual_dim=self.residual_dim, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, residual_dim=self.residual_dim, sigma=False)
        self.decoder = Decoder(self.n_parts, self.n_features)

    def forward(self, x, x_spatial_transform, x_appearance_transform, coord, vector):
        # Shape Stream
        shape_stream_parts, shape_stream_sum = self.E_sigma(x_appearance_transform)
        mu, L_inv = get_mu_and_prec(shape_stream_parts, self.device, self.scal)
        heat_map = get_heat_map(mu, L_inv, self.device)
        # Appearance Stream
        appearance_stream_parts, appearance_stream_sum = self.E_sigma(x_spatial_transform)
        f_xs = self.E_alpha(appearance_stream_sum)
        alpha = get_local_part_appearances(f_xs, appearance_stream_parts)
        # Decoder
        encoding = feat_mu_to_enc(alpha, mu, L_inv, self.device, self.covariance)
        reconstruction = self.decoder(encoding)
        # Loss
        loss = total_loss(x, reconstruction, shape_stream_parts, appearance_stream_parts, coord, vector, self.device,
                          self.L_mu, self.L_cov, self.scal)

        if self.mode == 'predict':
            return x, reconstruction, mu, heat_map

        elif self.mode == 'train':
            return reconstruction, loss