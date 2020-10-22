import torch
import torch.nn as nn
from architecture_ops import E, Decoder
import matplotlib.pyplot as plt
from ops import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec, total_loss
from DataLoader import ImageDataset, DataLoader
import numpy as np


class Model(nn.Module):
    def __init__(self, parts=16, n_features=32):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.E_sigma = E(3, parts, residual_dim=64, sigma=True)
        self.E_alpha = E(1, n_features, residual_dim=64, sigma=False)
        self.decoder = Decoder(parts, n_features)

    def forward(self, x, x_spatial_transform, x_appearance_transform, coord, vector):
        # Shape Stream
        shape_stream_parts, shape_stream_sum = self.E_sigma(x_appearance_transform)
        mu, L_inv = get_mu_and_prec(shape_stream_parts, self.device)
        # Appearance Stream
        appearance_stream_parts, appearance_stream_sum = self.E_sigma(x_spatial_transform)
        f_xs = self.E_alpha(appearance_stream_sum)
        alpha = get_local_part_appearances(f_xs, appearance_stream_parts)
        # Decoder
        encoding = feat_mu_to_enc(alpha, mu, L_inv, self.device)
        reconstruction = self.decoder(encoding)
        # Loss
        loss = total_loss(x, reconstruction, shape_stream_parts, appearance_stream_parts, coord, vector, self.device)

        return reconstruction, loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    model.train()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)
    image = np.expand_dims(plt.imread('Test.png'), 0).repeat(1, axis=0)
    train_set = ImageDataset(image, device=device)
    train_loader = DataLoader(train_set, batch_size=8)
    for batch_id, (original, spat, app, coord, vec) in enumerate(train_loader):
        coord, vec = coord.to(device), vec.to(device)
        original, spat, app = original.to(device), spat.to(device), app.to(device)
        for epoch in range(1000):
            optimizer.zero_grad()
            prediction, loss = model(original, spat, app, coord, vec)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')

train()
