
import typing as typ

import torch
import torch.nn as nn
import torch.nn.functional as F


# Initial implementation:
# https://github.com/facebookresearch/DeepSDF/blob/master/networks/deep_sdf_decoder.py

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        dims: typ.List[int],
        two_d: bool = False,
        dropout: typ.List[int] = None,
        dropout_prob: float = 0.0,
        norm_layers: typ.List[int] = (),
        latent_in=(),
        weight_norm: bool = False,
        xyz_in_all: bool = None,
        use_tanh: bool = False,
        latent_dropout: bool = False,
    ):
        """
        :param latent_size: size of latent vector Z (shape code) (ex.: 256, 128)
        :param dims: num of neurons per each layer (ex.: [512] * 8)
        :param two_d: if True use (x, y), else (x, y, z)
        :param dropout: layer indexes with dropout enabled
        :param dropout_prob: [0.0, 1.0]
        :param norm_layers: layer indexes with weight normalization enabled
        :param latent_in: layer index with latent vector skip connection
        """
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        self.n_points = 2 if two_d else 3
        dims = [latent_size + self.n_points] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.n_points

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer),
                        nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    def forward(self, input: torch.Tensor) -> float:
        """
        :param input: [N, latent_size + n_points]

        :rtype: sdf value
        """
        xyz = input[:, -self.n_points:]

        if input.shape[1] > self.n_points and self.latent_dropout:
            latent_vecs = input[:, :-self.n_points]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.training and self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob,
                                  training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
