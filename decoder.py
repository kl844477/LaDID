from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from einops import rearrange

Tensor = torch.Tensor
Module = nn.Module
Parameter = nn.parameter.Parameter

class CNNDecoder(Module):
    """Mapping from R^K to R^{NxDxn_param}."""
    def __init__(self, K: int, N: int, D: int, n_param: int, n_channels: int) -> None:
        super().__init__()

        self.K = K
        self.N = N
        self.D = D
        self.n_param = n_param

        self.n_channels = n_channels
        self.img_size = int(N**0.5)
        self.n_feat = (self.img_size//16)**2 * (8 * n_channels)

        self.lin_layer = nn.Linear(K, self.n_feat)

        self.f = nn.Sequential(
            nn.ConvTranspose2d(8*n_channels, 4*n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(4*n_channels),  # img_size/8

            nn.ConvTranspose2d(4*n_channels, 2*n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(2*n_channels),  # img_size/4

            nn.ConvTranspose2d(2*n_channels, n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size/2

            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size

            nn.Conv2d(n_channels, D*n_param, kernel_size=5, padding=2),
        )

    def forward(self, x) -> Tensor:
        # x: Tensor, shape (S, M, K)
        if type(x) == tuple:
            x, static_image = x
        S, M, _ = x.shape
        nc, h = 8*self.n_channels, self.img_size//16
        x = rearrange(self.lin_layer(x), "s m (nc h w) -> (s m) nc h w", nc=nc, h=h, w=h)
        if static_image is not None:
            x = torch.add(x, static_image)
        x = self.f(x)
        x = rearrange(x, "(s m) (d npar) h w -> s m (h w) d npar", s=S, m=M, d=self.D, npar=self.n_param)
        return x

class IDecoder(Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Maps latent state to parameters of p(y|x).

        Args:
            x: Latent state. Has shape (S, M, K).

        Returns:
            param: Parameters of p(y|x). Has shape (S, M, N, D, num. of param. groups in p(y|x)).
                For example, the number of parameter groups in a Normal p(y|x) is 2 (mean and variance).
        """
        pass

    @abstractmethod
    def set_param(self, param: Tensor) -> None:
        """Sets parameters to `param`.

        Args:
            param: New parameter values.
        """
        pass

    @abstractmethod
    def param_count(self) -> int:
        """Calculates the number of parameters.

        Returns:
            The number of parameters.
        """
        pass

class NeuralDecoder(IDecoder):
    """Neural-network-based decoder."""
    def __init__(self, decoder: Module, layers_to_count: list = []) -> None:
        super().__init__()
        self.decoder = decoder
        self.layers_to_count = [
            nn.Linear,
            nn.Conv2d, nn.ConvTranspose2d,
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
        ]
        self.layers_to_count.extend(layers_to_count)

    def forward(self, x: Tensor, static_image=None) -> Tensor:
        return self.decoder((x, static_image))

    def set_param(self, param: Tensor) -> None:
        assert self.param_count() == param.numel(), (
            f"The size of param ({param.numel()}) must be the same as self.param_count()"
            f"({self.param_count()})"
        )
        layers = self._get_layers(self.layers_to_count)
        self._set_layer_param_to_vec(layers, param)

    def param_count(self) -> int:
        param_count = 0
        layers = self._get_layers(self.layers_to_count)
        for layer in layers:
            self._check_weight_and_bias_of_layer(layer)
            layer_param_count = layer.weight.numel() + layer.bias.numel()
            param_count += layer_param_count
        return param_count

    def _get_layers(self, layer_types: list) -> list:
        """Returns all layers in `self.decoder` whose type is present in `layer_types`.

        Args:
            layer_types: A list with the requred layer types (e.g. nn.Linear).

        Returns:
            Layers of `self.decoder` whose type is in `layer_types`.
        """
        return_layers = []
        for layer in self.decoder.modules():
            if type(layer) in layer_types:
                return_layers.append(layer)
        return return_layers

    def _set_layer_param_to_vec(self, layers: list[Module], vec: Tensor) -> None:
        """Sets parameters of Modules in `layers` to elements of `vec`.

        Args:
            layers: List of Modules whose parameters need to be set.
            vec: 1D Tensor with parameters.
        """
        pointer = 0
        for layer in layers:
            self._check_weight_and_bias_of_layer(layer)

            layer_param_count = layer.weight.numel() + layer.bias.numel()
            layer_weight_count = layer.weight.numel()

            layer_param = vec[pointer:pointer + layer_param_count]
            layer_weight = layer_param[:layer_weight_count].view_as(layer.weight)
            layer_bias = layer_param[layer_weight_count:].view_as(layer.bias)

            self._del_set_layer_attr(layer, "weight", layer_weight)
            self._del_set_layer_attr(layer, "bias", layer_bias)

            pointer += layer_param_count

    def _del_set_layer_attr(self, layer, attr_name, attr_val):
        delattr(layer, attr_name)
        setattr(layer, attr_name, attr_val)

    def _check_weight_and_bias_of_layer(self, layer: Module) -> None:
        assert (type(layer.weight) is Tensor or type(layer.weight) is Parameter), (
            f"weight of layer {layer} must be Tensor or Parameter.")
        assert (type(layer.bias) is Tensor or type(layer.bias) is Parameter), (
            f"bias of layer {layer} must be Tensor or Parameter.")
