import numpy as np
import torch
import posterior

from types import SimpleNamespace

import torch
import torch.nn as nn

from pos_enc import (
    DiscreteSinCosPositionalEncoding,
    ContinuousSinCosPositionalEncoding,
    RelativePositionalEncodingInterp,
    RelativePositionalEncodingNN
)

from attention import (
    DotProductAttention,
    TemporalAttention,
    TemporalDotProductAttention,
    TemporalDotProductAttentionBaseline,
    SpatioTemporalDotProductAttention,
    SpatioTemporalDotProductAttention_V2
)

from tf_encoder import TFEncoder

ndarray = np.ndarray
Tensor = torch.Tensor
Module = nn.Module

class ToNormalParameters(Module):
    """Converts output of CNNDecoder to parameters of p(y|x)."""
    def __init__(self, sigY) -> None:
        super().__init__()
        self.sigY = sigY

    def forward(self, x):
        x[..., 0] = torch.sigmoid(x[..., 0])  # to keep mean \in (0, 1)
        x[..., 1] = self.sigY  # fix standard deviation
        return x


def get_inference_data(t: Tensor, y: Tensor, delta_inf: float) -> tuple[list[Tensor], list[Tensor]]:
    t_inf, y_inf = [], []
    for i in range(t.shape[0]):
        inf_inds = torch.argwhere(t[[i]] <= delta_inf)[:, 1]
        t_inf.append(t[[i]][:, inf_inds, :])
        y_inf.append(y[[i]][:, inf_inds, :, :])
    return t_inf, y_inf


def get_x0(elbo, t: list[Tensor], y: list[Tensor]) -> Tensor:
    x0 = []
    for ti, yi in zip(t, y):
        elbo.q.rec_net.update_time_grids(ti)
        gamma, tau = elbo.q.rec_net(yi)
        x0.append(gamma[:, [0], :] + tau[:, [0], :] * torch.randn_like(tau[:, [0], :]))
    return torch.cat(x0)


def _pred_full_traj(elbo, t: Tensor, x0: Tensor) -> Tensor:
    # elbo.p.set_theta(elbo.q.sample_theta())
    S, M, K = x0.shape[0], t.shape[1], x0.shape[2]

    x = torch.zeros((S, M, K), dtype=x0.dtype, device=x0.device)
    x[:, [0], :] = x0

    for i in range(1, M):
        x[:, [i], :] = elbo.p.F(x[:, [i-1], :], t=posterior.extract_time_grids(t[:, i-1:i+1, :], n_blocks=1))

    return elbo.p._sample_lik(x)


def pred_full_traj(param, elbo, t: Tensor, y: Tensor) -> Tensor:
    t_inf, y_inf = get_inference_data(t, y, param.delta_inf)
    x0 = get_x0(elbo, t_inf, y_inf)
    y_full_traj = _pred_full_traj(elbo, t, x0)
    return y_full_traj


def create_agg_net(param: SimpleNamespace, net_type: str) -> nn.Sequential:
    """Constructs aggregation network."""

    pos_enc_layers = {
        "dsc": DiscreteSinCosPositionalEncoding,
        "csc": ContinuousSinCosPositionalEncoding,
        "rpeNN": RelativePositionalEncodingNN,
        "rpeInterp": RelativePositionalEncodingInterp,
        "none": None,
    }

    attn_layers = {
        "dp": DotProductAttention,
        "t": TemporalAttention,
        "tdp": TemporalDotProductAttention,
        "tdp_b": TemporalDotProductAttentionBaseline,
        "stdp": SpatioTemporalDotProductAttention,
        "stdp2": SpatioTemporalDotProductAttention_V2,
    }

    attn_key, pos_enc_key = param.h_agg_attn, param.h_agg_pos_enc
    assert pos_enc_key in pos_enc_layers.keys(), f"Wrong position encoding name: {pos_enc_key}."
    assert attn_key in attn_layers.keys(), f"Wrong attention layer name: {attn_key}."

    t_init = torch.linspace(0, 1, 3).view(1, -1, 1)  # update it later
    pos_enc_args = {
        "d_model": param.m_h*param.K,
        "t": t_init,
        "max_tokens": param.h_agg_max_tokens,
        "max_time": param.h_agg_max_time,
        "delta_r": param.h_agg_delta_r,
        "f": nn.Linear(1, param.m_h*param.K, bias=False),
    }
    attn_args = {
        "d_model": param.m_h*param.K,
        "t": t_init,
        "eps": 1e-2,
        "delta_r": param.h_agg_delta_r,
        "p": param.h_agg_p,
        "n": param.n,
        "drop_prob": param.drop_prob,
    }

    if net_type == "static":
        param.h_agg_layers = param.h_agg_stat_layers
    elif net_type == "dynamic":
        param.h_agg_layers = param.h_agg_dyn_layers

    modules = []
    if pos_enc_key in ["dsc", "csc"]:  # absolute positional encodings
        pos_enc = pos_enc_layers[pos_enc_key](**pos_enc_args)
        tf_enc_blocks = []
        for _ in range(param.h_agg_layers):
            tf_enc_block = TFEncoder(
                d_model=param.m_h*param.K,
                self_attn=attn_layers[attn_key](**attn_args),
                t=t_init,
                dim_feedforward=2*param.m_h*param.K,
            )
            tf_enc_blocks.append(tf_enc_block)
        modules.extend([pos_enc, *tf_enc_blocks])
    else:  # relative positional encodings
        if pos_enc_key == "none":
            print("Using no positional encodings!")
            pos_enc = None
        else:
            pos_enc = pos_enc_layers[pos_enc_key](**pos_enc_args)
        tf_enc_blocks = []
        for i in range(param.h_agg_layers):
            if i == 0:
                self_attn = attn_layers["t"](rpe=pos_enc, **attn_args)
            else:
                self_attn = attn_layers[attn_key](rpe=pos_enc, **attn_args)

            tf_enc_block = TFEncoder(
                d_model=param.m_h*param.K,
                self_attn=self_attn,
                t=t_init,
                dim_feedforward=2*param.m_h*param.K,
            )
            tf_enc_blocks.append(tf_enc_block)

        modules.extend(tf_enc_blocks)

    return nn.Sequential(*modules)
