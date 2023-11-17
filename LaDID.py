
import torch
from torch import Tensor
import torch.nn as nn
from  torch.nn.parameter import Parameter
from einops import repeat, rearrange, reduce

from decoder import NeuralDecoder, CNNDecoder
from encoder import CNNEncoder
import utils
import math

from pos_enc import (
    RelativePositionalEncodingNN
)

Module = nn.Module

def positionalencoding1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

def rel_pos_encoding_sincos(t, t_max, t_min, n_waves):
    if n_waves % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(n_waves))

class RelPosEncodingSinCos(Module):
    def __init__(
        self,
        absolute_t_max,
        absolute_t_min
    ) -> None:
        super().__init__()
        self.absolute_t_min = absolute_t_min
        self.absolute_t_max = absolute_t_max
        dur = (absolute_t_max - absolute_t_min)*2
        dur = dur/torch.pow(2, torch.linspace(0, 8, 16))
        b = 2*torch.pi/dur
        self.b = repeat(b, 'd -> 1 1 d')
        self.b.requires_grad = False

    # @torch.no_grad
    def forward(self, t):       
        assert len(t.shape) == 3
        
        res = torch.stack((torch.sin(t*self.b.to(t.device)), torch.cos(t*self.b.to(t.device))), dim=0)
        res = rearrange(res, 'c n l d -> n l (d c)')
        return res

class SerializedModel(nn.Module):
    def __init__(self, args):
        super(SerializedModel, self).__init__()
        self.args = args

        if args.t_pos_enc == 'rel_pe_sincos':
            self.te_pe = RelPosEncodingSinCos(args.t_max, args.t_min)
        elif args.t_pos_enc == 'abs':
            self.te_pe = positionalencoding1d
        elif args.t_pos_enc == 'rpeNN':
            self.te_pe = RelativePositionalEncodingNN(
                f=nn.Linear(in_features=1, out_features=32, bias=False),
                delta_r=args.h_agg_delta_r,
                t=torch.linspace(0, 1, 3).view(1, -1, 1)  # update it later
            )
        else:
            raise NotImplementedError('Requested time encoding not implemented')
        
  
        self.phi_enc=CNNEncoder(args.m_h*args.K, args.N, args.D, args.h_enc_cnn_channels)

        self.phi_agg=utils.create_agg_net(args, "static")
        self.phi_linear=nn.Linear(args.m_h*args.K, args.K)

        if self.args.variational:
            self.phi_tau_linear=nn.Linear(args.m_h*args.K, args.K)

        self.g = NeuralDecoder(
            decoder=nn.Sequential(                    
                CNNDecoder(args.K, args.N, args.D, 2, args.g_cnn_channels),
                utils.ToNormalParameters(args.sigY),
            ),
        )

        nonlins = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "mish": nn.Mish,
        }

        self.F_psi = nn.Sequential(
                    nn.Linear(args.K + 32, args.m_F*args.K), nonlins[args.F_nonlin](),
                    nn.Dropout(p=self.args.F_drop),
                    nn.Linear(args.m_F*args.K, args.m_F*args.K), nonlins[args.F_nonlin](),
                    nn.Dropout(p=self.args.F_drop),
                    nn.Linear(args.m_F*args.K, args.K)
                )
        
        if args.variational:
            self.prior_param_dict = nn.ParameterDict({
                "mu0": Parameter(0.0 * torch.ones([args.K]), False),
                "sig0": Parameter(1.0 * torch.ones([args.K]), False),
                "sigXi": Parameter(args.Xi / args.K**0.5 * torch.ones([1]), False),
                "mu_theta": Parameter(0.0 * torch.ones([1]), False),
                "sig_theta": Parameter(1.0 * torch.ones([1]), False),
            })
        
        if args.static_representation:
            self.static_enc = nn.Sequential(
                nn.Conv2d(args.D, args.h_enc_cnn_channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(args.h_enc_cnn_channels),  # img_size/2

                nn.Conv2d(args.h_enc_cnn_channels, 2*args.h_enc_cnn_channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(2*args.h_enc_cnn_channels),  # img_size/4

                nn.Conv2d(2*args.h_enc_cnn_channels, 4*args.h_enc_cnn_channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(4*args.h_enc_cnn_channels),  # img_size/8

                nn.Conv2d(4*args.h_enc_cnn_channels, 8*args.h_enc_cnn_channels, kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(8*args.h_enc_cnn_channels),  # img_size/16

                nn.Conv2d(8*args.h_enc_cnn_channels, 8*args.h_enc_cnn_channels, kernel_size=3, stride=1, padding=1),
            )


    def kl_norm_norm(self, mu0: Tensor, mu1: Tensor, sig0: Tensor, sig1: Tensor) -> Tensor:
        """Calculates KL divergence between two K-dimensional Normal
            distributions with diagonal covariance matrices.

        Args:
            mu0: Mean of the first distribution. Has shape (*, K).
            mu1: Mean of the second distribution. Has shape (*, K).
            std0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
            std1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).

        Returns:
            KL divergence between the distributions. Has shape (*, 1).
        """
        assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
        a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
        b = ((mu1 - mu0).pow(2) / sig1**2).sum(-1, keepdim=True)
        c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
        kl = 0.5 * (a + b + c - mu0.shape[-1])
        return kl

    def calc_L1(self, x: Tensor, y: Tensor, static_image=None) -> Tensor:
        return self.loglik(y, x, static_image).sum()
    
    def calc_L1_MSE(self, x: Tensor, y: Tensor, static_image=None) -> Tensor:
        pred = self._sample_lik(x, static_image)
        mse = torch.mean((y - pred)**2, dim=(2)).sum() 
        return mse * (-1)
    
    def loglik(self, y: Tensor, x: Tensor, static_image=None) -> Tensor:
        return self._eval_loglik(y, x, static_image)
    
    def _sample_lik(self, x: Tensor, static_image) -> Tensor:
        param = self.g(x, static_image)
        mu, sig = param[..., 0], param[..., 1]
        y = torch.distributions.Normal(mu, sig).rsample()
        return y

    def _eval_loglik(self, y: Tensor, x: Tensor, static_image=None) -> Tensor:
        param = self.g(x, static_image)
        mu, sig = param[..., 0], param[..., 1]
        loglik = torch.distributions.Normal(mu, sig).log_prob(y)
        loglik = reduce(loglik, "s m n d -> s m ()", "sum")
        return loglik
    
    def calc_L2_smoothness(self, x: Tensor, gamma: Tensor, tau: Tensor, num_input_steps: int, num_query_points: int, scaler: float) -> Tensor:

        gamma_r = gamma[:, ::num_query_points,:]
        tau_r = tau[:, ::num_query_points,:]

        x_sub = x[:, ::num_query_points,:]
        S, B, K = x_sub.shape

        L2_0 = self.kl_norm_norm(
            gamma[:, 0, :],
            repeat(self.prior_param_dict["mu0"], "k -> s k", s=S, k=K),
            torch.exp(tau[:, 0, :]),
            repeat(self.prior_param_dict["sig0"], "k -> s k", s=S, k=K)
        ).sum()

        L2_1 = self.kl_norm_norm(
            gamma_r[:, :, :],
            x_sub[:, :, :],
            torch.exp(tau_r[:, :, :]),
            repeat(self.prior_param_dict["sigXi"], "() -> s b k", s=S, b=B, k=K)
        ).sum()

        return L2_0 + scaler * L2_1
    
    def calc_L2_smoothness_loglik(self, x: Tensor, gamma: Tensor, tau: Tensor,) -> Tensor:
        loglik = torch.distributions.Normal(gamma, torch.exp(tau)).log_prob(x)
        loglik = reduce(loglik, "n s d -> n s ()", "sum")
        return loglik.sum()
    
    def calc_L2(self):

        loglik = torch.distributions.Normal(torch.zeros_like(self.theta_r), torch.ones_like(self.theta_r)).log_prob(self.theta_r)
        loglik = reduce(loglik, "n d -> n ()", "sum")
        return loglik.sum()
    
    def calc_L2_overlap(self, x_sub1, x_sub2):

        loglik = torch.distributions.Normal(x_sub1, torch.ones_like(x_sub1)*10e-2).log_prob(x_sub2)
        loglik = reduce(loglik, "b n s d -> b n s ()", "sum")
        return loglik.sum()

    
    def update_time_grids(self, t: Tensor) -> None:
        """Updates all parts of aggregation net that depend on time grids."""
        for module in self.phi_agg.modules():
            if not hasattr(module, "update_time_grid"):
                continue
            if callable(getattr(module, "update_time_grid")):
                module.update_time_grid(t)  # type: ignore

    def set_rng(self, seed: int):
        self.rng = torch.Generator(device=torch.device("cuda:{}".format(0)))
        self.rng = self.rng.manual_seed(seed)
    
    def compute_static_image(self, y_blocks, num_input_steps):
        static_image = torch.mean(y_blocks[:,1:num_input_steps, :, :] - y_blocks[:,[0], :, :], dim=1, keepdim=True)
        static_image = static_image.repeat(1,y_blocks.shape[1]-num_input_steps, 1, 1)
        static_image = rearrange(static_image, 'b t (h w) d -> (b t) d h w', h=int(self.args.N**0.5), w=int(self.args.N**0.5))
        static_image_enc = self.static_enc(static_image)
        return static_image_enc

    def forward(self, t, y, batch_ids, block_size, scaler, num_input_steps, num_query_points, overlap=0, testing=False):
        n, l, *(_) = y.shape
        y_gt = y

        if isinstance(self.te_pe, (RelPosEncodingSinCos)):
            t_pe = self.te_pe(t).type_as(y)

            t_unf = t.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
            if self.args.relative_temporal_encoding:
                t_rel = t_unf - t_unf[:,:,:,[0]]
            t_rel = rearrange(t_rel, 'b n d t -> (b n) t d')
            t_pe_blocks = self.te_pe(t_rel).type_as(y)
        elif isinstance(self.te_pe, (RelativePositionalEncodingNN)):
            self.te_pe.update_time_grid(t)
            t_pe = self.te_pe().type_as(y).mean(2)
            t_pe_blocks = t_pe.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
            t_pe_blocks = rearrange(t_pe_blocks, 'b n d t -> (b n) t d')
        else:
            t_pe = self.te_pe(self.K, l)
            t_pe = repeat(t_pe, 'k d -> n k d', n=n).type_as(y)
            t_unf = t.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
            t_pe_blocks = t_pe.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
            

        if self.args.encode_full_trajectory:
            self.update_time_grids(t)        
            theta_full = self.phi_enc(y)
            theta_full = self.phi_agg(theta_full)
            gamma_full = self.phi_linear(theta_full)
            tau_full = self.phi_tau_linear(theta_full)  
        else:
            y_blocks = y.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
            y_blocks = rearrange(y_blocks, 'b n c d s -> (b n) s c d')

        t_blocks = t.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
        batch_size, blocks,  _, seq_len = t_blocks.shape
        t_blocks = rearrange(t_blocks, 'b n d s -> (b n) s d')

        if not self.args.encode_full_trajectory:
            self.update_time_grids(t_blocks[:,:num_input_steps, :])        
            theta = self.phi_enc(y_blocks[:,:num_input_steps, :, :])
            theta = self.phi_agg(theta)
        else:
            y_blocks = y.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points)
            y_blocks = rearrange(y_blocks, 'b n c d s -> (b n) s c d')
            self.update_time_grids(t_blocks[:,:num_input_steps, :])        
            theta = self.phi_enc(y_blocks[:,:num_input_steps, :, :])
            theta = self.phi_agg(theta)
        
        if self.args.static_representation:
            static_image_enc = self.compute_static_image(rearrange(y.unfold(dimension=1, size=num_input_steps + num_query_points + overlap, step=num_query_points), 'b n c d s -> (b n) s c d'), 
                                                         num_input_steps)
        else:
            static_image_enc = None

        if self.args.variational:
            gamma = self.phi_linear(theta)
            tau = self.phi_tau_linear(theta)
            noise = torch.randn(size=tau.size(), generator=self.rng, dtype=tau.dtype, device=tau.device, requires_grad=False)
            theta_r = gamma + tau * noise
            theta_r = torch.mean(theta_r[:, :num_input_steps, :], axis=1)

        else:
            theta = self.phi_linear(theta)
            theta_r = torch.mean(theta, axis=1)
        
        self.theta_r = theta_r
        theta_r = repeat(theta_r, 'n d -> n k d', k=t_pe_blocks.shape[1])
        theta_r = torch.cat((theta_r, t_pe_blocks), dim=-1)
        x = self.F_psi(theta_r[:,num_input_steps:, :])
        if overlap != 0:
            x_overlap = x[:,num_query_points:, :]
            x = x[:,:num_query_points, :]
            x_sub1 = rearrange(x, '(b n) s d -> b n s d', b=batch_size, n=blocks)[:, 1:, :overlap, :]
            x_sub2 = rearrange(x_overlap, '(b n) s d -> b n s d', b=batch_size, n=blocks)[:, :-1, :, :]
        x = rearrange(x, '(b n) s d -> b (n s) d', b=batch_size, n=blocks)

        if self.args.static_representation:
            L1 = self.calc_L1(x, y_gt[:,num_input_steps:y.shape[1]-overlap, :, :], static_image=static_image_enc)
        else:
            L1 = self.calc_L1(x, y_gt[:,num_input_steps:y.shape[1]-overlap, :, :])
        if self.args.variational:
            if self.args.encode_full_trajectory:
                L2_1 = self.calc_L2_smoothness_loglik(x, gamma_full[:, num_input_steps:, :], tau_full[:, num_input_steps:, :]) * (-1)
                if self.args.use_L2_2:
                    L2_2 = self.calc_L2() * (-1)
                    L2 = (L2_1 + L2_2) * self.args.smoothness_scaling
                else:
                    L2 = (L2_1) * self.args.smoothness_scaling
            else:
                L2 = self.calc_L2() * (-1)
            if overlap != 0:
                L2_overlap = self.calc_L2_overlap(x_sub1, x_sub2)
                L2 = L2 - L2_overlap
        else:
            L2 = torch.zeros_like(L1)

        if testing:
            trajectory = self._sample_lik(x, static_image_enc)
            return L1, L2, x, trajectory
        else:
            return L1, L2, x