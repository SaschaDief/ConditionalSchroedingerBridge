# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch
import math
from torch import nn
import numpy as np
import util


class VBLinear(nn.Module):
    """
    Bayesian linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_prec: float = 1.0,
        enable_map: bool = False,
        std_init: float = -9,
    ):
        """
        Constructs the Bayesian linear layer
        Args:
            in_features: Number of input dimensions
            out_features: Number of input dimensions
            prior_prec: Standard deviation of the Gaussian prior
            enable_map: If True, does not sample from posterior during evaluation
                        (maximum-a-posteriori)
            std_init: Logarithm of the initial standard deviation of the weights
        """
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = enable_map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of the layer.
        """
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        """
        Reset the random weights. New weights will be sampled the next time, forward is
        called in evaluation mode.
        """
        self.random = None

    def sample_random_state(self) -> np.ndarray:
        """
        Sample a random state and return it as a numpy array

        Returns:
            Tensor with sampled weights
        """
        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state: np.ndarray):
        """
        Replace the random state

        Args:
            state: Numpy array with random numbers
        """
        self.random = torch.tensor(
            state, device=self.logsig2_w.device, dtype=self.logsig2_w.dtype
        )

    def kl(self) -> torch.Tensor:
        """
        KL divergence between posterior and prior.

        Returns:
            KL divergence
        """
        logsig2_w = self.logsig2_w.clamp(-11, 11)
        kl = (
            0.5
            * (
                self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                - logsig2_w
                - 1
                - np.log(self.prior_prec)
            ).sum()
        )
        return kl

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bayesian linear layer. In training mode, use the local
        reparameterization trick.
        Args:
            input: Input tensor
        Returns:
            Output tensor
        """
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = nn.functional.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:
                return nn.functional.linear(input, self.mu_w, self.bias)

            logsig2_w = self.logsig2_w.clamp(-11, 11)
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return nn.functional.linear(input, weight, self.bias) + 1e-8


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torchttorchh.zeros_like(embedding[:, :1])], dim=-1)
    return embedding




class VBResNetDense(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1):
        super(VBResNetDense, self).__init__()
        
        #self.residual = nn.Linear(input_size, hidden_size)
        self.residual = VBLinear(input_size, hidden_size)
        
        layers = []
        for _ in range(nlayers):
            layers.extend([
                #nn.Linear(hidden_size, hidden_size),
                VBLinear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(),
                #nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        residual = self.residual(x)
        layer = self.layers(x)
        return residual + layer

class ResNetDense(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1):
        super(ResNetDense, self).__init__()
        
        self.residual = nn.Linear(input_size, hidden_size)
        
        layers = []
        for _ in range(nlayers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(),
                #nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        residual = self.residual(x)
        layer = self.layers(x)
        return residual + layer



class DenseNet(nn.Module):
    
    def linear(self, *args, **kwargs):
        if self.bayes:
            return VBLinear(*args, **kwargs)

        else:
            return torch.nn.Linear(*args, **kwargs)

    def __init__(self, log, noise_levels, x_dim=2,
                 hidden_dim=64, time_embed_dim=16,
                 nresnet = 4, nlayers_in_resnet=1,
                 use_fp16=False, cond=False, bayes=False):
        super(DenseNet, self).__init__()
        
        self.bayes = bayes

        self.cond = cond #condition model over reco level inputs
        self.noise_levels = noise_levels        

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.x_dim = x_dim
        self.nlayers_in_resnet = nlayers_in_resnet

        
        
        self.time_dense  = nn.Sequential(
            self.linear(self.time_embed_dim, self.hidden_dim),
            nn.LeakyReLU(),
            #self.linear(self.hidden_dim, self.hidden_dim),
            
        )
        
        self.inputs_dense = nn.Sequential(
            self.linear(self.x_dim if self.cond==False else 2*self.x_dim, self.hidden_dim),
            nn.LeakyReLU(),
            #self.linear(self.hidden_dim, self.hidden_dim),
        )

        
        self.residual = nn.Sequential(
            self.linear(self.hidden_dim,self.hidden_dim),
            nn.LeakyReLU(),
            self.linear(self.hidden_dim, self.hidden_dim),
        )

        if self.bayes:
            self.resnet_layers = nn.ModuleList([
                VBResNetDense(
                    self.hidden_dim, 
                    self.hidden_dim, 
                    nlayers=self.nlayers_in_resnet) for _ in range(nresnet)
            ])
        else:
            self.resnet_layers = nn.ModuleList([
                ResNetDense(
                    self.hidden_dim, 
                    self.hidden_dim, 
                    nlayers=self.nlayers_in_resnet) for _ in range(nresnet)
            ])
            
        self.final_layer = nn.Sequential(
            self.linear(self.hidden_dim, 2*self.hidden_dim),
            nn.LeakyReLU(),
            self.linear(2*self.hidden_dim, self.x_dim)
        )
        
    def forward(self, inputs,steps,cond=None):
        
        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == inputs.shape[0]
        
        embed = self.time_dense(timestep_embedding(t, self.time_embed_dim))
        if self.cond:
            x = torch.cat([inputs, cond], dim=1)
        else:
            x = inputs
        
        inputs_dense = self.inputs_dense(x)        
        residual = self.residual(inputs_dense+ embed)
        x = residual
        for layer in self.resnet_layers:
            x = layer(x)
            
        output = self.final_layer(x)
        output = output + inputs
        return output

    
    
    
    
class TransNet(nn.Module):
    
    def linear(self, *args, **kwargs):
        return torch.nn.Linear(*args, **kwargs)

    def __init__(self, log, noise_levels, x_dim,
                 time_embed_dim
                 
                 
                 
                 
                 use_fp16=False, cond=False, bayes=False):
        super(TransNet, self).__init__()
        
        self.cond = cond #condition model over reco level inputs
        self.noise_levels = noise_levels        

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.x_dim = x_dim

        
    

        
    def forward(self, inputs,steps,cond=None):
        
        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == inputs.shape[0]
        
        t = timestep_embedding(t, self.time_embed_dim)
        
        if self.cond_x1:
            x = torch.cat([t, inputs, cond], dim=-1)
        else:
            x = torch.cat([t, inputs], dim=-1)
        
        inputs_dense = self.inputs_dense(x)        
        residual = self.residual(inputs_dense+ embed)
        x = residual
        for layer in self.resnet_layers:
            x = layer(x)
            
        output = self.final_layer(x)
        output = output + inputs
        return output
