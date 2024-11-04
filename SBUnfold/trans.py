import torch.nn as nn
import torch

import math
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



class Transformer(nn.Module):

    def linear(self, *args, **kwargs):
        if self.bayes:
            return VBLinear(*args, **kwargs)

        else:
            return torch.nn.Linear(*args, **kwargs)

    def __init__(
        self, 
        dims_x, 
        dims_c, 
        dims_t,
        dim_embedding,
        n_head,
        n_encoder_layers,
        n_decoder_layers,
        dim_feedforward,
        bayes=False,
        dropout=0.0
    ):
        super().__init__()
        self.bayes = bayes
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.dims_t = dims_t

        self.dim_embedding = dim_embedding
        
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.final_layer = self.linear(self.dim_embedding+self.dims_t, 1)

    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if t is None:
            p = p.unsqueeze(-1)
        else:        
            if t.shape[1] == 1:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
            else:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        
        
        n_rest = self.dim_embedding - n_components - p.shape[-1]
        assert n_rest >= 0
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, inputs):

        t = inputs[:, :self.dims_t]
        x = inputs[:, self.dims_t:self.dims_t+self.dims_x]
        c = inputs[:, -self.dims_c:]

        x_embedding = self.compute_embedding(
                x,
                n_components=self.dims_x,
                t=t
            )

        c_embedding = self.compute_embedding(
                c,
                n_components=self.dims_c
            )

        transformer_out = self.transformer(
            src=c_embedding,
            tgt=x_embedding
        )

        v_pred = self.final_layer(torch.cat([t.unsqueeze(1).repeat(1, x.size(1), 1),
                                     transformer_out], dim=-1)).squeeze()

        return v_pred



class TransformerV2(nn.Module):

    def linear(self, *args, **kwargs):
        if self.bayes:
            return VBLinear(*args, **kwargs)

        else:
            return torch.nn.Linear(*args, **kwargs)

    def __init__(
        self, 
        dims_x, 
        dims_c, 
        dims_t,
        dim_embedding,
        n_head,
        n_encoder_layers,
        n_decoder_layers,
        dim_feedforward,
        bayes=False,
        dropout=0.0
    ):
        super().__init__()
        self.bayes = bayes
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.dims_t = dims_t

        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward
        
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        #self.final_layer = self.linear(self.dim_embedding+self.dims_t, 1)

        self.final_layer = nn.Sequential(
            self.linear(self.dim_embedding+self.dims_t+self.dims_c, self.dim_feedforward),
            nn.LeakyReLU(),
            self.linear(self.dim_feedforward, self.dim_feedforwardm),
            nn.LeakyReLU(),
            self.linear(self.dim_feedforward, self.dim_feedforwardm),
            nn.LeakyReLU(),
            self.linear(self.dim_feedforward, 1),
        )

    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if t is None:
            p = p.unsqueeze(-1)
        else:        
            if t.shape[1] == 1:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
            else:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        
        
        n_rest = self.dim_embedding - n_components - p.shape[-1]
        assert n_rest >= 0
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, inputs):

        t = inputs[:, :self.dims_t]
        x = inputs[:, self.dims_t:self.dims_t+self.dims_x]
        c = inputs[:, -self.dims_c:]

        x_embedding = self.compute_embedding(
                x,
                n_components=self.dims_x,
                t=t
            )

        c_embedding = self.compute_embedding(
                c,
                n_components=self.dims_c
            )

        transformer_out = self.transformer(
            src=c_embedding,
            tgt=x_embedding
        )

        t_out_t = torch.cat([t.unsqueeze(1).repeat(1, x.size(1), 1),transformer_out], dim=-1).squeeze()

        v_pred = self.final_layer(torch.cat([t_out_t, c], dim=-1))

        return v_pred



class TransformerV3(nn.Module):

    def linear(self, *args, **kwargs):
        if self.bayes:
            return VBLinear(*args, **kwargs)

        else:
            return torch.nn.Linear(*args, **kwargs)

    def __init__(
        self, 
        dims_x, 
        dims_c, 
        dims_t,
        dim_embedding,
        n_head,
        n_encoder_layers,
        n_decoder_layers,
        dim_feedforward,
        bayes=False,
        dropout=0.0
    ):
        super().__init__()
        self.bayes = bayes
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.dims_t = dims_t

        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward
        
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        #self.final_layer = self.linear(self.dim_embedding+self.dims_t, 1)

        self.final_layer = nn.Sequential(
            self.linear(self.dim_embedding+self.dims_t+self.dims_c, 1)
        )

    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if t is None:
            p = p.unsqueeze(-1)
        else:        
            if t.shape[1] == 1:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
            else:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        
        
        n_rest = self.dim_embedding - n_components - p.shape[-1]
        assert n_rest >= 0
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, inputs):

        t = inputs[:, :self.dims_t]
        x = inputs[:, self.dims_t:self.dims_t+self.dims_x]
        c = inputs[:, -self.dims_c:]

        x_embedding = self.compute_embedding(
                x,
                n_components=self.dims_x,
                t=t
            )

        c_embedding = self.compute_embedding(
                c,
                n_components=self.dims_c
            )

        transformer_out = self.transformer(
            src=c_embedding,
            tgt=x_embedding
        )

        t_out_t = torch.cat([t.unsqueeze(1).repeat(1, x.size(1), 1),transformer_out], dim=-1).squeeze()

        #v_pred = self.final_layer(torch.cat([t_out_t, c], dim=-1))
        v_pred = self.final_layer(t_out_t)

        return v_pred





















