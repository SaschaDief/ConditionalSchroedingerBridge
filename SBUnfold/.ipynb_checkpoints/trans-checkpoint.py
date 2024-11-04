import torch.nn as nn
import torch


class Transformer(nn.Module):

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
        dropout=0.0
    ):
        super().__init__()
        self.dims_x = dims_x
        self.dims_c = dims_c
        self.dims_t = dims_t

        self.dim_embedding = dim_embedding,
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.final_layer = nn.Linear(self.dim_embedding+self.dims_t, 1)

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
            p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
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